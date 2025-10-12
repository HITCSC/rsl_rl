import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,groups=1,bias=False):
        """
        :brief: similar with pytorch's group conv, but shared the weights between different groups. for examplem,
        assume input tensor with shape [B, H*C_in, L, W], then the output tensor will be [B, H*C_out, L, W]. Firstly, 
        the input tensor will be spilt into H groups with shape [B, C_in, L, W], then each group will be convolved with
        the same kernel with shape [C_out, C_in, kernel_size, kernel_size], and finally the output tensor will be concatenated
        along the channel dimension.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :param groups: number of groups
        """
        super(SharedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 定义一组可训练参数
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = None
        if (bias):
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # init weights 
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            nn.init.xavier_normal_(self.bias.data)

    def forward(self, x):
        # x 形状: [B, H, L, W]
        assert x.shape[1] == self.in_channels*self.groups, \
            "Require Input Tensor Shape [B, H, L, W] with H == C*groups"
        
        # 扩展权重和偏置：从 [C, 1, kH, kW] 到 [H*C, 1, kH, kW]
        expanded_weight = self.weight.repeat(self.groups, 1, 1, 1)
        expanded_bias = None if (self.bias is None) else self.bias.repeat(self.groups)
        
        # 使用卷积操作，但通过分组实现并行处理
        output = nn.functional.conv2d(x, expanded_weight, expanded_bias, 
                                      stride=self.stride, 
                                      padding=self.padding, 
                                      groups=self.groups)
        return output

class AttentionEncoderBlock(nn.Module):
    def __init__(self, d_obs:int,embedding_dim=64, h=16):
        """
        :param d_obs: 本体感觉观测的维度(单次观测)
        :param d: MHA模块的维度 (默认64)
        :param h: 注意力头数 (默认16)
        :param map_size: 地图扫描的尺寸 (L, W)
        """
        super(AttentionEncoderBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.h = h
        # self.L, self.W = map_size

        # CNN用于处理高度图 (z值)
        # self.cnn = nn.Sequential(
        #     SharedConv2d(1, 16, kernel_size=5, padding=2,groups=self.history_len),  # 保持空间维度不变
        #     nn.ReLU(),
        #     SharedConv2d(16, (self.embedding_dim - 3), kernel_size=5, padding=2,groups=self.history_len),  # 保持空间维度不变
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 保持空间维度不变
            nn.ReLU(),
            nn.Conv2d(16, (self.embedding_dim - 3), kernel_size=5, padding=2),  # 保持空间维度不变
        )

        # 本体感觉嵌入的线性层
        self.proprio_linear = nn.Linear(d_obs, embedding_dim) # 这里只对最后一个维度的向量进行操作, output shape (B,H,d)

        # 多头注意力模块
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=h, batch_first=True)

    def forward(self, map_scans, proprioception):
        """
        :param map_scans: height scan/high level input, shape (B, H, L, W, 3)
        :param proprioception: proprioception, shape (B,H,d_obs)
        :return: map_encoding: latent representation of the map, shape (B, H, embedding_dim)
        :return: proprio_embedding: proprioception embedding, shape (B, H, embedding_dim)
        :return: attn_weights: attention weights, shape (B, H, L, W)
        """
        B = map_scans.shape[0]
        H = proprioception.shape[1]
        L = map_scans.shape[2]
        W = map_scans.shape[3]
        high_dim_obs = map_scans.view(B*H,*map_scans.shape[2:]) # (B*H, L, W, 3)
        low_dim_obs = proprioception.view(B*H, *proprioception.shape[2:]) # (B*H, d_obs)

        # 1. 处理地图扫描
        # 提取z值 (高度)
        z_values = high_dim_obs[..., 2:3]  # (B*H, L, W, 1)

        # 转换为通道优先格式 (B*H, 1, L, W)
        z_values = z_values.permute(0, 3, 1, 2)

        # 通过CNN处理z值
        cnn_features = self.cnn(z_values)  # (B*H, d-3, L, W)

        # 转换回通道最后格式 (B*H, L, W, d-3)
        cnn_features = cnn_features.permute(0, 2, 3, 1)

        # 拼接CNN特征和原始坐标
        local_features = torch.cat([high_dim_obs, cnn_features], dim=-1)  # (B*H, L, W, d)

        # 重塑为点级特征 (B*H, L*W, d)
        pointwise_features = local_features.reshape(B*H, L*W, self.embedding_dim)

        # 2. 处理本体感觉
        proprio_embedding = self.proprio_linear(low_dim_obs)  # (B*H, d)
        # print(proprio_embedding.shape)
        proprio_embedding = proprio_embedding.unsqueeze(1)  # (B*H, 1, d)

        # 3. 多头注意力
        # 查询: proprio_embedding, 键值: pointwise_features
        map_encoding, attn_weights = self.mha(
            query=proprio_embedding,
            key=pointwise_features,
            value=pointwise_features
        )  # (B*H, 1, d) & (B*H, 1, L*W)

        # reshape to (B, H, d) & (B, H, L,W)
        history_map_enc = map_encoding.view(B,H,self.embedding_dim)
        history_attn_weights = attn_weights.view(B,H,L,W)

        return history_map_enc,proprioception,history_attn_weights

class AttentionMapEncoder(nn.Module):
    """
    完整的策略网络,包含编码器和后续MLP
    """

    def __init__(self, d_obs, embedding_dim=64, h=16):
        """
        :param d_obs: 本体感知向量的维度(单次观测)
        :param d: 编码维度

        """
        super(AttentionMapEncoder, self).__init__()
        # 这里需要对NaN的值进行处理,将其替换为0

        # 注意力地图编码模块
        self.encoder = AttentionEncoderBlock(d_obs, embedding_dim, h)

    def forward(self, map_scans, proprioception,embedding_only=False):
        """
        :param map_scans: height scan/high level input, shape (B, H, L, W, 3)
        :param proprioception: 本体感觉, 形状为 (B,H,d_obs)
        :return map_encoding: (B,H,d)
        :return attention: (B,H,L,W)
        """
        mask = torch.isnan(map_scans)
        # if (mask.any()):
        #     print("Warning: NaN values found in map_scans, replacing with 0.")
        map_scans[mask] = 0.0
        # 获取编码
        map_encoding, proprioception,attention = self.encoder(map_scans, proprioception)
        # [B,H,d], [B,H,d_obs], [B,H,L,W]
        # 拼接地图编码和原始本体感觉
        if (embedding_only):
            return map_encoding,attention
        else:
            combined = torch.cat([map_encoding, proprioception], dim=-1)  # (B, H, d + d_obs)
            return combined, attention


if __name__ == "__main__":
    d = 64  # MHA维度
    h = 16  # 注意力头数
    d_obs = 48  # 假设的本体感觉维度 (论文中未明确给出)
    map_size = (26, 16)  # ANYmal-D的地图尺寸
    horizon = 2
    # 创建模型
    model = AttentionMapEncoder(d_obs,d, h)

    # 创建示例输入
    batch_size = 4
    map_scans = torch.randn(batch_size, horizon, map_size[0], map_size[1], 3)  # (4, 26, 16, 3)
    proprioception = torch.randn(batch_size, horizon, d_obs)  # (4,2, 48)

    # 前向传播
    embedding,attention = model(map_scans, proprioception)

    print(f"输入地图扫描形状: {map_scans.shape}")
    print(f"输入本体感觉形状: {proprioception.shape}")
    print(f"output embedding shape: {embedding.shape}")
    print(f"output attention shape: {attention.shape}")