import torch
import torch.nn as nn
import torch.nn.functional as F


class _Learned2DPositionalEncoding(nn.Module):
    """Learned 2D positional encoding for a fixed max grid.

    Returns a tensor of shape [1, L*W, D].
    """

    def __init__(self, max_L: int, max_W: int, dim: int):
        super().__init__()
        self.max_L = int(max_L)
        self.max_W = int(max_W)
        self.dim = int(dim)
        self.pos_table = nn.Parameter(torch.zeros(1, self.max_L * self.max_W, self.dim))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

    def forward(self, L: int, W: int) -> torch.Tensor:
        if L > self.max_L or W > self.max_W:
            raise ValueError(f"Requested grid {(L, W)} exceeds max {(self.max_L, self.max_W)}")
        return self.pos_table[:, : L * W, :]


def _sincos_2d_positional_encoding(L: int, W: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """2D sin-cos positional encoding.

    Output: [1, L*W, dim]
    Note: dim must be even and is split equally for y/x.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for 2D sin-cos, got dim={dim}")
    half = dim // 2
    if half % 2 != 0:
        # we use standard 1D sincos which needs even dims per axis
        raise ValueError(f"dim/2 must be even for 2D sin-cos, got dim={dim}")

    yy = torch.arange(L, device=device)
    xx = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # [L,W]
    grid_y = grid_y.reshape(-1).to(dtype=torch.float32)
    grid_x = grid_x.reshape(-1).to(dtype=torch.float32)

    def _sincos_1d(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = torch.arange(d // 2, device=pos.device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (d // 2)))
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, d]

    pe_y = _sincos_1d(grid_y, half)  # [L*W, half]
    pe_x = _sincos_1d(grid_x, half)  # [L*W, half]
    pe = torch.cat([pe_y, pe_x], dim=1)  # [L*W, dim]
    pe = pe.to(device=device)
    if dtype is not None:
        pe = pe.to(dtype=dtype)
    return pe.unsqueeze(0)

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
    def __init__(
        self,
        d_obs: int,
        embedding_dim: int = 64,
        h: int = 16,
        enable_pos_encoding: bool = True,
        pos_encoding_type: str = "learned",
        pos_encoding_max_hw: tuple[int, int] = (64, 64),
    ):
        """
        :param d_obs: 本体感觉观测的维度(单次观测)
        :param d: MHA模块的维度 (默认64)
        :param h: 注意力头数 (默认16)
        :param map_size: 地图扫描的尺寸 (L, W)
        """
        super(AttentionEncoderBlock, self).__init__()
        self.use_single_prep = False
        self.embedding_dim = embedding_dim
        self.h = h
        self.enable_pos_encoding = enable_pos_encoding
        self.pos_encoding_type = pos_encoding_type
        # self.L, self.W = map_size

        # CNN用于处理高度图 (z值)
        # self.cnn = nn.Sequential(
        #     SharedConv2d(1, 16, kernel_size=5, padding=2,groups=self.history_len),  # 保持空间维度不变
        #     nn.ReLU(),
        #     SharedConv2d(16, (self.embedding_dim - 3), kernel_size=5, padding=2,groups=self.history_len),  # 保持空间维度不变
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 保持空间维度不变
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, (self.embedding_dim - 3), kernel_size=5, padding=2),  # 保持空间维度不变
            # nn.BatchNorm2d(self.embedding_dim - 3),
            # nn.ReLU(),
        )

        self.proprio_linear = nn.Linear(d_obs, embedding_dim) 

        # 2D positional encoding for map tokens (L*W)
        if self.enable_pos_encoding:
            if self.pos_encoding_type == "learned":
                max_L, max_W = pos_encoding_max_hw
                self.pos_encoding = _Learned2DPositionalEncoding(max_L=max_L, max_W=max_W, dim=self.embedding_dim)
            elif self.pos_encoding_type == "sincos":
                self.pos_encoding = None  # generated on the fly
            else:
                raise ValueError(
                    f"Unknown pos_encoding_type={self.pos_encoding_type}. Expected 'learned' or 'sincos'."
                )
        

        # 多头注意力模块
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=h, batch_first=True)
    # prop:prop_obs
    def forward(self, map_scans, proprioception):
        """
        :param map_scans: height scan/high level input, shape (B, H, L, W, 3)
        :param proprioception: proprioception, shape (B,H,d_obs)
        :return: map_encoding: latent representation of the map, shape (B, H, embedding_dim)
        :return: proprio_embedding: proprioception embedding, shape (B, H, embedding_dim)
        :return: attn_weights: attention weights, shape (B, H, L, W)
        """
        # TODO map_scans的H使用prop的，何意味 
        # TODO 单帧map_scan与多帧prop是否可行
        B = map_scans.shape[0]
        # H_high_dim_obs = map_scans.shape[1]
        H = proprioception.shape[1]
        L = map_scans.shape[2]
        W = map_scans.shape[3]
        # 需要解决输入维度不一样的问题——把H堆到obs的维度上,解耦map_scan，但是我传入的都是一帧的啊
        # TODO 把H堆到obs的维度上,解耦map_scan —— 保证mha输入的第一维度（batch）相同和embedding维度相同
        # flatten history dimension so each timestep becomes a token batch
        high_dim_obs = map_scans.reshape(B * H, *map_scans.shape[2:])  # (B*H, L, W, C)
        low_dim_obs = proprioception.reshape(B * H, *proprioception.shape[2:])  # (B*H, d_obs)
        # 因为actor，critic复用同一个encoder，目前仅actor的base_lin_vel传000，统一输入dim [batch*h,91]
        # 那么问题来了：因为critic中有真实速度传入，会不会影响Velocity estimator的训练
        # 目前可以attention，因为所有的obs都加了history
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

        # add 2D positional encoding so attention can distinguish locations
        if self.enable_pos_encoding:
            if self.pos_encoding_type == "learned":
                pe = self.pos_encoding(L, W).to(device=pointwise_features.device, dtype=pointwise_features.dtype)
            else:  # sincos
                pe = _sincos_2d_positional_encoding(
                    L, W, self.embedding_dim, device=pointwise_features.device, dtype=pointwise_features.dtype
                )
            pointwise_features = pointwise_features + pe

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

        # reshape to (B,H,d) & (B, H, d) & (B, H, L,W)
        history_proprio_embedding = proprio_embedding.view(B,H,self.embedding_dim)
        history_map_enc = map_encoding.view(B,H,self.embedding_dim)
        history_attn_weights = attn_weights.view(B,H,L,W)
        return history_map_enc,proprioception,history_attn_weights

class AttentionMapEncoder(nn.Module):
    """
    完整的策略网络,包含编码器和后续MLP
    """

    def __init__(
        self,
        d_obs,
        embedding_dim=64,
        h=16,
        enable_pos_encoding: bool = True,
        pos_encoding_type: str = "learned",
        pos_encoding_max_hw: tuple[int, int] = (64, 64),
    ):
        """
        :param d_obs: 本体感知向量的维度(单次观测)
        :param d: 编码维度

        """
        super(AttentionMapEncoder, self).__init__()
        # 这里需要对NaN的值进行处理,将其替换为0
        # 注意力地图编码模块
        self.encoder = AttentionEncoderBlock(
            d_obs,
            embedding_dim,
            h,
            enable_pos_encoding=enable_pos_encoding,
            pos_encoding_type=pos_encoding_type,
            pos_encoding_max_hw=pos_encoding_max_hw,
        )

    def forward(self, map_scans, proprioception, embedding_only=False):
        """
        :param map_scans: height scan/high level input, shape (B, H, L, W, 3)
        :param proprioception: 本体感觉, 形状为 (B,H,d_obs)
        :return map_encoding: (B,H,d)
        :return attention: (B,H,L,W)
        """
        # ONNX 不支持 torch.isnan，使用 torch.where 替换
        map_scans = torch.where(torch.isnan(map_scans), torch.zeros_like(map_scans), map_scans)


        # print("prop:",proprioception.shape)
        # print("prop:",proprioception[0,0,...])
        # proprioception = proprioception.view(B,H,:88)
        # 获取编码
        map_encoding, proprioception,attention = self.encoder(map_scans, proprioception)
        # attention [batch,h,L,W]
        # print("attention size",attention.shape)
        # attention = attention.view([...,-1,...])
        # print("attention dim", attention.shape)
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
    model = AttentionMapEncoder(d_obs,d, h,True)

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