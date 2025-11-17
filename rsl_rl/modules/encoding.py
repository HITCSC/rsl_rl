

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class AttentionBasedMapEncoding(nn.Module):
    def __init__(self, d=64, h=16, d_obs=None, map_size=None):
        """
        Attention-Based Map Encoding 模块

        参数:
            d: MHA模块的维度 (默认64)
            h: 注意力头数 (默认16)
            d_obs: 本体感觉观测的维度
            map_size: 地图扫描的尺寸 (L, W)
        """
        super(AttentionBasedMapEncoding, self).__init__()

        self.d = d
        self.h = h
        self.L, self.W, self.channel = map_size

        # CNN用于处理高度图 (z值)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 保持空间维度不变
            nn.ReLU(),
            nn.Conv2d(16, d - 3, kernel_size=5, padding=2)  # 保持空间维度不变
        )

        # 本体感觉嵌入的线性层
        self.proprio_linear = nn.Linear(d_obs, d)

        # 多头注意力模块
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=h, batch_first=True)

    def forward(self, map_scans, proprioception):
        """
        前向传播

        参数:
            map_scans: 地图扫描, 形状为 (batch_size, L*W, 3)
            proprioception: 本体感觉, 形状为 (batch_size, d_obs)

        返回:
            map_encoding: 地图编码, 形状为 (batch_size, 1, d)
            proprio_embedding: 本体感觉嵌入, 形状为 (batch_size, 1, d)
        """
        # print("map_scans shape: ", map_scans.shape)
        # print("self.W: ", self.W, "self.L: ", self.L)
        torch.where(torch.isnan(map_scans),torch.zeros_like(map_scans),map_scans)
        map_scans = map_scans.reshape(map_scans.shape[0], self.W, self.L, map_scans.shape[2])
        # torch.set_printoptions(threshold=sys.maxsize)
        # print("map_scans: ", map_scans[0, :, :, :])
        batch_size = map_scans.shape[0]

        # 1. 处理地图扫描
        # 提取z值 (高度)
        z_values = map_scans[..., 2:3]  # (batch_size, L, W, 1)

        # 转换为通道优先格式 (batch_size, 1, L, W)
        z_values = z_values.permute(0, 3, 1, 2)

        # 通过CNN处理z值
        cnn_features = self.cnn(z_values)  # (batch_size, d-3, L, W)

        # 转换回通道最后格式 (batch_size, L, W, d-3)
        cnn_features = cnn_features.permute(0, 2, 3, 1)

        # 拼接CNN特征和原始坐标
        local_features = torch.cat([map_scans, cnn_features], dim=-1)  # (batch_size, L, W, d)

        # 重塑为点级特征 (batch_size, L*W, d)
        pointwise_features = local_features.reshape(batch_size, self.L * self.W, self.d)

        # 2. 处理本体感觉
        proprio_embedding = self.proprio_linear(proprioception)  # (batch_size, d)
        # print(proprio_embedding.shape)
        proprio_embedding = proprio_embedding.unsqueeze(1)  # (batch_size, 1, d)

        # 3. 多头注意力
        # 查询: proprio_embedding, 键值: pointwise_features
        map_encoding, attn_weights = self.mha(
            query=proprio_embedding,
            key=pointwise_features,
            value=pointwise_features
        )  # (batch_size, 1, d)
        # print("map_encoding shape: ", map_encoding.shape)
        # print("attn_weights.shape: ", attn_weights.shape)
        attn_weights = attn_weights.reshape(batch_size, 1, self.W, self.L)
        # print("attn_weights: ", attn_weights)

        return map_encoding, proprioception, attn_weights


class Encoder(nn.Module):
    """
    完整的策略网络，包含编码器和后续MLP
    """

    def __init__(self, d=64, h=16, d_obs=None, map_size=(26, 16), action_dim=12):
        super(Encoder, self).__init__()

        # 注意力地图编码模块
        self.encoder = AttentionBasedMapEncoding(d, h, d_obs, map_size)

        # 后续MLP策略网络 (论文中未详细说明结构，根据常见实践设计)
        # self.mlp = nn.Sequential(
        #     nn.Linear(d + d_obs, 256),  # 拼接地图编码和原始本体感觉
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_dim),
        #     nn.Tanh()  # 假设动作在[-1, 1]范围内
        # )

    def forward(self, map_scans, proprioception):
        # 获取编码
        map_scans = torch.where(torch.isnan(map_scans), torch.zeros_like(map_scans), map_scans)
        map_encoding, proprioception = self.encoder(map_scans, proprioception)

        # 拼接地图编码和原始本体感觉
        # combined = torch.cat([map_encoding.squeeze(1), proprioception], dim=-1)  # (batch_size, d + d_obs)

        # 通过MLP生成动作
        # actions = self.mlp(combined)  # (batch_size, action_dim)

        # return combined
        return map_encoding


if __name__ == "__main__":

    d = 64  # MHA维度
    h = 16  # 注意力头数
    d_obs = 48  # 假设的本体感觉维度 (论文中未明确给出)
    map_size = (26, 16)  # ANYmal-D的地图尺寸
    action_dim = 12  # ANYmal-D的动作维度

    # 创建模型
    model = Encoder(d, h, d_obs, map_size, action_dim)

    # 创建示例输入
    batch_size = 4
    map_scans = torch.randn(batch_size, map_size[0], map_size[1], 3)  # (4, 26, 16, 3)
    proprioception = torch.randn(batch_size, d_obs)  # (4, 48)

    # 前向传播
    actions = model(map_scans, proprioception)

    print(f"输入地图扫描形状: {map_scans.shape}")
    print(f"输入本体感觉形状: {proprioception.shape}")
    print(f"输出动作形状: {actions.shape}")