import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class CENet(nn.Module):
    def __init__(self, 
                 n_features: int = 18,  # o_t的特征数
                 H: int = 5,             # 时间窗口长度
                 latent_dim: int = 48,   # 潜在向量z_t的维度
                 beta: float = 1.0,      # β-VAE的β参数
                 v_dim: int = 3):        # 身体速度维度
        super().__init__()
        self.n_features = n_features
        self.H = H
        self.latent_dim = latent_dim
        self.beta = beta
        self.v_dim = v_dim

        # -------------------------- 共享编码器（Encoder）--------------------------
        self.encoder = nn.Sequential(
            nn.Linear(n_features * H, 64),  # 输入：n_features*H（展平时间序列）
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 48),
            nn.ELU()
        )
        # 输出高斯分布参数（均值μ和对数方差log_var）
        self.fc_mu = nn.Linear(48, latent_dim)
        self.fc_log_var = nn.Linear(48, latent_dim)

        # -------------------------- 解码器1：速度估计头（Velocity Estimator）--------------------------
        self.decoder_velocity = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, v_dim)  # 输出：x/y/z方向速度
        )

        # -------------------------- 解码器2：观测重建头（Observation Reconstructor）--------------------------
        self.decoder_reconstruct = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, n_features)  # 输出：重建的下一时刻观测o_{t+1}
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码过程：将时间序列观测o_t^H编码为高斯分布参数（μ, log_var）
        Input: x -> (batch_size, H, n_features) 时间序列观测
        Output: mu, log_var -> (batch_size, latent_dim) 高斯分布参数
        """
        batch_size = x.shape[0]
        # 展平输入：(batch_size, H*n_features)
        x_flat = x.view(batch_size, -1)
        # 共享编码器特征提取
        features = self.encoder(x_flat)
        # 输出高斯分布参数
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：从高斯分布中采样潜在向量z_t（可微分）"""
        std = torch.exp(0.5 * log_var)  # σ = exp(0.5*log_var)
        eps = torch.randn_like(std)     # ε ~ N(0,1)
        return mu + eps * std           # z = μ + ε·σ

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        解码过程：从潜在向量z_t解码得到速度估计和观测重建
        Input: z -> (batch_size, latent_dim) 潜在向量
        Output: (v_est, o_recon) -> 速度估计、观测重建结果
        """
        v_est = self.decoder_velocity(z)
        o_recon = self.decoder_reconstruct(z)
        return v_est, o_recon

    def forward(self, 
                x: torch.Tensor, 
                v_true: torch.Tensor = None, 
                o_next_true: torch.Tensor = None) -> dict:
        """
        前向传播：完整的编码-采样-解码-损失计算流程
        Inputs:
            x: (batch_size, H, n_features) 输入时间序列观测o_t^H
            v_true: (batch_size, v_dim) 真实身体速度（训练时传入，测试时可省略）
            o_next_true: (batch_size, n_features) 真实下一时刻观测（训练时传入，测试时可省略）
        Output: 字典包含模型输出和损失（训练时）
        """
        # 1. 编码：得到高斯分布参数
        mu, log_var = self.encode(x)
        # 2. 重参数化采样z_t
        z = self.reparameterize(mu, log_var)
        # 3. 解码：得到速度估计和观测重建
        v_est, o_recon = self.decode(z)

        # 4. 计算损失（仅训练时）
        output = {"v_est": v_est, "o_recon": o_recon, "z": z, "mu": mu, "log_var": log_var}
        if v_true is not None and o_next_true is not None:
            # 速度估计损失 L_est
            loss_est = F.mse_loss(v_est, v_true)
            # VAE损失：重建损失 + β*KL散度
            loss_recon = F.mse_loss(o_recon, o_next_true)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
            loss_vae = loss_recon + self.beta * kl_div
            # 总损失
            total_loss = loss_est + loss_vae
            # 加入损失到输出
            output.update({
                "loss_est": loss_est,
                "loss_vae": loss_vae,
                "total_loss": total_loss
            })

        return output

# -------------------------- 训练示例 --------------------------
if __name__ == "__main__":
    batch_size = 64
    n_features = 18  # 假设o_t的特征维度为18（可根据实际传感器数据调整）
    H = 5            # 时间窗口长度
    latent_dim = 48
    beta = 1.0
    learning_rate = 1e-3
    epochs = 100

    # 2. 模拟训练数据（实际使用时替换为真实传感器数据）
    # 输入：o_t^H -> (batch_size, H, n_features)
    x = torch.randn(batch_size, H, n_features)  # 模拟归一化后的观测数据
    # 真实速度：v_true -> (batch_size, 3)
    v_true = torch.randn(batch_size, 3) * 0.5  # 模拟身体速度（-1~1 m/s）
    # 真实下一时刻观测：o_next_true -> (batch_size, n_features)
    o_next_true = torch.randn(batch_size, n_features)

    # 3. 初始化模型、优化器
    model = CENet(n_features=n_features, H=H, latent_dim=latent_dim, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 前向传播计算损失
        outputs = model(x, v_true=v_true, o_next_true=o_next_true)
        total_loss = outputs["total_loss"]
        # 反向传播与优化
        total_loss.backward()
        optimizer.step()
        # 打印训练日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Total Loss: {total_loss.item():.4f}, "
                  f"Loss Est: {outputs['loss_est'].item():.4f}, "
                  f"Loss VAE: {outputs['loss_vae'].item():.4f}")

    # 5. 测试推理（仅输入观测，输出速度估计和潜在向量）
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(1, H, n_features)  # 单个测试样本
        test_outputs = model(test_x)
        v_est = test_outputs["v_est"]  # 速度估计结果
        z = test_outputs["z"]          # 环境潜在表示
        print(f"\nTest Velocity Estimation: {v_est.squeeze().numpy():.4f}")
        print(f"Latent Context Vector Shape: {z.shape}")