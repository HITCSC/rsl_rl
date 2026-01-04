import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO 不加入command，只用关节及action
class Velocity_Estimator(nn.Module):
    def __init__(self, history_len:int=3, d_obs:int=84,output_dim:int=3):
        """param input_dim: 输入维度
        :param hidden_dims: 隐藏层维度
        :param activation: 激活函数
        """
        super(Velocity_Estimator, self).__init__()
        self.input_dim = history_len * d_obs
        self.estimator = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

    def forward(self, props):
        """
        如何返回[b,3],把
        :param x: 输入张量，形状为 (B, H,input_dim)
        :return: 估计的速度张量，形状为 (B, 3)
        """
        B = props.shape[0]
        H = props.shape[1]
        # 返回 [b,1,3]
        actor_proprioception = props.reshape(B, H*props.shape[2])  # (B, H*input_dim)
        return self.estimator(actor_proprioception)
    
class Critic_Estimator(nn.Module):
    def __init__(
        self,
        history_len: int = 3,
        d_obs: int = 84,
        output_dim: int = 8,
        force_dim: int = 6,
        height_dim: int = 2,
    ):
        """param input_dim: 输入维度
        :param hidden_dims: 隐藏层维度
        :param activation: 激活函数
        """
        super(Critic_Estimator, self).__init__()
        self.input_dim = history_len * d_obs
        self.force_dim = force_dim
        self.height_dim = height_dim
        expected = self.force_dim + self.height_dim
        if output_dim != expected:
            raise ValueError(
                f"Critic_Estimator output_dim must equal force_dim+height_dim ({expected}), got {output_dim}."
            )

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # two heads
        self.force_head = nn.Linear(128, self.force_dim)
        self.height_head = nn.Linear(128, self.height_dim)

    def forward(self, props, return_dict: bool = False):
        """
        如何返回[b,1],把
        :param x: 输入张量，形状为 (B, H,input_dim)
        :return: 估计的速度张量，形状为 (B, 1)
        """
        B = props.shape[0]
        H = props.shape[1]
        # 返回 [b,1,1]
        actor_proprioception = props.reshape(B, H*props.shape[2])  # (B, H*input_dim)
        feat = self.trunk(actor_proprioception)
        force = self.force_head(feat)
        height = self.height_head(feat)
        out = torch.cat([force, height], dim=-1)  # [B, 8] (force first, then height)
        if return_dict:
            return {"force": force, "height": height, "out": out}
        return out