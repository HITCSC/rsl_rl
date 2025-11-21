import torch
import torch.nn as nn
import torch.nn.functional as F

class Velocity_Estimator(nn.Module):
    def __init__(self, history_len:int=3, d_obs:int=88,output_dim:int=3):
        """param input_dim: 输入维度
        :param hidden_dims: 隐藏层维度
        :param activation: 激活函数
        """
        super(Velocity_Estimator, self).__init__()
        self.input_dim = history_len * d_obs
        self.estimator = nn.Sequential(
                nn.Linear(d_obs, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

    def forward(self, props):
        """
        :param x: 输入张量，形状为 (B, H,input_dim)
        :return: 估计的速度张量，形状为 (B, 3)
        """
        B = props.shape[0]
        H = props.shape[1]
        actor_proprioception = props.view(B*H, *props.shape[2:]) # (B*H, d_obs)
        return self.estimator(actor_proprioception)