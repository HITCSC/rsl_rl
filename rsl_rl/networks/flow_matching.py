# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple, Union
import math 
import numpy as np
import torch
import torch.nn as nn

from rsl_rl.networks import MLP
from rsl_rl.networks.flow_matching_utils import * 

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        :param hidden_size: 最终输出的time embedding的维度
        :param frequency_embedding_size: sinusoidal embedding的维度
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class adaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm1(x) * (1+scale) + shift
        return x
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FlowMatchingModel(nn.Module):
    """Simple conditional flow-matching network.

    The model predicts an instantaneous velocity field given the current
    (intermediate) action sample, timestep and observation context.
    """

    def __init__(
        self,
        obs_embed_dim: int,  # obs cond embedding dimension
        action_dim: int,
        embedded_dim: int,  # FiLM/AdaLN embedding dimension
        mlp_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "swish",
        parameterization: str ="velocity",  # flow matching parameterization ("velocity", "data")
        solver_step_size=0.1,  # for ODE Solver 
        zero_action_input: bool = False,  # albation for action cond, 但是不知道什么用
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.t_embed = TimestepEmbedder(embedded_dim)
        # 参考 FPO PHC 的骨架：纯 MLP -> AdaLN -> SiLU -> 线性投影到动作维
        # self.actor_mlp = MLP(
        #     input_dim=obs_embed_dim + action_dim,
        #     output_dim=embedded_dim,  # hidden size before AdaLN
        #     hidden_dims=mlp_dims,
        #     activation=activation,
        #     last_activation=None,  # keep last layer linear; non-linearity is applied after AdaLN
        # )
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_embed_dim + action_dim, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, embedded_dim)),
            nn.LayerNorm(embedded_dim),
            nn.SiLU(),
        )
        # TODO : adaLN会破坏等变性,这里或许换成FiLM有可能更好
        self.actor_norm = adaLN(embedded_dim)
        self.post_adaln_non_linearity = nn.SiLU()
        self.proj = layer_init(nn.Linear(embedded_dim, action_dim), std=0.01) # after FiLM/AdaLN, project to action space
        # self.proj = nn.Linear(embedded_dim, action_dim)  
        
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].bias, 0)

        self.solver : ODESolver = ODESolver()
        self.path : AffineProbPath = CondOTProbPath()  # TODO : 这里可以换成VP-SDE看看
        self.parameterization = parameterization
        self.solver_step_size = solver_step_size
        self.zero_action_input = zero_action_input


    def forward(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the instantaneous velocity at x_t.

        Args:
            obs: [..., obs_embed_dim] observation context.
            x_t: [..., action_dim] current point along the flow.
            t: [..., ] timestep (typically in [0, 1]).
        """
        # Broadcast observation if needed to match x_t rank.
        while obs.dim() < x_t.dim():
            obs = obs.unsqueeze(1)  
        t_embed = self.t_embed(t)
        if t_embed.shape[:-1] != x_t.shape[:-1]:
            # Broadcast along sample dimension if necessary.
            t_embed = t_embed.expand(*x_t.shape[:-1], t_embed.shape[-1])
        h = torch.cat((obs, x_t), dim=-1)
        hidden = self.actor_mlp(h)
        hidden = self.actor_norm(hidden,t_embed)
        hidden = self.post_adaln_non_linearity(hidden)
        action = self.proj(hidden)
        return action

    def sample(
        self,
        obs: torch.Tensor,
        noise: torch.Tensor | None = None,
        integrator: str = "euler",
        get_path: bool = False,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Draw a sample by integrating the learned velocity field.

        Args:
            obs: [batch, obs_embed_dim] observation context.
            flow_steps: Number of integration steps for the reverse process.
            sigma: Standard deviation for optional diffusion noise at each step.
            noise: Optional initial noise of shape [batch, action_dim].
            integrator: ODE solver for reverse flow (euler|heun).
            get_path: Whether to return the full path of intermediate states.
        Returns:
            action | path : [batch, action_dim] final point at t=0.
              [batch, flow_steps, action_dim] intermediate states.
        """
        device = obs.device
        batch_shape = obs.shape[:-1]
        B = obs.shape[0]
        time_grid = torch.tensor([0.0, 1.0], device=device)
        if noise is None:
            # 这里可以尝试放宽一些限制
            x_0 = torch.randn(*batch_shape, self.action_dim, device=device)
        else:
            x_0 = noise

        def velocity_fn(x,t,obs_embed):
            # Remember the normed obs to use in the critic
            x_eff = torch.zeros_like(x) if self.zero_action_input else x
            t_batch = torch.ones([B], device=obs.device) * t
            # Concatenate noised action and normed obs
            # TODO : 这里要确定一下是否使用CFG训练
            hidden = self.forward(obs_embed,x_eff,t_batch)

            if self.parameterization == "velocity":
                velocity = hidden
            elif self.parameterization == "data":
                x1 = hidden
                velocity = self.path.target_to_velocity(x_1=x1, x_t=x, t=t_batch.unsqueeze(-1))
            return velocity

        x_1 = self.solver.sample(
            velocity_fn,
            time_grid = time_grid,
            x_init = x_0,
            method = integrator,
            return_intermediates=get_path,  # 只返回最终结果
            atol=1e-5,
            rtol=1e-5,
            step_size=self.solver_step_size,
            obs_embed=obs,
        )
        return x_1

    def compute_cfm_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """Conditional flow matching loss from FPO reference implementation.

        Args:
            obs: [batch, obs_embed_dim] normalized observations.
            action: [batch, action_dim] action at t=0 used as target, w.r.t x0.
            noise: [batch, action_dim] base noise samples, w.r.t x1.
            t: [batch,] timesteps associated with eps samples.
            mode: Either 'u' or 'u_but_supervise_as_eps'.
        TODO:
            add equaivarient Loss 
        """
        while obs.dim() < noise.dim():
            obs = obs.unsqueeze(1)
        path_sample = self.path.sample(t=t, x_0=noise, x_1=actions)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        # TODO : add CFG training 
        pred = self.forward(obs, x_t, t)
        if self.parameterization == "velocity":
            x1 = self.path.velocity_to_target(x_t=x_t,velocity=pred,t=t.unsqueeze(-1))
            # TODO : 这里sigma应该作为超参数, 防止ratio爆炸,源代码给出的0.05直接爆了，但是现在这个参数也需要配合log ratio上的clip才能比较正常
            log_probs = -((u_t - pred) ** 2) / (2 * 0.05 ** 2)
            loss = - log_probs.reshape(-1).mean()
        elif self.parameterization == "data":
            x1 = pred 
            log_probs = -((x1 - actions) ** 2) / (2 * 0.05 ** 2)
            loss = - log_probs.reshape(-1).mean()
        # (TODO) Mean bound loss
        # if self.training:
            # self.mean_bound_loss = self.bound_loss(x1)
        return loss, log_probs.mean(-1).reshape(-1)
