# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class HeightMapCNNEncoder(nn.Module):
    """Encode flattened height scan with CNN and concatenate proprioceptive features."""

    def __init__(
        self,
        height_scan_dim: int,
        height_scan_shape: tuple[int, int],
        proprio_dim: int,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        cnn_output_dim: int = 64,
    ):
        super().__init__()

        self.height_scan_dim = int(height_scan_dim)
        self.height_scan_shape = tuple(height_scan_shape)
        self.proprio_dim = int(proprio_dim)
        self.output_dim = int(cnn_output_dim) + self.proprio_dim

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, cnn_output_dim)
        self.layernorm = nn.LayerNorm(cnn_output_dim)

    def _encode_height(self, height_flat: torch.Tensor) -> torch.Tensor:
        batch_size = height_flat.shape[0]
        height_h, height_w = self.height_scan_shape
      
        height_map = height_flat.view(batch_size, 1, height_h, height_w)
        height_map = height_map - height_map.amin(dim=(2, 3), keepdim=True)

        features = self.conv(height_map)
        features = self.pool(features).view(batch_size, -1)
        features = self.layernorm(self.fc(features))
        return features

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        height_flat = obs[..., : self.height_scan_dim]
        proprio = obs[..., self.height_scan_dim :]
        return torch.cat([self._encode_height(height_flat), proprio], dim=-1)


class HeightMapDiscriminator(nn.Module):
    """AMP discriminator: height scan -> CNN, concat proprio -> MLP."""

    def __init__(
        self,
        amp_obs_dim: int,
        amp_obs_layout: dict,
        amp_reward_coef: float,
        hidden_layer_sizes: list[int],
        device: str | torch.device,
        task_reward_lerp: float = 0.0,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        cnn_output_dim: int = 64,
    ):
        super().__init__()

        self.device = device
        self.amp_obs_dim = int(amp_obs_dim)
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        self.encoder = HeightMapCNNEncoder(
            height_scan_dim=amp_obs_layout["height_scan_dim"],
            height_scan_shape=amp_obs_layout["height_scan_shape"],
            proprio_dim=amp_obs_layout["proprio_dim"],
            cnn_channels=cnn_channels,
            cnn_output_dim=cnn_output_dim,
        )
        trunk_input_dim = self.encoder.output_dim * 2

        amp_layers: list[nn.Module] = []
        curr_in_dim = trunk_input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1)
        self.input_dim = trunk_input_dim

    def _encode_transition(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder.encode(state), self.encoder.encode(next_state)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[..., : self.amp_obs_dim]
        next_state = x[..., self.amp_obs_dim :]
        features = self._encode_transition(state, next_state)
        return self.amp_linear(self.trunk(features))

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        expert_state = expert_state.detach().requires_grad_(True)
        expert_next_state = expert_next_state.detach().requires_grad_(True)

        features = self._encode_transition(expert_state, expert_next_state)
        disc = self.amp_linear(self.trunk(features))
        ones = torch.ones(disc.size(), device=disc.device)

        grad_state = autograd.grad(
            outputs=disc,
            inputs=expert_state,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_next_state = autograd.grad(
            outputs=disc,
            inputs=expert_next_state,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (
            (grad_state.norm(2, dim=1) - 0).pow(2).mean() + (grad_next_state.norm(2, dim=1) - 0).pow(2).mean()
        )
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            features = self._encode_transition(state, next_state)
            d = self.amp_linear(self.trunk(features))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(), d
