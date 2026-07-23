# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn

from rsl_rl.modules.amp_discriminator import AmpDiscriminator
from rsl_rl.modules.height_map_discriminator import HeightMapDiscriminator


class MultiDiscriminator(nn.Module):
    """Collection of style-specific AMP discriminators routed by terrain/style id."""

    def __init__(
        self,
        amp_obs_dim: int,
        style_configs: dict[int, dict],
        hidden_layer_sizes: list[int],
        device: str | torch.device,
        task_reward_lerp: float = 0.0,
        amp_obs_layout: dict | None = None,
        use_height_map_cnn: bool = True,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        cnn_output_dim: int = 64,
    ):
        super().__init__()

        if not style_configs:
            raise ValueError("style_configs must contain at least one style entry.")

        self.device = device
        self.amp_obs_dim = amp_obs_dim
        self.use_height_map_cnn = use_height_map_cnn
        self.style_ids = sorted(int(style_id) for style_id in style_configs.keys())
        self.task_reward_lerp = task_reward_lerp

        if use_height_map_cnn:
            if amp_obs_layout is None:
                raise ValueError("amp_obs_layout is required when use_height_map_cnn=True.")
            self.input_dim = None
        else:
            self.input_dim = amp_obs_dim * 2

        discriminators = {}
        for style_id in self.style_ids:
            cfg = style_configs[style_id]
            if use_height_map_cnn:
                discriminators[str(style_id)] = HeightMapDiscriminator(
                    amp_obs_dim,
                    amp_obs_layout,
                    cfg["amp_reward_coef"],
                    hidden_layer_sizes,
                    device,
                    task_reward_lerp,
                    cnn_channels=cnn_channels,
                    cnn_output_dim=cnn_output_dim,
                )
            else:
                discriminators[str(style_id)] = AmpDiscriminator(
                    amp_obs_dim * 2,
                    cfg["amp_reward_coef"],
                    hidden_layer_sizes,
                    device,
                    task_reward_lerp,
                )
        self.discriminators = nn.ModuleDict(discriminators)

        if use_height_map_cnn:
            self.input_dim = self.discriminators[str(self.style_ids[0])].input_dim

        self.print_architecture()

    def print_architecture(self) -> None:
        """Print discriminator network structure for each style."""
        print("-" * 80)
        print("Multi-AMP Discriminator Architecture")
        print(f"  use_height_map_cnn: {self.use_height_map_cnn}")
        print(f"  amp_obs_dim (per state): {self.amp_obs_dim}")
        print(f"  transition input dim: {self.input_dim}")
        print(f"  style_ids: {self.style_ids}")
        for style_id in self.style_ids:
            disc = self.get_discriminator(style_id)
            num_params = sum(param.numel() for param in disc.parameters())
            print(f"\n  [Style {style_id}] {type(disc).__name__}")
            print(f"    amp_reward_coef: {disc.amp_reward_coef}")
            print(f"    task_reward_lerp: {disc.task_reward_lerp}")
            print(f"    num_parameters: {num_params}")
            if self.use_height_map_cnn:
                print(f"    Encoder:\n{disc.encoder}")
            print(f"    Trunk:\n{disc.trunk}")
            print(f"    Head:\n{disc.amp_linear}")
        print("-" * 80)

    def get_discriminator(self, style_id: int) -> AmpDiscriminator | HeightMapDiscriminator:
        return self.discriminators[str(int(style_id))]

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        style_ids: torch.Tensor,
        normalizers: dict[int, object],
    ) -> torch.Tensor:
        """Predict per-environment AMP rewards using the discriminator for each style."""
        rewards = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        style_ids = style_ids.view(-1).long()

        for style_id in self.style_ids:
            mask = style_ids == style_id
            if not mask.any():
                continue
            disc = self.get_discriminator(style_id)
            normalizer = normalizers[style_id]
            style_rewards, _ = disc.predict_amp_reward(
                state[mask], next_state[mask], task_reward[mask], normalizer=normalizer
            )
            rewards[mask] = style_rewards

        return rewards

    def parameters_by_style(self, style_id: int):
        disc = self.get_discriminator(style_id)
        if self.use_height_map_cnn:
            trunk_params = chain(disc.encoder.parameters(), disc.trunk.parameters())
            head_params = disc.amp_linear.parameters()
        else:
            trunk_params = disc.trunk.parameters()
            head_params = disc.amp_linear.parameters()
        return trunk_params, head_params
