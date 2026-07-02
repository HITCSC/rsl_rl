# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height-scan cross-attention model."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation


class HeightScanAttentionModel(MLPModel):
    """MLP model with proprioceptive-query attention over height-scan tokens."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        attention_cfg: dict[str, Any] | None = None,
    ) -> None:
        attention_cfg = dict(attention_cfg or {})
        self.token_dim = int(attention_cfg.pop("token_dim", 32))
        self.d_model = int(attention_cfg.pop("d_model", 64))
        self.n_heads = int(attention_cfg.pop("n_heads", 4))
        self.n_queries = int(attention_cfg.pop("n_queries", 4))
        self.attention_output_dim = int(attention_cfg.pop("output_dim", 128))
        self.attn_dropout = float(attention_cfg.pop("dropout", 0.0))
        if attention_cfg:
            raise ValueError(f"Unsupported HeightScanAttentionModel options: {sorted(attention_cfg)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        activation_fn = resolve_nn_activation(activation)
        self.token_projections = nn.ModuleDict(
            {
                group: nn.Linear(channels, self.token_dim)
                for group, channels in zip(self.obs_groups_height_scan, self.obs_channels_height_scan)
            }
        )
        self.position_embeddings = nn.ParameterDict(
            {
                group: nn.Parameter(torch.zeros(num_tokens, self.token_dim))
                for group, num_tokens in zip(self.obs_groups_height_scan, self.obs_tokens_height_scan)
            }
        )
        self.q_proj = nn.Linear(self.obs_dim, self.n_queries * self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            kdim=self.token_dim,
            vdim=self.token_dim,
            dropout=self.attn_dropout,
            batch_first=True,
        )
        self.attention_output = nn.Sequential(
            nn.Linear(self.n_queries * self.d_model, self.attention_output_dim),
            activation_fn,
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build latent from normalized 1D observations and attended height-scan tokens."""
        latent_1d = super().get_latent(obs)
        tokens = torch.cat([self._height_scan_tokens(obs, group) for group in self.obs_groups_height_scan], dim=1)
        queries = self.q_proj(latent_1d).view(latent_1d.shape[0], self.n_queries, self.d_model)
        attn_out, _ = self.cross_attn(queries, tokens, tokens)
        attn_latent = self.attention_output(attn_out.flatten(start_dim=1))
        return torch.cat([latent_1d, attn_latent], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization only from the 1D proprioceptive observations."""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchHeightScanAttentionModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxHeightScanAttentionModel(self, verbose)

    def _height_scan_tokens(self, obs: TensorDict, group: str) -> torch.Tensor:
        height_scan = obs[group]
        if len(height_scan.shape) != 4:
            raise ValueError(f"Height scan observation '{group}' must be [B, C, H, W], got {height_scan.shape}.")
        tokens = height_scan.flatten(start_dim=2).transpose(1, 2)
        tokens = self.token_projections[group](tokens)
        return tokens + self.position_embeddings[group].unsqueeze(0)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active groups and split 1D proprioception from height-scan images."""
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_groups_height_scan = []
        obs_channels_height_scan = []
        obs_tokens_height_scan = []
        obs_dims_height_scan = []

        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            if len(shape) == 2:
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            elif len(shape) == 4:
                obs_groups_height_scan.append(obs_group)
                obs_channels_height_scan.append(shape[1])
                obs_tokens_height_scan.append(shape[2] * shape[3])
                obs_dims_height_scan.append(shape[1:4])
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {shape}")

        if not obs_groups_1d:
            raise ValueError("HeightScanAttentionModel requires at least one 1D observation group for queries.")
        if not obs_groups_height_scan:
            raise ValueError("HeightScanAttentionModel requires at least one 2D height-scan observation group.")

        self.obs_groups_height_scan = obs_groups_height_scan
        self.obs_channels_height_scan = obs_channels_height_scan
        self.obs_tokens_height_scan = obs_tokens_height_scan
        self.obs_dims_height_scan = obs_dims_height_scan
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.attention_output_dim


class _TorchHeightScanAttentionModel(nn.Module):
    """Exportable height-scan attention model for JIT."""

    def __init__(self, model: HeightScanAttentionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.token_projections = nn.ModuleList(
            [copy.deepcopy(model.token_projections[group]) for group in model.obs_groups_height_scan]
        )
        self.position_embeddings = nn.ParameterList(
            [nn.Parameter(model.position_embeddings[group].detach().clone()) for group in model.obs_groups_height_scan]
        )
        self.q_proj = copy.deepcopy(model.q_proj)
        self.cross_attn = copy.deepcopy(model.cross_attn)
        self.attention_output = copy.deepcopy(model.attention_output)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.n_queries = model.n_queries
        self.d_model = model.d_model

    def forward(self, obs_1d: torch.Tensor, height_scans: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and height-scan inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        token_list = []
        for idx, height_scan in enumerate(height_scans):
            tokens = height_scan.flatten(start_dim=2).transpose(1, 2)
            tokens = self.token_projections[idx](tokens)
            token_list.append(tokens + self.position_embeddings[idx].unsqueeze(0))
        tokens = torch.cat(token_list, dim=1)
        queries = self.q_proj(latent_1d).view(latent_1d.shape[0], self.n_queries, self.d_model)
        attn_out, _ = self.cross_attn(queries, tokens, tokens)
        attn_latent = self.attention_output(attn_out.flatten(start_dim=1))
        out = self.mlp(torch.cat([latent_1d, attn_latent], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for attention exports)."""
        pass


class _OnnxHeightScanAttentionModel(nn.Module):
    """Exportable height-scan attention model for ONNX."""

    def __init__(self, model: HeightScanAttentionModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.torch_model = _TorchHeightScanAttentionModel(model)
        self.obs_dim_1d = model.obs_dim
        self.obs_dims_height_scan = model.obs_dims_height_scan
        self.obs_groups_height_scan = model.obs_groups_height_scan

    def forward(self, obs_1d: torch.Tensor, *height_scans: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        return self.torch_model(obs_1d, list(height_scans))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_height_scans = [torch.zeros(1, *shape) for shape in self.obs_dims_height_scan]
        return (dummy_1d, *dummy_height_scans)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_height_scan]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
