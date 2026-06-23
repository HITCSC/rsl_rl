# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.cached_encoder_model import CachedEncoderModelMixin
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class DepthAnythingV2Model(CachedEncoderModelMixin, MLPModel):
    """Depth-Anything-V2 RGB encoder with an MLP policy head.

    RGB observation groups are encoded with the frozen DA2 DINOv2 backbone. The
    selected patch tokens are reduced and concatenated with 1D observations
    before entering the MLP head.
    """

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
        da2_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        encoder_feature_cache: bool = True,
    ) -> None:
        self._get_obs_dim(obs, obs_groups, obs_set)
        self.encoder_feature_cache = encoder_feature_cache

        if cnns is not None:
            if not all(isinstance(encoder, _DepthAnythingV2Encoder) for encoder in cnns.values()):
                raise TypeError("Shared DA2 encoders must be created by DepthAnythingV2Model.")
            shared_encoders = list(cnns.values())
            if len(shared_encoders) != len(self.obs_groups_rgb):
                raise ValueError("Models sharing DA2 encoders must use the same number of RGB observations.")
            dimensions_match = all(
                encoder.input_dim == self.obs_dims_rgb[idx]  # type: ignore
                for idx, encoder in enumerate(shared_encoders)
            )
            if not dimensions_match:
                raise ValueError("Models sharing DA2 encoders must use RGB observations with identical dimensions.")
            cnns = {group: shared_encoders[idx] for idx, group in enumerate(self.obs_groups_rgb)}
            print("Sharing DA2 encoders between models, the DA2 configurations of the receiving model are ignored.")
        else:
            configs = self._resolve_da2_configs(da2_cfg)
            cnns = {
                group: _DepthAnythingV2Encoder(self.obs_dims_rgb[idx], **configs[group])
                for idx, group in enumerate(self.obs_groups_rgb)
            }

        self.cnn_latent_dim = sum(int(encoder.output_dim) for encoder in cnns.values())  # type: ignore

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

        self.cnns = cnns if isinstance(cnns, nn.ModuleDict) else nn.ModuleDict(cnns)
        self.encoder_obs_groups = self.obs_groups_rgb

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Combine normalized 1D observations with DA2 RGB features."""
        return self.get_latent_from_features(obs, self._encode_features(obs, detach=False))

    def train(self, mode: bool = True) -> DepthAnythingV2Model:
        """Set training mode while keeping frozen DA2 encoders in evaluation mode."""
        super().train(mode)
        for encoder in self.cnns.values():
            if not encoder.trainable:  # type: ignore
                encoder.eval()
        return self

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchDepthAnythingV2Model(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxDepthAnythingV2Model(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Separate 1D observations from three-channel RGB observations."""
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_rgb = []
        obs_groups_rgb = []

        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            if len(shape) == 2:
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            elif len(shape) == 4 and shape[1] == 3:
                obs_groups_rgb.append(obs_group)
                obs_dims_rgb.append(tuple(shape[2:4]))
            else:
                raise ValueError(
                    f"Invalid observation shape for '{obs_group}': {shape}. "
                    "DepthAnythingV2Model expects 1D observations or RGB shaped (B,3,H,W)."
                )

        if not obs_groups_rgb:
            raise ValueError("No RGB observations are provided. If this is intentional, use the MLP model instead.")

        self.obs_dims_rgb = obs_dims_rgb
        self.obs_groups_rgb = obs_groups_rgb
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.cnn_latent_dim

    def _resolve_da2_configs(self, da2_cfg: dict[str, dict] | dict[str, Any] | None) -> dict[str, dict]:
        """Expand a shared DA2 configuration into one configuration per RGB group."""
        cfg = {} if da2_cfg is None else da2_cfg
        if any(isinstance(value, dict) for value in cfg.values()):
            if set(cfg) != set(self.obs_groups_rgb) or not all(isinstance(cfg[group], dict) for group in cfg):
                raise ValueError("The number of DA2 configurations must match the number of RGB observation groups.")
            return {group: dict(cfg[group]) for group in self.obs_groups_rgb}  # type: ignore
        return {group: dict(cfg) for group in self.obs_groups_rgb}


class _DepthAnythingV2Encoder(nn.Module):
    """Depth-Anything-V2 DINOv2 encoder adapter producing flattened patch tokens."""

    patch_size: int = 14
    model_configs: dict[str, dict[str, Any]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        input_dim: tuple[int, int],
        encoder: str = "vits",
        repo_path: str | None = None,
        checkpoint_path: str | None = None,
        trainable: bool = False,
        target_size: int | tuple[int, int] | list[int] | None = None,
        token_feature_dim: int = 32,
        include_cls_token: bool = False,
    ) -> None:
        super().__init__()
        if encoder not in self.model_configs:
            raise ValueError(f"Unsupported DA2 encoder '{encoder}'. Valid encoders: {list(self.model_configs)}")

        self.input_dim = tuple(input_dim)
        self.target_dim = self._get_target_dim(self.input_dim, target_size)
        self.trainable = trainable
        self.include_cls_token = include_cls_token

        DepthAnythingV2 = self._resolve_da2_class(repo_path)
        self.model = DepthAnythingV2(**self.model_configs[encoder])
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.encoder = encoder
        self.layer_indices = self.model.intermediate_layer_idx[encoder]
        self.embed_dim = self.model.pretrained.embed_dim
        if token_feature_dim <= 0 or self.embed_dim % token_feature_dim != 0:
            raise ValueError(f"DA2 token_feature_dim must be a positive divisor of {self.embed_dim}.")
        self.token_feature_dim = token_feature_dim
        self.token_group_size = self.embed_dim // token_feature_dim
        num_patches = (self.target_dim[0] // self.patch_size) * (self.target_dim[1] // self.patch_size)
        num_tokens = num_patches + int(include_cls_token)
        self.output_dim = num_tokens * token_feature_dim

        if not trainable:
            self.model.requires_grad_(False)
            self.model.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB and return grouped, flattened DA2 backbone tokens."""
        x = self.preprocess(rgb)
        features = self.model.pretrained.get_intermediate_layers(
            x,
            self.layer_indices,
            return_class_token=True,
        )
        patch_tokens, cls_token = features[-1]
        if self.include_cls_token:
            patch_tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        grouped_tokens = patch_tokens.reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[1],
            self.token_feature_dim,
            self.token_group_size,
        ).mean(dim=-1)
        return grouped_tokens.flatten(start_dim=1)

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """Prepare RGB observations for the DA2 DINOv2 backbone."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("DA2 RGB input must have shape (B,3,H,W).")
        if tuple(rgb.shape[-2:]) != self.input_dim:
            raise ValueError(f"DA2 RGB input must keep its initialized spatial size {self.input_dim}.")

        rgb = torch.nan_to_num(rgb.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if rgb.max() > 2.0:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if tuple(rgb.shape[-2:]) != self.target_dim:
            rgb = F.interpolate(rgb, size=self.target_dim, mode="bilinear", align_corners=False, antialias=True)
        return (rgb - self.mean) / self.std

    @staticmethod
    def _resolve_da2_class(repo_path: str | None):
        if repo_path:
            path = str(Path(repo_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                "DepthAnythingV2Model requires the Depth Anything V2 package. "
                "Install rsl-rl-lib with the rgb extra, e.g. `rsl-rl-lib[rgb]`, "
                "or clone https://github.com/DepthAnything/Depth-Anything-V2 "
                "and pass repo_path in da2_cfg."
            ) from e
        return DepthAnythingV2

    @classmethod
    def _get_target_dim(
        cls, input_dim: tuple[int, int], target_size: int | tuple[int, int] | list[int] | None
    ) -> tuple[int, int]:
        """Resolve the target size and align it down to the DA2 patch size."""
        if target_size is None:
            target_dim = input_dim
        elif isinstance(target_size, int):
            target_dim = (target_size, target_size)
        elif len(target_size) == 2:
            target_dim = (int(target_size[0]), int(target_size[1]))
        else:
            raise ValueError("DA2 target_size must be an int or a two-element sequence.")

        target_dim = tuple((dim // cls.patch_size) * cls.patch_size for dim in target_dim)
        if min(target_dim) < cls.patch_size:
            raise ValueError(f"DA2 target dimensions must be at least {cls.patch_size} pixels.")
        return target_dim  # type: ignore


class _TorchDepthAnythingV2Model(nn.Module):
    """Exportable DA2 model for JIT."""

    def __init__(self, model: DepthAnythingV2Model) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group]) for group in model.obs_groups_rgb])
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs_1d: torch.Tensor, rgb: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and RGB inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_rgb = torch.cat([encoder(rgb[i]) for i, encoder in enumerate(self.cnns)], dim=-1)
        out = self.mlp(torch.cat([latent_1d, latent_rgb], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for DA2 exports)."""
        pass


class _OnnxDepthAnythingV2Model(nn.Module):
    """Exportable DA2 model for ONNX."""

    def __init__(self, model: DepthAnythingV2Model, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group]) for group in model.obs_groups_rgb])
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.obs_groups_rgb = model.obs_groups_rgb
        self.obs_dims_rgb = model.obs_dims_rgb
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *rgb: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from separated 1D and RGB inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_rgb = torch.cat([encoder(rgb[i]) for i, encoder in enumerate(self.cnns)], dim=-1)
        out = self.mlp(torch.cat([latent_1d, latent_rgb], dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        rgb = [torch.zeros(1, 3, h, w) for h, w in self.obs_dims_rgb]
        return (torch.zeros(1, self.obs_dim_1d), *rgb)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_rgb]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
