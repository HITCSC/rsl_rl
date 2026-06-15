# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories


class DefmModel(MLPModel):
    """DeFM-based neural model for mixed proprioceptive and metric-depth observations.

    Each depth observation is preprocessed using DeFM's metric-aware three-channel representation and encoded with a
    DeFM ViT-S/14 loaded through TorchHub. The resulting patch tokens are flattened and concatenated with any selected
    1D observations before being passed to the MLP head.
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
        defm_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the DeFM-based model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model.
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the 1D observations.
            distribution_cfg: Configuration dictionary for the output distribution.
            defm_cfg: Shared DeFM configuration or a configuration per depth observation group.
            cnns: DeFM encoders to share with another model. The name is kept consistent with :class:`CNNModel`.
        """
        self._get_obs_dim(obs, obs_groups, obs_set)

        if cnns is not None:
            if not all(isinstance(encoder, _DefmEncoder) for encoder in cnns.values()):
                raise TypeError("Shared DeFM encoders must be created by DefmModel.")
            shared_encoders = list(cnns.values())
            if len(shared_encoders) != len(self.obs_groups_2d):
                raise ValueError("Models sharing DeFM encoders must use the same number of depth observations.")
            dimensions_match = all(
                encoder.input_dim == self.obs_dims_2d[idx]  # type: ignore
                for idx, encoder in enumerate(shared_encoders)
            )
            if not dimensions_match:
                raise ValueError("Models sharing DeFM encoders must use depth observations with identical dimensions.")
            cnns = {group: shared_encoders[idx] for idx, group in enumerate(self.obs_groups_2d)}
            print("Sharing DeFM encoders between models, the DeFM configurations of the receiving model are ignored.")
        else:
            configs = self._resolve_defm_configs(defm_cfg)
            cnns = {
                group: _DefmEncoder(self.obs_dims_2d[idx], **configs[group])
                for idx, group in enumerate(self.obs_groups_2d)
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

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Combine normalized 1D observations with DeFM features."""
        return self.get_latent_from_features(obs, self._encode_features(obs, detach=False))

    @property
    def supports_feature_cache(self) -> bool:
        """Whether all DeFM encoders are frozen and their features can be cached safely."""
        return all(not encoder.trainable for encoder in self.cnns.values())  # type: ignore

    def encode_features(self, obs: TensorDict) -> TensorDict:
        """Encode depth observation groups into detached, transition-aligned features."""
        return self._encode_features(obs, detach=True)

    def _encode_features(self, obs: TensorDict, detach: bool) -> TensorDict:
        """Encode depth observation groups, optionally detaching the resulting tensors."""
        features = {group: self.cnns[group](obs[group]) for group in self.obs_groups_2d}
        if detach:
            features = {group: feature.detach() for group, feature in features.items()}
        return TensorDict(features, batch_size=obs.batch_size, device=obs.device)

    def get_latent_from_features(self, obs: TensorDict, features: TensorDict) -> torch.Tensor:
        """Combine normalized 1D observations with precomputed DeFM features."""
        latent_defm = torch.cat([features[group] for group in self.obs_groups_2d], dim=-1)
        if not self.obs_groups:
            return latent_defm
        return torch.cat([super().get_latent(obs), latent_defm], dim=-1)

    def forward_with_features(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, TensorDict]:
        """Run the model and return the frozen DeFM features used by the policy head."""
        features = self.encode_features(obs)
        output = self.forward_from_features(obs, features, masks, hidden_state, stochastic_output)
        return output, features

    def forward_from_features(
        self,
        obs: TensorDict,
        features: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the policy head from precomputed DeFM features."""
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
            features = unpad_trajectories(features, masks)
        mlp_output = self.mlp(self.get_latent_from_features(obs, features))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def train(self, mode: bool = True) -> DefmModel:
        """Set training mode while keeping frozen DeFM encoders in evaluation mode."""
        super().train(mode)
        for encoder in self.cnns.values():
            if not encoder.trainable:  # type: ignore
                encoder.eval()
        return self

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchDefmModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxDefmModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Separate 1D observations from single-channel metric-depth observations."""
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_groups_2d = []

        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            if len(shape) == 2:
                obs_groups_1d.append(obs_group)
                obs_dim_1d += shape[-1]
            elif len(shape) == 3:
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(tuple(shape[1:3]))
            elif len(shape) == 4 and shape[1] == 1:
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(tuple(shape[2:4]))
            else:
                raise ValueError(
                    f"Invalid observation shape for '{obs_group}': {shape}. "
                    "DefmModel expects 1D observations or metric depth shaped (B,H,W)/(B,1,H,W)."
                )

        if not obs_groups_2d:
            raise ValueError("No depth observations are provided. If this is intentional, use the MLP model instead.")

        self.obs_dims_2d = obs_dims_2d
        self.obs_groups_2d = obs_groups_2d
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.cnn_latent_dim

    def _resolve_defm_configs(self, defm_cfg: dict[str, dict] | dict[str, Any] | None) -> dict[str, dict]:
        """Expand a shared DeFM configuration into one configuration per depth group."""
        cfg = {} if defm_cfg is None else defm_cfg
        if any(isinstance(value, dict) for value in cfg.values()):
            if set(cfg) != set(self.obs_groups_2d) or not all(isinstance(cfg[group], dict) for group in cfg):
                raise ValueError("The number of DeFM configurations must match the number of depth observation groups.")
            return {group: dict(cfg[group]) for group in self.obs_groups_2d}  # type: ignore
        return {group: dict(cfg) for group in self.obs_groups_2d}


class _DefmEncoder(nn.Module):
    """TorchHub-loaded DeFM ViT-S/14 adapter producing flattened patch tokens."""

    patch_size: int = 14
    embed_dim: int = 384

    def __init__(
        self,
        input_dim: tuple[int, int],
        model_name: str = "defm_vit_s14",
        repo_or_dir: str = "leggedrobotics/defm:main",
        pretrained: bool = True,
        pretrained_path: str | None = None,
        trainable: bool = False,
        target_size: int | tuple[int, int] | list[int] | None = None,
        token_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        if model_name != "defm_vit_s14":
            raise ValueError("DefmModel currently supports only 'defm_vit_s14'.")

        self.input_dim = tuple(input_dim)
        self.target_dim = self._get_target_dim(self.input_dim, target_size)
        if token_feature_dim <= 0 or self.embed_dim % token_feature_dim != 0:
            raise ValueError(f"DeFM token_feature_dim must be a positive divisor of {self.embed_dim}.")
        self.token_feature_dim = token_feature_dim
        self.token_group_size = self.embed_dim // token_feature_dim
        num_patches = (self.target_dim[0] // self.patch_size) * (self.target_dim[1] // self.patch_size)
        self.output_dim = num_patches * self.token_feature_dim
        self.trainable = trainable

        hub_kwargs: dict[str, Any] = {"pretrained": pretrained}
        if pretrained_path is not None:
            hub_kwargs["pretrained_path"] = pretrained_path
        self.model = torch.hub.load(repo_or_dir, model_name, **hub_kwargs)

        if not trainable:
            self.model.requires_grad_(False)
            self.model.eval()

        self.register_buffer("mean", torch.tensor([0.248880, 0.495620, 0.492858]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.139357, 0.271314, 0.297177]).view(1, 3, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Preprocess metric depth and return grouped, flattened final-layer patch tokens."""
        x = self.preprocess(depth)
        patch_tokens = self.model.get_intermediate_layers(x, n=1)[0]
        grouped_tokens = patch_tokens.reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[1],
            self.token_feature_dim,
            self.token_group_size,
        ).mean(dim=-1)
        return grouped_tokens.flatten(start_dim=1)

    def preprocess(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply DeFM's batched metric-aware depth preprocessing using only Torch operations."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        elif depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("DeFM depth input must have shape (B,H,W) or (B,1,H,W).")
        if tuple(depth.shape[-2:]) != self.input_dim:
            raise ValueError(f"DeFM depth input must keep its initialized spatial size {self.input_dim}.")

        depth = torch.nan_to_num(depth.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        log_depth = torch.log1p(torch.clamp(depth, min=0.0, max=100.0))
        c1 = log_depth / math.log1p(100.0)
        c2 = torch.clamp(log_depth / math.log1p(9.0), min=0.0, max=1.0)

        flat = log_depth.flatten(start_dim=1)
        min_log = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_log = flat.max(dim=1).values.view(-1, 1, 1, 1)
        denominator = max_log - min_log
        c3 = (log_depth - min_log) / torch.where(denominator > 0.0, denominator, torch.ones_like(denominator))
        c3 = torch.where(denominator > 0.0, c3, torch.zeros_like(c3))

        x = torch.cat([c1, c2, c3], dim=1)
        if tuple(x.shape[-2:]) != self.target_dim:
            x = F.interpolate(x, size=self.target_dim, mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    @classmethod
    def _get_target_dim(
        cls, input_dim: tuple[int, int], target_size: int | tuple[int, int] | list[int] | None
    ) -> tuple[int, int]:
        """Resolve the target size and align it down to the ViT patch size."""
        if target_size is None:
            target_dim = input_dim
        elif isinstance(target_size, int):
            target_dim = (target_size, target_size)
        elif len(target_size) == 2:
            target_dim = (int(target_size[0]), int(target_size[1]))
        else:
            raise ValueError("DeFM target_size must be an int or a two-element sequence.")

        target_dim = tuple((dim // cls.patch_size) * cls.patch_size for dim in target_dim)
        if min(target_dim) < cls.patch_size:
            raise ValueError(f"DeFM target dimensions must be at least {cls.patch_size} pixels.")
        return target_dim  # type: ignore


class _TorchDefmModel(nn.Module):
    """Exportable DeFM model for JIT."""

    def __init__(self, model: DefmModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group]) for group in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs_1d: torch.Tensor, depth: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and depth inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_defm = torch.cat([encoder(depth[i]) for i, encoder in enumerate(self.cnns)], dim=-1)
        out = self.mlp(torch.cat([latent_1d, latent_defm], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for DeFM exports)."""
        pass


class _OnnxDefmModel(nn.Module):
    """Exportable DeFM model for ONNX."""

    def __init__(self, model: DefmModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[group]) for group in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *depth: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from separated 1D and depth inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_defm = torch.cat([encoder(depth[i]) for i, encoder in enumerate(self.cnns)], dim=-1)
        out = self.mlp(torch.cat([latent_1d, latent_defm], dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative raw metric-depth inputs for ONNX tracing."""
        depths = [torch.zeros(1, 1, h, w) for h, w in self.obs_dims_2d]
        return (torch.zeros(1, self.obs_dim_1d), *depths)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
