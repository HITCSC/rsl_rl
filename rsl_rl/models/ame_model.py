# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""AME (Attention-based Map Encoder) model ported from AME-Locomotion.

The model consumes a single flat 1-D observation group whose tail holds the
terrain elevation map ``[B, L*W*3]`` — per-ray 3-D coordinates of the height
scanner hits in the robot-yaw frame. A CNN downsamples the map into patch
features; the proprioceptive state is embedded as the *query* of a
Multi-Head-Attention over those patches (key/value = CNN features); the
attention output is concatenated with proprioception and fed to the policy MLP.

This mirrors ``AME_Locomotion``'s ``ActorCriticEncoder`` (``map_scan_dim``,
``mha_dim``, ``num_heads``, ``cnn_downsample``, ``attach_global``), adapted to
the rsl-rl model interface: the terrain encoder lives in ``self.cnns`` and can
be shared between actor and critic via the ``cnns`` parameter
(``algorithm.share_cnn_encoders``), while each branch keeps its own
``proprio_embedding`` and MLP head — exactly the sharing the source performs.
"""


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState, MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories


class AMEModel(MLPModel):
    """AME neural model for proprioception + terrain elevation-map observations."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        ame_cfg: dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the AME-based model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the policy MLP.
            activation: Activation function of the policy MLP.
            obs_normalization: Whether to normalize the proprioceptive block.
            distribution_cfg: Configuration dictionary for the output distribution.
            ame_cfg: AME terrain-encoder configuration (``map_scan_dim``,
                ``mha_dim``, ``num_heads``, ``cnn_downsample``, ``attach_global``).
            cnns: Shared terrain encoder to reuse (actor -> critic), consistent
                with :class:`CNNModel` / :class:`DefmModel`.
        """
        ame_cfg = {} if ame_cfg is None else dict(ame_cfg)
        self.map_scan_dim = tuple(ame_cfg.get("map_scan_dim", (33, 21, 3)))
        self.mha_dim = int(ame_cfg.get("mha_dim", 64))
        self.num_heads = int(ame_cfg.get("num_heads", 16))
        self.cnn_downsample = bool(ame_cfg.get("cnn_downsample", True))
        self.attach_global = bool(ame_cfg.get("attach_global", False))
        self.L, self.W, self.coord_dim = self.map_scan_dim
        self.map_scan_size = self.L * self.W * self.coord_dim
        self.cnn_output_dim = self.mha_dim

        # Resolve the flat 1-D observation dimension (map is embedded at the tail).
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.proprio_dim = self.obs_dim - self.map_scan_size
        if self.proprio_dim <= 0:
            raise ValueError(
                f"proprio_dim incorrect: obs dim {self.obs_dim} is not larger than "
                f"map_scan_size {self.map_scan_size}. Check that 'height_scan' is the "
                "last observation term and map_scan_dim matches the ray grid."
            )

        # Terrain encoder components. Built before ``super().__init__`` (which
        # calls ``_get_latent_dim``) but only registered as submodules after it,
        # mirroring ``DefmModel``'s encoder build.
        if cnns is not None:
            required = {"map_cnn", "mha"}
            if self.attach_global:
                required |= {"global_encoder", "query_projector"}
            if not required.issubset(set(cnns.keys())):
                raise ValueError(
                    f"Shared AME encoder missing components: got {list(cnns.keys())}, "
                    f"required {sorted(required)}."
                )
            map_cnn = cnns["map_cnn"]
            mha = cnns["mha"]
            global_encoder = cnns["global_encoder"] if "global_encoder" in cnns else None
            query_projector = cnns["query_projector"] if "query_projector" in cnns else None
        else:
            map_cnn, mha, global_encoder, query_projector = self._build_terrain_encoder()

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

        # Normalize the proprioceptive block only — the map is consumed raw.
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.proprio_dim)

        # Register the terrain-encoder submodules (after nn.Module init).
        self.map_cnn = map_cnn
        self.mha = mha
        self.global_encoder = global_encoder
        self.query_projector = query_projector
        self.proprio_embedding = nn.Linear(self.proprio_dim, self.mha_dim)

        # Expose the shared encoder so PPO can pass ``critic.cnns = actor.cnns``
        # when ``algorithm.share_cnn_encoders`` is enabled.
        shared = {"map_cnn": self.map_cnn, "mha": self.mha}
        if self.attach_global:
            shared["global_encoder"] = self.global_encoder
            shared["query_projector"] = self.query_projector
        self.cnns = nn.ModuleDict(shared)

    # -- terrain encoder ----------------------------------------------------

    def _build_terrain_encoder(self) -> tuple[nn.Module, nn.Module, nn.Module | None, nn.Module | None]:
        """Create the CNN + MHA terrain encoder (optionally with global context)."""
        if self.cnn_downsample:
            map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.cnn_output_dim),
            )
        else:
            map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(self.cnn_output_dim),
            )
        mha = nn.MultiheadAttention(embed_dim=self.mha_dim, num_heads=self.num_heads, batch_first=True)
        global_encoder = None
        query_projector = None
        if self.attach_global:
            global_encoder = MLP(self.mha_dim, self.mha_dim, [256, 128], "elu")
            query_projector = nn.Linear(self.mha_dim * 2, self.mha_dim)
        return map_cnn, mha, global_encoder, query_projector

    def _encode_terrain(self, proprio: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Encode the elevation-map tail of ``obs`` and attend it with ``proprio``."""
        # Stored order and reshape order differ, so swap W/L to keep spatial alignment.
        map_scan = obs[:, -self.map_scan_size:].reshape(-1, self.W, self.L, self.coord_dim)
        height_map = map_scan.permute(0, 3, 1, 2)  # [B, 3, W, L]
        cnn_features = self.map_cnn(height_map)
        if self.cnn_downsample:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
                -1, (self.L // 2 + 1) * (self.W // 2 + 1), self.cnn_output_dim
            )
        else:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
                -1, self.L * self.W, self.cnn_output_dim
            )

        proprio_embedding = self.proprio_embedding(proprio).unsqueeze(1)
        if self.attach_global:
            global_features = self.global_encoder(cnn_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_input = torch.cat([global_features_max, proprio_embedding.squeeze(1)], dim=-1)
            proprio_embedding = self.query_projector(query_input).unsqueeze(1)

        mha_output, _ = self.mha(
            query=proprio_embedding,
            key=cnn_features,
            value=cnn_features,
        )
        mha_output = mha_output.squeeze(1)

        encoded_obs = torch.cat([mha_output, proprio], dim=-1)
        if self.attach_global:
            encoded_obs = torch.cat([global_features_max, encoded_obs], dim=-1)
        return encoded_obs

    # -- model interface ----------------------------------------------------

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode the terrain map and concatenate the attention output with proprioception."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat = torch.cat(obs_list, dim=-1)
        proprio = flat[:, : self.proprio_dim]
        if self.obs_normalization:
            proprio = self.obs_normalizer(proprio)
        return self._encode_terrain(proprio, flat)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics on the proprioceptive block only."""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            flat = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(flat[:, : self.proprio_dim])  # type: ignore

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the policy MLP head."""
        extra = self.mha_dim if self.attach_global else 0
        return self.mha_dim + self.proprio_dim + extra

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchAmeModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxAmeModel(self, verbose)


class _TorchAmeModel(nn.Module):
    """Exportable AME model for JIT."""

    def __init__(self, model: AMEModel) -> None:
        super().__init__()
        self.map_scan_dim = model.map_scan_dim
        self.cnn_downsample = model.cnn_downsample
        self.attach_global = model.attach_global
        self.proprio_dim = model.proprio_dim
        self.map_scan_size = model.map_scan_size
        self.cnn_output_dim = model.cnn_output_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.map_cnn = copy.deepcopy(model.map_cnn)
        self.mha = copy.deepcopy(model.mha)
        # ``nn.Identity`` placeholders keep the wrapper TorchScript-scriptable
        # when ``attach_global=False`` (the branch is never taken at runtime).
        self.global_encoder = (
            copy.deepcopy(model.global_encoder) if model.global_encoder is not None else nn.Identity()
        )
        self.query_projector = (
            copy.deepcopy(model.query_projector) if model.query_projector is not None else nn.Identity()
        )
        self.proprio_embedding = copy.deepcopy(model.proprio_embedding)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        L, W, C = self.map_scan_dim
        map_scan = obs[:, -self.map_scan_size:].reshape(-1, W, L, C)
        height_map = map_scan.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(height_map)
        if self.cnn_downsample:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
                -1, (L // 2 + 1) * (W // 2 + 1), self.cnn_output_dim
            )
        else:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
                -1, L * W, self.cnn_output_dim
            )

        proprio = self.obs_normalizer(obs[:, : self.proprio_dim])
        proprio_embedding = self.proprio_embedding(proprio).unsqueeze(1)
        if self.attach_global:
            global_features = self.global_encoder(cnn_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_input = torch.cat([global_features_max, proprio_embedding.squeeze(1)], dim=-1)
            proprio_embedding = self.query_projector(query_input).unsqueeze(1)
            mha_output, _ = self.mha(query=proprio_embedding, key=cnn_features, value=cnn_features)
            mha_output = mha_output.squeeze(1)
            return torch.cat([global_features_max, mha_output, proprio], dim=-1)

        mha_output, _ = self.mha(query=proprio_embedding, key=cnn_features, value=cnn_features)
        mha_output = mha_output.squeeze(1)
        return torch.cat([mha_output, proprio], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on the flattened observation vector."""
        latent = self._encode(obs)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for AME exports)."""
        pass


class _OnnxAmeModel(_TorchAmeModel):
    """Exportable AME model for ONNX."""

    def __init__(self, model: AMEModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
