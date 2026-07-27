# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SSR visual-proprioceptive mixture-of-experts policy."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class _SSRDepthEncoder(nn.Sequential):
    def __init__(self, activation: str = "elu") -> None:
        activation_class = nn.ELU if activation == "elu" else nn.ReLU
        super().__init__(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            activation_class(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            activation_class(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            activation_class(),
            nn.Flatten(),
            nn.Linear(128, 128),
            activation_class(),
        )


class _SSRMoEActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str) -> None:
        super().__init__()
        self.gate = MLP(input_dim, 5, (128,), activation)
        self.experts = nn.ModuleList(
            [MLP(input_dim, output_dim, (1024, 512, 128), activation) for _ in range(5)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.sum(weights.unsqueeze(-1) * outputs, dim=-2)


class SSRModel(MLPModel):
    """Kuavo-compatible implementation of the SSR actor architecture.

    A five-frame proprioceptive sequence is encoded frame-wise and reduced by
    a GRU. Current 42x42 depth is encoded by the paper's three-layer CNN. A
    fusion MLP feeds three 16-D latent heads, while a separate estimator
    predicts base velocity. Current proprioception, estimated velocity and the
    48-D latent are consumed by a five-expert MoE actor.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        ssr_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the paper-sized encoder, estimators, decoders, and MoE."""
        del hidden_dims
        nn.Module.__init__(self)
        cfg = dict(ssr_cfg or {})
        self.history_length = int(cfg.get("history_length", 5))
        self.proprio_group = str(cfg.get("proprio_group", "actor"))
        self.depth_group = str(cfg.get("depth_group", "actor_depth"))
        self.foot_height_group = str(cfg.get("foot_height_group", "ssr_foot_heights"))
        self.body_height_group = str(cfg.get("body_height_group", "ssr_body_heights"))
        self.velocity_group = str(cfg.get("velocity_group", "ssr_base_velocity"))

        self.obs_groups = list(obs_groups[obs_set])
        if self.proprio_group not in self.obs_groups or self.depth_group not in self.obs_groups:
            raise ValueError(
                "SSRModel requires proprioception and depth groups in the actor observation set."
            )
        proprio_dim = int(obs[self.proprio_group].shape[-1])
        if proprio_dim % self.history_length:
            raise ValueError(
                f"Proprioception width {proprio_dim} is not divisible by history_length={self.history_length}."
            )
        self.frame_dim = proprio_dim // self.history_length
        if tuple(obs[self.depth_group].shape[-3:]) != (1, 42, 42):
            raise ValueError(
                f"SSRModel expects depth [B,1,42,42], got {tuple(obs[self.depth_group].shape)}."
            )
        self.obs_dim = proprio_dim
        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(proprio_dim) if obs_normalization else nn.Identity()
        )

        self.proprio_encoder = MLP(self.frame_dim, 128, (512, 256, 128), activation)
        self.depth_encoder = _SSRDepthEncoder(activation)
        self.temporal_encoder = nn.GRU(128, 256, num_layers=1, batch_first=True)
        self.fusion_encoder = MLP(384, 48, (512, 256, 128), activation)
        self.foot_latent_head = nn.Linear(48, 16)
        self.body_latent_head = nn.Linear(48, 16)
        self.motion_mu_head = nn.Linear(48, 16)
        self.motion_logvar_head = nn.Linear(48, 16)
        self.velocity_estimator = MLP(proprio_dim, 3, (512, 256, 128), activation)

        distribution_cfg = dict(distribution_cfg or {})
        dist_class: type[Distribution] = resolve_callable(
            distribution_cfg.pop("class_name", "GaussianDistribution")
        )
        self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
        actor_input_dim = self.frame_dim + 3 + 48
        self.mlp = _SSRMoEActor(actor_input_dim, self.distribution.input_dim, activation)

        self.foot_height_decoder = MLP(
            16, int(obs[self.foot_height_group].shape[-1]), (128, 128), activation
        )
        self.body_height_decoder = MLP(
            16, int(obs[self.body_height_group].shape[-1]), (128, 128), activation
        )
        self.next_proprio_decoder = MLP(16, self.frame_dim, (256, 128), activation)
        self.distribution.init_mlp_weights(self.mlp)

    def _encode(
        self, obs: TensorDict, sample_motion: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        proprio = self.obs_normalizer(obs[self.proprio_group])
        sequence = proprio.reshape(*proprio.shape[:-1], self.history_length, self.frame_dim)
        encoded_prop = self.proprio_encoder(sequence)
        temporal, _ = self.temporal_encoder(encoded_prop)
        temporal = temporal[..., -1, :]
        depth = self.depth_encoder(obs[self.depth_group])
        fusion = self.fusion_encoder(torch.cat((temporal, depth), dim=-1))
        z_foot = self.foot_latent_head(fusion)
        z_body = self.body_latent_head(fusion)
        motion_mu = self.motion_mu_head(fusion)
        motion_logvar = self.motion_logvar_head(fusion).clamp(-10.0, 10.0)
        if sample_motion:
            z_motion = motion_mu + torch.exp(0.5 * motion_logvar) * torch.randn_like(motion_mu)
        else:
            z_motion = motion_mu
        velocity = self.velocity_estimator(proprio)
        latent = torch.cat((z_foot, z_body, z_motion), dim=-1)
        return (
            proprio,
            latent,
            velocity,
            motion_mu,
            motion_logvar,
            z_foot,
            z_body,
            z_motion,
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the MoE input from current proprioception and learned context."""
        del masks, hidden_state
        proprio, latent, velocity, _, _, _, _, _ = self._encode(obs)
        current = proprio[..., -self.frame_dim :]
        return torch.cat((current, velocity, latent), dim=-1)

    def auxiliary_losses(
        self, obs: TensorDict, extra: TensorDict | None
    ) -> dict[str, tuple[torch.Tensor, float]]:
        """Return SSR Appendix A.5 hybrid prediction losses and coefficients."""
        (
            proprio,
            _,
            velocity,
            motion_mu,
            motion_logvar,
            z_foot,
            z_body,
            z_motion,
        ) = self._encode(obs, sample_motion=True)
        sequence = proprio.reshape(*proprio.shape[:-1], self.history_length, self.frame_dim)
        next_target = (
            extra["ssr_next_proprio"] if extra is not None and "ssr_next_proprio" in extra
            else sequence[..., -1, :].detach()
        )
        losses = {
            "ssr_body_height": (
                nn.functional.mse_loss(self.body_height_decoder(z_body), obs[self.body_height_group]),
                2.0,
            ),
            "ssr_foot_height": (
                nn.functional.mse_loss(self.foot_height_decoder(z_foot), obs[self.foot_height_group]),
                1.0,
            ),
            "ssr_next_proprio": (
                nn.functional.mse_loss(self.next_proprio_decoder(z_motion), next_target),
                5.0,
            ),
            "ssr_kl": (
                -0.5 * torch.mean(1.0 + motion_logvar - motion_mu.square() - motion_logvar.exp()),
                1.0,
            ),
            "ssr_velocity": (
                nn.functional.mse_loss(velocity, obs[self.velocity_group]),
                2.0,
            ),
        }
        return losses

    def update_normalization(self, obs: TensorDict) -> None:
        """Update proprioceptive observation statistics."""
        if self.obs_normalization:
            self.obs_normalizer.update(obs[self.proprio_group])

    def as_jit(self) -> nn.Module:
        """Return a deterministic two-input TorchScript export wrapper."""
        return _TorchSSRModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a deterministic two-input ONNX export wrapper."""
        del verbose
        return _TorchSSRModel(self)


class _TorchSSRModel(nn.Module):
    def __init__(self, model: SSRModel) -> None:
        super().__init__()
        self.history_length = model.history_length
        self.frame_dim = model.frame_dim
        self.proprio_dim = model.obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.temporal_encoder = copy.deepcopy(model.temporal_encoder)
        self.fusion_encoder = copy.deepcopy(model.fusion_encoder)
        self.foot_latent_head = copy.deepcopy(model.foot_latent_head)
        self.body_latent_head = copy.deepcopy(model.body_latent_head)
        self.motion_mu_head = copy.deepcopy(model.motion_mu_head)
        self.velocity_estimator = copy.deepcopy(model.velocity_estimator)
        self.moe = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        proprio = self.obs_normalizer(proprio)
        sequence = proprio.reshape(-1, self.history_length, self.frame_dim)
        temporal, _ = self.temporal_encoder(self.proprio_encoder(sequence))
        fusion = self.fusion_encoder(
            torch.cat((temporal[:, -1], self.depth_encoder(depth)), dim=-1)
        )
        latent = torch.cat(
            (
                self.foot_latent_head(fusion),
                self.body_latent_head(fusion),
                self.motion_mu_head(fusion),
            ),
            dim=-1,
        )
        velocity = self.velocity_estimator(proprio)
        output = self.moe(torch.cat((proprio[:, -self.frame_dim :], velocity, latent), dim=-1))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.proprio_dim),
            torch.zeros(1, 1, 36, 36),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprioception", "depth"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
