# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories


class CachedEncoderModelMixin:
    """Shared rollout feature-cache helpers for frozen encoder + MLP models."""

    encoder_feature_cache: bool
    encoder_obs_groups: list[str]

    @property
    def supports_feature_cache(self) -> bool:
        """Whether all encoders are frozen and their rollout features can be cached."""
        if not getattr(self, "encoder_feature_cache", True):
            return False
        return all(
            hasattr(encoder, "trainable") and not encoder.trainable  # type: ignore
            for encoder in self.cnns.values()  # type: ignore
        )

    def encode_features(self, obs: TensorDict) -> TensorDict:
        """Encode observation groups into detached, transition-aligned features."""
        return self._encode_features(obs, detach=True)

    def _encode_features(self, obs: TensorDict, detach: bool) -> TensorDict:
        """Encode cacheable observation groups, optionally detaching the resulting tensors."""
        features = {group: self.cnns[group](obs[group]) for group in self.encoder_obs_groups}  # type: ignore
        if detach:
            features = {group: feature.detach() for group, feature in features.items()}
        return TensorDict(features, batch_size=obs.batch_size, device=obs.device)

    def get_latent_from_features(self, obs: TensorDict, features: TensorDict) -> torch.Tensor:
        """Combine normalized 1D observations with precomputed encoder features."""
        latent_encoder = torch.cat([features[group] for group in self.encoder_obs_groups], dim=-1)
        if not self.obs_groups:  # type: ignore
            return latent_encoder
        return torch.cat([MLPModel.get_latent(self, obs), latent_encoder], dim=-1)

    def forward_with_features(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, TensorDict]:
        """Run the model and return the frozen encoder features used by the policy head."""
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
        """Run the policy head from precomputed encoder features."""
        if masks is not None and not self.is_recurrent:  # type: ignore
            obs = unpad_trajectories(obs, masks)
            features = unpad_trajectories(features, masks)
        mlp_output = self.mlp(self.get_latent_from_features(obs, features))  # type: ignore
        if self.distribution is not None:  # type: ignore
            if stochastic_output:
                self.distribution.update(mlp_output)  # type: ignore
                return self.distribution.sample()  # type: ignore
            return self.distribution.deterministic_output(mlp_output)  # type: ignore
        return mlp_output

    def get_representation_from_features(
        self,
        obs: TensorDict,
        features: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Return the hidden representation from precomputed encoder features."""
        del hidden_state
        if masks is not None and not self.is_recurrent:  # type: ignore
            obs = unpad_trajectories(obs, masks)
            features = unpad_trajectories(features, masks)
        return self.mlp.forward_features(self.get_latent_from_features(obs, features))  # type: ignore
