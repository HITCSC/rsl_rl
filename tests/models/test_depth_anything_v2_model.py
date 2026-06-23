# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Depth Anything V2 model using a lightweight mock encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import DepthAnythingV2Model
from rsl_rl.models.depth_anything_v2_model import _DepthAnythingV2Encoder
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 4
NUM_ACTIONS = 3


class _MockDa2Backbone(nn.Module):
    """Return deterministic patch tokens without loading DA2 weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 384
        self.forward_calls = 0

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        layer_indices: list[int],
        return_class_token: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        self.forward_calls += 1
        tokens = torch.arange(36 * self.embed_dim, dtype=x.dtype, device=x.device).reshape(1, 36, self.embed_dim)
        cls_token = torch.zeros(x.shape[0], self.embed_dim, dtype=x.dtype, device=x.device)
        return ((tokens.expand(x.shape[0], -1, -1), cls_token),)


class _MockDepthAnythingV2(nn.Module):
    """Minimal DA2 facade required by _DepthAnythingV2Encoder."""

    instances: list[_MockDepthAnythingV2] = []
    intermediate_layer_idx = {"vits": [2, 5, 8, 11]}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.pretrained = _MockDa2Backbone()
        self.instances.append(self)


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "actor": torch.randn(NUM_ENVS, 5),
            "critic": torch.randn(NUM_ENVS, 7),
            "actor_rgb": torch.rand(NUM_ENVS, 3, 84, 84),
            "critic_rgb": torch.rand(NUM_ENVS, 3, 84, 84),
        },
        batch_size=[NUM_ENVS],
    )


def _make_models(
    monkeypatch,
    *,
    trainable: bool = False,
    encoder_feature_cache: bool = True,
) -> tuple[DepthAnythingV2Model, DepthAnythingV2Model, TensorDict, _MockDa2Backbone]:
    _MockDepthAnythingV2.instances = []
    monkeypatch.setattr(_DepthAnythingV2Encoder, "_resolve_da2_class", staticmethod(lambda repo_path: _MockDepthAnythingV2))
    obs = _make_obs()
    obs_groups = {
        "actor": ["actor", "actor_rgb"],
        "critic": ["critic", "critic_rgb"],
    }
    cfg = {"encoder": "vits", "target_size": 84, "token_feature_dim": 32, "trainable": trainable}
    actor = DepthAnythingV2Model(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        da2_cfg=cfg,
        encoder_feature_cache=encoder_feature_cache,
    )
    critic = DepthAnythingV2Model(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=[16],
        cnns=actor.cnns,
        encoder_feature_cache=encoder_feature_cache,
    )
    return actor, critic, obs, _MockDepthAnythingV2.instances[0].pretrained


def test_grouped_patch_features_are_1152_dimensional(monkeypatch) -> None:
    """Each 384-dimensional DA2 token should become 32 fixed group means before flattening."""
    actor, _critic, obs, _mock_da2 = _make_models(monkeypatch)

    features = actor.encode_features(obs)["actor_rgb"]

    assert features.shape == (NUM_ENVS, 36 * 32)
    expected_first_token = torch.arange(384, dtype=features.dtype).reshape(32, 12).mean(dim=-1)
    assert torch.allclose(features[0, :32], expected_first_token)


def test_cached_features_match_regular_forward(monkeypatch) -> None:
    """Running the policy head from cached DA2 features should preserve deterministic output."""
    actor, _critic, obs, _mock_da2 = _make_models(monkeypatch)

    features = actor.encode_features(obs)
    regular = actor(obs)
    cached = actor.forward_from_features(obs, features)

    assert torch.allclose(regular, cached)


def test_ppo_update_does_not_reencode_cached_features(monkeypatch) -> None:
    """Frozen DA2 features collected with transitions should be reused throughout PPO update."""
    actor, critic, obs, mock_da2 = _make_models(monkeypatch)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    ppo = PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, schedule="fixed")

    with torch.inference_mode():
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        ppo.compute_returns(obs)
    calls_before_update = mock_da2.forward_calls

    ppo.update()

    assert mock_da2.forward_calls == calls_before_update
    assert ppo.storage.extra is not None
    assert not ppo.storage.extra["actor_features", "actor_rgb"].is_inference()


def test_trainable_da2_encoder_disables_feature_cache(monkeypatch) -> None:
    """Trainable DA2 encoders should keep the regular gradient path instead of caching rollout features."""
    actor, critic, obs, _mock_da2 = _make_models(monkeypatch, trainable=True)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    ppo = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1, schedule="fixed")

    with torch.inference_mode():
        ppo.act(obs)
        ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})

    assert not actor.supports_feature_cache
    assert not critic.supports_feature_cache
    assert not ppo.cache_features
    assert ppo.storage.extra is None
