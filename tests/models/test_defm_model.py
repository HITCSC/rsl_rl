# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DeFM model using a lightweight mock encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import DefmModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 4
NUM_ACTIONS = 3


class _MockDefm(nn.Module):
    """Return deterministic patch tokens without loading TorchHub weights."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1) -> tuple[torch.Tensor]:
        self.forward_calls += 1
        tokens = torch.arange(9 * 384, dtype=x.dtype, device=x.device).reshape(1, 9, 384)
        return (tokens.expand(x.shape[0], -1, -1),)


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "actor": torch.randn(NUM_ENVS, 5),
            "critic": torch.randn(NUM_ENVS, 7),
            "actor_depth": torch.rand(NUM_ENVS, 1, 42, 42),
            "critic_depth": torch.rand(NUM_ENVS, 1, 42, 42),
        },
        batch_size=[NUM_ENVS],
    )


def _make_models(monkeypatch) -> tuple[DefmModel, DefmModel, TensorDict, _MockDefm]:
    mock_defm = _MockDefm()
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: mock_defm)
    obs = _make_obs()
    obs_groups = {
        "actor": ["actor", "actor_depth"],
        "critic": ["critic", "critic_depth"],
    }
    cfg = {"pretrained": False, "target_size": 42, "token_feature_dim": 32, "trainable": False}
    actor = DefmModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        defm_cfg=cfg,
    )
    critic = DefmModel(obs, obs_groups, "critic", 1, hidden_dims=[16], cnns=actor.cnns)
    return actor, critic, obs, mock_defm


def test_grouped_patch_features_are_288_dimensional(monkeypatch) -> None:
    """Each 384-dimensional token should become 32 fixed group means before flattening."""
    actor, _critic, obs, _mock_defm = _make_models(monkeypatch)

    features = actor.encode_features(obs)["actor_depth"]

    assert features.shape == (NUM_ENVS, 9 * 32)
    expected_first_token = torch.arange(384, dtype=features.dtype).reshape(32, 12).mean(dim=-1)
    assert torch.allclose(features[0, :32], expected_first_token)


def test_cached_features_match_regular_forward(monkeypatch) -> None:
    """Running the policy head from cached features should preserve deterministic output."""
    actor, _critic, obs, _mock_defm = _make_models(monkeypatch)

    features = actor.encode_features(obs)
    regular = actor(obs)
    cached = actor.forward_from_features(obs, features)

    assert torch.allclose(regular, cached)


def test_ppo_update_does_not_reencode_cached_features(monkeypatch) -> None:
    """Frozen DeFM features collected with transitions should be reused throughout PPO update."""
    actor, critic, obs, mock_defm = _make_models(monkeypatch)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    ppo = PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, schedule="fixed")

    with torch.inference_mode():
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        ppo.compute_returns(obs)
    calls_before_update = mock_defm.forward_calls

    ppo.update()

    assert mock_defm.forward_calls == calls_before_update
    assert ppo.storage.extra is not None
    assert not ppo.storage.extra["actor_features", "actor_depth"].is_inference()
