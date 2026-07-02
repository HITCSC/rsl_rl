# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the HeightScanAttentionModel."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import HeightScanAttentionModel

NUM_ENVS = 3
OBS_DIM = 12
HEIGHT_SCAN_SHAPE = (1, 17, 11)
NUM_ACTIONS = 5


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, OBS_DIM),
            "height_scan": torch.randn(NUM_ENVS, *HEIGHT_SCAN_SHAPE),
        },
        batch_size=[NUM_ENVS],
    )


def _make_model(output_dim: int = NUM_ACTIONS) -> tuple[HeightScanAttentionModel, TensorDict]:
    obs = _make_obs()
    model = HeightScanAttentionModel(
        obs,
        {"actor": ["policy", "height_scan"]},
        "actor",
        output_dim,
        hidden_dims=[32, 32],
        activation="elu",
        attention_cfg={
            "token_dim": 8,
            "d_model": 16,
            "n_heads": 4,
            "n_queries": 2,
            "output_dim": 24,
        },
    )
    return model, obs


def test_height_scan_attention_output_shape() -> None:
    """Model output should match the requested output dimension."""
    model, obs = _make_model()
    output = model(obs)
    assert output.shape == (NUM_ENVS, NUM_ACTIONS)


def test_height_scan_attention_backpropagates() -> None:
    """A loss through the model should produce gradients in attention parameters."""
    model, obs = _make_model(output_dim=1)
    loss = model(obs).sum()
    loss.backward()
    assert model.q_proj.weight.grad is not None
    assert model.token_projections["height_scan"].weight.grad is not None


def test_height_scan_attention_latent_contains_1d_and_attention_features() -> None:
    """Latent dimension should include normalized 1D obs plus attention output."""
    model, obs = _make_model()
    latent = model.get_latent(obs)
    assert latent.shape == (NUM_ENVS, OBS_DIM + 24)
