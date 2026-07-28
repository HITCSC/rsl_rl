# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SSR visual-proprioceptive actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import SSRModel


def _make_model() -> tuple[SSRModel, TensorDict]:
    obs = TensorDict(
        {
            "actor": torch.randn(2, 450),
            "actor_depth": torch.randn(2, 1, 42, 42),
            "ssr_foot_heights": torch.randn(2, 110),
            "ssr_body_heights": torch.randn(2, 81),
            "ssr_base_velocity": torch.randn(2, 3),
        },
        batch_size=[2],
    )
    model = SSRModel(
        obs,
        {"actor": ["actor", "actor_depth"]},
        "actor",
        27,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_range": (0.2, 5.0),
        },
    )
    return model, obs


def test_ssr_structure_and_output_shape() -> None:
    """The actor should retain the paper's widths and emit Kuavo actions."""
    model, obs = _make_model()
    output = model(obs, stochastic_output=True)
    assert output.shape == (2, 27)
    assert len(model.mlp.experts) == 5
    assert model.temporal_encoder.hidden_size == 256
    assert model.temporal_encoder.input_size == 256
    assert model.foot_latent_head.out_features == 16
    assert model.body_latent_head.out_features == 16
    assert model.motion_mu_head.out_features == 16


def test_ssr_auxiliary_losses_backpropagate() -> None:
    """All five hybrid prediction objectives should support backpropagation."""
    model, obs = _make_model()
    extra = TensorDict({"ssr_next_proprio": torch.randn(2, 90)}, batch_size=[2])
    losses = model.auxiliary_losses(obs, extra)
    assert set(losses) == {
        "ssr_body_height",
        "ssr_foot_height",
        "ssr_next_proprio",
        "ssr_kl",
        "ssr_velocity",
    }
    total = sum(coefficient * loss for loss, coefficient in losses.values())
    total.backward()
    assert model.depth_encoder[0].weight.grad is not None


def test_ssr_export_wrapper_shape() -> None:
    """The deployment wrapper should use the S45-Rough ONNX port names."""
    model, _ = _make_model()
    exported = model.as_onnx()
    proprio, depth = exported.get_dummy_inputs()
    assert exported(proprio, depth).shape == (1, 27)
    assert exported.input_names == ["obs", "actor_depth"]
    assert exported.output_names == ["actions"]


def test_ssr_export_wrapper_preserves_oldest_to_newest_frame_order() -> None:
    """Deployment props should reach the temporal encoder oldest to newest."""
    model, _ = _make_model()
    exported = model.as_onnx()
    exported.obs_normalizer = torch.nn.Identity()
    captured: list[torch.Tensor] = []

    def capture_sequence(_module, inputs) -> None:
        captured.append(inputs[0].detach().clone())

    hook = exported.proprio_encoder.register_forward_pre_hook(capture_sequence)
    frames = torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(-1, -1, 90)
    exported(frames.reshape(1, -1), torch.zeros(1, 1, 42, 42))
    hook.remove()

    assert captured[0].shape == (1, 5, 90)
    assert torch.equal(captured[0][0, :, 0], torch.arange(5, dtype=torch.float32))
