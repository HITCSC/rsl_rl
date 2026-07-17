# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MoEModel."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import MoEModel

NUM_ENVS = 2
OBS_DIM_1D = 6
IMG_H, IMG_W = 16, 16
IMG_CHANNELS = 3
NUM_ACTIONS = 4

CNN_CFG = {
    "output_channels": [8, 16],
    "kernel_size": 3,
    "stride": 1,
    "activation": "elu",
}

MOE_CFG = {
    "num_experts": 3,
    "expert_hidden_dims": [32],
    "gate_hidden_dims": [16],
    "moe_output_dim": 24,
}


def _make_moe_model(**kwargs: object) -> tuple[MoEModel, TensorDict]:
    obs = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, OBS_DIM_1D),
            "image": torch.randn(NUM_ENVS, IMG_CHANNELS, IMG_H, IMG_W),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {"actor": ["policy", "image"]}
    defaults: dict[str, object] = dict(
        hidden_dims=[32, 32],
        cnn_cfg={"image": CNN_CFG},
        moe_cfg=dict(MOE_CFG),
    )
    defaults.update(kwargs)
    model = MoEModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


class TestMoEStructure:
    """Structural tests for the MoE head wiring."""

    def test_latent_dim_is_moe_output_dim(self) -> None:
        """get_latent should return the MoE-mixed latent of width moe_output_dim."""
        model, obs = _make_moe_model()
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, MOE_CFG["moe_output_dim"])

    def test_num_experts_and_gate(self) -> None:
        """The head should hold num_experts experts and a gate over them."""
        model, _obs = _make_moe_model()
        assert len(model.moe.experts) == MOE_CFG["num_experts"]
        # Gate output width equals number of experts.
        gate_out = model.moe.gate(torch.randn(NUM_ENVS, model.obs_dim + model.cnn_latent_dim))
        assert gate_out.shape == (NUM_ENVS, MOE_CFG["num_experts"])

    def test_forward_output_shape(self) -> None:
        """Deterministic forward should produce [B, num_actions]."""
        model, obs = _make_moe_model(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        out = model(obs)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_gate_weights_sum_to_one(self) -> None:
        """The softmax gate weights must form a valid distribution over experts."""
        model, obs = _make_moe_model()
        latent = model.obs_normalizer(obs["policy"])
        cnn = torch.cat([model.cnns[g](obs[g]) for g in model.obs_groups_2d], dim=-1)
        enc = torch.cat([latent, cnn], dim=-1)
        weights = torch.softmax(model.moe.gate(enc), dim=-1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(NUM_ENVS), atol=1e-6)

    def test_output_changes_with_image(self) -> None:
        """Changing the image input should change the MoE-mixed latent."""
        model, obs = _make_moe_model()
        before = model.get_latent(obs).detach().clone()
        obs["image"] = torch.randn_like(obs["image"])
        after = model.get_latent(obs).detach()
        assert not torch.allclose(before, after, atol=1e-6)


class TestMoESharing:
    """Sharing the CNN encoders (share_cnn_encoders) must still work for MoE."""

    def test_shared_cnns_are_same_object(self) -> None:
        """Passing cnns to a second MoE model shares the exact encoder objects."""
        model_a, obs = _make_moe_model()
        model_b = MoEModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            cnns=model_a.cnns,
            moe_cfg=dict(MOE_CFG),
        )
        for name in model_a.cnns:
            assert model_a.cnns[name] is model_b.cnns[name]

    def test_experts_not_shared(self) -> None:
        """MoE heads are per-model even when CNN encoders are shared."""
        model_a, obs = _make_moe_model()
        model_b = MoEModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            cnns=model_a.cnns,
            moe_cfg=dict(MOE_CFG),
        )
        assert model_a.moe is not model_b.moe


class TestMoEModelJITExport:
    """JIT export fidelity for MoE models."""

    def test_jit_export_model(self) -> None:
        """JIT-exported MoE model should match the eager deterministic output."""
        model, obs = _make_moe_model(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        original_output = model(obs).detach()

        jit_model = torch.jit.script(model.as_jit())
        jit_output = jit_model(obs["policy"], [obs["image"]])
        assert torch.allclose(original_output, jit_output, atol=1e-5)


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
class TestMoEModelONNXExport:
    """ONNX export fidelity for MoE models."""

    def test_onnx_export_model(self) -> None:
        """ONNX-exported MoE model should be a valid graph with correct I/O names."""
        model, _obs = _make_moe_model(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
            assert [i.name for i in loaded.graph.input] == ["obs", "image"]
            assert [o.name for o in loaded.graph.output] == ["actions"]
