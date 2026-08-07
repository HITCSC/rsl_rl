# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the AMEModel (terrain elevation-map encoder)."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import AMEModel

NUM_ENVS = 2
# Kuavo-S54 AME layout: 33 x 21 ray grid with per-ray (x, y, z) coordinates.
L, W = 33, 21
MAP_SCAN_SIZE = L * W * 3  # 2079
# S54 proprioception: ang_vel(3) + gravity(3) + cmd(3) + joints(27) + vel(27) + actions(27).
ACTOR_PROPRIO_DIM = 90
CRITIC_PROPRIO_DIM = 93
NUM_ACTIONS = 27

MHA_DIM = 64
NUM_HEADS = 16

AME_CFG = {
    "map_scan_dim": (L, W, 3),
    "mha_dim": MHA_DIM,
    "num_heads": NUM_HEADS,
    "cnn_downsample": True,
    "attach_global": False,
}


def _make_obs_and_groups() -> tuple[TensorDict, dict[str, list[str]]]:
    obs = TensorDict(
        {
            "actor": torch.randn(NUM_ENVS, ACTOR_PROPRIO_DIM + MAP_SCAN_SIZE),
            "critic": torch.randn(NUM_ENVS, CRITIC_PROPRIO_DIM + MAP_SCAN_SIZE),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    return obs, obs_groups


def _make_actor_model(**kwargs: object) -> AMEModel:
    obs, obs_groups = _make_obs_and_groups()
    defaults: dict[str, object] = dict(
        hidden_dims=[64, 64],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        ame_cfg=dict(AME_CFG),
    )
    defaults.update(kwargs)
    return AMEModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)


class TestAMEStructure:
    """Structural tests for the terrain encoder wiring."""

    def test_latent_dim_is_mha_plus_proprio(self) -> None:
        """get_latent concatenates the MHA output with proprioception."""
        obs, obs_groups = _make_obs_and_groups()
        model = AMEModel(
            obs, obs_groups, "actor", NUM_ACTIONS,
            hidden_dims=[64, 64], ame_cfg=dict(AME_CFG),
        )
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, MHA_DIM + ACTOR_PROPRIO_DIM)

    def test_actor_forward_output_shape(self) -> None:
        """Stochastic forward should produce [B, num_actions]."""
        model = _make_actor_model()
        out = model(_make_obs_and_groups()[0], stochastic_output=True)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_critic_forward_output_shape(self) -> None:
        """The value model (output_dim=1, no distribution) returns [B, 1]."""
        obs, obs_groups = _make_obs_and_groups()
        critic = AMEModel(
            obs, obs_groups, "critic", 1, hidden_dims=[64, 64], ame_cfg=dict(AME_CFG)
        )
        assert critic(obs).shape == (NUM_ENVS, 1)

    def test_attach_global_adds_global_context(self) -> None:
        """attach_global appends a max-pooled global feature to the latent."""
        cfg = dict(AME_CFG)
        cfg["attach_global"] = True
        model = _make_actor_model(ame_cfg=cfg)
        latent = model.get_latent(_make_obs_and_groups()[0])
        assert latent.shape == (NUM_ENVS, MHA_DIM + ACTOR_PROPRIO_DIM + MHA_DIM)


class TestAMESharing:
    """Sharing the terrain encoder (share_cnn_encoders) across actor/critic."""

    def test_shared_encoders_are_same_object(self) -> None:
        """Passing cnns to a second AME model reuses the exact encoder modules."""
        actor = _make_actor_model()
        obs, _obs_groups = _make_obs_and_groups()
        critic = AMEModel(
            obs,
            {"actor": ["actor"], "critic": ["critic"]},
            "critic",
            1,
            hidden_dims=[64, 64],
            ame_cfg=dict(AME_CFG),
            cnns=actor.cnns,
        )
        assert critic.map_cnn is actor.map_cnn
        assert critic.mha is actor.mha

    def test_proprio_embedding_not_shared(self) -> None:
        """Each branch keeps its own proprio embedding (different widths)."""
        actor = _make_actor_model()
        obs, _obs_groups = _make_obs_and_groups()
        critic = AMEModel(
            obs,
            {"actor": ["actor"], "critic": ["critic"]},
            "critic",
            1,
            hidden_dims=[64, 64],
            ame_cfg=dict(AME_CFG),
            cnns=actor.cnns,
        )
        assert critic.proprio_embedding.in_features == CRITIC_PROPRIO_DIM
        assert actor.proprio_embedding.in_features == ACTOR_PROPRIO_DIM


class TestAMEModelJITExport:
    """JIT export fidelity for AME models."""

    def test_jit_export_model(self) -> None:
        """JIT-exported AME model should match the eager deterministic output."""
        model = _make_actor_model()
        model.eval()
        obs, _obs_groups = _make_obs_and_groups()
        original_output = model(obs).detach()

        jit_model = torch.jit.script(model.as_jit())
        jit_output = jit_model(obs["actor"])
        assert torch.allclose(original_output, jit_output, atol=1e-5)


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
class TestAMEModelONNXExport:
    """ONNX export fidelity for AME models."""

    def test_onnx_export_model(self) -> None:
        """ONNX-exported AME model should be a valid graph with correct I/O names."""
        model = _make_actor_model()
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
            assert [i.name for i in loaded.graph.input] == ["obs"]
            assert [o.name for o in loaded.graph.output] == ["actions"]
