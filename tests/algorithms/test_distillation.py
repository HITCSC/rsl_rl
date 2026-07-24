# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Distillation algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.models import CNNModel, MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 12
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_distillation_setup(gradient_length: int = 3, num_learning_epochs: int = 1) -> tuple:
    """Build a Distillation instance with small networks."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32])
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])

    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    alg = Distillation(
        student,
        teacher,
        storage,
        num_learning_epochs=num_learning_epochs,
        gradient_length=gradient_length,
        learning_rate=1e-3,
    )
    return alg, obs, storage


def _fill_distillation_storage_steps(alg: Distillation, obs: TensorDict, steps: int) -> None:
    """Fill a configurable number of rollout slots for accumulation tests."""
    for _ in range(steps):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.privileged_actions = alg.teacher(obs).detach()
        t.rewards = torch.zeros(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


def _fill_distillation_storage(alg: Distillation, obs: TensorDict) -> None:
    """Fill the distillation storage with transitions."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.privileged_actions = alg.teacher(obs).detach()
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


class TestDistillationLoss:
    """Tests for distillation loss computation."""

    def test_loss_decreases_over_updates(self) -> None:
        """Behavior loss should decrease over repeated update() calls (learning signal works)."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3, num_learning_epochs=2)
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_distillation_storage(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["behavior"])

        # Loss should generally decrease; allow some noise — check first vs last
        assert losses[-1] < losses[0], f"Loss should decrease over updates, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_accumulation_step_count(self) -> None:
        """Optimizer should step floor(num_transitions / gradient_length) times per epoch."""
        gradient_length = 4
        alg, obs, _storage = _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=1)
        alg.train_mode()

        _fill_distillation_storage(alg, obs)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()

        expected_steps = NUM_STEPS // gradient_length
        assert step_count == expected_steps, f"Expected {expected_steps} optimizer steps, got {step_count}"

    def test_update_changes_student_but_not_teacher(self) -> None:
        """Student parameters should change after update, while teacher parameters remain frozen."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3)
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.student.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_distillation_storage(alg, obs)
        alg.update()

        any_student_changed = any(
            not torch.equal(p, student_before[name]) for name, p in alg.student.named_parameters()
        )
        assert any_student_changed, "Student parameters should change after an update"

        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during student update"

    def test_partial_gradient_window_is_optimized(self) -> None:
        """The last batches must not be dropped when steps are not divisible."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"student": ["policy"], "teacher": ["policy"]}
        student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32])
        teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])
        storage = RolloutStorage("distillation", NUM_ENVS, 5, obs, [NUM_ACTIONS])
        alg = Distillation(student, teacher, storage, gradient_length=3, learning_rate=1e-3)
        _fill_distillation_storage_steps(alg, obs, steps=5)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()
        assert step_count == 2

    def test_teacher_intervention_schedule_executes_teacher_actions(self) -> None:
        """A beta of one must execute complete teacher action vectors."""
        alg, obs, _storage = _make_distillation_setup()
        alg.teacher_intervention_start = 1.0
        alg.teacher_intervention_end = 0.0
        alg.teacher_intervention_decay_updates = 10

        with torch.no_grad():
            expected_teacher_actions = alg.teacher(obs)
            actions = alg.act(obs)

        assert torch.equal(actions, expected_teacher_actions)
        assert alg.transition.privileged_actions is not None
        assert torch.equal(alg.transition.privileged_actions, expected_teacher_actions)
        assert alg.teacher_intervention_beta == 1.0

    def test_action_loss_weights_change_behavior_objective(self) -> None:
        """Per-action weights should prioritize selected joints without changing scale."""
        alg, _obs, _storage = _make_distillation_setup()
        student_actions = torch.zeros(2, NUM_ACTIONS)
        teacher_actions = torch.zeros_like(student_actions)
        teacher_actions[:, 0] = 2.0

        unweighted = alg._compute_behavior_loss(student_actions, teacher_actions)
        alg.action_loss_weights = torch.tensor([4.0, 1.0, 1.0, 1.0])
        weighted = alg._compute_behavior_loss(student_actions, teacher_actions)

        assert weighted > unweighted

    def test_masked_terrain_loss_ignores_invalid_cells(self) -> None:
        """Invalid terrain targets must not contribute to either auxiliary loss."""
        target = torch.zeros(1, 2, 3)
        prediction = target.clone()
        prediction[0, 0, 0] = 10.0
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, 0, 0] = False

        height_loss, gradient_loss = Distillation._compute_terrain_losses(
            prediction, target, mask
        )

        assert height_loss.item() == 0.0
        assert gradient_loss.item() == 0.0

    def test_terrain_reconstruction_updates_spatial_encoder_and_decoder(self) -> None:
        """Terrain reconstruction should backpropagate through the depth CNN."""
        obs = TensorDict(
            {
                "policy": torch.randn(NUM_ENVS, OBS_DIM),
                "depth": torch.randn(NUM_ENVS, 1, 16, 16),
                "height": torch.randn(NUM_ENVS, 6),
                "height_valid": torch.ones(NUM_ENVS, 6),
            },
            batch_size=[NUM_ENVS],
        )
        obs_groups = {
            "student": ["policy", "depth"],
            "teacher": ["policy"],
        }
        student = CNNModel(
            obs,
            obs_groups,
            "student",
            NUM_ACTIONS,
            hidden_dims=[32],
            cnn_cfg={
                "depth": {
                    "output_channels": [8, 16],
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": "zeros",
                    "global_pool": "avg",
                }
            },
        )
        teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32])
        storage = RolloutStorage("distillation", NUM_ENVS, 2, obs, [NUM_ACTIONS])
        alg = Distillation(
            student,
            teacher,
            storage,
            gradient_length=2,
            terrain_reconstruction_loss_coef=0.1,
            terrain_gradient_loss_coef=0.05,
            terrain_target_obs_group="height",
            terrain_mask_obs_group="height_valid",
            terrain_target_shape=(2, 3),
        )
        cnn_before = next(student.cnns.parameters()).detach().clone()
        decoder_before = next(alg.terrain_decoder.parameters()).detach().clone()  # type: ignore[union-attr]
        for _ in range(2):
            transition = RolloutStorage.Transition()
            transition.observations = obs
            transition.actions = student(obs).detach()
            transition.privileged_actions = teacher(obs).detach()
            transition.rewards = torch.zeros(NUM_ENVS)
            transition.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(transition)

        losses = alg.update()

        assert "terrain_reconstruction" in losses
        assert "terrain_gradient" in losses
        assert not torch.equal(next(student.cnns.parameters()), cnn_before)
        assert not torch.equal(next(alg.terrain_decoder.parameters()), decoder_before)  # type: ignore[union-attr]
