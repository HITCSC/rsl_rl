# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SSR imagined-foothold guidance."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.extensions import ImaginedFoothold


def _observations(
    contacts: tuple[float, float],
    first_contacts: tuple[float, float],
) -> TensorDict:
    geometry = torch.tensor(
        [[
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.05,
            0.0,
            -0.05,
            *contacts,
            *first_contacts,
            1.0,
        ]]
    )
    # 3x3 constant-height map followed by nine valid flags.
    terrain = torch.cat((torch.zeros(1, 9), torch.ones(1, 9)), dim=-1)
    return TensorDict(
        {"state": torch.zeros(1, 3), "terrain": terrain, "geometry": geometry},
        batch_size=[1],
    )


def test_delayed_touchdown_supervision_and_guidance_reward() -> None:
    foothold = ImaginedFoothold(
        num_states=3,
        num_actions=2,
        num_envs=1,
        state_group="state",
        terrain_group="terrain",
        geometry_group="geometry",
        map_size=(0.2, 0.2),
        map_resolution=0.1,
        sole_size=(0.0, 0.0),
        sole_resolution=0.1,
        replay_capacity=8,
        batch_size=1,
        updates_per_iteration=1,
        max_pending_steps=2,
        train_min_samples=1,
        reward_min_samples=1,
        reward_min_updates=1,
        reward_min_terrain_level=0.0,
        reward_weight=0.25,
        step_dt=0.02,
    )
    actions = torch.zeros(1, 2)

    foothold.observe_action(_observations((0.0, 1.0), (0.0, 0.0)), actions)
    foothold.process_step(
        _observations((1.0, 1.0), (1.0, 0.0)), torch.zeros(1, dtype=torch.bool)
    )
    assert foothold.replay.size == 1
    assert foothold.replay.feet[0].item() == 0
    torch.testing.assert_close(foothold.replay.targets[0], torch.tensor([0.0, 0.05]))

    metrics = foothold.update()
    assert torch.isfinite(torch.tensor(metrics["foothold_nll"]))

    reward = foothold.observe_action(
        _observations((1.0, 1.0), (0.0, 0.0)), actions
    )
    torch.testing.assert_close(reward, torch.tensor([0.25 * 0.02]))
