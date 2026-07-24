# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EMP teacher model for teacher-student distillation.

The :class:`EMPTeacherModel` wraps the frozen :class:`~rsl_rl.modules.ActorCriticCNN`
from the EMP repository checkpoint (``doc/model_48350.pt``) so that it satisfies the
interface expected by the :class:`~rsl_rl.algorithms.Distillation` algorithm:

* ``forward(obs)`` — constructs the teacher's 486-dim flat input from the three
  teacher observation groups and returns actions.
* ``reset()`` / ``get_hidden_state()`` / ``detach_hidden_state()`` — no-ops
  (the teacher is a feed-forward CNN-MLP with no recurrent state).
* ``update_normalization()`` — no-op (the teacher's normalizer is frozen).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.emp_modules import (
    S45_LAB_TO_MJCF,
    build_teacher,
    joint_order_mjcf_to_lab_term_major,
)
from rsl_rl.modules.rnn import HiddenState


class EMPTeacherModel(nn.Module):
    """Frozen teacher model wrapper compatible with the Distillation algorithm.

    The teacher consumes three observation groups registered in the environment:

    * ``teacher_proprio`` — 5-frame history of proprioception ``[B, 420]``
    * ``teacher_cmd`` — current velocity command ``[B, 3]``
    * ``teacher_height`` — cropped height scan ``[B, 63]``

    These are concatenated into the 486-dim flat vector expected by
    :class:`~rsl_rl.modules.ActorCriticCNN`.
    """

    is_recurrent: bool = False
    """The teacher has no recurrent state."""

    privileged_latent_dim: int = 64
    """Dimensionality of the frozen EMP height-map representation."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        num_actions: int,
        checkpoint_path: str = "doc/model_48350.pt",
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.num_actions = num_actions
        self.teacher_net = build_teacher(checkpoint_path, device="cpu")
        # DistillationRunner uses this marker to reject accidentally uninitialized
        # teachers. Reaching this line means build_teacher loaded the checkpoint.
        self.is_loaded = True

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run teacher inference and return actions.

        Args:
            obs: Observation TensorDict with keys ``teacher_proprio``,
                ``teacher_cmd`` and ``teacher_height``.
            masks: Ignored (non-recurrent).
            hidden_state: Ignored (non-recurrent).
            stochastic_output: Ignored (teacher is always deterministic).

        Returns:
            Teacher actions ``[B, 26]`` (pre-scaling).
        """
        teacher_input = self._build_teacher_input(obs)
        actions = self.teacher_net(teacher_input)  # [B, 26] — Lab order
        return self._actions_to_mjcf(actions)

    def forward_with_latent(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MJCF-ordered teacher actions and the frozen terrain latent."""
        teacher_input = self._build_teacher_input(obs)
        actions, terrain_latent = self.teacher_net.forward_with_latent(teacher_input)
        return self._actions_to_mjcf(actions), terrain_latent

    def _build_teacher_input(self, obs: TensorDict) -> torch.Tensor:
        """Assemble the checkpoint-aligned 486-D EMP teacher input."""
        # The ``teacher_proprio`` group (420-dim) is in **term-major** order
        # from MJLab's observation manager — same as Isaac Lab during training.
        # We only need to reorder joint dims from MJCF (grouped) to Lab
        # (left-right interleaved) to match the checkpoint's training distribution.
        # Do NOT convert to frame-major here: the normalizer was trained on
        # term-major data and would subtract wrong per-dim means on frame-major.
        proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])
        # Build the 486-dim flat input in the exact order the teacher expects:
        #   [cmd(3), proprio(420), height(63)]
        return torch.cat(
            [obs["teacher_cmd"], proprio, obs["teacher_height"]],
            dim=-1,
        )

    @staticmethod
    def _actions_to_mjcf(actions: torch.Tensor) -> torch.Tensor:
        """Convert checkpoint USD/Lab action order to the MJCF environment order."""
        # The env expects actions in MJCF (qpos) order.  Convert here so that
        # distillation loss compares student (MJCF) and teacher (now MJCF) in
        # the same joint ordering.
        idx = torch.tensor(S45_LAB_TO_MJCF, device=actions.device, dtype=torch.long)
        return actions[..., idx]

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: teacher normaliser is frozen."""

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """No-op."""

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """No-op."""
