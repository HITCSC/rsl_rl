# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os

import wandb

from rsl_rl.algorithms import Distillation
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.onnx_export import export_policy_to_onnx


class DistillationRunner(OnPolicyRunner):
    """Distillation runner for student-teacher algorithms."""

    alg: Distillation
    """The distillation algorithm."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop after validating that the teacher model is loaded."""
        # Check if teacher is loaded
        if not self.alg.teacher_loaded:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the model checkpoint and export policy to ONNX.

        Delegates to the base :meth:`OnPolicyRunner.save` for the .pt checkpoint,
        then exports ``policy.onnx`` with deploy-ordered (frame-major) observation
        layout, attaches metadata, and uploads to WandB if configured.
        """
        # 1. Save model checkpoint (.pt)
        super().save(path, infos)

        # 2. Export ONNX with frame-major -> term-major obs reordering
        dir_path = os.path.dirname(path)
        obs_manager = self.env.unwrapped.observation_manager
        export_policy_to_onnx(obs_manager, self.alg.get_policy(), dir_path, "policy.onnx")

        # 3. Attach metadata (joint names, command names, etc.)
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

        run_name: str = (
            wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
        )
        onnx_path = os.path.join(dir_path, "policy.onnx")
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)

        # 4. Upload to WandB if enabled
        if self.logger.logger_type in ["wandb"]:
            wandb.save(onnx_path, base_path=dir_path)
