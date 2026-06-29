# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
from dataclasses import asdict

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("swanlab package is required to log to SwanLab.") from None


class SwanLabSummaryWriter:
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """Initialize a SwanLab run for logging."""
        del flush_secs

        run_name = os.path.split(log_dir)[-1]
        project = (
            os.environ.get("SWANLAB_PROJ_NAME")
            or cfg.get("swanlab_project")
            or cfg.get("wandb_project")
            or cfg.get("experiment_name")
        )
        if project is None:
            raise KeyError(
                "Please specify SWANLAB_PROJ_NAME in the environment or swanlab_project/wandb_project/"
                "experiment_name in the runner config."
            )

        init_kwargs = {
            "project": project,
            "name": os.environ.get("SWANLAB_EXP_NAME", run_name),
            "config": {"log_dir": log_dir},
            "log_dir": os.environ.get("SWANLAB_LOGDIR", os.path.join(log_dir, "swanlab")),
        }
        workspace = os.environ.get("SWANLAB_WORKSPACE")
        if workspace:
            init_kwargs["workspace"] = workspace

        self.run = swanlab.init(**init_kwargs)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to SwanLab."""
        env_dict = _to_dict(env_cfg)
        config = {"train_cfg": train_cfg, "env_cfg": env_dict}
        if hasattr(self.run, "config") and hasattr(self.run.config, "update"):
            self.run.config.update(config)
        else:
            swanlab.config.update(config)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to SwanLab."""
        del walltime, new_style
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """Finish the active SwanLab run."""
        swanlab.finish()

    def save_file(self, path: str) -> None:
        """SwanLab curve-only mode intentionally skips auxiliary file uploads."""
        del path

    def save_model(self, model_path: str, it: int) -> None:
        """SwanLab curve-only mode intentionally skips model uploads."""
        del model_path, it


def _to_dict(cfg: dict | object) -> dict:
    """Convert common config containers to dictionaries."""
    if isinstance(cfg, dict):
        return cfg
    try:
        return cfg.to_dict()  # type: ignore[no-any-return, union-attr]
    except Exception:
        return asdict(cfg)  # type: ignore[arg-type]
