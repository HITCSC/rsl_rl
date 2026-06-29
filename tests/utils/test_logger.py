# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logger backend resolution and fan-out behavior."""

from __future__ import annotations

from rsl_rl.utils.logger import FanoutSummaryWriter, Logger


class DummyWriter:
    """Small writer double that records calls."""

    def __init__(self) -> None:
        self.scalars = []
        self.models = []
        self.stopped = False

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False) -> None:
        self.scalars.append((tag, scalar_value, global_step, walltime, new_style))

    def save_model(self, model_path: str, it: int) -> None:
        self.models.append((model_path, it))

    def stop(self) -> None:
        self.stopped = True


def _make_logger(cfg: dict | None = None) -> Logger:
    """Create a logger without initializing external writers."""
    train_cfg = {
        "logger": "tensorboard",
        "num_steps_per_env": 1,
        "algorithm": {"rnd_cfg": None},
    }
    if cfg:
        train_cfg.update(cfg)
    return Logger(
        log_dir=None,
        cfg=train_cfg,
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )


def test_resolve_logger_types_falls_back_to_agent_cfg(monkeypatch) -> None:
    """Agent cfg logger is used when RSL_RL_LOGGERS is absent."""
    monkeypatch.delenv("RSL_RL_LOGGERS", raising=False)
    logger = _make_logger({"logger": "wandb"})

    assert logger._resolve_logger_types() == {"wandb"}


def test_resolve_logger_types_env_overrides_agent_cfg(monkeypatch) -> None:
    """RSL_RL_LOGGERS takes precedence over the agent cfg logger."""
    monkeypatch.setenv("RSL_RL_LOGGERS", "tensorboard,swanlab")
    logger = _make_logger({"logger": "wandb"})

    assert logger._resolve_logger_types() == {"tensorboard", "swanlab"}


def test_fanout_writer_broadcasts_scalars_and_stop() -> None:
    """Fanout writer forwards scalar and stop calls to each backend."""
    first = DummyWriter()
    second = DummyWriter()
    writer = FanoutSummaryWriter({"first": first, "second": second})

    writer.add_scalar("Train/mean_reward", 1.5, global_step=7)
    writer.stop()

    assert first.scalars == [("Train/mean_reward", 1.5, 7, None, False)]
    assert second.scalars == [("Train/mean_reward", 1.5, 7, None, False)]
    assert first.stopped
    assert second.stopped


def test_save_model_skips_external_upload_by_default(monkeypatch) -> None:
    """External model upload is disabled unless RSL_RL_UPLOAD_MODEL is true."""
    monkeypatch.delenv("RSL_RL_UPLOAD_MODEL", raising=False)
    logger = _make_logger()
    writer = DummyWriter()
    logger.writer = writer
    logger.logger_types = {"wandb"}
    logger.upload_model = False

    logger.save_model("model_0.pt", 0)

    assert writer.models == []


def test_save_model_uploads_when_enabled(monkeypatch) -> None:
    """External model upload runs when RSL_RL_UPLOAD_MODEL is enabled."""
    monkeypatch.setenv("RSL_RL_UPLOAD_MODEL", "1")
    logger = _make_logger()
    writer = DummyWriter()
    logger.writer = writer
    logger.logger_types = {"wandb"}
    logger.upload_model = True

    logger.save_model("model_0.pt", 0)

    assert writer.models == [("model_0.pt", 0)]
