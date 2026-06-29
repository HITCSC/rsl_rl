# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import git
import os
import pathlib
import statistics
import time
import torch
from collections import deque
from typing import Any

import rsl_rl


_EXTERNAL_LOGGERS = {"neptune", "wandb", "swanlab"}
_SUPPORTED_LOGGERS = {"neptune", "tensorboard", "wandb", "swanlab"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


class FanoutSummaryWriter:
    """Forward SummaryWriter-style calls to multiple logger backends."""

    def __init__(self, writers: dict[str, Any]) -> None:
        self.writers = writers

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to each configured backend."""
        for writer in self.writers.values():
            writer.add_scalar(
                tag,
                scalar_value,
                global_step=global_step,
                walltime=walltime,
                new_style=new_style,
            )

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Store configuration on backends that support it."""
        for writer in self.writers.values():
            if hasattr(writer, "store_config"):
                writer.store_config(env_cfg, train_cfg)

    def save_file(self, path: str) -> None:
        """Save an auxiliary file on backends that support it."""
        for writer in self.writers.values():
            if hasattr(writer, "save_file"):
                writer.save_file(path)

    def save_model(self, model_path: str, it: int) -> None:
        """Save a model artifact on backends that support it."""
        for writer in self.writers.values():
            if hasattr(writer, "save_model"):
                writer.save_model(model_path, it)

    def stop(self) -> None:
        """Stop or close each configured backend."""
        for writer in self.writers.values():
            if hasattr(writer, "stop"):
                writer.stop()
            elif hasattr(writer, "close"):
                writer.close()

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Save a video artifact on backends that support it."""
        for writer in self.writers.values():
            if hasattr(writer, "save_video"):
                writer.save_video(video, it)


class Logger:
    """Logger to save the learning metrics to different logging services."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
    ) -> None:
        """Initialize buffers and logging state for a training run."""
        self.log_dir = log_dir
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.device = device
        self.git_status_repos = [rsl_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0

        # Create buffers
        self.ep_extras = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Create RND buffers
        if self.cfg["algorithm"]["rnd_cfg"]:
            self.erewbuffer = deque(maxlen=100)
            self.irewbuffer = deque(maxlen=100)
            self.cur_ereward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cur_ireward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = is_distributed and gpu_global_rank != 0
        self.writer = None
        self.logger_types: set[str] = set()
        self.logger_type = "tensorboard"
        self.upload_model = os.environ.get("RSL_RL_UPLOAD_MODEL", "").lower() in _TRUE_VALUES

    def init_logging_writer(self) -> None:
        """Initialize the logging writer, which can be either Tensorboard, W&B or Neptune and save the code state.

        If the writer is either W&B or Neptune, the configuration and code state are uploaded as well.
        """
        if self.log_dir is not None and not self.disable_logs:
            self.logger_types = self._resolve_logger_types()
            self.logger_type = ",".join(sorted(self.logger_types))
            writers = {}
            if "neptune" in self.logger_types:
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                writers["neptune"] = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            if "wandb" in self.logger_types:
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                writers["wandb"] = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            if "swanlab" in self.logger_types:
                from rsl_rl.utils.swanlab_utils import SwanLabSummaryWriter

                writers["swanlab"] = SwanLabSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            if "tensorboard" in self.logger_types:
                from torch.utils.tensorboard import SummaryWriter

                writers["tensorboard"] = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self.writer = FanoutSummaryWriter(writers)
            self._print_logger_configuration()
        else:
            self.writer = None

        # Save code state
        files_to_upload = self._store_code_state()

        # Upload configuration and code state to external logging service if applicable
        if self.writer is not None and self.logger_types.intersection(_EXTERNAL_LOGGERS):
            self.writer.store_config(self.env_cfg, self.cfg)  # type: ignore
            for path in files_to_upload:
                self.writer.save_file(path)  # type: ignore

    def has_backend(self, name: str) -> bool:
        """Return whether a logger backend is active."""
        return name.lower() in self.logger_types

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.writer is not None:
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])

            # Update rewards and episode length
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            self.cur_episode_length += 1

            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console.

        If videos are available, they are uploaded to the logging service (W&B) as well.
        """
        if self.writer is not None:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            # Log episode extras
            extras_string = ""
            if self.ep_extras:
                # Iterate over all keys in the episode info dictionary
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    # Iterate over all steps
                    for ep_info in self.ep_extras:
                        # Handle missing, scalar, and zero dimensional tensors
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)  # type: ignore
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)  # type: ignore
                        extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

            # Log losses
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

            # Log std
            self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

            # Log performance
            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            # Log rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                if not self.has_backend("wandb") and not self.has_backend("swanlab"):
                    self.writer.add_scalar(
                        "Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time)
                    )
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            # Print to console
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            # Print run name if provided
            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            # Print performance
            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )

            # Print losses
            for key, value in loss_dict.items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

            # Print rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            # Print std
            log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

            # Print episode extras
            if not print_minimal:
                log_string += extras_string

            # Print footer
            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            # Upload available videos only when external artifact upload is explicitly enabled.
            if self.upload_model and self.has_backend("wandb"):
                for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                    self.writer.save_video(video, it)  # type: ignore

            # Clear extras buffer
            self.ep_extras.clear()

    def save_model(self, path: str, it: int) -> None:
        """Save the model to external logging services if specified."""
        if self.writer is not None and self.upload_model and self.logger_types.intersection(_EXTERNAL_LOGGERS):
            self.writer.save_model(path, it)  # type: ignore

    def stop_logging_writer(self) -> None:
        """Stop the logging writer."""
        if self.writer is not None:
            self.writer.stop()  # type: ignore

    def _store_code_state(self) -> list[str]:
        """Store the current git diff of the code repositories involved in the experiment."""
        files_to_upload = []
        if self.log_dir is not None and not self.disable_logs:
            git_log_dir = os.path.join(self.log_dir, "git")
            os.makedirs(git_log_dir, exist_ok=True)
            # Iterate over all repositories to log
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    t = repo.head.commit.tree
                    commit_hash = repo.head.commit.hexsha
                except Exception:
                    print(f"Could not find git repository in {repository_file_path}. Skipping.")
                    continue
                # Get the name of the repository
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
                # Check if the diff file already exists
                if os.path.isfile(diff_file_name):
                    continue
                # Write the diff file
                print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
                with open(diff_file_name, "x", encoding="utf-8") as f:
                    content = (
                        f"--- git commit ---\n{commit_hash}\n\n\n"
                        f"--- git status ---\n{repo.git.status()} \n\n\n"
                        f"--- git diff ---\n{repo.git.diff(t)}"
                    )
                    f.write(content)
                # Add the file path to the list of files to be uploaded
                files_to_upload.append(diff_file_name)
        return files_to_upload

    def _resolve_logger_types(self) -> set[str]:
        """Resolve logger backends from environment first, then runner config."""
        logger_value = os.environ.get("RSL_RL_LOGGERS", self.cfg.get("logger", "tensorboard"))
        logger_names = {
            name.strip().lower()
            for chunk in str(logger_value).split(",")
            for name in chunk.split("+")
            if name.strip()
        }
        if not logger_names:
            logger_names = {"tensorboard"}
        unknown_loggers = sorted(logger_names - _SUPPORTED_LOGGERS)
        if unknown_loggers:
            supported = ", ".join(sorted(_SUPPORTED_LOGGERS))
            unknown = ", ".join(unknown_loggers)
            raise ValueError(f"Logger type not found: {unknown}. Please choose from: {supported}.")
        return logger_names

    def _print_logger_configuration(self) -> None:
        """Print resolved logger settings so env/config precedence is visible."""
        logger_names = ", ".join(sorted(self.logger_types))
        print(f"[INFO] RSL-RL loggers: {logger_names}")
        print(f"[INFO] RSL-RL external model upload: {self.upload_model}")
        if self.has_backend("wandb"):
            project = os.environ.get("WANDB_PROJECT", self.cfg.get("wandb_project"))
            entity = os.environ.get("WANDB_USERNAME")
            print(f"[INFO] W&B project: {project}, entity: {entity}")
        if self.has_backend("swanlab"):
            project = (
                os.environ.get("SWANLAB_PROJ_NAME")
                or self.cfg.get("swanlab_project")
                or self.cfg.get("wandb_project")
                or self.cfg.get("experiment_name")
            )
            workspace = os.environ.get("SWANLAB_WORKSPACE")
            mode = os.environ.get("SWANLAB_MODE")
            api_host = os.environ.get("SWANLAB_API_HOST")
            web_host = os.environ.get("SWANLAB_WEB_HOST")
            print(
                f"[INFO] SwanLab project: {project}, workspace: {workspace}, mode: {mode}, "
                f"api_host: {api_host}, web_host: {web_host}"
            )
