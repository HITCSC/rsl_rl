# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    student: MLPModel
    """The student model."""

    teacher: MLPModel
    """The teacher model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded."""

    logs_action_std: bool = False
    """Behavior cloning uses deterministic rollout actions, so PPO's action-std metric is undefined."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        action_loss_weights: tuple[float, ...] | list[float] | None = None,
        latent_loss_coef: float = 0.0,
        teacher_intervention_start: float = 0.0,
        teacher_intervention_end: float = 0.0,
        teacher_intervention_decay_updates: int = 1,
        student_rollout_stochastic: bool = False,
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,  # handle unused config parameters
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.student`` / ``self.teacher``.
        self._raw_student = self.student
        self._raw_teacher = self.teacher

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        if action_loss_weights is None:
            self.action_loss_weights = None
        else:
            if len(action_loss_weights) != storage.actions_shape[0]:
                raise ValueError(
                    f"Expected {storage.actions_shape[0]} action loss weights, got {len(action_loss_weights)}."
                )
            weights = torch.as_tensor(action_loss_weights, dtype=torch.float, device=self.device)
            if torch.any(weights <= 0):
                raise ValueError("Action loss weights must all be positive.")
            self.action_loss_weights = weights

        self.latent_loss_coef = float(latent_loss_coef)
        if self.latent_loss_coef < 0:
            raise ValueError("latent_loss_coef must be non-negative.")
        self.latent_projection: nn.Module | None = None
        if self.latent_loss_coef > 0:
            student_latent_dim = getattr(student, "visual_latent_dim", None)
            teacher_latent_dim = getattr(teacher, "privileged_latent_dim", None)
            if student_latent_dim is None or not hasattr(student, "forward_with_visual_latent"):
                raise TypeError("Latent distillation requires a student with a visual latent interface.")
            if teacher_latent_dim is None or not hasattr(teacher, "forward_with_latent"):
                raise TypeError("Latent distillation requires a teacher with a privileged latent interface.")
            self.latent_projection = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)

        for name, value in (
            ("teacher_intervention_start", teacher_intervention_start),
            ("teacher_intervention_end", teacher_intervention_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if teacher_intervention_decay_updates <= 0:
            raise ValueError("teacher_intervention_decay_updates must be positive.")
        self.teacher_intervention_start = float(teacher_intervention_start)
        self.teacher_intervention_end = float(teacher_intervention_end)
        self.teacher_intervention_decay_updates = int(teacher_intervention_decay_updates)
        self._rollout_intervention_sum = 0.0
        self._rollout_intervention_steps = 0
        # Distillation must collect states from the student's deploy-time policy.
        # Sampling an untrained Gaussian policy sends the robot out of the frozen
        # teacher's state distribution before the behavior loss can correct it.
        self.student_rollout_stochastic = student_rollout_stochastic

        # Initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

        parameters = list(self.student.parameters())
        if self.latent_projection is not None:
            parameters.extend(self.latent_projection.parameters())
        self.optimizer = resolve_optimizer(optimizer)(parameters, lr=learning_rate)  # type: ignore

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        student_actions = self.student(obs, stochastic_output=self.student_rollout_stochastic).detach()
        if self.latent_projection is not None:
            teacher_actions, privileged_latent = self.teacher.forward_with_latent(obs)  # type: ignore
            self.transition.privileged_latent = privileged_latent.detach()
        else:
            teacher_actions = self.teacher(obs)
        teacher_actions = teacher_actions.detach()

        beta = self.teacher_intervention_beta
        if beta <= 0:
            intervention_mask = torch.zeros(
                student_actions.shape[0], 1, dtype=torch.bool, device=student_actions.device
            )
        elif beta >= 1:
            intervention_mask = torch.ones(
                student_actions.shape[0], 1, dtype=torch.bool, device=student_actions.device
            )
        else:
            intervention_mask = torch.rand(
                student_actions.shape[0], 1, device=student_actions.device
            ) < beta

        # Select a whole action vector per environment. Joint-wise blending can
        # create targets that neither policy considers dynamically consistent.
        self.transition.actions = torch.where(intervention_mask, teacher_actions, student_actions)
        self.transition.privileged_actions = teacher_actions
        self._rollout_intervention_sum += intervention_mask.float().mean().item()
        self._rollout_intervention_steps += 1
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    @property
    def teacher_intervention_beta(self) -> float:
        """Current probability that the teacher executes an environment step."""
        progress = min(self.num_updates / self.teacher_intervention_decay_updates, 1.0)
        return self.teacher_intervention_start + progress * (
            self.teacher_intervention_end - self.teacher_intervention_start
        )

    def _compute_behavior_loss(
        self, student_actions: torch.Tensor, teacher_actions: torch.Tensor
    ) -> torch.Tensor:
        element_loss = self.loss_fn(student_actions, teacher_actions, reduction="none")
        if self.action_loss_weights is None:
            return element_loss.mean()
        return (
            (element_loss * self.action_loss_weights).sum(dim=-1)
            / self.action_loss_weights.sum()
        ).mean()

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.student.update_normalization(obs)
        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op since distillation does not use return targets."""
        # Not needed for distillation
        pass

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        rollout_beta = self.teacher_intervention_beta
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_latent_loss = 0 if self.latent_projection is not None else None
        accumulated_loss: torch.Tensor | None = None
        accumulated_batches = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                # Inference of the student for gradient computation. When
                # enabled, reuse the same visual encoding for the action and
                # representation objectives.
                if self.latent_projection is not None:
                    actions, visual_latent = self.student.forward_with_visual_latent(  # type: ignore
                        batch.observations
                    )
                else:
                    actions = self.student(batch.observations)

                # Behavior cloning loss
                behavior_loss = self._compute_behavior_loss(actions, batch.privileged_actions)

                latent_loss = None
                if self.latent_projection is not None:
                    if batch.privileged_latent is None:
                        raise RuntimeError("Latent distillation target is missing from rollout storage.")
                    predicted_latent = self.latent_projection(visual_latent)
                    latent_loss = nn.functional.smooth_l1_loss(
                        predicted_latent, batch.privileged_latent
                    )

                # Total loss
                total_batch_loss = behavior_loss
                if latent_loss is not None:
                    total_batch_loss = total_batch_loss + self.latent_loss_coef * latent_loss
                accumulated_loss = (
                    total_batch_loss
                    if accumulated_loss is None
                    else accumulated_loss + total_batch_loss
                )
                accumulated_batches += 1
                mean_behavior_loss += behavior_loss.item()
                if mean_latent_loss is not None:
                    mean_latent_loss += latent_loss.item()  # type: ignore[union-attr]
                cnt += 1

                # Gradient step
                if accumulated_batches == self.gradient_length:
                    self.optimizer.zero_grad()
                    (accumulated_loss / accumulated_batches).backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self._trainable_parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    accumulated_loss = None
                    accumulated_batches = 0

                # Reset dones
                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        # Do not silently discard the final partial gradient window.
        if accumulated_loss is not None:
            self.optimizer.zero_grad()
            (accumulated_loss / accumulated_batches).backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self._trainable_parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.student.detach_hidden_state()

        mean_behavior_loss /= cnt
        if mean_latent_loss is not None:
            mean_latent_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        # Construct the loss dictionary
        intervention_rate = (
            self._rollout_intervention_sum / self._rollout_intervention_steps
            if self._rollout_intervention_steps > 0
            else 0.0
        )
        loss_dict = {
            "behavior": mean_behavior_loss,
            "teacher_intervention_beta": rollout_beta,
            "teacher_intervention_rate": intervention_rate,
        }
        if mean_latent_loss is not None:
            loss_dict["latent"] = mean_latent_loss
        self._rollout_intervention_sum = 0.0
        self._rollout_intervention_steps = 0

        return loss_dict

    def _trainable_parameters(self) -> list[nn.Parameter]:
        """Return student and auxiliary-head parameters for clipping/sync."""
        parameters = list(self.student.parameters())
        if self.latent_projection is not None:
            parameters.extend(self.latent_projection.parameters())
        return parameters

    def train_mode(self) -> None:
        """Set train mode for the student and keep the teacher in eval mode."""
        self.student.train()
        if self.latent_projection is not None:
            self.latent_projection.train()
        # Teacher is always in eval mode
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for student and teacher models."""
        self.student.eval()
        if self.latent_projection is not None:
            self.latent_projection.eval()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "student_state_dict": self._raw_student.state_dict(),
            "teacher_state_dict": self._raw_teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.latent_projection is not None:
            saved_dict["latent_projection_state_dict"] = self.latent_projection.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, determine what to load automatically
        if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):  # Load from RL training
            load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
        elif load_cfg is None:  # Load from distillation training
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("student"):
            self._raw_student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self._raw_teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self.latent_projection is not None and "latent_projection_state_dict" in loaded_dict:
            self.latent_projection.load_state_dict(loaded_dict["latent_projection_state_dict"], strict=strict)
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_student

    def compile(self, mode: str | None = None) -> None:
        """Compile student and teacher with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.student = compile_model(self._raw_student, mode)  # type: ignore
        self.teacher = compile_model(self._raw_teacher, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> Distillation:
        """Construct the distillation algorithm."""
        # Resolve class callables
        alg_class: type[Distillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["student", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Distillation is not compatible with RND and symmetry extensions
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with Distillation.")
        cfg["algorithm"]["symmetry_cfg"] = None

        # Initialize the policy
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        # Initialize the storage
        privileged_latent_dim = (
            getattr(teacher, "privileged_latent_dim", None)
            if cfg["algorithm"].get("latent_loss_coef", 0.0) > 0
            else None
        )
        storage = RolloutStorage(
            "distillation",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            privileged_latent_dim=privileged_latent_dim,
        )

        # Initialize the algorithm
        alg: Distillation = alg_class(
            student, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        alg.teacher_loaded = bool(getattr(teacher, "is_loaded", False))

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_student.state_dict(), self._raw_teacher.state_dict()]
        if self.latent_projection is not None:
            model_params.append(self.latent_projection.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_student.load_state_dict(model_params[0])
        self._raw_teacher.load_state_dict(model_params[1])
        if self.latent_projection is not None:
            self.latent_projection.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        parameters = self._trainable_parameters()
        grads = [param.grad.view(-1) for param in parameters if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in parameters:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
