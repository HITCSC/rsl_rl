# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class VisionDistillation(Distillation):
    """Vision-aware behavior distillation combining PPO with latent-space alignment.

    Extends the pure behavior-cloning :class:`Distillation` algorithm with three
    additional loss terms, following the "Now You See That" paper:

    * **PPO surrogate + value + entropy** — the student still learns from
      environment rewards (on-policy RL).
    * **Latent alignment** — MSE between student and teacher penultimate-layer
      features, aligning internal representations.
    * **Noise consistency** — MSE between clean-depth and noisy-depth student
      features, encouraging a representation that is invariant to depth noise.

    The teacher is frozen throughout and consumes privileged ``height_scan``
    observations while the student only sees noisy depth images.

    Reference:
        R.  Kolla  et  al.  "Now  You  See  That:  Learning  End-to-End
        Humanoid  Locomotion  from  Raw  Pixels."  arXiv, 2025.
    """

    critic: MLPModel
    """The critic model for value estimation (RL)."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # --- Distillation params (inherited) ---
        num_learning_epochs: int = 5,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        # --- PPO params ---
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.001,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        num_mini_batches: int = 4,
        # --- Vision distillation params ---
        teacher_obs_groups: dict[str, list[str]] | None = None,
        latent_loss_coef: float = 0.1,
        noise_consistency_coef: float = 0.05,
        noise_std: float = 0.02,
        # --- Misc ---
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the algorithm with student, teacher, critic, and storage."""
        # --- PPO / RL parameters (must be set before parent __init__) ---
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.num_mini_batches = num_mini_batches

        # --- Vision distillation params ---
        self.teacher_obs_groups = teacher_obs_groups or {"teacher": ["policy"]}
        self.latent_loss_coef = latent_loss_coef
        self.noise_consistency_coef = noise_consistency_coef
        self.noise_std = noise_std

        # Device
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Models
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.critic = critic.to(self.device)
        self._raw_student = self.student
        self._raw_teacher = self.teacher
        self._raw_critic = self.critic

        # Optimizer (student + critic, teacher is frozen)
        optimizer_parameters = {
            id(param): param
            for param in chain(self.student.parameters(), self.critic.parameters())
            if param.requires_grad
        }
        self.optimizer = resolve_optimizer(optimizer)(optimizer_parameters.values(), lr=learning_rate)

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # Distillation params
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Loss function
        loss_fn_dict = {
            "mse": F.mse_loss,
            "huber": F.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}.")

        self.num_updates = 0

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and store transition data (RL + teacher)."""
        # Record hidden states
        self.transition.hidden_states = (
            self.student.get_hidden_state(),
            self.teacher.get_hidden_state(),
        )

        # Student action (stochastic)
        self.transition.actions = self.student(obs, stochastic_output=True).detach()

        # Teacher action (deterministic, frozen — no grad)
        teacher_obs = self._select_teacher_obs(obs)
        teacher_actions = self.teacher(teacher_obs).detach()

        # Critic value
        values = self.critic(obs).detach()

        # Log-prob and distribution params (for PPO)
        actions_log_prob = self.student.get_output_log_prob(self.transition.actions).detach()
        distribution_params = tuple(p.detach() for p in self.student.output_distribution_params)

        # Store everything
        self.transition.values = values
        self.transition.actions_log_prob = actions_log_prob
        self.transition.distribution_params = distribution_params
        # Store teacher actions via extra (RL storage mode doesn't have privileged_actions)
        self.transition.extra = TensorDict(
            {"teacher_actions": teacher_actions},
            batch_size=obs.batch_size,
            device=obs.device,
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one env step, update normalizers, handle timeouts."""
        # Update normalizers
        self.student.update_normalization(obs)

        # Record rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on timeouts
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE returns and advantages (no-op in pure BC parent)."""
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run distillation + PPO optimization epochs over stored batches."""
        self.num_updates += 1
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_bc_loss = 0.0
        mean_latent_loss = 0.0
        mean_noise_loss = 0.0
        num_steps = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # --- Adaptive LR via KL ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # Run student forward to compute KL against old distribution
                    self.student(
                        batch.observations,
                        masks=batch.masks,
                        hidden_state=batch.hidden_states[0] if batch.hidden_states else None,
                        stochastic_output=True,
                    )
                    new_params = tuple(
                        p[:original_batch_size] for p in self.student.output_distribution_params
                    )
                    kl = self.student.get_kl_divergence(batch.old_distribution_params, new_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # --- Student forward (stochastic, for PPO) ---
            self.student(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0] if batch.hidden_states else None,
                stochastic_output=True,
            )
            actions_log_prob = self.student.get_output_log_prob(batch.actions)
            distribution_params = tuple(
                p[:original_batch_size] for p in self.student.output_distribution_params
            )
            entropy = self.student.output_entropy[:original_batch_size]

            # --- Critic forward ---
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1] if batch.hidden_states else None,
            )

            # --- PPO surrogate loss ---
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # --- Value loss ---
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # --- BC loss (student deterministic vs teacher stored actions) ---
            student_actions = self.student(batch.observations)  # deterministic forward
            teacher_actions = batch.extra["teacher_actions"]
            bc_loss = self.loss_fn(student_actions, teacher_actions.detach())

            # --- Latent alignment loss ---
            student_feat = self.student.get_penultimate_features(batch.observations)
            teacher_obs = self._select_teacher_obs(batch.observations)
            teacher_feat = self.teacher.get_penultimate_features(teacher_obs)
            latent_loss = F.mse_loss(student_feat, teacher_feat.detach())

            # --- Noise consistency loss ---
            noisy_obs = self._add_depth_noise(batch.observations, self.noise_std)
            noisy_feat = self.student.get_penultimate_features(noisy_obs)
            noise_loss = F.mse_loss(student_feat, noisy_feat)

            # --- Total loss ---
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + bc_loss
                + self.latent_loss_coef * latent_loss
                + self.noise_consistency_coef * noise_loss
            )

            # --- Gradient step ---
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(
                    chain(self.student.parameters(), self.critic.parameters()),
                    self.max_grad_norm,
                )
            self.optimizer.step()

            # --- Track losses ---
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_bc_loss += bc_loss.item()
            mean_latent_loss += latent_loss.item()
            mean_noise_loss += noise_loss.item()
            num_steps += 1

        # Average over steps
        mean_value_loss /= num_steps
        mean_surrogate_loss /= num_steps
        mean_entropy /= num_steps
        mean_bc_loss /= num_steps
        mean_latent_loss /= num_steps
        mean_noise_loss /= num_steps

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc": mean_bc_loss,
            "latent": mean_latent_loss,
            "noise_consistency": mean_noise_loss,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_teacher_obs(self, obs: TensorDict) -> TensorDict:
        """Select the subset of observation groups consumed by the teacher."""
        teacher_groups = self.teacher_obs_groups.get("teacher", ["policy"])
        return TensorDict(
            {g: obs[g] for g in teacher_groups},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def _add_depth_noise(self, obs: TensorDict, noise_std: float) -> TensorDict:
        """Return a copy of ``obs`` with Gaussian noise added to depth groups.

        Only groups whose name ends with ``_depth`` are noised; proprioceptive
        observations are passed through unchanged.
        """
        noisy = TensorDict({}, batch_size=obs.batch_size, device=obs.device)
        for key, value in obs.items():
            if key.endswith("_depth"):
                noisy[key] = value + torch.randn_like(value) * noise_std
            else:
                noisy[key] = value
        return noisy

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Student + critic in train mode; teacher stays in eval."""
        self.student.train()
        self.critic.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """All models in eval mode."""
        self.student.eval()
        self.critic.eval()
        self.teacher.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Return state dicts for student, teacher, critic, and optimizer."""
        return {
            "student_state_dict": self._raw_student.state_dict(),
            "teacher_state_dict": self._raw_teacher.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):
            # Load from RL training — teacher only
            load_cfg = {"teacher": True, "iteration": False}
        elif load_cfg is None:
            load_cfg = {
                "student": True,
                "teacher": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
            }

        if load_cfg.get("student"):
            self._raw_student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self._raw_teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"],
                strict=strict,
            )
            self.teacher_loaded = True
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(
                loaded_dict.get("critic_state_dict", {}),
                strict=strict,
            )
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Return the student policy for inference."""
        return self._raw_student

    def compile(self, mode: str | None = None) -> None:
        """Compile models with ``torch.compile``."""
        self.student = compile_model(self._raw_student, mode)
        self.critic = compile_model(self._raw_critic, mode)
        self.teacher = compile_model(self._raw_teacher, mode)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "VisionDistillation":
        """Construct the vision distillation algorithm from a config dict."""
        # Resolve class callables
        alg_class: type[VisionDistillation] = resolve_callable(
            cfg["algorithm"].pop("class_name")
        )
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["student", "teacher"]
        if "critic" in cfg:
            default_sets.append("critic")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Extract vision-distillation-specific configs
        teacher_obs_groups = cfg["algorithm"].pop("teacher_obs_groups", {"teacher": ["policy"]})
        teacher_checkpoint = cfg["algorithm"].pop("teacher_checkpoint", None)

        # Disable incompatible extensions
        cfg["algorithm"]["rnd_cfg"] = None
        cfg["algorithm"]["symmetry_cfg"] = None

        # Build student
        student: MLPModel = student_class(
            obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]
        ).to(device)
        print(f"Student Model: {student}")

        # Build teacher
        teacher: MLPModel = teacher_class(
            obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]
        ).to(device)
        if teacher_checkpoint:
            checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
            teacher.load_state_dict(checkpoint["actor_state_dict"])
            print(f"Teacher loaded from: {teacher_checkpoint}")
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        print(f"Teacher Model: {teacher} (frozen)")

        # Build critic
        critic_class: type[MLPModel] = resolve_callable(
            cfg.pop("critic", {}).pop("class_name", "MLPModel")
        )
        critic_obs_set = cfg.get("critic_obs_set", "student")
        critic_cfg = cfg.pop("critic", {
            "hidden_dims": cfg["student"].get("hidden_dims", (512, 256, 128)),
            "activation": cfg["student"].get("activation", "elu"),
            "obs_normalization": cfg["student"].get("obs_normalization", True),
        })
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], critic_obs_set, 1, **critic_cfg
        ).to(device)
        print(f"Critic Model: {critic}")

        # Build storage (RL mode — provides values, returns, advantages, log_prob)
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # Build algorithm
        alg: VisionDistillation = alg_class(
            student,
            teacher,
            critic,
            storage,
            device=device,
            teacher_obs_groups=teacher_obs_groups,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    # ------------------------------------------------------------------
    # Distributed training
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [
            self._raw_student.state_dict(),
            self._raw_teacher.state_dict(),
            self._raw_critic.state_dict(),
        ]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_student.load_state_dict(model_params[0])
        self._raw_teacher.load_state_dict(model_params[1])
        self._raw_critic.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Average gradients across GPUs."""
        all_params = list(chain(self.student.parameters(), self.critic.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
