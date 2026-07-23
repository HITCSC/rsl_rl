# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.D_PPO import DPPO
from rsl_rl.modules.multi_discriminator import MultiDiscriminator
from rsl_rl.storage import StyleReplayBuffer


class DMultiAMPPPO(DPPO):
    """Distillation PPO with multi-style AMP discriminators routed by terrain/style id."""

    def __init__(
        self,
        policy,
        multi_discriminator: MultiDiscriminator,
        amp_loaders: dict[int, object],
        amp_normalizers: dict[int, object],
        style_ids: list[int],
        amp_obs_layout: dict | None = None,
        amp_replay_buffer_size=100000,
        amploss_coef=1.0,
        grad_pen_lambda=10.0,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)

        self.amploss_coef = amploss_coef
        self.grad_pen_lambda = grad_pen_lambda
        self.multi_discriminator = multi_discriminator
        self.multi_discriminator.to(self.device)
        self.amp_loaders = amp_loaders
        self.amp_normalizers = amp_normalizers
        self.amp_obs_layout = amp_obs_layout
        self.style_ids = [int(style_id) for style_id in style_ids]

        amp_obs_dim = multi_discriminator.amp_obs_dim
        self.amp_storage = StyleReplayBuffer(amp_obs_dim, amp_replay_buffer_size, self.device)
        self._current_amp_obs = None
        self._current_style_ids = None

        params = [{"params": self.policy.parameters(), "name": "policy"}]
        for style_id in self.style_ids:
            trunk_params, head_params = self.multi_discriminator.parameters_by_style(style_id)
            params.append({"params": trunk_params, "weight_decay": 10e-4, "name": f"amp_trunk_{style_id}"})
            params.append({"params": head_params, "weight_decay": 10e-2, "name": f"amp_head_{style_id}"})
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def act(self, obs, amp_obs=None, style_ids=None):
        actions = super().act(obs)
        self._current_amp_obs = amp_obs
        self._current_style_ids = style_ids
        return actions

    def process_env_step_distillation(
        self, obs, rewards, dones, extras, student_flag, half_size, amp_obs=None, style_ids=None
    ):
        super().process_env_step_distillation(obs, rewards, dones, extras, student_flag, half_size)
        if self._current_amp_obs is not None and amp_obs is not None and style_ids is not None:
            self.amp_storage.insert(self._current_amp_obs, amp_obs, style_ids)
        self._current_amp_obs = None
        self._current_style_ids = None

    def _compute_amp_loss(self, style_id: int, policy_state, policy_next_state, expert_state, expert_next_state):
        normalizer = self.amp_normalizers[style_id]
        discriminator = self.multi_discriminator.get_discriminator(style_id)

        if normalizer is not None:
            with torch.no_grad():
                policy_state = normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = normalizer.normalize_torch(expert_next_state, self.device)

        policy_d = discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=self.grad_pen_lambda)

        if normalizer is not None:
            normalizer.update(policy_state.cpu().numpy())
            normalizer.update(expert_state.cpu().numpy())

        return amp_loss, grad_pen_loss, policy_d.mean().item(), expert_d.mean().item()

    def update_rl_distillation(self, student_flag):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_behavior_loss = 0
        mean_rma_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        alpha = 0.01

        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_amp_updates = self.num_learning_epochs * self.num_mini_batches
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_policy_generators = {
            style_id: self.amp_storage.feed_forward_generator(style_id, num_amp_updates, mini_batch_size)
            for style_id in self.style_ids
        }
        amp_expert_generators = {
            style_id: self.amp_loaders[style_id].feed_forward_generator(num_amp_updates, mini_batch_size)
            for style_id in self.style_ids
        }

        loss = 0
        for (
            obs_batch,
            teacher_obs_batch,
            actions_batch,
            privileged_actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            student_actions = self.policy.act_inference(teacher_obs_batch)
            behavior_loss = self.loss_fn(student_actions, privileged_actions_batch.detach())
            rma_loss = (
                self.policy.compute_rma_loss(obs_batch)
                if self.use_rma
                else obs_batch["policy"].new_tensor(0.0)
            )
            alpha = 0.2 if student_flag else 1.0
            loss = alpha * behavior_loss
            if self.use_rma:
                loss = loss + self.rma_loss_coef * rma_loss
            print("alpha: ", alpha, "use_rma: ", self.use_rma)
            if student_flag:
                num_aug = 1
                original_batch_size = obs_batch.batch_size[0]

                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                if self.symmetry and self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch,
                        actions=actions_batch,
                        env=self.symmetry["_env"],
                    )
                    num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)

                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
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

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss += surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            batch_amp_loss = 0.0
            batch_grad_pen_loss = 0.0
            batch_policy_pred = 0.0
            batch_expert_pred = 0.0
            for style_id in self.style_ids:
                sample_amp_policy = next(amp_policy_generators[style_id])
                sample_amp_expert = next(amp_expert_generators[style_id])
                amp_loss, grad_pen_loss, policy_pred, expert_pred = self._compute_amp_loss(
                    style_id, *sample_amp_policy, *sample_amp_expert
                )
                batch_amp_loss += amp_loss
                batch_grad_pen_loss += grad_pen_loss
                batch_policy_pred += policy_pred
                batch_expert_pred += expert_pred

            loss += self.amploss_coef * (batch_amp_loss + batch_grad_pen_loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if student_flag:
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                mean_behavior_loss += behavior_loss.item()
                mean_rma_loss += rma_loss.item()
            else:
                mean_behavior_loss += behavior_loss.item()
                mean_rma_loss += rma_loss.item()
            mean_amp_loss += batch_amp_loss.item()
            mean_grad_pen_loss += batch_grad_pen_loss.item()
            mean_policy_pred += batch_policy_pred / len(self.style_ids)
            mean_expert_pred += batch_expert_pred / len(self.style_ids)
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_behavior_loss /= num_updates
        mean_rma_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "behavior_loss": mean_behavior_loss,
            "rma": mean_rma_loss,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
            "guide_mix_ratio": getattr(self.policy, "guide_mix_ratio", 0.0),
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
