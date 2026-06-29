# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Next-observation prediction auxiliary loss
        next_obs_prediction_cfg: dict | None = None,
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

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.storage = storage
        self.cache_features = bool(
            getattr(self.actor, "supports_feature_cache", False)
            and getattr(self.critic, "supports_feature_cache", False)
        )
        if self.cache_features and self.symmetry:
            raise ValueError("Symmetry augmentation is not supported when caching encoder features.")
        if next_obs_prediction_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Next-observation prediction is not supported for recurrent policies.")

        # Next-observation prediction auxiliary loss
        self.next_obs_predictor: MLP | None = None
        self.next_obs_target_normalizer: EmpiricalNormalization | nn.Identity | None = None
        self.next_obs_groups: list[str] | None = None
        self.next_obs_loss_coef = 0.0
        self.next_obs_mask_dones = True
        self._rollout_final_obs: TensorDict | None = None
        if next_obs_prediction_cfg is not None:
            self._init_next_obs_prediction(next_obs_prediction_cfg)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer
        optimizer_parameters = {
            id(parameter): parameter
            for parameter in chain(
                self.actor.parameters(),
                self.critic.parameters(),
                self.next_obs_predictor.parameters() if self.next_obs_predictor is not None else (),
            )
            if parameter.requires_grad
        }
        self.optimizer = resolve_optimizer(optimizer)(optimizer_parameters.values(), lr=learning_rate)  # type: ignore

        # Add storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        if self.cache_features:
            actions, actor_features = self._raw_actor.forward_with_features(obs, stochastic_output=True)  # type: ignore
            values, critic_features = self._raw_critic.forward_with_features(obs)  # type: ignore
            self.transition.extra = TensorDict(
                {
                    "actor_features": actor_features,
                    "critic_features": critic_features,
                },
                batch_size=obs.batch_size,
                device=obs.device,
            )
            self.transition.actions = actions.detach()
            self.transition.values = values.detach()
            actor_model = self._raw_actor
        else:
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self.critic(obs).detach()
            actor_model = self.actor
        self.transition.actions_log_prob = actor_model.get_output_log_prob(  # type: ignore
            self.transition.actions
        ).detach()
        self.transition.distribution_params = tuple(p.detach() for p in actor_model.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
        if self.next_obs_groups is not None:
            self._rollout_final_obs = obs.select(*self.next_obs_groups)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_next_obs_loss = 0 if self.next_obs_predictor is not None else None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                next_obs_groups=self.next_obs_groups,
                final_obs=self._rollout_final_obs,
            )

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            if self.cache_features:
                if batch.extra is None:
                    raise RuntimeError("Cached-feature PPO requires rollout batch extra data.")
                self._raw_actor.forward_from_features(  # type: ignore
                    batch.observations,
                    batch.extra["actor_features"],
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                values = self._raw_critic.forward_from_features(  # type: ignore
                    batch.observations,
                    batch.extra["critic_features"],
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[1],
                )
                actor_model = self._raw_actor
            else:
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                actor_model = self.actor
            actions_log_prob = actor_model.get_output_log_prob(batch.actions)  # type: ignore
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in actor_model.output_distribution_params)
            entropy = actor_model.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = actor_model.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Next-observation prediction auxiliary loss
            next_obs_loss = self._compute_next_obs_prediction_loss(batch, original_batch_size)
            if next_obs_loss is not None:
                loss = loss + self.next_obs_loss_coef * next_obs_loss

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_next_obs_loss is not None:
                mean_next_obs_loss += next_obs_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_next_obs_loss is not None:
            mean_next_obs_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.next_obs_predictor is not None:
            loss_dict["next_obs_prediction"] = mean_next_obs_loss

        # Clear the storage
        self.storage.clear()
        self._rollout_final_obs = None

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        if self.next_obs_predictor is not None:
            self.next_obs_predictor.train()
        if self.next_obs_target_normalizer is not None:
            self.next_obs_target_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        if self.next_obs_predictor is not None:
            self.next_obs_predictor.eval()
        if self.next_obs_target_normalizer is not None:
            self.next_obs_target_normalizer.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        if self.next_obs_predictor is not None:
            saved_dict["next_obs_predictor_state_dict"] = self.next_obs_predictor.state_dict()
            if self.next_obs_target_normalizer is not None:
                saved_dict["next_obs_target_normalizer_state_dict"] = self.next_obs_target_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "next_obs_prediction": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            try:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except ValueError:
                if self.next_obs_predictor is None or "next_obs_predictor_state_dict" in loaded_dict:
                    raise
                print(
                    "Skipping optimizer state load because the checkpoint predates "
                    "next-observation prediction."
                )
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if (
            load_cfg.get("next_obs_prediction")
            and self.next_obs_predictor
            and "next_obs_predictor_state_dict" in loaded_dict
        ):
            self.next_obs_predictor.load_state_dict(loaded_dict["next_obs_predictor_state_dict"], strict=strict)
            if (
                self.next_obs_target_normalizer is not None
                and "next_obs_target_normalizer_state_dict" in loaded_dict
            ):
                self.next_obs_target_normalizer.load_state_dict(
                    loaded_dict["next_obs_target_normalizer_state_dict"],
                    strict=strict,
                )
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share visual encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        if self.next_obs_predictor is not None:
            model_params.append(self.next_obs_predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])
        if self.next_obs_predictor is not None:
            self.next_obs_predictor.load_state_dict(model_params[-1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.next_obs_predictor is not None:
            all_params = chain(all_params, self.next_obs_predictor.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def _init_next_obs_prediction(self, cfg: dict) -> None:
        """Initialize the optional next-observation prediction auxiliary head."""
        cfg = dict(cfg)
        target_groups = cfg.pop("target_obs_groups", None)
        if target_groups is None:
            target_groups = [
                group for group in self.actor.obs_groups if len(self.storage.observations[group].shape) == 3
            ]
        self.next_obs_groups = list(target_groups)
        if not self.next_obs_groups:
            raise ValueError("Next-observation prediction requires at least one 1D target observation group.")
        for group in self.next_obs_groups:
            if group not in self.storage.observations:
                raise ValueError(f"Next-observation target group '{group}' is not in the environment observations.")
            if len(self.storage.observations[group].shape) != 3:
                raise ValueError(
                    f"Next-observation target group '{group}' must be 1D, got {self.storage.observations[group].shape}."
                )

        target_dim = sum(self.storage.observations[group].shape[-1] for group in self.next_obs_groups)
        self.next_obs_loss_coef = float(cfg.pop("loss_coef", 1.0))
        hidden_dims = cfg.pop("hidden_dims", (256, 256))
        activation = cfg.pop("activation", "elu")
        target_normalization = bool(cfg.pop("target_normalization", True))
        self.next_obs_mask_dones = bool(cfg.pop("mask_dones", True))
        if cfg:
            raise ValueError(f"Unsupported next_obs_prediction_cfg keys: {sorted(cfg)}")

        self.next_obs_predictor = MLP(
            self.actor.mlp.feature_dim,
            target_dim,
            hidden_dims,
            activation,
        ).to(self.device)
        self.next_obs_target_normalizer = (
            EmpiricalNormalization(target_dim).to(self.device) if target_normalization else nn.Identity()
        )

    def _compute_next_obs_prediction_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
    ) -> torch.Tensor | None:
        """Compute the optional next-observation prediction auxiliary loss."""
        if self.next_obs_predictor is None:
            return None
        if batch.extra is None or "next_obs_prediction" not in batch.extra:
            raise RuntimeError("Next-observation prediction requires batch extra targets.")

        observations = batch.observations[:original_batch_size]  # type: ignore[index]
        if self.cache_features:
            representation = self._raw_actor.get_representation_from_features(  # type: ignore[attr-defined]
                observations,
                batch.extra["actor_features"][:original_batch_size],
                masks=None,
                hidden_state=None,
            )
        else:
            representation = self._raw_actor.get_representation(observations, masks=None, hidden_state=None)

        target = batch.extra["next_obs_prediction", "target"][:original_batch_size]
        assert self.next_obs_target_normalizer is not None
        if isinstance(self.next_obs_target_normalizer, EmpiricalNormalization):
            with torch.no_grad():
                self.next_obs_target_normalizer.update(target)
        target = self.next_obs_target_normalizer(target)

        prediction = self.next_obs_predictor(representation)
        sample_loss = (prediction - target).pow(2).mean(dim=-1)
        if self.next_obs_mask_dones:
            valid = ~batch.dones[:original_batch_size].bool().squeeze(-1)  # type: ignore[index]
            if not torch.any(valid):
                return sample_loss.mean() * 0.0
            sample_loss = sample_loss[valid]
        return sample_loss.mean()
