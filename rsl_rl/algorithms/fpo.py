
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import FlowActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class FPO:
    """Flow Policy Gradient algorithm adapted to the RSL-RL interface."""

    policy: FlowActorCritic

    def __init__(
        self,
        policy: FlowActorCritic,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.05,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        # entropy_coef: float = 0.0,  # not support now 
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # for Flow Matching 
        N_mc : int = 1,  # number of MC samples for ELBO clac ,
        **kwargs 
    ) -> None:
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
        # no use RND Loss now 
        self.rnd = None
        self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.policy = policy
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        # for FPO hyerparameters : 
        self.N_mc = N_mc # number of Monte Carlo samples for CFM Loss calc 

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Flow policy returns action and auxiliary flow info.
        actions = self.policy.act(obs)
        self.transition.actions = actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        # for flow matching :
        self.transition.extra = self.policy.sample_noise_t(actions.detach(),N_mc=self.N_mc) 
        self.transition.actions_log_prob = self.policy.compute_cfm_loss(obs, 
                                                                        actions,
                                                                        self.transition.extra["epsilon"],
                                                                        self.transition.extra["t"]).detach()

        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update observation normalizers.
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
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

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        # mean_entropy = 0.0
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_cfm_loss = 0.0  # evalute CFM Loss   

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # print("===== FPO update ====")
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            extra_batch,  # for FPO, record eps & timestamps 
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]
            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )
            
            # Perform symmetric augmentation
            symmetry_noise_batch = extra_batch["epsilon"]
            symmetry_t_batch = extra_batch["t"] 
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # For FPO , you need to do symmetry for noise :
                _, symmetry_noise_batch = data_augmentation_func(
                    obs=None,
                    actions=extra_batch["epsilon"],
                    env=self.symmetry["_env"],
                ) # shape = [M*B,N_mc,action]

                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch, Build Symmetry MDP 
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                # for extra : 
                symmetry_t_batch = extra_batch["t"].repeat(num_aug,1)

            # 原PPO实现为：
            # self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            # actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # 这里核心是需要计算replaybuffer中的action在新policy下的log_prob/ELBO, 
            # 针对FM这里只需要重新调用一下计算CFM即可
            actions_log_prob_batch = self.policy.compute_cfm_loss(obs_batch, actions_batch,
                                                                  symmetry_noise_batch,
                                                                  symmetry_t_batch).detach()
            value_batch = self.policy.evaluate(obs_batch)

            # Surrogate loss
            # print("cfm:",torch.isnan(actions_log_prob_batch).all())
            # print("cfm:",(actions_log_prob_batch-torch.squeeze(old_actions_log_prob_batch)).T)
            # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))  # exp(-\frac{1}{N_mc}(l_\theta-l_\theta^old))
            # 这么干特别容易出nan,what can i say
            log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            log_ratio = torch.clamp(log_ratio,-10,10)  # deal with inf, man  
            ratio = torch.exp(log_ratio)
            # print("ratio",ratio)
            print("adv mean/std/max:", advantages_batch.mean(), advantages_batch.std(), advantages_batch.abs().max())
            # print("L_old L_new stats:", old_actions_log_prob_batch.mean(), actions_log_prob_batch.mean(), log_ratio.mean(), log_ratio.abs().max())
            print("ratio before clip:", ratio.mean())

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    # 这里和Gaussian不同,原有的高斯策略设计保证均值等变,因此不在这里镜像action,但是FM得做
                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch,
                        actions=actions_batch,
                        env=self.symmetry["_env"],
                    )
                    _, symmetry_noise_batch = data_augmentation_func(obs=None,
                                                                    actions=extra_batch["epsilon"],
                                                                    env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    # timestamp is invarient under symmetry transformations
                    symmetry_t_batch = extra_batch["t"].repeat(num_aug,1)

                # Actions predicted by the actor for symmetrically-augmented observations
                velocity_batch = self.policy.velocity(
                    obs_batch[:original_batch_size],
                    actions_batch[:original_batch_size],
                    symmetry_noise_batch[:original_batch_size],
                    symmetry_t_batch[:original_batch_size]
                )  # shape = [B*N_mc,Da]
                velocity_symm_batch = self.policy.velocity(
                    obs_batch[original_batch_size:],
                    actions_batch[original_batch_size:],
                    symmetry_noise_batch[original_batch_size:],
                    symmetry_t_batch[original_batch_size:]
                )
                # Compute the symmetry velocity :
                _, symm_velocity_batch = data_augmentation_func(
                    obs=None, actions=velocity_batch, env=self.symmetry["_env"]
                )
                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(velocity_symm_batch, symm_velocity_batch[original_batch_size:])
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            
            # CFM Loss 
            cfm_loss = -actions_log_prob_batch[:original_batch_size].mean().detach()
            # print("Surrogate Loss : ",surrogate_loss)
            # print("Value Loss :",value_loss)
            # print("CFM Loss : ", cfm_loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cfm_loss += cfm_loss.item()
            # mean_entropy += entropy_batch.mean().item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # mean_entropy /= num_updates
        mean_cfm_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "cfm": mean_cfm_loss,
        }

        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
    

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
