# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class PolicyHeightMapCNN(nn.Module):
    def __init__(self, H=17, W=11, output_dim=64, use_output_layernorm: bool = True):
        super().__init__()

        self.H = H
        self.W = W
        self.use_output_layernorm = use_output_layernorm

        self.conv2 = nn.Sequential(
            # [B*T, 1, 17, 11] -> [B*T, 32, 17, 11]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # [B*T, 32, 17, 11] -> [B*T, 64, 9, 6]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # 消除空间维度
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, output_dim)
        self.layernorm = nn.LayerNorm(output_dim) if use_output_layernorm else nn.Identity()
        self.output_dim = output_dim

    def forward(self, height_map):
        """
        height_map: [B, 187, T]
        return:     [B, 64,  T]
        """
        # print("height_map.shape", height_map[1])
        B, C, T = height_map.shape
        min_height = height_map.min(dim=1, keepdim=True).values
        height_map = height_map - min_height
        h = height_map.permute(0, 2, 1)
        h = h.reshape(B * T, 1, self.W, self.H)
        h = h.permute(0, 1, 3, 2).contiguous()

        h = self.conv2(h)                    # [B*T, 64, h', w']
        h = self.pool(h).view(B * T, -1)     # [B*T, 64]
        h = self.fc(h)                       # [B*T, 64]
        h = self.layernorm(h)

        h = h.view(B, T, -1).permute(0, 2, 1)

        return h


class ActorCriticCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        height_map_cnn,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # initialize height map cnn
        self.height_map_cnn = height_map_cnn

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["teacher_policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        # add height map cnn output dim
        num_actor_obs_perceptive = num_actor_obs + self.height_map_cnn.output_dim
        num_critic_obs = 0
        for obs_group in obs_groups["teacher_critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        # add height map cnn output dim
        num_critic_obs_perceptive = num_critic_obs + self.height_map_cnn.output_dim

        # actor
        self.actor = MLP(num_actor_obs_perceptive, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs_perceptive, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, perceptive):
        # concatenate height map cnn output
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        obs_perceptive = torch.cat([obs, height_map_features], dim=-1)
        # compute mean
        mean = self.actor(obs_perceptive)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def get_distribution_params(self, obs, perceptive):
        # concatenate height map cnn output
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        obs_perceptive = torch.cat([obs, height_map_features], dim=-1)
        # compute mean
        mean = self.actor(obs_perceptive)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return mean, std

    def act(self, obs, **kwargs):
        # get actor obs and perception obs
        obs, perceptive = self.get_actor_obs(obs)
        # normalize actor obs
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs, perceptive)
        return self.distribution.sample()

    def act_inference(self, obs):
        # get actor obs and perception obs
        obs, perceptive = self.get_actor_obs(obs)
        # normalize actor obs
        obs = self.actor_obs_normalizer(obs)
        # concatenate height map cnn output
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        obs_perceptive = torch.cat([obs, height_map_features], dim=-1)
        # compute mean
        mean = self.actor(obs_perceptive)
        return mean

    def evaluate(self, obs, **kwargs):
        obs, perceptive = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        # concatenate height map cnn output
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        obs_perceptive = torch.cat([obs, height_map_features], dim=-1)
        return self.critic(obs_perceptive)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher_policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1), self.get_perception_obs(obs)

    def get_perception_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["perception"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher_critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1), self.get_precise_perception_obs(obs)
    
    def get_precise_perception_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["precise_perception"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
