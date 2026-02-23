# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, CNNEncoder


class CNNActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
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

        # actor cnn encoder
        self.actor_cnn_encoder = CNNEncoder()
        print(f"Actor Encoder : {self.actor_cnn_encoder}")

        # critic cnn encoder
        self.critic_cnn_encoder = CNNEncoder()
        print(f"Critic Encoder : {self.critic_cnn_encoder}")

        # get the observation dimensions
        self.obs_groups = obs_groups
        
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        # if actor_large_height_scan:
        #     num_actor_obs -= 161*101
        # else:
        #     num_actor_obs -= 17*11
        # num_actor_obs += self.actor_cnn_encoder.embedding_dim

        num_critic_obs = 0
        print("obs_groups critic:", obs_groups["critic"])
        for obs_group in obs_groups["critic"]:
            print("obs_group:", obs_group)
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        # if critic_large_height_scan:
        #     num_critic_obs -= 161*101
        # else:
        #     num_critic_obs -= 17*11
        # num_critic_obs += self.critic_cnn_encoder.embedding_dim

        # actor
        self.actor = MLP(num_actor_obs + self.actor_cnn_encoder.embedding_dim, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        
        # critic
        self.critic = MLP(num_critic_obs + self.critic_cnn_encoder.embedding_dim, 1, critic_hidden_dims, activation)
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

    def update_distribution(self, proprio_obs, perception_obs):
        # compute embedding
        emb = self.actor_cnn_encoder(self.perception_reshape(perception_obs))
        # compute mean
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        x = torch.cat([emb, proprio_obs], dim=-1)
        
        mean = self.actor(x)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        proprio_obs = self.get_actor_obs(obs)
        perception_obs = self.get_actor_perception(obs)
        self.update_distribution(proprio_obs, perception_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        proprio_obs = self.get_actor_obs(obs)
        perception = self.get_actor_perception(obs)
        # compute embedding
        emb = self.actor_cnn_encoder(self.perception_reshape(perception))
        # compute mean
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        x = torch.cat([emb, proprio_obs], dim=-1)
        
        return self.actor(x)

    def evaluate(self, obs, **kwargs):
        proprio_obs = self.get_critic_obs(obs)
        perception = self.get_critic_perception(obs)
        # compute embedding
        emb = self.critic_cnn_encoder(self.perception_reshape(perception))
        # compute critic input
        proprio_obs = self.critic_obs_normalizer(proprio_obs)
        x = torch.cat([emb, proprio_obs], dim=-1)
        
        return self.critic(x)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
    
    def get_actor_perception(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy_perception"]:
            obs_list.append(obs[obs_group])
        
        return torch.cat(obs_list, dim=-1)
        
    
    def perception_reshape(self, perception_obs):
        if perception_obs.shape[-1] == 161*101:
            perception_obs = perception_obs.view(-1, 1, 161, 101)
        elif perception_obs.shape[-1] == 17*11:
            perception_obs = perception_obs.view(-1, 1, 17, 11)
        else:
            raise ValueError(f"Unexpected height scan shape: {perception_obs.shape}")
        return perception_obs

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
    
    def get_critic_perception(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic_perception"]:
            obs_list.append(obs[obs_group])
        
        return torch.cat(obs_list, dim=-1)
        
        

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_proprio_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_proprio_obs)
        if self.critic_obs_normalization:
            critic_proprio_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_proprio_obs)

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
