# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from .encoding import AttentionBasedMapEncoding

from rsl_rl.utils import resolve_nn_activation

class ActorAttention(nn.Module):
    def __init__(
        self,
        encoder: AttentionBasedMapEncoding,
        mha_cfg: dict,
        num_actor_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorAttention.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Multi-head Attention Encoder
        self.encoder = encoder

        mlp_input_dim_a = mha_cfg["mha_dim"] + num_actor_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_layers)

    def forward(self, map_scans, observations):
        map_encoding, _ = self.encoder(map_scans, observations)
        # 拼接地图编码和原始本体感觉
        combined = torch.cat([map_encoding.squeeze(1), observations], dim=-1)  # (batch_size, d + d_obs)
        action_mean = self.actor_mlp(combined)
        return action_mean
        

class CriticAttention(nn.Module):
    def __init__(
        self,
        encoder: AttentionBasedMapEncoding,
        mha_cfg,
        num_critic_obs,
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Multi-head Attention Encoder
        self.encoder = encoder

        mlp_input_dim_c = mha_cfg["mha_dim"] + num_critic_obs

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_mlp = nn.Sequential(*critic_layers)

    def forward(self, map_scans, observations, critic_observations):
        map_encoding, _ = self.encoder(map_scans, observations)
        # 拼接地图编码和critic_observations
        combined = torch.cat([map_encoding.squeeze(1), critic_observations], dim=-1)  # (batch_size, d + d_critic_obs)
        value = self.critic_mlp(combined)
        return value


class ActorCriticAttention(nn.Module):
    is_recurrent = False

    """Actor-Critic model with attention-based encoding.

    Args:
        mha_cfg (dict): Configuration for the multi-head attention encoder. Keywords: "mha_dim", "mha_heads", "map_size"
        num_actor_obs (int): Number of observations for the actor.
        num_critic_obs (int): Number of observations for the critic.
        num_actions (int): Number of actions.
        actor_hidden_dims (list[int]): List of hidden dimensions for the actor MLP.
        critic_hidden_dims (list[int]): List of hidden dimensions for the critic MLP.
    """
    def __init__(
        self,
        mha_cfg,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # activation = resolve_nn_activation(activation)

        # Multi-head Attention Encoder
        self.encoder = AttentionBasedMapEncoding(
            d=mha_cfg["mha_dim"], 
            h=mha_cfg["mha_heads"], 
            d_obs=num_actor_obs, 
            map_size=mha_cfg["map_size"]
        )
        
        # Policy
        self.actor = ActorAttention(
            encoder=self.encoder,
            mha_cfg=mha_cfg,
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        # Value function
        self.critic = CriticAttention(
            encoder=self.encoder,
            mha_cfg=mha_cfg,
            num_critic_obs=num_critic_obs,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )

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

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

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

    def update_distribution(self, map_scans, observations):
        # compute mean
        mean = self.actor(map_scans, observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, map_scans, observations, **kwargs):
        self.update_distribution(map_scans, observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, map_scans, observations):
        actions_mean = self.actor(map_scans, observations)
        return actions_mean

    def evaluate(self, map_scans, observations, critic_observations, **kwargs):
        value = self.critic(map_scans, observations, critic_observations)
        return value

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
        return True
