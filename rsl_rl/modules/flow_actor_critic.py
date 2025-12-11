# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import EmpiricalNormalization, FlowMatchingModel, MLP


class FlowActorCritic(nn.Module):
    """Actor-critic with a flow-matching policy head.

    The actor samples actions by integrating a learned velocity field, while
    still exposing a Gaussian distribution so log probabilities and entropy are
    available for PPO-style objectives.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_embbed_dim: int = 128,  # for obs cond embedding
        actor_obs_encoder_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_obs_activation: str = "elu",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_activation: str = "elu",
        flow_embed_dim: int = 64,  # for FiLM/AdaLN embedding
        flow_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        flow_activation: str = "swish",
        flow_parameterization: str = "velocity",
        flow_solver_step_size: float = 0.1,
        flow_zero_action_input: bool = False,
        flow_schedule: str = "linear",
        flow_integrator: str = "euler",
        # average_losses_before_exp: bool = True,  # If True, mean CFM loss over samples before exponentiating (lower variance ratio).
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "FlowActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        # Build observation sizes.
        # For Flow Matching Policy, Actor Observation must contain "flow" group, which shape is as same as action,]
        assert "flow" in obs_groups, "FlowActorCritic requires 'flow' in obs_groups."
        assert len(obs_groups["flow"]) == 1, "FlowActorCritic only supports 1D observations, which shape is [B,Da]"
        assert obs_groups["flow"][0] in obs, "FlowActorCritic requires {} in obs TensorDict".format(obs_groups["flow"][0])

        actor_obs_dim = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "FlowActorCritic only supports 1D observations."
            actor_obs_dim += obs[obs_group].shape[-1]
        
        critic_obs_dim = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "FlowActorCritic only supports 1D observations."
            critic_obs_dim += obs[obs_group].shape[-1]

        # Flow policy.
        # self.average_losses_before_exp = average_losses_before_exp
        self.flow_embed_dim = flow_embed_dim
        self.flow_activation = flow_activation
        self.flow_hidden_dims = flow_hidden_dims
        self.flow_solver_step_size = flow_solver_step_size 
        self.flow_zero_action_input = flow_zero_action_input
        self.flow_schedule = flow_schedule
        self.flow_integrator = flow_integrator
        self.flow = FlowMatchingModel(
            actor_obs_embbed_dim,
            num_actions,
            flow_embed_dim,
            mlp_dims=flow_hidden_dims,
            activation=flow_activation,
            parameterization=flow_parameterization,
            solver_step_size=flow_solver_step_size,
            zero_action_input=flow_zero_action_input
        )
        # perturbation
        self.perturb_action_std = -0.05 

        # Observation normalization for actor and critic.
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_obs_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_obs_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        
        # Actor Encoder : 
        self.obs_encoder = MLP(actor_obs_dim, actor_obs_embbed_dim,
                               actor_obs_encoder_dims,actor_obs_activation)

        # Critic.
        self.critic = MLP(critic_obs_dim, 1, critic_hidden_dims, critic_activation)

        self.num_actions = num_actions
        self.distribtion_t = torch.distributions.Uniform(0,1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        # No recurrent state to reset.
        pass

    @property
    def action_std(self) -> torch.Tensor:
        # No use for Flow Matching 
        return torch.zeros((1,))

    @property
    def entropy(self) -> torch.Tensor:
        # TODO : 可以使用CFM Loss进行估算
        return torch.zeros((1,))
        # return self.distribution.entropy().sum(dim=-1)  # type: ignore

    def _sample_flow_action(self, obs_embed: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample action and auxiliary flow info."""
        action = self.flow.sample(
            obs_embed,
            noise=noise,
            integrator=self.flow_integrator,
            get_path=False,
        )
        return action

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        # obs_embed = self.obs_encoder(actor_obs)
        obs_embed = actor_obs
        # get prior 
        prior = obs[self.obs_groups["flow"][0]]  # prior noise of the flow
        action = self._sample_flow_action(obs_embed, prior)
        # print("action:",torch.isnan(action).all())
        if (self.perturb_action_std is not None and self.perturb_action_std > 0):
            std = self.perturb_action_std
            action = action + torch.randn_like(action) * std
        return action
    
    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)
    
    def velocity(self,obs:TensorDict,action:torch.Tensor, noise:torch.Tensor,stamps:torch.Tensor) -> torch.Tensor:
        """
        :brief : compute the velocity of the flow 
        :param obs: TensorDict
        :param action: torch.Tensor, shape = [B,Da] 
        :param noise: torch.Tensor, shape = [B,N,Da]
        :param stamps: torch.Tensor, shape = [B,N,]
        :return velocity: torch.Tensor, shape = [B,N,Da]
        TODO : 这里的repeat写法有点问题,只能支持[B,D]这种说是
        """
        # get velocity of the flow 
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        # obs_embed = self.obs_encoder(actor_obs)
        obs_embed = actor_obs
        N_mc = 1
        if (action.dim() != noise.dim()):
            N_mc = noise.shape[1]
        obs_embed_expand = obs_embed.repeat(N_mc,1)  # shape = [N_mc*B,Do]
        action_expand = action.repeat(N_mc,1)  # shape = [N_mc*B,Da]
        # noise -> [B*N_mc,Da]
        noise_expand = noise.view(-1,*noise.shape[2:])
        stamps_expand = stamps.view(-1)  # [B*N,]
        # build noise action 
        path_sample = self.flow.path.sample(t=stamps_expand,x_0=noise_expand,x_1=action_expand)
        x_t = path_sample.x_t # get noise action 
        pred = self.flow.forward(obs_embed_expand,x_t,stamps_expand)  # compute velocity/data 
        return pred.view(*noise.shape)

    def compute_cfm_loss(self,obs:TensorDict,action:torch.Tensor, noise:torch.Tensor,stamps:torch.Tensor) -> torch.Tensor:
        """
        计算Condiontional Flow Matching Loss, 用于计算ELBO
        :param noise: 预采样好的噪声,shape = [B, N_mc, Da]
        :param timestamp: 要计算的timestamp,shape = [B, N_mc,]
        :return: ELBO, shape = [B,1] \frac{1}{N_mc}\sum l_\theta(\tau_i,\epsilon_i)
        TODO : 这里的repeat写法有点问题,只能支持[B,D]这种说是
        """
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        # obs_embed = self.obs_encoder(actor_obs)
        obs_embed = actor_obs
        N_mc = 1
        batch_size = noise.shape[0]
        if (action.dim() != noise.dim()):
            N_mc = noise.shape[1]
        obs_embed_expand = obs_embed.repeat(N_mc,1)  # shape = [N_mc*B,Do]
        action_expand = action.repeat(N_mc,1)  # shape = [N_mc*B,Da]
        # noise -> [B*N_mc,Da]
        noise_expand = noise.view(-1,*noise.shape[2:])
        stamps_expand = stamps.view(-1)  # [B*N,]
        _,elbo = self.flow.compute_cfm_loss(
            obs_embed_expand,action_expand,noise_expand,stamps_expand
        )
        # print("elbo:",torch.isnan(elbo).all())
        elbo = elbo.view(batch_size,N_mc,1)
        return elbo.mean(dim=1) 

    def sample_noise_t(self,action:torch.Tensor,N_mc:int=1)->TensorDict:
        """
        预采样噪声以及对应的时间步, 用于计算CFM Loss/ELBO
        :param action: 动作,shape = [B, Da]
        :param Nmc: 采样数量
        :return: 预采样好的噪声,shape = [B, N_mc, Da]; 时间步,shape = [B, N_mc,]
        """
        B = action.shape[0]
        result = TensorDict(
            {
                "epsilon": torch.randn((B,N_mc,action.shape[-1])),
                "t":self.distribtion_t.sample((B,N_mc,)),
            },
            batch_size=[B],
            device=action.device
        )
        return result

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
