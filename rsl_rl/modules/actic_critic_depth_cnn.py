# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


# class PolicyHeightMapCNN(nn.Module):
#     def __init__(self, H=17, W=11, output_dim=64):
#         super().__init__()

#         self.H = H
#         self.W = W

#         self.conv2 = nn.Sequential(
#             # [B*T, 1, 17, 11] -> [B*T, 32, 17, 11]
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.2),
#             # [B*T, 32, 17, 11] -> [B*T, 64, 9, 6]
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#         )

#         # 消除空间维度
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(64, output_dim)
#         self.layernorm = nn.LayerNorm(output_dim)
#         self.output_dim = output_dim

#     def forward(self, height_map):
#         """
#         height_map: [B, 187, T]
#         return:     [B, 64,  T]
#         """
        
#         B, C, T = height_map.shape
#         min_height = height_map.min(dim=1, keepdim=True).values
#         height_map = height_map - min_height
#         h = height_map.permute(0, 2, 1)
#         h = h.reshape(B * T, 1, self.W, self.H)
#         h = h.permute(0, 1, 3, 2).contiguous()

#         h = self.conv2(h)                    # [B*T, 64, h', w']
#         h = self.pool(h).view(B * T, -1)     # [B*T, 64]
#         h = self.fc(h)                       # [B*T, 64]
#         h = self.layernorm(h)

#         h = h.view(B, T, -1).permute(0, 2, 1)

#         return h

class PolicyDepthCNN(nn.Module):
    def __init__(self, H=18, W=32, output_dim=64, use_output_layernorm: bool = False):
        super().__init__()

        self.H = H  # 高度，18
        self.W = W  # 宽度，32
        self.use_output_layernorm = use_output_layernorm

        self.conv2 = nn.Sequential(
            # [N, 1, 18, 32] -> [N, 32, 18, 32]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # [N, 32, 18, 32] -> [N, 64, 9, 16]
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
        height_map: [N, 576, 1]  其中 576 = H * W = 18 * 32
        return:     [N, 64]
        """
        
        
        B, C, T = height_map.shape  # B=N, C=576, T=1
        
        # 归一化
        # min_height = height_map.min(dim=1, keepdim=True).values
        # height_map = height_map - min_height
        
        # [N, 576, 1] -> [N, 1, 18, 32]
        h = height_map.permute(0, 2, 1)  # [N, 1, 576]
        h = h.reshape(B, 1, self.H, self.W)  # [N, 1, 18, 32]

        # 卷积处理
        h = self.conv2(h)  # [N, 64, 9, 16]
        h = self.pool(h).view(B, -1)  # [N, 64]
        
        # 全连接和层归一化
        h = self.fc(h)  # [N, 64]
        h = self.layernorm(h)
        
        return h


class RMAEncoder(nn.Module):
    """Map proprioceptive observations to body-frame velocity command ``[vx, vy, wz]``."""

    VELOCITY_DIM = 3

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        activation: str = "elu",
    ):
        super().__init__()
        self.input_dim = input_dim
        act = {"elu": nn.ELU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation.lower()]
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), act()])
            prev = dim
        layers.append(nn.Linear(prev, self.VELOCITY_DIM))
        self.mlp = nn.Sequential(*layers)
        self.output_dim = self.VELOCITY_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def prediction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, target)


class ActorCriticDepthCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        height_map_cnn,
        rma_encoder=None,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        use_rma: bool = False,
        use_multi_critic: bool = False,
        num_critics: int = 1,
        style_id_key: str = "style_id",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.use_rma = use_rma
        self.use_multi_critic = use_multi_critic
        self.num_critics = num_critics
        self.style_id_key = style_id_key

        if use_rma and rma_encoder is None:
            raise ValueError("use_rma=True requires a non-None rma_encoder.")
        if use_multi_critic and num_critics < 1:
            raise ValueError("num_critics must be >= 1 when use_multi_critic=True.")
        if use_multi_critic and style_id_key not in obs.keys():
            raise ValueError(
                f"use_multi_critic=True requires observation key '{style_id_key}' in env observations."
            )

        # initialize height map cnn
        self.height_map_cnn = height_map_cnn
        self.rma_encoder = rma_encoder

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        if use_rma:
            for obs_group in obs_groups["rma"]:
                assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
        velocity_dim = RMAEncoder.VELOCITY_DIM if use_rma else 0
        self.num_policy_obs = num_actor_obs
        self.velocity_dim = velocity_dim
        self._cnn_output_dim = self.height_map_cnn.output_dim
        num_actor_obs_perceptive = num_actor_obs + self._cnn_output_dim + velocity_dim
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        num_critic_obs_perceptive = num_critic_obs + self.height_map_cnn.output_dim

        # actor
        self.actor = MLP(num_actor_obs_perceptive, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        actor_norm_dim = num_actor_obs + velocity_dim
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_norm_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic(s): one critic per terrain type when use_multi_critic is enabled
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        if use_multi_critic:
            self.critics = nn.ModuleList(
                [
                    MLP(num_critic_obs_perceptive, 1, critic_hidden_dims, activation)
                    for _ in range(num_critics)
                ]
            )
            self.critic = None
            print(f"Multi-Critic MLPs ({num_critics} critics): {self.critics}")
        else:
            self.critic = MLP(num_critic_obs_perceptive, 1, critic_hidden_dims, activation)
            self.critics = None
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
        # 1.0 = 完全使用真实 guide_vel, 0.0 = 完全使用 RMA 估计速度
        self.guide_mix_ratio = 1.0
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    def compute_guide_mix_ratio(
        iteration: int,
        pure_guide_iters: int = 2000,
        blend_iters: int = 1000,
    ) -> float:
        """Guide velocity curriculum: 全真实 -> 线性过渡 -> 全估计."""
        if iteration < pure_guide_iters:
            return 1.0
        if iteration >= pure_guide_iters + blend_iters:
            return 0.0
        return 1.0 - (iteration - pure_guide_iters) / blend_iters

    def set_guide_mix_ratio(self, ratio: float) -> None:
        self.guide_mix_ratio = float(ratio)

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

    def _build_velocity_features(self, obs_dict):
        if not self.use_rma:
            raise RuntimeError("_build_velocity_features should not be called when use_rma=False.")

        rma_obs, guide_obs = self.get_rma_obs(obs_dict)
        guide_obs = guide_obs.detach()
        if self.guide_mix_ratio >= 1.0:
            return guide_obs
        if self.guide_mix_ratio <= 0.0:
            return self.rma_encoder(rma_obs)

        rma_pred = self.rma_encoder(rma_obs)
        mix = self.guide_mix_ratio
        return mix * guide_obs + (1.0 - mix) * rma_pred

    def _build_velocity_features_inference(self, obs_dict):
        if not self.use_rma:
            raise RuntimeError("_build_velocity_features_inference should not be called when use_rma=False.")

        rma_obs = self.get_rma_obs_inference(obs_dict)
        return self.rma_encoder(rma_obs)

    def _normalize_actor_obs(self, policy_obs, vel_features=None):
        if self.use_rma:
            actor_obs = torch.cat([policy_obs, vel_features], dim=-1)
        else:
            actor_obs = policy_obs
        return self.actor_obs_normalizer(actor_obs)

    def _build_actor_features(self, obs_dict, policy_obs, perceptive, for_inference: bool = False):
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        if self.use_rma:
            # print("use_rma: ", self.use_rma)
            if for_inference:
                vel_features = self._build_velocity_features_inference(obs_dict)
            else:
                vel_features = self._build_velocity_features(obs_dict)
            normalized_actor_obs = self._normalize_actor_obs(policy_obs, vel_features)
        else:
            normalized_actor_obs = self._normalize_actor_obs(policy_obs)
        return torch.cat([normalized_actor_obs, height_map_features], dim=-1)
    

    def _build_actor_features_inference(self, obs_dict, policy_obs, perceptive, for_inference: bool = False):
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        if self.use_rma:
            if for_inference:
                vel_features = self._build_velocity_features_inference(obs_dict)
            else:
                vel_features = self._build_velocity_features(obs_dict)
            normalized_actor_obs = self._normalize_actor_obs(policy_obs, vel_features)
        else:
            normalized_actor_obs = self._normalize_actor_obs(policy_obs)
        return torch.cat([normalized_actor_obs, height_map_features], dim=-1), vel_features

    def update_distribution(self, obs_dict, policy_obs, perceptive):
        obs_perceptive = self._build_actor_features(obs_dict, policy_obs, perceptive)
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

    def get_distribution_params(self, obs_dict, policy_obs, perceptive):
        obs_perceptive = self._build_actor_features(obs_dict, policy_obs, perceptive)
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
        policy_obs, perceptive = self.get_actor_obs(obs)
        self.update_distribution(obs, policy_obs, perceptive)
        return self.distribution.sample()

    def act_inference(self, obs):
        policy_obs, perceptive = self.get_actor_obs(obs)
       
        obs_perceptive = self._build_actor_features(obs, policy_obs, perceptive, for_inference=False)
        return self.actor(obs_perceptive)

    def act_rma_inference(self, obs):
        if not self.use_rma or self.rma_encoder is None:
            raise RuntimeError("act_rma_inference requires use_rma=True and a valid rma_encoder.")
        rma_obs, _ = self.get_rma_obs(obs)
        return self.rma_encoder(rma_obs)

    def compute_rma_loss(self, obs):
        if not self.use_rma or self.rma_encoder is None:
            return obs["policy"].new_tensor(0.0)
        rma_obs, guide_obs = self.get_rma_obs(obs)
        pred = self.rma_encoder(rma_obs)
        return self.rma_encoder.prediction_loss(pred, guide_obs)

    def get_style_id(self, obs):
        """Return terrain style indices as ``[N]`` long tensor from ``style_id`` observation."""
        style_id = obs[self.style_id_key]
        return style_id.squeeze(-1).long()

    def _evaluate_critic_features(self, obs_perceptive, style_ids):
        values = torch.empty(
            obs_perceptive.shape[0], 1, device=obs_perceptive.device, dtype=obs_perceptive.dtype
        )
        for critic_idx, critic in enumerate(self.critics):
            mask = style_ids == critic_idx
            if mask.any():
                values[mask] = critic(obs_perceptive[mask])
        return values

    def evaluate(self, obs, **kwargs):
        critic_obs, perceptive = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        obs_perceptive = torch.cat([critic_obs, height_map_features], dim=-1)
        if not self.use_multi_critic:
            return self.critic(obs_perceptive)
        return self._evaluate_critic_features(obs_perceptive, self.get_style_id(obs))

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            
            obs_list.append(obs[obs_group])
            # print(obs_group, obs[obs_group].shape)
        return torch.cat(obs_list, dim=-1), self.get_perception_obs(obs)

    def get_perception_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["student_perception"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_height_scan_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["student_height_scan"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1), self.get_precise_perception_obs(obs)

    def get_precise_perception_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["student_precise_perception"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_rma_obs(self, obs):
        rma_list = []
        for obs_group in self.obs_groups["rma"]:
            rma_list.append(obs[obs_group])
        guide_list = []
        for obs_group in self.obs_groups["guide"]:
            guide_list.append(obs[obs_group])
        return torch.cat(rma_list, dim=-1), torch.cat(guide_list, dim=-1)

    def get_rma_obs_inference(self, obs):
        rma_list = []
        for obs_group in self.obs_groups["rma"]:
            rma_list.append(obs[obs_group])
        return torch.cat(rma_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            policy_obs, _ = self.get_actor_obs(obs)
            if self.use_rma:
                vel_features = self._build_velocity_features(obs)
                actor_obs = torch.cat([policy_obs, vel_features], dim=-1)
                self.actor_obs_normalizer.update(actor_obs)
            else:
                self.actor_obs_normalizer.update(policy_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    @staticmethod
    def _is_actor_state_key(key: str) -> bool:
        """Return True for CNN, actor MLP, actor_obs_normalizer, and action std."""
        return key.startswith(("height_map_cnn.", "actor.", "actor_obs_normalizer.")) or key in ("std", "log_std")

    def _merge_actor_first_layer_weight(self, old_param: torch.Tensor, new_param: torch.Tensor) -> torch.Tensor | None:
        """Merge actor input layer when layout changes from [policy+cnn] to [policy+vel+cnn]."""
        if old_param.ndim != 2 or new_param.ndim != 2:
            return None
        if old_param.shape[0] != new_param.shape[0]:
            return None

        num_policy = self.num_policy_obs
        vel_dim = self.velocity_dim
        cnn_dim = self._cnn_output_dim
        expected_old_in = num_policy + cnn_dim
        expected_new_in = num_policy + vel_dim + cnn_dim

        if old_param.shape[1] == expected_old_in and new_param.shape[1] == expected_new_in:
            merged_weight = new_param.clone()
            merged_weight[:, :num_policy] = old_param[:, :num_policy]
            merged_weight[:, num_policy + vel_dim :] = old_param[:, num_policy:]
            return merged_weight

        if old_param.shape[1] < new_param.shape[1]:
            merged_weight = new_param.clone()
            merged_weight[:, : old_param.shape[1]] = old_param
            return merged_weight
        return None

    @staticmethod
    def _merge_actor_obs_normalizer_param(old_param: torch.Tensor, new_param: torch.Tensor) -> torch.Tensor | None:
        """Partially load actor normalizer stats when velocity dims are appended."""
        if old_param.shape == new_param.shape:
            return old_param
        if old_param.ndim == new_param.ndim and old_param.shape[:-1] == new_param.shape[:-1]:
            if old_param.shape[-1] < new_param.shape[-1]:
                merged = new_param.clone()
                merged[..., : old_param.shape[-1]] = old_param
                return merged
        return None

    def load_actor_state_dict(self, state_dict: dict) -> dict:
        """Load CNN, actor MLP, ``actor_obs_normalizer``, and action std from a checkpoint.

        Supports checkpoints trained without ``rma_encoder``. When RMA velocity is newly
        appended to the normalized actor observation, normalizer stats and the actor first
        layer are partially loaded.
        """
        current_sd = self.state_dict()
        merged_sd = {key: tensor.clone() for key, tensor in current_sd.items()}

        loaded_keys: list[str] = []
        partial_keys: list[tuple[str, torch.Size, torch.Size]] = []
        skipped_shape_keys: list[tuple[str, torch.Size, torch.Size]] = []
        ignored_keys: list[str] = []

        for key, old_param in state_dict.items():
            if not self._is_actor_state_key(key):
                if key.startswith(("rma_encoder.", "rma_obs_normalizer.", "critic", "critics.")):
                    ignored_keys.append(key)
                continue

            if key not in merged_sd:
                ignored_keys.append(key)
                continue

            new_param = merged_sd[key]
            if old_param.shape == new_param.shape:
                merged_sd[key] = old_param
                loaded_keys.append(key)
                continue

            if key == "actor.0.weight":
                merged_weight = self._merge_actor_first_layer_weight(old_param, new_param)
                if merged_weight is not None:
                    merged_sd[key] = merged_weight
                    loaded_keys.append(key)
                    partial_keys.append((key, old_param.shape, new_param.shape))
                    continue

            if key.startswith("actor_obs_normalizer."):
                merged_norm = self._merge_actor_obs_normalizer_param(old_param, new_param)
                if merged_norm is not None:
                    merged_sd[key] = merged_norm
                    loaded_keys.append(key)
                    if old_param.shape != new_param.shape:
                        partial_keys.append((key, old_param.shape, new_param.shape))
                    continue

            skipped_shape_keys.append((key, old_param.shape, new_param.shape))

        missing_keys, unexpected_keys = super().load_state_dict(merged_sd, strict=False)
        actor_missing = [key for key in missing_keys if self._is_actor_state_key(key)]

        return {
            "loaded_keys": loaded_keys,
            "partial_keys": partial_keys,
            "skipped_shape_keys": skipped_shape_keys,
            "ignored_keys": ignored_keys,
            "missing_actor_keys": actor_missing,
            "unexpected_keys": unexpected_keys,
        }

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
