# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd

import numpy as np
from typing import Callable, Tuple

class DiscHeightMapCNN(nn.Module):
    def __init__(self, H=17, W=11, output_dim=64):
        super().__init__()

        self.H = H
        self.W = W

        self.encoder = nn.Sequential(
            # ===== Stage 1: 局部结构 =====
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            # ===== Stage 2: 中尺度结构 =====
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 17x11 → 9x6
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            # ===== Stage 3: 全局结构 =====
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 9x6 → 5x3
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 3, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )
        self.layernorm = nn.LayerNorm(output_dim)

        self.output_dim = output_dim

    def forward(self, height_map):
        B, C, T = height_map.shape
        min_height = height_map.min(dim=1, keepdim=True).values
        height_map = height_map - min_height
        h = height_map.permute(0, 2, 1)
        h = h.reshape(B * T, 1, self.W, self.H)
        h = h.permute(0, 1, 3, 2).contiguous()

        h = self.encoder(h)
        h = h.flatten(1)
        h = self.fc(h)
        h = self.layernorm(h)

        h = h.view(B, T, -1).permute(0, 2, 1)
        return h


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip((input - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        amp_discr_hidden_dims (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        amp_task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        amp_task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self, 
                 obs, 
                 obs_groups, 
                 height_map_cnn,
                 device = "cpu",
                 amp_reward_coef = 0.3,
                 amp_motion_files = None,
                 amp_num_preload_transitions = 200000,
                 amp_task_reward_lerp = 0.7,
                 amp_discr_hidden_dims = [1024, 512, 256],
                 amp_use_nomalization = True,
                 **kwargs):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.device = device

        # initialize height map cnn
        self.height_map_cnn = height_map_cnn

        # get the observation dimensions
        self.obs_groups = obs_groups
        input_dim = 0
        for obs_group in obs_groups["discriminator"]:
            assert len(obs[obs_group].shape) == 2, "The Discriminator module only supports 1D observations."
            input_dim += obs[obs_group].shape[-1]
        input_style_obs_dim = input_dim
        # add height map cnn output dim
        input_dim += self.height_map_cnn.output_dim
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in amp_discr_hidden_dims:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(amp_discr_hidden_dims[-1], 1).to(device)

        # discriminator observation normalization
        self.discriminator_obs_normalization = amp_use_nomalization
        if amp_use_nomalization:
            self.discriminator_obs_normalizer = Normalizer(input_style_obs_dim)
        else:
            self.discriminator_obs_normalizer = torch.nn.Identity()
        print(f"Discriminator Network: \n{self.trunk}\n{self.amp_linear}")

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = amp_task_reward_lerp

    def get_discriminator_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["discriminator"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1), self.get_precise_perception_obs(obs)
    
    def get_precise_perception_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["precise_perception"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def forward(self, discriminator_obs, perceptive):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        # normalize
        discriminator_obs = self.discriminator_obs_normalizer.normalize_torch(discriminator_obs, self.device)
        # concatenate height map cnn output
        height_map_features = self.height_map_cnn(perceptive.unsqueeze(-1)).squeeze(-1)
        discriminator_obs = torch.cat([discriminator_obs, height_map_features], dim=-1)
        x = discriminator_obs
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, discriminator_obs, perceptive, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        discriminator_obs.requires_grad = True

        # disc = self.amp_linear(self.trunk(expert_data))
        disc = self.forward(discriminator_obs, perceptive)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=discriminator_obs, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, discriminator_obs, perceptive, task_reward, normalizer=None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            # next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            
            # get discriminator observations
            # discriminator_obs, perceptive = self.get_discriminator_obs(state)
            d = self.forward(discriminator_obs, perceptive)
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            amp_reward = reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), amp_reward, d
    
    def reorganize_style_obs(self, style_obs):
        """
        Style obs consists of 2 frame of style observations. 
        In isaaclab, it is concatenated by ITEM instead of by FRAME.
        So we need to reorganize the style obs, make sure they are concatenated by ITEM.

        Args:
            style_obs (torch.Tensor): Style observations concatenated by FRAME, tensor with shape (batch_size, style_dim).

        Returns:
            torch.Tensor: Style observations concatenated by ITEM, tensor with shape (batch_size, style_dim).
        """
        style_obs_terms = {
            # "base_lin_vel": 3,
            # "base_ang_vel": 3,
            "joint_pos": 26,
            "joint_vel": 26,
            "eef_positions": 12,
        }
        style_obs_frame_size = sum(style_obs_terms.values())
        style_obs_frame_num = 2

        # Validate input shape: expect 2D tensor (batch, frame_num * frame_size)
        if style_obs.dim() != 2:
            raise ValueError(
                f"style_obs must be 2-D tensor with shape (batch, {style_obs_frame_num * style_obs_frame_size}), got shape {tuple(style_obs.shape)}"
            )

        batch_size, total_dim = style_obs.shape
        expected_dim = style_obs_frame_num * style_obs_frame_size
        if total_dim != expected_dim:
            raise ValueError(
                f"style_obs last-dim must be {expected_dim} (frame_num * frame_size={style_obs_frame_num}*{style_obs_frame_size}), got {total_dim}"
            )

        # reshape to (batch, frame_num, frame_size)
        # use reshape to be robust to non-contiguous tensors
        style_obs_reshaped = style_obs.reshape(batch_size, style_obs_frame_num, style_obs_frame_size)

        # Swap the order of frames: originally concatenated by FRAME (frame0|frame1),
        # reverse to (frame1|frame0) along the frame dimension before ITEM reordering.
        # For frame_num==2 this simply flips the second dimension.
        # style_obs_reshaped = style_obs_reshaped.flip(dims=[1])
        # print("style_obs_reshaped_reverse frame: ", style_obs_reshaped)

        # For each item term in a frame, collect its values across frames and
        # concatenate frames for that item (i.e., ITEM-concatenated).
        parts = []
        offset = 0
        for term_name, term_dim in style_obs_terms.items():
            start = offset
            end = offset + term_dim
            # term across frames: shape (batch, frame_num, term_dim)
            term_across_frames = style_obs_reshaped[:, :, start:end]
            # flatten frames then features -> (batch, frame_num * term_dim)
            term_flat = term_across_frames.reshape(batch_size, style_obs_frame_num * term_dim)
            parts.append(term_flat)
            offset = end

        # Concatenate all item-blocks to form ITEM-concatenated vector
        reorganized = torch.cat(parts, dim=-1)

        # Ensure returned tensor is same device/dtype and shape (batch, style_dim)
        return reorganized

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
