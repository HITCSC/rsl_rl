# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store AMP state transitions."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str | torch.device):
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor):
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.states[sample_idxs], self.next_states[sample_idxs]


class StyleReplayBuffer(ReplayBuffer):
    """Replay buffer that stores style ids alongside AMP transitions."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str | torch.device):
        super().__init__(obs_dim, buffer_size, device)
        self.style_ids = torch.zeros(buffer_size, dtype=torch.long, device=device)

    def insert(self, states: torch.Tensor, next_states: torch.Tensor, style_ids: torch.Tensor):
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        style_ids = style_ids.view(-1).long()

        if end_idx > self.buffer_size:
            overflow = end_idx - self.buffer_size
            first_chunk = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:first_chunk]
            self.next_states[self.step : self.buffer_size] = next_states[:first_chunk]
            self.style_ids[self.step : self.buffer_size] = style_ids[:first_chunk]
            self.states[:overflow] = states[first_chunk:]
            self.next_states[:overflow] = next_states[first_chunk:]
            self.style_ids[:overflow] = style_ids[first_chunk:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states
            self.style_ids[start_idx:end_idx] = style_ids

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, style_id: int, num_mini_batch: int, mini_batch_size: int):
        style_id = int(style_id)
        valid_idxs = torch.where(self.style_ids[: self.num_samples] == style_id)[0]
        valid_count = valid_idxs.numel()

        for _ in range(num_mini_batch):
            if valid_count == 0:
                sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            elif valid_count < mini_batch_size:
                sample_idxs = valid_idxs[torch.randint(0, valid_count, (mini_batch_size,), device=self.device)]
            else:
                perm = torch.randperm(valid_count, device=self.device)[:mini_batch_size]
                sample_idxs = valid_idxs[perm]
            yield self.states[sample_idxs], self.next_states[sample_idxs]
