# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training-only imagined-foothold guidance for SSR policies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP


class _FootholdReplayBuffer:
    """CPU replay buffer for labels that become available at future touchdown."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.states = torch.empty(self.capacity, state_dim, dtype=torch.float32)
        self.actions = torch.empty(self.capacity, action_dim, dtype=torch.float32)
        self.targets = torch.empty(self.capacity, 2, dtype=torch.float32)
        self.feet = torch.empty(self.capacity, dtype=torch.long)
        self.position = 0
        self.size = 0

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        feet: torch.Tensor,
    ) -> None:
        count = states.shape[0]
        if count == 0:
            return
        if count > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            targets = targets[-self.capacity :]
            feet = feet[-self.capacity :]
            count = self.capacity
        indices = (torch.arange(count) + self.position) % self.capacity
        # Advanced indexing returns a temporary tensor, so ``.copy_()`` would
        # leave the replay storage untouched. Assignment dispatches index_put_.
        self.states[indices] = states
        self.actions[indices] = actions
        self.targets[indices] = targets
        self.feet[indices] = feet
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(
        self, batch_size: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(self.size, (batch_size,))
        return (
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.targets[indices].to(device),
            self.feet[indices].to(device),
        )


class ImaginedFoothold(nn.Module):
    """Predict future touchdown distributions and turn them into SSR guidance.

    The predictor is privileged and training-only.  For every swing-foot step it
    stores ``(s_t, a_t)`` until the next first-contact event supplies the actual
    sole-center position.  The policy never consumes the predicted footholds.

    ``geometry_group`` has the fixed layout documented by
    ``mdp.ssr_foothold_geometry``:

    ``root_xy, root_yaw_cos_sin, feet_xy_w, contacts, first_contacts, terrain_level``.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_envs: int,
        state_group: str,
        terrain_group: str,
        geometry_group: str,
        map_size: tuple[float, float],
        map_resolution: float,
        sole_size: tuple[float, float],
        sole_resolution: float,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        learning_rate: float = 5.0e-4,
        replay_capacity: int = 32768,
        batch_size: int = 1024,
        updates_per_iteration: int = 4,
        max_pending_steps: int = 32,
        train_min_samples: int = 1024,
        reward_min_samples: int = 4096,
        reward_min_updates: int = 100,
        reward_min_terrain_level: float = 5.0,
        reward_weight: float = 0.25,
        reward_variance: float = 0.0625,
        height_threshold: float = 0.03,
        stable_contact_max_deficiency: float = 0.25,
        min_std: float = 0.02,
        max_std: float = 0.25,
        max_target_distance: float = 1.5,
        step_dt: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.num_envs = int(num_envs)
        self.state_group = state_group
        self.terrain_group = terrain_group
        self.geometry_group = geometry_group
        self.map_size = tuple(float(v) for v in map_size)
        self.map_resolution = float(map_resolution)
        self.map_nx = round(self.map_size[0] / self.map_resolution) + 1
        self.map_ny = round(self.map_size[1] / self.map_resolution) + 1
        self.num_map_points = self.map_nx * self.map_ny
        self.reward_weight = float(reward_weight) * float(step_dt)
        self.reward_variance = float(reward_variance)
        self.height_threshold = float(height_threshold)
        self.stable_contact_max_deficiency = float(stable_contact_max_deficiency)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.max_target_distance = float(max_target_distance)
        self.batch_size = int(batch_size)
        self.updates_per_iteration = int(updates_per_iteration)
        self.train_min_samples = int(train_min_samples)
        self.reward_min_samples = int(reward_min_samples)
        self.reward_min_updates = int(reward_min_updates)
        self.reward_min_terrain_level = float(reward_min_terrain_level)
        self.max_pending_steps = int(max_pending_steps)

        self.state_normalizer = EmpiricalNormalization(self.num_states).to(device)
        self.predictor = MLP(
            self.num_states + self.num_actions, 6, hidden_dims, activation
        ).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.replay = _FootholdReplayBuffer(replay_capacity, self.num_states, self.num_actions)

        # A single state/action is shared by both feet at a given source step.
        # Delayed supervision lives on CPU to avoid a large persistent GPU buffer.
        pending_shape = (self.max_pending_steps, self.num_envs)
        self._pending_states = torch.empty(*pending_shape, self.num_states)
        self._pending_actions = torch.empty(*pending_shape, self.num_actions)
        self._pending_roots = torch.empty(*pending_shape, 4)
        self._pending_valid = torch.zeros(*pending_shape, 2, dtype=torch.bool)
        self._pending_cursor = 0
        self._last_reward = torch.zeros(self.num_envs, device=device)

        sole_nx = round(float(sole_size[0]) / float(sole_resolution)) + 1
        sole_ny = round(float(sole_size[1]) / float(sole_resolution)) + 1
        sole_x = torch.linspace(
            -sole_size[0] / 2, sole_size[0] / 2, sole_nx, device=device
        )
        sole_y = torch.linspace(
            -sole_size[1] / 2, sole_size[1] / 2, sole_ny, device=device
        )
        grid_x, grid_y = torch.meshgrid(sole_x, sole_y, indexing="xy")
        self.register_buffer(
            "sole_offsets", torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        )
        # Deterministic sigma points avoid injecting Monte-Carlo noise into PPO.
        self.register_buffer(
            "sigma_directions",
            torch.tensor(
                ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
                device=device,
            ),
        )
        self.register_buffer("predictor_updates", torch.zeros((), dtype=torch.long, device=device))

    def _state(self, obs: TensorDict) -> torch.Tensor:
        state = obs[self.state_group]
        if state.ndim != 2 or state.shape[-1] != self.num_states:
            raise ValueError(
                f"Foothold state '{self.state_group}' must be [B,{self.num_states}], got {tuple(state.shape)}."
            )
        return state

    def _distribution(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.state_normalizer(states)
        output = self.predictor(torch.cat((normalized, actions), dim=-1))
        mu = output[..., :4].reshape(-1, 2, 2)
        # A bounded sigmoid keeps uncertainty trainable at both limits. A
        # hard clamp would initialize near max_std and zero its gradient.
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(output[..., 4:])
        return mu, std

    @staticmethod
    def _world_to_source(
        target_w: torch.Tensor, source_root: torch.Tensor
    ) -> torch.Tensor:
        delta = target_w - source_root[..., :2]
        cos_yaw = source_root[..., 2]
        sin_yaw = source_root[..., 3]
        return torch.stack(
            (
                cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1],
                -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1],
            ),
            dim=-1,
        )

    def _support_deficiency(
        self, terrain: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate sole support for centers shaped ``[B, feet, samples, 2]``."""
        batch = terrain.shape[0]
        heights = terrain[:, : self.num_map_points].reshape(
            batch, 1, self.map_ny, self.map_nx
        )
        valid_map = terrain[:, self.num_map_points :].reshape(
            batch, 1, self.map_ny, self.map_nx
        )
        points = centers.unsqueeze(-2) + self.sole_offsets.view(1, 1, 1, -1, 2)
        norm_x = 2.0 * points[..., 0] / self.map_size[0]
        norm_y = 2.0 * points[..., 1] / self.map_size[1]
        sample_grid = torch.stack((norm_x, norm_y), dim=-1)
        output_shape = sample_grid.shape[1:-1]
        sample_grid = sample_grid.reshape(batch, -1, 1, 2)
        sampled_height = F.grid_sample(
            heights, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(batch, *output_shape)
        sampled_valid = F.grid_sample(
            valid_map, sample_grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).reshape(batch, *output_shape) > 0.5
        highest_ground = sampled_height.masked_fill(~sampled_valid, float("inf")).amin(
            dim=-1, keepdim=True
        )
        supported = sampled_valid & (
            sampled_height - highest_ground < self.height_threshold
        )
        return 1.0 - supported.float().mean(dim=-1)

    @staticmethod
    def _feet_local(geometry: torch.Tensor) -> torch.Tensor:
        root_xy = geometry[:, 0:2]
        cos_yaw = geometry[:, 2:3]
        sin_yaw = geometry[:, 3:4]
        feet_w = geometry[:, 4:8].reshape(-1, 2, 2)
        delta = feet_w - root_xy.unsqueeze(1)
        return torch.stack(
            (
                cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1],
                -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1],
            ),
            dim=-1,
        )

    def observe_action(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Predict guidance for ``(s_t,a_t)`` and enqueue its delayed label."""
        states = self._state(obs)
        geometry = obs[self.geometry_group]
        terrain = obs[self.terrain_group]
        with torch.no_grad():
            self.state_normalizer.update(states)
            mu, std = self._distribution(states, actions)

            # Geometry root yaw is represented as cos/sin. Convert current sole
            # centers to the same root-yaw frame used by predictor outputs.
            feet_local = self._feet_local(geometry)
            contacts = geometry[:, 8:10] > 0.5

            stance_rho = self._support_deficiency(terrain, feet_local.unsqueeze(2)).squeeze(2)
            stable_contacts = contacts & (stance_rho <= self.stable_contact_max_deficiency)
            imagined_centers = mu.unsqueeze(2) + std[..., None, None] * self.sigma_directions.view(
                1, 1, -1, 2
            )
            swing_rho = self._support_deficiency(terrain, imagined_centers).mean(dim=2)
            deficiency = torch.where(stable_contacts, stance_rho, swing_rho)
            reward = torch.exp(-deficiency.sum(dim=1).square() / self.reward_variance)
            active = (
                (geometry[:, 12] >= self.reward_min_terrain_level)
                & (self.replay.size >= self.reward_min_samples)
                & (self.predictor_updates >= self.reward_min_updates)
            )
            self._last_reward = reward * active.float() * self.reward_weight

            cursor = self._pending_cursor
            self._pending_states[cursor].copy_(states.detach().cpu())
            self._pending_actions[cursor].copy_(actions.detach().cpu())
            self._pending_roots[cursor].copy_(geometry[:, :4].detach().cpu())
            self._pending_valid[cursor].copy_((~stable_contacts).detach().cpu())
            self._pending_cursor = (cursor + 1) % self.max_pending_steps
        return self._last_reward

    @property
    def last_reward(self) -> torch.Tensor:
        """Guidance reward cached for the action most recently sent to the env."""
        return self._last_reward

    def process_step(self, obs: TensorDict, dones: torch.Tensor) -> None:
        """Resolve pending swing samples when the next stable touchdown arrives."""
        with torch.no_grad():
            geometry_device = obs[self.geometry_group]
            terrain = obs[self.terrain_group]
            feet_local = self._feet_local(geometry_device)
            contacts = geometry_device[:, 8:10] > 0.5
            stance_rho = self._support_deficiency(terrain, feet_local.unsqueeze(2)).squeeze(2)
            stable_contacts = contacts & (stance_rho <= self.stable_contact_max_deficiency)

        done_cpu = dones.detach().view(-1).bool().cpu()
        stable_contacts = stable_contacts.detach().cpu()
        stable_contacts[done_cpu] = False
        geometry = geometry_device.detach().cpu()
        feet_w = geometry[:, 4:8].reshape(-1, 2, 2)

        for foot in range(2):
            landed = stable_contacts[:, foot]
            if not torch.any(landed):
                continue
            mask = self._pending_valid[:, :, foot] & landed.unsqueeze(0)
            states = self._pending_states[mask]
            actions = self._pending_actions[mask]
            roots = self._pending_roots[mask]
            target_w = feet_w[:, foot].unsqueeze(0).expand(self.max_pending_steps, -1, -1)[mask]
            targets = self._world_to_source(target_w, roots)
            finite = torch.isfinite(targets).all(dim=-1)
            in_range = torch.linalg.vector_norm(targets, dim=-1) <= self.max_target_distance
            keep = finite & in_range
            self.replay.add(
                states[keep],
                actions[keep],
                targets[keep],
                torch.full((int(keep.sum()),), foot, dtype=torch.long),
            )
            self._pending_valid[:, landed, foot] = False

        if torch.any(done_cpu):
            self._pending_valid[:, done_cpu] = False

    def update(self, is_multi_gpu: bool = False, world_size: int = 1) -> dict[str, float]:
        """Train the privileged predictor from completed touchdown pairs."""
        ready = self.replay.size >= self.train_min_samples
        if is_multi_gpu:
            ready_tensor = torch.tensor(int(ready), device=self.device)
            torch.distributed.all_reduce(ready_tensor, op=torch.distributed.ReduceOp.MIN)
            ready = bool(ready_tensor.item())
        if not ready:
            return {
                "foothold_replay_size": float(self.replay.size),
                "foothold_reward_active": float(
                    self.replay.size >= self.reward_min_samples
                    and self.predictor_updates.item() >= self.reward_min_updates
                ),
            }
        mean_loss = 0.0
        mean_error = 0.0
        mean_std = 0.0
        self.predictor.train()
        for _ in range(self.updates_per_iteration):
            states, actions, targets, feet = self.replay.sample(self.batch_size, self.device)
            mu, std = self._distribution(states, actions)
            row = torch.arange(states.shape[0], device=states.device)
            selected_mu = mu[row, feet]
            selected_std = std[row, feet]
            squared_error = (targets - selected_mu).square().sum(dim=-1)
            nll = squared_error / (2.0 * selected_std.square()) + 2.0 * torch.log(selected_std)
            loss = nll.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if is_multi_gpu:
                for parameter in self.predictor.parameters():
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                        parameter.grad /= world_size
            nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()
            self.predictor_updates += 1
            mean_loss += loss.item()
            mean_error += torch.sqrt(squared_error).mean().item()
            mean_std += selected_std.mean().item()
        scale = 1.0 / self.updates_per_iteration
        return {
            "foothold_nll": mean_loss * scale,
            "foothold_position_error": mean_error * scale,
            "foothold_std": mean_std * scale,
            "foothold_replay_size": float(self.replay.size),
            "foothold_reward_active": float(
                self.replay.size >= self.reward_min_samples
                and self.predictor_updates.item() >= self.reward_min_updates
            ),
        }


def resolve_foothold_config(
    alg_cfg: dict, obs: TensorDict, num_actions: int, num_envs: int, step_dt: float
) -> dict:
    """Resolve dimensions and timing for the optional imagined-foothold module."""
    cfg = alg_cfg.get("foothold_cfg")
    if cfg is None:
        alg_cfg["foothold_cfg"] = None
        return alg_cfg
    state_group = cfg["state_group"]
    terrain_group = cfg["terrain_group"]
    geometry_group = cfg["geometry_group"]
    if obs[state_group].ndim != 2:
        raise ValueError(f"Foothold state '{state_group}' must be a flat observation group.")
    map_size = cfg["map_size"]
    resolution = cfg["map_resolution"]
    num_map_points = (round(map_size[0] / resolution) + 1) * (
        round(map_size[1] / resolution) + 1
    )
    if obs[terrain_group].shape[-1] != 2 * num_map_points:
        raise ValueError(
            f"Foothold terrain '{terrain_group}' has width {obs[terrain_group].shape[-1]}, "
            f"expected {2 * num_map_points}."
        )
    if obs[geometry_group].shape[-1] != 13:
        raise ValueError(f"Foothold geometry '{geometry_group}' must have width 13.")
    cfg.update(
        num_states=obs[state_group].shape[-1],
        num_actions=num_actions,
        num_envs=num_envs,
        step_dt=step_dt,
    )
    return alg_cfg
