# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP discriminator and S54 reference trajectory transition sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import autograd


class RunningMeanStd:
    """Running mean and variance estimator for discriminator observations."""

    def __init__(self, epsilon: float = 1e-4, shape: Union[int, Tuple[int, ...]] = ()) -> None:
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
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count


class Normalizer(RunningMeanStd):
    """Numpy-backed normalizer with a torch inference path."""

    def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0) -> None:
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: np.ndarray) -> np.ndarray:
        normalized = (input - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=device, dtype=input.dtype)
        std = torch.sqrt(torch.as_tensor(self.var + self.epsilon, device=device, dtype=input.dtype))
        return torch.clamp((input - mean) / std, -self.clip_obs, self.clip_obs)


class Discriminator(nn.Module):
    """AMP discriminator for state transitions ``concat(s_t, s_t+1)``."""

    def __init__(
        self,
        state_dim: Optional[int] = None,
        obs: Optional[dict[str, torch.Tensor]] = None,
        obs_groups: Optional[dict[str, Sequence[str]]] = None,
        device: Union[str, torch.device] = "cpu",
        amp_reward_coef: float = 0.3,
        amp_discr_hidden_dims: Sequence[int] = (1024, 512, 256),
        amp_use_normalization: bool = True,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.amp_reward_coef = amp_reward_coef

        self.obs_groups = obs_groups
        if state_dim is None and obs is not None and obs_groups is not None:
            state_dim = self._state_dim_from_obs(obs, obs_groups)
        if state_dim is None:
            raise ValueError("state_dim is required unless obs and obs_groups are provided.")

        self.state_dim = int(state_dim)
        self.input_dim = self.state_dim * 2

        layers: list[nn.Module] = []
        curr_dim = self.input_dim
        for hidden_dim in amp_discr_hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(self.device)
        self.amp_linear = nn.Linear(curr_dim, 1).to(self.device)

        self.discriminator_obs_normalization = amp_use_normalization
        if amp_use_normalization:
            self.discriminator_obs_normalizer = Normalizer(self.input_dim)
        else:
            self.discriminator_obs_normalizer = nn.Identity()

        print(f"Discriminator Network: \n{self.trunk}\n{self.amp_linear}")
        self.train()

    def _state_dim_from_obs(self, obs: dict[str, torch.Tensor], obs_groups: dict[str, Sequence[str]]) -> int:
        groups = obs_groups["discriminator"]
        state_dim = 0
        for obs_group in groups:
            if obs_group not in obs:
                raise KeyError(f"{obs_group} is not in obs")
            if len(obs[obs_group].shape) != 2:
                raise ValueError("The Discriminator module only supports 1D observations.")
            state_dim += obs[obs_group].shape[-1]
        return state_dim

    def _normalize(self, discriminator_obs: torch.Tensor) -> torch.Tensor:
        if self.discriminator_obs_normalization:
            return self.discriminator_obs_normalizer.normalize_torch(discriminator_obs, discriminator_obs.device)
        return self.discriminator_obs_normalizer(discriminator_obs)

    def forward(self, discriminator_obs: torch.Tensor) -> torch.Tensor:
        if discriminator_obs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected discriminator input dim {self.input_dim}, got {discriminator_obs.shape[-1]}")
        x = self._normalize(discriminator_obs)
        return self.amp_linear(self.trunk(x))

    def compute_grad_pen(self, discriminator_obs: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        discriminator_obs.requires_grad_(True)
        disc = self.forward(discriminator_obs)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=discriminator_obs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(
        self,
        discriminator_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            was_training = self.training
            self.eval()
            d = self.forward(discriminator_obs)
            amp_reward = self.amp_reward_coef * torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
            reward = amp_reward
            if was_training:
                self.train()
        return reward.squeeze(-1), amp_reward.squeeze(-1), d


class NpzStateTransitionDataset:
    """Random sampler for S54 reference trajectory ``(s_t, s_t+1)`` transitions."""

    DEFAULT_EEF_BODY_NAMES = ("leg_l6_link", "leg_r6_link", "zarm_l7_link", "zarm_r7_link")

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        device: Union[str, torch.device] = "cpu",
        eef_body_names: Sequence[str] = DEFAULT_EEF_BODY_NAMES,
        root_body_name: str = "base_link",
        dof_names: Sequence[str] | None = None,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(path) for path in paths]
        self.device = torch.device(device)
        self.eef_body_names = tuple(eef_body_names)
        self.root_body_name = root_body_name
        self.dof_names = tuple(dof_names) if dof_names is not None else None
        states = self._load_data(self.paths)
        self.state_dim = states[0].shape[-1]
        self.transitions = torch.cat([torch.cat((state[:-1], state[1:]), dim=-1) for state in states], dim=0)
        self.num_transitions = self.transitions.shape[0]
        print(
            f"Loaded {len(states)} trajectories from {len(self.paths)} npz files, "
            f"{self.num_transitions} transitions, state_dim={self.state_dim}"
        )

    def _load_data(self, paths: Sequence[Path]) -> list[torch.Tensor]:
        states: list[torch.Tensor] = []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                state = self._make_s54_state(data, path)
                if state.shape[0] < 2:
                    raise ValueError(f"Trajectory in {path} must contain at least 2 frames.")
                states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        if not states:
            raise ValueError("No valid trajectories were loaded from npz files.")
        first_dim = states[0].shape[-1]
        for state in states:
            if state.shape[-1] != first_dim:
                raise ValueError("All loaded trajectories must have the same state dimension.")
        return states

    def _make_s54_state(self, data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
        required = {"dof_positions", "dof_velocities", "dof_names", "body_positions", "body_names"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing required S54 reference trajectory keys: {missing}")

        q = np.asarray(data["dof_positions"])
        dot_q = np.asarray(data["dof_velocities"])
        q, dot_q = self._select_dofs(q, dot_q, data, path)

        body_positions = np.asarray(data["body_positions"])
        body_names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
        try:
            root_id = body_names.index(self.root_body_name)
            eef_ids = [body_names.index(name) for name in self.eef_body_names]
        except ValueError as exc:
            raise KeyError(
                f"{exc} in {path}. Available body_names: {body_names}"
            ) from exc

        eef_pos = body_positions[:, eef_ids, :] - body_positions[:, root_id : root_id + 1, :]
        return self._make_state(q, dot_q, eef_pos, path, "")

    def _select_dofs(
        self,
        q: np.ndarray,
        dot_q: np.ndarray,
        data: np.lib.npyio.NpzFile,
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dof_names is None:
            return q, dot_q
        names = [str(name) for name in np.asarray(data["dof_names"]).tolist()]
        try:
            indices = [names.index(name) for name in self.dof_names]
        except ValueError as exc:
            raise KeyError(f"{exc} in {path}. Available dof_names: {names}") from exc
        return q[:, indices], dot_q[:, indices]

    def _make_state(self, q: np.ndarray, dot_q: np.ndarray, eef_pos: np.ndarray, path: Path, prefix: str) -> np.ndarray:
        arrays = [self._flatten_time_array(arr) for arr in (q, dot_q, eef_pos)]
        length = arrays[0].shape[0]
        if any(arr.shape[0] != length for arr in arrays):
            raise ValueError(f"q, dot_q, and eef_pos lengths do not match for trajectory {prefix!r} in {path}")
        return np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)

    def _flatten_time_array(self, array: np.ndarray) -> np.ndarray:
        if array.ndim < 2:
            raise ValueError(f"Expected array with shape (T, D...), got {array.shape}")
        return array.reshape(array.shape[0], -1)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = self.sample_amp(batch_size)
        return transitions[:, : self.state_dim], transitions[:, self.state_dim :]

    def sample_amp(self, batch_size: int) -> torch.Tensor:
        indices = torch.randint(self.num_transitions, (batch_size,), device=self.device)
        return self.transitions[indices]

    def feed_forward_generator_amp(
        self,
        batch_size: int,
        num_batches: Optional[int] = None,
        return_pairs: bool = False,
    ) -> Iterator[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        batch_id = 0
        while num_batches is None or batch_id < num_batches:
            state, next_state = self.sample(batch_size)
            if return_pairs:
                yield state, next_state
            else:
                yield torch.cat((state, next_state), dim=-1)
            batch_id += 1


class ExpertLoader:
    """Small wrapper matching AMP code paths that expect ``loader.dataset``."""

    def __init__(self, dataset: NpzStateTransitionDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    @classmethod
    def from_npz(
        cls,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> "ExpertLoader":
        return cls(NpzStateTransitionDataset(paths, device=device, **kwargs), batch_size)
