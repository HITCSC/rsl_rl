# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

# Matches nav_distillation_cfg.AmpObsCfg concatenated order in recorded expert .txt files.
DEFAULT_AMP_OBS_LAYOUT: dict[str, Any] = {
    "height_scan_dim": 63,
    "height_scan_shape": [7, 9],
    "base_lin_vel_dim": 3,
    "base_ang_vel_dim": 3,
    "joint_pos_dim": 26,
    "joint_vel_dim": 26,
    "eef_pos_dim": 12,
    # Which terms are fed to the AMP discriminator / replay buffer.
    # None means use every term below (full recorded vector).
    "amp_active_terms": None,
}

AMP_OBS_TERM_SPECS: tuple[tuple[str, str], ...] = (
    ("height_scan", "height_scan_dim"),
    ("base_lin_vel", "base_lin_vel_dim"),
    ("base_ang_vel", "base_ang_vel_dim"),
    ("joint_pos", "joint_pos_dim"),
    ("joint_vel", "joint_vel_dim"),
    ("eef_positions", "eef_pos_dim"),
)

# Convenience preset: joint + eef only (no height scan / base velocities).
PROPRIO_ONLY_AMP_ACTIVE_TERMS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "eef_positions",
)


def resolve_amp_obs_layout(layout_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve AmpObs field layout, full file dim, and active discriminator dim."""
    layout = deepcopy(DEFAULT_AMP_OBS_LAYOUT)
    if layout_cfg:
        layout.update(layout_cfg)

    height_scan_dim = int(layout["height_scan_dim"])
    height_scan_shape = tuple(int(v) for v in layout["height_scan_shape"])
    if height_scan_shape[0] * height_scan_shape[1] != height_scan_dim:
        raise ValueError(
            f"height_scan_shape {height_scan_shape} does not match height_scan_dim={height_scan_dim}."
        )
    layout["height_scan_shape"] = height_scan_shape

    active_terms = layout.get("amp_active_terms")
    if active_terms is None:
        active_terms = [name for name, _ in AMP_OBS_TERM_SPECS]
    else:
        active_terms = list(active_terms)

    unknown = set(active_terms) - {name for name, _ in AMP_OBS_TERM_SPECS}
    if unknown:
        raise ValueError(f"Unknown amp_active_terms: {sorted(unknown)}")

    active_indices: list[int] = []
    proprio_dim = 0
    offset = 0
    for term_name, dim_key in AMP_OBS_TERM_SPECS:
        dim = int(layout[dim_key])
        if term_name in active_terms:
            active_indices.extend(range(offset, offset + dim))
            if term_name != "height_scan":
                proprio_dim += dim
        offset += dim

    if not active_indices:
        raise ValueError("amp_active_terms must include at least one observation term.")

    layout["amp_active_terms"] = active_terms
    layout["proprio_dim"] = proprio_dim
    layout["full_obs_dim"] = offset
    layout["active_obs_dim"] = len(active_indices)
    layout["active_indices"] = active_indices
    # Backward-compatible alias used by discriminators / replay buffer.
    layout["obs_dim"] = layout["active_obs_dim"]
    return layout


def slice_amp_obs(obs: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    """Select active AMP terms from a full recorded AmpObs vector."""
    active_indices = layout.get("active_indices")
    if active_indices is None:
        layout = resolve_amp_obs_layout(layout)
        active_indices = layout["active_indices"]

    if len(active_indices) == layout["full_obs_dim"]:
        return obs

    index_tensor = torch.tensor(active_indices, device=obs.device, dtype=torch.long)
    return obs.index_select(-1, index_tensor)


def slice_amp_obs_numpy(obs, layout: dict[str, Any]):
    """Numpy version for motion loader preprocessing."""
    import numpy as np

    active_indices = layout["active_indices"]
    obs = np.asarray(obs)
    if len(active_indices) == layout["full_obs_dim"]:
        return obs
    return obs[..., active_indices]
