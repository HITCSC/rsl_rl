# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EMP teacher model modules for teacher-student distillation.

Implements :class:`PolicyHeightMapCNN` and :class:`ActorCriticCNN` matching the
checkpoint structure from the EMP repository (``doc/model_48350.pt``). Used by
:class:`~rsl_rl.models.EMPTeacherModel` to run the frozen teacher during
distillation training.

The checkpoint uses the following key hierarchy::

    height_map_cnn.conv2.*       — PolicyHeightMapCNN
    height_map_cnn.fc.*
    height_map_cnn.layernorm.*
    actor.*                      — Actor MLP (MLP(487→512→256→128→26, ELU))
    actor_obs_normalizer.*       — EmpiricalNormalization(423)
    (critic/precise keys are ignored during distillation)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


# ---------------------------------------------------------------------------
# Proprioception ordering helpers
# ---------------------------------------------------------------------------

# MJLab / Isaac Lab ObservationManager with ``concatenate_terms=True +
# flatten_history_dim=True`` groups terms in **term-major** order::
#
#     [term1_h0..h4, term2_h0..h4, ..., term5_h0..h4]
#
# The EMP teacher was trained on **term-major** order directly from Isaac Lab's
# observation manager.  **Do NOT** convert to frame-major before the normalizer:
# the checkpoint's ``actor_obs_normalizer._mean`` was accumulated on term-major
# data; applying it to frame-major indices would subtract the wrong per-dim
# mean and break alignment.
#
# Each frame (one history step) contains 84 dims::
#
#     base_ang_vel(3) + projected_gravity(3) + joint_pos(26) +
#     joint_vel(26) + actions(26)

_TERM_DIMS = [3, 3, 26, 26, 26]   # per-term single-frame dimension
_HIST_LEN = 5                      # number of history frames (teacher)


def joint_order_mjcf_to_lab_term_major(proprio: torch.Tensor) -> torch.Tensor:
    """Convert joint order inside term-major proprio from MJCF to checkpoint USD order.

    The input ``[B, 420]`` term-major tensor has the structure::

        [base_ang_vel(15), gravity(15), joint_pos(130), joint_vel(130), actions(130)]

    where each term group contains 5 history frames concatenated along the last
    dim.  The three joint groups (index 2, 3, 4) are reordered from MJCF order
    ``[l1..l6, r1..r6, zl1..zl7, zr1..zr7]`` to the checkpoint's USD
    order using :data:`S45_MJCF_TO_LAB`.

    Non-joint dims (base_ang_vel, gravity) pass through unchanged.

    Args:
        proprio: ``[B, 420]`` — term-major proprio (native output of the
            ObservationManager with ``concatenate_terms=True``,
            ``flatten_history_dim=True``, ``history_length=5``).

    Returns:
        ``[B, 420]`` — term-major proprio with joints in checkpoint USD order,
        matching the distribution the checkpoint normalizer was trained on.
    """
    B = proprio.shape[0]
    splits = [d * _HIST_LEN for d in _TERM_DIMS]  # [15, 15, 130, 130, 130]
    term_groups = list(torch.split(proprio, splits, dim=-1))

    idx = torch.tensor(S45_MJCF_TO_LAB, device=proprio.device, dtype=torch.long)
    # Terms 2, 3, 4 = joint_pos, joint_vel, actions — each [B, 130] = [B, 5 * 26].
    for i in [2, 3, 4]:
        tg = term_groups[i].view(B, _HIST_LEN, -1)  # [B, 5, 26]
        term_groups[i] = tg[..., idx].reshape(B, -1)  # [B, 130]

    return torch.cat(term_groups, dim=-1)  # [B, 420]


# ---------------------------------------------------------------------------
# Legacy helpers: frame-major reorder + joint reorder (kept for reference)
#
# These two are designed to work as a PAIR:
#   1. reorder_proprio_to_frame_major (term → frame)
#   2. joint_order_mjcf_to_lab       (frame-major joint reorder)
#
# They are NOT used in teacher inference because the normalizer was trained on
# term-major data.  Keep them only if a caller explicitly needs frame-major
# ordering for analysis or debugging.
# ---------------------------------------------------------------------------


def reorder_proprio_to_frame_major(proprio: torch.Tensor) -> torch.Tensor:
    """Convert a 420-dim proprioception observation from term-major to
    frame-major ordering.

    .. note::
        The checkpoint normalizer was trained on **term-major** data.
        Do *not* call this function before the normalizer unless you also
        re-estimate the normalizer statistics for frame-major ordering.

    Args:
        proprio: ``[B, 420]`` — term-major (grouped by term, then by history).

    Returns:
        ``[B, 420]`` — frame-major (grouped by history frame, each frame
        containing all 5 terms in the standard EMP order).
    """
    B = proprio.shape[0]

    # Split into 5 term groups, each [B, hist_len * term_dim].
    splits = [d * _HIST_LEN for d in _TERM_DIMS]
    term_groups = torch.split(proprio, splits, dim=-1)

    # Reshape each to [B, hist_len, term_dim].
    terms_3d = [g.view(B, _HIST_LEN, d) for g, d in zip(term_groups, _TERM_DIMS)]

    # Reassemble frame-wise: for each history slot, concat all terms.
    frames = [torch.cat([t[:, h, :] for t in terms_3d], dim=-1) for h in range(_HIST_LEN)]

    return torch.cat(frames, dim=-1)  # [B, 420]


# ---------------------------------------------------------------------------
# Joint ordering: MJCF (grouped) ↔ Lab (interleaved left-right)
# ---------------------------------------------------------------------------

# S45 has 26 joints: 6 left leg + 6 right leg + 7 left arm + 7 right arm.
S45_NUM_JOINTS = 26

# MJCF order (MuJoCo qpos, the env's native order):
#   [leg_l1..leg_l6, leg_r1..leg_r6, zarm_l1..zarm_l7, zarm_r1..zarm_r7]
# USD/Lab order used by the EMP checkpoint:
#   [leg_l1, leg_r1, zarm_l1, zarm_r1, leg_l2, leg_r2, zarm_l2, zarm_r2,
#    ..., leg_l6, leg_r6, zarm_l6, zarm_r6, zarm_l7, zarm_r7]
#
# ``S45_MJCF_TO_LAB[i]`` = the MJCF index that goes to Lab slot i.
# ``S45_LAB_TO_MJCF[i]`` = the Lab index that goes to MJCF slot i.

S45_MJCF_TO_LAB = (
    0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25,
)
S45_LAB_TO_MJCF = (
    0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14,
    18, 22, 24, 3, 7, 11, 15, 19, 23, 25,
)


def joint_order_mjcf_to_lab(proprio: torch.Tensor) -> torch.Tensor:
    """Convert the joint dimensions inside frame-major proprioception from
    MJCF (grouped) order to the checkpoint's USD order.

    Operates on the 420-dim frame-major output of
    :func:`reorder_proprio_to_frame_major`.  Within each 84-dim frame, the
    three 26-dim joint blocks (joint_pos, joint_vel, actions) are reordered
    from grouped → interleaved.  The non-joint dims (base_ang_vel, gravity)
    pass through unchanged.

    Args:
        proprio: ``[B, 420]`` — frame-major proprio (output of
            :func:`reorder_proprio_to_frame_major`).

    Returns:
        ``[B, 420]`` — same shape, joint dims reordered to checkpoint USD order.
    """
    B = proprio.shape[0]
    # Reshape to [B, 5, 84] = [batch, frame, per-frame dim].
    frames = proprio.view(B, _HIST_LEN, -1)  # [B, 5, 84]
    # Within each 84-dim frame: [base_ang_vel(3), gravity(3),
    #                            joint_pos(26), joint_vel(26), actions(26)].
    # Non-joint dims (0-5) pass through; dims 6-83 (3×26) get reordered.
    idx = torch.tensor(S45_MJCF_TO_LAB, device=proprio.device, dtype=torch.long)
    # Split: non-joint (6) and joint section (78 = 3 joint groups × 26 dofs).
    non_joint = frames[..., :6]  # [B, 5, 6]
    joint_section = frames[..., 6:]  # [B, 5, 78]
    # Reshape joint section to [B, 5, 3, 26] so ``idx`` is applied per-group.
    joint_3d = joint_section.view(B, _HIST_LEN, 3, -1)  # [B, 5, 3, 26]
    joint_reordered = joint_3d[..., idx]  # [B, 5, 3, 26]
    joint_flat = joint_reordered.reshape(B, _HIST_LEN, -1)  # [B, 5, 78]
    return torch.cat([non_joint, joint_flat], dim=-1).view(B, -1)  # [B, 420]


class PolicyHeightMapCNN(nn.Module):
    """Height-map CNN processing a 7×9 height scan grid.

    Architecture:
        Conv2d(1→32, k=3, s=1, p=1) + LeakyReLU(0.2)
        Conv2d(32→64, k=3, s=2, p=1) + LeakyReLU(0.2)
        AdaptiveAvgPool2d(1)
        FC(64 → output_dim) + LayerNorm(output_dim)

    The ``conv2`` Sequential module name matches the checkpoint's
    ``height_map_cnn.conv2.*`` key prefix.
    """

    def __init__(self, H: int = 9, W: int = 7, output_dim: int = 64) -> None:
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            height_map: Tensor of shape ``[B, 63, T]`` where T is the temporal
                dimension (1 for single-frame inference).

        Returns:
            Tensor of shape ``[B, T, output_dim]`` — height-map features.
        """
        # --- pre-processing (matches EMP ActorCriticCNN forward) ---
        # Per-sample min subtraction (makes the scan translation-invariant).
        min_height = height_map.min(dim=1, keepdim=True).values
        height_map = height_map - min_height

        # Reshape the flat 63-dim scan into a 2-D grid.
        # The raw sensor output is 187 = 11(W) × 17(L); the teacher uses the
        # center-front 7(W) × 9(L) crop → 63.
        h = height_map.permute(0, 2, 1)  # [B, T, 63]
        B, T = h.shape[:2]
        h = h.reshape(B * T, 1, 7, 9)  # [B*T, 1, 7, 9]
        h = h.permute(0, 1, 3, 2).contiguous()  # → [B*T, 1, 9, 7] (H, W)

        # --- CNN forward ---
        h = self.conv2(h)  # [B*T, 64, 5, 4]
        h = self.adaptive_pool(h).flatten(1)  # [B*T, 64]
        h = self.fc(h)  # [B*T, output_dim]
        h = self.layernorm(h)  # [B*T, output_dim]

        # Restore batch and temporal dimensions.
        return h.view(B, T, -1)  # [B, T, output_dim]


class ActorCriticCNN(nn.Module):
    """EMP Actor-Critic network with a height-map CNN encoder.

    Only the **actor** branch is materialised; the critic and precise height-map
    CNN are discarded. This is sufficient for teacher-student distillation since
    only teacher actions are needed.

    The key structure loaded from the checkpoint::

        actor_obs_normalizer  — EmpiricalNormalization(423)
        height_map_cnn        — PolicyHeightMapCNN
        actor                 — MLP(487 → 512 → 256 → 128 → 26, activation="elu")
    """

    def __init__(self, num_actions: int = 26, obs_dim: int = 423, cnn_out: int = 64) -> None:
        super().__init__()
        self.actor_obs_normalizer = EmpiricalNormalization(obs_dim)
        self.height_map_cnn = PolicyHeightMapCNN(output_dim=cnn_out)
        # Actor MLP input = 423 (normalised) + 64 (CNN features) = 487.
        self.actor = MLP(obs_dim + cnn_out, num_actions, (512, 256, 128), activation="elu")

    def forward(self, teacher_input: torch.Tensor) -> torch.Tensor:
        """Forward pass — produce teacher actions from the 486-dim flat input.

        Args:
            teacher_input: Tensor ``[B, 486]`` — concatenation of
                ``[command(3), policy_obs(5×84=420), height_scan(63)]``.

        Returns:
            Actions ``[B, num_actions]`` — 26-dim joint target positions
            (pre-scaling; scale factor 0.25 is applied at the action term level).
        """
        # Split: first 423 dims → normalizer; last 63 dims → CNN.
        norm_obs = self.actor_obs_normalizer(teacher_input[:, :423])  # [B, 423]
        cnn_feat = self.encode_height(teacher_input[:, 423:])
        final_input = torch.cat([norm_obs, cnn_feat], dim=-1)  # [B, 487]
        return self.actor(final_input)  # [B, num_actions]

    def encode_height(self, height_scan: torch.Tensor) -> torch.Tensor:
        """Encode the 63-D privileged terrain scan into the teacher latent."""
        # Unsqueeze temporal dim → [B, 63, 1] for PolicyHeightMapCNN.
        return self.height_map_cnn(height_scan.unsqueeze(-1)).squeeze(1)  # [B, 64]

    def forward_with_latent(self, teacher_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher actions and the terrain representation that produced them."""
        norm_obs = self.actor_obs_normalizer(teacher_input[:, :423])
        cnn_feat = self.encode_height(teacher_input[:, 423:])
        actions = self.actor(torch.cat([norm_obs, cnn_feat], dim=-1))
        return actions, cnn_feat


def build_teacher(checkpoint_path: str, device: str = "cpu") -> ActorCriticCNN:
    """Construct and load the teacher model from an EMP checkpoint.

    Args:
        checkpoint_path: Path to ``model_X.pt`` containing the ``model_state_dict``.
        device: Target device (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        Loaded :class:`ActorCriticCNN` in evaluation mode.
    """
    net = ActorCriticCNN()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Filter out keys not related to the actor branch (critic, precise, log_std, etc.).
    state_dict = checkpoint["model_state_dict"]
    prefix_filter = ("actor.", "height_map_cnn.", "actor_obs_normalizer.")
    actor_keys = {k: v for k, v in state_dict.items() if k.startswith(prefix_filter)}

    missing, unexpected = net.load_state_dict(actor_keys, strict=False)
    if missing:
        print(f"[build_teacher] Missing keys (ignored): {missing}")
    if unexpected:
        print(f"[build_teacher] Unexpected keys (ignored): {unexpected}")

    # The normalizer statistics are part of the trained actor. Inference must
    # preserve them exactly; changing their scale changes the actor input domain.
    norm = net.actor_obs_normalizer
    print(
        f"[build_teacher] Normalizer count={norm.count.item():,}, "
        f"mean∈[{norm._mean.min():.3f}, {norm._mean.max():.3f}], "
        f"std∈[{norm._std.min():.6f}, {norm._std.max():.6f}]"
    )

    net.eval()
    return net.to(device)
