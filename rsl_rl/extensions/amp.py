# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial Motion Priors (AMP) discriminator — SCAFFOLD (default-off).

Ports the AMP framework from Hiking in the Wild §III-E (Peng et al. 2021):
a discriminator ``D(S)`` trained to tell reference motion transitions from
policy transitions, providing a "style reward" that shapes gait naturalness.

STATUS: this is an UNWIRED scaffold. The module implements the discriminator,
its least-squares loss (paper Eq. 7), the quadratic style reward (paper Eq. 8),
and a gradient penalty, but **no reference motion dataset exists** for Kuavo in
this repo, so AMP is never constructed by default (``PPO`` only builds it when
``amp_cfg`` is non-None, and no runner config passes one). See
``doc/amp_scaffold.md`` for the observation layout, dataset format, and how to
enable it once retargeted walk/run data is available.

Compatibility note: like symmetry augmentation, AMP is INCOMPATIBLE with the
DeFM encoder feature cache (mirrored/style-shaped observations cannot be aligned
with cached features). ``PPO`` raises if both are enabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class AMPDiscriminator(nn.Module):
    """Sequence discriminator for Adversarial Motion Priors.

    Consumes a flattened short state sequence ``S = [s_{t-n}, ..., s_t]`` and
    outputs a scalar logit. Trained with a least-squares (MSE) objective so the
    style reward has smooth, non-saturating gradients (paper §III-E).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        reward_scale: float = 1.0,
        grad_penalty_coeff: float = 10.0,
        weight_decay: float = 1.0e-4,
        learning_rate: float = 1.0e-3,
        device: str = "cpu",
    ) -> None:
        """Initialize the discriminator.

        Args:
            input_dim: flattened dimension of a state sequence ``S``.
            hidden_dims: discriminator trunk hidden sizes.
            activation: trunk activation.
            reward_scale: multiplier on the style reward (paper's task/style mix
                is applied by the caller).
            grad_penalty_coeff: coefficient for the gradient penalty on reference
                samples (training stability).
            weight_decay: L2 weight decay for the discriminator optimizer.
            learning_rate: discriminator optimizer learning rate.
            device: torch device.
        """
        super().__init__()
        self.device = device
        self.reward_scale = reward_scale
        self.grad_penalty_coeff = grad_penalty_coeff
        # Trunk outputs a single logit (MLP last layer -> 1).
        self.trunk = MLP(input_dim, 1, hidden_dims, activation).to(device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Return the discriminator logit ``D(S)`` for state sequences ``seq``."""
        return self.trunk(seq)

    def style_reward(self, policy_seq: torch.Tensor) -> torch.Tensor:
        """Quadratic AMP style reward (paper Eq. 8).

        ``r = max(0, 1 - 0.25 * (D(S) - 1)^2)``, scaled by ``reward_scale``.
        Returns a ``[B]`` tensor (no gradient; used for reward shaping).
        """
        with torch.no_grad():
            d = self.trunk(policy_seq).squeeze(-1)
            reward = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
        return self.reward_scale * reward

    def discriminator_loss(
        self, reference_seq: torch.Tensor, policy_seq: torch.Tensor
    ) -> torch.Tensor:
        """Least-squares AMP discriminator loss (paper Eq. 7) + gradient penalty.

        ``L_D = E_ref[(D(S) - 1)^2] + E_policy[(D(S) + 1)^2]`` plus a gradient
        penalty on the reference samples for stability.
        """
        ref_logit = self.trunk(reference_seq)
        pol_logit = self.trunk(policy_seq)
        loss = (ref_logit - 1.0).pow(2).mean() + (pol_logit + 1.0).pow(2).mean()
        if self.grad_penalty_coeff > 0.0:
            loss = loss + self.grad_penalty_coeff * self._gradient_penalty(reference_seq)
        return loss

    def _gradient_penalty(self, reference_seq: torch.Tensor) -> torch.Tensor:
        """Penalize the squared gradient norm of D w.r.t. reference inputs."""
        seq = reference_seq.detach().clone().requires_grad_(True)
        logit = self.trunk(seq)
        grad = torch.autograd.grad(
            outputs=logit,
            inputs=seq,
            grad_outputs=torch.ones_like(logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad.pow(2).sum(dim=-1).mean()
