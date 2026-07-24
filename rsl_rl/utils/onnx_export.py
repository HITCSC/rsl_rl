# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX export utilities with deployment-order (frame-major) input reordering.

Provides :class:`ReorderObsOnnx` (wraps ONNX model to convert frame-major → term-major
1-D observations), :func:`build_history_interleave_perm` (computes the permutation),
and :func:`export_policy_to_onnx` (full export pipeline).

These are shared by :class:`~rsl_rl.runners.OnPolicyRunner` subclasses so that both
PPO-based runners and the distillation runner export policy.onnx with the same
deployment-compatible observation layout.
"""

from __future__ import annotations

import os
import math

import torch
import torch.nn as nn


class ReorderObsOnnx(nn.Module):
    """Wrap an ONNX-export model so its 1D ``obs`` input is time-major.

    Training concatenates each observation term's ``history_length`` frames
    contiguously, i.e. the exported ``obs`` vector is *term-major*::

        [A1, A2, A3, A4, A5, B1, B2, B3, B4, B5]

    (A/B are terms such as ``joint_pos`` / ``joint_vel``; digits are frames,
    oldest→newest). Deployment instead provides the history *frame-major*
    (interleaved)::

        [A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]

    This wrapper gathers the deployment-ordered ``obs`` back into the
    term-major order the trained normalizer / MLP expects, then delegates to
    the original export model. Any extra inputs (e.g. depth) pass through
    untouched, so the ONNX input/output signature is unchanged.
    """

    def __init__(self, inner: nn.Module, perm: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("perm", perm)

    def forward(self, obs: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        return self.inner(obs.index_select(-1, self.perm), *rest)

    def get_dummy_inputs(self):
        return self.inner.get_dummy_inputs()

    @property
    def input_names(self):
        return self.inner.input_names

    @property
    def output_names(self):
        return self.inner.output_names


def build_history_interleave_perm(obs_manager, policy) -> torch.Tensor | None:
    """Map a deploy-ordered (frame-major) 1-D observation vector to training order.

    Training lays out the actor 1-D observation as term-major, frame-major
    within each term. Deployment feeds it frame-major (all terms at frame 0,
    then all terms at frame 1, ...). This returns a permutation ``perm`` such
    that ``training_obs = deploy_obs.index_select(-1, perm)``, or ``None`` when
    the two layouts already coincide (e.g. no history / ``history_length==1``).

    The interleaving is applied per 1-D observation group and only when all
    terms in that group share the same ``history_length > 1`` (the case for the
    ``actor`` proprioception group, where a uniform 5-frame history is typical).
    Non-uniform or history-free groups are left as identity blocks.

    Args:
        obs_manager: The environment's observation manager (provides
            ``active_terms``, ``group_obs_term_dim``, ``get_term_cfg``).
        policy: The policy model (provides ``obs_groups`` — a list of
            1-D observation group names, e.g. ``["actor"]``).

    Returns:
        A 1-D permutation tensor, or ``None`` if no reordering is needed.
    """
    perm: list[int] = []
    offset = 0
    for group in policy.obs_groups:
        term_names = obs_manager.active_terms[group]
        term_dims = obs_manager.group_obs_term_dim[group]
        bases: list[int] = []
        histories: list[int] = []
        for name, dim in zip(term_names, term_dims):
            term_cfg = obs_manager.get_term_cfg(group, name)
            total = int(math.prod(dim))
            if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                hist = term_cfg.history_length
            else:
                hist = 1
            bases.append(total // hist)
            histories.append(hist)

        group_dim = sum(b * h for b, h in zip(bases, histories))
        uniform_hist = len(set(histories)) == 1
        if uniform_hist and histories and histories[0] > 1:
            hist = histories[0]
            per_frame = sum(bases)
            term_off = 0
            for base in bases:
                for f in range(hist):
                    for k in range(base):
                        # training index (term-major, frame-major within term)
                        # <- deploy index (frame-major, terms interleaved per frame)
                        perm.append(offset + f * per_frame + term_off + k)
                term_off += base
        else:
            # No history (or non-uniform): layouts already match.
            perm.extend(range(offset, offset + group_dim))
        offset += group_dim

    perm_t = torch.tensor(perm, dtype=torch.long)
    if torch.equal(perm_t, torch.arange(perm_t.numel())):
        return None
    return perm_t


def export_policy_to_onnx(
    obs_manager,
    policy: nn.Module,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export the policy to ONNX with a deploy-ordered (frame-major) obs input.

    Builds the ONNX model from ``policy.as_onnx()``, optionally wraps it in
    :class:`ReorderObsOnnx` when a frame-major → term-major permutation is
    required, and exports via ``torch.onnx.export`` with ``opset_version=18``
    and ``dynamo=False``.

    Metadata attachment (e.g. joint names, command names) is **not** handled
    here — callers should attach it after this function returns.

    Args:
        obs_manager: The environment's observation manager (used by
            :func:`build_history_interleave_perm`).
        policy: The policy model (must have ``as_onnx()``, ``obs_groups``).
        path: Directory to write the ONNX file into.
        filename: ONNX file name (default ``"policy.onnx"``).
        verbose: Passed to ``torch.onnx.export`` and ``policy.as_onnx()``.
    """
    onnx_model = policy.as_onnx(verbose=verbose)
    perm = build_history_interleave_perm(obs_manager, policy)
    if perm is not None:
        onnx_model = ReorderObsOnnx(onnx_model, perm)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
