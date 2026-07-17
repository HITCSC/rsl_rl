# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.modules import MLP, HiddenState


class _MoEHead(nn.Module):
    """Gated mixture-of-experts feature mixer.

    Computes a softmax gate over ``num_experts`` expert MLPs and returns the
    gate-weighted sum of their outputs. All operations are plain tensor ops
    (no data-dependent control flow), so the module is TorchScript- and
    ONNX-traceable.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dims: tuple[int, ...] | list[int],
        gate_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate = MLP(input_dim, num_experts, gate_hidden_dims, activation)
        self.experts = nn.ModuleList(
            [MLP(input_dim, output_dim, expert_hidden_dims, activation) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the gate-weighted sum of expert outputs for input ``x``."""
        weights = torch.softmax(self.gate(x), dim=-1)  # [B, E]
        # [B, E, output_dim]
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Weighted sum over experts -> [B, output_dim]
        return (weights.unsqueeze(-1) * expert_outs).sum(dim=1)


class MoEModel(CNNModel):
    """Mixture-of-Experts policy model (Hiking in the Wild §I, MoE-Loco style).

    Extends :class:`CNNModel`: 2D depth groups are encoded by the (optionally
    shared) CNN encoders exactly as in the parent, producing a
    ``[1D‖cnn]`` latent. A gated mixture-of-experts head then mixes that latent
    into a fixed-width representation which the standard policy MLP
    (``self.mlp``) projects to the distribution parameters.

    Keeping ``self.mlp`` a plain sequential MLP means the output distribution's
    ``init_mlp_weights`` and the JIT/ONNX export path stay identical to the
    parent; the MoE only inserts a mixer in ``get_latent``. Encoder sharing
    (``cnns``) is inherited unchanged, so ``share_cnn_encoders`` works.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        moe_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the MoE model.

        Args:
            moe_cfg: Configuration of the mixture-of-experts head. Recognized
                keys (with defaults): ``num_experts`` (4), ``expert_hidden_dims``
                ((256,)), ``gate_hidden_dims`` ((64,)), ``moe_output_dim`` (256).
            All other args match :class:`CNNModel`.
        """
        moe_cfg = dict(moe_cfg or {})
        self._num_experts = int(moe_cfg.get("num_experts", 4))
        self._expert_hidden_dims = tuple(moe_cfg.get("expert_hidden_dims", (256,)))
        self._gate_hidden_dims = tuple(moe_cfg.get("gate_hidden_dims", (64,)))
        # Width of the mixed latent that feeds the shared policy MLP head.
        self._moe_output_dim = int(moe_cfg.get("moe_output_dim", 256))
        self._moe_activation = activation

        # Parent builds CNN encoders, self.mlp (input = self._get_latent_dim()),
        # and the distribution. self._get_latent_dim is overridden below to
        # return the MoE output width, so self.mlp consumes the mixed latent.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            cnn_cfg,
            cnns,
        )

        # Build the MoE mixer now that obs_dim + cnn_latent_dim are known.
        moe_input_dim = self.obs_dim + self.cnn_latent_dim
        self.moe = _MoEHead(
            input_dim=moe_input_dim,
            output_dim=self._moe_output_dim,
            num_experts=self._num_experts,
            expert_hidden_dims=self._expert_hidden_dims,
            gate_hidden_dims=self._gate_hidden_dims,
            activation=self._moe_activation,
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode with CNNs (parent), then mix through the MoE head."""
        encoder_latent = super().get_latent(obs, masks, hidden_state)
        return self.moe(encoder_latent)

    def _get_latent_dim(self) -> int:
        """The MLP head consumes the MoE-mixed latent."""
        return self._moe_output_dim

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMoEModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMoEModel(self, verbose)


class _TorchMoEModel(nn.Module):
    """Exportable MoE model for JIT."""

    def __init__(self, model: MoEModel) -> None:
        """Create a TorchScript-friendly copy of a MoEModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.moe = copy.deepcopy(model.moe)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent = self.moe(latent)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MoE exports)."""
        pass


class _OnnxMoEModel(nn.Module):
    """Exportable MoE model for ONNX."""

    def __init__(self, model: MoEModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a MoEModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.moe = copy.deepcopy(model.moe)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)
        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent = self.moe(latent)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
