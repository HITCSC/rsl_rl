# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_cnn import ActorCriticCNN, PolicyHeightMapCNN
from .actic_critic_depth_cnn import ActorCriticDepthCNN, PolicyDepthCNN, RMAEncoder
from .actor_critic_cnn_new import ActorCriticCNNNew
from .rnd import *
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import *
from .discriminator import *
from .multi_discriminator import MultiDiscriminator

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticCNN",
    "ActorCriticDepthCNN",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "PolicyDepthCNN",
    "PolicyHeightMapCNN",
    "RMAEncoder",
    "MultiDiscriminator",
]
