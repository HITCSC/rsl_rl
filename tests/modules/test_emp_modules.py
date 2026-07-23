"""Tests for EMP teacher observation and action ordering."""

import torch

from rsl_rl.modules.emp_modules import (
    S45_LAB_TO_MJCF,
    S45_MJCF_TO_LAB,
    joint_order_mjcf_to_lab_term_major,
)


def test_s45_joint_permutations_are_inverses():
    mjcf = torch.arange(26)
    lab = mjcf[list(S45_MJCF_TO_LAB)]

    assert torch.equal(lab[list(S45_LAB_TO_MJCF)], mjcf)


def test_term_major_joint_conversion_reorders_each_history_frame():
    proprio = torch.zeros(1, 420)
    expected = proprio.clone()
    source = torch.arange(5 * 26, dtype=torch.float).view(1, 5, 26)

    for start in (30, 160, 290):
        proprio[:, start : start + 130] = source.reshape(1, -1)
        expected[:, start : start + 130] = source[..., list(S45_MJCF_TO_LAB)].reshape(1, -1)

    converted = joint_order_mjcf_to_lab_term_major(proprio)

    assert torch.equal(converted, expected)
