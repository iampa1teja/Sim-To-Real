"""Grasp telemetry compares commands and measured joints in manifest order."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab")

from tools.diagnostics.grasp_diagnostics import _read_frame, _tracking_errors


def test_permuted_joint_order_does_not_create_false_tracking_error():
    robot = SimpleNamespace(data=SimpleNamespace(
        body_link_pos_w=torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]),
        body_link_quat_w=torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]]),
        joint_pos=torch.tensor([[20., 40., 10., 30.]]),
    ))
    command = torch.tensor([10., 20., 30., 40.])
    positions, orientations, cube, target, measured = _read_frame(
        torch, robot, [0, 1], [2, 0, 3, 1], torch.tensor([7., 8., 9.]), command,
    )
    np.testing.assert_array_equal(target, measured)
    np.testing.assert_array_equal(positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(orientations, [[1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(cube, [7, 8, 9])


def test_mixed_joint_tracking_errors_keep_degrees_and_metres_separate():
    error = _tracking_errors([np.pi / 2, 0.012, 0.01], [0, 0.01, 0.01], [True, False, False])
    assert error["max_joint_tracking_error_deg"] == pytest.approx(90)
    assert error["max_joint_tracking_error_m"] == pytest.approx(0.002)
    assert _tracking_errors([0.02], [0.01], [False]) == {"max_joint_tracking_error_m": 0.01}
