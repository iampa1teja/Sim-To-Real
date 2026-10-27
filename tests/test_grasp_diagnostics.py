"""Measurements use jaw contact points, live frames, and independent episodes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from sim_to_real_so101.utils.grasp_diagnostics import GraspTrace, measure_grasp, _tip_offsets, _read_frame


def test_live_measurements_share_one_host_transfer_and_preserve_body_joint_order():
    transfers = []

    class DeviceFrame:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            transfers.append(self.value.size)
            return self

        def numpy(self):
            return self.value

    robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=np.arange(12, dtype=np.float32).reshape(1, 4, 3),
        body_quat_w=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        joint_pos=np.arange(6, dtype=np.float32)[None]))
    cube = np.array([.1, .2, .3], dtype=np.float32)
    command = np.arange(6, dtype=np.float32) + 10
    torch = SimpleNamespace(cat=lambda arrays: DeviceFrame(np.concatenate(arrays)))
    positions, quaternions, observed_cube, observed_command, measured = _read_frame(
        torch, robot, [3, 1], cube, command)
    assert transfers == [29]
    np.testing.assert_array_equal(positions, robot.data.body_pos_w[0, [3, 1]])
    np.testing.assert_array_equal(quaternions, robot.data.body_quat_w[0, [3, 1]])
    np.testing.assert_array_equal(observed_cube, cube)
    np.testing.assert_array_equal(observed_command, command)
    np.testing.assert_array_equal(measured, robot.data.joint_pos[0])


def test_grasp_gap_uses_closing_axis_and_error_uses_gripper_frame():
    q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    # Gripper X points along world Y. Moving contact has lateral/depth offset;
    # neither affects the width available along the actual closing axis.
    result = measure_grasp(np.array([1., 2., 3.]), q, np.array([.996, 2.024, 3.003]), q,
                           np.array([1., 2.010, 3.]), np.zeros(3), np.zeros(3), .020)
    assert result['jaw_gap_mm'] == pytest.approx(24)
    assert result['tool_error_mm'] == pytest.approx(0, abs=1e-10)
    assert result['cube_width_mm'] == 20


def test_trace_preserves_closest_closing_event_and_resets_previous_command():
    trace = GraspTrace()
    for error, command in ((50., 30.), (25., 20.), (8., 10.), (5., 10.)):
        trace.observe(dict(tool_error_mm=error), command)
    result = trace.summary()
    assert result['closest']['tool_error_mm'] == 5
    assert result['first_closing']['tool_error_mm'] == 25
    assert result['closest_closing']['tool_error_mm'] == 8
    assert result['closing_events'] == 1
    trace.command = object()
    trace.step = 9000
    trace.reset()
    assert trace.command is None and trace.step == 0
    trace.observe(dict(tool_error_mm=100), 0.)
    assert trace.summary()['closing_events'] == 0
    assert trace.summary()['closest_closing'] is None


def test_canonical_arm_cad_tip_offsets_use_sibling_body_transforms():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(ROOT / 'source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd'))
    root = '/so101_new_calib'
    fixed_tip, moving_tip = _tip_offsets(stage, root, .01)
    cache = UsdGeom.XformCache()
    positions, quaternions = [], []
    for body in ('gripper', 'jaw'):
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root + '/' + body))
        positions.append(np.asarray(matrix.ExtractTranslation()))
        quaternion = matrix.ExtractRotationQuat()
        quaternions.append(np.array([quaternion.GetReal(), *quaternion.GetImaginary()]))
    value = measure_grasp(positions[0], quaternions[0], positions[1], quaternions[1], positions[0],
                          fixed_tip, moving_tip, .02)
    assert value['jaw_gap_mm'] == pytest.approx(15.8, abs=.1)
    assert np.linalg.norm(fixed_tip) < .15
    assert np.linalg.norm(moving_tip) < .15
