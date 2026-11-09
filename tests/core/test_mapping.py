"""Calibrated units preserve physical spans, independent of joint count/order."""
import numpy as np
import pytest
import torch

from core.arms.mapping import JointMapping, degrees_to_units, units_to_degrees


NAMES = ("pan", "left_finger", "elbow", "right_finger")
GRIPPERS = ("left_finger", "right_finger")
SPANS = {
    "pan": {"joint_min": -90.0, "joint_max": 180.0},
    "left_finger": {"joint_min": -10.0, "joint_max": 20.0},
    "elbow": {"joint_min": -180.0, "joint_max": 90.0},
    "right_finger": {"joint_min": -20.0, "joint_max": 10.0},
}


def test_explicit_joint_order_and_two_gripper_spans():
    # Grippers are interleaved and have arbitrary names; no positional sixth-joint rule.
    units = [[-100, 0, -100, 0], [0, 50, 0, 50], [100, 100, 100, 100]]
    expected = [[-90, -10, -180, -20], [45, 5, -45, -5], [180, 20, 90, 10]]
    angles = units_to_degrees(units, SPANS, NAMES, gripper_joints=GRIPPERS)
    np.testing.assert_equal(angles, expected)
    np.testing.assert_equal(degrees_to_units(angles, SPANS, NAMES, gripper_joints=GRIPPERS), units)


def test_unclamped_conversion_preserves_out_of_range_values_and_full_turns():
    spans = {"pan": {"joint_min": -180.0, "joint_max": 180.0}}
    np.testing.assert_equal(units_to_degrees([200], spans), [360])
    np.testing.assert_equal(degrees_to_units([720], spans), [400])


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_tensor_conversion_round_trip(dtype):
    values = torch.tensor([[-100, 0, 0, 25], [100, 75, 100, 100]], dtype=dtype)
    angles = units_to_degrees(values, SPANS, NAMES, gripper_joints=GRIPPERS)
    assert angles.device == values.device
    assert angles.is_floating_point()
    recovered = degrees_to_units(angles, SPANS, NAMES, gripper_joints=GRIPPERS)
    torch.testing.assert_close(recovered, values.to(dtype=recovered.dtype), rtol=0, atol=0)


def test_mapping_applies_runtime_clamp_after_calibrated_conversion():
    low = torch.tensor([-90.0, -10.0, -180.0, -20.0])
    high = torch.tensor([180.0, 20.0, 90.0, 10.0])
    joint_mapping = JointMapping(
        NAMES, low, high,
        clamp_mins=torch.tensor([-45.0, -5.0, -90.0, -10.0]),
        clamp_maxs=torch.tensor([90.0, 15.0, 45.0, 5.0]),
        gripper_mask=torch.tensor([False, True, False, True]),
    )
    units = np.array([[-100.0, 0.0, -100.0, 0.0], [100.0, 100.0, 100.0, 100.0]])
    expected = np.deg2rad([[-45.0, -5.0, -90.0, -10.0], [90.0, 15.0, 45.0, 5.0]])
    np.testing.assert_allclose(joint_mapping.to_radians(units).numpy(), expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(joint_mapping.to_radians_np(units), expected, rtol=0, atol=0)
    # Encoding has no runtime clamp: out-of-range physical observations stay observable.
    np.testing.assert_allclose(joint_mapping.to_raw(torch.deg2rad(high)), [100, 100, 100, 100], atol=1e-5)


@pytest.mark.parametrize("span", [(0, 0), (10, 0), (float("nan"), 10), (0, float("inf"))])
def test_rejects_invalid_calibration_bounds(span):
    mapping = {"pan": dict(joint_min=span[0], joint_max=span[1])}
    with pytest.raises(ValueError, match="finite and increasing"):
        units_to_degrees([0], mapping)
    with pytest.raises(ValueError, match="finite and increasing"):
        degrees_to_units([0], mapping)


def test_rejects_wrong_joint_dimension():
    with pytest.raises(ValueError, match="4 joint values"):
        units_to_degrees([0, 1], SPANS, NAMES, gripper_joints=GRIPPERS)
