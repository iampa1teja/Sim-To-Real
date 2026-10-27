"""Physical joint-frame anchors for the Molmo simulation adapter, CPU only.

These anchors are independent of the adapter's affine implementation. They
come from a shared-CAD-mesh alignment of the original Molmo wrist frame to the official SO-101
URDFs, matching base_motor_holder, rotation_pitch, upper_arm, under_arm,
wrist_roll_pitch, wrist_roll_follower and moving_jaw visual frames. The fitted
pan zero agrees within 0.001 degrees after aligning the physical base mesh;
the apparent 90-degree change of the *base coordinate frame* is not a motor
offset. Old motor degrees use a calibrated horizontal zero, independent of
the endpoints chosen for normalized commands.

Primary references:
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_old_calib.urdf
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
https://huggingface.co/docs/lerobot/main/en/backwardcomp#hardware-api-redesign
https://huggingface.co/docs/lerobot/main/en/molmoact2#joint-frame-transform-so-100101-zero-shot

No CAD package, renderer, hardware driver, torch or network access is needed:
the physical reference joint poses are frozen below, while both mapping
directions execute the actual production interface methods via AST loading.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]

JOINT_ORDER = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

# (physical USD arm degrees, old-convention Molmo motor degrees). In the
# horizontal old-zero pose, the USD lift is +90 and elbow -90. CAD's old URDF
# lift coordinate runs opposite to the old LeRobot motor convention, which is
# why that motor's correction has a sign flip. Gripper values remain percent.
PHYSICAL_ANCHORS = [
    pytest.param([0, 0, 0, 0, 90], [0, 90, 90, 0, 0], id="canonical-neutral"),
    pytest.param([0, 90, -90, 0, 90], [0, 0, 0, 0, 0], id="horizontal-old-zero"),
    pytest.param([30, -20, -30, 15, 70], [30, 110, 60, 15, -20], id="signed-bent-pose"),
    pytest.param([-35, 50, 50, -25, 140], [-35, 40, 140, -25, 50], id="opposite-quadrant"),
]


def _clamp(values, *, min, max):
    return np.clip(values, min, max)


@pytest.fixture
def interface():
    """Run the real two interface mapping methods without importing Isaac."""
    path = ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py"
    tree = ast.parse(path.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "LeRobotSO101Interface")
    wanted = {"get_mapped_actions_vectorized", "get_raw_actions_from_radians"}
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef)
               and node.name in wanted]
    assert {node.name for node in methods} == wanted
    bounds = next(ast.literal_eval(node.value) for node in source_class.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "SO101_USD_MAPPING"
                          for target in node.targets))
    test_class = ast.ClassDef(name="InterfaceMapping", bases=[], keywords=[],
                             body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(zeros_like=np.zeros_like, clamp=_clamp, pi=np.pi)}
    exec(compile(module, str(path), "exec"), namespace)
    iface = namespace["InterfaceMapping"]()
    names = [name.removesuffix(".pos") for name in JOINT_ORDER]
    iface.joint_mins = np.array([bounds[name]["joint_min"] for name in names])
    iface.joint_maxs = np.array([bounds[name]["joint_max"] for name in names])
    setup = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())
    mapping = {**bounds, **setup.get("sim_joint_mapping", {})}
    iface.mapping_joint_mins = np.array([mapping[name]["joint_min"] for name in names])
    iface.mapping_joint_maxs = np.array([mapping[name]["joint_max"] for name in names])
    return iface


def client_for_interface(iface, **overrides):
    from sim_to_real_so101.policy_client.contract import load_client_policy
    from sim_to_real_so101.policy_client.discovery import load_manifest
    span = iface.mapping_joint_maxs - iface.mapping_joint_mins
    midpoint = (iface.mapping_joint_maxs + iface.mapping_joint_mins) / 2
    options = dict(input_joint_units="usd_normalized",
                   arm_degrees_per_unit=",".join(str(value) for value in span[:5] / 200),
                   arm_zero_degrees=",".join(str(value) for value in midpoint[:5]),
                   wrist_roll_model_offset=-90)
    options.update(overrides)
    manifest = load_manifest(ROOT / "policies/molmoact2")
    return load_client_policy("molmoact2", manifest, options)


def gripper_angle(interface):
    return interface.mapping_joint_mins[-1] + .37 * (interface.mapping_joint_maxs[-1] - interface.mapping_joint_mins[-1])


def images():
    return {"room": np.zeros((2, 3, 3), np.uint8), "wrist": np.zeros((2, 3, 3), np.uint8)}


@pytest.mark.parametrize("usd_degrees,model_degrees", PHYSICAL_ANCHORS)
def test_actual_sim_observations_match_physical_frame_anchors(interface, usd_degrees, model_degrees):
    # The policy keeps percent gripper commands while the USD span follows calibration.
    physical_radians = np.deg2rad([*usd_degrees, gripper_angle(interface)])
    normalized = interface.get_raw_actions_from_radians(physical_radians)
    built = client_for_interface(interface).build_request(normalized, images(), "task")
    np.testing.assert_allclose(built["state"], [*model_degrees, 37], rtol=0, atol=2e-5)


@pytest.mark.parametrize("usd_degrees,model_degrees", PHYSICAL_ANCHORS)
def test_model_targets_drive_actual_sim_to_physical_frame_anchors(interface, usd_degrees, model_degrees):
    decoded = client_for_interface(interface).parse_response(
        {"actions": [[*model_degrees, 37]]}, JOINT_ORDER)[0]
    actual = interface.get_mapped_actions_vectorized(decoded)
    np.testing.assert_allclose(np.rad2deg(actual), [*usd_degrees, gripper_angle(interface)], rtol=0, atol=2e-5)


def test_asymmetric_sim_ranges_do_not_redefine_model_zero(interface):
    # Deliberately change the command-coordinate midpoints while keeping the
    # physical pose fixed. This would expose a mutually invertible but biased
    # build/parse transform, which a roundtrip-only test cannot detect.
    interface.mapping_joint_mins[:5] = [-140, -120, -80, -100, -175]
    interface.mapping_joint_maxs[:5] = [80, 80, 120, 90, 145]
    normalized = interface.get_raw_actions_from_radians(np.deg2rad([0, 0, 0, 0, 90, gripper_angle(interface)]))
    built = client_for_interface(interface).build_request(normalized, images(), "task")
    np.testing.assert_allclose(built["state"], [0, 90, 90, 0, 0, 37], rtol=0, atol=2e-5)


def test_clipped_native_pan_target_is_not_shifted_by_normalized_midpoint(interface):
    # Official SO101 action q99 is an absolute old-frame degree target, not an
    # angle to which the saved follower's +1.7578125 USD range midpoint is added.
    q99_pan = 48.55349563198916
    decoded = client_for_interface(interface).parse_response(
        {"actions": [[q99_pan, 90, 90, 0, 0, 0]]}, JOINT_ORDER)[0]
    physical = np.rad2deg(interface.get_mapped_actions_vectorized(decoded))
    np.testing.assert_allclose(physical[:5], [q99_pan, 0, 0, 0, 90], rtol=0, atol=2e-5)


def test_physical_usd_limits_still_clamp_large_model_targets(interface):
    decoded = client_for_interface(interface).parse_response(
        {"actions": [[200, 90, 90, 0, 0, 37]]}, JOINT_ORDER)[0]
    physical = np.rad2deg(interface.get_mapped_actions_vectorized(decoded))
    assert physical[0] == pytest.approx(110, abs=2e-5)


@pytest.mark.parametrize("zeros", ["", "0,0,0,0", "0,0,0,0,0,0", "0,0,nan,0,0", "0,inf,0,0,0", "bad,0,0,0,0"])
def test_sim_normalized_units_require_five_finite_zero_references(interface, zeros):
    with pytest.raises(ValueError, match="arm_zero_degrees"):
        client_for_interface(interface, arm_zero_degrees=zeros)


@pytest.mark.parametrize("scales", ["", "1,1,1,1", "1,1,1,1,1,1", "1,0,1,1,1", "1,1,nan,1,1", "1,inf,1,1,1"])
def test_sim_normalized_units_require_five_finite_nonzero_scales(interface, scales):
    with pytest.raises(ValueError, match="arm_degrees_per_unit"):
        client_for_interface(interface, arm_degrees_per_unit=scales)


def test_all_zero_midpoints_are_valid(interface):
    client = client_for_interface(interface, arm_zero_degrees="0,0,0,0,0")
    built = client.build_request(np.zeros(6), images(), "task")
    np.testing.assert_array_equal(built["state"], [0, 90, 90, 0, -90, 0])


def test_omitting_sim_zero_reference_is_rejected():
    from sim_to_real_so101.policy_client.contract import load_client_policy
    from sim_to_real_so101.policy_client.discovery import load_manifest
    manifest = load_manifest(ROOT / "policies/molmoact2")
    with pytest.raises(ValueError, match="arm_zero_degrees"):
        load_client_policy("molmoact2", manifest, {
            "input_joint_units": "usd_normalized", "arm_degrees_per_unit": "1,1,1,1,1",
        })


def test_real_normalized_client_keeps_midpoint_relative_degrees(interface):
    # The new simulation reconstruction must not change public real-client
    # modes, whose DEGREES zero is already the encoder calibration midpoint.
    client = client_for_interface(interface, input_joint_units="lerobot_normalized", wrist_roll_model_offset=0)
    built = client.build_request(np.zeros(6), images(), "task")
    np.testing.assert_array_equal(built["state"], [0, 90, 90, 0, 0, 0])


@pytest.mark.parametrize("value", ["bad", "nan", "inf", True])
def test_wrist_frame_offset_rejects_invalid_values(interface, value):
    with pytest.raises(ValueError, match="wrist_roll_model_offset"):
        client_for_interface(interface, wrist_roll_model_offset=value)


def test_original_molmo_frame_still_obeys_canonical_wrist_limit(interface):
    decoded = client_for_interface(interface).parse_response(
        {"actions": [[0, 90, 90, 0, 100, 37]]}, JOINT_ORDER)[0]
    physical = np.rad2deg(interface.get_mapped_actions_vectorized(decoded))
    assert physical[4] == pytest.approx(160, abs=2e-5)
