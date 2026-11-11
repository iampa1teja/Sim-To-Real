"""Typed arm contracts validate names and are selected at runtime."""
from dataclasses import replace
import tomllib

import pytest

from core.arms.spec import _parse_spec, load_arm_spec, load_arm_utils


def test_so101_owns_order_limits_and_optional_mount_camera(monkeypatch):
    monkeypatch.delenv("S2R_ARMS_DIR", raising=False)
    arm, directory = load_arm_spec("so101")
    assert arm.directory == directory
    assert arm.name == "so101"
    assert arm.joint_name_pairs[0] == ("shoulder_pan", "Rotation")
    assert arm.gripper_joint_names == ("gripper",)
    assert arm.cameras["wrist_cam"].parent_link == arm.links.ee
    assert arm.feature_names[-1] == "gripper.pos"
    assert arm.initial_joint_positions["Pitch"] == -0.6109
    assert arm.runtime_joint_limits["shoulder_pan"]["joint_min"] == -112.88476679176092
    assert load_arm_spec("so101", "no_camera")[0].usd_path.suffix == ".usda"


def test_four_joint_parallel_gripper_without_roll_or_camera(synthetic_arm, monkeypatch):
    assert synthetic_arm.joint_names == ("pan", "left", "slide", "right")
    assert synthetic_arm.gripper_joint_names == ("left", "right")
    assert synthetic_arm.wrist_roll is None
    assert synthetic_arm.cameras == {}
    assert [joint.unit for joint in synthetic_arm.joints] == ["deg", "m", "m", "m"]
    monkeypatch.setenv("S2R_ARM", "synthetic")
    assert load_arm_spec()[0].name == "synthetic"
    monkeypatch.setenv("S2R_ARM", "unknown")
    with pytest.raises(ValueError, match="Cannot load"):
        load_arm_spec()


@pytest.mark.parametrize("change,error", [
    (lambda data: data.update(name="wrong"), "folder name"),
    (lambda data: data["joints"][1].update(name="pan"), "Duplicate joint name"),
    (lambda data: data["joints"][1].update(usd="Pan"), "Duplicate joint usd"),
    (lambda data: data["joints"][0].update(type="continuous"), "joint type"),
    (lambda data: data["joints"][0].update(limits=[1, -1]), "increasing"),
    (lambda data: data["joints"][0].update(initial=20), "outside"),
    (lambda data: data["gripper"].update(joints=["left"]), "match exactly"),
    (lambda data: data.update(wrist_roll={"joint": "slide"}), "revolute arm joint"),
    (lambda data: data.update(wrist_roll={"joint": "left"}), "revolute arm joint"),
    (lambda data: data["actuators"]["drive"].update(joint_names_expr=["Unknown"]), "matches no USD joint"),
])
def test_invalid_contracts_fail_clearly(synthetic_arm, change, error):
    raw = tomllib.loads((synthetic_arm.directory / "arm.toml").read_text())["arm"]
    change(raw)
    with pytest.raises(ValueError, match=error):
        _parse_spec("synthetic", synthetic_arm.directory, raw)


def test_optional_utils_loads_by_explicit_path(synthetic_arm):
    path = synthetic_arm.directory / "synthetic_utils.py"
    path.write_text("def prepare_collisions(root):\n    root.append('prepared')\n")
    arm = replace(synthetic_arm, utils=path)
    hooks = load_arm_utils(arm)
    root = []
    hooks.prepare_collisions(root)
    assert root == ["prepared"]
    assert load_arm_utils(synthetic_arm) is None


@pytest.mark.parametrize('ik,error', [
    ({'orientation_axes': ['roll']}, 'select x, y'),
    ({'orientation_axes': ['x', 'x']}, 'duplicate'),
    ({'position_offset_m': [0, 0]}, 'three|3 values'),
    ({'method': 'pinv'}, 'ik.method must be dls or adaptive_dls'),
    ({'joint_limit_avoidance_gain': -0.1}, 'gain must be nonnegative'),
    ({'joint_limit_avoidance_gain': float('nan')}, 'gain must be a finite number'),
    ({'joint_limit_avoidance_margin': 0.0}, 'margin must be positive'),
    ({'joint_limit_avoidance_margin': float('inf')}, 'margin must be a finite number'),
    ({'joint_limit_avoidance_gain': 0.1}, 'no spare joint for limit avoidance'),
])
def test_invalid_ik_contract(synthetic_arm, ik, error):
    raw = tomllib.loads((synthetic_arm.directory / 'arm.toml').read_text())['arm']
    raw['ik'] = ik
    with pytest.raises(ValueError, match=error):
        _parse_spec('synthetic', synthetic_arm.directory, raw)


def test_optional_ik_defaults(synthetic_arm):
    assert synthetic_arm.ik.method == 'dls'
    assert synthetic_arm.ik.joint_limit_avoidance_gain == 0.0
    assert synthetic_arm.ik.joint_limit_avoidance_margin == 0.3


def test_ik_settings_with_a_spare_arm_joint(synthetic_arm):
    raw = tomllib.loads((synthetic_arm.directory / 'arm.toml').read_text())['arm']
    raw['joints'].extend(dict(name=f'arm_{index}', usd=f'Arm_{index}', type='revolute', role='arm',
                              limits=[-180.0, 180.0], initial=0.0) for index in range(5))
    raw['ik'] = dict(orientation_axes=['x', 'y', 'z'], position_offset_m=[0.01, 0.02, 0.03],
                     method='adaptive_dls', joint_limit_avoidance_gain=0.2, joint_limit_avoidance_margin=0.4)
    arm = _parse_spec('synthetic', synthetic_arm.directory, raw)
    assert sum(joint.role == 'arm' for joint in arm.joints) == 7
    assert arm.ik.orientation_axes == ('x', 'y', 'z')
    assert arm.ik.position_offset_m == (0.01, 0.02, 0.03)
    assert arm.ik.method == 'adaptive_dls'
    assert arm.ik.joint_limit_avoidance_gain == 0.2
    assert arm.ik.joint_limit_avoidance_margin == 0.4
    raw['joints'].pop()
    with pytest.raises(ValueError, match='no spare joint for limit avoidance'):
        _parse_spec('synthetic', synthetic_arm.directory, raw)
