"""Typed arm contracts validate names and are selected at runtime."""
from dataclasses import replace
import tomllib

import pytest

from utils.arms.spec import _parse_spec, load_arm_spec, load_arm_utils


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


def _with_mimic(synthetic_arm):
    raw = tomllib.loads((synthetic_arm.directory / 'arm.toml').read_text())['arm']
    raw['joints'][3]['mimic'] = dict(joint='left', multiplier=0.5, offset=0.01)
    raw['gripper']['joints'] = ['left']
    raw['gripper'].update(closed_gap_m=0.0, open_gap_m=0.08)
    raw['import'] = dict(version='test', report='import-report.json')
    return raw


def test_mimics_are_physical_joints_but_never_codec_or_action_axes(synthetic_arm):
    import numpy as np
    from utils.arms.mapping import JointMapping, raw_joint_limits
    from utils.arms.calibration import validate_mapping

    arm = _parse_spec('synthetic', synthetic_arm.directory, _with_mimic(synthetic_arm))
    assert arm.joint_names == ('pan', 'left', 'slide')
    assert arm.usd_joint_names == ('Pan', 'Left', 'Slide', 'Right')
    assert arm.action_usd_joint_names == ('Pan', 'Left', 'Slide')
    assert arm.mimic_joints == ((3, 1, 0.5, 0.01),)
    assert arm.gripper_joint_names == ('left',)
    assert arm.initial_joint_positions['Right'] == 0.02
    mapping = JointMapping.from_arm(arm)
    assert mapping.names == arm.joint_names
    assert set(raw_joint_limits(arm, {})) == set(arm.joint_names)
    assert set(validate_mapping(arm, arm.runtime_joint_limits)) == set(arm.joint_names)
    np.testing.assert_allclose(mapping.to_sim_np([0, 50, 0]), [0, 0.02, 0])
    assert arm.provenance['version'] == 'test'


@pytest.mark.parametrize('change,error', [
    (lambda raw: raw['joints'][3]['mimic'].update(joint='missing'), 'missing source'),
    (lambda raw: raw['joints'][1].update(mimic=dict(joint='right')), 'must not itself be a mimic'),
    (lambda raw: raw['joints'][3]['mimic'].update(joint='pan'), 'types must match'),
    (lambda raw: raw['joints'][3].update(role='arm'), 'only gripper mimics'),
    (lambda raw: raw['joints'][3]['mimic'].update(offset=float('nan')), 'finite'),
    (lambda raw: raw['joints'][3].update(initial=0.03), 'initial must follow'),
])
def test_invalid_mimics(synthetic_arm, change, error):
    raw = _with_mimic(synthetic_arm)
    change(raw)
    with pytest.raises(ValueError, match=error):
        _parse_spec('synthetic', synthetic_arm.directory, raw)


def test_gap_maps_include_reversed_open_direction_and_clamp(synthetic_arm):
    raw = _with_mimic(synthetic_arm)
    gripper = _parse_spec('synthetic', synthetic_arm.directory, raw).gripper
    assert gripper.gap_m(0.02) == pytest.approx(0.04)
    assert gripper.q_for_gap(0.04) == pytest.approx(0.02)
    assert gripper.q_for_gap(100) == pytest.approx(0.04)
    assert gripper.q_for_gap(-1) == 0.0
    raw['gripper'].update(closed_position=0.04, open_position=0.0)
    gripper = _parse_spec('synthetic', synthetic_arm.directory, raw).gripper
    assert gripper.q_for_gap(0.08) == pytest.approx(0.0)
    assert gripper.gap_m(0.04) == 0.0


def test_simulation_manifest_needs_no_hardware_driver(synthetic_arm):
    raw = _with_mimic(synthetic_arm)
    del raw['lerobot'], raw['hardware']
    arm = _parse_spec('synthetic', synthetic_arm.directory, raw)
    assert not arm.lerobot.use_degrees
    assert arm.lerobot.follower_type == 'unconfigured'


def test_simulation_codec_ignores_hardware_placeholders(synthetic_arm):
    from utils.arms.calibration import resolve_sim_calibration,resolve_calibration
    from utils.arms.spec import LeRobotSpec
    arm=replace(synthetic_arm,lerobot=LeRobotSpec('unconfigured','unconfigured',calibration='simulation'))
    selection=resolve_sim_calibration(arm,unit='unrelated_config_default')
    assert selection.mapping==arm.runtime_joint_limits
    assert selection.provenance['mapping_source']=='simulation_manifest'
    with pytest.raises(ValueError,match='simulation-only'):
        resolve_calibration(arm)
