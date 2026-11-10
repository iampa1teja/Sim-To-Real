"""Per-unit calibration is independent of scenes and supports non-SO101 layouts."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.arms.calibration import (
    factory_mapping, follower_mapping_from_calibration, load_unit, resolve_calibration,
)
from core.arms.spec import load_arm_spec
from core.arms.mapping import JointMapping
from core.datasets.alignment import recording_contract, validate_recording_root, RecordingAlignmentError, MARKER


def test_homing_warning_uses_selected_encoder_resolution():
    from core.calibration.check import static_check

    calibration = {'joint': dict(range_min=0, range_max=1000, homing_offset=3000)}
    low = SimpleNamespace(hardware=SimpleNamespace(encoder_counts_per_turn=4096))
    high = SimpleNamespace(hardware=SimpleNamespace(encoder_counts_per_turn=8192))
    assert static_check(calibration, {}, arm=low)[0]['status'] == 'WARN'
    assert static_check(calibration, {}, arm=high)[0]['status'] == 'PASS'


def test_production_units_preserve_distinct_measured_sources(monkeypatch):
    monkeypatch.delenv("S2R_ARMS_DIR", raising=False)
    arm, _ = load_arm_spec("so101")
    embedded, recorded = load_unit(arm, "rpd_robot"), load_unit(arm, "rpd_follower")
    assert embedded.source["follower_id"] == "rpd_robot"
    assert embedded.source["calibration_sha256"] == "86c6eee0eb97435082d5d6b820dafc3c4ad774b424ad84105c32cad96365be03"
    assert recorded.source["calibration_sha256"] == "7705cc34a847e7dd99a9549e9022fc164b640ddcf90f7a807d35fe0e579f7c26"
    assert embedded.mapping["gripper"] == dict(joint_min=-10.000003814697266, joint_max=119.63866806030273)
    assert recorded.mapping["gripper"] == dict(joint_min=-10.000003814697266, joint_max=124.29687118530273)
    for unit in (embedded, recorded):
        assert follower_mapping_from_calibration(unit.source["joints"], arm, parameters=unit.parameters) == unit.mapping
    with pytest.raises(ValueError, match="requires a calibration unit"):
        resolve_calibration(arm, environ={})


def test_factory_mapping_uses_encoder_range_and_declared_prismatic_stroke(synthetic_arm):
    selection = resolve_calibration(synthetic_arm, environ={})
    assert selection.unit is None
    assert selection.mapping == factory_mapping(synthetic_arm)
    assert selection.mapping["pan"] == dict(joint_min=-180.0, joint_max=179.912109375)
    assert selection.mapping["slide"] == dict(joint_min=-0.2, joint_max=0.2)
    assert selection.mapping["left"] == dict(joint_min=0.0, joint_max=0.04)
    assert selection.provenance["mapping_source"] == "factory"
    with pytest.raises(ValueError, match="does not use unit"):
        resolve_calibration(synthetic_arm, unit="invalid")


@pytest.fixture
def measured_unit(synthetic_arm):
    arm = replace(synthetic_arm, lerobot=replace(synthetic_arm.lerobot, calibration="per_unit"))
    source = {
        "parameters": {"left": {"metres_per_count": 0.0001}, "right": {"metres_per_count": 0.0001},
                       "slide": {"metres_per_count": 0.0001}},
        "joints": {"pan": dict(range_min=1024, range_max=3072), "left": dict(range_min=2048, range_max=2448),
                   "slide": dict(range_min=48, range_max=4048), "right": dict(range_min=2048, range_max=2448)},
    }
    mapping = follower_mapping_from_calibration(source["joints"], arm, parameters=source["parameters"])
    directory = arm.directory / "calibration/units"
    directory.mkdir(parents=True)
    path = directory / "physical-1.json"
    path.write_text(json.dumps(dict(schema_version=1, arm=arm.name, unit_id="physical-1", mapping=mapping, source=source)))
    return arm, path, source, mapping


def test_unit_metadata_and_mapping_survive_no_roll_parallel_gripper(measured_unit):
    arm, path, source, expected = measured_unit
    unit = load_unit(arm, "physical-1")
    assert unit.mapping == expected
    assert unit.source == source
    assert unit.parameters == source["parameters"]
    selection = resolve_calibration(arm, unit="physical-1", environ={})
    mapping = JointMapping.from_arm(arm, selection)
    np.testing.assert_allclose(mapping.to_sim_np([100, 100, 100, 100]), [np.pi/2, 0.04, 0.2, 0.04])
    assert selection.provenance["unit_calibration"]["path"] == str(path)


def test_unit_requires_explicit_linear_count_conversion(measured_unit):
    arm, _, source, _ = measured_unit
    with pytest.raises(ValueError, match="metres_per_count"):
        follower_mapping_from_calibration(source["joints"], arm)


def test_loaded_device_and_unit_files_are_verified_without_hardware(measured_unit, tmp_path):
    arm, unit_path, source, _ = measured_unit
    calibration = tmp_path / "follower.json"
    calibration.write_text(json.dumps(source["joints"]))
    selection = resolve_calibration(arm, unit="physical-1", follower_path=calibration, environ={})
    device = SimpleNamespace(calibration_fpath=calibration, calibration=source["joints"])
    selection.verify_loaded(device, "follower")
    modified = json.loads(calibration.read_text())
    modified["pan"]["range_max"] -= 1
    device.calibration = modified
    with pytest.raises(ValueError, match="in-memory calibration"):
        selection.verify_loaded(device, "follower")
    device.calibration = source["joints"]
    unit_path.write_text(unit_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Unit mapping changed"):
        selection.verify_loaded(device, "follower")


def test_new_recording_marker_is_self_describing_and_legacy_resume_is_refused(synthetic_arm, tmp_path):
    selection = resolve_calibration(synthetic_arm, environ={})
    marker = recording_contract(follower_id=None, leader_id=None, calibration_selection=selection)
    assert marker["joint_mapping"] == selection.mapping
    assert marker["arm"] == synthetic_arm.name
    assert marker["joint_names"] == list(synthetic_arm.joint_names)
    assert marker["joint_units"] == ["deg", "m", "m", "m"]
    assert not marker["sim_wrist_relative"]
    root = tmp_path / "dataset"
    path = root / MARKER
    path.parent.mkdir(parents=True)
    old = dict(version=3, calibration_sha256="old")
    path.write_text(json.dumps(old))
    with pytest.raises(RecordingAlignmentError):
        validate_recording_root(root, marker)
    assert json.loads(path.read_text()) == old


@pytest.mark.parametrize("role", ["leader", "follower"])
@pytest.mark.parametrize("field", ["range_min", "range_max", "homing_offset"])
def test_each_device_in_memory_calibration_matches_loaded_file(measured_unit, tmp_path, role, field):
    from copy import deepcopy
    arm, _, source, _ = measured_unit
    saved = deepcopy(source["joints"])
    for index, joint in enumerate(arm.joints):
        saved[joint.name].update(id=index + 1, drive_mode=0, homing_offset=0)
    path = tmp_path / (role + ".json")
    path.write_text(json.dumps(saved))
    selection = resolve_calibration(arm, unit="physical-1", environ={}, **{role + "_path": path})
    device = SimpleNamespace(calibration_fpath=path, calibration=deepcopy(saved))
    selection.verify_loaded(device, role)
    device.calibration[arm.joint_names[0]][field] += 1
    with pytest.raises(ValueError, match="in-memory calibration"):
        selection.verify_loaded(device, role)


def test_loaded_follower_drive_direction_cannot_change_under_same_physical_span(measured_unit, tmp_path):
    from copy import deepcopy
    arm, unit_path, source, mapping = measured_unit
    payload = json.loads(unit_path.read_text())
    payload["source"]["apply_drive_mode"] = True
    for row in payload["source"]["joints"].values():
        row["drive_mode"] = 0
    unit_path.write_text(json.dumps(payload))
    calibration = deepcopy(payload["source"]["joints"])
    calibration["pan"]["drive_mode"] = 1
    path = tmp_path / "different-drive.json"
    path.write_text(json.dumps(calibration))
    selection = resolve_calibration(arm, unit="physical-1", environ={})
    device = SimpleNamespace(calibration_fpath=path, calibration=calibration)
    with pytest.raises(ValueError, match="raw device codec"):
        selection.verify_loaded(device, "follower")
