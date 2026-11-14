"""Differential checks against the installed LeRobot encoder normalization API."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.arms.calibration import follower_mapping_from_calibration, load_unit, with_device_codec
from utils.arms.mapping import JointMapping, physical_to_units, units_to_physical, mapping_for_arm, raw_joint_limits
from utils.arms.arms import load_arm_spec


@pytest.fixture
def so101(monkeypatch):
    monkeypatch.delenv("S2R_ARMS_DIR", raising=False)
    return load_arm_spec("so101")[0]


def sdk_bus(arm, calibration, *, apply_drive_mode):
    # Invoke the real SDK methods on data-only bus state. No serial object is created.
    from lerobot.motors.motors_bus import MotorNormMode
    names = arm.joint_names
    norm = {joint.name: MotorNormMode.RANGE_0_100 if joint.role == "gripper" else
            MotorNormMode.DEGREES if arm.lerobot.use_degrees else MotorNormMode.RANGE_M100_100
            for joint in arm.joints}
    return SimpleNamespace(
        calibration={name: SimpleNamespace(**row) for name, row in calibration.items()},
        motors={name: SimpleNamespace(norm_mode=norm[name]) for name in names},
        apply_drive_mode=apply_drive_mode,
        model_resolution_table={"test": arm.hardware.encoder_counts_per_turn},
        _id_to_name=lambda index: names[index], _id_to_model=lambda index: "test",
    )


@pytest.mark.parametrize("use_degrees", [False, True])
@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize("apply_drive_mode", [False, True])
def test_calibration_codec_matches_actual_sdk_encoder_normalization(so101, use_degrees, drive_mode, apply_drive_mode):
    from lerobot.motors.motors_bus import MotorsBus
    arm = replace(so101, lerobot=replace(so101.lerobot, use_degrees=use_degrees))
    unit = load_unit(so101, "rpd_robot")
    calibration = deepcopy(unit.source["joints"])
    for row in calibration.values():
        row["drive_mode"] = drive_mode
    physical = follower_mapping_from_calibration(calibration, arm, parameters=unit.parameters,
                                                  apply_drive_mode=apply_drive_mode)
    mapping = with_device_codec(arm, physical, calibration, apply_drive_mode=apply_drive_mode)
    # JSON persistence must retain the direction/zero convention, not just physical limits.
    mapping = mapping_for_arm(arm, json.loads(json.dumps(mapping)))
    bus = sdk_bus(arm, calibration, apply_drive_mode=apply_drive_mode)
    for fraction in (0.0, 0.17, 0.5, 0.81, 1.0):
        counts = {index: int(row["range_min"] + fraction * (row["range_max"] - row["range_min"]))
                  for index, row in enumerate(calibration.values())}
        encoded = MotorsBus._normalize(bus, counts)
        raw = [encoded[index] for index in range(len(arm.joints))]
        decoded = units_to_physical(raw, mapping, arm.joint_names, gripper_joints=arm.gripper_joint_names)
        expected = []
        for index, joint in enumerate(arm.joints):
            if joint.role == "arm":
                value = (counts[index] - arm.hardware.zero_count) * 360 / arm.hardware.encoder_counts_per_turn
                value += unit.parameters.get(joint.name, {}).get("offset", 0)
            else:
                anchor = unit.parameters[joint.name]
                span = (calibration[joint.name]["range_max"] - calibration[joint.name]["range_min"]) * 360 / arm.hardware.encoder_counts_per_turn
                sign = -1 if drive_mode and apply_drive_mode else 1
                value = anchor["closed"] + sign * (raw[index] - anchor.get("contact_units", 0)) * span / 100
            expected.append(value)
        np.testing.assert_allclose(decoded, expected, atol=1e-12, rtol=0)
        round_trip = physical_to_units(decoded, mapping, arm.joint_names, gripper_joints=arm.gripper_joint_names)
        np.testing.assert_allclose(round_trip, raw, atol=1e-12, rtol=0)
        # SDK command encoding truncates to integer counts; preserve its quantization bound.
        recovered = MotorsBus._unnormalize(bus, {index: value for index, value in enumerate(round_trip)})
        assert all(abs(recovered[index] - value) <= 1 for index, value in counts.items())
        torch_mapping = JointMapping.from_arm(arm, mapping)
        np.testing.assert_allclose(torch_mapping.to_physical(raw).numpy(), expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("contact", [0.0, 25.0, 100.0])
def test_reversed_gripper_closed_anchor_is_at_the_declared_raw_contact(so101, contact):
    unit = load_unit(so101, "rpd_robot")
    calibration = deepcopy(unit.source["joints"])
    calibration["gripper"]["drive_mode"] = 1
    parameters = deepcopy(unit.parameters)
    parameters["gripper"].update(closed=-7.0, contact_units=contact)
    physical = follower_mapping_from_calibration(calibration, so101, parameters=parameters, apply_drive_mode=True)
    mapping = with_device_codec(so101, physical, calibration, apply_drive_mode=True)
    decoded = units_to_physical([contact], {"gripper": mapping["gripper"]}, ["gripper"], gripper_joints=("gripper",))
    assert decoded[0] == pytest.approx(-7.0)
    assert raw_joint_limits(so101, mapping)["gripper"] == (0.0, 100.0)


def test_reversed_encoder_requires_explicit_bus_capability_or_endpoints(so101):
    unit = load_unit(so101, "rpd_robot")
    calibration = deepcopy(unit.source["joints"])
    calibration["shoulder_pan"]["drive_mode"] = 1
    with pytest.raises(ValueError, match="apply_drive_mode"):
        with_device_codec(so101, unit.mapping, calibration)
    explicit = deepcopy(unit.mapping)
    explicit["shoulder_pan"].update(raw_min=100.0, raw_max=-100.0)
    assert with_device_codec(so101, explicit, calibration)["shoulder_pan"] == explicit["shoulder_pan"]


def test_redundant_normalized_endpoints_canonicalize_without_losing_degree_codecs(so101):
    from dataclasses import replace
    mapping = deepcopy(load_unit(so101, "rpd_robot").mapping)
    explicit = deepcopy(mapping)
    for joint in so101.joints:
        explicit[joint.name].update(raw_min=0.0 if joint.role == "gripper" else -100.0, raw_max=100.0)
    assert mapping_for_arm(so101, explicit) == mapping_for_arm(so101, mapping)
    degree_arm = replace(so101, lerobot=replace(so101.lerobot, use_degrees=True))
    assert "raw_min" in mapping_for_arm(degree_arm, explicit)["shoulder_pan"]


def test_explicit_reversed_gripper_codec_resolves_raw_contact_without_bus_guess(so101, tmp_path):
    unit = load_unit(so101, "rpd_robot")
    source = deepcopy(unit.source)
    source["joints"]["gripper"]["drive_mode"] = 1
    physical = follower_mapping_from_calibration(source["joints"], so101, parameters=unit.parameters,
                                                 apply_drive_mode=True)
    physical["gripper"].update(raw_min=100.0, raw_max=0.0)
    arm = replace(so101, directory=tmp_path)
    directory = tmp_path / "calibration/units"
    directory.mkdir(parents=True)
    (directory / "custom.json").write_text(json.dumps(dict(schema_version=1, arm=arm.name, unit_id="custom",
                                                            mapping=physical, source=source)))
    loaded = load_unit(arm, "custom")
    decoded = units_to_physical([0.0], {"gripper": loaded.mapping["gripper"]}, ["gripper"], gripper_joints=["gripper"])
    assert decoded[0] == pytest.approx(unit.parameters["gripper"]["closed"])
