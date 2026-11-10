"""Explicit arm-unit mappings and physical-device calibration provenance.

A unit file owns calibrated spans and optional per-joint anchors. Scene files are
never read here. Mapping values use degrees for revolute joints and metres for
prismatic joints; normalization is a separate concern in ``arms.mapping``.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path

from core.arms.spec import ArmSpec, load_arm_spec
from core.utils.hashing import file_sha256


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_mapping(arm: ArmSpec, mapping: dict) -> dict:
    if not isinstance(mapping, dict) or set(mapping) != set(arm.joint_names):
        raise ValueError(f"Mapping must specify exactly the arm joints {arm.joint_names}")
    result = {}
    for joint in arm.joints:
        row = mapping[joint.name]
        if not isinstance(row, dict):
            raise ValueError(f"Mapping for {joint.name} must contain joint_min and joint_max")
        low = _finite(row.get("joint_min"), f"{joint.name}.joint_min")
        high = _finite(row.get("joint_max"), f"{joint.name}.joint_max")
        if low >= high:
            raise ValueError(f"Mapping for {joint.name} must be increasing")
        result[joint.name] = dict(joint_min=low, joint_max=high)
        if "raw_min" in row or "raw_max" in row:
            raw_low = _finite(row.get("raw_min"), f"{joint.name}.raw_min")
            raw_high = _finite(row.get("raw_max"), f"{joint.name}.raw_max")
            if raw_low == raw_high:
                raise ValueError(f"Device codec for {joint.name} needs distinct raw endpoints")
            result[joint.name].update(raw_min=raw_low, raw_max=raw_high)
    return result


def raw_endpoints(mapping, joint_order, gripper_joints=()):
    """Oriented device endpoints, with normalized defaults for omitted metadata."""
    low, high = [], []
    for name in joint_order:
        span = mapping[name]
        if ("raw_min" in span) != ("raw_max" in span):
            raise ValueError(f"Device codec for {name} requires both raw_min and raw_max")
        a = _finite(span.get("raw_min", 0.0 if name in gripper_joints else -100.0), f"{name}.raw_min")
        b = _finite(span.get("raw_max", 100.0), f"{name}.raw_max")
        if a == b:
            raise ValueError(f"Device codec for {name} requires distinct raw endpoints")
        low.append(a)
        high.append(b)
    return tuple(low), tuple(high)


def with_device_codec(arm: ArmSpec, mapping: dict, calibration=None, *, apply_drive_mode=None):
    """Attach actual device endpoints without changing the physical encoder frame.

    SDK DEGREES is centred on measured encoder-range midpoint and uses resolution-1.
    It is not the arm's USD/physical zero. Normalized reversal depends on the bus
    capability, which a unit must declare when drive_mode is nonzero.
    """
    result = validate_mapping(arm, mapping)
    if apply_drive_mode is not None and type(apply_drive_mode) is not bool:
        raise ValueError("source.apply_drive_mode must be a boolean")
    if isinstance(calibration, (str, Path)):
        calibration = json.loads(Path(calibration).read_text())
    for joint in arm.joints:
        span = result[joint.name]
        if "raw_min" in span:
            continue
        row = (calibration or {}).get(joint.name, {})
        if arm.lerobot.use_degrees and joint.role == "arm":
            low = _finite(row.get("range_min"), f"{joint.name}.range_min for degree codec")
            high = _finite(row.get("range_max"), f"{joint.name}.range_max for degree codec")
            if low >= high:
                raise ValueError(f"Degree codec encoder span must increase for {joint.name}")
            midpoint = (low + high) / 2
            span.update(raw_min=(low - midpoint) * 360 / (arm.hardware.encoder_counts_per_turn - 1),
                        raw_max=(high - midpoint) * 360 / (arm.hardware.encoder_counts_per_turn - 1))
        elif row.get("drive_mode", 0):
            if apply_drive_mode is None:
                raise ValueError(f"{joint.name} drive_mode needs explicit source.apply_drive_mode or raw endpoints")
            if apply_drive_mode:
                span.update(raw_min=100.0, raw_max=0.0 if joint.role == "gripper" else -100.0)
    return result


def _source_parameters(arm, source):
    parameters = copy.deepcopy(source.get("parameters", {}))
    if not isinstance(parameters, dict):
        raise ValueError("Unit source.parameters must be a per-joint object")
    # Existing SO101 evidence uses scalar labels. Resolve them through declared
    # roles, never a hard-coded motor name or position in the action vector.
    if "gripper_closed_degrees" in source:
        gripper_axes = [joint for joint in arm.joints if joint.role == "gripper"]
        if len(gripper_axes) != 1 or gripper_axes[0].type != "revolute":
            raise ValueError("Scalar degree gripper anchor requires exactly one revolute gripper joint")
        parameters.setdefault(arm.gripper_joint_names[0], {}).update(
            closed=source["gripper_closed_degrees"], contact_units=source.get("gripper_contact_units", 0.0))
    if arm.wrist_roll is not None and "wrist_roll_zero_offset_degrees" in source:
        parameters.setdefault(arm.wrist_roll.joint, {})["offset"] = source["wrist_roll_zero_offset_degrees"]
    if not set(parameters) <= set(arm.joint_names):
        raise ValueError(f"Unknown anchored joint names: {sorted(set(parameters) - set(arm.joint_names))}")
    for name, values in parameters.items():
        if not isinstance(values, dict):
            raise ValueError(f"Calibration parameters for {name} must be an object")
        for key, value in values.items():
            _finite(value, f"{name}.{key}")
        if not 0 <= values.get("contact_units", 0) <= 100:
            raise ValueError(f"{name}.contact_units must be in 0..100")
        if "contact_units" in values and "closed" not in values:
            raise ValueError(f"{name}.contact_units requires a closed anchor")
    return parameters


@dataclass(frozen=True)
class CalibrationUnit:
    arm: str
    unit_id: str
    mapping: dict
    source: dict
    parameters: dict
    path: Path
    explicit_codecs: frozenset[str] = frozenset()


def load_unit(arm: ArmSpec, unit_id: str) -> CalibrationUnit:
    if not isinstance(unit_id, str) or not unit_id or Path(unit_id).name != unit_id or unit_id in (".", ".."):
        raise ValueError("Unit id must be a single nonempty filename stem")
    path = arm.directory / "calibration" / "units" / f"{unit_id}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read arm unit {path}: {error}") from error
    if data.get("schema_version") != 1 or data.get("arm") != arm.name or data.get("unit_id") != unit_id:
        raise ValueError(f"Unit identity/schema mismatch in {path}")
    mapping = validate_mapping(arm, data.get("mapping"))
    explicit_codecs = frozenset(name for name, span in mapping.items() if "raw_min" in span)
    source = data.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"Unit source must be an object in {path}")
    parameters = _source_parameters(arm, source)
    if "joints" in source:
        derived = follower_mapping_from_calibration(source["joints"], arm, parameters=parameters,
                                                    apply_drive_mode=source.get("apply_drive_mode"), raw_mapping=mapping)
        if any(mapping[name][key] != derived[name][key] for name in arm.joint_names
               for key in ("joint_min", "joint_max")):
            raise ValueError(f"Stale unit mapping in {path}; regenerate from its source calibration")
    mapping = with_device_codec(arm, mapping, source.get("joints"),
                                apply_drive_mode=source.get("apply_drive_mode"))
    return CalibrationUnit(arm.name, unit_id, mapping, source, parameters, path.resolve(), explicit_codecs)


def follower_mapping_from_calibration(calibration, arm: ArmSpec, *, parameters=None, apply_drive_mode=None, raw_mapping=None):
    """Convert measured encoder ranges using this arm's hardware and optional anchors."""
    parameters = {} if parameters is None else parameters
    if apply_drive_mode is not None and type(apply_drive_mode) is not bool:
        raise ValueError("apply_drive_mode must be a boolean")
    if not isinstance(parameters, dict) or not set(parameters) <= set(arm.joint_names):
        raise ValueError("Calibration parameters must name declared arm joints")
    for name, anchor in parameters.items():
        if not isinstance(anchor, dict) or set(anchor) - {"offset", "closed", "contact_units", "metres_per_count"}:
            raise ValueError(f"Unknown calibration anchor fields for {name}")
        if "contact_units" in anchor and "closed" not in anchor:
            raise ValueError(f"{name}.contact_units requires a closed anchor")
    label = "object"
    if isinstance(calibration, (str, Path)):
        label = str(calibration)
        try:
            calibration = json.loads(Path(calibration).read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read follower calibration {label}: {error}; run s2r calibrate first") from error
    if not isinstance(calibration, dict):
        raise ValueError(f"Follower calibration {label} must contain a joint object")
    result = {}
    for joint in arm.joints:
        row = calibration.get(joint.name)
        if not isinstance(row, dict):
            raise ValueError(f"Follower calibration {label} is missing joint {joint.name}")
        low = _finite(row.get("range_min"), f"{joint.name}.range_min")
        high = _finite(row.get("range_max"), f"{joint.name}.range_max")
        if high <= low:
            raise ValueError(f"{joint.name}.range_max must be greater than range_min")
        anchor = parameters.get(joint.name, {})
        if joint.type == "revolute":
            minimum = (low - arm.hardware.zero_count) * 360 / arm.hardware.encoder_counts_per_turn
            maximum = (high - arm.hardware.zero_count) * 360 / arm.hardware.encoder_counts_per_turn
            span = (high - low) * 360 / arm.hardware.encoder_counts_per_turn
        else:
            scale = _finite(anchor.get("metres_per_count"), f"{joint.name}.metres_per_count")
            if scale <= 0:
                raise ValueError(f"{joint.name}.metres_per_count must be positive")
            minimum = (low - arm.hardware.zero_count) * scale
            maximum = (high - arm.hardware.zero_count) * scale
            span = (high - low) * scale
        offset = _finite(anchor.get("offset", 0.0), f"{joint.name}.offset")
        minimum, maximum = minimum + offset, maximum + offset
        if "closed" in anchor:
            if joint.role != "gripper":
                raise ValueError(f"Closed anchor is only valid for gripper joints: {joint.name}")
            contact = _finite(anchor.get("contact_units", 0.0), f"{joint.name}.contact_units")
            if not 0 <= contact <= 100:
                raise ValueError(f"{joint.name}.contact_units must be in 0..100")
            codec = (raw_mapping or {}).get(joint.name, {})
            reversed_gripper = bool(row.get("drive_mode", 0))
            closed = _finite(anchor["closed"], f"{joint.name}.closed")
            if "raw_min" in codec or "raw_max" in codec:
                raw_low, raw_high = raw_endpoints({joint.name: codec}, [joint.name], [joint.name])
                minimum = closed - (contact - raw_low[0]) / (raw_high[0] - raw_low[0]) * span
            else:
                if reversed_gripper and apply_drive_mode is None:
                    raise ValueError(f"{joint.name} drive_mode needs explicit source.apply_drive_mode or raw endpoints")
                fraction = 1.0 - contact / 100 if reversed_gripper and apply_drive_mode else contact / 100
                minimum = closed - fraction * span if reversed_gripper and apply_drive_mode else closed - contact * span / 100
            maximum = minimum + span
        result[joint.name] = dict(joint_min=minimum, joint_max=maximum)
    return validate_mapping(arm, result)


def factory_mapping(arm: ArmSpec) -> dict:
    """Factory rotary joints use the full encoder range; slides use declared full stroke.

    An encoder turn cannot imply a linear travel without a measured transmission.
    A factory prismatic joint therefore declares that full stroke in joint.limits;
    its raw normalized endpoints map to those metres directly.
    """
    result = {}
    for joint in arm.joints:
        if joint.type == "prismatic":
            low, high = joint.limits
        else:
            low = -arm.hardware.zero_count * 360 / arm.hardware.encoder_counts_per_turn
            high = (arm.hardware.encoder_counts_per_turn - 1 - arm.hardware.zero_count) * 360 / arm.hardware.encoder_counts_per_turn
        result[joint.name] = dict(joint_min=low, joint_max=high)
    calibration = {joint.name: dict(range_min=0, range_max=arm.hardware.encoder_counts_per_turn - 1,
                                   drive_mode=0) for joint in arm.joints}
    return with_device_codec(arm, result, calibration)


def calibration_identity(path, *, required=True):
    if path is None:
        return {"path": None, "sha256": None}
    path = Path(path).expanduser().resolve()
    try:
        return {"path": str(path), "sha256": file_sha256(path)}
    except OSError as error:
        if required:
            raise ValueError(f"Cannot read calibration file {path}: {error}") from error
        return {"path": str(path), "sha256": None}


def load_device_calibration(robot, path):
    path = Path(path).expanduser().resolve()
    robot.calibration_fpath = path
    robot._load_calibration()
    robot.bus.calibration = robot.calibration


@dataclass(frozen=True)
class CalibrationSelection:
    arm: ArmSpec
    unit: CalibrationUnit | None
    mapping: dict
    follower_path: Path | None
    leader_path: Path | None
    provenance: dict

    def verify_loaded(self, robot, role):
        if role not in ("leader", "follower"):
            raise ValueError(f"Unknown device role {role!r}")
        if self.arm.lerobot.calibration == "factory":
            return
        expected = getattr(self, f"{role}_path")
        loaded = Path(robot.calibration_fpath).expanduser().resolve()
        if expected is not None and loaded != expected:
            raise ValueError(f"{role} calibration mismatch: loaded {loaded}; expected {expected}")
        actual_hash = calibration_identity(loaded)["sha256"]
        if expected is not None and actual_hash != self.provenance[f"{role}_calibration"]["sha256"]:
            raise ValueError(f"{role} calibration changed after selection: {loaded}")
        calibration = getattr(robot, "calibration", None)
        if calibration is not None:
            values = {name: asdict(value) if is_dataclass(value) else value for name, value in calibration.items()}
            saved = json.loads(loaded.read_text())
            fields = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
            if (set(values) != set(self.arm.joint_names) or set(saved) != set(self.arm.joint_names)
                    or any(not isinstance(values[name], dict) or not isinstance(saved[name], dict)
                           or any(values[name].get(key) != value for key, value in saved[name].items() if key in fields)
                           for name in self.arm.joint_names)):
                raise ValueError("LeRobot in-memory calibration differs from its file")
        if role == "follower":
            parameters = self.provenance["mapping_parameters"]
            actual = follower_mapping_from_calibration(loaded, self.arm, parameters=parameters,
                                                       apply_drive_mode=self.unit.source.get("apply_drive_mode"),
                                                       raw_mapping={name: self.unit.mapping[name] for name in self.unit.explicit_codecs})
            if any(abs(actual[name][key] - self.mapping[name][key]) > 1e-9
                   for name in self.arm.joint_names for key in ("joint_min", "joint_max")):
                raise ValueError(f"Follower calibration {loaded} differs from unit {self.unit.unit_id}; select its matching unit")
            for name in self.unit.explicit_codecs:
                actual[name].update({key: self.unit.mapping[name][key] for key in ("raw_min", "raw_max")})
            actual = with_device_codec(self.arm, actual, loaded,
                                       apply_drive_mode=self.unit.source.get("apply_drive_mode"))
            if raw_endpoints(actual, self.arm.joint_names, self.arm.gripper_joint_names) != raw_endpoints(
                    self.mapping, self.arm.joint_names, self.arm.gripper_joint_names):
                raise ValueError(f"Follower raw device codec differs from selected unit {self.unit.unit_id}")
        if self.unit is not None:
            identity = self.provenance["unit_calibration"]
            if file_sha256(self.unit.path) != identity["sha256"]:
                raise ValueError("Unit mapping changed after selection; restart the client")
        source = self.provenance["mapping_calibration"]
        if source["path"] and file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("Mapping calibration changed after selection; restart the client")

    def print_paths(self):
        for role in ("follower", "leader", "mapping"):
            key = "mapping_calibration" if role == "mapping" else f"{role}_calibration"
            item = self.provenance[key]
            print(f"[calibration] {role}: {item['path']} SHA-256={item['sha256']}", flush=True)


def resolve_calibration(arm: ArmSpec | None = None, *, unit=None, environ=None, follower_id=None,
                        leader_id=None, follower_path=None, leader_path=None) -> CalibrationSelection:
    arm = load_arm_spec()[0] if arm is None else arm
    environ = os.environ if environ is None else environ
    root = Path(environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()
    # Environment names are migrated with all commands in Phase 4.
    explicit_follower = follower_path or environ.get("S2R_FOLLOWER_CALIBRATION")
    explicit_leader = leader_path or environ.get("S2R_LEADER_CALIBRATION")
    selected_unit = None
    if arm.lerobot.calibration == "per_unit":
        if unit is None:
            candidates = [Path(explicit_follower).stem if explicit_follower else None, follower_id]
            unit = next((value for value in candidates if value and
                         (arm.directory / "calibration/units" / f"{value}.json").is_file()), None)
        if unit is None:
            raise ValueError(f"Arm {arm.name!r} requires a calibration unit id")
        selected_unit = load_unit(arm, unit) if isinstance(unit, str) else unit
        if selected_unit.arm != arm.name:
            raise ValueError(f"Unit belongs to {selected_unit.arm!r}, not {arm.name!r}")
    elif unit is not None:
        raise ValueError(f"Factory-calibrated arm {arm.name!r} does not use unit files")
    follower = (Path(explicit_follower).expanduser().resolve() if explicit_follower else
                (root / arm.lerobot.follower_calibration_dir / f"{follower_id}.json").resolve() if follower_id else None)
    leader = (Path(explicit_leader).expanduser().resolve() if explicit_leader else
              (root / arm.lerobot.leader_calibration_dir / f"{leader_id}.json").resolve() if leader_id else None)
    if selected_unit is None:
        mapping, parameters, source = factory_mapping(arm), {}, {}
        mapping_path = None
    else:
        parameters, source = selected_unit.parameters, selected_unit.source
        mapping_path = Path(explicit_follower).expanduser().resolve() if explicit_follower else selected_unit.path
        mapping = (with_device_codec(arm, follower_mapping_from_calibration(mapping_path, arm, parameters=parameters,
                                                                              apply_drive_mode=source.get("apply_drive_mode"),
                                                                              raw_mapping={name: selected_unit.mapping[name] for name in selected_unit.explicit_codecs}),
                                     mapping_path, apply_drive_mode=source.get("apply_drive_mode"))
                   if explicit_follower else selected_unit.mapping)
    per_unit = selected_unit is not None
    selected = dict(follower=calibration_identity(explicit_follower, required=per_unit)["sha256"],
                    leader=calibration_identity(explicit_leader, required=False)["sha256"])
    provenance = dict(
        arm=arm.name, unit=None if selected_unit is None else selected_unit.unit_id,
        follower_id=follower_id, leader_id=leader_id,
        follower_calibration=calibration_identity(follower, required=per_unit),
        leader_calibration=calibration_identity(leader, required=per_unit and leader_id is not None),
        unit_calibration=calibration_identity(selected_unit.path if selected_unit else None),
        mapping_calibration=calibration_identity(mapping_path), mapping=mapping, mapping_parameters=parameters,
        mapping_source="follower_file" if explicit_follower else "unit_file" if per_unit else "factory",
        explicit_selection=bool(explicit_follower or explicit_leader), selected_calibrations=selected,
        embedded_follower_sha256=source.get("calibration_sha256"),
        gripper_anchor_status=source.get("gripper_contact_fit_status", "not declared"),
    )
    return CalibrationSelection(arm, selected_unit, mapping, follower, leader, provenance)
