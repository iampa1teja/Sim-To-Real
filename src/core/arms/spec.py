"""Typed arm manifests and optional arm-folder hooks; no simulator imports."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import tomllib
from types import ModuleType
from typing import Any, Literal

from core.paths import DEFAULT_ARM, repo_dir


def arms_dir() -> Path:
    configured = os.environ.get("S2R_ARMS_DIR")
    return Path(configured).expanduser() if configured else repo_dir("arms")


def selected_arm_name() -> str:
    """Read the selector when requested, rather than freezing it during import."""
    return os.environ.get("S2R_ARM", DEFAULT_ARM)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    return tuple(_number(v, label) for v in value)


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _names(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of names")
    names = tuple(_text(v, label) for v in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names")
    return names


def _relative_file(directory, value, label):
    value = _text(value, label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative to the arm folder without '..'")
    path = directory / path
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path.resolve()


@dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    usd: str
    type: Literal["revolute", "prismatic"]
    role: Literal["arm", "gripper"]
    limits: tuple[float, float]
    initial: float

    @property
    def unit(self) -> str:
        return "deg" if self.type == "revolute" else "m"


@dataclass(frozen=True, slots=True)
class LinksSpec:
    base: str
    ee: str


@dataclass(frozen=True, slots=True)
class GripperSpec:
    kind: str
    joints: tuple[str, ...]
    contact_bodies: tuple[str, ...]
    friction_bodies: tuple[str, ...]
    tip_mesh: str | None = None


@dataclass(frozen=True, slots=True)
class WristRollSpec:
    joint: str
    relative_in_teleop: bool = True


@dataclass(frozen=True, slots=True)
class CameraSpec:
    parent_link: str
    position: tuple[float, float, float]
    euler_degrees: tuple[float, float, float]
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    calibration: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    data_types: tuple[str, ...] = ("rgb",)


@dataclass(frozen=True, slots=True)
class InitialStateSpec:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class LeRobotSpec:
    follower_type: str
    leader_type: str
    modules: tuple[str, ...] = ()
    use_degrees: bool = False
    calibration: str = "per_unit"
    follower_calibration_dir: str = ""
    leader_calibration_dir: str = ""


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    motor: str
    encoder_counts_per_turn: int
    zero_count: float


@dataclass(frozen=True, slots=True)
class IKSpec:
    # Rotation-vector components in the arm-base frame; omitted axes mean position-only IK.
    orientation_axes: tuple[str, ...] = ()
    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    directory: Path
    usd_path: Path
    utils: Path | None
    links: LinksSpec
    joints: tuple[JointSpec, ...]
    gripper: GripperSpec | None
    wrist_roll: WristRollSpec | None
    cameras: dict[str, CameraSpec]
    appearance: dict[str, Any]
    actuators: dict[str, dict[str, Any]]
    spawn: dict[str, Any]
    initial_state: InitialStateSpec
    lerobot: LeRobotSpec
    hardware: HardwareSpec
    ik: IKSpec = field(default_factory=IKSpec)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def usd_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.usd for joint in self.joints)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{name}.pos" for name in self.joint_names)

    @property
    def gripper_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.role == "gripper")

    @property
    def joint_name_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.joint_names, self.usd_joint_names))

    @property
    def runtime_joint_limits(self) -> dict[str, dict[str, float]]:
        return {joint.name: dict(joint_min=joint.limits[0], joint_max=joint.limits[1]) for joint in self.joints}

    @property
    def initial_joint_positions(self) -> dict[str, float]:
        return {joint.usd: joint.initial for joint in self.joints}


def load_arm_spec(name: str | None = None, variant: str | None = None) -> tuple[ArmSpec, Path]:
    name = selected_arm_name() if name is None else name
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Arm folder name {name!r} must be a public Python identifier")
    directory = (arms_dir() / name).resolve()
    try:
        with (directory / "arm.toml").open("rb") as handle:
            document = tomllib.load(handle)
        raw = document["arm"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot load arms/{name}/arm.toml: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"arms/{name}/arm.toml needs an [arm] table")
    variants = raw.pop("variants", {})
    if variant is not None:
        if variant not in variants:
            raise ValueError(f"Arm {name!r} has no variant {variant!r}; choices: {sorted(variants)}")
        raw = _deep_merge(raw, variants[variant])
    try:
        spec = _parse_spec(name, directory, raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"arms/{name}/arm.toml: {error}") from error
    return spec, directory


def _parse_spec(name, directory, raw):
    if raw["name"] != name:
        raise ValueError(f"arm.name {raw['name']!r} must equal folder name {name!r}")
    rows = raw["joints"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("arm.joints must be a nonempty list of [[arm.joints]] tables")
    joints = []
    for row in rows:
        kind, role = row["type"], row.get("role", "arm")
        if kind not in ("revolute", "prismatic") or role not in ("arm", "gripper"):
            raise ValueError("joint type must be revolute/prismatic and role must be arm/gripper")
        limits = _vector(row["limits"], 2, "joint.limits")
        if limits[0] >= limits[1]:
            raise ValueError("joint.limits must be increasing")
        initial = _number(row["initial"], "joint.initial")
        physical_initial = math.degrees(initial) if kind == "revolute" else initial
        if not limits[0] <= physical_initial <= limits[1]:
            raise ValueError(f"joint {row['name']!r} initial value lies outside its limits")
        joints.append(JointSpec(_text(row["name"], "joint.name"), _text(row["usd"], "joint.usd"),
                                kind, role, limits, initial))
    for field_name in ("name", "usd"):
        values = [getattr(joint, field_name) for joint in joints]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate joint {field_name}")
    gripper_names = {joint.name for joint in joints if joint.role == "gripper"}
    grip = raw.get("gripper")
    gripper = None
    if grip is not None:
        if grip["kind"] not in ("jaw", "parallel"):
            raise ValueError("gripper.kind must be jaw or parallel")
        grip_names = _names(grip["joints"], "gripper.joints")
        if set(grip_names) != gripper_names or not grip_names:
            raise ValueError("gripper.joints must match exactly the joints with role='gripper'")
        gripper = GripperSpec(grip["kind"], grip_names,
                              _names(grip.get("contact_bodies", []), "gripper.contact_bodies"),
                              _names(grip.get("friction_bodies", []), "gripper.friction_bodies"),
                              grip.get("tip_mesh"))
    elif gripper_names:
        raise ValueError("Joints with role='gripper' require an [arm.gripper] table")
    roll = raw.get("wrist_roll")
    wrist_roll = None
    if roll is not None:
        joint = next((joint for joint in joints if joint.name == roll["joint"]), None)
        if joint is None or joint.role != "arm" or joint.type != "revolute":
            raise ValueError("wrist_roll.joint must name a revolute arm joint")
        relative = roll.get("relative_in_teleop", True)
        if type(relative) is not bool:
            raise ValueError("wrist_roll.relative_in_teleop must be a boolean")
        wrist_roll = WristRollSpec(joint.name, relative)
    links = LinksSpec(_text(raw["links"]["base"], "links.base"), _text(raw["links"]["ee"], "links.ee"))
    initial = raw.get("initial_state", {})
    initial_state = InitialStateSpec(_vector(initial.get("position", [0, 0, 0]), 3, "initial_state.position"),
                                     _vector(initial.get("rotation_euler_degrees", [0, 0, 0]), 3,
                                             "initial_state.rotation_euler_degrees"))
    robot = raw["lerobot"]
    calibration = robot.get("calibration", "per_unit")
    if calibration not in ("per_unit", "factory"):
        raise ValueError("lerobot.calibration must be per_unit or factory")
    use_degrees = robot.get("use_degrees", False)
    if type(use_degrees) is not bool:
        raise ValueError("lerobot.use_degrees must be a boolean")
    lerobot = LeRobotSpec(_text(robot["follower_type"], "lerobot.follower_type"),
                          _text(robot["leader_type"], "lerobot.leader_type"),
                          _names(robot.get("modules", []), "lerobot.modules"), use_degrees, calibration,
                          robot.get("follower_calibration_dir", ""), robot.get("leader_calibration_dir", ""))
    hardware_raw = raw["hardware"]
    counts = hardware_raw["encoder_counts_per_turn"]
    if type(counts) is not int or counts < 2:
        raise ValueError("hardware.encoder_counts_per_turn must be an integer >= 2")
    hardware = HardwareSpec(_text(hardware_raw["motor"], "hardware.motor"), counts,
                             _number(hardware_raw["zero_count"], "hardware.zero_count"))
    cameras = {}
    for camera_name, camera in raw.get("cameras", {}).items():
        for dim in ("width", "height"):
            if type(camera[dim]) is not int or camera[dim] < 1:
                raise ValueError(f"cameras.{camera_name}.{dim} must be a positive integer")
        optics = [_number(camera[key], f"cameras.{camera_name}.{key}") for key in
                  ("focal_length", "horizontal_aperture", "vertical_aperture")]
        if min(optics) <= 0:
            raise ValueError(f"cameras.{camera_name} optics must be positive")
        cameras[camera_name] = CameraSpec(
            _text(camera["parent_link"], f"cameras.{camera_name}.parent_link"),
            _vector(camera["position"], 3, "camera.position"),
            _vector(camera["euler_degrees"], 3, "camera.euler_degrees"),
            camera["width"], camera["height"], *optics,
            copy.deepcopy(camera.get("calibration", {})), copy.deepcopy(camera.get("response", {})),
            _names(camera.get("data_types", ["rgb"]), "camera.data_types"),
        )
    actuators = copy.deepcopy(raw.get("actuators", {}))
    for group, actuator in actuators.items():
        expressions = _names(actuator["joint_names_expr"], f"actuators.{group}.joint_names_expr")
        for expression in expressions:
            try:
                matches = [joint.usd for joint in joints if re.fullmatch(expression, joint.usd)]
            except re.error as error:
                raise ValueError(f"Invalid actuator joint expression {expression!r}") from error
            if not matches:
                raise ValueError(f"Actuator {group!r} expression {expression!r} matches no USD joint")
    ik_raw = raw.get("ik", {})
    axes = _names(ik_raw.get("orientation_axes", []), "ik.orientation_axes")
    if set(axes) - {"x", "y", "z"}:
        raise ValueError("ik.orientation_axes must select x, y and/or z rotation-error components")
    ik = IKSpec(axes, _vector(ik_raw.get("position_offset_m", [0, 0, 0]), 3, "ik.position_offset_m"))
    return ArmSpec(name, directory, _relative_file(directory, raw["usd_path"], "usd_path"),
                   _relative_file(directory, raw["utils"], "utils") if raw.get("utils") else None,
                   links, tuple(joints), gripper, wrist_roll, cameras,
                   copy.deepcopy(raw.get("appearance", {})), actuators,
                   copy.deepcopy(raw.get("spawn", {})), initial_state, lerobot, hardware, ik)


def load_arm_utils(spec: ArmSpec) -> ModuleType | None:
    """Load optional hooks by path; importing this loader itself never imports Isaac."""
    if spec.utils is None:
        return None
    module_name = "_s2r_arm_utils_" + hashlib.sha256(str(spec.utils).encode()).hexdigest()
    if module_name not in sys.modules:
        module_spec = importlib.util.spec_from_file_location(module_name, spec.utils)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name]
