"""Typed arm manifests: dataclasses and validation of an already-read [arm] table."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Literal

from utils.workspace import boolean, number, relative_file, text, vector


def _names(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of names")
    names = tuple(text(v, label) for v in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names")
    return names


@dataclass(frozen=True, slots=True)
class MimicSpec:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0  # simulator radians/metres


@dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    usd: str
    type: Literal["revolute", "prismatic"]
    role: Literal["arm", "gripper"]
    limits: tuple[float, float]
    initial: float
    mimic: MimicSpec | None = None

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
    closed_gap_m: float | None = None
    open_gap_m: float | None = None
    # Source coordinates are simulator radians/metres; opening may decrease q.
    closed_position: float | None = None
    open_position: float | None = None
    closing_axis: tuple[float, float, float] | None = None
    approach_axis: tuple[float, float, float] | None = None

    def gap_m(self, q):
        self._require_gap_map()
        return self.closed_gap_m + (q - self.closed_position) / (
            self.open_position - self.closed_position) * (self.open_gap_m - self.closed_gap_m)

    def q_for_gap(self, gap):
        self._require_gap_map()
        gap = max(self.closed_gap_m, min(self.open_gap_m, number(gap, "gripper gap")))
        return self.closed_position + (gap - self.closed_gap_m) / (
            self.open_gap_m - self.closed_gap_m) * (self.open_position - self.closed_position)

    def _require_gap_map(self):
        if self.kind != "parallel" or any(value is None for value in (
                self.closed_gap_m, self.open_gap_m, self.closed_position, self.open_position)):
            raise ValueError("A parallel gripper with measured gap and source endpoints is required")


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
    rotation_relative_to_anchor: bool = False


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
    joint_ids: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IKSpec:
    # Rotation-vector components in the arm-base frame; omitted axes mean position-only IK.
    orientation_axes: tuple[str, ...] = ()
    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    method: str = "dls"
    joint_limit_avoidance_gain: float = 0.0
    joint_limit_avoidance_margin: float = 0.3


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
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def action_joints(self) -> tuple[JointSpec, ...]:
        return tuple(joint for joint in self.joints if joint.mimic is None)

    @property
    def mimic_joints(self) -> tuple[tuple[int, int, float, float], ...]:
        indices = {joint.name: index for index, joint in enumerate(self.joints)}
        return tuple((index, indices[joint.mimic.joint], joint.mimic.multiplier, joint.mimic.offset)
                     for index, joint in enumerate(self.joints) if joint.mimic is not None)

    @property
    def action_usd_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.usd for joint in self.action_joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.action_joints)

    @property
    def usd_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.usd for joint in self.joints)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{name}.pos" for name in self.joint_names)

    @property
    def gripper_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.action_joints if joint.role == "gripper")

    @property
    def joint_name_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.joint_names, self.action_usd_joint_names))

    @property
    def runtime_joint_limits(self) -> dict[str, dict[str, float]]:
        return {joint.name: dict(joint_min=joint.limits[0], joint_max=joint.limits[1]) for joint in self.action_joints}

    @property
    def initial_joint_positions(self) -> dict[str, float]:
        return {joint.usd: joint.initial for joint in self.joints}


def parse_arm_spec(name, directory, raw):
    """Validate an already-read [arm] table; ``directory`` resolves its relative files."""
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
        limits = vector(row["limits"], 2, "joint.limits")
        if limits[0] >= limits[1]:
            raise ValueError("joint.limits must be increasing")
        initial = number(row["initial"], "joint.initial")
        physical_initial = math.degrees(initial) if kind == "revolute" else initial
        if not limits[0] <= physical_initial <= limits[1]:
            raise ValueError(f"joint {row['name']!r} initial value lies outside its limits")
        mimic_raw = row.get("mimic")
        mimic = None if mimic_raw is None else MimicSpec(
            text(mimic_raw["joint"], "mimic.joint"),
            number(mimic_raw.get("multiplier", 1.0), "mimic.multiplier"),
            number(mimic_raw.get("offset", 0.0), "mimic.offset"))
        joints.append(JointSpec(text(row["name"], "joint.name"), text(row["usd"], "joint.usd"),
                                kind, role, limits, initial, mimic))
    for field_name in ("name", "usd"):
        values = [getattr(joint, field_name) for joint in joints]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate joint {field_name}")
    by_name = {joint.name: joint for joint in joints}
    for joint in joints:
        if joint.mimic is None:
            continue
        source = by_name.get(joint.mimic.joint)
        if source is None:
            raise ValueError(f"mimic joint {joint.name!r} has missing source {joint.mimic.joint!r}")
        if source is joint or source.mimic is not None:
            raise ValueError(f"mimic joint {joint.name!r}: source must not itself be a mimic")
        if source.type != joint.type:
            raise ValueError(f"mimic joint {joint.name!r}: source and follower types must match")
        if joint.role != "gripper" or source.role != "gripper":
            raise ValueError(f"mimic joint {joint.name!r}: only gripper mimics are supported")
        source_initial = source.initial * joint.mimic.multiplier + joint.mimic.offset
        if not math.isclose(joint.initial, source_initial, abs_tol=1e-8):
            raise ValueError(f"mimic joint {joint.name!r}: initial must follow its source")
    gripper_names = {joint.name for joint in joints if joint.role == "gripper" and joint.mimic is None}
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
        closed, opened = grip.get("closed_gap_m"), grip.get("open_gap_m")
        if (closed is None) != (opened is None):
            raise ValueError("gripper closed_gap_m and open_gap_m must be provided together")
        if closed is not None:
            closed = number(closed, "gripper.closed_gap_m")
            opened = number(opened, "gripper.open_gap_m")
            if closed < 0 or opened <= closed:
                raise ValueError("gripper gaps must be nonnegative and open_gap_m > closed_gap_m")
            source = by_name[grip_names[0]]
            limits = tuple(math.radians(v) for v in source.limits) if source.type == "revolute" else source.limits
            q_closed = number(grip.get("closed_position", limits[0]), "gripper.closed_position")
            q_open = number(grip.get("open_position", limits[1]), "gripper.open_position")
            if q_closed == q_open or not all(limits[0] <= v <= limits[1] for v in (q_closed, q_open)):
                raise ValueError("gripper source endpoints must be distinct and within limits")
            gripper = GripperSpec(gripper.kind, gripper.joints, gripper.contact_bodies,
                                  gripper.friction_bodies, gripper.tip_mesh, closed, opened, q_closed, q_open,
                                  vector(grip["closing_axis"], 3, "gripper.closing_axis") if "closing_axis" in grip else None,
                                  vector(grip["approach_axis"], 3, "gripper.approach_axis") if "approach_axis" in grip else None)
    elif gripper_names:
        raise ValueError("Joints with role='gripper' require an [arm.gripper] table")
    roll = raw.get("wrist_roll")
    wrist_roll = None
    if roll is not None:
        joint = next((joint for joint in joints if joint.name == roll["joint"]), None)
        if joint is None or joint.role != "arm" or joint.type != "revolute":
            raise ValueError("wrist_roll.joint must name a revolute arm joint")
        wrist_roll = WristRollSpec(joint.name, boolean(roll.get("relative_in_teleop", True), "wrist_roll.relative_in_teleop"))
    links = LinksSpec(text(raw["links"]["base"], "links.base"), text(raw["links"]["ee"], "links.ee"))
    initial = raw.get("initial_state", {})
    initial_state = InitialStateSpec(vector(initial.get("position", [0, 0, 0]), 3, "initial_state.position"),
                                     vector(initial.get("rotation_euler_degrees", [0, 0, 0]), 3,
                                             "initial_state.rotation_euler_degrees"),
                                     boolean(initial.get("rotation_relative_to_anchor", False),
                                             "initial_state.rotation_relative_to_anchor"))
    robot = raw.get("lerobot", {"follower_type": "unconfigured", "leader_type": "unconfigured", "calibration": "simulation"})
    calibration = robot.get("calibration", "per_unit")
    if calibration not in ("per_unit", "factory", "simulation"):
        raise ValueError("lerobot.calibration must be per_unit, factory or simulation")
    use_degrees = boolean(robot.get("use_degrees", False), "lerobot.use_degrees")
    lerobot = LeRobotSpec(text(robot["follower_type"], "lerobot.follower_type"),
                          text(robot["leader_type"], "lerobot.leader_type"),
                          _names(robot.get("modules", []), "lerobot.modules"), use_degrees, calibration,
                          robot.get("follower_calibration_dir", ""), robot.get("leader_calibration_dir", ""))
    hardware_raw = raw.get("hardware", {"motor": "unconfigured", "encoder_counts_per_turn": 4096, "zero_count": 2048})
    counts = hardware_raw["encoder_counts_per_turn"]
    if type(counts) is not int or counts < 2:
        raise ValueError("hardware.encoder_counts_per_turn must be an integer >= 2")
    hardware = HardwareSpec(text(hardware_raw["motor"], "hardware.motor"), counts,
                             number(hardware_raw["zero_count"], "hardware.zero_count"),
                             copy.deepcopy(hardware_raw.get("joint_ids", {})))
    cameras = {}
    for camera_name, camera in raw.get("cameras", {}).items():
        for dim in ("width", "height"):
            if type(camera[dim]) is not int or camera[dim] < 1:
                raise ValueError(f"cameras.{camera_name}.{dim} must be a positive integer")
        optics = [number(camera[key], f"cameras.{camera_name}.{key}") for key in
                  ("focal_length", "horizontal_aperture", "vertical_aperture")]
        if min(optics) <= 0:
            raise ValueError(f"cameras.{camera_name} optics must be positive")
        cameras[camera_name] = CameraSpec(
            text(camera["parent_link"], f"cameras.{camera_name}.parent_link"),
            vector(camera["position"], 3, "camera.position"),
            vector(camera["euler_degrees"], 3, "camera.euler_degrees"),
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
    method = ik_raw.get("method", "dls")
    if method not in ("dls", "adaptive_dls"):
        raise ValueError("ik.method must be dls or adaptive_dls")
    gain = number(ik_raw.get("joint_limit_avoidance_gain", 0.0), "ik.joint_limit_avoidance_gain")
    margin = number(ik_raw.get("joint_limit_avoidance_margin", 0.3), "ik.joint_limit_avoidance_margin")
    if gain < 0:
        raise ValueError("ik.joint_limit_avoidance_gain must be nonnegative")
    if margin <= 0:
        raise ValueError("ik.joint_limit_avoidance_margin must be positive")
    if gain > 0 and sum(joint.role == "arm" for joint in joints) <= 3 + len(axes):
        raise ValueError("arm has no spare joint for limit avoidance")
    ik = IKSpec(axes, vector(ik_raw.get("position_offset_m", [0, 0, 0]), 3, "ik.position_offset_m"),
                method, gain, margin)
    return ArmSpec(name, directory, relative_file(directory, raw["usd_path"], "usd_path", "relative to the arm folder"),
                   relative_file(directory, raw["utils"], "utils", "relative to the arm folder") if raw.get("utils") else None,
                   links, tuple(joints), gripper, wrist_roll, cameras,
                   copy.deepcopy(raw.get("appearance", {})), actuators,
                   copy.deepcopy(raw.get("spawn", {})), initial_state, lerobot, hardware, ik, copy.deepcopy(raw.get("import", {})))
