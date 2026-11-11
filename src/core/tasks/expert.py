"""Simulator-independent Cartesian expert commands in world metres and XYZW."""
from dataclasses import dataclass, field
import math

from core.tasks.base import Pose, StepState


@dataclass(frozen=True, slots=True)
class ExpertContext:
    state: StepState
    control_pose: Pose
    table_normal: tuple[float, float, float]
    options: dict = field(default_factory=dict)
    arm_base: Pose | None = None  # actual declared base-link pose in world coordinates
    arm: object | None = None


@dataclass(frozen=True, slots=True)
class Waypoint:
    name: str
    pose: Pose
    gripper_open_fraction: float | None = None
    duration_s: float = 2.0
    tolerance_m: float = .004
    timeout_s: float = 8.0
    gripper_open_gap_m: float | None = None

    def __post_init__(self):
        if (self.gripper_open_fraction is None) == (self.gripper_open_gap_m is None):
            raise ValueError("Exactly one of gripper_open_fraction and gripper_open_gap_m is required")
        opening = self.gripper_open_fraction if self.gripper_open_gap_m is None else self.gripper_open_gap_m
        values = (*self.pose.position, *self.pose.quat_xyzw, opening,
                  self.duration_s, self.tolerance_m, self.timeout_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Expert waypoints must contain finite poses and timing")
        if not self.name or sum(value * value for value in self.pose.quat_xyzw) < 1e-12:
            raise ValueError("Expert waypoints need a name and nonzero quaternion")
        if self.gripper_open_fraction is not None and not 0 <= self.gripper_open_fraction <= 1:
            raise ValueError("gripper_open_fraction must be within [0, 1]")
        if self.gripper_open_gap_m is not None and self.gripper_open_gap_m < 0:
            raise ValueError("gripper_open_gap_m must be nonnegative")
        if min(self.duration_s, self.tolerance_m, self.timeout_s) <= 0 or self.timeout_s < self.duration_s:
            raise ValueError("Waypoint timing/tolerance must be positive, with timeout >= duration")


def gripper_commands(context, object_width_m):
    """Expert opening options, with measured physical openings for imported arms."""
    options = context.options
    grip = context.arm.gripper if context.arm is not None else None
    measured = grip is not None and grip.open_gap_m is not None
    physical = measured or any(key in options for key in ('gripper_open_gap_m', 'gripper_closed_gap_m'))
    if physical:
        default_open = min(grip.open_gap_m, object_width_m + .02) if measured else object_width_m + .02
        return (dict(gripper_open_gap_m=float(options.get('gripper_open_gap_m', default_open))),
                dict(gripper_open_gap_m=float(options.get('gripper_closed_gap_m', 0.))))
    return (dict(gripper_open_fraction=float(options.get('gripper_open_fraction', .45))),
            dict(gripper_open_fraction=float(options.get('gripper_closed_fraction', 0.))))


def grasp_orientation(context, object_pose, orientation):
    """Nearest cube face normal about the table normal; preserve this after pickup."""
    import numpy as np
    from core.scenes.anchors import quaternion_matrix, quaternion_multiply

    arm = context.arm
    if arm is None or 'z' not in arm.ik.orientation_axes or arm.gripper is None:
        return orientation
    if arm.gripper.closing_axis is None:
        return orientation
    normal = np.asarray(context.table_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    closing = quaternion_matrix(orientation) @ arm.gripper.closing_axis
    closing -= np.dot(closing, normal) * normal
    if np.linalg.norm(closing) < 1e-8:
        raise ValueError('Gripper closing axis must lie in the table plane for grasp yaw alignment')
    closing /= np.linalg.norm(closing)
    rotation = quaternion_matrix(object_pose.quat_xyzw)
    angles = []
    for direction in (rotation[:, 0], -rotation[:, 0], rotation[:, 1], -rotation[:, 1]):
        direction = direction - np.dot(direction, normal) * normal
        if np.linalg.norm(direction) < 1e-8:
            continue
        direction /= np.linalg.norm(direction)
        angles.append(math.atan2(np.dot(normal, np.cross(closing, direction)), np.dot(closing, direction)))
    angle = min(angles, key=abs)
    increment = tuple(normal * math.sin(angle / 2)) + (math.cos(angle / 2),)
    return quaternion_multiply(increment, orientation)
