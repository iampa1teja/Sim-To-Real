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


@dataclass(frozen=True, slots=True)
class Waypoint:
    name: str
    pose: Pose
    gripper_open_fraction: float
    duration_s: float = 2.0
    tolerance_m: float = .004
    timeout_s: float = 8.0

    def __post_init__(self):
        values = (*self.pose.position, *self.pose.quat_xyzw, self.gripper_open_fraction,
                  self.duration_s, self.tolerance_m, self.timeout_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Expert waypoints must contain finite poses and timing")
        if not self.name or sum(value * value for value in self.pose.quat_xyzw) < 1e-12:
            raise ValueError("Expert waypoints need a name and nonzero quaternion")
        if not 0 <= self.gripper_open_fraction <= 1:
            raise ValueError("gripper_open_fraction must be within [0, 1]")
        if min(self.duration_s, self.tolerance_m, self.timeout_s) <= 0 or self.timeout_s < self.duration_s:
            raise ValueError("Waypoint timing/tolerance must be positive, with timeout >= duration")
