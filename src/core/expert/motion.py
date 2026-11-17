"""Arm-independent, bounded motion helpers."""

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass

import numpy as _np

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.arms.kinematics import KinematicChain as _KinematicChain
from core.arms.raw_mapping import RawMapping as _RawMapping
from core.generate.errors import ConfigError as _ConfigError


@_dataclass(frozen=True, eq=False)
class MotionLimits:
    max_joint_speed_rad_s: _np.ndarray
    max_gripper_speed_rad_s: float
    max_ee_speed_m_s: float
    __hash__ = None

    def __post_init__(self):
        speeds = _np.array(self.max_joint_speed_rad_s, dtype=_np.float64, order="C", copy=True)
        if speeds.ndim != 1 or not _np.all(_np.isfinite(speeds)) or _np.any(speeds <= 0):
            raise ValueError("Joint speeds must be a finite positive vector")
        speeds.flags.writeable = False
        object.__setattr__(self, "max_joint_speed_rad_s", speeds)
        for name in ("max_gripper_speed_rad_s", "max_ee_speed_m_s"):
            value = float(getattr(self, name))
            if not _np.isfinite(value) or value <= 0:
                raise ValueError("Motion speeds must be finite and positive")
            object.__setattr__(self, name, value)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return bool(_np.array_equal(self.max_joint_speed_rad_s, other.max_joint_speed_rad_s)
                    and self.max_gripper_speed_rad_s == other.max_gripper_speed_rad_s
                    and self.max_ee_speed_m_s == other.max_ee_speed_m_s)

    @classmethod
    def from_arm(cls, name=None):
        spec, _ = _load_arm_spec(name)
        try:
            expert = spec["expert"]
            names, jaw = spec["joints"]["names"], spec["gripper"]["joint"]
            speeds = expert["max_joint_speed_rad_s"]
            jaw_speed = expert["max_gripper_speed_rad_s"]
            values = [jaw_speed if joint == jaw else speeds[joint] if isinstance(speeds, _Mapping) else speeds
                      for joint in names]
            return cls(values, jaw_speed, expert["max_ee_speed_m_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _ConfigError(f"Incomplete expert motion limits: {exc}") from exc


def step_toward_joint(current, target, max_step) -> _np.ndarray:
    current, target, step = (_np.asarray(v, dtype=_np.float64) for v in (current, target, max_step))
    if current.ndim != 1 or target.shape != current.shape or step.shape not in ((), current.shape):
        raise ValueError("Joint motion requires matching vectors and a scalar or vector step")
    if not all(_np.all(_np.isfinite(v)) for v in (current, target, step)) or _np.any(step < 0):
        raise ValueError("Motion values must be finite and steps nonnegative")
    return current + _np.clip(target - current, -step, step)


def cartesian_waypoint(current, goal, max_step_m) -> _np.ndarray:
    current, goal = _np.asarray(current, dtype=_np.float64), _np.asarray(goal, dtype=_np.float64)
    if current.shape != (3,) or goal.shape != (3,):
        raise ValueError("Cartesian positions must have shape (3,)")
    if not _np.all(_np.isfinite(current)) or not _np.all(_np.isfinite(goal)) or not _np.isfinite(max_step_m) or max_step_m < 0:
        raise ValueError("Invalid Cartesian motion")
    delta = goal - current
    distance = float(_np.linalg.norm(delta))
    return goal.copy() if distance <= max_step_m else current + delta * (max_step_m / distance)


def expert_joint_limits(arm=None, *, setup=None):
    mapping, chain = _RawMapping.from_setup(setup, arm=arm), _KinematicChain.from_arm(arm)
    lower, upper = _np.maximum(mapping.lower_rad, chain.lower), _np.minimum(mapping.upper_rad, chain.upper)
    if _np.any(lower > upper):
        raise _ConfigError("Runtime and USD joint limits do not overlap")
    return lower, upper
