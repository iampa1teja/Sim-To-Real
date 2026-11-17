"""Arm-defined piecewise-linear jaw angle and opening-gap conversion."""

import math as _math
from dataclasses import dataclass as _dataclass

import numpy as _np

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.generate.errors import ConfigError as _ConfigError


def _finite(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _opening_table(rows):
    if (not isinstance(rows, (list, tuple)) or len(rows) < 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2 or not all(_finite(v) for v in row) for row in rows)):
        raise _ConfigError("opening_table requires at least two finite angle/gap rows")
    angles, gaps = tuple(float(r[0]) for r in rows), tuple(float(r[1]) for r in rows)
    if not all(b > a for a, b in zip(angles, angles[1:])):
        raise _ConfigError("Opening angles must strictly increase")
    if not (all(b > a for a, b in zip(gaps, gaps[1:])) or all(b < a for a, b in zip(gaps, gaps[1:]))):
        raise _ConfigError("Opening gaps must be strictly monotone")
    return angles, gaps


@_dataclass(frozen=True)
class Gripper:
    joint: str
    index: int
    open_rad: float
    closed_rad: float
    angles_rad: tuple[float, ...]
    gaps_m: tuple[float, ...]
    max_object_width_m: float

    @classmethod
    def from_table(cls, table, q_names):
        try:
            joint = table["joint"]
            if not isinstance(joint, str) or joint not in q_names:
                raise _ConfigError("Unknown gripper joint")
            if not all(_finite(table[key]) for key in ("open_rad", "closed_rad", "max_object_width_m")) or table["max_object_width_m"] <= 0:
                raise _ConfigError("Invalid gripper angle or width")
            angles, gaps = _opening_table(table["opening_table"])
            return cls(joint, tuple(q_names).index(joint), float(table["open_rad"]), float(table["closed_rad"]),
                       angles, gaps, float(table["max_object_width_m"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise _ConfigError(f"Invalid gripper table: {exc}") from exc

    @classmethod
    def from_arm(cls, name=None):
        spec, _ = _load_arm_spec(name)
        if "gripper" not in spec:
            raise _ConfigError("Arm has no gripper table")
        return cls.from_table(spec["gripper"], spec["joints"]["names"])

    def gap_from_angle(self, rad: float) -> float:
        return float(_np.interp(rad, self.angles_rad, self.gaps_m))

    def angle_from_gap(self, gap_m: float) -> float:
        gaps, angles = self.gaps_m, self.angles_rad
        if gaps[-1] < gaps[0]:
            gaps, angles = gaps[::-1], angles[::-1]
        return float(_np.interp(gap_m, gaps, angles))
