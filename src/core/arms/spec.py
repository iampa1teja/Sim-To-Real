"""Read arms/<name>/arm.toml (pure tomllib, no Isaac imports).

Both the sim (``sim.scene.arm.load_arm_cfg``) and hardware code (``core.arms.joint_calibration``)
read the arm from here, so the same file describes the simulated and the physical arm.
"""

from __future__ import annotations

import copy
import math as _math
import os
import tomllib
from pathlib import Path
from typing import Any

from core.paths import repo_dir
from core.generate.errors import ConfigError as _ConfigError

# Selected arm; set S2R_ARM before importing sim/core modules to change it.
ARM_NAME = os.environ.get("S2R_ARM", "so101")


def arms_dir() -> Path:
    """Return the arms/ folder: $S2R_ARMS_DIR, else the first arms/ above this file."""
    configured = os.environ.get("S2R_ARMS_DIR")
    return Path(configured) if configured else repo_dir("arms")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_arm_spec(name: str | None = None, variant: str | None = None) -> tuple[dict[str, Any], Path]:
    """Read arm.toml, apply a variant, and return (spec, arm_dir). ``name`` defaults to ARM_NAME."""
    name = name or ARM_NAME
    arm_dir = arms_dir() / name
    with open(arm_dir / "arm.toml", "rb") as handle:
        spec = tomllib.load(handle)["arm"]

    variants = spec.pop("variants", {})
    if variant is not None:
        if variant not in variants:
            raise KeyError(f"Arm '{name}' has no variant '{variant}'; choices: {sorted(variants)}")
        spec = _deep_merge(spec, variants[variant])
    _validate_v4(spec, arm_dir / "arm.toml")
    return spec, arm_dir


def _number(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(_number(v) for v in value)


def _validate_v4(spec, path):
    try:
        names = spec.get("joints", {}).get("names", [])
        links = spec.get("links", {})
        for key in ("moving_jaw", "moving_jaw_mesh"):
            if key in links and not isinstance(links[key], str):
                raise ValueError(f"links.{key} must be a string")
        kin = spec.get("kinematics", {})
        if not isinstance(kin, dict):
            raise ValueError("kinematics must be a table")
        if "file" in kin and not isinstance(kin["file"], str):
            raise ValueError("kinematics.file must be a string")
        if "ee_link" in kin and kin["ee_link"] != links["ee"]:
            raise ValueError("kinematics.ee_link must equal links.ee")
        for key, size in (("tcp_position_m", 3), ("tcp_quat_wxyz", 4), ("approach_axis", 3), ("closing_axis", 3)):
            if key in kin:
                value = kin[key]
                if not _vector(value, size):
                    raise ValueError(f"Invalid kinematics.{key}")
                if key != "tcp_position_m" and abs(_math.sqrt(sum(v*v for v in value)) - 1) > 1e-6:
                    raise ValueError(f"kinematics.{key} must be unit length")
        if "approach_axis" in kin and "closing_axis" in kin:
            if abs(sum(a*b for a, b in zip(kin["approach_axis"], kin["closing_axis"]))) > 1e-6:
                raise ValueError("Approach and closing axes must be perpendicular")
        grip = spec.get("gripper", {})
        if not isinstance(grip, dict):
            raise ValueError("gripper must be a table")
        if "joint" in grip and grip["joint"] not in names:
            raise ValueError("Unknown gripper joint")
        for key in ("open_rad", "closed_rad", "max_object_width_m"):
            if key in grip and (not _number(grip[key]) or (key == "max_object_width_m" and grip[key] <= 0)):
                raise ValueError(f"Invalid gripper.{key}")
        if "opening_table" in grip:
            from core.arms.gripper import _opening_table
            _opening_table(grip["opening_table"])
        expert = spec.get("expert", {})
        if not isinstance(expert, dict):
            raise ValueError("expert must be a table")
        if "arm_joints" in expert:
            joints = expert["arm_joints"]
            if (not isinstance(joints, list) or not all(isinstance(j, str) for j in joints)
                    or len(set(joints)) != len(joints) or set(joints) - set(names) or grip.get("joint") in joints):
                raise ValueError("Invalid expert.arm_joints")
        for key in ("home_rad", "max_joint_speed_rad_s"):
            if key not in expert:
                continue
            value = expert[key]
            if key == "max_joint_speed_rad_s" and _number(value) and value > 0:
                continue
            if (not isinstance(value, dict) or set(value) - set(names)
                    or any(not _number(v) or (key != "home_rad" and v <= 0) for v in value.values())):
                raise ValueError(f"Invalid expert.{key}")
        for key in ("max_gripper_speed_rad_s", "max_ee_speed_m_s", "ik_tol_pos_m", "ik_tol_rot_rad",
                    "ik_position_weight", "ik_axis_weight", "ik_roll_weight"):
            if key in expert:
                value = expert[key]
                if not _number(value) or value < 0 or ("weight" not in key and value == 0):
                    raise ValueError(f"Invalid expert.{key}")
    except (ValueError, TypeError, KeyError) as exc:
        raise _ConfigError(f"{path}: {exc}") from exc
