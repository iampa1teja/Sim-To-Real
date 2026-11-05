"""Read arms/<name>/arm.toml (pure tomllib, no Isaac imports).

Both the sim (``sim.scene.arm.load_arm_cfg``) and hardware code (``core.arms.joint_calibration``)
read the arm from here, so the same file describes the simulated and the physical arm.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

from core.paths import repo_dir

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
    return spec, arm_dir
