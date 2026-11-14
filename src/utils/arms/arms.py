"""Find, read and select arm folders and load their optional utils.py hooks; no simulator imports."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tomllib
from types import ModuleType

from utils.arms.spec import ArmSpec, parse_arm_spec
from utils.paths import DEFAULT_ARM
from utils.workspace import item_dir, load_file_module, read_toml


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


def load_arm_spec(name: str | None = None, variant: str | None = None) -> tuple[ArmSpec, Path]:
    name = selected_arm_name() if name is None else name
    directory = item_dir("arms", name)
    try:
        raw = read_toml(directory / "arm.toml")["arm"]
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
        spec = parse_arm_spec(name, directory, raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"arms/{name}/arm.toml: {error}") from error
    return spec, directory


def load_arm_utils(spec: ArmSpec) -> ModuleType | None:
    """Load optional hooks by path; importing this loader itself never imports Isaac."""
    if spec.utils is None:
        return None
    return load_file_module(spec.utils, "_s2r_arm_utils_" + hashlib.sha256(str(spec.utils).encode()).hexdigest())
