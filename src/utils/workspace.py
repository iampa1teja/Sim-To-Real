"""Shared lookup and validation for workspace items (arms, scenes, tasks); no simulator imports."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
import tomllib
from types import ModuleType

from utils.paths import repo_dir


def kind_dir(kind: str) -> Path:
    """workspace/<kind>, or the S2R_<KIND>_DIR override (arms, scenes, tasks)."""
    configured = os.environ.get(f"S2R_{kind.upper()}_DIR")
    if not configured:
        return repo_dir(kind)
    # Arm and scene overrides have always expanded "~"; the task override never has.
    return Path(configured) if kind == "tasks" else Path(configured).expanduser()


def available(kind: str, marker: str) -> list[str]:
    return sorted(p.name for p in kind_dir(kind).iterdir() if (p / marker).is_file())


def item_dir(kind: str, name) -> Path:
    """Resolve a selected arm or scene folder; folder names are public Python identifiers."""
    item_name(kind, name)
    return (kind_dir(kind) / name).resolve()


def item_name(kind: str, name):
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"{kind[:-1].capitalize()} folder name {name!r} must be a public Python identifier")
    return name


def read_toml(path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def load_file_module(path, name: str) -> ModuleType:
    """Import a file by path once per name; a failed import leaves nothing in sys.modules."""
    if name not in sys.modules:
        definition = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(definition)
        sys.modules[name] = module
        try:
            definition.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    return tuple(number(component, label) for component in value)


def text(value, label, *, empty=False):
    if not isinstance(value, str) or (not value and not empty):
        raise ValueError(f"{label} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def relative_file(directory, value, label, rule):
    """A manifest-relative file; ``rule`` is the caller's wording for the '..' error."""
    path = Path(text(value, label))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be {rule} without '..'")
    path = directory / path
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path.resolve()
