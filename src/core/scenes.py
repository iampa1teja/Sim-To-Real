"""Read scenes/<name>/scene.toml and its measured setup JSON (no Isaac / pxr imports)."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from core.paths import repo_dir

# Selected scene; set S2R_SCENE before importing sim/core modules to change it.
SCENE_NAME = os.environ.get("S2R_SCENE", "rpd_meeting_room")


def scene_dir(name: str | None = None) -> Path:
    """scenes/<name>/, or $S2R_SCENE_DIR when set (it then wins over ``name``)."""
    configured = os.environ.get("S2R_SCENE_DIR")
    return Path(configured) if configured else repo_dir("scenes") / (name or SCENE_NAME)


def load_scene_spec(name: str | None = None) -> dict[str, Any]:
    """The [scene] table of scene.toml; an empty dict when the scene has no scene.toml."""
    path = scene_dir(name) / "scene.toml"
    if not path.is_file():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle).get("scene", {})


def setup_path(name: str | None = None) -> Path:
    """The scene's measured setup JSON (scene.toml ``setup``, default real_setup.json)."""
    return scene_dir(name) / load_scene_spec(name).get("setup", "real_setup.json")


def load_setup(name: str | None = None) -> dict[str, Any]:
    """Parsed measured setup: robot pose, cameras, lighting, joint mapping and limits."""
    return json.loads(setup_path(name).read_text())
