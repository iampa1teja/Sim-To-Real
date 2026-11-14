"""Typed scene manifests, optional arm mounts and fitted appearance data."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
import os
import re
from pathlib import Path
import tomllib
from typing import Any

from utils.paths import DEFAULT_SCENE
from utils.workspace import boolean, item_name, kind_dir, number, read_toml, relative_file, text, vector


def selected_scene_name() -> str:
    return os.environ.get("S2R_SCENE", DEFAULT_SCENE)


def scene_dir(name: str | None = None) -> Path:
    configured = os.environ.get("S2R_SCENE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (kind_dir("scenes") / (selected_scene_name() if name is None else name)).resolve()


def _table(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table/object")
    return value


_SCENE_FILE = "a file relative to the scene folder"


def _axis(value, label):
    values = vector(value, 3, label)
    if math.hypot(*values) < 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return values


@dataclass(frozen=True, slots=True)
class AnchorSpec:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_degrees: float = 0.0
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    right: tuple[float, float, float] | None = None
    prim_name: str | None = None
    contact_material: bool = False
    xy_min: tuple[float, float] | None = None
    xy_max: tuple[float, float] | None = None
    contact_prim_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneCameraSpec:
    position: tuple[float, float, float]
    target: tuple[float, float, float] | None
    euler_degrees: tuple[float, float, float] | None
    up: tuple[float, float, float] | None
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    data_types: tuple[str, ...] = ("rgb",)
    depth_sensor: bool = False
    prim_name: str | None = None
    depth_prim_name: str | None = None
    depth_sensor_name: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerSpec:
    eye: tuple[float, float, float] = (1.8, -1.8, 1.85)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.75)
    camera: str | None = None


@dataclass(frozen=True, slots=True)
class MountSpec:
    anchor: str = "mount"
    translation_correction_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_joint_positions: dict[str, float] = field(default_factory=dict)
    source: dict[str, Any] | str = field(default_factory=dict)
    path: Path | None = None
    exists: bool = False


@dataclass(frozen=True, slots=True)
class SceneSpec:
    name: str
    directory: Path
    description: str
    room_usd: Path
    room_root: str
    anchors: dict[str, AnchorSpec]
    cameras: dict[str, SceneCameraSpec]
    lighting: dict[str, Any]
    viewer: ViewerSpec
    physics: dict[str, Any]
    renderer: dict[str, Any]
    appearance: dict[str, Any]
    appearance_path: Path | None
    hook: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _anchor(name, raw):
    raw = _table(raw, f"anchors.{name}")
    up = _axis(raw.get("up", [0, 0, 1]), f"anchors.{name}.up")
    right = _axis(raw["right"], f"anchors.{name}.right") if "right" in raw else None
    if right is not None:
        cross = tuple(up[(index+1) % 3]*right[(index+2) % 3] - up[(index+2) % 3]*right[(index+1) % 3]
                      for index in range(3))
        if math.hypot(*cross) < 1e-12:
            raise ValueError(f"anchors.{name}.right and up must not be parallel")
    low = vector(raw["xy_min"], 2, f"anchors.{name}.xy_min") if "xy_min" in raw else None
    high = vector(raw["xy_max"], 2, f"anchors.{name}.xy_max") if "xy_max" in raw else None
    if (low is None) != (high is None) or (low is not None and any(a >= b for a, b in zip(low, high))):
        raise ValueError(f"anchors.{name} xy_min/xy_max must be increasing pairs")
    pattern = raw.get("contact_prim_pattern")
    if pattern is not None:
        pattern = text(pattern, f"anchors.{name}.contact_prim_pattern")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"anchors.{name}.contact_prim_pattern is invalid: {error}") from error
    return AnchorSpec(
        vector(raw.get("position", [0, 0, 0]), 3, f"anchors.{name}.position"),
        number(raw.get("yaw_degrees", 0), f"anchors.{name}.yaw_degrees"), up, right,
        text(raw["prim_name"], f"anchors.{name}.prim_name") if "prim_name" in raw else None,
        boolean(raw.get("contact_material", False), f"anchors.{name}.contact_material"), low, high, pattern,
        copy.deepcopy(_table(raw.get("metadata", {}), "metadata")),
    )


def _camera(name, raw):
    raw = _table(raw, f"cameras.{name}")
    position = vector(raw["position"], 3, f"cameras.{name}.position")
    target = vector(raw["target"], 3, f"cameras.{name}.target") if "target" in raw else None
    angles = vector(raw["euler_degrees"], 3, f"cameras.{name}.euler_degrees") if "euler_degrees" in raw else None
    up = _axis(raw["up"], f"cameras.{name}.up") if "up" in raw else None
    if angles is None and (target is None or target == position):
        raise ValueError(f"cameras.{name} needs Euler angles or a target distinct from its position")
    if angles is None:
        direction = tuple(b-a for a, b in zip(position, target))
        normal = up or (0.0, 0.0, 1.0)
        cross = tuple(direction[(index+1) % 3]*normal[(index+2) % 3] - direction[(index+2) % 3]*normal[(index+1) % 3]
                      for index in range(3))
        if math.hypot(*cross) < 1e-12:
            raise ValueError(f"cameras.{name} target direction must not be parallel to up")
    for key in ("width", "height"):
        if type(raw[key]) is not int or raw[key] <= 0:
            raise ValueError(f"cameras.{name}.{key} must be a positive integer")
    optics = tuple(number(raw[key], f"cameras.{name}.{key}") for key in
                   ("focal_length", "horizontal_aperture", "vertical_aperture"))
    if min(optics) <= 0:
        raise ValueError(f"cameras.{name} optical dimensions must be positive")
    kinds = raw.get("data_types", ["rgb"])
    if not isinstance(kinds, list) or not kinds or any(not isinstance(kind, str) or not kind for kind in kinds):
        raise ValueError(f"cameras.{name}.data_types must be a nonempty list of names")
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"cameras.{name}.data_types contains duplicate names")
    metadata = copy.deepcopy(_table(raw.get("metadata", {}), "metadata"))
    if "orientation_convention" in raw:
        metadata["orientation_convention"] = raw["orientation_convention"]
    return SceneCameraSpec(position, target, angles, up, raw["width"], raw["height"], *optics, tuple(kinds),
                            boolean(raw.get("depth_sensor", False), f"cameras.{name}.depth_sensor"),
                            raw.get("prim_name"), raw.get("depth_prim_name"), raw.get("depth_sensor_name"),
                            copy.deepcopy(_table(raw.get("calibration", {}), f"cameras.{name}.calibration")), metadata)


def load_scene_spec(name: str | None = None) -> SceneSpec:
    name = selected_scene_name() if name is None else name
    directory = scene_dir(item_name("scenes", name))
    try:
        raw = _table(read_toml(directory / "scene.toml")["scene"], "scene")
        if raw["name"] != name or directory.name != name:
            raise ValueError(f"scene.name {raw['name']!r} must equal folder/selected name {name!r}")
        override = os.environ.get("S2R_ROOM_USD")
        if override:
            # Runtime overrides name files in the process filesystem. The
            # CLI runs simulation commands at /workspace/s2r, so relative
            # overrides are project-relative there, unlike manifest paths.
            room = Path(override).expanduser().resolve()
            if not room.is_file():
                raise ValueError(f"S2R_ROOM_USD file not found: {room}")
        else:
            room = relative_file(directory, raw["room_usd"], "scene.room_usd", _SCENE_FILE)
        root = text(raw.get("room_root", ""), "scene.room_root", empty=True)
        if root.startswith("/") or ".." in Path(root).parts:
            raise ValueError("scene.room_root must be a relative prim path")
        appearance_path = relative_file(directory, raw["appearance"], "scene.appearance", _SCENE_FILE) if raw.get("appearance") else None
        appearance = json.loads(appearance_path.read_text()) if appearance_path else {}
        if not isinstance(appearance, dict):
            raise ValueError("Scene appearance JSON must contain an object")
        anchors = {key: _anchor(key, value) for key, value in _table(raw.get("anchors", {}), "scene.anchors").items()}
        cameras = {key: _camera(key, value) for key, value in _table(raw.get("cameras", {}), "scene.cameras").items()}
        view = _table(raw.get("viewer", {}), "scene.viewer")
        viewer = ViewerSpec(vector(view.get("eye", [1.8, -1.8, 1.85]), 3, "viewer.eye"),
                            vector(view.get("lookat", [0, 0, 0.75]), 3, "viewer.lookat"), view.get("camera"))
        if viewer.camera is not None and viewer.camera not in cameras:
            raise ValueError(f"viewer.camera {viewer.camera!r} is not a fixed scene camera")
        hook = raw.get("hook")
        if hook is not None:
            filename, separator, function = hook.partition(":")
            if not separator or not function.isidentifier():
                raise ValueError("scene.hook must be a relative file.py:function entry point")
            relative_file(directory, filename, "scene.hook", _SCENE_FILE)
        for forbidden in ("env_cfg", "setup", "room_usd_runtime"):
            if forbidden in raw:
                raise ValueError(f"Legacy scene key {forbidden!r} is not part of the declarative contract")
        return SceneSpec(name, directory, raw.get("description", ""), room, root, anchors, cameras,
                         copy.deepcopy(_table(raw.get("lighting", {}), "scene.lighting")), viewer, copy.deepcopy(_table(raw.get("physics", {}), "scene.physics")),
                         copy.deepcopy(_table(raw.get("renderer", {}), "scene.renderer")), appearance, appearance_path, hook,
                         copy.deepcopy(_table(raw.get("metadata", {}), "metadata")))
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"scenes/{name}/scene.toml: {error}") from error


def load_mount_spec(scene: SceneSpec, arm) -> MountSpec:
    name = arm if isinstance(arm, str) else arm.name
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Invalid mount arm name {name!r}")
    path = scene.directory / "mounts" / f"{name}.toml"
    if not path.exists():
        return MountSpec()
    try:
        raw = _table(read_toml(path)["mount"], "mount")
        anchor = text(raw.get("anchor", "mount"), "mount.anchor")
        if anchor not in scene.anchors:
            raise ValueError(f"Mount anchor {anchor!r} does not exist in scene {scene.name!r}")
        starts = {key: number(value, f"mount.start_joint_positions.{key}")
                  for key, value in _table(raw.get("start_joint_positions", {}), "mount.start_joint_positions").items()}
        if not isinstance(arm, str) and not set(starts) <= set(arm.joint_names):
            raise ValueError(f"Mount start pose has unknown joints for arm {arm.name!r}")
        source = raw.get("source", {})
        if not isinstance(source, (dict, str)):
            raise ValueError("mount.source must be text or a provenance object")
        return MountSpec(anchor, vector(raw.get("translation_correction_m", [0, 0, 0]), 3, "mount.translation_correction_m"),
                          starts, copy.deepcopy(source), path.resolve(), True)
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot load mount {path}: {error}") from error
