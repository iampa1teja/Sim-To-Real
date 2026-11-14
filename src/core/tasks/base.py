from __future__ import annotations

import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.paths import DEFAULT_TASK, repo_dir

TASK_NAME = os.environ.get("S2R_TASK", DEFAULT_TASK)
TASK_PACKAGE = "s2r_tasks"


def tasks_dir() -> Path:
    configured = os.environ.get("S2R_TASKS_DIR")
    return Path(configured) if configured else repo_dir("tasks")


def available_tasks() -> list[str]:
    return sorted(p.name for p in tasks_dir().iterdir() if (p / "task.toml").is_file())


def gym_id(task: str, variant: str) -> str:
    return f"S2R-{task}-{variant}"


def _target(path: Path, key: str, target: Any) -> str:
    if not isinstance(target, str) or target.count(":") != 1:
        raise ValueError(f"{path}: {key} must be 'module:function', got {target!r}")
    return target


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_lists(cls, position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> Pose:
        return cls(tuple(float(v) for v in position), tuple(float(v) for v in quat_xyzw))


@dataclass(frozen=True, slots=True)
class StepState:
    objects: dict[str, Pose]
    ee: Pose
    gripper: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    name: str
    usd: Path | None = None
    size: tuple[float, float, float] | None = None
    rigid: bool = True
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, name: str, table: dict[str, Any], task_dir: Path) -> ObjectSpec:
        table = dict(table)
        usd = table.pop("usd", None)
        size = table.pop("size", None)
        if usd is None and size is None:
            raise ValueError(f"Task object {name!r} needs either usd or size")
        if usd is not None and not (task_dir / usd).is_file():
            raise FileNotFoundError(f"Task object {name!r}: {task_dir / usd} does not exist")
        return cls(
            name=name,
            usd=(task_dir / usd).resolve() if usd is not None else None,
            size=tuple(float(v) for v in size) if size is not None else None,
            rigid=bool(table.pop("rigid", True)),
            position=tuple(float(v) for v in table.pop("position", (0.0, 0.0, 0.0))),
            rotation_euler_degrees=tuple(float(v) for v in table.pop("rotation_euler_degrees", (0.0, 0.0, 0.0))),
            params=table,
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    task_dir: Path
    instruction: str
    variants: dict[str, str]
    default_variant: str
    objects: dict[str, ObjectSpec]
    raw: dict[str, Any]
    stages: tuple[str, ...]
    tracked_objects: tuple[str, ...]
    start: dict[str, Any]

    def entry_point(self, variant: str) -> str:
        if variant not in self.variants:
            raise KeyError(f"Task {self.name!r} has no variant {variant!r}; choices: {sorted(self.variants)}")
        return f"{TASK_PACKAGE}.{self.name}.{self.variants[variant]}"

    def all_variants(self) -> list[str]:
        return list(self.variants)


def load_task_spec(name: str | None = None) -> TaskSpec:
    name = name or TASK_NAME
    task_dir = tasks_dir() / name
    path = task_dir / "task.toml"
    if not path.is_file():
        raise FileNotFoundError(f"No task.toml for task {name!r}; available: {available_tasks()}")
    with open(path, "rb") as handle:
        table = tomllib.load(handle).get("task", {})
    if table.get("name", name) != name:
        raise ValueError(f"{path}: [task] name = {table['name']!r} but the folder is {name!r}")

    variants = {key: _target(path, f"[task.variants] {key}", value) for key, value in table.get("variants", {}).items()}
    if not variants:
        raise ValueError(f"{path}: [task.variants] needs at least one variant (e.g. teleop)")
    default_variant = table.get("default_variant", next(iter(variants)))
    if default_variant not in variants:
        raise ValueError(f"{path}: default_variant {default_variant!r} is not in [task.variants]")

    objects = {key: ObjectSpec.from_toml(key, value, task_dir) for key, value in table.get("objects", {}).items()}
    stages = tuple(table.get("stages", ()))
    tracked = tuple(table.get("tracked_objects", objects))
    for label, values in (("stages", stages), ("tracked_objects", tracked)):
        if not values or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"{path}: [task] {label} must be a nonempty list of unique names")
    if set(tracked) - set(objects):
        raise ValueError(f"{path}: tracked_objects contains unknown objects {sorted(set(tracked) - set(objects))}")
    start = table.get("start", {})
    if not isinstance(start, dict):
        raise ValueError(f"{path}: [task.start] must be a table")
    if start.get("object", tracked[0]) not in objects:
        raise ValueError(f"{path}: task.start.object must name a declared object")
    return TaskSpec(
        name=name,
        task_dir=task_dir,
        instruction=table.get("instruction", name.replace("_", " ")),
        variants=variants,
        default_variant=default_variant,
        objects=objects,
        raw=table,
        stages=stages,
        tracked_objects=tracked,
        start=dict(start),
    )


class Task(ABC):
    def __init__(self, spec: TaskSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def instruction(self) -> str:
        return self.spec.instruction

    def gym_id(self, variant: str | None = None) -> str:
        variant = variant or self.spec.default_variant
        self.spec.entry_point(variant)
        return gym_id(self.name, variant)

    @property
    def stages(self) -> tuple[str, ...]:
        return self.spec.stages

    def sample_start(self, rng: np.random.Generator) -> dict[str, Pose]:
        """Sample the primary object in the frame declared by task.start."""
        start = self.spec.start
        if "xy_min" not in start or "xy_max" not in start:
            raise ValueError(f"Task {self.name!r} random starts need task.start.xy_min and xy_max")
        name = start.get("object", self.spec.tracked_objects[0])
        x, y = rng.uniform(start["xy_min"], start["xy_max"])
        yaw = np.deg2rad(rng.uniform(*start.get("yaw_deg", (0.0, 90.0))))
        pose = Pose.from_lists((x, y, start.get("z", 0.0)), (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)))
        return {name: pose}

    def start_pose_to_base(self, pose, scene=None):
        frame = self.spec.start.get("frame", "arm_base")
        if frame == "arm_base":
            return pose
        if scene is None:
            raise ValueError(f"Task {self.name!r} starts in frame {frame!r} require captured scene geometry")
        from core.eval.geometry import rotation_matrix
        from utils.geometry import quaternion_multiply

        source = {"position_world": (0, 0, 0), "orientation_world_wxyz": (1, 0, 0, 0)} if frame == "world" else scene.get("frames", {}).get(frame)
        if source is None:
            raise ValueError(f"Captured scene has no start frame {frame!r}")
        source_p = np.asarray(source["position_world"])
        source_q = np.asarray(source["orientation_world_wxyz"])
        base_q = np.asarray(scene["base_quaternion_wxyz"])
        base_inverse_xyzw = (-base_q[1], -base_q[2], -base_q[3], base_q[0])
        world = source_p + rotation_matrix(source_q) @ np.asarray(pose.position)
        position = rotation_matrix(base_q).T @ (world - scene["base_position_world"])
        orientation = quaternion_multiply(base_inverse_xyzw, quaternion_multiply(
            (source_q[1], source_q[2], source_q[3], source_q[0]), pose.quat_xyzw))
        return Pose.from_lists(position, orientation)

    @abstractmethod
    def signals(self, env) -> dict[str, Any]:
        """Optional live sensor signals; simulator imports belong to task folders."""
        raise NotImplementedError

    def episode_metadata(self, env, index=0) -> dict[str, Any]:
        return {}

    def expert(self):
        """Load the optional task-folder expert without hiding its import errors."""
        if not (self.spec.task_dir / "expert.py").is_file():
            raise NotImplementedError(f"Task {self.name!r} has no expert.py scripted expert")
        from core.tasks.loader import load_task_module

        module = load_task_module("expert", self.name)
        factory = getattr(module, "create", None)
        if not callable(factory):
            raise TypeError(f"Task {self.name!r} expert.py must export create(task)")
        return factory(self)

    @abstractmethod
    def observe(self, state: StepState) -> dict[str, bool]:
        raise NotImplementedError

    def start_is_valid(self, start: dict[str, Pose], scene: dict[str, Any]) -> bool:
        """Require sampled footprints to fit the table and clear other objects."""
        from core.eval.geometry import footprint, footprints_overlap, inside_polygon, resting_pose, rotation_matrix, upright_orientation

        footprints = {}
        base_rotation = rotation_matrix(scene["base_quaternion_wxyz"])
        for name, pose in start.items():
            if name not in self.spec.objects:
                return False
            spec = self.spec.objects[name]
            size = scene.get("objects", {}).get(name, {}).get("size", spec.size or spec.params.get("size_outer"))
            if size is None:
                continue
            x, y, z, w = pose.quat_xyzw
            q = (w, x, y, z)
            position = np.asarray(pose.position)
            rotation = rotation_matrix(q)
            if "table_plane_world" in scene:
                q = upright_orientation(scene["base_quaternion_wxyz"], scene["table_plane_world"], q)
                world, world_rotation = resting_pose(position, q, scene["base_position_world"],
                                                     scene["base_quaternion_wxyz"], scene["table_plane_world"], size)
                position = base_rotation.T @ (world - scene["base_position_world"])
                rotation = base_rotation.T @ world_rotation
            polygon = footprint(position, rotation, size, bottom_origin=bool(spec.params.get("bottom_origin", False)))
            if "table_outline_base_xy" in scene and not all(inside_polygon(point, scene["table_outline_base_xy"]) for point in polygon):
                return False
            footprints[name] = polygon
        fixed = {}
        for name, obj in scene.get("objects", {}).items():
            if name in start:
                continue
            if "footprint_base_xy" in obj:
                fixed[name] = np.asarray(obj["footprint_base_xy"])
            elif all(key in obj for key in ("position", "orientation_wxyz", "size")):
                fixed[name] = footprint(obj["position"], rotation_matrix(obj["orientation_wxyz"]), obj["size"],
                                        bottom_origin=bool(obj.get("bottom_origin", False)))
        items = list(footprints.items())
        for index, (_, polygon) in enumerate(items):
            if any(footprints_overlap(polygon, other) for other in fixed.values()):
                return False
            if any(footprints_overlap(polygon, other) for _, other in items[:index]):
                return False
        return True

    def failure_mode(self, flags: dict[str, bool], state: StepState | None) -> str | None:
        return None if self.is_success(flags) else "timeout"

    def progress_label(self, flags: dict[str, bool], state: StepState | None) -> str:
        stage = self.furthest_stage(flags) or "approaching"
        return "stage=" + stage.replace("_", " ")

    def is_success(self, flags: dict[str, bool]) -> bool:
        return bool(flags.get(self.stages[-1], False))

    def furthest_stage(self, flags: dict[str, bool]) -> str | None:
        return next((stage for stage in reversed(self.stages) if flags.get(stage, False)), None)


def sample_start_in_base(task: Task, rng: np.random.Generator, scene: dict[str, Any]) -> dict[str, Pose]:
    """Keep plugin sampling small and convert its declared frame at the boundary."""
    return {name: task.start_pose_to_base(pose, scene) for name, pose in task.sample_start(rng).items()}
