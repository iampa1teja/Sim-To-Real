"""Two-file task specifications, explicit cached loading, stages and expert composition.

Task modules import checks/moves without torch or Isaac. Selection is always explicit;
there is no import-time task or legacy package loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import hashlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from utils.geometry import Pose
from utils.workspace import available, kind_dir, read_toml, load_file_module

# Variants of two-file tasks, all applied by the generic task environment builder.
GENERIC_VARIANTS = {variant: f"sim.workspace.envs.task_env:apply_{variant}" for variant in ("teleop", "eval", "eval_dr")}


def gym_id(task: str, variant: str) -> str:
    return f"S2R-{task}-{variant}"


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
    task_py: Path

    def entry_point(self, variant: str) -> str:
        if variant not in self.variants:
            raise KeyError(f"Task {self.name!r} has no variant {variant!r}; choices: {sorted(self.variants)}")
        return self.variants[variant]

    def all_variants(self) -> list[str]:
        return list(self.variants)


def load_task_spec(name: str) -> TaskSpec:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Task folder name {name!r} must be a valid Python identifier")
    task_dir = kind_dir("tasks") / name
    path = task_dir / "task.toml"
    if not path.is_file():
        raise FileNotFoundError(f"No task.toml for task {name!r}; available: {available('tasks', 'task.toml')}")
    table = read_toml(path).get("task", {})
    if table.get("name", name) != name:
        raise ValueError(f"{path}: [task] name = {table['name']!r} but the folder is {name!r}")

    task_py = task_dir / "task.py"
    if not task_py.is_file():
        raise FileNotFoundError(f"tasks/{name}/ needs task.py")
    # Two-file layout: every variant is built by the generic task environment.
    names = table.get("variants", list(GENERIC_VARIANTS))
    if not isinstance(names, list) or not names or set(names) - set(GENERIC_VARIANTS) or len(set(names)) != len(names):
        raise ValueError(f"{path}: [task] variants must be a nonempty list from {list(GENERIC_VARIANTS)}")
    if "eval_dr" in names and "dr" not in table:
        raise ValueError(f"{path}: variant eval_dr needs a [task.dr] table")
    variants = {name: GENERIC_VARIANTS[name] for name in names}
    if not variants:
        raise ValueError(f"{path}: [task] variants needs at least one variant (e.g. teleop)")
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
        task_py=task_py,
    )


def load_task_file(path: Path):
    """Import a task folder's task.py under a unique module name (no sys.path changes)."""
    path = Path(path).resolve()
    return load_file_module(path, "_s2r_task_" + hashlib.sha256(str(path).encode()).hexdigest()[:16])


def _walk(checks):
    seen, ordered = set(), []

    def visit(check):
        if id(check) in seen:
            return
        seen.add(id(check))
        for child in check.children():
            visit(child)
        ordered.append(check)

    for check in checks:
        visit(check)
    return tuple(ordered)


class Task:
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

    def _sample_start(self, rng: np.random.Generator) -> dict[str, Pose]:
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
        from utils.geometry import quaternion_multiply, rotation_matrix_wxyz, wxyz_to_xyzw

        source = {"position_world": (0, 0, 0), "orientation_world_wxyz": (1, 0, 0, 0)} if frame == "world" else scene.get("frames", {}).get(frame)
        if source is None:
            raise ValueError(f"Captured scene has no start frame {frame!r}")
        source_p = np.asarray(source["position_world"])
        source_q = np.asarray(source["orientation_world_wxyz"])
        base_q = np.asarray(scene["base_quaternion_wxyz"])
        base_inverse_xyzw = (-base_q[1], -base_q[2], -base_q[3], base_q[0])
        world = source_p + rotation_matrix_wxyz(source_q) @ np.asarray(pose.position)
        position = rotation_matrix_wxyz(base_q).T @ (world - scene["base_position_world"])
        orientation = quaternion_multiply(base_inverse_xyzw, quaternion_multiply(
            wxyz_to_xyzw(source_q), pose.quat_xyzw))
        return Pose.from_lists(position, orientation)

    def _start_is_valid(self, start: dict[str, Pose], scene: dict[str, Any]) -> bool:
        """Require sampled footprints to fit the table and clear other objects."""
        from utils.geometry import footprint, footprints_overlap, inside_polygon, resting_pose, rotation_matrix_wxyz, upright_orientation_wxyz, xyzw_to_wxyz

        footprints = {}
        base_rotation = rotation_matrix_wxyz(scene["base_quaternion_wxyz"])
        for name, pose in start.items():
            if name not in self.spec.objects:
                return False
            spec = self.spec.objects[name]
            size = scene.get("objects", {}).get(name, {}).get("size", spec.size or spec.params.get("size_outer"))
            if size is None:
                continue
            q = xyzw_to_wxyz(pose.quat_xyzw)
            position = np.asarray(pose.position)
            rotation = rotation_matrix_wxyz(q)
            if "table_plane_world" in scene:
                q = upright_orientation_wxyz(scene["base_quaternion_wxyz"], scene["table_plane_world"], q)
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
                fixed[name] = footprint(obj["position"], rotation_matrix_wxyz(obj["orientation_wxyz"]), obj["size"],
                                        bottom_origin=bool(obj.get("bottom_origin", False)))
        items = list(footprints.items())
        for index, (_, polygon) in enumerate(items):
            if any(footprints_overlap(polygon, other) for other in fixed.values()):
                return False
            if any(footprints_overlap(polygon, other) for _, other in items[:index]):
                return False
        return True

    def is_success(self, flags: dict[str, bool]) -> bool:
        return bool(flags.get(self.stages[-1], False))

    def furthest_stage(self, flags: dict[str, bool]) -> str | None:
        return next((stage for stage in reversed(self.stages) if flags.get(stage, False)), None)

    def __deepcopy__(self, memo):
        """Copy per-env task/check state while retaining the imported code module."""
        from copy import deepcopy

        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        # Runtime owners are live simulator environments, not task configuration.
        # Keep their identity; a frame from another environment initializes fresh state.
        for check in self.checks:
            owner = getattr(check, "_owner", None)
            if owner is not None:
                memo[id(owner)] = owner
        for name, value in vars(self).items():
            setattr(copied, name, value if name == "module" else deepcopy(value, memo))
        return copied

    def __init__(self, spec: TaskSpec, module=None):
        if spec.task_py is None:
            raise ValueError(f"Task {spec.name!r} has no task.py")
        self.spec = spec
        self.module = load_task_file(spec.task_py) if module is None else module
        stages = getattr(self.module, "stages", None)
        if not isinstance(stages, dict) or tuple(stages) != spec.stages:
            raise ValueError(f"{spec.task_py}: stages must be a dict with the task.toml stages {list(spec.stages)} "
                             f"in order, got {list(stages) if isinstance(stages, dict) else stages!r}")
        self.stage_checks = dict(stages)
        # Every distinct check, children first; the env resets each of them once.
        self.checks = _walk(self.stage_checks.values())
        terms = spec.raw.get("observations", {}).get("subtask_terms", {})
        for stage in (spec.stages[-1], *terms.values()):
            if stage not in self.stage_checks or not self.stage_checks[stage].in_sim:
                raise ValueError(f"{spec.task_py}: stage {stage!r} drives a simulator term and needs a "
                                 "simulator-evaluated check")

    def _hook(self, name):
        return getattr(self.module, name, None)

    def signals(self, env) -> dict[str, Any]:
        from sim.workspace.envs.task_env import task_signals

        return task_signals(env, self)

    def observe(self, state: StepState) -> dict[str, bool]:
        return {name: bool(check.observe(state, name, self.spec.objects)) for name, check in self.stage_checks.items()}

    def sample_start(self, rng):
        rule = self._hook("sample_start")
        if rule is None:
            return self._sample_start(rng)
        name = self.spec.start.get("object", self.spec.tracked_objects[0])
        return rule(self, rng, lambda: self._sample_start(rng)[name])

    def start_is_valid(self, start, scene) -> bool:
        if not self._start_is_valid(start, scene):
            return False
        rule = self._hook("start_is_valid")
        return True if rule is None else bool(rule(self, start, scene))

    def failure_mode(self, flags, state):
        from .stages import failure_mode

        holding = bool(state is not None and state.signals.get("holding", False))
        return failure_mode(self.stages, flags, holding)["failure_mode"]

    def progress_label(self, flags, state):
        stage = self.furthest_stage(flags) or "approaching"
        label = "stage=" + stage.replace("_", " ")
        rule = self._hook("progress_label")
        return label if rule is None else rule(label, flags, state)

    def object_colors(self) -> dict[str, dict]:
        """``[task.dr] object_colors``: object name -> {colour name: RGB or "from_setup"}."""
        return dict(self.spec.raw.get("dr", {}).get("object_colors", {}))

    def episode_metadata(self, env, index=0) -> dict[str, Any]:
        metadata = {}
        colors = self.object_colors()
        if colors:
            # The colour-randomized object names the episode colour (its first palette entry by default).
            name, palette = next(iter(colors.items()))
            recorded = getattr(env, "_object_colors", {}).get(name)
            metadata["color"] = recorded[index] if recorded is not None and recorded[index] is not None else next(iter(palette))
        rule = self._hook("episode_metadata")
        return metadata if rule is None else rule(metadata)

    def expert(self):
        moves = getattr(self.module, "expert", None)
        if moves is None:
            raise NotImplementedError(f"Task {self.name!r} has no expert in task.py")
        from sim.actions.action import Expert

        return Expert(moves, self.spec.objects, getattr(self.module, "expert_options", None))


def sample_start_in_base(task: Task, rng: np.random.Generator, scene: dict[str, Any]) -> dict[str, Pose]:
    """Keep plugin sampling small and convert its declared frame at the boundary."""
    return {name: task.start_pose_to_base(pose, scene) for name, pose in task.sample_start(rng).items()}


@cache
def load_task(name: str) -> Task:
    """Load a task.toml + task.py by an explicit folder name; never read environment selection."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Task folder name {name!r} must be a valid Python identifier")
    if name not in available("tasks", "task.toml"):
        raise FileNotFoundError(f"Unknown task {name!r}; available: {available('tasks', 'task.toml')}")
    return Task(load_task_spec(name))


def resolve(entry_point: str) -> Any:
    module_name, _, attr = entry_point.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{module_name} has no {attr!r} (from entry point {entry_point!r})") from None
