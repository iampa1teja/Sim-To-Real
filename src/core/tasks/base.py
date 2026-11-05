from __future__ import annotations

import os
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.paths import repo_dir

TASK_NAME = os.environ.get("S2R_TASK", "pick_place_cube")
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


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_lists(cls, position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> Pose:
        return cls(tuple(float(v) for v in position), tuple(float(v) for v in quat_xyzw))


@dataclass(frozen=True)
class StepState:
    objects: dict[str, Pose]
    ee: Pose
    gripper: float
    signals: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_dir: Path
    instruction: str
    variants: dict[str, str]
    default_variant: str
    objects: dict[str, ObjectSpec]
    raw: dict[str, Any]

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

    return TaskSpec(
        name=name,
        task_dir=task_dir,
        instruction=table.get("instruction", name.replace("_", " ")),
        variants=variants,
        default_variant=default_variant,
        objects={key: ObjectSpec.from_toml(key, value, task_dir) for key, value in table.get("objects", {}).items()},
        raw=table,
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
    @abstractmethod
    def stages(self) -> tuple[str, ...]:
        ...

    @abstractmethod
    def sample_start(self, rng: np.random.Generator) -> dict[str, Pose]:
        ...

    @abstractmethod
    def observe(self, state: StepState) -> dict[str, bool]:
        ...

    def start_from_row(self, row: dict[str, Any]) -> dict[str, Pose]:
        objects = row.get("objects", {})
        missing = set(self.spec.objects) - set(objects)
        if missing:
            raise KeyError(f"Eval-set row is missing objects {sorted(missing)} for task {self.name!r}")
        return {name: Pose.from_lists(value["position"], value.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0)))
                for name, value in objects.items()}

    def is_success(self, flags: dict[str, bool]) -> bool:
        return bool(flags.get(self.stages[-1], False))

    def furthest_stage(self, flags: dict[str, bool]) -> str | None:
        reached = None
        for stage in self.stages:
            if not flags.get(stage, False):
                break
            reached = stage
        return reached
