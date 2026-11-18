from __future__ import annotations

import math as _math
import os
import string as _string
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from dataclasses import dataclass, field, replace as _replace
from pathlib import Path
from types import MappingProxyType as _MappingProxyType
from typing import Any

import numpy as np

from core.generate.errors import ConfigError as _ConfigError
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
    tcp: Pose | None = None
    joints: _Mapping[str, float] | None = None


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
class ObjectSetSpec:
    name: str
    prefix: str
    count_max: int
    template: ObjectSpec

    def slot_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_{i}" for i in range(self.count_max))


@dataclass(frozen=True)
class HoldoutSpec:
    colors: tuple[str, ...] = ()
    counts: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "colors", tuple(self.colors))
        object.__setattr__(self, "counts", tuple(self.counts))


class _Palette(_Mapping):
    def __init__(self, values):
        self._values = _MappingProxyType(dict(values))

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        value = self._values[key]
        if isinstance(value, str):
            from core.scenes import load_setup
            value = load_setup()["object_materials"][value.removeprefix("from_setup:")]["color"]
        return tuple(float(c) for c in value)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_dir: Path
    instruction: str
    variants: dict[str, str]
    default_variant: str
    objects: dict[str, ObjectSpec]
    raw: dict[str, Any]
    object_sets: dict[str, ObjectSetSpec] = field(default_factory=dict)
    palette: _Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    holdout: HoldoutSpec = field(default_factory=HoldoutSpec)
    instruction_templates: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.palette, _Palette):
            object.__setattr__(self, "palette", _Palette(self.palette))

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

    spec = TaskSpec(
        name=name,
        task_dir=task_dir,
        instruction=table.get("instruction", name.replace("_", " ")),
        variants=variants,
        default_variant=default_variant,
        objects={key: ObjectSpec.from_toml(key, value, task_dir) for key, value in table.get("objects", {}).items()},
        raw=table,
    )
    return _load_v2(spec)


def _strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _failure_rules(raw):
    table = raw.get("failures", {})
    if not isinstance(table, dict):
        raise _ConfigError("task.failures must be a table")
    rules = table.get("rules", [])
    if not isinstance(rules, list):
        raise _ConfigError("failure rules must be a list")
    result = []
    for rule in rules:
        if (not isinstance(rule, dict) or set(rule) - {"name", "if", "unless"}
                or not isinstance(rule.get("name"), str)
                or not _strings(rule.get("if", [])) or not _strings(rule.get("unless", []))):
            raise _ConfigError("Invalid failure rule")
        result.append(FailureRule(rule["name"], tuple(rule.get("if", ())), tuple(rule.get("unless", ()))))
    if result and (result[-1].if_ or result[-1].unless):
        raise _ConfigError("The last failure rule must be unconditional")
    return tuple(result)


def _load_v2(spec):
    from core.tasks.sampling import _validate_start

    table, objects = spec.raw, dict(spec.objects)
    try:
        sets = {}
        names = set(objects)
        set_tables = table.get("object_sets", {})
        if not isinstance(set_tables, dict) or names.intersection(set_tables):
            raise _ConfigError("Object and set names must be distinct")
        names.update(set_tables)
        for name, data in set_tables.items():
            data = dict(data)
            prefix, count = data.pop("prefix", name), data.pop("count_max", None)
            if type(count) is not int or count < 1 or not isinstance(prefix, str) or not prefix:
                raise _ConfigError("Object set needs a prefix and integer count_max >= 1")
            oset = ObjectSetSpec(name, prefix, count, ObjectSpec.from_toml(name, data, spec.task_dir))
            sets[name] = oset
            for i, slot in enumerate(oset.slot_names()):
                if slot in names:
                    raise _ConfigError(f"Object name clash: {slot}")
                names.add(slot)
                objects[slot] = _replace(oset.template, name=slot,
                                         params={**oset.template.params, "set": name, "slot": i})
        palette = {}
        for name, value in table.get("palette", {}).items():
            if isinstance(value, str) and value.startswith("from_setup:") and value.removeprefix("from_setup:"):
                palette[name] = value
            elif (isinstance(value, list) and len(value) == 3
                  and all(type(c) in (int, float) and _math.isfinite(c) and 0 <= c <= 1 for c in value)):
                palette[name] = tuple(float(c) for c in value)
            else:
                raise _ConfigError(f"Invalid palette colour: {name}")
        roles = dict(table.get("roles", {}))
        for target in roles.values():
            if not isinstance(target, str) or target not in set(spec.objects) | set(sets):
                raise _ConfigError(f"Role must name a plain object or set: {target}")
        holdout = table.get("holdout", {})
        if (not isinstance(holdout, dict) or set(holdout) - {"colors", "counts"}
                or not _strings(holdout.get("colors", []))
                or not isinstance(holdout.get("counts", []), list)
                or any(type(c) is not int for c in holdout.get("counts", []))):
            raise _ConfigError("Invalid holdout table")
        templates = ()
        if "instruction_template" in table:
            texts = table["instruction_template"].get("texts")
            if not _strings(texts) or not texts:
                raise _ConfigError("instruction_template.texts must be non-empty strings")
            allowed = {f"{role}.{part}" for role in roles for part in ("color", "name")}
            allowed.update(f"goal.{key}" for key in table.get("start", {}).get("goal", {}))
            for text in texts:
                for _, key, fmt, conversion in _string.Formatter().parse(text):
                    if key is not None and (key not in allowed or fmt or conversion):
                        raise _ConfigError(f"Invalid instruction placeholder: {key}")
            templates = tuple(texts)
        _failure_rules(table)
        result = _replace(spec, objects=objects, object_sets=sets, palette=_Palette(palette), roles=roles,
                          holdout=HoldoutSpec(tuple(holdout.get("colors", ())), tuple(holdout.get("counts", ()))),
                          instruction_templates=templates)
        _validate_start(result)
        return result
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        if isinstance(exc, _ConfigError):
            raise
        raise _ConfigError(f"{spec.task_dir}: {exc}") from exc


def _freeze_value(value):
    if isinstance(value, _Mapping):
        return _MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CheckContext:
    roles: _Mapping[str, str]
    start: _Mapping[str, Pose]
    goal: _Mapping[str, Any]
    objects: _Mapping[str, ObjectSpec]
    active: frozenset[str]

    def __post_init__(self):
        for name in ("roles", "start", "objects"):
            object.__setattr__(self, name, _MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "goal", _freeze_value(self.goal))
        object.__setattr__(self, "active", frozenset(self.active))

    @classmethod
    def from_spec(cls, spec, task_spec) -> CheckContext:
        return cls(spec.roles, {name: state.pose for name, state in spec.objects.items()}, spec.goal,
                   episode_objects(task_spec, spec), frozenset(name for name, state in spec.objects.items() if state.active))


def episode_objects(task_spec, spec):
    """Return object definitions with the sizes drawn for this episode."""
    objects = dict(task_spec.objects)
    for name, state in spec.objects.items():
        if state.size is not None:
            objects[name] = _replace(objects[name], size=state.size)
    return objects


@dataclass(frozen=True)
class FailureRule:
    name: str
    if_: tuple[str, ...] = ()
    unless: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "if_", tuple(self.if_))
        object.__setattr__(self, "unless", tuple(self.unless))


@dataclass(frozen=True)
class Outcome:
    success: bool
    failure_mode: str | None
    furthest_stage: str | None


class StageTracker:
    def __init__(self, stages: _Sequence[str], hold_until: _Mapping[str, str] = {}):
        self._stages, self._hold = tuple(stages), dict(hold_until)
        if any(s not in self._stages or h not in self._stages for s, h in self._hold.items()):
            raise ValueError("hold_until must name known stages")
        self._ever = dict.fromkeys(self._stages, False)
        self._current = dict(self._ever)
        self._frame = 0
        self.stage_index = self.furthest_index = -1
        self.regressions: list[tuple[int, str]] = []

    def update(self, checks):
        from core.generate.protocols import StageProgress

        for stage in self._stages:
            value = bool(checks.get(stage, False))
            self._ever[stage] |= value
            self._current[stage] |= value
        regressed = None
        for i, stage in enumerate(self._stages):
            hold = self._hold.get(stage)
            if hold is not None and self._current[stage] and not checks.get(stage, False) and not self._current[hold]:
                for later in self._stages[i:]:
                    self._current[later] = False
                self.regressions.append((self._frame, stage))
                regressed = stage
                break
        self.stage_index = self._prefix(self._current)
        self.furthest_index = self._prefix(self._ever)
        result = StageProgress(self._frame, self.stage_index, self.furthest_index,
                               {key: bool(value) for key, value in checks.items()}, regressed)
        self._frame += 1
        return result

    def _prefix(self, flags):
        index = -1
        for i, stage in enumerate(self._stages):
            if not flags[stage]:
                break
            index = i
        return index

    def rewind_from(self, stage: str) -> None:
        index = self._stages.index(stage)
        if index <= self.stage_index:
            for later in self._stages[index:]:
                self._current[later] = False
            self.stage_index = index - 1

    def flags(self) -> dict[str, bool]:
        return dict(self._ever)

    @property
    def furthest_stage(self) -> str | None:
        return self._stages[self.furthest_index] if self.furthest_index >= 0 else None


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

    def sample_start(self, rng: np.random.Generator) -> dict[str, Pose]:
        from core.tasks.sampling import _uses_workspace, validate_workspace

        workspace = None
        if _uses_workspace(self.spec.raw.get("start", {})):
            from core.arms.spec import load_arm_spec
            workspace = validate_workspace(load_arm_spec()[0].get("workspace"))
        return {name: state.pose for name, state in self.sample_episode_start(rng, workspace=workspace).objects.items()
                if state.active}

    def sample_episode_start(self, rng, *, reach_fn=None, home_q=None, joint_limits=None,
                             instruction_rng=None, joint_names=None, arm_joints=None, workspace=None, tcp_fn=None):
        from core.tasks.sampling import sample_start
        return sample_start(self.spec, rng, reach_fn=reach_fn, home_q=home_q, joint_limits=joint_limits,
                            instruction_rng=instruction_rng, joint_names=joint_names, arm_joints=arm_joints,
                            workspace=workspace, tcp_fn=tcp_fn)

    def observe_checks(self, state: StepState, ctx: CheckContext) -> dict[str, bool]:
        return self.observe(state)

    def make_tracker(self) -> StageTracker:
        return StageTracker(self.stages, self.hold_until)

    @property
    def hold_until(self) -> dict[str, str]:
        return dict(self.spec.raw.get("stages", {}).get("hold_until", {}))

    @property
    def failure_rules(self) -> tuple[FailureRule, ...]:
        return _failure_rules(self.spec.raw)

    @property
    def failure_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.failure_rules)

    def failure_mode(self, flags: _Mapping[str, bool], holding: bool, success: bool) -> str | None:
        if success:
            return None

        def value(name):
            return bool(holding if name == "holding" else flags.get(name, False))

        for rule in self.failure_rules:
            if all(value(name) for name in rule.if_) and not any(value(name) for name in rule.unless):
                return rule.name
        return None

    def outcome(self, tracker: StageTracker, final_checks: _Mapping[str, bool], final_state: StepState) -> Outcome:
        flags = tracker.flags()
        flags.update({key: bool(value) for key, value in final_checks.items() if key not in self.stages})
        holding = bool(final_checks.get("holding", final_state.signals.get("holding", False)))
        success = self.is_success(flags)
        if self.spec.raw.get("stages", {}).get("success_requires_final", True):
            success = success and bool(final_checks.get(self.stages[-1], False))
        return Outcome(success, self.failure_mode(flags, holding, success), tracker.furthest_stage)

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
        required = self.spec.raw.get("stages", {}).get("success_requires_stages", ())
        return bool(flags.get(self.stages[-1], False)) and all(flags.get(stage, False) for stage in required)

    def furthest_stage(self, flags: dict[str, bool]) -> str | None:
        reached = None
        for stage in self.stages:
            if not flags.get(stage, False):
                break
            reached = stage
        return reached
