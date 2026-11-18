"""Validate task expert programs before an episode starts."""

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from itertools import product as _product
from types import MappingProxyType as _MappingProxyType
import math as _math

from core.expert.skills import SKILLS as _SKILLS
from core.generate.errors import ConfigError as _ConfigError
from core.generate.specs import _variation
from core.tasks.base import _freeze_value


@_dataclass(frozen=True)
class ProgramStep:
    skill: str
    name: str
    stage: str | None
    budget: int
    params: _Mapping

    def __post_init__(self):
        object.__setattr__(self, "params", _freeze_value(self.params))


@_dataclass(frozen=True)
class ExpertProgram:
    steps: tuple[ProgramStep, ...]
    max_steps: int
    settle_steps: int
    max_retries: int
    objects: _Mapping
    approach_axis: tuple[float, ...]
    closing_axis: tuple[float, ...]
    speed_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "objects", _MappingProxyType(dict(self.objects)))
        object.__setattr__(self, "approach_axis", tuple(self.approach_axis))
        object.__setattr__(self, "closing_axis", tuple(self.closing_axis))


def _require(condition, message):
    if not condition:
        raise _ConfigError(message)


def _number(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _vector(value, lengths):
    return isinstance(value, (tuple, list)) and len(value) in lengths and all(_number(v) for v in value)


def _goal_ref(value, task):
    _require(isinstance(value, str) and value.startswith("@goal.") and value.split(".")[1]
             in task.spec.raw.get("start", {}).get("goal", {}), f"Unknown goal reference: {value}")


def _ref(value, task, *, goal=True, region=False, points=()):
    if isinstance(value, (list, tuple)):
        _require(_vector(value, points), "Invalid target point")
        return
    if isinstance(value, _Mapping):
        _require(region and value.keys() == {"xy_min", "xy_max"}
                 and _vector(value["xy_min"], (2,)) and _vector(value["xy_max"], (2,))
                 and all(a <= b for a, b in zip(value["xy_min"], value["xy_max"])), "Invalid target region")
        return
    _require(isinstance(value, str), "Invalid reference")
    if value.startswith("@goal."):
        _require(goal, "This skill requires an object target")
        _goal_ref(value, task)
    elif value.startswith("@"):
        parts = value[1:].split(".")
        _require(parts[0] in task.spec.roles, f"Unknown role: {value}")
        obj = task.spec.roles[parts[0]]
        obj = task.spec.object_sets[obj].template if obj in task.spec.object_sets else task.spec.objects[obj]
        if len(parts) > 1:
            _require(region and parts[1:] == ["inner"] and "inner_half_size" in obj.params, "Invalid container reference")
    else:
        _require(value in task.spec.objects, f"Unknown object: {value}")


def _options(value, *, integer=False):
    variation = _variation(value, integer=integer)
    if variation is not None:
        form, values = variation
        if form == "choice":
            return values[0], values
        lo, hi = values
        return (lo if integer else lo / 2 + hi / 2), (lo, hi)
    if isinstance(value, (list, tuple)):
        items = [_options(item, integer=integer) for item in value]
        return [default for default, _ in items], _product(*(options for _, options in items))
    return value, (value,)


def _validated_params(params, kind, *, task, arm_spec, joint_limits, names, arm_joints, gaps):
    params = dict(params)
    if "target" in params:
        _ref(params["target"], task, goal=kind not in ("grasp", "push"), region=kind in ("transport", "place"),
             points=(3,) if kind in ("reach", "approach", "touch") else ())
    if "goal" in params:
        _ref(params["goal"], task, region=True, points=(2, 3))
    for key, value in params.items():
        if key == "offset_m":
            _require(_vector(value, (3,)), "offset_m must be a finite 3-vector")
        elif key in ("tol_m", "tol_rad", "contact_tol_m", "height_m", "clearance_m", "standoff_m", "z_offset_m", "depth_m", "dz_m"):
            _require(_number(value), f"Invalid {key}")
            if key in ("tol_m", "tol_rad", "contact_tol_m", "height_m"):
                _require(value > 0, f"{key} must be positive")
            if key in ("clearance_m", "standoff_m"):
                _require(value >= 0, f"{key} must be nonnegative")
        elif key in ("dwell_steps", "steps"):
            _require(type(value) is int and value >= (1 if key == "steps" else 0), f"Invalid {key}")
        elif key == "orientation":
            _require(value in ("down", "keep"), "Invalid orientation")
        elif key == "yaw":
            _require(value in ("object", "keep") or _number(value), "Invalid yaw")
        elif key in ("gap_m", "close_gap_m"):
            if isinstance(value, str) and value.startswith("@"):
                _goal_ref(value, task)
            elif key == "gap_m" and value in ("open", "closed"):
                pass
            else:
                _require(_number(value) and min(gaps) <= value <= max(gaps), "Gap is outside the opening table")
    if "q" in params:
        q = params["q"]
        if isinstance(q, str) and q == "home":
            home = {**arm_spec["initial_state"]["joint_positions"], **arm_spec["expert"].get("home_rad", {})}
            q = {joint: float(home[joint]) for joint in arm_joints}
        if isinstance(q, str):
            _goal_ref(q, task)
        else:
            _require(isinstance(q, _Mapping) and bool(q) and not q.keys() - set(names) and all(_number(v) for v in q.values()), "Invalid joint target")
            q = {key: float(value) for key, value in q.items()}
            if joint_limits is not None:
                _require(all(joint_limits[0][names.index(j)] <= v <= joint_limits[1][names.index(j)] for j, v in q.items()), "Joint target outside limits")
        params["q"] = q
    if kind == "sweep":
        joint, angles = params["joint"], params["angles"]
        if isinstance(joint, str) and joint.startswith("@"):
            _goal_ref(joint, task)
        else:
            _require(joint in arm_joints, "Sweep must command an arm joint")
        if isinstance(angles, str):
            _goal_ref(angles, task)
        else:
            _require(isinstance(angles, (list, tuple)) and bool(angles) and all(_number(v) for v in angles), "Invalid sweep angles")
            if joint_limits is not None and joint in arm_joints:
                i = names.index(joint)
                _require(all(joint_limits[0][i] <= v <= joint_limits[1][i] for v in angles), "Sweep outside limits")
    return params


def parse_program(table, *, task, arm_spec, joint_limits=None) -> ExpertProgram:
    try:
        _require(isinstance(table, _Mapping) and not table.keys() - {"steps", "max_steps", "settle_steps", "max_retries", "speed_scale"}, "Invalid expert table")
        raw_steps = table.get("steps")
        _require(isinstance(raw_steps, list) and bool(raw_steps), "Expert steps must be a non-empty list")
        counts = [table.get("max_steps", 600), table.get("settle_steps", 10), table.get("max_retries", 2)]
        _require(all(type(v) is int and v >= minimum for v, minimum in zip(counts, (1, 0, 0))), "Invalid expert budgets")
        names = arm_spec["joints"]["names"]
        arm_joints = arm_spec["expert"]["arm_joints"]
        approach, closing = arm_spec["kinematics"]["approach_axis"], arm_spec["kinematics"]["closing_axis"]
        _require(_vector(approach, (3,)) and _vector(closing, (3,)), "Missing tool axes")
        gaps = [row[1] for row in arm_spec["gripper"]["opening_table"]]
        speed_scale, speed_options = _options(table.get("speed_scale", 1.0))
        _require(_number(speed_scale) and all(_number(v) and 0 < v <= 1 for v in speed_options), "Invalid speed_scale")
        steps = []
        for raw in raw_steps:
            _require(isinstance(raw, _Mapping) and isinstance(raw.get("skill"), str) and raw["skill"] in _SKILLS, "Missing or unknown skill")
            skill = _SKILLS[raw["skill"]]
            _require(not raw.keys() - ({"skill", "name", "stage", "budget"} | set(skill.required) | skill.defaults.keys()), "Unknown skill parameter")
            _require(all(key in raw for key in skill.required), "Missing required skill parameter")
            name, stage = raw.get("name", skill.name), raw.get("stage")
            budget = raw.get("budget", 1 if skill.budget is None else skill.budget)
            _require(type(budget) is int and budget >= 1 and isinstance(name, str) and bool(name), "Invalid step budget or name")
            _require(stage is None or stage in task.stages, "Unknown stage")
            params = {**skill.defaults, **{key: value for key, value in raw.items() if key not in ("skill", "name", "stage", "budget")}}
            if skill.name == "grasp" and "close_gap_m" not in raw:
                # Some jaws retain a measured gap at their closed limit.
                params["close_gap_m"] = max(min(gaps), params["close_gap_m"])
            options = {key: _options(value, integer=key in ("steps", "dwell_steps")) for key, value in params.items()}
            checks = dict(task=task, arm_spec=arm_spec, joint_limits=joint_limits, names=names, arm_joints=arm_joints, gaps=gaps)
            nominal = _validated_params({key: default for key, (default, _) in options.items()}, skill.name, **checks)
            for values in _product(*(variants for _, variants in options.values())):
                _validated_params(dict(zip(options, values)), skill.name, **checks)
            if skill.budget is None and "budget" not in raw:
                _, durations = _options(params["steps"], integer=True)
                budget = max(durations)
            steps.append(ProgramStep(skill.name, name, stage, budget, nominal))
        return ExpertProgram(tuple(steps), *counts, task.spec.objects, tuple(approach), tuple(closing), float(speed_scale))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        if isinstance(exc, _ConfigError):
            raise
        raise _ConfigError(f"Invalid expert program: {exc}") from exc
