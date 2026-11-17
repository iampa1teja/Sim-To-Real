"""Validated perturbation draws and one-shot world disturbances."""

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from types import MappingProxyType as _MappingProxyType
import math as _math

import numpy as _np

from core.generate.errors import ConfigError as _ConfigError, EnvError as _EnvError, SamplingError as _SamplingError
from core.generate.protocols import Kind as _Kind, PerturbationGroup as _Group, PerturbationSpec as _PerturbationSpec
from core.tasks.base import Pose as _Pose


@_dataclass(frozen=True)
class PerturbationType:
    name: str
    group: _Group
    kinds: frozenset[_Kind]
    params: _Mapping

    def __post_init__(self):
        object.__setattr__(self, "kinds", frozenset(self.kinds))
        object.__setattr__(self, "params", _MappingProxyType({
            key: tuple(value) if isinstance(value, (list, tuple)) else value for key, value in self.params.items()}))

    def allowed_stages(self, task):
        stages = tuple(task.stages)
        steps = task.spec.raw.get("expert", {}).get("steps", [])
        if self.name == "wrong_target":
            return ()
        if self.name == "target_offset":
            tagged = {step.get("stage") for step in steps if any(key in step for key in ("target", "q", "gap_m"))}
            return tuple(stage for stage in stages if stage in tagged)
        if self.name in ("object_shift", "container_shift"):
            skill = "grasp" if self.name == "object_shift" else "release"
            for step in steps:
                if step.get("skill") == skill and step.get("stage") is not None:
                    return stages[:stages.index(step["stage"])]
        if self.name == "object_drop":
            checks = task.spec.raw.get("checks", {})

            def implies(name, visiting):
                check = checks.get(name, {})
                if name in visiting or check.get("negate", False) or check.get("check") == "any":
                    return False
                if name == "holding" or check.get("check") == "held":
                    return True
                return check.get("check") == "all" and any(
                    implies(child, visiting | {name}) for child in check.get("of", ()))

            return tuple(stage for stage in stages if implies(stage, set()))
        return stages


_FR = frozenset((_Kind.FAILURE, _Kind.RECOVERY))
PERTURBATION_TYPES = _MappingProxyType({name: PerturbationType(name, group, kinds, params)
    for name, group, kinds, params in (
        ("grasp_offset", _Group.EXPERT, _FR, {"xy_m": None, "z_m": (0., 0.), "yaw_rad": (0., 0.)}),
        ("early_close", _Group.EXPERT, _FR, {"delay_steps": None}),
        ("partial_close", _Group.EXPERT, _FR, {"extra_gap_m": (0.004, 0.012)}),
        ("early_release", _Group.EXPERT, _FR, {"delay_steps": None}),
        ("place_offset", _Group.EXPERT, _FR, {"xy_m": None}),
        ("wrong_target", _Group.EXPERT, _FR, {}),
        ("target_offset", _Group.EXPERT, _FR, {"xy_m": (0., 0.), "z_m": (0., 0.), "joint_rad": (0., 0.), "gap_m": (0., 0.)}),
        ("stall", _Group.EXPERT, {_Kind.FAILURE}, {"delay_steps": (0, 0), "steps": None}),
        ("object_shift", _Group.WORLD, _FR, {"object": "@target", "delay_steps": (0, 0), "xy_m": None, "yaw_rad": (0., 0.)}),
        ("object_drop", _Group.WORLD, _FR, {"delay_steps": (0, 0), "xy_jitter_m": (0., 0.)}),
        ("container_shift", _Group.WORLD, _FR, {"object": "@container", "delay_steps": (0, 0), "xy_m": None}),
    )})


def _require(condition, message):
    if not condition:
        raise _ConfigError(message)


def _number(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _validate(task, override):
    table = task.spec.raw.get("perturbations", {})
    _require(isinstance(table, _Mapping) and not table.keys() - {"failure", "recovery"}, "Invalid perturbation table")
    _require(isinstance(override, _Mapping), "Weights override must be a table")
    for name, weight in override.items():
        _require(name in PERTURBATION_TYPES and _number(weight) and weight >= 0, "Invalid perturbation weight override")
    skills = {step.get("skill") for step in task.spec.raw.get("expert", {}).get("steps", [])}
    needs = {"grasp_offset": "grasp", "partial_close": "grasp", "early_close": "approach",
             "early_release": "transport", "place_offset": "place"}
    for kind, entries in table.items():
        _require(isinstance(entries, list), "Perturbations must be lists")
        for entry in entries:
            _require(isinstance(entry, _Mapping) and isinstance(entry.get("type"), str)
                     and entry["type"] in PERTURBATION_TYPES, "Missing or unknown perturbation type")
            ptype = PERTURBATION_TYPES[entry["type"]]
            name = ptype.name
            _require(_Kind(kind) in ptype.kinds, f"{name} does not support {kind}")
            weight = entry.get("weight", 1.)
            _require(_number(weight) and weight >= 0, "Invalid perturbation weight")
            _require(not entry.keys() - (ptype.params.keys() | {"type", "weight", "stages"}), "Unknown perturbation parameter")
            allowed = ptype.allowed_stages(task)
            if "stages" in entry:
                stages = entry["stages"]
                _require(isinstance(stages, list) and bool(stages) and all(isinstance(s, str) and s in allowed for s in stages), "Invalid perturbation stages")
            if name in ("stall", "object_shift", "object_drop", "container_shift"):
                _require("stages" in entry, f"{name} requires stages")
            if name in needs:
                _require(needs[name] in skills, f"{name} needs a {needs[name]} step")
            if name == "target_offset":
                _require(bool(allowed), "target_offset needs a tagged target step")
            if name == "object_drop":
                _require("target" in task.spec.roles, "object_drop needs a target role")
            if name == "wrong_target":
                target = task.spec.roles.get("target")
                _require(target in task.spec.object_sets, "wrong_target needs an object set target")
                count = task.spec.raw.get("start", {}).get("count", (0, task.spec.object_sets[target].count_max))
                _require(count[1] >= 2, "wrong_target needs at least two possible objects")
            for key, default in ptype.params.items():
                value = entry.get(key, default)
                _require(value is not None, f"Missing required parameter {key}")
                if key == "object":
                    _require(isinstance(value, str) and (value[1:] in task.spec.roles if value.startswith("@")
                                                        else value in task.spec.objects), "Unknown object reference")
                    continue
                integer = key in ("delay_steps", "steps")
                _require(isinstance(value, (tuple, list)) and len(value) == 2
                         and all(type(v) is int if integer else _number(v) for v in value)
                         and value[0] <= value[1], f"Invalid range for {key}")
                if key in ("xy_m", "xy_jitter_m", "extra_gap_m", "delay_steps", "steps"):
                    _require(value[0] >= (1 if key == "steps" else 0), f"Invalid lower bound for {key}")
    return table


def draw_perturbation(task, kind, rng, weights_override=None, *, start=None):
    override = {} if weights_override is None else weights_override
    table = _validate(task, override)
    kind = _Kind(kind)
    if kind is _Kind.SUCCESS:
        return None
    entries = table.get(kind.value, [])
    _require(bool(entries), f"No {kind.value} perturbations")
    weights = _np.array([override.get(e["type"], e.get("weight", 1.)) for e in entries], dtype=float)
    _require(float(weights.sum()) > 0, "Perturbation weights sum to zero")
    if start is not None:
        target_set = task.spec.roles.get("target")
        count = sum(o.active and task.spec.objects[name].params.get("set") == target_set
                    for name, o in start.objects.items() if name in task.spec.objects)
        if count < 2:
            for i, entry in enumerate(entries):
                if entry["type"] == "wrong_target":
                    weights[i] = 0.
    if not weights.sum() > 0:
        raise _SamplingError("No applicable perturbations for this start")
    entry = entries[int(rng.choice(len(weights), p=weights / weights.sum()))]
    ptype = PERTURBATION_TYPES[entry["type"]]
    onset = str(rng.choice(entry["stages"])) if "stages" in entry else None
    params, delay = {}, 0
    for key, default in ptype.params.items():
        value = entry.get(key, default)
        if isinstance(value, str):
            drawn = value
        elif key in ("delay_steps", "steps"):
            drawn = int(rng.integers(value[0], value[1] + 1))
        else:
            drawn = float(rng.uniform(*value))
        if key == "delay_steps":
            delay = drawn
        else:
            params[key] = drawn
        if key == "xy_m":
            params["dir_rad"] = float(rng.uniform(0., 2 * _math.pi))
    return _PerturbationSpec(ptype.name, ptype.group, onset, delay, params)


def _yaw(quat):
    x, y, z, w = quat
    return _math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def _dims(obj):
    return obj.size if obj.size is not None else obj.params.get("size_outer", (0., 0., 0.))


def _bottom(obj):
    return obj.size is None and obj.params.get("origin") == "bottom"


class WorldPerturber:
    def __init__(self, spec, task_spec, rng):
        self._spec, self._task, self._rng = spec, task_spec, rng
        self.applied = False
        self.onset_frame = None
        self._latch = self._due_t = None
        self._disabled = self._checked_drop = False

    def due(self, progress, t):
        p = self._spec.perturbation
        if p is None or p.group is not _Group.WORLD or self.applied or self._disabled:
            return False
        stages = self._task.raw["stages"]["order"]
        index = stages.index(p.onset_stage) if p.onset_stage is not None else -1
        if self._latch is None and progress.furthest_index >= index:
            self._latch = t
        if self._latch is None or t < self._latch + p.delay_steps:
            return False
        if p.type == "object_drop" and not self._checked_drop:
            self._checked_drop = True
            if not progress.satisfied.get("holding", False):
                self._disabled = True
                return False
        self._due_t = t
        return True

    def apply(self, env, obs):
        if self._due_t is None or self.applied:
            raise RuntimeError("World perturbation is not due")
        p = self._spec.perturbation
        ref = "@target" if p.type == "object_drop" else p.params["object"]
        name = self._spec.roles[ref[1:]] if ref.startswith("@") else ref
        if name not in obs.objects or not obs.objects[name].active:
            raise _EnvError(f"Missing or inactive perturbation object: {name}")
        old = obs.objects[name].pose
        x, y, z = old.position
        if p.type == "object_drop":
            direction = float(self._rng.uniform(0., 2 * _math.pi))
            radius = p.params["xy_jitter_m"]
            x, y = x + radius * _math.cos(direction), y + radius * _math.sin(direction)
            support = self._task.raw.get("start", {}).get("table_z_m", 0.)
            for other, state in obs.objects.items():
                if other == name or not state.active or other not in self._task.objects:
                    continue
                obj = self._task.objects[other]
                ox, oy, oz = state.pose.position
                yaw = _yaw(state.pose.quat_xyzw)
                c, s = _math.cos(yaw), _math.sin(yaw)
                local = (c*(x-ox) + s*(y-oy), -s*(x-ox) + c*(y-oy))
                inner = obj.params.get("inner_half_size")
                if inner is not None and all(abs(v) <= h for v, h in zip(local, inner)):
                    support = max(support, oz + obj.params.get("floor_z", 0.))
                else:
                    dims = _dims(obj)
                    if all(abs(v) <= h/2 for v, h in zip(local, dims[:2])):
                        support = max(support, oz + dims[2] * (1 if _bottom(obj) else 0.5))
            obj = self._task.objects[name]
            z = support + (0 if _bottom(obj) else _dims(obj)[2] / 2)
            yaw = _yaw(old.quat_xyzw)
            quat = (0., 0., _math.sin(yaw/2), _math.cos(yaw/2))
        else:
            radius, direction = p.params["xy_m"], p.params["dir_rad"]
            x, y = x + radius * _math.cos(direction), y + radius * _math.sin(direction)
            a, b, c, d = old.quat_xyzw
            angle = p.params.get("yaw_rad", 0.) if p.type == "object_shift" else 0.
            s, co = _math.sin(angle/2), _math.cos(angle/2)
            quat = (co*a - s*b, co*b + s*a, co*c + s*d, co*d - s*c)
        pose = _Pose((x, y, z), quat)
        env.set_object_pose(name, pose)
        self.applied = True
        self.onset_frame = self._due_t + 1
        return pose


def ou_noise(noise, arm_mask, steps, rng):
    mask = _np.asarray(arm_mask)
    if mask.ndim != 1 or mask.dtype != _np.bool_:
        raise TypeError("arm_mask must be a one-dimensional boolean mask")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    out = _np.zeros((steps, len(mask)), dtype=_np.float64)
    current = _np.zeros(int(mask.sum()))
    for t in range(steps - 1):
        current = _np.clip(current - noise.theta * current + noise.sigma_rad * rng.standard_normal(len(current)),
                           -noise.max_abs_rad, noise.max_abs_rad)
        out[t + 1, mask] = current
    return out
