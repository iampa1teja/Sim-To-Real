"""TOML-driven task checks, evaluated in the robot base frame."""

from __future__ import annotations

import math as _math
from collections.abc import Mapping as _Mapping
from types import MappingProxyType as _MappingProxyType
from typing import Callable as _Callable

import numpy as _np

from core.generate.errors import ConfigError as _ConfigError
from core.tasks.base import CheckContext as _CheckContext, StepState as _StepState, Task as _Task

PREDICATES: dict[str, _Callable[[_StepState, _CheckContext, _Mapping], bool]] = {}


def register_predicate(name):
    """Register a check callable without changing the callable."""
    def register(function):
        PREDICATES[name] = function
        return function
    return register


def resolve_ref(ref, ctx):
    """Resolve an object, container marker or nested episode goal reference."""
    if isinstance(ref, (list, tuple)):
        return tuple(ref)
    if not isinstance(ref, str):
        raise TypeError("A reference must be a string or sequence")
    try:
        if ref.startswith("@goal."):
            parts = ref.split(".")[1:]
            value = ctx.goal[parts[0]]
            for part in parts[1:]:
                if isinstance(value, _Mapping):
                    value = value[part]
                elif isinstance(value, (list, tuple)) and part.isdecimal():
                    value = value[int(part)]
                else:
                    raise KeyError(part)
            return value
        if ref.startswith("@"):
            parts = ref[1:].split(".")
            name = ctx.roles[parts[0]]
            if name not in ctx.objects:
                raise KeyError(name)
            if len(parts) == 1:
                return name
            if parts[1:] == ["inner"] and "inner_half_size" in ctx.objects[name].params:
                return _MappingProxyType({"inner": name})
        elif ref in ctx.objects:
            return ref
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise _ConfigError(f"Unresolvable reference: {ref}") from exc
    raise _ConfigError(f"Unresolvable reference: {ref}")


def _dims(spec):
    return _np.asarray(spec.size if spec.size is not None else spec.params.get("size_outer", (0, 0, 0)),
                       dtype=_np.float64)


def _bottom_origin(spec):
    return spec.size is None and spec.params.get("origin", "center") == "bottom"


def _yaw(pose):
    x, y, z, w = pose.quat_xyzw
    return _math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def _position(state, ctx, ref):
    name = resolve_ref(ref, ctx)
    if name not in state.objects or name not in ctx.active:
        return name, None
    return name, _np.asarray(state.objects[name].position, dtype=_np.float64)


def _tcp(state):
    return _np.asarray((state.tcp or state.ee).position, dtype=_np.float64)


def _value(value, ctx):
    return resolve_ref(value, ctx) if isinstance(value, str) and value.startswith("@") else value


@register_predicate("ee_near")
def _ee_near(state, ctx, params):
    if ("object" in params) == ("point" in params):
        raise _ConfigError("ee_near requires exactly one of object and point")
    if not state.signals.get(params.get("contact_signal"), True):
        return False
    if "object" in params:
        _, target = _position(state, ctx, params["object"])
        if target is None:
            return False
    else:
        target = _np.asarray(resolve_ref(params["point"], ctx), dtype=_np.float64)
    target = target + _np.asarray(params.get("offset_m", (0, 0, 0)))
    return bool(_np.linalg.norm(_tcp(state) - target) <= params.get("tol_m", 0.02))


@register_predicate("object_lifted")
def _object_lifted(state, ctx, params):
    name, position = _position(state, ctx, params["object"])
    return bool(position is not None and position[2] - ctx.start[name].position[2] >= params.get("min_lift_m", 0.01))


@register_predicate("object_in_region")
def _object_in_region(state, ctx, params):
    name, position = _position(state, ctx, params["object"])
    if position is None:
        return False
    region = params["region"]
    if isinstance(region, str):
        region = resolve_ref(region, ctx)
    margin = params.get("margin_m", 0)
    if "inner" in region:
        container = region["inner"]
        if container not in state.objects or container not in ctx.active:
            return False
        pose, spec = state.objects[container], ctx.objects[container]
        xy = position[:2] - _np.asarray(pose.position[:2])
        yaw = _yaw(pose)
        c, s = _math.cos(yaw), _math.sin(yaw)
        local = _np.array([c*xy[0] + s*xy[1], -s*xy[0] + c*xy[1]])
        inside = bool(_np.all(_np.abs(local) <= _np.asarray(spec.params["inner_half_size"]) - margin))
        if inside and params.get("check_z", True):
            bottom = position[2] - (0 if _bottom_origin(ctx.objects[name]) else _dims(ctx.objects[name])[2]/2)
            inside = bool(bottom - (pose.position[2] + spec.params.get("floor_z", 0)) <= params.get("z_tol_m", 0.01))
        return inside
    if "xy_min" in region:
        return bool(_np.all(position[:2] >= _np.asarray(region["xy_min"]) + margin)
                    and _np.all(position[:2] <= _np.asarray(region["xy_max"]) - margin))
    radius = _math.hypot(*position[:2])
    bearing = _math.degrees(_math.atan2(position[1], position[0]))
    return bool(region["r_m"][0] + margin <= radius <= region["r_m"][1] - margin
                and region["bearing_deg"][0] <= bearing <= region["bearing_deg"][1])


@register_predicate("objects_in_region")
def _objects_in_region(state, ctx, params):
    excluded = {resolve_ref(ref, ctx) for ref in params.get("exclude", ())}
    count = sum(_object_in_region(state, ctx, {**params, "object": name})
                for name, spec in ctx.objects.items()
                if spec.params.get("set") == params["set"] and name in ctx.active and name not in excluded)
    return bool(count >= params.get("min_count", 1))


@register_predicate("object_on_object")
def _object_on_object(state, ctx, params):
    name, position = _position(state, ctx, params["object"])
    base, support = _position(state, ctx, params["base"])
    if position is None or support is None:
        return False
    top = _dims(ctx.objects[base])[2] * (1 if _bottom_origin(ctx.objects[base]) else 0.5)
    offset = 0 if _bottom_origin(ctx.objects[name]) else _dims(ctx.objects[name])[2]/2
    return bool(_np.linalg.norm(position[:2] - support[:2]) <= params.get("xy_tol_m", 0.01)
                and abs(position[2] - (support[2] + top + offset)) <= params.get("z_tol_m", 0.005))


@register_predicate("nested_in")
def _nested_in(state, ctx, params):
    _, position = _position(state, ctx, params["object"])
    _, base = _position(state, ctx, params["base"])
    return bool(position is not None and base is not None
                and _np.linalg.norm(position[:2] - base[:2]) <= params.get("xy_tol_m", 0.01)
                and 0 <= position[2] - base[2] <= params["max_rise_m"])


@register_predicate("held")
def _held(state, ctx, params):
    name, position = _position(state, ctx, params["object"])
    if position is None:
        return False
    contact_signal = params.get("contact_signal")
    if contact_signal in state.signals:
        return state.signals[contact_signal]
    width = min(_dims(ctx.objects[name])[:2])
    return bool(width - params.get("gap_low_m", 0.006) <= state.gripper <= width + params.get("gap_high_m", 0.004)
                and _np.linalg.norm(_tcp(state) - position) <= params.get("max_dist_m", 0.03))


@register_predicate("joint_near")
def _joint_near(state, ctx, params):
    joint = _value(params["joint"], ctx)
    if isinstance(joint, _Mapping):
        joint = joint["joint"]
    return bool(state.joints is not None and joint in state.joints
                and abs(state.joints[joint] - _value(params["value_rad"], ctx)) <= params.get("tol_rad", 0.02))


@register_predicate("gap_near")
def _gap_near(state, ctx, params):
    return bool(abs(state.gripper - _value(params["value_m"], ctx)) <= params.get("tol_m", 0.003))


@register_predicate("object_displaced")
def _object_displaced(state, ctx, params):
    name, position = _position(state, ctx, params["object"])
    if position is None:
        return False
    axes = params.get("axes", "xy")
    if not isinstance(axes, str) or set(axes) - set("xyz"):
        raise _ConfigError("object_displaced.axes must be a subset of xyz")
    delta = position - _np.asarray(ctx.start[name].position)
    distance = _np.linalg.norm(delta[["xyz".index(axis) for axis in axes]])
    return bool(params["min_m"] <= distance <= params.get("max_m", _math.inf))


@register_predicate("signal")
def _signal(state, ctx, params):
    return bool(state.signals.get(params["name"], False))


@register_predicate("all")
def _all(state, ctx, params):
    return all(params["of"])


@register_predicate("any")
def _any(state, ctx, params):
    return any(params["of"])


_REQUIRED = {
    "object_lifted": ("object",), "held": ("object",), "object_displaced": ("object", "min_m"),
    "object_in_region": ("object", "region"), "objects_in_region": ("set", "region"),
    "object_on_object": ("object", "base"), "nested_in": ("object", "base", "max_rise_m"),
    "joint_near": ("joint", "value_rad"), "gap_near": ("value_m",), "signal": ("name",),
    "all": ("of",), "any": ("of",),
}


def _static_ref(ref, spec):
    if isinstance(ref, (tuple, list)):
        return
    if not isinstance(ref, str):
        raise _ConfigError(f"Invalid reference: {ref}")
    if ref.startswith("@goal."):
        if ref.split(".")[1] not in spec.raw.get("start", {}).get("goal", {}):
            raise _ConfigError(f"Unknown goal: {ref}")
    elif ref.startswith("@"):
        parts = ref[1:].split(".")
        if parts[0] not in spec.roles:
            raise _ConfigError(f"Unknown role: {ref}")
        target = spec.roles[parts[0]]
        obj = spec.object_sets[target].template if target in spec.object_sets else spec.objects[target]
        if len(parts) > 1 and (parts[1:] != ["inner"] or "inner_half_size" not in obj.params):
            raise _ConfigError(f"Invalid container reference: {ref}")
    elif ref not in spec.objects:
        raise _ConfigError(f"Unknown object: {ref}")


class GenericTask(_Task):
    """A task whose checks and failure rules are supplied by task.toml."""

    def __init__(self, spec):
        super().__init__(spec)
        try:
            stages = spec.raw.get("stages", {}).get("order")
            if not isinstance(stages, list) or not stages or not all(isinstance(s, str) for s in stages):
                raise _ConfigError("task.stages.order must be a non-empty list of names")
            self._stages = tuple(stages)
            if len(set(stages)) != len(stages):
                raise _ConfigError("Duplicate stages")
            required = spec.raw["stages"].get("success_requires_stages", [])
            if not isinstance(required, list) or any(stage not in stages for stage in required):
                raise _ConfigError("success_requires_stages must list known stages")
            self._checks = {key: dict(value) for key, value in spec.raw.get("checks", {}).items()}
            if set(stages) - self._checks.keys():
                raise _ConfigError("Every stage needs a check")
            for name, table in self._checks.items():
                kind = table.get("check")
                if kind not in PREDICATES:
                    raise _ConfigError(f"Unknown predicate: {kind}")
                if any(key not in table for key in _REQUIRED.get(kind, ())):
                    raise _ConfigError(f"Missing parameter for {name}")
                if kind == "ee_near" and (("object" in table) == ("point" in table)):
                    raise _ConfigError("ee_near requires exactly one target")
                if "contact_signal" in table and (
                        kind not in ("ee_near", "held") or not isinstance(table["contact_signal"], str)
                        or not table["contact_signal"]):
                    raise _ConfigError("contact_signal must name an ee_near or held signal")
                for key in ("object", "base", "point"):
                    if key in table:
                        _static_ref(table[key], spec)
                if isinstance(table.get("region"), str):
                    _static_ref(table["region"], spec)
                for ref in table.get("exclude", ()):
                    _static_ref(ref, spec)
                for key in ("joint", "value_rad", "value_m"):
                    if isinstance(table.get(key), str) and table[key].startswith("@"):
                        _static_ref(table[key], spec)
                if kind == "objects_in_region" and table["set"] not in spec.object_sets:
                    raise _ConfigError("Unknown object set")
                if kind == "object_displaced" and (not isinstance(table.get("axes", "xy"), str)
                                                     or set(table.get("axes", "xy")) - set("xyz")):
                    raise _ConfigError("Invalid displacement axes")
                if kind in ("all", "any") and (not isinstance(table["of"], list)
                                                or any(ref not in self._checks for ref in table["of"])):
                    raise _ConfigError("Unknown check dependency")
            for stage, hold in self.hold_until.items():
                if stage not in stages or hold not in stages or stages.index(hold) <= stages.index(stage):
                    raise _ConfigError("hold_until must name a later stage")
            known = set(self._checks) | set(stages) | {"holding"}
            for rule in self.failure_rules:
                if set(rule.if_ + rule.unless) - known:
                    raise _ConfigError("Unknown failure-rule flag")
            self._order = []
            visiting, visited = set(), set()
            def visit(name):
                if name in visiting:
                    raise _ConfigError("Check dependency cycle")
                if name in visited:
                    return
                visiting.add(name)
                table = self._checks[name]
                if table["check"] in ("all", "any"):
                    for child in table["of"]:
                        visit(child)
                visiting.remove(name)
                visited.add(name)
                self._order.append(name)
            for name in self._checks:
                visit(name)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            if isinstance(exc, _ConfigError):
                raise
            raise _ConfigError(f"Invalid task checks: {exc}") from exc

    @property
    def stages(self) -> tuple[str, ...]:
        return self._stages

    def observe(self, state):
        raise TypeError("GenericTask requires observe_checks(state, ctx)")

    def observe_checks(self, state, ctx) -> dict[str, bool]:
        result = {}
        for name in self._order:
            table = self._checks[name]
            params = {key: value for key, value in table.items() if key not in ("check", "negate")}
            if table["check"] in ("all", "any"):
                params["of"] = [result[child] for child in params["of"]]
            value = bool(PREDICATES[table["check"]](state, ctx, params))
            result[name] = not value if table.get("negate", False) else value
        return result
