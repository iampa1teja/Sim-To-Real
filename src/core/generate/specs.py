"""Construct episode specs using independent deterministic random streams."""

import math as _math
from collections.abc import Mapping as _Mapping

from core.generate.errors import ConfigError as _ConfigError
from core.generate.perturbations import draw_perturbation as _draw_perturbation
from core.generate.protocols import EpisodeSpec as _EpisodeSpec, Kind as _Kind, NoiseSpec as _NoiseSpec
from core.generate.seeds import rng_for as _rng_for


_STEP_KEYS = {"skill", "name", "stage", "budget"}
_PHYSICS_KEYS = ("mass_kg", "static_friction", "dynamic_friction")


def _number(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _variation(value, *, integer=False):
    if not isinstance(value, _Mapping) or not value.keys() & {"range", "choice"}:
        return None
    form = next(iter(value))
    values = value[form]
    if len(value) != 1 or not isinstance(values, (tuple, list)) or not values:
        raise _ConfigError("Expected a range or a non-empty choice")
    if form == "range" and (len(values) != 2 or not all(_number(v) for v in values)
                            or values[0] > values[1] or integer and not all(type(v) is int for v in values)):
        raise _ConfigError("Invalid expert parameter range")
    return form, values


def _draw(value, rng, *, integer=False):
    variation = _variation(value, integer=integer)
    if variation is not None:
        form, values = variation
        if form == "choice":
            return values[rng.integers(len(values))], True
        lo, hi = values
        return (int(rng.integers(lo, hi + 1)) if integer else float(rng.uniform(lo, hi))), True
    if isinstance(value, (list, tuple)):
        items = [_draw(item, rng, integer=integer) for item in value]
        if any(varied for _, varied in items):
            return [item for item, _ in items], True
    return value, False


def draw_expert_params(expert_table, rng) -> dict:
    result = {}
    speed, varied = _draw(expert_table.get("speed_scale", 1.0), rng)
    if varied:
        result["speed_scale"] = float(speed)
    steps = {}
    for index, step in enumerate(expert_table.get("steps", ())):
        params = {}
        for key in sorted(step.keys() - _STEP_KEYS):
            value, varied = _draw(step[key], rng, integer=key in ("dwell_steps", "steps"))
            if varied:
                params[key] = value
        if params:
            steps[str(index)] = params
    if steps:
        result["steps"] = steps
    return result


def validate_physics(task_spec):
    table = task_spec.raw.get("physics", {})
    if not isinstance(table, _Mapping) or table.keys() - (task_spec.objects.keys() | task_spec.object_sets.keys()):
        raise _ConfigError("Physics must name task objects or object sets")
    for name, entry in table.items():
        if not isinstance(entry, _Mapping) or not entry or entry.keys() - set(_PHYSICS_KEYS):
            raise _ConfigError(f"Invalid physics entry for {name}")
        for key, bounds in entry.items():
            if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2 or not all(_number(v) for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1] or key == "mass_kg" and bounds[0] == 0):
                raise _ConfigError(f"Invalid physics range for {name}.{key}")


def draw_physics(task_spec, objects, rng) -> dict:
    table = task_spec.raw.get("physics", {})
    physics = {}
    for name, obj in task_spec.objects.items():
        if name not in objects or not objects[name].active:
            continue
        entry = {**table.get(obj.params.get("set"), {}), **table.get(name, {})}
        if entry:
            values = {key: float(rng.uniform(*entry[key])) for key in _PHYSICS_KEYS if key in entry}
            if "dynamic_friction" in values and "static_friction" in values:
                values["dynamic_friction"] = min(values["dynamic_friction"], values["static_friction"])
            physics[name] = values
    return physics


def build_spec(task, kind, attempt_index, seed, *, arm, scene, home_q, reach_fn=None, noise=None,
               weights_override=None, fps=30, joint_names=None, joint_limits=None, arm_joints=None,
               workspace=None, tcp_fn=None):
    kind = _Kind(kind)
    start = task.sample_episode_start(_rng_for(seed, "start"), reach_fn=reach_fn, home_q=home_q,
                                      joint_limits=joint_limits, instruction_rng=_rng_for(seed, "instruction"),
                                      joint_names=joint_names, arm_joints=arm_joints, workspace=workspace, tcp_fn=tcp_fn)
    perturbation = _draw_perturbation(task, kind, _rng_for(seed, "perturbation"), weights_override, start=start)
    drawn_noise = None
    if kind is _Kind.SUCCESS and noise is not None and _rng_for(seed, "noise").random() < noise.fraction:
        drawn_noise = _NoiseSpec(noise.theta, noise.sigma_rad, noise.max_abs_rad)
    expert = task.spec.raw.get("expert", {})
    return _EpisodeSpec(task=task.name, arm=arm, scene=scene, kind=kind, attempt_index=attempt_index, seed=seed,
                        objects=start.objects, roles=start.roles, goal=start.goal, instruction=start.instruction,
                        initial_q=tuple(home_q) if start.initial_q is None else start.initial_q,
                        held_object=start.held_object, perturbation=perturbation, noise=drawn_noise,
                        max_steps=expert.get("max_steps", 600), fps=fps,
                        expert_params=draw_expert_params(expert, _rng_for(seed, "expert_params")),
                        physics=draw_physics(task.spec, start.objects, _rng_for(seed, "physics")))
