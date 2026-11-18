"""Deterministic episode starts, with injected reachability checks."""

from __future__ import annotations

import math as _math
import re as _re
import string as _string
from collections import Counter as _Counter
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass, replace as _replace
from types import MappingProxyType as _MappingProxyType
from typing import Any as _Any

import numpy as _np

from core.generate.errors import ConfigError as _ConfigError, SamplingError as _SamplingError
from core.generate.protocols import ObjectState as _ObjectState
from core.tasks.base import Pose as _Pose, _freeze_value
from core.tasks.generic import _bottom_origin, _dims, _yaw


@_dataclass(frozen=True)
class StartSample:
    objects: _Mapping[str, _ObjectState]
    roles: _Mapping[str, str]
    goal: _Mapping[str, _Any]
    instruction: str
    initial_q: tuple[float, ...] | None
    held_object: str | None

    def __post_init__(self):
        object.__setattr__(self, "objects", _MappingProxyType(dict(self.objects)))
        object.__setattr__(self, "roles", _MappingProxyType(dict(self.roles)))
        object.__setattr__(self, "goal", _freeze_value(self.goal))
        if self.initial_q is not None:
            object.__setattr__(self, "initial_q", tuple(float(value) for value in self.initial_q))


def park_positions(names):
    return {name: (0.1 * i, 0.0, -2.0) for i, name in enumerate(names)}


def _format_goal(value):
    if isinstance(value, (float, _np.floating)):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_goal(item) for item in value)
    return str(value)


def render_instruction(template, roles, colors, goal):
    parts = []
    try:
        for literal, key, fmt, conversion in _string.Formatter().parse(template):
            parts.append(literal)
            if key is None:
                continue
            if fmt or conversion:
                raise _ConfigError("Instruction placeholders do not take format specifications")
            head, tail = key.split(".")
            if head == "goal":
                parts.append(_format_goal(goal[tail]))
            elif tail == "color":
                parts.append(str(colors[roles[head]]))
            elif tail == "name":
                parts.append(_re.sub(r"_\d+$", "", roles[head]).replace("_", " "))
            else:
                raise _ConfigError(f"Invalid instruction field: {key}")
    except (ValueError, KeyError, TypeError) as exc:
        raise _ConfigError(f"Cannot render instruction: {exc}") from exc
    return "".join(parts)


def _number(value):
    return type(value) in (int, float) and _math.isfinite(value)


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(v) for v in value)


def _range(value, *, nonnegative=False):
    return _vector(value, 2) and value[0] <= value[1] and (not nonnegative or value[0] >= 0)


def validate_workspace(table):
    keys = {"r_m", "bearing_deg"}
    clip_keys = {"xy_min", "xy_max"}
    if (not isinstance(table, _Mapping) or set(table) not in (keys, keys | clip_keys)
            or not _range(table["r_m"], nonnegative=True) or table["r_m"][0] == table["r_m"][1]
            or not _range(table["bearing_deg"]) or table["bearing_deg"][0] == table["bearing_deg"][1]):
        raise _ConfigError("Invalid arm workspace")
    if "xy_min" in table and (not all(_vector(table[key], 2) for key in clip_keys)
                              or not all(lo < hi for lo, hi in zip(table["xy_min"], table["xy_max"]))):
        raise _ConfigError("Invalid arm workspace clip")
    return table


def in_workspace(xy, workspace):
    radius = _math.hypot(*xy)
    lo, hi = workspace["bearing_deg"]
    bearing = lo + (_math.degrees(_math.atan2(xy[1], xy[0])) - lo) % 360.0
    return (workspace["r_m"][0] <= radius <= workspace["r_m"][1] and bearing <= hi
            and ("xy_min" not in workspace or all(lo <= x <= hi for x, lo, hi in
                                                   zip(xy, workspace["xy_min"], workspace["xy_max"]))))


def _workspace_region(region):
    return region == "workspace" or isinstance(region, _Mapping) and set(region) == {"workspace_inset_m"}


def _uses_workspace(start):
    return (any(_workspace_region(place["region"]) for place in start.get("place", {}).values())
            or any(_workspace_region(goal.get("region", goal.get("center_region")))
                   for goal in start.get("goal", {}).values()))


def _region_workspace(region, workspace):
    if not _workspace_region(region):
        return None
    if workspace is None:
        raise ValueError("Workspace region requires an arm workspace")
    if region == "workspace":
        return workspace
    inset = region["workspace_inset_m"]
    lo, hi = workspace["r_m"]
    result = {**workspace, "r_m": [lo + inset, hi - inset]}
    if "xy_min" in workspace:
        result.update(xy_min=[v + inset for v in workspace["xy_min"]],
                      xy_max=[v - inset for v in workspace["xy_max"]])
    return validate_workspace(result)


def _validate_region(region):
    if region == "workspace":
        return
    if _workspace_region(region) and _number(region["workspace_inset_m"]) and region["workspace_inset_m"] >= 0:
        return
    if not isinstance(region, dict):
        raise _ConfigError("Region must be a table")
    if set(region) == {"xy_min", "xy_max"}:
        if (_vector(region["xy_min"], 2) and _vector(region["xy_max"], 2)
                and all(a <= b for a, b in zip(region["xy_min"], region["xy_max"]))):
            return
    elif set(region) == {"r_m", "bearing_deg"}:
        if _range(region["r_m"], nonnegative=True) and _range(region["bearing_deg"]):
            return
    raise _ConfigError("Invalid rectangular or polar region")


def _validate_start(spec):
    """Validate only present v4 tables, preserving legacy start layouts."""
    start = spec.raw.get("start", {})
    if not isinstance(start, dict):
        raise _ConfigError("task.start must be a table")
    target_set = spec.object_sets.get(spec.roles.get("target"))
    role_counts = _Counter(spec.roles.values())
    for name, obj_set in spec.object_sets.items():
        if role_counts[name] > obj_set.count_max:
            raise _ConfigError(f"Too many roles for object set {name}")
    if "count" in start and target_set is None:
        raise _ConfigError("start.count requires a target set")
    if any(color not in spec.palette for color in spec.holdout.colors):
        raise _ConfigError("Unknown holdout colour")
    if spec.holdout.counts and target_set is None:
        raise _ConfigError("Holdout counts require a target set")
    if target_set is not None:
        count = start.get("count", (target_set.count_max, target_set.count_max))
        if (not isinstance(count, (list, tuple)) or len(count) != 2 or any(type(v) is not int for v in count)
                or not 1 <= count[0] <= count[1] <= target_set.count_max):
            raise _ConfigError("Invalid start.count range")
        if any(not count[0] <= c <= count[1] for c in spec.holdout.counts):
            raise _ConfigError("Holdout count outside start.count")
        allowed = [c for c in range(count[0], count[1] + 1) if c not in spec.holdout.counts]
        if not allowed:
            raise _ConfigError("No allowed object count")
        if allowed[0] < role_counts[target_set.name]:
            raise _ConfigError(f"Not enough active slots for roles in {target_set.name}")
        palette = [c for c in spec.palette if c not in spec.holdout.colors]
        if spec.palette and (not palette or (start.get("distinct_colors", True) and len(palette) < target_set.count_max)
                             or (count[1] > 1 and len(palette) < 2)):
            raise _ConfigError("Insufficient training colours")
    if "max_tries" in start and (type(start["max_tries"]) is not int or start["max_tries"] < 1):
        raise _ConfigError("max_tries must be a positive integer")
    for key in ("min_separation_m", "clearance_m"):
        if key in start and (not _number(start[key]) or start[key] < 0):
            raise _ConfigError(f"Invalid {key}")
    if "table_z_m" in start and not _number(start["table_z_m"]):
        raise _ConfigError("Invalid table height")
    if "distinct_colors" in start and type(start["distinct_colors"]) is not bool:
        raise _ConfigError("distinct_colors must be boolean")
    plain = {name for name, obj in spec.objects.items() if "set" not in obj.params}
    for name, obj in spec.objects.items():
        if "size_variants" in obj.params:
            variants = obj.params["size_variants"]
            if ((obj.size is None and name in plain) or not isinstance(variants, (tuple, list)) or len(variants) < 2
                    or not all(_vector(size, 3) and min(size) > 0 for size in variants)):
                raise _ConfigError(f"Invalid size variants for {name}")
    places = start.get("place", {})
    if not isinstance(places, dict):
        raise _ConfigError("start.place must be a table")
    for name, place in places.items():
        if name not in plain | spec.object_sets.keys() or not isinstance(place, dict) or "region" not in place:
            raise _ConfigError(f"Invalid placement entry: {name}")
        _validate_region(place["region"])
        if "yaw_deg" in place and not _range(place["yaw_deg"]):
            raise _ConfigError("Invalid yaw range")
    goals = start.get("goal", {})
    if not isinstance(goals, dict):
        raise _ConfigError("start.goal must be a table")
    def valid_object(ref):
        return isinstance(ref, str) and (ref in spec.objects or (ref.startswith("@") and ref[1:] in spec.roles))

    for name, goal in goals.items():
        if not isinstance(goal, dict) or goal.get("type") not in ("point", "points", "grid", "gap", "region", "joint"):
            raise _ConfigError(f"Unknown goal type: {name}")
        kind = goal["type"]
        if kind in ("point", "points"):
            _validate_region(goal.get("region"))
            if not _range(goal.get("z_m")):
                raise _ConfigError("Point goal requires a z_m range")
            if kind == "points" and (type(goal.get("count")) is not int or goal["count"] < 1):
                raise _ConfigError("Points goal needs a positive count")
            if "min_separation_m" in goal and (not _number(goal["min_separation_m"]) or goal["min_separation_m"] < 0):
                raise _ConfigError("Invalid point separation")
        elif kind == "region":
            region = goal.get("center_region")
            if isinstance(region, dict) and "around" in region:
                if (set(region) != {"around", "distance_m"} or not valid_object(region["around"])
                        or not _range(region["distance_m"], nonnegative=True)):
                    raise _ConfigError("Invalid object-relative region goal")
            else:
                _validate_region(region)
            if not _vector(goal.get("half_size_m"), 2) or any(v < 0 for v in goal["half_size_m"]):
                raise _ConfigError("Region goal requires nonnegative half extents")
        elif kind == "grid":
            if not isinstance(goal.get("points"), list) or not goal["points"] or not all(_vector(p, 3) for p in goal["points"]):
                raise _ConfigError("Grid goal requires 3-D points")
        elif kind == "gap":
            if not _range(goal.get("gap_m"), nonnegative=True):
                raise _ConfigError("Invalid gap range")
        elif kind == "joint":
            joints = goal.get("joints")
            if joints != "arm" and (not isinstance(joints, list) or not joints or not all(isinstance(j, str) for j in joints)):
                raise _ConfigError("Joint goal requires joints")
            fraction = goal.get("fraction", (0.1, 0.9))
            if not _range(fraction) or not 0 <= fraction[0] <= fraction[1] <= 1:
                raise _ConfigError("Invalid joint fraction range")
            if type(goal.get("count", 2)) is not int or goal.get("count", 2) < 1:
                raise _ConfigError("Invalid joint goal count")
    reachable = start.get("reachable", [])
    if not isinstance(reachable, list):
        raise _ConfigError("reachable must be a list")
    for ref in reachable:
        if not valid_object(ref) and not (isinstance(ref, str) and ref.startswith("@goal.") and ref[6:] in goals):
            raise _ConfigError(f"Invalid reachable reference: {ref}")
    if "held" in start and not valid_object(start["held"]):
        raise _ConfigError("Invalid held reference")
    initial = start.get("initial_q", "home")
    if isinstance(initial, dict) and "around" in initial:
        optional = {"min_tcp_height_m", "min_tcp_clearance_m"}
        if (initial.keys() - {"around", "spread"} - optional or initial["around"] != "home"
                or not _number(initial.get("spread")) or not 0 < initial["spread"] <= 0.5
                or any(not _number(initial[key]) or initial[key] < 0 for key in optional & initial.keys())):
            raise _ConfigError("Invalid random initial_q")
    elif initial != "home" and (not isinstance(initial, dict) or not all(_number(v) for v in initial.values())):
        raise _ConfigError("initial_q must be home or a joint table")


def _xy(rng, region, workspace=None):
    if workspace is not None:
        lo, hi = workspace["r_m"]
        radius = _math.sqrt(rng.uniform(lo**2, hi**2))
        bearing = _math.radians(rng.uniform(*workspace["bearing_deg"]))
        xy = _np.array([radius * _math.cos(bearing), radius * _math.sin(bearing)])
        return xy if in_workspace(xy, workspace) else None
    if "xy_min" in region:
        return rng.uniform(region["xy_min"], region["xy_max"])
    radius = rng.uniform(*region["r_m"])
    bearing = _math.radians(rng.uniform(*region["bearing_deg"]))
    return _np.array([radius * _math.cos(bearing), radius * _math.sin(bearing)], dtype=_np.float64)


def _pose(xy, z, yaw):
    return _Pose((float(xy[0]), float(xy[1]), float(z)),
                 (0.0, 0.0, _math.sin(yaw / 2), _math.cos(yaw / 2)))


def _footprint(spec, pose):
    hx, hy = _dims(spec)[:2] / 2
    yaw = _yaw(pose)
    c, s = _math.cos(yaw), _math.sin(yaw)
    rotation = _np.array([[c, -s], [s, c]])
    return _np.array([(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]) @ rotation.T + pose.position[:2]


def _footprint_distance(left, right):
    separated = False
    for polygon in (left, right):
        for edge in _np.roll(polygon, -1, axis=0) - polygon:
            if not _np.any(edge):
                continue
            axis = _np.array([-edge[1], edge[0]])
            a, b = left @ axis, right @ axis
            if a.max() < b.min() or b.max() < a.min():
                separated = True
                break
    if not separated:
        return 0.0
    distance = _math.inf
    for points, polygon in ((left, right), (right, left)):
        for a, b in zip(polygon, _np.roll(polygon, -1, axis=0)):
            edge = b - a
            length2 = float(edge @ edge)
            t = _np.clip((points - a) @ edge / length2, 0, 1) if length2 else _np.zeros(4)
            distance = min(distance, float(_np.linalg.norm(points - (a + t[:, None] * edge), axis=1).min()))
    return distance


def _object_ref(ref, roles):
    return roles[ref[1:]] if ref.startswith("@") else ref


def _goal(rng, entry, *, reachable, reach_fn, max_tries, joint_names, joint_limits, arm_joints,
          workspace, poses, roles):
    kind = entry["type"]
    if kind in ("point", "points"):
        region = entry["region"]
        clip = _region_workspace(region, workspace)
        points = []
        for _ in range(entry.get("count", 1) if kind == "points" else 1):
            for _ in range(max_tries):
                xy = _xy(rng, region, clip)
                if xy is None:
                    continue
                point = (float(xy[0]), float(xy[1]), float(rng.uniform(*entry["z_m"])))
                if any(_np.linalg.norm(_np.subtract(point, previous)) < entry.get("min_separation_m", 0) for previous in points):
                    continue
                if reachable and reach_fn is not None and not reach_fn(_Pose(point, (1.0, 0.0, 0.0, 0.0))):
                    continue
                points.append(point)
                break
            else:
                raise _SamplingError("Point goal exceeded max_tries")
        return points[0] if kind == "point" else points
    if kind == "grid":
        return [float(v) for v in entry["points"][rng.integers(len(entry["points"]))]]
    if kind == "gap":
        return float(rng.uniform(*entry["gap_m"]))
    if kind == "region":
        region = entry["center_region"]
        clip = _region_workspace(region, workspace)
        relative = isinstance(region, _Mapping) and "around" in region
        if relative:
            name = _object_ref(region["around"], roles)
            if name not in poses:
                raise _SamplingError(f"Region goal references inactive object {name}")
            anchor = _np.asarray(poses[name].position[:2])
        for _ in range(max_tries):
            if relative:
                radius = rng.uniform(*region["distance_m"])
                angle = rng.uniform(0, 2 * _math.pi)
                center = anchor + radius * _np.array([_math.cos(angle), _math.sin(angle)])
                if workspace is not None and not in_workspace(center, workspace):
                    continue
            else:
                center = _xy(rng, region, clip)
                if center is None:
                    continue
            return {"xy_min": (center - entry["half_size_m"]).tolist(),
                    "xy_max": (center + entry["half_size_m"]).tolist()}
        raise _SamplingError("Region goal exceeded max_tries")
    if joint_names is None or joint_limits is None or (entry["joints"] == "arm" and arm_joints is None):
        raise ValueError("Joint goals require joint_names, joint_limits, and arm_joints for arm goals")
    candidates = arm_joints if entry["joints"] == "arm" else entry["joints"]
    if not candidates:
        raise ValueError("Joint goal has no candidates")
    joint = candidates[rng.integers(len(candidates))]
    index = list(joint_names).index(joint)
    lower, upper = joint_limits[0][index], joint_limits[1][index]
    fractions = rng.uniform(*entry.get("fraction", (0.1, 0.9)), size=entry.get("count", 2))
    return {"joint": joint, "angles": [float(lower + f * (upper - lower)) for f in fractions]}


def _initial_q(start, rng, poses, home_q, joint_limits, joint_names, arm_joints, tcp_fn, max_tries):
    initial = start.get("initial_q", "home")
    q = list(home_q) if home_q is not None else None
    if not isinstance(initial, dict):
        return q
    if "around" not in initial:
        if q is None or joint_names is None or len(q) != len(joint_names):
            raise ValueError("initial_q overrides require matching home_q and joint_names")
        for name, value in initial.items():
            q[list(joint_names).index(name)] = float(value)
        return q
    if q is None:
        return None
    if joint_limits is None or joint_names is None or arm_joints is None or tcp_fn is None:
        raise ValueError("Random initial_q requires joint limits, joint names, arm joints and tcp_fn")
    lo, hi = joint_limits
    if len(lo) != len(hi) or len(hi) != len(q) or len(q) != len(joint_names):
        raise ValueError("Random initial_q requires one home and limit per joint")
    indices = [i for i, name in enumerate(joint_names) if name in arm_joints]
    min_z = start.get("table_z_m", 0.0) + initial.get("min_tcp_height_m", 0.05)
    clearance = initial.get("min_tcp_clearance_m", 0.05)
    for _ in range(max_tries):
        for i in indices:
            delta = rng.uniform(-1, 1) * initial["spread"] * (hi[i] - lo[i])
            q[i] = float(_np.clip(home_q[i] + delta, lo[i], hi[i]))
        x, y, z = tcp_fn(tuple(q)).position
        if z >= min_z and all(_math.hypot(x - pose.position[0], y - pose.position[1]) >= clearance
                             for pose in poses.values()):
            return q
    raise _SamplingError("Random initial_q exceeded max_tries")


def sample_start(task_spec, rng, *, reach_fn=None, home_q=None, joint_limits=None,
                 instruction_rng=None, joint_names=None, arm_joints=None, workspace=None, tcp_fn=None):
    """Draw a deterministic episode start."""
    start = task_spec.raw.get("start", {})
    target_set = task_spec.object_sets.get(task_spec.roles.get("target"))
    active = set(task_spec.objects)
    colors = {name: None for name in task_spec.objects}
    available_slots = {name: list(obj_set.slot_names()) for name, obj_set in task_spec.object_sets.items()}
    if target_set is not None:
        lo, hi = start.get("count", (target_set.count_max, target_set.count_max))
        allowed = [count for count in range(lo, hi + 1) if count not in task_spec.holdout.counts]
        count = allowed[rng.integers(len(allowed))]
        target_color, distractors = None, [None] * (count - 1)
        if task_spec.palette:
            palette = [color for color in task_spec.palette if color not in task_spec.holdout.colors]
            target_color = palette[rng.integers(len(palette))]
            distractors = [str(c) for c in rng.choice([c for c in palette if c != target_color], count - 1,
                                                      replace=not start.get("distinct_colors", True))]
        slots = rng.permutation(target_set.count_max)[:count]
        names = target_set.slot_names()
        chosen = [names[int(i)] for i in slots]
        available_slots[target_set.name] = chosen.copy()
        active.difference_update(set(names) - set(chosen))
    role_counts = _Counter(task_spec.roles.values())
    roles = {}
    for role, name in task_spec.roles.items():
        if name not in available_slots:
            roles[role] = name
        elif role_counts[name] == 1:
            roles[role] = available_slots[name][0]
        else:
            slots = available_slots[name]
            roles[role] = slots.pop(int(rng.integers(len(slots))))
    if target_set is not None:
        target_slot = roles["target"]
        colors[target_slot] = target_color
        for name, color in zip((slot for slot in chosen if slot != target_slot), distractors):
            colors[name] = color
    drawn_sizes = {}
    definitions = dict(task_spec.objects)
    for name, obj in task_spec.objects.items():
        variants = obj.params.get("size_variants")
        if name in active and variants:
            size = tuple(float(value) for value in variants[rng.integers(len(variants))])
            drawn_sizes[name] = size
            definitions[name] = _replace(obj, size=size)
    reachable = start.get("reachable", ["@target"] if "target" in roles else [])
    reachable_objects = {_object_ref(ref, roles) for ref in reachable if not ref.startswith("@goal.")}
    max_tries = start.get("max_tries", 200)
    places = start.get("place", {})
    placement_order = [(name, place) for name, place in places.items() if name not in task_spec.object_sets]
    for name, place in places.items():
        if name in task_spec.object_sets:
            placement_order.extend((slot, place) for slot in task_spec.object_sets[name].slot_names() if slot in active)
    moving = {name for name, _ in placement_order}
    poses = {}
    for name, obj in task_spec.objects.items():
        if name in active and name not in moving:
            poses[name] = _pose(obj.position[:2], obj.position[2], _math.radians(obj.rotation_euler_degrees[2]))
    for name, place in placement_order:
        obj = definitions[name]
        height = _dims(obj)[2]
        z = start.get("table_z_m", 0.0) + (0 if _bottom_origin(obj) else height / 2)
        region = place["region"]
        clip = _region_workspace(region, workspace)
        for _ in range(max_tries):
            xy = _xy(rng, region, clip)
            if xy is None:
                continue
            yaw = _math.radians(rng.uniform(*place.get("yaw_deg", (0, 0))))
            pose = _pose(xy, z, yaw)
            valid = True
            for other, other_pose in poses.items():
                other_obj = definitions[other]
                if "inner_half_size" in obj.params or "inner_half_size" in other_obj.params:
                    valid = _footprint_distance(_footprint(obj, pose), _footprint(other_obj, other_pose)) >= start.get("clearance_m", 0.01)
                else:
                    valid = _math.hypot(xy[0] - other_pose.position[0], xy[1] - other_pose.position[1]) >= start.get("min_separation_m", 0.035)
                if not valid:
                    break
            if not valid:
                continue
            if reach_fn is not None and name in reachable_objects:
                center_z = z + (height / 2 if _bottom_origin(obj) else 0)
                if not reach_fn(_Pose((float(xy[0]), float(xy[1]), float(center_z + 0.06)), (1.0, 0.0, 0.0, 0.0))):
                    continue
            poses[name] = pose
            break
        else:
            raise _SamplingError(f"Placement of {name} exceeded max_tries")
    parked = park_positions(list(task_spec.objects))
    objects = {name: _ObjectState(poses[name], True, colors[name], drawn_sizes.get(name)) if name in active
               else _ObjectState(_Pose(parked[name]), False, None) for name in task_spec.objects}
    goal = {name: _goal(rng, entry, reachable=f"@goal.{name}" in reachable, reach_fn=reach_fn,
                        max_tries=max_tries, joint_names=joint_names, joint_limits=joint_limits, arm_joints=arm_joints,
                        workspace=workspace, poses=poses, roles=roles)
            for name, entry in start.get("goal", {}).items()}
    q = _initial_q(start, rng, poses, home_q, joint_limits, joint_names, arm_joints, tcp_fn, max_tries)
    held = _object_ref(start["held"], roles) if "held" in start else None
    instruction = task_spec.instruction
    if task_spec.instruction_templates:
        index = int(instruction_rng.integers(len(task_spec.instruction_templates))) if instruction_rng is not None else 0
        instruction = render_instruction(task_spec.instruction_templates[index], roles,
                                         {name: c for name, c in colors.items() if name in active and c is not None}, goal)
    return StartSample(objects, roles, goal, instruction, None if q is None else tuple(q), held)
