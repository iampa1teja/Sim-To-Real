"""Deterministic, parent-distinct DR appearance draws and sidecar copies."""

import copy as _copy
import string as _string
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from types import MappingProxyType as _MappingProxyType

import numpy as _np

from core.generate.config import _validate_camera_jitter
from core.generate.errors import GenerationError as _GenerationError
from core.generate.fingerprint import appearance_fp as _appearance_fp, episode_fp as _episode_fp
from core.generate.seeds import derive_seed as _derive_seed
from core.tasks.base import _freeze_value


@_dataclass(frozen=True)
class AppearanceDraw:
    colors: _Mapping
    instruction: str
    light_exposure: float | None
    robot_color: str | None
    seed: int
    appearance_fp: str
    camera_offsets: _Mapping | None = None

    def __post_init__(self):
        object.__setattr__(self, "colors", _MappingProxyType(dict(self.colors)))
        if self.camera_offsets is not None:
            object.__setattr__(self, "camera_offsets", _freeze_value(self.camera_offsets))


def _render(template, roles, colors, goal):
    result = []
    try:
        for literal, field, fmt, conversion in _string.Formatter().parse(template):
            result.append(literal)
            if field is None:
                continue
            if fmt or conversion:
                raise ValueError("Unsupported instruction format")
            head, tail = field.split(".")
            if head == "goal":
                value = str(goal[tail])
            elif tail == "name":
                value = roles[head]
            elif tail == "color" and colors[roles[head]] is not None:
                value = colors[roles[head]]
            else:
                raise ValueError(f"Unresolvable field: {field}")
            result.append(value)
    except (ValueError, TypeError, KeyError) as exc:
        raise _GenerationError(f"Cannot render DR instruction: {exc}") from exc
    return "".join(result)


def draw_appearance(parent, *, task_spec, terms, exposure_range, robot_colors, recolor, dr_seed, copy_index,
                     taken, max_redraws=100, camera_jitter=None, cameras=()) -> AppearanceDraw:
    fingerprints = parent.get("fingerprints")
    if fingerprints is None or fingerprints.get("traj_fp") is None:
        raise ValueError("DR requires a parent trajectory fingerprint")
    if any(term not in ("light", "robot_color") for term in terms):
        raise ValueError("Unknown appearance term")
    if "robot_color" in terms and not robot_colors:
        raise ValueError("robot_color requires a non-empty palette")
    if exposure_range[0] > exposure_range[1]:
        raise ValueError("Exposure range must increase")
    if camera_jitter is not None:
        jitter = _validate_camera_jitter(camera_jitter)
        cameras = sorted(set(cameras))
        if not cameras:
            raise ValueError("Camera jitter requires camera names")
    names = sorted(name for name, obj in parent["objects"].items() if obj["active"])
    base = {name: parent["objects"][name]["color"] for name in names}
    palette = [name for name in task_spec.palette if name not in task_spec.holdout.colors]
    roles = parent["roles"]
    target = roles.get("target")
    group = []
    if recolor and palette and target in base and base[target] is not None:
        target_set = parent["objects"][target]["set"]
        group = [target] + sorted(name for name in names if name != target and target_set is not None
                                   and parent["objects"][name]["set"] == target_set)
    distinct = task_spec.raw.get("start", {}).get("distinct_colors", True)
    if group and ((distinct and len(palette) < len(group)) or (len(group) > 1 and len(palette) < 2)):
        raise _GenerationError("Insufficient colours for a unique target and distractors")
    excluded = set(taken) | {fingerprints["appearance_fp"]}
    for redraw in range(max_redraws):
        seed = _derive_seed(dr_seed, fingerprints["traj_fp"], copy_index, redraw)
        rng = _np.random.default_rng(seed)
        colors = dict(base)
        if group:
            target_color = palette[rng.integers(len(palette))]
            distractors = rng.choice([name for name in palette if name != target_color], len(group) - 1, replace=not distinct)
            colors[group[0]] = target_color
            colors.update({name: str(color) for name, color in zip(group[1:], distractors)})
        templates = task_spec.instruction_templates
        instruction = (_render(templates[rng.integers(len(templates))], roles, colors, parent.get("goal", {}))
                       if templates else parent["instruction"])
        light = float(rng.uniform(*exposure_range)) if "light" in terms else None
        robot = str(robot_colors[rng.integers(len(robot_colors))]) if "robot_color" in terms else None
        camera_offsets = None
        if camera_jitter is not None:
            camera_offsets = {name: {key: rng.uniform(-limit, limit, size=3).tolist()
                                     for key, limit in jitter.items()} for name in cameras}
        fp = _appearance_fp(colors, instruction, light, robot, camera_offsets)
        if fp not in excluded:
            return AppearanceDraw(colors, instruction, light, robot, seed, fp, camera_offsets)
    raise _GenerationError("Appearance space exhausted without a distinct draw")


def dr_copy_sidecar(parent, draw, *, episode_index, source_repo_id) -> dict:
    result = _copy.deepcopy(parent)
    result["episode_index"] = episode_index
    for name, color in draw.colors.items():
        result["objects"][name]["color"] = color
    result["instruction"] = draw.instruction
    result["appearance"] = {"light_exposure": draw.light_exposure, "robot_color": draw.robot_color}
    if draw.camera_offsets:
        result["appearance"]["camera_offsets"] = {name: {key: list(value) for key, value in offsets.items()}
                                                   for name, offsets in draw.camera_offsets.items()}
    fp = parent["fingerprints"]
    result["fingerprints"] = {key: fp[key] for key in ("scenario_key", "spec_fp", "traj_fp")}
    result["fingerprints"].update(appearance_fp=draw.appearance_fp, episode_fp=_episode_fp(fp["traj_fp"], draw.appearance_fp))
    result["source"] = {"repo_id": source_repo_id, "episode_index": parent["episode_index"],
                        "traj_fp": fp["traj_fp"], "episode_fp": fp["episode_fp"]}
    result["generator"] = {"name": "dr_replay", "git_commit": parent.get("generator", {}).get("git_commit") or "unknown"}
    result["legacy_v1"] = False
    return result
