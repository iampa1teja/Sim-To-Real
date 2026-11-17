"""Pure-Python planning for domain-randomized replay (no Isaac / LeRobot imports)."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from core.datasets.task_meta import validate_sidecar
from core.generate.appearance import AppearanceDraw, dr_copy_sidecar, draw_appearance


def __getattr__(name):
    if name != "COLORS":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Only legacy cube replay needs the measured scene palette.
    from core.scenes import load_setup

    colors = {
        "blue": tuple(load_setup()["object_materials"]["blue_cube"]["color"]),
        "red": (0.8, 0.05, 0.05),
    }
    globals()[name] = colors
    return colors


DR_TERMS = ("light", "robot_color", "camera_external", "camera_wrist")
# Cameras are calibrated to the real setup, so the default keeps camera views fixed.
DEFAULT_DR = ("light", "robot_color")


@dataclass
class DRRanges:
    exposure: tuple[float, float]
    robot_colors: dict[str, tuple]
    camera_pos: dict[str, tuple[float, float]]
    camera_rot: dict[str, tuple[float, float]]
    focal_length: tuple[float, float] | None  # None when camera DR is not configured


@dataclass
class DRDraw:
    exposure: float | None
    robot_color: str | None
    camera_external: dict | None
    camera_wrist_focal: float | None


@dataclass
class PlannedEpisode:
    dst_index: int
    src_index: int
    color: str
    dr_seed: int
    task_text: str
    draw: DRDraw | None = None


def parse_csv(text, allowed) -> list[str]:
    values = [s.strip() for s in text.split(",")]
    allowed = set(allowed)
    if not text or any(not value or value not in allowed for value in values):
        raise ValueError(f"Invalid CSV value: {text!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate CSV value: {text!r}")
    return values


def episode_seed(seed, src_index, color) -> int:
    data = f"{seed}:{src_index}:{color}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def draw_dr(terms, ranges, rng: random.Random) -> DRDraw:
    unknown_terms = set(terms) - set(DR_TERMS)
    if unknown_terms:
        raise ValueError(f"Unknown DR term(s): {sorted(unknown_terms)}")
    exposure = (rng.uniform(*ranges.exposure) if "light" in terms else None)
    robot_color = rng.choice(list(ranges.robot_colors.keys())) if "robot_color" in terms else None
    camera_external = None
    if "camera_external" in terms:
        pos = {k: rng.uniform(*v) for k, v in ranges.camera_pos.items()}
        rot = {k: rng.uniform(*v) for k, v in ranges.camera_rot.items()}
        focal = rng.uniform(*ranges.focal_length)
        camera_external = {"pos": pos, "rot": rot, "focal": focal}
    camera_wrist_focal = rng.uniform(*ranges.focal_length) if "camera_wrist" in terms else None
    return DRDraw(exposure, robot_color, camera_external, camera_wrist_focal)


def task_text(color, lang_blue, lang_red) -> str:
    if color == "blue":
        return lang_blue
    elif color == "red":
        return lang_red
    else:
        raise ValueError(f"Unknown color: {color}")


def plan_episodes(src_indices, colors, terms, ranges, seed, include_originals, lang_blue, lang_red) -> list[PlannedEpisode]:
    episodes = []
    dst_index = 0

    if include_originals:
        for src_index in src_indices:
            color = "blue"
            episodes.append(PlannedEpisode(
                dst_index=dst_index,
                src_index=src_index,
                color=color,
                dr_seed=episode_seed(seed, src_index, color),
                task_text=task_text(color, lang_blue, lang_red),
                draw=None,
            ))
            dst_index += 1

    for src_index in src_indices:
        for color in colors:
            dr_seed = episode_seed(seed, src_index, color)
            rng = random.Random(dr_seed)
            draw = draw_dr(terms, ranges, rng)
            episodes.append(PlannedEpisode(
                dst_index=dst_index,
                src_index=src_index,
                color=color,
                dr_seed=dr_seed,
                task_text=task_text(color, lang_blue, lang_red),
                draw=draw,
            ))
            dst_index += 1
    return episodes


def build_manifest(plan, args: dict, git_commit: str) -> dict:
    originals = sum(episode.draw is None for episode in plan)
    generated = [episode for episode in plan if episode.draw is not None]
    return {
        "git_commit": git_commit,
        "args": args,
        "episodes": [
            {
                "dst_index": episode.dst_index,
                "src_index": episode.src_index,
                "color": episode.color,
                "dr_seed": episode.dr_seed,
                "task_text": episode.task_text,
                "draw": (
                    {
                        "exposure": episode.draw.exposure,
                        "robot_color": episode.draw.robot_color,
                        "camera_external": episode.draw.camera_external,
                        "camera_wrist_focal": episode.draw.camera_wrist_focal,
                    }
                    if episode.draw is not None
                    else None
                ),
            }
            for episode in plan
        ],
        "counts": {
            "total": len(plan),
            "originals": originals,
            "blue": sum(episode.color == "blue" for episode in generated),
            "red": sum(episode.color == "red" for episode in generated),
        },
    }


def write_json_atomic(path, data) -> None:
    directory = os.path.dirname(os.path.abspath(path))

    fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception as e:
        os.remove(temp_path)
        raise e


@dataclass(frozen=True)
class PlannedEpisodeV2:
    dst_index: int
    src_index: int
    draw: AppearanceDraw | None = None
    copy_index: int | None = None


def plan_episodes_v2(sources, *, task_spec, cfg, exposure_range, robot_colors, include_originals):
    sources = list(sources)
    if any(source.get("legacy_v1") or not source.get("fingerprints") for source in sources):
        raise ValueError("V2 replay requires source sidecars with fingerprints")
    episodes = []
    if include_originals:
        episodes.extend(PlannedEpisodeV2(index, source["episode_index"]) for index, source in enumerate(sources))
    for parent in sources:
        taken = {parent["fingerprints"]["appearance_fp"]}
        for copy_index in range(cfg.copies):
            draw = draw_appearance(parent, task_spec=task_spec, terms=cfg.terms, exposure_range=exposure_range,
                                   robot_colors=robot_colors, recolor=cfg.recolor, dr_seed=cfg.seed,
                                   copy_index=copy_index, taken=taken)
            episodes.append(PlannedEpisodeV2(len(episodes), parent["episode_index"], draw, copy_index))
            taken.add(draw.appearance_fp)
    return episodes


def robot_color_names():
    path = Path(__file__).resolve().parents[1] / "envs" / "mdp" / "resets.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ROBOT_COLORS" for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"ROBOT_COLORS is missing from {path}")


def exposure_range(task_spec):
    bounds = task_spec.raw.get("dr", {}).get("room_light_exposure_range", (-3.0, 1.0))
    return tuple(map(float, bounds))


def sidecar_v2(planned, parent, *, source_repo_id):
    if planned.draw is None:
        payload = deepcopy(parent)
        payload["episode_index"] = planned.dst_index
    else:
        payload = dr_copy_sidecar(parent, planned.draw, episode_index=planned.dst_index, source_repo_id=source_repo_id)
    validate_sidecar(payload)
    return payload


def build_manifest_v2(plan, args, git_commit):
    episodes = []
    for planned in plan:
        draw = planned.draw
        episodes.append(dict(dst_index=planned.dst_index, src_index=planned.src_index, copy_index=planned.copy_index,
                             draw=None if draw is None else dict(
                                 colors=dict(draw.colors), instruction=draw.instruction,
                                 light_exposure=draw.light_exposure, robot_color=draw.robot_color,
                                 seed=draw.seed, appearance_fp=draw.appearance_fp)))
    originals = sum(item["draw"] is None for item in episodes)
    return dict(git_commit=git_commit, args=args, episodes=episodes,
                counts=dict(total=len(episodes), originals=originals, copies=len(episodes) - originals))


def print_plan_v2(plan):
    originals = 0
    for planned in plan:
        draw = planned.draw
        originals += draw is None
        print(json.dumps(dict(dst=planned.dst_index, src=planned.src_index, copy=planned.copy_index,
                              robot_color=draw.robot_color if draw else None,
                              light=draw.light_exposure if draw else None,
                              instruction=draw.instruction if draw else None)))
    print(f"{len(plan)} episodes: {originals} originals, {len(plan) - originals} copies")
