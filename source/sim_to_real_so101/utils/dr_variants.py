"""Pure-Python planning for domain-randomized replay (no Isaac / LeRobot imports)."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

_REAL_SETUP = Path(__file__).resolve().parents[1] / "assets" / "real_setup.json"

COLORS: dict[str, tuple[float, float, float]] = {
    # Measured real cube colour; follows any recalibration of real_setup.json.
    "blue": tuple(json.loads(_REAL_SETUP.read_text(encoding="utf-8"))["object_materials"]["blue_cube"]["color"]),
    "red": (0.8, 0.05, 0.05),
}
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