#!/usr/bin/env python3
"""Validate an evaluation YAML and print shell assignments for docker/eval.sh.

Usage: eval_config.py <config.yaml>      (output is meant for `eval "$(...)"`)
Needs PyYAML on the host (pip install pyyaml).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required on the host: pip install pyyaml")

TASKS = {
    "pick_place": "Lerobot-So101-Teleop-Pick-Place-Eval",
    "pick_place_dr": "Lerobot-So101-Teleop-Pick-Place-DR-Eval",
}
SPLITS = ("id", "ood", "yaw", "train")
DEFAULT_LANG = "Pick up the blue cube and place it in the white box"
DEFAULT_RENAME = {"realsense_rgb": "room", "wrist_cam": "wrist"}

SECTIONS = {
    "task": None,
    "policy": {"name", "checkpoint", "models_dir", "port", "server_image", "options"},
    "eval": {"dataset", "episodes", "start_mode", "random_fraction", "robot_start", "render_warmup",
             "episode_length_s", "action_horizon", "action_repeat", "molmo_overhead_camera", "eval_set", "splits", "repeats", "lang",
             "lang_by_color", "rename_map", "results_json", "robot_usd", "save_videos", "video_dir", "video_encoder"},
    "grasp": {"gripper_latch", "log_gripper", "seed_per_episode", "latch_close_below",
              "latch_open_above", "latch_confirm_steps", "latch_release_confirm", "latch_min_hold",
              "latch_dataset", "gripper_csv", "latch_steady_steps", "latch_steady_max_delta",
              "gripper_empty_closed", "latch_empty_margin", "latch_empty_confirm", "latch_reopen_steps"},
    "run": {"gui", "rerun"},
    "smoke_test": {"selection", "reach_timeout_s", "progress_timeout_s"},
}
GRASP_FLAGS = {"gripper_latch", "log_gripper", "seed_per_episode"}


def fail(message: str) -> None:
    sys.exit(f"ERROR: {message}")


def load(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        fail(f"Cannot read {path}: {error}")
    if not isinstance(data, dict):
        fail(f"{path} must contain a YAML mapping.")
    for key, value in data.items():
        if key not in SECTIONS:
            fail(f"Unknown top-level key {key!r}; allowed: {', '.join(SECTIONS)}.")
        allowed = SECTIONS[key]
        if allowed is None:
            continue
        if value is None:
            data[key] = value = {}
        if not isinstance(value, dict):
            fail(f"{key!r} must be a mapping.")
        unknown = sorted(set(value) - allowed)
        if unknown:
            fail(f"Unknown key(s) in {key!r}: {', '.join(unknown)}.")
    return data


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        fail(f"{name} must be a positive integer.")
    return value


def text(value, name: str) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        fail(f"{name} must be a string.")
    return str(value)


def options_list(options, name: str) -> list[str]:
    if options is None:
        return []
    if not isinstance(options, dict):
        fail(f"{name} must be a mapping of option names to values.")
    pairs = []
    for key, value in options.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            fail(f"Invalid option name {key!r} in {name}.")
        pairs.append(f"{key}={value}")
    return pairs


def build(data: dict, *, molmo_overhead_camera: bool = False) -> dict:
    out: dict = {}
    task = data.get("task", "pick_place")
    if task not in TASKS:
        fail(f"task must be one of: {', '.join(TASKS)}.")
    out["dr"] = "dr" if task == "pick_place_dr" else "nodr"
    out["task"] = TASKS[task]

    policy = data.get("policy", {})
    name = text(policy.get("name", "groot_n17"), "policy.name")
    if not re.fullmatch(r"[a-z0-9_]+", name):
        fail("policy.name must be lowercase letters, digits and underscores.")
    checkpoint = policy.get("checkpoint")
    if not checkpoint:
        fail("policy.checkpoint is required (path relative to policy.models_dir).")
    checkpoint = text(checkpoint, "policy.checkpoint").rstrip("/")
    if checkpoint.startswith("/") or any(part in (".", "..") for part in checkpoint.split("/")):
        fail("policy.checkpoint must be a relative path under policy.models_dir, without . or .. components.")
    port = policy.get("port")
    port_text = ""
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port <= 65535:
            fail("policy.port must be an integer from 1 to 65535.")
        port_text = str(port)
    image = text(policy.get("server_image", "policy-server:latest"), "policy.server_image")
    if not image or image.startswith("-") or re.search(r"\s", image):
        fail("policy.server_image must be a Docker image name.")
    out.update(policy_name=name, model=checkpoint, port=port_text, server_image=image,
               models_dir=text(policy.get("models_dir", "~/models"), "policy.models_dir"))
    out["policy_options"] = options_list(policy.get("options"), "policy.options")

    run = data.get("eval", {})
    dataset = run.get("dataset")
    if not dataset or not str(dataset).startswith("/"):
        fail("eval.dataset is required and must be an absolute path inside teleop.")
    out["dataset"] = str(dataset).rstrip("/")
    out["episodes"] = positive_int(run.get("episodes", 20), "eval.episodes")
    out["repeats"] = positive_int(run.get("repeats", 1), "eval.repeats")
    start_mode = run.get("start_mode", "cycle")
    if start_mode not in ("cycle", "random"):
        fail("eval.start_mode must be cycle or random.")
    out["start_mode"] = start_mode
    fraction = run.get("random_fraction", 0.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        fail("eval.random_fraction must be in [0, 1].")
    out["random_fraction"] = fraction
    robot_start = run.get("robot_start", "recorded")
    if robot_start not in ("recorded", "default"):
        fail("eval.robot_start must be recorded or default.")
    out["robot_start"] = robot_start
    robot_usd = run.get("robot_usd")
    if robot_usd is not None and (not isinstance(robot_usd, str) or not robot_usd.startswith("/")):
        fail("eval.robot_usd must be an absolute USD path inside teleop.")
    out["robot_usd"] = robot_usd or ""
    warmup = run.get("render_warmup", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        fail("eval.render_warmup must be a non-negative integer.")
    out["render_warmup"] = warmup
    length = run.get("episode_length_s")
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
            fail("eval.episode_length_s must be finite and positive.")
    out["episode_length_s"] = "" if length is None else length
    max_horizon = 30 if name == "molmoact2" else 16
    horizon = run.get("action_horizon", max_horizon)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= max_horizon:
        fail(f"eval.action_horizon must be an integer in 1..{max_horizon} for {name}.")
    out["action_horizon"] = horizon
    out["action_repeat"] = positive_int(run.get("action_repeat", 1), "eval.action_repeat")
    overhead = run.get("molmo_overhead_camera", False)
    if not isinstance(overhead, bool):
        fail("eval.molmo_overhead_camera must be true or false.")
    overhead = overhead or molmo_overhead_camera
    if overhead and (name != "molmoact2" or task not in ("pick_place", "pick_place_dr")):
        fail("eval.molmo_overhead_camera requires molmoact2 and a pick-place task.")
    out["molmo_overhead_camera"] = "true" if overhead else "false"

    eval_set = run.get("eval_set")
    splits = run.get("splits")
    if isinstance(splits, str):
        splits = [item for item in splits.split(",") if item]
    splits = splits or []
    if (splits or out["repeats"] != 1) and not eval_set:
        fail("eval.splits and eval.repeats require eval.eval_set.")
    if eval_set and not str(eval_set).startswith("/"):
        fail("eval.eval_set must be an absolute path inside teleop.")
    bad = [item for item in splits if item not in SPLITS]
    if bad:
        fail(f"Invalid eval.splits {bad}; allowed: {', '.join(SPLITS)}.")
    out["eval_set"] = str(eval_set or "")
    out["splits"] = ",".join(splits)

    lang = run.get("lang", DEFAULT_LANG)
    out["lang"] = text(lang, "eval.lang")
    by_color = run.get("lang_by_color")
    if by_color is not None:
        required = ("blue", "red") if out["dr"] == "dr" else ("blue",)
        if not isinstance(by_color, dict) or not all(
                isinstance(by_color.get(c), str) and by_color[c].strip() for c in required):
            fail(f"eval.lang_by_color needs an instruction for: {', '.join(required)}.")
    out["lang_by_color"] = json.dumps(by_color) if by_color else ""
    rename = run.get("rename_map", DEFAULT_RENAME)
    if not isinstance(rename, dict) or not all(
            isinstance(rename.get(c), str) and rename[c].strip() for c in DEFAULT_RENAME) \
            or rename["realsense_rgb"] == rename["wrist_cam"]:
        fail("eval.rename_map must map realsense_rgb and wrist_cam to distinct nonempty camera names.")
    if overhead:
        rename = dict(rename, realsense_rgb="original_room", molmo_room="room", wrist_cam="wrist")
    elif name == "molmoact2" and rename.get("molmo_room") == "room" and rename.get("realsense_rgb") == "original_room":
        rename = dict(rename, realsense_rgb="room")
        rename.pop("molmo_room")
    out["rename_map"] = json.dumps(rename)
    results = run.get("results_json")
    if results and not str(results).startswith("/"):
        fail("eval.results_json must be an absolute path inside teleop.")
    out["results_json"] = str(results or "")
    save_videos = run.get("save_videos", False)
    if not isinstance(save_videos, bool):
        fail("eval.save_videos must be true or false.")
    out["save_videos"] = "true" if save_videos else "false"
    video_dir = run.get("video_dir")
    if video_dir is not None and (not isinstance(video_dir, str) or not video_dir.startswith("/")):
        fail("eval.video_dir must be an absolute path inside teleop.")
    out["video_dir"] = video_dir or ""
    video_encoder = run.get("video_encoder", "libx264")
    if video_encoder not in ("libx264", "h264_nvenc"):
        fail("eval.video_encoder must be libx264 or h264_nvenc.")
    out["video_encoder"] = video_encoder

    grasp_args: list[str] = []
    for key, value in (data.get("grasp") or {}).items():
        if key in GRASP_FLAGS:
            if not isinstance(value, bool):
                fail(f"grasp.{key} must be true or false.")
            if value:
                grasp_args.append(f"--{key}")
        elif value is not None:
            grasp_args += [f"--{key}", str(value)]
    out["seed_per_episode"] = "true" if "--seed_per_episode" in grasp_args else "false"
    out["grasp_args"] = grasp_args

    smoke = data.get("smoke_test") or {}
    selection = smoke.get("selection", "first_five")
    if selection not in ("first_five", "one_per_zone"):
        fail("smoke_test.selection must be first_five or one_per_zone.")
    if selection == "one_per_zone" and not eval_set:
        fail("smoke_test.selection=one_per_zone requires eval.eval_set.")
    out["smoke_selection"] = selection
    reach, progress = smoke.get("reach_timeout_s"), smoke.get("progress_timeout_s")
    if (reach is None) != (progress is None):
        fail("Set both smoke_test.reach_timeout_s and progress_timeout_s.")
    for key, value in (("reach_timeout_s", reach), ("progress_timeout_s", progress)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            fail(f"smoke_test.{key} must be finite and positive.")
        out["smoke_" + key] = "" if value is None else value
    if reach is not None and (name != "molmoact2" or not eval_set):
        fail("Smoke progress deadlines require molmoact2 and a fixed eval.eval_set.")

    flags = data.get("run") or {}
    for key in ("gui", "rerun"):
        if not isinstance(flags.get(key, False), bool):
            fail(f"run.{key} must be true or false.")
        out[key] = "true" if flags.get(key, False) else "false"
    ignored = {"gui", "rerun", "results_json", "save_videos", "video_dir", "video_encoder", "port", "server_image",
               "smoke_selection", "smoke_reach_timeout_s", "smoke_progress_timeout_s"}
    identity = {key: value for key, value in out.items() if key not in ignored}
    out["config_fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return out


def emit(values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, list):
            print(f"{key}=({' '.join(shlex.quote(str(item)) for item in value)})")
        else:
            print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--molmo-overhead-camera", action="store_true")
    args = parser.parse_args()
    emit(build(load(args.config), molmo_overhead_camera=args.molmo_overhead_camera))
