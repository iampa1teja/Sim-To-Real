# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from tqdm import tqdm

from isaaclab.app import AppLauncher


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_int(value):
    number = int(value)
    if isinstance(value, bool) or number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _ema_alpha(value):
    number = _positive_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Eval Client (remote GR00T inference server).")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-MyRoom", help="Name of the task.")
parser.add_argument("--seed", type=int, default=1984, help="Environment seed")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument(
    "--rename_map",
    type=str,
    required=False,
    default=None,
    help=(
        'JSON mapping for renaming camera keys to match policy/feature config: key is simulation feature name, value is policy feature name '
        'e.g. \'{"sim_name1": "policy_name1", "sim_name2": "policy_name2"}\'. '
    ),
)
parser.add_argument("--policy", default="groot_n17", help="Policy plugin name (a folder under policies/)")
parser.add_argument("--policy_option", action="append", default=[], metavar="KEY=VALUE",
                    help="Policy-specific client option; repeatable")
parser.add_argument("--policy_host", "--host", type=str, default="localhost", help="Policy server host")
parser.add_argument("--policy_port", "--port", type=int, default=None,
                    help="Policy server port (default: the plugin manifest's default_port)")
parser.add_argument("--policy_api_token", default=os.environ.get("POLICY_API_TOKEN"), help="Server API token")
parser.add_argument("--action_horizon", type=int, default=None,
                    help="Action steps per query: MolmoAct2 1..30 (default 30), other policies 1..16 (default 16)")
parser.add_argument("--action_repeat", type=_positive_int, default=1,
                    help="Control ticks to hold each predicted joint target (default 1)")
parser.add_argument("--prefetch_steps", type=_nonnegative_int, default=0,
                    help="Query P steps early; sim delivers P steps later and drops the first P actions (0: off)")
parser.add_argument("--blend_steps", type=_nonnegative_int, default=0,
                    help="Cross-fade old unexecuted predictions into new chunks over K steps (0: off)")
parser.add_argument("--ema_alpha", type=_ema_alpha, default=1.0,
                    help="Arm EMA coefficient in (0,1]; 1: off")
parser.add_argument("--smooth_gripper", action="store_true", help="Also apply EMA to the gripper")
parser.add_argument("--temporal_ensemble", type=_nonnegative_int, default=0,
                    help="Query every M steps and average overlapping predictions, favouring older chunks (0: off)")
parser.add_argument("--te_decay", type=_nonnegative_float, default=0.01,
                    help="Temporal ensemble weights exp(-k*i), i=0 oldest")
parser.add_argument("--log_timing", action="store_true", help="Write per-action CSV and print latency/jerk summary")
parser.add_argument("--timing_csv", default="", help="Timing CSV path (default: outputs/timing/sim_<timestamp>.csv)")
parser.add_argument("--episode_length_s", type=_positive_float, default=None,
                    help="Episode timeout in simulation seconds (default: task configuration; 15 s for Pick-Place-Eval)")
parser.add_argument(
    "--lang_instruction",
    type=str,
    default="Pick up the vial and place it in the rack",
    help="Language instruction for the policy",
)
parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")
parser.add_argument("--smoke-test", action="store_true",
                    help="Run five episodes with Rerun, without saving evaluation reports, videos or CSVs")
parser.add_argument("--molmo-overhead-camera", action="store_true",
                    help="Add a behind/above-arm room view for Molmo pick-place, preserving the existing cameras")
parser.add_argument("--save_videos", action="store_true", help="Save room/wrist MP4 videos per attempt")
parser.add_argument("--video_dir", type=Path, default=None, help="Video output directory (default: beside results JSON)")
parser.add_argument("--video_encoder", choices=("libx264", "h264_nvenc"), default="libx264",
                    help="Recording encoder: CPU libx264 or NVIDIA NVENC (requires video driver support)")
parser.add_argument("--render_warmup", type=int, default=0,
                    help="0 (default): render the cameras every control step. N>0: render only the N steps before each "
                         "policy query (the policy reads images once per action chunk); the extra N-1 frames refresh "
                         "the renderer's temporal history (DLAA, denoisers). Physics is unchanged.")

parser.add_argument("--eval_set", type=Path, help="Fixed benchmark JSON (overrides num_episodes)")
parser.add_argument("--splits", help="Comma-separated splits, default all in the file")
parser.add_argument("--repeats", type=int, default=1)

parser.add_argument("--cube_starts", help="Directory containing episode_*.json cube trajectories")
parser.add_argument("--start_mode", choices=("cycle", "random"), default="cycle")
parser.add_argument("--random_fraction", type=float, default=0.0)
parser.add_argument("--robot_start", choices=("recorded", "default"), default="recorded",
                    help="recorded: arm starts at each episode's recorded first-frame state (demo start pose); "
                         "default: the scene's calibrated reset pose")
parser.add_argument("--robot_usd", type=Path, default=None,
                    help="Override the robot USD before creating the environment (default: task asset)")
parser.add_argument("--lang_instruction_by_color", type=json.loads, help="JSON colour to instruction map")
parser.add_argument("--results_json", type=Path)
parser.add_argument("--resume", action="store_true", help="Continue completed attempts from --results_json (fixed benchmark only)")
parser.add_argument("--config_fingerprint", default=None, help=argparse.SUPPRESS)
parser.add_argument("--checkpoint", default=os.environ.get("PICK_PLACE_EVAL_CHECKPOINT", os.environ.get("MODEL", "unknown")))

from source.sim_to_real_so101.policy_client.grasp import GraspConfig, add_grasp_arguments
add_grasp_arguments(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.molmo_overhead_camera and (args_cli.policy != "molmoact2" or args_cli.task not in (
        "Lerobot-So101-Teleop-Pick-Place-Eval", "Lerobot-So101-Teleop-Pick-Place-DR-Eval")):
    parser.error("molmo_overhead_camera requires molmoact2 and a pick-place Eval task")
max_horizon = 30 if args_cli.policy == "molmoact2" else 16
if args_cli.action_horizon is None:
    args_cli.action_horizon = max_horizon
if not 1 <= args_cli.action_horizon <= max_horizon:
    parser.error(f"action_horizon must be in 1..{max_horizon} for {args_cli.policy}")
if args_cli.prefetch_steps >= args_cli.action_horizon * args_cli.action_repeat:
    parser.error("prefetch_steps must be less than action_horizon * action_repeat")
if args_cli.temporal_ensemble and args_cli.blend_steps:
    parser.error("temporal_ensemble and blend_steps are mutually exclusive")

# always enable cameras to record video
args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import torch

importlib.import_module("isaaclab_tasks")
from isaaclab_tasks.utils import parse_env_cfg

importlib.import_module("sim_to_real_so101.tasks")
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import (
    LeRobotSO101Interface,
    SimRemotePolicy,
)


def _evaluate():
    from time import monotonic
    from sim_to_real_so101.utils.eval_resume import load_episodes, read_report, write_report, next_video_attempt
    smoke_test = getattr(args_cli, "smoke_test", False)
    resume = getattr(args_cli, "resume", False)
    if resume and (smoke_test or not getattr(args_cli, "eval_set", None) or not args_cli.results_json):
        raise ValueError("--resume requires --eval_set and --results_json, and cannot be used with --smoke-test")
    if smoke_test:
        args_cli.num_episodes = 5
        args_cli.rerun = True
        args_cli.results_json = None
        args_cli.save_videos = False
        args_cli.video_dir = None
        args_cli.log_timing = False
        args_cli.timing_csv = ""
        args_cli.log_gripper = False
        args_cli.gripper_csv = ""
        print("[SMOKE TEST] Five episodes with Rerun; evaluation output is console-only.")
    max_horizon = 30 if args_cli.policy == "molmoact2" else 16
    if args_cli.action_horizon is None:
        args_cli.action_horizon = max_horizon
    if (isinstance(args_cli.action_horizon, bool) or not isinstance(args_cli.action_horizon, int)
            or not 1 <= args_cli.action_horizon <= max_horizon):
        raise ValueError(f"action_horizon must be in 1..{max_horizon} for {args_cli.policy}")
    action_repeat = getattr(args_cli, "action_repeat", 1)
    if isinstance(action_repeat, bool) or not isinstance(action_repeat, int) or action_repeat < 1:
        raise ValueError("action_repeat must be a positive integer")
    args_cli.action_repeat = action_repeat
    if getattr(args_cli, "prefetch_steps", 0) >= args_cli.action_horizon * action_repeat:
        raise ValueError("prefetch_steps must be less than action_horizon * action_repeat")
    render_warmup = getattr(args_cli, "render_warmup", 0)
    if render_warmup < 0:
        raise ValueError("render_warmup must be >= 0")
    if args_cli.episode_length_s is not None and (
            not math.isfinite(args_cli.episode_length_s) or args_cli.episode_length_s <= 0):
        raise ValueError("episode_length_s must be finite and greater than 0")
    benchmark = None
    if getattr(args_cli, 'eval_set', None):
        from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts, StageTrace, benchmark_report, save_heatmap, polygons_match
        from sim_to_real_so101.utils.pick_place_benchmark_runtime import install_stage_tracking, capture_scene
        benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed)
        if smoke_test and len(benchmark.schedule) < 5:
            # Repeat small selections so smoke mode still executes exactly five attempts.
            args_cli.repeats *= math.ceil(5 / len(benchmark.schedule))
            benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed)
        args_cli.num_episodes = 5 if smoke_test else len(benchmark.schedule)
        if args_cli.random_fraction != 0:
            raise ValueError('--eval_set cannot be combined with random_fraction')
    elif getattr(args_cli, 'splits', None) or getattr(args_cli, 'repeats', 1) != 1:
        raise ValueError('--splits and --repeats require --eval_set')
    saved_report = read_report(args_cli.results_json) if resume else None
    if saved_report and saved_report.get("save_videos"):
        args_cli.save_videos = True
        selected_directory = getattr(args_cli, "video_dir", None)
        if selected_directory is not None and str(selected_directory.resolve()) != saved_report.get("video_dir"):
            raise ValueError("Cannot resume with a different video_dir")
        args_cli.video_dir = Path(saved_report["video_dir"])
    if not resume and args_cli.results_json and args_cli.results_json.exists():
        raise ValueError("Results JSON already exists; use --resume or select a new results path")
    keyboard_control = KeyboardControl()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    if args_cli.robot_usd is not None:
        robot_usd = args_cli.robot_usd.resolve()
        if not robot_usd.is_file():
            raise ValueError(f"Robot USD does not exist: {robot_usd}")
        env_cfg.scene.robot.spawn.usd_path = str(robot_usd)
    print(f"[INFO]: Robot USD: {env_cfg.scene.robot.spawn.usd_path}")
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    episode_length_s = float(env_cfg.episode_length_s)
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and greater than 0")
    pick_place = args_cli.task in ("Lerobot-So101-Teleop-Pick-Place-Eval", "Lerobot-So101-Teleop-Pick-Place-DR-Eval")
    use_molmo_overhead = getattr(args_cli, "molmo_overhead_camera", False)
    if not isinstance(use_molmo_overhead, bool):
        raise ValueError("molmo_overhead_camera must be true or false")
    if use_molmo_overhead and (args_cli.policy != "molmoact2" or not pick_place):
        raise ValueError("molmo_overhead_camera requires molmoact2 and a pick-place Eval task")
    if args_cli.num_episodes < 1 or not 0 <= args_cli.random_fraction <= 1:
        raise ValueError("num_episodes must be positive and random_fraction must be in [0, 1]")
    if pick_place:
        if env_cfg.scene.num_envs != 1:
            raise ValueError("Pick-place policy evaluation requires --num_envs 1 (one remote policy state)")
        params = env_cfg.events.spawn_cube.params
        params.update(mode=args_cli.start_mode, random_fraction=args_cli.random_fraction, seed=args_cli.seed,
                      reset_robot=args_cli.robot_start == "recorded")
        if args_cli.cube_starts:
            params["starts_dir"] = args_cli.cube_starts
        if benchmark:
            params.update(eval_set=str(args_cli.eval_set), splits=args_cli.splits, repeats=args_cli.repeats,
                          start_offset=len(saved_report["episodes"]) if resume else 0)
    elif benchmark or args_cli.cube_starts or args_cli.lang_instruction_by_color is not None:
        raise ValueError("Cube starts and colour instructions require a pick-place Eval task")

    # Seed all RNGs for reproducible episode resets
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    molmo_overhead_metadata = None
    if use_molmo_overhead:
        from sim_to_real_so101.utils.molmo_camera import add_molmo_overhead_camera
        molmo_overhead_metadata = add_molmo_overhead_camera(env_cfg)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    step_dt = float(env.unwrapped.step_dt)
    replan_interval_s = step_dt * args_cli.action_horizon * action_repeat
    if action_repeat > 1 or args_cli.policy == "molmoact2":
        print(f"[INFO]: Replanning every {replan_interval_s:.2f}s sim "
              f"({args_cli.action_horizon} targets held for {action_repeat} control ticks each).", flush=True)
    if benchmark:
        install_stage_tracking(env.unwrapped)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Click 'R' to reset the world")

    # cameras
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print("[Info]: No cameras found - videos will not be recorded")

    # lerobot interface (provides sim↔real coordinate transforms)
    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    if use_molmo_overhead:
        rename_map = dict(rename_map or {}, realsense_rgb="original_room", molmo_room="room", wrist_cam="wrist")
    elif (args_cli.policy == "molmoact2" and rename_map and rename_map.get("molmo_room") == "room"
          and rename_map.get("realsense_rgb") == "original_room"):
        rename_map = dict(rename_map, realsense_rgb="room")
        rename_map.pop("molmo_room")
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=None,
        id="leader_arm_1",
        cameras=cameras,
        fps=30,
        kind="follower",
        rename_map=rename_map,
        joint_mapping=getattr(env.unwrapped.cfg, "sim_joint_mapping", None),
    )
    print(f"[INFO]: Initializing device with Rerun visualization: {args_cli.rerun}")
    robot_iface.init_device(visualize=args_cli.rerun)

    # remote policy plugin
    smoothing_defaults = dict(prefetch_steps=0, blend_steps=0, ema_alpha=1.0,
                              smooth_gripper=False, temporal_ensemble=0, te_decay=0.01,
                              log_timing=False, timing_csv="")
    smoothing_kwargs = {key: getattr(args_cli, key) for key, default in smoothing_defaults.items()
                        if getattr(args_cli, key, default) != default}
    grasp_config = GraspConfig(**{name: getattr(args_cli, name) for name in GraspConfig.__dataclass_fields__})
    grasp_kwargs = {}
    if grasp_config.enabled or args_cli.seed_per_episode:
        grasp_kwargs = dict(grasp_config=grasp_config, seed_per_episode=args_cli.seed_per_episode,
                            episode_seed=args_cli.seed, control_dt=step_dt)
    policy_options = {}
    for item in args_cli.policy_option:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --policy_option {item!r}: expected KEY=VALUE")
        policy_options[key] = value
    policy = SimRemotePolicy(
        robot_iface=robot_iface,
        policy_name=args_cli.policy,
        policy_options=policy_options,
        api_token=args_cli.policy_api_token,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        action_repeat=action_repeat,
        lang_instruction=args_cli.lang_instruction,
        **smoothing_kwargs,
        **grasp_kwargs,
    )
    if grasp_config.enabled:
        policy.grasp.clock = lambda: step * step_dt
    policy.connect()

    grasp_diagnostics = None
    if smoke_test and pick_place and args_cli.policy == "molmoact2":
        from sim_to_real_so101.utils.grasp_diagnostics import install_grasp_diagnostics
        grasp_diagnostics = install_grasp_diagnostics(env.unwrapped, robot_iface, step_dt)
        print("[INFO]: Smoke grasp diagnostics enabled (millimetres; in memory only).", flush=True)

    from sim_to_real_so101.utils.pick_place_eval import select_instruction, results_report

    def episode_metadata():
        raw = env.unwrapped
        index = getattr(raw, "_pick_place_start_index", [None])[0]
        color = getattr(raw, "_pick_place_cube_color", ["unknown"])[0]
        instruction = select_instruction(color, args_cli.lang_instruction, args_cli.lang_instruction_by_color)
        policy._lang_instruction = instruction
        extra = {}
        if benchmark:
            start = raw._benchmark_start
            extra = dict(start_id=start['id'], split=start['split'], repeat=start['repeat'],
                         policy_seed=start['policy_seed'], x=start['x'], y=start['y'], yaw=start['yaw'])
        return {**extra, "start_index": index,
                "start_kind": "random" if index == -1 else "recorded" if index is not None else "unknown",
                "zone": getattr(raw, "_pick_place_start_zone", ["unknown"])[0],
                "cube_color": color, "instruction": instruction}

    # ManagerBasedRLEnv auto-resets on termination. Snapshot metadata BEFORE
    # stepping, then use the returned reset observation for the next episode.
    episodes = []
    successes = 0
    pbar = None
    video_recorder = None
    video_attempt = 0
    video_dir = getattr(args_cli, "video_dir", None) if resume else None

    def progress_summary():
        completed = len(episodes)
        rate = f"{successes / completed:.1%}" if completed else "N/A"
        return f"success={rate} ({successes}/{completed})"

    def build_report():
        report_fn = results_report
        report_extra = {}
        if benchmark:
            report_fn = benchmark_report
            report_extra = dict(eval_set=benchmark.data, repeats=args_cli.repeats)
        report = report_fn(
            episodes, task=args_cli.task, checkpoint=args_cli.checkpoint, seed=args_cli.seed, **report_extra,
            random_fraction=args_cli.random_fraction, episode_length_s=episode_length_s,
            action_horizon=args_cli.action_horizon, step_dt=step_dt,
        )
        report['robot_start'] = args_cli.robot_start
        report['action_repeat'] = action_repeat
        report['replan_interval_s'] = replan_interval_s
        report['robot_usd'] = env_cfg.scene.robot.spawn.usd_path
        if molmo_overhead_metadata is not None:
            report['molmo_overhead_camera'] = molmo_overhead_metadata
        if video_recorder is not None or (resume and saved_report.get("save_videos")):
            report["save_videos"] = True
            report["video_dir"] = str(video_dir)
            if video_recorder is not None:
                report["video_encoder"] = video_recorder.encoder
            elif "video_encoder" in saved_report:
                report["video_encoder"] = saved_report["video_encoder"]
        if getattr(policy, "joint_transform", None) is not None:
            report["policy_joint_transform"] = policy.joint_transform
        if getattr(policy, "policy_cameras", None) is not None:
            report["policy_cameras"] = policy.policy_cameras
        if grasp_kwargs:
            from dataclasses import asdict
            report["gripper_control"] = asdict(grasp_config)
            if grasp_config.enabled:
                report["gripper_calibration"] = policy.grasp.calibration
            report["gripper_latch"] = grasp_config.gripper_latch
            report["seed_per_episode"] = args_cli.seed_per_episode
        if smoothing_kwargs:
            report['smoothing'] = {key: getattr(args_cli, key, default)
                                   for key, default in smoothing_defaults.items()}
        report['requested_episodes'] = args_cli.num_episodes
        report['complete'] = len(episodes) == args_cli.num_episodes
        report["policy"] = args_cli.policy
        fingerprint = getattr(args_cli, "config_fingerprint", None)
        if fingerprint is not None:
            report["config_fingerprint"] = fingerprint
        from dataclasses import asdict
        report["resume_settings"] = dict(
            policy=args_cli.policy, policy_options=policy_options,
            cube_starts=args_cli.cube_starts, rename_map=rename_map,
            lang=args_cli.lang_instruction, lang_by_color=args_cli.lang_instruction_by_color,
            render_warmup=render_warmup, molmo_overhead_camera=use_molmo_overhead,
            smoothing={key: getattr(args_cli, key, default) for key, default in smoothing_defaults.items()
                       if key not in ("log_timing", "timing_csv")},
            grasp={key: value for key, value in asdict(grasp_config).items()
                   if key not in ("log_gripper", "gripper_csv")},
            seed_per_episode=args_cli.seed_per_episode)
        return report

    if resume:
        # Validate before opening a recorder or replacing any existing report.
        expected = build_report()
        identity_keys = ("task", "checkpoint", "seed", "random_fraction", "episode_length_s",
                         "action_horizon", "action_repeat", "step_dt", "robot_start", "robot_usd",
                         "eval_set_id", "repeats", "requested_episodes", "policy_joint_transform",
                         "policy_cameras", "resume_settings", "config_fingerprint")
        try:
            saved_report = load_episodes(args_cli.results_json,
                                         {key: expected[key] for key in identity_keys if key in expected},
                                         benchmark.schedule)
        except BaseException:
            env.close()
            raise
        episodes = saved_report["episodes"].copy()
        successes = sum(row["success"] for row in episodes)
        print(f"[RESUME] Retained {len(episodes)}/{args_cli.num_episodes} completed episodes; "
              "continuing at the next unfinished attempt.", flush=True)

    try:
        if len(episodes) == args_cli.num_episodes:
            print("[RESUME] Evaluation is already complete; no attempts to run.", flush=True)
            return
        if getattr(args_cli, "save_videos", False):
            from sim_to_real_so101.utils.eval_video import EpisodeVideoRecorder
            video_dir = getattr(args_cli, "video_dir", None)
            if video_dir is None:
                if args_cli.results_json:
                    video_dir = args_cli.results_json.with_name(args_cli.results_json.stem + "_videos")
                else:
                    from datetime import datetime, timezone
                    video_dir = Path("outputs/eval_videos") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            video_dir = video_dir.expanduser().resolve()
            video_recorder = EpisodeVideoRecorder(video_dir, fps=1 / step_dt, rename_map=rename_map,
                                                 encoder=getattr(args_cli, "video_encoder", "libx264"))
            if resume:
                video_attempt = next_video_attempt(video_dir)
            print(f"[INFO]: Saving attempt videos to {video_dir} "
                  f"(background {video_recorder.encoder} encoding)", flush=True)
        # Persist an empty initial checkpoint too, so an interruption in episode 1 is resumable.
        if args_cli.results_json:
            write_report(args_cli.results_json, build_report())
        obs, _ = env.reset()
        metadata = episode_metadata()
        if benchmark:
            live = capture_scene(env.unwrapped)
            expected = benchmark.data['scene']
            # Float32 poses can change the number of nearly collinear hull
            # vertices. Compare boundaries while retaining the geometry guard.
            for key in ('box_footprint_base_xy', 'table_outline_base_xy'):
                if not polygons_match(live[key], expected[key], atol=1e-5):
                    raise ValueError(f'Live scene differs from eval set: {key}; recapture geometry')
            if (np.shape(live['cube_size_m']) != np.shape(expected['cube_size_m'])
                    or not np.allclose(live['cube_size_m'], expected['cube_size_m'], atol=1e-5, rtol=0)):
                raise ValueError('Live scene differs from eval set: cube_size_m; recapture geometry')
            policy.reset(seed=metadata['policy_seed'])
        else:
            policy.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        initial_action = torch.tensor(
            [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465],
            device=env.unwrapped.device,
        )

        def settling_pose(observation):
            if pick_place and args_cli.robot_start == "recorded":
                return observation["policy"]["joint_pos_obs"][0].clone()
            return initial_action

        # Hold the reset pose for all ten settling steps, rather than following
        # joint drift or moving a recorded start to the calibrated default pose.
        settling_action = settling_pose(obs)
        step = 0
        episode_wall_start = monotonic()
        render_every = env.unwrapped.cfg.sim.render_interval if render_warmup else None
        while simulation_app.is_running() and len(episodes) < args_cli.num_episodes:
            with torch.inference_mode():
                if step == 0:
                    if video_recorder is not None:
                        video_recorder.start_episode(video_attempt)
                        video_attempt += 1
                    pbar = tqdm(total=env.unwrapped.max_episode_length,
                                desc=f"Rollout (ep {len(episodes) + 1}/{args_cli.num_episodes})", unit="step")
                    pbar.set_postfix_str(progress_summary())
                if step < 10:
                    actions[:] = settling_action
                else:
                    joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                    actions[:] = policy.get_action(joint_positions, obs["visual"], log=args_cli.rerun)
                    if grasp_diagnostics is not None:
                        grasp_diagnostics.command = actions[0].clone()
                        grasp_diagnostics.step = step + 1
                if render_warmup:
                    # The first query uses the obs after settling step 9; later
                    # queries follow the policy's chunk/prefetch/ensemble clock.
                    steps_to_query = 9 - step if step < 10 else policy.steps_until_query
                    # Isaac Lab renders at physics substeps where counter % render_interval == 0; a huge
                    # interval skips this step's render (physics and termination checks are unaffected).
                    env.unwrapped.cfg.sim.render_interval = render_every if steps_to_query < render_warmup else 10**12
                if video_recorder is not None:
                    # The next observation may already belong to an automatic reset.
                    video_recorder.write(obs["visual"])
                obs, _, terminated, truncated, _ = env.step(actions)
                step += 1
                if grasp_config.enabled and benchmark and env.unwrapped._benchmark_trace.stages["grasped"]:
                    policy.grasp.mark_grasp(step * step_dt)
                pbar.update(1)
                is_terminated = bool(terminated.any().item())
                is_truncated = bool(truncated.any().item())
                if is_terminated or is_truncated:
                    row = {**metadata, "episode": len(episodes), "steps": step,
                           "success": is_terminated and not is_truncated,
                           "success_step": step if is_terminated and not is_truncated else None,
                           "inference_calls": getattr(policy, "inference_calls", None),
                           "wall_time_s": monotonic() - episode_wall_start}
                    # Both clocks start with the episode rollout, including the
                    # ten settling steps. Wall time also includes inference/rendering.
                    row["elapsed_sim_s"] = step * step_dt
                    row["time_to_success_s"] = row["elapsed_sim_s"] if row["success"] else None
                    row["wall_time_to_success_s"] = row["wall_time_s"] if row["success"] else None
                    if video_recorder is not None:
                        row["videos"] = video_recorder.finish_episode()
                        row["video_encoder"] = video_recorder.encoder
                    if benchmark:
                        row.update(env.unwrapped._benchmark_trace.finish(row['success']))
                        env.unwrapped._benchmark_trace = StageTrace()
                    if grasp_config.enabled:
                        row.update(policy.grasp.finish_episode(reason="success" if row["success"] else "timeout"))
                    if grasp_diagnostics is not None:
                        row['grasp_diagnostics'] = grasp_diagnostics.summary()
                        grasp_diagnostics.reset()
                    episodes.append(row)
                    successes += int(row['success'])
                    if args_cli.results_json:
                        write_report(args_cli.results_json, build_report())
                    pbar.set_postfix_str(progress_summary())
                    pbar.close()
                    pbar = None
                    start = "random" if row['start_index'] == -1 else row['start_index']
                    timing_label = "time_to_success" if row["success"] else "timeout_elapsed"
                    print(f"[EPISODE {len(episodes)}] start={start} zone={row['zone']} "
                          f"cube_color={row['cube_color']} {'success' if row['success'] else 'fail'} steps={step} "
                          f"{timing_label}={row['elapsed_sim_s']:.2f}s sim "
                          f"({row['wall_time_s']:.2f}s wall)", flush=True)
                    if grasp_diagnostics is not None:
                        print(f"[GRASP {len(episodes)}] " + json.dumps(row['grasp_diagnostics']), flush=True)
                    if len(episodes) == args_cli.num_episodes:
                        break
                    metadata = episode_metadata()
                    if benchmark:
                        policy.reset(seed=metadata['policy_seed'])
                    else:
                        policy.reset()
                    settling_action = settling_pose(obs)
                    step = 0
                    episode_wall_start = monotonic()
                elif keyboard_control.reset_world:
                    if benchmark:
                        raise RuntimeError('Manual reset interrupted fixed benchmark; saving incomplete report')
                    keyboard_control.reset_world = False
                    pbar.close()
                    pbar = None
                    print(f"[MANUAL RESET] Episode interrupted at step {step}; excluded from results")
                    if video_recorder is not None:
                        video_recorder.finish_episode()
                    if grasp_diagnostics is not None:
                        grasp_diagnostics.reset()
                    obs, _ = env.reset()
                    policy.reset()
                    metadata = episode_metadata()
                    settling_action = settling_pose(obs)
                    step = 0
                    episode_wall_start = monotonic()
        if len(episodes) != args_cli.num_episodes:
            raise RuntimeError(f"Evaluation stopped after {len(episodes)}/{args_cli.num_episodes} episodes")
    except Exception as error:
        # Kit cleanup can exit before Python displays an uncaught exception.
        # Emit the original failure before closing the environment or app.
        traceback.print_exc()
        sys.stderr.flush()
        error._so101_traceback_printed = True
        raise
    finally:
        if pbar is not None:
            pbar.close()
        report = build_report()
        overall = report['overall']
        rate = overall['success_rate']
        rate_text = f"{rate:.1%}" if rate is not None else 'n/a'
        status = 'complete' if report['complete'] else 'incomplete'
        print(f"[RESULT] {status}: {len(episodes)}/{args_cli.num_episodes} episodes; "
              f"success {overall['successes']}/{overall['episodes']} ({rate_text})")
        if overall.get("median_time_to_success_s") is not None:
            print(f"[RESULT] Median time to success: {overall['median_time_to_success_s']:.2f}s sim")
        for group_key, label in (('by_split', 'Split'), ('by_zone', 'Zone'), ('by_cube_color', 'Colour')):
            groups = report.get(group_key, {})
            if groups:
                values = ', '.join(f"{name} {stats['successes']}/{stats['episodes']}"
                                   for name, stats in groups.items())
                print(f"[RESULT] {label}: {values}")
        if args_cli.results_json:
            write_report(args_cli.results_json, report)
            if benchmark:
                save_heatmap(report, args_cli.results_json.with_suffix('.png'))
            print(f"Results JSON: {args_cli.results_json}")
        elif not smoke_test:
            print(json.dumps(report, indent=2, allow_nan=False))
        # Kit may exit without flushing Python's redirected stdout during shutdown.
        sys.stdout.flush()
        try:
            if video_recorder is not None:
                video_recorder.close()
            if smoothing_kwargs.get("log_timing") or grasp_config.enabled:
                policy.close_timing()
        finally:
            try:
                if args_cli.rerun:
                    import rerun as rr
                    recording = rr.get_global_data_recording()
                    if recording is not None:
                        recording.flush(timeout_sec=5)
                    rr.disconnect()
            finally:
                env.close()


def main():
    try:
        _evaluate()
    except Exception as error:
        # Setup failures happen before the rollout guard; avoid printing a
        # rollout failure twice when normal cleanup returns to Python.
        if not getattr(error, "_so101_traceback_printed", False):
            traceback.print_exc()
            sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
