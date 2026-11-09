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
"""Evaluate a served policy in simulation (ported from v1 scripts/lerobot_eval.py).

    lerobot_eval --config_path configs/eval/pick_place.yaml [--flag value ...]

The env is S2R-<task>-<variant> (default variant: eval). Flags override the YAML.
"""
# Changes from v1:
# - --arm/--scene/--task/--variant pick the env through sim.envs (S2R_* exported before any core/sim import);
#   v1's --task gym id and --robot_usd are gone (the arm USD comes from arms/<arm>/arm.toml).
# - --config_path reads the configs/eval YAML directly (v1 docker/eval_config.py turned it into flags).
# - Joint units go through core.arms.mapping.JointMapping and sim.eval.sim_remote_policy.SimRemotePolicy
#   (v1 LeRobotSO101Interface is gone). Settling holds the reset pose from the first observation.
# - Benchmark, results and instruction helpers come from tasks/<task>/{benchmark,eval}.py.
import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

SERVER_ONLY_KEYS = {"models_dir", "server_image"}  # read by `s2r serve`, ignored here
_RENAMES = {
    "policy": {"name": "policy", "host": "policy_host", "port": "policy_port", "timeout_ms": "policy_timeout_ms",
               "api_token": "policy_api_token", "checkpoint": "checkpoint"},
    "eval": {"episodes": "num_episodes", "lang": "lang_instruction", "lang_by_color": "lang_instruction_by_color"},
    "run": {},
    "grasp": {},
    "smoke_test": {"enabled": "smoke_test", "selection": "smoke_selection",
                   "reach_timeout_s": "smoke_reach_timeout_s", "progress_timeout_s": "smoke_progress_timeout_s"},
}
_TOP_LEVEL = {"arm", "scene", "task", "variant"}


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


def load_yaml_config(path):
    """Flatten a configs/eval YAML into argparse destinations (see configs/eval/pick_place.yaml)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    out = {}
    for key, value in data.items():
        if key in _TOP_LEVEL:
            out[key] = value
            continue
        if key not in _RENAMES:
            raise ValueError(f"{path}: unknown section {key!r}; expected {sorted(_TOP_LEVEL | set(_RENAMES))}")
        if not isinstance(value, dict):
            raise ValueError(f"{path}: {key} must be a mapping")
        for name, item in value.items():
            if key == "policy" and name in SERVER_ONLY_KEYS:
                continue
            if key == "policy" and name == "options":
                out["policy_option"] = [f"{k}={v}" for k, v in (item or {}).items()]
            elif key == "eval" and name == "dataset":
                # Recorded starts and robot states live in <dataset>/pick_place_meta (v1 eval.sh).
                out["cube_starts"] = str(Path(item) / "pick_place_meta")
            elif key == "eval" and name == "rename_map":
                out["rename_map"] = json.dumps(item) if item else None
            elif key == "eval" and name == "splits":
                out["splits"] = ",".join(item) if isinstance(item, (list, tuple)) else item
            elif key == "run" and name == "gui":
                out["headless"] = not item
            else:
                out[_RENAMES[key].get(name, name)] = item
    for name in ("eval_set", "video_dir", "results_json"):
        if out.get(name) is not None:
            out[name] = Path(out[name])
    return out


def rendering_kit_args(existing: str = "") -> str:
    """Enable the installed material context-menu dependency before Kit starts.

    The material library initializes its menus asynchronously during startup.
    Enabling this extension after app creation can miss that initialization.
    """
    extension = "omni.kit.context_menu"
    words = existing.split()
    for index, word in enumerate(words):
        if word == f"--enable={extension}" or (
            word == "--enable" and words[index + 1:index + 2] == [extension]
        ):
            return existing
    return f"{existing} --enable {extension}".strip()


def _build_parser():
    parser = argparse.ArgumentParser(description="S2R policy evaluation in simulation (remote policy server).")
    parser.add_argument("--config_path", type=Path, default=None, help="configs/eval YAML; flags override it")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", "so101"))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "rpd_meeting_room"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", "pick_place_cube"))
    parser.add_argument("--variant", default="eval", help="Task variant (eval, eval_dr, ...)")
    parser.add_argument("--disable_fabric", action="store_true", default=False,
                        help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=1984, help="Environment seed")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--policy_timeout_ms", type=_positive_int, default=15000,
                        help="Policy request timeout; increase for CPU inference (default 15000 ms).")
    parser.add_argument("--rename_map", type=str, default=None,
                        help='JSON sim camera -> policy camera map, e.g. \'{"realsense_rgb": "room"}\'')
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
    parser.add_argument("--ema_alpha", type=_ema_alpha, default=1.0, help="Arm EMA coefficient in (0,1]; 1: off")
    parser.add_argument("--smooth_gripper", action="store_true", help="Also apply EMA to the gripper")
    parser.add_argument("--temporal_ensemble", type=_nonnegative_int, default=0,
                        help="Query every M steps and average overlapping predictions, favouring older chunks (0: off)")
    parser.add_argument("--te_decay", type=_nonnegative_float, default=0.01,
                        help="Temporal ensemble weights exp(-k*i), i=0 oldest")
    parser.add_argument("--log_timing", action="store_true", help="Write per-action CSV and print latency/jerk summary")
    parser.add_argument("--timing_csv", default="", help="Timing CSV path (default: outputs/timing/sim_<timestamp>.csv)")
    parser.add_argument("--episode_length_s", type=_positive_float, default=None,
                        help="Episode timeout in simulation seconds (default: task.toml [task.episode] eval_steps)")
    parser.add_argument("--lang_instruction", type=str, default=None,
                        help="Language instruction for the policy (default: task.toml instruction)")
    parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run five episodes with Rerun, without saving evaluation reports, videos or CSVs")
    parser.add_argument("--save_videos", action="store_true", help="Save room/wrist MP4 videos per attempt")
    parser.add_argument("--video_dir", type=Path, default=None, help="Video output directory (default: beside results JSON)")
    parser.add_argument("--video_encoder", choices=("libx264", "h264_nvenc"), default="libx264",
                        help="Recording encoder: CPU libx264 or NVIDIA NVENC (requires video driver support)")
    parser.add_argument("--render_warmup", type=int, default=0,
                        help="0 (default): render the cameras every control step. N>0: render only the N steps before "
                             "each policy query; physics is unchanged.")
    parser.add_argument("--eval_set", type=Path, help="Fixed benchmark JSON (overrides num_episodes)")
    parser.add_argument("--splits", help="Comma-separated splits, default all in the file")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--smoke-selection", choices=("first_five", "one_per_zone"), default="first_five")
    parser.add_argument("--smoke-reach-timeout-s", type=float, default=None)
    parser.add_argument("--smoke-progress-timeout-s", type=float, default=None)
    parser.add_argument("--cube_starts", help="Directory containing episode_*.json cube trajectories")
    parser.add_argument("--start_mode", choices=("cycle", "random"), default="cycle")
    parser.add_argument("--random_fraction", type=float, default=0.0)
    parser.add_argument("--robot_start", choices=("recorded", "default"), default="recorded",
                        help="recorded: arm starts at each episode's recorded first-frame state; "
                             "default: the scene's calibrated reset pose")
    parser.add_argument("--lang_instruction_by_color", type=json.loads, help="JSON colour to instruction map")
    parser.add_argument("--results_json", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Continue completed attempts from --results_json (fixed benchmark only)")
    parser.add_argument("--config_fingerprint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint",
                        default=os.environ.get("PICK_PLACE_EVAL_CHECKPOINT", os.environ.get("MODEL", "unknown")))
    return parser


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path", type=Path, default=None)
    for name in _TOP_LEVEL:
        pre.add_argument(f"--{name}", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = load_yaml_config(known.config_path) if known.config_path else {}

    # Arm/scene/task must be exported before anything imports core or sim (they read S2R_* at import).
    from sim.teleop.config import apply_selection

    parser = _build_parser()
    selection = SimpleNamespace(**{name: getattr(known, name) or defaults.get(name) or parser.get_default(name)
                                   for name in _TOP_LEVEL})
    if not selection.task:
        raise SystemExit("lerobot_eval needs a task (evaluation has no scene-only mode).")
    apply_selection(selection)

    from isaaclab.app import AppLauncher
    from core.policy_client.grasp import add_grasp_arguments

    add_grasp_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    dests = {action.dest for action in parser._actions}
    unknown = set(defaults) - dests
    if unknown:
        parser.error(f"{known.config_path}: unknown keys {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in _TOP_LEVEL:
        setattr(args, name, getattr(selection, name))
    # A YAML smoke_test section only applies when the run is a smoke test (v1 eval.sh --smoke-test).
    if not args.smoke_test:
        for name in ("smoke_selection", "smoke_reach_timeout_s", "smoke_progress_timeout_s"):
            if name in defaults and getattr(args, name) == defaults[name]:
                setattr(args, name, _build_parser().get_default(name))

    max_horizon = 30 if args.policy == "molmoact2" else 16
    if args.action_horizon is None:
        args.action_horizon = max_horizon
    if not 1 <= args.action_horizon <= max_horizon:
        parser.error(f"action_horizon must be in 1..{max_horizon} for {args.policy}")
    if args.prefetch_steps >= args.action_horizon * args.action_repeat:
        parser.error("prefetch_steps must be less than action_horizon * action_repeat")
    if args.temporal_ensemble and args.blend_steps:
        parser.error("temporal_ensemble and blend_steps are mutually exclusive")
    # Always enable cameras: the policy reads them and videos record them.
    args.enable_cameras = True
    args.kit_args = rendering_kit_args(args.kit_args)
    return args, AppLauncher


def _evaluate(args_cli, simulation_app):  # noqa: C901
    from time import monotonic

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from tqdm import tqdm

    import sim.envs
    from core.arms.mapping import JointMapping
    from core.policy_client.grasp import GraspConfig
    from core.results.eval_resume import load_episodes, next_video_attempt, read_report, write_report
    from core.tasks.loader import load_task_module
    from sim.eval.sim_remote_policy import SimRemotePolicy
    from sim.teleop.keyboard import KeyboardControl

    task = sim.envs.TASK
    env_id = task.gym_id(args_cli.variant)
    if args_cli.lang_instruction is None:
        args_cli.lang_instruction = task.instruction

    smoke_test = getattr(args_cli, "smoke_test", False)
    smoke_selection = getattr(args_cli, "smoke_selection", "first_five")
    smoke_reach = getattr(args_cli, "smoke_reach_timeout_s", None)
    smoke_progress = getattr(args_cli, "smoke_progress_timeout_s", None)
    smoke_budget = None
    if (smoke_reach is None) != (smoke_progress is None):
        raise ValueError("Set both smoke deadline durations")
    if not smoke_test and (smoke_selection != "first_five" or smoke_reach is not None):
        raise ValueError("Smoke selection and deadlines require --smoke-test")
    if smoke_reach is not None:
        from core.results.eval_smoke import MilestoneDeadline
        smoke_budget = MilestoneDeadline(smoke_reach, smoke_progress)
        if args_cli.policy != "molmoact2" or not getattr(args_cli, "eval_set", None):
            raise ValueError("Smoke progress deadlines require Molmo with a fixed benchmark")
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
        coverage = "One example per zone" if smoke_selection == "one_per_zone" else "Five episodes"
        print(f"[SMOKE TEST] {coverage} with Rerun; evaluation output is console-only.")
    action_repeat = args_cli.action_repeat
    render_warmup = getattr(args_cli, "render_warmup", 0)
    if render_warmup < 0:
        raise ValueError("render_warmup must be >= 0")

    benchmark = None
    benchmark_module = None
    if getattr(args_cli, "eval_set", None):
        benchmark_module = load_task_module("benchmark", task.name)
        from sim.eval.pick_place_benchmark_runtime import capture_scene, install_stage_tracking
        EvalSetStarts = benchmark_module.EvalSetStarts
        selection = "one_per_zone" if smoke_test and smoke_selection == "one_per_zone" else "all"
        benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed,
                                  selection=selection)
        if smoke_test and smoke_selection == "first_five" and len(benchmark.schedule) < 5:
            # Repeat small selections so smoke mode still executes exactly five attempts.
            args_cli.repeats *= math.ceil(5 / len(benchmark.schedule))
            benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed)
        args_cli.num_episodes = 5 if smoke_test and smoke_selection == "first_five" else len(benchmark.schedule)
        if args_cli.random_fraction != 0:
            raise ValueError("--eval_set cannot be combined with random_fraction")
    elif getattr(args_cli, "splits", None) or getattr(args_cli, "repeats", 1) != 1:
        raise ValueError("--splits and --repeats require --eval_set")
    elif smoke_selection == "one_per_zone":
        raise ValueError("One-per-zone smoke coverage requires --eval_set")
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

    env_cfg = parse_env_cfg(env_id, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    print(f"[INFO]: Env {env_id}; robot USD: {env_cfg.scene.robot.spawn.usd_path}")
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if smoke_budget is not None:
        env_cfg.episode_length_s = smoke_budget.maximum_duration_s
    episode_length_s = float(env_cfg.episode_length_s)
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and greater than 0")
    spawn_term = getattr(env_cfg.events, "spawn_cube", None)
    pick_place = getattr(getattr(spawn_term, "func", None), "__name__", "") == "reset_cube_from_recorded_starts"
    if args_cli.num_episodes < 1 or not 0 <= args_cli.random_fraction <= 1:
        raise ValueError("num_episodes must be positive and random_fraction must be in [0, 1]")
    if pick_place:
        if env_cfg.scene.num_envs != 1:
            raise ValueError("Pick-place policy evaluation requires --num_envs 1 (one remote policy state)")
        params = spawn_term.params
        params.update(mode=args_cli.start_mode, random_fraction=args_cli.random_fraction, seed=args_cli.seed,
                      reset_robot=args_cli.robot_start == "recorded")
        if args_cli.cube_starts:
            params["starts_dir"] = args_cli.cube_starts
        if benchmark:
            params.update(eval_set=str(args_cli.eval_set), splits=args_cli.splits, repeats=args_cli.repeats,
                          start_offset=len(saved_report["episodes"]) if resume else 0)
            if smoke_test and smoke_selection == "one_per_zone":
                params["selection"] = "one_per_zone"
    elif benchmark or args_cli.cube_starts or args_cli.lang_instruction_by_color is not None:
        raise ValueError(f"Cube starts and colour instructions need a recorded-starts variant; {env_id} has none")

    # Seed all RNGs for reproducible episode resets
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(env_id, cfg=env_cfg)
    step_dt = float(env.unwrapped.step_dt)
    replan_interval_s = step_dt * args_cli.action_horizon * action_repeat
    if action_repeat > 1 or args_cli.policy == "molmoact2":
        print(f"[INFO]: Replanning every {replan_interval_s:.2f}s sim "
              f"({args_cli.action_horizon} targets held for {action_repeat} control ticks each).", flush=True)
    if benchmark:
        install_stage_tracking(env.unwrapped)
    smoke_deadline = None
    if smoke_budget is not None:
        from core.results.eval_smoke import install_milestone_deadline
        smoke_deadline = install_milestone_deadline(env.unwrapped, smoke_reach, smoke_progress)
        print(f"[SMOKE TEST] {smoke_reach:g}s sim to reach; {smoke_progress:g}s per new milestone; "
              "repeated milestones do not extend the deadline.", flush=True)
    if smoke_test and smoke_selection == "one_per_zone":
        print("[SMOKE TEST] Fixed starts: " + json.dumps([
            dict(start_id=r["id"], zone=r["zone"], policy_seed=r["policy_seed"]) for r in benchmark.schedule]), flush=True)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Click 'R' to reset the world")

    cameras = []
    for key in env.unwrapped.scene.keys():
        if key.startswith("camera_"):
            cameras.append(key[len("camera_"):])
            print(f"[INFO]: Found Camera: {cameras[-1]}")
    if not cameras:
        print("[Info]: No cameras found - videos will not be recorded")

    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    # The env's mapping (real_setup.json sim_joint_mapping) converts sim radians <-> policy raw units.
    mapping = JointMapping.from_setup(SimpleNamespace(mapping=getattr(env.unwrapped.cfg, "sim_joint_mapping", None)),
                                      device=env.unwrapped.device)
    if args_cli.rerun:
        from sim.teleop.devices import init_rerun
        init_rerun("s2r_eval")
    print(f"[INFO]: Rerun visualization: {args_cli.rerun}")

    smoothing_defaults = dict(prefetch_steps=0, blend_steps=0, ema_alpha=1.0,
                              smooth_gripper=False, temporal_ensemble=0, te_decay=0.01,
                              log_timing=False, timing_csv="")
    smoothing_kwargs = {key: getattr(args_cli, key) for key, default in smoothing_defaults.items()
                        if getattr(args_cli, key, default) != default}
    grasp_config = GraspConfig(**{name: getattr(args_cli, name) for name in GraspConfig.__dataclass_fields__})
    if smoke_budget is not None and (grasp_config.gripper_latch or smoothing_kwargs):
        raise ValueError("Unassisted Molmo smoke trials require no gripper override or action smoothing")
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
        mapping, cameras, rename_map,
        policy_name=args_cli.policy,
        policy_options=policy_options,
        api_token=args_cli.policy_api_token,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_ms=args_cli.policy_timeout_ms,
        action_horizon=args_cli.action_horizon,
        action_repeat=action_repeat,
        lang_instruction=args_cli.lang_instruction,
        **smoothing_kwargs,
        **grasp_kwargs,
    )
    step = 0
    if grasp_config.enabled:
        policy.grasp.clock = lambda: step * step_dt
    policy.connect()

    grasp_diagnostics = None
    if smoke_test and pick_place and args_cli.policy == "molmoact2":
        from sim.eval.grasp_diagnostics import install_grasp_diagnostics
        grasp_diagnostics = install_grasp_diagnostics(env.unwrapped, mapping, step_dt)
        print("[INFO]: Smoke grasp diagnostics enabled (millimetres; in memory only).", flush=True)

    task_eval = load_task_module("eval", task.name)
    select_instruction, results_report = task_eval.select_instruction, task_eval.results_report

    def episode_metadata():
        raw = env.unwrapped
        index = getattr(raw, "_pick_place_start_index", [None])[0]
        color = getattr(raw, "_pick_place_cube_color", ["unknown"])[0]
        instruction = select_instruction(color, args_cli.lang_instruction, args_cli.lang_instruction_by_color)
        policy.lang_instruction = instruction
        extra = {}
        if benchmark:
            start = raw._benchmark_start
            extra = dict(start_id=start["id"], split=start["split"], repeat=start["repeat"],
                         policy_seed=start["policy_seed"], x=start["x"], y=start["y"], yaw=start["yaw"])
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

    def progress_summary(stage_label=None):
        completed = len(episodes)
        rate = f"{successes / completed:.1%}" if completed else "N/A"
        if benchmark and stage_label is None:
            stage_label = env.unwrapped._benchmark_trace.progress_label()
        return (f"{stage_label}, " if stage_label else "") + f"success={rate} ({successes}/{completed})"

    def build_report():
        report_fn = results_report
        report_extra = {}
        if benchmark:
            report_fn = benchmark_module.benchmark_report
            report_extra = dict(eval_set=benchmark.data, repeats=args_cli.repeats)
        report = report_fn(
            episodes, task=env_id, checkpoint=args_cli.checkpoint, seed=args_cli.seed, **report_extra,
            random_fraction=args_cli.random_fraction, episode_length_s=episode_length_s,
            action_horizon=args_cli.action_horizon, step_dt=step_dt,
        )
        report["robot_start"] = args_cli.robot_start
        report["action_repeat"] = action_repeat
        report["replan_interval_s"] = replan_interval_s
        report["robot_usd"] = env_cfg.scene.robot.spawn.usd_path
        report["selection"] = dict(arm=args_cli.arm, scene=args_cli.scene, task=args_cli.task, variant=args_cli.variant)
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
            report["smoothing"] = {key: getattr(args_cli, key, default) for key, default in smoothing_defaults.items()}
        report["requested_episodes"] = args_cli.num_episodes
        if smoke_test:
            report["smoke_test"] = dict(selection=smoke_selection, reach_timeout_s=smoke_reach,
                                        progress_timeout_s=smoke_progress)
        report["complete"] = len(episodes) == args_cli.num_episodes
        report["policy"] = args_cli.policy
        fingerprint = getattr(args_cli, "config_fingerprint", None)
        if fingerprint is not None:
            report["config_fingerprint"] = fingerprint
        from dataclasses import asdict
        report["resume_settings"] = dict(
            policy=args_cli.policy, policy_options=policy_options,
            cube_starts=args_cli.cube_starts, rename_map=rename_map,
            lang=args_cli.lang_instruction, lang_by_color=args_cli.lang_instruction_by_color,
            render_warmup=render_warmup,
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
            from sim.eval.eval_video import EpisodeVideoRecorder
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
            expected = benchmark.data["scene"]
            # Float32 poses can change the number of nearly collinear hull
            # vertices. Compare boundaries while retaining the geometry guard.
            for key in ("box_footprint_base_xy", "table_outline_base_xy"):
                if not benchmark_module.polygons_match(live[key], expected[key], atol=1e-5):
                    raise ValueError(f"Live scene differs from eval set: {key}; recapture geometry")
            if (np.shape(live["cube_size_m"]) != np.shape(expected["cube_size_m"])
                    or not np.allclose(live["cube_size_m"], expected["cube_size_m"], atol=1e-5, rtol=0)):
                raise ValueError("Live scene differs from eval set: cube_size_m; recapture geometry")
            policy.reset(seed=metadata["policy_seed"])
        else:
            policy.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        def settling_pose(observation):
            # Hold the reset pose (recorded start or the arm's calibrated default) for the ten settling
            # steps, rather than following joint drift. v1 hard-coded SO-101 radians for the default start.
            return observation["policy"]["joint_pos_obs"][0].clone()

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
                    progress_text = progress_summary()
                    pbar.set_postfix_str(progress_text)
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
                    terminal_stage = env.unwrapped._benchmark_trace.progress_label() if benchmark else None
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
                        row.update(env.unwrapped._benchmark_trace.finish(row["success"]))
                        env.unwrapped._benchmark_trace = benchmark_module.StageTrace()
                    if grasp_config.enabled:
                        row.update(policy.grasp.finish_episode(reason="success" if row["success"] else "timeout"))
                    if grasp_diagnostics is not None:
                        row["grasp_diagnostics"] = grasp_diagnostics.summary()
                        grasp_diagnostics.reset()
                    if smoke_deadline is not None:
                        row["smoke_deadline"] = smoke_deadline.summary()
                    episodes.append(row)
                    successes += int(row["success"])
                    if args_cli.results_json:
                        write_report(args_cli.results_json, build_report())
                    pbar.set_postfix_str(progress_summary(terminal_stage))
                    pbar.close()
                    pbar = None
                    start = "random" if row["start_index"] == -1 else row["start_index"]
                    timing_label = "time_to_success" if row["success"] else "timeout_elapsed"
                    final_status = f" {terminal_stage}" if terminal_stage else ""
                    if row.get("failure_mode"):
                        final_status += f" failure={row['failure_mode']}"
                    if row.get("smoke_deadline", {}).get("stop_reason"):
                        final_status += " stop=" + row["smoke_deadline"]["stop_reason"]
                    print(f"[EPISODE {len(episodes)}] start={start} zone={row['zone']} "
                          f"cube_color={row['cube_color']} {'success' if row['success'] else 'fail'} steps={step} "
                          f"{timing_label}={row['elapsed_sim_s']:.2f}s sim "
                          f"({row['wall_time_s']:.2f}s wall){final_status}", flush=True)
                    if grasp_diagnostics is not None:
                        print(f"[GRASP {len(episodes)}] " + json.dumps(row["grasp_diagnostics"]), flush=True)
                    if len(episodes) == args_cli.num_episodes:
                        break
                    metadata = episode_metadata()
                    if smoke_deadline is not None:
                        smoke_deadline.reset()
                    if benchmark:
                        policy.reset(seed=metadata["policy_seed"])
                    else:
                        policy.reset()
                    settling_action = settling_pose(obs)
                    step = 0
                    episode_wall_start = monotonic()
                elif keyboard_control.reset_world:
                    if benchmark:
                        raise RuntimeError("Manual reset interrupted fixed benchmark; saving incomplete report")
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
                else:
                    latest_progress = progress_summary()
                    if latest_progress != progress_text:
                        pbar.set_postfix_str(latest_progress, refresh=False)
                        progress_text = latest_progress
        if len(episodes) != args_cli.num_episodes:
            raise RuntimeError(f"Evaluation stopped after {len(episodes)}/{args_cli.num_episodes} episodes")
    except Exception as error:
        # Kit cleanup can exit before Python displays an uncaught exception.
        # Emit the original failure before closing the environment or app.
        traceback.print_exc()
        sys.stderr.flush()
        error._s2r_traceback_printed = True
        raise
    finally:
        if pbar is not None:
            pbar.close()
        report = build_report()
        overall = report["overall"]
        rate = overall["success_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        status = "complete" if report["complete"] else "incomplete"
        print(f"[RESULT] {status}: {len(episodes)}/{args_cli.num_episodes} episodes; "
              f"success {overall['successes']}/{overall['episodes']} ({rate_text})")
        if overall.get("median_time_to_success_s") is not None:
            print(f"[RESULT] Median time to success: {overall['median_time_to_success_s']:.2f}s sim")
        for group_key, label in (("by_split", "Split"), ("by_zone", "Zone"), ("by_cube_color", "Colour")):
            groups = report.get(group_key, {})
            if groups:
                values = ", ".join(f"{name} {stats['successes']}/{stats['episodes']}" for name, stats in groups.items())
                print(f"[RESULT] {label}: {values}")
        if args_cli.results_json:
            write_report(args_cli.results_json, report)
            if benchmark:
                benchmark_module.save_heatmap(report, args_cli.results_json.with_suffix(".png"))
            print(f"Results JSON: {args_cli.results_json}")
        elif smoke_test and (smoke_selection == "one_per_zone" or smoke_budget is not None):
            # Machine-readable console output lets research runners compare
            # candidates without creating smoke report/video/log files.
            smoke_result = {key: report.get(key) for key in (
                "complete", "requested_episodes", "checkpoint", "seed", "step_dt", "episode_length_s", "action_horizon",
                "action_repeat", "policy_cameras", "policy_joint_transform", "resume_settings", "smoke_test", "overall")}
            smoke_result["episodes"] = [{key: row.get(key) for key in (
                "start_id", "zone", "policy_seed", "success", "elapsed_sim_s", "wall_time_s",
                "failure_mode", "stages", "smoke_deadline")} for row in episodes]
            for item, row in zip(smoke_result["episodes"], episodes):
                diagnostics = row.get("grasp_diagnostics") or {}
                item["grasp_diagnostics"] = diagnostics
                item["closest_tool_error_mm"] = (diagnostics.get("closest") or {}).get("tool_error_mm")
                item["closest_closing_tool_error_mm"] = (diagnostics.get("closest_closing") or {}).get("tool_error_mm")
            print("[SMOKE_RESULT] " + json.dumps(smoke_result, separators=(",", ":"), allow_nan=False), flush=True)
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


def main(argv=None):
    args_cli, AppLauncher = parse_args(argv)
    simulation_app = AppLauncher(args_cli).app
    try:
        _evaluate(args_cli, simulation_app)
    except Exception as error:
        # Setup failures happen before the rollout guard; avoid printing a
        # rollout failure twice when normal cleanup returns to Python.
        if not getattr(error, "_s2r_traceback_printed", False):
            traceback.print_exc()
            sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
