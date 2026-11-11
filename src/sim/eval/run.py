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
"""Evaluate a served policy in simulation.

    s2r eval --config_path configs/eval/pick_place.yaml [--flag value ...]

The env is S2R-<task>-<variant> (default variant: eval). Flags override the YAML.
"""

# - --arm/--scene/--task/--variant pick the env through sim.envs. S2R_* are exported before importing any
#   core/sim module that reads them at import. The arm USD comes from arms/<arm>/arm.toml.
# - --config_path reads the configs/eval YAML directly.
# - Joint units go through core.arms.mapping.JointMapping and sim.eval.sim_remote_policy.SimRemotePolicy.
#   Settling holds the reset pose from the first observation.
# - Task plugins supply signals/stages; benchmark and dataset handling are shared core modules.
import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from sim.app import apply_selection, add_launch_arguments, launch

SERVER_ONLY_KEYS = {"models_dir", "server_image"}  # read by `s2r serve`, ignored here
_RENAMES = {
    "policy": {
        "name": "policy",
        "host": "policy_host",
        "port": "policy_port",
        "timeout_ms": "policy_timeout_ms",
        "api_token": "policy_api_token",
        "checkpoint": "checkpoint",
    },
    "eval": {"episodes": "num_episodes", "lang": "lang_instruction", "lang_by_color": "lang_instruction_by_color"},
    "run": {},
    "grasp": {},
    "smoke_test": {
        "enabled": "smoke_test",
        "selection": "smoke_selection",
        "reach_timeout_s": "smoke_reach_timeout_s",
        "progress_timeout_s": "smoke_progress_timeout_s",
    },
}
_TOP_LEVEL = {"arm", "scene", "task", "variant", "unit"}


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
                # The generic reader accepts both current and legacy dataset sidecars.
                out["cube_starts"] = str(Path(item))
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


def _build_parser():
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    parser = argparse.ArgumentParser(description="S2R policy evaluation in simulation (remote policy server).")
    parser.add_argument("--config_path", type=Path, default=None, help="configs/eval YAML; flags override it")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
    parser.add_argument("--unit", default=os.environ.get("S2R_UNIT"), help="arm calibration unit ID")
    parser.add_argument("--variant", default="eval", help="Task variant (eval, eval_dr, ...)")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=1984, help="Environment seed")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument(
        "--policy_timeout_ms",
        type=_positive_int,
        default=15000,
        help="Policy request timeout; increase for CPU inference (default 15000 ms).",
    )
    parser.add_argument(
        "--rename_map",
        type=str,
        default=None,
        help='JSON sim camera -> policy camera map, e.g. \'{"realsense_rgb": "room"}\'',
    )
    parser.add_argument("--policy", default="groot_n17", help="Policy plugin name (a folder under policies/)")
    parser.add_argument(
        "--policy_option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Policy-specific client option; repeatable",
    )
    parser.add_argument("--policy_host", "--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument(
        "--policy_port",
        "--port",
        type=int,
        default=None,
        help="Policy server port (default: the plugin manifest's default_port)",
    )
    parser.add_argument("--policy_api_token", default=os.environ.get("POLICY_API_TOKEN"), help="Server API token")
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=None,
        help="Action steps per query: MolmoAct2 1..30 (default 30), other policies 1..16 (default 16)",
    )
    parser.add_argument(
        "--action_repeat",
        type=_positive_int,
        default=1,
        help="Control ticks to hold each predicted joint target (default 1)",
    )
    parser.add_argument(
        "--prefetch_steps",
        type=_nonnegative_int,
        default=0,
        help="Query P steps early; sim delivers P steps later and drops the first P actions (0: off)",
    )
    parser.add_argument(
        "--blend_steps",
        type=_nonnegative_int,
        default=0,
        help="Cross-fade old unexecuted predictions into new chunks over K steps (0: off)",
    )
    parser.add_argument("--ema_alpha", type=_ema_alpha, default=1.0, help="Arm EMA coefficient in (0,1]; 1: off")
    parser.add_argument("--smooth_gripper", action="store_true", help="Also apply EMA to the gripper")
    parser.add_argument(
        "--temporal_ensemble",
        type=_nonnegative_int,
        default=0,
        help="Query every M steps and average overlapping predictions, favouring older chunks (0: off)",
    )
    parser.add_argument(
        "--te_decay", type=_nonnegative_float, default=0.01, help="Temporal ensemble weights exp(-k*i), i=0 oldest"
    )
    parser.add_argument("--log_timing", action="store_true", help="Write per-action CSV and print latency/jerk summary")
    parser.add_argument(
        "--timing_csv", default="", help="Timing CSV path (default: outputs/timing/sim_<timestamp>.csv)"
    )
    parser.add_argument(
        "--episode_length_s",
        type=_positive_float,
        default=None,
        help="Episode timeout in simulation seconds (default: task.toml [task.episode] eval_steps)",
    )
    parser.add_argument(
        "--lang_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy (default: task.toml instruction)",
    )
    parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run five episodes with Rerun, without saving evaluation reports, videos or CSVs",
    )
    parser.add_argument("--save_videos", action="store_true", help="Save room/wrist MP4 videos per attempt")
    parser.add_argument(
        "--video_dir", type=Path, default=None, help="Video output directory (default: beside results JSON)"
    )
    parser.add_argument(
        "--video_encoder",
        choices=("libx264", "h264_nvenc"),
        default="libx264",
        help="Recording encoder: CPU libx264 or NVIDIA NVENC (requires video driver support)",
    )
    parser.add_argument(
        "--render_warmup",
        type=int,
        default=0,
        help="0 (default): render the cameras every control step. N>0: render only the N steps before "
        "each policy query; physics is unchanged.",
    )
    parser.add_argument("--eval_set", type=Path, help="Fixed benchmark JSON (overrides num_episodes)")
    parser.add_argument("--splits", help="Comma-separated splits, default all in the file")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--smoke-selection", choices=("first_five", "one_per_zone"), default="first_five")
    parser.add_argument("--smoke-reach-timeout-s", type=float, default=None)
    parser.add_argument("--smoke-progress-timeout-s", type=float, default=None)
    parser.add_argument("--cube_starts", help="Directory containing episode_*.json cube trajectories")
    parser.add_argument("--start_mode", choices=("cycle", "random"), default="cycle")
    parser.add_argument("--random_fraction", type=float, default=0.0)
    parser.add_argument(
        "--robot_start",
        choices=("recorded", "default"),
        default="recorded",
        help="recorded: arm starts at each episode's recorded first-frame state; "
        "default: the scene's calibrated reset pose",
    )
    parser.add_argument("--lang_instruction_by_color", type=json.loads, help="JSON colour to instruction map")
    parser.add_argument("--results_json", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Continue completed attempts from --results_json (fixed benchmark only)"
    )
    parser.add_argument("--config_fingerprint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=os.environ.get("S2R_EVAL_CHECKPOINT", "unknown"))
    return parser


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path", type=Path, default=None)
    for name in _TOP_LEVEL:
        pre.add_argument(f"--{name}", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = load_yaml_config(known.config_path) if known.config_path else {}

    # Arm/scene/task must be exported before importing any core or sim module that reads S2R_* at import.

    parser = _build_parser()
    selection = SimpleNamespace(
        **{name: getattr(known, name) or defaults.get(name) or parser.get_default(name) for name in _TOP_LEVEL}
    )
    if not selection.task:
        raise SystemExit("s2r eval needs a task (evaluation has no scene-only mode).")
    apply_selection(selection)

    from core.policy_client.grasp import add_grasp_arguments

    add_grasp_arguments(parser)
    add_launch_arguments(parser)
    dests = {action.dest for action in parser._actions}
    # Lab 3 no longer registers a --headless flag, but its launcher still accepts
    # the option in a Namespace. run.gui remains part of the evaluation YAML.
    unknown = set(defaults) - dests - {"headless"}
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

    from core.policy_client.discovery import get_manifest

    if args.action_horizon is None:
        args.action_horizon = get_manifest(args.policy).default_horizon
    if args.action_horizon < 1:
        parser.error("action_horizon must be positive")
    if args.prefetch_steps >= args.action_horizon * args.action_repeat:
        parser.error("prefetch_steps must be less than action_horizon * action_repeat")
    if args.temporal_ensemble and args.blend_steps:
        parser.error("temporal_ensemble and blend_steps are mutually exclusive")
    # Always enable cameras: the policy reads them and videos record them.
    args.enable_cameras = True
    return args


def setup(args_cli, simulation_app, resources):
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    import sim.envs
    from core.arms.mapping import JointMapping
    from core.arms.spec import load_arm_spec
    from core.arms.calibration import resolve_calibration
    from core.policy_client.grasp import GraspConfig
    from core.benchmark.resume import read_report
    from core.benchmark.eval_set import EvalSetStarts
    from sim.eval.stage_tracking import install_stage_tracking
    from sim.eval.sim_remote_policy import SimRemotePolicy
    from sim.teleop.controls.keyboard import KeyboardControl

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
        from core.benchmark.smoke import MilestoneDeadline

        smoke_budget = MilestoneDeadline(smoke_reach, smoke_progress, milestones=task.stages[:-1])
        if not getattr(args_cli, "eval_set", None):
            raise ValueError("Smoke progress deadlines require a fixed benchmark")
    resume = getattr(args_cli, "resume", False)
    if resume and (smoke_test or not getattr(args_cli, "eval_set", None) or (not args_cli.results_json)):
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
    if getattr(args_cli, "eval_set", None):
        selection = "one_per_zone" if smoke_test and smoke_selection == "one_per_zone" else "all"
        benchmark = EvalSetStarts(
            args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed, selection=selection, task=task
        )
        if smoke_test and smoke_selection == "first_five" and (len(benchmark.schedule) < 5):
            args_cli.repeats *= math.ceil(5 / len(benchmark.schedule))
            benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed, task=task)
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
    env_cfg = parse_env_cfg(
        env_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    arm, _ = load_arm_spec(args_cli.arm)
    calibration = resolve_calibration(arm, unit=args_cli.unit)
    env_cfg.sim_joint_mapping = dict(calibration.mapping)
    print(f"[INFO]: Env {env_id}; robot USD: {env_cfg.scene.robot.spawn.usd_path}")
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if smoke_budget is not None:
        env_cfg.episode_length_s = smoke_budget.maximum_duration_s
    episode_length_s = float(env_cfg.episode_length_s)
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and greater than 0")
    from sim.envs.mdp.starts import reset_from_starts

    spawn_term = getattr(env_cfg.events, "reset_from_starts", None)
    starts_enabled = spawn_term is not None and spawn_term.func is reset_from_starts
    if args_cli.num_episodes < 1 or not 0 <= args_cli.random_fraction <= 1:
        raise ValueError("num_episodes must be positive and random_fraction must be in [0, 1]")
    if env_cfg.scene.num_envs != 1:
        raise ValueError("Remote policy evaluation requires --num_envs 1 (one policy state)")
    if starts_enabled:
        params = spawn_term.params
        params.update(
            mode=args_cli.start_mode,
            random_fraction=args_cli.random_fraction,
            seed=args_cli.seed,
            reset_robot=args_cli.robot_start == "recorded",
        )
        if args_cli.cube_starts:
            params["starts_dir"] = args_cli.cube_starts
        if benchmark:
            params.update(
                eval_set=str(args_cli.eval_set),
                splits=args_cli.splits,
                repeats=args_cli.repeats,
                start_offset=len(saved_report["episodes"]) if resume else 0,
            )
            if smoke_test and smoke_selection == "one_per_zone":
                params["benchmark_selection"] = "one_per_zone"
    elif benchmark or args_cli.cube_starts or args_cli.lang_instruction_by_color is not None:
        raise ValueError(f"Recorded starts need an evaluation variant with reset_from_starts; {env_id} has none")
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env = gym.make(env_id, cfg=env_cfg)
    resources.callback(env.close)
    step_dt = float(env.unwrapped.step_dt)
    replan_interval_s = step_dt * args_cli.action_horizon * action_repeat
    if action_repeat > 1:
        print(
            f"[INFO]: Replanning every {replan_interval_s:.2f}s sim ({args_cli.action_horizon} targets held for {action_repeat} control ticks each).",
            flush=True,
        )
    install_stage_tracking(env.unwrapped, task)
    smoke_deadline = None
    if smoke_budget is not None:
        from core.benchmark.smoke import install_milestone_deadline

        smoke_deadline = install_milestone_deadline(
            env.unwrapped, smoke_reach, smoke_progress, milestones=task.stages[:-1]
        )
        print(
            f"[SMOKE TEST] {smoke_reach:g}s sim to reach; {smoke_progress:g}s per new milestone; repeated milestones do not extend the deadline.",
            flush=True,
        )
    if smoke_test and smoke_selection == "one_per_zone":
        print(
            "[SMOKE TEST] Fixed starts: "
            + json.dumps(
                [dict(start_id=r["id"], zone=r["zone"], policy_seed=r["policy_seed"]) for r in benchmark.schedule]
            ),
            flush=True,
        )
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Click 'R' to reset the world")
    cameras = []
    for key in env.unwrapped.scene.keys():
        if key.startswith("camera_"):
            cameras.append(key[len("camera_") :])
            print(f"[INFO]: Found Camera: {cameras[-1]}")
    if not cameras:
        print("[Info]: No cameras found - videos will not be recorded")
    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    mapping = JointMapping.from_arm(arm, calibration, device=env.unwrapped.device)
    if args_cli.rerun:
        from sim.teleop.devices import init_rerun

        init_rerun("s2r_eval")
    print(f"[INFO]: Rerun visualization: {args_cli.rerun}")
    smoothing_defaults = dict(
        prefetch_steps=0,
        blend_steps=0,
        ema_alpha=1.0,
        smooth_gripper=False,
        temporal_ensemble=0,
        te_decay=0.01,
        log_timing=False,
        timing_csv="",
    )
    smoothing_kwargs = {
        key: getattr(args_cli, key)
        for key, default in smoothing_defaults.items()
        if getattr(args_cli, key, default) != default
    }
    grasp_config = GraspConfig(**{name: getattr(args_cli, name) for name in GraspConfig.__dataclass_fields__})
    if smoke_budget is not None and (grasp_config.gripper_latch or smoothing_kwargs):
        raise ValueError("Unassisted smoke trials require no gripper override or action smoothing")
    grasp_kwargs = {}
    if grasp_config.enabled or args_cli.seed_per_episode:
        grasp_kwargs = dict(
            grasp_config=grasp_config,
            seed_per_episode=args_cli.seed_per_episode,
            episode_seed=args_cli.seed,
            control_dt=step_dt,
        )
    policy_options = {}
    for item in args_cli.policy_option:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --policy_option {item!r}: expected KEY=VALUE")
        policy_options[key] = value
    policy = SimRemotePolicy(
        mapping,
        cameras,
        rename_map,
        arm_spec=arm,
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
    policy.connect()
    from sim.eval.loop import EvaluationContext

    return EvaluationContext(
        args_cli=args_cli,
        simulation_app=simulation_app,
        task=task,
        env_id=env_id,
        env_cfg=env_cfg,
        env=env,
        benchmark=benchmark,
        resume=resume,
        saved_report=saved_report,
        smoke_test=smoke_test,
        smoke_selection=smoke_selection,
        smoke_reach=smoke_reach,
        smoke_progress=smoke_progress,
        smoke_budget=smoke_budget,
        smoke_deadline=smoke_deadline,
        render_warmup=render_warmup,
        episode_length_s=episode_length_s,
        step_dt=step_dt,
        replan_interval_s=replan_interval_s,
        keyboard_control=keyboard_control,
        policy=policy,
        grasp_config=grasp_config,
        grasp_kwargs=grasp_kwargs,
        smoothing_defaults=smoothing_defaults,
        smoothing_kwargs=smoothing_kwargs,
        policy_options=policy_options,
        rename_map=rename_map,
    )


def main(argv=None):
    from contextlib import ExitStack

    args = parse_args(argv)
    app = launch(args, selection=args, enable_cameras=True)
    exit_code = 0
    try:
        from sim.eval.loop import run_loop

        with ExitStack() as resources:
            run_loop(setup(args, app, resources))
    except Exception as error:
        exit_code = 1
        if not getattr(error, "_s2r_traceback_printed", False):
            traceback.print_exc()
            sys.stderr.flush()
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
