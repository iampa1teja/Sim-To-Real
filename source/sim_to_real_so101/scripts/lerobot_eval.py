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
parser.add_argument("--policy_host", type=str, default="localhost", help="GR00T policy server host")
parser.add_argument("--policy_port", type=int, default=5555, help="GR00T policy server port")
parser.add_argument("--action_horizon", type=int, choices=range(1, 17), default=16,
                    help="Number of action steps per server query (1..16; SO-arm checkpoint chunk length is 16)")
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
parser.add_argument("--lang_instruction_by_color", type=json.loads, help="JSON colour to instruction map")
parser.add_argument("--results_json", type=Path)
parser.add_argument("--checkpoint", default=os.environ.get("PICK_PLACE_EVAL_CHECKPOINT", os.environ.get("MODEL", "unknown")))

from sim_to_real_so101.gr00t_client.grasp import GraspConfig, add_grasp_arguments
add_grasp_arguments(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.prefetch_steps >= args_cli.action_horizon:
    parser.error("prefetch_steps must be less than action_horizon")
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
    GR00TRemotePolicy,
)


def _evaluate():
    from time import monotonic
    if not 1 <= args_cli.action_horizon <= 16:
        raise ValueError("action_horizon must be in 1..16")
    render_warmup = getattr(args_cli, "render_warmup", 0)
    if render_warmup < 0:
        raise ValueError("render_warmup must be >= 0")
    if args_cli.episode_length_s is not None and (
            not math.isfinite(args_cli.episode_length_s) or args_cli.episode_length_s <= 0):
        raise ValueError("episode_length_s must be finite and greater than 0")
    benchmark = None
    if getattr(args_cli, 'eval_set', None):
        from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts, StageTrace, benchmark_report, save_heatmap
        from sim_to_real_so101.utils.pick_place_benchmark_runtime import install_stage_tracking, capture_scene
        benchmark = EvalSetStarts(args_cli.eval_set, args_cli.splits, args_cli.repeats, args_cli.seed)
        args_cli.num_episodes = len(benchmark.schedule)
        if args_cli.random_fraction != 0:
            raise ValueError('--eval_set cannot be combined with random_fraction')
    elif getattr(args_cli, 'splits', None) or getattr(args_cli, 'repeats', 1) != 1:
        raise ValueError('--splits and --repeats require --eval_set')
    keyboard_control = KeyboardControl()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    episode_length_s = float(env_cfg.episode_length_s)
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and greater than 0")
    pick_place = args_cli.task in ("Lerobot-So101-Teleop-Pick-Place-Eval", "Lerobot-So101-Teleop-Pick-Place-DR-Eval")
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
            params.update(eval_set=str(args_cli.eval_set), splits=args_cli.splits, repeats=args_cli.repeats)
    elif benchmark or args_cli.cube_starts or args_cli.lang_instruction_by_color is not None:
        raise ValueError("Cube starts and colour instructions require a pick-place Eval task")

    # Seed all RNGs for reproducible episode resets
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    step_dt = float(env.unwrapped.step_dt)
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

    # remote GR00T policy
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
    policy = GR00TRemotePolicy(
        robot_iface=robot_iface,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        lang_instruction=args_cli.lang_instruction,
        **smoothing_kwargs,
        **grasp_kwargs,
    )
    if grasp_config.enabled:
        policy.grasp.clock = lambda: step * step_dt
    policy.connect()

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
    pbar = None
    try:
        obs, _ = env.reset()
        metadata = episode_metadata()
        if benchmark:
            live = capture_scene(env.unwrapped)
            expected = benchmark.data['scene']
            # Float32 PhysX poses may differ at the micrometre level. A changed
            # scene must never silently reuse a benchmark's footprint mask.
            for key in ('box_footprint_base_xy', 'table_outline_base_xy', 'cube_size_m'):
                if np.shape(live[key]) != np.shape(expected[key]) or not np.allclose(live[key], expected[key], atol=1e-5, rtol=0):
                    raise ValueError(f'Live scene differs from eval set: {key}; recapture geometry')
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
                    pbar = tqdm(total=env.unwrapped.max_episode_length,
                                desc=f"Rollout (ep {len(episodes) + 1})", unit="step")
                if step < 10:
                    actions[:] = settling_action
                else:
                    joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                    actions[:] = policy.get_action(joint_positions, obs["visual"], log=args_cli.rerun)
                if render_warmup:
                    # The first query uses the obs after settling step 9; later
                    # queries follow the policy's chunk/prefetch/ensemble clock.
                    steps_to_query = 9 - step if step < 10 else policy.steps_until_query
                    # Isaac Lab renders at physics substeps where counter % render_interval == 0; a huge
                    # interval skips this step's render (physics and termination checks are unaffected).
                    env.unwrapped.cfg.sim.render_interval = render_every if steps_to_query < render_warmup else 10**12
                obs, _, terminated, truncated, _ = env.step(actions)
                step += 1
                if grasp_config.enabled and benchmark and env.unwrapped._benchmark_trace.stages["grasped"]:
                    policy.grasp.mark_grasp(step * step_dt)
                pbar.update(1)
                is_terminated = bool(terminated.any().item())
                is_truncated = bool(truncated.any().item())
                if is_terminated or is_truncated:
                    pbar.close()
                    pbar = None
                    row = {**metadata, "episode": len(episodes), "steps": step,
                           "success": is_terminated and not is_truncated,
                           "success_step": step if is_terminated and not is_truncated else None,
                           "inference_calls": getattr(policy, "inference_calls", None),
                           "wall_time_s": monotonic() - episode_wall_start}
                    if benchmark:
                        row.update(env.unwrapped._benchmark_trace.finish(row['success']))
                        env.unwrapped._benchmark_trace = StageTrace()
                    if grasp_config.enabled:
                        row.update(policy.grasp.finish_episode(reason="success" if row["success"] else "timeout"))
                    episodes.append(row)
                    start = "random" if row['start_index'] == -1 else row['start_index']
                    print(f"[EPISODE {len(episodes)}] start={start} zone={row['zone']} "
                          f"cube_color={row['cube_color']} {'success' if row['success'] else 'fail'} steps={step}", flush=True)
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
        print(json.dumps(report, indent=2))
        if args_cli.results_json:
            args_cli.results_json.parent.mkdir(parents=True, exist_ok=True)
            args_cli.results_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if benchmark:
                save_heatmap(report, args_cli.results_json.with_suffix('.png'))
            print(f"Results JSON: {args_cli.results_json}")
        try:
            if smoothing_kwargs.get("log_timing") or grasp_config.enabled:
                policy.close_timing()
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
