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
"""Paired sim/real evaluation page: spawn one start in Isaac, place the real cube to match, run the policy in sim
and then on the real arm, and label the stage the real arm reached.

Run inside the teleop container after a policy server is up (`./so101 serve ...` or
`./docker/eval.sh <config>.yaml --server_only`):

    sim_real_eval --eval_set eval_sets/pick_place_v1_seed1984.json --policy groot_n17 \
        --lang_instruction "Pick up the blue cube and place it in the white box"

Open http://localhost:8766. The real arm is never commanded without --enable-motion. Sim and real rollouts run one
after the other on this thread and share one policy server, so only one policy instance holds GPU memory.
"""
import hashlib
import importlib
import json
import math
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

from sim_to_real_so101.scripts import real_rollout as real_cli
from sim_to_real_so101.utils import real_rollout

parser = real_cli.build_parser()
parser.description = __doc__
parser.add_argument("--task", default="Lerobot-So101-Teleop-Pick-Place-Eval", help="Pick-place Eval task to spawn in sim.")
parser.add_argument("--eval_set", type=Path, required=True, help="eval set JSON, e.g. eval_sets/*.json")
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--sim_rename_map", default=None,
                    help="JSON sim camera name -> policy camera name (same meaning as lerobot_eval --rename_map)")
parser.add_argument("--robot_usd", type=Path, default=None)
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="sim episode length in seconds; default from the env cfg")
parser.add_argument("--policy_api_token", default=os.environ.get("POLICY_API_TOKEN"))
parser.add_argument("--gui_host", default="127.0.0.1")
parser.add_argument("--gui_port", type=int, default=8766)
parser.add_argument("--resume", type=Path, default=None, help="an existing output folder to continue")
parser.add_argument("--no_real", action="store_true", help="sim only; no arm connected")
parser.add_argument("--num_envs", type=int, default=None, help="must be 1 or unset (one remote policy state)")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True


def parse_policy_options(args):
    options = {}
    for item in args.policy_option:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --policy_option {item!r}: expected KEY=VALUE")
        options[key] = value
    return options


def check_args(args):
    """Argument errors found before touching any hardware or launching Isaac."""
    errors = [] if args.no_real else real_cli.validate(args)
    if not args.eval_set.is_file():
        errors.append(f"--eval_set not found: {args.eval_set}")
    if args.num_envs not in (None, 1):
        errors.append("--num_envs must be 1 (one remote policy state)")
    if args.resume is not None and not args.resume.is_dir():
        errors.append(f"--resume folder does not exist: {args.resume}")
    if args.episode_length_s is not None and not (math.isfinite(args.episode_length_s) and args.episode_length_s > 0):
        errors.append("--episode_length_s must be finite and greater than 0")
    try:
        parse_policy_options(args)
    except ValueError as error:
        errors.append(str(error))
    if args.sim_rename_map:
        try:
            json.loads(args.sim_rename_map)
        except ValueError as error:
            errors.append(f"--sim_rename_map is not valid JSON: {error}")
    return errors


_errors = check_args(args_cli)
if _errors:
    parser.error("; ".join(_errors))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

importlib.import_module("isaaclab_tasks")
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

importlib.import_module("sim_to_real_so101.tasks")
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface, SimRemotePolicy  # noqa: E402
from sim_to_real_so101.utils.pick_place_benchmark import StageTrace  # noqa: E402
from sim_to_real_so101.utils.pick_place_benchmark_runtime import install_stage_tracking  # noqa: E402
from sim_to_real_so101.utils.sim_real_eval import SimRealSession, furthest_stage, load_trials  # noqa: E402
from sim_to_real_so101.utils.sim_real_web_gui import SimRealWebGui  # noqa: E402


def sim_frames(obs):
    """Sim camera frames for the page: the room and wrist RGB images of env 0."""
    frames = {"sim_room": None, "sim_wrist": None}
    for key, value in obs["visual"].items():
        if "rgb" in key:
            frames["sim_wrist" if "wrist" in key else "sim_room"] = value[0]
    return frames


def real_frames(observation):
    return {"real_room": observation.get("external"), "real_wrist": observation.get("gripper")}


class FramePublishingRobot:
    """Forwards to the LeRobot follower and mirrors each observation's camera frames to the page."""

    def __init__(self, robot, gui):
        self._robot = robot
        self._gui = gui

    def get_observation(self):
        obs = self._robot.get_observation()
        self._gui.update_images(real_frames(obs))
        return obs

    def send_action(self, action):
        # Only reached through real_rollout.run_episode, which calls it only when motion is enabled.
        return self._robot.send_action(action)

    def __getattr__(self, name):
        return getattr(self._robot, name)


def build_sim(args):
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    if args.robot_usd is not None:
        robot_usd = args.robot_usd.resolve()
        if not robot_usd.is_file():
            raise ValueError(f"Robot USD does not exist: {robot_usd}")
        env_cfg.scene.robot.spawn.usd_path = str(robot_usd)
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    params = env_cfg.events.spawn_cube.params
    params.update(mode="cycle", random_fraction=0.0, seed=args.seed, reset_robot=True,
                  eval_set=str(args.eval_set), splits=None, repeats=1, start_offset=0)

    # Seed all RNGs for reproducible episode resets
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args.task, cfg=env_cfg)
    install_stage_tracking(env.unwrapped)

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}

    rename_map = json.loads(args.sim_rename_map) if args.sim_rename_map else None
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
    robot_iface.init_device(visualize=False)

    # No smoothing and no grasp assistance: the real side has neither, so the comparison stays fair.
    sim_policy = SimRemotePolicy(
        robot_iface=robot_iface,
        policy_name=args.policy,
        policy_options=parse_policy_options(args),
        api_token=args.policy_api_token,
        host=args.policy_host,
        port=args.policy_port,
        action_horizon=args.action_horizon,
        action_repeat=1,
        lang_instruction=args.lang_instruction,
    )
    sim_policy.connect()
    return SimpleNamespace(env=env, sim_policy=sim_policy, step_dt=float(env.unwrapped.step_dt),
                           episode_length_s=float(env_cfg.episode_length_s))


def build_real(args):
    if args.no_real:
        return None
    interface = real_cli.make_interface(args)
    real_cli.connect_follower(interface)
    policy = real_cli.make_policy(args, interface)
    policy.connect()
    cameras = [json.loads(args.rename_map)[name] for name in interface.cameras]
    settings = real_rollout.RolloutSettings(args.fps, args.episode_seconds, args.enable_motion, args.max_start_error)
    return SimpleNamespace(interface=interface, policy=policy, cameras=cameras, settings=settings)


def shutdown_step(label, action):
    """Run one shutdown action; a failure is reported and does not skip the remaining steps."""
    try:
        action()
    except Exception as error:  # noqa: BLE001 - shutdown must keep going
        print(f"[WARN]: {label} failed during shutdown: {error}", flush=True)


def main():
    sim = real = gui = None
    try:
        sim = build_sim(args_cli)
        real = build_real(args_cli)
        env, sim_policy, step_dt = sim.env, sim.sim_policy, sim.step_dt

        data = json.loads(args_cli.eval_set.read_text(encoding="utf-8"))
        starts = [{"start_id": str(r["id"]), "split": r["split"], "x": r["x"], "y": r["y"], "yaw": r["yaw"],
                   "arm_start_state": list(r["arm_start_state"])} for r in data["starts"]]
        start_by_id = {s["start_id"]: s for s in starts}
        scene = {"table_outline_base_xy": data["scene"]["table_outline_base_xy"],
                 "box_footprint_base_xy": data["scene"]["box_footprint_base_xy"]}

        def nearest_arm_state(x, y):
            nearest = min(starts, key=lambda s: (s["x"] - x) ** 2 + (s["y"] - y) ** 2)
            return nearest["arm_start_state"]

        state = {}  # last sim observation, kept between runs

        def spawner(start):
            env.unwrapped._sim_real_forced_start = dict(start)
            obs, _ = env.reset()
            state["obs"] = obs
            gui.update_images(sim_frames(obs))

        def sim_runner(start):
            raw = env.unwrapped
            raw._sim_real_forced_start = dict(start)
            gui.clear_stop()
            publish()
            obs, _ = env.reset()
            raw._benchmark_trace = StageTrace()
            sim_policy.reset()
            actions = torch.zeros(env.action_space.shape, device=raw.device)
            settling = obs["policy"]["joint_pos_obs"][0].clone()
            step = 0
            stopped = done_t = done_f = False
            with torch.inference_mode():
                while simulation_app.is_running():
                    if gui.stop_requested():
                        stopped = True
                        break
                    if step < 10:
                        actions[:] = settling
                    else:
                        actions[:] = sim_policy.get_action(obs["policy"]["joint_pos_obs"][0].clone(), obs["visual"])
                    obs, _, terminated, truncated, _ = env.step(actions)
                    step += 1
                    gui.update_images(sim_frames(obs))
                    if step % 15 == 0:
                        gui.set_status(f"Sim running: step {step}, {raw._benchmark_trace.progress_label()}",
                                       busy=True)
                    done_t, done_f = bool(terminated.any().item()), bool(truncated.any().item())
                    if done_t or done_f:
                        break
            success = done_t and not done_f and not stopped
            result = raw._benchmark_trace.finish(success)
            raw._benchmark_trace = StageTrace()
            state["obs"] = obs
            return {"stage": furthest_stage(result["stages"]), "success": success, "steps": step,
                    "seconds": round(step * step_dt, 3),
                    "failure_mode": "stopped" if stopped else result["failure_mode"]}

        def real_runner(start, should_stop):
            if real is None:
                raise RuntimeError("Started with --no_real; no arm connected")
            gui.clear_stop()
            publish()
            real.policy.reset()
            robot = FramePublishingRobot(real.interface.robot, gui)
            trial_file = out_dir / f"real_trial_{session.snapshot()['next_trial_id']:03d}.jsonl"
            with open(trial_file, "w", encoding="utf-8") as trace:
                result = real_rollout.run_episode(
                    robot, real.policy, real.interface.SO101_JOINT_ORDER, real.cameras, real.settings,
                    should_stop=should_stop, log=lambda row: trace.write(json.dumps(row) + "\n"))
            return {"status": result.status, "steps": result.steps, "seconds": result.seconds,
                    "detail": result.detail}

        out_dir = args_cli.resume or Path(
            args_cli.output or f"outputs/sim_real_eval/{datetime.now():%Y%m%d_%H%M%S}")
        trials = load_trials(out_dir / "trials.jsonl") if args_cli.resume else []
        header = {
            "policy": args_cli.policy, "checkpoint": args_cli.checkpoint,
            "lang_instruction": args_cli.lang_instruction, "task": args_cli.task,
            "eval_set": str(args_cli.eval_set),
            "eval_set_sha256": hashlib.sha256(args_cli.eval_set.read_bytes()).hexdigest(),
            "seed": args_cli.seed, "action_horizon": args_cli.action_horizon,
            "motion": args_cli.enable_motion, "real": not args_cli.no_real, "fps": args_cli.fps,
            "episode_seconds": args_cli.episode_seconds, "sim_episode_length_s": sim.episode_length_s,
            "max_relative_target": args_cli.max_relative_target, "max_start_error": args_cli.max_start_error,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        info = {"motion": args_cli.enable_motion and not args_cli.no_real, "policy": args_cli.policy,
                "checkpoint": args_cli.checkpoint, "instruction": args_cli.lang_instruction,
                "eval_set": str(args_cli.eval_set), "starts": starts, "scene": scene}
        gui = SimRealWebGui(info, host=args_cli.gui_host, port=args_cli.gui_port)
        print(f"Sim/real evaluation page: {gui.url}", flush=True)
        session = SimRealSession(spawner, sim_runner, real_runner, out_dir, header=header, trials=trials)

        def publish():
            gui.set_session(session.snapshot(), [t.to_dict() for t in session.trials], session.summary())

        publish()
        gui.set_status("Ready: pick a start and spawn it")
        while simulation_app.is_running():
            request = gui.pop_request()
            if request is None:
                if real is not None and session.snapshot()["state"] not in ("sim_running", "real_running"):
                    gui.update_images(real_frames(real.interface.robot.get_observation()))  # live preview
                else:
                    time.sleep(0.05)
                continue
            cmd = request["cmd"]
            if cmd == "exit":
                break
            try:
                if cmd == "spawn":
                    if "start_id" in request:
                        if request["start_id"] not in start_by_id:
                            raise ValueError(f"Unknown start {request['start_id']}")
                        start = start_by_id[request["start_id"]]
                    else:
                        start = {"start_id": "manual", "split": "manual", "x": request["x"], "y": request["y"],
                                 "yaw": request["yaw"],
                                 "arm_start_state": nearest_arm_state(request["x"], request["y"])}
                    gui.set_status("Spawning in sim...", busy=True)
                    session.spawn(start)
                    gui.set_status("Spawned: place the real cube to match the overlay, then click Placed")
                elif cmd == "placed":
                    session.mark_placed()
                    gui.set_status("Real cube placed")
                elif cmd == "run_sim":
                    gui.set_status("Sim running...", busy=True)
                    result = session.run_sim()
                    gui.set_status(f"Sim finished: {result['stage']}")
                elif cmd == "run_real":
                    gui.set_status("Real arm running..." if args_cli.enable_motion
                                   else "Real shadow run (no motion)...", busy=True)
                    result = session.run_real(gui.stop_requested)
                    gui.set_status(f"Real finished: {result['status']} - label the stage")
                elif cmd == "label":
                    session.label_real(request["stage"], request["note"])
                    gui.set_status("Labelled; Save to record")
                elif cmd == "save":
                    trial = session.save()
                    gui.set_status(f"Saved trial {trial.trial_id}")
                elif cmd == "discard":
                    session.discard()
                    gui.set_status("Trial discarded")
            except (RuntimeError, ValueError) as error:
                gui.set_status(f"Error: {error}")
            except Exception as error:  # noqa: BLE001 - keep the page alive; e.g. a policy server timeout in sim
                traceback.print_exc()
                gui.set_status(f"Error ({type(error).__name__}): {error}. See the terminal for the traceback.")
            publish()
        print(f"Results: {out_dir / 'report.json'}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if gui is not None:
            shutdown_step("web page", gui.destroy)
        if real is not None:
            shutdown_step("real policy", real.policy.close_timing)
            shutdown_step("real arm", real.interface.disconnect)
        if sim is not None:
            shutdown_step("sim policy", sim.sim_policy.close_timing)
            shutdown_step("sim env", sim.env.close)
        shutdown_step("simulation app", simulation_app.close)


if __name__ == "__main__":
    main()
