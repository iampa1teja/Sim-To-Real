"""Run a served policy on the physical follower. Shadow mode (no motion) unless --enable-motion is given.

Run it in the sim/teleop container after the policy server is up (`s2r serve <policy>`). No simulator starts.

    s2r rollout --policy molmoact2 --lang_instruction "Pick up the blue cube and place it in the white box"

The arm is configured from the same variables as the recorder: ROBOT_PORT, ROBOT_ID, CAMERA_GRIPPER,
CAMERA_EXTERNAL, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOURCC and CONTROL_FPS. The arm must already be calibrated
(`lerobot-calibrate`); this script never starts calibration. On exit the arm keeps holding its last goal.

Units are raw LeRobot units, the same ones the simulator mapping uses: -100..100 for the body joints and 0..100
for the gripper. Nothing is sent to the arm unless motion is enabled. Stopping never releases torque unless
--release-on-exit is given.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

DEFAULT_RENAME_MAP = '{"external": "room", "gripper": "wrist"}'


# ---------------------------------------------------------------------------
# Episode runner (numpy only: robot and policy are injected)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutSettings:
    fps: int = 30
    episode_seconds: float = 30.0
    enable_motion: bool = False
    max_start_error: float = 20.0  # largest allowed gap, raw units, between the pose and the first target
    expected_start_roll: float | None = None  # shared sim-start wrist roll in this follower's units
    roll_key: str | None = None
    raw_ranges: tuple[tuple[float, float], ...] = ()


@dataclass
class EpisodeResult:
    status: str  # completed | stopped | policy_error | robot_error | start_too_far
    steps: int
    seconds: float
    detail: str = ""


def joint_ranges(arm, mapping):
    """Raw action bounds from the selected calibration's actual device codec."""
    from core.arms.mapping import raw_joint_limits
    bounds = raw_joint_limits(arm, mapping)
    return tuple(bounds[name] for name in arm.joint_names)


def sim_start_roll_units(mapping, mount=None, arm=None):
    """Optional sim-start roll expressed in the selected unit's raw convention."""
    from core.arms.spec import load_arm_spec
    from core.arms.mapping import physical_to_units
    from core.scenes import load_scene_spec, load_mount_spec

    arm = arm or load_arm_spec()[0]
    if arm.wrist_roll is None:
        return None
    joint = next(joint for joint in arm.joints if joint.name == arm.wrist_roll.joint)
    mount = load_mount_spec(load_scene_spec(), arm) if mount is None else mount
    initial = mount.start_joint_positions.get(joint.name, joint.initial)
    physical = math.degrees(initial) if joint.type == "revolute" else initial
    return float(physical_to_units([physical], {joint.name: mapping[joint.name]}, [joint.name],
                                   gripper_joints=arm.gripper_joint_names)[0])


def clamp_action(action, joint_keys, ranges):
    """Target vector inside each joint's raw range. Raises ValueError for a missing or non-finite value."""
    if len(ranges) != len(joint_keys):
        raise ValueError("One raw range per arm joint is required")
    values = []
    for key, (low, high) in zip(joint_keys, ranges):
        if key not in action:
            raise ValueError(f"policy action has no value for {key}")
        value = float(action[key])
        if not np.isfinite(value):
            raise ValueError(f"policy action for {key} is not finite ({value})")
        values.append(min(max(value, low), high))
    return np.asarray(values, dtype=np.float32)


def run_episode(robot, policy, joint_keys, cameras, settings, *, should_stop=lambda: False, log=None,
                clock=time.monotonic, sleep=time.sleep):
    """Run one episode at settings.fps for at most episode_seconds; hold the last goal when it ends or fails.

    cameras maps the robot camera name to the policy camera name.
    """
    period = 1.0 / settings.fps
    max_steps = int(settings.episode_seconds * settings.fps)
    started = clock()
    next_tick = started
    step = 0
    status, detail = "completed", ""
    while step < max_steps:
        if should_stop():
            status = "stopped"
            break
        try:
            observation = robot.get_observation()
            state = np.asarray([observation[key] for key in joint_keys], dtype=np.float32)
            images = {policy_name: observation[robot_name] for robot_name, policy_name in cameras.items()}
        except Exception as error:
            status, detail = "robot_error", f"observation failed: {error!r}"
            break
        if step == 0 and settings.expected_start_roll is not None:
            measured_roll = float(state[joint_keys.index(settings.roll_key)])
            roll_gap = abs(measured_roll - settings.expected_start_roll)
            if roll_gap > settings.max_start_error:
                status = "start_too_far"
                detail = (f"wrist roll is {roll_gap:.1f} raw units from the mapped sim start "
                          f"({measured_roll:.3f} measured, {settings.expected_start_roll:.3f} expected; "
                          f"limit {settings.max_start_error})")
                break
        try:
            target = clamp_action(policy.get_raw_action(state, images), joint_keys, settings.raw_ranges)
        except Exception as error:
            status, detail = "policy_error", repr(error)
            break
        if step == 0:
            gap = float(np.max(np.abs(target - state)))
            if gap > settings.max_start_error:
                status = "start_too_far"
                detail = f"first target is {gap:.1f} raw units from the current pose (limit {settings.max_start_error})"
                break
        if settings.enable_motion:
            try:
                robot.send_action(dict(zip(joint_keys, target.tolist())))
            except Exception as error:
                status, detail = "robot_error", f"send_action failed: {error!r}"
                break
        if log is not None:
            log({"step": step, "t": round(clock() - started, 4), "state": state.tolist(), "target": target.tolist(),
                 "sent": settings.enable_motion})
        step += 1
        next_tick += period
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)
        else:
            next_tick = clock()  # overran the tick: do not catch up with a burst of commands
    return EpisodeResult(status, step, round(clock() - started, 3), detail)


def summarize(episodes):
    """Only labelled episodes count towards the success rate."""
    labelled = [e for e in episodes if e["outcome"] in ("success", "failure")]
    successes = sum(e["outcome"] == "success" for e in labelled)
    return {"episodes": len(episodes), "labelled": len(labelled), "successes": successes,
            "success_rate": (successes / len(labelled)) if labelled else None}


def write_report(path, header, episodes):
    report = {"schema": "so101_real_rollout_v1", **header, "episodes": episodes, "summary": summarize(episodes)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_path", type=Path, help="rollout YAML; explicit flags override its defaults")
    parser.add_argument("--arm", default=env("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=env("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=env("S2R_TASK", DEFAULT_TASK))
    parser.add_argument("--variant", default=env("S2R_VARIANT", "teleop"))
    parser.add_argument("--unit", default=env("S2R_UNIT"), help="arm calibration unit ID")
    parser.add_argument("--policy", required=True, help="policy folder name under policies/")
    parser.add_argument("--policy_host", default="localhost")
    parser.add_argument("--policy_port", type=int, default=None, help="default: the policy's default_port")
    parser.add_argument("--policy_option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", default="", help="label stored in the report; the server decides what is loaded")
    parser.add_argument("--lang_instruction", default=None, help="default: the selected task's instruction")
    parser.add_argument("--action_horizon", type=int, default=None, help="default: the policy manifest horizon")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode_seconds", type=float, default=30.0)
    parser.add_argument("--enable-motion", dest="enable_motion", action="store_true",
                        help="actually move the arm; without it the policy is queried and logged only")
    parser.add_argument("--max_start_error", type=float, default=20.0,
                        help="refuse an episode whose first target is farther than this (raw units) from the pose")
    parser.add_argument("--max_relative_target", type=float, default=10.0,
                        help="per-command step limit, raw units, enforced by the follower driver")
    parser.add_argument("--release-on-exit", dest="release_on_exit", action="store_true",
                        help="disable torque on exit; the arm will drop")
    parser.add_argument("--follower_port", default=env("ROBOT_PORT"))
    parser.add_argument("--follower_id", default=env("ROBOT_ID"))
    parser.add_argument("--camera_gripper", default=env("CAMERA_GRIPPER"))
    parser.add_argument("--camera_external", default=env("CAMERA_EXTERNAL"))
    parser.add_argument("--camera_width", type=int, default=int(env("CAMERA_WIDTH", "640")))
    parser.add_argument("--camera_height", type=int, default=int(env("CAMERA_HEIGHT", "480")))
    parser.add_argument("--camera_fourcc", default=env("CAMERA_FOURCC") or None)
    parser.add_argument("--rename_map", default=DEFAULT_RENAME_MAP, help="robot camera name -> policy camera name (JSON)")
    parser.add_argument("--fps", type=int, default=int(env("CONTROL_FPS", "30")))
    parser.add_argument("--output", default="", help="report folder (default outputs/real_rollout/<timestamp>)")
    parser.add_argument("--no_prompt", action="store_true", help="do not ask for resets or outcomes (outcomes unlabeled)")
    return parser


def parse_args(argv=None):
    """Load YAML defaults, then apply explicit flags and the task's instruction."""
    from core.utils.config import parse_yaml_args

    args = parse_yaml_args(build_parser(), argv, json_fields=("rename_map",))
    if args.lang_instruction is None:
        from core.tasks.loader import load_task

        args.lang_instruction = load_task(args.task).instruction
    return args


def camera_specs(args):
    """Robot camera name -> spec for sim.teleop.devices (only the cameras that are set)."""
    from sim.recording.recording_setup import real_camera_specs

    return real_camera_specs({"gripper": args.camera_gripper, "external": args.camera_external}, fps=args.fps,
                             width=args.camera_width, height=args.camera_height, fourcc=args.camera_fourcc)


def validate(args):
    """Argument errors found before touching any hardware."""
    from core.arms.calibration import resolve_calibration
    from core.arms.spec import load_arm_spec

    errors = []
    if not args.follower_port:
        errors.append("set ROBOT_PORT or pass --follower_port")
    if not args.follower_id:
        errors.append("set ROBOT_ID or pass --follower_id")
    try:
        args.arm_spec = load_arm_spec(args.arm)[0]
        args.calibration_selection = resolve_calibration(args.arm_spec, unit=args.unit, follower_id=args.follower_id)
        args.joint_mapping = args.calibration_selection.mapping
    except (ValueError, FileNotFoundError) as error:
        errors.append(str(error))
    if not (args.camera_gripper or args.camera_external):
        errors.append("set CAMERA_GRIPPER and/or CAMERA_EXTERNAL, or pass --camera_gripper / --camera_external")
    if args.episodes < 1 or args.episode_seconds <= 0 or args.fps < 1:
        errors.append("--episodes, --episode_seconds and --fps must be positive")
    if args.max_relative_target <= 0 or args.max_start_error <= 0:
        errors.append("--max_relative_target and --max_start_error must be positive")
    try:
        rename_map = json.loads(args.rename_map)
        missing = sorted(set(camera_specs(args)) - set(rename_map))
        if missing:
            errors.append(f"--rename_map has no policy name for camera(s) {missing}")
    except ValueError as error:
        errors.append(f"--rename_map is not valid JSON: {error}")
    for option in args.policy_option:
        if "=" not in option:
            errors.append(f"--policy_option must look like KEY=VALUE, got {option!r}")
    return errors


def connect_follower(args):
    """Connect without ever entering LeRobot's calibration; refuses an arm whose calibration does not match."""
    from sim.teleop.devices import connect_device

    return connect_device("follower", args.follower_port, args.follower_id, cameras=camera_specs(args), fps=args.fps,
                          selection=args.calibration_selection, calibrate=False,
                          options={"max_relative_target": args.max_relative_target,
                                   "disable_torque_on_disconnect": args.release_on_exit})


def make_policy(args):
    from core.policy_client.contract import ArmInfo
    from core.policy_client.remote_policy import RemotePolicy

    options = dict(option.split("=", 1) for option in args.policy_option)
    arm_info = ArmInfo.from_arm(args.arm_spec, args.calibration_selection.mapping)
    return RemotePolicy(list(args.arm_spec.feature_names), policy_name=args.policy, host=args.policy_host,
                        port=args.policy_port, action_horizon=args.action_horizon,
                        lang_instruction=args.lang_instruction, control_dt=1 / args.fps, policy_options=options,
                        arm_info=arm_info)


def ask(prompt, input_fn, allowed):
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in allowed:
            return answer


def main(argv=None, *, follower_factory=connect_follower, policy_factory=make_policy, input_fn=input,
         now=datetime.now):
    args = parse_args(argv)
    os.environ.update(S2R_ARM=args.arm, S2R_SCENE=args.scene)
    errors = validate(args)
    if errors:
        print("real_rollout: " + "; ".join(errors), file=sys.stderr)
        return 2
    args.calibration_selection.print_paths()
    if not args.no_prompt and not sys.stdin.isatty() and input_fn is input:
        print("real_rollout: needs an interactive terminal (docker exec -it) or --no_prompt", file=sys.stderr)
        return 2
    started = now()
    output = Path(args.output or f"outputs/real_rollout/{started:%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    print("MOTION ENABLED: the arm will move. Keep a hand on the power switch." if args.enable_motion else
          "SHADOW MODE: the policy is queried and logged, no command is sent to the arm.", flush=True)
    expected_roll = sim_start_roll_units(args.joint_mapping, arm=args.arm_spec)
    header = {"policy": args.policy, "checkpoint": args.checkpoint, "lang_instruction": args.lang_instruction,
              "motion": args.enable_motion, "fps": args.fps, "episode_seconds": args.episode_seconds,
              "max_relative_target": args.max_relative_target, "max_start_error": args.max_start_error,
              "started": started.isoformat(timespec="seconds"), "sim_start_roll_units": expected_roll,
              "calibration": args.calibration_selection.provenance}
    joint_keys = list(args.arm_spec.feature_names)
    settings = RolloutSettings(args.fps, args.episode_seconds, args.enable_motion, args.max_start_error,
                               expected_start_roll=expected_roll,
                               roll_key=(f"{args.arm_spec.wrist_roll.joint}.pos" if args.arm_spec.wrist_roll else None),
                               raw_ranges=joint_ranges(args.arm_spec, args.calibration_selection.mapping))
    rename_map = json.loads(args.rename_map)
    cameras = {name: rename_map[name] for name in camera_specs(args)}
    follower, policy, episodes, failed = None, None, [], False
    try:
        follower = follower_factory(args)
        policy = policy_factory(args)
        policy.connect()
        for index in range(args.episodes):
            if not args.no_prompt:
                input_fn(f"Episode {index + 1}/{args.episodes}: reset the scene, then press Enter to start (Ctrl-C quits) ")
            policy.reset()
            with open(output / f"episode_{index:03d}.jsonl", "w", encoding="utf-8") as trace:
                result = run_episode(follower, policy, joint_keys, cameras, settings,
                                     log=lambda row: trace.write(json.dumps(row) + "\n"))
            outcome = "unlabeled"
            if not args.no_prompt and result.status == "completed" and args.enable_motion:
                outcome = {"y": "success", "n": "failure", "s": "unlabeled"}[
                    ask("Outcome? [y=success, n=failure, s=skip] ", input_fn, ("y", "n", "s"))]
            episodes.append({"index": index, "status": result.status, "steps": result.steps, "seconds": result.seconds,
                             "detail": result.detail, "outcome": outcome, "inference_calls": policy.inference_calls})
            print(f"Episode {index + 1}: {result.status}, {result.steps} steps"
                  + (f" ({result.detail})" if result.detail else ""), flush=True)
            write_report(output / "report.json", header, episodes)
            if result.status not in ("completed", "stopped"):
                failed = True
                break
    except KeyboardInterrupt:
        print("\nInterrupted: the arm holds its last goal.", flush=True)
        failed = True
    finally:
        if episodes or policy is not None:
            write_report(output / "report.json", header, episodes)
        if policy is not None:
            policy.close_timing()
        if follower is not None:
            follower.disconnect()
    print(f"Report: {output / 'report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
