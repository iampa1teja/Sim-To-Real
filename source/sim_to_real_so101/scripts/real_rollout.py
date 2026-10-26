"""Run a served policy on the physical SO-101 follower. Shadow mode (no motion) unless --enable-motion is given.

Run it from the repo root inside the teleop container, after `so101 serve` (or docker/eval.sh --server_only)
has started the policy server. No simulator is started.

    real_rollout --policy molmoact2 --lang_instruction "Pick up the blue cube and place it in the white box"

The arm is configured from the same variables as the recorder: ROBOT_PORT, ROBOT_ID, CAMERA_GRIPPER,
CAMERA_EXTERNAL, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOURCC and CONTROL_FPS. The arm must already be calibrated
(`lerobot-calibrate`); this script never starts calibration. On exit the arm keeps holding its last goal.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from sim_to_real_so101.utils import real_rollout as rr

DEFAULT_RENAME_MAP = '{"external": "room", "gripper": "wrist"}'


def build_parser():
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="policy folder name under policies/")
    parser.add_argument("--policy_host", default="localhost")
    parser.add_argument("--policy_port", type=int, default=None, help="default: the policy's default_port")
    parser.add_argument("--policy_option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", default="", help="label stored in the report; the server decides what is loaded")
    parser.add_argument("--lang_instruction", required=True)
    parser.add_argument("--action_horizon", type=int, default=30)
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
    parser.add_argument("--calibration_file", help="follower mapping JSON (default: "
                        "$HF_LEROBOT_CALIBRATION/robots/so101_follower/<follower_id>.json, "
                        "or ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<follower_id>.json)")
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


def validate(args):
    """Argument errors found before touching any hardware."""
    errors = []
    if not args.follower_port:
        errors.append("set ROBOT_PORT or pass --follower_port")
    if not args.follower_id:
        errors.append("set ROBOT_ID or pass --follower_id")
    if args.calibration_file is None:
        calibration_root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", "~/.cache/huggingface/lerobot/calibration"))
        args.calibration_file = calibration_root.expanduser() / "robots" / "so101_follower" / f"{args.follower_id}.json"
    else:
        args.calibration_file = Path(args.calibration_file).expanduser()
    try:
        args.joint_mapping = rr.follower_mapping_from_calibration(args.calibration_file)
    except ValueError as error:
        errors.append(str(error))
    if not (args.camera_gripper or args.camera_external):
        errors.append("set CAMERA_GRIPPER and/or CAMERA_EXTERNAL, or pass --camera_gripper / --camera_external")
    if args.episodes < 1 or args.episode_seconds <= 0 or args.fps < 1:
        errors.append("--episodes, --episode_seconds and --fps must be positive")
    if args.max_relative_target <= 0 or args.max_start_error <= 0:
        errors.append("--max_relative_target and --max_start_error must be positive")
    try:
        rename_map = json.loads(args.rename_map)
        cameras = camera_specs(args)
        missing = sorted(set(cameras) - set(rename_map))
        if missing:
            errors.append(f"--rename_map has no policy name for camera(s) {missing}")
    except ValueError as error:
        errors.append(f"--rename_map is not valid JSON: {error}")
    for option in args.policy_option:
        if "=" not in option:
            errors.append(f"--policy_option must look like KEY=VALUE, got {option!r}")
    return errors


def camera_specs(args):
    common = {"width": args.camera_width, "height": args.camera_height, "fourcc": args.camera_fourcc}
    specs = {}
    if args.camera_gripper:
        specs["gripper"] = {**common, "index_or_path": args.camera_gripper}
    if args.camera_external:
        specs["external"] = {**common, "index_or_path": args.camera_external}
    return specs


def make_interface(args):
    from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface

    return LeRobotSO101Interface(
        device="cpu", port=args.follower_port, id=args.follower_id, cameras=camera_specs(args), fps=args.fps,
        kind="follower", rename_map=json.loads(args.rename_map), joint_mapping=args.joint_mapping,
        follower_options={"max_relative_target": args.max_relative_target,
                          "disable_torque_on_disconnect": args.release_on_exit})


def make_policy(args, interface):
    from sim_to_real_so101.utils.lerobot_interface import SimRemotePolicy

    options = dict(option.split("=", 1) for option in args.policy_option)
    return SimRemotePolicy(interface, host=args.policy_host, port=args.policy_port, action_horizon=args.action_horizon,
                           lang_instruction=args.lang_instruction, control_dt=1 / args.fps, policy_name=args.policy,
                           policy_options=options)


def connect_follower(interface):
    """Connect without ever entering LeRobot's interactive calibration, and refuse an uncalibrated arm."""
    interface.init_device()
    calibration_file = Path(interface.robot.calibration_fpath)
    if not calibration_file.is_file():
        raise SystemExit(f"No calibration file for follower {interface.id!r} ({calibration_file}). "
                         "Run lerobot-calibrate for this arm first.")
    interface.connect(calibrate=False)
    if not interface.robot.is_calibrated:
        raise SystemExit("The arm's stored calibration does not match its calibration file; recalibrate it with "
                         "lerobot-calibrate before running a policy.")


def ask(prompt, input_fn, allowed):
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in allowed:
            return answer


def main(argv=None, *, interface_factory=make_interface, policy_factory=make_policy, input_fn=input,
         now=datetime.now):
    args = build_parser().parse_args(argv)
    errors = validate(args)
    if errors:
        print("real_rollout: " + "; ".join(errors), file=sys.stderr)
        return 2
    if not args.no_prompt and not sys.stdin.isatty() and input_fn is input:
        print("real_rollout: needs an interactive terminal (docker exec -it) or --no_prompt", file=sys.stderr)
        return 2
    started = now()
    output = Path(args.output or f"outputs/real_rollout/{started:%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    print("MOTION ENABLED: the arm will move. Keep a hand on the power switch." if args.enable_motion else
          "SHADOW MODE: the policy is queried and logged, no command is sent to the arm.", flush=True)
    header = {"policy": args.policy, "checkpoint": args.checkpoint, "lang_instruction": args.lang_instruction,
              "motion": args.enable_motion, "fps": args.fps, "episode_seconds": args.episode_seconds,
              "max_relative_target": args.max_relative_target, "max_start_error": args.max_start_error,
              "started": started.isoformat(timespec="seconds")}
    settings = rr.RolloutSettings(args.fps, args.episode_seconds, args.enable_motion, args.max_start_error)
    interface = interface_factory(args)
    policy, episodes, failed = None, [], False
    try:
        connect_follower(interface)
        policy = policy_factory(args, interface)
        policy.connect()
        cameras = list(json.loads(args.rename_map)[name] for name in interface.cameras)
        for index in range(args.episodes):
            if not args.no_prompt:
                input_fn(f"Episode {index + 1}/{args.episodes}: reset the scene, then press Enter to start (Ctrl-C quits) ")
            policy.reset()
            with open(output / f"episode_{index:03d}.jsonl", "w", encoding="utf-8") as trace:
                result = rr.run_episode(interface.robot, policy, interface.SO101_JOINT_ORDER, cameras, settings,
                                        log=lambda row: trace.write(json.dumps(row) + "\n"))
            outcome = "unlabeled"
            if not args.no_prompt and result.status == "completed" and args.enable_motion:
                outcome = {"y": "success", "n": "failure", "s": "unlabeled"}[
                    ask("Outcome? [y=success, n=failure, s=skip] ", input_fn, ("y", "n", "s"))]
            episodes.append({"index": index, "status": result.status, "steps": result.steps, "seconds": result.seconds,
                             "detail": result.detail, "outcome": outcome, "inference_calls": policy.inference_calls})
            print(f"Episode {index + 1}: {result.status}, {result.steps} steps"
                  + (f" ({result.detail})" if result.detail else ""), flush=True)
            rr.write_report(output / "report.json", header, episodes)
            if result.status not in ("completed", "stopped"):
                failed = True
                break
    except KeyboardInterrupt:
        print("\nInterrupted: the arm holds its last goal.", flush=True)
        failed = True
    finally:
        if episodes or policy is not None:
            rr.write_report(output / "report.json", header, episodes)
        if policy is not None:
            policy.close_timing()
        interface.disconnect()
    print(f"Report: {output / 'report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
