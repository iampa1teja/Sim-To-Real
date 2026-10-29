#!/usr/bin/env python3
"""Summarize/watch passive telemetry, or manually match a rendered torque-off pose."""
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.calibration_check import (
    LIMITATION, Thresholds, pose_measurement, reference_poses, summarize,
)
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS, resolve_calibration
from sim_to_real_so101.utils.joint_angle_display import JointAngleTable
from sim_to_real_so101.utils.calibration_poses import POSE_SET_PATH, PoseThresholds, load_pose_set, summarize_poses


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    report = modes.add_parser("summarize", help="Recompute both summary sidecars offline")
    report.add_argument("--log", type=Path, required=True)
    report.add_argument("--output", type=Path, help="Also write the JSON summary to this path")
    report.add_argument("--pass-deg", type=float, default=1.)
    report.add_argument("--warn-deg", type=float, default=3.)
    report.add_argument("--static-velocity-deg-s", type=float, default=1.)
    report.add_argument("--static-window-s", type=float, default=.25)
    watch = modes.add_parser("watch", help="Show a fixed live table; Ctrl-C stops this observer only")
    watch.add_argument("--log", type=Path, required=True)
    poses = modes.add_parser("poses", help="Operator-started reference-pose session (requires Isaac/LeRobot)")
    poses.add_argument("--log", type=Path, help="NEW pose evidence JSONL; never overwritten")
    poses.add_argument("--preview", action="store_true", help="Isaac-only clearance checks and room-camera contact sheet")
    poses.add_argument("--headless", action="store_true", help="Render without a display (also usable for preview)")
    poses.add_argument("--output", type=Path, default=ROOT / "outputs/calibration-check/preview")
    poses.add_argument("--pose-set", type=Path, default=POSE_SET_PATH)
    poses.add_argument("--pose-pass-deg", type=float, default=1.)
    poses.add_argument("--pose-warn-deg", type=float, default=2.)
    poses.add_argument("--pose-spread-deg", type=float, default=1.)
    poses.add_argument("--pose-min-poses", type=int, default=5)
    poses.add_argument("--follower-port", default=os.environ.get("ROBOT_PORT"))
    poses.add_argument("--follower-id", "--follower_id", default=os.environ.get("ROBOT_ID"))
    poses.add_argument("--supported", action="store_true", help="Confirm the real arm is physically supported with torque off")
    poses.add_argument("--task", default="Lerobot-So101-Teleop-MyRoom")
    poses.add_argument("--device", default="cuda:0")
    report = modes.add_parser("report", help="Combine independent poses and tracking evidence offline")
    report.add_argument("--poses", type=Path, required=True)
    report.add_argument("--gap", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    for flag, kind in (("pass-deg", float), ("warn-deg", float), ("spread-deg", float), ("min-poses", int)):
        report.add_argument("--pose-" + flag, type=kind, default=None, help="Override the recorded pose threshold")
    lock = modes.add_parser("lock", help="Lock a LOCKABLE report into simulation; print git commands")
    lock.add_argument("--report", type=Path, required=True)
    return parser


def latest_frame(path):
    """Ignore a partial final line and provenance-only startup files."""
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        data = stream.read()
    lines = data.split(b"\n")[:-1]
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if row.get("type") == "frame" or "sim_physx_degrees" in row:
            return row
    return None


def watch_log(path):
    last = None
    table = JointAngleTable(FOLLOWER_JOINTS)
    try:
        while True:
            row = latest_frame(path)
            identity = None if row is None else (row["frame"], row.get("timestamp_unix_ns", row.get("read_finished_unix_ns")))
            if row is not None and identity != last:
                last = identity
                table.update(row)
            time.sleep(1)
    except KeyboardInterrupt:
        return
    finally:
        table.close()


class PoseRenderer:
    """Render a kinematically held sim pose; owns no real motor interface."""

    def __init__(self, args, selection):
        from isaaclab.app import AppLauncher

        self.env = None
        self.app = AppLauncher(headless=args.headless, device=args.device, enable_cameras=True).app
        try:
            import gymnasium as gym
            import torch
            from importlib import import_module
            from isaaclab_tasks.utils import parse_env_cfg
            from sim_to_real_so101.utils.joint_limits import USD_JOINTS

            import_module("isaaclab_tasks")
            import_module("sim_to_real_so101.tasks")
            cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
            if cfg.sim_joint_mapping != selection.mapping:
                raise ValueError("Rendered environment differs from the selected calibration")
            # Widen only this reference renderer's room camera, preserving its
            # calibrated location/orientation and all production task defaults.
            camera = cfg.scene.camera_realsense_rgb.spawn
            camera.intrinsics = dict(camera.intrinsics)
            for key in ("fx", "fy"):
                camera.intrinsics[key] *= .35
            camera.focal_length *= .35
            self.env = gym.make(args.task, cfg=cfg)
            self.env.reset()
            self.robot = self.env.unwrapped.scene["robot"]
            self.joint_ids = [self.robot.joint_names.index(name) for name in USD_JOINTS]
            self.torch = torch
        except BaseException:
            import traceback
            traceback.print_exc()
            self.close(exit_code=1)
            raise

    def show(self, name, target):
        q = self.torch.deg2rad(self.torch.tensor(target, device=self.env.unwrapped.device, dtype=self.torch.float32))[None, :]
        self.robot.write_joint_state_to_sim(q, self.torch.zeros_like(q), joint_ids=self.joint_ids)
        self.env.unwrapped.sim.forward()
        self.render_for(.1)
        actual = self.robot.root_physx_view.get_dof_positions()[0, self.joint_ids].detach().cpu().numpy()
        actual = actual * 180 / self.torch.pi
        if max(abs(actual - target)) > .05:
            raise RuntimeError("Rendered pose's PhysX read-back differs from the requested reference")
        print(f"\n{name}: " + ", ".join(f"{joint}={value:+.2f}°" for joint, value in zip(FOLLOWER_JOINTS, target)), flush=True)
        return actual.tolist()

    def render_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.app.is_running():
                raise KeyboardInterrupt
            self.env.unwrapped.sim.render()
            time.sleep(min(.02, max(0, deadline - time.monotonic())))

    def wait_for_match(self):
        answers = queue.Queue()

        def read_enter():
            try:
                answers.put(input("Support and match the real arm to the sim, then Enter to sample (q + Enter quits): "))
            except EOFError:
                answers.put("q")

        threading.Thread(target=read_enter, daemon=True).start()
        while answers.empty():
            self.render_for(.05)
        if answers.get().strip().lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

    def close(self, exit_code=0):
        self.app._app.post_quit(exit_code)
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.app.close()


def run_pose_session(robot, selection, renderer, stream, *, supported, pose_set=POSE_SET_PATH, thresholds=None):
    """Collect operator matches through the guarded session, including finalization."""
    from sim_to_real_so101.utils.joint_angle_check import (
        observation_only_session, sample_pose,
    )
    from sim_to_real_so101.utils.joint_limits import runtime_joint_limits

    selection.verify_loaded(robot, "follower")
    bounds = runtime_joint_limits()
    limits = [[bounds[name]["joint_min"], bounds[name]["joint_max"]] for name in FOLLOWER_JOINTS]
    measurements, evidence = [], {}
    try:
        with observation_only_session(robot, selection.follower_path, evidence, supported=supported) as reader:
            for name, target in reference_poses(FOLLOWER_JOINTS, limits, pose_set):
                shown = renderer.show(name, target)
                renderer.wait_for_match()
                samples = sample_pose(reader, sleep=renderer.render_for)
                measurement = pose_measurement(name, target, samples, selection.mapping, FOLLOWER_JOINTS, limits)
                measurement["shown_physx_deg"] = shown
                measurements.append(measurement)
                stream.write(json.dumps(dict(type="pose", **measurement), allow_nan=False) + "\n")
                print("Accepted" if measurement["accepted"] else "Rejected: motion exceeded 0.3 normalized units std", flush=True)
    finally:
        result = summarize_poses(measurements, thresholds)
        result["applied"] = False
        result.update(limitation=LIMITATION, connection_evidence=evidence,
                      measurements=measurements, real_minus_target_sign="measurement", candidate_sign="target minus real; additive to mapping")
        stream.write(json.dumps(dict(type="pose_summary", **result), allow_nan=False) + "\n")
        for name, joint in result["joints"].items():
            print(f"{name}: real-target median={joint['median_offset_deg']}° spread={joint['spread_deg']}° "
                  f"poses={joint['accepted_poses']} {joint['verdict']} {joint['reason']}", flush=True)
        gripper = result["gripper"]
        print(f"Gripper (nonblocking): real-target median={gripper['median_offset_deg']}° "
              f"spread={gripper['spread_deg']}° poses={gripper['accepted_poses']} {gripper['verdict']}", flush=True)
        print(f"Calibration: {result['decision']} (gripper never blocks; applied=false)", flush=True)
        Path(str(stream.name) + ".candidates.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def poses(args, parser):
    from dataclasses import asdict
    from sim_to_real_so101.utils.joint_limits import runtime_joint_limits

    bounds = runtime_joint_limits()
    limits = [[bounds[n]["joint_min"], bounds[n]["joint_max"]] for n in FOLLOWER_JOINTS]
    references, pose_identity = load_pose_set(FOLLOWER_JOINTS, limits, args.pose_set)
    thresholds = PoseThresholds(args.pose_pass_deg, args.pose_warn_deg, args.pose_spread_deg, args.pose_min_poses)
    if args.preview:
        from sim_to_real_so101.utils.calibration_preview import preview
        renderer = PoseRenderer(args, resolve_calibration())
        exit_code = 1
        try:
            result = preview(renderer, references, pose_identity, args.output)
            exit_code = 0 if result["all_ok"] else 1
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            renderer.close(exit_code=exit_code)
        if not result["all_ok"]:
            raise SystemExit(1)
        return
    if not args.supported or not args.follower_port or not args.follower_id or not args.log:
        parser.error("poses requires --supported, --log, --follower-port / ROBOT_PORT and --follower-id / ROBOT_ID")
    from sim_to_real_so101.utils.joint_angle_check import make_observation_only_robot, telemetry_provenance

    selection = resolve_calibration(follower_id=args.follower_id)
    selection.print_paths()
    renderer = None
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x", encoding="utf-8", buffering=1) as stream:
        header = telemetry_provenance(selection, type="provenance", mode="operator_poses", limitation=LIMITATION,
                                      joint_order=list(FOLLOWER_JOINTS), runtime_limits=limits, pose_set=pose_identity,
                                      pose_thresholds=asdict(thresholds))
        stream.write(json.dumps(header, allow_nan=False) + "\n")
        try:
            renderer = PoseRenderer(args, selection)
            robot = make_observation_only_robot(args.follower_port, args.follower_id, selection.follower_path)
            run_pose_session(robot, selection, renderer, stream, supported=args.supported,
                             pose_set=args.pose_set, thresholds=thresholds)
        except KeyboardInterrupt:
            print("Pose session stopped. Motors remain off; no offset was applied.", flush=True)
        finally:
            if renderer is not None:
                renderer.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "report":
        from sim_to_real_so101.utils.calibration_report import generate_report
        from sim_to_real_so101.utils.calibration_poses import read_pose_log
        header, _, _ = read_pose_log(args.poses)
        values = dict(header.get("pose_thresholds", {}))
        for key in ("pass_deg", "warn_deg", "spread_deg", "min_poses"):
            value = getattr(args, "pose_" + key)
            if value is not None:
                values[key] = value
        result = generate_report(args.poses, args.gap, args.output, PoseThresholds(**values))
        print(f"Calibration: {result['calibration']['decision']}. Reports: {args.output}/report.md and report.json")
    elif args.mode == "lock":
        from sim_to_real_so101.utils.calibration_report import lock_report
        lock_report(args.report)
    elif args.mode == "watch":
        watch_log(args.log)
    elif args.mode == "poses":
        poses(args, parser)
    else:
        if args.output and args.output.resolve() == args.log.resolve():
            parser.error("--output must not replace the raw log")
        thresholds = Thresholds(pass_deg=args.pass_deg, warn_deg=args.warn_deg,
                                static_velocity_deg_s=args.static_velocity_deg_s, static_window_s=args.static_window_s)
        result = summarize(args.log, thresholds)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, joint in {**result["joints"], **({"gripper": result["gripper"]} if result["gripper"] else {})}.items():
            gap = joint["static_not_at_limit"]["sim_minus_real_deg"]
            print(f"{name}: mean={gap['mean_signed_deg']}° mean_abs={gap['mean_abs_deg']}° "
                  f"n={gap['count']} real-tracking={joint['real_tracking_verdict']} sim-tracking={joint['sim_tracking_verdict']}")
        print(f"Reports: {args.log}.summary.json and {args.log}.summary.md")


if __name__ == "__main__":
    main()
