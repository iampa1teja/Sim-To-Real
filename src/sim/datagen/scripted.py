"""Generate physical expert rollouts and commit only task-confirmed successes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

from core.tasks.base import Pose
from core.tasks.expert import ExpertContext
from core.utils.config import parse_yaml_args
from sim.app import add_launch_arguments, apply_selection, close_app, launch
from sim.datagen.rollout import LiveRollout, SuccessRecorder, write_json


def add_generation_launch_arguments(parser):
    add_launch_arguments(parser)
    if not any(action.dest == "headless" for action in parser._actions):
        parser.add_argument("--headless", action="store_true")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--source", choices=("scripted",), default="scripted")
    for key, default in (("arm", "so101"), ("scene", "rpd_meeting_room"), ("task", "pick_place_cube")):
        parser.add_argument("--" + key, default=os.environ.get("S2R_" + key.upper(), default))
    parser.add_argument("--variant", default="teleop")
    parser.add_argument("--unit")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo_id", default="local/scripted")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1, help="Stop after this many accepted episodes; zero runs all attempts")
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1984)
    parser.add_argument("--starts_dir", type=Path)
    parser.add_argument("--expert_options", default="{}", help="Task-specific expert options as a JSON object")
    parser.add_argument("--control_offset", default=None, help="Optional EE-local TCP offset in metres, JSON vector")
    parser.add_argument("--disable_fabric", action="store_true")
    add_generation_launch_arguments(parser)
    return parser


def parse_args(argv=None, *, parser=None):
    parser = parser or build_parser()
    args = parse_yaml_args(parser, argv, json_fields=("expert_options",))
    if min(args.attempts, args.max_steps, args.fps) < 1 or args.episodes < 0 or args.warmup_steps < 0:
        parser.error("attempts/max_steps/fps must be positive; episodes/warmup_steps must be nonnegative")
    args.expert_options = json.loads(args.expert_options) if isinstance(args.expert_options, str) else args.expert_options
    if not isinstance(args.expert_options, dict):
        parser.error("expert_options must be a JSON object")
    if isinstance(args.control_offset, str):
        args.control_offset = json.loads(args.control_offset)
    if args.control_offset is not None and (np.shape(args.control_offset) != (3,) or not np.isfinite(args.control_offset).all()):
        parser.error("control_offset must contain three finite metres")
    return args


def reset_task_tracking(raw):
    term = getattr(raw.cfg.events, "reset_tracking", None)
    if term is not None:
        prepared = raw.event_manager.get_term_cfg("reset_tracking")
        prepared.func(raw, None, **prepared.params)


def _matrix_pose(matrix):
    import torch
    from isaaclab.utils.math import quat_from_matrix
    quaternion = quat_from_matrix(torch.as_tensor(matrix[:3, :3], dtype=torch.float64, device="cpu"))
    return Pose.from_lists(matrix[:3, 3], quaternion.numpy())


@dataclass
class GenerationContext:
    args: object
    env: object
    task: object
    arm: object
    mapping: object
    calibration: object
    cameras: dict
    writer: object
    ik: object
    kinematics: object
    source: str = "scripted"

    @property
    def raw(self):
        return getattr(self.env, "unwrapped", self.env)

    def target_pose(self, target):
        from core.scenes import quaternion_multiply
        from core.benchmark.geometry import rotation_matrix

        matrix = self.kinematics.fk(LiveRollout._numpy(target).reshape(-1))
        local = _matrix_pose(matrix)
        robot = self.raw.scene["robot"]
        base_id = robot.find_bodies(self.arm.links.base)[0][0]
        p = LiveRollout._numpy(robot.data.body_link_pos_w)[0, base_id]
        q = LiveRollout._numpy(robot.data.body_link_quat_w)[0, base_id]
        world = p + rotation_matrix((q[3], q[0], q[1], q[2])) @ local.position
        return Pose.from_lists(world, quaternion_multiply(q, local.quat_xyzw))

    def gripper_target(self, target, fraction):
        """Interpolate each gripper's declared physical closed/open codec endpoints."""
        result = target.clone()
        raw = self.mapping.to_raw(result)
        ids = [index for index, joint in enumerate(self.arm.joints) if joint.role == "gripper"]
        raw[:, ids] = self.mapping.raw_mins[ids] + float(fraction) * (
            self.mapping.raw_maxs[ids] - self.mapping.raw_mins[ids])
        decoded = self.mapping.to_sim(raw)
        result[:, ids] = decoded[:, ids]
        return result

    def begin_attempt(self, seed, *, reset=True, warmup=True, observer=None):
        import torch
        from sim.envs.mdp._common import as_torch

        self.raw.cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if reset:
            self.env.reset()
        if warmup:
            ids = self.raw.scene["robot"].find_joints(list(self.arm.usd_joint_names), preserve_order=True)[0]
            hold = as_torch(self.raw.scene["robot"].data.joint_pos)[:, ids].clone()
            hold = self.gripper_target(hold, self.args.expert_options.get("gripper_open_fraction", .45))
            for _ in range(self.args.warmup_steps):
                self.env.step(hold)
            reset_task_tracking(self.raw)
        sizes = {name: self.task.spec.objects[name].size or self.task.spec.objects[name].params.get("size_outer")
                 for name in self.task.spec.tracked_objects}
        recorder = SuccessRecorder(self.writer, self.task, self.arm, sizes, self.args.fps,
                                   control_offset_m=self.ik.position_offset_m)
        return LiveRollout(self.env, self.task, self.arm, self.mapping, recorder,
                           control_offset=self.ik.position_offset_m, observer=observer, target_pose_fn=self.target_pose)

    def close(self):
        try:
            self.writer.finalize()
        finally:
            self.env.close()


def setup_context(args, *, env_factory=None, sample_starts=True, source=None):
    """Build a task environment after Kit launch; Mimic can supply its real env class."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import EventTermCfg
    import sim.envs
    from core.arms.spec import load_arm_spec
    from core.arms.calibration import resolve_calibration
    from core.arms.mapping import JointMapping
    from core.datasets.alignment import mark_recording_root, recording_contract
    from sim.envs.mdp.starts import reset_from_starts
    from sim.envs.scene_builder import drop_unused_camera_outputs
    from sim.datagen.dr_replay import discover_sim_cameras
    from sim.datagen.ik import ArticulationIK, ArmKinematics
    from sim.recording.lerobot_recorder import LeRobotRecorder

    if Path(args.destination).exists():
        raise FileExistsError(args.destination)
    task = sim.envs.register_envs(scene=args.scene, task=args.task)
    arm, _ = load_arm_spec(args.arm)
    calibration = resolve_calibration(arm, unit=args.unit)
    cfg = parse_env_cfg(task.gym_id(args.variant), device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    cfg.seed = args.seed
    cfg.sim_joint_mapping = calibration.mapping
    cfg.terminations = None
    cfg.episode_length_s = (args.max_steps + getattr(args, "warmup_steps", 0) + 1) / args.fps
    if not math.isclose(1 / (cfg.decimation * cfg.sim.dt), args.fps, abs_tol=1e-6):
        raise ValueError("Generation FPS must match the task's physical control timestep")
    cfg.sim.render_interval = cfg.decimation
    cfg.events.reset_from_starts = (EventTermCfg(func=reset_from_starts, mode="reset", params={
        "task": task.name, "starts_dir": str(args.starts_dir) if args.starts_dir else None,
        "mode": "random", "seed": None, "reset_robot": False,
    }) if sample_starts else None)
    drop_unused_camera_outputs(cfg)
    env = env_factory(cfg) if env_factory is not None else gym.make(task.gym_id(args.variant), cfg=cfg)
    raw = getattr(env, "unwrapped", env)
    writer = None
    try:
        cameras = discover_sim_cameras(env)
        mapping = JointMapping.from_arm(arm, calibration.mapping, device=raw.device)
        writer = LeRobotRecorder(task_name=task.instruction, repo_id=args.repo_id, dataset_root=str(args.destination),
                                 fps=args.fps, device=raw.device, cameras=cameras, arm=arm)
        writer.init_dataset()
        producer = source or getattr(args, "source", "scripted")
        contract = recording_contract(follower_id=None, leader_id=f"datagen:{producer}", calibration_selection=calibration)
        contract.update(start_sync="sim_state", wrist_rezero=False, sim_wrist_relative=False,
                        recorded_action="absolute_sim_target_in_device_units",
                        note="Generated actions and observations use the selected device codec; no relative teleop wrist shift.")
        mark_recording_root(args.destination, contract)
        offset = getattr(args, "control_offset", None)
        ik = ArticulationIK(raw.scene["robot"], arm, position_offset_m=offset)
        kinematics = ArmKinematics.from_arm(arm, position_offset_m=ik.position_offset_m)
        return GenerationContext(args, env, task, arm, mapping, calibration, cameras, writer, ik, kinematics, producer)
    except BaseException:
        try:
            if writer is not None:
                writer.finalize()
        finally:
            env.close()
        raise


def make_report(context, *, source_provenance=None):
    from core.utils.provenance import git_commit
    arm, args = context.arm, context.args
    return {"schema_version": 1, "source": context.source, "dataset": str(Path(args.destination).resolve()),
            "repo_id": args.repo_id, "arm": arm.name, "task": context.task.name, "scene": args.scene,
            "variant": args.variant, "fps": args.fps, "joint_names": list(arm.joint_names),
            "joint_units": [joint.unit for joint in arm.joints], "joint_roles": [joint.role for joint in arm.joints],
            "joint_mapping": context.calibration.mapping, "calibration": context.calibration.provenance,
            "control_offset_m": np.asarray(context.ik.position_offset_m).tolist(),
            "seed": args.seed, "git_commit": git_commit(Path(__file__).resolve().parents[3]),
            "source_provenance": source_provenance or {}, "episodes": [], "accepted_episodes": 0,
            "requested_episodes": getattr(args, "episodes", None),
            "attempt_limit": getattr(args, "attempts", None), "attempts_completed": 0, "complete": False}


def publish_report(context, report):
    report["accepted_episodes"] = sum(row["accepted"] for row in report["episodes"])
    report["attempts_completed"] = len(report["episodes"])
    destination = context.args.report or Path(str(context.args.destination) + ".report.json")
    write_json(destination, report)
    write_json(Path(context.args.destination) / "meta/datagen.json", report)


def execute_expert(context, rollout, expert):
    from sim.eval.scene import capture_scene

    initial = rollout.snapshot()
    geometry = capture_scene(context.raw, context.task)
    a, b, _ = geometry["table_plane_world"]
    normal = np.array([-a, -b, 1.])
    normal /= np.linalg.norm(normal)
    waypoints = expert.plan(ExpertContext(initial, rollout.control_pose(initial), tuple(normal), context.args.expert_options))
    dt = 1 / context.args.fps
    steps = 0
    for waypoint in waypoints:
        start = rollout.control_pose(rollout.last_state or initial)
        for tick in range(1, math.ceil(waypoint.timeout_s / dt) + 1):
            elapsed = tick * dt
            alpha = min(1., elapsed / waypoint.duration_s)
            alpha = alpha * alpha * (3 - 2 * alpha)
            position = np.asarray(start.position) * (1 - alpha) + np.asarray(waypoint.pose.position) * alpha
            first, last = np.asarray(start.quat_xyzw), np.asarray(waypoint.pose.quat_xyzw)
            last = last if np.dot(first, last) >= 0 else -last
            quaternion = first * (1 - alpha) + last * alpha
            quaternion /= np.linalg.norm(quaternion)
            solved = context.ik.solve(position, quaternion)
            target = context.gripper_target(solved.joint_positions, waypoint.gripper_open_fraction)
            rollout.step(target, target_eef_pose=Pose.from_lists(position, quaternion), phase=waypoint.name)
            steps += 1
            if context.task.is_success(rollout.recorder.trace.stages):
                return "task_success"
            if steps >= context.args.max_steps:
                return "max_steps"
            if elapsed >= waypoint.duration_s:
                final = context.ik.solve(waypoint.pose.position, waypoint.pose.quat_xyzw)
                if (float(final.position_error[0]) <= waypoint.tolerance_m and
                        float(final.orientation_error[0]) <= context.ik.options.orientation_tolerance_rad):
                    break
        else:
            return "waypoint_timeout:" + waypoint.name
    return "expert_finished_without_success"


def main(argv=None):
    args = parse_args(argv)
    apply_selection(args)
    app = launch(args, enable_cameras=True)
    context = None
    report = None
    succeeded = False
    try:
        import torch

        context = setup_context(args)
        expert = context.task.expert()
        report = make_report(context, source_provenance={"expert": str(context.task.spec.task_dir / "expert.py"),
                                                        "options": args.expert_options})
        publish_report(context, report)
        with torch.inference_mode():
            for attempt in range(args.attempts):
                seed = args.seed + attempt
                rollout = context.begin_attempt(seed)
                try:
                    reason = execute_expert(context, rollout, expert)
                    row = rollout.recorder.finish(attempt=attempt, seed=seed, reason=reason)
                except BaseException:
                    rollout.recorder.abort()
                    raise
                report["episodes"].append(row)
                publish_report(context, report)
                print("[DATAGEN_ATTEMPT] " + json.dumps(row), flush=True)
                if args.episodes and report["accepted_episodes"] >= args.episodes:
                    break
        succeeded = report["accepted_episodes"] > 0 and (not args.episodes or report["accepted_episodes"] >= args.episodes)
        report["attempts_completed"] = len(report["episodes"])
        report["requested_episodes"] = args.episodes
        report["complete"] = (report["accepted_episodes"] >= args.episodes if args.episodes else
                              len(report["episodes"]) == args.attempts)
        publish_report(context, report)
        return 0 if succeeded else 1
    finally:
        try:
            if context is not None:
                try:
                    if report is not None:
                        publish_report(context, report)
                finally:
                    context.close()
        finally:
            close_app(app, success=succeeded)


if __name__ == "__main__":
    raise SystemExit(main())
