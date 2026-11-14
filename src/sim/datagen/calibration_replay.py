"""Replay audited device commands under a selected mapping; keep physical successes.

Only the initial robot/object poses are restored. All subsequent object motion is
physical. Source hashes and saved source calibration remain strict audit checks.
"""
import argparse
import json
import os
from pathlib import Path

from sim import apply_selection, close_app, launch
from sim.datagen.scripted import add_generation_launch_arguments, make_report, publish_report, reset_task_tracking, setup_context


def build_parser():
    from utils.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source_repo_id")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo_id")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected_source_episodes", type=int, default=0, help="Optional explicit audit count")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1984)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disable_fabric", action="store_true")
    for key, default in (("arm", DEFAULT_ARM), ("scene", DEFAULT_SCENE), ("task", DEFAULT_TASK)):
        parser.add_argument("--" + key, default=os.environ.get("S2R_" + key.upper(), default))
    parser.add_argument("--variant", default="teleop")
    parser.add_argument("--unit", help="Target unit; source interpretation comes from the audit")
    add_generation_launch_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.destination.exists():
        raise FileExistsError(args.destination)
    source_root = args.source.resolve()
    if args.destination.resolve().is_relative_to(source_root) or args.report.resolve().is_relative_to(source_root):
        raise ValueError("Replay destination and report must be outside the audited source dataset")
    if args.limit < 0 or args.expected_source_episodes < 0 or args.fps <= 0:
        raise ValueError("Episode limits/counts must be nonnegative; FPS must be positive")
    args.repo_id = args.repo_id or f"local/{args.destination.name}"
    args.source_repo_id = args.source_repo_id or f"local/{args.source.name}"
    args.starts_dir, args.control_offset, args.expert_options = None, None, {}
    args.warmup_steps = 0
    apply_selection(args)
    app = launch(args, enable_cameras=True)
    context = None
    report = None
    completed = False
    try:
        import numpy as np
        import torch
        from utils.arms.mapping import physical_to_units, units_to_physical
        from utils.arms.arms import load_arm_spec
        from utils.datasets.sources import load_source, recorded_joint_mapping, _mapping_bounds
        from utils.hashing import tree_sha256
        from sim.workspace.envs.mdp._common import as_torch
        from sim.recording.replay_scene import SceneReplayer

        arm, _ = load_arm_spec(args.arm)
        audit = json.loads(args.audit.read_text())
        if tree_sha256(args.source) != audit["source_hashes"]:
            raise RuntimeError("Source differs from the pre-change snapshot")
        saved, saved_path = recorded_joint_mapping(args.source, arm)
        if saved is not None and saved != _mapping_bounds(audit["old_mapping"], arm):
            raise ValueError(f"Audit old_mapping differs from source dataset metadata in {saved_path}")
        from core.tasks.loader import load_task
        task = load_task(args.task)
        source = load_source(str(args.source), args.source_repo_id, set(), arm=arm, task=task)
        count = len(source)
        expected = args.expected_source_episodes or audit.get("expected_source_episodes", count)
        if count != expected or not source:
            raise ValueError(f"Expected {expected} nonempty source episodes; got {count}")
        if any(episode.fps is not None and episode.fps != args.fps for episode in source):
            raise ValueError("Source FPS differs from the replay control FPS")
        source = source[:args.limit] if args.limit else source
        args.max_steps = max(episode.length for episode in source)
        context = setup_context(args, sample_starts=False, source="calibration_replay")
        replayer = SceneReplayer(context.raw, context.mapping, None, context.cameras, task=task.name, arm=arm)
        report = make_report(context, source_provenance={
            "method": "original device commands, selected mapping, physical action replay",
            "source": str(args.source.resolve()), "source_repo_id": args.source_repo_id,
            "audit": str(args.audit.resolve()), "source_hashes": audit["source_hashes"],
            "old_mapping": audit["old_mapping"], "expected_source_episodes": expected,
            "selected_source_episodes": [episode.index for episode in source],
            "initial_positions": "Recorded physical joint/object positions; velocities zero",
        })
        publish_report(context, report)
        with torch.inference_mode():
            for attempt, episode in enumerate(source):
                context.writer.task_name = episode.task
                context.env.reset()
                conversion = dict(joint_order=arm.joint_names, gripper_joints=arm.gripper_joint_names)
                initial = physical_to_units(units_to_physical(episode.states[0], audit["old_mapping"], **conversion),
                                            context.calibration.mapping, **conversion)
                replayer.pose_frame(initial, {name: poses[0] for name, poses in episode.objects_base.items()})
                q = context.mapping.to_sim(initial).reshape(1, -1)
                replayer.robot.set_joint_position_target_index(target=q, joint_ids=replayer.joint_ids, env_ids=replayer.env_ids)
                context.raw.scene.write_data_to_sim()
                context.raw.sim.forward()
                context.raw.scene.update(0.)
                reset_task_tracking(context.raw)
                # Flush the restored initial poses to RTX before the first
                # pre-action frame; subsequent images come from physical steps.
                replayer.render_images()
                rollout = context.begin_attempt(args.seed + attempt, reset=False, warmup=False)
                maximum_errors = torch.zeros_like(context.mapping.mins)
                try:
                    for frame, action in enumerate(episode.actions):
                        target = context.mapping.to_sim(action)
                        rollout.step(target, action_raw=action, phase="calibration_replay")
                        measured = as_torch(replayer.robot.data.joint_pos)[0, replayer.joint_ids]
                        maximum_errors = torch.maximum(maximum_errors, torch.abs(target - measured))
                        if frame % 100 == 0:
                            print(f"[REPLAY_FRAME] episode={episode.index} frame={frame}/{episode.length}", flush=True)
                    row = rollout.recorder.finish(attempt=attempt, seed=args.seed + attempt, reason="source_trajectory_complete")
                except BaseException:
                    rollout.recorder.abort()
                    raise
                tracking = {joint.name: value * 180 / np.pi if joint.type == "revolute" else value
                            for joint, value in zip(arm.action_joints, maximum_errors.cpu().tolist())}
                row.update(source_episode=episode.index, maximum_tracking_error_by_joint=tracking,
                           maximum_tracking_error_degrees=max((tracking[j.name] for j in arm.action_joints if j.type == "revolute"), default=0.),
                           maximum_tracking_error_metres=max((tracking[j.name] for j in arm.action_joints if j.type == "prismatic"), default=0.),
                           tracking_units={joint.name: joint.unit for joint in arm.action_joints})
                report["episodes"].append(row)
                publish_report(context, report)
                print("[REPLAY_EPISODE] " + json.dumps(row), flush=True)
        unchanged = tree_sha256(args.source) == audit["source_hashes"]
        report["source_provenance"]["source_unchanged"] = unchanged
        if not unchanged:
            raise RuntimeError("Original source changed during replay")
        report["complete"] = True
        publish_report(context, report)
        completed = True
        return 0
    finally:
        try:
            if context is not None:
                try:
                    if report is not None:
                        publish_report(context, report)
                finally:
                    context.close()
        finally:
            close_app(app, success=completed)


if __name__ == "__main__":
    raise SystemExit(main())
