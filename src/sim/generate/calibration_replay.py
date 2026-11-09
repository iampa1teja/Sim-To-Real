# sim/generate/calibration_replay.py  (ported from v1 scripts/calibration_replay.py)
"""Method B: replay original commands under the new mapping and retain physical successes.

Only the initial robot/cube poses are restored. Subsequent cube motion is
simulated, never copied from the source trajectory. The existing benchmark
requires contact acquisition, lift, release inside the box, and 25 stable ticks.

  calibration_replay --source <dataset> --destination <new root> --audit <audit.json> --report <report.json>
"""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Joint units go through core.arms.mapping.JointMapping (v1 LeRobotSO101Interface is gone); old -> new mapping
#   relabeling uses the local _units_to_degrees / _degrees_to_units (v1 utils/calibration_dataset was not ported).
# - Source snapshot hashes come from core.utils.hashing.tree_sha256 (same keys as v1 snapshot on Linux).
# - Success params come from the task (tasks/pick_place_cube/env_cfg.py) instead of PickPlaceTerminationsCfg.
# - Live quaternions are XYZW; sidecar poses are written WXYZ (v1 format).
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

EXPECTED_SOURCE_EPISODES = 75
SOURCE_REPO_ID = "local/pick_place_v1"


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

def _bounds(mapping):
    import numpy as np

    from core.arms.joint_calibration import FOLLOWER_JOINTS, GRIPPER_JOINT

    low = np.array([mapping[n]["joint_min"] for n in FOLLOWER_JOINTS], dtype=np.float64)
    high = np.array([mapping[n]["joint_max"] for n in FOLLOWER_JOINTS], dtype=np.float64)
    if not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Mapping bounds must be finite and increasing")
    gripper = np.array([n == GRIPPER_JOINT for n in FOLLOWER_JOINTS])
    # Body joints span -100..100 raw units, the gripper 0..100.
    return low, high, np.where(gripper, 100.0, 200.0), np.where(gripper, 0.0, 100.0)


def _units_to_degrees(units, mapping):
    import numpy as np

    low, high, scale, offset = _bounds(mapping)
    return low + (np.asarray(units, dtype=np.float64) + offset) / scale * (high - low)


def _degrees_to_units(degrees, mapping):
    import numpy as np

    low, high, scale, offset = _bounds(mapping)
    return (np.asarray(degrees, dtype=np.float64) - low) / (high - low) * scale - offset


def main(argv=None):
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1984)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.destination.exists():
        raise FileExistsError(args.destination)
    args.enable_cameras = True
    args.num_envs = 1
    args.variant = "teleop"
    args.fps = 30
    args.kit_args = rendering_kit_args(args.kit_args)
    app = AppLauncher(args).app
    env = writer = None
    try:
        import numpy as np
        import torch
        import isaaclab.utils.math as math_utils

        import sim.envs
        from core.arms.mapping import JointMapping
        from core.tasks.loader import load_task_module
        from core.utils.hashing import tree_sha256
        from sim.envs.mdp.resets import as_torch
        from sim.generate.dr_replay import discover_sim_cameras, make_env
        from sim.recording.episode_metadata import save_cube_trajectory
        from sim.recording.lerobot_recorder import LeRobotRecorder
        from sim.recording.replay_scene import SceneReplayer
        from sim.recording.replay_source import load_source

        task = sim.envs.TASK
        mdp = load_task_module("mdp", task.name)
        env_cfg_module = load_task_module("env_cfg", task.name)
        audit = json.loads(args.audit.read_text())
        if tree_sha256(args.source) != audit["source_hashes"]:
            raise RuntimeError("Source differs from the pre-change snapshot")
        source = load_source(str(args.source), SOURCE_REPO_ID, set())
        if len(source) != EXPECTED_SOURCE_EPISODES:
            raise ValueError(f"Expected {EXPECTED_SOURCE_EPISODES} source episodes; got {len(source)}")
        if args.limit:
            source = source[:args.limit]
        env = make_env(args)
        raw = env.unwrapped
        cameras = discover_sim_cameras(env)
        mapping = raw.cfg.sim_joint_mapping
        joints = JointMapping.from_setup(SimpleNamespace(mapping=mapping), device=raw.device)
        replayer = SceneReplayer(raw, joints, None, cameras, task=task.name)
        writer = LeRobotRecorder(task_name=source[0].task, repo_id="local/pick_place_v1_calibrated_B",
                                 dataset_root=str(args.destination), fps=args.fps, device=raw.device, cameras=cameras)
        writer.init_dataset()
        success_params = dict(env_cfg_module._place_params(task),
                              confirm_steps=task.spec.raw.get("success", {}).get("confirm_steps", 25))
        report = dict(method="B: original units, new mapping, physics action replay",
                      expected_source_episodes=EXPECTED_SOURCE_EPISODES,
                      source=str(args.source), destination=str(args.destination), old_mapping=audit["old_mapping"],
                      new_mapping=mapping, initial_positions="Recorded physical joint and cube positions; velocities zero",
                      success_criterion=({k: v for k, v in success_params.items() if k != "contact_sensor_cfg"}),
                      control_fps=args.fps, seed=args.seed, episodes=[], complete=False)

        def publish():
            report["successes"] = sum(row["success"] for row in report["episodes"])
            args.report.write_text(json.dumps(report, indent=2) + "\n")

        robot, cube = replayer.robot, replayer.cube
        with torch.inference_mode():
            for episode in source:
                env.reset()
                initial = _degrees_to_units(_units_to_degrees(episode.states[0], audit["old_mapping"]), mapping)
                replayer.pose_frame(initial, episode.poses_base[0])
                q = joints.to_radians(initial).reshape(1, -1)
                robot.set_joint_position_target_index(target=q, joint_ids=replayer.joint_ids, env_ids=replayer.env_ids)
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                raw.scene.update(0.)
                mdp.reset_pick_place_state(raw, torch.arange(raw.num_envs, device=raw.device))
                poses, success, first_success = [], False, None
                maximum_tracking = 0.
                for frame, action in enumerate(episode.actions):
                    joint_pos = as_torch(robot.data.joint_pos)[0, replayer.joint_ids]
                    state = joints.to_raw(joint_pos)
                    images = replayer.render_images()
                    writer.add_frame(writer.make_sim_frame(action, state, images))
                    base_pos = as_torch(robot.data.body_link_pos_w)[:, replayer.base_body_id]
                    base_quat = as_torch(robot.data.body_link_quat_w)[:, replayer.base_body_id]
                    position, quaternion = math_utils.subtract_frame_transforms(
                        base_pos, base_quat, as_torch(cube.data.root_pos_w), as_torch(cube.data.root_quat_w))
                    # Sidecars keep v1's WXYZ order.
                    poses.append(torch.cat((position[0], quaternion[0, 3:4], quaternion[0, :3])).cpu().tolist())
                    target = joints.to_radians(action)
                    env.step(target.reshape(1, -1))
                    joint_pos = as_torch(robot.data.joint_pos)[0, replayer.joint_ids]
                    maximum_tracking = max(maximum_tracking,
                                           float(torch.max(torch.abs(target - joint_pos)).cpu()) * 180 / np.pi)
                    if bool(mdp.object_placed_in_container_termination(raw, **success_params)[0]):
                        success = True
                        if first_success is None:
                            first_success = frame
                    if frame % 100 == 0:
                        print(f"[REPLAY_FRAME] episode={episode.index} frame={frame}/{episode.length}", flush=True)
                destination_index = None
                if success:
                    destination_index = writer.save_episode()
                    save_cube_trajectory(args.destination, destination_index, poses, episode.cube_size, args.fps)
                else:
                    writer.clear_episode_buffer()
                row = dict(source_episode=episode.index, success=success, destination_episode=destination_index,
                           frames=episode.length, first_success_frame=first_success,
                           ever_grasped=bool(raw._pick_place_ever_grasped[0]),
                           maximum_tracking_error_degrees=maximum_tracking)
                report["episodes"].append(row)
                publish()
                print("[REPLAY_EPISODE] " + json.dumps(row), flush=True)
        report["source_unchanged"] = tree_sha256(args.source) == audit["source_hashes"]
        if not report["source_unchanged"]:
            raise RuntimeError("Original source changed during replay")
        report["complete"] = len(report["episodes"]) == EXPECTED_SOURCE_EPISODES
        publish()
        (args.destination / "meta/calibration_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.finalize()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
