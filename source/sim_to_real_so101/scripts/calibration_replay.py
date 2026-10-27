"""Method B: replay original commands under the new mapping and retain physical successes.

Only the initial robot/cube poses are restored. Subsequent cube motion is
simulated, never copied from the source trajectory. The existing benchmark
requires contact acquisition, lift, release inside the box, and 25 stable ticks.
"""
import argparse
import json
from pathlib import Path
import traceback


def main():
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
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    args.enable_cameras = True
    args.num_envs = 1
    args.task = "Lerobot-So101-Teleop-Pick-Place"
    args.fps = 30
    app = AppLauncher(args).app
    env = writer = None
    try:
        import numpy as np
        import torch
        from isaaclab.managers import SceneEntityCfg
        from sim_to_real_so101.scripts.dr_replay import make_env, make_interface, discover_sim_cameras
        from sim_to_real_so101.utils.calibration_dataset import snapshot, units_to_degrees, degrees_to_units
        from sim_to_real_so101.utils.replay_source import load_source
        from sim_to_real_so101.utils.replay_scene import SceneReplayer
        from sim_to_real_so101.utils.replay_math import world_to_base
        from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder
        from sim_to_real_so101.utils.episode_metadata import save_cube_trajectory
        from sim_to_real_so101.mdp.terms import reset_pick_place_state, object_placed_in_container_termination
        from sim_to_real_so101.tasks.pick_place_env_cfg import PickPlaceTerminationsCfg
        audit = json.loads(args.audit.read_text())
        if snapshot(args.source) != audit["source_hashes"]:
            raise RuntimeError("Source differs from the pre-change snapshot")
        source = load_source(str(args.source), "local/pick_place_v1", set())
        if len(source) != 75:
            raise ValueError(f"Expected 75 source episodes; got {len(source)}")
        if args.limit:
            source = source[:args.limit]
        env = make_env(args)
        raw = env.unwrapped
        cameras = discover_sim_cameras(env)
        interface = make_interface(env, cameras, args.fps)
        mapping = raw.cfg.sim_joint_mapping
        replayer = SceneReplayer(raw, interface, None)
        writer = LeRobotRecorder(task_name=source[0].task, repo_id="local/pick_place_v1_calibrated_B",
                                 dataset_root=str(args.destination), fps=30, device=raw.device, cameras=cameras)
        writer.init_dataset()
        success_params = dict(PickPlaceTerminationsCfg().success.params)
        success_params["contact_sensor_cfg"] = SceneEntityCfg("contact_grasp")
        report = dict(method="B: original units, new mapping, physics action replay", expected_source_episodes=75,
                      source=str(args.source), destination=str(args.destination), old_mapping=audit["old_mapping"],
                      new_mapping=mapping, initial_positions="Recorded physical joint and cube positions; velocities zero",
                      success_criterion=({k: v for k, v in success_params.items() if k != "contact_sensor_cfg"}),
                      control_fps=30, seed=args.seed, episodes=[], complete=False)
        def publish():
            report["successes"] = sum(row["success"] for row in report["episodes"])
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        with torch.inference_mode():
            for episode in source:
                env.reset()
                initial = degrees_to_units(units_to_degrees(episode.states[0], audit["old_mapping"]), mapping)
                replayer.pose_frame(initial, episode.poses_base[0])
                q = interface.get_mapped_actions_vectorized(torch.tensor(initial, device=raw.device, dtype=torch.float32))
                replayer.robot.set_joint_position_target(q.reshape(1, -1))
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                raw.scene.update(0.)
                reset_pick_place_state(raw, torch.arange(raw.num_envs, device=raw.device))
                poses, success, first_success = [], False, None
                maximum_tracking = 0.
                for frame, action in enumerate(episode.actions):
                    state = interface.get_raw_actions_from_radians(replayer.robot.data.joint_pos[0])
                    images = replayer.render_images()
                    writer.add_frame(writer.make_sim_frame(action, state, images))
                    bp = replayer.robot.data.body_pos_w[0, replayer.base_body_id]
                    bq = replayer.robot.data.body_quat_w[0, replayer.base_body_id]
                    position, quaternion = world_to_base(bp, bq, replayer.cube.data.root_pos_w[0],
                                                         replayer.cube.data.root_quat_w[0])
                    poses.append(torch.cat((position, quaternion)).cpu().tolist())
                    target = interface.get_mapped_actions_vectorized(torch.tensor(action, device=raw.device))
                    env.step(target.reshape(1, -1))
                    maximum_tracking = max(maximum_tracking, float(torch.max(torch.abs(
                        target - replayer.robot.data.joint_pos[0])).cpu()) * 180 / np.pi)
                    if bool(object_placed_in_container_termination(raw, **success_params)[0]):
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
        report["source_unchanged"] = snapshot(args.source) == audit["source_hashes"]
        if not report["source_unchanged"]:
            raise RuntimeError("Original source changed during replay")
        report["complete"] = len(report["episodes"]) == 75
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
