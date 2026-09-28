# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-process SO-101 real + simulation teleoperation and recording."""

import argparse
from importlib import import_module
import os

from isaaclab.app import AppLauncher
from sim_to_real_so101.utils.recording_setup import (
    cleanup as _cleanup,
    connect_interface,
    exit_cleanly_on_termination,
    disconnect as _disconnect,
    make_recorders,
    real_camera_specs,
    validate_serial_ports,
)


parser = argparse.ArgumentParser(description="Isaac Lab SO-101 teleoperation agent.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-MyRoom")
parser.add_argument("--spawn_port", type=int, default=6001,
                    help="Local spawn.py listener port; 0 disables it.")
parser.add_argument("--spawn_authkey", default="so101-spawn")
parser.add_argument("--control_input", choices=("terminal", "keyboard"), default="terminal",
                    help="Episode controls: Python input() commands or the original keyboard listener.")

# Physical leader
parser.add_argument(
    "--port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Physical leader/teleop serial port.",
)
parser.add_argument(
    "--robot_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="Physical leader/teleop calibration ID.",
)

# Physical follower
parser.add_argument(
    "--enable_real_follower",
    action="store_true",
    default=False,
    help="Send the original leader action to a physical follower.",
)
parser.add_argument(
    "--follower_port",
    type=str,
    default=os.getenv("ROBOT_PORT", "/dev/ttyACM1"),
    help="Physical follower serial port.",
)
parser.add_argument(
    "--follower_id",
    type=str,
    default=os.getenv("ROBOT_ID", "follower_arm_1"),
    help="Physical follower calibration ID.",
)

# LeRobot datasets. The existing repo arguments continue to name the sim
# dataset. Real defaults are derived by appending "-real".
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--real_repo_id", type=str, default=None)
parser.add_argument("--real_repo_root", type=str, default=None)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--instance_id_seg", action="store_true", default=False)
parser.add_argument(
    "--image_writer_threads_per_camera",
    type=int,
    default=4,
    help="LeRobot asynchronous PNG writer threads per camera.",
)

# Real cameras are attached to the physical follower through the official
# SO101FollowerConfig.cameras field. Unset sources simply omit that camera.
parser.add_argument(
    "--real_gripper_camera",
    type=str,
    default=os.getenv("CAMERA_GRIPPER"),
    help="OpenCV index or /dev/video path for the real gripper camera.",
)
parser.add_argument(
    "--real_external_camera",
    type=str,
    default=os.getenv("CAMERA_EXTERNAL"),
    help="OpenCV index or /dev/video path for the real external camera.",
)
parser.add_argument(
    "--real_camera_width",
    type=int,
    default=int(os.getenv("CAMERA_WIDTH", "640")),
)
parser.add_argument(
    "--real_camera_height",
    type=int,
    default=int(os.getenv("CAMERA_HEIGHT", "480")),
)
parser.add_argument(
    "--real_camera_fourcc",
    type=str,
    default=os.getenv("CAMERA_FOURCC") or None,
)

parser.add_argument(
    "--fps",
    type=int,
    default=int(os.getenv("CONTROL_FPS", "30")),
    help="One master teleoperation and recording loop rate.",
)
parser.add_argument("--seed", type=int, default=101)

# MyRoom currently has no task RigidObject. This optional hook only acts on a
# caller-supplied Isaac Lab scene entity and therefore invents no prim path.
parser.add_argument(
    "--sim_reset_object_name",
    type=str,
    default=None,
    help="Existing Isaac Lab scene entity to place near the simulated gripper.",
)
parser.add_argument(
    "--sim_reset_object_offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="World-frame XYZ offset in metres from the simulated gripper.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

dataset_args = (args_cli.repo_id, args_cli.repo_root, args_cli.task_name)
if any(dataset_args) and not all(dataset_args):
    parser.error("--repo_id, --repo_root, and --task_name must be supplied together.")
if (args_cli.real_repo_id or args_cli.real_repo_root) and not all(dataset_args):
    parser.error("Real dataset overrides require the base dataset arguments.")
if args_cli.fps <= 0:
    parser.error("--fps must be positive.")
if args_cli.image_writer_threads_per_camera < 0:
    parser.error("--image_writer_threads_per_camera cannot be negative.")
if args_cli.enable_real_follower and os.path.realpath(args_cli.port) == os.path.realpath(args_cli.follower_port):
    parser.error("Leader and follower serial ports must be different.")


validate_serial_ports(args_cli, parser)
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


try:
    import time

    import gymnasium as gym
    import torch

    import_module("isaaclab_tasks")  # Registers Isaac Lab tasks.
    from isaaclab_tasks.utils import parse_env_cfg

    import_module("sim_to_real_so101.tasks")  # Registers workshop tasks.
    from sim_to_real_so101.utils.terminal_control import TerminalInputControl
    from sim_to_real_so101.utils.cube_spawner import CubeSpawnServer, spawn_blue_cube

except BaseException:
    simulation_app.close()
    raise


def _place_configured_object_near_ee(env) -> bool:
    """Place a configured existing scene entity at the simulated EE world pose."""
    object_name = args_cli.sim_reset_object_name
    if not object_name:
        return False

    scene = env.unwrapped.scene
    if object_name not in scene.keys():
        raise ValueError(
            f"Scene entity {object_name!r} does not exist. Available entities: "
            f"{list(scene.keys())}"
        )
    if "ee_frame" not in scene.keys():
        raise ValueError("The task has no 'ee_frame' sensor for object placement.")

    sim_object = scene[object_name]
    if not hasattr(sim_object, "write_root_pose_to_sim"):
        raise TypeError(
            f"Scene entity {object_name!r} is not a writable rigid object/articulation."
        )

    ee_frame = scene["ee_frame"]
    ee_position_w = ee_frame.data.target_pos_w[:, 0, :].clone()
    offset = torch.tensor(
        args_cli.sim_reset_object_offset,
        dtype=ee_position_w.dtype,
        device=ee_position_w.device,
    )
    ee_position_w += offset

    # Retain the object's configured orientation; only spatial placement is
    # coupled to the simulated end effector.
    object_quat_w = sim_object.data.root_quat_w.clone()
    root_pose = torch.cat((ee_position_w, object_quat_w), dim=-1)
    sim_object.write_root_pose_to_sim(root_pose)
    sim_object.write_root_velocity_to_sim(
        torch.zeros((root_pose.shape[0], 6), device=root_pose.device)
    )
    print(
        f"[INFO]: Placed simulation entity {object_name!r} near the end effector "
        f"with world offset {tuple(args_cli.sim_reset_object_offset)}."
    )
    return True


def _reset_for_buffer(env, keyboard_control) -> None:
    env.reset()
    keyboard_control.set_recording(False)
    print(
        "[INFO]: RESET/BUFFER phase. Reposition the real scene and leader, "
        "then use S (Enter in terminal mode) to start the next synchronized episode."
    )


@torch.inference_mode()
def main():
    spawn_server = None
    keyboard_control = None
    env = None
    leader_iface = None
    follower_iface = None
    recorder_group = None
    recording = False

    try:
        exit_cleanly_on_termination()
        # Use Fabric consistently with zero/eval agents so rendered links and
        # the attached wrist camera follow the live articulation.
        use_fabric = not args_cli.disable_fabric
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=use_fabric,
        )
        env_cfg.seed = args_cli.seed
        if env_cfg.scene.num_envs != 1:
            raise ValueError("Manual recorder scripts require num_envs == 1.")
        if env_cfg.terminations is not None and any(
            value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
        ):
            raise ValueError("Manual recording requires a teleop task without automatic terminations.")
        physics_steps = 1.0 / (args_cli.fps * env_cfg.sim.dt)
        if abs(physics_steps - round(physics_steps)) > 1e-6 or physics_steps < 1:
            raise ValueError("--fps must divide the physics rate exactly.")
        env_cfg.decimation = round(physics_steps)
        env_cfg.sim.render_interval = env_cfg.decimation
        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        env.reset()

        sim_cameras = {}
        for obj in env.unwrapped.scene.keys():
            if obj.startswith("camera_"):
                camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
                camera_name = obj.replace("camera_", "")
                sim_cameras[camera_name] = {
                    "height": camera_cfg.height,
                    "width": camera_cfg.width,
                }
                print(f"[INFO]: Found simulation camera: {camera_name}")
        if not sim_cameras:
            print("[WARNING]: No simulation cameras found.")

        leader_iface = connect_interface(
            args_cli, env.unwrapped.device, sim_cameras, "leader"
        )

        recording_mode = bool(all(dataset_args))
        # Preserve follower-only teleoperation: physical cameras are opened
        # only when a real dataset will actually consume them.
        real_cameras = real_camera_specs(args_cli) if recording_mode else {}
        if args_cli.enable_real_follower:
            follower_iface = connect_interface(args_cli, env.unwrapped.device, real_cameras, "follower")

        if recording_mode:
            recorder_group = make_recorders(
                args_cli, env.unwrapped.device, sim_cameras, follower_iface, real_cameras,
            )
            recorder_group.init_datasets()
            print(
                f"[INFO]: Recording ready at synchronized episode "
                f"{recorder_group.episode_index}: "
                f"{ {name: os.fspath(rec.dataset_root) for name, rec in recorder_group.recorders.items()} }"
            )
        else:
            print(
                "[INFO]: Dataset recording disabled. Supply --repo_id, "
                "--repo_root, and --task_name to enable it."
            )

        if args_cli.sim_reset_object_name is None:
            print(
                "[INFO]: Automatic simulation-object placement is disabled. "
                "MyRoom defines no task RigidObject; add one to the task and pass "
                "--sim_reset_object_name to enable the reset hook."
            )
        elif args_cli.sim_reset_object_name not in env.unwrapped.scene.keys():
            raise ValueError(
                f"--sim_reset_object_name={args_cli.sim_reset_object_name!r} is "
                f"not in this task: {list(env.unwrapped.scene.keys())}"
            )

        actions = torch.zeros(
            env.action_space.shape,
            device=env.unwrapped.device,
        )
        wrist_roll_index = 4
        leader_wrist_start = None

        # Start input only after device connection/calibration prompts finish.
        if args_cli.control_input == "keyboard":
            from sim_to_real_so101.utils.keyboard import KeyboardControl
            keyboard_control = KeyboardControl()
            print("[controls] S: start/save; Right: save; Left/C: discard; R: reset; Escape: quit.")
        else:
            keyboard_control = TerminalInputControl()
        print("[INFO]: RESET/BUFFER phase. Use S (Enter in terminal mode) when ready.")

        if args_cli.spawn_port:
            try:
                spawn_server = CubeSpawnServer(args_cli.spawn_port, args_cli.spawn_authkey)
                print(f"[INFO]: spawn.py listener ready on localhost:{args_cli.spawn_port}.")
            except OSError as exc:
                print(f"[WARNING]: Cube spawner unavailable: {exc}. Use --spawn_port to select another port.")

        def spawn_requested_cube():
            eef = env.unwrapped.scene['ee_frame'].data
            return spawn_blue_cube(
                env.unwrapped.sim.stage,
                eef.target_pos_w[0, 0].detach().cpu().tolist(),
                eef.target_quat_w[0, 0].detach().cpu().tolist(),
            )

        while simulation_app.is_running():
            loop_start = time.perf_counter()
            if spawn_server is not None:
                spawn_server.poll(spawn_requested_cube)
            requests = keyboard_control.consume_requests()

            if requests["stop"]:
                if recording and recorder_group is not None:
                    if requests["rerecord"]:
                        recorder_group.cancel_episode()
                    else:
                        recorder_group.save_episode()
                    recording = False
                    keyboard_control.set_recording(False)
                break

            if requests["rerecord"]:
                if recording and recorder_group is not None:
                    recorder_group.cancel_episode()
                recording = False
                _reset_for_buffer(env, keyboard_control)
                leader_wrist_start = None

            elif requests["end"]:
                if recording and recorder_group is not None:
                    recorder_group.save_episode()
                recording = False
                _reset_for_buffer(env, keyboard_control)
                leader_wrist_start = None

            elif requests["reset"]:
                if recording and recorder_group is not None:
                    recorder_group.save_episode()
                recording = False
                _reset_for_buffer(env, keyboard_control)
                leader_wrist_start = None

            if (requests["start"] and not recording
                    and not any(requests[key] for key in ("end", "rerecord", "reset"))):
                if recorder_group is None:
                    print(
                        "[WARNING]: Recording is disabled; restart with dataset "
                        "arguments before pressing S."
                    )
                else:
                    _place_configured_object_near_ee(env)
                    recording = True
                    keyboard_control.set_recording(True)
                    print(
                        f"[INFO]: Recording synchronized episode "
                        f"{recorder_group.episode_index}."
                    )

            try:
                # Keep this dictionary unchanged for the physical follower
                # and the real LeRobot action frame.
                leader_action = leader_iface.robot.get_action()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read the physical leader: {exc}"
                ) from exc

            _, mapped_action = (
                leader_iface.real_to_sim_obs_processor(leader_action)
            )
            if leader_wrist_start is None:
                leader_wrist_start = mapped_action[wrist_roll_index].clone()
                sim_wrist_start = env.unwrapped.observation_manager.compute()["policy"]["joint_pos_obs"][
                    0, wrist_roll_index
                ].clone()
            mapped_action[wrist_roll_index] = (
                sim_wrist_start
                + mapped_action[wrist_roll_index]
                - leader_wrist_start
            )
            mapped_action[wrist_roll_index] = torch.clamp(
                mapped_action[wrist_roll_index],
                leader_iface.joint_mins[wrist_roll_index] * torch.pi / 180,
                leader_iface.joint_maxs[wrist_roll_index] * torch.pi / 180,
            )

            obs = env.unwrapped.observation_manager.compute()

            if recording and recorder_group is not None:
                visual_obs = obs.get("visual")
                if visual_obs is None:
                    raise RuntimeError(
                        "The task has no 'visual' observation group required "
                        "for simulation recording."
                    )

                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                (
                    sim_observation,
                    visual_buffers,
                    depth_buffers,
                    instance_id_seg_buffers,
                ) = leader_iface.sim_to_real_dataset_processor(
                    joint_pos_obs,
                    visual_obs,
                )
                sim_recorder = recorder_group.recorders["sim"]
                frames = {
                    "sim": sim_recorder.make_sim_frame(
                        leader_iface.get_raw_actions_from_radians(mapped_action),
                        sim_observation,
                        visual_buffers,
                    )
                }
                auxiliary = {
                    "sim": sim_recorder.make_auxiliary_frame(
                        visual_buffers,
                        depth_buffers,
                        instance_id_seg_buffers,
                    )
                }

                if follower_iface is not None:
                    try:
                        follower_observation = (
                            follower_iface.robot.get_observation()
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Physical follower observation failed; the current "
                            f"synchronized episode will be discarded: {exc}"
                        ) from exc
                    frames["real"] = follower_iface.make_real_dataset_frame(
                        leader_action,
                        follower_observation,
                        args_cli.task_name,
                    )

                recorder_group.add_frames(frames, auxiliary)

            if follower_iface is not None:
                try:
                    # Critical invariant: this is the original LeRobot
                    # action, not the mapped Isaac Sim tensor.
                    follower_iface.robot.send_action(leader_action)
                except Exception as exc:
                    raise RuntimeError(
                        "Physical follower action write failed; the current "
                        f"synchronized episode will be discarded: {exc}"
                    ) from exc

            actions[:] = mapped_action
            try:
                obs, _, _, _, _ = env.step(actions)
            except Exception as exc:
                raise RuntimeError(
                    f"Isaac Sim action application failed: {exc}"
                ) from exc

            remaining = (1.0 / args_cli.fps) - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("[INFO]: Interrupted; shutting down cleanly.")
    finally:
        if keyboard_control is not None:
            _cleanup("Keyboard cleanup", keyboard_control.cleanup)
        _cleanup("Follower disconnect", lambda: _disconnect(follower_iface, "physical follower"))
        _cleanup("Leader disconnect", lambda: _disconnect(leader_iface, "physical leader"))
        if spawn_server is not None:
            _cleanup("Spawn server shutdown", spawn_server.close)
        if recorder_group is not None:
            _cleanup("Dataset finalization", recorder_group.finalize)
        if env is not None:
            _cleanup("Environment close", env.close)
        _cleanup("Simulation app close", simulation_app.close)


if __name__ == "__main__":
    main()
