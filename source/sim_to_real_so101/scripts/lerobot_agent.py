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

import argparse
import os

from isaaclab.app import AppLauncher


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Isaac Lab SO-101 Teleop agent."
)

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate.",
)

parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task.",
)


# ============================================================
# PHYSICAL LEADER / TELEOP ARM
# Same convention as:
#
# --teleop.port=$TELEOP_PORT
# --teleop.id=$TELEOP_ID
# ============================================================

parser.add_argument(
    "--port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Port of the physical leader/teleop robot.",
)

parser.add_argument(
    "--robot_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="ID of the physical leader/teleop robot.",
)


# ============================================================
# PHYSICAL FOLLOWER
# Same convention as:
#
# --robot.port=$ROBOT_PORT
# --robot.id=$ROBOT_ID
# ============================================================

parser.add_argument(
    "--enable_real_follower",
    action="store_true",
    default=False,
    help="Enable simultaneous physical follower teleoperation.",
)

parser.add_argument(
    "--follower_port",
    type=str,
    default=os.getenv("ROBOT_PORT", "/dev/ttyACM1"),
    help="Port of the physical follower robot.",
)

parser.add_argument(
    "--follower_id",
    type=str,
    default=os.getenv("ROBOT_ID", "follower_arm_1"),
    help="ID of the physical follower robot.",
)


# ============================================================
# DATASET ARGUMENTS
# ============================================================

parser.add_argument(
    "--repo_id",
    type=str,
    default=None,
    help="Repository ID to store the dataset.",
)

parser.add_argument(
    "--repo_root",
    type=str,
    default=None,
    help="Repository root to store the dataset.",
)

parser.add_argument(
    "--save_mp4",
    action="store_true",
    default=False,
    help="Save depth and RGB as mp4 videos.",
)

parser.add_argument(
    "--depth",
    action="store_true",
    default=False,
    help="Save depth as mp4 video.",
)

parser.add_argument(
    "--instance_id_seg",
    action="store_true",
    default=False,
    help="Save instance id segmentation as mp4 video.",
)

parser.add_argument(
    "--task_name",
    type=str,
    default=None,
    help="Name of the task.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=101,
    help="Environment seed",
)


# ============================================================
# ISAAC LAB APP ARGUMENTS
# ============================================================

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Always enable cameras to record video
args_cli.enable_cameras = True


# ============================================================
# LAUNCH OMNIVERSE APP
# ============================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401

from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder


def main():

    # ========================================================
    # KEYBOARD CONTROL
    # ========================================================

    keyboard_control = KeyboardControl()


    # ========================================================
    # ISAAC LAB ENVIRONMENT CONFIGURATION
    # ========================================================

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Create environment
    env_cfg.seed = args_cli.seed

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )


    # ========================================================
    # PRINT ENVIRONMENT INFORMATION
    # ========================================================

    print(
        f"[INFO]: Gym observation space: "
        f"{env.observation_space}"
    )

    print(
        f"[INFO]: Gym action space: "
        f"{env.action_space}"
    )

    print(
        "[INFO]: Click 'R' to reset the world"
    )

    print(
        "[INFO]: Click 'S' to start/stop recording; "
        "'R' will also stop recording"
    )


    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    env.reset()


    # ========================================================
    # FIND CAMERAS
    # ========================================================

    cameras = {}

    for obj in env.unwrapped.scene.keys():

        if obj.startswith("camera_"):

            camera_cfg = getattr(
                env.unwrapped.scene.cfg,
                obj,
            )

            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }

            print(
                f"[INFO]: Found Camera: "
                f"{obj.replace('camera_', '')}"
            )

    if len(cameras) == 0:

        print(
            "[Info]: No cameras found - "
            "videos will not be recorded"
        )


    # ========================================================
    # PHYSICAL LEADER
    # ========================================================

    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras=cameras,
        fps=30,
        kind="leader",
    )

    robot_iface.init_device()
    robot_iface.connect()

    print(
        f"[INFO]: Physical leader connected:"
    )

    print(
        f"[INFO]:   Port = {args_cli.port}"
    )

    print(
        f"[INFO]:   ID   = {args_cli.robot_id}"
    )


    # ========================================================
    # PHYSICAL FOLLOWER
    # ========================================================

    follower_iface = None

    if args_cli.enable_real_follower:

        print(
            "[INFO]: Physical follower teleoperation enabled."
        )

        print(
            f"[INFO]: Connecting follower:"
        )

        print(
            f"[INFO]:   Port = {args_cli.follower_port}"
        )

        print(
            f"[INFO]:   ID   = {args_cli.follower_id}"
        )

        follower_iface = LeRobotSO101Interface(
            device=env.unwrapped.device,
            port=args_cli.follower_port,
            id=args_cli.follower_id,
            cameras={},
            fps=30,
            kind="follower",
        )

        follower_iface.init_device()
        follower_iface.connect()

        print(
            "[INFO]: Physical follower connected."
        )


    # ========================================================
    # ISAAC SIM ACTION TENSOR
    # ========================================================

    actions = torch.zeros(
        env.action_space.shape,
        device=env.unwrapped.device,
    )

    # Use the leader's wrist pose at startup as the zero point. The simulated
    # wrist starts at +90 degrees and then follows leader motion relatively.
    wrist_roll_index = 4
    leader_wrist_start = None
    sim_wrist_start = -torch.pi / 2


    # ========================================================
    # DATASET RECORDING
    # ========================================================

    if all(
        [
            args_cli.repo_id,
            args_cli.repo_root,
            args_cli.task_name,
        ]
    ):

        recording_mode = True

    else:

        recording_mode = False


    if recording_mode:

        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=args_cli.save_mp4,
            depth=args_cli.depth,
            instance_id_seg=args_cli.instance_id_seg,
        )

        try:

            recorder.init_dataset()

        except ValueError:

            print(
                "[ERROR]: Failed to initialize dataset. "
                "folder already exists"
            )

            env.close()
            simulation_app.close()


    # ========================================================
    # MAIN TELEOPERATION LOOP
    # ========================================================

    while simulation_app.is_running():

        # Run everything in inference mode
        with torch.inference_mode():

            # =================================================
            # 1. READ PHYSICAL LEADER
            # =================================================

            real_action = robot_iface.robot.get_action()


            # =================================================
            # 2. SEND LEADER ACTION TO PHYSICAL FOLLOWER
            #
            # IMPORTANT:
            # Send real_action directly.
            #
            # Do NOT send mapped_action here.
            # mapped_action is specifically converted
            # for the Isaac Lab simulation.
            # =================================================

            if follower_iface is not None:

                follower_iface.robot.send_action(
                    real_action
                )


            # =================================================
            # 3. MAP LEADER ACTION FOR ISAAC SIM
            # =================================================

            real_action, mapped_action = (
                robot_iface.real_to_sim_obs_processor(
                    real_action
                )
            )

            if leader_wrist_start is None:

                leader_wrist_start = mapped_action[
                    wrist_roll_index
                ].clone()

            mapped_action[wrist_roll_index] = (
                sim_wrist_start
                + mapped_action[wrist_roll_index]
                - leader_wrist_start
            )

            mapped_action[wrist_roll_index] = torch.clamp(
                mapped_action[wrist_roll_index],
                robot_iface.joint_mins[wrist_roll_index]
                * torch.pi
                / 180,
                robot_iface.joint_maxs[wrist_roll_index]
                * torch.pi
                / 180,
            )


            # =================================================
            # 4. SEND ACTION TO SIMULATED FOLLOWER
            # =================================================

            actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)


            # =================================================
            # 5. RESET WORLD
            # =================================================

            if keyboard_control.reset_world:

                keyboard_control.reset_world = False

                env.reset()

                # Re-zero at the leader's current wrist pose after a reset.
                leader_wrist_start = None

                continue


            # =================================================
            # 6. DATASET RECORDING
            # =================================================

            if (
                recording_mode
                and keyboard_control.recording
            ):

                visual_obs = obs.get(
                    "visual",
                    None,
                )

                if visual_obs is None:

                    print(
                        "[WARNING]: No 'visual' observation group - "
                        "recording requires a task with cameras"
                    )

                    keyboard_control.recording = False

                    continue


                # ---------------------------------------------
                # Extract joint positions from policy
                # observation dictionary
                # ---------------------------------------------

                joint_pos_obs = (
                    obs["policy"]["joint_pos_obs"][0]
                )

                visual_obs = obs["visual"]


                # ---------------------------------------------
                # Convert simulation observations into
                # LeRobot dataset format
                # ---------------------------------------------

                (
                    real_obs,
                    visual_buffers,
                    depth_buffers,
                    instance_id_seg_buffers,
                ) = (
                    robot_iface.sim_to_real_dataset_processor(
                        joint_pos_obs,
                        visual_obs,
                    )
                )


                # ---------------------------------------------
                # Store frame
                # ---------------------------------------------

                recorder.push_frame_to_buffer(
                    real_action,
                    real_obs,
                    visual_buffers,
                    depth_buffers,
                    instance_id_seg_buffers,
                )


    # ========================================================
    # CLEANUP
    # ========================================================

    if follower_iface is not None:

        follower_iface.robot.disconnect()

    robot_iface.robot.disconnect()

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

    while True:

        simulation_app.update()

    simulation_app.close()
