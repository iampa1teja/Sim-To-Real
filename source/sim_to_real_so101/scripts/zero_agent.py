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
#
# This file contains code derived from Isaac Lab
# (https://github.com/isaac-sim/IsaacLab)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
# Licensed under the BSD-3-Clause License.

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=None, help="Stop after this many steps (default: run until closed).")
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-MyRoom", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.num_steps is not None and args_cli.num_steps <= 0:
    parser.error("--num_steps must be positive.")
args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Zero actions agent with Isaac Lab environment."""
    try:
        _run_environment()
    except Exception:
        # Kit may exit the process during close(), before Python prints an
        # uncaught exception. Preserve the diagnostic before releasing it.
        traceback.print_exc()
        raise
    finally:
        # Console entry points call main() directly, bypassing __main__ below.
        # Release the renderer/CUDA context even when environment creation fails.
        simulation_app.close()


def _run_environment():
    """Create, step, and close the environment."""
    # Keep task imports inside main's cleanup guard: configuration import
    # errors must also release the simulator and its GPU allocations.
    import sim_to_real_so101.tasks  # noqa: F401

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()
        step_count = 0
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            while simulation_app.is_running():
                env.step(actions)
                step_count += 1
                if args_cli.num_steps is not None and step_count >= args_cli.num_steps:
                    break
        print(f"[INFO]: Completed {step_count} simulation steps.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
