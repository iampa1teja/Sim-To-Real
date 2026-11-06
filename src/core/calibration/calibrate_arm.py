# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# core/calibration/calibrate_arm.py
"""Run LeRobot's interactive calibration for the leader or follower, saving where s2r looks for it.

LeRobot 0.6.1 saves every SO arm under robots/so_follower/ and teleoperators/so_leader/, but s2r reads the
folders named in arms/<arm>/arm.toml ([arm.lerobot] follower_calibration_dir / leader_calibration_dir). This
passes that folder as calibration_dir, so the file lands in the right place. The prompts are LeRobot's own:
move every joint to the middle of its range (wrist roll too), press Enter, sweep each joint through its full
range, press Enter.

Usage:
    s2r calibrate leader   --port /dev/ttyACM0 --id leader_arm_1
    s2r calibrate follower --port /dev/ttyACM1 --id follower_arm_1
Port and id default to $TELEOP_PORT/$TELEOP_ID (leader) and $ROBOT_PORT/$ROBOT_ID (follower).
Afterwards: s2r calibration-check --id <follower id>
"""

import argparse
import os
import shutil
import sys

from core.arms.joint_calibration import FOLLOWER_CALIBRATION_DIR, LEADER_CALIBRATION_DIR
from core.arms.spec import load_arm_spec
from core.calibration.check_calibration import calibration_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("leader", "follower"))
    parser.add_argument("--port")
    parser.add_argument("--id")
    args = parser.parse_args(argv)

    lerobot = load_arm_spec()[0]["lerobot"]
    if args.role == "leader":
        prefix, kind = "teleop", lerobot["leader_type"]
        port = args.port or os.getenv("TELEOP_PORT", "/dev/ttyACM0")
        device_id = args.id or os.getenv("TELEOP_ID", "leader_arm_1")
        folder = calibration_root() / LEADER_CALIBRATION_DIR
    else:
        prefix, kind = "robot", lerobot["follower_type"]
        port = args.port or os.getenv("ROBOT_PORT", "/dev/ttyACM1")
        device_id = args.id or os.getenv("ROBOT_ID", "follower_arm_1")
        folder = calibration_root() / FOLLOWER_CALIBRATION_DIR

    executable = shutil.which("lerobot-calibrate")
    if executable is None:
        raise SystemExit("lerobot-calibrate not found; run this in the sim container or the sim2real env.")
    command = [executable, f"--{prefix}.type={kind}", f"--{prefix}.port={port}", f"--{prefix}.id={device_id}",
               f"--{prefix}.use_degrees={str(lerobot['use_degrees']).lower()}", f"--{prefix}.calibration_dir={folder}"]
    print(f"[calibrate] {args.role} {device_id} on {port}; the file will be saved to {folder / (device_id + '.json')}")
    sys.stdout.flush()
    os.execv(executable, command)


if __name__ == "__main__":
    main()
