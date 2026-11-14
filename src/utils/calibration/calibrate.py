# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# utils/calibration/calibrate.py
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
from dataclasses import fields, is_dataclass
import os
import importlib

from utils.arms.spec import load_arm_spec
from utils.calibration.check import calibration_root


def calibrate_device(role, lerobot, *, port, device_id, directory):
    """Use the installed LeRobot calibration lifecycle after registering arm modules."""
    for module in lerobot.modules:
        importlib.import_module(module)
    if role == "leader":
        from lerobot.teleoperators import TeleoperatorConfig
        from lerobot.teleoperators.utils import make_teleoperator_from_config
        config_class = TeleoperatorConfig.get_choice_class(lerobot.leader_type)
        make_device = make_teleoperator_from_config
    elif role == "follower":
        from lerobot.robots import RobotConfig, make_robot_from_config
        config_class = RobotConfig.get_choice_class(lerobot.follower_type)
        make_device = make_robot_from_config
    else:
        raise ValueError(f"Unknown device role {role!r}")
    options = dict(port=port, id=device_id, calibration_dir=directory)
    if not is_dataclass(config_class) or "use_degrees" in {field.name for field in fields(config_class)}:
        options["use_degrees"] = lerobot.use_degrees
    config = config_class(**options)
    device = make_device(config)
    try:
        device.connect(calibrate=False)
        device.calibrate()
    except BaseException as error:
        try:
            device.disconnect()
        except Exception as cleanup_error:
            error.add_note(f"Device cleanup also failed: {cleanup_error!r}")
        raise
    else:
        device.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("leader", "follower"))
    parser.add_argument("--port")
    parser.add_argument("--id")
    args = parser.parse_args(argv)

    lerobot = load_arm_spec()[0].lerobot
    if args.role == "leader":
        port = args.port or os.getenv("TELEOP_PORT", "/dev/ttyACM0")
        device_id = args.id or os.getenv("TELEOP_ID", "leader_arm_1")
        folder = calibration_root() / lerobot.leader_calibration_dir
    else:
        port = args.port or os.getenv("ROBOT_PORT", "/dev/ttyACM1")
        device_id = args.id or os.getenv("ROBOT_ID", "follower_arm_1")
        folder = calibration_root() / lerobot.follower_calibration_dir

    print(f"[calibrate] {args.role} {device_id} on {port}; the file will be saved to {folder / (device_id + '.json')}",
          flush=True)
    calibrate_device(args.role, lerobot, port=port, device_id=device_id, directory=folder)


if __name__ == "__main__":
    main()
