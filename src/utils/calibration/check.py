# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# utils/calibration/check.py  (ported from the workshop's so101_check_calibration.py)
"""Quality check for a follower calibration file.

1. Static check: each joint's calibrated range (raw encoder counts, range_max - range_min) is compared with the
   mean and std of many known-good arms (arms/<arm>/calibration/stats.json, see s2r calibration-stats).
   A joint more than N_STD standard deviations away gets a WARN; so does a very large homing offset.
2. Live check (needs the follower connected): the current raw encoder position of every joint lies inside the
   calibrated range. Only the motor bus is opened, so torque and the arm are left exactly as they are.

Usage (in the sim container started with compose.hardware.yaml):
    s2r calibration-check --id follower_arm_1 --port /dev/ttyACM1
    s2r calibration-check --file /path/to/follower_arm_1.json --no-live
"""

import argparse
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path

from utils.arms.spec import load_arm_spec

# Standard deviations from the mean motion range before a joint is flagged.
N_STD = 2.0


# Folder LeRobot calibration files live under (same default as utils.arms.calibration.resolve_calibration).
def calibration_root():
    return Path(os.environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()


# arms/<arm>/calibration/stats.json, written by s2r calibration-stats.
def default_stats_path():
    _, arm_dir = load_arm_spec()
    return arm_dir / "calibration" / "stats.json"


# Calibration file as {joint: {id, drive_mode, homing_offset, range_min, range_max}}.
def load_calibration(path):
    return json.loads(Path(path).read_text())


# Compare each joint's motion range with mean +/- N_STD * std from the stats file.
def static_check(calibration, stats, *, arm=None):
    arm = arm or load_arm_spec()[0]
    homing_offset_warn = arm.hardware.encoder_counts_per_turn / 2
    results = []
    for joint, calib in calibration.items():
        motion_range = abs(calib["range_max"] - calib["range_min"])
        joint_stats = stats.get(joint)
        issues = []
        deviation_std = None
        if joint_stats:
            mean, std = joint_stats["mean"], joint_stats["std"]
            deviation_std = (motion_range - mean) / std if std > 0 else 0.0
            if abs(deviation_std) > N_STD:
                issues.append(f"motion range {motion_range} deviates {deviation_std:+.1f}σ from mean "
                              f"(mean={mean:.0f}, std={std:.0f}, threshold=±{N_STD}σ)")
        if abs(calib["homing_offset"]) > homing_offset_warn:
            issues.append(f"large homing_offset ({calib['homing_offset']}): "
                          "arm was likely not near the midpoint during calibration")
        results.append({
            "joint": joint, "motion_range": motion_range,
            "mean": joint_stats["mean"] if joint_stats else None,
            "std": joint_stats["std"] if joint_stats else None,
            "deviation_std": deviation_std, "homing_offset": calib["homing_offset"],
            "status": "PASS" if not issues else "WARN", "issues": issues,
        })
    return results


# Read raw encoder positions and check each is inside the calibrated range.
def live_check(bus, calibration):
    results = []
    for joint, position in bus.sync_read("Present_Position", normalize=False).items():
        calib = calibration.get(joint)
        if calib is None:
            continue
        results.append({
            "joint": joint, "raw_pos": position, "range_min": calib["range_min"], "range_max": calib["range_max"],
            "in_range": calib["range_min"] <= position <= calib["range_max"],
        })
    return results


# Open only the follower's motor bus (no configure, no torque change, no cameras) and read it.
def read_live(port, robot_id, calibration_path, calibration):
    from lerobot.robots import RobotConfig, make_robot_from_config

    lerobot = load_arm_spec()[0].lerobot
    import importlib
    for module in lerobot.modules:
        importlib.import_module(module)
    config_class = RobotConfig.get_choice_class(lerobot.follower_type)
    options = dict(port=port, id=robot_id, calibration_dir=Path(calibration_path).parent)
    if not is_dataclass(config_class) or "use_degrees" in {field.name for field in fields(config_class)}:
        options["use_degrees"] = lerobot.use_degrees
    config = config_class(**options)
    robot = make_robot_from_config(config)
    robot.bus.connect()
    try:
        return live_check(robot.bus, calibration)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


def print_report(calibration_path, stats_path, static_results, live_results):
    width = 76
    print("\n" + "=" * width)
    print("  CALIBRATION CHECK REPORT")
    print(f"  File:  {calibration_path}")
    print(f"  Stats: {stats_path}")
    print("=" * width)

    print(f"\n[1] Motion Range vs Stats (threshold ±{N_STD}σ)\n")
    print(f"  {'Joint':<18} {'Range':>6}  {'Mean':>7} {'Std':>6} {'Deviation':>10}  {'Offset':>8}  Status")
    print("  " + "-" * (width - 2))
    all_pass = True
    for r in static_results:
        all_pass &= r["status"] == "PASS"
        mean = f"{r['mean']:.0f}" if r["mean"] is not None else "-"
        std = f"{r['std']:.0f}" if r["std"] is not None else "-"
        deviation = f"{r['deviation_std']:+.2f}σ" if r["deviation_std"] is not None else "-"
        print(f"  {r['joint']:<18} {r['motion_range']:>6}  {mean:>7} {std:>6} {deviation:>10}  "
              f"{r['homing_offset']:>8}  {'✓' if r['status'] == 'PASS' else '⚠'} {r['status']}")
        for issue in r["issues"]:
            print(f"    {'':18}  → {issue}")

    if live_results:
        print("\n[2] Live Encoder Positions\n")
        print(f"  {'Joint':<18} {'Position':>9}  {'Calibrated Range':^20}  {'In Range':>9}")
        print("  " + "-" * (width - 2))
        for r in live_results:
            all_pass &= r["in_range"]
            print(f"  {r['joint']:<18} {r['raw_pos']:>9}  {str(r['range_min']) + ' – ' + str(r['range_max']):^20}  "
                  f"{'✓ OK' if r['in_range'] else '✗ OUT OF RANGE'}")
    else:
        print("\n[2] Live check skipped\n")

    print("\n" + "=" * width)
    print("  Overall: " + ("✓ PASS — calibration looks good." if all_pass else "⚠ WARN — review the issues above."))
    print("=" * width + "\n")
    return all_pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default=os.getenv("ROBOT_ID", "follower_arm_1"), help="follower id (default $ROBOT_ID)")
    parser.add_argument("--port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM1"), help="follower port (default $ROBOT_PORT)")
    parser.add_argument("--file", help="calibration JSON to check instead of <calibration root>/<arm dir>/<id>.json")
    parser.add_argument("--stats", help="stats JSON (default arms/<arm>/calibration/stats.json, or $STATS_JSON)")
    parser.add_argument("--no-live", action="store_true", help="skip the live encoder check (no hardware needed)")
    args = parser.parse_args(argv)

    calibration_path = Path(args.file) if args.file else calibration_root() / load_arm_spec()[0].lerobot.follower_calibration_dir / f"{args.id}.json"
    if not calibration_path.is_file():
        raise SystemExit(f"Calibration file not found: {calibration_path}\n"
                         "Run `s2r calibrate follower` first, or pass --id / --file.")
    stats_path = Path(args.stats or os.getenv("STATS_JSON") or default_stats_path())
    if not stats_path.is_file():
        raise SystemExit(f"Stats file not found: {stats_path}\nRun `s2r calibration-stats` first, or pass --stats.")

    calibration = load_calibration(calibration_path)
    stats = json.loads(stats_path.read_text())
    static_results = static_check(calibration, stats)

    live_results = []
    if not args.no_live:
        try:
            live_results = read_live(args.port, args.id, calibration_path, calibration)
            print("[INFO]: Follower connected; read live encoder positions.")
        except Exception as exc:
            print(f"[WARNING]: Could not read the follower ({exc}); showing the file-only report.")

    return 0 if print_report(calibration_path, stats_path, static_results, live_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
