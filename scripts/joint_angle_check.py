#!/usr/bin/env python3
"""Inspect passive joint telemetry from lerobot_agent; never opens a robot port.

The operator chose teleoperation instead of phone angle measurements. The
existing agent owns all control; this tool summarizes encoder/PhysX tracking.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'source'))
from sim_to_real_so101.utils.joint_angle_check import summarize_teleop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--watch', action='store_true', help='Update the joint table once per second; Ctrl-C exits the observer only')
    parser.add_argument('--physx-check', action='store_true',
                        help='Offline replay into a separate sim using production mapping, zero physics steps, actual PhysX getters; requires Isaac')
    args = parser.parse_args()
    if args.physx_check:
        if not args.output or args.watch:
            parser.error('--physx-check requires --output and cannot be combined with --watch')
        from sim_to_real_so101.utils.joint_angle_check_physx import check_in_physx
        check_in_physx(args.log, args.output)
        return
    if args.watch:
        from scripts.calibration_check import watch_log
        watch_log(args.log)
        return
    result = summarize_teleop(args.log)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        if args.output.resolve() == args.log.resolve():
            parser.error('--output must not replace the raw log')
        args.output.write_text(encoded)
    print(encoded)


if __name__ == '__main__':
    main()
