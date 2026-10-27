"""Regenerate simulation mapping from an unchanged follower file, using pxr in teleop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from pxr import Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.joint_calibration import (
    ENCODER_COUNTS_PER_TURN, ZERO_COUNT, follower_mapping_from_calibration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--follower-id", default="rpd_robot")
    args = parser.parse_args()
    asset_dir = ROOT / "source/sim_to_real_so101/assets"
    path = asset_dir / "real_setup.json"
    setup = json.loads(path.read_text())
    calibration = json.loads(args.calibration.read_text())
    stage = Usd.Stage.Open(str(asset_dir / "usd/SO-ARM101-USD.usd"))
    jaw = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/so101_new_calib/joints/Jaw"))
    closed = jaw.GetLowerLimitAttr().Get()
    setup["sim_joint_mapping"] = follower_mapping_from_calibration(calibration, gripper_closed_degrees=closed)
    source = dict(follower_id=args.follower_id, calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                  encoder_counts_per_turn=ENCODER_COUNTS_PER_TURN, zero_count=ZERO_COUNT, gripper_closed_degrees=closed,
                  gripper_closed_source="SO-ARM101-USD.usd:/so101_new_calib/joints/Jaw.physics:lowerLimit",
                  note="Unclipped follower ranges; the USD joint limits still clamp simulation targets. "
                       "No added wrist zero offset. Gripper unit 0 is the USD closing end-stop.", joints={})
    for name, values in calibration.items():
        source["joints"][name] = dict(values, method=(
            "closed_degrees + (count - range_min) * 360 / 4096" if name == "gripper"
            else "(count - 2048) * 360 / 4096"))
    setup["sim_joint_mapping_source"] = source
    path.write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps(setup["sim_joint_mapping"], indent=2))


if __name__ == "__main__":
    main()
