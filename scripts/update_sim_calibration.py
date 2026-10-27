"""Regenerate simulation mapping from an unchanged follower file, using pxr in teleop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.joint_calibration import (
    ENCODER_COUNTS_PER_TURN, ZERO_COUNT, follower_mapping_from_calibration, sim_mapping_parameters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--follower-id", default="rpd_robot")
    parser.add_argument("--wrist-roll-zero-offset-degrees", type=float)
    parser.add_argument("--gripper-closed-degrees", type=float)
    parser.add_argument("--gripper-contact-units", type=float)
    args = parser.parse_args()
    asset_dir = ROOT / "source/sim_to_real_so101/assets"
    path = asset_dir / "real_setup.json"
    setup = json.loads(path.read_text())
    calibration = json.loads(args.calibration.read_text())
    source = dict(setup["sim_joint_mapping_source"])
    anchors = sim_mapping_parameters(source)
    for name in anchors:
        value = getattr(args, name)
        if value is not None:
            anchors[name] = value
    setup["sim_joint_mapping"] = follower_mapping_from_calibration(calibration, **anchors)
    source.update(follower_id=args.follower_id,
                  calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                  encoder_counts_per_turn=ENCODER_COUNTS_PER_TURN, zero_count=ZERO_COUNT,
                  **anchors, joints={})
    source["note"] = ("Unclipped encoder spans, with explicit USD-vs-arm roll zero and gripper contact anchor. "
                      "Anchor values and their verification status are independent of runtime articulation limits. "
                      "Molmo's wrist_roll_model_offset remains a separate policy convention.")
    for name, values in calibration.items():
        source["joints"][name] = dict(values, method=(
            "closed_degrees + ((count - range_min) / (range_max - range_min) * 100 - contact_units) "
            "* (range_max - range_min) * 360 / 4096 / 100" if name == "gripper"
            else "(count - 2048) * 360 / 4096 + wrist_roll_zero_offset_degrees" if name == "wrist_roll"
            else "(count - 2048) * 360 / 4096"))
    setup["sim_joint_mapping_source"] = source
    path.write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps(setup["sim_joint_mapping"], indent=2))


if __name__ == "__main__":
    main()
