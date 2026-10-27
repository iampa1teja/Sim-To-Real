"""One encoder-to-angle derivation for simulation and physical follower clients."""
import json
import math
from pathlib import Path

FOLLOWER_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ENCODER_COUNTS_PER_TURN = 4096
ZERO_COUNT = 2048
SETUP_PATH = Path(__file__).resolve().parents[1] / "assets/real_setup.json"


def follower_mapping_from_calibration(calibration, *, gripper_closed_degrees=None):
    """Derive unclipped bounds; optionally express gripper angles in the USD jaw frame.

    Body zero is always encoder 2048, including the full-turn wrist. Gripper
    unit zero is range_min; its measured span is retained when anchoring it to
    the USD closing end-stop. Omitting the anchor returns motor encoder angles.
    Accept a calibration file or the same joint object embedded in provenance.
    """
    label = "object"
    if isinstance(calibration, (str, Path)):
        label = str(calibration)
        try:
            calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read follower calibration file {label}: {error}. Run lerobot-calibrate first.") from error
    if not isinstance(calibration, dict):
        raise ValueError(f"Follower calibration {label} must contain a joint object")
    mapping = {}
    for name in FOLLOWER_JOINTS:
        if name not in calibration:
            raise ValueError(f"Follower calibration {label} is missing joint {name}")
        joint = calibration[name]
        if not isinstance(joint, dict):
            raise ValueError(f"Follower calibration {label}: {name} must contain range_min and range_max")
        low, high = joint.get("range_min"), joint.get("range_max")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)):
            raise ValueError(f"Follower calibration {label}: {name} range_min and range_max must be finite numbers")
        if high <= low:
            raise ValueError(f"Follower calibration {label}: {name} range_max must be greater than range_min")
        mapping[name] = {"joint_min": (low - ZERO_COUNT) * 360 / ENCODER_COUNTS_PER_TURN,
                         "joint_max": (high - ZERO_COUNT) * 360 / ENCODER_COUNTS_PER_TURN}
        if name == "gripper" and gripper_closed_degrees is not None:
            if (isinstance(gripper_closed_degrees, bool) or not isinstance(gripper_closed_degrees, (int, float))
                    or not math.isfinite(gripper_closed_degrees)):
                raise ValueError("gripper_closed_degrees must be finite")
            mapping[name] = {"joint_min": float(gripper_closed_degrees),
                             "joint_max": float(gripper_closed_degrees) + (high - low) * 360 / ENCODER_COUNTS_PER_TURN}
    return mapping


def sim_gripper_closed_degrees():
    """The audited USD closing end-stop, persisted alongside the source calibration."""
    return json.loads(SETUP_PATH.read_text())["sim_joint_mapping_source"]["gripper_closed_degrees"]
