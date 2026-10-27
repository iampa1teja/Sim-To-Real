"""One encoder-to-angle derivation for simulation and physical follower clients."""
import json
import math
from pathlib import Path

FOLLOWER_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ENCODER_COUNTS_PER_TURN = 4096
ZERO_COUNT = 2048
SETUP_PATH = Path(__file__).resolve().parents[1] / "assets/real_setup.json"


def follower_mapping_from_calibration(calibration, *, gripper_closed_degrees=None,
                                      gripper_contact_units=0.0, wrist_roll_zero_offset_degrees=0.0):
    """Derive unclipped bounds with explicit arm-to-USD zero anchors.

    Body encoder zero is 2048. The wrist offset expresses this arm's middle
    pose in the USD frame; policy convention offsets belong to the policy.
    The gripper anchor is the angle at gripper_contact_units, independent of
    any articulation limit. Omitting it returns motor encoder angles.
    Accept a calibration file or the same joint object embedded in provenance.
    """
    for name, value in (("wrist_roll_zero_offset_degrees", wrist_roll_zero_offset_degrees),
                        ("gripper_contact_units", gripper_contact_units)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= gripper_contact_units <= 100:
        raise ValueError("gripper_contact_units must be in 0..100")
    if gripper_closed_degrees is None and gripper_contact_units != 0:
        raise ValueError("gripper_contact_units requires gripper_closed_degrees")
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
        if name == "wrist_roll":
            mapping[name] = {key: value + wrist_roll_zero_offset_degrees
                             for key, value in mapping[name].items()}
        if name == "gripper" and gripper_closed_degrees is not None:
            if (isinstance(gripper_closed_degrees, bool) or not isinstance(gripper_closed_degrees, (int, float))
                    or not math.isfinite(gripper_closed_degrees)):
                raise ValueError("gripper_closed_degrees must be finite")
            span = (high - low) * 360 / ENCODER_COUNTS_PER_TURN
            minimum = float(gripper_closed_degrees) - gripper_contact_units * span / 100
            mapping[name] = {"joint_min": minimum, "joint_max": minimum + span}
    return mapping


def sim_mapping_parameters(source=None):
    """Persisted anchors shared by setup validation, regeneration and real rollout."""
    if source is None:
        source = json.loads(SETUP_PATH.read_text())["sim_joint_mapping_source"]
    return {"gripper_closed_degrees": source["gripper_closed_degrees"],
            "gripper_contact_units": source.get("gripper_contact_units", 0.0),
            "wrist_roll_zero_offset_degrees": source.get("wrist_roll_zero_offset_degrees", 0.0)}


def sim_gripper_closed_degrees():
    """The retained/measured gripper anchor; this is not a joint limit."""
    return sim_mapping_parameters()["gripper_closed_degrees"]


def sim_wrist_roll_zero_offset_degrees():
    return sim_mapping_parameters()["wrist_roll_zero_offset_degrees"]
