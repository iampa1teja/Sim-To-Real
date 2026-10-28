"""One encoder-to-angle derivation for simulation and physical follower clients."""
import json
import math
import hashlib
import os
from dataclasses import dataclass
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


def file_sha256(path):
    """Hash input bytes without modifying the file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CalibrationSelection:
    """The exact files and angle transform selected before opening a device."""

    mapping: dict
    follower_path: Path | None
    leader_path: Path | None
    provenance: dict

    def verify_loaded(self, robot, role):
        """Refuse a different file, including a changed file at the same path."""
        expected = getattr(self, f"{role}_path")
        if expected is None:
            return
        loaded = Path(robot.calibration_fpath).expanduser().resolve()
        if loaded != expected:
            raise ValueError(f"{role} calibration mismatch: LeRobot loaded {loaded}; sim selected {expected}")
        if file_sha256(loaded) != self.provenance[f"{role}_calibration"]["sha256"]:
            raise ValueError(f"{role} calibration changed after the sim mapping was resolved: {loaded}")

    def print_paths(self):
        for role in ("follower", "leader"):
            item = self.provenance[f"{role}_calibration"]
            print(f"[calibration] {role}: {item['path']} SHA-256={item['sha256']}", flush=True)


def resolve_calibration(*, environ=None, setup=None, follower_id=None, leader_id=None, follower_path=None):
    """Select files and derive the sim mapping through one shared code path.

    Explicit paths win, followed by IDs in the environment. With no selection,
    retain the stored mapping and its original embedded-calibration stale check.
    Explicit IDs are useful for command-line clients that already require an ID.
    """
    environ = os.environ if environ is None else environ
    setup = json.loads(SETUP_PATH.read_text()) if setup is None else setup
    root = Path(environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()

    def selected_path(role, explicit, identifier, group, kind):
        value = explicit or environ.get(f"SO101_{role.upper()}_CALIBRATION")
        if value:
            return Path(value).expanduser().resolve()
        if identifier:
            return (root / group / kind / f"{identifier}.json").resolve()
        return None

    follower = selected_path("follower", follower_path, follower_id or environ.get("ROBOT_ID"),
                             "robots", "so101_follower")
    leader = selected_path("leader", None, leader_id or environ.get("TELEOP_ID"),
                           "teleoperators", "so101_leader")

    def identity(path, role):
        if path is None:
            return {"path": None, "sha256": None}
        try:
            return {"path": str(path), "sha256": file_sha256(path)}
        except OSError as error:
            raise ValueError(f"Cannot read {role} calibration file {path}: {error}") from error

    follower_identity = identity(follower, "follower")
    leader_identity = identity(leader, "leader")
    source = setup.get("sim_joint_mapping_source", {})
    if follower is not None:
        mapping = follower_mapping_from_calibration(follower, **sim_mapping_parameters(source))
    else:
        mapping = setup["sim_joint_mapping"]
        if "joints" in source:
            derived = follower_mapping_from_calibration(source["joints"], **sim_mapping_parameters(source))
            if mapping != derived:
                raise ValueError("Stale sim_joint_mapping; regenerate with scripts/update_sim_calibration.py")
            mapping = derived

    for path, details in ((follower, follower_identity), (leader, leader_identity)):
        if path is not None and file_sha256(path) != details["sha256"]:
            raise ValueError(f"Calibration changed while resolving the sim mapping: {path}")
    provenance = {"follower_calibration": follower_identity, "leader_calibration": leader_identity,
                  "real_setup": {"path": str(SETUP_PATH), "sha256": file_sha256(SETUP_PATH)},
                  "mapping": mapping, "mapping_parameters": sim_mapping_parameters(source),
                  "mapping_source": "follower_file" if follower else "embedded_real_setup",
                  "embedded_follower_sha256": source.get("calibration_sha256"),
                  "gripper_anchor_status": source.get("gripper_contact_fit_status", "unresolved; see real_setup.json")}
    return CalibrationSelection(mapping, follower, leader, provenance)
