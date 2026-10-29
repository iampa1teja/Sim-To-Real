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


def calibration_identity(path, *, required=True):
    """Identify input bytes; optional leader inputs never break a sim import."""
    if path is None:
        return {"path": None, "sha256": None}
    path = Path(path).expanduser().resolve()
    try:
        return {"path": str(path), "sha256": file_sha256(path)}
    except OSError as error:
        if required:
            raise ValueError(f"Cannot read calibration file {path}: {error}") from error
        return {"path": str(path), "sha256": None}


def load_device_calibration(robot, path):
    """Load a caller-selected file into LeRobot memory, preserving its config ID.

    No serial connection or register write occurs here. LeRobot constructs its
    bus before this override, so both in-memory copies must be updated.
    """
    path = Path(path).expanduser().resolve()
    robot.calibration_fpath = path
    robot._load_calibration()
    robot.bus.calibration = robot.calibration


@dataclass(frozen=True)
class CalibrationSelection:
    """Independent simulation transform and files to be loaded by hardware."""

    mapping: dict
    follower_path: Path | None
    leader_path: Path | None
    provenance: dict

    def verify_loaded(self, robot, role):
        """Validate the actual LeRobot input before any connection or movement."""
        expected = getattr(self, f"{role}_path")
        loaded = Path(robot.calibration_fpath).expanduser().resolve()
        if expected is not None and loaded != expected:
            raise ValueError(f"{role} calibration mismatch: LeRobot loaded {loaded}; expected {expected}")
        actual_hash = calibration_identity(loaded)["sha256"]
        if expected is not None:
            identity = self.provenance[f"{role}_calibration"]
            if actual_hash != identity["sha256"]:
                raise ValueError(f"{role} calibration changed after selection: {loaded}")
        if role != "follower":
            return
        # Use the file LeRobot actually loaded, never an ID-derived assumption.
        actual = follower_mapping_from_calibration(loaded, **self.provenance["mapping_parameters"])
        rows = []
        for name in FOLLOWER_JOINTS:
            sim, real = self.mapping[name], actual[name]
            delta = [real[key] - sim[key] for key in ("joint_min", "joint_max")]
            rows.append((name, sim, real, delta))
        if any(abs(d) > 1e-9 for _, _, _, ds in rows for d in ds):
            source = self.provenance["mapping_calibration"]
            lines = ["Follower calibration mapping differs from the sim mapping; refusing to connect.",
                     f"Loaded follower: {loaded} SHA-256={actual_hash}",
                     f"Sim source: {source['path']} SHA-256={source['sha256']}",
                     f"Embedded follower SHA-256={self.provenance['embedded_follower_sha256']}",
                     "Joint             sim min/max (deg)    loaded min/max (deg)   delta min/max (deg)"]
            for name, sim, real, delta in rows:
                lines.append(f"{name:17s} {sim['joint_min']:8.3f} {sim['joint_max']:8.3f}  "
                             f"{real['joint_min']:8.3f} {real['joint_max']:8.3f}  {delta[0]:+8.3f} {delta[1]:+8.3f}")
            raise ValueError("\n".join(lines))
        # Check the in-memory normalization too, when the installed driver exposes it.
        calibration = getattr(robot, "calibration", None)
        if calibration:
            from dataclasses import asdict, is_dataclass
            values = {name: asdict(v) if is_dataclass(v) else v for name, v in calibration.items()}
            if follower_mapping_from_calibration(values, **self.provenance["mapping_parameters"]) != actual:
                raise ValueError("LeRobot in-memory calibration differs from its file; refusing to connect")
        sim_source = self.provenance["mapping_calibration"]
        if file_sha256(sim_source["path"]) != sim_source["sha256"]:
            raise ValueError("Sim mapping source changed after selection; restart the client")

    def print_paths(self):
        for role in ("follower", "leader"):
            item = self.provenance[f"{role}_calibration"]
            print(f"[calibration] {role}: {item['path']} SHA-256={item['sha256']}", flush=True)
        item = self.provenance["mapping_calibration"]
        print(f"[calibration] sim mapping: {item['path']} SHA-256={item['sha256']}", flush=True)


def resolve_calibration(*, environ=None, setup=None, follower_id=None, leader_id=None, follower_path=None):
    """Keep the locked mapping unless a follower path is explicitly selected.

    IDs passed by hardware clients select only their LeRobot input files. The
    environment IDs have no effect here: CLI parsers supply their fallbacks.
    An explicit client path also selects that hardware input, without changing
    its logical/config ID. A missing leader is checked only by a leader client.
    """
    environ = os.environ if environ is None else environ
    setup = json.loads(SETUP_PATH.read_text()) if setup is None else setup
    root = Path(environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()
    explicit_follower = follower_path or environ.get("SO101_FOLLOWER_CALIBRATION")
    explicit_leader = environ.get("SO101_LEADER_CALIBRATION")
    mapping_path = Path(explicit_follower).expanduser().resolve() if explicit_follower else None
    follower = (Path(follower_path).expanduser().resolve() if follower_path else
                (root / "robots/so101_follower" / f"{follower_id}.json").resolve() if follower_id else mapping_path)
    leader = ((root / "teleoperators/so101_leader" / f"{leader_id}.json").resolve() if leader_id else
              Path(explicit_leader).expanduser().resolve() if explicit_leader else None)
    follower_identity = calibration_identity(follower)
    leader_identity = calibration_identity(leader, required=leader_id is not None)
    mapping_identity = calibration_identity(mapping_path or SETUP_PATH)
    source = setup.get("sim_joint_mapping_source", {})
    if mapping_path is not None:
        mapping = follower_mapping_from_calibration(mapping_path, **sim_mapping_parameters(source))
    else:
        mapping = setup["sim_joint_mapping"]
        if "joints" in source:
            derived = follower_mapping_from_calibration(source["joints"], **sim_mapping_parameters(source))
            if mapping != derived:
                raise ValueError("Stale sim_joint_mapping; regenerate with scripts/update_sim_calibration.py")
            mapping = derived
    selected = {"follower": calibration_identity(mapping_path)["sha256"],
                "leader": calibration_identity(explicit_leader, required=False)["sha256"]}
    provenance = {"follower_calibration": follower_identity, "leader_calibration": leader_identity,
                  "real_setup": calibration_identity(SETUP_PATH), "mapping_calibration": mapping_identity,
                  "mapping": mapping, "mapping_parameters": sim_mapping_parameters(source),
                  "mapping_source": "follower_file" if mapping_path else "embedded_real_setup",
                  "explicit_selection": bool(explicit_follower or explicit_leader),
                  "selected_calibrations": selected,
                  "embedded_follower_sha256": source.get("calibration_sha256"),
                  "gripper_anchor_status": source.get("gripper_contact_fit_status", "unresolved; see real_setup.json")}
    return CalibrationSelection(mapping, follower, leader, provenance)
