import hashlib
import json 
import math 
from pathlib import Path

from ..arms.spec import load_arm_spec
from ..arms.joint_calibration import resolve_calibration
from ..scenes import setup_path

_ARM, _ = load_arm_spec()
_LEROBOT = _ARM["lerobot"]
FOLLOWER_JOINTS = tuple(_LEROBOT["names"])
FOLLOWER_CALIBRATION_DIR = _LEROBOT["follower_calibration_dir"]
LEADER_CALIBRATION_DIR = _LEROBOT["leader_calibration_dir"]
SETUP_PATH = setup_path()
MARKER = "meta/recording_alignment.json"


class RecordingAlignmentError(ValueError):
    """An existing dataset cannot be resumed with the selected calibration."""


def recording_contract(*, follower_id, leader_id, setup=None, calibration_selection=None):
    setup = json.loads(SETUP_PATH.read_text()) if setup is None else setup
    selection = calibration_selection or resolve_calibration(setup=setup)
    resolved = setup["sim_joint_mapping_source"].get("wrist_roll_fit_status") == "verified"
    calibration = {key: setup[key] for key in ("sim_joint_mapping", "sim_joint_mapping_source", "runtime_joint_limits")}
    if selection.provenance["explicit_selection"]:
        calibration = dict(calibration, sim_joint_mapping=selection.mapping)
    # Record the files actually loaded by both devices, including the leader's default file.
    selected = {role: selection.provenance[f"{role}_calibration"]["sha256"]
                for role in ("follower", "leader")}
    if any(selected.values()):
        calibration["selected_calibrations"] = selected
    source = setup["sim_joint_mapping_source"]
    contract = {
        "version": 3, "follower_id": follower_id, "leader_id": leader_id,
        "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        "start_sync": "follower_state", "roll_resolved": resolved,
        "synced_joints": list(FOLLOWER_JOINTS),
        "wrist_rezero": False, "recorded_action": "unshifted_leader",
        "wrist_roll_zero_offset_degrees": source["wrist_roll_zero_offset_degrees"],
        "roll_zero_measurement": "verified constant" if resolved else "unmeasured constant",
        "note": "All six joints sync from the follower; every sim target uses the mapped leader action. "
        "The shared roll-zero constant is independent of episode starts; no sim-only re-zero."
        }
    if "selected_calibrations" in calibration:
        contract["selected_calibrations"] = calibration["selected_calibrations"]
    return contract

def validate_recording_root(root, contract):
    root = Path(root)
    if not root.exists() or not any(root.iterdir()):
        return
    marker = root / MARKER
    try:
        saved = json.loads(marker.read_text()) if marker.is_file() else None
    except json.JSONDecodeError:
        saved = None
    if saved != contract:
        raise RecordingAlignmentError(
            f"Recording alignment differs or is missing in {root}. "
            "Choose a new dataset root or set dataset.on_alignment_mismatch=new "
            "to keep recordings from different calibrations in separate folders.")


def mark_recording_root(root, contract):
    """Only called after the dataset owner has successfully initialized the new root."""
    path = Path(root) / MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n")


def synced_start_pose(mapped_follower):
    if len(mapped_follower) != 6:
        raise ValueError("Episode alignment needs six joints")
    if not all(math.isfinite(float(v)) for v in mapped_follower):
        raise ValueError("Follower start state must be finite")
    return mapped_follower.clone() if hasattr(mapped_follower, "clone") else mapped_follower.copy()
