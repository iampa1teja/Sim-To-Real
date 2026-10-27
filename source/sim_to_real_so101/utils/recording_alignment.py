"""Episode alignment and dataset-version guards, independent of Isaac and hardware."""
import hashlib
import json
import math
from pathlib import Path

from .joint_calibration import SETUP_PATH

MARKER = "meta/recording_alignment.json"


def recording_contract(*, follower_id, leader_id, setup=None):
    setup = json.loads(SETUP_PATH.read_text()) if setup is None else setup
    resolved = setup["sim_joint_mapping_source"].get("wrist_roll_fit_status") == "verified"
    calibration = {key: setup[key] for key in ("sim_joint_mapping", "sim_joint_mapping_source", "runtime_joint_limits")}
    return {"version": 2, "follower_id": follower_id, "leader_id": leader_id,
            "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
            "start_sync": "follower_state", "roll_resolved": resolved,
            "wrist_rezero": not resolved, "recorded_action": "unshifted_leader",
            "note": "When roll is unresolved, the executed sim wrist target retains re-zero; logged actions stay paired."}


def validate_recording_root(root, contract):
    root = Path(root)
    if not root.exists() or not any(root.iterdir()):
        return
    marker = root / MARKER
    if not marker.is_file() or json.loads(marker.read_text()) != contract:
        raise ValueError(f"Recording alignment differs or is missing in {root}. "
                         "Choose a NEW dataset root, e.g. pick_place_v2_start_synced; legacy roots cannot be resumed.")


def mark_recording_root(root, contract):
    """Only called after the dataset owner has successfully initialized the new root."""
    path = Path(root) / MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n")


def synced_start_pose(mapped_follower, current_sim, *, roll_resolved):
    if len(mapped_follower) != 6 or len(current_sim) != 6:
        raise ValueError("Episode alignment needs six joints")
    if not all(math.isfinite(float(v)) for v in mapped_follower):
        raise ValueError("Follower start state must be finite")
    result = mapped_follower.clone() if hasattr(mapped_follower, "clone") else mapped_follower.copy()
    if not roll_resolved:
        result[4] = current_sim[4]
    return result


def wrist_target(mapped_roll, sim_start, leader_start, *, roll_resolved):
    return mapped_roll if roll_resolved else sim_start + mapped_roll - leader_start
