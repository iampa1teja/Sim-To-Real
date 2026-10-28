"""Episode alignment and dataset-version guards, independent of Isaac and hardware."""
import hashlib
import json
import math
from pathlib import Path

from .joint_calibration import FOLLOWER_JOINTS, SETUP_PATH, resolve_calibration

MARKER = "meta/recording_alignment.json"


def recording_contract(*, follower_id, leader_id, setup=None, calibration_selection=None):
    setup = json.loads(SETUP_PATH.read_text()) if setup is None else setup
    selection = calibration_selection or resolve_calibration(setup=setup)
    resolved = setup["sim_joint_mapping_source"].get("wrist_roll_fit_status") == "verified"
    calibration = {key: setup[key] for key in ("sim_joint_mapping", "sim_joint_mapping_source", "runtime_joint_limits")}
    if selection.follower_path is not None or selection.leader_path is not None:
        calibration = dict(calibration, sim_joint_mapping=selection.mapping,
                           selected_calibrations={role: selection.provenance[f"{role}_calibration"]["sha256"]
                                                  for role in ("follower", "leader")})
    source = setup["sim_joint_mapping_source"]
    contract = {"version": 3, "follower_id": follower_id, "leader_id": leader_id,
                "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
                "start_sync": "follower_state", "roll_resolved": resolved,
                "synced_joints": list(FOLLOWER_JOINTS),
                "wrist_rezero": False, "recorded_action": "unshifted_leader",
                "wrist_roll_zero_offset_degrees": source["wrist_roll_zero_offset_degrees"],
                "roll_zero_measurement": "verified constant" if resolved else "unmeasured constant",
                "note": "All six joints sync from the follower; every sim target uses the mapped leader action. "
                "The shared roll-zero constant is independent of episode starts; no sim-only re-zero."}
    if "selected_calibrations" in calibration:
        contract["selected_calibrations"] = calibration["selected_calibrations"]
    return contract


def validate_recording_root(root, contract):
    root = Path(root)
    if not root.exists() or not any(root.iterdir()):
        return
    marker = root / MARKER
    if not marker.is_file() or json.loads(marker.read_text()) != contract:
        raise ValueError(f"Recording alignment differs or is missing in {root}. "
                         "Choose a NEW dataset root, e.g. pick_place_v3_start_synced; legacy and v2 roots cannot be resumed.")


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
