"""Record the selected arm/mapping and reject calibration changes on dataset resume."""
import json
from pathlib import Path

from utils.arms.calibration import resolve_calibration
from utils.arms.arms import load_arm_spec
from utils.hashing import legacy_json_sha256

MARKER = "meta/recording_alignment.json"


class RecordingAlignmentError(ValueError):
    """An existing dataset cannot be resumed with the selected calibration."""


def recording_contract(*, follower_id, leader_id, calibration_selection=None, arm=None, unit=None):
    arm = (getattr(calibration_selection, "arm", None) or load_arm_spec()[0]) if arm is None else arm
    selection = calibration_selection or resolve_calibration(arm, unit=unit)
    source = selection.unit.source if getattr(selection, "unit", None) is not None else {}
    selected = {role: selection.provenance[f"{role}_calibration"]["sha256"] for role in ("follower", "leader")}
    calibration = dict(arm=arm.name, mapping=selection.mapping, runtime_joint_limits=arm.runtime_joint_limits,
                       parameters=selection.provenance["mapping_parameters"], selected_calibrations=selected)
    contract = dict(
        version=4, arm=arm.name, unit=selection.provenance.get("unit"),
        joint_names=list(arm.joint_names), joint_units=[joint.unit for joint in arm.action_joints],
        joint_roles=[joint.role for joint in arm.action_joints], use_degrees=arm.lerobot.use_degrees,
        joint_mapping=selection.mapping, follower_id=follower_id, leader_id=leader_id,
        calibration_sha256=legacy_json_sha256(calibration),
        start_sync="follower_state", synced_joints=list(arm.joint_names),
        wrist_rezero=bool(arm.wrist_roll and arm.wrist_roll.relative_in_teleop),
        sim_wrist_relative=bool(arm.wrist_roll and arm.wrist_roll.relative_in_teleop),
        recorded_action="unshifted_leader", selected_calibrations=selected,
        note="Sim targets use the selected mapping and declared relative-roll behavior; recorded actions preserve device units.",
    )
    if arm.wrist_roll is not None:
        contract.update(roll_resolved=source.get("wrist_roll_fit_status") == "verified",
                        wrist_roll_zero_offset_degrees=selection.provenance["mapping_parameters"].get(
                            arm.wrist_roll.joint, {}).get("offset", 0.0))
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
    # Legacy markers did not store enough arm/mapping information to establish
    # equivalence. Keep them readable for replay; resume needs a new root.
    if saved != contract:
        raise RecordingAlignmentError(
            f"Recording alignment differs or is missing in {root}. "
            "Choose a new dataset root or set dataset.on_alignment_mismatch=new "
            "to keep recordings from different calibrations in separate folders.")


def mark_recording_root(root, contract):
    path = Path(root) / MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n")
