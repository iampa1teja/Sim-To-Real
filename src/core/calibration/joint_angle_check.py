"""Passive per-joint telemetry over devices already owned by the caller."""
from pathlib import Path
import time

import numpy as np

from core.arms.calibration import resolve_calibration
from core.utils.provenance import git_identity
from .calibration_check import CalibrationGapLogger
from .joint_angle_display import JointAngleTable


def telemetry_provenance(selection, **extra):
    details = dict(selection.provenance)
    details.update(joint_units=[joint.unit for joint in selection.arm.joints],
                   joint_roles=[joint.role for joint in selection.arm.joints],
                   gripper_joints=list(selection.arm.gripper_joint_names))
    root = Path(__file__).resolve().parents[3]
    details.update(git_identity(root, timeout=5, untracked_files="normal"))
    details.update(extra)
    return details


class JointAngleTelemetry:
    """Observe encoder and articulation state without creating a device connection."""
    def __init__(self, path, *, selection=None, arm=None, unit=None, provenance=None, thresholds=None):
        self.selection = selection or resolve_calibration(arm, unit=unit)
        self.arm = self.selection.arm
        self.names = self.arm.joint_names
        self.keys = self.arm.feature_names
        self.units = tuple(joint.unit for joint in self.arm.joints)
        self.legacy_degrees = all(unit == "deg" for unit in self.units)
        details = telemetry_provenance(self.selection, **(provenance or {}))
        self.logger = CalibrationGapLogger(path, joint_order=self.names,
                                           runtime_limits=[joint.limits for joint in self.arm.joints],
                                           provenance=details, thresholds=thresholds)
        self.path, self.stream = self.logger.path, self.logger.stream
        self.frames, self.last_print = 0, 0.0
        self.table = JointAngleTable(self.names, gripper_joints=self.arm.gripper_joint_names, joint_units=self.units)

    def observe(self, follower_observation, mapping, sim_robot, previous_target=None, *, previous_command=None):
        def values(tensor):
            return tensor.detach().cpu().numpy().astype(float)
        start_ns = time.time_ns()
        raw = mapping.raw_from_actions(follower_observation)
        target = mapping.to_sim(raw)
        if not bool(target.isfinite().all()):
            raise ValueError("Mapped follower joints must be finite")
        ids = [sim_robot.joint_names.index(usd) for usd in self.arm.usd_joint_names]
        state = sim_robot.data.joint_pos
        state = state.torch if hasattr(state, "torch") else state
        actual = state[0, ids]
        real_values = values(mapping.to_physical(raw))
        target_values = values(mapping.sim_to_physical(target))
        actual_values = values(mapping.sim_to_physical(actual))
        commanded = None if previous_command is None else values(mapping.to_physical(
            [previous_command[key] for key in self.keys]))
        previous = None if previous_target is None else values(mapping.sim_to_physical(previous_target))
        extra = dict(read_started_unix_ns=start_ns, read_finished_unix_ns=time.time_ns(),
                     joint_order=list(self.names), follower_units=values(raw).tolist(),
                     interpretation="encoder/mapping/tracking comparison; no independent physical zero measurement")
        if self.legacy_degrees:
            extra.update(mapped_follower_degrees=target_values.tolist(), sim_physx_degrees=actual_values.tolist(),
                         sim_minus_mapped_follower_degrees=(actual_values-target_values).tolist(),
                         previous_sim_target_degrees=None if previous is None else previous.tolist())
        else:
            extra.update(mapped_follower=target_values.tolist(), actual_sim=actual_values.tolist(),
                         sim_minus_mapped_follower=(actual_values-target_values).tolist(),
                         previous_sim_target=None if previous is None else previous.tolist())
        row = self.logger.record(commanded=commanded, real=real_values, sim=actual_values, sim_target=previous,
                                 timestamp_ns=start_ns, extra=extra)
        self.frames += 1
        now = time.monotonic()
        if now - self.last_print >= 1:
            self.last_print = now
            self.table.update(row)
        return row

    def close(self):
        try:
            self.table.close()
        finally:
            self.logger.close()
        return self.logger.summary
