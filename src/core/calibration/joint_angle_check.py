"""Read-only follower measurements and unapplied joint-angle candidates.

This module has no Isaac, LeRobot or serial imports. It observes the devices
already owned by the caller and never creates a hardware connection.

Ported to Isaac Lab 3.0 / LeRobot 0.6.1: joints come from arm.toml, the mapping is
``core.arms.mapping.JointMapping``, and sim angles are read from ``robot.data.joint_pos``
(Newton has no PhysX tensor view). Log keys keep their v1 names (``sim_physx_degrees``,
``physx_readback_degrees``) so old and new telemetry stay comparable.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from core.arms.joint_calibration import FOLLOWER_JOINTS, GRIPPER_JOINT, resolve_calibration

from core.arms.mapping import units_to_degrees
from core.utils.provenance import git_identity

from .calibration_check import CalibrationGapLogger
from .joint_angle_display import JointAngleTable

ORDER = tuple(name + '.pos' for name in FOLLOWER_JOINTS)
BODY_JOINTS = tuple(name for name in FOLLOWER_JOINTS if name != GRIPPER_JOINT)
# Base yaw joint: its offset needs an external pan reference (SO-101: shoulder_pan).
PAN_JOINT = FOLLOWER_JOINTS[0]
JOINT_COUNT = len(FOLLOWER_JOINTS)
SAMPLE_COUNT = 30
STABILITY_LIMIT = 0.3


def finite_vector(values, length=JOINT_COUNT):
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {length} finite joint values')
    return result


def sample_pose(robot, *, sleep=time.sleep, clock=time.time_ns):
    rows, stamps = [], []
    for _ in range(SAMPLE_COUNT):
        observation = robot.get_observation()
        rows.append(finite_vector([observation[name] for name in ORDER]).tolist())
        stamps.append(int(clock()))
        sleep(1 / 30)
    values = np.asarray(rows)
    std = values.std(axis=0, ddof=1)
    return dict(count=SAMPLE_COUNT, observations=rows, timestamps_unix_ns=stamps,
                median=np.median(values, axis=0).tolist(), std=std.tolist(),
                stable=bool(np.all(std <= STABILITY_LIMIT)),
                rejected_joints=[name for name, spread in zip(FOLLOWER_JOINTS, std) if spread > STABILITY_LIMIT])


def summarize_candidates(pose_results, *, pan_reference=None):
    result = {'applied': False, 'gripper': 'OUT OF SCOPE', 'joints': {}}
    for joint in BODY_JOINTS:
        observations, seen = [], set()
        for row in pose_results:
            if (not row.get('accepted') or joint not in row.get('offsets_degrees', {})
                    or row.get('at_limit', {}).get(joint, False)):
                continue
            target = row.get('target_deg')
            reference = row['name'] if target is None else round(float(target[FOLLOWER_JOINTS.index(joint)]), 6)
            if reference in seen:
                continue
            seen.add(reference)
            observations.append(dict(pose=row['name'], degrees=float(row['offsets_degrees'][joint])))
        values = np.asarray([row['degrees'] for row in observations])
        if joint == PAN_JOINT and pan_reference is None:
            status = 'NOT MEASURED'
            observations, values = [], np.array([])
        else:
            status = 'UNRESOLVED'
        if len(values):
            spread = float(np.ptp(values))
            offset = float(np.median(values))
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            if len(values) >= 5 and spread <= 1.0:
                status = 'CONSISTENT'
        else:
            offset = spread = std = None
        branch_error = bool(np.any(abs(values) > 180))
        if branch_error:
            status = 'UNRESOLVED'
        result['joints'][joint] = dict(offset_degrees=offset, spread_degrees=spread,
                                       std_degrees=std, poses=len(values), status=status,
                                       samples=observations, suspected_branch_error=branch_error)
    result['spread_definition'] = 'max minus min of linear per-pose joint offsets; no angular wrapping'
    result['consistency_gate'] = 'at least 5 distinct accepted joint targets and spread <= 1 degree; saturated joints excluded'
    result['sign_convention'] = 'candidate is additive to the current mapped USD joint angle; nothing is applied'
    result['pan_reference'] = pan_reference
    return result


def telemetry_provenance(selection, **extra):
    """Collect software identity without importing simulation or hardware packages."""
    details = dict(selection.provenance)
    root = Path(__file__).resolve().parents[3]
    details.update(git_identity(root, timeout=5, untracked_files="normal"))
    details.update(extra)
    return details


class JointAngleTelemetry:
    """Passive tap on the existing teleoperation loop; owns no robot connection.

    Actual sim angles come from the articulation joint state (``data.joint_pos``),
    not the target. No physical motor APIs are called here.
    """
    def __init__(self, path, *, selection=None, provenance=None, thresholds=None):
        from core.arms.joint_limits import runtime_joint_limits

        self.selection = selection or resolve_calibration()
        details = telemetry_provenance(self.selection, **(provenance or {}))
        bounds = runtime_joint_limits()
        self.logger = CalibrationGapLogger(path, joint_order=FOLLOWER_JOINTS,
                                           runtime_limits=[[bounds[n]['joint_min'], bounds[n]['joint_max']]
                                                           for n in FOLLOWER_JOINTS],
                                           provenance=details, thresholds=thresholds)
        self.path, self.stream = self.logger.path, self.logger.stream
        self.frames = 0
        self.last_print = 0.
        self.table = JointAngleTable(FOLLOWER_JOINTS)

    def observe(self, follower_observation, mapping, sim_robot, previous_target=None, *, previous_command=None):
        """Log one tick. ``mapping`` is the agent's ``core.arms.mapping.JointMapping``."""
        from core.arms.joint_limits import joint_name_pairs

        def values(tensor):
            return tensor.detach().cpu().numpy().astype(float)

        start_ns = time.time_ns()
        raw = mapping.raw_from_actions(follower_observation)
        # The same mapping the agent uses for every sim target.
        mapped = mapping.to_radians(raw)
        if not bool(mapped.isfinite().all()):
            raise ValueError('Mapped follower joints must be finite')
        ids = [sim_robot.joint_names.index(usd) for _, usd in joint_name_pairs()]
        joint_pos = sim_robot.data.joint_pos
        joint_pos = joint_pos.torch if hasattr(joint_pos, 'torch') else joint_pos
        actual = joint_pos[0, ids]
        follower = values(raw)
        mapped_degrees = np.rad2deg(values(mapped))
        actual_degrees = np.rad2deg(values(actual))
        real_degrees = units_to_degrees(follower, self.selection.mapping, FOLLOWER_JOINTS, gripper_joints=(GRIPPER_JOINT,))
        commanded = None if previous_command is None else units_to_degrees(
            [previous_command[name] for name in ORDER], self.selection.mapping, FOLLOWER_JOINTS,
            gripper_joints=(GRIPPER_JOINT,))
        legacy = dict(read_started_unix_ns=start_ns, read_finished_unix_ns=time.time_ns(),
                      joint_order=list(FOLLOWER_JOINTS), follower_units=follower.tolist(),
                      mapped_follower_degrees=mapped_degrees.tolist(), sim_physx_degrees=actual_degrees.tolist(),
                      sim_minus_mapped_follower_degrees=(actual_degrees - mapped_degrees).tolist(),
                      previous_sim_target_degrees=None if previous_target is None else np.rad2deg(values(previous_target)).tolist(),
                      at_runtime_limit=(np.minimum(abs(mapped_degrees - values(mapping.clamp_mins)),
                                                   abs(mapped_degrees - values(mapping.clamp_maxs))) <= .05).tolist(),
                      interpretation='encoder/mapping/tracking comparison; not independent physical angle calibration')
        row = self.logger.record(commanded_deg=commanded, real_deg=real_degrees, sim_deg=actual_degrees,
                                 sim_target_deg=None if previous_target is None else np.rad2deg(values(previous_target)),
                                 timestamp_ns=start_ns, extra=legacy)
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
