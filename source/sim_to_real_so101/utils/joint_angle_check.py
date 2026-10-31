"""Read-only follower measurements and unapplied joint-angle candidates.

This module has no Isaac, LeRobot or serial imports. Hardware dependencies are
loaded only by ``make_observation_only_robot`` after an explicit live request.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .joint_calibration import FOLLOWER_JOINTS, load_device_calibration, resolve_calibration
from .calibration_check import CalibrationGapLogger, units_to_degrees
from .joint_angle_display import JointAngleTable

ORDER = tuple(name + '.pos' for name in FOLLOWER_JOINTS)
BODY_JOINTS = FOLLOWER_JOINTS[:5]
SAMPLE_COUNT = 30
STABILITY_LIMIT = 0.3
READBACK_LIMIT_DEGREES = 0.05


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite_vector(values, length=6):
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {length} finite joint values')
    return result


def compare_readback(follower, mapped_degrees, readback_degrees, limits):
    """Compare an actual PhysX getter result; never wrap away a full-turn error."""
    follower = finite_vector(follower)
    mapped = finite_vector(mapped_degrees)
    actual = finite_vector(readback_degrees)
    bounds = np.asarray(limits, dtype=float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Expected six finite, increasing runtime limit pairs')
    difference = actual - mapped
    at_limit = np.minimum(abs(mapped - bounds[:, 0]), abs(mapped - bounds[:, 1])) <= READBACK_LIMIT_DEGREES
    return dict(passed=bool(np.all(abs(difference) <= READBACK_LIMIT_DEGREES)),
                max_abs_difference_degrees=float(max(abs(difference))),
                joints={name: dict(follower_units=float(follower[i]), mapped_degrees=float(mapped[i]),
                                   physx_readback_degrees=float(actual[i]),
                                   difference_degrees=float(difference[i]), at_limit=bool(at_limit[i]),
                                   in_scope=name != 'gripper')
                        for i, name in enumerate(FOLLOWER_JOINTS)})


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


def angular_delta(measured, expected):
    return (np.asarray(measured) - np.asarray(expected) + 180) % 360 - 180


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
        if joint == 'shoulder_pan' and pan_reference is None:
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


def _forbidden(*args, **kwargs):
    raise RuntimeError('Read-only joint check forbids motion, torque enable, configuration and calibration')


def guard_observation_only_robot(robot):
    """Fail closed on writes, including indirect writes during installed connect().

    Feetech.disable_torque() writes Torque_Enable=0 and Lock=0. Those are the
    only allowed writes. A depth guard also rejects bypassing the register API.
    """
    bus = robot.bus
    original_write = bus.write
    original_low_write = getattr(bus, '_write', None)
    permit = [False]
    writes = []

    def guarded_write(data_name, motor, value, *args, **kwargs):
        if data_name not in {'Torque_Enable', 'Lock'} or isinstance(value, bool) or value != 0:
            return _forbidden()
        permit[0] = True
        try:
            result = original_write(data_name, motor, value, *args, **kwargs)
            writes.append(dict(register=data_name, motor=motor, value=0, timestamp_unix_ns=time.time_ns()))
            return result
        finally:
            permit[0] = False

    def guarded_low_write(*args, **kwargs):
        if not permit[0]:
            return _forbidden()
        return original_low_write(*args, **kwargs)

    # connect(calibrate=False) still invokes configure() in installed LeRobot.
    # Skip it completely: its torque_disabled() context unconditionally enables
    # torque on exit and it writes operating modes, gains and motor limits.
    robot.configure = lambda: None
    for name in ('send_action', 'calibrate', '_save_calibration', 'setup_motors'):
        setattr(robot, name, _forbidden)
    bus.write = guarded_write
    if original_low_write is not None:
        bus._write = guarded_low_write
    for name in ('enable_torque', 'write_calibration', 'configure_motors', 'setup_motor',
                 'set_half_turn_homings', 'record_ranges_of_motion', 'sync_write', '_sync_write'):
        setattr(bus, name, _forbidden)
    return writes


def make_observation_only_robot(port, robot_id, calibration_path):
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    path = Path(calibration_path)
    if not path.is_file():
        raise ValueError('An existing follower calibration file is required; none will be created')
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id,
                                              calibration_dir=path.parent, cameras={}, use_degrees=False,
                                              disable_torque_on_disconnect=False))
    load_device_calibration(robot, path)
    if robot.cameras or tuple(robot.bus.motors) != FOLLOWER_JOINTS:
        raise RuntimeError('Read-only mode requires exactly six motors and zero sensors')
    return robot


@contextmanager
def observation_only_session(robot, calibration_path, evidence, *, supported=False):
    if not supported:
        raise ValueError('Support the arm first: torque-off lets it drop under gravity')
    evidence['calibration_path'] = str(calibration_path)
    evidence['calibration_sha256_before'] = sha256(calibration_path)
    evidence['allowed_writes'] = guard_observation_only_robot(robot)
    try:
        robot.connect(calibrate=False)
        # Do this immediately, before any observation or further checks.
        robot.bus.disable_torque(num_retry=5)
        torque = {name: int(robot.bus.read('Torque_Enable', name, normalize=False))
                  for name in FOLLOWER_JOINTS}
        evidence['torque_enable_readback'] = torque
        if set(torque) != set(FOLLOWER_JOINTS) or any(torque.values()):
            raise RuntimeError(f'Torque-off read-back failed: {torque}')
        if not robot.is_calibrated:
            raise RuntimeError('Motor calibration differs from the existing file; refusing to calibrate or proceed')
        yield robot
    finally:
        try:
            if robot.bus.is_connected:
                robot.bus.disable_torque(num_retry=5)
                evidence['torque_enable_final'] = {
                    name: int(robot.bus.read('Torque_Enable', name, normalize=False)) for name in FOLLOWER_JOINTS}
                if any(evidence['torque_enable_final'].values()):
                    raise RuntimeError('Torque did not remain off')
        finally:
            try:
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=False)
            finally:
                evidence['calibration_sha256_after'] = sha256(calibration_path)
                evidence['calibration_unchanged'] = evidence['calibration_sha256_before'] == evidence['calibration_sha256_after']
                if not evidence['calibration_unchanged']:
                    raise RuntimeError('Calibration file changed during the read-only check')


def driver_provenance(robot):
    paths = {inspect.getsourcefile(type(robot)), inspect.getsourcefile(type(robot.bus))}
    # Include the installed motor-bus base that implements torque_disabled().
    paths.update(inspect.getsourcefile(cls) for cls in type(robot.bus).__mro__ if cls.__module__.startswith('lerobot.'))
    return {str(path): sha256(path) for path in sorted(path for path in paths if path)}


def telemetry_provenance(selection, **extra):
    """Collect software identity without importing simulation or hardware packages."""
    details = dict(selection.provenance)
    root = Path(__file__).resolve().parents[3]
    git = ['git', '-c', f'safe.directory={root}', '-C', str(root)]
    try:
        details['git_commit'] = subprocess.check_output(
            [*git, 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        details['git_dirty'] = bool(subprocess.check_output(
            [*git, 'status', '--porcelain', '--untracked-files=normal'],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        details['git_commit'] = None
        details['git_dirty'] = None
    details.update(extra)
    return details


class JointAngleTelemetry:
    """Passive tap on the existing teleoperation loop; owns no robot connection.

    Actual sim angles come from the PhysX getter, not the target or the Isaac
    cached joint-position buffer. No physical motor APIs are called here.
    """
    def __init__(self, path, *, selection=None, provenance=None, thresholds=None):
        from .joint_limits import runtime_joint_limits

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

    def observe(self, follower_observation, mapping_interface, sim_robot, previous_target=None, *, previous_command=None):
        from .joint_limits import USD_JOINTS

        def values(tensor):
            return tensor.detach().cpu().numpy().astype(float)

        start_ns = time.time_ns()
        raw = mapping_interface.get_raw_actions_tensor(follower_observation)
        # These are the SAME production functions used at recorder start sync.
        from .recording_alignment import synced_start_pose
        mapped = synced_start_pose(mapping_interface.get_mapped_actions_vectorized(raw))
        ids = [sim_robot.joint_names.index(name) for name in USD_JOINTS]
        actual = sim_robot.root_physx_view.get_dof_positions()[0, ids]
        follower = values(raw)
        mapped_degrees = np.rad2deg(values(mapped))
        actual_degrees = np.rad2deg(values(actual))
        real_degrees = units_to_degrees(follower, self.selection.mapping, FOLLOWER_JOINTS)
        commanded = None if previous_command is None else units_to_degrees(
            [previous_command[name] for name in ORDER], self.selection.mapping, FOLLOWER_JOINTS)
        legacy = dict(read_started_unix_ns=start_ns, read_finished_unix_ns=time.time_ns(),
                      joint_order=list(FOLLOWER_JOINTS), follower_units=follower.tolist(),
                      mapped_follower_degrees=mapped_degrees.tolist(), sim_physx_degrees=actual_degrees.tolist(),
                      sim_minus_mapped_follower_degrees=(actual_degrees - mapped_degrees).tolist(),
                      previous_sim_target_degrees=None if previous_target is None else np.rad2deg(values(previous_target)).tolist(),
                      at_runtime_limit=(np.minimum(abs(mapped_degrees - values(mapping_interface.joint_mins)),
                                                   abs(mapped_degrees - values(mapping_interface.joint_maxs))) <= .05).tolist(),
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


def summarize_teleop(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get('type') != 'provenance']
    if not rows:
        raise ValueError('No teleoperation frames were captured')
    raw = np.asarray([row['follower_units'] for row in rows])
    actual = np.asarray([row['sim_physx_degrees'] for row in rows])
    mapped = np.asarray([row['mapped_follower_degrees'] for row in rows])
    gaps = actual - mapped
    stable = []
    for start in range(0, len(rows) - SAMPLE_COUNT + 1, SAMPLE_COUNT):
        stop = start + SAMPLE_COUNT
        std = raw[start:stop].std(axis=0, ddof=1)
        stable.append(dict(first_frame=start, last_frame=stop - 1,
                           accepted=bool(np.all(std <= STABILITY_LIMIT)), std_follower_units=std.tolist(),
                           median_follower_units=np.median(raw[start:stop], axis=0).tolist(),
                           median_tracking_difference_degrees=np.median(gaps[start:stop], axis=0).tolist()))
    return dict(frames=len(rows), duration_seconds=(rows[-1]['read_finished_unix_ns'] - rows[0]['read_started_unix_ns']) / 1e9,
                evidence_sha256=sha256(path), stable_windows=stable,
                joints={name: dict(follower_units_min=float(raw[:, i].min()), follower_units_max=float(raw[:, i].max()),
                                   mapped_degrees_min=float(mapped[:, i].min()), mapped_degrees_max=float(mapped[:, i].max()),
                                   median_sim_minus_follower_degrees=float(np.median(gaps[:, i])),
                                   p95_absolute_tracking_difference_degrees=float(np.percentile(abs(gaps[:, i]), 95)))
                        for i, name in enumerate(FOLLOWER_JOINTS)},
                physical_offsets='UNRESOLVED: no independent inclinometer or pan reference measurements')
