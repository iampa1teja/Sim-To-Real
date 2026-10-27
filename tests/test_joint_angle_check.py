"""Joint-angle checks must never actuate or calibrate the follower."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from sim_to_real_so101.utils.joint_angle_check import (
    FOLLOWER_JOINTS, ORDER, compare_readback, guard_observation_only_robot,
    observation_only_session, sample_pose, summarize_candidates,
)


def comparison(actual):
    return compare_readback(np.arange(6), np.zeros(6), actual, [[-90, 90]] * 6)


def test_readback_tolerance_and_limit_flags():
    assert comparison([0.049, 0, 0, 0, 0, 0])['passed']
    assert not comparison([0.051, 0, 0, 0, 0, 0])['passed']
    assert not comparison([360, 0, 0, 0, 0, 0])['passed']
    result = compare_readback([0] * 6, [90, 89.9, -90, 0, 0, 0],
                              [90, 89.9, -90, 0, 0, 0], [[-90, 90]] * 6)
    assert result['joints']['shoulder_pan']['at_limit']
    assert result['joints']['elbow_flex']['at_limit']
    assert not result['joints']['shoulder_lift']['at_limit']
    assert not result['joints']['gripper']['in_scope']


@pytest.mark.parametrize('invalid', [[0] * 5, [0] * 5 + [np.nan], [np.inf] * 6])
def test_readback_rejects_invalid_values(invalid):
    with pytest.raises(ValueError):
        comparison(invalid)


class FakeBus:
    def __init__(self):
        self.is_connected = False
        self.motors = dict.fromkeys(FOLLOWER_JOINTS)
        self.torque = dict.fromkeys(FOLLOWER_JOINTS, 1)
        self.events = []

    def forbidden(self, *args, **kwargs):
        raise AssertionError('A motion, torque-enable or calibration function was called')

    enable_torque = write_calibration = configure_motors = sync_write = forbidden

    def write(self, register, motor, value, **kwargs):
        assert register in {'Torque_Enable', 'Lock'} and value == 0
        self.events.append(('write', register, motor, value))
        self._write(register, motor, value)
        if register == 'Torque_Enable':
            self.torque[motor] = value

    def _write(self, register, motor, value):
        assert register in {'Torque_Enable', 'Lock'} and value == 0

    def connect(self):
        self.events.append(('connect',))
        self.is_connected = True

    def disable_torque(self, **kwargs):
        self.events.append(('disable_torque',))
        for motor in self.motors:
            self.write('Torque_Enable', motor, 0)
            self.write('Lock', motor, 0)

    def read(self, register, motor, **kwargs):
        assert register == 'Torque_Enable' and kwargs['normalize'] is False
        self.events.append(('read_torque', motor))
        return self.torque[motor]

    def disconnect(self, *, disable_torque):
        assert disable_torque is False
        self.is_connected = False


class FakeRobot:
    def __init__(self):
        self.bus = FakeBus()
        self.count = 0
        self.is_calibrated = True

    def forbidden(self, *args, **kwargs):
        raise AssertionError('A motion, torque-enable or calibration function was called')

    send_action = calibrate = _save_calibration = forbidden

    def configure(self):
        # Emulate the installed driver's dangerous configure() exit.
        self.bus.enable_torque()

    def connect(self, *, calibrate):
        assert calibrate is False
        self.bus.connect()
        self.configure()

    def get_observation(self):
        assert not any(self.bus.torque.values())
        self.bus.events.append(('observation',))
        self.count += 1
        return dict.fromkeys(ORDER, 12.)


def test_live_workflow_never_calls_motion_enable_or_calibration(tmp_path):
    path = tmp_path / 'rpd_robot.json'
    path.write_text('{"unchanged": true}\n')
    before = path.read_bytes()
    robot, evidence = FakeRobot(), {}
    with observation_only_session(robot, path, evidence, supported=True) as reader:
        result = sample_pose(reader, sleep=lambda _: None, clock=lambda: 123)
    assert result['count'] == robot.count == 30
    assert result['stable'] and result['std'] == [0.] * 6
    assert result['median'] == [12.] * 6
    assert path.read_bytes() == before
    assert evidence['calibration_unchanged']
    assert set(evidence['torque_enable_readback'].values()) == {0}
    assert set(evidence['torque_enable_final'].values()) == {0}
    first_observation = robot.bus.events.index(('observation',))
    assert len([row for row in robot.bus.events[:first_observation] if row[0] == 'read_torque']) == 6
    assert robot.bus.events[:2] == [('connect',), ('disable_torque',)]
    assert all(row['value'] == 0 for row in evidence['allowed_writes'])
    assert not robot.bus.is_connected


def test_requires_support_before_connection(tmp_path):
    robot = FakeRobot()
    with pytest.raises(ValueError, match='Support'):
        with observation_only_session(robot, tmp_path / 'missing', {}, supported=False):
            pass
    assert not robot.bus.events


@pytest.mark.parametrize('method,args', [
    ('write', ('Goal_Position', 'shoulder_pan', 0)),
    ('write', ('Torque_Enable', 'shoulder_pan', 1)),
    ('write', ('Homing_Offset', 'shoulder_pan', 0)),
    ('write', ('Min_Position_Limit', 'shoulder_pan', 0)),
    ('_write', (40, 1, 1, 1)),
    ('sync_write', ('Goal_Position', {'shoulder_pan': 0})),
    ('enable_torque', ()), ('write_calibration', ({},)), ('configure_motors', ()),
])
def test_guard_refuses_every_non_safety_write(method, args):
    robot = FakeRobot()
    guard_observation_only_robot(robot)
    with pytest.raises(RuntimeError, match='Read-only'):
        getattr(robot.bus, method)(*args)
    assert not robot.bus.events


def test_unstable_pose_rejects_even_out_of_scope_gripper():
    robot = FakeRobot()
    robot.bus.torque = dict.fromkeys(FOLLOWER_JOINTS, 0)
    def observation():
        robot.count += 1
        result = dict.fromkeys(ORDER, 0.)
        result['gripper.pos'] = float(robot.count % 2)
        return result
    robot.get_observation = observation
    result = sample_pose(robot, sleep=lambda _: None)
    assert not result['stable'] and result['rejected_joints'] == ['gripper']


def test_exception_keeps_torque_off_and_checks_calibration(tmp_path):
    path = tmp_path / 'rpd_robot.json'
    path.write_text('{}')
    robot, evidence = FakeRobot(), {}
    with pytest.raises(LookupError):
        with observation_only_session(robot, path, evidence, supported=True):
            raise LookupError('interrupted measurement')
    assert evidence['calibration_unchanged'] and not robot.bus.is_connected
    assert not any(evidence['torque_enable_final'].values())


def test_calibration_mismatch_refuses_without_recalibrating(tmp_path):
    path = tmp_path / 'rpd_robot.json'
    path.write_text('{}')
    robot = FakeRobot()
    robot.is_calibrated = False
    with pytest.raises(RuntimeError, match='differs'):
        with observation_only_session(robot, path, {}, supported=True):
            pytest.fail('Mismatch must not permit observations')
    assert not robot.bus.is_connected


def test_consistency_requires_five_poses_and_range_not_just_std():
    rows = [dict(name=str(i), accepted=True,
                 offsets_degrees={'shoulder_lift': value, 'elbow_flex': .5}, at_limit={})
            for i, value in enumerate([0., 0., 0., 0., 1.1])]
    result = summarize_candidates(rows)
    assert result['joints']['shoulder_lift']['std_degrees'] < 1
    assert result['joints']['shoulder_lift']['status'] == 'UNRESOLVED'
    assert result['joints']['elbow_flex']['status'] == 'CONSISTENT'
    assert result['joints']['shoulder_pan']['status'] == 'NOT MEASURED'
    assert result['joints']['wrist_roll']['offset_degrees'] is None
    assert summarize_candidates(rows[:4])['joints']['elbow_flex']['status'] == 'UNRESOLVED'


def test_limit_saturation_cannot_produce_a_consistent_offset():
    rows = [dict(name=str(i), accepted=True, offsets_degrees={'shoulder_lift': 0.},
                 at_limit={'shoulder_lift': True}) for i in range(5)]
    assert summarize_candidates(rows)['joints']['shoulder_lift']['poses'] == 0


def test_installed_lerobot_connect_with_fake_motors_does_not_enable_torque(tmp_path):
    module = pytest.importorskip('lerobot.robots.so101_follower')
    # Execute the installed production connect(), not an imitation of it.
    robot = object.__new__(module.SO101Follower)
    robot.bus = FakeBus()
    robot.cameras = {}
    robot.id = 'read_only_test'
    robot.calibration = {}
    # The real property delegates to the bus; do not replace connect().
    robot.bus.is_calibrated = True
    path = tmp_path / 'rpd_robot.json'
    path.write_text('{}')
    with observation_only_session(robot, path, {}, supported=True):
        assert robot.is_connected
        assert not any(robot.bus.torque.values())


def test_teleop_observer_reads_physx_and_never_controls_a_robot(tmp_path):
    torch = pytest.importorskip('torch')
    import json
    from sim_to_real_so101.utils.joint_angle_check import JointAngleTelemetry, summarize_teleop
    from sim_to_real_so101.utils.joint_limits import USD_JOINTS
    from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
    setup = json.loads((Path(__file__).resolve().parents[1] / 'source/sim_to_real_so101/assets/real_setup.json').read_text())
    iface = LeRobotSO101Interface('cpu', '', '', {}, 30, joint_mapping=setup['sim_joint_mapping'])
    observation = dict(zip(ORDER, [10., 20., 30., 40., 99., 8.]))
    expected = iface.get_mapped_actions_vectorized(iface.get_raw_actions_tensor(observation))
    actual = expected.clone()
    actual[2] += np.deg2rad(2.)
    def forbid(*args, **kwargs):
        raise AssertionError('Observer tried to control a robot')
    iface.robot = SimpleNamespace(send_action=forbid, connect=forbid, calibrate=forbid)
    # Deliberately disagree with both the target and the cached state.
    sim = SimpleNamespace(joint_names=list(USD_JOINTS),
                          root_physx_view=SimpleNamespace(get_dof_positions=lambda: actual.unsqueeze(0)),
                          data=SimpleNamespace(joint_pos=torch.ones(1,6)*100),
                          write_joint_state_to_sim=forbid, set_joint_position_target=forbid)
    log = tmp_path / 'teleop.jsonl'
    trace = JointAngleTelemetry(log)
    try:
        for _ in range(30):
            row = trace.observe(observation, iface, sim, torch.zeros(6))
    finally:
        trace.close()
    assert row['sim_minus_mapped_follower_degrees'][2] == pytest.approx(2., abs=1e-5)
    assert row['at_runtime_limit'][4]
    assert row['follower_units'] == list(observation.values())
    result = summarize_teleop(log)
    assert result['frames'] == 30 and result['stable_windows'][0]['accepted']
    assert result['physical_offsets'].startswith('UNRESOLVED')
    assert result['joints']['elbow_flex']['median_sim_minus_follower_degrees'] == pytest.approx(2., abs=1e-5)


def test_surface_tilts_use_longitudinal_link_axes_and_flag_vertical_roll_axis():
    from sim_to_real_so101.utils.joint_angle_check_physx import expected_tilts, rotation_xyzw
    identity = rotation_xyzw([0.,0.,0.,1.])
    result = expected_tilts(dict.fromkeys(('upper_arm','lower_arm','gripper'), identity))
    assert result['upper_arm']['world_pitch_degrees'] == 0
    assert result['forearm']['world_pitch_degrees'] == 0
    assert result['housing']['world_pitch_degrees'] == -90
    assert result['housing']['roll_axis_horizontal'] is False
    theta = np.deg2rad(45.)
    rotated = rotation_xyzw([0.,np.sin(theta/2),0.,np.cos(theta/2)])
    result = expected_tilts(dict.fromkeys(('upper_arm','lower_arm','gripper'), rotated))
    assert result['upper_arm']['world_pitch_degrees'] == pytest.approx(45.)
    assert result['forearm']['world_pitch_degrees'] == pytest.approx(45.)
    assert result['housing']['world_pitch_degrees'] == pytest.approx(-45.)
    theta = np.deg2rad(90.)
    horizontal = rotation_xyzw([0.,np.sin(theta/2),0.,np.cos(theta/2)])
    result = expected_tilts(dict.fromkeys(('upper_arm','lower_arm','gripper'), horizontal))
    assert result['housing']['roll_axis_horizontal'] is True


def test_invalid_physx_quaternion_refuses_geometry_prediction():
    from sim_to_real_so101.utils.joint_angle_check_physx import rotation_xyzw
    for quaternion in ([0.,0.,0.,0.], [0.,0.,np.nan,1.]):
        with pytest.raises(ValueError):
            rotation_xyzw(quaternion)


def test_summary_does_not_promote_constant_tracking_error_to_physical_offset(tmp_path):
    import json
    from sim_to_real_so101.utils.joint_angle_check import summarize_teleop
    log = tmp_path / 'teleop.jsonl'
    rows = [dict(follower_units=[0.]*6, sim_physx_degrees=[12.]*6,
                 mapped_follower_degrees=[0.]*6,
                 read_started_unix_ns=i*1_000_000, read_finished_unix_ns=i*1_000_000+100)
            for i in range(180)]
    log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = summarize_teleop(log)
    assert len([row for row in summary['stable_windows'] if row['accepted']]) == 6
    assert summary['joints']['wrist_roll']['median_sim_minus_follower_degrees'] == 12
    assert summary['physical_offsets'].startswith('UNRESOLVED')
    assert summarize_candidates([])['joints']['wrist_roll']['offset_degrees'] is None


def test_physx_only_path_contains_no_hardware_connection_or_physics_step():
    import ast
    from sim_to_real_so101.utils import joint_angle_check_physx as module
    tree = ast.parse(Path(module.__file__).read_text())
    names = {node.func.attr for node in ast.walk(tree)
             if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert not names & {'connect','init_device','send_action','enable_torque','calibrate','step','render'}
    assert {'get_dof_positions','write_joint_state_to_sim','update_articulations_kinematic'} <= names

