"""Real-device dispatch and cancellation contracts, using injected hardware only."""
from types import SimpleNamespace
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from utils.arms.arms import load_arm_spec
from core.rollout.testing_gui import RealArmSession, parse_args
from core.rollout.testing_http import TestingHttpServer as ConsoleServer


def wait_for(predicate, timeout=4.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for the test state')


class Robot:
    def __init__(self, arm):
        self.values = {j.name + '.pos': 50. if j.role == 'gripper' else 0. for j in arm.action_joints}
        self.sent, self.threads = [], []
        self.disconnected = False
        self.bad_observation = False

    def get_observation(self):
        self.threads.append(threading.get_ident())
        if self.bad_observation:
            raise RuntimeError('camera unplugged')
        image = np.full((24, 32, 3), 100, dtype=np.uint8)
        return {**self.values, 'external': image, 'gripper': image}

    def send_action(self, action):
        self.threads.append(threading.get_ident())
        self.sent.append(dict(action))
        self.values.update(action)
        return action

    def disconnect(self):
        self.threads.append(threading.get_ident())
        self.disconnected = True


class Policy:
    def __init__(self, args, robot, *, block=None, invalid=False):
        self.args, self.robot, self.block, self.invalid = args, robot, block, invalid
        self.entered = threading.Event()
        self.closed = False
        self.threads = []

    def connect(self):
        self.threads.append(threading.get_ident())

    def reset(self):
        self.threads.append(threading.get_ident())

    def get_raw_action(self, state, images):
        self.threads.append(threading.get_ident())
        assert set(images) == {'room', 'wrist'}
        self.entered.set()
        if self.block:
            assert self.block.wait(5), 'test must release blocked inference'
        if self.invalid:
            return {key: np.nan for key in self.robot.values}
        return {key: float(value) + .1 for key, value in zip(self.robot.values, state)}

    def close(self):
        self.threads.append(threading.get_ident())
        self.closed = True


@pytest.fixture
def rig(tmp_path):
    args = parse_args(['--arm', 'so101', '--policy', 'example', '--follower_id', 'test-follower',
                       '--follower_port', 'test-port', '--camera_gripper', 'test-gripper',
                       '--camera_external', 'test-external', '--output', str(tmp_path)])
    arm = load_arm_spec('so101')[0]
    robot = Robot(arm)
    policies, settings = [], {}
    def validate(selected):
        selected.arm_spec = arm
        selected.calibration_selection = SimpleNamespace(mapping=arm.runtime_joint_limits,
            provenance={'test': True, 'follower_calibration': {'sha256': 'f' * 64}})
        return []
    def factory(selected):
        policy = Policy(selected, robot, **settings)
        policies.append(policy)
        return policy
    manifest = SimpleNamespace(description='A test policy', action_dim='arm', cameras=('room', 'wrist'),
                               default_port=5600, language=True)
    session = RealArmSession(args, follower_factory=lambda _: robot, policy_factory=factory,
                             manifests={'example': manifest}, validator=validate)
    yield SimpleNamespace(session=session, robot=robot, policies=policies, settings=settings,
                          manifest=manifest, args=args)
    for policy in policies:
        if policy.block:
            policy.block.set()
    session.close()
    assert not session._thread.is_alive()


def connect(rig):
    rig.session.request_connect()
    wait_for(lambda: rig.session.snapshot()['phase'] == 'ready')


def run_body(**kw):
    return dict(policy='example', prompt='Place the blue cube in the box', client_id='test-browser',
                host='localhost', port=5600, seconds=.3, **kw)


def test_connect_is_read_only_and_prompt_reaches_selected_policy(rig):
    assert not rig.session.snapshot()['connected']
    assert not rig.policies and not rig.robot.sent
    connect(rig)
    wait_for(lambda: rig.session.frame('external') is not None)
    assert not rig.robot.sent and not rig.policies
    rig.session.request_run(run_body())
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    summary = rig.session.snapshot()['history'][0]
    assert summary['status'] == 'completed' and summary['steps'] > 0
    assert rig.robot.sent and rig.policies[0].args.lang_instruction == 'Place the blue cube in the box'
    assert rig.policies[0].args.policy == 'example' and rig.policies[0].args.policy_port == 5600
    assert rig.policies[0].closed
    assert len(set(rig.policies[0].threads)) == 1
    assert len(set(rig.robot.threads)) == 1
    assert set(rig.policies[0].threads).isdisjoint(rig.robot.threads)
    report = json.loads(rig.session.report(summary['id']))
    assert report['motion'] and report['outcome'] == 'unlabeled'


def test_stop_during_blocked_inference_never_sends_late_reply(rig):
    release = threading.Event()
    rig.settings['block'] = release
    connect(rig)
    rig.session.request_run({**run_body(), 'seconds': 10})
    wait_for(lambda: rig.policies and rig.policies[0].entered.is_set())
    sequence = rig.session.snapshot()['frame_seq']
    rig.session.request_stop()
    wait_for(lambda: rig.session.snapshot()['frame_seq'] > sequence)
    assert not rig.robot.sent
    release.set()
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert rig.session.snapshot()['history'][0]['status'] == 'stopped'
    assert not rig.robot.sent


def test_browser_lease_expiry_cancels_pending_motion(rig):
    release = threading.Event(); rig.settings['block'] = release
    connect(rig); rig.session.request_run({**run_body(), 'seconds': 10})
    wait_for(lambda: rig.policies and rig.policies[0].entered.is_set())
    rig.session._heartbeat_timeout = .12
    rig.session.heartbeat('test-browser')
    wait_for(lambda: rig.session.snapshot()['phase'] == 'stopping')
    release.set(); wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert not rig.robot.sent
    assert rig.session.snapshot()['history'][0]['detail'] == 'Browser heartbeat lost.'


def test_stop_before_policy_preparation_is_a_valid_zero_command_run(rig):
    connect(rig)
    with rig.session._lock:
        rig.session.request_run(run_body())
        rig.session.request_stop()
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert not rig.robot.sent and all(p.closed for p in rig.policies)
    assert rig.session.snapshot()['history'][0]['status'] == 'stopped'


def test_invalid_targets_fail_without_movement_and_allow_a_new_run(rig):
    rig.settings['invalid'] = True
    connect(rig); rig.session.request_run(run_body())
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert rig.session.snapshot()['history'][0]['status'] == 'action_error'
    assert not rig.robot.sent and rig.policies[0].closed
    rig.settings['invalid'] = False
    rig.session.request_run(run_body()); wait_for(lambda: len(rig.session.snapshot()['history']) == 2)
    assert rig.robot.sent


def test_oversized_first_target_never_moves(rig):
    rig.args.max_start_error = .01
    connect(rig); rig.session.request_run(run_body())
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert rig.session.snapshot()['history'][0]['status'] == 'start_too_far'
    assert not rig.robot.sent


def test_incompatible_policy_and_nonfinite_requests_are_rejected(rig):
    connect(rig)
    rig.manifest.action_dim = 8
    with pytest.raises(ValueError, match='Requires 8 joints'):
        rig.session.request_run(run_body())
    rig.manifest.action_dim = 'arm'
    for changes in ({'seconds': float('nan')}, {'seconds': True}, {'port': True}, {'prompt': '  '}):
        with pytest.raises(ValueError):
            rig.session.request_run({**run_body(), **changes})
    assert not rig.robot.sent


def test_observation_failure_stops_and_disconnects(rig):
    connect(rig); rig.session.request_run({**run_body(), 'seconds': 10})
    wait_for(lambda: bool(rig.robot.sent))
    rig.robot.bad_observation = True
    wait_for(lambda: rig.robot.disconnected)
    assert rig.session.snapshot()['history'][0]['status'] == 'robot_error'


def test_output_failure_sends_no_actions_and_leaves_the_console_usable(rig, tmp_path):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('occupied')
    rig.args.output = blocked
    connect(rig)
    rig.session.request_run(run_body())
    wait_for(lambda: bool(rig.session.snapshot()['history']))
    assert rig.session.snapshot()['history'][0]['status'] == 'report_error'
    assert not rig.robot.sent and not rig.policies
    assert 'Could not save' in rig.session.snapshot()['error']
    rig.args.output = tmp_path / 'recovered'
    rig.session.request_run(run_body())
    wait_for(lambda: len(rig.session.snapshot()['history']) == 2)
    assert rig.robot.sent


def test_http_commands_cannot_be_sent_by_another_origin(rig):
    server = ConsoleServer(rig.session, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        assert b'Task prompt' in urlopen(base).read()
        assert json.load(urlopen(base + '/api/config'))['arm'] == 'so101'
        req = Request(base + '/api/connect', data=b'{}', headers={'Content-Type':'application/json', 'Origin':'https://another.example'})
        with pytest.raises(HTTPError) as e:urlopen(req)
        assert e.value.code == 403
        req = Request(base + '/api/connect', data=b'{}', headers={'Content-Type':'text/plain'})
        with pytest.raises(HTTPError) as e:urlopen(req)
        assert e.value.code == 415
        req = Request(base + '/api/connect', data=b'{}', headers={'Content-Type':'application/json'})
        assert urlopen(req).status == 202
        wait_for(lambda: rig.session.snapshot()['connected'])
        assert not rig.robot.sent
        with pytest.raises(HTTPError):urlopen(base + '/api/report/../../anything')
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_gui_cli_defaults_use_local_hardware_settings_and_keep_overrides(monkeypatch):
    from core.rollout import testing_gui
    monkeypatch.setattr(testing_gui, 'hardware_defaults', lambda: {'ROBOT_PORT':'/dev/configured','ROBOT_ID':'configured'})
    args = testing_gui.parse_args(['--follower_port', '/dev/override'])
    assert args.follower_port == '/dev/override' and args.follower_id == 'configured'
    assert args.release_on_exit is False and args.enable_motion is True


def test_pinned_follower_hash_is_checked_before_opening_hardware(rig):
    rig.args.expected_follower_sha256 = '0' * 64
    rig.session.request_connect()
    wait_for(lambda: bool(rig.session.snapshot()['error']))
    assert 'SHA-256 differs' in rig.session.snapshot()['error']
    assert not rig.robot.threads and not rig.robot.sent
    rig.args.expected_follower_sha256 = 'f' * 64
    connect(rig)
    assert rig.session.snapshot()['connected'] and not rig.robot.sent


def test_busy_gui_port_reports_how_to_open_existing_gui(capsys, monkeypatch):
    import socket
    from core.rollout import testing_gui

    sessions = []
    factory = testing_gui.RealArmSession
    def make_session(args):
        session = factory(args)
        sessions.append(session)
        return session
    monkeypatch.setattr(testing_gui, 'RealArmSession', make_session)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = testing_gui.main(['--arm', 'so101', '--gui_port', str(port)])
    assert result == 1
    error = capsys.readouterr().err
    assert 'already in use' in error
    assert f'http://127.0.0.1:{port}' in error
    assert '--gui_port 8767' in error
    assert len(sessions) == 1 and not sessions[0]._thread.is_alive()
    assert not sessions[0].snapshot()['connected']
