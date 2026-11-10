"""Paired comparison guards and the trusted local server codec."""
import io
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sim.eval.sim_real import limited_target


def test_first_action_guard_precedes_step_limit():
    with pytest.raises(ValueError, match='First policy target'):
        limited_target({'a.pos': 50}, [0], ['a.pos'], [(-100, 100)],
                       first=True, start_limit=20, step_limit=10)


def test_each_side_limits_relative_to_its_own_observed_state():
    action = {'a.pos': 90, 'b.pos': -200}
    kwargs = dict(keys=list(action), ranges=[(-100, 100), (-100, 100)],
                  first=False, start_limit=20, step_limit=10)
    np.testing.assert_equal(limited_target(action, [0, 0], **kwargs), [10, -10])
    np.testing.assert_equal(limited_target(action, [60, 40], **kwargs), [70, 30])
    with pytest.raises(ValueError, match='finite'):
        limited_target(action, [float('nan'), 0], **kwargs)


def test_local_server_numpy_response_and_disallowed_classes():
    from policies.molmoact2.local_tcp import _ArrayUnpickler
    payload = {'chunk': np.ones((30, 6), dtype=np.float32)}
    result = _ArrayUnpickler(io.BytesIO(pickle.dumps(payload))).load()
    np.testing.assert_equal(result['chunk'], payload['chunk'])
    with pytest.raises(pickle.UnpicklingError):
        _ArrayUnpickler(io.BytesIO(pickle.dumps(SimpleNamespace()))).load()


def test_local_server_codec_does_not_double_transform(monkeypatch):
    from core.arms.spec import load_arm_spec
    from core.arms.calibration import resolve_calibration
    from core.policy_client.contract import ArmInfo
    from policies.molmoact2.local_tcp import LocalLeRobotPolicy
    import struct

    arm = load_arm_spec('so101')[0]
    selection = resolve_calibration(arm, unit='rpd_follower')
    policy = LocalLeRobotPolicy(ArmInfo.from_arm(arm, selection.mapping))
    state = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.float32)
    class Socket:
        def sendall(self, payload):
            request = pickle.loads(payload[4:])
            raw = request['raw_obs']
            degrees = np.asarray([raw[key] for key in policy.keys], dtype=np.float32)
            # A server returning the input pose after its own frame round-trip.
            response = pickle.dumps({'chunk': degrees[None, :]})
            self.response = io.BytesIO(struct.pack('>I', len(response)) + response)
        def recv(self, count):
            return self.response.read(count)
    policy.socket = Socket()
    images = {key: np.zeros((16, 16, 3), np.uint8) for key in ('room', 'wrist')}
    action = policy.get_raw_action(state, images)
    np.testing.assert_allclose(list(action.values()), state, atol=1e-5)


def test_video_preserves_wall_time_when_capture_drops_frames(tmp_path):
    cv2 = pytest.importorskip('cv2')
    from sim.eval.sim_real import TrialVideo
    path = tmp_path / 'trial.mp4'
    images = {key: np.zeros((32, 32, 3), dtype=np.uint8) for key in ('room', 'wrist')}
    writer = TrialVideo(path, 10)
    writer.write(images, 'REAL', 0)
    writer.write(images, 'REAL', .5)
    writer.close(1.0)
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened()
    assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    assert reader.get(cv2.CAP_PROP_FPS) == 10
    reader.release()


def test_leader_positions_before_start_and_is_not_read_when_control_hands_off():
    from sim.eval.sim_real import position_with_leader
    commands = []
    reads = []
    def read():
        reads.append(True)
        return {'a.pos': 50, 'b.pos': -50}
    leader = SimpleNamespace(get_action=read)
    follower = SimpleNamespace(send_action=commands.append)
    kwargs = dict(leader=leader, follower=follower, measured=[0, 0], keys=['a.pos', 'b.pos'],
                  ranges=[(-100, 100)] * 2, step_limit=10, motion=True)
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=True), [10, -10])
    assert commands == [{'a.pos': 10, 'b.pos': -10}]
    # Policy takes over at Start; Stop and completion also disable operator control.
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=False), [0, 0])
    assert len(commands) == len(reads) == 1
    kwargs['motion'] = False
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=True), [10, -10])
    assert len(commands) == 1  # Shadow still lets the sim follow, without physical writes.


def test_gui_can_stop_leader_positioning_before_a_trial():
    from sim.recording.web_gui import RecordingWebGui
    gui = RecordingWebGui([], port=0)
    try:
        gui.set_state(False, False, 'Positioning', positioning=True)
        assert gui._command('stop')
        assert gui.consume_requests()['stop']
        gui.set_state(False, False, 'Holding', positioning=False)
        assert not gui._command('stop')
    finally:
        gui.destroy()


def test_failed_policy_pair_closes_connections_and_can_retry():
    from concurrent.futures import ThreadPoolExecutor
    from sim.eval.sim_real import connect_trial_policies
    events = []
    attempts = []
    class Policy:
        def __init__(self):
            self.index = len(attempts)
            attempts.append(self)
        def connect(self):
            if self.index == 1:
                raise ConnectionRefusedError('Server is warming up')
        def close(self):
            events.append(('close', self.index))
        def reset(self):
            events.append(('reset', self.index))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ConnectionRefusedError):
            connect_trial_policies(Policy, executor)
        assert events == [('close', 0), ('close', 1)]
        result = connect_trial_policies(Policy, executor)
        assert len(result) == 2
        assert events[-2:] == [('reset', 2), ('reset', 3)]
