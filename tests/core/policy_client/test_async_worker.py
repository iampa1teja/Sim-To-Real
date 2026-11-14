"""Deterministic clocks and controlled completions exercise async scheduling."""
from dataclasses import replace
from queue import Queue
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from core.policy_client.async_worker import AlignedPlayback, InferenceFault, Observation
from core.policy_client.remote_policy import RemotePolicy
from core.policy_client.smoothing import SmoothingConfig, TimingLog


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeClient:
    def __init__(self):
        self.events = []
        self.requests, self.replies = Queue(), Queue()
        self.policy = SimpleNamespace(cameras=[])
        self.manifest = SimpleNamespace(name='example', cameras=[])
        self.client = SimpleNamespace(close=lambda: self.record('close'))
        self.seed_ack = True
        self.active = self.max_active = 0

    def record(self, operation):
        self.events.append((operation, get_ident()))

    def ping(self):
        self.record('ping')
        return True

    def describe(self):
        self.record('describe')
        return {'name': 'example'}

    def reset(self):
        self.record('reset')

    def call_endpoint(self, endpoint, data):
        assert endpoint == 'seed_policy'
        self.record(('seed', data['seed']))
        return data if self.seed_ack else {'seed': -1}

    def get_action(self, observation):
        self.record('get_action')
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        self.requests.put(observation)
        try:
            reply = self.replies.get(timeout=5)
            if isinstance(reply, Exception):
                raise reply
            return reply
        finally:
            self.active -= 1


@pytest.fixture
def connected(monkeypatch):
    policies = []
    def make(**kwargs):
        clock, client = FakeClock(), FakeClient()
        def factory(*args, **kw):
            client.record('connect')
            return client
        monkeypatch.setattr('core.policy_client.remote_policy.build_plugin_client', factory)
        policy = RemotePolicy(['a.pos'], policy_name='example', action_horizon=10,
                              async_inference=True, control_dt=.1, clock=clock, **kwargs)
        policy.connect()
        policy.reset()
        policies.append((policy, client))
        return policy, client, clock
    yield make
    for policy, client in policies:
        client.replies.put(np.zeros((10, 1)))
        policy.close()


def tick(policy, state=0., instruction=None):
    return policy.get_raw_action([state], {})['a.pos']


def complete(policy, client, clock, delay, values=None):
    client.requests.get(timeout=1)
    clock.advance(delay)
    client.replies.put(np.arange(10, dtype=float)[:, None] if values is None else values)
    policy._async.pending[0].result(timeout=1)


def test_no_steady_state_holds_when_latency_fits_playback(connected):
    policy, client, clock = connected(refill_seconds=.4)
    pending_id = None
    due = None
    held = []
    for step in range(100):
        clock.now = step * .1
        if due is not None and clock.now + 1e-9 >= due:
            client.replies.put(np.zeros((10, 1)))
            policy._async.pending[0].result(timeout=1)
            due = None
        tick(policy)
        held.append(policy.last_tick.held)
        pending = policy._async.pending
        if pending is not None and pending[1].chunk_id != pending_id:
            pending_id = pending[1].chunk_id
            client.requests.get(timeout=1)
            due = clock.now + .2
    assert held[:2] == [True, True]
    assert not any(held[2:])
    assert client.max_active == 1


def test_elapsed_wall_time_and_control_steps_trim_the_prefix(connected):
    policy, client, clock = connected(refill_seconds=0)
    assert tick(policy) == 0
    complete(policy, client, clock, .3)
    assert tick(policy) == 3
    assert policy.last_tick.action_age_ms == pytest.approx(300)
    assert policy.last_tick.turnaround == pytest.approx(.3)


def test_step_index_also_trims_when_fake_time_does_not_advance(connected):
    policy, client, clock = connected(refill_seconds=0)
    tick(policy); tick(policy); tick(policy)
    complete(policy, client, clock, 0)
    assert tick(policy) == 3


def test_control_pause_expires_buffered_targets(connected):
    policy, client, clock = connected(refill_seconds=0)
    tick(policy)
    complete(policy, client, clock, .1)
    assert tick(policy) == 1
    clock.advance(.4)
    assert tick(policy) == 5


def test_capture_jitter_does_not_delay_the_refill_boundary(connected):
    policy, client, clock = connected(refill_seconds=.4)
    clock.now = .0002
    tick(policy)
    complete(policy, client, clock, .2)
    tick(policy)
    for step in [3,4,5]:
        clock.now = step * .1 + .0001
        tick(policy)
    assert len(policy._async.playback.rows) == 4
    assert policy._async.pending is not None
    client.requests.get(timeout=1)


def test_pause_does_not_restart_starvation_grace(connected):
    policy, client, clock = connected(refill_seconds=0, starvation_grace=.25)
    tick(policy)
    complete(policy, client, clock, .1)
    assert tick(policy) == 1
    clock.advance(2)
    with pytest.raises(InferenceFault, match='starvation'):
        tick(policy)


def test_blend_window_and_gripper_exclusion():
    options = SmoothingConfig(blend_steps=3)
    playback = AlignedPlayback(['arm', 'grip'], 10, .1, options, gripper_indices=(1,))
    obs = Observation(0, 0, 0., 0, {})
    playback.add_chunk([dict(arm=0., grip=0.)] * 10, obs, 0, 0., 0.)
    playback.action(0., 0)
    playback.add_chunk([dict(arm=9., grip=9.)] * 10, replace(obs, captured_at=.1, step=1, chunk_id=1), 1, .1, 0.)
    sent = [playback.action(.1 + i * .1, 0)[1] for i in range(4)]
    np.testing.assert_allclose(sent, [[3, 9], [6, 9], [9, 9], [9, 9]])


def test_ema_and_held_applied_target_are_preserved():
    playback = AlignedPlayback(['a'], 2, .1, SmoothingConfig(ema_alpha=.5))
    playback.add_chunk([dict(a=0), dict(a=10)], Observation(0, 0, 0., 0, {}), 0, 0., 0.)
    assert playback.action(0., 0)[1] == [0]
    assert playback.action(.1, 0)[1] == [5]
    playback.last_target = np.array([4.])  # dispatcher clamped the applied target
    assert playback.action(.2, 0)[1] == [4]
    assert playback.action(.3, 0)[1] == [4]


def test_starvation_holds_then_latches_fault(connected):
    policy, client, clock = connected(starvation_grace=.25, timeout_ms=1000)
    assert tick(policy, 7.) == 7
    clock.advance(.2)
    assert tick(policy, 9.) == 7
    generation = policy._async.generation
    clock.advance(.1)
    with pytest.raises(InferenceFault, match='starvation'):
        tick(policy)
    assert policy._async.generation == generation + 1
    with pytest.raises(InferenceFault, match='starvation'):
        tick(policy)
    assert policy._async.generation == generation + 1


def test_request_deadline_fault_is_nonblocking(connected):
    policy, client, clock = connected(timeout_ms=200, starvation_grace=10)
    tick(policy)
    client.requests.get(timeout=1)
    clock.advance(.2)
    with pytest.raises(InferenceFault, match='deadline'):
        tick(policy)
    assert client.active == 1  # control faulted without waiting for the socket


def test_reply_completed_after_deadline_is_rejected(connected):
    policy, client, clock = connected(timeout_ms=200, starvation_grace=10)
    tick(policy)
    complete(policy, client, clock, .3)
    with pytest.raises(InferenceFault, match='deadline'):
        tick(policy)


@pytest.mark.parametrize('change', ['cancel', 'instruction'])
def test_late_reply_discarded_on_stop_or_instruction(connected, change):
    policy, client, clock = connected()
    tick(policy, 7)
    generation = policy._async.generation
    if change == 'cancel':
        policy.cancel()
    else:
        policy.set_instruction('new task')
    assert policy._async.generation == generation + 1
    complete(policy, client, clock, .2, np.full((10, 1), 999.))
    if change == 'cancel':
        with pytest.raises(RuntimeError, match='stopped'):
            tick(policy)
    else:
        assert tick(policy, 8) == 7  # retain the applied target while fresh work starts
        assert policy.last_tick.held
        assert client.requests.get(timeout=1)['instruction'] == 'new task'


def test_reset_invalidates_before_waiting_for_the_old_request(connected, monkeypatch):
    policy, client, clock = connected()
    tick(policy, 7)
    client.requests.get(timeout=1)
    invalidated = Event()
    original = policy._async._invalidate
    def invalidate(*args, **kwargs):
        original(*args, **kwargs)
        invalidated.set()
    monkeypatch.setattr(policy._async, '_invalidate', invalidate)
    reset = Thread(target=policy.reset)
    reset.start()
    assert invalidated.wait(1)
    client.replies.put(np.full((10, 1), 999.))
    reset.join(timeout=1)
    assert not reset.is_alive()
    assert tick(policy, 8) == 8
    assert policy.last_tick.held


def test_action_repeat_is_expanded_before_stale_trimming(connected):
    policy, client, clock = connected(action_repeat=2, refill_seconds=0)
    tick(policy)
    complete(policy, client, clock, .3)
    assert tick(policy) == 1
    clock.advance(.1)
    assert tick(policy) == 2
    clock.advance(.1)
    assert tick(policy) == 2


def test_grasp_observes_chunk_before_process_and_receives_measured_values(connected):
    policy, client, clock = connected()
    events = []
    grasp = SimpleNamespace(measured=None, gripper_index=0,
        observe_chunk=lambda actions, identity: events.append(('observe', identity)),
        process=lambda raw, sent, identity, held: (events.append(('process', identity, held, grasp.measured)) or sent))
    policy.grasp = policy._async.grasp = policy._async.playback.grasp = grasp
    tick(policy, 12)
    assert events == [('process', -1, True, 12.)]
    complete(policy, client, clock, .1)
    tick(policy, 13)
    assert events[-2:] == [('observe', 0), ('process', 0, False, 13.)]
    policy.grasp = None  # fake hook deliberately has no lifecycle methods


def test_episode_seed_ack_and_every_query_use_the_worker(connected):
    policy, client, clock = connected(seed_per_episode=True)
    policy.reset(seed=123)
    tick(policy)
    client.requests.get(timeout=1)
    operations = [event[0] for event in client.events]
    assert operations[-3:] == [('seed', 123), ('seed', 123), 'get_action']
    client.replies.put(np.zeros((10, 1)))
    policy._async.pending[0].result(timeout=1)
    policy.close()
    owners = {owner for _, owner in client.events}
    assert len(owners) == 1
    assert get_ident() not in owners
    assert operations[0] == 'connect'
    assert client.events[-1][0] == 'close'


def test_seed_without_ack_faults(connected):
    policy, client, clock = connected()
    client.seed_ack = False
    with pytest.raises(InferenceFault, match='ACK'):
        policy.reset(seed=123)


def test_exactly_one_request_even_across_instruction_change(connected):
    policy, client, clock = connected()
    tick(policy)
    client.requests.get(timeout=1)
    for _ in range(5):
        tick(policy)
    policy.set_instruction('fresh')
    for _ in range(5):
        tick(policy)
    assert client.requests.empty() and client.active == 1
    client.replies.put(np.zeros((10, 1)))
    policy._async.pending[0].result(timeout=1)
    tick(policy)
    assert client.requests.get(timeout=1)['instruction'] == 'fresh'
    assert client.max_active == 1


def test_instruction_change_does_not_restart_a_stopped_policy(connected):
    policy, client, clock = connected()
    policy.cancel()
    policy.set_instruction('fresh')
    with pytest.raises(RuntimeError, match='stopped'):
        tick(policy)
    assert client.requests.empty()


def test_temporal_ensemble_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='temporal_ensemble.*unsupported'):
        RemotePolicy(['a.pos'], policy_name='example', async_inference=True, temporal_ensemble=1)


def test_timing_records_async_provenance(tmp_path):
    import csv
    log = TimingLog(tmp_path / 'ticks.csv', ['a'])
    log.record(1, 3, .05, False, np.array([1]), np.array([1]),
               held=True, action_age_ms=125., turnaround=.05)
    log.close()
    with (tmp_path / 'ticks.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['held'] == '1' and row['chunk_id'] == '3'
    assert row['action_age_ms'] == '125.0' and row['turnaround_s'] == '0.05'


@pytest.mark.parametrize('motion', [False, True])
def test_rollout_checks_first_fresh_target_after_startup_holds(motion):
    from core.rollout.real_rollout import RolloutSettings, run_episode
    info = SimpleNamespace(held=True, action_age_ms=None, chunk_id=-1, turnaround=0.)
    sent, applied, cancelled = [], [], []
    calls = []
    def action(*_):
        calls.append(True)
        info.held = len(calls) < 3
        return {'a.pos': 0. if info.held else 80.}
    policy = SimpleNamespace(async_inference=True, last_tick=info, get_raw_action=action,
                             record_applied_target=applied.append, cancel=lambda: cancelled.append(True))
    robot = SimpleNamespace(get_observation=lambda: {'a.pos': 0}, send_action=sent.append)
    result = run_episode(robot, policy, ['a.pos'], {},
        RolloutSettings(fps=10, episode_seconds=1, enable_motion=motion,
                        max_start_error=20, raw_ranges=((-100, 100),)),
        clock=lambda: 0., sleep=lambda _: None)
    assert result.status == 'start_too_far'
    assert sent == ([{'a.pos': 0.}, {'a.pos': 0.}] if motion else [])
    assert cancelled == [True]


def gui_session(policy):
    import io
    import threading
    import time
    from core.rollout.testing_gui import RealArmSession
    session = RealArmSession.__new__(RealArmSession)
    session.args = SimpleNamespace(max_start_error=20., rename_map={})
    session.arm = SimpleNamespace(feature_names=['a.pos'])
    session._policy, session._future, session._kind = policy, None, None
    session._motion_lock, session._lock = threading.Lock(), threading.RLock()
    session._shutdown = Event()
    session._heartbeat_timeout = 10.
    session._ranges = ((-100., 100.),)
    session._run = dict(cancel=Event(), heartbeat=time.monotonic(), started=time.monotonic(),
                        seconds=10., steps=0, trace=io.StringIO(), stop_status='stopped', detail='')
    session.catalog = lambda: {'cameras': []}
    session._set = lambda *args: None
    session.finishes = []
    session._finish = lambda status, detail='': session.finishes.append((status, detail))
    session.sent = []
    session._robot = SimpleNamespace(send_action=session.sent.append)
    return session


def test_gui_ticks_dispatch_holds_and_keep_first_fresh_guard(connected):
    policy, client, clock = connected()
    session = gui_session(policy)
    session._drive({'a.pos': 7.})
    session._drive({'a.pos': 7.})
    assert session.sent == [{'a.pos': 7.}, {'a.pos': 7.}]
    complete(policy, client, clock, .1, np.full((10, 1), 80.))
    session._drive({'a.pos': 7.})
    assert session.finishes[0][0] == 'start_too_far'
    assert len(session.sent) == 2


def test_gui_stop_acknowledgement_prevents_any_late_dispatch(connected):
    policy, client, clock = connected()
    session = gui_session(policy)
    session._drive({'a.pos': 7.})
    session.request_stop()
    before = list(session.sent)
    complete(policy, client, clock, .1, np.full((10, 1), 8.))
    session._drive({'a.pos': 7.})
    assert session.sent == before
    assert session.finishes[0][0] == 'stopped'
