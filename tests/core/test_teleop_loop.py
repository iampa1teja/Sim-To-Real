"""Recording requests preserve save/discard/reset precedence without starting a simulator."""
from types import SimpleNamespace

import pytest

from sim.teleop.loop import LoopResources, RecordingController, RecordingPhase


@pytest.fixture
def session():
    events = []

    class Recorder:
        target = None
        frame_count = 1
        episode_index = 3

        def save_episode(self, sidecar):
            assert sidecar is None
            events.append('save')
            return self.episode_index

        def cancel_episode(self):
            events.append('discard')

    cfg = SimpleNamespace(fps=30, sim_reset_object_name=None, sim_reset_object_offset=(0., 0., 0.))
    resources = LoopResources(
        env=SimpleNamespace(reset=lambda: events.append('reset_env')), env_id='fixture', leader=None,
        mapping=None, arm=None, task=None,
        controls=SimpleNamespace(set_recording=lambda active: events.append(('recording', active))),
        spawner=SimpleNamespace(reset=lambda: events.append('reset_spawner'),
                                spawn=lambda name, offset: events.append(('spawn', name, offset))),
        sim_cameras={}, real_cameras={}, recorders=Recorder(),
    )
    return RecordingController(cfg, resources), events


@pytest.mark.parametrize('ending,expected', [
    ({'end': True}, ['save', 'reset_env', 'reset_spawner', ('recording', False)]),
    ({'reset': True}, ['save', 'reset_env', 'reset_spawner', ('recording', False)]),
    ({'rerecord': True, 'end': True}, ['discard', 'reset_env', 'reset_spawner', ('recording', False)]),
    ({'stop': True}, ['save', ('recording', False)]),
    ({'stop': True, 'rerecord': True, 'reset': True}, ['discard', ('recording', False)]),
])
def test_recording_endings_save_or_discard_once_with_correct_reset_order(session, ending, expected):
    controller, events = session
    transition = controller.handle({'start': True})
    assert controller.recording and transition.invalidate_observation
    assert events == [('recording', True)]
    controller.trajectory.append('frame')
    events.clear()
    # A simultaneous start must not begin a second episode after saving/discarding.
    transition = controller.handle(dict(ending, start=True))
    assert events == expected
    assert controller.trajectory == []
    assert not controller.recording
    assert transition.stop == bool(ending.get('stop'))
    assert transition.reset_roll == (not transition.stop)
    assert controller.phase is (RecordingPhase.STOPPED if transition.stop else RecordingPhase.BUFFER)


def test_buffer_reset_cancels_staged_replacement_without_saving(session):
    controller, events = session
    controller.resources.recorders.target = 1
    transition = controller.handle({'reset': True})
    assert events == ['discard', 'reset_env', 'reset_spawner', ('recording', False)]
    assert transition.invalidate_observation and transition.reset_roll
    assert controller.note == 'Re-record cancelled; original episode kept.'


def test_spawn_then_start_uses_declared_object_and_offset(session):
    controller, events = session
    controller.cfg.sim_reset_object_name = 'cube_a'
    controller.cfg.sim_reset_object_offset = (0.01, 0., 0.)
    transition = controller.handle({'spawn': True})
    assert transition.invalidate_observation and not controller.recording
    assert events == [('spawn', 'cube_a', (0.01, 0., 0.))]
    events.clear()
    controller.handle({'start': True})
    assert events == [('spawn', 'cube_a', (0.01, 0., 0.)), ('recording', True)]
