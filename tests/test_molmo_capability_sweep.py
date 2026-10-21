"""Research result guards prevent unfair or assisted comparisons."""
import importlib.util
from pathlib import Path
import os
import socket
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capability_sweep', ROOT/'scripts/molmo_capability_sweep.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    expected = [dict(id='id-000', zone='MC', policy_seed=42)]
    options = dict(norm_tag='so100_so101_molmoact2', include_original_room=False)
    report = dict(complete=True, requested_episodes=1, checkpoint='molmoact2_so101',
                  episode_length_s=210, action_horizon=30, action_repeat=1, policy_cameras=['room','wrist'],
                  resume_settings=dict(policy_options={k:str(v) for k,v in options.items()}, grasp={}, smoothing={}),
                  smoke_test=dict(selection='one_per_zone', reach_timeout_s=60., progress_timeout_s=30.),
                  episodes=[dict(start_id='id-000', zone='MC', policy_seed=42,
                    smoke_deadline=dict(reach_timeout_s=60., progress_timeout_s=30.))])
    return report, expected, options


def test_complete_matching_released_trial_is_accepted():
    report, expected, options = fixture()
    assert m.validate_result(report, expected, 30, options) is report


def test_three_camera_variant_must_be_declared_explicitly():
    report, expected, options = fixture()
    report['policy_cameras'] = ['room','wrist','original_room']
    with pytest.raises(ValueError, match='camera'):
        m.validate_result(report, expected, 30, options)
    report['resume_settings']['molmo_overhead_camera'] = True
    assert m.validate_result(report, expected, 30, options, overhead=True) is report


@pytest.mark.parametrize('change', ['seed','start','zone','duration','horizon','camera','checkpoint',
                                   'incomplete','latch','smoothing','options','deadline'])
def test_changed_comparison_or_assistance_cannot_be_scored(change):
    report, expected, options = fixture()
    if change == 'seed': report['episodes'][0]['policy_seed'] += 1
    elif change == 'start': report['episodes'][0]['start_id'] = 'id-001'
    elif change == 'zone': report['episodes'][0]['zone'] = 'OUT'
    elif change == 'duration': report['episode_length_s'] = 300
    elif change == 'horizon': report['action_horizon'] = 16
    elif change == 'camera': report['policy_cameras'] = ['room','wrist','original_room']
    elif change == 'checkpoint': report['checkpoint'] = 'experimental_finetuned'
    elif change == 'incomplete': report['complete'] = False
    elif change == 'latch': report['resume_settings']['grasp']['gripper_latch'] = True
    elif change == 'smoothing': report['resume_settings']['smoothing']['ema_alpha'] = .5
    elif change == 'options': report['resume_settings']['policy_options']['norm_tag'] = 'other'
    elif change == 'deadline': report['episodes'][0]['smoke_deadline']['reach_timeout_s'] = 90
    with pytest.raises(ValueError):
        m.validate_result(report, expected, 30, options)


def test_port_probe_does_not_connect_to_or_replace_existing_listener():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not m.port_free(port)
        assert listener.getsockname()[1] == port
    assert m.port_free(port)


def test_process_identity_distinguishes_current_and_missing_process():
    identity = m.process_identity(os.getpid())
    assert identity is not None and len(identity) == 3
    assert m.process_identity(999999999) is None


def test_wait_preserves_existing_job_and_advances_only_after_its_identity_disappears(monkeypatch):
    worker = m.Investigation.__new__(m.Investigation)
    worker.args = SimpleNamespace(wait_pid=17)
    worker.wait_identity = ('start','command','cwd')
    identities = iter([worker.wait_identity, worker.wait_identity, None])
    phases, sleeps = [], []
    worker.save = phases.append
    monkeypatch.setattr(m, 'process_identity', lambda pid: next(identities))
    monkeypatch.setattr(m, 'port_free', lambda port: True)
    monkeypatch.setattr(m.time, 'sleep', sleeps.append)
    monkeypatch.setattr(m, 'run', lambda *a, **k: pytest.fail('Waiting must not run Docker or stop processes'))
    worker.wait_for_existing()
    assert phases == ['waiting_for_existing_evaluation','waiting_for_existing_evaluation','ready']
    assert sleeps == [30,30]


def test_gpu_memory_wait_does_not_replace_other_work(monkeypatch):
    worker = m.Investigation.__new__(m.Investigation)
    values = iter(['18778\n', '32\n'])
    phases, sleeps = [], []
    worker.save = phases.append
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *args, **kwargs: next(values))
    monkeypatch.setattr(m.time, 'sleep', sleeps.append)
    worker.wait_for_gpu_memory()
    assert phases == ['waiting_for_gpu_memory'] and sleeps == [30]
