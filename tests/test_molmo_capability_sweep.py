"""Research result guards prevent unfair or assisted comparisons."""
import importlib.util
from pathlib import Path
import os
import socket
from types import SimpleNamespace
import json
import tempfile
import tomllib

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


def test_two_external_views_are_validated_against_their_actual_declared_inputs():
    report, expected, options = fixture()
    report['policy_cameras'] = ['room', 'original_room']
    report['resume_settings']['molmo_overhead_camera'] = True
    with pytest.raises(ValueError, match='camera'):
        m.validate_result(report, expected, 30, options, overhead=True)
    assert m.validate_result(report, expected, 30, options, overhead=True,
                             camera_inputs=('room', 'original_room')) is report
    report['policy_cameras'] = ['room', 'wrist', 'original_room']
    with pytest.raises(ValueError, match='camera'):
        m.validate_result(report, expected, 30, options, overhead=True,
                          camera_inputs=('room', 'original_room'))


def test_camera_selection_changes_only_the_private_snapshot_and_its_digest():
    manifest = ROOT/'policies/molmoact2/policy.toml'
    original = manifest.read_bytes()
    with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
        args = SimpleNamespace(state_dir=Path(directory)/'control', wait_pid=None, control_repo=ROOT)
        control = m.Investigation(args)
        args = SimpleNamespace(state_dir=Path(directory)/'external', wait_pid=None, control_repo=ROOT)
        external = m.Investigation(args, camera_inputs=('room', 'original_room'))
        assert (control.snapshot/'policies/molmoact2/policy.toml').read_bytes() == original
        selected = tomllib.loads((external.snapshot/'policies/molmoact2/policy.toml').read_text())
        assert selected['observation']['cameras'] == ['room', 'original_room']
        assert external.source_digest != control.source_digest
        state = json.loads((external.root/'state.json').read_text())
        assert state['manifest_camera_inputs'] == ['room', 'original_room']
        assert state['source_digest'] == external.source_digest
    assert manifest.read_bytes() == original


@pytest.mark.parametrize('cameras', [[], ['room', 'room'], ['unknown'], 'room', ['room', None]])
def test_invalid_research_camera_selection_fails_before_creating_resources(cameras):
    with pytest.raises(ValueError, match='camera'):
        m.Investigation(SimpleNamespace(), camera_inputs=cameras)


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


def test_fast_failure_does_not_improve_the_speed_ranking():
    def candidate(success_time, failure_time):
        return dict(report=dict(episodes=[
            dict(success=True, elapsed_sim_s=success_time, stages=dict(grasped=True)),
            dict(success=False, elapsed_sim_s=failure_time, stages=dict(grasped=False))]))
    slower_placement = candidate(20, 60)
    faster_placement = candidate(10, 90)
    assert m.rank_result(faster_placement) > m.rank_result(slower_placement)


def test_runtime_exception_is_visible_without_printing_numeric_error_metrics():
    assert m.FAILURE_LINE.search('RuntimeError: Expected all tensors on the same device')
    assert m.FAILURE_LINE.search('torch.OutOfMemoryError: CUDA out of memory')
    assert not m.FAILURE_LINE.search('[GRASP 1] {"tool_error_mm": 8.4, "max_joint_tracking_error_deg": 1.2}')


def test_zero_exit_incomplete_trial_retains_cause_before_shutdown_noise(monkeypatch, capsys):
    report, expected, options = fixture()
    report['complete'] = False
    output = ('RuntimeError: CUDA out of memory\n'
              + '\n'.join('[Warning] shutdown diagnostic' for _ in range(100))
              + '\n' + m.MARKER + json.dumps(report) + '\n').encode()
    read_fd, write_fd = os.pipe()
    os.write(write_fd, output)
    os.close(write_fd)
    reader = os.fdopen(read_fd, 'rb')
    worker = m.Investigation.__new__(m.Investigation)
    worker.camera_inputs = ('room', 'wrist')
    worker.cfg = {'eval': dict(lang='task', dataset='/dataset', robot_usd='/robot.usd',
                               rename_map={'realsense_rgb': 'room', 'wrist_cam': 'wrist'}, eval_set='/set.json')}
    worker.snapshot = Path('/dev/shm/private-test')
    worker.viewer = SimpleNamespace(poll=lambda: None)
    worker.server = 'owned-test-policy'
    worker.results = []
    worker.expected_by_seed = {1984: expected}
    worker.save = lambda *args: None
    worker.start_server = lambda *args: None
    cleaned = []
    worker.cleanup_client = lambda: cleaned.append(True)
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(stdout=reader, wait=lambda: 0))
    def logs(args, **kwargs):
        assert args == ['docker', 'logs', '--tail', '80', 'owned-test-policy']
        return SimpleNamespace(stdout='policy request failed: allocation exceeded shared GPU memory', stderr='')
    monkeypatch.setattr(m.subprocess, 'run', logs)
    try:
        with pytest.raises(RuntimeError) as caught:
            worker.trial('failure', options)
        assert 'CUDA out of memory' in str(caught.value)
        assert 'allocation exceeded shared GPU memory' in str(caught.value)
        assert 'shutdown diagnostic' not in str(caught.value)
        assert m.MARKER not in str(caught.value)
        assert not worker.results and cleaned == [True]
        assert 'RuntimeError: CUDA out of memory' in capsys.readouterr().out
    finally:
        reader.close()


def test_episode_summary_joined_to_progress_is_printed_and_saved(monkeypatch, capsys):
    report, expected, options = fixture()
    report['overall'] = dict(successes=0, episodes=1)
    summary = '[EPISODE 1] start=0 zone=MC fail stage=reached stop=progress_timeout'
    output = ('Rollout (ep 1/1): 14% 901/6300 [00:30] ' + summary
              + '\r\n' + m.MARKER + json.dumps(report) + '\n').encode()
    read_fd, write_fd = os.pipe()
    os.write(write_fd, output)
    os.close(write_fd)
    reader = os.fdopen(read_fd, 'rb')
    worker = m.Investigation.__new__(m.Investigation)
    worker.camera_inputs = ('room', 'wrist')
    worker.cfg = {'eval': dict(lang='task', dataset='/dataset', robot_usd='/robot.usd',
                               rename_map={'realsense_rgb': 'room', 'wrist_cam': 'wrist'}, eval_set='/set.json')}
    worker.snapshot = Path('/dev/shm/private-test')
    worker.viewer = SimpleNamespace(poll=lambda: None)
    worker.results = []
    worker.expected_by_seed = {1984: expected}
    saved = []
    worker.save = lambda *args: saved.append(dict(worker.current or {}))
    worker.start_server = lambda *args: None
    worker.cleanup_client = lambda: None
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(stdout=reader, wait=lambda: 0))
    try:
        result = worker.trial('complete', options)
        assert result['last_episode'] == summary
        assert any(s.get('last_episode') == summary for s in saved)
        output = capsys.readouterr().out
        assert '\n' + summary + '\n' in output
        assert 'Rollout (ep 1/1)' not in output
        assert m.MARKER not in output
    finally:
        reader.close()
