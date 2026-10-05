"""CPU-only grasp-control tests, including the actual sim/real client methods."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim_to_real_so101.gr00t_client.grasp import (
    GraspConfig, GraspController, GripperLatch, dither_score, seed_query, training_bounds,
)
from sim_to_real_so101.gr00t_client.smoothing import ActionSmoother, SmoothingConfig
from test_smoothing import KEYS, FakeClient, FakeClock, FakeRobot, actions, make_sim_policy, real_code


@pytest.fixture
def dataset(tmp_path):
    meta = tmp_path / 'train/meta'
    meta.mkdir(parents=True)
    # Deliberately not 0..100; state and action extrema differ.
    (meta / 'info.json').write_text(json.dumps({'features': {
        name: {'names': list(KEYS)} for name in ('action', 'observation.state')}}))
    (meta / 'stats.json').write_text(json.dumps({
        'action': {'min': [0]*5 + [10], 'max': [20]*5 + [90]},
        'observation.state': {'min': [0]*5 + [12], 'max': [20]*5 + [89]}}))
    return meta.parent


def controller(dataset, **kwargs):
    return GraspController(GraspConfig(latch_dataset=str(dataset), **kwargs), KEYS)


def test_thresholds_from_tiny_training_dataset(dataset, capsys):
    assert training_bounds(dataset) == (10, 90)
    grasp = controller(dataset, gripper_latch=True)
    assert grasp.latch.close_below == 50
    assert grasp.latch.open_above == 74
    assert grasp.latch.empty_closed == 12
    printed = capsys.readouterr().out
    assert 'C=50' in printed and 'O=74' in printed and 'gripper=12' in printed
    override = controller(dataset, gripper_latch=True, latch_close_below=42,
                          latch_open_above=79, gripper_empty_closed=11)
    assert (override.latch.close_below, override.latch.open_above, override.latch.empty_closed) == (42, 79, 11)


def test_latch_noisy_confirm_hold_and_release():
    latch = GripperLatch(0, 100, GraspConfig())
    trace = [49, 50, 49, 51, 49, 49, 49]
    for step, raw in enumerate(trace):
        sent, event = latch.apply(raw, raw, step)
        assert latch.latched == (step == 6)
    assert sent == 0 and event == 'engage'
    for step in range(7, 36):
        assert latch.apply(90, 90, step)[0] == 0  # min hold beats release confirms
    assert latch.apply(90, 90, 36) == (90, 'release')
    for step in range(37, 40):
        latch.apply(49, 49, step)
    for step, raw in enumerate([81, 81, 80, 81, 81, 79, 81], 70):
        assert latch.apply(raw, raw, step)[0] == 0
    for step in range(77, 81):
        result = latch.apply(81, 81, step)
    assert result == (81, 'release')


def test_held_predictions_cannot_engage_or_release():
    latch = GripperLatch(0, 100, GraspConfig(latch_min_hold=0))
    for step in range(20):
        latch.apply(0, 0, step, held=True)
    assert not latch.latched
    for step in range(20, 23):
        latch.apply(0, 0, step)
    for step in range(23, 40):
        assert latch.apply(100, 100, step, held=True)[0] == 0


def test_ten_step_gate_and_arm_unaffected(dataset):
    grasp = controller(dataset, gripper_latch=True)
    for step in range(40):
        # Alternating >2-unit arm changes block engagement; then steady arm.
        arm = float(step % 2 * 4) if step < 20 else 0.
        raw = np.array([arm]*5 + [20.])
        sent = grasp.process(raw, raw, 0)
        assert sent[:5].tobytes() == raw[:5].tobytes()
        if step < 32:
            assert not grasp.latch.latched
    assert grasp.latch.latched
    assert grasp.rows[32]['latch_event'] == 'engage'


def test_measured_empty_escape_reopens_then_requires_normal_retrigger(dataset, caplog):
    grasp = controller(dataset, gripper_latch=True)
    raw = np.array([0.]*5 + [20.])
    grasp.measured = 12
    for _ in range(13):
        grasp.process(raw, raw, 0)
    assert grasp.latch.latched
    for _ in range(14):
        assert grasp.process(raw, raw, 0)[-1] == 10
    assert grasp.process(raw, raw, 0)[-1] == 90
    assert not grasp.latch.latched
    assert 'missed grasp' in caplog.text
    for _ in range(14):
        assert grasp.process(raw, raw, 0)[-1] == 90
    for _ in range(2):
        assert grasp.process(raw, raw, 0)[-1] == 20
    assert grasp.process(raw, raw, 0)[-1] == 10


def test_object_stall_never_triggers_timer_release(dataset):
    grasp = controller(dataset, gripper_latch=True)
    grasp.measured = 15.001  # just above empty + 3
    raw = np.array([0.]*5 + [20.])
    for _ in range(1000):
        sent = grasp.process(raw, raw, 0)
    assert grasp.latch.latched and sent[-1] == 10
    raw[-1] = 80
    for _ in range(5):
        sent = grasp.process(raw, raw, 0)
    assert not grasp.latch.latched and sent[-1] == 80


def test_empty_requires_consecutive_measured_steps(dataset):
    grasp = controller(dataset, gripper_latch=True)
    raw = np.array([0.]*5 + [20.])
    for step in range(150):
        grasp.measured = 12 if step % 15 < 14 else 16
        grasp.process(raw, raw, 0)
    assert grasp.latch.latched
    assert all(row['latch_event'] != 'missed grasp' for row in grasp.rows)


def test_logging_only_is_byte_identical_after_smoothing(dataset, tmp_path):
    grasp = controller(dataset, log_gripper=True, gripper_csv=str(tmp_path/'log.csv'))
    plain = ActionSmoother(SmoothingConfig(blend_steps=3, ema_alpha=.5), 8, KEYS)
    logged = ActionSmoother(SmoothingConfig(blend_steps=3, ema_alpha=.5), 8, KEYS, grasp=grasp)
    for step in range(24):
        if step % 8 == 0:
            chunk = actions(20-step)
            for smoother in (plain, logged):
                smoother.add_chunk(chunk, step, step, step//8)
        expected = plain.action(step)[1]
        actual = logged.action(step)[1]
        assert actual.tobytes() == expected.tobytes()
    grasp.close()
    rows = list(csv.DictReader((tmp_path/'log.csv').open()))
    assert len(rows) == 24
    assert float(rows[8]['endpoint_delta_norm']) == pytest.approx(np.sqrt(5*8**2))
    assert float(rows[8]['predicted_gripper']) == pytest.approx(12.5)


def test_dither_ignores_gripper_zero_deltas_and_outside_window():
    assert dither_score([], []) == 0
    arm = np.array([[0,0], [1,1], [1,2], [0,3], [1,4], [0,5]])
    assert dither_score(np.arange(6), arm, 5, window_s=10) == 3
    assert dither_score(np.arange(6), arm, 5) == 1
    assert dither_score(np.arange(6), arm, 3) == 0
    assert dither_score([0, 1, 2], [[0], [1e-8], [0]]) == 0


def test_csv_summary_uses_two_seconds_before_actual_grasp(dataset, tmp_path, capsys):
    grasp = controller(dataset, log_gripper=True, gripper_csv=str(tmp_path/'gripper.csv'))
    grasp.measured = 25
    for step in range(180):
        raw = np.array([step % 2]*5 + [float(step)])
        grasp.process(raw, raw, step//16)
    grasp.mark_grasp(3)
    result = grasp.finish_episode('success')
    assert result['predicted_gripper_min_2s'] == 30
    assert result['predicted_gripper_max_2s'] == 90
    assert result['time_to_grasp_s'] == 3
    assert 'before grasp: [30.0, 90.0]' in capsys.readouterr().out
    grasp.close()
    rows = list(csv.DictReader((tmp_path/'gripper.csv').open()))
    assert all(float(r['measured_gripper']) == 25 for r in rows)
    assert all(float(r['grasp_time_s']) == 3 for r in rows)
    script = Path(__file__).resolve().parents[1]/'scripts/plot_grasp.py'
    subprocess.run([sys.executable, str(script), str(tmp_path/'gripper.csv')], check=True)
    assert (tmp_path/'gripper_ep0.png').stat().st_size > 1000


def test_fixed_noise_resets_before_every_query_and_rejects_missing_ack():
    policy = make_sim_policy(seed_per_episode=True, action_horizon=8)
    calls = []
    policy._client.call_endpoint = lambda endpoint, payload: calls.append(payload['seed']) or payload
    for step in range(24):
        policy.get_action(np.zeros(6), {})
    assert calls == [1984]*3
    policy.reset(seed=33)
    policy.get_action(np.zeros(6), {})
    assert calls[-2:] == [33, 33]
    policy._client.call_endpoint = lambda *args: None
    with pytest.raises(RuntimeError, match='ACK'):
        seed_query(policy._client, 33)


def test_real_latch_reads_joints_each_step_without_extra_camera_reads(dataset):
    clock = FakeClock()
    code = real_code(clock)
    robot = FakeRobot(40, clock)
    reads = []
    robot.get_joint_positions = lambda: reads.append(1) or {'gripper.pos': 25}
    policy = code.So100Adapter(FakeClient(clock=clock), ['room', 'wrist'])
    policy.grasp = controller(dataset, gripper_latch=True)
    policy.grasp_started = 0
    with pytest.raises(KeyboardInterrupt):
        code.run_control_loop(code.EvalConfig(action_horizon=8), robot, policy, [], [])
    assert len(reads) == 41  # final attempted send is interrupted by the test robot
    assert robot.observations == 6  # chunk queries only
    assert all(r['measured_gripper'] == 25 for r in policy.grasp.rows)


def test_sim_latch_uses_measured_joint_and_reset(dataset):
    policy = make_sim_policy(grasp_config=GraspConfig(gripper_latch=True, latch_dataset=str(dataset)))
    policy._iface.get_raw_actions_from_radians = lambda joints: joints
    for _ in range(20):
        policy.get_action(np.array([0.]*5 + [25.]), {})
    assert all(r['measured_gripper'] == 25 for r in policy.grasp.rows)
    policy.reset()
    assert not policy.grasp.latch.latched
    assert not policy.grasp.rows


@pytest.mark.parametrize('kwargs', [dict(latch_confirm_steps=0), dict(latch_steady_steps=0),
    dict(latch_min_hold=-1), dict(latch_close_below=90, latch_open_above=80),
    dict(latch_steady_max_delta=float('nan')), dict(latch_empty_margin=-1)])
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        GripperLatch(0, 100, GraspConfig(**kwargs))


def test_default_off_client_bytes_and_query_count():
    baseline = make_sim_policy(action_horizon=8)
    explicit_off = make_sim_policy(action_horizon=8, grasp_config=GraspConfig(), seed_per_episode=False)
    for _ in range(30):
        a = baseline.get_action(np.zeros(6), {})
        b = explicit_off.get_action(np.zeros(6), {})
        assert a.tobytes() == b.tobytes()
    assert baseline.inference_calls == explicit_off.inference_calls == 4
    assert explicit_off.grasp is None


@pytest.mark.parametrize('smoothing', [{}, {'ema_alpha': .5, 'blend_steps': 3}, {'temporal_ensemble': 2}])
def test_sim_latch_and_logging_preserve_arm_mapping(dataset, smoothing):
    baseline = make_sim_policy(action_horizon=8, **smoothing)
    latched = make_sim_policy(action_horizon=8,
        grasp_config=GraspConfig(gripper_latch=True, latch_dataset=str(dataset)), **smoothing)
    latched._iface.get_raw_actions_from_radians = lambda joints: joints
    def predict(_):
        return {'single_arm': np.zeros((1,16,5)), 'gripper': np.full((1,16,1), 20.)}, {}
    baseline._client.get_action = latched._client.get_action = predict
    for step in range(50):
        a = baseline.get_action(np.array([0.]*5 + [25.]), {})
        b = latched.get_action(np.array([0.]*5 + [25.]), {})
        assert a[:5].tobytes() == b[:5].tobytes()
        if step >= 12:
            assert b[-1] == 70.125  # unchanged FakeInterface mapping: 10*7+.125
    assert baseline.inference_calls == latched.inference_calls


def test_latch_sweep_summary_pairs_schedule_and_reports_metrics(tmp_path):
    from test_summarize_sweep import report
    from sim_to_real_so101.scripts.summarize_sweep import summarize_folder
    off = report(8, [True, False, True, False])
    on = report(8, [True, True, True, False])
    variants = []
    for mode, value, dither in [('off', off, 12), ('on', on, 7)]:
        value['gripper_latch'] = mode == 'on'
        for row in value['episodes']:
            row.update(time_to_grasp_s=1.5 if row['stages']['grasped'] else None, dither_score=dither)
        folder = tmp_path / f'ah_8_latch_{mode}'
        folder.mkdir()
        (folder/'results.json').write_text(json.dumps(value))
        variants.append(dict(horizon=8, latch=mode, folder=folder.name, settings={'gripper_latch': mode == 'on'}))
    config = dict(horizons=[8], variants=variants, selected_splits=['id','ood','yaw'],
                  eval_set_id=off['eval_set_id'],
                  expected_schedule=[(r['start_id'],r['repeat']) for r in off['episodes']])
    (tmp_path/'sweep_config.json').write_text(json.dumps(config))
    summary = summarize_folder(tmp_path)
    comparison = next(r for r in summary['latch_comparisons'] if r['split'] == 'all')
    assert comparison['on_only_successes'] == 1
    assert comparison['success_rate_on_minus_off'] == .25
    assert comparison['grasped_rate_on_minus_off'] == .25
    assert comparison['mean_dither_score_on_minus_off'] == -5
    # An off report in the on folder must not qualify for comparisons or resume.
    (tmp_path/'ah_8_latch_on/results.json').write_text(json.dumps(off))
    summary = summarize_folder(tmp_path)
    assert summary['latch_comparisons'] == [{'action_horizon': 8, 'available': False}]


def test_runner_forwards_grasp_controls_and_seeded_server_without_eval_set():
    root = Path(__file__).resolve().parents[1]
    base = [str(root/'docker/eval_pick_place.sh'), '--model', 'example/checkpoint-200000',
            '--models_dir', '/tmp/models', '--dataset', '/workspace/data', '--dry_run',
            '--seed_per_episode', '--gripper_latch', '--log_gripper', '--latch_steady_steps', '10',
            '--gripper_empty_closed', '2']
    server = subprocess.run(base + ['--server_only'], capture_output=True, text=True, check=True).stdout
    assert 'benchmark_server.py' in server
    client = subprocess.run(base + ['--external_server'], capture_output=True, text=True, check=True).stdout
    assert '--latch_steady_steps 10' in client
    assert '--gripper_empty_closed 2' in client
    assert '--gripper_latch' in client and '--seed_per_episode' in client
