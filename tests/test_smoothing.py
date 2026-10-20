"""CPU-only tests of the shared smoother and the actual simulation client path."""

import ast
import argparse
from collections import deque
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.policy_client.smoothing import (
    ActionSmoother, FixedRateScheduler, InferenceWorker, JointEMA,
    SimulationSmoother, SmoothingConfig, TimingLog, blend_actions,
)

KEYS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")


def actions(base=0, length=16):
    return [{key: float(base + step + i / 10) for i, key in enumerate(KEYS)}
            for step in range(length)]


def load_client_code(path, names, context):
    """Execute production classes/functions without importing hardware/Isaac.

    This exercises their real method bodies and control flow. No torch, CUDA,
    renderer, LeRobot bus, or camera driver is initialized by these tests.
    """
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), context)
    return SimpleNamespace(**context)


def sim_code():
    context = dict(LeRobotSO101Interface=object, np=np, torch=SimpleNamespace(Tensor=np.ndarray),
                   deque=deque, SmoothingConfig=SmoothingConfig, SimulationSmoother=SimulationSmoother,
                   TimingLog=TimingLog, datetime=datetime, log_rerun_data=lambda **kw: None)
    return load_client_code(ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py",
                            {"SimRemotePolicy"}, context)


@pytest.mark.parametrize("policy_name", ["molmoact2", "groot_n17"])
def test_sim_connection_supplies_its_actual_joint_scales_only_to_molmo(monkeypatch, policy_name):
    class CpuArray(np.ndarray):
        def cpu(self):
            return self

    iface = SimpleNamespace(SO101_JOINT_ORDER=KEYS,
        mapping_joint_mins=np.array([-116.279296875, -100, -100, -95, -160, -10]).view(CpuArray),
        mapping_joint_maxs=np.array([119.794921875, 100, 90, 95, 160, 100]).view(CpuArray))
    options = []
    client = SimpleNamespace(client=SimpleNamespace(port=5556),
                             policy=SimpleNamespace(cameras=("room", "wrist")))

    def build(*args, **kwargs):
        options.append(kwargs["policy_options"])
        return client

    monkeypatch.setitem(sys.modules, "sim_to_real_so101.policy_client.adapter",
                        SimpleNamespace(build_plugin_client=build, check_server=lambda _: None))
    policy = sim_code().SimRemotePolicy(iface, policy_name=policy_name, policy_options={"num_steps": "10"})
    policy.connect()
    assert options[0]["num_steps"] == "10"
    if policy_name == "molmoact2":
        assert options[0]["input_joint_units"] == "usd_normalized"
        scales = [float(v) for v in options[0]["arm_degrees_per_unit"].split(",")]
        np.testing.assert_allclose(scales, [1.18037109375, 1, .95, .95, 1.6])
        zeros = [float(v) for v in options[0]["arm_zero_degrees"].split(",")]
        np.testing.assert_allclose(zeros, [1.7578125, 0, -5, 0, 0])
    else:
        assert options[0] == {"num_steps": "10"}


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    monotonic = time

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class FakeClient:
    def __init__(self, latency=0.0, clock=time):
        self.latency = latency
        self.clock = clock
        self.calls = []
        self.threads = []
        self.finished = threading.Event()
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.calls.clear()

    def get_action(self, observation):
        self.finished.clear()
        chunk_id = len(self.calls)
        self.calls.append(observation)
        self.threads.append(threading.get_ident())
        if self.latency:
            self.clock.sleep(self.latency)
        chunk = np.array([[a[k] for k in KEYS] for a in actions(100 * chunk_id)])
        self.finished.set()
        return chunk


class FakeInterface:
    SO101_JOINT_ORDER = KEYS

    def get_raw_actions_tensor(self, action):
        return np.array([action[k] for k in KEYS])

    def get_mapped_actions_vectorized(self, action):
        # Deliberately distinguish the untouched mapping from the smoother.
        return action * np.array([2, 3, 4, 5, 6, 7]) + .125


def make_sim_policy(**options):
    policy = sim_code().SimRemotePolicy(FakeInterface(), **options)
    policy._client = FakeClient()
    policy._sim_obs_to_inputs = lambda joints, visual: {"joints": joints.copy(), "visual": visual}
    return policy


class ThirtyTargetClient(FakeClient):
    def get_action(self, observation):
        chunk_id = len(self.calls)
        self.calls.append(observation)
        return np.array([[row[key] for key in KEYS] for row in actions(100 * chunk_id, length=30)])


def test_repeated_native_chunk_replans_every_five_seconds_and_resets_its_clock():
    policy = make_sim_policy(action_horizon=30, action_repeat=5)
    policy._client = ThirtyTargetClient()
    sent, remaining = [], []
    for step in range(301):
        remaining.append(policy.steps_until_query)
        sent.append(policy.get_action(np.full(6, step), {}))
    query_steps = [int(call["joints"][0]) for call in policy._client.calls]
    assert query_steps == [0, 150, 300]
    assert [step / 30 for step in query_steps] == [0, 5, 10]
    assert remaining[:7] == [0, 149, 148, 147, 146, 145, 144]
    assert remaining[149:152] == [1, 0, 149]
    expected = []
    for step in range(301):
        row = actions(100 * (step // 150), length=30)[step % 150 // 5]
        expected.append(FakeInterface().get_mapped_actions_vectorized(
            FakeInterface().get_raw_actions_tensor(row)))
    np.testing.assert_array_equal(sent, expected)
    assert policy.inference_calls == 3
    policy.reset()
    assert policy.steps_until_query == 0
    np.testing.assert_array_equal(policy.get_action(np.zeros(6), {}), expected[0])
    assert policy.inference_calls == 1


def test_repeated_targets_and_ema_use_the_same_physical_control_clock():
    policy = make_sim_policy(action_horizon=30, action_repeat=5, ema_alpha=.6)
    policy._client = ThirtyTargetClient()
    sent = [policy.get_action(np.full(6, step), {}) for step in range(301)]
    assert [int(call["joints"][0]) for call in policy._client.calls] == [0, 150, 300]
    assert policy._smoothing.smoother.horizon == 150
    previous, reference = None, []
    for step in range(301):
        row = actions(100 * (step // 150), length=30)[step % 150 // 5]
        raw = FakeInterface().get_raw_actions_tensor(row)
        filtered = raw.copy()
        if previous is not None:
            filtered[:5] = .6 * raw[:5] + .4 * previous[:5]
        previous = filtered
        reference.append(FakeInterface().get_mapped_actions_vectorized(filtered))
    np.testing.assert_allclose(sent, reference)
    policy.reset()
    assert policy.steps_until_query == 0
    np.testing.assert_array_equal(policy.get_action(np.zeros(6), {}), reference[0])


def test_grasp_controller_observes_expanded_chunk_in_control_ticks():
    policy = make_sim_policy(action_horizon=8, action_repeat=5)
    policy._client = ThirtyTargetClient()
    policy._iface.get_raw_actions_from_radians = lambda joints: joints
    observed = []
    policy.grasp = SimpleNamespace(gripper_index=5,
        observe_chunk=lambda chunk, chunk_id: observed.append((chunk, chunk_id)),
        process=lambda raw, sent, chunk_id: raw)
    policy.get_action(np.zeros(6), {})
    chunk, chunk_id = observed[0]
    assert chunk_id == 0 and len(chunk) == 150
    assert [row[KEYS[0]] for row in chunk[:10]] == [0] * 5 + [1] * 5
    assert policy.steps_until_query == 39


@pytest.mark.parametrize("repeat", [0, -1, True, False, 1.5, "5", None])
def test_invalid_action_repeat_fails_before_querying(repeat):
    with pytest.raises(ValueError, match="action_repeat"):
        make_sim_policy(action_repeat=repeat)


@pytest.mark.parametrize("horizon", [1, 8, 12, 16])
def test_default_sim_policy_matches_original_sequence_exactly(horizon):
    policy = make_sim_policy(action_horizon=horizon)
    actual = [policy.get_action(np.zeros(6), {}) for _ in range(35)]
    reference = []
    for step in range(35):
        row = actions(100 * (step // horizon))[step % horizon]
        reference.append(FakeInterface().get_mapped_actions_vectorized(
            FakeInterface().get_raw_actions_tensor(row)))
    np.testing.assert_array_equal(actual, reference)
    assert policy.inference_calls == (35 + horizon - 1) // horizon
    assert policy._smoothing is None
    assert policy._timing is None


def test_stale_chunk_expires_without_replaying_old_actions():
    smoother = ActionSmoother(SmoothingConfig(), 12, KEYS)
    smoother.add_chunk(actions(), 0, 0, 0)
    smoother.action(0)
    assert not smoother.add_chunk(actions(100), 1, 17, 1)
    assert smoother.stale_actions_dropped == 16
    _, sent, _, _, held = smoother.action(17)
    assert held
    np.testing.assert_array_equal(sent, [actions()[0][k] for k in KEYS])


def test_blend_uses_old_unexecuted_tail_and_k_zero_is_noop():
    old = np.array([[4.], [5.], [6.], [7.]])
    new = np.full((6, 1), 10.)
    assert blend_actions(old, new, 0) is new
    np.testing.assert_array_equal(blend_actions(old, new, 4).ravel(), [5.5, 7.5, 9, 10, 10, 10])
    smoother = ActionSmoother(SmoothingConfig(blend_steps=4), 4, KEYS)
    smoother.add_chunk(actions(length=8), 0, 0, 0)
    previous = [smoother.action(t)[1] for t in range(4)]
    smoother.add_chunk(actions(10), 4, 4, 1)
    raw, sent, *_ = smoother.action(4)
    assert raw[0] == 10  # CSV records the model output before the cross-fade.
    assert sent[0] == 5.5  # Old row 4, rather than repeating old row 3.
    assert abs(sent[0] - previous[-1][0]) < abs(raw[0] - previous[-1][0])


def test_blend_after_tail_exhaustion_fades_from_last_target():
    smoother = ActionSmoother(SmoothingConfig(blend_steps=2), 2, KEYS)
    smoother.add_chunk(actions(length=2), 0, 0, 0)
    smoother.action(0)
    last = smoother.action(1)[1]
    smoother.add_chunk(actions(11), 3, 3, 1)
    sent = smoother.action(3)[1]
    np.testing.assert_array_equal(sent, (last + np.array([actions(11)[0][k] for k in KEYS])) / 2)


def test_ema_gripper_passthrough_and_reset():
    ema = JointEMA(.6)
    first = np.array([1., 2., 3., 4., 5., 6.])
    second = first + 10
    np.testing.assert_array_equal(ema.apply(first), first)
    result = ema.apply(second)
    np.testing.assert_allclose(result[:5], first[:5] + 6)
    assert result[5] == second[5]
    ema.reset()
    np.testing.assert_array_equal(ema.apply(second), second)
    ema = JointEMA(.6, smooth_gripper=True)
    ema.apply(first)
    np.testing.assert_allclose(ema.apply(second), first + 6)


def test_temporal_ensemble_oldest_has_highest_weight():
    smoother = ActionSmoother(SmoothingConfig(temporal_ensemble=1, te_decay=math_log_two()), 4, KEYS)
    smoother.add_chunk(actions(0), 0, 0, 0)
    smoother.add_chunk(actions(10), 1, 1, 1)
    smoother.add_chunk(actions(20), 2, 2, 2)
    _, sent, chunk_id, _, held = smoother.action(2)
    # Predictions at step 2 are 2, 11, 20, weighted 1, 1/2, 1/4.
    expected = (2 + .5 * 11 + .25 * 20) / 1.75
    assert sent[0] == pytest.approx(expected)
    assert chunk_id == 2 and not held
    assert smoother.action(16)[1][0] == pytest.approx((25 + .5 * 34) / 1.5)
    assert smoother.action(17)[1][0] == 35  # Only the third prediction still covers step 17.


def math_log_two():
    return float(np.log(2))


def test_sim_prefetch_queries_early_and_drops_exactly_p_actions():
    policy = make_sim_policy(action_horizon=12, prefetch_steps=4)
    sent = []
    query_steps = []
    for step in range(30):
        calls = policy.inference_calls
        sent.append(policy.get_action(np.full(6, step), {}))
        if policy.inference_calls != calls:
            query_steps.append(step)
    assert query_steps == [0, 8, 20]
    expected = ([*actions(0)[:12], *actions(100)[4:16], *actions(200)[4:10]])
    reference = [FakeInterface().get_mapped_actions_vectorized(
        FakeInterface().get_raw_actions_tensor(row)) for row in expected]
    np.testing.assert_array_equal(sent, reference)
    assert policy._smoothing.smoother.stale_actions_dropped == 8
    assert [int(call["joints"][0]) for call in policy._client.calls] == query_steps


def test_sim_temporal_queries_every_m_with_overlapping_delayed_deliveries():
    policy = make_sim_policy(action_horizon=12, prefetch_steps=4, temporal_ensemble=2, te_decay=0)
    sent = []
    for step in range(9):
        sent.append(policy.get_action(np.full(6, step), {}))
    assert [int(call["joints"][0]) for call in policy._client.calls] == [0, 2, 4, 6, 8]
    assert sent[6][0] == pytest.approx((6 + 104) / 2 * 2 + .125)
    assert sent[8][0] == pytest.approx((8 + 106 + 204) / 3 * 2 + .125)


def test_sim_ema_and_ensemble_state_resets_per_episode():
    policy = make_sim_policy(action_horizon=4, ema_alpha=.6, temporal_ensemble=2)
    first = [policy.get_action(np.full(6, step), {}) for step in range(9)]
    policy.reset()
    second = [policy.get_action(np.full(6, step), {}) for step in range(9)]
    np.testing.assert_array_equal(first, second)
    assert policy._client.resets == 1


def test_sim_blend_and_ema_run_before_the_existing_joint_mapping():
    policy = make_sim_policy(action_horizon=4, blend_steps=2, ema_alpha=.5)
    sent = [policy.get_action(np.zeros(6), {}) for _ in range(6)]
    # Old arm tail row 4 and new row 100 cross-fade to 52, then EMA with
    # previous target 2.125 gives 27.0625, then the untouched mapping applies.
    assert sent[4][0] == 54.25
    assert sent[5][0] == 128.1875
    # Gripper participates in blending but passes through EMA.
    assert sent[4][5] == 52.5 * 7 + .125
    assert sent[5][5] == 101.5 * 7 + .125


def test_sim_timing_alone_does_not_change_actions(tmp_path, capsys):
    baseline = make_sim_policy(action_horizon=8)
    timed = make_sim_policy(action_horizon=8, log_timing=True, timing_csv=str(tmp_path / "sim.csv"))
    try:
        expected = [baseline.get_action(np.zeros(6), {}) for _ in range(25)]
        actual = [timed.get_action(np.zeros(6), {}) for _ in range(25)]
        np.testing.assert_array_equal(actual, expected)
        assert timed.inference_calls == baseline.inference_calls == 4
    finally:
        timed.close_timing()
    assert len(list(csv.DictReader((tmp_path / "sim.csv").open()))) == 25
    assert "overruns=0" in capsys.readouterr().out


def test_new_sim_cli_flags_and_early_validation():
    path = ROOT / "source/sim_to_real_so101/scripts/lerobot_eval.py"
    tree = ast.parse(path.read_text())
    boundary = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Import)
                    and any(alias.name == "gymnasium" for alias in node.names))
    body = [node for node in tree.body[:boundary] if not isinstance(node, (ast.Import, ast.ImportFrom))]
    launches = []

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, args):
            launches.append(args)
            self.app = SimpleNamespace()

    def parse(arguments):
        from sim_to_real_so101.policy_client.grasp import add_grasp_arguments
        context = dict(argparse=argparse, json=json, math=math, os=os, Path=Path, AppLauncher=Launcher,
                       add_grasp_arguments=add_grasp_arguments)
        with patch.object(sys, "argv", ["lerobot_eval", *arguments]):
            exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), context)
        return context["args_cli"]

    defaults = parse([])
    assert (defaults.prefetch_steps, defaults.blend_steps, defaults.ema_alpha,
            defaults.temporal_ensemble, defaults.te_decay, defaults.log_timing) == (0, 0, 1, 0, .01, False)
    preset = parse(["--action_horizon", "12", "--prefetch_steps", "4", "--blend_steps", "4",
                    "--ema_alpha", ".6", "--smooth_gripper", "--log_timing"])
    assert (preset.prefetch_steps, preset.blend_steps, preset.ema_alpha) == (4, 4, .6)
    assert preset.smooth_gripper and preset.log_timing
    invalid = [["--prefetch_steps=-1"], ["--action_horizon=4", "--prefetch_steps=4"],
               ["--blend_steps=-1"], ["--ema_alpha=0"], ["--ema_alpha=1.1"], ["--ema_alpha=nan"],
               ["--temporal_ensemble=-1"], ["--te_decay=inf"], ["--te_decay=-.1"],
               ["--temporal_ensemble=2", "--blend_steps=4"]]
    for arguments in invalid:
        before = len(launches)
        with pytest.raises(SystemExit) as error:
            parse(arguments)
        assert error.value.code == 2
        assert len(launches) == before


@pytest.mark.parametrize("options", [dict(prefetch_steps=-1), dict(prefetch_steps=12),
                                       dict(blend_steps=-1), dict(ema_alpha=0), dict(ema_alpha=1.1),
                                       dict(ema_alpha=float("nan")), dict(te_decay=float("inf")),
                                       dict(te_decay=-1), dict(temporal_ensemble=-1),
                                       dict(temporal_ensemble=2, blend_steps=2)])
def test_invalid_options_fail_before_a_rollout(options):
    with pytest.raises(ValueError):
        SmoothingConfig(**options).validate(12)


def test_fixed_rate_uses_absolute_deadlines_and_counts_overruns():
    clock = FakeClock()
    scheduler = FixedRateScheduler(clock=clock.monotonic, sleep=clock.sleep)
    clock.now += .01
    assert not scheduler.finish_tick()
    assert clock.now == pytest.approx(1 / 30)
    clock.now += .04
    assert scheduler.finish_tick()
    assert scheduler.overruns == 1
    assert scheduler.next_tick == pytest.approx(2 / 30)
    assert not scheduler.finish_tick()
    assert clock.now == pytest.approx(3 / 30)


def test_inference_worker_captures_fresh_observation_once_and_never_blocks_submit():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    step = [3]
    seen = []

    def observe():
        return {"step": step[0]}

    def query(obs):
        seen.append(obs)
        entered.set()
        assert release.wait(1)
        finished.set()
        return actions()

    worker = InferenceWorker(observe, query, lambda: step[0])
    try:
        assert worker.submit()
        assert entered.wait(1)
        assert not worker.submit()
        assert worker.poll() is None
        step[0] = 8
        release.set()
        assert finished.wait(1)
        # Synchronize on the result rather than depending on OS thread timing.
        ready = worker.results.get(timeout=1)
        worker.results.put(ready)
        result = worker.poll()
        assert result.observation_step == 3
        assert seen == [{"step": 3}]
        smoother = ActionSmoother(SmoothingConfig(), 12, KEYS)
        smoother.add_chunk(result.actions, result.observation_step, 8, result.chunk_id)
        assert smoother.stale_actions_dropped == 5
        assert smoother.action(8)[1][0] == 5
    finally:
        release.set()
        worker.close()
        worker.thread.join(1)
    assert not worker.thread.is_alive()


def test_inference_worker_propagates_failures_without_waiting_at_shutdown():
    entered = threading.Event()
    release = threading.Event()

    def fail(_):
        entered.set()
        assert release.wait(1)
        raise RuntimeError("fake client failure")

    worker = InferenceWorker(lambda: {}, fail, lambda: 0)
    try:
        assert worker.submit()
        assert entered.wait(1)
        worker.close()
        assert worker.thread.is_alive()  # close did not wait for slow inference.
        release.set()
        result = worker.results.get(timeout=1)
        worker.results.put(result)
        with pytest.raises(RuntimeError, match="fake client failure"):
            worker.poll()
        assert not worker.submit()
    finally:
        release.set()
        worker.thread.join(1)
    assert not worker.thread.is_alive()


def test_timing_summary_uses_once_per_inference_latency_and_arm_jerk(tmp_path, capsys):
    log = TimingLog(tmp_path / "timing.csv", KEYS)
    log.inference(.1)
    log.inference(.2)
    for step, pan in enumerate([0, 1, 3]):
        row = np.array([pan, 0, 0, 0, 0, 100 * step * step], dtype=float)
        log.record(step, 0, .1, step == 1, row, row)
    assert log.jerk_sum == 1  # Ignore gripper second difference.
    log.reset_episode()
    log.record(3, 1, .2, False, row, row)
    assert log.jerk_count == 1
    log.close()
    summary = capsys.readouterr().out
    assert "mean=0.150000s p95=0.195000s" in summary
    assert "overruns=1; mean joint jerk=1.000000" in summary
