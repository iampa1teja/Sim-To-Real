"""CPU-only tests of the shared smoother and the actual real/sim client paths."""

import ast
import argparse
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.gr00t_client.smoothing import (
    ActionSmoother, FixedRateScheduler, InferenceResult, InferenceWorker, JointEMA,
    SimulationSmoother, SmoothingConfig, TimingLog, blend_actions, log_late,
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


def real_code(clock=time):
    context = dict(dataclass=dataclass, datetime=datetime, RobotConfig=object, PolicyClient=object,
                   Dict=Dict, List=List, Any=Any, np=np, time=clock, logging=logging,
                   ActionSmoother=ActionSmoother, SmoothingConfig=SmoothingConfig,
                   FixedRateScheduler=FixedRateScheduler, InferenceWorker=InferenceWorker,
                   TimingLog=TimingLog, log_late=log_late)
    return load_client_code(ROOT / "docker/real/scripts/so101_eval.py",
                            {"EvalConfig", "So100Adapter", "recursive_add_extra_dim", "smoothing_config",
                             "run_control_loop", "run_smooth_control_loop"}, context)


def sim_code():
    context = dict(LeRobotSO101Interface=object, np=np, torch=SimpleNamespace(Tensor=np.ndarray),
                   deque=deque, SmoothingConfig=SmoothingConfig, SimulationSmoother=SimulationSmoother,
                   TimingLog=TimingLog, datetime=datetime, log_rerun_data=lambda **kw: None)
    return load_client_code(ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py",
                            {"GR00TRemotePolicy"}, context)


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
        return {"single_arm": chunk[None, :, :5], "gripper": chunk[None, :, 5:]}, {}


class FakeRobot:
    def __init__(self, steps, clock=time):
        self.steps = steps
        self.clock = clock
        self.sent = []
        self.ticks = []
        self.logged = []
        self.observations = 0

    def get_observation(self):
        self.observations += 1
        return {**dict.fromkeys(KEYS, float(len(self.sent))),
                "room": np.zeros((2, 2, 3), dtype=np.uint8),
                "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}

    get_observation_cameras_unlocked = get_observation

    def send_action(self, action):
        if len(self.sent) == self.steps:
            raise KeyboardInterrupt
        self.ticks.append(self.clock.monotonic())
        self.sent.append(action.copy())

    def update_log_action(self, action):
        self.logged.append(action.copy())


class FakeInterface:
    SO101_JOINT_ORDER = KEYS

    def get_raw_actions_tensor(self, action):
        return np.array([action[k] for k in KEYS])

    def get_mapped_actions_vectorized(self, action):
        # Deliberately distinguish the untouched mapping from the smoother.
        return action * np.array([2, 3, 4, 5, 6, 7]) + .125


def make_sim_policy(**options):
    policy = sim_code().GR00TRemotePolicy(FakeInterface(), **options)
    policy._client = FakeClient()
    policy._sim_obs_to_groot_inputs = lambda joints, visual: {"joints": joints.copy(), "visual": visual}
    return policy


def test_default_real_path_matches_original_actions_and_timing_exactly():
    clock = FakeClock()
    code = real_code(clock)
    client = FakeClient(.150, clock)
    robot = FakeRobot(27, clock)
    cfg = code.EvalConfig(action_horizon=8)
    policy = code.So100Adapter(client, camera_keys=["room", "wrist"])
    with pytest.raises(KeyboardInterrupt):
        code.run_control_loop(cfg, robot, policy, [], [])

    # Run the original loop as an independent reference, including its sleeps.
    reference_clock = FakeClock()
    reference = FakeRobot(27, reference_clock)
    reference_policy = code.So100Adapter(FakeClient(.150, reference_clock), ["room", "wrist"])
    with pytest.raises(KeyboardInterrupt):
        while True:
            observation = reference.get_observation()
            observation["lang"] = cfg.lang_instruction
            predicted = reference_policy.get_action(observation)
            for action in predicted[:cfg.action_horizon]:
                tic = reference_clock.time()
                reference.send_action(action)
                reference.update_log_action(action)
                toc = reference_clock.time()
                if toc - tic < 1.0 / 30:
                    reference_clock.sleep(1.0 / 30 - (toc - tic))
    assert robot.sent == reference.sent
    assert robot.logged == reference.logged
    assert robot.ticks == reference.ticks
    assert clock.sleeps == reference_clock.sleeps
    assert robot.observations == reference.observations


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


def test_150ms_client_does_not_leave_gaps_in_real_30hz_ticks(tmp_path, capsys, caplog):
    code = real_code()
    cfg = code.EvalConfig(action_horizon=12, prefetch_steps=4, blend_steps=4,
                          ema_alpha=.6, log_timing=True, timing_csv=str(tmp_path / "ticks.csv"))
    client = FakeClient(.150)
    robot = FakeRobot(32)
    policy = code.So100Adapter(client, ["room", "wrist"])
    with caplog.at_level(logging.INFO), pytest.raises(KeyboardInterrupt):
        code.run_control_loop(cfg, robot, policy, [], [])
    assert client.finished.wait(1)
    intervals = np.diff(robot.ticks)
    # Allow OS scheduling jitter, but not an inference-sized 150ms gap.
    assert intervals.max() < 1 / 30 + .025
    assert abs(intervals.mean() - 1 / 30) < .002
    assert len(set(client.threads[1:])) == 1
    assert client.threads[0] != client.threads[1]
    rows = list(csv.DictReader((tmp_path / "ticks.csv").open()))
    assert len(rows) == len(robot.sent)
    assert any(int(row["chunk_id"]) > 0 for row in rows)
    for row, sent in zip(rows, robot.sent):
        assert [float(row[f"sent_{k}"]) for k in KEYS] == [sent[k] for k in KEYS]
    assert "dropping" in caplog.text
    assert "inference mean=" in capsys.readouterr().out


def test_late_real_response_holds_last_sent_target_and_drops_elapsed_steps(tmp_path, caplog):
    clock = FakeClock()
    code = real_code(clock)
    robot = FakeRobot(20, clock)
    client = FakeClient()
    policy = code.So100Adapter(client, ["room", "wrist"])
    smoother_instances = []

    class RecordedSmoother(ActionSmoother):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            smoother_instances.append(self)

    class StepWorker:
        def __init__(self, observe, query, current_step):
            self.observe, self.query, self.current_step = observe, query, current_step
            self.pending = None
            self.chunk_id = 1

        def submit(self):
            if self.pending is not None:
                return False
            origin = self.current_step()
            chunk = self.query(self.observe())
            self.pending = (origin + 5, InferenceResult(origin, chunk, .150, self.chunk_id))
            self.chunk_id += 1
            return True

        def poll(self):
            if self.pending is not None and self.current_step() >= self.pending[0]:
                _, result = self.pending
                self.pending = None
                return result

        def close(self):
            pass

    code.run_smooth_control_loop.__globals__.update(
        ActionSmoother=RecordedSmoother, InferenceWorker=StepWorker,
        FixedRateScheduler=lambda: FixedRateScheduler(clock=clock.monotonic, sleep=clock.sleep))
    cfg = code.EvalConfig(action_horizon=12, prefetch_steps=4, ema_alpha=.6,
                          log_timing=True, timing_csv=str(tmp_path / "late.csv"))
    with caplog.at_level(logging.INFO), pytest.raises(KeyboardInterrupt):
        code.run_control_loop(cfg, robot, policy, [], [])
    rows = list(csv.DictReader((tmp_path / "late.csv").open()))
    assert rows[12]["held"] == "1"
    assert robot.sent[12] == robot.sent[11]  # EMA must not creep during a hold.
    assert rows[13]["chunk_id"] == "1"
    assert float(rows[13]["raw_shoulder_pan.pos"]) == 105
    assert smoother_instances[0].stale_actions_dropped == 5
    np.testing.assert_allclose(np.diff(robot.ticks), 1 / 30, rtol=0, atol=1e-15)
    assert "holding last target" in caplog.text
    assert "dropping 5 stale actions" in caplog.text


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
        from sim_to_real_so101.gr00t_client.grasp import add_grasp_arguments
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


def test_worker_capture_keeps_camera_waits_outside_the_hardware_lock():
    """A slow camera read on the inference thread must not delay send_action on the control thread."""
    path = ROOT / "docker/real/scripts/so101_control.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SO101Control")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("send_action", "get_observation_cameras_unlocked")]
    context = {"Dict": Dict, "Any": Any}
    cls.body, cls.bases, cls.decorator_list = methods, [], []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), context)

    class Bus:
        def sync_read(self, register):
            return {key[:-4]: 1.0 for key in KEYS}

    class SlowCamera:
        def async_read(self):
            time.sleep(0.2)
            return np.zeros((2, 2, 3), dtype=np.uint8)

    sent = []
    control = object.__new__(context["SO101Control"])
    control._hw_lock = threading.Lock()
    control.robot = SimpleNamespace(bus=Bus(), cameras={"room": SlowCamera()},
                                    send_action=lambda action: sent.append(time.monotonic()))
    capture = threading.Thread(target=lambda: control.get_observation_cameras_unlocked())
    capture.start()
    time.sleep(0.05)  # capture is now inside the slow camera read
    started = time.monotonic()
    control.send_action({})
    assert sent and sent[0] - started < 0.05
    capture.join()
    observation = control.get_observation_cameras_unlocked()
    assert set(observation) == set(KEYS) | {"room"}
