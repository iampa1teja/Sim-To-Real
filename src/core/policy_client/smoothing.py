"""Client-side chunk smoothing, with no simulator, robot, or torch dependency.

Actions are in the existing LeRobot joint units. Prediction timestamps are
control-step indices, so stale prefixes and temporal overlaps share one clock.
"""

from dataclasses import dataclass
import csv
import logging
import math
from pathlib import Path
from queue import Empty, Queue
import threading
import time

import numpy as np


@dataclass(frozen=True)
class SmoothingConfig:
    prefetch_steps: int = 0
    blend_steps: int = 0
    ema_alpha: float = 1.0
    smooth_gripper: bool = False
    temporal_ensemble: int = 0
    te_decay: float = 0.01
    log_timing: bool = False

    def validate(self, action_horizon):
        for name in ("prefetch_steps", "blend_steps", "temporal_ensemble"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.prefetch_steps >= action_horizon:
            raise ValueError("prefetch_steps must be less than action_horizon")
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        if not math.isfinite(self.te_decay) or self.te_decay < 0:
            raise ValueError("te_decay must be finite and nonnegative")
        if self.temporal_ensemble and self.blend_steps:
            raise ValueError("temporal_ensemble and blend_steps are mutually exclusive")

    @property
    def enabled(self):
        return bool(self.prefetch_steps or self.blend_steps or self.ema_alpha != 1.0
                    or self.temporal_ensemble or self.log_timing)


def blend_actions(old, new, steps):
    """Cross-fade using weights 1/K, 2/K, ..., 1; hold old's last tail if short.

    A missing old tail is supplied by the caller as the last sent target.
    K=0 returns the original array without even a copy or arithmetic.
    """
    if steps == 0 or len(old) == 0:
        return new
    result = new.copy()
    count = min(steps, len(new))
    weights = (np.arange(count) + 1)[:, None] / steps
    previous = old[np.minimum(np.arange(count), len(old) - 1)]
    result[:count] = (1 - weights) * previous + weights * new[:count]
    return result


class JointEMA:
    def __init__(self, alpha=1.0, smooth_gripper=False, gripper_index=-1):
        self.alpha = alpha
        self.smooth_gripper = smooth_gripper
        self.gripper_index = gripper_index
        self.reset()

    def reset(self):
        self.previous = None

    def apply(self, action):
        if self.previous is None or self.alpha == 1.0:
            sent = action.copy()
        else:
            sent = self.alpha * action + (1 - self.alpha) * self.previous
            if not self.smooth_gripper:
                sent[self.gripper_index] = action[self.gripper_index]
        self.previous = sent.copy()
        return sent


@dataclass
class Prediction:
    start: int
    actions: np.ndarray
    chunk_id: int
    latency: float = 0.0
    raw_actions: np.ndarray | None = None

    @property
    def end(self):
        return self.start + len(self.actions)


class ActionSmoother:
    """Blend/ensemble and then EMA; keep full tails but execute at most H rows.

    On arrival at step t, a chunk observed at s loses its first t-s rows.
    Chunk mode executes min(H, T-(t-s)) rows from that point. Its unused model
    tail remains available for a cross-fade. Ensemble mode uses all T rows.
    """

    def __init__(self, config, action_horizon, joint_keys, grasp=None):
        config.validate(action_horizon)
        self.config = config
        self.grasp = grasp
        self.horizon = action_horizon
        self.joint_keys = tuple(joint_keys)
        gripper = next((i for i, key in enumerate(joint_keys) if key == "gripper.pos"), -1)
        self.ema = JointEMA(config.ema_alpha, config.smooth_gripper, gripper)
        self.reset()

    def reset(self, initial_target=None):
        self.active = None
        self.execution_end = 0
        self.predictions = []
        self.last_sent = None if initial_target is None else np.asarray(initial_target).copy()
        self.last_chunk_id = -1
        self.last_latency = 0.0
        self.stale_actions_dropped = 0
        self.ema.reset()

    def remaining(self, step):
        return max(0, self.execution_end - step)

    def add_chunk(self, actions, observation_step, step, chunk_id, latency=0.0):
        values = np.asarray([[a[k] for k in self.joint_keys] for a in actions], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.joint_keys) or len(values) == 0:
            raise ValueError("Policy returned an empty or malformed action chunk")
        if self.grasp is not None:
            self.grasp.observe_chunk(actions, chunk_id)
        stale = max(0, step - observation_step)
        self.stale_actions_dropped += min(stale, len(values))
        values = values[stale:]
        if len(values) == 0:
            return False
        if self.config.temporal_ensemble:
            self.predictions.append(Prediction(step, values, chunk_id, latency))
        else:
            old = np.empty((0, len(self.joint_keys)))
            if self.active is not None:
                old = self.active.actions[max(0, step - self.active.start):]
            if len(old) == 0 and self.last_sent is not None:
                old = self.last_sent[None, :]
            raw_values = values
            values = blend_actions(old, values, self.config.blend_steps)
            self.active = Prediction(step, values, chunk_id, latency, raw_values)
            self.execution_end = step + min(self.horizon, len(values))
        return True

    def action(self, step):
        prediction = None
        raw = None
        if self.config.temporal_ensemble:
            self.predictions = [p for p in self.predictions if p.end > step]
            overlaps = [p for p in self.predictions if p.start <= step]
            if overlaps:
                # Insertion order is oldest first, as in ACT (older gets weight 1).
                weights = np.exp(-self.config.te_decay * np.arange(len(overlaps)))
                values = np.stack([p.actions[step - p.start] for p in overlaps])
                raw = np.sum(values * weights[:, None], axis=0) / weights.sum()
                prediction = overlaps[-1]
        elif self.active is not None and step < self.execution_end:
            prediction = self.active
            raw = prediction.actions[step - prediction.start].copy()
        held = raw is None
        logged_raw = raw
        if held:
            if self.last_sent is None:
                raise RuntimeError("No prediction or initial hold target")
            # A late response holds the sent target exactly, including with EMA.
            raw = self.last_sent.copy()
            sent = raw.copy()
            logged_raw = raw
        else:
            sent = self.ema.apply(raw)
            if prediction.raw_actions is not None:
                logged_raw = prediction.raw_actions[step - prediction.start].copy()
            self.last_chunk_id = prediction.chunk_id
            self.last_latency = prediction.latency
        if self.grasp is not None:
            if prediction is not None and self.config.temporal_ensemble:
                logged_raw = prediction.actions[step - prediction.start].copy()
            sent = self.grasp.process(logged_raw, sent, self.last_chunk_id, held)
        self.last_sent = sent.copy()
        return logged_raw, sent, self.last_chunk_id, self.last_latency, held

    def as_dict(self, action):
        return {key: float(action[i]) for i, key in enumerate(self.joint_keys)}


class TimingLog:
    """Write CSV per tick and aggregate latency/jerk without retaining the rollout."""

    def __init__(self, path, joint_keys):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.joint_keys = tuple(joint_keys)
        self.arm_indices = [i for i, k in enumerate(joint_keys) if k != "gripper.pos"]
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "chunk_id", "inference_latency_s", "overrun", "held",
                              *[f"raw_{k}" for k in joint_keys], *[f"sent_{k}" for k in joint_keys]])
        self.latencies = []
        self.overruns = 0
        self.jerk_sum = 0.0
        self.jerk_count = 0
        self.reset_episode()

    def reset_episode(self):
        self.previous = []

    def inference(self, latency):
        self.latencies.append(latency)

    def record(self, step, chunk_id, latency, overrun, raw, sent, held=False):
        self.writer.writerow([step, chunk_id, latency, int(overrun), int(held), *raw, *sent])
        self.overruns += int(overrun)
        if len(self.previous) == 2:
            jerk = sent - 2 * self.previous[-1] + self.previous[-2]
            self.jerk_sum += float(np.abs(jerk[self.arm_indices]).sum())
            self.jerk_count += 1
        self.previous = (self.previous + [sent.copy()])[-2:]

    def close(self):
        if self.file.closed:
            return
        self.file.close()
        mean = float(np.mean(self.latencies)) if self.latencies else 0.0
        p95 = float(np.percentile(self.latencies, 95)) if self.latencies else 0.0
        jerk = self.jerk_sum / self.jerk_count if self.jerk_count else 0.0
        print(f"Timing CSV: {self.path}; inference mean={mean:.6f}s p95={p95:.6f}s; "
              f"overruns={self.overruns}; mean joint jerk={jerk:.6f}")


class SimulationSmoother:
    """Synchronous queries, deterministic P-step delivery delay after startup.

    A query at s completes at s+P, so P rows are dropped on delivery. No wall
    clock or thread affects actions. The initial query is delivered immediately.
    """

    def __init__(self, config, action_horizon, joint_keys, timing=None, grasp=None):
        self.config = config
        self.smoother = ActionSmoother(config, action_horizon, joint_keys, grasp=grasp)
        self.timing = timing
        self.total_steps = 0
        self.reset()

    def reset(self):
        self.step = 0
        self.chunk_id = 0
        self.pending = []
        self.smoother.reset()
        if self.timing:
            self.timing.reset_episode()

    @property
    def steps_until_query(self):
        if self.config.temporal_ensemble:
            return (-self.step) % self.config.temporal_ensemble
        if self.pending:
            due, _, actions, _, _ = self.pending[0]
            retained = min(self.smoother.horizon, len(actions) - self.config.prefetch_steps)
            return max(0, due - self.step + retained - self.config.prefetch_steps)
        return max(0, self.smoother.remaining(self.step) - self.config.prefetch_steps)

    def get_action(self, query):
        t = self.step
        while self.pending and self.pending[0][0] <= t:
            _, origin, actions, latency, chunk_id = self.pending.pop(0)
            self.smoother.add_chunk(actions, origin, t, chunk_id, latency)
        due = (t % self.config.temporal_ensemble == 0 if self.config.temporal_ensemble
               else self.smoother.remaining(t) <= self.config.prefetch_steps)
        if due and (self.config.temporal_ensemble or not self.pending):
            started = time.monotonic()
            actions = query()
            latency = time.monotonic() - started
            if self.timing:
                self.timing.inference(latency)
            delay = self.config.prefetch_steps if t else 0
            if delay:
                self.pending.append((t + delay, t, actions, latency, self.chunk_id))
            else:
                self.smoother.add_chunk(actions, t, t, self.chunk_id, latency)
            self.chunk_id += 1
        raw, sent, chunk_id, latency, held = self.smoother.action(t)
        if self.timing:
            self.timing.record(self.total_steps, chunk_id, latency, False, raw, sent, held)
        self.step += 1
        self.total_steps += 1
        return self.smoother.as_dict(sent)


@dataclass
class InferenceResult:
    observation_step: int
    actions: object
    latency: float
    chunk_id: int
    error: BaseException | None = None


class InferenceWorker:
    """One persistent daemon thread owns observation capture and policy requests.

    At most one request is outstanding. Polling and submission never wait for
    inference; slow requests cannot build a queue of obsolete observations.
    """

    def __init__(self, observe, query, current_step, clock=time.monotonic):
        self.observe = observe
        self.query = query
        self.current_step = current_step
        self.clock = clock
        self.requests = Queue()
        self.results = Queue()
        self.busy = False
        self.closed = False
        self.chunk_id = 0
        self.thread = threading.Thread(target=self._run, name="gr00t-inference", daemon=True)
        self.thread.start()

    def submit(self):
        if self.busy or self.closed:
            return False
        self.busy = True
        self.requests.put(self.chunk_id)
        self.chunk_id += 1
        return True

    def _run(self):
        while True:
            chunk_id = self.requests.get()
            if chunk_id is None or self.closed:
                return
            origin = self.current_step()
            started = self.clock()
            try:
                obs = self.observe()
                origin = self.current_step()
                started = self.clock()
                actions = self.query(obs)
                result = InferenceResult(origin, actions, self.clock() - started, chunk_id)
            except BaseException as error:
                result = InferenceResult(origin, None, self.clock() - started, chunk_id, error)
            self.results.put(result)

    def poll(self):
        try:
            result = self.results.get_nowait()
        except Empty:
            return None
        self.busy = False
        if result.error is not None:
            raise result.error
        return result

    def close(self):
        self.closed = True
        self.requests.put(None)
        # Never wait for an unresponsive server at shutdown. The worker never
        # sends motor commands and won't capture another observation after close.


class FixedRateScheduler:
    def __init__(self, hz=30.0, clock=time.monotonic, sleep=time.sleep):
        self.period = 1.0 / hz
        self.clock = clock
        self.sleep = sleep
        self.next_tick = clock()
        self.overruns = 0

    def finish_tick(self):
        self.next_tick += self.period
        remaining = self.next_tick - self.clock()
        overrun = remaining < 0
        if overrun:
            self.overruns += 1
        elif remaining > 0:
            self.sleep(remaining)
        return overrun


def log_late(result, step, horizon):
    elapsed = step - result.observation_step
    logging.info("Chunk %s inference %.3fs, dropping %s stale actions%s", result.chunk_id,
                 result.latency, min(elapsed, len(result.actions)),
                 " (past execution horizon)" if elapsed >= horizon else "")
