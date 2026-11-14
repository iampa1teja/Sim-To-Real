"""Single-owner network worker and clock-driven, aligned action playback.

Only lifecycle calls (connect/reset/close) wait. A control tick polls a Future;
it never waits for network I/O. Playback is independent of threads and accepts
an injected monotonic clock, including a manually advanced clock in tests.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import math
from queue import Queue
from threading import RLock, Thread
import time

import numpy as np

from .smoothing import JointEMA, blend_actions


class SystemClock:
    monotonic = staticmethod(time.monotonic)


class InferenceFault(RuntimeError):
    """Latched failure: reset is required before any more actions can play."""


@dataclass(frozen=True)
class Observation:
    generation: int
    step: int
    captured_at: float
    chunk_id: int
    data: dict


@dataclass(frozen=True)
class TickInfo:
    held: bool = True
    action_age_ms: float | None = None
    chunk_id: int = -1
    turnaround: float = 0.0
    generation: int = 0


class PolicyWorker:
    """Create, use and close the entire plugin client on one thread."""

    def __init__(self, factory, initialize):
        self._queue = Queue()
        self.ready = Future()
        self._closed = False
        self.thread = Thread(target=self._run, args=(factory, initialize),
                             name='policy-inference', daemon=True)
        self.thread.start()

    def _run(self, factory, initialize):
        client = None
        try:
            client = factory()
            self.ready.set_result(initialize(client))
            while True:
                command = self._queue.get()
                if command is None:
                    break
                operation, future = command
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(operation(client))
                    except BaseException as error:
                        future.set_exception(error)
        except BaseException as error:
            if not self.ready.done():
                self.ready.set_exception(error)
        finally:
            if client is not None:
                client.client.close()

    def submit(self, operation):
        if self._closed:
            raise RuntimeError('Policy worker is closed')
        future = Future()
        self._queue.put((operation, future))
        return future

    def close(self):
        if not self._closed:
            self._closed = True
            # Drain an already executing request (bounded by the socket timeout).
            # No socket operation, including close, occurs on this calling thread.
            self._queue.put(None)
        self.thread.join()


class AlignedPlayback:
    """Pure playback state; observation provenance survives blending and holds."""

    def __init__(self, keys, horizon, dt, options, *, gripper_indices=(), grasp=None):
        self.keys, self.horizon, self.dt = tuple(keys), horizon, dt
        self.options, self.grasp = options, grasp
        self.gripper_indices = tuple(gripper_indices)
        self.ema = JointEMA(options.ema_alpha, options.smooth_gripper, gripper_indices)
        self.reset()

    def reset(self):
        self.rows = deque()
        self.last_target = None
        self.last_origin = None
        self.ema_origin = None
        self.chunk_id = -1
        self.turnaround = 0.0
        self.accepted_chunks = 0
        self.ema.reset()

    def add_chunk(self, actions, observation, step, now, turnaround):
        values = np.asarray([[a[k] for k in self.keys] for a in actions], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.keys) or not len(values) or not np.isfinite(values).all():
            raise ValueError('Policy returned an empty, non-finite or malformed action chunk')
        stale = max(step - observation.step,
                    int(math.floor((now - observation.captured_at) / self.dt + 1e-9)), 0)
        values = values[:self.horizon][stale:]
        if not len(values):
            return False
        if self.grasp is not None:
            self.grasp.observe_chunk(actions, observation.chunk_id)
        old = np.asarray([row[1] for row in self.rows])
        old_origins = [row[2] for row in self.rows]
        if not len(old) and self.last_target is not None:
            old, old_origins = self.last_target[None, :], [self.last_origin]
        blended = blend_actions(old, values, self.options.blend_steps)
        if self.options.blend_steps and not self.options.smooth_gripper:
            blended[:, self.gripper_indices] = values[:, self.gripper_indices]
        rows = []
        for i, (raw, target) in enumerate(zip(values, blended)):
            origin = observation.captured_at
            if len(old) and i < self.options.blend_steps - 1:
                previous_origin = old_origins[min(i, len(old_origins) - 1)]
                if previous_origin is not None:
                    origin = min(origin, previous_origin)
            valid_until = observation.captured_at + (stale + i + 1) * self.dt
            rows.append((raw.copy(), target.copy(), origin, observation.chunk_id, turnaround, valid_until))
        self.rows = deque(rows)
        self.accepted_chunks += 1
        return True

    def action(self, now, generation):
        # A delayed camera/control tick must not replay already expired targets.
        while self.rows and self.rows[0][5] <= now + 1e-9:
            self.rows.popleft()
        held = not self.rows
        if held:
            if self.last_target is None:
                raise RuntimeError('An initial measured hold target is required')
            raw = sent = self.last_target.copy()
        else:
            raw, target, origin, self.chunk_id, self.turnaround, _ = self.rows.popleft()
            sent = self.ema.apply(target)
            if self.options.ema_alpha != 1.0 and self.ema_origin is not None:
                origin = min(origin, self.ema_origin)
            self.ema_origin = self.last_origin = origin
        if self.grasp is not None:
            processed = self.grasp.process(raw, sent, self.chunk_id, held)
            if not held:
                sent = processed
        self.last_target = sent.copy()
        age = None if self.last_origin is None else max(0.0, (now - self.last_origin) * 1000)
        return raw, sent, TickInfo(held, age, self.chunk_id, self.turnaround, generation)


class AsyncInference:
    def __init__(self, factory, initialize, decode, keys, horizon, dt, options, *,
                 timeout, refill_seconds=.5, starvation_grace=1., clock=None,
                 gripper_indices=(), grasp=None, timing=None, seed_per_episode=False,
                 episode_seed=1984):
        if options.temporal_ensemble:
            raise ValueError('temporal_ensemble is unsupported with async_inference')
        for name, value in [('control_dt', dt), ('timeout', timeout),
                            ('refill_seconds', refill_seconds), ('starvation_grace', starvation_grace)]:
            if not math.isfinite(value) or value < 0 or (name in ('control_dt', 'timeout') and value == 0):
                raise ValueError(f'{name} must be finite and {"positive" if name in ("control_dt", "timeout") else "nonnegative"}')
        self.factory, self.initialize, self.decode = factory, initialize, decode
        self.clock, self.dt, self.timeout = clock or SystemClock(), dt, timeout
        self.refill_seconds, self.starvation_grace = refill_seconds, starvation_grace
        self.playback = AlignedPlayback(keys, horizon, dt, options,
                                        gripper_indices=gripper_indices, grasp=grasp)
        self.grasp, self.timing = grasp, timing
        self.seed_per_episode, self.episode_seed = seed_per_episode, episode_seed
        self.lock = RLock()
        self.worker = None
        self.pending = None
        self.generation = self.step = self.calls = self.submitted = 0
        self.active = False
        self.fault = None
        self.held_since = None
        self.turnarounds = deque(maxlen=32)
        self.last_tick = TickInfo()

    def connect(self):
        if self.worker is not None:
            raise RuntimeError('Policy is already connected')
        generation = self.generation
        self.worker = PolicyWorker(self.factory, self.initialize)
        try:
            metadata = self.worker.ready.result()
        except BaseException:
            self.worker.close()
            raise
        with self.lock:
            if generation == self.generation:
                self.active = True
        return metadata

    def _invalidate(self, active=False):
        self.generation += 1
        self.active = active
        self.playback.reset()
        self.held_since = None
        self.last_tick = TickInfo(generation=self.generation)

    def cancel(self):
        with self.lock:
            self._invalidate()

    def instruction_changed(self):
        with self.lock:
            target, origin = self.playback.last_target, self.playback.last_origin
            chunk_id, turnaround = self.playback.chunk_id, self.playback.turnaround
            self._invalidate(active=self.active and self.fault is None)
            self.playback.last_target, self.playback.last_origin = target, origin
            self.playback.chunk_id, self.playback.turnaround = chunk_id, turnaround

    def reset(self, seed=None):
        with self.lock:
            self._invalidate()
            generation = self.generation
            self.fault = None
            self.step = self.calls = self.submitted = 0
            self.turnarounds.clear()
            if seed is not None:
                self.episode_seed = int(seed)
        def reset_client(client):
            client.reset()
            if seed is not None:
                self._seed(client, seed)
        try:
            self.worker.submit(reset_client).result()
        except Exception as error:
            with self.lock:
                if generation == self.generation:
                    self._fail(f'Policy reset failed: {error}')
        with self.lock:
            if generation == self.generation:
                self.pending = None
                self.active = True
                if self.grasp is not None:
                    self.grasp.reset()
                if self.timing is not None:
                    self.timing.reset_episode()

    @staticmethod
    def _seed(client, seed):
        from .grasp import seed_query
        seed_query(client, seed)

    def _fail(self, message):
        if self.fault is None:
            self.fault = InferenceFault(message)
            self._invalidate()
        raise self.fault

    def tick(self, make_input, measured_gripper=None):
        with self.lock:
            if self.fault is not None:
                raise self.fault
            if not self.active:
                raise RuntimeError('Policy inference is stopped; reset before playback')
            now = self.clock.monotonic()
            if self.pending is not None:
                future, observation = self.pending
                if future.done():
                    self.pending = None
                    if observation.generation == self.generation:
                        try:
                            actions, finished = future.result()
                            turnaround = finished - observation.captured_at
                            if turnaround >= self.timeout:
                                self._fail('Policy request deadline exceeded')
                            self.calls += 1
                            self.turnarounds.append(turnaround)
                            self.playback.add_chunk(self.decode(actions), observation, self.step, now, turnaround)
                            if self.timing is not None:
                                self.timing.inference(turnaround)
                        except Exception as error:
                            self._fail(f'Policy request failed: {error}')
                elif observation.generation == self.generation and now - observation.captured_at >= self.timeout:
                    self._fail('Policy request deadline exceeded')
            model_input = None
            captured_at = now
            if self.playback.last_target is None:
                model_input = make_input()
                initial = np.asarray(model_input['state'], dtype=float)
                if initial.shape != (len(self.playback.keys),) or not np.isfinite(initial).all():
                    self._fail('Initial measured hold target is malformed or non-finite')
                self.playback.last_target = initial.copy()
            if self.grasp is not None:
                self.grasp.measured = float(measured_gripper)
            raw, sent, info = self.playback.action(now, self.generation)
            self.last_tick = info
            if info.held:
                if self.held_since is None:
                    self.held_since = now
                if now - self.held_since > self.starvation_grace:
                    self._fail('Policy action starvation grace exceeded')
            else:
                self.held_since = None
            recent = float(np.percentile(self.turnarounds, 95)) if self.turnarounds else 0.
            threshold = max(self.refill_seconds, recent + self.dt)
            remaining = max(0., self.playback.rows[-1][5] - now - self.dt) if self.playback.rows else 0.
            if self.pending is None and remaining <= threshold:
                if model_input is None:
                    captured_at = self.clock.monotonic()
                    model_input = make_input()
                # Freeze caller-owned camera/state buffers before crossing threads.
                data = dict(model_input, state=np.asarray(model_input['state']).copy(),
                            images={k: np.asarray(v).copy() for k, v in model_input['images'].items()})
                observation = Observation(self.generation, self.step, captured_at, self.submitted, data)
                seed = self.episode_seed
                def query(client):
                    if self.seed_per_episode:
                        self._seed(client, seed)
                    actions = client.get_action(observation.data)
                    return actions, self.clock.monotonic()
                self.pending = self.worker.submit(query), observation
                self.submitted += 1
            if self.timing is not None:
                self.timing.record(self.step, info.chunk_id, info.turnaround, False, raw, sent,
                                   info.held, action_age_ms=info.action_age_ms,
                                   turnaround=info.turnaround)
            self.step += 1
            return dict(zip(self.playback.keys, map(float, sent)))

    def applied(self, action):
        """Remember the bounded target actually dispatched by the caller."""
        with self.lock:
            if self.active:
                values = np.asarray([action[k] for k in self.playback.keys], dtype=float)
                if not np.isfinite(values).all():
                    self._fail('Applied target contains a non-finite value')
                self.playback.last_target = values

    def close(self):
        self.cancel()
        if self.worker is not None:
            self.worker.close()
