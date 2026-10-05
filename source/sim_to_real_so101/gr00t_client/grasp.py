"""Optional gripper control and diagnostics in LeRobot action units (CPU only)."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import logging
import math

import numpy as np


@dataclass(frozen=True)
class GraspConfig:
    gripper_latch: bool = False
    latch_close_below: float | None = None
    latch_confirm_steps: int = 3
    latch_open_above: float | None = None
    latch_release_confirm: int = 5
    latch_min_hold: int = 30
    latch_dataset: str = ""
    latch_steady_steps: int = 10
    latch_steady_max_delta: float = 2.0
    gripper_empty_closed: float | None = None
    latch_empty_margin: float = 3.0
    latch_empty_confirm: int = 15
    latch_reopen_steps: int = 15
    log_gripper: bool = False
    gripper_csv: str = ""

    @property
    def enabled(self):
        return self.gripper_latch or self.log_gripper


def training_bounds(dataset):
    """Read gripper action extrema and named column from LeRobot training metadata."""
    root = Path(dataset).expanduser()
    info = json.loads((root / 'meta/info.json').read_text())
    names = info['features']['action']['names']
    if isinstance(names, dict):
        names = names['motors']
    index = names.index('gripper.pos')
    stats = json.loads((root / 'meta/stats.json').read_text())['action']
    closed, opened = float(stats['min'][index]), float(stats['max'][index])
    if not math.isfinite(closed) or not math.isfinite(opened) or closed >= opened:
        raise ValueError('Training gripper extrema must be finite and closed < open')
    return closed, opened


class GripperLatch:
    def __init__(self, closed, opened, config):
        self.closed, self.opened = closed, opened
        self.empty_closed = config.gripper_empty_closed
        self.close_below = (closed + .50 * (opened - closed)
                            if config.latch_close_below is None else config.latch_close_below)
        self.open_above = (closed + .80 * (opened - closed)
                           if config.latch_open_above is None else config.latch_open_above)
        for name in ('latch_confirm_steps', 'latch_release_confirm', 'latch_min_hold',
                     'latch_steady_steps', 'latch_empty_confirm', 'latch_reopen_steps'):
            value = getattr(config, name)
            if type(value) is not int or value < (0 if name == 'latch_min_hold' else 1):
                raise ValueError(f'{name} must be a positive integer (min_hold may be zero)')
        if not (math.isfinite(self.close_below) and math.isfinite(self.open_above)
                and self.close_below <= self.open_above):
            raise ValueError('Require finite close threshold <= open threshold')
        for value in (config.latch_steady_max_delta, config.latch_empty_margin):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Steadiness limit and empty margin must be finite and nonnegative")
        self.config = config
        self.reset()

    def reset(self):
        self.latched = False
        self.close_count = self.open_count = 0
        self.engaged_step = None
        self.empty_count = 0
        self.reopen_until = -1

    def apply(self, predicted, commanded, step, held=False, steady=True, measured=None):
        event = ''
        if step < self.reopen_until:
            return self.opened, event
        if self.latched and measured is not None and self.empty_closed is not None:
            empty = math.isfinite(measured) and measured <= self.empty_closed + self.config.latch_empty_margin
            self.empty_count = self.empty_count + 1 if empty else 0
            if self.empty_count >= self.config.latch_empty_confirm:
                self.latched = False
                self.close_count = self.open_count = self.empty_count = 0
                self.reopen_until = step + self.config.latch_reopen_steps
                logging.warning('Gripper missed grasp at step %d: measured=%.6f; reopening', step, measured)
                return self.opened, 'missed grasp'
        # A held target is not a new model prediction and cannot confirm either transition.
        if held or not math.isfinite(predicted):
            self.close_count = self.open_count = 0
        elif self.latched:
            self.open_count = self.open_count + 1 if predicted > self.open_above else 0
            if (step - self.engaged_step >= self.config.latch_min_hold
                    and self.open_count >= self.config.latch_release_confirm):
                self.latched = False
                self.open_count = self.close_count = 0
                event = 'release'
        else:
            self.close_count = self.close_count + 1 if steady and predicted < self.close_below else 0
            if self.close_count >= self.config.latch_confirm_steps:
                self.latched = True
                self.engaged_step = step
                self.empty_count = 0
                self.close_count = self.open_count = 0
                event = 'engage'
        if event:
            print(f'Gripper latch {event} at step {step}: predicted={predicted:.6f}', flush=True)
        return (self.closed if self.latched else commanded), event


def dither_score(times, arm, end_time=None, window_s=2.0, epsilon=1e-6):
    """Count direction reversals, summed over arm joints in the final 2 s.

    Use commanded arm positions, omit zero/epsilon-sized velocities, and never
    count a velocity crossing the window boundary. No FK or gripper is involved.
    """
    times, arm = np.asarray(times), np.asarray(arm)
    if len(times) < 3:
        return 0
    end = times[-1] if end_time is None else end_time
    values = arm[(times >= end - window_s) & (times <= end)]
    if len(values) < 3:
        return 0
    delta = np.diff(values, axis=0)
    total = 0
    for joint in delta.T:
        signs = np.sign(joint[np.abs(joint) > epsilon])
        total += int(np.count_nonzero(signs[1:] != signs[:-1]))
    return total


def seed_query(client, seed):
    response = client.call_endpoint('seed_policy', {'seed': int(seed)})
    if response != {'seed': int(seed)}:
        raise RuntimeError('Fixed episode noise requires seed_policy ACK; start benchmark_server.py')


class GraspController:
    def __init__(self, config, joint_keys, dt=1/30):
        self.config, self.keys, self.dt = config, tuple(joint_keys), dt
        self.gripper_index = self.keys.index('gripper.pos')
        self.arm_indices = [i for i in range(len(self.keys)) if i != self.gripper_index]
        dataset = config.latch_dataset or str(Path(__file__).resolve().parents[3] / 'datasets/pick_place_sim')
        closed, opened = training_bounds(dataset)
        self.latch = GripperLatch(closed, opened, config)
        if config.gripper_empty_closed is None:
            info = json.loads((Path(dataset).expanduser() / 'meta/info.json').read_text())
            names = info['features']['observation.state']['names']
            if isinstance(names, dict):
                names = names['motors']
            state_stats = json.loads((Path(dataset).expanduser() / 'meta/stats.json').read_text())
            self.latch.empty_closed = float(state_stats['observation.state']['min'][names.index('gripper.pos')])
        if not math.isfinite(self.latch.empty_closed):
            raise ValueError('gripper_empty_closed must be finite')
        print(f'Empty-closed measured gripper={self.latch.empty_closed:.9g}; '
              f'empty threshold={self.latch.empty_closed + config.latch_empty_margin:.9g}', flush=True)
        print(f'Gripper thresholds from {dataset}: closed={closed:.9g}, open={opened:.9g}, '
              f'C={self.latch.close_below:.9g}, O={self.latch.open_above:.9g}; '
              f'latch={config.gripper_latch}', flush=True)
        self.calibration = dict(dataset=str(dataset), closed=closed, opened=opened,
                                close_below=self.latch.close_below, open_above=self.latch.open_above,
                                empty_closed=self.latch.empty_closed)
        self.path = None
        self.file = self.writer = None
        if config.log_gripper:
            self.path = Path(config.gripper_csv or f'outputs/gripper/grasp_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open('w', newline='')
            print(f'Gripper CSV: {self.path}', flush=True)
        self.clock = None
        self.episode = -1
        self.reset()

    def reset(self):
        if getattr(self, 'rows', None):
            self.finish_episode(reason='reset')
        self.episode += 1
        self.rows = []
        self.previous_endpoint = None
        self.endpoint_deltas = {}
        self.grasp_time = None
        self.measured = None
        self.arm_history = []
        self.latch.reset()

    def observe_chunk(self, actions, chunk_id):
        endpoint = np.asarray([actions[-1][self.keys[i]] for i in self.arm_indices])
        delta = (np.full(len(endpoint), np.nan) if self.previous_endpoint is None
                 else endpoint - self.previous_endpoint)
        self.endpoint_deltas[chunk_id] = delta
        self.previous_endpoint = endpoint.copy()

    def process(self, raw, sent, chunk_id, held=False, timestamp=None):
        step = len(self.rows)
        if timestamp is None:
            timestamp = self.clock() if self.clock is not None else step * self.dt
        result = sent.copy()
        predicted = float(raw[self.gripper_index]) if not held else float('nan')
        event = ''
        arm = np.asarray(sent)[self.arm_indices]
        self.arm_history.append(arm.copy())
        self.arm_history = self.arm_history[-(self.config.latch_steady_steps + 1):]
        steady = (len(self.arm_history) > self.config.latch_steady_steps and
                  np.max(np.abs(np.diff(self.arm_history, axis=0))) <= self.config.latch_steady_max_delta)
        if self.config.gripper_latch:
            result[self.gripper_index], event = self.latch.apply(
                predicted, result[self.gripper_index], step, held, steady, self.measured)
        delta = self.endpoint_deltas.get(chunk_id, np.full(len(self.arm_indices), np.nan))
        row = dict(episode=self.episode, step=step, time_s=timestamp,
                   predicted_gripper=predicted, measured_gripper=self.measured, arm_steady=int(steady),
                   commanded_gripper=float(result[self.gripper_index]),
                   latch_state=int(self.latch.latched), latch_event=event, chunk_id=chunk_id,
                   held=int(held), endpoint_delta_norm=float(np.linalg.norm(delta)),
                   close_below=self.latch.close_below, open_above=self.latch.open_above,
                   empty_closed=self.latch.empty_closed,
                   empty_threshold=self.latch.empty_closed + self.config.latch_empty_margin)
        for i, index in enumerate(self.arm_indices):
            row['arm_' + self.keys[index]] = float(result[index])
            row['endpoint_delta_' + self.keys[index]] = float(delta[i])
        self.rows.append(row)
        return result

    def mark_grasp(self, timestamp):
        if self.grasp_time is None:
            self.grasp_time = timestamp

    def finish_episode(self, reason='timeout'):
        times = [r['time_s'] for r in self.rows]
        arm = [[r['arm_' + self.keys[i]] for i in self.arm_indices] for r in self.rows]
        score = dither_score(times, arm, self.grasp_time)
        end = self.grasp_time if self.grasp_time is not None else (times[-1] if times else 0)
        predictions = [r["predicted_gripper"] for r in self.rows
                       if end - 2 <= r["time_s"] <= end and math.isfinite(r["predicted_gripper"])]
        low, high = (min(predictions), max(predictions)) if predictions else (None, None)
        print(f"Episode {self.episode}: predicted gripper range in final 2 s before "
              f"{'grasp' if self.grasp_time is not None else reason}: [{low}, {high}]; dither={score}", flush=True)
        if self.file and self.rows:
            for row in self.rows:
                row.update(grasp_time_s=self.grasp_time, end_reason=reason, dither_score=score)
                if self.writer is None:
                    self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            self.file.flush()
        self.rows = []
        return dict(dither_score=score, time_to_grasp_s=self.grasp_time,
                    predicted_gripper_min_2s=low, predicted_gripper_max_2s=high)

    def close(self):
        if self.rows:
            self.finish_episode(reason='interrupted')
        if self.file:
            self.file.close()
            self.file = None


def add_grasp_arguments(parser, include_latch=True):
    if include_latch:
        parser.add_argument('--gripper_latch', action='store_true')
    parser.add_argument('--latch_close_below', type=float)
    parser.add_argument('--latch_open_above', type=float)
    parser.add_argument('--latch_confirm_steps', type=int, default=3)
    parser.add_argument('--latch_release_confirm', type=int, default=5)
    parser.add_argument('--latch_min_hold', type=int, default=30)
    parser.add_argument('--latch_dataset', default='', help='Training dataset with meta/info.json and meta/stats.json')
    parser.add_argument('--latch_steady_steps', type=int, default=10)
    parser.add_argument('--latch_steady_max_delta', type=float, default=2.0)
    parser.add_argument('--gripper_empty_closed', type=float)
    parser.add_argument('--latch_empty_margin', type=float, default=3.0)
    parser.add_argument('--latch_empty_confirm', type=int, default=15)
    parser.add_argument('--latch_reopen_steps', type=int, default=15)
    parser.add_argument('--log_gripper', action='store_true')
    parser.add_argument('--gripper_csv', default='')
    parser.add_argument('--seed_per_episode', action='store_true')
