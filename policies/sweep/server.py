"""NumPy-only trajectory playback; deliberately independent of the repo's core package."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from policy_server.contract import ServerPolicy


class SweepServerPolicy(ServerPolicy):
    def __init__(self):
        self.horizon = 30
        self.tick = 0
        self.observed_start = None
        self.rows = np.empty((0, 0), dtype=np.float32)
        self.streams = {}

    def load(self, checkpoint, device, options):
        data = json.loads(Path(options.get('trajectory', '/policies/sweep/sweep.json')).read_text())
        self.horizon = int(options.get('horizon', 30))
        self.fps = float(data['fps'])
        self.home = np.asarray(data['home_raw'], dtype=np.float32)
        self.trajectory = np.asarray(data['ticks'], dtype=np.float32)
        self.factors = np.asarray(data['deg_per_raw'], dtype=np.float64)
        self.low, self.high = (np.asarray(data[key], dtype=np.float32) for key in ('raw_min', 'raw_max'))
        if (self.horizon < 1 or self.fps <= 0 or self.trajectory.ndim != 2
                or self.trajectory.shape[1:] != self.home.shape or not len(self.trajectory)
                or self.factors.shape != self.home.shape or np.any(self.factors <= 0)
                or not all(np.isfinite(x).all() for x in (self.trajectory, self.home, self.factors))
                or np.any(self.trajectory < self.low) or np.any(self.trajectory > self.high)):
            raise ValueError('Invalid sweep trajectory')
        self.reset()

    def reset(self, seed=None):
        self.tick = 0
        self.observed_start = None
        self.rows = getattr(self, 'trajectory', np.empty((0, 0), dtype=np.float32))
        self.streams = {}
        return {'seed': seed}

    def infer(self, request):
        stream = request.get('sweep_stream', 'default')
        self.tick, self.observed_start, self.rows = self.streams.get(
            stream, (0, None, self.trajectory))
        if self.observed_start is None:
            state = np.asarray(request['state'], dtype=np.float32)
            if state.shape != self.home.shape or not np.isfinite(state).all():
                raise ValueError('Expected a finite start state in arm order')
            if np.any(state < self.low) or np.any(state > self.high):
                raise ValueError('Observed start is outside calibrated raw bounds')
            self.observed_start = state.copy()
            count = max(1, math.ceil(1.5 * np.max(np.abs(self.home - state) * self.factors) / 15 * self.fps))
            t = np.arange(count, dtype=np.float64) / count
            ramp = state + (self.home - state) * (t*t*(3 - 2*t))[:, None]
            self.rows = np.concatenate([ramp.astype(np.float32), self.trajectory])
        actions = np.tile(self.home, (self.horizon, 1))
        available = self.rows[self.tick:self.tick + self.horizon]
        actions[:len(available)] = available
        self.tick += self.horizon
        self.streams[stream] = (self.tick, self.observed_start, self.rows)
        return {'actions': actions.astype(np.float32)}

    def info(self):
        return dict(name='sweep', horizon=self.horizon, tick=self.tick, total_ticks=len(self.rows))


def create():
    return SweepServerPolicy()
