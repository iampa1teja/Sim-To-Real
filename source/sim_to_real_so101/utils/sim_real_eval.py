"""Paired sim/real evaluation: trial records, the per-trial state machine and the gap report. CPU only."""
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import os
from pathlib import Path
import threading

from .pick_place_benchmark import STAGES, mcnemar

STAGE_ORDER = ('none',) + STAGES
REPORT_SCHEMA = 'so101_sim_real_eval_v1'


def stage_rank(stage):
    """Index of a stage in STAGE_ORDER. Raises ValueError for an unknown stage."""
    if stage not in STAGE_ORDER:
        raise ValueError(f'unknown stage {stage!r}; expected one of {STAGE_ORDER}')
    return STAGE_ORDER.index(stage)


def furthest_stage(stages):
    """Last name in STAGES whose value is truthy, else 'none'."""
    reached = 'none'
    for name in STAGES:
        if stages.get(name):
            reached = name
    return reached


@dataclass
class Trial:
    trial_id: int
    start_id: str
    split: str
    x: float
    y: float
    yaw: float
    sim: dict
    real: dict
    created: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls)})


def append_trial(path, trial):
    """Append one trial as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(trial.to_dict()) + '\n')


def load_trials(path):
    """Trials from a JSON-lines file. Missing file gives []; for a repeated trial_id the last line wins."""
    path = Path(path)
    if not path.exists():
        return []
    by_id = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = Trial.from_dict(json.loads(line))
            by_id[trial.trial_id] = trial
    return list(by_id.values())


def summarize_trials(trials):
    """Sim/real stage rates, gap, stage confusion and paired success over fully labelled trials."""
    paired = [t for t in trials if t.sim is not None and t.real is not None and 'stage' in t.real]
    n = len(paired)
    sim_ranks = [stage_rank(t.sim['stage']) for t in paired]
    real_ranks = [stage_rank(t.real['stage']) for t in paired]

    stage_rates = {}
    for stage in STAGES:
        rank = stage_rank(stage)
        sim_rate = sum(r >= rank for r in sim_ranks) / n if n else None
        real_rate = sum(r >= rank for r in real_ranks) / n if n else None
        gap = sim_rate - real_rate if n else None
        stage_rates[stage] = {'sim': sim_rate, 'real': real_rate, 'gap': gap}

    confusion = {s: {r: 0 for r in STAGE_ORDER} for s in STAGE_ORDER}
    for t in paired:
        confusion[t.sim['stage']][t.real['stage']] += 1

    sim_bools = [t.sim['stage'] == 'success' for t in paired]
    real_bools = [t.real['stage'] == 'success' for t in paired]
    paired_success = {
        'both': sum(a and b for a, b in zip(sim_bools, real_bools)),
        'sim_only': sum(a and not b for a, b in zip(sim_bools, real_bools)),
        'real_only': sum(b and not a for a, b in zip(sim_bools, real_bools)),
        'neither': sum(not a and not b for a, b in zip(sim_bools, real_bools)),
        'mcnemar': mcnemar(sim_bools, real_bools),
    }
    return {'n': n, 'stage_rates': stage_rates, 'confusion': confusion, 'paired_success': paired_success}


def write_trials_report(path, header, trials):
    """Atomically write the JSON gap report (temp file, then os.replace)."""
    payload = {'schema': REPORT_SCHEMA, 'header': header,
               'trials': [t.to_dict() for t in trials], 'summary': summarize_trials(trials)}
    path = os.fspath(path)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


class SimRealSession:
    """One paired trial at a time: spawn, mark placed, run sim, run real, label, save or discard.

    Runners are called without the lock held; the state names a run in progress so a second run_* raises.
    """

    def __init__(self, spawner, sim_runner, real_runner, out_dir, header=None, trials=None, now=datetime.now):
        self._spawner = spawner
        self._sim_runner = sim_runner
        self._real_runner = real_runner
        self._now = now
        self._lock = threading.Lock()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.header = dict(header or {})
        self.trials = list(trials or [])
        self.state = 'idle'
        self.real_placed = False
        self.start = None
        self.sim = None
        self.real = None
        self.label = None

    def _next_id(self):
        return max((t.trial_id for t in self.trials), default=-1) + 1

    def _is_running(self):
        return self.state in ('sim_running', 'real_running')

    def _clear(self):
        self.real_placed = False
        self.start = None
        self.sim = None
        self.real = None
        self.label = None

    def spawn(self, start):
        with self._lock:
            if self._is_running():
                raise RuntimeError(f'spawn needs no run in progress (state={self.state})')
            self._spawner(start)
            self._clear()
            self.start = dict(start)
            self.state = 'spawned'

    def mark_placed(self):
        with self._lock:
            if self.start is None or self._is_running():
                raise RuntimeError(f'mark_placed needs a spawned start and no run in progress '
                                   f'(state={self.state}, start_set={self.start is not None})')
            self.real_placed = True

    def run_sim(self):
        with self._lock:
            if self.state not in ('spawned', 'sim_done', 'real_done'):
                raise RuntimeError(f'run_sim needs a spawned start or a finished sim run (state={self.state})')
            previous, start = self.state, dict(self.start)
            self.state = 'sim_running'
        try:
            result = self._sim_runner(start)
        except BaseException:
            with self._lock:
                self.state = previous
            raise
        with self._lock:
            self.sim = dict(result)
            self.real = None
            self.label = None
            self.state = 'sim_done'

    def run_real(self, should_stop):
        with self._lock:
            if self.state not in ('sim_done', 'real_done') or not self.real_placed:
                raise RuntimeError('run_real needs a finished sim run and the real cube marked placed '
                                   f'(state={self.state}, placed={self.real_placed})')
            start = dict(self.start)
            self.state = 'real_running'
        try:
            result = self._real_runner(start, should_stop)
        except BaseException:
            with self._lock:
                self.state = 'sim_done'
            raise
        with self._lock:
            self.real = dict(result)
            self.label = None
            self.state = 'real_done'

    def label_real(self, stage, note=''):
        with self._lock:
            if self.state != 'real_done':
                raise RuntimeError(f'label_real needs a finished real run (state={self.state})')
            if stage not in STAGE_ORDER:
                raise ValueError(f'label_real got unknown stage {stage!r}; expected one of {STAGE_ORDER}')
            note = str(note)
            self.label = {'stage': stage, 'note': note}
            self.real = {**self.real, 'stage': stage, 'note': note}

    def save(self):
        with self._lock:
            if self.state != 'real_done' or self.label is None:
                raise RuntimeError(f'save needs a labelled real run (state={self.state}, '
                                   f'labelled={self.label is not None})')
            trial = Trial(trial_id=self._next_id(), start_id=str(self.start['start_id']),
                          split=str(self.start['split']), x=float(self.start['x']), y=float(self.start['y']),
                          yaw=float(self.start['yaw']), sim=dict(self.sim), real=dict(self.real),
                          created=self._now().isoformat(timespec='seconds'))
            append_trial(self.out_dir / 'trials.jsonl', trial)
            self.trials.append(trial)
            write_trials_report(self.out_dir / 'report.json', self.header, self.trials)
            self._clear()
            self.state = 'idle'
            return trial

    def discard(self):
        with self._lock:
            if self._is_running():
                raise RuntimeError(f'discard needs no run in progress (state={self.state})')
            self._clear()
            self.state = 'idle'

    def snapshot(self):
        with self._lock:
            return {
                'state': self.state,
                'real_placed': self.real_placed,
                'start': dict(self.start) if self.start is not None else None,
                'sim': dict(self.sim) if self.sim is not None else None,
                'real': dict(self.real) if self.real is not None else None,
                'label': dict(self.label) if self.label is not None else None,
                'trial_count': len(self.trials),
                'next_trial_id': self._next_id(),
            }

    def summary(self):
        with self._lock:
            return summarize_trials(list(self.trials))
