"""Measure matched target holds and build room-camera contact sheets, without a simulator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_log(path, width):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    times = np.asarray([r['seconds'] for r in rows], dtype=float)
    state = np.asarray([r['state'] for r in rows], dtype=float)
    target = np.asarray([r['target'] for r in rows], dtype=float)
    if (len(rows) < 2 or state.shape != (len(rows), width) or target.shape != state.shape
            or not all(np.isfinite(x).all() for x in (times, state, target)) or np.any(np.diff(times) <= 0)):
        raise ValueError(f'{path}: expected finite, ordered joint logs')
    return dict(times=times, state=state, target=target)


def align_hold(log, raw, *, cursor=0, hold_s=2.0, fps=30):
    """Find the next contiguous target interval; timestamps measure its duration.

    Log targets remain commanded until the next row. End times therefore use that
    next row (or one median frame at EOF), matching TrialVideo's wall-time origin.
    """
    times = log['times']
    dt = float(np.median(np.diff(times)))
    mask = np.max(np.abs(log['target'] - raw), axis=1) <= .01
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    for start in starts:
        if start < cursor:
            continue
        stop = int(start) + 1
        while stop < len(mask) and mask[stop] and times[stop] - times[stop - 1] <= max(3 * dt, 3 / fps):
            stop += 1
        end = times[stop] if stop < len(times) else times[-1] + dt
        if end - times[start] + 1e-6 < hold_s:
            continue
        selected = np.arange(start, stop)[times[start:stop] >= end - 1.0 - 1e-6]
        if len(selected) < 2:
            raise ValueError('Hold has insufficient samples in the final second')
        return dict(indices=selected, end_seconds=float(end), next_cursor=stop,
                    start_seconds=float(times[start]))
    raise ValueError(f'No target hold lasting {hold_s:g}s after row {cursor}: {np.asarray(raw).tolist()}')


def statistics(values, factors):
    def stats(array):
        return dict(median=np.median(array, axis=0).tolist(), max=np.max(array, axis=0).tolist(),
                    max_abs=np.max(np.abs(array), axis=0).tolist())
    return dict(raw=stats(values), degrees=stats(values * factors))


def analyze(sweep, run, *, hold_s=None, unit='rpd_follower', unit_path=None):
    if sweep['unit'] != unit:
        raise ValueError('Sweep unit does not match --unit')
    unit_path = unit_path or ROOT / 'arms' / sweep.get('arm', 'so101') / 'calibration' / 'units' / f'{unit}.json'
    payload = Path(unit_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sweep['unit_sha256']:
        raise ValueError('Calibration unit has changed since trajectory generation')
    calibration = json.loads(payload)
    names = sweep['joint_names']
    factors = np.asarray([(calibration['mapping'][n]['joint_max'] - calibration['mapping'][n]['joint_min'])
                          / (100 if n == 'gripper' else 200) for n in names])
    logs = {side: read_log(Path(run) / f'{side}.jsonl', len(names)) for side in ('real', 'sim')}
    cursor = dict(real=0, sim=0)
    results = []
    for hold in sweep['holds']:
        aligned = {}
        for side, log in logs.items():
            try:
                aligned[side] = align_hold(log, hold['raw'], cursor=cursor[side],
                                           hold_s=sweep['hold_s'] if hold_s is None else hold_s, fps=sweep['fps'])
            except ValueError as error:
                raise ValueError(f'{side} {hold["joint"]}:{hold["index"]}: {error}') from error
            cursor[side] = aligned[side]['next_cursor']
        states = {side: logs[side]['state'][a['indices']] for side, a in aligned.items()}
        # Pair samples by relative position in the final second of each hold,
        # independently of the two trials' wall times or inference latency.
        sim_offsets = logs['sim']['times'][aligned['sim']['indices']] - aligned['sim']['end_seconds']
        real_offsets = logs['real']['times'][aligned['real']['indices']] - aligned['real']['end_seconds']
        real_at_sim = np.column_stack([np.interp(sim_offsets, real_offsets, states['real'][:, j])
                                       for j in range(len(names))])
        result = dict(joint=hold['joint'], index=hold['index'], angle_deg=hold['angle_deg'],
                      target_raw=hold['raw'], alignment={side: {k: v for k, v in a.items() if k != 'indices'}
                                                       for side, a in aligned.items()},
                      real_target=statistics(states['real'] - hold['raw'], factors),
                      sim_target=statistics(states['sim'] - hold['raw'], factors),
                      sim_real=statistics(states['sim'] - real_at_sim, factors))
        results.append(result)
    summary = {}
    for j, name in enumerate(names):
        own = [r for r in results if r['joint'] == name]
        worst = max((r['sim_real']['degrees']['max_abs'][j] for r in results), default=None)
        medians = np.asarray([r['sim_real']['degrees']['median'][j] for r in own])
        angles = np.asarray([r['angle_deg'] for r in own])
        constant = bool(len(own) >= 3 and np.ptp(medians) <= .25)
        grows = False
        if len(own) >= 3 and np.ptp(angles) > 0 and not constant:
            ends = np.abs(angles - (angles.max() + angles.min()) / 2)
            slope = np.polyfit(ends, np.abs(medians), 1)[0]
            grows = bool(slope > 0 and slope * np.ptp(ends) > .25)
        summary[name] = dict(worst_abs_sim_real_deg=worst, constant_across_holds=constant,
                             grows_toward_range_ends=grows,
                             pattern='insufficient_holds' if len(own) < 3 else
                             'constant' if constant else 'grows_toward_range_ends' if grows else 'variable')
    return dict(unit=unit, unit_sha256=sweep['unit_sha256'], joint_names=names, holds=results,
                per_joint=summary, summary_method='Median gap variation <=0.25deg: constant; otherwise absolute-gap '
                'linear slope against distance from range centre must predict >0.25deg growth.')


def room_frames(path, timestamps):
    import av
    desired = sorted(enumerate(timestamps), key=lambda pair: pair[1])
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        pending = 0
        last = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * frame.time_base) - origin
            pixels = frame.to_ndarray(format='rgb24')[:, :640, :]
            while pending < len(desired) and seconds >= desired[pending][1]:
                index, stamp = desired[pending]
                # Nearest presentation timestamp, not nominal frame index.
                chosen = last[1] if last is not None and stamp - last[0] < seconds - stamp else pixels
                frames[index] = chosen.copy()
                pending += 1
            last = (seconds, pixels)
            if pending == len(desired):
                break
    if len(frames) != len(desired):
        raise ValueError(f'{path}: video ends before a requested hold frame')
    return [frames[i] for i in range(len(timestamps))]


def write_outputs(report, run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gap.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = []
    for hold in report['holds']:
        for j, name in enumerate(report['joint_names']):
            row = dict(hold_joint=hold['joint'], hold_index=hold['index'], angle_deg=hold['angle_deg'],
                       joint=name, target_raw=hold['target_raw'][j])
            for metric in ('real_target', 'sim_target', 'sim_real'):
                for units in ('raw', 'degrees'):
                    for stat, values in hold[metric][units].items():
                        row[f'{metric}_{stat}_{units}'] = values[j]
            rows.append(row)
    if rows:
        with (out / 'gap.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for j, name in enumerate(report['joint_names']):
        holds = [h for h in report['holds'] if h['joint'] == name]
        fig, ax = plt.subplots(figsize=(8, 4))
        for metric in ('real_target', 'sim_target', 'sim_real'):
            ax.plot([h['angle_deg'] for h in holds], [h[metric]['degrees']['median'][j] for h in holds],
                    'o-', label=metric.replace('_', ' - '))
        ax.axhline(0, color='gray', linewidth=.5)
        ax.set(xlabel='Hold angle (degrees)', ylabel='Median gap (degrees)', title=name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'gap_{name}.png', dpi=150)
        plt.close(fig)
        if not holds:
            continue
        images = {side: room_frames(Path(run) / f'{side}.mp4',
                                   [max(0, h['alignment'][side]['end_seconds'] - .2) for h in holds])
                  for side in ('real', 'sim')}
        fig, axes = plt.subplots(len(holds), 3, figsize=(15, 3.9 * len(holds)), squeeze=False)
        for i, h in enumerate(holds):
            real, sim = images['real'][i], images['sim'][i]
            blend = ((real.astype(float) + sim.astype(float)) / 2).astype(np.uint8)
            for col, (label, pixels) in enumerate(zip(('real', 'sim', '50% blend'), (real, sim, blend))):
                axes[i, col].imshow(pixels)
                axes[i, col].set_title(f'{name}:{h["index"]} {h["angle_deg"]:.2f}deg; '
                                       f'raw={h["target_raw"][j]:.2f} | {label}', fontsize=9)
                axes[i, col].axis('off')
        fig.tight_layout()
        fig.savefig(out / f'contact_sheet_{name}.png', dpi=100)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--unit', default='rpd_follower')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hold-s', type=float, default=None, help='Minimum matched hold duration; default: sweep hold_s')
    args = parser.parse_args()
    if args.hold_s is not None and args.hold_s < 1:
        parser.error('--hold-s must be at least one second')
    report = analyze(json.loads(args.sweep.read_text()), args.run, hold_s=args.hold_s, unit=args.unit)
    write_outputs(report, args.run, args.out)
    print(json.dumps(report['per_joint'], indent=2))


if __name__ == '__main__':
    main()
