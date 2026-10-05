#!/usr/bin/env python3
"""Read-only, CPU grasp audit. Writes reports only, never a training dataset."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from check_pick_place_demo_alignment import SETUP_PATH, cube_face_yaw, eval_utils, square_angle

FORCED_BAD = {0, 6, 3, 51, 72}
METRICS = ('hover_s', 'close_attempts', 'dither', 'close_duration_frames', 'roll_alignment_deg')


def episode_metrics(actions, joints_deg, sidecar, closed, opened, speed_threshold, fps=30, min_lift=.01):
    """joints_deg is calibrated state; action gripper stays in training units."""
    actions, joints_deg = np.asarray(actions), np.asarray(joints_deg)
    n = len(actions)
    g = (actions[:, 5] - closed) / (opened - closed)
    below = g < .5
    crossings = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    sustained = [int(i) for i in crossings if i + 10 <= n and below[i:i + 10].all()]
    close = sustained[0] if sustained else None
    z = np.asarray(sidecar['position'])[:, 2]
    lifts = np.flatnonzero((z - z[0] >= min_lift) & (np.arange(n) > (close if close is not None else n)))
    lift = int(lifts[0]) if len(lifts) else None
    end = lift if lift is not None else n
    # Every below-midpoint -> at/above-midpoint transition counts, even brief attempts.
    reopens = np.flatnonzero(below[:-1] & ~below[1:]) + 1
    attempts = sum(int(np.any((reopens > i) & (reopens < end))) for i in crossings if i < end)
    row = dict(frames=n, gripper_max=float(actions[:, 5].max()),
               reaches_90pct_open=bool(np.any(g >= .9)),
               cube_rise_anywhere_m=float(np.max(z - z[0])), close_frame=close, lift_frame=lift, lift_ok=lift is not None,
               close_attempts=attempts, hover_s=None, dither=None,
               close_duration_frames=None, roll_alignment_deg=None)
    if close is None:
        return row
    velocity = np.diff(joints_deg[:, :5], axis=0) * fps
    row['hover_s'] = float(np.sum(np.linalg.norm(velocity[:close], axis=1) < speed_threshold) / fps)
    # Raw sign changes per joint, ignoring exact stationary samples; no hidden smoothing.
    changes = velocity[max(0, close - 60):close]
    row['dither'] = sum(int(np.sum(s[1:] != s[:-1])) for j in range(5)
                        for s in [np.sign(changes[:, j][changes[:, j] != 0])])
    starts = np.flatnonzero(g[:close] >= .9)
    if len(starts):
        start = int(starts[-1])
        finishes = np.flatnonzero(g[close:] <= .1) + close
        reopens = np.flatnonzero(g[close:] >= .9) + close
        if len(finishes) and (not len(reopens) or finishes[0] < reopens[0]):
            row['close_duration_frames'] = int(finishes[0] - start)
    yaw = cube_face_yaw(sidecar['orientation_wxyz'][close])
    row['roll_alignment_deg'] = abs(float(square_angle(joints_deg[close, 4] - square_angle(yaw - joints_deg[close, 0]))))
    return row


def classify(rows):
    # Hard failures do not set the reference population's behavioural percentiles.
    reference = [r for r in rows if r['episode'] not in FORCED_BAD and r['lift_ok'] and r['close_frame'] is not None]
    if not reference:
        raise ValueError('No lifted episodes with a sustained close for percentile thresholds')
    thresholds = {k: float(np.percentile([r[k] for r in reference if r[k] is not None], 75))
                  for k in METRICS if any(r[k] is not None for r in reference)}
    for row in rows:
        bad = []
        if row['episode'] in FORCED_BAD:
            bad.append('mandated exclusion')
        if row['close_frame'] is None:
            bad.append('no sustained midpoint crossing')
        if not row['lift_ok']:
            bad.append('no post-close lift')
        hesitant = [f'{k}>P75' for k, threshold in thresholds.items()
                    if row[k] is not None and row[k] > threshold]
        if row['close_frame'] is not None and row['reaches_90pct_open'] and row['close_duration_frames'] is None:
            hesitant.append('incomplete 90%-10% transition')
        row['classification'] = 'bad' if bad else 'hesitant' if hesitant else 'clean'
        row['reasons'] = '; '.join(bad or hesitant)
    return thresholds


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def read_dataset(root):
    info = json.loads((root / 'meta/info.json').read_text())
    if info['fps'] != 30:
        raise ValueError('Audit requires 30 fps')
    for key in ('action', 'observation.state'):
        if info['features'][key]['names'] != [j + '.pos' for j in eval_utils.SO101_JOINTS]:
            raise ValueError('Unexpected joint ordering')
    frames = {}
    for path in sorted(root.glob('data/*/*.parquet')):
        for row in pq.read_table(path).to_pylist():
            key = (int(row['episode_index']), int(row['frame_index']))
            if key in frames:
                raise ValueError(f'Duplicate frame {key}')
            frames[key] = row
    mapping = json.loads(SETUP_PATH.read_text()).get('sim_joint_mapping', {})
    episodes = []
    for ep in range(info['total_episodes']):
        sidecar = json.loads((root / f'pick_place_meta/episode_{ep:06d}.json').read_text())
        n = sidecar['num_frames']
        if (sidecar['episode_index'] != ep or sidecar['fps'] != 30 or n < 2
                or sidecar['reference_frame'] != 'robot base link (base)' or sidecar['units'] != 'm'):
            raise ValueError(f'Invalid sidecar {ep}')
        rows = [frames.pop((ep, i)) for i in range(n)]
        np.testing.assert_allclose([r['timestamp'] for r in rows], np.arange(n) / 30, atol=1e-5)
        actions = np.asarray([r['action'] for r in rows])
        states = np.asarray([r['observation.state'] for r in rows])
        positions = np.asarray(sidecar['position'])
        quats = np.asarray(sidecar['orientation_wxyz'])
        if (actions.shape != (n, 6) or states.shape != (n, 6) or positions.shape != (n, 3)
                or quats.shape != (n, 4) or not all(np.isfinite(a).all() for a in (actions, states, positions, quats))
                or np.any(np.linalg.norm(quats, axis=1) == 0)):
            raise ValueError(f'Invalid arrays in episode {ep}')
        joints = np.degrees([eval_utils.dataset_state_to_radians(s, mapping) for s in states])
        episodes.append((ep, actions, joints, sidecar))
    if frames or sum(len(e[1]) for e in episodes) != info['total_frames']:
        raise ValueError('Dataset totals disagree')
    if len(list((root / 'pick_place_meta').glob('episode_*.json'))) != len(episodes):
        raise ValueError('Extra sidecars')
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/pick_place_v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    out = (args.output or Path('datasets/analysis/grasp_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if out == root or root in out.parents or out in root.parents or out.exists():
        parser.error('Output must be new and separate from source')
    before = snapshot(root)
    episodes = read_dataset(root)
    stats = json.loads((root / 'meta/stats.json').read_text())['action']
    closed, opened = float(stats['min'][5]), float(stats['max'][5])
    actual = np.concatenate([e[1][:, 5] for e in episodes])
    if not np.isfinite([closed, opened]).all() or closed >= opened:
        raise ValueError('Invalid training gripper extrema')
    np.testing.assert_allclose([closed, opened], [actual.min(), actual.max()], atol=1e-5)
    speed = np.concatenate([np.linalg.norm(np.diff(e[2][:, :5], axis=0) * 30, axis=1) for e in episodes])
    threshold = float(np.percentile(speed, 25))
    rows = [dict(episode=ep, **episode_metrics(a, j, s, closed, opened, threshold)) for ep, a, j, s in episodes]
    thresholds = classify(rows)
    if before != snapshot(root):
        raise RuntimeError('Source changed during audit')
    out.mkdir(parents=True)
    with (out / 'audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'source_sha256.json').write_text(json.dumps(before, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, (*METRICS, 'close_frame')):
        ax.hist([r[key] for r in rows if r[key] is not None], bins=15)
        ax.set_title(key)
        ax.set_ylabel('Episodes')
        if key in thresholds:
            ax.axvline(thresholds[key], color='red', label='P75')
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'distributions.png', dpi=150)
    plt.close(fig)
    lines = ['# Grasp audit — STEP 1 review', '', f'Source: `{root}`. {len(rows)} episodes; {sum(r["frames"] for r in rows)} frames; 30 fps.',
             f'Gripper training closed/open: {closed:.8g} / {opened:.8g}; midpoint {(closed + opened)/2:.8g}.',
             f'Arm-speed P25: {threshold:.6g} degrees/s (five-joint Euclidean norm, state differences).', '',
             'Definitions: close is the first downward midpoint crossing with 10 consecutive closed frames; initially closed is not a crossing. Hover sums all low-speed intervals from episode start to close (includes initial settling, not just time near the cube). Dither counts per-joint nonzero velocity sign reversals in the preceding 60 intervals; exact zeros are ignored, with no smoothing. Attempts count downward-crossing→reopen cycles before lift, or episode end if no lift; initial opening is excluded. Duration runs from the last ≥90%-open sample before close to the first ≤10%-open sample before reopening to 90%; durations are missing when those boundaries are not observed; absence of a 90%-open sample is unmeasurable, not evidence of hesitation.', '',
             'Alignment is abs(fold90(wrist_roll − fold90(cube_face_yaw − shoulder_pan))) using the existing calibrated joint mapping and base-frame cube quaternion. This is the requested joint-angle proxy, not gripper forward kinematics. Lift is ≥0.01 m base-Z rise above the initial cube position after close; it does not establish contact or task success.', '',
             'Classification: bad = mandated IDs 0,6,3,51,72, missing sustained close, or missing post-close lift. Otherwise hesitant = any metric strictly above its P75, or an incomplete 90%-10% transition after reaching 90% open. Otherwise clean. P75 is computed on lifted, sustained-close episodes excluding mandated bad IDs: a conservative upper-quartile review screen, not evidence that all flagged episodes should be deleted. No percentile-only bad exclusions.', '',
             '| Metric | N | P0 | P25 | P50 | P75 | P95 | P100 | Hesitant threshold |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in (*METRICS, 'close_frame'):
        vals = [r[key] for r in rows if r[key] is not None]
        quantiles = np.percentile(vals, [0, 25, 50, 75, 95, 100]) if vals else [float('nan')] * 6
        lines.append(f'| {key} | {len(vals)} | ' + ' | '.join(f'{v:.4g}' for v in quantiles) + f' | {thresholds.get(key, "—")} |')
    for category in ('bad', 'hesitant', 'clean'):
        selected = [r for r in rows if r['classification'] == category]
        lines += ['', f'{category.capitalize()}: {len(selected)} episodes / {sum(r["frames"] for r in selected)} frames.', ','.join(str(r['episode']) for r in selected)]
    lines += ['', f'Lift detected: {sum(r["lift_ok"] for r in rows)}/{len(rows)}.', '', '**Threshold limitation:** the global training maximum sets 90% open at ' + f'{closed + .9 * (opened - closed):.5g}. Only {sum(r["reaches_90pct_open"] for r in rows)} episodes reach it; duration is measurable in {sum(r["close_duration_frames"] is not None for r in rows)}. ' + f'{sum(r["gripper_max"] < (closed + opened)/2 for r in rows)} episodes never reach the midpoint, whereas {sum(r["cube_rise_anywhere_m"] >= .01 for r in rows)} show a cube rise somewhere. Missing close/lift metrics under these thresholds are not proof of a failed physical grasp. Treat non-mandated bad IDs as provisional review exclusions; do not blindly remove them. Approve a revised threshold definition if desired before building data.', '', '![Metric distributions](distributions.png)', '',
              'Source SHA-256 snapshot matched before/after (all source files, including videos). No training data built. STOP: approval required before STEP 2.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
