#!/usr/bin/env python3
"""Plot shared sim/real gripper CSVs, one figure per episode."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from sim_to_real_so101.gr00t_client.grasp import dither_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out_dir', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.csv.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error('CSV contains no completed episode samples')
    output = args.out_dir or args.csv.parent
    output.mkdir(parents=True, exist_ok=True)
    for episode in dict.fromkeys(r['episode'] for r in rows):
        trace = [r for r in rows if r['episode'] == episode]
        times = [float(r['time_s']) for r in trace]
        keys = [k for k in trace[0] if k.startswith('arm_')]
        arm = [[float(r[k]) for k in keys] for r in trace]
        grasp = trace[0]['grasp_time_s']
        end = float(grasp) if grasp else times[-1]
        score = dither_score(times, arm, end)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, [float(r['predicted_gripper']) for r in trace], label='Predicted (raw)')
        ax.plot(times, [float(r['commanded_gripper']) for r in trace], label='Commanded')
        ax.plot(times, [float(r['measured_gripper']) if r['measured_gripper'] else float('nan')
                        for r in trace], label='Measured', alpha=.8)
        ax.axhline(float(trace[0]['empty_threshold']), linestyle=':', label='Empty-closure threshold')
        for field, label in [('close_below', 'Close threshold'), ('open_above', 'Release threshold')]:
            ax.axhline(float(trace[0][field]), linestyle='--', label=label)
        previous = None
        for row in trace:
            if row['chunk_id'] != previous:
                ax.axvline(float(row['time_s']), color='gray', alpha=.18, linewidth=.6)
                previous = row['chunk_id']
        ax.axvspan(max(times[0], end - 2), end, alpha=.10, color='orange', label='Dither window')
        if grasp:
            ax.axvline(end, color='green', label='Observed grasp')
        ax.set(xlabel='Episode time (s)', ylabel='Gripper (LeRobot units)',
               title=f'Episode {episode}: dither score {score} (arm direction reversals)')
        ax.legend()
        fig.tight_layout()
        path = output / f'{args.csv.stem}_ep{episode}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f'{path}: dither_score={score}; end={"grasp" if grasp else trace[0]["end_reason"]}')


if __name__ == '__main__':
    main()
