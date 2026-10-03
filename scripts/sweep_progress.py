#!/usr/bin/env python3
"""Live tqdm bars for a parallel checkpoint sweep (scripts/sweep_checkpoints.sh).

One bar per checkpoint log in LOGS (datasets/hyperparameters/parallel_logs/<run>/), showing episodes
done / total and successes so far. Attach to a running sweep at any time:
    python3 scripts/sweep_progress.py                      # newest parallel_logs folder
    python3 scripts/sweep_progress.py <logs dir> [steps...]
Exits when every listed checkpoint has finished, or when <logs dir>/.done appears.
"""
import os
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BAR = re.compile(r'AH \d+ \(\d+/\d+\):\s+\d+%\|.*?\|\s*(\d+)/(\d+)')
END = re.compile(r'AH \d+: (complete|failed)|skipping complete')


def state(log):
    """(done, total, successes, status) parsed from one sweep log and its run.log."""
    text = log.read_text(errors='replace').replace('\r', '\n') if log.exists() else ''
    if not text:
        return 0, None, 0, 'queued'
    bars = BAR.findall(text)
    done, total = (int(bars[-1][0]), int(bars[-1][1])) if bars else (0, None)
    folder = re.findall(r'Sweep folder: (\S+)', text)
    successes = 0
    if folder:
        for run in Path(folder[-1]).glob('ah_*/run.log'):
            successes += len(re.findall(r'\[EPISODE \d+\][^\n]* success ', run.read_text(errors='replace')))
    end = END.findall(text)
    status = end[-1] or 'complete' if end else ('failed' if 'Traceback' in text and not bars else 'running')
    return done, total, successes, status


def main(argv):
    if argv:
        logs = Path(argv[0])
    else:
        runs = sorted((ROOT/'datasets/hyperparameters/parallel_logs').glob('*/'), key=lambda p: p.stat().st_mtime)
        if not runs:
            sys.exit('No parallel_logs folder yet')
        logs = runs[-1]
    steps = argv[1:] or sorted((p.stem.split('-')[-1] for p in logs.glob('checkpoint-*.log')), key=int, reverse=True)
    if not steps:
        sys.exit(f'No checkpoint logs in {logs}')
    print(f'Logs: {logs}')
    parent = os.getppid() if os.environ.get('SWEEP_DRIVER') else None
    bars = {s: tqdm(total=200, desc=f'ckpt-{s:>6}', unit='ep', position=i, dynamic_ncols=True, leave=True,
                    bar_format='{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] {postfix}')
            for i, s in enumerate(steps)}
    try:
        while True:
            finished = 0
            for s, bar in bars.items():
                done, total, successes, status = state(logs/f'checkpoint-{s}.log')
                if total and bar.total != total:
                    bar.total = total
                if done > bar.n:
                    bar.update(done-bar.n)
                rate = f'{successes}/{done} = {successes/done:.0%}' if done else '-'
                bar.set_postfix_str(f'{status}, success {rate}', refresh=True)
                finished += status in ('complete', 'failed')
            # Also stop if the logs folder vanished or the launching driver exited (we were orphaned).
            if finished == len(bars) or (logs/'.done').exists() or not logs.is_dir() or (parent and os.getppid() != parent):
                break
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in bars.values():
            bar.close()


if __name__ == '__main__':
    main(sys.argv[1:])
