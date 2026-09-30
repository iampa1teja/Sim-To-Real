#!/usr/bin/env python3
"""Compare schema-2 results on a common fixed benchmark.

McNemar uses one repeat (default 0), not correlated pseudo-replicates. Pass
--repeat to compare another paired repeat. All-repeat rates remain descriptive.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from sim_to_real_so101.utils.eval_comparison import compare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args(argv)
    reports = [json.loads(p.read_text()) for p in args.results]
    pairs = compare(reports, args.repeat)
    print('Checkpoint | Split | Successes/episodes | Success rate [95% Wilson CI]')
    for r in reports:
        for split, value in r['by_split'].items():
            lo, hi = value['success_rate_ci95']
            print(f"{r['checkpoint']} | {split} | {value['successes']}/{value['episodes']} | "
                  f"{value['success_rate']:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f'\nExact paired McNemar (repeat {args.repeat}; unadjusted p-values):')
    for pair in pairs:
        print(f"{reports[pair['first']]['checkpoint']} vs {reports[pair['second']]['checkpoint']} "
              f"| {pair['split']} | n={pair['shared_starts']} | discordant={pair['first_only']}/{pair['second_only']} "
              f"| p={pair['p_value']:.6g}")


if __name__ == '__main__':
    main()
