#!/usr/bin/env python3
"""Compare schema-2 results on a common fixed benchmark.

McNemar uses one repeat (default 0), not correlated pseudo-replicates. Pass
--repeat to compare another paired repeat. All-repeat rates remain descriptive.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from sim_to_real_so101.utils.pick_place_benchmark import mcnemar, validate_eval_set


def compare(reports, repeat=0):
    if len(reports) < 2:
        raise ValueError('Provide at least two results JSONs')
    for report in reports:
        if report.get('schema_version') != 2 or not report.get('complete'):
            raise ValueError('Comparison requires complete schema-2 benchmark results')
        validate_eval_set(report['eval_set'])
        if report['eval_set_id'] != report['eval_set']['eval_set_id']:
            raise ValueError('Result eval_set_id mismatch')
    if len({r['eval_set_id'] for r in reports}) != 1:
        raise ValueError('Results must use the same eval set')
    for key in ('task', 'episode_length_s', 'action_horizon', 'robot_start', 'seed'):
        if len({r.get(key) for r in reports}) != 1:
            raise ValueError(f'Incomparable run settings: {key}')
    pairs = []
    for (i, first), (j, second) in combinations(enumerate(reports), 2):
        def indexed(report):
            rows = [r for r in report['episodes'] if r['repeat'] == repeat]
            result = {r['start_id']: r for r in rows}
            if len(result) != len(rows):
                raise ValueError('Duplicate start/repeat rows')
            return result
        a, b = indexed(first), indexed(second)
        shared = sorted(a.keys() & b.keys())
        if not shared:
            raise ValueError(f'No shared starts for repeat {repeat}')
        if any(a[k]['policy_seed'] != b[k]['policy_seed'] for k in shared):
            raise ValueError('Paired starts have different policy seeds')
        for split in ['all', *sorted({a[k]['split'] for k in shared})]:
            keys = [k for k in shared if split == 'all' or a[k]['split'] == split]
            pairs.append(dict(first=i, second=j, split=split, repeat=repeat,
                              **mcnemar([a[k]['success'] for k in keys], [b[k]['success'] for k in keys])))
    return pairs


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
