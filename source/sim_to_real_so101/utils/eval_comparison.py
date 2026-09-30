"""Shared validation and paired McNemar comparisons for fixed-benchmark reports."""
from itertools import combinations
from .pick_place_benchmark import mcnemar, validate_eval_set


def compare(reports, repeat=0, *, varying_action_horizon=False):
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
    keys = ['task', 'episode_length_s', 'robot_start', 'seed']
    keys.append('checkpoint' if varying_action_horizon else 'action_horizon')
    for key in keys:
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
