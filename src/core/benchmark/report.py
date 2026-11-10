"""Task-independent episode, benchmark and spatial reports."""
from datetime import datetime, timezone
import math
from statistics import median
import numpy as np
from .stats import wilson_interval

def select_instruction(color, default, by_color=None):
    if by_color is None:
        return default
    if not isinstance(by_color, dict) or color not in by_color or not isinstance(by_color[color], str) or not by_color[color].strip():
        raise ValueError(f'--lang_instruction_by_color requires a non-empty instruction for {color!r}')
    return by_color[color]


def results_report(episodes, task, checkpoint, seed, *, random_fraction=0.0,
                   episode_length_s=None, action_horizon=16, step_dt=None):
    """Add simulated success timing without changing existing schema-1 fields.

    Success steps count control transitions from the episode reset, including
    settling steps. Median time includes successful episodes only; the 15-second
    rate uses every completed episode as its denominator. Without step_dt, old
    callers retain their rates and receive null timing metrics.
    """
    def summarize(rows):
        successes = sum(bool(row['success']) for row in rows)
        success_times = ([row['success_step'] * step_dt for row in rows if row['success']]
                         if step_dt is not None else None)
        within_15 = (sum(time <= 15 or math.isclose(time, 15, rel_tol=0, abs_tol=1e-12)
                         for time in success_times) if success_times is not None else None)
        return {'episodes': len(rows), 'successes': successes,
                'success_rate': successes / len(rows) if rows else None,
                'successes_within_15s': within_15,
                'success_rate_within_15s': within_15 / len(rows) if rows and within_15 is not None else None,
                'median_time_to_success_s': median(success_times) if success_times else None}

    def grouped(key):
        return {value: summarize([row for row in episodes if row.get(key) == value])
                for value in sorted({row[key] for row in episodes if key in row})}

    return {'schema_version': 1, 'task': task, 'checkpoint': checkpoint, 'seed': seed,
            'random_fraction': random_fraction, 'episode_length_s': episode_length_s,
            'action_horizon': action_horizon, 'step_dt': step_dt,
            'time': datetime.now(timezone.utc).isoformat(), 'overall': summarize(episodes),
            'by_start': {kind: summarize([row for row in episodes if row.get('start_kind') == kind])
                         for kind in ('recorded', 'random')},
            'by_zone': grouped('zone'), 'by_cube_color': grouped('cube_color'), 'episodes': episodes}


def summarize(rows, step_dt, *, stages=None, failures=None):
    stages = tuple(stages) if stages is not None else tuple(dict.fromkeys(
        stage for row in rows for stage in row.get('stages', {})))
    failures = tuple(failures) if failures is not None else tuple(sorted({
        row['failure_mode'] for row in rows if row.get('failure_mode') is not None}))
    n = len(rows)
    successes = sum(r['success'] for r in rows)
    times = [r['success_step'] * step_dt for r in rows if r['success']]
    return dict(episodes=n, successes=successes, success_rate=successes / n if n else None,
                success_rate_ci95=wilson_interval(successes, n),
                stage_rates={s: sum(r['stages'].get(s, False) for r in rows) / n if n else None for s in stages},
                failure_modes={f: sum(r.get('failure_mode') == f for r in rows) for f in failures},
                median_time_to_success_s=median(times) if times else None)


def benchmark_report(episodes, eval_set, repeats, *, stages=None, failures=None, **kwargs):
    report = results_report(episodes, **kwargs)
    report.update(schema_version=2, eval_set_id=eval_set['eval_set_id'], eval_set=eval_set, repeats=repeats,
                  ci_method='95% Wilson, episode-level; repeats are not independent starts',
                  overall=summarize(episodes, kwargs['step_dt'], stages=stages, failures=failures))
    if eval_set.get('legacy_eval_set_id'):
        report['legacy_eval_set_id'] = eval_set['legacy_eval_set_id']
    for key in ('split', 'zone'):
        report['by_' + key] = {name: summarize([r for r in episodes if r[key] == name], kwargs['step_dt'],
                                             stages=stages, failures=failures)
                               for name in sorted({r[key] for r in episodes})}
    report['per_start_consistency'] = {
        name: dict(successes=sum(r['success'] for r in episodes if r['start_id'] == name),
                   repeats=sum(r['start_id'] == name for r in episodes),
                   success_fraction=float(np.mean([r['success'] for r in episodes if r['start_id'] == name])))
        for name in sorted({r['start_id'] for r in episodes})}
    return report


def save_heatmap(report, path):
    """Plot the primary sampled object and named static footprints in the base frame."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from .eval_set import normalize_scene, start_objects
    from .starts import primary_object, task_instance

    data = report['eval_set']
    task = task_instance(data.get('task'))
    primary = primary_object(task)
    scene = normalize_scene(data['scene'], task)
    fig, (ax, overview) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
    outlines = [('Table', scene.get('table_outline_base_xy', []))]
    outlines += [(name, values.get('footprint_base_xy', [])) for name, values in scene['objects'].items()
                 if name != primary]
    for label, points in outlines:
        polygon = np.asarray(points)
        if polygon.ndim != 2 or not len(polygon):
            continue
        polygon = np.vstack([polygon, polygon[0]])
        for target in (ax, overview):
            target.plot(*polygon.T, label=label, color='tan' if label == 'Table' else 'black')

    def xy(start):
        return start_objects(start, task=task)[primary]['position'][:2]

    train = np.array([xy(start) for start in data['starts'] if start['split'] == 'train'])
    if len(train):
        ax.scatter(*train.T, c='gray', s=15, label='Training starts', alpha=.5)
        overview.scatter(*train.T, c='gray', s=5)
    overview.set(title='Full tabletop', aspect='equal', xlabel='Base X (m)', ylabel='Base Y (m)')
    cmap = LinearSegmentedColormap.from_list('success', ['#c43c39', '#eac75a', '#22965c'])
    for split, marker in zip(('id', 'ood', 'yaw', 'train'), ('o', '^', 's', 'D')):
        starts = [start for start in data['starts']
                  if start['split'] == split and start['id'] in report['per_start_consistency']]
        if starts:
            points = np.array([xy(start) for start in starts])
            ax.scatter(*points.T, c=[report['per_start_consistency'][start['id']]['success_fraction'] for start in starts],
                       cmap=cmap, vmin=0, vmax=1, marker=marker,
                       s={'id': 110, 'yaw': 25, 'ood': 65, 'train': 30}[split],
                       edgecolors='black', linewidths=.4, alpha=.8, label=split)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, label='Success fraction', shrink=.7)
    region = np.array([xy(start) for start in data['starts']])
    low, high = region.min(0), region.max(0)
    pad = np.maximum(high - low, .01) * .1
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set(xlabel='Base-link X (m)', ylabel='Base-link Y (m)',
           title=task.name.replace('_', ' ') + ' fixed-start benchmark', aspect='equal')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
