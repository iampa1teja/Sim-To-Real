"""Task-independent episode, benchmark and spatial reports."""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median
import sys

import numpy as np

from .eval_set import validate_eval_set
from .resume import write_report


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


def wilson_interval(successes, episodes):
    if not 0 <= successes <= episodes:
        raise ValueError('Require 0 <= successes <= episodes')
    if episodes == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / episodes
    denom = 1 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denom
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denom
    return [max(0., centre - half), min(1., centre + half)]


def mcnemar(first, second):
    """Exact two-sided paired binomial McNemar, one Boolean per shared start."""
    if len(first) != len(second):
        raise ValueError('Paired observations must have equal length')
    b = sum(bool(a) and not bool(c) for a, c in zip(first, second))
    c = sum(not bool(a) and bool(d) for a, d in zip(first, second))
    n = b + c
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)) if n else 1.
    return dict(shared_starts=len(first), first_only=b, second_only=c, discordant=n,
                p_value=p, method='exact two-sided McNemar')


def eval_set_identities(report):
    data = report.get('eval_set', report)
    return {value for value in (data.get('eval_set_id'), data.get('legacy_eval_set_id')) if value}


def compare(reports, repeat=0, *, varying_action_horizon=False, task=None):
    if len(reports) < 2:
        raise ValueError('Provide at least two results JSONs')
    for report in reports:
        if report.get('schema_version') != 2 or not report.get('complete'):
            raise ValueError('Comparison requires complete schema-2 benchmark results')
        validate_eval_set(report['eval_set'], task=task)
        if report['eval_set_id'] != report['eval_set']['eval_set_id']:
            raise ValueError('Result eval_set_id mismatch')
    if not set.intersection(*(eval_set_identities(report) for report in reports)):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--task', default=None, help='Task folder whose benchmark to use (default: S2R_TASK)')
    args = parser.parse_args(argv)
    reports = [json.loads(p.read_text()) for p in args.results]
    pairs = compare(reports, args.repeat, task=args.task)
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


def build_report(context, state):
    report_fn = results_report
    report_extra = {}
    if context.benchmark:
        report_fn = benchmark_report
        report_extra = dict(
            eval_set=context.benchmark.data, repeats=context.args_cli.repeats, stages=context.task.stages
        )
    report = report_fn(
        state.episodes,
        task=context.env_id,
        checkpoint=context.args_cli.checkpoint,
        seed=context.args_cli.seed,
        **report_extra,
        random_fraction=context.args_cli.random_fraction,
        episode_length_s=context.episode_length_s,
        action_horizon=context.args_cli.action_horizon,
        step_dt=context.step_dt,
    )
    report["robot_start"] = context.args_cli.robot_start
    report["action_repeat"] = context.args_cli.action_repeat
    report["replan_interval_s"] = context.replan_interval_s
    report["robot_usd"] = context.env_cfg.scene.robot.spawn.usd_path
    report["selection"] = dict(
        arm=context.args_cli.arm,
        scene=context.args_cli.scene,
        task=context.args_cli.task,
        variant=context.args_cli.variant,
    )
    if state.video_recorder is not None or (context.resume and context.saved_report.get("save_videos")):
        report["save_videos"] = True
        report["video_dir"] = str(state.video_dir)
        if state.video_recorder is not None:
            report["video_encoder"] = state.video_recorder.encoder
        elif "video_encoder" in context.saved_report:
            report["video_encoder"] = context.saved_report["video_encoder"]
    if getattr(context.policy, "joint_transform", None) is not None:
        report["policy_joint_transform"] = context.policy.joint_transform
    if getattr(context.policy, "policy_cameras", None) is not None:
        report["policy_cameras"] = context.policy.policy_cameras
    if context.grasp_kwargs:
        from dataclasses import asdict

        report["gripper_control"] = asdict(context.grasp_config)
        if context.grasp_config.enabled:
            report["gripper_calibration"] = context.policy.grasp.calibration
        report["gripper_latch"] = context.grasp_config.gripper_latch
        report["seed_per_episode"] = context.args_cli.seed_per_episode
    if context.smoothing_kwargs:
        report["smoothing"] = {
            key: getattr(context.args_cli, key, default) for key, default in context.smoothing_defaults.items()
        }
    report["requested_episodes"] = context.args_cli.num_episodes
    if context.smoke_test:
        report["smoke_test"] = dict(
            selection=context.smoke_selection,
            reach_timeout_s=context.smoke_reach,
            progress_timeout_s=context.smoke_progress,
        )
    report["complete"] = len(state.episodes) == context.args_cli.num_episodes
    report["policy"] = context.args_cli.policy
    fingerprint = getattr(context.args_cli, "config_fingerprint", None)
    if fingerprint is not None:
        report["config_fingerprint"] = fingerprint
    from dataclasses import asdict

    report["resume_settings"] = dict(
        policy=context.args_cli.policy,
        policy_options=context.policy_options,
        cube_starts=context.args_cli.cube_starts,
        rename_map=context.rename_map,
        lang=context.args_cli.lang_instruction,
        lang_by_color=context.args_cli.lang_instruction_by_color,
        render_warmup=context.render_warmup,
        smoothing={
            key: getattr(context.args_cli, key, default)
            for key, default in context.smoothing_defaults.items()
            if key not in ("log_timing", "timing_csv")
        },
        grasp={
            key: value
            for key, value in asdict(context.grasp_config).items()
            if key not in ("log_gripper", "gripper_csv")
        },
        seed_per_episode=context.args_cli.seed_per_episode,
    )
    return report


def finish_report(context, state):
    report = build_report(context, state)
    overall = report["overall"]
    rate = overall["success_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "n/a"
    status = "complete" if report["complete"] else "incomplete"
    print(
        f"[RESULT] {status}: {len(state.episodes)}/{context.args_cli.num_episodes} episodes; success {overall['successes']}/{overall['episodes']} ({rate_text})"
    )
    if overall.get("median_time_to_success_s") is not None:
        print(f"[RESULT] Median time to success: {overall['median_time_to_success_s']:.2f}s sim")
    for group_key, label in (("by_split", "Split"), ("by_zone", "Zone"), ("by_cube_color", "Colour")):
        groups = report.get(group_key, {})
        if groups:
            values = ", ".join((f"{name} {stats['successes']}/{stats['episodes']}" for name, stats in groups.items()))
            print(f"[RESULT] {label}: {values}")
    if context.args_cli.results_json:
        write_report(context.args_cli.results_json, report)
        if context.benchmark:
            save_heatmap(report, context.args_cli.results_json.with_suffix(".png"))
        print(f"Results JSON: {context.args_cli.results_json}")
    elif context.smoke_test and (context.smoke_selection == "one_per_zone" or context.smoke_budget is not None):
        smoke_result = {
            key: report.get(key)
            for key in (
                "complete",
                "requested_episodes",
                "checkpoint",
                "seed",
                "step_dt",
                "episode_length_s",
                "action_horizon",
                "action_repeat",
                "policy_cameras",
                "policy_joint_transform",
                "resume_settings",
                "smoke_test",
                "overall",
            )
        }
        smoke_result["episodes"] = [
            {
                key: row.get(key)
                for key in (
                    "start_id",
                    "zone",
                    "policy_seed",
                    "success",
                    "elapsed_sim_s",
                    "wall_time_s",
                    "failure_mode",
                    "stages",
                    "smoke_deadline",
                )
            }
            for row in state.episodes
        ]
        print("[SMOKE_RESULT] " + json.dumps(smoke_result, separators=(",", ":"), allow_nan=False), flush=True)
    elif not context.smoke_test:
        print(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
