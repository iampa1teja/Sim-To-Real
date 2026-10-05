"""Summarize fixed-benchmark action-horizon sweeps without simulator imports."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from sim_to_real_so101.utils.pick_place_benchmark import summarize, FAILURES, STAGES, validate_eval_set
from sim_to_real_so101.utils.pick_place_eval import results_report


def complete_report(report, expected, eval_set_id=None, horizon=None, settings=None):
    """Require the exact unique (start, repeat) schedule, not just a row count."""
    try:
        if report.get('schema_version') != 2 or report.get('complete') is not True:
            return False
        validate_eval_set(report['eval_set'])
        if report['eval_set_id'] != report['eval_set']['eval_set_id']:
            return False
        if eval_set_id and report['eval_set_id'] != eval_set_id:
            return False
        if horizon is not None and report.get('action_horizon') != horizon:
            return False
        if settings and any(report.get(k) != v for k, v in settings.items()):
            return False
        keys = [(r['start_id'], r['repeat']) for r in report['episodes']]
        return (report.get('requested_episodes') == len(expected) and len(keys) == len(expected)
                and len(set(keys)) == len(keys) and set(keys) == set(expected))
    except (KeyError, TypeError, ValueError):
        return False


def paired(first, second):
    # Same comparison helper as scripts/compare_eval_results.py; only horizon varies.
    from sim_to_real_so101.utils.eval_comparison import compare
    return compare([first, second], varying_action_horizon=True)


def aggregate(horizon, split, rows, report, status):
    value = summarize(rows, report.get('step_dt', 1))
    timing = results_report(rows, task='', checkpoint='', seed=0, step_dt=report.get('step_dt', 1))['overall']
    times = [r['success_step'] * report.get('step_dt', 1) for r in rows if r['success']]
    def avg(key):
        # Missing telemetry must never be represented as measured zero.
        return mean(r[key] for r in rows) if rows and all(r.get(key) is not None for r in rows) else None
    result = dict(action_horizon=horizon, split=split, status=status,
                  episodes=value['episodes'], successes=value['successes'], success_rate=value['success_rate'],
                  ci95_low=value['success_rate_ci95'][0], ci95_high=value['success_rate_ci95'][1],
                  **{s+'_rate': value['stage_rates'][s] for s in STAGES if s != 'success'},
                  **value['failure_modes'], median_time_to_success_s=value['median_time_to_success_s'],
                  mean_time_to_success_s=mean(times) if times else None,
                  success_within_15s=timing['success_rate_within_15s'],
                  mean_inference_calls_per_episode=avg('inference_calls'), mean_wall_time_per_episode_s=avg('wall_time_s'))
    grasp_times = [r['time_to_grasp_s'] for r in rows if r.get('time_to_grasp_s') is not None]
    result['mean_time_to_grasp_s'] = mean(grasp_times) if grasp_times else None
    result['mean_dither_score'] = avg('dither_score')
    if report.get('repeats', 1) > 1:
        result['per_start_consistency'] = {k: mean(r['success'] for r in rows if r['start_id'] == k)
                                           for k in sorted({r['start_id'] for r in rows})}
    return result


def summarize_folder(folder):
    folder = Path(folder)
    config = json.loads((folder/'sweep_config.json').read_text())
    if 'variants' in config:
        return summarize_latch_variants(folder, config)
    expected = [tuple(k) for k in config['expected_schedule']]
    rows, reports = [], {}
    for horizon in config['horizons']:
        run = folder/f'ah_{horizon}'
        try:
            report = json.loads((run/'results.json').read_text())
        except (OSError, ValueError):
            report = {}
        status = 'complete' if complete_report(report, expected, config['eval_set_id'], horizon, config.get('report_settings')) else 'failed'
        if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('status') == 'failed':
            status = 'failed'
        if status == 'complete':
            reports[horizon] = report
        episodes = report.get('episodes', [])
        for split in [*config['selected_splits'], 'all']:
            subset = [r for r in episodes if split == 'all' or r['split'] == split]
            rows.append(aggregate(horizon, split, subset, report, status))
    def rank(h):
        all_row = next(r for r in rows if r['action_horizon'] == h and r['split'] == 'all')
        preferred = 'id_near' if 'id_near' in reports[h].get('by_split', {}) else 'id'
        tie_rate = reports[h].get('by_split', {}).get(preferred, {}).get('success_rate')
        return (-(all_row['success_rate'] or 0), -(tie_rate or 0),
                all_row['median_time_to_success_s'] if all_row['median_time_to_success_s'] is not None else float('inf'), h)
    ranked = sorted(reports, key=rank)
    best = ranked[0] if ranked else None
    comparisons = {}
    for label, other in [('runner_up', ranked[1] if len(ranked)>1 else None), ('horizon_16', 16 if 16 in reports else None)]:
        if best is None or other is None:
            comparisons[label] = {'available': False, 'reason': 'Complete paired reports unavailable'}
            continue
        try:
            tests = paired(reports[best], reports[other])
        except ValueError as error:
            comparisons[label] = {'available': False, 'reason': str(error)}
            continue
        overall = next(p for p in tests if p['split'] == 'all')
        comparisons[label] = dict(available=True, first=best, second=other, tests=tests,
            statistically_better=overall['p_value'] < .05 and overall['first_only'] > overall['second_only'])
    output = dict(rows=rows, best_horizon=best, paired_comparisons=comparisons,
                  note='95% Wilson intervals are descriptive episode-level intervals; repeats are correlated. McNemar uses repeat 0; p-values are unadjusted.')
    (folder/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (folder/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, columns); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k,v in row.items()})
    def pct(v):
        return '—' if v is None else f'{v:.1%}'
    lines = ['# Action-horizon sweep', '', output['note'], '',
             f'Best horizon: **{best}**.' if best is not None else 'No complete horizon; no best-horizon verdict.', '']
    for label, value in comparisons.items():
        if not value['available']:
            lines.append(f"{label}: unavailable ({value['reason']}).")
        else:
            lines.append(f"{label}: AH {best} vs {value['second']}; statistically better: **{value['statistically_better']}**.")
            for p in value['tests']:
                lines.append(f"- {p['split']}: shared starts={p['shared_starts']}, discordant={p['first_only']}/{p['second_only']}, p={p['p_value']:.6g}.")
    lines += ['', 'Time to success includes settling and uses successes only. Missing telemetry is shown as —.', '',
              '| AH | Split | Status | Successes/episodes | Rate [95% CI] | Median / mean success s | ≤15 s |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        t = lambda key: '—' if r[key] is None else f'{r[key]:.2f}'
        lines.append(f"| {r['action_horizon']} | {r['split']} | {r['status']} | {r['successes']}/{r['episodes']} | {pct(r['success_rate'])} [{pct(r['ci95_low'])}, {pct(r['ci95_high'])}] | {t('median_time_to_success_s')} / {t('mean_time_to_success_s')} | {pct(r['success_within_15s'])} |")
    lines += ['', '| AH | Split | Reached | Grasped | Lifted | Over box | Placed |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append('| '+ ' | '.join([str(r['action_horizon']),r['split'],*[pct(r[s+'_rate']) for s in STAGES if s!='success']])+' |')
    lines += ['', '| AH | Split | '+' | '.join(FAILURES)+' |', '| --- | --- | '+' | '.join(['---']*len(FAILURES))+' |']
    for r in rows:
        lines.append('| '+' | '.join([str(r['action_horizon']),r['split'],*[str(r[f]) for f in FAILURES]])+' |')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    make_plots(folder, rows)
    return output


def make_plots(folder, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = folder/'plots'; plots.mkdir(exist_ok=True)
    valid = [r for r in rows if r['status']=='complete']
    fig, ax = plt.subplots()
    for split in sorted({r['split'] for r in valid}):
        group = sorted([r for r in valid if r['split']==split], key=lambda r:r['action_horizon'])
        ax.errorbar([r['action_horizon'] for r in group], [r['success_rate'] for r in group],
                    yerr=[[r['success_rate']-r['ci95_low'] for r in group], [r['ci95_high']-r['success_rate'] for r in group]], marker='o', label=split)
    ax.set(xlabel='Action horizon', ylabel='Success rate', ylim=(0,1.05));
    if valid: ax.legend()
    fig.tight_layout(); fig.savefig(plots/'success_vs_horizon.png'); plt.close(fig)
    all_rows = sorted([r for r in valid if r['split']=='all'], key=lambda r:r['action_horizon'])
    x=[r['action_horizon'] for r in all_rows]
    for filename, fields, ylabel in [('stages_vs_horizon.png',[s+'_rate' for s in STAGES if s!='success'],'Stage rate'), ('time_to_success_vs_horizon.png',['median_time_to_success_s','mean_time_to_success_s'],'Simulation seconds')]:
        fig,ax=plt.subplots()
        for field in fields: ax.plot(x,[r[field] if r[field] is not None else float('nan') for r in all_rows],marker='o',label=field)
        ax.set(xlabel='Action horizon',ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(plots/filename); plt.close(fig)
    fig,ax=plt.subplots(); bottom=[0]*len(x)
    for failure in FAILURES:
        values=[r[failure] for r in all_rows]; ax.bar([str(h) for h in x],values,bottom=bottom,label=failure)
        bottom=[a+b for a,b in zip(bottom,values)]
    ax.set(xlabel='Action horizon',ylabel='Failed episodes'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots/'failure_modes_vs_horizon.png'); plt.close(fig)


def summarize_latch_variants(folder, config):
    """Keep both on/off runs visible; never rank or compare incomplete schedules."""
    expected = [tuple(k) for k in config['expected_schedule']]
    rows, reports = [], {}
    for variant in config['variants']:
        run = folder / variant['folder']
        try:
            report = json.loads((run / 'results.json').read_text())
        except (OSError, ValueError):
            report = {}
        complete = complete_report(report, expected, config['eval_set_id'],
                                   variant['horizon'], variant['settings'])
        if (run / 'status.json').exists():
            complete &= json.loads((run / 'status.json').read_text()).get('status') == 'complete'
        if complete:
            reports[(variant['horizon'], variant['latch'])] = report
        for split in [*config['selected_splits'], 'all']:
            subset = [r for r in report.get('episodes', []) if split == 'all' or r['split'] == split]
            row = aggregate(variant['horizon'], split, subset, report, 'complete' if complete else 'failed')
            row['gripper_latch'] = variant['latch']
            rows.append(row)
    comparisons = []
    for horizon in config['horizons']:
        off, on = reports.get((horizon, 'off')), reports.get((horizon, 'on'))
        if off is None or on is None:
            comparisons.append(dict(action_horizon=horizon, available=False))
            continue
        for split in [*config['selected_splits'], 'all']:
            first, second = [next(r for r in rows if r['action_horizon'] == horizon and
                                 r['split'] == split and r['gripper_latch'] == mode) for mode in ('off', 'on')]
            differences = {}
            for key in ('success_rate', 'grasped_rate', 'mean_time_to_grasp_s', 'mean_dither_score'):
                differences[key + '_on_minus_off'] = (second[key] - first[key]
                                                      if first[key] is not None and second[key] is not None else None)
            pairs = {(r['start_id'], r['repeat']): r for r in off['episodes']}
            off_only = on_only = 0
            for r in on['episodes']:
                if split != 'all' and r['split'] != split:
                    continue
                before = pairs[(r['start_id'], r['repeat'])]['success']
                on_only += int(r['success'] and not before)
                off_only += int(before and not r['success'])
            comparisons.append(dict(action_horizon=horizon, split=split, available=True,
                                    on_only_successes=on_only, off_only_successes=off_only, **differences))
    result = dict(rows=rows, latch_comparisons=comparisons)
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (folder / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| H | Latch | Split | Status | Success | Grasp rate | Time to grasp (s) | Dither |',
             '|---|---|---|---|---|---|---|---|']
    def fmt(value):
        return '—' if value is None else f'{value:.4g}'
    for r in rows:
        lines.append(f"| {r['action_horizon']} | {r['gripper_latch']} | {r['split']} | {r['status']} | "
                     f"{fmt(r['success_rate'])} | {fmt(r['grasped_rate'])} | "
                     f"{fmt(r['mean_time_to_grasp_s'])} | {fmt(r['mean_dither_score'])} |")
    lines += ['', 'Time to grasp averages observed grasps only; dither averages all completed episodes.',
              'Paired on/off success counts and metric differences are in summary.json.']
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('folder',type=Path)
    args=parser.parse_args(argv); summarize_folder(args.folder)
    print((args.folder/'summary.md').read_text())


if __name__=='__main__': main()
