"""Audit 36 async B runs and compare summaries with the published reference."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / 'raw_async/control'
REFERENCE = 'e1b941727dbe51c234c136df72f6c1a0e3723aee'


def key(spec):
    return tuple(int(spec[k]) for k in ['height', 'width', 'cameras', 'chunk', 'infer_ms'])


def distribution(values):
    return {k: float(np.percentile(values, p)) for k, p in [('p50',50),('p95',95),('p99',99),('max',100)]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    expected = {(h,w,c,n,d) for h,w in [(480,640),(224,224)]
                for c,n,d in itertools.product([1,2,3],[16,30,50],[50,150])}
    rows, problems, hashes = [], [], {}
    for path in sorted(RAW.glob('B-*.json')):
        row = json.loads(path.read_text())
        if path.name.endswith('.server.json'):
            continue
        spec = row['spec']
        if spec['setup'] != 'async' or spec['stack'] != 'zmq_npy' or spec['seconds'] != 60 or spec['real_forward']:
            problems.append(f'{path.name}: unexpected benchmark configuration')
        with gzip.open(path.with_suffix('.ticks.csv.gz'), 'rt') as stream:
            ticks = list(csv.DictReader(stream))
        with gzip.open(path.with_suffix('.chunks.csv.gz'), 'rt') as stream:
            chunks = list(csv.DictReader(stream))
        with gzip.open(path.with_suffix('.turnarounds.json.gz'), 'rt') as stream:
            turnarounds = json.load(stream)
        if len(ticks) != 1800 or [int(t['tick']) for t in ticks] != list(range(1800)) or row['elapsed_s'] < 60:
            problems.append(f'{path.name}: control schedule mismatch')
        held = sum(int(t['held']) for t in ticks)
        startup = sum(int(t['held']) for t in ticks if not t['age_ms'])
        missed = sum(int(t['missed_deadline']) for t in ticks)
        if (held,startup,held-startup,missed) != tuple(row[k] for k in
                ['held_ticks','startup_held_ticks','steady_held_ticks','missed_dispatch_ticks']):
            problems.append(f'{path.name}: hold accounting mismatch')
        accepted = [c for c in chunks if 0 <= float(c['accepted_elapsed_s']) < 60]
        if len(accepted) - bool(accepted) != row['chunk_replacements']:
            problems.append(f'{path.name}: replacement accounting mismatch')
        for name, values in [('action_age_ms',[float(t['age_ms']) for t in ticks if t['age_ms']]),
                             ('turnaround_ms', turnarounds)]:
            if not values or min(values) < 0 or len(values) != row[name]['n']:
                problems.append(f'{path.name}: {name} sample count/range mismatch')
            elif any(not np.isclose(v,row[name][k]) for k,v in distribution(values).items()):
                problems.append(f'{path.name}: {name} statistics mismatch')
        rows.append(row)
        for source in [path,path.with_suffix('.ticks.csv.gz'),path.with_suffix('.chunks.csv.gz'),
                       path.with_suffix('.turnarounds.json.gz')]:
            hashes[str(source.relative_to(HERE))] = hashlib.sha256(source.read_bytes()).hexdigest()
    actual = {key(r['spec']) for r in rows}
    if actual != expected or len(rows) != 36:
        problems.append(f'{len(rows)}/36 cases; {len(expected-actual)} missing; {len(actual-expected)} extra')
    with (HERE/'reference_control.csv').open() as stream:
        baseline = list(csv.DictReader(stream))
    if len(baseline) != 216:
        problems.append('Published reference must contain 216 B summaries')
    protected = ['src/core/policy_client/transport.py','src/sim/eval/loop.py',
                 'src/sim/eval/sim_remote_policy.py','policies']
    diff = subprocess.check_output(['git','diff','--name-only','origin/v2','--',*protected],cwd=ROOT,text=True)
    if diff.strip():
        problems.append('Protected transport/default simulator/policy files changed: '+diff.strip())
    env_path = RAW/'environment.json'
    if env_path.exists():
        implementation = json.loads(env_path.read_text())['implementation_commit']
        runtime = ['src/core/policy_client/async_worker.py','src/core/policy_client/remote_policy.py',
                   'src/core/policy_client/smoothing.py','tools/bench_transport/bench.py',
                   'tools/bench_transport/common.py','tools/bench_transport/servers.py']
        drift = subprocess.check_output(['git','diff','--name-only',implementation,'--',*runtime],cwd=ROOT,text=True)
        if drift.strip():
            problems.append('Benchmark runtime differs from its recorded commit: '+drift.strip())
    cpu = (HERE/'cpu-suite.txt').read_text()
    cpu_verified = (HERE/'cpu-suite.exitcode').read_text().strip() == '0' and '656 passed, 2 skipped' in cpu
    if not cpu_verified:
        problems.append('Full CPU suite evidence missing')
    status = dict(complete=not problems,cells=len(rows),expected_cells=36,problems=problems,
                  reference_commit=REFERENCE,raw_sha256=hashes)
    (HERE/'async_audit.json').write_text(json.dumps(status,indent=2)+'\n')
    flat = []
    for r in rows:
        spec = r['spec']
        item = {k:spec[k] for k in ['height','width','cameras','chunk','infer_ms','refill']}
        item.update({k:r[k] for k in ['ticks','held_ticks','startup_held_ticks','steady_held_ticks',
                                      'chunk_replacements','elapsed_s','client_cpu_percent','server_cpu_percent']})
        for name in ['action_age_ms','turnaround_ms']:
            item.update({name+'_'+k:v for k,v in r[name].items()})
        flat.append(item)
    if flat:
        with (HERE/'async_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(flat)
    report = ['# Background policy worker: CPU control-loop results','',
        f'Status: {"complete" if not problems else "INCOMPLETE"}. {len(rows)}/36 new cases. [Audit](async_audit.json).','',
        'The new runs use the production async RemotePolicy, original np.save/msgpack ZMQ transport, '
        'fixed zero actions, and 50/150 ms artificial inference sleeps. Each run schedules 1,800 ticks '
        'over 60 real seconds. This measures client scheduling and CPU/loopback overhead, not learned '
        'task success, GPU latency or physical smoothness. [Commands and implementation](README.md).','',
        f'The 216 reference summaries come unchanged from benchmark commit `{REFERENCE}` '
        '([published report](https://github.com/iampa1teja/Sim-To-Real/blob/'+REFERENCE+'/tools/bench_transport/REPORT.md)). '
        'They are a prior serial run on this workstation, not concurrent paired trials. '
        'New hardware/software and the measured runtime commit are recorded in '
        '[environment](async_environment.json); the audit rejects runtime drift after collection.','',
        '## What changed','',
        'RemotePolicy now has an opt-in single-owner socket worker. Control ticks poll completed '
        'requests and continue dispatching buffered actions. Refill uses remaining control slots '
        'and max(configured refill, recent turnaround p95 + one period), with one inference in flight. '
        'Copied observation snapshots carry generation, monotonic capture time and control step; '
        'arrivals trim elapsed repeated steps and replace the aligned buffer.','',
        'Starvation repeats the applied, bounded target and faults after grace; request deadlines '
        'latch faults without waiting on the network. Reset, Stop, instruction change and faults '
        'invalidate older replies. Optional blending/EMA, role-based gripper exclusions, grasp '
        'feedback and seed acknowledgements remain available. Real callers keep range clamps, '
        'first-fresh-target checks and motion locks. Simulator defaults remain deterministic.','',
        'An initial pilot exposed a one-period refill delay caused by sub-period capture jitter. '
        'The accepted matrix is remeasured after correcting the control-slot boundary. Clock tests '
        'also verify that a long control pause expires targets and does not restart starvation '
        'grace. Pilot data is ignored and is not mixed with these summaries.','',
        '## Target checks','']
    lr = {key(b):b for b in baseline if b['stack']=='lerobot' and b['encoding']=='raw' and b['setup']=='default'}
    if rows:
        age_deltas = [(key(r['spec']),p,r['action_age_ms'][p]-float(lr[key(r['spec'])]['action_age_ms_'+p]))
                      for r in rows for p in ['p50','p95','p99','max']]
        violations = [(k,p,d) for k,p,d in age_deltas if d>0]
        holds_ok=sum(r['steady_held_ticks']<=9 for r in rows)
        report += [f'{holds_ok}/{len(rows)} collected runs meet the single-digit hold target after startup; '
                   f'{sum(r["steady_held_ticks"] for r in rows)} steady-state held ticks in total. '
                   f'Startup holds range from {min(r["startup_held_ticks"] for r in rows)} to '
                   f'{max(r["startup_held_ticks"] for r in rows)} ticks.','',
                   'Age is compared separately at p50, p95, p99 and max against the matched '
                   'LeRobot raw/default reference (its raw encoding is the primary comparison). '
                   'Differences are computed per payload/delay; no pooled percentile is claimed.', '',
                   '| Age statistic | New cells no worse than reference |', '|---|---:|',
                   *[f'| {p} | {sum(q==p and delta<=0 for _,q,delta in age_deltas)}/{len(rows)} |'
                     for p in ['p50','p95','p99','max']]]
        if violations:
            report += ['', 'Statistics above the reference (new minus reference, ms):', '',
                       *[f'- {h}×{w}, {c} cameras, chunk {n}, delay {d}, {p}: {delta:+.3f}' for (h,w,c,n,d),p,delta in violations],
                       '', 'Single runs have scheduler/copy jitter and no CPU affinity. These differences '
                       'remain strict single-run target misses; they are not erased by a tolerance. Review turnaround '
                       'and per-tick traces before tuning refill or making a hardware claim.']
        report += ['']
    report += ['## Side-by-side measurements','',
        'Held includes startup. Steady held is available only for the new runs; reference raw traces '
        'are retained on the benchmark branch. Age includes retained targets during holds and excludes '
        'startup slots before the first prediction. Turnaround is observation capture through completion '
        'of the worker request, including input copies/seed RPC when enabled; the old B summaries did '
        'not record a turnaround distribution. Replacements count accepted chunks after the first '
        'inside the window; new chunks activate on the control tick. All times are milliseconds.','',
        '| H×W / cameras / chunk | Delay | Setup | Held / 1800 | Steady held | Age p50 | Age p95 | Age p99 | Age max | Replacements | Turnaround p50 / p95 / p99 / max |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    by_key={key(r['spec']):r for r in rows}
    for k in sorted(expected,reverse=True):
        h,w,c,n,d=k
        for b in sorted((b for b in baseline if key(b)==k),key=lambda b:(b['stack'],b['encoding'],b['setup'])):
            setup=('zmq '+b['setup'] if b['stack']=='zmq_npy' else 'lerobot '+b['encoding']+' '+b['setup'])
            report.append(f'| {h}×{w} / {c} / {n} | {d} | {setup} | {int(float(b["held_ticks"]))} | — | '+
                          ' | '.join(f'{float(b["action_age_ms_"+p]):.3f}' for p in ['p50','p95','p99','max'])+
                          f' | {int(float(b["chunk_replacements"]))} | — |')
        if k in by_key:
            r=by_key[k]
            report.append(f'| {h}×{w} / {c} / {n} | {d} | **zmq async** | {r["held_ticks"]} | {r["steady_held_ticks"]} | '+
                          ' | '.join(f'{r["action_age_ms"][p]:.3f}' for p in ['p50','p95','p99','max'])+
                          f' | {r["chunk_replacements"]} | '+
                          ' / '.join(f'{r["turnaround_ms"][p]:.3f}' for p in ['p50','p95','p99','max'])+' |')
    cpu_status = ('Full CPU suite: **656 passed, 22 subtests passed, 2 existing skips**, exit 0 '
                  '([log](cpu-suite.txt)). ' if cpu_verified else
                  'The final full-suite rerun is pending; [last saved suite log](cpu-suite.txt). ')
    report += ['', '## Validation and limits','',cpu_status+
        'The two baseline skips are the original inapplicable jaw cases in '
        '`test_mimic_joints.py` and `test_parallel_gripper_geometry.py`; neither test was changed. '
        'No new skips, ignore flags or deselections were introduced. The only pre-existing test file '
        'modified is `test_transport.py`, which gained a real-ZMQ async example-server test; its '
        'existing assertions were retained. New worker/integration tests use controlled completions '
        'and a fake monotonic clock.','',
        'Default simulator evaluation remains step-driven: `SimulationSmoother`, fixed schedules, '
        'seeds, eval loop and simulator policy wrapper retain their existing default behavior. '
        'Temporal ensembling is explicitly rejected in async mode. Socket close drains the pending '
        'RPC sequence on the worker (each RPC retains its configured transport timeout); cancellation invalidates playback immediately '
        'and never waits on that socket.','',
        'Both documented client modes also passed against the real ZMQ example server on CPU '
        '([utility validation](gpu-client-smoke.txt)); the GPU-serving command was checked through '
        'the CLI dry-run planner. This did not execute a GPU model or open a physical robot.','',
        'Hardware validation still required: real-arm clamp/first-target/Stop behavior under device '
        'latency and operator races, real GPU inference tail latency, calibrated action/units parity, '
        'trajectory jerk and physical smoothness. Fixed synthetic RGB/actions and one run per cell '
        'cannot establish these. The benchmark changes only the example plugin instance camera '
        'list for 1–3 cameras; production policy folders and transport are unchanged. Raw captures '
        'are kept locally under ignored `raw_async/`; committed files contain summaries and hashes.']
    (HERE/'ASYNC_RESULTS.md').write_text('\n'.join(report)+'\n')
    env=RAW/'environment.json'
    if env.exists():
        (HERE/'async_environment.json').write_bytes(env.read_bytes())
    print(json.dumps({k:v for k,v in status.items() if k!='raw_sha256'},indent=2))
    if args.require_complete and problems:
        raise SystemExit(1)


if __name__=='__main__':
    main()
