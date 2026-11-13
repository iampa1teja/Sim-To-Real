"""Generate readable result tables and CSV; audit full matrix and raw sample coverage."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'


def load():
    rows = []
    for directory in ['localhost', 'netem-1ms', 'netem-5ms', 'control']:
        for path in sorted((RESULTS / directory).glob('[AB]-*.json')):
            if path.name.endswith('.server.json'):
                continue
            data = json.loads(path.read_text())
            data['_path'] = path
            data['_network'] = directory
            rows.append(data)
    return rows


def flatten(row):
    result = {'network': row['_network']}
    result.update({k: row['spec'].get(k) for k in ['mode', 'height', 'width', 'cameras', 'chunk', 'stack',
        'encoding', 'infer_ms', 'setup', 'iterations', 'warmup', 'warmup_timeout_budget', 'seconds', 'prefetch', 'blend', 'refill', 'netem_ms']})
    if row['spec']['mode'] == 'A':
        result.update(warmup_successes=row.get('warmup_successes', row['spec']['warmup']),
                      warmup_attempts=row.get('warmup_attempts', row['spec']['warmup']),
                      warmup_timeouts=len(row.get('warmup_failures', [])))
    for key in ['client_cpu_percent', 'server_cpu_percent', 'elapsed_s', 'held_ticks', 'ticks',
                'missed_dispatch_ticks', 'chunk_replacements', 'executed_source_transitions', 'effective_refill_s',
                'late_chunk_arrivals']:
        if key in row:
            result[key] = row[key]
    for key in ['rtt_ms', 'action_age_ms', 'tick_call_ms']:
        for metric, value in row.get(key, {}).items():
            result[key + '_' + metric] = value
    for side in ['client', 'server']:
        for key, stats in row.get(side + '_codec', {}).items():
            for metric, value in stats.items():
                result[f'{side}_{key}_{metric}'] = value
    for direction, stats in (row.get('wire') or {}).get('per_exchange', {}).items():
        for key, value in stats.items():
            result[f'wire_{direction}_{key}'] = value
    return result


def audit(rows):
    from bench import matrix
    problems = []
    for directory, mode, latency in [('localhost', 'A', 0), ('netem-1ms', 'A', 1),
                                     ('netem-5ms', 'A', 5), ('control', 'B', 0)]:
        actual = [row for row in rows if row['_network'] == directory]
        expected = list(matrix(argparse.Namespace(mode=mode)))
        def key(spec):
            return tuple(spec.get(k) for k in ['height', 'width', 'cameras', 'chunk', 'stack',
                                               'encoding', 'infer_ms', 'setup'])
        missing = set(map(key, expected)) - {key(r['spec']) for r in actual}
        extra = {key(r['spec']) for r in actual} - set(map(key, expected))
        if missing or extra or len(actual) != len(expected):
            problems.append(f'{directory}: {len(actual)}/{len(expected)} cells, {len(missing)} missing, {len(extra)} extra')
        for row in actual:
            spec, path = row['spec'], row['_path']
            if spec['netem_ms'] != latency:
                problems.append(f'{path.name}: unexpected latency metadata')
            if mode == 'A':
                if spec['iterations'] < 1000 or spec['warmup'] < 50:
                    problems.append(f'{path.name}: insufficient repetitions')
                failures = row.get('warmup_failures', [])
                successes = row.get('warmup_successes', spec['warmup'])
                attempts = row.get('warmup_attempts', spec['warmup'])
                if successes != spec['warmup'] or attempts != successes + len(failures):
                    problems.append(f'{path.name}: warmup accounting mismatch')
                if len(failures) > spec.get('warmup_timeout_budget', 0):
                    problems.append(f'{path.name}: warmup timeout budget exceeded')
                if not row.get('wire') or row['wire']['kernel_dropped_packets']:
                    problems.append(f'{path.name}: missing/lossy wire data')
                with gzip.open(path.with_suffix('.samples.json.gz'), 'rt') as stream:
                    raw = json.load(stream)
                if len(raw['rtt_ms']) != spec['iterations'] or not np.isclose(
                        np.percentile(raw['rtt_ms'], 99), row['rtt_ms']['p99']):
                    problems.append(f'{path.name}: RTT sample mismatch')
                for side in ['client_codec', 'server_codec']:
                    for name, values in raw[side].items():
                        if len(values) != spec['iterations'] or not np.isclose(np.mean(values), row[side][name]['mean']):
                            problems.append(f'{path.name}: codec sample mismatch {side}/{name}')
                if raw['client_codec']['encoded_bytes'] != raw['server_codec']['decoded_bytes']:
                    problems.append(f'{path.name}: request byte mismatch')
                if raw['client_codec']['decoded_bytes'] != raw['server_codec']['encoded_bytes']:
                    problems.append(f'{path.name}: response byte mismatch')
                with gzip.open(path.with_suffix('.packets.csv.gz'), 'rt') as stream:
                    packets = list(csv.DictReader(stream))
                for direction in ['request', 'response']:
                    selected = [p for p in packets if (int(p['dst_port']) == spec['port']) == (direction == 'request')]
                    for metric in ['l2_bytes', 'ip_bytes', 'tcp_payload_bytes']:
                        if sum(int(p[metric]) for p in selected) != row['wire']['totals'][direction][metric]:
                            problems.append(f'{path.name}: wire total mismatch')
                    app_bytes = sum(raw['client_codec']['encoded_bytes' if direction == 'request' else 'decoded_bytes'])
                    if row['wire']['totals'][direction]['tcp_payload_bytes'] < app_bytes:
                        problems.append(f'{path.name}: wire capture misses application bytes')
            else:
                with gzip.open(path.with_suffix('.ticks.csv.gz'), 'rt') as stream:
                    ticks = list(csv.DictReader(stream))
                if spec['seconds'] != 60 or row['ticks'] != 1800 or len(ticks) != 1800 or row['elapsed_s'] < 60:
                    problems.append(f'{path.name}: incomplete 60 s / 1800 tick run')
                if [int(t['tick']) for t in ticks] != list(range(1800)):
                    problems.append(f'{path.name}: tick indices missing')
                if sum(int(t['held']) for t in ticks) != row['held_ticks']:
                    problems.append(f'{path.name}: hold count mismatch')
                if sum(int(t['missed_deadline']) for t in ticks) != row['missed_dispatch_ticks']:
                    problems.append(f'{path.name}: missed-deadline mismatch')
                with gzip.open(path.with_suffix('.chunks.csv.gz'), 'rt') as stream:
                    arrivals = list(csv.DictReader(stream))
                accepted = [event for event in arrivals if 0 <= float(event['accepted_elapsed_s']) < spec['seconds']]
                if len(accepted) - bool(accepted) != row['chunk_replacements']:
                    problems.append(f'{path.name}: chunk-replacement sample mismatch')
                if sum(float(event['accepted_elapsed_s']) >= spec['seconds'] for event in arrivals) != row['late_chunk_arrivals']:
                    problems.append(f'{path.name}: late chunk-arrival mismatch')
                ages = [float(t['age_ms']) for t in ticks if t['age_ms']]
                if ages and (min(ages) < 0 or not np.isclose(np.percentile(ages, 95), row['action_age_ms']['p95'])):
                    problems.append(f'{path.name}: action-age sample mismatch')
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'origin/v2', '--',
                                      'src', 'policies', 'docker', 'configs'], cwd=ROOT, text=True).strip()
    if changed:
        problems.append('Production files changed: ' + changed)
    exit_path = RESULTS / 'cpu-tests-final.exitcode'
    if not exit_path.exists() or exit_path.read_text().strip() != '0':
        problems.append('CPU suite successful exit code is unverified')
    test_path = RESULTS / 'cpu-tests-final.txt'
    if not test_path.exists():
        problems.append('Zero-skip CPU test evidence is pending')
    else:
        import re
        tests = test_path.read_text()
        if not re.search(r'\d+ passed', tests) or re.search(r'\d+ (failed|error|skipped|deselected)', tests):
            problems.append('CPU suite lacks a clean zero-skip result')
    return problems


def f(number):
    return f'{number:.3f}' if number is not None else '—'


def label(spec):
    return f"{spec['height']}×{spec['width']} / {spec['cameras']} / {spec['chunk']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    rows = load()
    flat = [flatten(row) for row in rows]
    keys = list(dict.fromkeys(k for row in flat for k in row))
    with (RESULTS / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    problems = audit(rows)
    status = {'complete': not problems, 'cells': len(rows), 'expected_cells': 432, 'problems': problems,
              'result_sha256': {str(r['_path'].relative_to(HERE)): hashlib.sha256(r['_path'].read_bytes()).hexdigest()
                                for r in rows}}
    (RESULTS / 'audit.json').write_text(json.dumps(status, indent=2))
    report = ['# ZMQ versus LeRobot 0.7 remote inference', '',
              f"Status: {'complete' if not problems else 'INCOMPLETE — measurements/acceptance still in progress'}. "
              f'{len(rows)}/432 matrix cells recorded. See [audit](results/audit.json).', '',
              'These CPU numbers isolate transport, processors and client scheduling. Neural inference is replaced '
              'by fixed zero chunks after 0/50/150 ms sleep. They do not establish GPU end-to-end latency, policy '
              'quality, physical smoothness or useful language behavior. [Exact commands and metric definitions](README.md).', '',
              '## Measured facts', '',
              'Software/hardware are recorded per run in `results/*/environment.json`. '
              'Client/server CPU percentages use process CPU time divided by wall time; one busy core is 100%. '
              'Request/response sizes below include captured loopback Ethernet/IP/TCP framing and ACKs, amortized '
              'over measured exchanges. This is packet accounting on `lo`, not physical link serialization. '
              'Application bytes and all four codec timing distributions are in [summary.csv](results/summary.csv); '
              'every RTT, codec sample, packet header/length and control tick is retained in compressed raw files.', '']
    baseline = [r for r in rows if r['_network'] == 'localhost']
    if baseline:
        report += ['### A — synchronous transport, zero inference delay', '',
                   'Each row has 50 warmups and 1,000 measured exchanges. Times are milliseconds. '
                   'Sizes are bytes/exchange. C-enc/C-dec/S-enc/S-dec are mean codec times.', '',
                   '| H×W / cameras / chunk | Stack / encoding | p50 | p95 | p99 | max | C-enc | C-dec | S-enc | S-dec | Request B | Response B | Client CPU% | Server CPU% |',
                   '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in baseline:
            spec = r['spec']; wire = r['wire']['per_exchange']
            vals = [r['rtt_ms'][k] for k in ['p50', 'p95', 'p99', 'max']]
            vals += [r[side + '_codec'][operation + '_ms']['mean'] for side, operation in
                     [('client', 'encode'), ('client', 'decode'), ('server', 'encode'), ('server', 'decode')]]
            vals += [wire[d]['l2_bytes'] for d in ['request', 'response']]
            vals += [r['client_cpu_percent'], r['server_cpu_percent']]
            report.append('| ' + label(spec) + f" | {spec['stack']} / {spec['encoding']} | " + ' | '.join(map(f, vals)) + ' |')
    controls = [r for r in rows if r['_network'] == 'control']
    report += ['', '### B — 30 Hz control, 60 seconds per cell', '',
               'Held includes startup, exhausted-buffer holds and deadlines missed during blocking calls. '
               'Missed is the blocking subset. Age is elapsed client monotonic time since the source observation, '
               'including the retained last target on holds; startup holds without a source are excluded. '
               'Blended actions use the older contributing observation. Replacements count accepted chunks after '
               'the first within the 60 s window, even if an early replacement supersedes an unexecuted chunk. '
               'A final blocking call can return after the window; its arrival is retained separately and excluded '
               'from replacement counts.', '',
               'Our synchronous driver and prefetch setup call the real production `RemotePolicy.next_raw_action`. '
               '`SimulationSmoother` still queries synchronously and schedules synthetic P-step delivery; '
               'prefetch does not create a background network worker. Prefetch P=ceil(delay×30)+2 (4 or 7), '
               'blend=2. LeRobot uses its actual engine/dispatch with aligned chunks, zero blending, default '
               'refill 0.5 s or tuned delay+2/30 s (0.1167/0.2167 s). Its effective refill has an adaptive '
               'turnaround+one-action floor. This is a wall-clock paced control schedule with a retained last '
               'target; the unpaced Isaac simulator loop has different wall-clock behavior.', '',
               '| H×W / cameras / chunk | Stack / encoding / setup | Delay ms | Held / 1800 | Missed | Age p50 ms | Age p95 ms | Age p99 ms | Age max ms | Replacements |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in controls:
        spec = r['spec']
        vals = [spec['infer_ms'], r['held_ticks'], r['missed_dispatch_ticks']]
        vals += [r['action_age_ms'].get(k) for k in ['p50', 'p95', 'p99', 'max']]
        vals += [r['chunk_replacements']]
        report.append('| ' + label(spec) + f" | {spec['stack']} / {spec['encoding']} / {spec['setup']} | " + ' | '.join(map(f, vals)) + ' |')
    report += ['', '### C — added loopback latency', '',
               '`tc netem` runs only inside the dedicated namespace. A configured delay of 1 or 5 ms applies '
               'on each traversal of `lo`, so a request/reply adds approximately 2 or 10 ms RTT before '
               'segmentation/ACK effects. Packet offloads remain at their defaults. This tests added delay, '
               'not a bandwidth-constrained WAN. All 72 A cells are repeated at each setting.', '',
               '| H×W / cameras / chunk | Stack / encoding | Added delay per direction ms | p50 ms | p95 ms | p99 ms | max ms |',
               '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if not r['_network'].startswith('netem-'):
            continue
        spec = r['spec']
        vals = [spec['netem_ms'], *[r['rtt_ms'][k] for k in ['p50', 'p95', 'p99', 'max']]]
        report.append('| ' + label(spec) + f" | {spec['stack']} / {spec['encoding']} | " + ' | '.join(map(f, vals)) + ' |')
    report += ['', '### Cold-session delivery and warmup', '',
               'The initial 5 ms / three-camera 480×640 / LeRobot raw attempt timed out during warmup '
               'and stopped the supervisor. A fresh diagnostic reproduced a timeout on request 0: the '
               'client encoded one observation and the server decoded none. In a separate six-request '
               'diagnostic using one session, request 0 timed out at about 5,010 ms and the next five '
               'independent requests completed in 27–61 ms. Netem and the packet sniffer both reported '
               'zero drops. Captured client TCP payload, including admission, was smaller than the '
               'six encoded observation payloads. These observations are consistent with the transport\'s '
               'documented nonblocking DROP publication under startup backpressure; the exact internal '
               'Zenoh drop mechanism was not instrumented.', '',
               'Original failures, counters and the diagnostic packet lengths are retained in '
               '[failures/](results/failures/), including '
               '[six-request evidence](results/failures/netem5-cold-session-delivery.json). '
               'The benchmark now records up to three warmup timeouts while still requiring 50 successful '
               'warmups before sampling. Each subsequent observation has a new request ID; no request is '
               'replayed. The 1,000 measured requests remain fail-fast, so successful steady-state rows '
               'do not discard failed measured samples. This diagnostic/manual-client warmup behavior is '
               'not evidence that RemoteInferenceEngine automatically recovers from such a startup timeout.', '',
               '| Network | H×W / cameras / chunk | Stack / encoding | Successful warmups | Attempts | Warmup timeouts |',
               '|---|---|---|---:|---:|---:|']
    for r in rows:
        if r.get('warmup_failures'):
            s = r['spec']
            report.append(f"| {r['_network']} | {label(s)} | {s['stack']} / {s['encoding']} | "
                          f"{r['warmup_successes']} | {r['warmup_attempts']} | {len(r['warmup_failures'])} |")
    report += ['', '## Compatibility', '', (HERE / 'COMPATIBILITY.md').read_text(), '',
               '## Recommendation and migration work', '']
    if len(baseline) == 72 and len(controls) == 216:
        def key(r):
            s = r['spec']; return s['height'], s['cameras'], s['chunk']
        npy = {key(r): r for r in baseline if r['spec']['stack'] == 'zmq_npy'}
        raw = {key(r): r for r in baseline if r['spec']['stack'] == 'zmq_raw'}
        lr = {key(r): r for r in baseline if r['spec']['stack'] == 'lerobot' and r['spec']['encoding'] == 'raw'}
        savings = np.array([100 * (1 - raw[k]['rtt_ms']['p50'] / npy[k]['rtt_ms']['p50']) for k in npy])
        saving = np.median(savings)
        ratio = np.median([lr[k]['rtt_ms']['p50'] / npy[k]['rtt_ms']['p50'] for k in npy])
        report += [f'Across matched localhost cells, the median relative change from the raw-array codec is '
                   f'{saving:.1f}% lower p50 RTT; LeRobot raw has a median p50 ratio of {ratio:.2f}× versus '
                   'our npy serializer. These summarize per-cell ratios, not pooled latency percentiles.', '',
                   f'Raw-codec p50 improvements range from {savings.min():.1f}% to {savings.max():.1f}% '
                   f'(positive means lower latency); {int(np.sum(savings < 0))}/{len(savings)} cells regress. '
                   'The serializer is therefore a candidate improvement whose per-payload distributions '
                   'need review, rather than a universal latency win.', '']
        report += ['| Stack / encoding / setup | Delay ms | Mean held ticks per run | Median per-run age p95 ms |',
                   '|---|---:|---:|---:|']
        for delay in [50, 150]:
            groups = sorted({(r['spec']['stack'], r['spec']['encoding'], r['spec']['setup']) for r in controls})
            for group in groups:
                selected = [r for r in controls if r['spec']['infer_ms'] == delay and
                            tuple(r['spec'][k] for k in ['stack', 'encoding', 'setup']) == group]
                report.append('| ' + ' / '.join(group) + f' | {delay} | ' +
                              f"{np.mean([r['held_ticks'] for r in selected]):.1f} | " +
                              f"{np.median([r['action_age_ms']['p95'] for r in selected]):.3f} |")
        report += ['', 'Recommendation (engineering inference): **keep ZMQ plus specific improvements**. '
                   'Before promoting the raw dtype/shape serializer, add production validation/bounds and '
                   'wire-version negotiation, and confirm gains for the payloads with measured regressions. '
                   'Implement a real single-owner background inference worker '
                   'with monotonic observation provenance, stale-prefix trimming, deadline handling and '
                   'retained-target dispatch; the present prefetch helper alone does not overlap blocking '
                   'network calls. Preserve ArmInfo calibration/role/type checks and benchmark the new '
                   'worker separately. LeRobot offers a substantially richer lifecycle/admission and '
                   'asynchronous execution implementation, but adopting it entails two model/processor '
                   'exports or adapters, simulator dispatch/hold integration, contract/calibration mapping, '
                   'separate serving dependencies, and new reset/fault/prompt parity validation. The observed '
                   'cold-session delivery failure also needs a verified transport/startup solution for large raw '
                   'images under latency; the benchmark warmup allowance is not a production recovery design. '
                   'The CPU comparison does not establish a GPU latency or task-success reason to undertake '
                   'that migration. Validate these decisions against the separate GPU budget before rollout.']
    else:
        report += ['Recommendation is pending the complete A/B matrix. No final migration decision is claimed '
                   'from smoke tests or partially collected cells.']
    report += ['', '## Acceptance and limits', '',
               'One run per cell, seeded synthetic high-entropy RGB, localhost TCP, no CPU affinity or hard '
               'real-time guarantees. JPEG q90 reduces bytes at a lossy encoding/decoding cost; real camera '
               'compressibility may differ. The observer adds lightweight timing instrumentation and '
               'packet capture overhead. Fixed actions measure continuity/freshness, not trajectory jerk '
               'or policy responsiveness. Tiny ACT is loaded/warmed with saved processors, then only its '
               'chunk prediction entry point is replaced; processor and protocol costs remain real.', '',
               'The full CPU suite log is [cpu-tests-final.txt](results/cpu-tests-final.txt), with its '
               '[process exit code](results/cpu-tests-final.exitcode). The approved fixture correction retains '
               'both import embodiments and applies the two parallel-specific checks only to their matching fixture. '
               'Production invariance, raw-data checks, matrix coverage and test evidence are audited by '
               '`report.py --require-complete`. The GPU follow-up commands in README are intentionally '
               'not part of the CPU acceptance run.', '']
    if problems:
        report += ['Outstanding acceptance items:', '', *['- ' + p for p in problems], '']
    (HERE / 'REPORT.md').write_text('\n'.join(report))
    print(json.dumps({'cells': len(rows), 'complete': not problems, 'problems': problems}, indent=2))
    if args.require_complete and problems:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
