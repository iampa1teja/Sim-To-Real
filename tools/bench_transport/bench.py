"""Run each benchmark cell in fresh client/server processes; append resumable raw results."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import psutil
from common import Meter, NAMES, TASK, canonical, environment, instrument_lerobot, instrument_zmq, payload, summary
from servers import serve

HERE = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, spec):
        self.pipe, child = mp.get_context('spawn').Pipe()
        self.process = mp.get_context('spawn').Process(target=serve, args=(spec, child))
        self.process.start()
        if not self.pipe.poll(90):
            self.close()
            raise RuntimeError('Server readiness timed out')
        assert self.pipe.recv()['ready']

    def command(self, command):
        self.pipe.send(command)
        if not self.pipe.poll(10):
            raise RuntimeError('Server metrics unavailable')
        return self.pipe.recv()

    def close(self):
        if self.process.is_alive():
            self.pipe.send('stop')
            self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()


def remote_client(spec):
    from lerobot.inference import RemoteInferenceConfig
    from lerobot.remote_inference import RemoteClient
    config = RemoteInferenceConfig(endpoint=f"tcp/127.0.0.1:{spec['port']}", deployment='bench',
        semantics='benchmark-raw-joint-units-v1', chunk_merge='aligned', encoding=spec['encoding'],
        jpeg_quality=90, refill_seconds=spec.get('refill', .5))
    client = RemoteClient.connect(config)
    caps = client.capabilities
    client.admit(features=caps.features, action_feature=caps.action_feature, semantics=config.semantics,
                 action_interval=caps.action_interval, mode=config.mode)
    return client, config


def sync_query(spec, client, obs, index):
    if spec['stack'].startswith('zmq'):
        return client.get_action(obs)['actions']
    from lerobot.inference import ChunkRequest, ObservationSnapshot, ExecutionMode
    from lerobot.policies.rtc.action_queue import QueueSnapshot
    snapshot = ObservationSnapshot(canonical(obs), time.monotonic(), TASK, observation_id=str(index),
                                   action_cursor=0, execution_generation=0)
    request = ChunkRequest(snapshot, QueueSnapshot(0, 0, 0, None, None, ()), str(index),
                           client.generation, ExecutionMode.CHUNK, 0, time.monotonic())
    return client.infer(request).canonical_actions.numpy()


def cpu_time(process):
    t = process.cpu_times()
    return t.user + t.system


def transport(spec, server, client, meter, obs):
    for i in range(spec['warmup']):
        values = sync_query(spec, client, obs, i)
        assert values.shape == (spec['chunk'], 6) and np.isfinite(values).all()
        if not spec.get('real_forward'):
            assert np.all(values == 0)
    server.command('reset')
    meter.reset()
    cproc, sproc = psutil.Process(), psutil.Process(server.process.pid)
    rtts = []
    capture = None
    if spec.get('capture'):
        from wire import Capture
        capture = Capture(spec['destination'], spec['port'])
    c0, s0 = cpu_time(cproc), cpu_time(sproc)
    start = time.monotonic()
    for i in range(spec['iterations']):
        before = time.perf_counter_ns()
        values = sync_query(spec, client, obs, i + spec['warmup'])
        rtts.append((time.perf_counter_ns() - before) / 1e6)
        assert values.shape == (spec['chunk'], 6) and np.isfinite(values).all()
        if not spec.get('real_forward'):
            assert np.all(values == 0)
    elapsed = time.monotonic() - start
    server_values = server.command('stats')
    for key in ('encode_ms', 'decode_ms', 'encoded_bytes', 'decoded_bytes'):
        assert len(meter.values[key]) == spec['iterations'], (key, len(meter.values[key]))
        assert len(server_values[key]) == spec['iterations'], (key, len(server_values[key]))
    assert meter.values['encoded_bytes'] == server_values['decoded_bytes']
    assert meter.values['decoded_bytes'] == server_values['encoded_bytes']
    c1, s1 = cpu_time(cproc), cpu_time(sproc)
    wire = capture.stop(spec['iterations']) if capture else None
    return {'rtt_ms': summary(rtts), 'elapsed_s': elapsed, 'wire': wire,
            'client_cpu_percent': 100 * (c1 - c0) / elapsed,
            'server_cpu_percent': 100 * (s1 - s0) / elapsed,
            'client_codec': meter.summary(),
            'server_codec': {k: summary(v) for k, v in server_values.items()},
            'raw': {'rtt_ms': rtts, 'client_codec': {k: list(v) for k, v in meter.values.items()}, 'server_codec': server_values}}


def control(spec, server, client, meter, obs):
    """A fixed wall-clock 30 Hz schedule; skipped deadlines retain the last target."""
    from core.policy_client.remote_policy import RemotePolicy
    captures, accepts, tick_rows = {}, [], []
    next_id = 0
    last_source = None
    engine = None
    if spec['stack'].startswith('zmq'):
        driver = RemotePolicy(NAMES, policy_name='example', action_horizon=spec['chunk'],
                              prefetch_steps=spec['prefetch'], blend_steps=spec['blend'])
        class Adapter:
            def get_action(self, observation):
                nonlocal next_id
                captures[next_id] = time.monotonic()
                next_id += 1
                result = client.get_action(observation)['actions']
                if driver._smoothing is None:
                    accepts.append((time.monotonic(), next_id - 1))
                return result
        driver._client = Adapter()
        if driver._smoothing:
            original_add = driver._smoothing.smoother.add_chunk
            def add_chunk(actions, observation_step, step, chunk_id, latency=0):
                accepted = original_add(actions, observation_step, step, chunk_id, latency)
                if accepted:
                    accepts.append((time.monotonic(), chunk_id))
                return accepted
            driver._smoothing.smoother.add_chunk = add_chunk
        def tick():
            action = driver.next_raw_action(lambda: obs)
            assert np.isfinite(list(action.values())).all()
            if not spec.get('real_forward'):
                assert np.array_equal(list(action.values()), np.zeros(6))
            if driver._smoothing:
                smoother = driver._smoothing.smoother
                return smoother.last_chunk_id, smoother.last_chunk_id >= 0 and smoother.remaining(
                    driver._smoothing.step - 1) > 0
            return next_id - 1, True
        def source(identifier):
            captured = captures.get(identifier)
            if driver._smoothing and driver._smoothing.config.blend_steps:
                smoother = driver._smoothing.smoother
                index = driver._smoothing.step - 1 - smoother.active.start if smoother.active else -1
                if 0 <= index < driver._smoothing.config.blend_steps - 1 and identifier > 0:
                    captured = min(captured, captures[identifier - 1])
            return captured
    else:
        from lerobot.inference import dispatch_tick
        from lerobot.remote_inference import RemoteInferenceEngine
        from lerobot.utils.action_interpolator import ActionInterpolator
        config = client.config
        engine = RemoteInferenceEngine(client, config, task=TASK)
        interpolator = ActionInterpolator(multiplier=1)
        original_accept = engine.runtime.accept
        def accept(request, chunk, **kwargs):
            accepted = original_accept(request, chunk, **kwargs)
            if accepted:
                accepts.append((time.monotonic(), request.request_id))
            return accepted
        engine.runtime.accept = accept
        def tick():
            engine.submit_observation(canonical(obs), capture_time=time.monotonic())
            target = dispatch_tick(engine, interpolator, send=lambda x: None, hold=lambda: None)
            if engine.failed:
                raise RuntimeError(engine.failure_traceback or engine.runtime.failure)
            current = engine.runtime.current
            if target is not None:
                assert np.isfinite(target.numpy()).all()
                if not spec.get('real_forward'):
                    assert np.array_equal(target.numpy(), np.zeros(6))
            return (current.request_id if current else None), target is not None
        def source(identifier):
            current = engine.runtime.current
            return current.capture_time if current and current.request_id == identifier else last_source
        engine.start()
        engine.resume()
    server.command('reset')
    meter.reset()
    cproc, sproc = psutil.Process(), psutil.Process(server.process.pid)
    c0, s0 = cpu_time(cproc), cpu_time(sproc)
    started = time.monotonic()
    count = round(spec['seconds'] * 30)
    held, missed, transitions, last_id = 0, 0, 0, None
    i = 0
    try:
        while i < count:
            due = started + i / 30
            time.sleep(max(0, due - time.monotonic()))
            call_start = time.monotonic()
            identifier, fresh = tick()
            now = time.monotonic()
            finish_slot = max(i, int((now - started) * 30))
            # Until the blocking call returns, the actuator retains the PREVIOUS target.
            while i < min(count, finish_slot):
                tick_rows.append([i, i / 30, 1, 1, last_id,
                                  None if last_source is None else (started + i / 30 - last_source) * 1000, 0])
                held += 1
                missed += 1
                i += 1
            if i >= count:
                break
            if fresh:
                if identifier != last_id and last_id is not None:
                    transitions += 1
                last_id = identifier
                last_source = source(identifier)
            else:
                held += 1
            tick_rows.append([i, (now - started), int(not fresh), 0, identifier,
                              None if last_source is None else (now - last_source) * 1000,
                              (now - call_start) * 1000])
            i += 1
        time.sleep(max(0, started + spec['seconds'] - time.monotonic()))
        elapsed = time.monotonic() - started
        ages = [row[5] for row in tick_rows if row[5] is not None]
        # A final synchronous call may return after the last scheduled tick.
        # Preserve that arrival as evidence, but exclude it from the 60 s count.
        arrivals = [(when - started, identifier) for when, identifier in list(accepts)]
        in_window = [event for event in arrivals if 0 <= event[0] < spec['seconds']]
        result = {'ticks': count, 'held_ticks': held, 'missed_dispatch_ticks': missed,
                  'chunk_replacements': len(in_window) - bool(in_window), 'executed_source_transitions': transitions,
                  'late_chunk_arrivals': sum(event[0] >= spec['seconds'] for event in arrivals),
                  'action_age_ms': summary(ages), 'elapsed_s': elapsed,
                  'client_cpu_percent': 100 * (cpu_time(cproc) - c0) / elapsed,
                  'server_cpu_percent': 100 * (cpu_time(sproc) - s0) / elapsed,
                  'effective_refill_s': engine.runtime.effective_refill if engine else None,
                  'tick_call_ms': summary([row[6] for row in tick_rows if not row[3]]),
                  'raw': {'ticks': tick_rows, 'chunk_arrivals': arrivals}}
        return result
    finally:
        if engine:
            engine.stop()


def run_case(spec):
    import torch
    torch.set_num_threads(1)
    meter = Meter()
    if spec['stack'].startswith('zmq'):
        instrument_zmq(spec['stack'], meter)
    else:
        instrument_lerobot(meter)
    spec['port'] = free_port()
    server = Server(spec)
    client = None
    try:
        if spec['stack'].startswith('zmq'):
            from core.policy_client.transport import PolicyClient
            client = PolicyClient(host='127.0.0.1', port=spec['port'])
            assert client.ping()
        else:
            client, _ = remote_client(spec)
        obs = payload(spec['cameras'], spec['height'], spec['width'])
        result = (transport if spec['mode'] == 'A' else control)(spec, server, client, meter, obs)
        result['spec'] = spec
        return result
    finally:
        if client:
            client.close()
        server.close()


def matrix(args):
    for height, width in [(480, 640), (224, 224)]:
        for cameras in [1, 2, 3]:
            for chunk in [16, 30, 50]:
                for stack, encoding in [('zmq_npy', 'raw'), ('zmq_raw', 'raw'),
                                         ('lerobot', 'raw'), ('lerobot', 'jpeg')]:
                    common = dict(height=height, width=width, cameras=cameras, chunk=chunk,
                                  stack=stack, encoding=encoding)
                    if args.mode == 'A':
                        yield dict(common, mode='A', infer_ms=0, setup='sync')
                    else:
                        for infer_ms in [50, 150]:
                            if stack == 'zmq_raw':
                                continue  # Raw codec is the transport-only ablation.
                            setups = [('sync', 0, 0), ('prefetch', min(chunk - 1, int(np.ceil(infer_ms / 1000 * 30)) + 2), 2)]
                            if stack == 'lerobot':
                                for setup, refill in [('default', .5), ('tuned', infer_ms / 1000 + 2 / 30)]:
                                    yield dict(common, mode='B', infer_ms=infer_ms, setup=setup, refill=refill)
                            else:
                                for setup, prefetch, blend in setups:
                                    yield dict(common, mode='B', infer_ms=infer_ms, setup=setup,
                                               prefetch=prefetch, blend=blend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['A', 'B'], default='A')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, default=HERE / 'results/localhost')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--filter', default='', help='Substring in case name')
    parser.add_argument('--case', help=argparse.SUPPRESS)
    parser.add_argument('--capture', action='store_true', help='Capture loopback TCP packets; requires tcpdump privileges')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--real-forward', action='store_true', help='Include ACT model compute (GPU follow-up)')
    parser.add_argument('--netem-ms', type=float, default=0, help='Metadata; apply tc externally in isolated namespace')
    args = parser.parse_args()
    if args.case:
        spec = json.loads(args.case)
        result = run_case(spec)
        raw = result.pop('raw')
        destination = Path(spec['destination'])
        if 'ticks' in raw:
            with gzip.open(destination.with_suffix('.ticks.csv.gz'), 'wt') as f:
                writer = csv.writer(f)
                writer.writerow(['tick', 'dispatch_elapsed_s', 'held', 'missed_deadline', 'source_id',
                                 'age_ms', 'call_ms'])
                writer.writerows(raw['ticks'])
            with gzip.open(destination.with_suffix('.chunks.csv.gz'), 'wt') as f:
                writer = csv.writer(f)
                writer.writerow(['accepted_elapsed_s', 'source_id'])
                writer.writerows(raw['chunk_arrivals'])
        else:
            with gzip.open(destination.with_suffix('.samples.json.gz'), 'wt') as f:
                json.dump(raw, f)
        destination.write_text(json.dumps(result, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env_path = args.output / 'environment.json'
    if not env_path.exists():
        env_path.write_text(json.dumps(environment(), indent=2))
    completed = 0
    for spec in matrix(args):
        name = (f"{spec['mode']}-{spec['height']}x{spec['width']}-c{spec['cameras']}-h{spec['chunk']}-"
                f"{spec['stack']}-{spec['encoding']}-d{spec['infer_ms']}-{spec['setup']}")
        if args.filter not in name:
            continue
        destination = args.output / (name + '.json')
        if destination.exists():
            continue
        spec.update(iterations=args.iterations, warmup=args.warmup, seconds=args.seconds,
                    checkpoint_dir=str(HERE / 'checkpoints'), device=args.device, real_forward=args.real_forward,
                    netem_ms=args.netem_ms, capture=args.capture, config_path=str(args.output / (name + '.server.json')),
                    destination=str(destination))
        print('START', name, flush=True)
        with (args.output / (name + '.log')).open('w') as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case', json.dumps(spec)],
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            print((args.output / (name + '.log')).read_text(), file=sys.stderr)
            raise SystemExit(f'Cell failed: {name}')
        print('DONE', name, flush=True)
        completed += 1
        if args.limit and completed >= args.limit:
            break

if __name__ == '__main__':
    main()
