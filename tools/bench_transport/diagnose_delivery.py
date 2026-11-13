"""Observe independent requests after a cold-session timeout, without replaying them.

Diagnostic only: preserves admission traffic, all request outcomes, codec counters,
and packet lengths. Results are separate from the accepted latency matrix.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from bench import Server, free_port, remote_client, sync_query
from common import Meter, instrument_lerobot, payload
from wire import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = dict(stack='lerobot', encoding='raw', cameras=3, height=480, width=640, chunk=16,
                infer_ms=0, port=free_port(), checkpoint_dir=str(here / 'checkpoints'),
                config_path=str(args.output.with_suffix('.server.json')), device='cpu', real_forward=False)
    meter = Meter()
    instrument_lerobot(meter)
    server = Server(spec)
    client = capture = None
    outcomes = []
    try:
        # Start before admission so capture startup cannot pace the first observation.
        capture = Capture(args.output, spec['port'])
        client, config = remote_client(spec)
        server.command('reset')
        meter.reset()
        observation = payload(3, 480, 640)
        for index in range(args.requests):
            before = time.monotonic()
            outcome = {'request_id': index}
            try:
                actions = sync_query(spec, client, observation, index)
                assert actions.shape == (16, 6)
                outcome['status'] = 'ok'
            except TimeoutError as exc:
                outcome.update(status='timeout', error=str(exc))
            outcome['elapsed_ms'] = (time.monotonic() - before) * 1000
            outcome['client_encoded'] = len(meter.values['encode_ms'])
            outcome['client_decoded'] = len(meter.values['decode_ms'])
            stats = server.command('stats')
            outcome['server_encoded'] = len(stats['encode_ms'])
            outcome['server_decoded'] = len(stats['decode_ms'])
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
        result = {'spec': spec, 'action_timeout_s': config.action_timeout_s, 'outcomes': outcomes,
                  'client_codec': {key: list(values) for key, values in meter.values.items()},
                  'server_codec': stats,
                  'qdisc': subprocess.check_output(['tc', '-s', 'qdisc', 'show', 'dev', 'lo'], text=True),
                  'wire': capture.stop(args.requests, require_exchange=False),
                  'note': 'Diagnostic traffic includes admission. Each observation has a new request ID; '
                          'no request is replayed. Default transport settings and 5 s deadlines are retained.'}
        capture = None
        result['wire']['note'] = ('Diagnostic packet accounting includes admission and six independent observations; '
                                  'it is not a steady-state latency matrix cell. Loopback headers/ACKs are included.')
        args.output.write_text(json.dumps(result, indent=2))
    finally:
        if capture is not None:
            capture.stop(max(1, len(outcomes)), require_exchange=False)
        if client is not None:
            client.close()
        server.close()

if __name__ == '__main__':
    main()
