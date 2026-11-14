"""Real server implementations with benchmark-only fixed prediction hooks."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from threading import Thread
import time
import types
import numpy as np
from common import Meter, NAMES, TASK, canonical, instrument_zmq, instrument_lerobot

class DummyPolicy:
    def __init__(self, chunk, infer_ms):
        self.chunk, self.infer_ms = chunk, infer_ms

    def infer(self, request):
        # Production server passes its nested get_action data to infer.
        observation = request if getattr(self, 'direct_observation', False) else request['observation']
        assert observation['state'].shape == (6,)
        if self.infer_ms:
            time.sleep(self.infer_ms / 1000)
        if getattr(self, 'runner', None) is not None:
            from lerobot.inference import ObservationSnapshot
            snapshot = ObservationSnapshot(canonical(observation), time.monotonic(), observation['instruction'])
            actions = self.runner.predict(snapshot).canonical_actions.cpu().numpy()
            return {'actions': actions}
        return {'actions': np.zeros((self.chunk, 6), dtype=np.float32)}

    def info(self):
        return {'name': 'benchmark', 'horizon': self.chunk}

    def reset(self):
        if getattr(self, 'runner', None) is not None:
            self.runner.reset(full=True)
        return {}


def lerobot_config(cameras, height, width, chunk, port, checkpoint, device='cpu'):
    from lerobot.inference import FeatureSpec
    from lerobot.remote_inference.configs import ServerConfig, ModelConfig, ExecutionConfig
    from lerobot.transport import ZenohConfig
    return ServerConfig(deployment='bench', model=ModelConfig(str(checkpoint), device=device, tf32=None),
        execution=ExecutionConfig(n_action_steps=chunk, warmup_calls=1),
        zenoh=ZenohConfig(listen_endpoints=[f'tcp/127.0.0.1:{port}']),
        semantics='benchmark-raw-joint-units-v1', robot_type='benchmark',
        features=[FeatureSpec('observation.state', (6,), 'float32', names=NAMES,
                              semantics='benchmark-raw-joint-units-v1'), *[
            FeatureSpec(f'observation.images.cam{i}', (height, width, 3), 'uint8', kind='rgb',
                        semantics=f'benchmark-camera-{i}-rgb-v1') for i in range(cameras)]],
        action_feature=FeatureSpec('action', (6,), 'float32', names=NAMES,
                                  semantics='benchmark-raw-joint-units-v1'))


def make_checkpoint(cameras, height, width, directory):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    directory = Path(directory) / f'act-{cameras}-{height}-{width}'
    if (directory / 'policy_postprocessor.json').exists():
        return directory
    torch.manual_seed(20261114)
    config = ACTConfig(input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)), **{
        f'observation.images.cam{i}': PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        for i in range(cameras)}},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={kind: NormalizationMode.IDENTITY for kind in
                               [FeatureType.STATE, FeatureType.VISUAL, FeatureType.ACTION]},
        device='cpu', chunk_size=50, n_action_steps=50, use_vae=False, dim_model=32,
        n_heads=2, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
        vision_backbone='resnet18', pretrained_backbone_weights=None)
    ACTPolicy(config).save_pretrained(directory)
    pre, post = make_act_pre_post_processors(config)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    return directory


def load_runner(spec):
    from lerobot.remote_inference.deployment import load_deployment
    checkpoint = make_checkpoint(spec['cameras'], spec['height'], spec['width'], spec['checkpoint_dir'])
    config = lerobot_config(spec['cameras'], spec['height'], spec['width'], spec['chunk'], spec['port'],
                            checkpoint, spec.get('device', 'cpu'))
    Path(spec['config_path']).write_text(json.dumps(asdict(config), indent=2))
    runner, identity = load_deployment(config)
    return config, runner, identity


def serve(spec, connection):
    import torch
    torch.set_num_threads(1)
    meter = Meter()
    if spec['stack'].startswith('zmq'):
        instrument_zmq(spec['stack'], meter)
        from core.policy_server.server import PolicyServer
        policy = DummyPolicy(spec['chunk'], spec['infer_ms'])
        if spec.get('setup') == 'async':
            # Public RemotePolicy admission uses the real example plugin schema.
            policy.info = lambda: {'name': 'example', 'horizon': spec['chunk']}
            policy.direct_observation = True
        if spec.get('real_forward'):
            _, policy.runner, _ = load_runner(spec)
        server = PolicyServer(policy, host='127.0.0.1', port=spec['port'])
        stop = server._kill_server
        run = server.run
    else:
        from lerobot.remote_inference.server import SessionWorker, PolicyServer
        from lerobot.transport import ZenohTransport
        config, runner, identity = load_runner(spec)
        if not spec.get('real_forward'):
            def fixed_prediction(policy, batch, **kwargs):
                if spec['infer_ms']:
                    time.sleep(spec['infer_ms'] / 1000)
                return torch.zeros((1, policy.config.chunk_size, 6), dtype=torch.float32,
                                   device=policy.config.device)
            runner.policy.predict_action_chunk = types.MethodType(fixed_prediction, runner.policy)
        elif spec['infer_ms']:
            original_prediction = runner.policy.predict_action_chunk
            def delayed_prediction(*args, **kwargs):
                time.sleep(spec['infer_ms'] / 1000)
                return original_prediction(*args, **kwargs)
            runner.policy.predict_action_chunk = delayed_prediction
        instrument_lerobot(meter)
        worker = SessionWorker(runner, deployment='bench', artifact_identity=identity, semantics=config.semantics)
        transport = ZenohTransport(config.zenoh)
        server = PolicyServer(worker, transport)
        stop, run = server.stop, server.serve
        original_token = transport.declare_token
        def token(key):
            result = original_token(key)
            connection.send({'ready': True})
            return result
        transport.declare_token = token
    def commands():
        while True:
            command = connection.recv()
            if command == 'reset':
                meter.reset()
                connection.send(True)
            elif command == 'stats':
                connection.send(meter.values)
            elif command == 'stop':
                stop()
                return
    Thread(target=commands, daemon=True).start()
    if spec['stack'].startswith('zmq'):
        connection.send({'ready': True})
    run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack', choices=['zmq', 'lerobot'], default='zmq')
    parser.add_argument('--cameras', type=int, default=1)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--real-forward', action='store_true')
    parser.add_argument('--port', type=int, default=5558)
    parser.add_argument('--chunk', type=int, default=30)
    parser.add_argument('--infer-ms', type=float, default=0)
    parser.add_argument('--serializer', choices=['npy', 'raw'], default='npy')
    args = parser.parse_args()
    if args.stack == 'lerobot':
        import multiprocessing
        parent, child = multiprocessing.Pipe()
        here = Path(__file__).resolve().parent
        spec = dict(stack='lerobot', cameras=args.cameras, height=args.height, width=args.width,
                    chunk=args.chunk, infer_ms=args.infer_ms, port=args.port,
                    checkpoint_dir=str(here / 'checkpoints'), config_path=str(here / 'standalone.server.json'),
                    device=args.device, real_forward=args.real_forward)
        serve(spec, child)
        raise SystemExit(0)
    from core.policy_server.server import PolicyServer
    if args.serializer == 'raw':
        instrument_zmq('zmq_raw', Meter())
    PolicyServer(DummyPolicy(args.chunk, args.infer_ms), host='127.0.0.1', port=args.port).run()
