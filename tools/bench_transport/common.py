"""Shared payloads and measurements. Instrumentation is local to benchmark processes."""
from __future__ import annotations
import importlib.metadata as metadata
import platform
import subprocess
from pathlib import Path
import sys
import time
import numpy as np
import msgpack
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/core')]
NAMES = tuple(f'joint_{i}.pos' for i in range(6))
TASK = 'Pick up the red cube and place it in the bowl.'

class RawSerializer:
    @staticmethod
    def to_bytes(value):
        def encode(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype.hasobject or obj.dtype.fields:
                    raise ValueError('Only plain numeric arrays are supported')
                return {'__raw__': True, 'dtype': obj.dtype.str, 'shape': obj.shape,
                        'data': obj.tobytes(order='C')}
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(type(obj).__name__)
        return msgpack.packb(value, default=encode, use_bin_type=True)

    @staticmethod
    def from_bytes(value):
        def decode(obj):
            if obj.get('__raw__') is True:
                dtype = np.dtype(obj['dtype'])
                if dtype.hasobject or dtype.fields:
                    raise ValueError('Invalid array dtype')
                return np.frombuffer(obj['data'], dtype=dtype).reshape(obj['shape']).copy()
            return obj
        return msgpack.unpackb(value, raw=False, object_hook=decode)

class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {'encode_ms': [], 'decode_ms': [], 'encoded_bytes': [], 'decoded_bytes': []}

    def wrap(self, fn, operation):
        def measured(value, *args, **kwargs):
            start = time.perf_counter_ns()
            result = fn(value, *args, **kwargs)
            self.values[operation + '_ms'].append((time.perf_counter_ns() - start) / 1e6)
            self.values['encoded_bytes' if operation == 'encode' else 'decoded_bytes'].append(
                len(result if operation == 'encode' else value))
            return result
        return measured

    def summary(self):
        return {key: summary(values) for key, values in self.values.items()}


def summary(values):
    if not values:
        return {'n': 0}
    a = np.asarray(values)
    return {'n': len(a), 'mean': float(a.mean()), **{
        key: float(np.percentile(a, p)) for key, p in [('p50', 50), ('p95', 95), ('p99', 99)]},
        'max': float(a.max())}


def instrument_zmq(variant, meter):
    from core.policy_client.transport import MsgSerializer
    serializer = RawSerializer if variant == 'zmq_raw' else MsgSerializer
    class MeasuredSerializer(serializer):
        to_bytes = staticmethod(meter.wrap(serializer.to_bytes, 'encode'))
        from_bytes = staticmethod(meter.wrap(serializer.from_bytes, 'decode'))
    # Server imports the historical policy_client alias; patch both aliases if loaded.
    import core.policy_client.transport as client
    import core.policy_server.server as server
    client.MsgSerializer = server.MsgSerializer = MeasuredSerializer


def instrument_lerobot(meter):
    import lerobot.remote_inference.client
    import lerobot.remote_inference.server
    import lerobot.remote_inference.codec as codec
    originals = [(codec.encode_message, meter.wrap(codec.encode_message, 'encode')),
                 (codec.decode_message, meter.wrap(codec.decode_message, 'decode'))]
    # Decode is also imported by reply validation; replace references, not source files.
    for name, module in list(sys.modules.items()):
        if name.startswith('lerobot.') and module:
            for attr in ('encode_message', 'decode_message'):
                current = getattr(module, attr, None)
                for original, wrapped in originals:
                    if current is original:
                        setattr(module, attr, wrapped)


def payload(cameras, height, width):
    # Fixed seeded, high entropy images: deliberately no JPEG-friendly all-black input.
    rng = np.random.default_rng(20261114)
    return {'state': np.linspace(-1, 1, 6, dtype=np.float32),
            'images': {f'cam{i}': rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                       for i in range(cameras)}, 'instruction': TASK}


def canonical(observation):
    return {'observation.state': observation['state'], **{
        'observation.images.' + name: value for name, value in observation['images'].items()}}


def environment():
    return {'implementation_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'platform': platform.platform(), 'os_release': platform.freedesktop_os_release(),
            'python': sys.version, 'machine': platform.machine(),
            'cpu': Path('/proc/cpuinfo').read_text().split('model name\t: ')[1].splitlines()[0],
            'logical_cpus': psutil.cpu_count(), 'physical_cpus': psutil.cpu_count(logical=False),
            'ram_bytes': psutil.virtual_memory().total,
            'versions': {p: metadata.version(p) for p in
                         ['numpy', 'torch', 'torchvision', 'pyzmq', 'msgpack', 'eclipse-zenoh', 'lerobot', 'psutil']}}
