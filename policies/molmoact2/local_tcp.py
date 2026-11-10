"""Client for the trusted local molmo_serve.py LeRobot socket server.

That server performs SO-101 model-frame transforms itself. Send calibrated
LeRobot degrees, and decode its postprocessed degrees back into device units.
The pickle protocol is restricted to loopback; never expose it to untrusted peers.
"""
from collections import deque
import io
import pickle
import socket
import struct

import numpy as np


class LocalLeRobotPolicy:
    def __init__(self, arm_info, *, port=5005, instruction='', horizon=30):
        from policies.molmoact2.client import MolmoAct2ClientPolicy
        from core.policy_client.discovery import get_manifest

        self.codec = MolmoAct2ClientPolicy.from_manifest(
            get_manifest('molmoact2'), {'input_joint_units': 'lerobot_normalized'})
        self.codec.configure(arm_info)
        self.keys = list(arm_info.feature_names)
        self.port, self.instruction, self.horizon = port, instruction, horizon
        self.queue = deque()
        self.socket = None

    def connect(self):
        self.socket = socket.create_connection(('127.0.0.1', self.port), timeout=30)

    def reset(self):
        self.queue.clear()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _read(self, count):
        data = bytearray()
        while len(data) < count:
            block = self.socket.recv(count - len(data))
            if not block:
                raise ConnectionError('Molmo server disconnected')
            data.extend(block)
        return bytes(data)

    def get_raw_action(self, state, images):
        if not self.queue:
            request = self.codec.build_request(state, images, self.instruction)
            # Inverse of the server's own SO-101 frame correction.
            signs = np.asarray([1, -1, 1, 1, 1, 1], dtype=np.float32)
            offsets = np.asarray([0, 90, 90, 0, 0, 0], dtype=np.float32)
            degrees = (request['state'] - offsets) / signs
            raw = dict(zip(self.keys, map(float, degrees)))
            raw.update(cam0=images['room'], cam1=images['wrist'], task=self.instruction)
            features = {'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': self.keys}}
            for camera in ('cam0', 'cam1'):
                features[f'observation.images.{camera}'] = {
                    'dtype': 'image', 'shape': raw[camera].shape,
                    'names': ['height', 'width', 'channels'], 'info': {'is_depth_map': False}}
            payload = pickle.dumps({'raw_obs': raw, 'lerobot_features': features})
            self.socket.sendall(struct.pack('>I', len(payload)) + payload)
            count = struct.unpack('>I', self._read(4))[0]
            if not 0 < count < 16 * 1024 * 1024:
                raise ValueError('Invalid local Molmo response size')
            response = _ArrayUnpickler(io.BytesIO(self._read(count))).load()
            model = np.asarray(response['chunk'], dtype=np.float32) * signs + offsets
            actions = self.codec.parse_response({'actions': model}, self.keys)
            if len(actions) == 0:
                raise ValueError('Empty Molmo action chunk')
            self.queue.extend(dict(zip(self.keys, map(float, row))) for row in actions[:self.horizon])
        return self.queue.popleft()


class _ArrayUnpickler(pickle.Unpickler):
    """Only accept the NumPy payload classes used by this local server."""
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'), ('numpy', 'dtype'),
            ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
            ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported response class {module}.{name}')
        return super().find_class(module, name)
