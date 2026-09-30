"""CPU transport compatibility with N1.6 NPY and N1.7 msgpack_numpy wires."""
import ast
from collections import deque
import importlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAVE_TRANSPORT = all(importlib.util.find_spec(name) is not None for name in ('msgpack', 'zmq'))
HAVE_MSGPACK_NUMPY = importlib.util.find_spec('msgpack_numpy') is not None
msgpack = mnp = MsgSerializer = ModalityConfig = None


def setUpModule():
    global msgpack, mnp, MsgSerializer, ModalityConfig
    if not HAVE_TRANSPORT:
        return
    msgpack = importlib.import_module('msgpack')
    if HAVE_MSGPACK_NUMPY:
        mnp = importlib.import_module('msgpack_numpy')
    with patch.object(sys, 'path', [str(ROOT / 'source'), *sys.path]):
        transport = importlib.import_module('sim_to_real_so101.gr00t_client.server_client')
    MsgSerializer = transport.MsgSerializer
    ModalityConfig = transport.ModalityConfig


@unittest.skipUnless(HAVE_TRANSPORT, 'msgpack/pyzmq unavailable; run in the GR00T environment')
class Gr00tTransportTests(unittest.TestCase):
    def decode_envelope(self, payload):
        return MsgSerializer.from_bytes(msgpack.packb(payload))

    def assert_array_equal(self, actual, expected):
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_old_npy_requests_and_modality_roundtrip_remain_unchanged(self):
        state = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
        camera = np.arange(72, dtype=np.uint8).reshape(1, 1, 4, 6, 3)
        modality = ModalityConfig(delta_indices=list(range(16)), modality_keys=['arm', 'gripper'])
        request = {'endpoint': 'get_action', 'data': {'state': state, 'camera': camera}, 'modality': modality}
        packed = MsgSerializer.to_bytes(request)
        wire = msgpack.unpackb(packed, raw=False)
        self.assertEqual(wire['data']['state']['__ndarray_class__'], True)
        self.assertNotIn(b'nd', wire['data']['state'])
        self.assertTrue(wire['data']['state']['as_npy'].startswith(b'\x93NUMPY'))
        self.assert_array_equal(np.load(io.BytesIO(wire['data']['state']['as_npy']), allow_pickle=False), state)
        self.assertIn('__ModalityConfig_class__', wire['modality'])
        actual = MsgSerializer.from_bytes(packed)
        self.assert_array_equal(actual['data']['state'], state)
        self.assert_array_equal(actual['data']['camera'], camera)
        self.assertEqual(actual['modality'], modality)

    @unittest.skipUnless(HAVE_MSGPACK_NUMPY, 'msgpack_numpy needed only to generate official response fixtures')
    def test_real_msgpack_numpy_state_cameras_and_action_matrix(self):
        state = np.linspace(-1, 1, 6, dtype=np.float32).reshape(1, 1, 6)
        camera = np.arange(72, dtype=np.uint8).reshape(1, 1, 4, 6, 3)
        wrist = camera[:, :, :, ::-1, :]  # Encoder must also handle a non-contiguous view.
        action = np.linspace(-100, 100, 96, dtype=np.float32).reshape(1, 16, 6)
        response = [{'arm': action[..., :5], 'gripper': action[..., 5:]},
                    {'state': state, 'room': camera, 'wrist': wrist}]
        # This is the same numeric encoder used by the official N1.7 serializer.
        actual = MsgSerializer.from_bytes(msgpack.packb(response, default=mnp.encode))
        for key, expected in response[0].items():
            self.assert_array_equal(actual[0][key], expected)
        for key, expected in response[1].items():
            self.assert_array_equal(actual[1][key], expected)

    @unittest.skipUnless(HAVE_MSGPACK_NUMPY, 'msgpack_numpy needed only to generate official response fixtures')
    def test_real_msgpack_numpy_scalars_empty_and_structured_numeric_arrays(self):
        for scalar in (np.float32(1.25), np.float64(-2.5), np.int64(-7), np.uint32(11),
                       np.bool_(True), np.complex64(1 + 2j)):
            with self.subTest(scalar=scalar):
                actual = self.decode_envelope(mnp.encode(scalar))
                self.assertIsInstance(actual, np.generic)
                self.assertEqual(actual.dtype, scalar.dtype)
                self.assertEqual(actual, scalar)
        arrays = [np.zeros((0, 6), dtype=np.float32), np.array(3.5, dtype=np.float32),
                  np.zeros(2, dtype=[('position', [('x', '<f4'), ('y', '<f4')]), ('indices', 'u1', (2,))])]
        for array in arrays:
            with self.subTest(dtype=array.dtype, shape=array.shape):
                self.assert_array_equal(self.decode_envelope(mnp.encode(array)), array)
        self.assertEqual(self.decode_envelope(mnp.encode(1 + 2j)), 1 + 2j)

    @unittest.skipUnless(HAVE_MSGPACK_NUMPY, 'msgpack_numpy needed only to generate official response fixtures')
    def test_remote_policy_horizon_eight_and_sixteen_query_frequency_and_absolute_actions(self):
        # Load the real policy class without importing LeRobot hardware modules.
        path = ROOT / 'source/sim_to_real_so101/utils/lerobot_interface.py'
        tree = ast.parse(path.read_text())
        policy_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                            and node.name == 'GR00TRemotePolicy')
        context = dict(np=np, deque=deque, torch=SimpleNamespace(Tensor=object), LeRobotSO101Interface=object)
        exec(compile(ast.Module(body=[policy_class], type_ignores=[]), str(path), 'exec'), context)
        joint_names = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
        iface = SimpleNamespace(SO101_JOINT_ORDER=joint_names,
                                get_raw_actions_tensor=lambda row: np.array([row[key] for key in joint_names]),
                                get_mapped_actions_vectorized=np.deg2rad)
        for horizon in (8, 16):
            with self.subTest(horizon=horizon):
                policy = context['GR00TRemotePolicy'](iface, action_horizon=horizon)
                requests = []

                def query(observation):
                    requests.append(observation)
                    chunk = (np.arange(16, dtype=np.float32)[:, None]
                             + np.arange(6, dtype=np.float32)[None, :] / 10 + (len(requests) - 1) * 20)[None]
                    reply = [{'single_arm': chunk[..., :5], 'gripper': chunk[..., 5:]}, {}]
                    return MsgSerializer.from_bytes(msgpack.packb(reply, default=mnp.encode))

                policy._client = SimpleNamespace(get_action=query)
                policy._sim_obs_to_groot_inputs = lambda joints, visual: {'observed_joints': joints}
                for step in range(16):
                    action = policy.get_action(np.full(6, step), {})
                    expected_degrees = (step // horizon) * 20 + step % horizon + np.arange(6) / 10
                    np.testing.assert_allclose(action, np.deg2rad(expected_degrees), atol=1e-7)
                self.assertEqual(len(requests), 16 // horizon)
                self.assertEqual([request['observed_joints'][0] for request in requests], list(range(0, 16, horizon)))
                self.assertEqual(len(policy._action_queue), 0)

    def test_numpy_envelopes_accept_string_or_byte_keys_and_dtype_strings(self):
        array = np.array([1.25, -2.5], dtype='>f4')
        for key_type in (str, lambda name: name.encode()):
            for dtype in (array.dtype.str, array.dtype.str.encode()):
                with self.subTest(key_type=key_type, dtype=dtype):
                    payload = {key_type('nd'): True, key_type('type'): dtype, key_type('kind'): b'',
                               key_type('shape'): list(array.shape), key_type('data'): array.tobytes()}
                    self.assert_array_equal(self.decode_envelope(payload), array)

    def test_both_modality_markers_accept_mapping_json_string_and_bytes(self):
        payload = {'delta_indices': list(range(16)), 'modality_keys': ['arm', 'gripper'],
                   'action_configs': [{'rep': 'ABSOLUTE', 'type': 'NON_EEF', 'format': 'DEFAULT'}] * 2}
        expected = ModalityConfig(**payload)
        for marker in ('__ModalityConfig__', b'__ModalityConfig__',
                       '__ModalityConfig_class__', b'__ModalityConfig_class__'):
            for value in (payload, json.dumps(payload), json.dumps(payload).encode()):
                with self.subTest(marker=marker, payload_type=type(value)):
                    key = b'as_json' if isinstance(marker, bytes) else 'as_json'
                    self.assertEqual(self.decode_envelope({marker: True, key: value}), expected)

    def test_missing_modality_and_npy_payloads_raise_clear_errors(self):
        for marker in ('__ModalityConfig__', b'__ModalityConfig__',
                       '__ModalityConfig_class__', b'__ModalityConfig_class__'):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'as_json.*missing'):
                self.decode_envelope({marker: True})
        for marker in ('__ndarray_class__', b'__ndarray_class__'):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'as_npy.*missing'):
                self.decode_envelope({marker: True})
        with self.assertRaisesRegex(ValueError, 'must contain an object'):
            self.decode_envelope({'__ModalityConfig__': True, 'as_json': '[]'})

    def test_object_and_nested_structured_object_dtypes_refused_before_buffer_or_pickle(self):
        descriptors = [('|O', b''), ('<f4', b'O'),
                       ([['value', '|O']], b'V'),
                       ([[b'nested', [[b'value', b'|O']]], ['numeric', '<f4']], b'V'),
                       ([['nested', [['value', '|O']], [2]]], b'')]
        for dtype, kind in descriptors:
            for nd in (True, False, 1):
                for key_type in (str, lambda name: name.encode()):
                    with self.subTest(dtype=dtype, kind=kind, nd=nd, key_type=key_type):
                        payload = {key_type('nd'): nd, key_type('type'): dtype, key_type('kind'): kind,
                                   key_type('shape'): [1], key_type('data'): b'not-pickle-or-object-pointers'}
                        with patch.object(pickle, 'loads') as unpickle, patch.object(np, 'frombuffer') as buffer:
                            with self.assertRaisesRegex(ValueError, 'Refusing.*object'):
                                self.decode_envelope(payload)
                            unpickle.assert_not_called()
                            buffer.assert_not_called()

    def test_legacy_object_arrays_retain_allow_pickle_false_boundary(self):
        for array in (np.array([object()], dtype=object), np.zeros(1, dtype=[('value', object)])):
            with self.subTest(dtype=array.dtype), self.assertRaises(ValueError):
                MsgSerializer.to_bytes(array)
        output = io.BytesIO()
        np.lib.format.write_array_header_1_0(output, {'descr': '|O', 'fortran_order': False, 'shape': (1,)})
        output.write(b'not-a-pickle')
        with patch.object(pickle, 'load') as unpickle, self.assertRaisesRegex(ValueError, 'allow_pickle=False'):
            self.decode_envelope({'__ndarray_class__': True, 'as_npy': output.getvalue()})
        unpickle.assert_not_called()

    def test_malformed_numeric_envelopes_fail_and_plain_dicts_pass_through(self):
        valid = {b'nd': True, b'type': '<f4', b'kind': b'', b'shape': [2],
                 b'data': np.ones(2, dtype=np.float32).tobytes()}
        malformed = [{key: value for key, value in valid.items() if key != missing}
                     for missing in (b'type', b'shape', b'data')]
        malformed += [{**valid, b'nd': 1}, {**valid, b'type': 'invalid'}, {**valid, b'shape': [-1]},
                      {**valid, b'shape': [True]}, {**valid, b'shape': [2.0]}, {**valid, b'shape': [3]},
                      {**valid, b'data': b'x'}, {**valid, b'nd': False}]
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.decode_envelope(payload)
        ordinary = {'status': 'ok', 'type': 'metadata', 'data': [1, 2], 'shape': [2]}
        self.assertEqual(self.decode_envelope(ordinary), ordinary)


if __name__ == '__main__':
    unittest.main()
