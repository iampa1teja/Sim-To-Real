"""CPU-only checkpoint validation; fixtures contain a few bytes, never model weights.

Set GROOT_N17_CHECKOUT to test configuration registration against actual pinned
N1.7 types in an isolated subprocess. This does not load a model or dataset.
"""
import copy
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / 'scripts/gr00t_model_profiles.py'
spec = importlib.util.spec_from_file_location('workshop_gr00t_profiles', MODULE_PATH)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n')


def tiny_shard(path, tensor_name, values=(1.0, 2.0)):
    """Write the safetensors binary format with one real F32 CPU tensor."""
    payload = struct.pack('<' + 'f' * len(values), *values)
    header = json.dumps({tensor_name: {'dtype': 'F32', 'shape': [len(values)],
                                      'data_offsets': [0, len(payload)]}}).encode()
    header += b' ' * (-len(header) % 8)
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)


class CheckpointProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gr00t-profile-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.modalities = {
            'video': {'modality_keys': ['room', 'wrist'], 'delta_indices': [0]},
            'state': {'modality_keys': ['single_arm', 'gripper'], 'delta_indices': [0]},
            'action': {'modality_keys': ['single_arm', 'gripper'], 'delta_indices': list(range(16)),
                       'action_configs': [dict(rep='ABSOLUTE', type='NON_EEF', format='DEFAULT', state_key=None)
                                          for _ in range(2)]},
            'language': {'modality_keys': ['annotation.human.task_description'], 'delta_indices': [0]},
        }
        self.config = {'model_type': 'Gr00tN1d7', 'action_horizon': 40}
        self.processor = {'processor_class': 'Gr00tN1d7Processor', 'processor_kwargs': {
            'max_action_horizon': 40, 'modality_configs': {'new_embodiment': self.modalities}}}
        self.statistics = {'new_embodiment': {
            kind: {key: {name: [0.0] * size for name in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
                   for key, size in (('single_arm', 5), ('gripper', 1))}
            for kind in ('state', 'action')}}
        self.shards = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
        self.index = {'metadata': {'total_size': 12},
                      'weight_map': {'arm.weight': self.shards[0], 'gripper.weight': self.shards[1]}}
        self.write_metadata()
        tiny_shard(self.root / self.shards[0], 'arm.weight')
        tiny_shard(self.root / self.shards[1], 'gripper.weight', (3.0,))

    def write_metadata(self):
        for name, value in [('config.json', self.config), ('processor_config.json', self.processor),
                            ('embodiment_id.json', {'new_embodiment': 10}),
                            ('statistics.json', self.statistics), ('model.safetensors.index.json', self.index)]:
            write_json(self.root / name, value)

    def test_valid_checkpoint_contract_and_read_only_inspection(self):
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        report = profiles.inspect_so_arm_checkpoint(self.root)
        self.assertEqual(report['embodiment_tag'], 'NEW_EMBODIMENT')
        self.assertEqual(report['embodiment_id'], 10)
        self.assertEqual(report['video_keys'], ['room', 'wrist'])
        self.assertEqual(report['state_dimensions'], {'single_arm': 5, 'gripper': 1})
        self.assertEqual(report['action_representation'], {'single_arm': 'ABSOLUTE', 'gripper': 'ABSOLUTE'})
        self.assertEqual(report['embodiment_action_horizon'], 16)
        self.assertEqual(report['model_action_capacity'], 40)
        self.assertEqual(report['weight_shards'], self.shards)
        self.assertTrue(report['weights_checked'])
        self.assertEqual({path.name: path.read_bytes() for path in self.root.iterdir()}, before)

    def test_profile_pins_and_legacy_representation_are_distinct(self):
        self.assertEqual(profiles.PROFILES['so100_n16']['revision'], 'ead52833afbbf4243f8cd5e7664f48a94de03b19')
        self.assertEqual(profiles.PROFILES['so_arm_n17']['revision'], '51d4c89f72fda44cbf77285c6a8114b52676b8a1')
        self.assertEqual(profiles.PROFILES['so100_n16']['cameras'], ['front', 'wrist'])
        self.assertTrue(profiles.PROFILES['so100_n16']['relative_actions'])
        self.assertEqual(profiles.PROFILES['so_arm_n17']['cameras'], ['room', 'wrist'])
        self.assertFalse(profiles.PROFILES['so_arm_n17']['relative_actions'])
        self.assertEqual(profiles.PROFILES['so_arm_n17']['video_backend'], 'torchcodec')

    @unittest.skipUnless(importlib.util.find_spec('safetensors'), 'Optional safetensors CPU reader unavailable')
    def test_tiny_weight_fixtures_are_valid_safetensors(self):
        from safetensors.numpy import load_file
        self.assertEqual(load_file(self.root / self.shards[0])['arm.weight'].tolist(), [1.0, 2.0])
        self.assertEqual(load_file(self.root / self.shards[1])['gripper.weight'].tolist(), [3.0])

    def test_rejects_model_or_processor_version_mismatch(self):
        for filename, key, value, message in (
            ('config.json', 'model_type', 'Gr00tN1d6', 'N1.7 model_type'),
            ('processor_config.json', 'processor_class', 'Gr00tN1d6Processor', 'N1.7 processor'),
        ):
            with self.subTest(filename=filename):
                self.write_metadata()
                data = json.loads((self.root / filename).read_text())
                data[key] = value
                write_json(self.root / filename, data)
                with self.assertRaisesRegex(ValueError, message):
                    profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_camera_and_lowdim_key_mismatch(self):
        for kind, keys in [('video', ['front', 'wrist']), ('video', ['room']),
                           ('state', ['single_arm']), ('action', ['gripper', 'single_arm']),
                           ('language', ['task'])]:
            with self.subTest(kind=kind, keys=keys):
                processor = copy.deepcopy(self.processor)
                processor['processor_kwargs']['modality_configs']['new_embodiment'][kind]['modality_keys'] = keys
                write_json(self.root / 'processor_config.json', processor)
                with self.assertRaisesRegex(ValueError, f'Unexpected {kind} keys'):
                    profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_horizon_mismatch(self):
        for kind, horizon in [('action', list(range(8))), ('action', list(range(40))),
                              ('video', [-1, 0]), ('state', [1]), ('language', [])]:
            with self.subTest(kind=kind, horizon=horizon):
                processor = copy.deepcopy(self.processor)
                processor['processor_kwargs']['modality_configs']['new_embodiment'][kind]['delta_indices'] = horizon
                write_json(self.root / 'processor_config.json', processor)
                with self.assertRaisesRegex(ValueError, f'Unexpected {kind} horizon'):
                    profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_model_capacity_mismatch(self):
        for filename, path in [('config.json', ('action_horizon',)),
                               ('processor_config.json', ('processor_kwargs', 'max_action_horizon'))]:
            with self.subTest(filename=filename):
                self.write_metadata()
                data = json.loads((self.root / filename).read_text())
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = 16
                write_json(self.root / filename, data)
                with self.assertRaisesRegex(ValueError, 'model capacity 40'):
                    profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_relative_or_eef_actions_and_state_binding(self):
        for index in (0, 1):
            for key, value in [('rep', 'RELATIVE'), ('rep', 'DELTA'), ('type', 'EEF'),
                               ('format', 'XYZ_ROT6D'), ('state_key', 'single_arm')]:
                with self.subTest(index=index, key=key, value=value):
                    processor = copy.deepcopy(self.processor)
                    processor['processor_kwargs']['modality_configs']['new_embodiment']['action']['action_configs'][index][key] = value
                    write_json(self.root / 'processor_config.json', processor)
                    with self.assertRaisesRegex(ValueError, 'absolute joint targets'):
                        profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_statistics_dimension_mismatch(self):
        for kind in ('state', 'action'):
            for key in ('single_arm', 'gripper'):
                with self.subTest(kind=kind, key=key):
                    statistics = copy.deepcopy(self.statistics)
                    statistics['new_embodiment'][kind][key]['q99'].append(0.0)
                    write_json(self.root / 'statistics.json', statistics)
                    with self.assertRaisesRegex(ValueError, f'Unexpected {kind}.{key} dimension'):
                        profiles.inspect_so_arm_checkpoint(self.root)

    def test_rejects_wrong_embodiment_id(self):
        write_json(self.root / 'embodiment_id.json', {'new_embodiment': 0})
        with self.assertRaisesRegex(ValueError, 'ID 10'):
            profiles.inspect_so_arm_checkpoint(self.root)

    def test_missing_weight_shard_rejected_but_metadata_only_allowed(self):
        (self.root / self.shards[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            profiles.inspect_so_arm_checkpoint(self.root)
        self.assertFalse(profiles.inspect_so_arm_checkpoint(self.root, check_weights=False)['weights_checked'])

    def test_truncated_weights_and_invalid_header_are_rejected(self):
        path = self.root / self.shards[0]
        original = path.read_bytes()
        for data in (b'', original[:7], struct.pack('<Q', 0), struct.pack('<Q', 16 * 1024 * 1024 + 1),
                     original[:-1], original + b'extra', original[:9]):
            with self.subTest(size=len(data)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    profiles.inspect_so_arm_checkpoint(self.root)

    def test_missing_indexed_tensor_rejected(self):
        tiny_shard(self.root / self.shards[0], 'unrelated.weight')
        with self.assertRaisesRegex(ValueError, 'Indexed tensors missing'):
            profiles.inspect_so_arm_checkpoint(self.root)

    def test_invalid_or_empty_shard_index_rejected(self):
        for mapping in ({}, {'arm.weight': '../escape.safetensors'},
                        {'arm.weight': '/absolute.safetensors'}, {'arm.weight': 'optimizer.pt'}):
            with self.subTest(mapping=mapping):
                write_json(self.root / 'model.safetensors.index.json', {'weight_map': mapping})
                with self.assertRaises(ValueError):
                    profiles.inspect_so_arm_checkpoint(self.root, check_weights=False)

    def test_cli_reports_success_and_actionable_failure(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        command = [sys.executable, str(MODULE_PATH), '--checkpoint', str(self.root)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['model_version'], 'N1.7')
        (self.root / 'processor_config.json').unlink()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Checkpoint validation failed', result.stderr)
        self.assertIn('processor_config.json', result.stderr)

    def test_download_selects_pinned_root_weights_without_training_state(self):
        destination = self.root / 'space and $(literal) checkpoint'
        args = shlex.split(profiles.download_command(destination))
        self.assertEqual(args[:3], ['hf', 'download', 'nvidia/SO_ARM_Starter_Gr00tN17'])
        self.assertEqual(args[args.index('--revision') + 1], '93a5a88f78a395939f784a5fe3685184913dcc60')
        self.assertEqual(args[args.index('--local-dir') + 1], str(destination))
        patterns = [args[index + 1] for index, value in enumerate(args) if value == '--include']
        self.assertEqual(len(patterns), 6)
        # Current hf CLI consumes one pattern per --include. Every remaining
        # argument must belong to an option/value pair, not positional filenames.
        remaining = args[3:]
        self.assertEqual(len(remaining) % 2, 0)
        self.assertTrue(all(option in ('--revision', '--include', '--local-dir')
                            for option in remaining[::2]), remaining)
        selected = lambda filename: any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)
        for filename in ('config.json', 'processor_config.json', 'embodiment_id.json', 'statistics.json',
                         'model.safetensors.index.json', *self.shards):
            self.assertTrue(selected(filename), filename)
        for filename in ('optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'training_args.bin',
                         'checkpoint-100/config.json', 'checkpoint-100/model-00001-of-00002.safetensors'):
            self.assertFalse(selected(filename), filename)

    @unittest.skipUnless(os.environ.get('GROOT_N17_CHECKOUT'), 'Set GROOT_N17_CHECKOUT for actual pinned N1.7 types')
    def test_so_arm_config_registers_actual_n17_types(self):
        checkout = Path(os.environ['GROOT_N17_CHECKOUT']).resolve()
        revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(revision, profiles.N17_PIN)
        code = '''
import importlib.util, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import gr00t
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ModalityConfig
assert pathlib.Path(gr00t.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
spec = importlib.util.spec_from_file_location('workshop_so_arm', sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.so100_config
assert all(isinstance(value, ModalityConfig) for value in config.values())
assert MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value] is config
assert all(isinstance(value, ActionConfig) for value in config['action'].action_configs)
print(json.dumps({'video_keys': config['video'].modality_keys,
                  'action_keys': config['action'].modality_keys,
                  'horizon': config['action'].delta_indices,
                  'representations': [value.rep.name for value in config['action'].action_configs],
                  'state_keys': [value.state_key for value in config['action'].action_configs]}))
'''
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(checkout)}
        result = subprocess.run([sys.executable, '-c', code, str(checkout), str(REPO / 'scripts/so_arm_n17_config.py')],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['video_keys'], ['room', 'wrist'])
        self.assertEqual(report['action_keys'], ['single_arm', 'gripper'])
        self.assertEqual(report['horizon'], list(range(16)))
        self.assertEqual(report['representations'], ['ABSOLUTE', 'ABSOLUTE'])
        self.assertEqual(report['state_keys'], [None, None])


class SetupScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gr00t-setup-flags-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.checkout = self.root / 'checkout'
        self.checkout.mkdir()
        self.cuda = self.root / 'cuda'
        (self.cuda / 'bin').mkdir(parents=True)
        self.log = self.root / 'uv.jsonl'
        self.env = {**os.environ, 'GROOT': str(self.checkout), 'CUDA_HOME': str(self.cuda),
                    'PATH': f'{self.bin}:{os.environ["PATH"]}', 'FAKE_UV_LOG': str(self.log)}
        for key in ('FAKE_GIT_DIRTY', 'FAKE_GIT_REVISION'):
            self.env.pop(key, None)
        self.write_executable(self.cuda / 'bin/nvcc', '#!/bin/sh\nexit 0\n')
        self.write_executable(self.bin / 'python3.12', '#!/bin/sh\nexit 0\n')
        self.write_executable(self.bin / 'ffmpeg', '#!/bin/sh\nprintf "ffmpeg version 6.1.0\\n"\n')
        self.write_executable(self.bin / 'git', f'''#!{sys.executable}
import os, sys
if 'rev-parse' in sys.argv:
    print(os.environ.get('FAKE_GIT_REVISION', '{profiles.N17_PIN}'))
elif 'diff' in sys.argv:
    assert sys.argv[-3:] == ['--', 'pyproject.toml', 'uv.lock'], sys.argv
    sys.exit(int(os.environ.get('FAKE_GIT_DIRTY', '0')))
else:
    raise AssertionError(sys.argv)
''')
        # Exit immediately at uv: this fixture never installs dependencies or
        # launches the checkout interpreter, even if orchestration is broken.
        self.write_executable(self.bin / 'uv', f'''#!{sys.executable}
import json, os, sys
with open(os.environ['FAKE_UV_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(97)
''')

    @staticmethod
    def write_executable(path, contents):
        path.write_text(contents)
        path.chmod(0o755)

    def run_script(self, *extra):
        return subprocess.run(['bash', str(REPO / 'scripts/setup_gr00t_n17.sh'), *extra],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_uses_frozen_pinned_lock_for_current_platform(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 97, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls, [['sync', '--frozen', '--python', '3.12', '--project', str(self.checkout)]])
        self.assertNotIn('--locked', result.stdout)

    def test_dirty_dependency_files_and_wrong_revision_fail_before_uv(self):
        for variable, value, diagnostic in [('FAKE_GIT_DIRTY', '1', 'dependency files have local changes'),
                                            ('FAKE_GIT_REVISION', profiles.N16_PIN, 'checkout differs')]:
            with self.subTest(variable=variable):
                self.env[variable] = value
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.env.pop(variable)
        self.assertFalse(self.log.exists())

    def test_dry_run_does_not_call_uv(self):
        result = self.run_script('--dry_run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('uv sync --frozen --python 3.12', result.stdout)
        self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
