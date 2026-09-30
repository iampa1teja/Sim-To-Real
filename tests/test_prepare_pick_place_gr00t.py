"""CPU integration tests. Set GROOT_CHECKOUT to the pinned GR00T checkout.

Set GROOT_N17_CHECKOUT and GROOT_N17_PYTHON (or supply checkout/.venv/bin/python)
for actual N1.7 reader/preparation integration. No dependency mocks are used.

Optional PREPARATION_BASELINE points to the pre-change preparation script for
byte-for-byte default-output regression testing (except provenance artifacts).
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
GROOT = os.environ.get('GROOT_CHECKOUT')
GROOT_N17 = os.environ.get('GROOT_N17_CHECKOUT')
N17_PIN = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def make_source_fixture(cls):
    import av
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    cls.temp = tempfile.TemporaryDirectory(prefix='pick-place-exclusion-')
    cls.root = Path(cls.temp.name)
    cls.source = cls.root / 'source'
    cls.filtered = cls.root / 'filtered'
    cls.default = cls.root / 'default'
    cls.keys = ['observation.images.realsense_rgb', 'observation.images.wrist_cam']
    cls.lengths = [20, 21, 22, 23, 24]
    cls.tables = []
    records = []
    offset = 0
    features = {key: {'dtype': 'float32', 'shape': [6]} for key in ('observation.state', 'action')}
    features.update({key: {'dtype': 'int64', 'shape': [1]}
                     for key in ('episode_index', 'index', 'frame_index', 'task_index')})
    features['timestamp'] = {'dtype': 'float32', 'shape': [1]}
    features.update({key: {'dtype': 'video', 'shape': [480, 640, 3]} for key in cls.keys})
    info = dict(codebase_version='v3.0', total_episodes=5, total_frames=sum(cls.lengths),
                total_tasks=3, chunks_size=2, fps=30, robot_type='so101_follower',
                splits={'train': '0:3', 'test': '3:5'}, features=features,
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    write_json(cls.source / 'meta/info.json', info)
    pd.DataFrame({'task_index': [0, 1, 2]}, index=['failed only', 'retained', 'also retained']).to_parquet(
        cls.source / 'meta/tasks.parquet')
    for ep, length in enumerate(cls.lengths):
        task = 0 if ep in (0, 3) else (2 if ep == 4 else 1)
        state = np.arange(length * 6, dtype=np.float32).reshape(length, 6) / 100 + ep * 10
        action = state + ep + np.arange(length, dtype=np.float32)[:, None] / 10
        table = pa.table({'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32(), 6)),
                          'action': pa.array(action.tolist(), type=pa.list_(pa.float32(), 6)),
                          'timestamp': pa.array(np.arange(length, dtype=np.float32) / 30),
                          'episode_index': np.full(length, ep, dtype=np.int64),
                          'index': np.arange(offset, offset + length, dtype=np.int64),
                          'frame_index': np.arange(length, dtype=np.int64),
                          'task_index': np.full(length, task, dtype=np.int64)})
        cls.tables.append(table)
        record = dict(episode_index=ep, length=length, tasks=[['failed only', 'retained', 'also retained'][task]],
                      dataset_from_index=offset, dataset_to_index=offset + length)
        record.update({'data/chunk_index': 0, 'data/file_index': int(ep >= 3),
                       'stats/index/mean': [offset + (length - 1) / 2],
                       'stats/episode_index/mean': [ep]})
        for camera, key in enumerate(cls.keys):
            path = cls.source / info['video_path'].format(video_key=key, chunk_index=0, file_index=ep)
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('libx264', rate=30)
                stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
                for frame in range(length):
                    pixels = np.full((480, 640, 3), 20 + ep * 30 + camera * 5 + frame, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            record.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': ep,
                           f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': length / 30})
        records.append(record)
        write_json(cls.source / f'pick_place_meta/episode_{ep:06d}.json',
                   {'episode_index': ep, 'cube_pose': [ep, 2, 3], 'success': ep not in (0, 3)})
        offset += length
    for file, tables in enumerate((cls.tables[:3], cls.tables[3:])):
        path = cls.source / f'data/chunk-000/file-{file:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(tables), path)
    path = cls.source / 'meta/episodes/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    # Valid but deliberately distinct global stats: filtering must not reuse these.
    stats = {}
    for key in ('observation.state', 'action', 'timestamp'):
        values = np.asarray(pa.concat_tables(cls.tables)[key].to_pylist())
        values = values.reshape(len(values), -1)
        stats[key] = {name: value.tolist() for name, value in (
            ('mean', values.mean(0)), ('std', values.std(0)), ('min', values.min(0)),
            ('max', values.max(0)), ('q01', np.quantile(values, .01, axis=0)),
            ('q99', np.quantile(values, .99, axis=0)))}
    write_json(cls.source / 'meta/stats.json', stats)
    cls.before = snapshot(cls.source)


@unittest.skipUnless(GROOT, 'Set GROOT_CHECKOUT for real GR00T CPU integration tests')
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_source_fixture(cls)
        cls.prepare(cls.filtered, '--exclude_episodes', '0,3')
        cls.prepare(cls.default)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def run_script(cls, script, *args, success=True):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.pathsep.join([GROOT, os.environ.get('PYTHONPATH', '')]))
        result = subprocess.run([sys.executable, str(script), *map(str, args)], env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if success and result.returncode:
            raise AssertionError(result.stdout)
        if not success and not result.returncode:
            raise AssertionError('Expected rejection: ' + result.stdout)
        return result.stdout

    @classmethod
    def prepare(cls, output, *args, success=True, script=None):
        return cls.run_script(script or REPO / 'scripts/prepare_pick_place_gr00t.py',
                              '--groot', GROOT, '--source', cls.source, '--output', output,
                              *args, success=success)

    def test_filtered_values_paths_sidecars_tasks_and_stats(self):
        import numpy as np
        import pyarrow.parquet as pq

        offset = 0
        for new, old in enumerate([1, 2, 4]):
            actual = pq.read_table(self.filtered / f'data/chunk-{new // 2:03d}/episode_{new:06d}.parquet')
            original = self.tables[old]
            for key in original.column_names:
                if key not in ('episode_index', 'index'):
                    self.assertTrue(actual[key].equals(original[key]), (new, old, key))
            np.testing.assert_array_equal(actual['episode_index'], np.full(self.lengths[old], new))
            np.testing.assert_array_equal(actual['index'], np.arange(offset, offset + self.lengths[old]))
            offset += self.lengths[old]
            for key in self.keys:
                src = self.source / f'videos/{key}/chunk-000/file-{old:03d}.mp4'
                dst = self.filtered / f'videos/chunk-{new // 2:03d}/{key}/episode_{new:06d}.mp4'
                self.assertEqual(src.read_bytes(), dst.read_bytes())
            expected = json.loads((self.source / f'pick_place_meta/episode_{old:06d}.json').read_text())
            expected['episode_index'] = new
            self.assertEqual(expected, json.loads((self.filtered / f'pick_place_meta/episode_{new:06d}.json').read_text()))
        self.assertEqual(len(list(self.filtered.glob('data/*/*.parquet'))), 3)
        self.assertEqual(len(list(self.filtered.glob('pick_place_meta/*.json'))), 3)
        info = json.loads((self.filtered / 'meta/info.json').read_text())
        self.assertEqual([info[k] for k in ('total_episodes', 'total_frames', 'total_tasks', 'total_chunks', 'total_videos')],
                         [3, offset, 2, 2, 6])
        self.assertEqual(info['splits'], {'train': '0:2', 'test': '2:3'})
        tasks = [json.loads(line) for line in (self.filtered / 'meta/tasks.jsonl').read_text().splitlines()]
        self.assertEqual([task['task_index'] for task in tasks], [1, 2])
        episodes = [json.loads(line) for line in (self.filtered / 'meta/episodes.jsonl').read_text().splitlines()]
        self.assertEqual([row['episode_index'] for row in episodes], [0, 1, 2])
        episode_stats = [json.loads(line) for line in (self.filtered / 'meta/episodes_stats.jsonl').read_text().splitlines()]
        offset = 0
        for new, old in enumerate((1, 2, 4)):
            self.assertEqual(episode_stats[new]['episode_index'], new)
            self.assertEqual(episode_stats[new]['stats']['episode_index']['mean'], [new])
            self.assertEqual(episode_stats[new]['stats']['index']['mean'], [offset + (self.lengths[old] - 1) / 2])
            offset += self.lengths[old]
        stats = json.loads((self.filtered / 'meta/stats.json').read_text())
        for key in ('observation.state', 'action'):
            values = np.vstack([np.asarray(self.tables[old][key].to_pylist(), dtype=np.float32) for old in (1, 2, 4)])
            for stat, expected in [('mean', values.mean(0)), ('std', values.std(0)), ('min', values.min(0)),
                                   ('max', values.max(0)), ('q01', np.quantile(values, .01, axis=0)),
                                   ('q99', np.quantile(values, .99, axis=0))]:
                np.testing.assert_allclose(stats[key][stat], expected)
        relative = []
        for old in (1, 2, 4):
            state = np.asarray(self.tables[old]['observation.state'].to_pylist())[:, :5]
            action = np.asarray(self.tables[old]['action'].to_pylist())[:, :5]
            relative.extend([(action[i:i + 16] - state[i]).astype(np.float32) for i in range(len(state) - 15)])
        stats = json.loads((self.filtered / 'meta/relative_stats.json').read_text())['single_arm']
        np.testing.assert_allclose(stats['mean'], np.mean(relative, axis=0), rtol=1e-6)
        report = json.loads((self.filtered / 'preparation_report.json').read_text())
        self.assertEqual(report['excluded'], [0, 3])
        self.assertEqual(report['index_map'], {'0': 1, '1': 2, '2': 4})
        self.assertTrue(report['groot_loader_all_episodes_checked'])
        self.assertEqual(snapshot(self.source), self.before)

    def test_verifier_every_episode(self):
        for output in (self.filtered, self.default):
            result = self.run_script(REPO / 'scripts/verify_pick_place_training.py', '--dataset', output)
            print(result.strip(), flush=True)

    def test_default_preserves_all_source_values(self):
        import pyarrow.parquet as pq
        for ep, expected in enumerate(self.tables):
            self.assertTrue(expected.equals(pq.read_table(self.default / f'data/chunk-{ep // 2:03d}/episode_{ep:06d}.parquet')))
        self.assertEqual((self.source / 'meta/stats.json').read_bytes(), (self.default / 'meta/stats.json').read_bytes())
        self.assertEqual(snapshot(self.source), self.before)

    @unittest.skipUnless(os.environ.get('PREPARATION_BASELINE'), 'Optional pre-change script not supplied')
    def test_default_matches_prechange_output(self):
        baseline = self.root / 'baseline'
        self.prepare(baseline, script=os.environ['PREPARATION_BASELINE'])
        ignored = {'prepare_pick_place_gr00t.py', 'prepare_pick_place_gr00t_baseline.py',
                   'preparation_report.json', 'prepared_sha256.json', 'gr00t_model_profiles.py'}
        self.assertEqual({k: v for k, v in snapshot(baseline).items() if k not in ignored},
                         {k: v for k, v in snapshot(self.default).items() if k not in ignored})
        reports = [json.loads((root / 'preparation_report.json').read_text()) for root in (baseline, self.default)]
        for report in reports:
            for key in ('time_utc', 'output'):
                report.pop(key)
        self.assertEqual(*reports)

    def test_resume_rejects_changed_selection_and_rebuilds_stats(self):
        output = self.root / 'resumed'
        building = output.with_name(output.name + '.building')
        shutil.copytree(self.filtered, building)
        before = snapshot(building)
        self.assertIn('same --exclude_episodes', self.prepare(output, '--resume', success=False))
        self.assertEqual(snapshot(building), before)
        for name in ('stats.json', 'relative_stats.json'):
            shutil.copy2(self.default / 'meta' / name, building / 'meta' / name)
        self.prepare(output, '--resume', '--exclude_episodes', '3,0,0')
        for name in ('stats.json', 'relative_stats.json'):
            self.assertEqual((output / 'meta' / name).read_bytes(), (self.filtered / 'meta' / name).read_bytes())
        print(self.run_script(REPO / 'scripts/verify_pick_place_training.py', '--dataset', output).strip())
        self.assertEqual(snapshot(self.source), self.before)

    def test_invalid_exclusions_and_source_hash_resume(self):
        for value in ('-1', '5', '0,1,2,3,4', 'a', '0,'):
            output = self.root / 'invalid'
            self.prepare(output, '--exclude_episodes', value, success=False)
            self.assertFalse(output.with_name(output.name + '.building').exists())
        output = self.root / 'bad-hash'
        building = output.with_name(output.name + '.building')
        shutil.copytree(self.filtered, building)
        write_json(building / 'source_sha256.json', {})
        self.prepare(output, '--resume', '--exclude_episodes', '0,3', success=False)
        self.assertEqual(snapshot(self.source), self.before)

    def test_n17_profile_rejects_the_legacy_checkout_before_creating_output(self):
        output = self.root / 'wrong-runtime'
        self.prepare(output, '--model_profile', 'so_arm_n17', success=False)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + '.building').exists())
        self.assertEqual(snapshot(self.source), self.before)

    def test_resume_rejects_a_different_recorded_model_profile(self):
        output = self.root / 'wrong-resume-profile'
        building = output.with_name(output.name + '.building')
        shutil.copytree(self.filtered, building)
        write_json(building / 'model_profile.json', {'model_profile': 'so_arm_n17'})
        before = snapshot(building)
        result = self.prepare(output, '--resume', '--exclude_episodes', '0,3', success=False)
        self.assertIn('same --model_profile', result)
        self.assertEqual(snapshot(building), before)
        self.assertEqual(snapshot(self.source), self.before)


@unittest.skipUnless(GROOT_N17, 'Set GROOT_N17_CHECKOUT and a matching N1.7 runtime for reader integration')
class N17PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkout = Path(GROOT_N17).resolve()
        cls.python = Path(os.environ.get('GROOT_N17_PYTHON', cls.checkout / '.venv/bin/python'))
        if not cls.python.is_file():
            raise unittest.SkipTest('N1.7 interpreter unavailable; set GROOT_N17_PYTHON after runtime setup')
        revision = subprocess.check_output(['git', '-C', str(cls.checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != N17_PIN:
            raise AssertionError(f'Expected pinned N1.7 source {N17_PIN}, got {revision}')
        # An explicitly selected runtime must match. An automatically discovered
        # environment may still be under installation, so gate it until ready.
        try:
            cls.run_code('''
import importlib.metadata as metadata, sys
assert sys.version_info[:2] == (3, 12), sys.version
for name, expected in [('torch', '2.9.0'), ('transformers', '4.57.3'), ('torchcodec', '0.8.0')]:
    actual = metadata.version(name).split('+')[0]
    assert actual == expected, (name, expected, actual)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
''')
        except AssertionError as error:
            if os.environ.get('GROOT_N17_PYTHON'):
                raise
            reason = str(error).strip().splitlines()[-1]
            raise unittest.SkipTest(f'N1.7 runtime unavailable: {reason}') from None
        make_source_fixture(cls)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.prepare(cls.filtered, '--exclude_episodes', '0,3')
        cls.prepare(cls.default)

    @classmethod
    def environment(cls):
        return {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(cls.checkout),
                'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    @classmethod
    def run_code(cls, code, *args):
        result = subprocess.run([str(cls.python), '-c', code, *map(str, args)], env=cls.environment(),
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result.stdout

    @classmethod
    def prepare(cls, output, *extra, success=True):
        result = subprocess.run([str(cls.python), str(REPO / 'scripts/prepare_pick_place_gr00t.py'),
                                 '--groot', str(cls.checkout), '--source', str(cls.source), '--output', str(output),
                                 '--model_profile', 'so_arm_n17', *map(str, extra)], env=cls.environment(),
                                capture_output=True, text=True, timeout=120)
        if (result.returncode == 0) != success:
            raise AssertionError(result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_profile_preserves_absolute_targets_stats_videos_and_source(self):
        import numpy as np
        import pyarrow.parquet as pq

        for output, retained in ((self.filtered, (1, 2, 4)), (self.default, (0, 1, 2, 3, 4))):
            report = json.loads((output / 'preparation_report.json').read_text())
            self.assertEqual(report['model_profile'], 'so_arm_n17')
            self.assertEqual(report['groot_revision'], N17_PIN)
            self.assertEqual(report['action_representation'], {'single_arm': 'ABSOLUTE', 'gripper': 'ABSOLUTE'})
            self.assertEqual(report['action_horizon'], 16)
            self.assertEqual(report['video_keys'], ['room', 'wrist'])
            self.assertEqual(report['groot_validation_video_backend'], 'torchcodec')
            self.assertTrue(report['groot_loader_all_episodes_checked'])
            self.assertFalse(report['relative_action_statistics_generated'])
            self.assertFalse((output / 'meta/relative_stats.json').exists())
            modality = json.loads((output / 'meta/modality.json').read_text())
            self.assertEqual(set(modality['video']), {'room', 'wrist'})
            self.assertEqual(modality['video']['room']['original_key'], self.keys[0])
            self.assertEqual(modality['video']['wrist']['original_key'], self.keys[1])
            for new, old in enumerate(retained):
                actual = pq.read_table(output / f'data/chunk-{new // 2:03d}/episode_{new:06d}.parquet')
                self.assertTrue(actual['action'].equals(self.tables[old]['action']))
                self.assertTrue(actual['observation.state'].equals(self.tables[old]['observation.state']))
                for key in self.keys:
                    source_video = self.source / f'videos/{key}/chunk-000/file-{old:03d}.mp4'
                    output_video = output / f'videos/chunk-{new // 2:03d}/{key}/episode_{new:06d}.mp4'
                    self.assertEqual(source_video.read_bytes(), output_video.read_bytes())
            expected = np.concatenate([np.asarray(self.tables[old]['action'].to_pylist()) for old in retained])
            stats = json.loads((output / 'meta/stats.json').read_text())['action']
            np.testing.assert_allclose(stats['mean'], expected.mean(0), rtol=1e-6)
            np.testing.assert_allclose(stats['q01'], np.quantile(expected, .01, axis=0), rtol=1e-6)
        self.assertEqual(snapshot(self.source), self.before)

    def test_actual_n17_reader_and_relative_stats_are_a_noop(self):
        code = '''
import importlib.util, json, pathlib, numpy as np, pandas as pd, sys
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('prepared_config', root / 'so100_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loader = LeRobotEpisodeLoader(root, module.so100_config)
for ep in range(len(loader)):
    raw = loader._load_parquet_data(ep)
    original = pd.read_parquet(root / loader.data_path_pattern.format(episode_chunk=ep // loader.chunk_size,
                                                                     episode_index=ep))
    expected = np.stack(original['action'].values)
    np.testing.assert_array_equal(np.stack(raw['action.single_arm']), expected[:, :5])
    np.testing.assert_array_equal(np.stack(raw['action.gripper']), expected[:, 5:])
generate_rel_stats(root, EmbodimentTag.NEW_EMBODIMENT)
relative = json.loads((root / 'meta/relative_stats.json').read_text())
assert relative == {'__fingerprints__': {}}, relative
print('PASS: actual N1.7 reader returns absolute actions; no relative action statistics generated')
'''
        before = snapshot(self.filtered)
        # The GA stats helper emits an empty fingerprint cache even when every
        # action is absolute. Probe that behavior on an independent fixture copy.
        probe = self.root / 'absolute-stats-probe'
        shutil.copytree(self.filtered, probe)
        self.assertIn('PASS: actual N1.7 reader', self.run_code(code, probe))
        self.assertEqual(snapshot(self.filtered), before)
        result = subprocess.run([str(self.python), str(REPO / 'scripts/verify_pick_place_training.py'),
                                 '--dataset', str(self.filtered)], env=self.environment(),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS: all prepared checksums', result.stdout)
        self.assertEqual(snapshot(self.source), self.before)

    def test_resume_rejects_changed_profile_without_modifying_source(self):
        output = self.root / 'changed-profile'
        building = output.with_name(output.name + '.building')
        shutil.copytree(self.filtered, building)
        write_json(building / 'model_profile.json', {'model_profile': 'so100_n16'})
        before = snapshot(building)
        result = self.prepare(output, '--resume', '--exclude_episodes', '0,3', success=False)
        self.assertIn('same --model_profile', result)
        self.assertEqual(snapshot(building), before)
        self.assertEqual(snapshot(self.source), self.before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
