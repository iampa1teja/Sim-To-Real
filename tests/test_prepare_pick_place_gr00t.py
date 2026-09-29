"""CPU integration tests. Set GROOT_CHECKOUT to the pinned GR00T checkout.

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


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@unittest.skipUnless(GROOT, 'Set GROOT_CHECKOUT for real GR00T CPU integration tests')
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        sys.path.insert(0, GROOT)
        from gr00t.data.stats import calculate_dataset_statistics
        stats = calculate_dataset_statistics(list(cls.source.glob('data/*/*.parquet')),
                                             ['observation.state', 'action', 'timestamp'])
        write_json(cls.source / 'meta/stats.json', stats)
        cls.before = snapshot(cls.source)
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
                   'preparation_report.json', 'prepared_sha256.json'}
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
