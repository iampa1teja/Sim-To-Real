"""Prepare a verified, independent GR00T v2.1 copy of this recorded v3 dataset.

Usage: python scripts/prepare_pick_place_gr00t.py --groot /path/to/pinned/Isaac-GR00T
Original episodes/videos/metadata are never modified. Requires the existing
LeRobot dependencies, PyAV and the pinned GR00T source tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gr00t_model_profiles import PROFILES

PIN = 'ead52833afbbf4243f8cd5e7664f48a94de03b19'


def episode_list(value):
    try:
        episodes = [int(item) for item in value.split(',')]
        if any(ep < 0 for ep in episodes):
            raise ValueError
        return sorted(set(episodes))
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated nonnegative episode indices') from None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exclude_episodes', type=episode_list, default=[],
                    help='Comma-separated source episode indices to omit, e.g. 0,6')
parser.add_argument('--groot', type=Path, required=True)
parser.add_argument('--resume', action='store_true', help='Retry an incomplete build from the same unchanged source')
parser.add_argument('--source', type=Path, default=Path('datasets/pick_place_v1'))
parser.add_argument('--output', type=Path, default=Path('datasets/pick_place_v1_gr00t'))
parser.add_argument('--model_profile', choices=PROFILES, default='so100_n16',
                    help='Use so_arm_n17 for the official NVIDIA SO-arm N1.7 checkpoint')
args = parser.parse_args()
profile = PROFILES[args.model_profile]
PIN = profile['revision']
source, output, groot = args.source.resolve(), args.output.resolve(), args.groot.resolve()
assert subprocess.check_output(['git', '-C', str(groot), 'rev-parse', 'HEAD'], text=True).strip() == PIN
assert not output.exists(), f'Refusing to overwrite {output}'
build = output.with_name(output.name + '.building')
assert not build.exists() or args.resume, f'Inspect previous build before retrying: {build}'
assert source != output and not output.is_relative_to(source)
assert build != source and not build.is_relative_to(source) and not source.is_relative_to(build)

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

before = snapshot(source)
if args.resume:
    assert json.loads((build / 'source_sha256.json').read_text()) == before
    selection_path = build / 'excluded_episodes.json'
    previous_excluded = json.loads(selection_path.read_text()) if selection_path.exists() else []
    assert previous_excluded == args.exclude_episodes, 'Resume requires the same --exclude_episodes'
    profile_path = build / 'model_profile.json'
    previous_profile = json.loads(profile_path.read_text())['model_profile'] if profile_path.exists() else 'so100_n16'
    assert previous_profile == args.model_profile, 'Resume requires the same --model_profile'
spec = importlib.util.spec_from_file_location('groot_v3_converter', groot / 'scripts/lerobot_conversion/convert_v3_to_v2.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
converter.validate_local_dataset_version(source)
records = converter.load_episode_records(source)
info = converter.load_info(source)
assert [int(row['episode_index']) for row in records] == list(range(info['total_episodes']))
assert sum(int(row['length']) for row in records) == info['total_frames']
assert set(args.exclude_episodes) <= set(range(len(records))), 'Unknown excluded episode index'
source_records = [row for row in records if int(row['episode_index']) not in args.exclude_episodes]
assert source_records, 'Cannot exclude every episode'
index_map = {new: int(row['episode_index']) for new, row in enumerate(source_records)}
video_keys = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
assert set(video_keys) == {'observation.images.realsense_rgb', 'observation.images.wrist_cam'}
chunks_size = info['chunks_size']
build.mkdir(exist_ok=args.resume)
if args.model_profile != 'so100_n16':
    write_json(build / 'model_profile.json', {'model_profile': args.model_profile, 'groot_revision': PIN})
write_json(build / 'source_sha256.json', before)
if args.exclude_episodes:
    write_json(build / 'excluded_episodes.json', args.exclude_episodes)
# Use the upstream conversion functions without its destructive source swap.
converter.convert_info(source, build, records, video_keys)
converter.convert_tasks(source, build)
if not args.exclude_episodes:
    converter.copy_global_stats(source, build)
    converter.convert_data(source, build, records, chunks_size)
    shutil.copytree(source / 'pick_place_meta', build / 'pick_place_meta', dirs_exist_ok=args.resume)
else:
    # Do not pass a filtered list to upstream convert_data: it infers a file's
    # row offset from its first episode, which may now have been excluded.
    records = []
    offset = 0
    used_tasks = set()
    (build / 'pick_place_meta').mkdir(exist_ok=True)
    for ep, original in enumerate(source_records):
        old_ep, length = int(original['episode_index']), int(original['length'])
        record = dict(original, episode_index=ep, dataset_from_index=offset,
                      dataset_to_index=offset + length)
        # Per-episode statistics remain valid except for the two shifted IDs.
        for key, shift in [('episode_index', ep - old_ep),
                           ('index', offset - int(original['dataset_from_index']))]:
            for stat in ('mean', 'min', 'max', 'q01', 'q99'):
                name = f'stats/{key}/{stat}'
                if name in record:
                    record[name] = (np.asarray(record[name]) + shift).tolist()
        records.append(record)
        path = source / info['data_path'].format(chunk_index=int(original['data/chunk_index']),
                                               file_index=int(original['data/file_index']))
        table = pq.read_table(path)
        table = table.filter(pc.equal(table['episode_index'], old_ep))
        assert table.num_rows == length
        np.testing.assert_array_equal(table['index'].to_numpy(),
                                      np.arange(original['dataset_from_index'], original['dataset_to_index']))
        for key, values in [('episode_index', np.full(length, ep)),
                            ('index', np.arange(offset, offset + length))]:
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
        dest = build / converter.LEGACY_DATA_PATH_TEMPLATE.format(episode_chunk=ep // chunks_size, episode_index=ep)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        used_tasks.update(table['task_index'].to_pylist())
        sidecar = json.loads((source / f'pick_place_meta/episode_{old_ep:06d}.json').read_text())
        assert sidecar['episode_index'] == old_ep
        sidecar['episode_index'] = ep
        write_json(build / f'pick_place_meta/episode_{ep:06d}.json', sidecar)
        offset += length
    tasks_path = build / 'meta/tasks.jsonl'
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    tasks = [task for task in tasks if task['task_index'] in used_tasks]
    assert {task['task_index'] for task in tasks} == used_tasks
    tasks_path.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
    target_info = json.loads((build / 'meta/info.json').read_text())
    splits = {}
    for name, bounds in info['splits'].items():
        start, stop = map(int, bounds.split(':'))
        assert 0 <= start <= stop <= info['total_episodes'], (name, bounds)
        splits[name] = f'{sum(old < start for old in index_map.values())}:{sum(old < stop for old in index_map.values())}'
    target_info.update(total_episodes=len(records), total_frames=offset, total_tasks=len(tasks),
                       total_chunks=(len(records) + chunks_size - 1) // chunks_size,
                       total_videos=len(records) * len(video_keys), splits=splits)
    write_json(build / 'meta/info.json', target_info)
    # Both pinned generators skip valid existing stats, including on resume.
    for name in ('stats.json', 'relative_stats.json'):
        (build / 'meta' / name).unlink(missing_ok=True)
converter.convert_episodes_metadata(build, records)

video_checks = []
for record in records:
    ep, length = int(record['episode_index']), int(record['length'])
    data_src = source / info['data_path'].format(chunk_index=int(record['data/chunk_index']), file_index=int(record['data/file_index']))
    original_table = pq.read_table(data_src)
    expected = original_table.filter(pc.equal(original_table['episode_index'], index_map[ep]))
    if args.exclude_episodes:
        for key, values in [('episode_index', np.full(length, ep)),
                            ('index', np.arange(record['dataset_from_index'], record['dataset_to_index']))]:
            field = expected.schema.field(key)
            expected = expected.set_column(expected.schema.get_field_index(key), field, pa.array(values, type=field.type))
    target_table = pq.read_table(build / converter.LEGACY_DATA_PATH_TEMPLATE.format(episode_chunk=ep // chunks_size, episode_index=ep))
    assert target_table.num_rows == length and expected.equals(target_table), f'Parquet mismatch: {ep}'
    frame_indices = target_table['frame_index'].to_numpy()
    np.testing.assert_array_equal(frame_indices, np.arange(length))
    np.testing.assert_array_equal(target_table['index'].to_numpy(), np.arange(record['dataset_from_index'], record['dataset_to_index']))
    np.testing.assert_allclose(target_table['timestamp'].to_numpy(), np.arange(length) / info['fps'], atol=1e-6)
    for key in ('observation.state', 'action'):
        values = np.array(target_table[key].to_pylist())
        assert values.shape == (length, 6) and np.isfinite(values).all(), (ep, key)
    for key in video_keys:
        start, end = record[f'videos/{key}/from_timestamp'], record[f'videos/{key}/to_timestamp']
        assert start == 0 and abs(end * info['fps'] - length) < 1e-6, 'This lossless-copy recipe requires one episode per source video'
        src = source / info['video_path'].format(video_key=key, chunk_index=int(record[f'videos/{key}/chunk_index']), file_index=int(record[f'videos/{key}/file_index']))
        dst = build / converter.LEGACY_VIDEO_PATH_TEMPLATE.format(episode_chunk=ep // chunks_size, video_key=key, episode_index=ep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(dst) == before[str(src.relative_to(source))]
        video_checks.append((dst, ep, key, length))

modality = json.loads((groot / 'examples/SO100/modality.json').read_text())
modality['video']['front']['original_key'] = 'observation.images.realsense_rgb'
modality['video']['wrist']['original_key'] = 'observation.images.wrist_cam'
if args.model_profile == 'so_arm_n17':
    modality['video']['room'] = modality['video'].pop('front')
write_json(build / 'meta/modality.json', modality)
config_source = (Path(__file__).with_name('so_arm_n17_config.py') if args.model_profile == 'so_arm_n17'
                 else groot / 'examples/SO100/so100_config.py')
shutil.copy2(config_source, build / 'so100_config.py')
shutil.copy2(Path(__file__), build / 'prepare_pick_place_gr00t.py')
shutil.copy2(Path(__file__).with_name('gr00t_model_profiles.py'), build / 'gr00t_model_profiles.py')
if args.model_profile == 'so_arm_n17':
    shutil.copy2(config_source, build / 'so_arm_n17_config.py')
write_json(build / 'source_sha256.json', before)

def decode_check(item):
    path, ep, key, length = item
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'NONE'
        stream.codec_context.thread_count = 1
        assert float(stream.average_rate) == info['fps']
        for frame in container.decode(stream):
            assert frame.width == 640 and frame.height == 480
            assert abs(float(frame.pts * frame.time_base) - count / info['fps']) < 1e-6
            # Force pixel materialization to catch decoding/conversion failures.
            pixels = frame.to_ndarray(format='rgb24')
            assert pixels.shape == (480, 640, 3)
            count += 1
    assert count == length, (path, count, length)
    return {'episode': ep, 'camera': key, 'frames': count, 'sha256': sha(path)}

print(f'Decoding and checking every frame in {len(video_checks)} videos...', flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    decoded = []
    for result in pool.map(decode_check, video_checks):
        decoded.append(result)
        if len(decoded) % 10 == 0:
            print(f'Validated videos: {len(decoded)}/{len(video_checks)}', flush=True)

# Exercise the actual reader with the selected checkpoint's action conventions.
sys.path.insert(0, str(groot))
config_spec = importlib.util.spec_from_file_location('pick_place_so100_config', build / 'so100_config.py')
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_stats, generate_rel_stats

generate_stats(build)
if profile['relative_actions']:
    generate_rel_stats(build, EmbodimentTag.NEW_EMBODIMENT)
loader_kwargs = {'video_backend': 'ffmpeg'} if args.model_profile == 'so100_n16' else {}
loader = LeRobotEpisodeLoader(build, config_module.so100_config, **loader_kwargs)
assert len(loader) == len(records)
for ep, length in enumerate(loader.episode_lengths):
    lowdim = loader._load_parquet_data(ep)
    assert len(lowdim) == length
    assert np.stack(lowdim['state.single_arm']).shape == (length, 5)
    assert np.stack(lowdim['action.gripper']).shape == (length, 1)
    assert set(lowdim['language.annotation.human.task_description']) == set(records[ep]['tasks'])
    frames = loader._load_video_data(ep, np.array([0, length // 2, length - 1]))
    assert set(frames) == set(profile['cameras'])
    assert all(value.shape == (3, 480, 640, 3) for value in frames.values())
assert snapshot(source) == before, 'Original dataset changed during preparation'
report = {'status': 'passed', 'time_utc': datetime.now(timezone.utc).isoformat(),
          'source': str(source), 'output': str(output), 'groot_revision': PIN,
          'format': 'v2.1', 'episodes': len(records), 'frames': sum(int(row['length']) for row in records),
          'videos': len(decoded), 'decoded_video_frames': sum(row['frames'] for row in decoded),
          'all_parquet_values_preserved': True, 'all_videos_byte_identical': True,
          'all_video_frames_decoded': True, 'all_video_timestamps_checked': True,
          'groot_loader_all_episodes_checked': True, 'groot_validation_video_backend': profile['video_backend'],
          'relative_action_statistics_generated': profile['relative_actions'],
          'original_dataset_sha256_unchanged': True, 'video_checks': decoded}
if args.model_profile != 'so100_n16':
    report.update(model_profile=args.model_profile, embodiment_tag='NEW_EMBODIMENT',
                  video_keys=profile['cameras'], action_horizon=16,
                  action_representation={'single_arm': 'ABSOLUTE', 'gripper': 'ABSOLUTE'})
if args.exclude_episodes:
    report.update(excluded=args.exclude_episodes, index_map=index_map,
                  statistics_policy=('Recomputed GR00T global and relative-action stats from retained parquet; '
                                     if profile['relative_actions'] else
                                     'Recomputed GR00T global absolute-action stats from retained parquet; ') +
                  'retained per-episode stats with shifted episode_index/index stats.',
                  all_parquet_values_preserved=False,
                  all_parquet_values_except_renumbered_indices_preserved=True)
write_json(build / 'preparation_report.json', report)
(build / 'prepared_sha256.json').unlink(missing_ok=True)
write_json(build / 'prepared_sha256.json', snapshot(build))
build.rename(output)
print(json.dumps({k:v for k,v in report.items() if k != 'video_checks'}, indent=2), flush=True)
