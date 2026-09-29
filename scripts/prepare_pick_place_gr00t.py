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
import pyarrow.parquet as pq

PIN = 'ead52833afbbf4243f8cd5e7664f48a94de03b19'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--groot', type=Path, required=True)
parser.add_argument('--resume', action='store_true', help='Retry an incomplete build from the same unchanged source')
parser.add_argument('--source', type=Path, default=Path('datasets/pick_place_v1'))
parser.add_argument('--output', type=Path, default=Path('datasets/pick_place_v1_gr00t'))
args = parser.parse_args()
source, output, groot = args.source.resolve(), args.output.resolve(), args.groot.resolve()
assert subprocess.check_output(['git', '-C', str(groot), 'rev-parse', 'HEAD'], text=True).strip() == PIN
assert not output.exists(), f'Refusing to overwrite {output}'
build = output.with_name(output.name + '.building')
assert not build.exists() or args.resume, f'Inspect previous build before retrying: {build}'
assert source != output and not output.is_relative_to(source)

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
spec = importlib.util.spec_from_file_location('groot_v3_converter', groot / 'scripts/lerobot_conversion/convert_v3_to_v2.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
converter.validate_local_dataset_version(source)
records = converter.load_episode_records(source)
info = converter.load_info(source)
assert [int(row['episode_index']) for row in records] == list(range(info['total_episodes']))
assert sum(int(row['length']) for row in records) == info['total_frames']
video_keys = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
assert set(video_keys) == {'observation.images.realsense_rgb', 'observation.images.wrist_cam'}
chunks_size = info['chunks_size']
build.mkdir(exist_ok=args.resume)
# Use the upstream conversion functions without its destructive source swap.
converter.convert_info(source, build, records, video_keys)
converter.copy_global_stats(source, build)
converter.convert_tasks(source, build)
converter.convert_data(source, build, records, chunks_size)
converter.convert_episodes_metadata(build, records)
shutil.copytree(source / 'pick_place_meta', build / 'pick_place_meta', dirs_exist_ok=args.resume)

video_checks = []
for record in records:
    ep, length = int(record['episode_index']), int(record['length'])
    data_src = source / info['data_path'].format(chunk_index=int(record['data/chunk_index']), file_index=int(record['data/file_index']))
    original_table = pq.read_table(data_src)
    expected = original_table.filter(__import__('pyarrow').compute.equal(original_table['episode_index'], ep))
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
write_json(build / 'meta/modality.json', modality)
shutil.copy2(groot / 'examples/SO100/so100_config.py', build / 'so100_config.py')
shutil.copy2(Path(__file__), build / 'prepare_pick_place_gr00t.py')
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

print('Decoding and checking every frame in 100 videos...', flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    decoded = []
    for result in pool.map(decode_check, video_checks):
        decoded.append(result)
        if len(decoded) % 10 == 0:
            print(f'Validated videos: {len(decoded)}/{len(video_checks)}', flush=True)

# Exercise the actual pinned GR00T reader and its relative-action statistics.
sys.path.insert(0, str(groot))
config_spec = importlib.util.spec_from_file_location('pick_place_so100_config', build / 'so100_config.py')
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_stats, generate_rel_stats

generate_stats(build)
generate_rel_stats(build, EmbodimentTag.NEW_EMBODIMENT)
loader = LeRobotEpisodeLoader(build, config_module.so100_config, video_backend='ffmpeg')
assert len(loader) == info['total_episodes']
for ep, length in enumerate(loader.episode_lengths):
    lowdim = loader._load_parquet_data(ep)
    assert len(lowdim) == length
    assert np.stack(lowdim['state.single_arm']).shape == (length, 5)
    assert np.stack(lowdim['action.gripper']).shape == (length, 1)
    assert set(lowdim['language.annotation.human.task_description']) == set(records[ep]['tasks'])
    frames = loader._load_video_data(ep, np.array([0, length // 2, length - 1]))
    assert set(frames) == {'front', 'wrist'}
    assert all(value.shape == (3, 480, 640, 3) for value in frames.values())
assert snapshot(source) == before, 'Original dataset changed during preparation'
report = {'status': 'passed', 'time_utc': datetime.now(timezone.utc).isoformat(),
          'source': str(source), 'output': str(output), 'groot_revision': PIN,
          'format': 'v2.1', 'episodes': len(records), 'frames': info['total_frames'],
          'videos': len(decoded), 'decoded_video_frames': sum(row['frames'] for row in decoded),
          'all_parquet_values_preserved': True, 'all_videos_byte_identical': True,
          'all_video_frames_decoded': True, 'all_video_timestamps_checked': True,
          'groot_loader_all_episodes_checked': True, 'groot_validation_video_backend': 'ffmpeg', 'relative_action_statistics_generated': True,
          'original_dataset_sha256_unchanged': True, 'video_checks': decoded}
write_json(build / 'preparation_report.json', report)
write_json(build / 'prepared_sha256.json', snapshot(build))
build.rename(output)
print(json.dumps({k:v for k,v in report.items() if k != 'video_checks'}, indent=2), flush=True)
