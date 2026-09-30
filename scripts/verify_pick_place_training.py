"""Verify prepared files and exercise the pinned GR00T reader against PyAV pixels.
Run: PYTHONPATH=/path/to/Isaac-GR00T python scripts/verify_pick_place_training.py
     --dataset datasets/pick_place_v1_gr00t
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from gr00t_model_profiles import PROFILES

import av
import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, default=Path('datasets/pick_place_v1_gr00t'))
root = parser.parse_args().dataset.resolve()
report = json.loads((root / 'preparation_report.json').read_text())
profile_name = report.get('model_profile', 'so100_n16')
profile = PROFILES[profile_name]
import gr00t
groot_root = Path(gr00t.__file__).resolve().parents[1]
revision = subprocess.check_output(['git', '-C', str(groot_root), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == profile['revision'] == report['groot_revision'], 'GR00T runtime/profile revision differs'
for name, expected in json.loads((root / 'prepared_sha256.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
spec = importlib.util.spec_from_file_location('pick_place_modality', root / 'so100_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loader_kwargs = {'video_backend': 'ffmpeg'} if profile_name == 'so100_n16' else {}
loader = LeRobotEpisodeLoader(root, module.so100_config, **loader_kwargs)

def check_episode(ep):
    length = loader.get_episode_length(ep)
    indices = [0, length // 2, length - 1]
    frames = loader._load_video_data(ep, np.array(indices))
    assert set(frames) == set(profile['cameras'])
    max_difference = 0
    for key, actual in frames.items():
        original = loader.modality_meta['video'][key]['original_key']
        path = root / loader.video_path_pattern.format(episode_chunk=ep // loader.chunk_size,
                                                       video_key=original, episode_index=ep)
        expected = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for index, frame in enumerate(container.decode(stream)):
                if index in indices:
                    expected.append(frame.to_ndarray(format='rgb24'))
        expected = np.stack(expected)
        assert actual.shape == expected.shape
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        # Independent FFmpeg builds can round YUV->RGB conversion slightly differently.
        assert difference.max() <= 2, (ep, key, int(difference.max()))
        max_difference = max(max_difference, int(difference.max()))
    return max_difference

with ThreadPoolExecutor(max_workers=2) as pool:
    differences = []
    for value in pool.map(check_episode, range(len(loader))):
        differences.append(value)
        if len(differences) % 10 == 0:
            print(f'GR00T/PyAV frame comparisons: {len(differences)}/{len(loader)} episodes', flush=True)
print(f'PASS: all prepared checksums, all {len(loader)} episodes, both cameras, first/middle/last frames; maximum RGB difference {max(differences)}', flush=True)
