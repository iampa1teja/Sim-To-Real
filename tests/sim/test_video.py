"""Encode and decode arbitrary camera names through the bounded production video queue."""
from pathlib import Path

import av
import numpy as np

from sim.eval.video import EpisodeVideoRecorder


def test_generic_camera_names_encode_every_frame_and_snapshot_reused_buffers(tmp_path):
    recorder = EpisodeVideoRecorder(tmp_path, fps=20, queue_size=1)
    recorder.start_episode(2)
    buffers = {name: np.zeros((1, 32, 48, 3), dtype=np.uint8) for name in ('camera_A', 'camera_B')}
    expected = {name: [] for name in buffers}
    try:
        for index in range(7):
            colors = {'camera_A': (30 + index * 20, 40, 60), 'camera_B': (20, 200 - index * 20, 90)}
            for name, color in colors.items():
                buffers[name][:] = color
                expected[name].append(color)
            recorder.write({f'rgb_{name}': values for name, values in buffers.items()})
            # Simulator sensor arrays are reused immediately after write().
            for values in buffers.values():
                values[:] = 0
        paths = recorder.finish_episode()
    finally:
        recorder.close()
    assert set(paths) == {'camera_A', 'camera_B'}
    for name, path in paths.items():
        assert Path(path).name == f'episode_000002_{name}.mp4'
        with av.open(path) as stream:
            assert stream.streams.video[0].codec_context.name == 'h264'
            decoded = [frame.to_ndarray(format='rgb24') for frame in stream.decode(video=0)]
        assert len(decoded) == 7
        assert all(frame.shape == (32, 48, 3) for frame in decoded)
        actual = np.array([frame.mean(axis=(0, 1)) for frame in decoded])
        np.testing.assert_allclose(actual, expected[name], atol=5, rtol=0)
