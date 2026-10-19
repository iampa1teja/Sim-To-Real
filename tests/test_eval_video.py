"""Real CPU encode/decode checks for optional evaluation camera recordings."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
MODULE_PATH = Path(__file__).resolve().parents[1] / "source/sim_to_real_so101/utils/eval_video.py"
spec = importlib.util.spec_from_file_location("eval_video", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EpisodeVideoRecorder = module.EpisodeVideoRecorder


def observations(value):
    frame = np.zeros((1, 16, 24, 3), dtype=np.uint8)
    frame[..., 0] = value
    return {"rgb_realsense_rgb": frame, "rgb_wrist_cam": frame.copy()}


def decode(path):
    with av.open(str(path)) as container:
        rate = container.streams.video[0].average_rate
        images = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return rate, images


def test_streams_distinct_attempts_and_two_camera_views(tmp_path):
    directory = tmp_path / "videos"
    recorder = EpisodeVideoRecorder(directory, fps=30)
    assert not directory.exists()
    recorder.start_episode(0)
    for value in (180, 210):
        recorder.write(observations(value))
    first = recorder.finish_episode()
    assert recorder.finish_episode() == first
    recorder.start_episode(1)
    for value in (20, 30, 40):
        recorder.write(observations(value))
    second = recorder.finish_episode()
    recorder.close()
    assert set(first) == set(second) == {"room", "wrist"}
    assert len(set(first.values()) | set(second.values())) == 4
    for paths, count, low, high in ((first, 2, 170, 220), (second, 3, 10, 50)):
        for camera, path in paths.items():
            rate, images = decode(path)
            assert rate == 30
            assert len(images) == count
            assert all(image.shape == (16, 24, 3) for image in images)
            assert all(low < image[..., 0].mean() < high for image in images)
            assert Path(path).name.endswith(f"_{camera}.mp4")


def test_partial_attempt_close_finalizes_playable_files(tmp_path):
    recorder = EpisodeVideoRecorder(tmp_path, fps=1/.03333333333333333)
    recorder.start_episode(0)
    recorder.write(observations(128))
    with pytest.raises(KeyError, match="Missing video camera"):
        try:
            recorder.write({})
        finally:
            recorder.close()
    for file in tmp_path.glob("*.mp4"):
        assert len(decode(file)[1]) == 1
    assert len(list(tmp_path.glob("*.mp4"))) == 2


def test_accepts_batched_torch_and_preserves_camera_rename_map(tmp_path):
    torch = pytest.importorskip("torch")
    recorder = EpisodeVideoRecorder(tmp_path, 20, {"realsense_rgb": "external camera", "wrist_cam": "hand"})
    recorder.start_episode(3)
    recorder.write({key: torch.from_numpy(value).float() for key, value in observations(120).items()})
    paths = recorder.finish_episode()
    assert set(paths) == {"external camera", "hand"}
    assert Path(paths["external camera"]).name == "episode_000003_external_camera.mp4"
    assert all(len(decode(path)[1]) == 1 for path in paths.values())


def test_existing_recording_is_preserved(tmp_path):
    existing = tmp_path / "episode_000000_room.mp4"
    existing.write_bytes(b"previous recording")
    recorder = EpisodeVideoRecorder(tmp_path, 30)
    recorder.start_episode(0)
    with pytest.raises(FileExistsError):
        recorder.write(observations(100))
    recorder.close()
    assert existing.read_bytes() == b"previous recording"


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_is_rejected_without_creating_files(tmp_path, fps):
    directory = tmp_path / "videos"
    with pytest.raises(ValueError, match="fps"):
        EpisodeVideoRecorder(directory, fps)
    assert not directory.exists()
