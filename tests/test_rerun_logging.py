"""Camera playback must retain successive frames on the recording timeline."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def logger(monkeypatch):
    calls = []

    class Image:
        def __init__(self, pixels):
            self.pixels = np.array(pixels, copy=True)

        def compress(self, jpeg_quality):
            return SimpleNamespace(pixels=self.pixels, jpeg_quality=jpeg_quality)

    rerun = SimpleNamespace(Image=Image, Scalars=lambda value: SimpleNamespace(scalar=value),
                            log=lambda key, value, **kwargs: calls.append((key, value, kwargs)))
    monkeypatch.setitem(sys.modules, "rerun", rerun)
    spec = importlib.util.spec_from_file_location(
        "rollout_rerun_logging", ROOT / "source/sim_to_real_so101/utils/rerun_logging.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.log_rerun_data, calls


def test_successive_camera_frames_are_temporal_compressed_and_written_only_to_rerun(monkeypatch):
    log, calls = logger(monkeypatch)
    first = np.zeros((8, 10, 3), dtype=np.uint8)
    second = np.full_like(first, 217)
    with patch("builtins.open", side_effect=AssertionError("logging must not write files")):
        log(observation={"room": first})
        log(observation={"room": second})
    assert [key for key, _, _ in calls] == ["observation.room", "observation.room"]
    for _, image, options in calls:
        assert image.jpeg_quality == 90
        assert options.get("static", False) is False
    np.testing.assert_array_equal(calls[0][1].pixels, first)
    np.testing.assert_array_equal(calls[1][1].pixels, second)


def test_chw_camera_and_scalar_names_match_existing_recordings(monkeypatch):
    log, calls = logger(monkeypatch)
    chw = np.arange(3 * 8 * 10, dtype=np.uint8).reshape(3, 8, 10)
    log(observation={"observation.wrist": chw, "shoulder_lift.pos": np.float32(-32.5),
                     "joint_vector": np.array([1., 2.]), "absent": None},
        action={"action.gripper.pos": 12., "wrist_roll.pos": -4.})
    values = {key: value for key, value, _ in calls}
    np.testing.assert_array_equal(values["observation.wrist"].pixels, chw.transpose(1, 2, 0))
    assert values["observation.shoulder_lift.pos"].scalar == -32.5
    assert values["observation.joint_vector_0"].scalar == 1.
    assert values["observation.joint_vector_1"].scalar == 2.
    assert values["action.gripper.pos"].scalar == 12.
    assert values["action.wrist_roll.pos"].scalar == -4.
    assert "observation.absent" not in values
