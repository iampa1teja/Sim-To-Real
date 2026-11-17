"""WP8: label columns in the LeRobot recorder and the RecorderEpisodeWriter adapter (contracts §9.3, §12, §3.3).

``sim.recording.lerobot_recorder`` imports LeRobot at module level, so these tests load it with stub ``lerobot``
and ``sim.recording.recording_dataset`` modules (contracts §13 "sys.modules stubs"); only the pure parts are used.

Decisions pinned here (docs/v4/prompts/WP8.md):
* ``LeRobotRecorder(..., extra_features=None)``: ``None`` keeps ``dataset_features`` exactly as today; a dict is
  appended after the default features (same key order otherwise).
* ``make_sim_frame(action, observation, visual_buffers, extra=None)``: ``None`` keeps today's frame; each extra
  column is copied as a numpy array of the feature's dtype; a key that is not a dataset feature raises ``KeyError``.
* ``RecorderEpisodeWriter(recorder)`` (``sim/generate/isaac_gen_env.py``): ``root = Path(recorder.dataset_root)``;
  ``episode_count() = len(recorder.episode_lengths())``; ``add_frame`` = ``recorder.make_sim_frame(action_raw,
  state_raw, images, extra=label_columns(labels))``, ``frame["task"] = task``, ``recorder.add_frame(frame)``;
  ``discard_episode`` = ``recorder.clear_episode_buffer()``; ``save_episode`` returns ``recorder.save_episode()``;
  ``finalize`` = ``recorder.finalize()``. ``sim.generate.isaac_gen_env`` imports no Isaac / LeRobot module at top level.
"""

import contextlib
import importlib
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from core.datasets.task_meta import LABEL_FEATURES
from core.generate.labels import label_columns
from core.generate.protocols import EpisodeWriter, FrameLabels
from v4_support import SRC_DIR, TESTS_DIR


@contextlib.contextmanager
def stub_lerobot():
    """Import sim.recording.lerobot_recorder with LeRobot and RecordingDataset replaced by stubs."""
    names = ["lerobot", "lerobot.datasets", "lerobot.datasets.feature_utils", "lerobot.utils",
             "lerobot.utils.constants", "sim.recording.recording_dataset", "sim.recording.lerobot_recorder"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names[:-1]:
            sys.modules[n] = types.ModuleType(n)
        sys.modules["lerobot.datasets.feature_utils"].validate_episode_buffer = lambda *a, **k: None
        sys.modules["lerobot.utils.constants"].DEFAULT_FEATURES = {}
        sys.modules["sim.recording.recording_dataset"].RecordingDataset = object
        sys.modules.pop("sim.recording.lerobot_recorder", None)
        yield importlib.import_module("sim.recording.lerobot_recorder")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def labels(t=0):
    q = np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3])
    q[3:] /= np.linalg.norm(q[3:])
    return FrameLabels(stage=t - 1, imitate=bool(t % 2), ee_pose_wxyz=q)


class RecorderExtraFeatureTests(unittest.TestCase):
    """WP8 §9.3 extra_features and make_sim_frame(extra=...)."""

    CAMS = {"front": {"height": 4, "width": 6}}

    def recorder(self, mod, **kw):
        return mod.LeRobotRecorder(task_name="t", repo_id="local/x", dataset_root="/tmp/x", fps=30, device="cpu",
                                   cameras=self.CAMS, robot_type="r", **kw)

    def test_default_unchanged(self):
        with stub_lerobot() as mod:
            r = self.recorder(mod)
            self.assertEqual(r.dataset_features, mod.LeRobotRecorder._make_sim_features(self.CAMS))
            self.assertEqual(list(r.dataset_features), list(mod.LeRobotRecorder._make_sim_features(self.CAMS)))
            J = len(mod.JOINT_FEATURE_NAMES)
            frame = r.make_sim_frame(np.zeros(J), np.ones(J), {"front": np.zeros((4, 6, 3), np.uint8)})
            self.assertEqual(set(frame), {"action", "observation.state", "task", "observation.images.front"})

    def test_extra_features(self):
        with stub_lerobot() as mod:
            r = self.recorder(mod, extra_features=LABEL_FEATURES)
            base = mod.LeRobotRecorder._make_sim_features(self.CAMS)
            self.assertEqual(list(r.dataset_features), list(base) + list(LABEL_FEATURES))
            for key, spec in LABEL_FEATURES.items():
                self.assertEqual(r.dataset_features[key]["dtype"], spec["dtype"])
                self.assertEqual(tuple(r.dataset_features[key]["shape"]), tuple(spec["shape"]))
            J = len(mod.JOINT_FEATURE_NAMES)
            lab = labels(3)
            frame = r.make_sim_frame(np.zeros(J), np.ones(J), {"front": np.zeros((4, 6, 3), np.uint8)},
                                     extra=label_columns(lab))
            self.assertEqual(frame["label.stage"].dtype, np.int64)
            self.assertEqual(frame["label.imitate"].dtype, np.int64)
            self.assertEqual(frame["label.ee_pose"].dtype, np.float32)
            np.testing.assert_array_equal(frame["label.stage"], [lab.stage])
            np.testing.assert_array_equal(frame["label.imitate"], [1])
            np.testing.assert_allclose(frame["label.ee_pose"], lab.ee_pose_wxyz, atol=1e-7)
            with self.assertRaises(KeyError):
                r.make_sim_frame(np.zeros(J), np.ones(J), {"front": np.zeros((4, 6, 3), np.uint8)},
                                 extra={"label.unknown": np.zeros(1)})


class StubRecorder:
    def __init__(self, root):
        self.dataset_root = str(root)
        self.lengths, self.buffer, self.calls = [], [], []

    def episode_lengths(self):
        self.calls.append("episode_lengths")
        return list(self.lengths)

    def make_sim_frame(self, action, observation, visual_buffers, extra=None):
        self.calls.append("make_sim_frame")
        frame = {"action": np.asarray(action, np.float32), "observation.state": np.asarray(observation, np.float32),
                 "task": "default"}
        frame.update({f"observation.images.{k}": v for k, v in visual_buffers.items()})
        frame.update(dict(extra or {}))
        return frame

    def add_frame(self, frame, auxiliary=None):
        self.calls.append("add_frame")
        self.buffer.append(frame)

    def clear_episode_buffer(self):
        self.calls.append("clear_episode_buffer")
        self.buffer = []

    def save_episode(self):
        self.calls.append("save_episode")
        self.lengths.append(len(self.buffer))
        self.buffer = []
        return len(self.lengths) - 1

    def finalize(self):
        self.calls.append("finalize")


class RecorderEpisodeWriterTests(unittest.TestCase):
    """WP8 §12 RecorderEpisodeWriter maps EpisodeWriter onto LeRobotRecorder."""

    def test_adapter(self):
        from sim.generate.isaac_gen_env import RecorderEpisodeWriter
        rec = StubRecorder("/data/x")
        w = RecorderEpisodeWriter(rec)
        self.assertIsInstance(w, EpisodeWriter)
        self.assertEqual(w.root, Path("/data/x"))
        self.assertEqual(w.episode_count(), 0)
        for t in range(3):
            w.add_frame(np.full(6, t, float), np.zeros(6), {"cam": np.zeros((2, 2, 3), np.uint8)}, "say it", labels(t))
        self.assertEqual(len(rec.buffer), 3)
        f = rec.buffer[2]
        self.assertEqual(f["task"], "say it")
        for key, value in label_columns(labels(2)).items():
            np.testing.assert_array_equal(f[key], value)
        np.testing.assert_array_equal(f["action"], np.full(6, 2, np.float32))
        self.assertEqual(w.save_episode(), 0)
        self.assertEqual((w.episode_count(), w.episode_lengths()), (1, [3]))
        w.add_frame(np.zeros(6), np.zeros(6), {}, "x", labels())
        w.discard_episode()
        self.assertEqual(rec.buffer, [])
        self.assertEqual(rec.calls[-1], "clear_episode_buffer")
        w.finalize()
        self.assertEqual(rec.calls[-1], "finalize")

    def test_module_imports_without_isaac(self):
        code = ("import importlib.abc, sys\n"
                "class B(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in {'isaaclab','isaacsim','omni','carb','pxr','warp','lerobot'}:\n"
                "            raise ImportError(name)\n"
                "sys.meta_path.insert(0, B())\n"
                "import sim.generate.isaac_gen_env, sim.generate.scripted_generate\n"
                "print('ok')\n")
        env = dict(os.environ, PYTHONPATH=f"{SRC_DIR}:{TESTS_DIR}")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
