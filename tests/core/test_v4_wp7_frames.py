"""WP7: the lite LeRobot-v3 dataset writer (LiteDatasetWriter / MemoryWriter) and read_frames (contracts §10.1,
§11, §9.3, §3.3 EpisodeWriter).

Decisions pinned here (docs/v4/prompts/WP7.md):
* ``LiteDatasetWriter(root, *, fps, joint_names)`` writes the lite layout; ``MemoryWriter`` is the same writer
  with crash hooks. ``info.json`` ``features``: ``action`` / ``observation.state`` ``{"dtype": "float32",
  "shape": [J], "names": joint_names}`` and the three ``LABEL_FEATURES`` (shape as a list). Images are
  validated (uint8, (H, W, 3)) but not stored; no camera features.
* Opening an existing root checks ``codebase_version``, ``fps`` and ``features`` (``ValueError`` if they
  differ), deletes data files with index >= ``total_episodes`` and leftover ``*.tmp`` files.
* ``save_episode`` on an empty buffer raises ``RuntimeError``; ``add_frame`` raises ``ValueError`` for a wrong
  action/state shape or a bad image and ``TypeError`` for labels that are not ``FrameLabels``.
* ``crash_on_episode=None`` arms the crash for the first ``save_episode``; a crash fires once per writer.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.datasets.frames import LiteDatasetWriter, MemoryWriter, read_frames
from core.datasets.task_meta import LABEL_FEATURES
from core.generate.fake_env import SimulatedCrash
from core.generate.protocols import EpisodeWriter, FrameLabels
from v4_support import sandbox

NAMES = ("j1", "j2", "j3", "grip")
J = len(NAMES)


def labels(t, n):
    q = np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3])
    q[3:] /= np.linalg.norm(q[3:])
    return FrameLabels(stage=min(t, 2) - 1, imitate=t % 3 != 1, ee_pose_wxyz=q + [t * 1e-3, 0, 0, 0, 0, 0, 0])


def episode(rng, n):
    return [(rng.uniform(-100, 100, J), rng.uniform(-100, 100, J), labels(t, n)) for t in range(n)]


def put(writer, frames, task="do it", images=None):
    for a, s, lab in frames:
        writer.add_frame(a, s, images or {}, task, lab)


class WriterBase(unittest.TestCase):
    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()
        self.root = self.box.root / "ds"
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def writer(self, cls=LiteDatasetWriter, **kw):
        return cls(self.root, fps=30, joint_names=NAMES, **kw)


class LayoutTests(WriterBase):
    """WP7 §11 lite layout: info.json is the commit point; one parquet file per episode with fixed columns."""

    def test_protocol_and_empty_root(self):
        w = self.writer()
        self.assertIsInstance(w, EpisodeWriter)
        self.assertEqual(Path(w.root), self.root)
        self.assertEqual(w.episode_count(), 0)
        self.assertEqual(w.episode_lengths(), [])
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.assertEqual(info["codebase_version"], "v3.0-lite")
        self.assertEqual(info["fps"], 30)
        self.assertEqual(info["total_episodes"], 0)
        feats = info["features"]
        for key in ("action", "observation.state"):
            self.assertEqual(feats[key], {"dtype": "float32", "shape": [J], "names": list(NAMES)})
        for key, spec in LABEL_FEATURES.items():
            self.assertEqual(feats[key], {"dtype": spec["dtype"], "shape": list(spec["shape"]),
                                          "names": list(spec["names"])})
        self.assertEqual(set(feats), {"action", "observation.state", *LABEL_FEATURES})

    def test_episode_files_and_columns(self):
        w = self.writer()
        eps = [episode(self.rng, n) for n in (5, 3, 7)]
        for k, ep in enumerate(eps):
            put(w, ep, task=f"task {k}")
            self.assertEqual(w.save_episode(), k)
        self.assertEqual(w.episode_count(), 3)
        self.assertEqual(w.episode_lengths(), [5, 3, 7])
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.assertEqual(info["total_episodes"], 3)
        start = 0
        for k, ep in enumerate(eps):
            path = self.root / "data" / "chunk-000" / f"file-{k:06d}.parquet"
            t = pq.read_table(path)
            with self.subTest(episode=k):
                self.assertEqual(set(t.column_names), {"action", "observation.state", "episode_index", "frame_index",
                                                       "index", "timestamp", "task", *LABEL_FEATURES})
                self.assertEqual(t.schema.field("action").type, pa.list_(pa.float32()))
                self.assertEqual(t.schema.field("observation.state").type, pa.list_(pa.float32()))
                for col in ("episode_index", "frame_index", "index"):
                    self.assertEqual(t.schema.field(col).type, pa.int64())
                self.assertEqual(t.schema.field("timestamp").type, pa.float32())
                self.assertEqual(t.schema.field("task").type, pa.string())
                self.assertEqual(t.schema.field("label.stage").type, pa.list_(pa.int64()))
                self.assertEqual(t.schema.field("label.imitate").type, pa.list_(pa.int64()))
                self.assertEqual(t.schema.field("label.ee_pose").type, pa.list_(pa.float32()))
                d = t.to_pydict()
                n = len(ep)
                self.assertEqual(d["episode_index"], [k] * n)
                self.assertEqual(d["frame_index"], list(range(n)))
                self.assertEqual(d["index"], list(range(start, start + n)))
                np.testing.assert_array_equal(np.array(d["timestamp"], np.float32), np.arange(n, dtype=np.float32) / np.float32(30))
                self.assertEqual(d["task"], [f"task {k}"] * n)
                np.testing.assert_array_equal(np.array(d["action"], np.float32), np.array([a for a, _, _ in ep], np.float32))
                np.testing.assert_array_equal(np.array(d["observation.state"], np.float32),
                                              np.array([s for _, s, _ in ep], np.float32))
                self.assertEqual(d["label.stage"], [[lab.stage] for _, _, lab in ep])
                self.assertEqual(d["label.imitate"], [[int(lab.imitate)] for _, _, lab in ep])
                np.testing.assert_array_equal(np.array(d["label.ee_pose"], np.float32),
                                              np.array([lab.ee_pose_wxyz for _, _, lab in ep], np.float32))
            start += n

    def test_discard_and_empty_save(self):
        w = self.writer()
        put(w, episode(self.rng, 4))
        w.discard_episode()
        w.discard_episode()                         # no-op when empty
        with self.assertRaises(RuntimeError):
            w.save_episode()
        put(w, episode(self.rng, 2))
        self.assertEqual(w.save_episode(), 0)
        self.assertEqual(w.episode_lengths(), [2])

    def test_add_frame_validation(self):
        w = self.writer()
        lab = labels(0, 1)
        with self.assertRaises(ValueError):
            w.add_frame(np.zeros(J + 1), np.zeros(J), {}, "x", lab)
        with self.assertRaises(ValueError):
            w.add_frame(np.zeros(J), np.zeros(J - 1), {}, "x", lab)
        with self.assertRaises(TypeError):
            w.add_frame(np.zeros(J), np.zeros(J), {}, "x", {"stage": 0})
        with self.assertRaises(ValueError):
            w.add_frame(np.zeros(J), np.zeros(J), {"cam": np.zeros((4, 4, 3), np.float32)}, "x", lab)
        with self.assertRaises(ValueError):
            w.add_frame(np.zeros(J), np.zeros(J), {"cam": np.zeros((4, 4), np.uint8)}, "x", lab)
        w.add_frame(np.zeros(J), np.zeros(J), {"cam": np.zeros((4, 4, 3), np.uint8)}, "x", lab)
        self.assertEqual(w.save_episode(), 0)

    def test_reopen_continues_and_cleans(self):
        w = self.writer()
        for n in (3, 4):
            put(w, episode(self.rng, n))
            w.save_episode()
        data = self.root / "data" / "chunk-000"
        (data / "file-000002.parquet").write_bytes((data / "file-000000.parquet").read_bytes())   # uncommitted
        (data / "x.tmp").write_bytes(b"partial")
        w2 = self.writer()
        self.assertEqual(w2.episode_count(), 2)
        self.assertEqual(w2.episode_lengths(), [3, 4])
        self.assertFalse((data / "file-000002.parquet").exists())
        self.assertEqual(list(data.glob("*.tmp")), [])
        put(w2, episode(self.rng, 5))
        self.assertEqual(w2.save_episode(), 2)
        self.assertEqual(sorted(read_frames(self.root)), [0, 1, 2])

    def test_reopen_rejects_other_layout(self):
        self.writer()
        with self.assertRaises(ValueError):
            LiteDatasetWriter(self.root, fps=15, joint_names=NAMES)
        with self.assertRaises(ValueError):
            LiteDatasetWriter(self.root, fps=30, joint_names=NAMES[:-1] + ("jaw",))


class CrashTests(WriterBase):
    """WP7 §11 MemoryWriter crash hooks: before_save writes nothing; after_save commits, then raises."""

    def test_before_save(self):
        w = self.writer(MemoryWriter, crash="before_save", crash_on_episode=1)
        put(w, episode(self.rng, 3))
        self.assertEqual(w.save_episode(), 0)
        put(w, episode(self.rng, 3))
        with self.assertRaises(SimulatedCrash):
            w.save_episode()
        self.assertEqual(self.writer().episode_count(), 1)
        self.assertFalse((self.root / "data" / "chunk-000" / "file-000001.parquet").exists())

    def test_after_save(self):
        w = self.writer(MemoryWriter, crash="after_save")
        put(w, episode(self.rng, 3))
        with self.assertRaises(SimulatedCrash):
            w.save_episode()
        again = self.writer()
        self.assertEqual(again.episode_count(), 1)
        self.assertEqual(again.episode_lengths(), [3])

    def test_crash_fires_once(self):
        w = self.writer(MemoryWriter, crash="before_save")
        put(w, episode(self.rng, 2))
        with self.assertRaises(SimulatedCrash):
            w.save_episode()
        self.assertEqual(w.save_episode(), 0)          # the buffer is still there; the hook is spent

    def test_bad_crash_mode(self):
        with self.assertRaises(ValueError):
            self.writer(MemoryWriter, crash="during_save")


class ReadFramesTests(WriterBase):
    """WP7 §10.1 read_frames: episode -> arrays, on the lite layout and on a real LeRobot v3 layout."""

    def test_lite_round_trip(self):
        w = self.writer()
        eps = [episode(self.rng, n) for n in (4, 6)]
        for ep in eps:
            put(w, ep)
            w.save_episode()
        out = read_frames(self.root)
        self.assertEqual(list(out), [0, 1])
        for k, ep in enumerate(eps):
            f = out[k]
            with self.subTest(episode=k):
                self.assertEqual(set(f), {"action", "observation.state", "label.stage", "label.imitate",
                                          "label.ee_pose", "frame_index"})
                self.assertEqual(f["action"].dtype, np.float32)
                self.assertEqual(f["action"].shape, (len(ep), J))
                self.assertEqual(f["label.stage"].dtype, np.int64)
                self.assertEqual(f["label.stage"].shape, (len(ep),))
                self.assertEqual(f["label.imitate"].dtype, np.int64)
                self.assertEqual(f["label.ee_pose"].shape, (len(ep), 7))
                np.testing.assert_array_equal(f["frame_index"], np.arange(len(ep)))
                np.testing.assert_array_equal(f["action"], np.array([a for a, _, _ in ep], np.float32))
                np.testing.assert_array_equal(f["label.stage"], [lab.stage for _, _, lab in ep])
                np.testing.assert_array_equal(f["label.imitate"], [int(lab.imitate) for _, _, lab in ep])

    def test_v3_layout(self):
        """Several episodes per file, rows shuffled, fixed-size-list vectors and scalar label columns."""
        from test_v4_wp12_support import make_sidecar, write_frames
        sidecars = [make_sidecar(k, index=i, n=8 + i) for i, k in enumerate(("success", "failure", "recovery", "success"))]
        rng = np.random.default_rng(5)
        written = write_frames(self.root, sidecars, rng=rng, layout="v3", files=2)
        out = read_frames(self.root)
        self.assertEqual(sorted(out), [0, 1, 2, 3])
        for i, arrays in written.items():
            with self.subTest(episode=i):
                np.testing.assert_array_equal(out[i]["frame_index"], arrays["frame_index"])
                np.testing.assert_array_equal(out[i]["action"], arrays["action"])
                np.testing.assert_array_equal(out[i]["label.stage"], arrays["label.stage"])
                np.testing.assert_array_equal(out[i]["label.ee_pose"], arrays["label.ee_pose"])


if __name__ == "__main__":
    unittest.main()
