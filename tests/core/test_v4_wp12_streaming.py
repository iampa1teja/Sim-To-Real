"""WP12: episode selection and streaming helpers (contracts §10.4, §9.2, §9.3, §11, §1.6).

``iter_episodes(root, *, episodes=None)`` is the local counterpart of ``make_streaming_dataset``
asked for by the WP12 brief (a contract addition, docs/v4/prompts/WP12.md): it yields
``(sidecar, frames)`` per episode in ascending ``episode_index``; ``frames`` has the §10.1
``read_frames`` keys with these shapes and dtypes: ``action`` / ``observation.state`` ``(N, J)``
float32, ``label.stage`` / ``label.imitate`` ``(N,)`` int64, ``label.ee_pose`` ``(N, 7)`` float32,
``frame_index`` ``(N,)`` int64, rows in ``frame_index`` order. It reads lite datasets (§11, list
columns, one file per episode) and LeRobot v3 data files (fixed-size lists, scalar shape-(1,)
columns, several episodes per file, any row order) alike.

``select_episodes`` decisions: the result is sorted ascending without repeats; ``kinds`` accepts
``Kind`` members or their values (unknown → ValueError); ``success=True/False`` keeps rows whose
``success`` is exactly that value (null never matches); ``exclude_holdout_of`` (duck-typed:
``.holdout.colors``, ``.holdout.counts``) drops rows where an object's colour is held out or where the
number of objects of one object set (same non-null ``set``) is a held-out count.
"""

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from core.datasets.consolidate import consolidate, read_consolidated
from core.datasets.streaming import iter_episodes, make_streaming_dataset, select_episodes
from core.datasets.task_meta import read_sidecar
from core.generate.errors import SchemaError
from core.generate.protocols import Kind
from test_v4_wp12_support import KINDS, build_dataset, make_dr_copy, make_legacy, make_sidecar

FRAME_KEYS = ("action", "observation.state", "label.stage", "label.imitate", "label.ee_pose", "frame_index")


def dataset_sidecars(rng, count=5):
    items = [make_sidecar(KINDS[i % 3], index=i, n=int(rng.integers(8, 30)), rng=rng, tag=f"st{i}")
             for i in range(count)]
    items.append(make_dr_copy(items[0], index=count, tag="st-copy"))
    return items


def collect(root, **kwargs):
    return list(iter_episodes(root, **kwargs))


def holdout(colors=(), counts=()):
    return types.SimpleNamespace(holdout=types.SimpleNamespace(colors=tuple(colors), counts=tuple(counts)))


class IterEpisodesTests(unittest.TestCase):
    """WP12 §10.4 iter_episodes over a local dataset."""

    def test_order_labels_and_sidecars(self):
        """§10.4 episodes in order; frames as written; labels equal the sidecar stage/imitate/ee_pose (§9.3)."""
        rng = np.random.default_rng(0)
        sidecars = dataset_sidecars(rng)
        with tempfile.TemporaryDirectory() as tmp:
            frames = build_dataset(tmp, sidecars, rng=rng, consolidate=True)
            items = collect(tmp)
            self.assertEqual([sc["episode_index"] for sc, _ in items], list(range(len(sidecars))))
            for sc, fr in items:
                e = sc["episode_index"]
                with self.subTest(episode=e):
                    self.assertEqual(sc, read_sidecar(tmp, e))
                    self.assertEqual(tuple(sorted(fr)), tuple(sorted(FRAME_KEYS)))
                    n = sc["num_frames"]
                    shapes = {"action": (n, 6), "observation.state": (n, 6), "label.stage": (n,),
                              "label.imitate": (n,), "label.ee_pose": (n, 7), "frame_index": (n,)}
                    for key in FRAME_KEYS:
                        self.assertEqual(fr[key].shape, shapes[key], key)
                        self.assertEqual(fr[key].dtype, frames[e][key].dtype, key)
                        np.testing.assert_array_equal(fr[key], frames[e][key], key)
                    np.testing.assert_array_equal(fr["label.stage"], sc["stage"])
                    np.testing.assert_array_equal(fr["label.imitate"], np.asarray(sc["imitate"], np.int64))
                    ee = np.concatenate([sc["ee_pose"]["position"], sc["ee_pose"]["orientation_wxyz"]], axis=1)
                    np.testing.assert_allclose(fr["label.ee_pose"], ee.astype(np.float32), rtol=0, atol=0)
                    np.testing.assert_array_equal(fr["frame_index"], np.arange(n))

    def test_lite_and_v3_layouts_agree(self):
        """§10.1/§11 a lite dataset and a v3-style dataset with the same content yield equal episodes."""
        sidecars = dataset_sidecars(np.random.default_rng(1), count=6)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            build_dataset(a, sidecars, rng=np.random.default_rng(7), layout="lite")
            build_dataset(b, sidecars, rng=np.random.default_rng(7), layout="v3")
            lite, v3 = collect(a), collect(b)
            self.assertEqual(len(lite), len(v3))
            for (sa, fa), (sb, fb) in zip(lite, v3):
                with self.subTest(episode=sa["episode_index"]):
                    self.assertEqual(sa, sb)
                    for key in FRAME_KEYS:
                        self.assertEqual(fa[key].dtype, fb[key].dtype, key)
                        np.testing.assert_array_equal(fa[key], fb[key], key)

    def test_with_and_without_consolidated_parquet(self):
        """§9.2 sidecars come from episodes.parquet when present, else from the JSON sidecars: same result."""
        sidecars = dataset_sidecars(np.random.default_rng(2))
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, sidecars, rng=np.random.default_rng(3))
            plain = collect(tmp)
            consolidate(tmp)
            self.assertEqual(read_consolidated(tmp), [sc for sc, _ in plain])
            with_parquet = collect(tmp)
            self.assertEqual([sc for sc, _ in plain], [sc for sc, _ in with_parquet])

    def test_episode_subset(self):
        """§10.4 episodes= restricts the output, always in ascending order; an unknown index → ValueError."""
        sidecars = dataset_sidecars(np.random.default_rng(4))
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, sidecars, rng=np.random.default_rng(5))
            self.assertEqual([sc["episode_index"] for sc, _ in collect(tmp, episodes=[4, 1, 2])], [1, 2, 4])
            self.assertEqual(collect(tmp, episodes=[]), [])
            with self.assertRaises(ValueError):
                collect(tmp, episodes=[99])

    def test_mismatches(self):
        """§11 frame count ≠ sidecar num_frames, or an episode without data → SchemaError."""
        sidecars = dataset_sidecars(np.random.default_rng(6), count=3)
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, sidecars, rng=np.random.default_rng(6))
            (Path(tmp) / "data" / "chunk-000" / "file-000001.parquet").unlink()
            with self.assertRaises(SchemaError):
                collect(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            short = copy.deepcopy(sidecars)
            build_dataset(tmp, short, rng=np.random.default_rng(6))
            longer = make_sidecar(short[1]["kind"], index=1, n=short[1]["num_frames"] + 1, tag="st1")
            from core.datasets.task_meta import write_sidecar
            write_sidecar(tmp, longer)
            with self.assertRaises(SchemaError):
                collect(tmp)


class SelectEpisodesTests(unittest.TestCase):
    """WP12 §10.4 select_episodes is a pure filter over read_consolidated rows."""

    def rows(self):
        rows = []
        for i in range(9):
            sc = make_sidecar(KINDS[i % 3], index=i, tag=f"sel{i}", colors=(("red", "blue", "purple")[i // 3], None))
            rows.append(sc)
        rows[4] = dict(rows[4], success=None, failure_mode=None)
        return rows

    def test_filters(self):
        """§10.4 kinds (Kind or value), success (exact match, null never matches) and no filter."""
        rows = self.rows()
        self.assertEqual(select_episodes(rows), list(range(9)))
        self.assertEqual(select_episodes(rows, kinds=[Kind.FAILURE]), [1, 4, 7])
        self.assertEqual(select_episodes(rows, kinds=["success", "recovery"]), [0, 2, 3, 5, 6, 8])
        self.assertEqual(select_episodes(rows, success=False), [1, 7])
        self.assertEqual(select_episodes(rows, success=True), [0, 2, 3, 5, 6, 8])
        self.assertEqual(select_episodes(rows, kinds=["recovery"], success=True), [2, 5, 8])
        self.assertEqual(select_episodes(rows, kinds=[]), [])
        with self.assertRaises(ValueError):
            select_episodes(rows, kinds=["teleop"])

    def test_holdout(self):
        """§10.4/§4.5 exclude_holdout_of drops held-out colours and held-out object-set counts."""
        rows = self.rows()
        self.assertEqual(select_episodes(rows, exclude_holdout_of=holdout(colors=["purple"])), list(range(6)))
        self.assertEqual(select_episodes(rows, exclude_holdout_of=holdout(colors=["blue", "green"])),
                         [0, 1, 2, 6, 7, 8])
        two = copy.deepcopy(rows[0])
        two["episode_index"] = 9
        two["objects"]["cube_1"] = dict(two["objects"]["cube_0"], role=None, color="green")
        rows.append(two)
        self.assertEqual(select_episodes(rows, exclude_holdout_of=holdout(counts=[2])), list(range(9)))
        self.assertEqual(select_episodes(rows, exclude_holdout_of=holdout(counts=[1])), [9])
        self.assertEqual(select_episodes(rows, exclude_holdout_of=holdout()), list(range(10)))

    def test_sorted_unique_and_pure(self):
        """§10.4 the result is sorted without repeats, and the rows are not mutated."""
        rows = self.rows()
        shuffled = [rows[i] for i in (5, 0, 8, 3, 1)] + [rows[0]]
        snapshot = copy.deepcopy(shuffled)
        self.assertEqual(select_episodes(shuffled), [0, 1, 3, 5, 8])
        self.assertEqual(shuffled, snapshot)

    def test_property_against_definition(self):
        """§10.4 random rows: the result equals the filter definition applied row by row."""
        palette = ["red", "green", "blue", "purple", "orange"]
        for k in range(20):
            rng = np.random.default_rng(300 + k)
            rows = []
            for i in range(int(rng.integers(0, 15))):
                sc = make_sidecar(KINDS[int(rng.integers(3))], index=i, tag=f"p{k}-{i}",
                                  colors=(palette[int(rng.integers(5))], None))
                if rng.random() < 0.4:
                    sc["objects"]["cube_1"] = dict(sc["objects"]["cube_0"], role=None,
                                                   color=palette[int(rng.integers(5))])
                if rng.random() < 0.2:
                    sc = dict(sc, success=None)
                rows.append(sc)
            kinds = None if rng.random() < 0.3 else list(rng.choice(KINDS, int(rng.integers(1, 3)), replace=False))
            success = [None, True, False][int(rng.integers(3))]
            spec = holdout(colors=list(rng.choice(palette, int(rng.integers(0, 3)), replace=False)),
                           counts=[int(c) for c in rng.choice([1, 2, 3], int(rng.integers(0, 2)), replace=False)])

            def keep(row):
                if kinds is not None and row["kind"] not in kinds:
                    return False
                if success is not None and row["success"] is not success:
                    return False
                if any(o["color"] in spec.holdout.colors for o in row["objects"].values()):
                    return False
                sets = {}
                for o in row["objects"].values():
                    if o["set"] is not None:
                        sets[o["set"]] = sets.get(o["set"], 0) + 1
                return not any(c in spec.holdout.counts for c in sets.values())

            with self.subTest(k=k):
                self.assertEqual(select_episodes(rows, kinds=kinds, success=success, exclude_holdout_of=spec),
                                 sorted(r["episode_index"] for r in rows if keep(r)))

    def test_over_consolidated_rows(self):
        """§10.4 works directly on read_consolidated output, legacy rows included."""
        rng = np.random.default_rng(9)
        sidecars = dataset_sidecars(rng, count=3) + [make_legacy(index=4)]
        with tempfile.TemporaryDirectory() as tmp:
            from core.datasets.task_meta import write_sidecar
            for sc in sidecars:
                write_sidecar(tmp, sc)
            consolidate(tmp)
            rows = read_consolidated(tmp)
            self.assertEqual(select_episodes(rows, kinds=["success"]), [0, 3, 4])
            self.assertEqual(select_episodes(rows, success=True), [0, 2, 3])


class MakeStreamingDatasetTests(unittest.TestCase):
    """WP12 §10.4/§1.6 make_streaming_dataset imports LeRobot lazily (container only)."""

    def _fake_lerobot(self):
        calls = []

        class StreamingLeRobotDataset:
            def __init__(self, repo_id, *args, **kwargs):
                calls.append((repo_id, args, kwargs))

        lerobot = types.ModuleType("lerobot")
        datasets = types.ModuleType("lerobot.datasets")
        streaming = types.ModuleType("lerobot.datasets.streaming_dataset")
        datasets.StreamingLeRobotDataset = streaming.StreamingLeRobotDataset = StreamingLeRobotDataset
        datasets.streaming_dataset = streaming
        lerobot.datasets = datasets
        modules = {"lerobot": lerobot, "lerobot.datasets": datasets, "lerobot.datasets.streaming_dataset": streaming}
        return modules, StreamingLeRobotDataset, calls

    def test_passes_arguments_through(self):
        """§10.4 StreamingLeRobotDataset(repo_id, episodes=[...], **kwargs) is built and returned."""
        modules, cls, calls = self._fake_lerobot()
        with mock.patch.dict(sys.modules, modules):
            ds = make_streaming_dataset("org/fx", episodes=(np.int64(3), 1), buffer_size=10, seed=5)
            self.assertIsInstance(ds, cls)
            make_streaming_dataset("org/fx")
        (repo, args, kwargs), (repo2, args2, kwargs2) = calls
        self.assertEqual(repo, "org/fx")
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"episodes": [3, 1], "buffer_size": 10, "seed": 5})
        self.assertTrue(all(type(e) is int for e in kwargs["episodes"]))
        self.assertEqual((repo2, args2, kwargs2.get("episodes")), ("org/fx", (), None))

    def test_without_lerobot(self):
        """§1.6 on the host (no lerobot) the call raises ImportError."""
        blocked = {name: None for name in ("lerobot", "lerobot.datasets", "lerobot.datasets.streaming_dataset")}
        with mock.patch.dict(sys.modules, blocked), self.assertRaises(ImportError):
            make_streaming_dataset("org/fx")


if __name__ == "__main__":
    unittest.main()
