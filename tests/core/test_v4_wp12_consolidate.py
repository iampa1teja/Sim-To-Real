"""WP12: consolidated sidecar table task_meta/episodes.parquet (contracts §9.2, §9.1, §1.3).

Decisions pinned here (docs/v4/prompts/WP12.md): the column order is the §9.2 table read top to
bottom, left to right; ``episode_index``, ``num_frames``, ``kind`` and ``sidecar_json`` are
non-nullable, every other column is nullable; ``read_consolidated`` validates every sidecar; the
file is written to a temporary file in ``task_meta/`` and moved into place with ``os.replace``.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.datasets.consolidate import consolidate, read_consolidated
from core.datasets.task_meta import list_sidecars, read_sidecar, write_sidecar
from core.generate.errors import SchemaError
from core.utils.hashing import canonical_json
from test_v4_wp12_support import KINDS, make_dr_copy, make_legacy, make_sidecar

SCHEMA = [
    ("episode_index", pa.int64(), False), ("num_frames", pa.int64(), False), ("attempt_index", pa.int64(), True),
    ("seed", pa.uint64(), True), ("kind", pa.string(), False), ("instruction", pa.string(), True),
    ("failure_mode", pa.string(), True), ("furthest_stage", pa.string(), True),
    ("perturbation_type", pa.string(), True), ("success", pa.bool_(), True), ("scenario_key", pa.string(), True),
    ("spec_fp", pa.string(), True), ("traj_fp", pa.string(), True), ("appearance_fp", pa.string(), True),
    ("episode_fp", pa.string(), True), ("source_episode_fp", pa.string(), True), ("sidecar_json", pa.string(), False),
]


def expected_row(sc):
    """§9.2 column values derived from one sidecar."""
    fp, att, src, pert = sc["fingerprints"] or {}, sc["attempt"], sc["source"], sc["perturbation"]
    return {"episode_index": sc["episode_index"], "num_frames": sc["num_frames"],
            "attempt_index": att["index"] if att else None, "seed": att["seed"] if att else None, "kind": sc["kind"],
            "instruction": sc["instruction"], "failure_mode": sc["failure_mode"],
            "furthest_stage": sc["furthest_stage"], "perturbation_type": pert["type"] if pert else None,
            "success": sc["success"], "scenario_key": fp.get("scenario_key"), "spec_fp": fp.get("spec_fp"),
            "traj_fp": fp.get("traj_fp"), "appearance_fp": fp.get("appearance_fp"),
            "episode_fp": fp.get("episode_fp"), "source_episode_fp": src["episode_fp"] if src else None,
            "sidecar_json": canonical_json(sc)}


def mixed_sidecars(rng):
    """Success / failure / recovery originals, a DR copy and a legacy sidecar, varied lengths."""
    items = [make_sidecar(KINDS[i % 3], index=i, n=int(rng.integers(8, 30)), rng=rng, tag=f"m{i}",
                          seed=int(rng.integers(0, 2 ** 64, dtype=np.uint64))) for i in range(5)]
    items.append(make_dr_copy(items[1], index=5, tag="m5copy"))
    items.append(make_legacy(index=6))
    return items


def write_all(root, sidecars, order=None):
    for k in (range(len(sidecars)) if order is None else order):
        write_sidecar(root, sidecars[k])


def parquet_path(root):
    return Path(root) / "task_meta" / "episodes.parquet"


class SchemaTests(unittest.TestCase):
    """WP12 §9.2 the parquet has exactly the documented columns and arrow types."""

    def test_schema(self):
        """§9.2 column names, order, arrow types and nullability."""
        with tempfile.TemporaryDirectory() as tmp:
            write_all(tmp, mixed_sidecars(np.random.default_rng(0)))
            path = consolidate(tmp)
            self.assertEqual(path, parquet_path(tmp))
            schema = pq.read_schema(path)
            self.assertEqual([(f.name, f.type, f.nullable) for f in schema], SCHEMA)

    def test_empty_dataset(self):
        """§9.2 no sidecars → a zero-row table with the same schema; read_consolidated returns []."""
        with tempfile.TemporaryDirectory() as tmp:
            path = consolidate(tmp)
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 0)
            self.assertEqual([(f.name, f.type, f.nullable) for f in table.schema], SCHEMA)
            self.assertEqual(read_consolidated(tmp), [])


class RowTests(unittest.TestCase):
    """WP12 §9.2 one row per sidecar, equal to the JSON sidecars row for row."""

    def test_rows_match_sidecars(self):
        """§9.2 every column equals the value derived from its sidecar; sidecar_json is canonical_json."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecars = mixed_sidecars(np.random.default_rng(1))
            write_all(tmp, sidecars)
            rows = pq.read_table(consolidate(tmp)).to_pylist()
            self.assertEqual(len(rows), len(sidecars))
            for row, i in zip(rows, list_sidecars(tmp)):
                with self.subTest(episode=i):
                    self.assertEqual(row, expected_row(read_sidecar(tmp, i)))

    def test_nulls_for_legacy_and_values_for_copies(self):
        """§9.2 legacy rows have null attempt/seed/fingerprints; a DR copy has source_episode_fp."""
        with tempfile.TemporaryDirectory() as tmp:
            sidecars = mixed_sidecars(np.random.default_rng(2))
            write_all(tmp, sidecars)
            rows = {r["episode_index"]: r for r in pq.read_table(consolidate(tmp)).to_pylist()}
            legacy, copy_ = rows[6], rows[5]
            for key in ("attempt_index", "seed", "scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp",
                        "source_episode_fp", "success", "instruction"):
                with self.subTest(key=key):
                    self.assertIsNone(legacy[key])
            self.assertEqual(copy_["source_episode_fp"], sidecars[1]["fingerprints"]["episode_fp"])
            self.assertEqual(copy_["traj_fp"], sidecars[1]["fingerprints"]["traj_fp"])
            self.assertEqual(rows[0]["perturbation_type"], None)
            self.assertEqual(rows[1]["perturbation_type"], sidecars[1]["perturbation"]["type"])

    def test_seed_is_exact_uint64(self):
        """§1.3 seeds up to 2**64 − 1 survive as exact Python ints (uint64, not int64 or float)."""
        with tempfile.TemporaryDirectory() as tmp:
            seeds = [0, 2 ** 63, 2 ** 64 - 1, 2 ** 53 + 1]
            for i, seed in enumerate(seeds):
                write_sidecar(tmp, make_sidecar("success", index=i, seed=seed, tag=f"s{i}"))
            column = pq.read_table(consolidate(tmp)).column("seed").to_pylist()
            self.assertEqual(column, seeds)
            self.assertTrue(all(type(v) is int for v in column))

    def test_sorted_by_episode_index(self):
        """§9.2 rows are sorted by episode_index whatever order the sidecars were written in."""
        rng = np.random.default_rng(3)
        sidecars = mixed_sidecars(rng)
        with tempfile.TemporaryDirectory() as tmp:
            write_all(tmp, sidecars, order=rng.permutation(len(sidecars)))
            column = pq.read_table(consolidate(tmp)).column("episode_index").to_pylist()
            self.assertEqual(column, sorted(sc["episode_index"] for sc in sidecars))

    def test_round_trip_property(self):
        """§9.2 read_consolidated(root) == [read_sidecar(root, i) for i in list_sidecars(root)] (random datasets)."""
        for k in range(10):
            rng = np.random.default_rng(100 + k)
            with tempfile.TemporaryDirectory() as tmp, self.subTest(k=k):
                count = int(rng.integers(1, 9))
                indices = sorted(rng.choice(40, count, replace=False).tolist())
                sidecars = [make_sidecar(KINDS[int(rng.integers(3))], index=i, n=int(rng.integers(8, 40)), rng=rng,
                                         tag=f"{k}-{i}", seed=int(rng.integers(0, 2 ** 64, dtype=np.uint64)))
                            for i in indices]
                write_all(tmp, sidecars, order=rng.permutation(count))
                consolidate(tmp)
                back = read_consolidated(tmp)
                self.assertEqual(back, [read_sidecar(tmp, i) for i in indices])
                self.assertEqual(back, sidecars)
                self.assertEqual(json.dumps(back, sort_keys=True), json.dumps(sidecars, sort_keys=True))


class DeterminismTests(unittest.TestCase):
    """WP12 §9.2 consolidation is deterministic."""

    def test_same_bytes_twice_and_across_write_orders(self):
        """§9.2 the same sidecars give a byte-identical parquet, whatever the sidecar write order."""
        rng = np.random.default_rng(4)
        sidecars = mixed_sidecars(rng)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write_all(a, sidecars)
            write_all(b, sidecars, order=list(reversed(range(len(sidecars)))))
            first = consolidate(a).read_bytes()
            self.assertEqual(consolidate(a).read_bytes(), first)
            self.assertEqual(consolidate(b).read_bytes(), first)


class AtomicRebuildTests(unittest.TestCase):
    """WP12 §9.2 the parquet is rebuilt from the sidecars and replaced atomically."""

    def test_rebuild_reflects_new_sidecars(self):
        """§9.2 consolidate after adding a sidecar rewrites the table (no stale rows, no appends)."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0, tag="r0"))
            consolidate(tmp)
            write_sidecar(tmp, make_sidecar("failure", index=1, tag="r1"))
            self.assertEqual(len(read_consolidated(tmp)), 1, "read_consolidated must not rebuild")
            consolidate(tmp)
            self.assertEqual([sc["episode_index"] for sc in read_consolidated(tmp)], [0, 1])

    def test_failed_replace_keeps_old_file(self):
        """§9.2 if os.replace fails, the old parquet is byte-identical and no temporary file is left."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0, tag="a0"))
            old = consolidate(tmp).read_bytes()
            write_sidecar(tmp, make_sidecar("recovery", index=1, tag="a1"))
            before = sorted(p.name for p in (Path(tmp) / "task_meta").iterdir())
            with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                consolidate(tmp)
            self.assertEqual(parquet_path(tmp).read_bytes(), old)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "task_meta").iterdir()), before)

    def test_failed_first_write_leaves_nothing(self):
        """§9.2 a failed first consolidation leaves no episodes.parquet and no temporary file."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0, tag="b0"))
            with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                consolidate(tmp)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "task_meta").iterdir()), ["episode_000000.json"])

    def test_invalid_sidecar_writes_nothing(self):
        """§9.1 an invalid sidecar makes consolidate raise SchemaError before anything is written."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0, tag="c0"))
            old = consolidate(tmp).read_bytes()
            bad = make_sidecar("success", index=1, tag="c1")
            bad["num_frames"] += 1
            (Path(tmp) / "task_meta" / "episode_000001.json").write_text(json.dumps(bad))
            with self.assertRaises(SchemaError):
                consolidate(tmp)
            self.assertEqual(parquet_path(tmp).read_bytes(), old)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "task_meta").iterdir()),
                             ["episode_000000.json", "episode_000001.json", "episodes.parquet"])


class ReadTests(unittest.TestCase):
    """WP12 §9.2 read_consolidated."""

    def test_missing_file(self):
        """§9.2 no task_meta/episodes.parquet → FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0))
            with self.assertRaises(FileNotFoundError):
                read_consolidated(tmp)

    def test_invalid_sidecar_json(self):
        """§9.1 a row whose sidecar_json is not a valid sidecar → SchemaError."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", index=0, tag="d0"))
            table = pq.read_table(consolidate(tmp))
            bad = json.loads(table.column("sidecar_json")[0].as_py())
            bad["kind"] = "not_a_kind"
            cases = {"invalid sidecar": canonical_json(bad), "not json": "{"}
            for name, text in cases.items():
                with self.subTest(case=name):
                    column = pa.array([text], pa.string())
                    patched = table.set_column(table.schema.get_field_index("sidecar_json"),
                                               table.schema.field("sidecar_json"), column)
                    pq.write_table(patched, parquet_path(tmp))
                    with self.assertRaises(SchemaError):
                        read_consolidated(tmp)

    def test_str_and_path_roots(self):
        """§9.2 root may be str or Path."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("failure", index=0, tag="e0"))
            self.assertEqual(consolidate(str(tmp)), consolidate(Path(tmp)))
            self.assertEqual(read_consolidated(str(tmp)), read_consolidated(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
