"""WP7 main gate: the generated datasets never contain duplicate episodes (contracts §13; the owner's hard
requirement).

``v4_support.harness.run_no_duplicates_suite`` runs, for each fixture task and all three kinds, on FakeEnv +
MemoryWriter: 3 shards with random kill points (env crash mid-episode, crash before / after the episode is
committed, a torn manifest line) and resume; WP12 merge; the WP7 audit (must pass); an independent
recomputation of every fingerprint from the stored data (``assert_no_duplicates``); then a fake DR pass with 2
appearance copies per episode, audited against its source and checked again. Quotas must be met exactly.

WP9, WP10 and WP13 call the same suite for the 17 real tasks (``test_generation_{primitives,cube_tasks,
stack_bowls}.py``).
"""

import contextlib
import json
import shutil
import unittest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from core.datasets.audit import audit_dataset
from core.datasets.task_meta import read_sidecar, sidecar_path
from v4_support import sandbox
from v4_support import harness as H

TASKS = ("fx7_grip", "fx7_reach", "fx7_pick", "fx7_color")


class NoDuplicatesGate(unittest.TestCase):
    """WP7 §13 the no-duplicates gate on the WP7 fixture tasks (fx_arm5; the fast task also on fx_arm6)."""

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, TASKS)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def test_all_kinds_all_tasks(self):
        out = H.run_no_duplicates_suite(self, TASKS, work=self.box.root / "gate", seed=0)
        self.assertEqual(sorted(out), sorted(TASKS))
        for task, res in out.items():
            self.assertEqual(res["counts"]["originals"], 9, task)

    def test_second_seed_and_arm(self):
        H.run_no_duplicates_suite(self, ("fx7_grip",), work=self.box.root / "gate6", arm="fx_arm6", seed=1,
                                  quotas={"success": 4, "failure": 3, "recovery": 2}, kills_per_shard=3)

    def test_injected_copy_is_caught(self):
        """A duplicated episode (same frames and sidecar under a new index) fails the audit and the independent
        check."""
        out = H.run_no_duplicates_suite(self, ("fx7_grip",), work=self.box.root / "gate_dup", seed=2,
                                        kills_per_shard=1, dr_copies=1)
        src = out["fx7_grip"]["merged"]
        root = self.box.root / "dup"
        shutil.copytree(src, root)
        n = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
        sc = read_sidecar(root, 2)
        sc["episode_index"] = n
        sidecar_path(root, n).write_text(json.dumps(sc))
        files = sorted((root / "data").rglob("*.parquet"))
        table = pq.read_table(files[0])
        for f in files:
            t = pq.read_table(f)
            if 2 in set(t.column("episode_index").to_pylist()):
                table = t.filter(pc.equal(t.column("episode_index"), 2))
        table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index",
                                 pa.array([n] * table.num_rows, pa.int64()))
        pq.write_table(table, files[0].parent / f"file-{n:06d}.parquet")
        info = json.loads((root / "meta" / "info.json").read_text())
        info["total_episodes"] = n + 1
        (root / "meta" / "info.json").write_text(json.dumps(info))
        report = audit_dataset(root)
        self.assertFalse(report.passed)
        self.assertIn("episode_fp_unique", {c["name"] for c in report.checks if not c["passed"]})
        with self.assertRaises(AssertionError):
            H.assert_no_duplicates(self, root)


if __name__ == "__main__":
    unittest.main()
