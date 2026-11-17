"""WP7: the dataset audit — recomputed fingerprints, uniqueness, labels, counts, hold-outs, DR checks, and the
report WP12 reads (contracts §10.1, §15 WP12, §8.2, §9).

Decisions pinned here (docs/v4/prompts/WP7.md):
* ``audit_dataset`` returns ``AuditReport`` with the ten checks in the §10.1 order; ``details`` ≤ 50 strings.
* ``counts_match``: ``info.json`` ``total_episodes`` = data episodes = sidecars; indices are 0..E−1; per-episode
  frame counts equal ``num_frames``; and, when ``meta/generation_manifest.jsonl`` exists and ``check_manifest``,
  the accepted lines are exactly the originals (``source`` null) with the same fingerprints, kind, attempt
  index, seed and ``num_frames``.
* ``labels_match``: ``label.stage`` / ``label.imitate`` equal the sidecar exactly; ``label.ee_pose`` within 1e-5.
* ``fingerprints_recomputed``: traj from the data (``action``, ``observation.state``) and the sidecar's frame-0
  object positions; appearance from the sidecar colours, instruction and ``appearance``; episode from both.
* ``holdout_absent``: hold-outs from ``generation_plan.json`` when present, else from the task's ``[task.holdout]``;
  fails for a held-out colour on any object or a held-out count of the target's set.
* ``dr_sources_exist`` passes with a "skipped" detail when ``source_root`` is None.
* ``dr_appearance_distinct``: copies of one parent (``source.episode_fp``) differ from each other and from the
  parent (found in the dataset or in ``source_root``) in ``appearance_fp`` and keep the source ``traj_fp``.
* ``write_audit_report`` writes ``meta/audit_report.json`` (keys exactly the eight fields, ``checks`` as a list of
  ``{name, passed, details}``); ``read_audit_report`` raises ``SchemaError`` on anything else.
"""

import copy
import json
import shutil
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.datasets.audit import AuditReport, audit_dataset, content_digest, read_audit_report, write_audit_report
from core.datasets.task_meta import read_sidecar, sidecar_path
from core.generate.errors import AuditFailed, SchemaError
from core.utils.hashing import json_sha256
from test_v4_wp7_support import CHECKS, FAST, SandboxCase, sidecar_dicts
from v4_support import harness as H


def failed(report):
    return {c["name"] for c in report.checks if not c["passed"]}


def data_file(root, i):
    return root / "data" / "chunk-000" / f"file-{i:06d}.parquet"


def rewrite_column(root, i, name, fn):
    path = data_file(root, i)
    t = pq.read_table(path)
    col = t.column(name).to_pylist()
    t = t.set_column(t.schema.get_field_index(name), name, pa.array(fn(col), t.schema.field(name).type))
    pq.write_table(t, path)


def write_raw_sidecar(root, payload):
    sidecar_path(root, payload["episode_index"]).write_text(json.dumps(payload))


class AuditBase(SandboxCase):
    TASKS = ("fx7_grip", "fx7_color")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        q = {"success": 2, "failure": 2, "recovery": 2}
        cls.clean, _, _ = H.run_shard(H.write_config(cls.work / "g.yaml", tasks=[FAST], dst_root=cls.work / "g",
                                                     quotas=q), FAST, rig=cls.rig)
        cls.color, _, _ = H.run_shard(H.write_config(cls.work / "c.yaml", tasks=["fx7_color"], dst_root=cls.work / "c",
                                                     quotas={"success": 2, "failure": 0, "recovery": 0}),
                                      "fx7_color", rig=cls.rig)

    def assert_caught(self, root, expected, *, exactly=False, **kw):
        r = audit_dataset(root, **kw)
        self.assertFalse(r.passed)
        if exactly:
            self.assertEqual(failed(r), set(expected))
        else:
            self.assertTrue(set(expected) <= failed(r), (expected, failed(r)))
        for c in r.checks:
            self.assertLessEqual(len(c["details"]), 50)
            if not c["passed"]:
                self.assertTrue(c["details"])
        return r

    def copy(self, src=None):
        dst = self.fresh("copy") / "ds"
        shutil.copytree(src or self.clean, dst)
        return dst


class CleanDatasetTests(AuditBase):
    """WP7 §10.1 a generated dataset passes every check; the report fields and digests are as specified."""

    def test_all_checks_pass_in_order(self):
        for root in (self.clean, self.color):
            r = audit_dataset(root)
            with self.subTest(root=root.name):
                self.assertIsInstance(r, AuditReport)
                self.assertEqual(tuple(c["name"] for c in r.checks), CHECKS)
                self.assertEqual(failed(r), set())
                self.assertTrue(r.passed)
                self.assertEqual(r.v, 1)
                self.assertEqual(r.episodes, len(sidecar_dicts(root)))
                self.assertEqual(r.root, str(root))
                self.assertTrue(all(set(c) == {"name", "passed", "details"} for c in r.checks))

    def test_digests(self):
        from test_v4_wp12_support import content_digest as wp12_digest
        r = audit_dataset(self.clean)
        self.assertEqual(r.content_digest, wp12_digest(self.clean))
        self.assertEqual(content_digest(self.clean), r.content_digest)
        fps = sorted(sc["fingerprints"]["episode_fp"] for sc in sidecar_dicts(self.clean).values())
        self.assertEqual(r.fingerprints_digest, json_sha256(fps))

    def test_report_round_trip_and_digest_excludes_itself(self):
        root = self.copy()
        r = audit_dataset(root)
        path = write_audit_report(root, r)
        self.assertEqual(path, root / "meta" / "audit_report.json")
        d = json.loads(path.read_text())
        self.assertEqual(set(d), {"v", "root", "passed", "episodes", "checks", "content_digest", "fingerprints_digest",
                                  "created_at"})
        self.assertIsInstance(d["checks"], list)
        self.assertEqual(read_audit_report(root), r)
        self.assertEqual(content_digest(root), r.content_digest)

    def test_read_rejects_malformed_reports(self):
        root = self.copy()
        write_audit_report(root, audit_dataset(root))
        path = root / "meta" / "audit_report.json"
        good = json.loads(path.read_text())
        for bad in ({**good, "extra": 1}, {k: v for k, v in good.items() if k != "passed"}, {**good, "passed": "yes"},
                    {**good, "content_digest": "abc"}, {**good, "checks": [{"name": "x"}]}):
            path.write_text(json.dumps(bad))
            with self.subTest(bad=sorted(bad)):
                with self.assertRaises(SchemaError):
                    read_audit_report(root)
        path.write_text("{not json")
        with self.assertRaises(SchemaError):
            read_audit_report(root)

    def test_wp12_accepts_the_report(self):
        """§15 WP12: merge and upload read the real audit record; a later change makes it stale."""
        from core.datasets.consolidate import consolidate
        from core.datasets.merge import ShardSummary, plan_merge
        from core.datasets.upload import HubStorage, plan_upload
        from test_v4_wp12_support import FakeHubApi
        root = self.copy()
        consolidate(root)
        write_audit_report(root, audit_dataset(root))
        plan = plan_merge([ShardSummary.from_root(root)])
        self.assertEqual(plan.total, len(sidecar_dicts(root)))
        up = plan_upload(root, HubStorage(FakeHubApi("org/fx"), "org/fx"))
        self.assertTrue(up.items)
        rewrite_column(root, 0, "task", lambda col: [c + "!" for c in col])
        with self.assertRaises(AuditFailed):
            plan_merge([ShardSummary.from_root(root)])
        with self.assertRaises(AuditFailed):
            plan_upload(root, HubStorage(FakeHubApi("org/fx"), "org/fx"))


class CorruptionTests(AuditBase):
    """WP7 §10.1 each corruption is caught by the check named for it (and the report does not pass)."""

    def test_copied_episode(self):
        """The same frames and sidecar under a new index (a duplicate episode)."""
        root = self.copy()
        n = len(sidecar_dicts(root))
        sc = read_sidecar(root, 1)
        sc["episode_index"] = n
        write_raw_sidecar(root, sc)
        t = pq.read_table(data_file(root, 1))
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                         pa.array([n] * t.num_rows, pa.int64()))
        pq.write_table(t, data_file(root, n))
        info = json.loads((root / "meta" / "info.json").read_text())
        info["total_episodes"] = n + 1
        (root / "meta" / "info.json").write_text(json.dumps(info))
        self.assert_caught(root, {"episode_fp_unique", "traj_fp_unique_originals", "scenario_key_unique_originals"})

    def test_modified_frame(self):
        root = self.copy()

        def bump(col):
            col[3] = [v + 1.0 for v in col[3]]
            return col
        rewrite_column(root, 1, "action", bump)
        self.assert_caught(root, {"fingerprints_recomputed"}, exactly=True)

    def test_modified_state(self):
        root = self.copy()

        def bump(col):
            col[0] = [v + 0.5 for v in col[0]]
            return col
        rewrite_column(root, 0, "observation.state", bump)
        self.assert_caught(root, {"fingerprints_recomputed"}, exactly=True)

    def test_deleted_sidecar(self):
        root = self.copy()
        last = max(sidecar_dicts(root))
        sidecar_path(root, last).unlink()
        self.assert_caught(root, {"counts_match"})

    def test_swapped_label_column(self):
        root = self.copy()
        sc = read_sidecar(root, 0)
        self.assertNotEqual(sc["stage"], sc["stage"][::-1])
        rewrite_column(root, 0, "label.stage", lambda col: col[::-1])
        self.assert_caught(root, {"labels_match"}, exactly=True)

    def test_imitate_column(self):
        root = self.copy()
        rewrite_column(root, 2, "label.imitate", lambda col: [[1 - v[0]] for v in col])
        self.assert_caught(root, {"labels_match"}, exactly=True)

    def test_edited_manifest_line(self):
        """An accepted line that disagrees with its sidecar in a fingerprint, the seed or the frame count."""
        for key, value in (("episode_fp", "f" * 64), ("traj_fp", "e" * 64), ("seed", 12345), ("num_frames", 1)):
            with self.subTest(key=key):
                root = self.copy()
                path = root / "meta" / "generation_manifest.jsonl"
                lines = [json.loads(x) for x in path.read_text().splitlines()]
                k = next(i for i, d in enumerate(lines) if d["status"] == "accepted")
                lines[k][key] = value
                path.write_text("".join(json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n" for d in lines))
                self.assert_caught(root, {"counts_match"}, exactly=True)
                self.assertTrue(audit_dataset(root, check_manifest=False).passed)

    def test_missing_manifest_line(self):
        root = self.copy()
        path = root / "meta" / "generation_manifest.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        k = max(i for i, x in enumerate(lines) if '"status":"accepted"' in x)
        path.write_text("".join(lines[:k] + lines[k + 1:]))
        self.assert_caught(root, {"counts_match"}, exactly=True)

    def test_held_out_colour(self):
        root = self.copy(self.color)
        sc = read_sidecar(root, 0)
        other = next(n for n, o in sc["objects"].items() if o["role"] != "target" and o["set"] == "cubes")
        sc["objects"][other]["color"] = "purple"
        write_raw_sidecar(root, sc)
        self.assert_caught(root, {"holdout_absent"})

    def test_held_out_count(self):
        """Hold-outs come from generation_plan.json when present: declaring the dataset's count as held out
        makes the check fail."""
        root = self.copy(self.color)
        sc = read_sidecar(root, 0)
        n = sum(1 for o in sc["objects"].values() if o["set"] == "cubes")
        plan = json.loads((root / "meta" / "generation_plan.json").read_text())
        plan["holdout"]["counts"] = [n]
        (root / "meta" / "generation_plan.json").write_text(json.dumps(plan))
        self.assert_caught(root, {"holdout_absent"}, exactly=True)


class DRTests(AuditBase):
    """WP7 §10.1 DR datasets: originals plus copies; sources exist; copies differ in appearance."""

    def dr(self, src=None, **kw):
        return H.fake_dr(src or self.color, self.fresh("dr") / "ds", **kw)

    def test_fake_dr_passes(self):
        for src in (self.clean, self.color):
            dr = self.dr(src)
            with self.subTest(src=src.name):
                r = audit_dataset(dr, source_root=src)
                self.assertEqual(failed(r), set())
                sides = sidecar_dicts(dr)
                self.assertEqual(len(sides), 3 * len(sidecar_dicts(src)))
                skipped = audit_dataset(dr)
                self.assertTrue(skipped.passed)
                det = [c for c in skipped.checks if c["name"] == "dr_sources_exist"][0]
                self.assertTrue(det["passed"])
                self.assertTrue(any("skip" in d.lower() for d in det["details"]))

    def test_copy_without_source(self):
        dr = self.dr(self.color)
        self.assert_caught(dr, {"dr_sources_exist"}, exactly=True, source_root=self.clean)

    def test_copy_with_parent_appearance(self):
        dr = self.dr(self.color, include_originals=False)
        sides = sidecar_dicts(dr)
        i, sc = next((i, sc) for i, sc in sides.items() if sc["source"] is not None)
        parent = read_sidecar(self.color, sc["source"]["episode_index"])
        bad = copy.deepcopy(sc)
        for key in ("instruction", "objects", "appearance"):
            bad[key] = copy.deepcopy(parent[key])
        bad["fingerprints"]["appearance_fp"] = parent["fingerprints"]["appearance_fp"]
        bad["fingerprints"]["episode_fp"] = parent["fingerprints"]["episode_fp"]
        write_raw_sidecar(dr, bad)
        r = audit_dataset(dr, source_root=self.color)
        self.assertIn("dr_appearance_distinct", failed(r))
        self.assertFalse(r.passed)

    def test_sibling_copies_with_same_appearance(self):
        dr = self.dr(self.color, include_originals=False)
        sides = sidecar_dicts(dr)
        groups = {}
        for i, sc in sorted(sides.items()):
            groups.setdefault(sc["source"]["episode_fp"], []).append(i)
        a, b = next(g for g in groups.values() if len(g) >= 2)[:2]
        first, second = sides[a], copy.deepcopy(sides[b])
        for key in ("instruction", "objects", "appearance"):
            second[key] = copy.deepcopy(first[key])
        second["fingerprints"] = dict(first["fingerprints"])
        write_raw_sidecar(dr, second)
        r = audit_dataset(dr, source_root=self.color)
        self.assertTrue({"dr_appearance_distinct", "episode_fp_unique"} <= failed(r), failed(r))


if __name__ == "__main__":
    unittest.main()
