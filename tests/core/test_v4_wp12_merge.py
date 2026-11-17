"""WP12: merging shard datasets (contracts §10.2, §8.3, §8.5, §9.1, §11, §1.5).

Shards are small lite datasets (§11) written by ``test_v4_wp12_support.build_shard`` with a passing
audit-record stub (§10.1 fields, current content digest). Decisions pinned here
(docs/v4/prompts/WP12.md):

* ``ShardSummary.from_root(root)`` builds a summary: ``episode_fps`` of all episodes and
  ``scenario_keys`` of the originals (``source`` null), in episode order; ``report`` is the audit record.
* Canonical order: shards are sorted by (``generation_plan.json`` shard index, or 0 without a plan;
  then the audit ``content_digest``). Offsets are cumulative in that order, so the merge does not
  depend on the order the shards are listed in; ``MergePlan.index_maps[k]`` belongs to ``shards[k]``.
* Refusals: ``AuditFailed`` (missing / failed / stale audit), ``ConfigError`` (shards differ in
  ``meta/info.json`` codebase_version / fps / features / robot_type, or in the generation plan's
  task / config_sha256 / task_toml_sha256 / arm_files_sha256, or only some have a plan),
  ``DuplicateError`` (an ``episode_fp``, an original's ``scenario_key`` or a manifest
  ``(task, kind, attempt_index)`` in two shards). Nothing is written when ``plan_merge`` refuses.
* A shard without ``meta/generation_manifest.jsonl`` has an empty manifest (a shard whose quota was
  0 writes no line); the merged manifest exists iff some shard has one.
* Lite shards (``codebase_version "v3.0-lite"``) are merged on the host by ``merge_frames``:
  one file per episode, ``episode_index`` renumbered, ``index`` contiguous over the merged dataset,
  every other column unchanged; ``meta/info.json`` gets ``total_episodes``.
"""

import io
import itertools
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from core.datasets.consolidate import read_consolidated
from core.datasets.merge import MergePlan, ShardSummary, apply_merge_meta, merge_frames, plan_merge
from core.datasets.task_meta import list_sidecars, read_sidecar
from core.generate.errors import AuditFailed, ConfigError, DuplicateError, GenerationError, SchemaError
from core.generate.manifest import Manifest
from core.scripts import merge_shards
from test_v4_wp12_support import (AUDIT_REL, MANIFEST_REL, build_shard, content_digest, features, make_dr_copy,
                                  make_legacy, make_sidecar, other_line, read_table_by_episode, tree_bytes,
                                  write_audit)
from v4_support import assert_frozen

COUNTS = (2, 3, 2)


def make_shards(tmp, counts=COUNTS, *, n=None, per_shard=None, **kwargs):
    """Shards ``i`` of ``n`` (default len(counts)) at ``tmp/shard{i}``; ``per_shard[i]`` adds build_shard kwargs."""
    n = len(counts) if n is None else n
    roots, sidecars, frames = [], [], []
    for i, count in enumerate(counts):
        root = Path(tmp) / f"shard{i}"
        extra = dict(kwargs, **(per_shard or {}).get(i, {}))
        sc, fr = build_shard(root, i, n, episodes=count, **extra)
        roots.append(root)
        sidecars.append(sc)
        frames.append(fr)
    return roots, sidecars, frames


def run_script(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = merge_shards.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def merge_api(roots, dst, repo_id="local/fx_merged"):
    shards = [ShardSummary.from_root(r) for r in roots]
    plan = plan_merge(shards)
    merge_frames(plan, shards, dst, repo_id)
    apply_merge_meta(plan, shards, dst)
    return shards, plan


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


class ShardSummaryTests(unittest.TestCase):
    """WP12 §10.2 ShardSummary and its from_root constructor."""

    def test_from_root_fields(self):
        """§10.2 episodes, episode_fps (all, episode order), scenario_keys (originals only), report."""
        with tempfile.TemporaryDirectory() as tmp:
            def add_copy(sidecars):
                sidecars.append(make_dr_copy(sidecars[0], index=len(sidecars), tag="copy-of-0"))
            roots, sidecars, _ = make_shards(tmp, (3,), per_shard={0: {"sidecar_hook": add_copy}})
            summary = ShardSummary.from_root(roots[0])
            assert_frozen(summary, deep=False)
            self.assertEqual(summary.root, roots[0])
            self.assertIsInstance(summary.root, Path)
            self.assertEqual(summary.episodes, 4)
            self.assertEqual(summary.episode_fps, tuple(sc["fingerprints"]["episode_fp"] for sc in sidecars[0]))
            self.assertEqual(summary.scenario_keys, tuple(sc["fingerprints"]["scenario_key"] for sc in sidecars[0][:3]))
            self.assertIs(summary.report.passed, True)
            self.assertEqual(summary.report.content_digest, content_digest(roots[0]))
            self.assertEqual(ShardSummary.from_root(str(roots[0])), summary)

    def test_from_root_refuses(self):
        """§1.5 no or malformed audit record → AuditFailed; a sidecar gap or a legacy sidecar → SchemaError."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (2,), audit=False)
            with self.assertRaises(AuditFailed):
                ShardSummary.from_root(roots[0])
            for text in ("{", json.dumps({"v": 1, "passed": True})):
                with self.subTest(record=text):
                    (roots[0] / AUDIT_REL).write_text(text)
                    with self.assertRaises(AuditFailed):
                        ShardSummary.from_root(roots[0])
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (3,))
            (roots[0] / "task_meta" / "episode_000001.json").unlink()
            write_audit(roots[0])
            with self.assertRaises(SchemaError):
                ShardSummary.from_root(roots[0])
        with tempfile.TemporaryDirectory() as tmp:
            def legacy(sidecars):
                sidecars.append(make_legacy(index=len(sidecars)))
            roots, _, _ = make_shards(tmp, (1,), per_shard={0: {"sidecar_hook": legacy}})
            with self.assertRaises(SchemaError):
                ShardSummary.from_root(roots[0])


class PlanTests(unittest.TestCase):
    """WP12 §10.2 plan_merge index maps."""

    def test_index_maps_every_listing_order(self):
        """§10.2 new = offset + old with offsets cumulative in shard order, for every listing order."""
        counts = (2, 3, 1)
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, counts)
            offsets = [0, 2, 5]
            for perm in itertools.permutations(range(3)):
                with self.subTest(order=perm):
                    shards = [ShardSummary.from_root(roots[i]) for i in perm]
                    plan = plan_merge(shards)
                    self.assertIsInstance(plan, MergePlan)
                    self.assertEqual(plan.total, 6)
                    self.assertEqual(len(plan.index_maps), 3)
                    for k, i in enumerate(perm):
                        self.assertEqual(dict(plan.index_maps[k]), {old: offsets[i] + old for old in range(counts[i])})

    def test_single_and_empty(self):
        """§10.2 one shard maps onto itself; no shard → ValueError; an empty shard takes no indices."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (3, 0, 2))
            plan = plan_merge([ShardSummary.from_root(roots[0])])
            self.assertEqual((plan.total, dict(plan.index_maps[0])), (3, {0: 0, 1: 1, 2: 2}))
            plan = plan_merge([ShardSummary.from_root(r) for r in roots])
            self.assertEqual([dict(m) for m in plan.index_maps], [{0: 0, 1: 1, 2: 2}, {}, {0: 3, 1: 4}])
        with self.assertRaises(ValueError):
            plan_merge([])

    def test_report_is_duck_typed(self):
        """§10.2 plan_merge reads report.passed and report.content_digest of any report object."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (1, 1))
            good = [ShardSummary.from_root(r) for r in roots]
            current = content_digest(roots[1])
            cases = {"failed": types.SimpleNamespace(passed=False, content_digest=current),
                     "stale": types.SimpleNamespace(passed=True, content_digest="0" * 64)}
            for name, report in cases.items():
                with self.subTest(case=name):
                    bad = ShardSummary(root=roots[1], episodes=good[1].episodes, episode_fps=good[1].episode_fps,
                                       scenario_keys=good[1].scenario_keys, report=report)
                    with self.assertRaises(AuditFailed):
                        plan_merge([good[0], bad])
            ok = ShardSummary(root=roots[1], episodes=1, episode_fps=good[1].episode_fps,
                              scenario_keys=good[1].scenario_keys,
                              report=types.SimpleNamespace(passed=True, content_digest=current))
            self.assertEqual(plan_merge([good[0], ok]).total, 2)


class MergedOutputTests(unittest.TestCase):
    """WP12 §10.2 merged sidecars, frames, info.json and manifest."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.roots, cls.sidecars, cls.frames = make_shards(tmp, COUNTS)
        cls.before = [tree_bytes(r) for r in cls.roots]
        cls.dst = tmp / "merged"
        cls.code, cls.out, cls.err = run_script(*[cls.roots[i] for i in (2, 0, 1)], "--dst_root", cls.dst,
                                                "--repo_id", "local/fx_merged")
        cls.offsets = [0, COUNTS[0], COUNTS[0] + COUNTS[1]]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_script_succeeds(self):
        """§10.2 merge_shards exits 0 and leaves the shards untouched."""
        self.assertEqual(self.code, 0, self.err)
        self.assertEqual([tree_bytes(r) for r in self.roots], self.before)

    def test_sidecars_renumbered_and_preserved(self):
        """§10.2 contiguous indices in shard order; each sidecar equal to its source except episode_index."""
        self.assertEqual(list_sidecars(self.dst), list(range(sum(COUNTS))))
        for i, shard in enumerate(self.sidecars):
            for old, sc in enumerate(shard):
                with self.subTest(shard=i, episode=old):
                    new = self.offsets[i] + old
                    merged = read_sidecar(self.dst, new)
                    self.assertEqual(merged["episode_index"], new)
                    self.assertEqual(without(merged, "episode_index"), without(sc, "episode_index"))
                    self.assertEqual(merged["fingerprints"], sc["fingerprints"])
                    self.assertEqual(merged["attempt"], sc["attempt"])

    def test_frames_preserved(self):
        """§10.2/§11 every episode's frames equal its source; episode_index renumbered; index contiguous."""
        merged = read_table_by_episode(self.dst)
        self.assertEqual(sorted(merged), list(range(sum(COUNTS))))
        sources = [read_table_by_episode(r) for r in self.roots]
        start = 0
        for i, shard in enumerate(self.sidecars):
            for old in range(len(shard)):
                new = self.offsets[i] + old
                with self.subTest(shard=i, episode=old):
                    table, source = merged[new], sources[i][old]
                    self.assertEqual(table.schema, source.schema)
                    self.assertEqual(table.column("episode_index").to_pylist(), [new] * table.num_rows)
                    self.assertEqual(table.column("index").to_pylist(), list(range(start, start + table.num_rows)))
                    for name in table.column_names:
                        if name not in ("episode_index", "index"):
                            self.assertEqual(table.column(name).to_pylist(), source.column(name).to_pylist(), name)
                    np.testing.assert_array_equal(np.asarray(table.column("label.stage").to_pylist()).ravel(),
                                                  self.frames[i][old]["label.stage"])
                    start += table.num_rows
        files = sorted(p.relative_to(self.dst).as_posix() for p in (self.dst / "data").rglob("*") if p.is_file())
        self.assertEqual(files, [f"data/chunk-000/file-{e:06d}.parquet" for e in range(sum(COUNTS))])

    def test_info_json(self):
        """§11 merged meta/info.json: same codebase_version, fps and features; total_episodes = total."""
        info = json.loads((self.dst / "meta" / "info.json").read_text())
        source = json.loads((self.roots[0] / "meta" / "info.json").read_text())
        self.assertEqual(info, dict(source, total_episodes=sum(COUNTS)))

    def test_manifest_concatenated(self):
        """§8.3/§10.2 manifests concatenated in shard order; no line lost or duplicated; episode_index remapped."""
        merged = [line.to_json() for line in Manifest(self.dst / MANIFEST_REL).lines()]
        expected = []
        for i, root in enumerate(self.roots):
            for line in Manifest(root / MANIFEST_REL).lines():
                d = line.to_json()
                if d["episode_index"] is not None:
                    d["episode_index"] += self.offsets[i]
                expected.append(d)
        self.assertEqual(merged, expected)
        accepted = Manifest(self.dst / MANIFEST_REL).accepted()
        self.assertEqual([l.episode_index for l in accepted], list(range(sum(COUNTS))))
        for line in accepted:
            self.assertEqual(line.episode_fp, read_sidecar(self.dst, line.episode_index)["fingerprints"]["episode_fp"])

    def test_consolidated(self):
        """§9.2 merge_shards consolidates the destination: read_consolidated equals the merged sidecars."""
        self.assertEqual(read_consolidated(self.dst), [read_sidecar(self.dst, i) for i in list_sidecars(self.dst)])

    def test_no_audit_copied(self):
        """§10.1 the merged dataset carries no audit record (it must be audited itself)."""
        self.assertFalse((self.dst / AUDIT_REL).exists())


class DeterminismTests(unittest.TestCase):
    """WP12 §10.2 the merge does not depend on the order the shards are listed in."""

    def _merged_trees(self, roots, tmp):
        trees = []
        for k, perm in enumerate([(0, 1, 2), (2, 1, 0), (1, 2, 0)]):
            dst = Path(tmp) / f"out{k}"
            code, _, err = run_script(*[roots[i] for i in perm], "--dst_root", dst, "--repo_id", "local/fx")
            self.assertEqual(code, 0, err)
            trees.append(tree_bytes(dst))
        return trees

    def test_byte_identical_for_every_order(self):
        """§10.2 merged trees are byte-identical whatever the listing order (generation shards)."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, COUNTS)
            trees = self._merged_trees(roots, tmp)
            self.assertEqual(trees[1], trees[0])
            self.assertEqual(trees[2], trees[0])

    def test_byte_identical_without_plans(self):
        """§10.2 shards without generation_plan.json are merged in a deterministic (content) order."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, COUNTS, with_plan=False)
            trees = self._merged_trees(roots, tmp)
            self.assertEqual(trees[1], trees[0])
            self.assertEqual(trees[2], trees[0])

    def test_api_matches_script(self):
        """§10.2 plan_merge + merge_frames + apply_merge_meta give the script's sidecars and frames."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, COUNTS)
            code, _, err = run_script(*roots, "--dst_root", Path(tmp) / "script", "--repo_id", "local/fx")
            self.assertEqual(code, 0, err)
            merge_api(list(reversed(roots)), Path(tmp) / "api")
            script = tree_bytes(Path(tmp) / "script")
            script.pop("task_meta/episodes.parquet")
            self.assertEqual(tree_bytes(Path(tmp) / "api"), script)


def _collide_episode_fp(sidecars):
    sidecars[0]["fingerprints"]["episode_fp"] = make_sidecar(tag="fx_pick-0-1")["fingerprints"]["episode_fp"]


def _collide_scenario(sidecars):
    sidecars[1]["fingerprints"]["scenario_key"] = make_sidecar(tag="fx_pick-0-0")["fingerprints"]["scenario_key"]


class RefusalTests(unittest.TestCase):
    """WP12 §10.2/§1.5 refusals: the contract error is raised and nothing is written to the destination."""

    def _refused(self, error, counts=COUNTS, mutate=None, **make_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, counts, **make_kwargs)
            if mutate is not None:
                mutate(roots)
            before = [tree_bytes(r) for r in roots]
            dst = Path(tmp) / "merged"
            code, out, err = run_script(*roots, "--dst_root", dst, "--repo_id", "local/fx")
            self.assertEqual(code, 1, out + err)
            self.assertIn(error.__name__, err)
            self.assertFalse(dst.exists(), "nothing may be written to the destination")
            self.assertEqual([tree_bytes(r) for r in roots], before, "shards must stay untouched")
            with self.assertRaises(error) as cm:
                plan_merge([ShardSummary.from_root(r) for r in roots])
            self.assertIs(type(cm.exception), error)

    def test_episode_fp_collision(self):
        """§10.2 the same episode_fp in two shards → DuplicateError."""
        self._refused(DuplicateError, per_shard={2: {"sidecar_hook": _collide_episode_fp}})

    def test_scenario_key_collision(self):
        """§10.2 the same original scenario_key in two shards → DuplicateError."""
        self._refused(DuplicateError, per_shard={1: {"sidecar_hook": _collide_scenario}})

    def test_manifest_line_collision(self):
        """§8.3 the same (task, kind, attempt_index) in two shards' manifests → DuplicateError."""
        line = other_line("fx_pick", "failure", 999, (1, 3), "rejected", "dup")
        self._refused(DuplicateError, per_shard={0: {"extra_lines": [line]}, 1: {"extra_lines": [line]}})

    def test_config_sha_mismatch(self):
        """§8.5 shards of different runs (config_sha256) → ConfigError."""
        self._refused(ConfigError, per_shard={1: {"plan_kwargs": {"config_sha": "e" * 64}}})

    def test_task_toml_mismatch(self):
        """§8.5 shards with different task_toml_sha256 → ConfigError."""
        self._refused(ConfigError, per_shard={2: {"plan_kwargs": {"task_toml_sha": "f" * 64}}})

    def test_plan_in_some_shards_only(self):
        """§8.5 generation_plan.json in some shards but not all → ConfigError."""
        self._refused(ConfigError, per_shard={1: {"with_plan": False}})

    def test_fps_mismatch(self):
        """§11 different fps in meta/info.json → ConfigError."""
        self._refused(ConfigError, per_shard={1: {"fps": 15}})

    def test_features_mismatch(self):
        """§11 different features in meta/info.json → ConfigError."""
        self._refused(ConfigError, per_shard={0: {"feats": features(cameras=("front",))}})

    def test_codebase_version_mismatch(self):
        """§11 a lite shard and a LeRobot v3 shard → ConfigError."""
        self._refused(ConfigError, per_shard={2: {"layout": "v3"}})

    def test_no_audit(self):
        """§10.2 a shard without meta/audit_report.json → AuditFailed."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, COUNTS, per_shard={1: {"audit": False}})
            dst = Path(tmp) / "merged"
            code, out, err = run_script(*roots, "--dst_root", dst, "--repo_id", "local/fx")
            self.assertEqual(code, 1, out + err)
            self.assertIn("AuditFailed", err)
            self.assertFalse(dst.exists())

    def test_failed_audit(self):
        """§10.2 a shard whose audit did not pass → AuditFailed."""
        self._refused(AuditFailed, mutate=lambda roots: write_audit(roots[1], passed=False))

    def test_stale_audit(self):
        """§10.2 a shard changed after its audit (data, sidecar folder, manifest) → AuditFailed."""
        edits = {
            "data": lambda r: (r / "data" / "chunk-000" / "file-000000.parquet").write_bytes(b"not parquet"),
            "new file": lambda r: (r / "task_meta" / "notes.txt").write_text("x"),
            "manifest": lambda r: (r / MANIFEST_REL).write_text((r / MANIFEST_REL).read_text()[:-10]),
        }
        for name, edit in edits.items():
            with self.subTest(edit=name):
                self._refused(AuditFailed, mutate=lambda roots, edit=edit: edit(roots[2]))

    def test_dr_copy_shares_scenario_key(self):
        """§10.1 scenario keys are unique among originals only: a DR copy in another shard is fine."""
        def add_copy(sidecars):
            parent = make_sidecar(tag="fx_pick-0-0")
            copy_ = make_dr_copy(parent, index=len(sidecars), tag="dr")
            copy_["attempt"]["shard"] = [1, 3]
            sidecars.append(copy_)
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, COUNTS, per_shard={1: {"sidecar_hook": add_copy}})
            self.assertEqual(plan_merge([ShardSummary.from_root(r) for r in roots]).total, sum(COUNTS) + 1)


class DestinationTests(unittest.TestCase):
    """WP12 §10.2 the merge never overwrites an existing destination."""

    def test_script_refuses_existing_destination(self):
        """§10.2 merge_shards refuses an existing --dst_root (even an empty folder) and changes nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (1, 1))
            dst = Path(tmp) / "merged"
            dst.mkdir()
            code, _, err = run_script(*roots, "--dst_root", dst, "--repo_id", "local/fx")
            self.assertEqual(code, 1)
            self.assertEqual(list(dst.iterdir()), [])

    def test_api_refuses_existing_output(self):
        """§10.2 apply_merge_meta refuses existing sidecars; merge_frames refuses an existing info.json."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (1, 2))
            dst = Path(tmp) / "merged"
            shards, plan = merge_api(roots, dst)
            before = tree_bytes(dst)
            with self.assertRaises(GenerationError):
                apply_merge_meta(plan, shards, dst)
            with self.assertRaises(GenerationError):
                merge_frames(plan, shards, dst, "local/fx")
            self.assertEqual(tree_bytes(dst), before)

    def test_arguments(self):
        """§10.2 merge_shards needs at least one shard, --dst_root and --repo_id (argparse exit 2)."""
        cases = [[], ["a", "--dst_root", "d"], ["a", "--repo_id", "r"], ["--dst_root", "d", "--repo_id", "r"]]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                merge_shards.main(argv)
            self.assertEqual(cm.exception.code, 2)

    def test_lerobot_shards_need_lerobot(self):
        """§1.6 non-lite shards are merged by LeRobot (container only): without it ImportError, nothing written."""
        with tempfile.TemporaryDirectory() as tmp:
            roots, _, _ = make_shards(tmp, (1, 2), layout="v3")
            shards = [ShardSummary.from_root(r) for r in roots]
            plan = plan_merge(shards)
            dst = Path(tmp) / "merged"
            blocked = {name: None for name in ("lerobot", "lerobot.datasets", "lerobot.datasets.aggregate")}
            with mock.patch.dict(sys.modules, blocked), self.assertRaises(ImportError):
                merge_frames(plan, shards, dst, "local/fx")
            self.assertFalse(dst.exists() and any(dst.rglob("*")))


if __name__ == "__main__":
    unittest.main()
