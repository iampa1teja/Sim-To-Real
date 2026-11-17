"""WP5: ``core.generate.manifest`` — manifest lines, append-only file, reconcile, generation_plan.json
(contracts §8.3, §8.4, §8.5)."""

import dataclasses
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from core.generate.errors import ResumeError, SchemaError
from core.generate.manifest import (Manifest, ManifestLine, ReconcileResult, read_plan_json, reconcile,
                                    write_plan_json)
from core.generate.protocols import AttemptStatus, Kind, RejectionReason, SkipReason
from core.utils.hashing import canonical_json
from v4_support import factories as fx
from v4_support.stubs import ListWriter

MANIFEST_REL = Path("meta") / "generation_manifest.jsonl"
LINE_KEYS = ["v", "task", "kind", "attempt_index", "seed", "shard", "status", "reason", "episode_index", "scenario_key",
             "spec_fp", "traj_fp", "appearance_fp", "episode_fp", "success", "failure_mode", "furthest_stage",
             "perturbation", "num_frames", "wall_s"]
KIND_ORDER = (Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY)


def line_json(index=0, kind="success", status="accepted", *, episode_index=None, reason=..., tag=None, **over):
    """A valid manifest line dict (§8.3) for attempt ``index``."""
    tag = f"{kind}{index}" if tag is None else tag
    accepted = status == "accepted"
    skipped = status == "skipped"
    if reason is ...:
        reason = None if accepted else ("scenario_seen" if skipped else "expert_failed")
    sampling_failed = reason == "sampling_failed"
    d = {"v": 1, "task": "fx_pick", "kind": kind, "attempt_index": index, "seed": 2**64 - 1 - index, "shard": [0, 1],
         "status": status, "reason": reason, "episode_index": episode_index if accepted else None,
         "scenario_key": None if sampling_failed else fx.hex_digest(f"s{tag}"),
         "spec_fp": None if sampling_failed else fx.hex_digest(f"p{tag}"),
         "traj_fp": None if skipped else fx.hex_digest(f"t{tag}"),
         "appearance_fp": None if skipped else fx.hex_digest(f"a{tag}"),
         "episode_fp": None if skipped else fx.hex_digest(f"e{tag}"),
         "success": None if skipped else kind == "success",
         "failure_mode": None if (skipped or kind == "success") else "missed_grasp",
         "furthest_stage": None if skipped else "reached",
         "perturbation": None if (skipped or kind == "success") else "grasp_offset",
         "num_frames": None if skipped else 30 + index, "wall_s": 1.25}
    if accepted and episode_index is None:
        d["episode_index"] = 0
    d.update(over)
    return d


def line(*args, **kwargs):
    return ManifestLine.from_json(line_json(*args, **kwargs))


def sidecar_payload(episode_index, *, kind="failure", index=7, seed=99, shard=(1, 4), tag="sc"):
    """A §9.1 v2 sidecar (all keys) for an accepted generated episode."""
    n = 3
    pos = [[0.2, 0.05, 0.01]] * n
    quat = [[1.0, 0.0, 0.0, 0.0]] * n
    pert = None if kind == "success" else {**fx.make_perturbation().to_json(), "onset_frame": 1, "applied": True}
    return {
        "schema_version": 2, "episode_index": episode_index, "num_frames": n, "fps": 30, "task": "fx_pick",
        "arm": "fx_arm5", "scene": "fx_scene", "reference_frame": "robot base link (base)", "units": "m",
        "angle_units": "rad", "joint_names": ["j1", "j2", "j3", "j4", "j5", "jaw"],
        "objects": {"cube_0": {"set": "cubes", "size_m": [0.02, 0.02, 0.02], "color": "red", "active": True,
                               "role": "target", "position": pos, "orientation_wxyz": quat}},
        "ee_pose": {"position": pos, "orientation_wxyz": quat}, "instruction": "Pick up the red cube",
        "roles": {"target": "cube_0"}, "kind": kind, "success": kind == "success",
        "failure_mode": None if kind == "success" else "missed_grasp", "furthest_stage": "reached",
        "stages": ["reached", "success"], "stage": [-1, 0, 0], "imitate": [True, True, True], "phases": [],
        "segments": [{"label": "nominal", "start": 0, "end": n, "phase": None}], "perturbation": pert,
        "recovery": None, "noise": None,
        "attempt": {"index": index, "seed": seed, "shard": list(shard), "spec_fp": fx.hex_digest(f"p{tag}")},
        "fingerprints": {"scenario_key": fx.hex_digest(f"s{tag}"), "spec_fp": fx.hex_digest(f"p{tag}"),
                         "traj_fp": fx.hex_digest(f"t{tag}"), "appearance_fp": fx.hex_digest(f"a{tag}"),
                         "episode_fp": fx.hex_digest(f"e{tag}")},
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "scripted_expert", "git_commit": "abc"}, "legacy_v1": False,
    }


def write_sidecar(root, index, payload=None):
    path = Path(root) / "task_meta" / f"episode_{index:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload if payload is not None else sidecar_payload(index)))
    return path


def sidecar_indices(root):
    folder = Path(root) / "task_meta"
    return sorted(int(p.stem.split("_")[1]) for p in folder.glob("episode_*.json")) if folder.is_dir() else []


def writer_with(count):
    w = ListWriter()
    w.episodes = [[{"frame": 0}] for _ in range(count)]
    return w


class TempRoot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="wp5_")
        self.root = Path(self._tmp.name)
        self.path = self.root / MANIFEST_REL
        self.manifest = Manifest(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, records):
        """Append ``records`` = [(index, kind, status, **kw)] in order, numbering accepted episodes 0, 1, ..."""
        accepted = 0
        for index, kind, status, *rest in records:
            kw = rest[0] if rest else {}
            if status == "accepted":
                kw = {"episode_index": accepted, **kw}
                accepted += 1
            self.manifest.append(line(index, kind, status, **kw))
        return accepted


class ManifestLineTests(unittest.TestCase):
    """WP5 §8.3 ManifestLine."""

    def test_json_keys_and_round_trip(self):
        """§8.3 to_json has exactly the table keys with v = 1; from_json(to_json()) == line; plain builtins."""
        for status, reason in (("accepted", None), ("rejected", "duplicate_trajectory"), ("skipped", "scenario_seen"),
                               ("skipped", "sampling_failed")):
            d = line_json(3, "recovery", status, reason=reason)
            ml = ManifestLine.from_json(d)
            out = ml.to_json()
            with self.subTest(status=status, reason=reason):
                self.assertEqual(sorted(out), sorted(LINE_KEYS))
                self.assertEqual(out["v"], 1)
                self.assertEqual(json.loads(canonical_json(out)), d)
                self.assertEqual(ManifestLine.from_json(json.loads(canonical_json(out))), ml)
                self.assertIs(ml.kind, Kind.RECOVERY)
                self.assertIs(ml.status, AttemptStatus(status))
                self.assertIsInstance(out["shard"], list)
                self.assertIs(type(out["kind"]), str)
                self.assertIs(type(out["status"]), str)
        self.assertIs(line(0, "failure", "rejected", reason="env_error").reason, RejectionReason.ENV_ERROR)
        self.assertIs(line(0, "failure", "skipped", reason="sampling_failed").reason, SkipReason.SAMPLING_FAILED)

    def test_frozen(self):
        """§8.3 ManifestLine is a frozen dataclass."""
        ml = line()
        self.assertTrue(dataclasses.is_dataclass(ml))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ml.seed = 1

    def test_seed_range_is_exact(self):
        """§1.3 seeds are JSON integers up to 2**64 - 1 and survive the line round-trip exactly."""
        for seed in (0, 2**63, 2**64 - 1):
            ml = line(seed=seed)
            self.assertEqual(json.loads(canonical_json(ml.to_json()))["seed"], seed)

    def test_rejects_invalid(self):
        """§8.3 from_json raises SchemaError for missing/unknown keys, wrong v, bad types and inconsistent
        status/reason/episode_index/fingerprint combinations."""
        good = line_json()
        bad = {
            "missing_key": {k: v for k, v in good.items() if k != "wall_s"},
            "unknown_key": {**good, "extra": 1},
            "v2": {**good, "v": 2},
            "kind": {**good, "kind": "nominal"},
            "status": {**good, "status": "done"},
            "seed_str": {**good, "seed": "5"},
            "seed_float": {**good, "seed": 5.0},
            "seed_neg": {**good, "seed": -1},
            "seed_big": {**good, "seed": 2**64},
            "shard_order": {**good, "shard": [1, 1]},
            "shard_len": {**good, "shard": [0]},
            "index_neg": {**good, "attempt_index": -1},
            "hex_upper": {**good, "traj_fp": good["traj_fp"].upper()},
            "hex_short": {**good, "scenario_key": "abc"},
            "accepted_reason": {**good, "reason": "expert_failed"},
            "accepted_no_episode": {**good, "episode_index": None},
            "accepted_no_traj": {**good, "traj_fp": None},
            "rejected_skip_reason": line_json(0, "failure", "rejected", reason="scenario_seen"),
            "rejected_no_reason": line_json(0, "failure", "rejected", reason=None),
            "rejected_episode": {**line_json(0, "failure", "rejected"), "episode_index": 0},
            "skipped_rejection_reason": line_json(0, "failure", "skipped", reason="expert_failed"),
            "skipped_traj": {**line_json(0, "failure", "skipped"), "traj_fp": fx.hex_digest(1)},
            "sampling_failed_key": {**line_json(0, "failure", "skipped", reason="sampling_failed"),
                                    "scenario_key": fx.hex_digest(1)},
            "scenario_seen_no_key": {**line_json(0, "failure", "skipped"), "scenario_key": None},
            "wall_nan": {**good, "wall_s": float("nan")},
            "success_str": {**good, "success": "yes"},
            "not_a_dict": [1, 2],
        }
        for name, payload in bad.items():
            with self.subTest(case=name), self.assertRaises(SchemaError):
                ManifestLine.from_json(payload)

    def test_from_result(self):
        """§8.3 ManifestLine.from_result fills the line from an EpisodeResult (WP7 uses it)."""
        res = fx.make_episode_result("failure", "accepted", num_frames=12, attempt_index=4, seed=77, episode_index=2)
        ml = ManifestLine.from_result(res, shard=(1, 3), perturbation="grasp_offset", wall_s=2.5)
        self.assertEqual((ml.task, ml.kind, ml.attempt_index, ml.seed, ml.shard, ml.status, ml.reason, ml.episode_index),
                         ("fx_pick", Kind.FAILURE, 4, 77, (1, 3), AttemptStatus.ACCEPTED, None, 2))
        fp = res.fingerprints
        self.assertEqual((ml.scenario_key, ml.spec_fp, ml.traj_fp, ml.appearance_fp, ml.episode_fp),
                         (fp.scenario_key, fp.spec_fp, fp.traj_fp, fp.appearance_fp, fp.episode_fp))
        self.assertEqual((ml.success, ml.failure_mode, ml.furthest_stage, ml.perturbation, ml.num_frames, ml.wall_s),
                         (res.success, res.failure_mode, res.furthest_stage, "grasp_offset", 12, 2.5))
        skipped = ManifestLine.from_result(fx.make_episode_result("success", "skipped"), shard=(0, 1))
        self.assertEqual((skipped.status, skipped.reason, skipped.num_frames, skipped.traj_fp, skipped.perturbation),
                         (AttemptStatus.SKIPPED, SkipReason.SCENARIO_SEEN, None, None, None))


class ManifestFileTests(TempRoot):
    """WP5 §8.3 Manifest: append-only JSONL, torn lines."""

    def test_missing_file_is_empty(self):
        """§8.3 a manifest that does not exist yet has no lines."""
        self.assertEqual(self.manifest.lines(), [])
        self.assertEqual(self.manifest.accepted(), [])
        self.assertIsNone(self.manifest.last_index(Kind.SUCCESS))

    def test_one_canonical_line_per_append(self):
        """§8.3 each append writes canonical_json(line) + '\\n' (creating meta/); nothing else."""
        records = [line(0, "success", "accepted", episode_index=0), line(0, "failure", "rejected"),
                   line(0, "recovery", "skipped", reason="sampling_failed"), line(1, "success", "skipped")]
        for i, ml in enumerate(records):
            self.manifest.append(ml)
            text = self.path.read_text(encoding="utf-8")
            with self.subTest(i=i):
                self.assertEqual(text, "".join(canonical_json(r.to_json()) + "\n" for r in records[:i + 1]))
        self.assertEqual(self.manifest.lines(), records)
        self.assertEqual(Manifest(self.path).lines(), records)

    def test_never_rewrites_earlier_bytes(self):
        """§8.3/§8.4 appends only extend the file: every earlier byte prefix is preserved."""
        prefixes = []
        for i in range(12):
            status = "accepted" if i % 3 == 0 else "rejected"
            self.manifest.append(line(i, "success", status, episode_index=i // 3))
            data = self.path.read_bytes()
            for p in prefixes:
                self.assertTrue(data.startswith(p))
            prefixes.append(data)

    def test_accepted_and_last_index(self):
        """§8.3 accepted() returns accepted lines in file order; last_index(kind) is the highest attempt index of
        that kind with any line (None when there is none)."""
        self.build([(0, "success", "accepted"), (0, "failure", "rejected"), (1, "success", "skipped"),
                    (1, "failure", "accepted"), (3, "failure", "skipped", {"reason": "sampling_failed"}),
                    (4, "success", "accepted")])
        self.assertEqual([(l.attempt_index, l.kind) for l in self.manifest.accepted()],
                         [(0, Kind.SUCCESS), (1, Kind.FAILURE), (4, Kind.SUCCESS)])
        self.assertEqual([l.episode_index for l in self.manifest.accepted()], [0, 1, 2])
        self.assertEqual(self.manifest.last_index(Kind.SUCCESS), 4)
        self.assertEqual(self.manifest.last_index("failure"), 3)
        self.assertIsNone(self.manifest.last_index(Kind.RECOVERY))

    def test_torn_final_line_dropped_and_truncated(self):
        """§8.3 a final line without '\\n' (any prefix of a real line, including complete JSON) is dropped on
        read and truncated before the next append; earlier lines are untouched."""
        self.build([(0, "success", "accepted"), (0, "failure", "rejected")])
        good = self.path.read_bytes()
        torn_full = (canonical_json(line(1, "success", "accepted", episode_index=1).to_json())).encode()
        for cut in (1, 10, len(torn_full) // 2, len(torn_full) - 1, len(torn_full)):
            self.path.write_bytes(good + torn_full[:cut])
            with self.subTest(cut=cut):
                reopened = Manifest(self.path)
                self.assertEqual(len(reopened.lines()), 2)
                nxt = line(1, "failure", "rejected")
                reopened.append(nxt)
                self.assertEqual(self.path.read_bytes(), good + (canonical_json(nxt.to_json()) + "\n").encode())
                self.assertEqual(Manifest(self.path).lines()[-1], nxt)

    def test_final_invalid_json_line_is_torn(self):
        """§8.3 a final newline-terminated line that is not valid JSON is torn: dropped and truncated."""
        self.build([(0, "success", "accepted")])
        good = self.path.read_bytes()
        self.path.write_bytes(good + b'{"v": 1, "task": \n')
        self.assertEqual(len(self.manifest.lines()), 1)
        nxt = line(0, "failure", "rejected")
        self.manifest.append(nxt)
        self.assertEqual(self.path.read_bytes(), good + (canonical_json(nxt.to_json()) + "\n").encode())

    def test_invalid_non_final_line_raises(self):
        """§8.3 an invalid non-final line raises SchemaError (on read and on append); the file is not modified."""
        self.build([(0, "success", "accepted")])
        good = self.path.read_bytes()
        tail = (canonical_json(line(0, "failure", "rejected").to_json()) + "\n").encode()
        for name, junk in (("not_json", b"garbage\n"), ("empty", b"\n"),
                           ("schema", (canonical_json({**line_json(0, "recovery", "rejected"), "v": 9}) + "\n").encode())):
            self.path.write_bytes(good + junk + tail)
            with self.subTest(case=name):
                with self.assertRaises(SchemaError):
                    Manifest(self.path).lines()
                with self.assertRaises(SchemaError):
                    Manifest(self.path).append(line(1, "success", "rejected"))
                self.assertEqual(self.path.read_bytes(), good + junk + tail)

    def test_duplicate_attempt_rejected(self):
        """§8.4 'a line, once written, is never executed again': a second line for the same (task, kind, index)
        raises SchemaError on append (file unchanged) and on read."""
        self.build([(0, "success", "accepted"), (0, "failure", "rejected")])
        before = self.path.read_bytes()
        with self.assertRaises(SchemaError):
            self.manifest.append(line(0, "failure", "skipped"))
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_bytes(before + (canonical_json(line_json(0, "failure", "rejected", tag="z")) + "\n").encode())
        with self.assertRaises(SchemaError):
            self.manifest.lines()

    def test_fsync_after_each_line(self):
        """§8.3 each append is followed by flush and os.fsync."""
        calls = []
        real = os.fsync

        def spy(fd):
            calls.append(fd)
            return real(fd)
        os.fsync = spy
        try:
            self.manifest.append(line())
            self.manifest.append(line(0, "failure", "rejected"))
        finally:
            os.fsync = real
        self.assertGreaterEqual(len(calls), 2)


class ReconcileTests(TempRoot):
    """WP5 §8.4 reconcile: E = writer.episode_count(), M = accepted lines, S = sidecar indices."""

    def assert_untouched(self, before_manifest, before_sidecars):
        self.assertEqual(self.path.read_bytes() if self.path.exists() else None, before_manifest)
        self.assertEqual(sidecar_indices(self.root), before_sidecars)

    def snapshot(self):
        return (self.path.read_bytes() if self.path.exists() else None), sidecar_indices(self.root)

    def test_result_type(self):
        """§8.4 ReconcileResult(rebuilt_lines, deleted_sidecars, next_position)."""
        self.assertEqual([f.name for f in dataclasses.fields(ReconcileResult)],
                         ["rebuilt_lines", "deleted_sidecars", "next_position"])

    def test_fresh_dataset(self):
        """§8.4 no manifest, no episodes, no sidecars: nothing to do; start at (0, 0)."""
        result = reconcile(self.root, writer_with(0), self.manifest)
        self.assertEqual(result, ReconcileResult(rebuilt_lines=0, deleted_sidecars=[], next_position=(0, 0)))

    def test_consistent(self):
        """§8.4 E = M with sidecars 0..E-1: no change; next position follows the last line in plan order."""
        m = self.build([(0, "success", "accepted"), (0, "failure", "rejected"), (0, "recovery", "accepted"),
                        (1, "success", "skipped"), (1, "failure", "accepted")])
        for i in range(m):
            write_sidecar(self.root, i)
        before = self.snapshot()
        result = reconcile(self.root, writer_with(m), self.manifest)
        self.assertEqual(result, ReconcileResult(0, [], (1, 2)))
        self.assert_untouched(*before)

    def test_next_position_wraps(self):
        """§8.4 rule 6: the position after (i, recovery) is (i + 1, success); positions are (index, KINDS.index)."""
        self.build([(0, "success", "accepted"), (2, "recovery", "rejected")])
        write_sidecar(self.root, 0)
        self.assertEqual(reconcile(self.root, writer_with(1), self.manifest).next_position, (3, 0))

    def test_next_position_sharded(self):
        """§8.4 rule 6 for a shard 1/3 whose lines sit at indices 1 and 4 only."""
        self.build([(1, "success", "accepted", {"shard": [1, 3]}), (4, "success", "rejected", {"shard": [1, 3]}),
                    (4, "failure", "rejected", {"shard": [1, 3]})])
        write_sidecar(self.root, 0)
        self.assertEqual(reconcile(self.root, writer_with(1), self.manifest).next_position, (4, 2))

    def test_dataset_ahead_by_one_rebuilds_line(self):
        """§8.4 rule 3: E = M + 1 (crash after save_episode, before the line) rebuilds the line from sidecar E-1
        with wall_s = 0.0, appended at the end; all other bytes are kept."""
        m = self.build([(0, "success", "accepted"), (0, "failure", "rejected")])
        write_sidecar(self.root, 0)
        payload = sidecar_payload(1, kind="failure", index=1, seed=2**64 - 5, shard=(0, 1), tag="rebuilt")
        write_sidecar(self.root, 1, payload)
        before = self.path.read_bytes()
        result = reconcile(self.root, writer_with(m + 1), self.manifest)
        self.assertEqual(result, ReconcileResult(1, [], (1, 2)))
        self.assertTrue(self.path.read_bytes().startswith(before))
        rebuilt = self.manifest.lines()[-1]
        expected = ManifestLine.from_json({
            "v": 1, "task": "fx_pick", "kind": "failure", "attempt_index": 1, "seed": 2**64 - 5, "shard": [0, 1],
            "status": "accepted", "reason": None, "episode_index": 1, "scenario_key": fx.hex_digest("srebuilt"),
            "spec_fp": fx.hex_digest("prebuilt"), "traj_fp": fx.hex_digest("trebuilt"),
            "appearance_fp": fx.hex_digest("arebuilt"), "episode_fp": fx.hex_digest("erebuilt"), "success": False,
            "failure_mode": "missed_grasp", "furthest_stage": "reached", "perturbation": "grasp_offset",
            "num_frames": 3, "wall_s": 0.0})
        self.assertEqual(rebuilt, expected)
        self.assertEqual(len(self.manifest.accepted()), m + 1)
        self.assertEqual(sidecar_indices(self.root), [0, 1])
        again = reconcile(self.root, writer_with(m + 1), self.manifest)
        self.assertEqual(again, ReconcileResult(0, [], (1, 2)))

    def test_torn_accepted_line_is_rebuilt(self):
        """§8.3 + §8.4 a torn final accepted line is dropped, so E = M + 1 and the line is rebuilt from its sidecar."""
        self.build([(0, "success", "accepted")])
        write_sidecar(self.root, 0)
        write_sidecar(self.root, 1, sidecar_payload(1, kind="success", index=1, shard=(0, 1), tag="torn"))
        full = canonical_json(line_json(1, "success", "accepted", episode_index=1, tag="torn")).encode()
        self.path.write_bytes(self.path.read_bytes() + full[:-7])
        result = reconcile(self.root, writer_with(2), self.manifest)
        self.assertEqual(result.rebuilt_lines, 1)
        lines = Manifest(self.path).lines()
        self.assertEqual([l.episode_index for l in lines], [0, 1])
        self.assertEqual(lines[-1].traj_fp, fx.hex_digest("ttorn"))
        self.assertEqual(lines[-1].wall_s, 0.0)

    def test_dataset_ahead_without_sidecar(self):
        """§8.4 rule 3: E = M + 1 but sidecar E-1 is missing → ResumeError, nothing modified."""
        m = self.build([(0, "success", "accepted")])
        write_sidecar(self.root, 0)
        before = self.snapshot()
        with self.assertRaises(ResumeError):
            reconcile(self.root, writer_with(m + 1), self.manifest)
        self.assert_untouched(*before)

    def test_dataset_ahead_by_two(self):
        """§8.4 rule 2: E > M + 1 → ResumeError, nothing modified."""
        m = self.build([(0, "success", "accepted")])
        for i in range(3):
            write_sidecar(self.root, i)
        before = self.snapshot()
        with self.assertRaises(ResumeError):
            reconcile(self.root, writer_with(m + 2), self.manifest)
        self.assert_untouched(*before)

    def test_manifest_ahead_of_dataset(self):
        """§8.4 rule 2: E < M (the dataset lost a committed episode) → ResumeError, nothing modified."""
        m = self.build([(0, "success", "accepted"), (1, "success", "accepted")])
        for i in range(m):
            write_sidecar(self.root, i)
        before = self.snapshot()
        for e in (0, m - 1):
            with self.subTest(E=e), self.assertRaises(ResumeError):
                reconcile(self.root, writer_with(e), self.manifest)
            self.assert_untouched(*before)

    def test_orphan_sidecars_deleted(self):
        """§8.4 rule 4: sidecars with index >= E are deleted (crash after write_sidecar, before save_episode)."""
        m = self.build([(0, "success", "accepted"), (0, "failure", "rejected")])
        for i in (0, 1, 2, 5):
            write_sidecar(self.root, i)
        before = self.path.read_bytes()
        result = reconcile(self.root, writer_with(m), self.manifest)
        self.assertEqual(result, ReconcileResult(0, [1, 2, 5], (0, 2)))
        self.assertEqual(sidecar_indices(self.root), [0])
        self.assertEqual(self.path.read_bytes(), before)

    def test_rebuild_and_orphans_together(self):
        """§8.4 rules 3 + 4: E = M + 1 rebuilds line E-1 and deletes sidecars >= E."""
        m = self.build([(0, "success", "accepted")])
        write_sidecar(self.root, 0)
        write_sidecar(self.root, 1, sidecar_payload(1, kind="success", index=2, shard=(0, 1), tag="r"))
        write_sidecar(self.root, 2)
        result = reconcile(self.root, writer_with(m + 1), self.manifest)
        self.assertEqual(result, ReconcileResult(1, [2], (2, 1)))
        self.assertEqual(sidecar_indices(self.root), [0, 1])

    def test_missing_sidecar_below_e(self):
        """§8.4 rule 5: a missing sidecar with index < E raises ResumeError; nothing is deleted or appended."""
        m = self.build([(0, "success", "accepted"), (1, "success", "accepted"), (2, "success", "accepted")])
        write_sidecar(self.root, 0)
        write_sidecar(self.root, 2)
        write_sidecar(self.root, 7)
        before = self.snapshot()
        with self.assertRaises(ResumeError):
            reconcile(self.root, writer_with(m), self.manifest)
        self.assert_untouched(*before)

    def test_episode_indices_must_be_0_to_m_minus_1(self):
        """§8.4 rule 1: accepted episode_index values that are not exactly 0..M-1 raise ResumeError."""
        for name, indices in (("gap", [0, 2]), ("duplicate", [0, 0]), ("start_at_1", [1, 2])):
            with self.subTest(case=name):
                self.path.unlink(missing_ok=True)
                for i, e in enumerate(indices):
                    self.manifest.append(line(i, "success", "accepted", episode_index=e))
                for i in range(3):
                    write_sidecar(self.root, i)
                before = self.snapshot()
                with self.assertRaises(ResumeError):
                    reconcile(self.root, writer_with(2), self.manifest)
                self.assert_untouched(*before)

    def test_rejected_and_skipped_do_not_count(self):
        """§8.4 M counts accepted lines only."""
        m = self.build([(0, "success", "rejected"), (0, "failure", "skipped"), (0, "recovery", "accepted"),
                        (1, "success", "skipped", {"reason": "sampling_failed"})])
        self.assertEqual(m, 1)
        write_sidecar(self.root, 0)
        self.assertEqual(reconcile(self.root, writer_with(1), self.manifest), ReconcileResult(0, [], (1, 1)))

    def test_bad_sidecar_for_rebuild(self):
        """§8.4 rule 3: a sidecar E-1 that cannot give a line (legacy, no attempt/fingerprints, wrong
        episode_index) raises ResumeError and modifies nothing."""
        bad = {"no_attempt": {**sidecar_payload(0), "attempt": None},
               "no_fingerprints": {**sidecar_payload(0), "fingerprints": None},
               "wrong_index": sidecar_payload(5),
               "not_json": None}
        for name, payload in bad.items():
            with self.subTest(case=name):
                path = write_sidecar(self.root, 0, payload if payload is not None else {})
                if payload is None:
                    path.write_text("{not json")
                before = self.snapshot()
                with self.assertRaises(ResumeError):
                    reconcile(self.root, writer_with(1), self.manifest)
                self.assert_untouched(*before)

    def test_rebuilt_line_must_not_duplicate_an_attempt(self):
        """§8.4 'a line, once written, is never executed again': a sidecar whose attempt already has a line
        cannot be rebuilt → ResumeError."""
        self.build([(0, "success", "accepted"), (1, "failure", "rejected")])
        write_sidecar(self.root, 0)
        write_sidecar(self.root, 1, sidecar_payload(1, kind="failure", index=1, shard=(0, 1)))
        before = self.snapshot()
        with self.assertRaises(ResumeError):
            reconcile(self.root, writer_with(2), self.manifest)
        self.assert_untouched(*before)

    def test_random_kill_points(self):
        """§8.4 property: simulate the commit order (sidecar → save → line) and stop at random points; reconcile
        always restores M == E, sidecars == 0..E-1, and the next position is after every committed line."""
        for k in range(30):
            rng = np.random.default_rng(k)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = Manifest(root / MANIFEST_REL)
                writer = writer_with(0)
                positions = [(i, kind) for i in range(int(rng.integers(1, 6))) for kind in KIND_ORDER]
                kill = int(rng.integers(0, 4 * len(positions)))
                step = 0
                for index, kind in positions:
                    outcome = rng.choice(["accepted", "rejected", "skipped"])
                    if outcome == "accepted":
                        e = writer.episode_count()
                        stages = ("sidecar", "save", "line")
                        for st in stages:
                            if step == kill:
                                break
                            step += 1
                            if st == "sidecar":
                                write_sidecar(root, e, sidecar_payload(e, kind=kind.value, index=index, shard=(0, 1),
                                                                       tag=f"{k}-{index}-{kind.value}"))
                            elif st == "save":
                                writer.episodes.append([{"frame": 0}])
                            else:
                                manifest.append(line(index, kind.value, "accepted", episode_index=e,
                                                     tag=f"{k}-{index}-{kind.value}"))
                        else:
                            continue
                        break
                    if step == kill:
                        break
                    step += 1
                    manifest.append(line(index, kind.value, str(outcome)))
                with self.subTest(k=k, kill=kill):
                    result = reconcile(root, writer, manifest)
                    accepted = manifest.accepted()
                    self.assertEqual(len(accepted), writer.episode_count())
                    self.assertEqual(sidecar_indices(root), list(range(writer.episode_count())))
                    lines = manifest.lines()
                    if lines:
                        last = max((l.attempt_index, KIND_ORDER.index(l.kind)) for l in lines)
                        self.assertGreater(result.next_position, last)
                    self.assertEqual(reconcile(root, writer, manifest).rebuilt_lines, 0)


def plan_payload(**over):
    p = {"v": 1, "config_sha256": fx.hex_digest("cfg"), "config": {"arm": "fx_arm5", "fps": 30}, "master_seed": 0,
         "git_commit": "0123abcd", "shard": [0, 1], "task": "fx_pick", "task_toml_sha256": fx.hex_digest("toml"),
         "arm": "fx_arm5", "arm_files_sha256": {"arm.toml": fx.hex_digest("arm"), "kinematics.json": fx.hex_digest("k")},
         "scene": "fx_scene", "quotas": {"success": 10, "failure": 2, "recovery": 3},
         "holdout": {"colors": ["purple"], "counts": [4]}, "created_at": "2026-01-01T00:00:00+00:00"}
    p.update(over)
    return p


class PlanJsonTests(TempRoot):
    """WP5 §8.5 generation_plan.json."""

    def plan_path(self):
        return self.root / "meta" / "generation_plan.json"

    def test_write_and_read(self):
        """§8.5 write_plan_json writes <root>/meta/generation_plan.json (no *.tmp left); read_plan_json returns it."""
        p = plan_payload()
        self.assertEqual(write_plan_json(self.root, p), p)
        self.assertTrue(self.plan_path().is_file())
        self.assertEqual(json.loads(self.plan_path().read_text()), p)
        self.assertEqual(read_plan_json(self.root), p)
        self.assertEqual(list((self.root / "meta").glob("*.tmp")), [])

    def test_resume_same_plan_keeps_file(self):
        """§8.5 resume with identical guarded keys keeps the original file (created_at unchanged)."""
        write_plan_json(self.root, plan_payload())
        before = self.plan_path().read_bytes()
        out = write_plan_json(self.root, plan_payload(created_at="2027-01-01T00:00:00+00:00", config={"other": 1}))
        self.assertEqual(out["created_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(self.plan_path().read_bytes(), before)

    def test_resume_guarded_mismatch(self):
        """§8.5 a different config_sha256, task_toml_sha256, arm_files_sha256, master_seed or shard raises ResumeError."""
        write_plan_json(self.root, plan_payload())
        before = self.plan_path().read_bytes()
        changes = {"config_sha256": fx.hex_digest("cfg2"), "task_toml_sha256": fx.hex_digest("toml2"),
                   "arm_files_sha256": {"arm.toml": fx.hex_digest("arm"), "kinematics.json": fx.hex_digest("k2")},
                   "master_seed": 1, "shard": [0, 2]}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ResumeError):
                write_plan_json(self.root, plan_payload(**{key: value}))
            self.assertEqual(self.plan_path().read_bytes(), before)

    def test_resume_git_commit_warns_only(self):
        """§8.5 a different git_commit only warns."""
        write_plan_json(self.root, plan_payload())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_plan_json(self.root, plan_payload(git_commit="fedcba98"))
        self.assertTrue(caught)
        self.assertEqual(read_plan_json(self.root)["git_commit"], "0123abcd")

    def test_schema(self):
        """§8.5/§1.5 missing or unknown keys and bad values raise SchemaError (write and read)."""
        bad = {"missing": {k: v for k, v in plan_payload().items() if k != "holdout"},
               "unknown": plan_payload(extra=1), "v": plan_payload(v=2),
               "seed": plan_payload(master_seed=-1), "shard": plan_payload(shard=[2, 2]),
               "quotas": plan_payload(quotas={"success": 1, "failure": 1}),
               "quota_neg": plan_payload(quotas={"success": -1, "failure": 1, "recovery": 1}),
               "sha": plan_payload(config_sha256="xyz")}
        for name, p in bad.items():
            with self.subTest(case=name), self.assertRaises(SchemaError):
                write_plan_json(self.root, p)
            self.assertFalse(self.plan_path().exists())
        self.plan_path().parent.mkdir(parents=True, exist_ok=True)
        self.plan_path().write_text(json.dumps(plan_payload(extra=1)))
        with self.assertRaises(SchemaError):
            read_plan_json(self.root)


if __name__ == "__main__":
    unittest.main()
