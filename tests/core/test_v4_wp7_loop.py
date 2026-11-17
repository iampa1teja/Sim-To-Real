"""WP7: run_generation — plan, quotas, shards, commit order, resume, budgets and the duplicate-trajectory path
(contracts §8.1, §8.3, §8.4, §8.5, §11.1).

Decisions pinned here (docs/v4/prompts/WP7.md):
* The task comes from ``load_task(task_name)``; the dataset root is ``writer.root``. On start the loop writes
  ``generation_plan.json`` (``write_plan_json``: guarded keys raise ``ResumeError`` on resume) and calls
  ``reconcile``.
* It walks ``shard_attempts(iter_plan(task, master_seed, spec_fn, unique_across_kinds=...), i, n)`` from the
  start. Per-kind shard quota = ``shard_quota(total, i, n)``. It stops when every kind's quota is met. A
  position of a kind whose quota is met is passed over without a line. A position that already has a line is
  not executed again; its status updates the counters (accepted, executed, consecutive skips) exactly as when
  it was first run, so a resumed run makes the same decisions as an uninterrupted one.
* A skipped position writes one ``skipped`` line (``scenario_key`` / ``spec_fp`` from the plan, null for
  ``sampling_failed``); ``max_consecutive_skips`` consecutive skips of one kind raise ``ScenarioSpaceExhausted``
  after the line is written.
* Before executing an attempt of kind k: if k has already executed ``ceil(max_attempt_factor × shard quota)``
  non-skipped attempts, ``AttemptBudgetExceeded``.
* An accepted result whose ``traj_fp`` is the ``traj_fp`` of an accepted line of this dataset becomes
  ``rejected / duplicate_trajectory``.
* Accepted: ``write_sidecar`` → ``save_episode`` (must return ``episode_count()`` from before) → manifest
  line. Rejected: ``discard_episode`` → manifest line. ``wall_s`` is the attempt's wall time.
* ``GenerationSummary`` counts the whole manifest after the run (so a resumed run reports the same numbers):
  ``accepted`` per Kind, ``rejected`` per reason value, ``skipped`` lines, ``attempts`` = all lines.
* Each attempt is a pure function of (master seed, task, kind, index): the same attempt gives the same line,
  sidecar and frames whatever the shard count.
"""

import hashlib
import json
import math
import unittest

import numpy as np

from core.datasets.frames import read_frames
from core.generate.config import load_generate_config
from core.generate.errors import AttemptBudgetExceeded, ResumeError, ScenarioSpaceExhausted
from core.generate.manifest import Manifest, read_plan_json
from core.generate.plan import KINDS, attempt_seed, shard_quota
from core.generate.protocols import AttemptStatus, Kind
from test_v4_wp7_support import (ARM, FAST, ConstantExpert, SandboxCase, manifest_dicts, sidecar_dicts, task_table,
                                 tree_bytes, write_variant)
from v4_support import harness as H
from v4_support.fx_arms import fx_home_q

Q3 = {"success": 3, "failure": 3, "recovery": 3}


def plan_order(lines):
    return [(d["attempt_index"], [k.value for k in KINDS].index(d["kind"])) for d in lines]


class LoopBase(SandboxCase):
    TASKS = ("fx7_grip", "fx7_reach")

    def run_cfg(self, cfg, task=FAST, **kw):
        return H.run_shard(cfg, task, rig=self.rig, **kw)


class QuotaTests(LoopBase):
    """WP7 §8.1/§8.3/§8.4 one shard: quotas met exactly, one line per attempt in plan order, commit order."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cfg_path = H.write_config(cls.work / "q.yaml", tasks=[FAST], dst_root=cls.work / "q", quotas=Q3)
        cls.root, cls.summary, _ = H.run_shard(cls.cfg_path, FAST, rig=cls.rig)
        cls.lines = manifest_dicts(cls.root)
        cls.sidecars = sidecar_dicts(cls.root)

    def test_quotas_exact(self):
        kinds = [sc["kind"] for sc in self.sidecars.values()]
        self.assertEqual({k: kinds.count(k) for k in Q3}, Q3)
        self.assertEqual(self.summary.accepted, {Kind(k): v for k, v in Q3.items()})
        self.assertEqual(len(read_frames(self.root)), 9)

    def test_lines_in_plan_order(self):
        order = plan_order(self.lines)
        self.assertEqual(order, sorted(order))
        self.assertEqual(len(set(order)), len(order))
        for d in self.lines:
            with self.subTest(line=(d["kind"], d["attempt_index"])):
                self.assertEqual(d["seed"], attempt_seed(0, FAST, Kind(d["kind"]), d["attempt_index"]))
                self.assertEqual(d["shard"], [0, 1])
                self.assertEqual(d["task"], FAST)

    def test_no_lines_after_quota(self):
        for kind, quota in Q3.items():
            acc = [d["attempt_index"] for d in self.lines if d["kind"] == kind and d["status"] == "accepted"]
            last = acc[quota - 1]
            later = [d for d in self.lines if d["kind"] == kind and d["attempt_index"] > last]
            self.assertEqual(later, [], kind)

    def test_accepted_lines_match_sidecars(self):
        acc = [d for d in self.lines if d["status"] == "accepted"]
        self.assertEqual([d["episode_index"] for d in acc], list(range(len(acc))))
        for d in acc:
            sc = self.sidecars[d["episode_index"]]
            with self.subTest(episode=d["episode_index"]):
                self.assertEqual(sc["kind"], d["kind"])
                self.assertEqual(sc["attempt"], {"index": d["attempt_index"], "seed": d["seed"], "shard": [0, 1],
                                                 "spec_fp": d["spec_fp"]})
                for key in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp"):
                    self.assertEqual(sc["fingerprints"][key], d[key])
                self.assertEqual(sc["generator"], {"name": "scripted_expert", "git_commit": "0" * 40})
                self.assertEqual(sc["num_frames"], d["num_frames"])
                self.assertEqual(sc["joint_names"], list(self.rig.chain.q_names))
                self.assertEqual(sc["reference_frame"], "robot base link (fx5_base)")
                self.assertEqual((sc["task"], sc["arm"], sc["scene"]), (FAST, ARM, "fx_scene"))
                self.assertIsNone(sc["source"])
                self.assertFalse(sc["legacy_v1"])
                self.assertEqual(sc["appearance"], {"light_exposure": None, "robot_color": None})

    def test_rejected_attempts_leave_no_trace(self):
        rejected = [d for d in self.lines if d["status"] == "rejected"]
        self.assertTrue(rejected, "the fixture should reject some recovery attempts")
        stored = {sc["fingerprints"]["traj_fp"] for sc in self.sidecars.values()}
        for d in rejected:
            self.assertIsNone(d["episode_index"])
            if d["traj_fp"] is not None:
                self.assertNotIn(d["traj_fp"], stored)
        files = sorted((self.root / "data").rglob("*.parquet"))
        self.assertEqual(len(files), len(self.sidecars))

    def test_summary_counts_manifest(self):
        rej = {}
        for d in self.lines:
            if d["status"] == "rejected":
                rej[d["reason"]] = rej.get(d["reason"], 0) + 1
        self.assertEqual(dict(self.summary.rejected), rej)
        self.assertEqual(self.summary.skipped, sum(d["status"] == "skipped" for d in self.lines))
        self.assertEqual(self.summary.attempts, len(self.lines))

    def test_plan_json(self):
        plan = read_plan_json(self.root)
        cfg = load_generate_config(self.cfg_path)
        self.assertEqual(plan["config_sha256"], cfg.sha256)
        self.assertEqual(plan["task"], FAST)
        self.assertEqual(plan["shard"], [0, 1])
        self.assertEqual(plan["master_seed"], 0)
        self.assertEqual(plan["quotas"], Q3)
        self.assertEqual(plan["arm"], ARM)
        self.assertEqual(plan["git_commit"], "0" * 40)
        toml = (self.box.tasks / FAST / "task.toml").read_bytes()
        self.assertEqual(plan["task_toml_sha256"], hashlib.sha256(toml).hexdigest())
        arm_dir = self.box.arms / ARM
        self.assertEqual(plan["arm_files_sha256"], {n: hashlib.sha256((arm_dir / n).read_bytes()).hexdigest()
                                                    for n in ("arm.toml", "kinematics.json")})
        self.assertEqual(plan["holdout"], {"colors": [], "counts": []})

    def test_rerun_after_completion_is_a_no_op(self):
        before = tree_bytes(self.root)
        _, summary, _ = H.run_shard(self.cfg_path, FAST, rig=self.rig)
        self.assertEqual(tree_bytes(self.root), before)
        self.assertEqual(summary, self.summary)

    def test_changed_config_refused(self):
        other = H.write_config(self.work / "q2.yaml", tasks=[FAST], dst_root=self.work / "q", quotas=Q3, master_seed=5)
        with self.assertRaises(ResumeError):
            H.run_shard(other, FAST, rig=self.rig)


class ShardTests(LoopBase):
    """WP7 §8.1 shards own index % n == i, split the quotas with shard_quota, and give attempt-wise identical
    results to a single shard."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.single = H.run_shard(H.write_config(cls.work / "s1.yaml", tasks=[FAST], dst_root=cls.work / "s1",
                                                quotas=Q3), FAST, rig=cls.rig)[0]
        cls.shards = [H.run_shard(H.write_config(cls.work / f"s3_{i}.yaml", tasks=[FAST], dst_root=cls.work / f"s3_{i}",
                                                 quotas=Q3, shard=f"{i}/3"), FAST, rig=cls.rig)[0] for i in range(3)]

    def test_ownership_and_quota_split(self):
        for i, root in enumerate(self.shards):
            lines = manifest_dicts(root)
            with self.subTest(shard=i):
                self.assertTrue(all(d["attempt_index"] % 3 == i and d["shard"] == [i, 3] for d in lines))
                kinds = [sc["kind"] for sc in sidecar_dicts(root).values()]
                self.assertEqual({k: kinds.count(k) for k in Q3}, {k: shard_quota(q, i, 3) for k, q in Q3.items()})

    def test_zero_quota_shard_writes_no_lines(self):
        cfg = H.write_config(self.work / "z.yaml", tasks=[FAST], dst_root=self.work / "z",
                             quotas={"success": 2, "failure": 1, "recovery": 0}, shard="2/3")
        root, summary, _ = H.run_shard(cfg, FAST, rig=self.rig)
        self.assertEqual(manifest_dicts(root), [])
        self.assertEqual(summary.attempts, 0)

    def test_attempts_identical_across_shard_counts(self):
        single = {(d["kind"], d["attempt_index"]): d for d in manifest_dicts(self.single)}
        sharded = {(d["kind"], d["attempt_index"]): d for r in self.shards for d in manifest_dicts(r)}
        common = sorted(set(single) & set(sharded))
        self.assertGreaterEqual(len(common), 6)
        side1 = {(sc["kind"], sc["attempt"]["index"]): (self.single, i, sc)
                 for i, sc in sidecar_dicts(self.single).items()}
        side3 = {(sc["kind"], sc["attempt"]["index"]): (r, i, sc) for r in self.shards for i, sc in sidecar_dicts(r).items()}
        strip = lambda d: {k: v for k, v in d.items() if k not in ("shard", "episode_index")}  # noqa: E731
        for key in common:
            with self.subTest(attempt=key):
                self.assertEqual(strip(single[key]), strip(sharded[key]))
                if key in side1:
                    r1, i1, a = side1[key]
                    r3, i3, b = side3[key]
                    norm = lambda sc: {**sc, "episode_index": 0, "attempt": {**sc["attempt"], "shard": None}}  # noqa: E731
                    self.assertEqual(norm(a), norm(b))
                    f1, f3 = read_frames(r1)[i1], read_frames(r3)[i3]
                    for col in f1:
                        np.testing.assert_array_equal(f1[col], f3[col])


class ResumeTests(LoopBase):
    """WP7 §8.4 crash anywhere, resume: the result equals an uninterrupted run (data, sidecars and info.json
    byte for byte; manifest line for line except ``wall_s``; plan except ``created_at``)."""

    KILLS = [("env", 25), ("after_save", 0), ("torn", None), ("before_save", 1), ("env", 140), ("after_save", 3),
             ("torn", None), ("env", 260)]

    def check_equal(self, task):
        clean = H.run_shard(self.config(task, quotas=Q3), task, rig=self.rig)
        crashed = H.run_shard(self.config(task, quotas=Q3), task, rig=self.rig, kills=self.KILLS)
        (ra, sa, _), (rb, sb, crashes) = clean, crashed
        self.assertGreaterEqual(crashes, 4, "kill points did not fire")
        a, b = tree_bytes(ra), tree_bytes(rb)
        self.assertEqual(sorted(a), sorted(b))
        for path in a:
            if path in ("meta/generation_manifest.jsonl", "meta/generation_plan.json"):
                continue
            self.assertEqual(a[path], b[path], path)
        self.assertEqual(manifest_dicts(ra), manifest_dicts(rb))
        pa, pb = (json.loads(x["meta/generation_plan.json"]) for x in (a, b))
        pa.pop("created_at"), pb.pop("created_at")
        pa["config"].pop("dst_root", None), pb["config"].pop("dst_root", None)
        self.assertEqual(pa, pb)
        self.assertEqual(sa, sb)

    def test_resume_equals_uninterrupted_fast(self):
        self.check_equal(FAST)

    def test_resume_equals_uninterrupted_reach(self):
        self.check_equal("fx7_reach")

    def test_crash_points_leave_reconcilable_state(self):
        """before_save leaves an orphan sidecar and no committed episode; after_save leaves a committed episode
        without its line, which the next run rebuilds from the sidecar (``wall_s`` 0.0)."""
        from core.datasets.frames import MemoryWriter
        from core.generate.fake_env import FakeEnv, SimulatedCrash
        from core.generate.loop import make_expert_factory, make_spec_fn, run_generation
        cfg_path = self.config(FAST, quotas=Q3)
        cfg = load_generate_config(cfg_path)
        task = self.task(FAST)
        root = cfg.dataset_root(FAST)

        def crash_run(mode):
            w = MemoryWriter(root, fps=30, joint_names=tuple(self.rig.chain.q_names), crash=mode, crash_on_episode=0)
            with self.assertRaises(SimulatedCrash):
                run_generation(cfg, FAST, FakeEnv(task, self.rig.chain, self.rig.gripper), w,
                               expert_factory=make_expert_factory(task, ARM), mapping=self.rig.mapping,
                               spec_fn=make_spec_fn(task, ARM, scene="fx_scene", noise=cfg.noise), git_commit="0" * 40)

        def total():
            return json.loads((root / "meta" / "info.json").read_text())["total_episodes"]

        crash_run("before_save")
        self.assertEqual([d for d in manifest_dicts(root) if d["status"] == "accepted"], [])
        self.assertTrue((root / "task_meta" / "episode_000000.json").exists())
        self.assertEqual(total(), 0)
        crash_run("after_save")
        self.assertEqual([d for d in manifest_dicts(root) if d["status"] == "accepted"], [])
        self.assertEqual(total(), 1)
        H.run_shard(cfg_path, FAST, rig=self.rig)
        lines = [json.loads(x) for x in (root / "meta" / "generation_manifest.jsonl").read_text().splitlines()]
        first = [d for d in lines if d["status"] == "accepted"][0]
        self.assertEqual(first["episode_index"], 0)
        self.assertEqual(first["wall_s"], 0.0)
        self.assertEqual(total(), 9)


class BudgetAndDuplicateTests(LoopBase):
    """WP7 §8.1 budgets, §7.2 duplicate_trajectory and ScenarioSpaceExhausted."""

    def test_attempt_budget(self):
        """An expert that never succeeds: every attempt is rejected; the run stops with AttemptBudgetExceeded
        before executing attempt ceil(factor × quota) + 1 of a kind."""
        cfg = self.config(FAST, quotas={"success": 1, "failure": 1, "recovery": 1}, max_attempt_factor=2.0)
        with self.assertRaises(AttemptBudgetExceeded):
            H.run_shard(cfg, FAST, rig=self.rig, expert_factory=lambda spec: ConstantExpert(fx_home_q(ARM), frames=3))
        root = load_generate_config(cfg).dataset_root(FAST)
        lines = manifest_dicts(root)
        self.assertEqual(len(lines), 6)
        for kind in ("success", "failure", "recovery"):
            self.assertEqual(sum(d["kind"] == kind for d in lines), math.ceil(2.0 * 1))
        self.assertTrue(all(d["status"] == "rejected" for d in lines))
        self.assertEqual(sidecar_dicts(root), {})

    def test_duplicate_trajectory_rejected(self):
        """A stub expert gives identical trajectories for different specs: the first is kept, the others are
        rejected as duplicate_trajectory (also after a restart), and the dataset holds one episode."""
        t = task_table(FAST)
        t["checks"]["opened"]["tol_m"] = 1.0                       # every gap passes
        name = write_variant(self.box, "fx7_dup", FAST, checks=t["checks"])
        cfg = self.config(name, quotas={"success": 3, "failure": 0, "recovery": 0}, max_attempt_factor=2.0,
                          noise_fraction=0.0)
        stub = lambda spec: ConstantExpert(fx_home_q(ARM), frames=4)  # noqa: E731
        for _ in range(2):
            with self.assertRaises(AttemptBudgetExceeded):
                H.run_shard(cfg, name, rig=self.rig, expert_factory=stub)
        root = load_generate_config(cfg).dataset_root(name)
        lines = manifest_dicts(root)
        self.assertEqual([d["status"] for d in lines], ["accepted"] + ["rejected"] * 5)
        self.assertTrue(all(d["reason"] == "duplicate_trajectory" for d in lines[1:]))
        self.assertTrue(all(d["traj_fp"] == lines[0]["traj_fp"] for d in lines))
        self.assertEqual(len(set(d["scenario_key"] for d in lines)), 6)
        self.assertEqual(list(sidecar_dicts(root)), [0])
        self.assertEqual(sorted(read_frames(root)), [0])

    def test_scenario_space_exhausted(self):
        """A task with one possible start: after the first attempt every position is a scenario_seen skip;
        max_consecutive_skips skips of one kind raise ScenarioSpaceExhausted."""
        t = task_table(FAST)
        start = {"goal": {"gap": {"type": "gap", "gap_m": [0.02, 0.02]}}}
        expert = {"max_steps": 60, "settle_steps": 5,
                  "steps": [{"skill": "set_gripper", "gap_m": "@goal.gap", "stage": "opened", "budget": 30},
                            {"skill": "wait", "steps": 5, "stage": "success"}]}
        name = write_variant(self.box, "fx7_one", FAST, start=start, expert=expert)
        cfg = self.config(name, quotas={"success": 3, "failure": 0, "recovery": 0}, max_consecutive_skips=5,
                          noise_fraction=0.0)
        with self.assertRaises(ScenarioSpaceExhausted):
            H.run_shard(cfg, name, rig=self.rig)
        lines = manifest_dicts(load_generate_config(cfg).dataset_root(name))
        self.assertEqual([d["status"] for d in lines], ["accepted"] + ["skipped"] * 5)
        for d in lines[1:]:
            self.assertEqual(d["reason"], "scenario_seen")
            self.assertEqual(d["scenario_key"], lines[0]["scenario_key"])
            self.assertIsNotNone(d["spec_fp"])
            self.assertIsNone(d["traj_fp"])
            self.assertIsNone(d["num_frames"])


if __name__ == "__main__":
    unittest.main()
