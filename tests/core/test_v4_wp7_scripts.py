"""WP7: the ``dataset-audit`` and ``generate-plan`` scripts, public names and import hygiene (contracts §1.6, §2,
§8.1, §10.1, §12).

Decisions pinned here (docs/v4/prompts/WP7.md):
* ``core.scripts.audit_dataset.main(argv) -> int``: positional ``root``, ``--source_root``, ``--no_manifest``;
  writes ``meta/audit_report.json``; prints one line per check, ``PASS <name>`` or ``FAIL <name>: <details>``;
  returns 0 when the audit passed, else 1.
* ``core.scripts.plan_generation.main(argv) -> int``: ``--config_path`` (required), ``--task``, ``--shard i/n``;
  prints one JSON object per task (keys ``task``, ``shard``, ``quotas``, ``shard_quotas``, ``non_skipped``,
  ``horizon``; kinds by value) from ``check_capacity`` with the config's spec function; on a
  ``GenerationError`` prints ``<ErrorClass>: <message>`` to stderr and returns 1. It never imports Isaac.
* ``core.generate.loop`` also exports ``make_spec_fn(task, arm, *, scene, noise=None, weights_override=None,
  fps=30, setup=None, reach_fn=None)`` and ``home_q(arm, *, setup=None)`` (initial joint positions in
  ``[arm.joints].names`` order, overridden by ``[arm.expert].home_rad``), so the runner (WP8) and the plan
  script build specs the same way. ``core.datasets.audit.CHECKS`` holds the ten check names in order.
"""

import ast
import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path

from core.generate.config import load_generate_config
from core.generate.loop import home_q, make_spec_fn
from core.generate.plan import KINDS, check_capacity, shard_quota
from core.scripts import audit_dataset as audit_script
from core.scripts import plan_generation as plan_script
from test_v4_wp7_support import ARM, CHECKS, FAST, SandboxCase, import_clean, task_table, write_variant
from v4_support import REPO_ROOT, SRC_DIR
from v4_support import harness as H
from v4_support.fx_arms import fx_home_q

MODULES = {
    "core.generate.fake_env": {"FakeEnv", "SimulatedCrash"},
    "core.datasets.frames": {"LiteDatasetWriter", "read_frames", "MemoryWriter"},
    "core.generate.loop": {"run_episode", "run_generation", "GenerationSummary", "make_expert_factory",
                           "make_spec_fn", "home_q"},
    "core.datasets.audit": {"AuditReport", "audit_dataset", "write_audit_report", "read_audit_report",
                            "content_digest", "CHECKS"},
    "core.scripts.audit_dataset": {"main"},
    "core.scripts.plan_generation": {"main"},
}
OPTIONAL = {"core.scripts.audit_dataset": {"build_parser"}, "core.scripts.plan_generation": {"build_parser"},
            "core.datasets.frames": {"LITE_VERSION"}}


def module_path(name):
    return SRC_DIR / (name.replace(".", "/") + ".py")


def run_main(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class AuditScriptTests(SandboxCase):
    """WP7 §10.1 ``dataset-audit``."""

    TASKS = (FAST,)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root, _, _ = H.run_shard(H.write_config(cls.work / "a.yaml", tasks=[FAST], dst_root=cls.work / "a",
                                                    quotas={"success": 2, "failure": 1, "recovery": 1}),
                                     FAST, rig=cls.rig)

    def test_pass(self):
        root = self.fresh("s") / "ds"
        shutil.copytree(self.root, root)
        code, out, _ = run_main(audit_script.main, [str(root)])
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual([ln.split()[1].rstrip(":") for ln in lines[:len(CHECKS)]], list(CHECKS))
        self.assertTrue(all(ln.startswith("PASS ") for ln in lines[:len(CHECKS)]))
        report = json.loads((root / "meta" / "audit_report.json").read_text())
        self.assertTrue(report["passed"])

    def test_fail(self):
        root = self.fresh("s") / "ds"
        shutil.copytree(self.root, root)
        (root / "task_meta" / "episode_000000.json").unlink()
        code, out, _ = run_main(audit_script.main, [str(root)])
        self.assertEqual(code, 1)
        self.assertTrue(any(ln.startswith("FAIL counts_match") for ln in out.splitlines()))
        self.assertFalse(json.loads((root / "meta" / "audit_report.json").read_text())["passed"])

    def test_flags(self):
        root = self.fresh("s") / "ds"
        shutil.copytree(self.root, root)
        path = root / "meta" / "generation_manifest.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(ln for ln in lines if '"status":"accepted"' not in ln))
        self.assertEqual(run_main(audit_script.main, [str(root)])[0], 1)
        self.assertEqual(run_main(audit_script.main, [str(root), "--no_manifest"])[0], 0)
        self.assertEqual(run_main(audit_script.main, [str(root), "--source_root", str(self.root)])[0], 1)


class PlanScriptTests(SandboxCase):
    """WP7 §8.1/§12 ``generate-plan``: capacity report per task without Isaac."""

    TASKS = (FAST, "fx7_reach")

    def test_report(self):
        cfg_path = H.write_config(self.work / "p.yaml", tasks=[FAST, "fx7_reach"], dst_root=self.work / "p",
                                  quotas={"success": 4, "failure": 2, "recovery": 3})
        code, out, err = run_main(plan_script.main, ["--config_path", str(cfg_path), "--shard", "1/3"])
        self.assertEqual(code, 0, err)
        rows = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
        self.assertEqual([r["task"] for r in rows], [FAST, "fx7_reach"])
        cfg = load_generate_config(cfg_path)
        for row in rows:
            with self.subTest(task=row["task"]):
                self.assertEqual(set(row), {"task", "shard", "quotas", "shard_quotas", "non_skipped", "horizon"})
                self.assertEqual(row["shard"], [1, 3])
                self.assertEqual(row["quotas"], {"success": 4, "failure": 2, "recovery": 3})
                self.assertEqual(row["shard_quotas"], {k: shard_quota(v, 1, 3) for k, v in row["quotas"].items()})
                task = self.task(row["task"])
                fn = make_spec_fn(task, ARM, scene=cfg.scene, noise=cfg.noise, fps=cfg.fps)
                rep = check_capacity(row["task"], 0, fn, cfg.quotas[row["task"]],
                                     max_attempt_factor=cfg.max_attempt_factor,
                                     max_consecutive_skips=cfg.max_consecutive_skips,
                                     near_duplicates=True)  # WP14 §16.6: generation skips near-duplicates
                self.assertEqual(row["non_skipped"], {k.value: rep.non_skipped[k] for k in KINDS})
                self.assertEqual(row["horizon"], rep.horizon)
        code, out, _ = run_main(plan_script.main, ["--config_path", str(cfg_path), "--task", "fx7_reach"])
        self.assertEqual([json.loads(ln)["task"] for ln in out.splitlines() if ln.strip()], ["fx7_reach"])

    def test_exhausted(self):
        start = {"goal": {"gap": {"type": "gap", "gap_m": [0.02, 0.02]}}}
        expert = {"max_steps": 60, "settle_steps": 5,
                  "steps": [{"skill": "set_gripper", "gap_m": "@goal.gap", "stage": "opened", "budget": 30},
                            {"skill": "wait", "steps": 5, "stage": "success"}]}
        name = write_variant(self.box, "fx7_one_plan", FAST, start=start, expert=expert)
        cfg_path = H.write_config(self.work / "x.yaml", tasks=[name], dst_root=self.work / "x",
                                  quotas={"success": 3, "failure": 0, "recovery": 0}, max_consecutive_skips=5)
        code, out, err = run_main(plan_script.main, ["--config_path", str(cfg_path)])
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("ScenarioSpaceExhausted:"), err)

    def test_home_q(self):
        import numpy as np
        np.testing.assert_array_equal(np.array(home_q(ARM)), fx_home_q(ARM))


class SurfaceTests(unittest.TestCase):
    """WP7 §1.6/§2: public names, Python 3.11 syntax, no heavy imports, no SO-101 literals."""

    def test_public_names(self):
        for mod, required in MODULES.items():
            tree = ast.parse(module_path(mod).read_text())
            names = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
            public = {n for n in names if not n.startswith("_")}
            with self.subTest(module=mod):
                self.assertTrue(required <= public, required - public)
                self.assertTrue(public <= required | OPTIONAL.get(mod, set()), public - required)

    def test_python_311_syntax(self):
        for mod in MODULES:
            with self.subTest(module=mod):
                ast.parse(module_path(mod).read_text(), feature_version=(3, 11))

    def test_import_hygiene(self):
        import_clean(self, list(MODULES))

    def test_no_so101_literals(self):
        import tomllib
        arm = tomllib.loads((REPO_ROOT / "arms" / "so101" / "arm.toml").read_text())["arm"]
        bad = {"so101", "jaw", "gripper", "moving_jaw_so101_v1", *arm["joints"]["names"], *arm["lerobot"]["names"]}
        for mod in MODULES:
            tree = ast.parse(module_path(mod).read_text())
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            found = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and id(n) not in docs and n.value in bad}
            with self.subTest(module=mod):
                self.assertEqual(found, set())


if __name__ == "__main__":
    unittest.main()
