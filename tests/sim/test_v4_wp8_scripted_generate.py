"""WP8: the ``generate`` runner — argument parsing and the Isaac-free ``--dry_run`` path (contracts §12).

Decisions pinned here (docs/v4/prompts/WP8.md):
* ``build_parser()``: ``--config_path`` (required), ``--task``, ``--shard i/n``, ``--dst_root``, ``--master_seed``
  (int), ``--dry_run``, ``--limit_attempts`` (int, default 0 = no limit), ``--disable_fabric``. ``main`` uses
  ``parse_known_args`` first, so AppLauncher arguments may follow; AppLauncher is imported only after the
  ``--dry_run`` branch.
* ``--shard``, ``--dst_root`` and ``--master_seed`` override the config (``load_generate_config(path, overrides)``).
* ``--dry_run`` prints exactly what ``generate-plan`` prints for the resolved config (one JSON object per task)
  and returns its exit code; it imports neither Isaac nor LeRobot.
* The real path builds an ``IsaacGenEnv`` and a ``RecorderEpisodeWriter`` and calls ``run_generation`` (checked on
  the AST here; it runs in the container).
"""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest

from core.scripts import plan_generation
from sim.generate import scripted_generate
from v4_support import SRC_DIR, TESTS_DIR, sandbox
from v4_support import harness as H

BLOCKER = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'isaaclab', 'isaacsim', 'omni', 'carb', 'pxr', 'warp', 'lerobot', 'gymnasium'}:
            raise ImportError('blocked: ' + name)
        return None
sys.meta_path.insert(0, _Block())
"""


def run_main(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ParserTests(unittest.TestCase):
    """WP8 §12 build_parser."""

    def test_arguments(self):
        p = scripted_generate.build_parser()
        a = p.parse_args(["--config_path", "c.yaml"])
        self.assertEqual((a.config_path, a.task, a.shard, a.dst_root, a.master_seed, a.dry_run, a.limit_attempts,
                          a.disable_fabric), ("c.yaml", None, None, None, None, False, 0, False))
        a = p.parse_args(["--config_path", "c.yaml", "--task", "t", "--shard", "1/4", "--dst_root", "/d",
                          "--master_seed", "7", "--dry_run", "--limit_attempts", "3", "--disable_fabric"])
        self.assertEqual((a.task, a.shard, a.dst_root, a.master_seed, a.dry_run, a.limit_attempts, a.disable_fabric),
                         ("t", "1/4", "/d", 7, True, 3, True))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                p.parse_args([])
        args, rest = p.parse_known_args(["--config_path", "c.yaml", "--headless", "--device", "cuda:0"])
        self.assertEqual(rest, ["--headless", "--device", "cuda:0"])

    def test_real_path_wiring(self):
        """main builds IsaacGenEnv and RecorderEpisodeWriter and calls run_generation; AppLauncher comes after the
        dry-run branch."""
        tree = ast.parse((SRC_DIR / "sim" / "generate" / "scripted_generate.py").read_text())
        calls = {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertTrue({"run_generation", "IsaacGenEnv", "RecorderEpisodeWriter"} <= calls, calls)
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) or "" for n in top}
        self.assertFalse({m for m in mods if m.split(".")[0] in ("isaaclab", "isaacsim", "omni", "lerobot", "torch")})


class DryRunTests(unittest.TestCase):
    """WP8 §12 --dry_run equals generate-plan for the resolved config, without Isaac."""

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, ("fx7_grip", "fx7_reach"))
        cls.cfg = H.write_config(cls.box.root / "g.yaml", tasks=["fx7_grip", "fx7_reach"], dst_root=cls.box.root / "d",
                                 quotas={"success": 4, "failure": 2, "recovery": 2})

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def test_same_as_generate_plan(self):
        for extra in ([], ["--shard", "2/3"], ["--task", "fx7_reach"]):
            with self.subTest(extra=extra):
                a = run_main(scripted_generate.main, ["--config_path", str(self.cfg), "--dry_run", *extra])
                b = run_main(plan_generation.main, ["--config_path", str(self.cfg), *extra])
                self.assertEqual(a[0], 0, a[2])
                self.assertEqual(a[:2], b[:2])

    def test_master_seed_override(self):
        other = H.write_config(self.box.root / "g7.yaml", tasks=["fx7_grip", "fx7_reach"], dst_root=self.box.root / "d",
                               quotas={"success": 4, "failure": 2, "recovery": 2}, master_seed=7)
        a = run_main(scripted_generate.main, ["--config_path", str(self.cfg), "--dry_run", "--master_seed", "7"])
        b = run_main(plan_generation.main, ["--config_path", str(other)])
        self.assertEqual(a[:2], b[:2])

    def test_dry_run_without_isaac(self):
        code = BLOCKER + (
            "from sim.generate import scripted_generate\n"
            f"rc = scripted_generate.main(['--config_path', {str(self.cfg)!r}, '--dry_run'])\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'lerobot', 'omni', 'pxr'))\n"
            "print('RC', rc, bad)\n")
        env = dict(os.environ, PYTHONPATH=f"{SRC_DIR}:{TESTS_DIR}")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines[-1], "RC 0 []")
        rows = [json.loads(x) for x in lines[:-1]]
        self.assertEqual([r["task"] for r in rows], ["fx7_grip", "fx7_reach"])


if __name__ == "__main__":
    unittest.main()
