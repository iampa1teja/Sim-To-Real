"""Steps 7-8 - scripts/dr_replay.py + pyproject console script. Static (AST) checks only:
the script launches Isaac Sim, so its behaviour is covered by the GPU runs in the docs
(fidelity check, then a 2-episode red/DR run, then the full run).

CPU only: python -m unittest tests.test_dr_replay_script -v

Contract under test
- argparse flags and defaults:
    --src_root --src_repo_id --dst_root --dst_repo_id   (required, no default)
    --exclude_episodes            default ""        (e.g. "0,6")
    --colors                      default "blue,red"
    --dr                          default "light,robot_color"   (camera views are never randomized)
    --include_originals           store_true
    --lang_blue / --lang_red      default the blue / red pick-and-place instruction
    --seed                        default 0
    --dry_run                     store_true
- `--dry_run` is handled BEFORE AppLauncher is imported or constructed (so it runs on a
  machine without Isaac Sim and without touching the GPU)
- defines: read_dr_ranges, make_env, make_writer, replay_episode, main
- main is decorated with @torch.inference_mode()
- pyproject: lerobot_dr_replay = "sim_to_real_so101.scripts.dr_replay:main"
"""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "source/sim_to_real_so101/scripts/dr_replay.py"
PYPROJECT = REPO / "source/sim_to_real_so101/pyproject.toml"
LANG_BLUE = "Pick up the blue cube and place it in the white box"
LANG_RED = "Pick up the red cube and place it in the white box"


def literal(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


@unittest.skipUnless(SCRIPT.is_file(), "scripts/dr_replay.py not implemented yet")
class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.flags = {}
        for node in ast.walk(cls.tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("--")):
                cls.flags[node.args[0].value] = {k.arg: literal(k.value) for k in node.keywords}

    def test_flags_and_defaults(self):
        for flag in ("--src_root", "--src_repo_id", "--dst_root", "--dst_repo_id"):
            self.assertIn(flag, self.flags)
            self.assertTrue(self.flags[flag].get("required"), f"{flag} must be required")
        expected = {
            "--exclude_episodes": "",
            "--colors": "blue,red",
            "--dr": "light,robot_color",
            "--lang_blue": LANG_BLUE,
            "--lang_red": LANG_RED,
            "--seed": 0,
        }
        for flag, default in expected.items():
            self.assertIn(flag, self.flags)
            self.assertEqual(self.flags[flag].get("default"), default, flag)
        for flag in ("--include_originals", "--dry_run"):
            self.assertEqual(self.flags[flag].get("action"), "store_true", flag)

    def _first_line(self, predicate):
        lines = [n.lineno for n in ast.walk(self.tree) if predicate(n)]
        return min(lines) if lines else None

    def test_dry_run_before_isaac(self):
        dry = self._first_line(lambda n: isinstance(n, ast.Attribute) and n.attr == "dry_run"
                               and isinstance(n.ctx, ast.Load))
        launcher_import = self._first_line(
            lambda n: isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")
        launcher_call = self._first_line(
            lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", None) == "AppLauncher")
        self.assertIsNotNone(dry, "--dry_run is never read")
        self.assertIsNotNone(launcher_call, "AppLauncher(...) not found")
        self.assertLess(dry, launcher_call, "--dry_run must exit before AppLauncher(...)")
        if launcher_import is not None:
            self.assertLess(dry, launcher_import, "import isaaclab.app only after the --dry_run exit")

    def test_functions_and_inference_mode(self):
        functions = {n.name: n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        for name in ("read_dr_ranges", "make_env", "make_writer", "replay_episode", "main"):
            self.assertIn(name, functions)
        decorators = [ast.unparse(d) for d in functions["main"].decorator_list]
        self.assertTrue(any("inference_mode" in d for d in decorators), "decorate main with @torch.inference_mode()")

    def test_no_hardware_connection(self):
        self.assertNotIn(".connect(", self.source, "the replay must never connect to the robot")
        self.assertNotIn("init_device(", self.source)

    def test_refuses_existing_destination(self):
        self.assertRegex(self.source, r"dst_root.*exists\(\)|exists\(\).*dst_root",
                         "refuse to write into an existing --dst_root")


class PyprojectTests(unittest.TestCase):
    def test_console_script(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        if "lerobot_dr_replay" not in text:
            self.skipTest("console script not added yet")
        self.assertRegex(text, r'lerobot_dr_replay\s*=\s*"sim_to_real_so101\.scripts\.dr_replay:main"')


if __name__ == "__main__":
    unittest.main()
