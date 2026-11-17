"""WP8: CLI wiring for the v4 programs, the smoke config, and SO-101-free sim code (contracts §12, §2, §1.6).

Checked without Isaac Sim (AST, tomllib and the config loader only):
* ``src/cli/paths.py`` ``SIM_PROGRAMS`` gains six rows and keeps every existing row;
* ``pyproject.toml`` ``[project.scripts]`` gains the six console scripts and keeps every existing one;
* every target module exists and defines ``main`` at top level;
* ``configs/generate/smoke.yaml`` loads with ``load_generate_config`` and names an existing arm, scene and tasks;
* no SO-101 string literal remains in the files §12 lists, nor in ``sim/scene/arm.py``,
  ``sim/scene/contact_materials.py``, ``sim/eval/grasp_diagnostics.py`` and ``tasks/pick_place_cube/env_cfg.py``
  (link and mesh names come from ``[arm.links]``: ``ee``, ``moving_jaw``, ``fixed_jaw_mesh``, ``moving_jaw_mesh``).
"""

import ast
import tomllib
import unittest

from v4_support import REPO_ROOT, SRC_DIR

NEW_PROGRAMS = {"generate": "scripted_generate", "generate-plan": "plan_generation", "arm-kinematics": "extract_kinematics",
                "dataset-audit": "audit_dataset", "dataset-merge": "merge_shards", "dataset-upload": "upload_dataset"}
OLD_PROGRAMS = {"record": "lerobot_agent", "eval": "lerobot_eval", "rollout": "real_rollout", "spawn": "spawn",
                "envs": "list_envs", "dr-replay": "dr_replay", "calibration-replay": "calibration_replay",
                "calibration-report": "calibration_report", "eval-set": "make_eval_set", "calibrate": "calibrate_arm",
                "calibration-check": "check_calibration", "calibration-stats": "calibration_stats"}
NEW_SCRIPTS = {"scripted_generate": "sim.generate.scripted_generate:main",
               "plan_generation": "core.scripts.plan_generation:main",
               "extract_kinematics": "sim.generate.extract_kinematics:main",
               "audit_dataset": "core.scripts.audit_dataset:main",
               "merge_shards": "core.scripts.merge_shards:main",
               "upload_dataset": "core.scripts.upload_dataset:main"}
OLD_SCRIPTS = {"s2r": "cli.main:main", "lerobot_agent": "sim.teleop.lerobot_agent:main",
               "spawn": "core.scripts.spawn:main", "list_envs": "core.scripts.list_envs:main",
               "lerobot_eval": "sim.eval.lerobot_eval:main", "dr_replay": "sim.generate.dr_replay:main",
               "calibration_replay": "sim.generate.calibration_replay:main"}
NO_LITERAL_FILES = ("src/core/expert", "src/core/generate", "src/sim/generate/isaac_gen_env.py",
                    "src/sim/generate/scripted_generate.py", "src/sim/envs/generic_task_cfg.py",
                    "src/sim/envs/mdp/objects.py", "src/sim/scene/arm.py", "src/sim/scene/contact_materials.py",
                    "src/sim/eval/grasp_diagnostics.py", "tasks/pick_place_cube/env_cfg.py")


def sim_programs():
    tree = ast.parse((SRC_DIR / "cli" / "paths.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SIM_PROGRAMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("SIM_PROGRAMS not found")


def forbidden_literals():
    arm = tomllib.loads((REPO_ROOT / "arms" / "so101" / "arm.toml").read_text())["arm"]
    exact = {"so101", "jaw", "gripper", "moving_jaw_so101_v1", "wrist_roll_follower_so101_v1",
             *arm["joints"]["names"], *arm["lerobot"]["names"]}
    return exact


def string_constants(path):
    """String constants of a file; docstrings and mapping keys (``x["gripper"]``, ``x.get("gripper")``: the
    ``[arm.gripper]`` table) are excluded; f-string pieces are included."""
    tree = ast.parse(path.read_text())
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            skip.add(id(n.body[0].value))
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
            skip.add(id(n.slice))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                and isinstance(n.args[0], ast.Constant):
            skip.add(id(n.args[0]))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


class CliWiringTests(unittest.TestCase):
    """WP8 §12 SIM_PROGRAMS rows and console scripts."""

    def test_sim_programs(self):
        rows = sim_programs()
        for key, value in {**OLD_PROGRAMS, **NEW_PROGRAMS}.items():
            with self.subTest(program=key):
                self.assertEqual(rows.get(key), value)

    def test_console_scripts(self):
        scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
        for key, value in {**OLD_SCRIPTS, **NEW_SCRIPTS}.items():
            with self.subTest(script=key):
                self.assertEqual(scripts.get(key), value)
        self.assertEqual(set(NEW_PROGRAMS.values()), set(NEW_SCRIPTS))

    def test_targets_define_main(self):
        for target in NEW_SCRIPTS.values():
            module, func = target.split(":")
            path = SRC_DIR / (module.replace(".", "/") + ".py")
            with self.subTest(target=target):
                self.assertTrue(path.is_file(), path)
                names = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
                self.assertIn(func, names)


class SmokeConfigTests(unittest.TestCase):
    """WP8 configs/generate/smoke.yaml: a valid generate config for the real arm, scene and existing tasks."""

    def test_smoke_config(self):
        from core.generate.config import load_generate_config
        path = REPO_ROOT / "configs" / "generate" / "smoke.yaml"
        cfg = load_generate_config(path)
        self.assertTrue((REPO_ROOT / "arms" / cfg.arm / "arm.toml").is_file())
        self.assertTrue((REPO_ROOT / "scenes" / cfg.scene).is_dir())
        self.assertIn("pick_place_cube", cfg.tasks)
        for task in cfg.tasks:
            with self.subTest(task=task):
                self.assertTrue((REPO_ROOT / "tasks" / task / "task.toml").is_file())
                q = cfg.quotas[task]
                self.assertTrue(1 <= q.success <= 20 and 1 <= q.failure <= 20 and 1 <= q.recovery <= 20)
        self.assertEqual(tuple(cfg.shard), (0, 1))


class NoSo101LiteralTests(unittest.TestCase):
    """WP8 §12/§2: arm-specific names come from arm.toml, never from string literals."""

    def test_no_literals(self):
        bad = forbidden_literals()
        files = []
        for rel in NO_LITERAL_FILES:
            p = REPO_ROOT / rel
            files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for path in files:
            with self.subTest(file=str(path.relative_to(REPO_ROOT))):
                self.assertTrue(path.is_file(), path)
                found = [s for s in string_constants(path)
                         if s in bad or "so101" in s.lower() or s.rstrip("/").endswith(("/jaw", "/gripper"))]
                self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
