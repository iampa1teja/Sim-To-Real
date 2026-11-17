"""WP5: public surface and module hygiene of plan / fingerprint / manifest / config / appearance
(contracts §1.4, §1.6, §2, §12 SO-101 literals)."""

import ast
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

from core.generate import appearance, config, fingerprint, manifest, plan
from v4_support import REPO_ROOT

GENERATE_DIR = Path(plan.__file__).resolve().parent
SRC_ROOT = GENERATE_DIR.parents[1]
PUBLIC = {
    plan: {"KINDS", "attempt_seed", "shard_quota", "owns", "PlannedAttempt", "iter_plan", "shard_attempts",
           "check_capacity", "CapacityReport"},
    fingerprint: {"quantize", "scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp"},
    manifest: {"ManifestLine", "Manifest", "reconcile", "ReconcileResult", "write_plan_json", "read_plan_json"},
    config: {"Quotas", "NoiseConfig", "DRConfig", "GenerateConfig", "load_generate_config"},
    appearance: {"AppearanceDraw", "draw_appearance", "dr_copy_sidecar"},
}
FILES = {m: Path(m.__file__).name for m in PUBLIC}
FORBIDDEN_ROOTS = {"isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch", "random", "pyarrow",
                   "huggingface_hub", "fsspec", "sim", "tasks", "s2r_tasks"}
#: WP5 may import WP0 and existing code only (§2): not WP1 (generic/sampling), WP2 (arms), WP4/WP12 (datasets) ...
FORBIDDEN_CORE = ("core.tasks.generic", "core.tasks.sampling", "core.arms", "core.datasets", "core.expert",
                  "core.generate.perturbations", "core.generate.labels", "core.generate.specs", "core.generate.loop",
                  "core.generate.fake_env")
SO101_LITERALS = {"so101", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
                  "jaw", "moving_jaw_so101_v1", "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"}


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node, node.module


class SurfaceTests(unittest.TestCase):
    """WP5 §2 module map and §1.6 imports."""

    def test_public_names(self):
        """§2 each WP5 module defines exactly its contract names (classes, functions, KINDS); helpers are private."""
        for module, expected in PUBLIC.items():
            defined = {name for name, obj in vars(module).items()
                       if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))
                       and getattr(obj, "__module__", None) == module.__name__}
            tree = ast.parse((GENERATE_DIR / FILES[module]).read_text())
            constants = {t.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                         for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                         if isinstance(t, ast.Name) and not t.id.startswith("_")}
            with self.subTest(module=module.__name__):
                self.assertEqual(defined | constants, expected)

    def test_imports(self):
        """§1.6 stdlib, numpy and core only; yaml only in config.py; no WP1/WP2/WP4+ modules, no sim, no random."""
        for module, filename in FILES.items():
            tree = ast.parse((GENERATE_DIR / filename).read_text())
            for node, name in _imports(tree):
                root = name.split(".")[0]
                allowed = set(sys.stdlib_module_names) | {"numpy", "core"} | ({"yaml"} if filename == "config.py" else set())
                with self.subTest(file=filename, line=node.lineno, module=name):
                    self.assertNotIn(root, FORBIDDEN_ROOTS)
                    self.assertIn(root, allowed)
                    self.assertFalse(any(name == f or name.startswith(f + ".") for f in FORBIDDEN_CORE))

    def test_no_global_numpy_random(self):
        """§1.4 only np.random.default_rng / Generator are used."""
        for filename in FILES.values():
            tree = ast.parse((GENERATE_DIR / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "random":
                    with self.subTest(file=filename, line=node.lineno):
                        self.assertIn(node.attr, {"default_rng", "Generator"})

    def test_python_311_syntax(self):
        """§1.6 the WP5 modules parse as Python 3.11."""
        for filename in FILES.values():
            with self.subTest(file=filename):
                ast.parse((GENERATE_DIR / filename).read_text(), feature_version=(3, 11))

    def test_no_so101_literals(self):
        """§12 no SO-101 literal appears in src/core/generate (string constants)."""
        for filename in FILES.values():
            tree = ast.parse((GENERATE_DIR / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    with self.subTest(file=filename, line=node.lineno):
                        self.assertNotIn(node.value, SO101_LITERALS)

    def test_import_pulls_no_heavy_modules(self):
        """§1.6 importing the WP5 modules loads no Isaac, torch, lerobot, pyarrow or sim; yaml only via config."""
        heavy = sorted(FORBIDDEN_ROOTS - {"random", "tasks"} | {"scipy"})
        code = ("import sys; import core.generate.plan, core.generate.fingerprint, core.generate.manifest, "
                "core.generate.appearance; "
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {heavy + ['yaml']!r}))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(SRC_ROOT), "PATH": "/usr/bin:/bin"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
