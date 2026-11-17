"""WP12: public surface and module hygiene of consolidate / merge / upload / streaming and the two
scripts (contracts §1.4, §1.6, §2, §9.2, §10.2–§10.4, §12 SO-101 literals).

Contract additions pinned here (docs/v4/prompts/WP12.md): ``ShardSummary.from_root`` (classmethod),
``streaming.iter_episodes`` and the ``storage`` keyword of ``upload_dataset.main``.
"""

import ast
import dataclasses
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

from core.datasets import consolidate, merge, streaming, upload
from core.scripts import merge_shards, upload_dataset
from test_v4_wp12_support import HEAVY_ROOTS
from v4_support import REPO_ROOT

DATASETS_DIR = Path(consolidate.__file__).resolve().parent
SRC_ROOT = DATASETS_DIR.parents[1]
PUBLIC = {
    consolidate: {"consolidate", "read_consolidated"},
    merge: {"ShardSummary", "MergePlan", "plan_merge", "apply_merge_meta", "merge_frames"},
    upload: {"StorageClient", "HubStorage", "BucketStorage", "UploadItem", "UploadPlan", "plan_upload", "run_upload"},
    streaming: {"select_episodes", "make_streaming_dataset", "iter_episodes"},
    merge_shards: {"main"},
    upload_dataset: {"main"},
}
#: Allowed only inside functions (container-only code paths, §1.6).
LAZY_ONLY = {"lerobot", "huggingface_hub", "fsspec"}
FORBIDDEN_ROOTS = {"isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "torch", "random", "yaml", "sim", "tasks",
                   "s2r_tasks", "boto3", "botocore", "s3fs", "datasets", "pandas"}
#: WP12 may import WP0, WP4, WP5 and existing code (§2), not WP1/WP2/WP3/WP6/WP7 modules.
FORBIDDEN_CORE = ("core.tasks.generic", "core.tasks.sampling", "core.arms", "core.expert", "core.generate.perturbations",
                  "core.generate.labels", "core.generate.specs", "core.generate.loop", "core.generate.fake_env",
                  "core.datasets.audit", "core.datasets.frames")
SO101_LITERALS = {"so101", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
                  "jaw", "moving_jaw_so101_v1", "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"}


def _file(module):
    return Path(module.__file__).resolve()


def _imports(tree):
    """(node, module name, inside a function) for every import statement."""
    out = []

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            inside = in_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                out.extend((child, alias.name, in_function) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                out.append((child, child.module, in_function))
            visit(child, inside)
    visit(tree, False)
    return out


def _params(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


P = inspect.Parameter


class SurfaceTests(unittest.TestCase):
    """WP12 §2 module map and §1.6 imports."""

    def test_files(self):
        """§2 the WP12 modules live at the contract paths."""
        self.assertEqual({_file(m).name for m in (consolidate, merge, upload, streaming)},
                         {"consolidate.py", "merge.py", "upload.py", "streaming.py"})
        self.assertEqual(_file(consolidate).parent, SRC_ROOT / "core" / "datasets")
        self.assertEqual({_file(merge_shards), _file(upload_dataset)},
                         {SRC_ROOT / "core" / "scripts" / "merge_shards.py", SRC_ROOT / "core" / "scripts" / "upload_dataset.py"})

    def test_public_names(self):
        """§2 each module defines exactly its contract names (plus iter_episodes); helpers are private."""
        for module, expected in PUBLIC.items():
            defined = {name for name, obj in vars(module).items()
                       if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))
                       and getattr(obj, "__module__", None) == module.__name__}
            tree = ast.parse(_file(module).read_text())
            constants = {t.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                         for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                         if isinstance(t, ast.Name) and not t.id.startswith("_")}
            with self.subTest(module=module.__name__):
                self.assertEqual(defined | constants, expected)

    def test_imports(self):
        """§1.6 stdlib, numpy, pyarrow, core at module level; lerobot / huggingface_hub / fsspec only inside functions."""
        for module in PUBLIC:
            tree = ast.parse(_file(module).read_text())
            for node, name, in_function in _imports(tree):
                root = name.split(".")[0]
                allowed = set(sys.stdlib_module_names) | {"numpy", "pyarrow", "core"} | (LAZY_ONLY if in_function else set())
                with self.subTest(module=module.__name__, line=node.lineno, imported=name):
                    self.assertNotIn(root, FORBIDDEN_ROOTS)
                    self.assertIn(root, allowed)
                    self.assertFalse(any(name == f or name.startswith(f + ".") for f in FORBIDDEN_CORE))

    def test_no_global_numpy_random(self):
        """§1.4 only np.random.default_rng / Generator are used."""
        for module in PUBLIC:
            for node in ast.walk(ast.parse(_file(module).read_text())):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "random":
                    with self.subTest(module=module.__name__, line=node.lineno):
                        self.assertIn(node.attr, {"default_rng", "Generator"})

    def test_python_311_syntax(self):
        """§1.6 the WP12 files parse as Python 3.11."""
        for module in PUBLIC:
            with self.subTest(module=module.__name__):
                ast.parse(_file(module).read_text(), feature_version=(3, 11))

    def test_no_so101_literals(self):
        """§12 no SO-101 literal appears in the WP12 files (string constants)."""
        for module in PUBLIC:
            for node in ast.walk(ast.parse(_file(module).read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    with self.subTest(module=module.__name__, line=node.lineno):
                        self.assertNotIn(node.value, SO101_LITERALS)

    def test_import_pulls_no_heavy_modules(self):
        """§1.6 importing every WP12 module loads no Isaac, torch, lerobot, huggingface_hub, fsspec, boto3 or sim."""
        code = ("import sys; import core.datasets.consolidate, core.datasets.merge, core.datasets.upload, "
                "core.datasets.streaming, core.scripts.merge_shards, core.scripts.upload_dataset; "
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(HEAVY_ROOTS)!r}))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(SRC_ROOT), "PATH": "/usr/bin:/bin"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


class SignatureTests(unittest.TestCase):
    """WP12 §9.2, §10.2–§10.4 parameter names, keyword-only flags and defaults."""

    def test_functions(self):
        """§9.2/§10 function signatures."""
        empty = P.empty
        expect = {
            consolidate.consolidate: [("root", P.POSITIONAL_OR_KEYWORD, empty)],
            consolidate.read_consolidated: [("root", P.POSITIONAL_OR_KEYWORD, empty)],
            merge.plan_merge: [("shards", P.POSITIONAL_OR_KEYWORD, empty)],
            merge.apply_merge_meta: [("plan", P.POSITIONAL_OR_KEYWORD, empty), ("shards", P.POSITIONAL_OR_KEYWORD, empty),
                                     ("dst_root", P.POSITIONAL_OR_KEYWORD, empty)],
            merge.merge_frames: [("plan", P.POSITIONAL_OR_KEYWORD, empty), ("shards", P.POSITIONAL_OR_KEYWORD, empty),
                                 ("dst_root", P.POSITIONAL_OR_KEYWORD, empty), ("repo_id", P.POSITIONAL_OR_KEYWORD, empty)],
            upload.plan_upload: [("root", P.POSITIONAL_OR_KEYWORD, empty), ("storage", P.POSITIONAL_OR_KEYWORD, empty),
                                 ("allow_overwrite", P.KEYWORD_ONLY, False)],
            upload.run_upload: [("plan", P.POSITIONAL_OR_KEYWORD, empty), ("storage", P.POSITIONAL_OR_KEYWORD, empty),
                                ("message", P.KEYWORD_ONLY, empty)],
            streaming.select_episodes: [("rows", P.POSITIONAL_OR_KEYWORD, empty), ("kinds", P.KEYWORD_ONLY, None),
                                        ("success", P.KEYWORD_ONLY, None), ("exclude_holdout_of", P.KEYWORD_ONLY, None)],
            streaming.make_streaming_dataset: [("repo_id", P.POSITIONAL_OR_KEYWORD, empty), ("episodes", P.KEYWORD_ONLY, None),
                                               ("kwargs", P.VAR_KEYWORD, empty)],
            streaming.iter_episodes: [("root", P.POSITIONAL_OR_KEYWORD, empty), ("episodes", P.KEYWORD_ONLY, None)],
            merge_shards.main: [("argv", P.POSITIONAL_OR_KEYWORD, None)],
            upload_dataset.main: [("argv", P.POSITIONAL_OR_KEYWORD, None), ("storage", P.KEYWORD_ONLY, None)],
        }
        for fn, params in expect.items():
            with self.subTest(fn=fn.__qualname__):
                self.assertEqual(_params(fn), params)

    def test_dataclasses(self):
        """§10.2/§10.3 frozen dataclasses with the contract fields in order."""
        expect = {merge.ShardSummary: ["root", "episodes", "episode_fps", "scenario_keys", "report"],
                  merge.MergePlan: ["index_maps", "total"],
                  upload.UploadItem: ["local", "remote", "sha256", "size"],
                  upload.UploadPlan: ["items", "skipped"]}
        for cls, fields in expect.items():
            with self.subTest(cls=cls.__name__):
                self.assertTrue(dataclasses.is_dataclass(cls))
                self.assertTrue(cls.__dataclass_params__.frozen)
                self.assertEqual([f.name for f in dataclasses.fields(cls)], fields)
        self.assertIsInstance(inspect.getattr_static(merge.ShardSummary, "from_root"), classmethod)

    def test_storage_classes(self):
        """§10.3 StorageClient protocol members; HubStorage / BucketStorage constructors."""
        self.assertTrue(getattr(upload.StorageClient, "_is_protocol", False))
        self.assertTrue(getattr(upload.StorageClient, "_is_runtime_protocol", False))
        for name in ("list_files", "upload_file", "finalize"):
            with self.subTest(member=name):
                self.assertTrue(callable(getattr(upload.StorageClient, name)))
        self.assertEqual(_params(upload.StorageClient.upload_file)[1:],
                         [("local", P.POSITIONAL_OR_KEYWORD, P.empty), ("remote", P.POSITIONAL_OR_KEYWORD, P.empty)])
        self.assertEqual(_params(upload.HubStorage.__init__)[1:],
                         [("api", P.POSITIONAL_OR_KEYWORD, P.empty), ("repo_id", P.POSITIONAL_OR_KEYWORD, P.empty),
                          ("revision", P.KEYWORD_ONLY, "main"), ("repo_type", P.KEYWORD_ONLY, "dataset")])
        self.assertEqual(_params(upload.BucketStorage.__init__)[1:],
                         [("fs", P.POSITIONAL_OR_KEYWORD, P.empty), ("prefix", P.POSITIONAL_OR_KEYWORD, P.empty)])
        for cls in (upload.HubStorage, upload.BucketStorage):
            for name in ("list_files", "upload_file", "finalize"):
                with self.subTest(cls=cls.__name__, member=name):
                    self.assertTrue(callable(getattr(cls, name)))


if __name__ == "__main__":
    unittest.main()
