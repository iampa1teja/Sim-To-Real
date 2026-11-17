"""WP0: the GenEnv / EpisodeWriter / Expert protocols and module hygiene (contracts §1.6, §2, §3.3)."""

import ast
import inspect
import subprocess
import sys
import typing
import unittest
from pathlib import Path

import numpy as np

import core.generate
from core.generate import errors, protocols, seeds
from core.generate.protocols import EpisodeWriter, Expert, GenEnv
from v4_support import REPO_ROOT
from v4_support.factories import make_expert_action, make_expert_obs
from v4_support.stubs import ListWriter

GENERATE_DIR = Path(protocols.__file__).resolve().parent          # src/core/generate (the imported package)
SRC_ROOT = GENERATE_DIR.parents[1]
WP0_FILES = ("__init__.py", "errors.py", "protocols.py", "seeds.py")
PROTOCOL_PUBLIC = {"Kind", "SegmentLabel", "AttemptStatus", "SkipReason", "RejectionReason", "PerturbationGroup",
                   "ObjectState", "PerturbationSpec", "NoiseSpec", "EpisodeSpec", "StageProgress", "ExpertObs",
                   "ExpertAction", "FrameLabels", "StepRecord", "Segment", "RecoveryInfo", "Fingerprints",
                   "EpisodeResult", "GenEnv", "EpisodeWriter", "Expert"}
FORBIDDEN_ROOTS = {"isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch", "random", "yaml",
                   "pyarrow", "scipy", "huggingface_hub", "fsspec"}


class _Env:
    fps = 30
    joint_names = ("a", "b")
    camera_names = ()

    def reset(self, spec):
        return make_expert_obs()

    def step(self, q_target):
        return make_expert_obs()

    def observe(self):
        return make_expert_obs()

    def set_object_pose(self, name, pose):
        return None

    def render(self):
        return {}

    def close(self):
        return None


class _EnvWithoutRender(_Env):
    render = None


class _Expert:
    def reset(self, spec, obs):
        self._phase = "start"

    def act(self, obs, progress):
        return make_expert_action()

    @property
    def phase(self):
        return "start"

    @property
    def done(self):
        return False


class _ExpertWithoutAct:
    def reset(self, spec, obs):
        pass

    phase = "x"
    done = False


def _params(func):
    return [p for p in inspect.signature(func).parameters if p != "self"]


class ProtocolShapeTests(unittest.TestCase):
    """WP0 §3.3 protocol members and signatures."""

    def test_protocols_are_runtime_checkable(self):
        """§3.3 the three protocols are typing.Protocol and runtime_checkable."""
        for proto in (GenEnv, EpisodeWriter, Expert):
            with self.subTest(proto=proto.__name__):
                self.assertTrue(getattr(proto, "_is_protocol", False))
                self.assertTrue(getattr(proto, "_is_runtime_protocol", False))
                self.assertIn(typing.Protocol, proto.__mro__)

    def test_gen_env_members(self):
        """§3.3 GenEnv attributes (fps, joint_names, camera_names) and method signatures."""
        hints = typing.get_type_hints(GenEnv)
        self.assertEqual(hints["fps"], int)
        self.assertEqual(hints["joint_names"], tuple[str, ...])
        self.assertEqual(hints["camera_names"], tuple[str, ...])
        expected = {"reset": ["spec"], "step": ["q_target"], "observe": [], "set_object_pose": ["name", "pose"],
                    "render": [], "close": []}
        for name, params in expected.items():
            with self.subTest(method=name):
                self.assertEqual(_params(getattr(GenEnv, name)), params)

    def test_episode_writer_members(self):
        """§3.3 EpisodeWriter.root and method signatures (keyword names are part of the contract)."""
        self.assertEqual(typing.get_type_hints(EpisodeWriter)["root"], Path)
        expected = {"episode_count": [], "episode_lengths": [], "discard_episode": [], "save_episode": [],
                    "finalize": [], "add_frame": ["action_raw", "state_raw", "images", "task", "labels"]}
        for name, params in expected.items():
            with self.subTest(method=name):
                self.assertEqual(_params(getattr(EpisodeWriter, name)), params)

    def test_expert_members(self):
        """§3.3 Expert.reset(spec, obs), act(obs, progress), properties phase and done."""
        self.assertEqual(_params(Expert.reset), ["spec", "obs"])
        self.assertEqual(_params(Expert.act), ["obs", "progress"])
        self.assertIsInstance(inspect.getattr_static(Expert, "phase"), property)
        self.assertIsInstance(inspect.getattr_static(Expert, "done"), property)

    def test_return_annotations(self):
        """§3.3 return types: ExpertObs from env reset/step/observe, ExpertAction from act, int from save."""
        env = typing.get_type_hints(GenEnv.step)
        self.assertIs(env["return"], protocols.ExpertObs)
        self.assertIs(typing.get_type_hints(GenEnv.reset)["return"], protocols.ExpertObs)
        self.assertIs(typing.get_type_hints(GenEnv.observe)["return"], protocols.ExpertObs)
        self.assertIs(typing.get_type_hints(Expert.act)["return"], protocols.ExpertAction)
        self.assertIs(typing.get_type_hints(EpisodeWriter.save_episode)["return"], int)
        self.assertIs(typing.get_type_hints(EpisodeWriter.episode_count)["return"], int)
        self.assertEqual(typing.get_type_hints(EpisodeWriter.episode_lengths)["return"], list[int])


class ProtocolConformanceTests(unittest.TestCase):
    """WP0 §3.3 isinstance checks."""

    def test_conforming_objects_pass_isinstance(self):
        """§3.3 structural implementations satisfy isinstance()."""
        self.assertIsInstance(_Env(), GenEnv)
        self.assertIsInstance(_Expert(), Expert)
        self.assertIsInstance(ListWriter(), EpisodeWriter)

    def test_non_conforming_objects_fail_isinstance(self):
        """§3.3 an object missing a protocol member is not an instance."""
        self.assertNotIsInstance(_EnvWithoutRender(), GenEnv)
        self.assertNotIsInstance(_ExpertWithoutAct(), Expert)
        self.assertNotIsInstance(object(), EpisodeWriter)
        self.assertNotIsInstance(_Expert(), GenEnv)

    def test_list_writer_follows_save_contract(self):
        """§3.3 save_episode returns the index == episode_count() before the call; discard drops the buffer."""
        writer = ListWriter()
        labels = None
        writer.discard_episode()
        for episode in range(3):
            for _ in range(episode + 1):
                writer.add_frame(np.zeros(2), np.zeros(2), {}, "t", labels)
            before = writer.episode_count()
            self.assertEqual(writer.save_episode(), before)
        writer.add_frame(np.zeros(2), np.zeros(2), {}, "t", labels)
        writer.discard_episode()
        self.assertEqual(writer.episode_lengths(), [1, 2, 3])


class ModuleHygieneTests(unittest.TestCase):
    """WP0 §1.6/§2 module layout and imports."""

    def test_package_init_is_empty(self):
        """§2 core/generate/__init__.py is empty (a docstring at most)."""
        tree = ast.parse((GENERATE_DIR / "__init__.py").read_text())
        body = [node for node in tree.body
                if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))]
        self.assertEqual(body, [])
        self.assertEqual(Path(core.generate.__file__).resolve(), (GENERATE_DIR / "__init__.py").resolve())

    def test_protocols_public_names(self):
        """§2 protocols.py defines exactly the contract's public classes (helpers are private)."""
        defined = {name for name, obj in vars(protocols).items()
                   if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))
                   and getattr(obj, "__module__", None) == protocols.__name__}
        self.assertEqual(defined, PROTOCOL_PUBLIC)

    def test_seeds_public_names(self):
        """§2 seeds.py defines exactly derive_seed and rng_for."""
        defined = {name for name, obj in vars(seeds).items()
                   if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))
                   and getattr(obj, "__module__", None) == seeds.__name__}
        self.assertEqual(defined, {"derive_seed", "rng_for"})

    def test_pose_is_the_existing_core_type(self):
        """§2 WP0 reuses core.tasks.base.Pose rather than defining its own."""
        from core.tasks.base import Pose
        self.assertNotIn("Pose", {n for n, o in vars(protocols).items()
                                  if inspect.isclass(o) and o.__module__ == protocols.__name__})
        self.assertIs(typing.get_type_hints(protocols.ObjectState)["pose"], Pose)

    def test_no_forbidden_imports(self):
        """§1.4/§1.6 WP0 modules import only stdlib, numpy and existing core (no Isaac, torch, random, yaml)."""
        for filename in WP0_FILES:
            tree = ast.parse((GENERATE_DIR / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    roots = {node.module.split(".")[0]}
                else:
                    continue
                with self.subTest(file=filename, line=node.lineno):
                    self.assertFalse(roots & FORBIDDEN_ROOTS, f"{filename}:{node.lineno} imports {roots}")
                    self.assertTrue(roots <= set(sys.stdlib_module_names) | {"numpy", "core"},
                                    f"{filename}:{node.lineno} imports {roots}")

    def test_no_global_numpy_random(self):
        """§1.4 only np.random.default_rng / Generator are used (no global np.random draws)."""
        for filename in WP0_FILES:
            tree = ast.parse((GENERATE_DIR / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) \
                        and node.value.attr == "random":
                    with self.subTest(file=filename, line=node.lineno):
                        self.assertIn(node.attr, {"default_rng", "Generator"})

    def test_python_311_syntax(self):
        """§1.6 the WP0 modules parse as Python 3.11."""
        for filename in WP0_FILES:
            with self.subTest(file=filename):
                ast.parse((GENERATE_DIR / filename).read_text(), feature_version=(3, 11))

    def test_import_pulls_no_heavy_modules(self):
        """§1.6 importing the WP0 modules loads no Isaac, torch, lerobot, pyarrow, scipy or yaml."""
        code = ("import sys; import core.generate.protocols, core.generate.errors, core.generate.seeds; "
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FORBIDDEN_ROOTS - {'random'})!r}))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(SRC_ROOT), "PATH": "/usr/bin:/bin"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_errors_module_is_importable_alone(self):
        """§1.5 errors.py has no project dependencies (importable from every package)."""
        tree = ast.parse((GENERATE_DIR / "errors.py").read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in imports:
            module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            self.assertNotEqual((module or "").split(".")[0], "core")
        self.assertTrue(issubclass(errors.GenerationError, Exception))


if __name__ == "__main__":
    unittest.main()
