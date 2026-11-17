"""WP7: private helpers shared by the WP7 test files (imported as ``from test_v4_wp7_support import ...``; pytest
puts ``tests/core`` on ``sys.path`` and basenames are unique, contracts §13). The checks at the bottom test the
helpers themselves.

Fixture tasks (``tests/fixtures/v4_wp7/tasks``) run on the fx arms inside a sandbox:

* ``fx7_grip``  joint sweep then a gripper gap (no IK; ~30 frames): the fast task for loop mechanics.
* ``fx7_reach`` ``fx_reach`` with the point region inside the fx workspace.
* ``fx7_pick``  ``fx_pick`` with the table at z = -0.05 m (the fx arms reach a down-pointing TCP only for
  z ≈ -0.06…0.10 m) and the start regions inside the workspace.
* ``fx7_color`` ``fx_color`` likewise (2–6 cubes, palette, hold-out ``purple`` and count 4).
"""

import contextlib
import json
import math
import shutil
import unittest
from pathlib import Path

import numpy as np

from v4_support import FIXTURES_DIR, sandbox
from v4_support import harness as H
from v4_support.tasks import write_task

ARM = "fx_arm5"
FAST = "fx7_grip"
ALL_TASKS = ("fx7_grip", "fx7_reach", "fx7_pick", "fx7_color")
WP7_TASKS = FIXTURES_DIR / "v4_wp7" / "tasks"
CHECKS = ("sidecars_valid", "counts_match", "labels_match", "fingerprints_recomputed", "episode_fp_unique",
          "traj_fp_unique_originals", "scenario_key_unique_originals", "holdout_absent", "dr_sources_exist",
          "dr_appearance_distinct")
HEAVY = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch", "huggingface_hub", "boto3",
         "fsspec", "sim")


def tree_bytes(root):
    """{posix relative path: bytes} of every file under ``root``."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def manifest_dicts(root, *, drop=("wall_s",)):
    path = Path(root) / "meta" / "generation_manifest.jsonl"
    if not path.exists():
        return []
    return [{k: v for k, v in json.loads(line).items() if k not in drop}
            for line in path.read_text().splitlines() if line.strip()]


def sidecar_dicts(root):
    from core.datasets.task_meta import list_sidecars, read_sidecar
    return {i: read_sidecar(root, i) for i in list_sidecars(root)}


def task_table(name):
    """The parsed ``task.toml`` table of a WP7 fixture task (to derive variants with ``write_task``)."""
    import tomllib
    return tomllib.loads((WP7_TASKS / name / "task.toml").read_text())["task"]


def write_variant(box, name, base, **changes):
    """Write task ``name`` = fixture ``base`` with top-level ``[task]`` keys replaced by ``changes``."""
    from v4_support import clear_task_cache
    table = task_table(base)
    table.update(changes)
    table["name"] = name
    files = {}
    usd = WP7_TASKS / base / "usd"
    if usd.is_dir():
        files = {f"usd/{p.name}": p.read_text() for p in usd.iterdir()}
    write_task(box.tasks, name, table, files=files)
    clear_task_cache(name)
    return name


class SandboxCase(unittest.TestCase):
    """A sandbox with both fx arms, ``fx_scene`` and the WP7 fixture tasks, alive for the whole class."""

    TASKS = ALL_TASKS

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, cls.TASKS)
        cls.rig = H.Rig(ARM)
        cls.work = cls.box.root / "work"
        cls.work.mkdir()
        cls._n = 0

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def fresh(self, name="run"):
        """A new empty directory under the sandbox."""
        type(self)._n += 1
        path = self.work / f"{name}_{type(self)._n}"
        path.mkdir()
        return path

    def config(self, task=FAST, *, shard="0/1", quotas=None, **kw):
        d = self.fresh("cfg")
        return H.write_config(d / "cfg.yaml", tasks=[task], dst_root=d / "data", shard=shard,
                              quotas=quotas or {"success": 3, "failure": 3, "recovery": 3}, **kw)

    def task(self, name):
        from core.tasks.loader import load_task
        return load_task(name)

    def spec(self, task_name, kind, index=0, *, master_seed=0):
        """The EpisodeSpec the plan gives attempt (kind, index); raises SamplingError like ``spec_fn``."""
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        from core.generate.protocols import Kind
        kind = Kind(kind)
        fn = make_spec_fn(self.task(task_name), ARM, scene=H.SCENE)
        return fn(kind, index, attempt_seed(master_seed, task_name, kind, index))

    def env(self, task_name, **kw):
        from core.generate.fake_env import FakeEnv
        return FakeEnv(self.task(task_name), self.rig.chain, self.rig.gripper, **kw)

    def expert(self, task_name, spec):
        from core.generate.loop import make_expert_factory
        return make_expert_factory(self.task(task_name), ARM)(spec)

    def writer(self, root=None, **kw):
        from core.datasets.frames import MemoryWriter
        return MemoryWriter(root or self.fresh("w"), fps=30, joint_names=tuple(self.rig.chain.q_names), **kw)

    def first_spec(self, task_name, kind, *, start=0, accept=None):
        """The first attempt index >= start whose spec samples (and passes ``accept(spec)``)."""
        from core.generate.errors import SamplingError
        for i in range(start, start + 200):
            try:
                s = self.spec(task_name, kind, i)
            except SamplingError:
                continue
            if accept is None or accept(s):
                return s
        raise AssertionError("no spec")


class RecordingEnv:
    """Wraps a GenEnv and records every ``step`` target and ``set_object_pose`` call."""

    def __init__(self, env):
        self._env = env
        self.targets = []
        self.teleports = []
        self.fps, self.joint_names, self.camera_names = env.fps, env.joint_names, env.camera_names

    def reset(self, spec):
        return self._env.reset(spec)

    def step(self, q):
        self.targets.append(np.array(q, dtype=np.float64))
        return self._env.step(q)

    def observe(self):
        return self._env.observe()

    def set_object_pose(self, name, pose):
        self.teleports.append((len(self.targets), name, pose))
        return self._env.set_object_pose(name, pose)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


class FailingEnv(RecordingEnv):
    """Raises ``exc`` on step number ``at`` (0-based)."""

    def __init__(self, env, at, exc):
        super().__init__(env)
        self.at, self.exc = at, exc

    def step(self, q):
        if len(self.targets) == self.at:
            self.targets.append(np.array(q))
            raise self.exc
        return super().step(q)


class NaNEnv(RecordingEnv):
    """Returns an observation with a NaN joint after step number ``at``."""

    def __init__(self, env, at):
        super().__init__(env)
        self.at = at

    def step(self, q):
        obs = super().step(q)
        if len(self.targets) == self.at:
            import dataclasses
            bad = np.array(obs.q)
            bad[0] = np.nan
            obs = dataclasses.replace(obs, q=bad)
        return obs


class ConstantExpert:
    """An Expert that ignores the spec: holds ``q`` and the given gap angle, done after ``frames`` frames."""

    def __init__(self, q, *, frames=8):
        self.q = np.array(q, dtype=np.float64)
        self.frames = frames
        self._t = 0
        self._ready = False

    @property
    def phase(self):
        return "hold"

    @property
    def done(self):
        return self._t >= self.frames

    def reset(self, spec, obs):
        self._t = 0
        self._ready = True

    def act(self, obs, progress):
        from core.generate.protocols import ExpertAction
        if not self._ready:
            raise RuntimeError("act before reset")
        self._t += 1
        return ExpertAction(q_target=self.q.copy(), phase="hold", done=self.done, injected=False, reentered=False,
                            resumed_stage=None)


def import_clean(test, modules, forbidden=HEAVY):
    """Import ``modules`` in a fresh interpreter and assert none of ``forbidden`` got imported."""
    import subprocess
    import sys
    from v4_support import SRC_DIR, TESTS_DIR
    code = ("import sys, json\n"
            f"for m in {list(modules)!r}: __import__(m)\n"
            f"bad = sorted(n for n in sys.modules if n.split('.')[0] in {list(forbidden)!r})\n"
            "print(json.dumps(bad))\n")
    env = dict(__import__("os").environ, PYTHONPATH=f"{SRC_DIR}:{TESTS_DIR}")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    test.assertEqual(out.returncode, 0, out.stderr[-2000:])
    test.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), [])


class SupportSelfTests(SandboxCase):
    """WP7 test support: the fixture tasks load and the derived configs parse (contracts §8.5, §13)."""

    TASKS = ALL_TASKS

    def test_fixture_tasks_load(self):
        for name in ALL_TASKS:
            with self.subTest(task=name):
                t = self.task(name)
                self.assertEqual(t.name, name)
                self.assertIn("expert", t.spec.raw)

    def test_config_parses(self):
        from core.generate.config import load_generate_config
        cfg = load_generate_config(self.config(shard="1/3"))
        self.assertEqual(tuple(cfg.shard), (1, 3))
        self.assertEqual(cfg.tasks, (FAST,))

    def test_harness_imports_without_v4_modules(self):
        """``v4_support.harness`` imports only stdlib + numpy at module level (v4 imports are lazy)."""
        import ast
        tree = ast.parse(Path(H.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) or "" for n in top}
        self.assertFalse({m for m in names if m.split(".")[0] in ("core", "sim")}, names)
