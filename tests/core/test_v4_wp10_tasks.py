"""WP10: the cube tasks — ``pick_place_cube`` additions, ``stack_two_cubes`` and ``pick_place_color_cube``
(contracts §4, §6, §7, §14 decisions 7, 9, 18; owner decisions in docs/platform_v4.md).

The real task folders are copied into a sandbox and run on ``fx_arm5l`` (tests/fixtures/v4_wp10): fx_arm5 scaled
×2.2 and turned −90° about z, so that, like the SO-101 base frame, the arm faces −y and reaches the recorded
cube starts (x −0.3…0.33 m, y −0.42…0.04 m) from table height (z = 0). Grasp physics is not checked here
(container check: expert success ≥ 80 %).

Decisions pinned here (docs/v4/prompts/WP10.md):
* ``pick_place_cube`` keeps ``PickPlaceTask`` and its contact-sensor signals, gains the ``generate`` variant,
  ``[task.roles]``, ``[task.start]`` (box at its measured place), ``[task.stages].order`` (= ``STAGES``, no
  ``hold_until``), ``[task.failures]`` (reproducing the six names), ``[task.expert]`` and ``[task.perturbations]``;
  it loses its own ``sample_start`` and ``failure_mode``; ``reached`` measures from the TCP when the state has one
  (generation) and from the ee otherwise (eval, unchanged); ``observe`` no longer imports ``eval.py``.
* ``pick_place_color_cube``: 2 cm cubes (set ``cubes``, 2–6), 7-colour palette, hold-out colour ``purple`` and count
  4, a 4 cm-inner box ``small_box`` whose position and yaw are randomised, instruction names the target colour,
  ``wrong_object_placed`` when a distractor ends in the box.
* ``stack_two_cubes``: ``red_cube`` (target) stacked on ``blue_cube`` (role ``base``), success via
  ``object_on_object``.
"""

import ast
import contextlib
import math
import re
import unittest

import numpy as np

from core.generate.errors import SamplingError
from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
from core.generate.fake_env import FakeEnv
from core.generate.plan import attempt_seed
from core.generate.protocols import AttemptStatus, Kind
from core.tasks.base import Pose, StepState
from core.tasks.loader import load_task
from v4_support import REPO_ROOT, sandbox
from v4_support import harness as H
from v4_support.stubs import ListWriter

ARM = "fx_arm5l"
TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube")
PALETTE = {"red", "green", "blue", "yellow", "orange", "purple", "pink"}


class CubeTaskBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, TASKS)
        cls.rig = H.Rig(ARM)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def spec_fn(self, name):
        return make_spec_fn(load_task(name), ARM, scene=H.SCENE)

    def specs(self, name, kind=Kind.SUCCESS, n=50, start=0):
        fn, out = self.spec_fn(name), []
        for i in range(start, start + n):
            try:
                out.append(fn(kind, i, attempt_seed(0, name, kind, i)))
            except SamplingError:
                pass
        return out

    def run_spec(self, name, spec):
        task = load_task(name)
        return run_episode(spec, task, FakeEnv(task, self.rig.chain, self.rig.gripper),
                           make_expert_factory(task, ARM)(spec), ListWriter(), self.rig.mapping)[0]


class PickPlaceCubeTests(CubeTaskBase):
    """WP10 pick_place_cube additions (decisions 7, 18)."""

    def test_spec_tables(self):
        task = load_task("pick_place_cube")
        self.assertEqual(type(task).__name__, "PickPlaceTask")
        spec = task.spec
        self.assertTrue({"teleop", "eval", "eval_dr", "generate"} <= set(spec.variants))
        self.assertEqual(spec.variants["generate"], "env_cfg:apply_generate")
        self.assertEqual(dict(spec.roles), {"target": "blue_cube", "container": "white_box"})
        self.assertEqual(spec.raw["stages"]["order"], list(task.stages))
        self.assertEqual(task.hold_until, {})
        self.assertEqual(set(task.failure_names), {"never_reached", "missed_grasp", "dropped_before_box",
                                                   "dropped_outside", "placed_not_confirmed", "timeout_holding"})
        self.assertEqual(float(spec.raw["start"].get("table_z_m", 0.0)), 0.0)

    def test_class_overrides_removed(self):
        tree = ast.parse((REPO_ROOT / "tasks" / "pick_place_cube" / "__init__.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PickPlaceTask")
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse({"sample_start", "failure_mode"} & methods)
        env_cfg = ast.parse((REPO_ROOT / "tasks" / "pick_place_cube" / "env_cfg.py").read_text())
        self.assertIn("apply_generate", {n.name for n in env_cfg.body if isinstance(n, ast.FunctionDef)})

    def test_reached_uses_tcp_when_present(self):
        task = load_task("pick_place_cube")
        objs = {"blue_cube": Pose((0.0, -0.3, 0.01)), "white_box": Pose((-0.247, 0.006, 0.0))}
        far_ee, near_tcp = Pose((0.0, -0.3, 0.2)), Pose((0.0, -0.3, 0.015))
        self.assertTrue(task.observe(StepState(objs, far_ee, 0.03, {}, tcp=near_tcp))["reached"])
        self.assertFalse(task.observe(StepState(objs, far_ee, 0.03, {}))["reached"])
        self.assertTrue(task.observe(StepState(objs, near_tcp, 0.03, {}))["reached"])

    def test_over_box_without_the_default_arm(self):
        """observe with holding=True works in a sandbox that has no SO-101 arm (no eval.py import-time arm read)."""
        task = load_task("pick_place_cube")
        box = Pose((-0.247, 0.006, 0.0))
        inside = task.observe(StepState({"blue_cube": Pose((-0.25, 0.0, 0.1)), "white_box": box}, Pose((0, 0, 0.3)),
                                        0.02, {"holding": True}))
        outside = task.observe(StepState({"blue_cube": Pose((0.0, -0.3, 0.1)), "white_box": box}, Pose((0, 0, 0.3)),
                                         0.02, {"holding": True}))
        self.assertTrue(inside["over_box"])
        self.assertFalse(outside["over_box"])

    def test_starts(self):
        """WP14 §16.1-2: cube and box anywhere in the arm's workspace; the cube rests on the table at its drawn size."""
        from core.arms.spec import load_arm_spec
        from core.tasks.sampling import in_workspace
        specs = self.specs("pick_place_cube", n=40)
        self.assertGreaterEqual(len(specs), 35)
        workspace = load_arm_spec(ARM)[0]["workspace"]
        for s in specs:
            box = s.objects["white_box"].pose.position
            cube = s.objects["blue_cube"].pose.position
            with self.subTest(seed=s.seed):
                self.assertTrue(in_workspace(box[:2], workspace))
                self.assertTrue(in_workspace(cube[:2], workspace))
                self.assertAlmostEqual(cube[2], s.objects["blue_cube"].size[2] / 2, places=9)
                self.assertEqual(s.instruction, "Pick up the cube and place it in the box")
        xs = np.array([s.objects["blue_cube"].pose.position[0] for s in specs])
        bx = np.array([s.objects["white_box"].pose.position[0] for s in specs])
        self.assertGreater(bx.max() - bx.min(), 0.2, "the box moves too")
        self.assertGreater(xs.max() - xs.min(), 0.2)


class ColorCubeTests(CubeTaskBase):
    """WP10 pick_place_color_cube (owner decisions; contracts §4.5, §4.6, §14 decision 9)."""

    def test_tables(self):
        spec = load_task("pick_place_color_cube").spec
        self.assertEqual(type(load_task("pick_place_color_cube")).__name__, "GenericTask")
        self.assertEqual(set(spec.palette), PALETTE)
        self.assertEqual(tuple(spec.holdout.colors), ("purple",))
        self.assertEqual(tuple(spec.holdout.counts), (4,))
        cubes = spec.object_sets["cubes"]
        self.assertEqual(cubes.count_max, 6)
        self.assertEqual(tuple(cubes.template.size), (0.02, 0.02, 0.02))
        box = spec.objects["small_box"]
        self.assertEqual(tuple(box.params["inner_half_size"]), (0.02, 0.02))
        self.assertEqual(box.params.get("origin"), "bottom")
        self.assertEqual(dict(spec.roles), {"target": "cubes", "container": "small_box"})
        self.assertIn("wrong_object_placed", load_task("pick_place_color_cube").failure_names)
        self.assertTrue(any("{target.color}" in t for t in spec.instruction_templates))

    def test_box_asset(self):
        """The 4 cm box is a hand-written USDA whose extent matches size_outer with a bottom origin."""
        spec = load_task("pick_place_color_cube").spec
        box = spec.objects["small_box"]
        path = REPO_ROOT / "tasks" / "pick_place_color_cube" / spec.raw["objects"]["small_box"]["usd"]
        text = path.read_text()
        self.assertTrue(text.startswith("#usda 1.0"))
        self.assertRegex(text, r'defaultPrim\s*=\s*"\w+"')
        m = re.search(r"float3\[\]\s+extent\s*=\s*\[\(([^)]*)\),\s*\(([^)]*)\)\]", text)
        self.assertIsNotNone(m, "the root prim declares its extent")
        lo = [float(v) for v in m.group(1).split(",")]
        hi = [float(v) for v in m.group(2).split(",")]
        w, d, h = box.params["size_outer"]
        np.testing.assert_allclose(lo, [-w / 2, -d / 2, 0.0], atol=1e-9)
        np.testing.assert_allclose(hi, [w / 2, d / 2, h], atol=1e-9)
        inner = box.params["inner_half_size"]
        self.assertTrue(inner[0] < w / 2 and inner[1] < d / 2 and 0 < box.params["floor_z"] < h)

    def test_colour_binding_and_holdouts(self):
        specs = self.specs("pick_place_color_cube", n=200)
        self.assertGreaterEqual(len(specs), 180)
        counts, targets = set(), set()
        for s in specs:
            active = {n: o for n, o in s.objects.items() if o.active and n.startswith("cube_")}
            colors = [o.color for o in active.values()]
            target = s.roles["target"]
            with self.subTest(seed=s.seed):
                self.assertIn(target, active)
                self.assertNotIn("purple", colors)
                self.assertEqual(colors.count(active[target].color), 1)
                self.assertIn(active[target].color, s.instruction)
                self.assertNotEqual(len(active), 4)
                self.assertTrue(set(colors) <= PALETTE)
            counts.add(len(active))
            targets.add(active[target].color)
        self.assertEqual(counts, {2, 3, 5, 6})
        self.assertEqual(targets, PALETTE - {"purple"})

    def test_box_randomised(self):
        specs = self.specs("pick_place_color_cube", n=60)
        pos = np.array([s.objects["small_box"].pose.position[:2] for s in specs])
        self.assertGreater(np.ptp(pos[:, 0]), 0.03)
        self.assertGreater(np.ptp(pos[:, 1]), 0.03)

    def test_wrong_object_placed(self):
        fn = self.spec_fn("pick_place_color_cube")
        seen = 0
        for i in range(80):
            try:
                spec = fn(Kind.FAILURE, i, attempt_seed(0, "pick_place_color_cube", Kind.FAILURE, i))
            except SamplingError:
                continue
            if spec.perturbation.type != "wrong_target":
                continue
            r = self.run_spec("pick_place_color_cube", spec)
            if r.status is AttemptStatus.ACCEPTED:
                self.assertEqual(r.failure_mode, "wrong_object_placed")
                seen += 1
            if seen >= 2:
                break
        self.assertGreaterEqual(seen, 2)


class StackTests(CubeTaskBase):
    """WP10 stack_two_cubes."""

    def test_tables(self):
        task = load_task("stack_two_cubes")
        self.assertEqual(type(task).__name__, "GenericTask")
        self.assertEqual(dict(task.spec.roles), {"target": "red_cube", "base": "blue_cube"})
        checks = task.spec.raw["checks"]
        stacked = checks[task.stages[-2]]
        names = stacked.get("of", [task.stages[-2]])
        self.assertTrue(any(checks[n]["check"] == "object_on_object" for n in names if n in checks))
        for s in self.specs("stack_two_cubes", n=30):
            a, b = (np.array(s.objects[n].pose.position[:2]) for n in ("red_cube", "blue_cube"))
            self.assertGreaterEqual(np.linalg.norm(a - b), 0.05)


class SuccessRateTests(CubeTaskBase):
    """WP10: the scripted expert succeeds on ≥ 80 % of success attempts of each task in FakeEnv."""

    def test_success_rate(self):
        for name in TASKS:
            specs = [s for s in self.specs(name, n=14) if s.noise is None][:10]
            ok = sum(self.run_spec(name, s).status is AttemptStatus.ACCEPTED for s in specs)
            with self.subTest(task=name):
                self.assertGreaterEqual(ok, math.ceil(0.8 * len(specs)), f"{ok}/{len(specs)}")


if __name__ == "__main__":
    unittest.main()
