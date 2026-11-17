"""WP13: ``stack_bowls`` — put the small bowl into the large bowl (contracts §4.4 ``nested_in``, §4.6, §14 open
point "bowl geometry"; docs/v4/prompts/WP13.md).

Proposed geometry (owner to confirm): a small bowl 4 cm across and 2.5 cm high, narrow enough for the gripper to
close on its body, nested into a large bowl 8 cm across and 3.5 cm high. Both are procedural USDA assets with a
bottom origin. Run on ``fx_arm5l`` (SO-101 base-frame conventions, see test_v4_wp10_tasks.py). Concave collision
and grasp physics are container checks.
"""

import contextlib
import math
import re
import unittest

import numpy as np

from core.arms.gripper import Gripper
from core.generate.errors import SamplingError
from core.generate.fake_env import FakeEnv
from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
from core.generate.plan import attempt_seed
from core.generate.protocols import AttemptStatus, Kind
from core.tasks.loader import load_task
from v4_support import REPO_ROOT, sandbox
from v4_support import harness as H
from v4_support.stubs import ListWriter

ARM = "fx_arm5l"
NAME = "stack_bowls"


class StackBowlsTests(unittest.TestCase):
    """WP13 stack_bowls task, assets and expert."""

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, (NAME,))
        cls.rig = H.Rig(ARM)
        cls.task = load_task(NAME)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def specs(self, kind=Kind.SUCCESS, n=40):
        fn, out = make_spec_fn(self.task, ARM, scene=H.SCENE), []
        for i in range(n):
            try:
                out.append(fn(kind, i, attempt_seed(0, NAME, kind, i)))
            except SamplingError:
                pass
        return out

    def test_tables(self):
        spec = self.task.spec
        self.assertEqual(type(self.task).__name__, "GenericTask")
        self.assertEqual(dict(spec.roles), {"target": "small_bowl", "container": "large_bowl"})
        checks = spec.raw["checks"]
        self.assertTrue(any(c["check"] == "nested_in" and c.get("object") == "@target" and c.get("base") == "@container"
                            for c in checks.values()))
        small, large = spec.objects["small_bowl"], spec.objects["large_bowl"]
        for o in (small, large):
            self.assertEqual(o.params.get("origin"), "bottom")
        sw = min(small.params["size_outer"][:2])
        self.assertLessEqual(sw, Gripper.from_arm(ARM).max_object_width_m, "the gripper must close on the small bowl")
        inner = large.params["inner_half_size"]
        self.assertGreaterEqual(min(inner), sw / 2 + 0.005, "the small bowl must fit inside the large one")
        self.assertIn("generate", spec.variants)

    def test_assets(self):
        for name in ("small_bowl", "large_bowl"):
            o = self.task.spec.objects[name]
            path = REPO_ROOT / "tasks" / NAME / self.task.spec.raw["objects"][name]["usd"]
            text = path.read_text()
            with self.subTest(object=name):
                self.assertTrue(text.startswith("#usda 1.0"))
                self.assertRegex(text, r'defaultPrim\s*=\s*"\w+"')
                m = re.search(r"float3\[\]\s+extent\s*=\s*\[\(([^)]*)\),\s*\(([^)]*)\)\]", text)
                self.assertIsNotNone(m)
                w, d, h = o.params["size_outer"]
                np.testing.assert_allclose([float(v) for v in m.group(1).split(",")], [-w / 2, -d / 2, 0], atol=1e-9)
                np.testing.assert_allclose([float(v) for v in m.group(2).split(",")], [w / 2, d / 2, h], atol=1e-9)

    def test_starts(self):
        specs = self.specs()
        self.assertGreaterEqual(len(specs), 35)
        sep = self.task.spec.raw["start"]["min_separation_m"]
        for s in specs:
            a = np.array(s.objects["small_bowl"].pose.position)
            b = np.array(s.objects["large_bowl"].pose.position)
            with self.subTest(seed=s.seed):
                self.assertGreaterEqual(np.linalg.norm(a[:2] - b[:2]), sep - 1e-9)
                self.assertAlmostEqual(a[2], 0.0, places=9)
                self.assertAlmostEqual(b[2], 0.0, places=9)

    def test_success_rate(self):
        specs = [s for s in self.specs(n=14) if s.noise is None][:10]
        fac = make_expert_factory(self.task, ARM)
        ok = 0
        for s in specs:
            r, _ = run_episode(s, self.task, FakeEnv(self.task, self.rig.chain, self.rig.gripper), fac(s), ListWriter(),
                               self.rig.mapping)
            ok += r.status is AttemptStatus.ACCEPTED
        self.assertGreaterEqual(ok, math.ceil(0.8 * len(specs)), f"{ok}/{len(specs)}")


if __name__ == "__main__":
    unittest.main()
