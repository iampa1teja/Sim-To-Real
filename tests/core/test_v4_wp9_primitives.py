"""WP9: the 13 primitive tasks (owner-approved meanings; docs/v4/prompts/WP9.md).

Every primitive is a ``GenericTask`` folder under ``tasks/`` with ``teleop`` and ``generate`` variants, a scripted
expert program and failure and recovery perturbations. Run on ``fx_arm5l`` (SO-101 base-frame conventions, see
test_v4_wp10_tasks.py). Each test checks what the primitive means on successful episodes in FakeEnv:

=========================  ==========================================================================================
prim_joint_sweep           one arm joint passes 3 random angles in order, then the gripper is set to a random width
prim_gripper_gap           at a random point, the gripper opens to a random width
prim_reach_point           the TCP goes to a random free-space point
prim_test_reach            the TCP goes 4 cm beside the cube without moving it
prim_calibration_reach     the TCP visits three points of a fixed 9-point calibration grid, in order
prim_waypoint_path         the TCP passes 3 random waypoints (≥ 6 cm apart) in order
prim_touch_object          the TCP touches the cube's top and withdraws; the cube does not move
prim_nudge                 the cube is pushed 0.5–5 cm into an area just in front of it
prim_push_to_region        the cube is pushed into a random target area
prim_lift_test             the cube is grasped, raised about 1 cm and set down on the table
prim_grasp_lift            the cube is grasped and held at least 6 cm up
prim_place_at              the cube is picked up and set down inside a random target spot
prim_regrasp_after_drop    the cube is lifted, let go, and grasped and lifted again
=========================  ==========================================================================================

Two meanings were adapted so every primitive has enough distinct start scenarios for the no-duplicates rule
(a fixed start has one scenario): ``calibration_reach`` draws its three points from the fixed grid, and
``gripper_gap`` sets the width at a random reach point.
"""

import contextlib
import math
import re
import unittest

import numpy as np

from core.generate.errors import SamplingError
from core.generate.fake_env import FakeEnv
from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
from core.generate.plan import attempt_seed
from core.generate.protocols import AttemptStatus, Kind
from core.tasks.loader import load_task
from v4_support import sandbox
from v4_support import harness as H
from v4_support.stubs import ListWriter

ARM = "fx_arm5l"
PRIMITIVES = ("prim_joint_sweep", "prim_gripper_gap", "prim_reach_point", "prim_test_reach", "prim_calibration_reach",
              "prim_waypoint_path", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
              "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")
OBJECT_PRIMS = ("prim_test_reach", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
                "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")


class PrimitiveBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, PRIMITIVES)
        cls.rig = H.Rig(ARM)
        cls.episodes = {}

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def specs(self, name, kind=Kind.SUCCESS, n=10):
        fn, out = make_spec_fn(load_task(name), ARM, scene=H.SCENE), []
        for i in range(n + 10):
            try:
                s = fn(kind, i, attempt_seed(0, name, kind, i))
            except SamplingError:
                continue
            if s.noise is None:
                out.append(s)
            if len(out) == n:
                break
        return out

    def run_specs(self, name):
        """(spec, result, records, env) for 10 success specs, cached per task."""
        if name not in self.episodes:
            task = load_task(name)
            fac = make_expert_factory(task, ARM)
            out = []
            for s in self.specs(name):
                env = FakeEnv(task, self.rig.chain, self.rig.gripper)
                r, rec = run_episode(s, task, env, fac(s), ListWriter(), self.rig.mapping)
                out.append((s, r, rec, env))
            type(self).episodes[name] = out
        return self.episodes[name]

    def accepted(self, name):
        return [(s, r, rec, env) for s, r, rec, env in self.run_specs(name) if r.status is AttemptStatus.ACCEPTED]


class StructureTests(PrimitiveBase):
    """WP9 every primitive folder loads, has the two variants, a valid program and both perturbation lists."""

    def test_structure(self):
        for name in PRIMITIVES:
            task = load_task(name)
            with self.subTest(task=name):
                self.assertEqual(type(task).__name__, "GenericTask")
                self.assertEqual(task.name, name)
                self.assertEqual(set(task.spec.variants), {"teleop", "generate"})
                make_expert_factory(task, ARM)
                pert = task.spec.raw["perturbations"]
                self.assertTrue(pert.get("failure") and pert.get("recovery"))
                self.assertTrue(task.failure_names)
                self.assertEqual(task.stages[-1], "success")
                for text in task.spec.instruction_templates or (task.spec.instruction,):
                    self.assertNotRegex(text, r"\{goal\.")
                if name in OBJECT_PRIMS:
                    self.assertEqual(task.spec.roles.get("target"), "cube")
                    self.assertEqual(tuple(task.spec.objects["cube"].size), (0.02, 0.02, 0.02))

    def test_success_rate(self):
        for name in PRIMITIVES:
            runs = self.run_specs(name)
            ok = len(self.accepted(name))
            with self.subTest(task=name):
                self.assertEqual(len(runs), 10)
                self.assertGreaterEqual(ok, 9, f"{name}: {ok}/10")


def tcp_path(rec, env, chain, mapping):
    return np.array([chain.fk(mapping.to_radians(r.state_raw), "tcp")[:3, 3] for r in rec])


class MeaningTests(PrimitiveBase):
    """WP9 the observed behaviour of successful episodes matches each primitive's meaning."""

    def tcp(self, rec):
        return tcp_path(rec, None, self.rig.chain, self.rig.mapping)

    def cube(self, rec):
        return np.array([r.objects["cube"].position for r in rec])

    def test_joint_sweep(self):
        names = list(self.rig.chain.q_names)
        for s, r, rec, env in self.accepted("prim_joint_sweep"):
            goal = s.goal["sweep"]
            j = names.index(goal["joint"])
            q = np.array([self.rig.mapping.to_radians(x.state_raw)[j] for x in rec])
            self.assertEqual(len(goal["angles"]), 3)
            t = -1
            for a in goal["angles"]:
                hits = np.nonzero(np.abs(q[t + 1:] - a) <= 0.03)[0]
                self.assertTrue(len(hits), f"angle {a} never reached after frame {t}")
                t = t + 1 + int(hits[0])
            self.assertLess(abs(env.observe().gripper_gap_m - s.goal["gap"]), 0.004)

    def test_gripper_gap(self):
        for s, r, rec, env in self.accepted("prim_gripper_gap"):
            self.assertLess(abs(env.observe().gripper_gap_m - s.goal["gap"]), 0.004)
            self.assertLess(np.linalg.norm(np.array(env.observe().tcp.position) - s.goal["point"]), 0.015)

    def test_reach_point(self):
        for s, r, rec, env in self.accepted("prim_reach_point"):
            self.assertLess(np.linalg.norm(np.array(env.observe().tcp.position) - s.goal["point"]), 0.012)

    def test_test_reach(self):
        for s, r, rec, env in self.accepted("prim_test_reach"):
            c = self.cube(rec)
            self.assertLess(np.max(np.linalg.norm(c - c[0], axis=1)), 0.002, "the cube must not move")
            tcp = np.array(env.observe().tcp.position)
            self.assertGreater(np.linalg.norm(tcp[:2] - c[-1][:2]), 0.025)
            self.assertLess(np.linalg.norm(tcp[:2] - c[-1][:2]), 0.055)

    def test_calibration_reach(self):
        grids = []
        for s, r, rec, env in self.accepted("prim_calibration_reach"):
            pts = [np.array(s.goal[f"c{i}"]) for i in range(3)]
            grids += [tuple(np.round(p, 6)) for p in pts]
            path, t = self.tcp(rec), -1
            for p in pts:
                hits = np.nonzero(np.linalg.norm(path[t + 1:] - p, axis=1) <= 0.012)[0]
                self.assertTrue(len(hits))
                t = t + 1 + int(hits[0])
        self.assertLessEqual(len(set(grids)), 9, "points come from one fixed grid")

    def test_waypoint_path(self):
        for s, r, rec, env in self.accepted("prim_waypoint_path"):
            pts = np.array(s.goal["points"])
            self.assertEqual(pts.shape, (3, 3))
            for a in range(3):
                for b in range(a + 1, 3):
                    self.assertGreaterEqual(np.linalg.norm(pts[a] - pts[b]), 0.06 - 1e-9)
            path, t = self.tcp(rec), -1
            for p in pts:
                hits = np.nonzero(np.linalg.norm(path[t + 1:] - p, axis=1) <= 0.012)[0]
                self.assertTrue(len(hits))
                t = t + 1 + int(hits[0])

    def test_touch_object(self):
        for s, r, rec, env in self.accepted("prim_touch_object"):
            c, path = self.cube(rec), self.tcp(rec)
            self.assertLess(np.max(np.linalg.norm(c - c[0], axis=1)), 0.005)
            top = c[0] + [0, 0, 0.01]
            self.assertLess(np.min(np.linalg.norm(path - top, axis=1)), 0.008)
            self.assertGreater(np.linalg.norm(path[-1] - top), 0.04)

    def in_area(self, xy, area, margin=0.0):
        return all(area["xy_min"][k] - margin <= xy[k] <= area["xy_max"][k] + margin for k in range(2))

    def test_nudge(self):
        for s, r, rec, env in self.accepted("prim_nudge"):
            c = self.cube(rec)
            d = np.linalg.norm(c[-1][:2] - c[0][:2])
            self.assertTrue(0.005 <= d <= 0.05, d)
            self.assertTrue(self.in_area(c[-1], s.goal["area"]))

    def test_push_to_region(self):
        for s, r, rec, env in self.accepted("prim_push_to_region"):
            c = self.cube(rec)
            self.assertFalse(self.in_area(c[0], s.goal["area"]))
            self.assertTrue(self.in_area(c[-1], s.goal["area"]))
            self.assertLess(np.max(c[:, 2]) - c[0][2], 0.002, "pushed, not lifted")

    def test_lift_test(self):
        for s, r, rec, env in self.accepted("prim_lift_test"):
            c = self.cube(rec)
            rise = np.max(c[:, 2]) - c[0][2]
            self.assertTrue(0.006 <= rise <= 0.03, rise)
            self.assertAlmostEqual(c[-1][2], c[0][2], places=6)
            self.assertFalse(env.observe().signals["holding"])

    def test_grasp_lift(self):
        for s, r, rec, env in self.accepted("prim_grasp_lift"):
            self.assertTrue(env.observe().signals["holding"])
            self.assertGreaterEqual(self.cube(rec)[-1][2] - self.cube(rec)[0][2], 0.06)

    def test_place_at(self):
        for s, r, rec, env in self.accepted("prim_place_at"):
            c = self.cube(rec)
            self.assertFalse(self.in_area(c[0], s.goal["spot"]))
            self.assertTrue(self.in_area(c[-1], s.goal["spot"]))
            self.assertAlmostEqual(c[-1][2], c[0][2], places=6)
            self.assertFalse(env.observe().signals["holding"])

    def test_regrasp_after_drop(self):
        for s, r, rec, env in self.accepted("prim_regrasp_after_drop"):
            c = self.cube(rec)
            z = c[:, 2] - c[0][2]
            up = np.nonzero(z > 0.015)[0]
            self.assertTrue(len(up))
            down = np.nonzero(np.abs(z[up[0]:]) < 1e-6)[0]
            self.assertTrue(len(down), "the cube is let go and lands")
            again = np.nonzero(z[up[0] + down[0]:] > 0.05)[0]
            self.assertTrue(len(again), "and is lifted again")
            self.assertTrue(env.observe().signals["holding"])


if __name__ == "__main__":
    unittest.main()
