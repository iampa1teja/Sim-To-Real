"""WP3: motion helpers — MotionLimits, step_toward_joint, cartesian_waypoint, expert_joint_limits (contracts §5.1, §5.4, §6.2).

``expert_joint_limits`` is the public name WP3 pins for the "expert joint limits" left open in contracts
§15 (WP2): ``core.expert.motion.expert_joint_limits(arm=None, *, setup=None) -> (lower, upper)`` =
``[max(runtime, USD), min(runtime, USD)]`` per joint, radians, joint-vector order.
"""

import math
import unittest

import numpy as np

from core.arms.kinematics import KinematicChain
from core.arms.raw_mapping import RawMapping
from core.expert.motion import MotionLimits, cartesian_waypoint, expert_joint_limits, step_toward_joint
from core.generate.errors import ConfigError
from v4_support import assert_frozen, copy_fixture, sandbox
from v4_support.fx_arms import FX_ARMS, fx_arm_table, fx_joint_names, fx_kinematics, fx_limits, fx_setup, write_arm
from v4_support.tasks import write_scene


def _fx_box(box):
    for arm in FX_ARMS:
        copy_fixture(f"arms/{arm}", box.root)
    copy_fixture("scenes/fx_scene", box.root)


class MotionLimitsTests(unittest.TestCase):
    """WP3 §6.2 MotionLimits / §5.1 [arm.expert] speed keys."""

    def test_from_arm_fx(self):
        """§5.1 scalar max_joint_speed_rad_s applies to every non-gripper joint; the gripper entry of the (J,)
        array is max_gripper_speed_rad_s (decision); ee speed copied."""
        with sandbox() as box:
            _fx_box(box)
            for arm in FX_ARMS:
                with self.subTest(arm=arm):
                    lim = MotionLimits.from_arm(arm)
                    names = fx_joint_names(arm)
                    grip = fx_arm_table(arm)["gripper"]["joint"]
                    expected = np.array([3.0 if n == grip else 1.5 for n in names])
                    self.assertEqual(lim.max_joint_speed_rad_s.dtype, np.float64)
                    np.testing.assert_array_equal(lim.max_joint_speed_rad_s, expected)
                    self.assertEqual(lim.max_gripper_speed_rad_s, 3.0)
                    self.assertEqual(lim.max_ee_speed_m_s, 0.15)
                    assert_frozen(lim)

    def test_from_arm_per_joint_table(self):
        """§5.1 max_joint_speed_rad_s may be a {joint = float} table; values land at the joint's index."""
        with sandbox() as box:
            table = fx_arm_table("fx_arm5")
            names = fx_joint_names("fx_arm5")
            speeds = {n: 0.5 + 0.25 * i for i, n in enumerate(names[:-1])}
            table["expert"]["max_joint_speed_rad_s"] = speeds
            write_arm(box.arms, "fx_wp3_speeds", table, kinematics=fx_kinematics("fx_arm5"))
            lim = MotionLimits.from_arm("fx_wp3_speeds")
            np.testing.assert_array_equal(lim.max_joint_speed_rad_s, [speeds[n] for n in names[:-1]] + [3.0])

    def test_from_arm_missing_values(self):
        """§1.5 a missing [arm.expert] table, a missing speed key, or a table without every arm joint -> ConfigError."""
        names = fx_joint_names("fx_arm5")
        cases = {}
        t = fx_arm_table("fx_arm5")
        del t["expert"]
        cases["no_expert"] = t
        for key in ("max_joint_speed_rad_s", "max_gripper_speed_rad_s", "max_ee_speed_m_s"):
            t = fx_arm_table("fx_arm5")
            del t["expert"][key]
            cases[f"no_{key}"] = t
        t = fx_arm_table("fx_arm5")
        t["expert"]["max_joint_speed_rad_s"] = {n: 1.0 for n in names[:3]}
        cases["partial_table"] = t
        with sandbox() as box:
            for i, (label, table) in enumerate(cases.items()):
                with self.subTest(case=label):
                    write_arm(box.arms, f"fx_wp3_bad{i}", table, kinematics=fx_kinematics("fx_arm5"))
                    with self.assertRaises(ConfigError):
                        MotionLimits.from_arm(f"fx_wp3_bad{i}")

    def test_direct_construction(self):
        """§1.3 the speed array is stored as a read-only float64 copy; scalar fields are floats."""
        speeds = [1.0, 2.0, 3.0]
        lim = MotionLimits(max_joint_speed_rad_s=speeds, max_gripper_speed_rad_s=2, max_ee_speed_m_s=0.1)
        speeds[0] = 9.0
        np.testing.assert_array_equal(lim.max_joint_speed_rad_s, [1.0, 2.0, 3.0])
        self.assertIsInstance(lim.max_gripper_speed_rad_s, float)
        assert_frozen(lim)


class StepTowardJointTests(unittest.TestCase):
    """WP3 §6.2 per-joint speed limit helper."""

    def test_property(self):
        """§6.2 |out − current| ≤ max_step per joint; out == target when within reach; else exactly max_step toward it."""
        for k in range(40):
            rng = np.random.default_rng(k)
            n = int(rng.integers(1, 8))
            cur, tgt = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
            step = rng.uniform(0.0, 1.0, n) if k % 2 else float(rng.uniform(0.0, 1.0))
            cur0, tgt0 = cur.copy(), tgt.copy()
            with self.subTest(seed=k):
                out = step_toward_joint(cur, tgt, step)
                s = np.broadcast_to(step, (n,))
                self.assertEqual(out.shape, (n,))
                self.assertEqual(out.dtype, np.float64)
                self.assertTrue(np.all(np.abs(out - cur) <= s + 1e-12))
                near = np.abs(tgt - cur) <= s
                np.testing.assert_allclose(out[near], tgt[near], atol=1e-12)
                far = ~near
                np.testing.assert_allclose(out[far], cur[far] + np.sign(tgt[far] - cur[far]) * s[far], atol=1e-12)
                np.testing.assert_array_equal(cur, cur0)
                np.testing.assert_array_equal(tgt, tgt0)

    def test_shape_mismatch(self):
        """§1.6 wrong argument shapes are ValueError."""
        with self.assertRaises(ValueError):
            step_toward_joint(np.zeros(3), np.zeros(4), 0.1)


class CartesianWaypointTests(unittest.TestCase):
    """WP3 §6.2 the Cartesian goal advances ≤ max_ee_speed_m_s / fps per step."""

    def test_property(self):
        """§6.2 straight-line step of at most max_step toward the goal; the goal itself once within reach."""
        for k in range(40):
            rng = np.random.default_rng(100 + k)
            cur, goal = rng.uniform(-0.5, 0.5, 3), rng.uniform(-0.5, 0.5, 3)
            step = float(rng.uniform(0.001, 0.2))
            with self.subTest(seed=k):
                out = cartesian_waypoint(cur, goal, step)
                self.assertEqual(out.shape, (3,))
                d = np.linalg.norm(goal - cur)
                self.assertLessEqual(np.linalg.norm(out - cur), step + 1e-12)
                if d <= step:
                    np.testing.assert_allclose(out, goal, atol=1e-12)
                else:
                    self.assertAlmostEqual(float(np.linalg.norm(out - cur)), step, places=12)
                    # collinear with the segment
                    self.assertLess(np.linalg.norm(np.cross(out - cur, goal - cur)), 1e-12)
                    self.assertLess(np.linalg.norm(goal - out), d)

    def test_reaches_goal_in_ceil_steps(self):
        """§6.2 repeated waypoints arrive after exactly ceil(distance / step) steps."""
        cur, goal, step = np.array([0.1, 0.0, 0.2]), np.array([0.2, 0.05, 0.05]), 0.005
        n = math.ceil(np.linalg.norm(goal - cur) / step)
        for i in range(n):
            self.assertFalse(np.allclose(cur, goal))
            cur = cartesian_waypoint(cur, goal, step)
        np.testing.assert_allclose(cur, goal, atol=1e-12)

    def test_bad_shape(self):
        """§1.6 positions must be (3,)."""
        with self.assertRaises(ValueError):
            cartesian_waypoint(np.zeros(2), np.zeros(3), 0.1)


class ExpertJointLimitsTests(unittest.TestCase):
    """WP3 resolves contracts §15 (WP2 open point): ``expert_joint_limits`` = runtime ∩ USD (§5.4)."""

    def test_fx_scene(self):
        """§5.4 per joint [max(runtime, USD), min(runtime, USD)] in rad, joint-vector order, float64 (J,)."""
        with sandbox() as box:
            _fx_box(box)
            for arm in FX_ARMS:
                with self.subTest(arm=arm):
                    lo, hi = expert_joint_limits(arm)
                    raw = RawMapping.from_setup(arm=arm)
                    usd_lo, usd_hi = fx_limits(arm)
                    self.assertEqual(lo.shape, (len(fx_joint_names(arm)),))
                    self.assertEqual(lo.dtype, np.float64)
                    np.testing.assert_allclose(lo, np.maximum(raw.lower_rad, usd_lo), atol=1e-12)
                    np.testing.assert_allclose(hi, np.minimum(raw.upper_rad, usd_hi), atol=1e-12)
                    self.assertTrue(np.all(lo < hi))
                    # fx_scene runtime limits are the USD range shrunk to 95 %: strictly inside
                    self.assertTrue(np.all(lo > usd_lo) and np.all(hi < usd_hi))

    def test_intersection_takes_each_side_separately(self):
        """§5.4 a runtime bound beyond the USD limit is replaced by the USD limit, per side and per joint."""
        with sandbox() as box:
            copy_fixture("arms/fx_arm6", box.root)
            setup = fx_setup(["fx_arm6"])
            block = setup["arms"]["fx_arm6"]["runtime_joint_limits"]
            first, last = list(block)[0], list(block)[-2]
            block[first]["joint_min"] = -300.0       # beyond USD -2.8 rad (-160.4 deg)
            block[last]["joint_max"] = 300.0
            write_scene(box.scene, setup)
            lo, hi = expert_joint_limits("fx_arm6", setup=setup)
            usd_lo, usd_hi = fx_limits("fx_arm6")
            self.assertEqual(lo[0], usd_lo[0])
            self.assertEqual(hi[-2], usd_hi[-2])
            self.assertAlmostEqual(hi[0], math.radians(block[first]["joint_max"]), places=12)
            # the same answer when the setup is read from the scene folder
            lo2, hi2 = expert_joint_limits("fx_arm6")
            np.testing.assert_array_equal(lo, lo2)
            np.testing.assert_array_equal(hi, hi2)

    def test_disjoint_limits(self):
        """§1.5 runtime and USD ranges that do not overlap for some joint -> ConfigError."""
        with sandbox() as box:
            copy_fixture("arms/fx_arm6", box.root)
            setup = fx_setup(["fx_arm6"])
            first = list(setup["arms"]["fx_arm6"]["runtime_joint_limits"])[0]
            setup["arms"]["fx_arm6"]["runtime_joint_limits"][first] = {"joint_min": 170.0, "joint_max": 175.0}
            with self.assertRaises(ConfigError):
                expert_joint_limits("fx_arm6", setup=setup)

    def test_chain_limits_are_usd(self):
        """§5.3 sanity: KinematicChain.lower/upper are the USD limits the intersection uses."""
        with sandbox() as box:
            _fx_box(box)
            for arm in FX_ARMS:
                chain = KinematicChain.from_arm(arm)
                np.testing.assert_array_equal(chain.lower, fx_limits(arm)[0])


if __name__ == "__main__":
    unittest.main()
