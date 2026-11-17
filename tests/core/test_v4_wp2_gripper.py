"""WP2: Gripper — opening gap (m) <-> jaw angle (rad) from the [arm.gripper] table (contracts §5.1, §5.4)."""

import unittest

import numpy as np

from core.arms.gripper import Gripper
from core.generate.errors import ConfigError
from v4_support import assert_frozen, sandbox
from v4_support.fx_arms import FX_ARMS, FX_OPENING_TABLE, fx_arm_table, fx_joint_names, fx_kinematics, write_arm, write_fx_arm


def table(**overrides):
    out = dict(fx_arm_table("fx_arm6")["gripper"])
    out.update(overrides)
    return out


class GripperFromTableTests(unittest.TestCase):
    """WP2 §5.4 Gripper.from_table."""

    def test_fields(self):
        """§5.4 fields: joint, index into q (q_names order), open/closed rad, table columns as float tuples, width."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                grip_table = fx_arm_table(name)["gripper"]
                g = Gripper.from_table(grip_table, fx_joint_names(name))
                self.assertEqual(g.joint, grip_table["joint"])
                self.assertEqual(g.index, fx_joint_names(name).index(grip_table["joint"]))
                self.assertEqual((g.open_rad, g.closed_rad), (1.0, 0.0))
                self.assertEqual(g.angles_rad, tuple(float(r[0]) for r in FX_OPENING_TABLE))
                self.assertEqual(g.gaps_m, tuple(float(r[1]) for r in FX_OPENING_TABLE))
                self.assertIsInstance(g.angles_rad, tuple)
                self.assertTrue(all(isinstance(v, float) for v in g.angles_rad + g.gaps_m))
                self.assertEqual(g.max_object_width_m, 0.045)
                assert_frozen(g)

    def test_index_follows_q_names(self):
        """§1.2 index is the gripper's position in the given joint vector order."""
        names = list(fx_joint_names("fx_arm6"))
        names.insert(0, names.pop())          # gripper first
        self.assertEqual(Gripper.from_table(table(), names).index, 0)

    def test_invalid_tables_raise_config_error(self):
        """§1.5/§5.1 joint not in q, < 2 rows, angles not strictly increasing, gaps not strictly monotone."""
        names = fx_joint_names("fx_arm6")
        cases = {
            "unknown_joint": table(joint="fx6_nojoint"),
            "one_row": table(opening_table=[[0.0, 0.0]]),
            "angles_equal": table(opening_table=[[0.0, 0.0], [0.5, 0.01], [0.5, 0.02]]),
            "angles_decreasing": table(opening_table=[[0.5, 0.0], [0.0, 0.02]]),
            "gaps_not_monotone": table(opening_table=[[0.0, 0.0], [0.5, 0.03], [1.0, 0.02]]),
            "gaps_flat": table(opening_table=[[0.0, 0.01], [0.5, 0.01], [1.0, 0.02]]),
        }
        for label, bad in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ConfigError):
                    Gripper.from_table(bad, names)


class GripperMappingTests(unittest.TestCase):
    """WP2 §5.4 gap_from_angle (piecewise linear, clamped) and angle_from_gap (inverse, round trip ≤ 1e-9)."""

    def setUp(self):
        self.g = Gripper.from_table(table(), fx_joint_names("fx_arm6"))

    def test_table_points_and_midpoints(self):
        """§5.4 piecewise linear through the table rows (golden values from FX_OPENING_TABLE)."""
        for angle, gap in FX_OPENING_TABLE:
            self.assertAlmostEqual(self.g.gap_from_angle(angle), gap, delta=1e-12)
            self.assertAlmostEqual(self.g.angle_from_gap(gap), angle, delta=1e-12)
        self.assertAlmostEqual(self.g.gap_from_angle(0.125), 0.006, delta=1e-12)
        self.assertAlmostEqual(self.g.gap_from_angle(0.875), 0.0425, delta=1e-12)
        self.assertAlmostEqual(self.g.angle_from_gap(0.0185), 0.375, delta=1e-12)
        self.assertIsInstance(self.g.gap_from_angle(0.3), float)
        self.assertIsInstance(self.g.angle_from_gap(0.01), float)

    def test_clamped_to_table_ends(self):
        """§5.4 values outside the table clamp to the end rows."""
        self.assertEqual(self.g.gap_from_angle(-0.7), 0.0)
        self.assertEqual(self.g.gap_from_angle(3.0), 0.048)
        self.assertEqual(self.g.angle_from_gap(-0.01), 0.0)
        self.assertEqual(self.g.angle_from_gap(0.2), 1.0)

    def test_round_trips(self):
        """§5.4 angle→gap→angle and gap→angle→gap within 1e-9 inside the table."""
        rng = np.random.default_rng(0)
        for k, angle in enumerate(rng.uniform(0.0, 1.0, 200)):
            with self.subTest(k=k, angle=angle):
                self.assertAlmostEqual(self.g.angle_from_gap(self.g.gap_from_angle(float(angle))), angle, delta=1e-9)
        for k, gap in enumerate(rng.uniform(0.0, 0.048, 200)):
            with self.subTest(k=k, gap=gap):
                self.assertAlmostEqual(self.g.gap_from_angle(self.g.angle_from_gap(float(gap))), gap, delta=1e-9)

    def test_monotone(self):
        """§5.4 both maps are monotone (strictly inside the table) for an increasing table."""
        angles = np.linspace(-0.2, 1.2, 701)
        gaps = np.array([self.g.gap_from_angle(float(a)) for a in angles])
        self.assertTrue(np.all(np.diff(gaps) >= 0))
        inside = (angles > 0) & (angles < 1)
        self.assertTrue(np.all(np.diff(gaps[inside]) > 0))
        g_grid = np.linspace(-0.01, 0.06, 701)
        back = np.array([self.g.angle_from_gap(float(v)) for v in g_grid])
        self.assertTrue(np.all(np.diff(back) >= 0))
        self.assertTrue(np.all(np.diff(back[(g_grid > 0) & (g_grid < 0.048)]) > 0))

    def test_decreasing_gap_table(self):
        """§5.1 gaps may be strictly decreasing (a jaw that closes with growing angle): inverse and round trip hold."""
        rows = [[a, 0.048 - gap] for a, gap in FX_OPENING_TABLE]
        g = Gripper.from_table(table(opening_table=rows, open_rad=0.0, closed_rad=1.0), fx_joint_names("fx_arm6"))
        self.assertAlmostEqual(g.gap_from_angle(0.0), 0.048, delta=1e-12)
        self.assertAlmostEqual(g.gap_from_angle(1.0), 0.0, delta=1e-12)
        self.assertEqual(g.angle_from_gap(1.0), 0.0)
        self.assertEqual(g.angle_from_gap(-1.0), 1.0)
        rng = np.random.default_rng(1)
        for k, angle in enumerate(rng.uniform(0.0, 1.0, 100)):
            with self.subTest(k=k):
                self.assertAlmostEqual(g.angle_from_gap(g.gap_from_angle(float(angle))), angle, delta=1e-9)
        gaps = [g.gap_from_angle(float(a)) for a in np.linspace(0.0, 1.0, 101)]
        self.assertTrue(np.all(np.diff(gaps) < 0))


class GripperFromArmTests(unittest.TestCase):
    """WP2 §5.4 Gripper.from_arm reads [arm.gripper] and [arm.joints].names of the named arm."""

    def test_from_arm_equals_from_table(self):
        """§5.4 from_arm(name) == from_table([arm.gripper], [arm.joints].names)."""
        for name in FX_ARMS:
            with self.subTest(arm=name), sandbox() as box:
                write_fx_arm(box.arms, name)
                self.assertEqual(Gripper.from_arm(name),
                                 Gripper.from_table(fx_arm_table(name)["gripper"], fx_joint_names(name)))

    def test_from_arm_without_gripper_table(self):
        """§1.5 an arm without [arm.gripper] raises ConfigError at from_arm."""
        name = "fx_arm5"
        arm_table = fx_arm_table(name)
        del arm_table["gripper"]
        with sandbox() as box:
            write_arm(box.arms, name, arm_table, kinematics=fx_kinematics(name))
            with self.assertRaises(ConfigError):
                Gripper.from_arm(name)


if __name__ == "__main__":
    unittest.main()
