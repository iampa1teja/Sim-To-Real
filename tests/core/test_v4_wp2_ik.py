"""WP2: KinematicChain.ik — damped least squares with bounds (contracts §5.3, §1.5).

Error definitions used to check the reported numbers (§5.3): with R the TCP rotation of the result and
R* the target rotation, a = R·approach, a* = R*·approach, c = R·closing, c* = R*·closing:
position_error_m = ‖p − p*‖; orientation_error_rad = rotation angle of Rᵀ·R*; axis_error_rad = ∠(a, a*);
roll_error_rad = |signed angle about a* from c* to c projected onto the plane ⟂ a*|.
"""

import math
import unittest

import numpy as np

from core.arms.kinematics import IKResult, IKWeights, KinematicChain
from core.tasks.base import Pose
from v4_support import assert_frozen
from v4_support.fx_arms import (FX_ARMS, axis_rotation, frame_matrix, fx_arm_table, fx_joint_names, fx_kinematics,
                                fx_limits, matrix_to_quat_wxyz, random_q)

TOL_POS, TOL_ROT = 5e-4, 0.02   # §5.3 ik() defaults


def make_chain(name, *, chain_joints=None):
    table = fx_arm_table(name)
    kin = table["kinematics"]
    return KinematicChain.from_dict(
        fx_kinematics(name), q_names=fx_joint_names(name),
        chain_joints=table["expert"]["arm_joints"] if chain_joints is None else chain_joints,
        ee_link=kin["ee_link"], tcp_position_m=kin["tcp_position_m"], tcp_quat_wxyz=kin["tcp_quat_wxyz"],
        approach_axis=kin["approach_axis"], closing_axis=kin["closing_axis"])


def to_pose(matrix) -> Pose:
    w, x, y, z = matrix_to_quat_wxyz(matrix)
    return Pose(tuple(float(v) for v in matrix[:3, 3]), (float(x), float(y), float(z), float(w)))


def pose_matrix(pose: Pose) -> np.ndarray:
    x, y, z, w = pose.quat_xyzw
    return frame_matrix(pose.position, [w, x, y, z])


def expected_errors(name, chain, q, target: Pose):
    kin = fx_arm_table(name)["kinematics"]
    approach, closing = np.asarray(kin["approach_axis"], float), np.asarray(kin["closing_axis"], float)
    T, Tt = chain.fk(q), pose_matrix(target)
    R, Rt = T[:3, :3], Tt[:3, :3]
    a, at, c, ct = R @ approach, Rt @ approach, R @ closing, Rt @ closing
    cp = c - (c @ at) * at
    roll = math.atan2(float(at @ np.cross(ct, cp)), float(ct @ cp))
    cos = float(np.clip((np.trace(R.T @ Rt) - 1.0) / 2.0, -1.0, 1.0))
    return {"position": float(np.linalg.norm(T[:3, 3] - Tt[:3, 3])), "orientation": math.acos(cos),
            "axis": math.acos(float(np.clip(a @ at, -1.0, 1.0))), "roll": abs(roll)}


def random_quat_xyzw(rng):
    v = rng.normal(size=4)
    return tuple(float(x) for x in v / np.linalg.norm(v))


class _IKCase(unittest.TestCase):
    def check_result(self, name, chain, target, result, *, weights=None, lower=None, upper=None, seed=None,
                     tol_pos=TOL_POS, tol_rot=TOL_ROT):
        """Shape, limits, honest error report and the §5.3 success rule."""
        weights = weights or IKWeights()
        self.assertIsInstance(result, IKResult)
        self.assertEqual(result.q.shape, (len(chain.q_names),))
        self.assertEqual(result.q.dtype, np.float64)
        lo = chain.lower if lower is None else np.maximum(lower, chain.lower)
        hi = chain.upper if upper is None else np.minimum(upper, chain.upper)
        idx = [chain.q_names.index(j) for j in chain.chain_joints]
        self.assertTrue(np.all(result.q[idx] >= lo[idx]) and np.all(result.q[idx] <= hi[idx]),
                        f"q {result.q} outside [{lo}, {hi}]")
        if seed is not None:
            rest = [i for i in range(len(chain.q_names)) if i not in idx]
            np.testing.assert_array_equal(result.q[rest], np.asarray(seed, float)[rest])
        want = expected_errors(name, chain, result.q, target)
        self.assertAlmostEqual(result.position_error_m, want["position"], delta=1e-9)
        self.assertAlmostEqual(result.orientation_error_rad, want["orientation"], delta=1e-5)
        self.assertAlmostEqual(result.axis_error_rad, want["axis"], delta=1e-5)
        if result.axis_error_rad < 1e-3:
            self.assertAlmostEqual(result.roll_error_rad, want["roll"], delta=1e-4)
        self.assertGreaterEqual(result.roll_error_rad, 0.0)
        for field in ("position_error_m", "orientation_error_rad", "axis_error_rad", "roll_error_rad"):
            self.assertIsInstance(getattr(result, field), float)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.iterations, int)
        rule = (result.position_error_m <= tol_pos
                and (weights.axis <= 0 or result.axis_error_rad <= tol_rot)
                and (weights.roll <= 0 or result.roll_error_rad <= tol_rot))
        self.assertEqual(result.success, rule)


class IKTypesTests(unittest.TestCase):
    """WP2 §5.3 IKWeights / IKResult."""

    def test_weights_defaults_and_frozen(self):
        """§5.3 IKWeights(position=1.0, axis=0.3, roll=0.1), frozen."""
        w = IKWeights()
        self.assertEqual((w.position, w.axis, w.roll), (1.0, 0.3, 0.1))
        assert_frozen(w)
        self.assertEqual(IKWeights(2.0, 0.0, 0.0).position, 2.0)

    def test_negative_weight_is_value_error(self):
        """§5.1 weights are ≥ 0: a negative weight raises ValueError."""
        for kwargs in ({"position": -1.0}, {"axis": -0.1}, {"roll": -0.1}, {"axis": float("nan")}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                with self.assertRaises(ValueError):
                    IKWeights(**kwargs)

    def test_result_is_frozen_with_readonly_q(self):
        """§1.3/§5.3 IKResult is a frozen dataclass; q is float64, C-contiguous and read-only."""
        name = "fx_arm6"
        chain = make_chain(name)
        q = random_q(name, np.random.default_rng(0), margin=0.35)
        result = chain.ik(chain.fk_pose(q), q)
        assert_frozen(result)
        self.assertEqual([f for f in IKResult.__dataclass_fields__],
                         ["q", "success", "position_error_m", "orientation_error_rad", "axis_error_rad",
                          "roll_error_rad", "iterations"])


class IKRoundTripTests(_IKCase):
    """WP2 §5.3 tolerances: ik(fk(q)) from a seed perturbed by ±0.3 rad."""

    N = 40

    def test_round_trip_position_and_orientation(self):
        """§5.3 table: position ≤ 2e-4 m on both arms; orientation ≤ 1e-3 rad on the 6-DoF arm (full weights)."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(100)
            for k in range(self.N):
                q = random_q(name, rng, margin=0.35)
                seed = q + rng.uniform(-0.3, 0.3, size=q.shape)
                target = chain.fk_pose(q)
                with self.subTest(arm=name, k=k):
                    result = chain.ik(target, seed)
                    self.check_result(name, chain, target, result, seed=seed)
                    self.assertTrue(result.success)
                    self.assertLessEqual(result.position_error_m, 2e-4)
                    if name == "fx_arm6":
                        self.assertLessEqual(result.orientation_error_rad, 1e-3)

    def test_seed_outside_limits(self):
        """§5.3 the result is always within the limits, even when the seed is not."""
        name = "fx_arm6"
        chain = make_chain(name)
        lower, upper = fx_limits(name)
        rng = np.random.default_rng(101)
        for k in range(10):
            q = random_q(name, rng, margin=0.35)
            seed = np.where(rng.random(q.shape) < 0.5, lower - 0.5, upper + 0.5)
            target = chain.fk_pose(q)
            with self.subTest(k=k):
                self.check_result(name, chain, target, chain.ik(target, seed), seed=seed)

    def test_non_chain_joints_copied_from_seed(self):
        """§5.3 joints outside chain_joints (gripper; an excluded arm joint) are copied from seed_q unchanged."""
        name = "fx_arm5"
        arm_joints = [j for j in fx_arm_table(name)["expert"]["arm_joints"] if j != "fx5_roll"]
        chain = make_chain(name, chain_joints=arm_joints)
        rng = np.random.default_rng(102)
        for k in range(10):
            q = random_q(name, rng, margin=0.35)
            seed = q + rng.uniform(-0.2, 0.2, size=q.shape)
            seed[fx_joint_names(name).index("fx5_roll")] = q[fx_joint_names(name).index("fx5_roll")]
            seed[-1] = 0.123
            target = chain.fk_pose(q)
            with self.subTest(k=k):
                result = chain.ik(target, seed, weights=IKWeights(1.0, 0.0, 0.0))
                self.check_result(name, chain, target, result, weights=IKWeights(1.0, 0.0, 0.0), seed=seed)
                self.assertEqual(result.q[-1], 0.123)

    def test_deterministic(self):
        """§5.3 identical inputs give identical outputs (also across chain instances)."""
        name = "fx_arm6"
        rng = np.random.default_rng(103)
        q = random_q(name, rng, margin=0.35)
        seed = q + rng.uniform(-0.3, 0.3, size=q.shape)
        target = make_chain(name).fk_pose(q)
        a = make_chain(name).ik(target, seed)
        b = make_chain(name).ik(target, seed)
        np.testing.assert_array_equal(a.q, b.q)
        self.assertEqual((a.success, a.position_error_m, a.orientation_error_rad, a.axis_error_rad, a.roll_error_rad,
                          a.iterations),
                         (b.success, b.position_error_m, b.orientation_error_rad, b.axis_error_rad, b.roll_error_rad,
                          b.iterations))

    def test_does_not_mutate_seed(self):
        """§1.3 ik leaves seed_q unchanged."""
        name = "fx_arm5"
        chain = make_chain(name)
        q = random_q(name, np.random.default_rng(104), margin=0.35)
        seed = q + 0.2
        before = seed.copy()
        chain.ik(chain.fk_pose(q), seed)
        np.testing.assert_array_equal(seed, before)


class IKLimitsTests(_IKCase):
    """WP2 §5.3 result ∈ [max(lower, self.lower), min(upper, self.upper)]."""

    def test_user_bounds_are_respected(self):
        """§5.3 random user boxes (tighter and wider than USD): the result stays inside their intersection."""
        for name in FX_ARMS:
            chain = make_chain(name)
            usd_lo, usd_hi = fx_limits(name)
            rng = np.random.default_rng(200)
            for k in range(15):
                center = random_q(name, rng, margin=0.35)
                lower = center - rng.uniform(0.05, 0.6, size=center.shape)
                upper = center + rng.uniform(0.05, 0.6, size=center.shape)
                lower[0], upper[1] = usd_lo[0] - 1.0, usd_hi[1] + 1.0      # wider than USD on two joints
                target = chain.fk_pose(random_q(name, rng, margin=0.35))
                with self.subTest(arm=name, k=k):
                    result = chain.ik(target, center, lower=lower, upper=upper)
                    self.check_result(name, chain, target, result, lower=lower, upper=upper, seed=center)

    def test_reachable_inside_box(self):
        """§5.3 a target reachable inside the user box is solved inside it."""
        name = "fx_arm6"
        chain = make_chain(name)
        rng = np.random.default_rng(201)
        for k in range(10):
            q = random_q(name, rng, margin=0.5)
            lower, upper = q - 0.3, q + 0.3
            seed = q + rng.uniform(-0.25, 0.25, size=q.shape)
            target = chain.fk_pose(q)
            with self.subTest(k=k):
                result = chain.ik(target, seed, lower=lower, upper=upper)
                self.check_result(name, chain, target, result, lower=lower, upper=upper, seed=seed)
                self.assertTrue(result.success)


class IKUnreachableTests(_IKCase):
    """WP2 §5.3/§1.5 unreachable targets return success=False with honest errors; IK never raises for them."""

    def test_target_out_of_workspace(self):
        """§5.3 a target 1 m away: success False, position error reported as the real distance, q inside limits."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(300)
            for k in range(5):
                seed = random_q(name, rng, margin=0.35)
                direction = rng.normal(size=3)
                position = 1.0 * direction / np.linalg.norm(direction)
                target = Pose(tuple(float(v) for v in position), random_quat_xyzw(rng))
                with self.subTest(arm=name, k=k):
                    result = chain.ik(target, seed)
                    self.check_result(name, chain, target, result, seed=seed)
                    self.assertFalse(result.success)
                    self.assertGreater(result.position_error_m, 0.5)

    def test_target_outside_user_box_is_not_clamped_to_success(self):
        """§5.3 a target reachable only outside the user box: success False, errors measured at the clamped q."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(301)
            for k in range(5):
                q0 = random_q(name, rng, margin=0.35)
                lower, upper = q0 - 0.02, q0 + 0.02
                far = chain.fk(q0)
                far[:3, 3] += np.array([0.0, 0.0, 0.15]) if far[2, 3] < 0.2 else np.array([0.0, 0.0, -0.15])
                target = to_pose(far)
                with self.subTest(arm=name, k=k):
                    result = chain.ik(target, q0, lower=lower, upper=upper)
                    self.check_result(name, chain, target, result, lower=lower, upper=upper, seed=q0)
                    self.assertFalse(result.success)
                    self.assertGreater(result.position_error_m, 0.05)

    def test_nan_input_raises_value_error(self):
        """§5.3 NaN input raises ValueError."""
        name = "fx_arm6"
        chain = make_chain(name)
        q = random_q(name, np.random.default_rng(302), margin=0.35)
        good = chain.fk_pose(q)
        nan_seed = q.copy()
        nan_seed[2] = np.nan
        cases = {
            "position": (Pose((float("nan"), 0.1, 0.1), good.quat_xyzw), q),
            "quaternion": (Pose(good.position, (0.0, float("nan"), 0.0, 1.0)), q),
            "seed": (good, nan_seed),
        }
        for label, (target, seed) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    chain.ik(target, seed)


class IKReducedOrientationTests(_IKCase):
    """WP2 §5.3 success counts the axis only when w_a > 0 and the roll only when w_r > 0 (5-DoF arm)."""

    NAME = "fx_arm5"

    def _rotated_about_approach(self, chain, q, angle):
        """fk(q) rotated about the TCP approach axis (+z of the TCP frame for the fx arms) by ``angle``."""
        return to_pose(chain.fk(q) @ axis_rotation(fx_arm_table(self.NAME)["kinematics"]["approach_axis"], angle))

    def test_position_only(self):
        """§5.3 weights (1, 0, 0): any orientation; success depends on the position only."""
        chain = make_chain(self.NAME)
        rng = np.random.default_rng(400)
        weights = IKWeights(1.0, 0.0, 0.0)
        big_axis_error = 0
        for k in range(25):
            q = random_q(self.NAME, rng, margin=0.35)
            seed = q + rng.uniform(-0.3, 0.3, size=q.shape)
            target = Pose(chain.fk_pose(q).position, random_quat_xyzw(rng))
            with self.subTest(k=k):
                result = chain.ik(target, seed, weights=weights)
                self.check_result(self.NAME, chain, target, result, weights=weights, seed=seed)
                self.assertTrue(result.success)
                self.assertLessEqual(result.position_error_m, 2e-4)
                big_axis_error += result.axis_error_rad > TOL_ROT
        self.assertGreater(big_axis_error, 10, "position-only targets must not be judged on orientation")

    def test_axis_without_roll(self):
        """§5.3 weights (1, 0.3, 0) with the roll joint boxed: success needs position + approach axis, not roll."""
        chain = make_chain(self.NAME)
        roll_index = fx_joint_names(self.NAME).index("fx5_roll")
        rng = np.random.default_rng(401)
        weights = IKWeights(1.0, 0.3, 0.0)
        for k in range(15):
            q = random_q(self.NAME, rng, margin=0.35)
            angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0))
            target = self._rotated_about_approach(chain, q, angle)
            lower, upper = np.full(q.shape, -np.inf), np.full(q.shape, np.inf)
            lower[roll_index], upper[roll_index] = q[roll_index] - 0.05, q[roll_index] + 0.05
            seed = q + rng.uniform(-0.2, 0.2, size=q.shape)
            with self.subTest(k=k):
                result = chain.ik(target, seed, weights=weights, lower=lower, upper=upper)
                self.check_result(self.NAME, chain, target, result, weights=weights, lower=lower, upper=upper, seed=seed)
                self.assertTrue(result.success)
                self.assertLessEqual(result.axis_error_rad, TOL_ROT)
                self.assertGreater(result.roll_error_rad, TOL_ROT)

    def test_unreachable_roll_with_roll_weight(self):
        """§5.3 the same boxed targets with default weights (roll weight > 0) are reported unreachable."""
        chain = make_chain(self.NAME)
        roll_index = fx_joint_names(self.NAME).index("fx5_roll")
        rng = np.random.default_rng(401)
        for k in range(8):
            q = random_q(self.NAME, rng, margin=0.35)
            angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0))
            target = self._rotated_about_approach(chain, q, angle)
            lower, upper = np.full(q.shape, -np.inf), np.full(q.shape, np.inf)
            lower[roll_index], upper[roll_index] = q[roll_index] - 0.05, q[roll_index] + 0.05
            seed = q + rng.uniform(-0.2, 0.2, size=q.shape)
            with self.subTest(k=k):
                result = chain.ik(target, seed, lower=lower, upper=upper)
                self.check_result(self.NAME, chain, target, result, lower=lower, upper=upper, seed=seed)
                self.assertFalse(result.success)

    def test_any_positive_weight_enables_its_check(self):
        """§5.3 the axis / roll checks apply whenever w_a > 0 / w_r > 0, however small the weight."""
        chain = make_chain(self.NAME)
        roll_index = fx_joint_names(self.NAME).index("fx5_roll")
        rng = np.random.default_rng(404)
        for k in range(8):
            q = random_q(self.NAME, rng, margin=0.35)
            seed = q + rng.uniform(-0.2, 0.2, size=q.shape)
            angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0))
            lower, upper = np.full(q.shape, -np.inf), np.full(q.shape, np.inf)
            lower[roll_index], upper[roll_index] = q[roll_index] - 0.05, q[roll_index] + 0.05
            rolled = self._rotated_about_approach(chain, q, angle)
            tiny_roll = IKWeights(1.0, 0.0, 1e-9)
            with self.subTest(k=k, check="roll"):
                result = chain.ik(rolled, seed, weights=tiny_roll, lower=lower, upper=upper)
                self.check_result(self.NAME, chain, rolled, result, weights=tiny_roll, lower=lower, upper=upper,
                                  seed=seed)
                self.assertLessEqual(result.position_error_m, TOL_POS)
                self.assertGreater(result.roll_error_rad, TOL_ROT)
                self.assertFalse(result.success)
            tilted = Pose(chain.fk_pose(q).position, to_pose(chain.fk(q) @ axis_rotation([1.0, 0.0, 0.0], 0.6)).quat_xyzw)
            tiny_axis = IKWeights(1.0, 1e-9, 0.0)
            with self.subTest(k=k, check="axis"):
                result = chain.ik(tilted, seed, weights=tiny_axis, lower=lower, upper=upper)
                self.check_result(self.NAME, chain, tilted, result, weights=tiny_axis, lower=lower, upper=upper,
                                  seed=seed)
                self.assertLessEqual(result.position_error_m, TOL_POS)
                self.assertGreater(result.axis_error_rad, TOL_ROT)
                self.assertFalse(result.success)

    def test_full_weights_reach_rolled_targets(self):
        """§5.3 default weights on the 5-DoF arm: a target rolled about the approach axis is solved by the roll joint."""
        chain = make_chain(self.NAME)
        roll_index = fx_joint_names(self.NAME).index("fx5_roll")
        rng = np.random.default_rng(402)
        for k in range(15):
            q = random_q(self.NAME, rng, margin=0.35)
            q[roll_index] = rng.uniform(-1.5, 1.5)
            target = self._rotated_about_approach(chain, q, float(rng.uniform(-1.0, 1.0)))
            seed = q + rng.uniform(-0.3, 0.3, size=q.shape)
            with self.subTest(k=k):
                result = chain.ik(target, seed)
                self.check_result(self.NAME, chain, target, result, seed=seed)
                self.assertTrue(result.success)
                self.assertLessEqual(result.position_error_m, 2e-4)
                self.assertLessEqual(result.roll_error_rad, TOL_ROT)

    def test_tolerance_arguments_drive_success(self):
        """§5.3 tol_pos_m / tol_rot_rad are the thresholds of the success rule."""
        chain = make_chain(self.NAME)
        rng = np.random.default_rng(403)
        q = random_q(self.NAME, rng, margin=0.35)
        target = to_pose(chain.fk(q) @ frame_matrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        far = Pose((target.position[0] + 2.0, target.position[1], target.position[2]), target.quat_xyzw)
        result = chain.ik(far, q, weights=IKWeights(1.0, 0.0, 0.0), tol_pos_m=10.0)
        self.check_result(self.NAME, chain, far, result, weights=IKWeights(1.0, 0.0, 0.0), tol_pos=10.0, seed=q)
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
