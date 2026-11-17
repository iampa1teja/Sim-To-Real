"""WP6: group-B ``WorldPerturber`` (onset rule, new poses, object_drop support) and group-C ``ou_noise``
(contracts §7.1, §11 support rule, §11.1 loop step 5, §14 decision 14, docs/v4/prompts/WP6.md)."""

import math
import unittest

import numpy as np

from core.generate.perturbations import WorldPerturber, ou_noise
from core.generate.protocols import ExpertObs, Kind, ObjectState, PerturbationGroup, PerturbationSpec
from core.tasks.base import Pose, load_task_spec
from v4_support import sandbox
from v4_support.factories import make_episode_spec, make_noise, make_pose, make_stage_progress
from v4_support.tasks import STUB_USDA, contract_color_task_table, write_task

BOX = (0.25, 0.0, 0.0)          # small_box origin (bottom), size_outer 0.05 x 0.05 x 0.035, floor_z 0.002
CUBE_H = 0.02                   # cube_* size 0.02 (centre origin)


def color_task_spec():
    with sandbox() as box:
        table = contract_color_task_table()
        table["start"]["table_z_m"] = 0.0
        write_task(box.tasks, "fx6_world_color", table, files={"usd/Small-Box-Physics.usda": STUB_USDA})
        return load_task_spec("fx6_world_color")


def objects(target=(0.15, 0.05, 0.01), yaw=0.0, cube_1=(0.15, -0.08, 0.01), box_yaw=0.0):
    """cube_0 (target), cube_1 (distractor), small_box active; cubes 2..5 parked."""
    out = {"cube_0": ObjectState(make_pose(*target, yaw=yaw), color="red"),
           "cube_1": ObjectState(make_pose(*cube_1), color="green"),
           "small_box": ObjectState(make_pose(*BOX, yaw=box_yaw))}
    for i in range(2, 6):
        out[f"cube_{i}"] = ObjectState(make_pose(0.1 * i, 0.0, -2.0), active=False)
    return out


def episode_spec(ptype, params, onset_stage="reached", delay=0, kind=Kind.FAILURE, group=PerturbationGroup.WORLD):
    pert = PerturbationSpec(type=ptype, group=group, onset_stage=onset_stage, delay_steps=delay, params=params)
    return make_episode_spec(kind, task="fx6_world_color", objects=objects(),
                             roles={"target": "cube_0", "container": "small_box"}, perturbation=pert)


def obs_with(objs, frame=5):
    return ExpertObs(frame=frame, q=np.zeros(6), ee=make_pose(0.2, 0.0, 0.15), tcp=make_pose(0.2, 0.0, 0.06),
                     gripper_gap_m=0.02, objects=objs)


def progress(frame, furthest, stage=None, holding=True):
    return make_stage_progress(frame=frame, stage_index=furthest if stage is None else stage, furthest_index=furthest,
                               satisfied={"holding": holding})


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class StubEnv:
    def __init__(self):
        self.calls = []

    def set_object_pose(self, name, pose):
        self.calls.append((name, pose))


class OnsetTests(unittest.TestCase):
    """WP6 §7.1 group B: due on the first frame with ``onset_stage`` latched (``furthest_index`` ≥ its index)
    plus ``delay_steps``; the loop applies after ``env.step`` at frame t, so the onset frame is t + 1 (D14)."""

    @classmethod
    def setUpClass(cls):
        cls.task_spec = color_task_spec()

    def test_due_after_latch_plus_delay(self):
        """Onset stage ``lifted`` (index 2) latches at frame 4; delay 3 → due at frame 7 only; never before."""
        for delay in (0, 1, 3, 10):
            spec = episode_spec("object_shift", {"object": "@target", "xy_m": 0.03, "dir_rad": 0.0, "yaw_rad": 0.0},
                                onset_stage="lifted", delay=delay)
            perturber = WorldPerturber(spec, self.task_spec, np.random.default_rng(0))
            furthest = [-1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4]
            due_at = None
            for t, f in enumerate(furthest):
                if perturber.due(progress(t, f), t):
                    due_at = t
                    pose = perturber.apply(StubEnv(), obs_with(objects()))
                    self.assertIsInstance(pose, Pose)
                    break
            with self.subTest(delay=delay):
                self.assertEqual(due_at, 4 + delay)
                self.assertTrue(perturber.applied)
                self.assertEqual(perturber.onset_frame, 4 + delay + 1)
                later = [perturber.due(progress(t, 4), t) for t in range(due_at + 1, due_at + 20)]
                self.assertFalse(any(later), "a world perturbation fires once")

    def test_latch_is_ever_reached_not_current(self):
        """'latched': a regression of the current stage after the latch does not postpone the onset."""
        spec = episode_spec("object_shift", {"object": "@target", "xy_m": 0.03, "dir_rad": 0.0, "yaw_rad": 0.0},
                            onset_stage="reached", delay=2)
        perturber = WorldPerturber(spec, self.task_spec, np.random.default_rng(0))
        frames = [(-1, -1), (0, -1), (0, -1), (0, -1), (0, -1)]          # (furthest, current): regressed at once
        due = [perturber.due(make_stage_progress(frame=t, stage_index=c, furthest_index=f,
                                                 satisfied={"holding": False}), t) for t, (f, c) in enumerate(frames)]
        self.assertEqual(due, [False, False, False, True, True], "due from latch + delay until applied")

    def test_never_due_for_group_a_or_success(self):
        """Group-A specs run inside the expert; a success spec has no perturbation: never due, never applied."""
        specs = [episode_spec("grasp_offset", {"xy_m": 0.01, "dir_rad": 0.0, "z_m": 0.0, "yaw_rad": 0.0},
                              onset_stage=None, group=PerturbationGroup.EXPERT),
                 episode_spec("stall", {"steps": 5}, onset_stage="reached", group=PerturbationGroup.EXPERT),
                 make_episode_spec("success")]
        for spec in specs:
            perturber = WorldPerturber(spec, self.task_spec, np.random.default_rng(0))
            with self.subTest(type=getattr(spec.perturbation, "type", None)):
                self.assertFalse(any(perturber.due(progress(t, 5), t) for t in range(30)))
                self.assertFalse(perturber.applied)
                self.assertIsNone(perturber.onset_frame)

    def test_object_drop_needs_holding(self):
        """§7.1 object_drop is 'not applied if nothing is held then': ``satisfied["holding"]`` false on the due
        frame → never due afterwards, ``applied`` stays false (the loop rejects ``perturbation_not_applied``)."""
        spec = episode_spec("object_drop", {"xy_jitter_m": 0.0}, onset_stage="lifted", delay=2)
        perturber = WorldPerturber(spec, self.task_spec, np.random.default_rng(0))
        due = [perturber.due(progress(t, 2, holding=(t < 2)), t) for t in range(10)]
        self.assertEqual(due, [False] * 10)
        self.assertFalse(perturber.applied)
        held = WorldPerturber(spec, self.task_spec, np.random.default_rng(0))
        due = [held.due(progress(t, 2, holding=True), t) for t in range(3)]
        self.assertEqual(due, [False, False, True])


class NewPoseTests(unittest.TestCase):
    """WP6 §7.1 new poses; ``apply`` teleports with ``env.set_object_pose(name, pose)`` and returns the pose."""

    @classmethod
    def setUpClass(cls):
        cls.task_spec = color_task_spec()

    def fire(self, spec, objs, seed=0, holding=True):
        perturber = WorldPerturber(spec, self.task_spec, np.random.default_rng(seed))
        t = 0
        while not perturber.due(progress(t, 5, holding=holding), t):
            t += 1
            self.assertLess(t, 100)
        env = StubEnv()
        pose = perturber.apply(env, obs_with(objs))
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(env.calls[0][1], pose)
        return env.calls[0][0], pose

    def test_object_shift(self):
        """Current xy + xy_m·(cos dir, sin dir), yaw + yaw_rad, z unchanged; the object is ``@target``."""
        rng = np.random.default_rng(3)
        for _ in range(25):
            r, d, dyaw, yaw0 = rng.uniform(0.02, 0.05), rng.uniform(0, 2 * np.pi), rng.uniform(-0.5, 0.5), rng.uniform(-3, 3)
            spec = episode_spec("object_shift", {"object": "@target", "xy_m": r, "dir_rad": d, "yaw_rad": dyaw})
            name, pose = self.fire(spec, objects(target=(0.15, 0.05, 0.01), yaw=yaw0))
            with self.subTest(r=r, d=d):
                self.assertEqual(name, "cube_0")
                np.testing.assert_allclose(pose.position, (0.15 + r * math.cos(d), 0.05 + r * math.sin(d), 0.01),
                                           atol=1e-12)
                self.assertAlmostEqual(math.remainder(yaw_of(pose.quat_xyzw) - (yaw0 + dyaw), 2 * math.pi), 0.0,
                                       places=9)
                self.assertAlmostEqual(float(np.linalg.norm(pose.quat_xyzw)), 1.0, places=9)

    def test_container_shift(self):
        """Default object ``@container``: the box moves by xy_m along dir; z and orientation unchanged."""
        spec = episode_spec("container_shift", {"object": "@container", "xy_m": 0.04, "dir_rad": math.pi / 2},
                            onset_stage="lifted")
        name, pose = self.fire(spec, objects(box_yaw=0.3))
        self.assertEqual(name, "small_box")
        np.testing.assert_allclose(pose.position, (BOX[0], BOX[1] + 0.04, BOX[2]), atol=1e-12)
        self.assertAlmostEqual(math.remainder(yaw_of(pose.quat_xyzw) - 0.3, 2 * math.pi), 0.0, places=9)

    def test_object_drop_supports(self):
        """§11 support below the dropped target (yaw kept, roll/pitch zero): the table (``table_z_m`` + h/2), the
        box floor inside its inner rectangle (box z + floor_z + h/2), the box top on its rim (box z + 0.035 + h/2),
        another cube's top (its z + h/2 + h/2)."""
        cases = {
            "table": ((0.15, 0.05, 0.08), 0.0 + CUBE_H / 2),
            "box_floor": ((BOX[0] + 0.01, BOX[1] - 0.01, 0.08), BOX[2] + 0.002 + CUBE_H / 2),
            "box_rim": ((BOX[0] + 0.023, BOX[1], 0.08), BOX[2] + 0.035 + CUBE_H / 2),
            "on_cube_1": ((0.15 + 0.005, -0.08, 0.08), 0.01 + CUBE_H / 2 + CUBE_H / 2),
        }
        for case, (held_at, z) in cases.items():
            spec = episode_spec("object_drop", {"xy_jitter_m": 0.0}, onset_stage="lifted")
            tilted = objects(target=held_at)
            tilted["cube_0"] = ObjectState(Pose(held_at, (0.1, 0.05, math.sin(0.35), math.cos(0.35))), color="red")
            name, pose = self.fire(spec, tilted)
            with self.subTest(case=case):
                self.assertEqual(name, "cube_0")
                np.testing.assert_allclose(pose.position, (held_at[0], held_at[1], z), atol=1e-9)
                x, y, _, _ = pose.quat_xyzw
                self.assertAlmostEqual(x, 0.0, places=12)
                self.assertAlmostEqual(y, 0.0, places=12)
                expected_yaw = yaw_of((0.1, 0.05, math.sin(0.35), math.cos(0.35)))
                self.assertAlmostEqual(math.remainder(yaw_of(pose.quat_xyzw) - expected_yaw, 2 * math.pi), 0.0,
                                       places=9)

    def test_object_drop_jitter_direction_from_rng(self):
        """WP6.md: the jitter direction is one ``rng.uniform(0, 2π)`` drawn in ``apply`` from the perturber rng."""
        for seed in range(10):
            r = 0.015
            spec = episode_spec("object_drop", {"xy_jitter_m": r}, onset_stage="lifted")
            _, pose = self.fire(spec, objects(target=(0.15, 0.05, 0.08)), seed=seed)
            d = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi)
            with self.subTest(seed=seed):
                np.testing.assert_allclose(pose.position[:2], (0.15 + r * math.cos(d), 0.05 + r * math.sin(d)),
                                           atol=1e-12)


class OUNoiseTests(unittest.TestCase):
    """WP6 §7.1 group C: ``ou_noise(noise, arm_mask, steps, rng) -> (steps, J)``."""

    MASK = (True, True, True, True, True, False)

    @staticmethod
    def reference(noise, mask, steps, rng):
        mask = np.asarray(mask, bool)
        out = np.zeros((steps, mask.size))
        cur = np.zeros(int(mask.sum()))
        for t in range(steps - 1):
            cur = np.clip(cur - noise.theta * cur + noise.sigma_rad * rng.standard_normal(int(mask.sum())),
                          -noise.max_abs_rad, noise.max_abs_rad)
            out[t + 1, mask] = cur
        return out

    def test_shape_dtype_bounds(self):
        """n[0] = 0; gripper (mask False) columns 0; |n| ≤ max_abs_rad; float64 C-contiguous (§1.3)."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            noise = make_noise(theta=float(rng.uniform(0, 1)), sigma_rad=float(rng.uniform(0, 0.05)),
                               max_abs_rad=float(rng.uniform(0.001, 0.05)))
            mask = tuple(bool(v) for v in rng.integers(0, 2, 7))
            n = ou_noise(noise, mask, 120, np.random.default_rng(seed))
            with self.subTest(seed=seed):
                self.assertEqual(n.shape, (120, 7))
                self.assertEqual(n.dtype, np.float64)
                self.assertTrue(n.flags.c_contiguous)
                np.testing.assert_array_equal(n[0], 0.0)
                np.testing.assert_array_equal(n[:, ~np.asarray(mask)], 0.0)
                self.assertLessEqual(float(np.abs(n).max()), noise.max_abs_rad + 1e-15)

    def test_exact_recursion(self):
        """§7.1 n[t+1] = clip(n[t] − θ·n[t] + σ·N(0,1), ±max_abs): one ``rng.standard_normal(k)`` per step over the
        k arm columns (WP6.md), so noise is reproducible across implementations."""
        for seed in range(10):
            noise = make_noise(theta=0.15, sigma_rad=0.01, max_abs_rad=0.02)
            got = ou_noise(noise, self.MASK, 200, np.random.default_rng(seed))
            want = self.reference(noise, self.MASK, 200, np.random.default_rng(seed))
            with self.subTest(seed=seed):
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
        self.assertTrue(np.any(got != 0.0))
        self.assertTrue(np.any(np.abs(got) == 0.02), "clipping is active for these values")

    def test_degenerate_cases(self):
        """σ = 0 → all zeros; steps 0 / 1 → shapes (0, J) / zeros; an int instead of a mask → ``TypeError``."""
        np.testing.assert_array_equal(ou_noise(make_noise(sigma_rad=0.0), self.MASK, 50, np.random.default_rng(0)), 0.0)
        self.assertEqual(ou_noise(make_noise(), self.MASK, 0, np.random.default_rng(0)).shape, (0, 6))
        np.testing.assert_array_equal(ou_noise(make_noise(), self.MASK, 1, np.random.default_rng(0)), np.zeros((1, 6)))
        with self.assertRaises(TypeError):
            ou_noise(make_noise(), 5, 10, np.random.default_rng(0))

    def test_deterministic_and_rng_only(self):
        """§1.4 same rng seed → identical array; the global numpy RNG is not used."""
        np.random.seed(1)
        a = ou_noise(make_noise(), self.MASK, 80, np.random.default_rng(42))
        np.random.seed(2)
        b = ou_noise(make_noise(), self.MASK, 80, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
