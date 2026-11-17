"""WP7: FakeEnv, the kinematic GenEnv for CPU tests (contracts §11, §3.3).

Decisions pinned here (docs/v4/prompts/WP7.md):
* ``reset`` fills every task object; objects missing from the spec are inactive at the origin. Unknown object
  names in the spec or a wrong ``initial_q`` length raise ``EnvError``.
* The jaw box is in the TCP frame: ``hypot(d_x, d_y) <= w/2`` and ``|d_z| <= h/2`` with ``d = R_tcpᵀ(c − p_tcp)``
  and ``c`` the object centre; ``w`` = min footprint side, ``h`` = height.
* ``success`` is true on a frame when ``placed`` has been true on the last ``confirm_steps`` observed frames,
  counting the current one (the reset frame counts).
* ``crash_at_step = k`` raises ``SimulatedCrash`` on the k-th ``step`` call (0-based, counted across resets)
  before anything moves.
"""

import dataclasses
import math
import unittest

import numpy as np

from core.generate.errors import EnvError, GenerationError
from core.generate.fake_env import FakeEnv, SimulatedCrash
from core.generate.protocols import EpisodeSpec, GenEnv, Kind, ObjectState
from core.tasks.base import Pose
from test_v4_wp7_support import ARM, SandboxCase
from v4_support.fx_arms import fx_home_q

TABLE_Z = -0.05
CUBE = 0.02


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class FakeEnvBase(SandboxCase):
    TASKS = ("fx7_pick", "fx7_grip")

    def setUp(self):
        self.pick = self.task("fx7_pick")
        self.home = np.array(fx_home_q(ARM))
        self.tcp0 = self.rig.chain.fk(self.home, "tcp")[:3, 3]
        self.g = self.rig.gripper.index

    def make_spec(self, *, cube=None, box=None, held=None, q=None, cube_active=True, yaw=0.3, cube_color="red"):
        cube = (0.15, -0.08, TABLE_Z + CUBE / 2) if cube is None else cube
        box = (0.15, 0.06, TABLE_Z) if box is None else box
        objs = {"cube": ObjectState(Pose(tuple(map(float, cube)), yaw_quat(yaw)), cube_active, cube_color),
                "box": ObjectState(Pose(tuple(map(float, box)), yaw_quat(0.0)), True, None)}
        return EpisodeSpec(task="fx7_pick", arm=ARM, scene="fx_scene", kind=Kind.SUCCESS, attempt_index=0, seed=1,
                           objects=objs, roles={"target": "cube", "container": "box"} if cube_active else {"container": "box"},
                           goal={}, instruction="Pick up the cube and place it in the box",
                           initial_q=tuple(float(v) for v in (self.home if q is None else q)), held_object=held)

    def env(self, **kw):
        return FakeEnv(self.pick, self.rig.chain, self.rig.gripper, **kw)

    def q_with_gap(self, gap, q=None):
        q = np.array(self.home if q is None else q, dtype=float)
        q[self.g] = self.rig.gripper.angle_from_gap(gap)
        return q

    def at_tcp(self, q=None):
        return tuple(float(v) for v in self.rig.chain.fk(self.home if q is None else q, "tcp")[:3, 3])


class ProtocolTests(FakeEnvBase):
    """WP7 §3.3 FakeEnv implements GenEnv; reset returns frame 0 with the spec applied."""

    def test_conforms_to_genenv(self):
        env = self.env(cameras={"top": (4, 6), "wrist": (2, 3)})
        self.assertIsInstance(env, GenEnv)
        self.assertEqual(env.fps, 30)
        self.assertEqual(tuple(env.joint_names), tuple(self.rig.chain.q_names))
        self.assertEqual(tuple(env.camera_names), ("top", "wrist"))

    def test_reset_applies_spec(self):
        env = self.env()
        spec = self.make_spec()
        obs = env.reset(spec)
        self.assertEqual(obs.frame, 0)
        np.testing.assert_array_equal(obs.q, np.array(spec.initial_q))
        self.assertEqual(obs.objects["cube"].pose.position, spec.objects["cube"].pose.position)
        self.assertEqual(obs.objects["cube"].color, "red")
        self.assertTrue(obs.objects["box"].active)
        self.assertFalse(obs.signals["holding"])
        for key in ("holding", "grasped", "lifted", "placed", "success"):
            self.assertIn(key, obs.signals)

    def test_inactive_objects_stay_parked(self):
        env = self.env()
        park = (1.0, 2.0, -3.0)
        obs = env.reset(self.make_spec(cube=park, cube_active=False))
        self.assertFalse(obs.objects["cube"].active)
        for _ in range(3):
            obs = env.step(self.home)
        self.assertEqual(tuple(obs.objects["cube"].pose.position), park)

    def test_reset_errors(self):
        env = self.env()
        spec = self.make_spec()
        bad = dataclasses.replace(spec, objects=dict(spec.objects, ghost=ObjectState(Pose((0.0, 0.0, 0.0)))))
        with self.assertRaises(EnvError):
            env.reset(bad)
        with self.assertRaises(EnvError):
            env.reset(dataclasses.replace(spec, initial_q=tuple(spec.initial_q[:-1])))

    def test_render(self):
        env = self.env(cameras={"top": (4, 6)})
        env.reset(self.make_spec())
        img = env.render()
        self.assertEqual(set(img), {"top"})
        self.assertEqual(img["top"].shape, (4, 6, 3))
        self.assertEqual(img["top"].dtype, np.uint8)
        env2 = self.env()
        env2.reset(self.make_spec())
        self.assertEqual(env2.render(), {})

    def test_simulated_crash_is_not_a_generation_error(self):
        """§11 SimulatedCrash is never caught by run_generation, so it is not a GenerationError."""
        self.assertTrue(issubclass(SimulatedCrash, Exception))
        self.assertFalse(issubclass(SimulatedCrash, GenerationError))

    def test_crash_at_step(self):
        env = self.env(crash_at_step=3)
        env.reset(self.make_spec())
        for _ in range(3):                          # calls 0, 1, 2
            env.step(self.home)
        env.reset(self.make_spec())                 # the count continues across resets: the next call is 3
        before = env.observe()
        with self.assertRaises(SimulatedCrash):
            env.step(self.home + 0.3)
        self.assertEqual(env.observe().frame, before.frame)
        np.testing.assert_array_equal(env.observe().q, before.q)

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        targets = [self.home + rng.normal(0, 0.2, self.home.shape) for _ in range(20)]
        runs = []
        for _ in range(2):
            env = self.env()
            obs = [env.reset(self.make_spec())] + [env.step(t) for t in targets]
            runs.append([(o.q.tobytes(), o.ee.position, o.tcp.quat_xyzw, o.gripper_gap_m,
                          {n: s.pose.position for n, s in o.objects.items()}, dict(o.signals)) for o in obs])
        self.assertEqual(runs[0], runs[1])


class JointTests(FakeEnvBase):
    """WP7 §11 step 1–2: first-order tracking to the clipped target; ee/TCP from the chain; gap from the angle."""

    def test_tracking_formula(self):
        lo, hi = self.rig.chain.lower, self.rig.chain.upper
        for alpha in (0.5, 1.0, 0.2):
            env = self.env(alpha=alpha)
            spec = self.make_spec(cube=(5.0, 5.0, 0.0))      # far away: no attach, no push
            obs = env.reset(spec)
            q = np.array(spec.initial_q)
            rng = np.random.default_rng(int(alpha * 10))
            for t in range(25):
                target = q + rng.normal(0, 0.8, q.shape)
                obs = env.step(target)
                q = q + alpha * (np.clip(target, lo, hi) - q)
                with self.subTest(alpha=alpha, t=t):
                    np.testing.assert_allclose(obs.q, q, rtol=0, atol=1e-12)
                    self.assertEqual(obs.frame, t + 1)

    def test_poses_from_chain(self):
        env = self.env(alpha=1.0)
        obs = env.reset(self.make_spec(cube=(5.0, 5.0, 0.0)))
        rng = np.random.default_rng(4)
        for t in range(10):
            obs = env.step(obs.q + rng.normal(0, 0.3, obs.q.shape))
            for frame, pose in (("tcp", obs.tcp), ("ee", obs.ee)):
                T = self.rig.chain.fk(obs.q, frame)
                with self.subTest(t=t, frame=frame):
                    np.testing.assert_allclose(pose.position, T[:3, 3], atol=1e-9)
                    np.testing.assert_allclose(rot(pose.quat_xyzw), T[:3, :3], atol=1e-9)
            self.assertAlmostEqual(obs.gripper_gap_m, self.rig.gripper.gap_from_angle(float(obs.q[self.g])), places=12)


class GraspTests(FakeEnvBase):
    """WP7 §11 steps 3–5: attach on closing inside the jaw box, rigid carry, drop to the support below."""

    def closed_on_cube(self, *, offset=(0.0, 0.0, 0.0)):
        """Reset with the cube centred at the open TCP (+ offset in the TCP frame), then close in one step."""
        q = self.q_with_gap(0.04)
        T = self.rig.chain.fk(q, "tcp")
        c = T[:3, 3] + T[:3, :3] @ np.asarray(offset, float)
        env = self.env(alpha=1.0)
        env.reset(self.make_spec(cube=tuple(c), q=q))
        obs = env.step(self.q_with_gap(0.0, q))
        return env, obs, q

    def test_attach_inside_jaw_box(self):
        env, obs, q = self.closed_on_cube()
        self.assertTrue(obs.signals["holding"])
        self.assertTrue(obs.signals["grasped"])
        self.assertAlmostEqual(obs.gripper_gap_m, CUBE, places=9)
        # the gripper cannot close past the object while it is held
        obs = env.step(self.q_with_gap(0.0, q))
        self.assertAlmostEqual(obs.gripper_gap_m, CUBE, places=9)

    def test_jaw_box_boundary(self):
        """Offsets in the TCP frame: radial > w/2 or |z| > h/2 never attach; inside both always attaches."""
        for off in ((0.0, 0.0105, 0.0), (0.0105, 0.0, 0.0), (0.008, -0.008, 0.0), (0.0, 0.0, 0.0105),
                    (0.0, 0.0, -0.0105)):
            with self.subTest(offset=off):
                _, obs, _ = self.closed_on_cube(offset=off)
                self.assertFalse(obs.signals["holding"])
        for off in ((0.006, 0.006, 0.0095), (0.0, 0.0095, -0.0095), (-0.007, 0.0, 0.0)):
            with self.subTest(offset=off):
                _, obs, _ = self.closed_on_cube(offset=off)
                self.assertTrue(obs.signals["holding"])

    def test_no_attach_when_gap_does_not_decrease(self):
        q = self.q_with_gap(0.021)
        env = self.env(alpha=1.0)
        env.reset(self.make_spec(cube=self.at_tcp(q), q=q))
        obs = env.step(q)
        self.assertFalse(obs.signals["holding"])

    def test_rigid_carry_and_lift_signal(self):
        env, obs, q = self.closed_on_cube()
        start = np.array(obs.objects["cube"].pose.position)
        rel = np.array(obs.objects["cube"].pose.position) - np.array(obs.tcp.position)
        Rt = rot(obs.tcp.quat_xyzw).T @ rel
        q2 = q.copy()
        q2[1] -= 0.25                                  # lift joint: the TCP rises
        q2[self.g] = self.rig.gripper.angle_from_gap(0.0)
        obs = env.step(q2)
        self.assertTrue(obs.signals["holding"])
        now = rot(obs.tcp.quat_xyzw) @ Rt + np.array(obs.tcp.position)
        np.testing.assert_allclose(obs.objects["cube"].pose.position, now, atol=1e-9)
        rise = obs.objects["cube"].pose.position[2] - start[2]
        self.assertEqual(obs.signals["lifted"], rise >= 0.01)

    def test_drop_to_table(self):
        env, obs, q = self.closed_on_cube()
        before = obs.objects["cube"].pose
        obs = env.step(self.q_with_gap(0.04, q))
        self.assertFalse(obs.signals["holding"])
        p = obs.objects["cube"].pose
        self.assertAlmostEqual(p.position[0], before.position[0], places=9)
        self.assertAlmostEqual(p.position[1], before.position[1], places=9)
        self.assertAlmostEqual(p.position[2], TABLE_Z + CUBE / 2, places=9)
        self.assertAlmostEqual(abs(p.quat_xyzw[0]) + abs(p.quat_xyzw[1]), 0.0, places=9)   # roll, pitch = 0

    def test_held_object_at_reset_and_drop_into_box(self):
        """§11 reset attaches ``held_object`` at the TCP with gap w; opening drops it onto the container floor."""
        q = self.q_with_gap(0.04)
        tcp = self.at_tcp(q)
        env = self.env(alpha=1.0)
        obs = env.reset(self.make_spec(held="cube", q=q, box=(tcp[0], tcp[1], TABLE_Z)))
        self.assertTrue(obs.signals["holding"])
        self.assertAlmostEqual(obs.gripper_gap_m, CUBE, places=9)
        np.testing.assert_allclose(obs.objects["cube"].pose.position, tcp, atol=1e-9)
        obs = env.step(self.q_with_gap(0.045, q))
        floor = TABLE_Z + self.pick.spec.objects["box"].params["floor_z"]
        self.assertAlmostEqual(obs.objects["cube"].pose.position[2], floor + CUBE / 2, places=9)
        self.assertTrue(obs.signals["placed"])

    def test_drop_onto_another_object(self):
        """§11 supports include the top of any active object under the xy (a centre-origin cube here)."""
        q = self.q_with_gap(0.04)
        tcp = self.at_tcp(q)
        env = self.env(alpha=1.0)
        # the box (bottom origin, 0.045 high) is the object under the held cube, but outside its inner rectangle
        inner = self.pick.spec.objects["box"].params["inner_half_size"]
        box = (tcp[0] - inner[0] - 0.001, tcp[1], TABLE_Z)        # on the 1.9 mm wall
        obs = env.reset(self.make_spec(held="cube", q=q, box=box))
        obs = env.step(self.q_with_gap(0.045, q))
        top = TABLE_Z + self.pick.spec.objects["box"].params["size_outer"][2]
        self.assertAlmostEqual(obs.objects["cube"].pose.position[2], top + CUBE / 2, places=9)
        self.assertFalse(obs.signals["placed"])

    def test_success_after_confirm_steps(self):
        q = self.q_with_gap(0.04)
        tcp = self.at_tcp(q)
        env = self.env(alpha=1.0)
        env.reset(self.make_spec(held="cube", q=q, box=(tcp[0], tcp[1], TABLE_Z)))
        confirm = self.pick.spec.raw["success"]["confirm_steps"]
        obs = env.step(self.q_with_gap(0.045, q))                   # placed on this frame: streak 1
        for k in range(2, confirm):
            obs = env.step(self.q_with_gap(0.045, q))
            self.assertTrue(obs.signals["placed"])
            self.assertFalse(obs.signals["success"], k)
        obs = env.step(self.q_with_gap(0.045, q))
        self.assertTrue(obs.signals["success"])


class TeleportPushTests(FakeEnvBase):
    """WP7 §11: set_object_pose is exact and detaches; the TCP pushes objects it passes through."""

    def test_set_object_pose_detaches(self):
        q = self.q_with_gap(0.04)
        env = self.env(alpha=1.0)
        obs = env.reset(self.make_spec(held="cube", q=q))
        self.assertTrue(obs.signals["holding"])
        target = Pose((0.12, 0.1, 0.3), yaw_quat(0.7))
        env.set_object_pose("cube", target)
        obs = env.observe()
        self.assertFalse(obs.signals["holding"])
        self.assertEqual(tuple(obs.objects["cube"].pose.position), target.position)     # no drop
        np.testing.assert_allclose(obs.objects["cube"].pose.quat_xyzw, target.quat_xyzw, atol=1e-12)
        self.assertEqual(obs.frame, 0)
        with self.assertRaises(EnvError):
            env.set_object_pose("ghost", target)

    def test_push_moves_object_with_tcp(self):
        q = self.q_with_gap(0.045)
        env = self.env(alpha=1.0)
        obs = env.reset(self.make_spec(cube=self.at_tcp(q), q=q, yaw=0.0))
        c0 = np.array(obs.objects["cube"].pose.position)
        t0 = np.array(obs.tcp.position)
        q2 = q.copy()
        q2[0] += 0.02                                   # pan: horizontal TCP motion, gap unchanged
        obs = env.step(q2)
        self.assertFalse(obs.signals["holding"])
        d = np.array(obs.tcp.position) - t0
        np.testing.assert_allclose(np.array(obs.objects["cube"].pose.position)[:2], c0[:2] + d[:2], atol=1e-12)
        self.assertAlmostEqual(obs.objects["cube"].pose.position[2], c0[2], places=12)

    def test_containers_are_not_pushed(self):
        q = self.q_with_gap(0.045)
        tcp = self.at_tcp(q)
        env = self.env(alpha=1.0)
        box = (tcp[0], tcp[1], tcp[2] - 0.02)
        obs = env.reset(self.make_spec(q=q, box=box))
        q2 = q.copy()
        q2[0] += 0.02
        obs = env.step(q2)
        self.assertEqual(tuple(obs.objects["box"].pose.position), box)


if __name__ == "__main__":
    unittest.main()
