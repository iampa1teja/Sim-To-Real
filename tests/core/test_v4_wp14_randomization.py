"""WP14: wide randomisation of generated episodes (contracts §16; docs/v4/prompts/WP14.md).

Every generated episode now draws, from its attempt seed:

* object and goal positions anywhere in the arm's table workspace (``[arm.workspace]``, §16.1), containers included;
* object sizes from ``size_variants`` (§16.2);
* the arm's start pose around home (``initial_q = {around = "home", spread}``, §16.3);
* expert motion parameters written as ``{range = [lo, hi]}`` / ``{choice = [...]}`` and a ``speed_scale`` (§16.4);
* object mass and friction (``[task.physics]``, §16.5);
* and DR copies draw camera pose offsets (§16.7).

Starts closer than one 5 mm / 5° cell to an earlier start are skipped as near-duplicates (§16.6). The task tables
of all 17 tasks use all of this (§16.9). The 5-episodes-per-task smoke run is tests/core/test_generation_all_tasks.py.
"""

import contextlib
import dataclasses
import json
import math
import unittest

import numpy as np

from core.generate.errors import ConfigError, SamplingError
from core.generate.protocols import EpisodeSpec, Kind, ObjectState, SkipReason
from core.tasks.base import Pose, load_task_spec
from v4_support import REPO_ROOT, sandbox
from v4_support import harness as H
from v4_support.tasks import write_task

ARM = "fx_arm5l"
WS = {"r_m": [0.15, 0.35], "bearing_deg": [-60.0, 60.0]}
CUBE = {"size": [0.02, 0.02, 0.02], "mass": 0.008}
VARIANTS5 = [[0.018, 0.018, 0.018], [0.02, 0.02, 0.02], [0.022, 0.022, 0.022], [0.024, 0.024, 0.024],
             [0.026, 0.026, 0.026]]
ALL_TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube", "stack_bowls",
             "prim_joint_sweep", "prim_gripper_gap", "prim_reach_point", "prim_test_reach", "prim_calibration_reach",
             "prim_waypoint_path", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
             "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")
OBJECT_TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube", "stack_bowls", "prim_test_reach",
                "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test", "prim_grasp_lift",
                "prim_place_at", "prim_regrasp_after_drop")
CUBE_TASKS = tuple(t for t in OBJECT_TASKS if t != "stack_bowls")
CONTAINERS = {"pick_place_cube": "white_box", "pick_place_color_cube": "small_box", "stack_bowls": "large_bowl"}
VARIANTS = {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"}


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _in_ws(xy, ws, tol=1e-9):
    r, b = math.hypot(xy[0], xy[1]), math.degrees(math.atan2(xy[1], xy[0]))
    lo, hi = ws["bearing_deg"]
    b = lo + (b - lo) % 360.0
    ok = ws["r_m"][0] - tol <= r <= ws["r_m"][1] + tol and b <= hi + tol
    if "xy_min" in ws:
        ok = ok and all(ws["xy_min"][k] - tol <= xy[k] <= ws["xy_max"][k] + tol for k in range(2))
    return ok


def _inset(ws, d):
    out = {"r_m": [ws["r_m"][0] + d, ws["r_m"][1] - d], "bearing_deg": list(ws["bearing_deg"])}
    if "xy_min" in ws:
        out["xy_min"] = [v + d for v in ws["xy_min"]]
        out["xy_max"] = [v - d for v in ws["xy_max"]]
    return out


class _Sandboxed(unittest.TestCase):
    TASKS = ()

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, cls.TASKS)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def spec(self, name, table):
        write_task(self.box.tasks, name, dict({"variants": VARIANTS}, **table))
        return load_task_spec(name)


# ---------------------------------------------------------------------------------------------------- §16.1

class WorkspaceTests(_Sandboxed):
    """WP14 §16.1 ``[arm.workspace]`` and ``region = "workspace"`` / ``{workspace_inset_m}``."""

    def test_validate_workspace(self):
        from core.tasks.sampling import validate_workspace
        self.assertEqual(validate_workspace(dict(WS)), WS)
        ok = dict(WS, xy_min=[-0.3, -0.4], xy_max=[0.3, 0.1])
        self.assertEqual(validate_workspace(ok), ok)
        for bad in ({"r_m": [0.15, 0.35]}, dict(WS, r_m=[0.35, 0.15]), dict(WS, r_m=[-0.1, 0.3]),
                    dict(WS, bearing_deg=[10.0, 10.0]), dict(WS, xy_min=[0.0, 0.0]),
                    dict(WS, xy_min=[0.1, 0.0], xy_max=[0.0, 0.1]), dict(WS, extra=1)):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                validate_workspace(bad)

    def test_in_workspace(self):
        from core.tasks.sampling import in_workspace
        ws = {"r_m": [0.15, 0.40], "bearing_deg": [-182.0, 2.0]}
        for xy, inside in (((0.0, -0.3), True), ((-0.3, 0.005), True), ((0.3, 0.005), True), ((0.0, 0.3), False),
                           ((0.0, -0.1), False), ((0.0, -0.45), False), ((-0.3, 0.02), False)):
            with self.subTest(xy=xy):
                self.assertEqual(in_workspace(xy, ws), inside)
        clipped = dict(ws, xy_min=[-0.2, -0.42], xy_max=[0.3, 0.03])
        self.assertFalse(in_workspace((-0.25, -0.1), clipped))
        self.assertTrue(in_workspace((-0.15, -0.2), clipped))

    def test_sampled_positions_cover_the_workspace_uniformly(self):
        """Positions are uniform over the annulus area: half of them fall inside the equal-area radius."""
        from core.tasks.sampling import sample_start
        spec = self.spec("fx14_ws", {"objects": {"cube": CUBE},
                                     "start": {"place": {"cube": {"region": "workspace", "yaw_deg": [0.0, 90.0]}}}})
        pts = np.array([sample_start(spec, np.random.default_rng(k), workspace=WS).objects["cube"].pose.position
                        for k in range(1500)])
        for p in pts:
            self.assertTrue(_in_ws(p[:2], WS))
        r = np.hypot(pts[:, 0], pts[:, 1])
        r_half = math.sqrt((WS["r_m"][0] ** 2 + WS["r_m"][1] ** 2) / 2)
        self.assertAlmostEqual(float(np.mean(r <= r_half)), 0.5, delta=0.05, msg="radius must be area-uniform")
        b = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        self.assertAlmostEqual(float(np.mean(b <= 0.0)), 0.5, delta=0.05)
        np.testing.assert_allclose(pts[:, 2], 0.01)

    def test_clip_and_inset(self):
        from core.tasks.sampling import sample_start
        ws = dict(WS, xy_min=[0.0, -0.1], xy_max=[0.4, 0.1])
        spec = self.spec("fx14_inset", {"objects": {"cube": CUBE},
                                        "start": {"place": {"cube": {"region": {"workspace_inset_m": 0.03}}}}})
        for k in range(300):
            p = sample_start(spec, np.random.default_rng(k), workspace=ws).objects["cube"].pose.position
            self.assertTrue(_in_ws(p[:2], _inset(ws, 0.03)), p)
        with self.assertRaises((ConfigError, SamplingError)):
            bad = self.spec("fx14_inset_bad", {"objects": {"cube": CUBE},
                                               "start": {"place": {"cube": {"region": {"workspace_inset_m": 0.2}}}}})
            sample_start(bad, np.random.default_rng(0), workspace=ws)
        with self.assertRaises(ConfigError):
            self.spec("fx14_inset_neg", {"objects": {"cube": CUBE},
                                         "start": {"place": {"cube": {"region": {"workspace_inset_m": -0.01}}}}})

    def test_workspace_region_needs_a_workspace(self):
        from core.tasks.sampling import sample_start
        spec = self.spec("fx14_nows", {"objects": {"cube": CUBE}, "start": {"place": {"cube": {"region": "workspace"}}}})
        with self.assertRaises(ValueError):
            sample_start(spec, np.random.default_rng(0))

    def test_goal_regions(self):
        """point / points use the workspace; a region goal's centre may be relative to an object."""
        from core.tasks.sampling import sample_start
        spec = self.spec("fx14_goals", {
            "objects": {"cube": CUBE}, "roles": {"target": "cube"},
            "start": {"place": {"cube": {"region": {"workspace_inset_m": 0.05}}},
                      "goal": {"p": {"type": "point", "region": "workspace", "z_m": [0.05, 0.1]},
                               "ps": {"type": "points", "count": 3, "region": "workspace", "z_m": [0.05, 0.1],
                                      "min_separation_m": 0.05},
                               "area": {"type": "region", "half_size_m": [0.01, 0.02],
                                        "center_region": {"around": "@target", "distance_m": [0.03, 0.06]}}}}})
        for k in range(200):
            s = sample_start(spec, np.random.default_rng(k), workspace=WS)
            cube = np.array(s.objects["cube"].pose.position[:2])
            with self.subTest(k=k):
                self.assertTrue(_in_ws(s.goal["p"][:2], WS))
                for q in s.goal["ps"]:
                    self.assertTrue(_in_ws(q[:2], WS))
                lo, hi = np.array(s.goal["area"]["xy_min"]), np.array(s.goal["area"]["xy_max"])
                np.testing.assert_allclose(hi - lo, [0.02, 0.04], atol=1e-12)
                centre = (lo + hi) / 2
                self.assertTrue(0.03 - 1e-9 <= np.linalg.norm(centre - cube) <= 0.06 + 1e-9)
                self.assertTrue(_in_ws(centre, WS))
        with self.assertRaises(ConfigError):
            self.spec("fx14_goal_bad", {"objects": {"cube": CUBE}, "roles": {"target": "cube"},
                                        "start": {"goal": {"a": {"type": "region", "half_size_m": [0.01, 0.01],
                                                                 "center_region": {"around": "@nobody",
                                                                                   "distance_m": [0.03, 0.06]}}}}})

    def test_test_arm_and_so101_have_workspaces(self):
        from core.arms.spec import load_arm_spec
        from core.tasks.sampling import validate_workspace
        validate_workspace(load_arm_spec(ARM)[0]["workspace"])
        import tomllib
        so101 = tomllib.loads((REPO_ROOT / "arms" / "so101" / "arm.toml").read_text())["arm"]
        ws = validate_workspace(so101["workspace"])
        # The container reach check measured 11 cm of radial depth for the SO-101.
        # Preserve that depth, at least 90 degrees of bearing, and coverage on both sides.
        min_depth_m = 0.11
        self.assertGreaterEqual(ws["r_m"][1] - ws["r_m"][0], min_depth_m)
        self.assertGreaterEqual(ws["bearing_deg"][1] - ws["bearing_deg"][0], 90.0)
        grid = [(x / 100, y / 100) for x in range(-45, 46) for y in range(-45, 46)]
        inside = [p for p in grid if _in_ws(p, ws)]
        self.assertTrue(any(p[0] <= -0.1 for p in inside) and any(p[0] >= 0.1 for p in inside), "both sides of the base")
        depths = sorted(math.hypot(*p) for p in inside)
        self.assertGreaterEqual(depths[-1] - depths[0], min_depth_m)


# ---------------------------------------------------------------------------------------------------- §16.2

class SizeVariantTests(_Sandboxed):
    """WP14 §16.2 ``size_variants`` → ``ObjectState.size`` and every consumer of object sizes."""

    TASKS = ("prim_grasp_lift",)

    def test_validation(self):
        for name, obj in (("fx14_sv_one", dict(CUBE, size_variants=[[0.02, 0.02, 0.02]])),
                          ("fx14_sv_neg", dict(CUBE, size_variants=[[0.02, 0.02, 0.02], [0.02, -0.02, 0.02]])),
                          ("fx14_sv_two", dict(CUBE, size_variants=[[0.02, 0.02], [0.03, 0.03]]))):
            with self.subTest(name=name), self.assertRaises(ConfigError):
                self.spec(name, {"objects": {"cube": obj}})

    def test_drawn_sizes(self):
        from core.tasks.sampling import sample_start
        spec = self.spec("fx14_sv", {"objects": {"cube": dict(CUBE, size_variants=VARIANTS5)},
                                     "object_sets": {"balls": {"prefix": "ball", "count_max": 2, "size": [0.03, 0.03, 0.03],
                                                               "size_variants": [[0.03, 0.03, 0.03], [0.04, 0.04, 0.04]]}},
                                     "start": {"place": {"cube": {"region": "workspace"}, "balls": {"region": "workspace"}},
                                               "min_separation_m": 0.05}})
        seen = set()
        for k in range(200):
            s = sample_start(spec, np.random.default_rng(k), workspace=WS)
            cube = s.objects["cube"]
            self.assertIn(list(cube.size), VARIANTS5)
            self.assertAlmostEqual(cube.pose.position[2], cube.size[2] / 2, places=12)
            for b in ("ball_0", "ball_1"):
                self.assertIn(list(s.objects[b].size), [[0.03] * 3, [0.04] * 3])
            seen.add(cube.size)
        self.assertEqual(len(seen), 5, "every variant is drawn")
        plain = self.spec("fx14_sv_none", {"objects": {"cube": CUBE}, "start": {"place": {"cube": {"region": "workspace"}}}})
        self.assertIsNone(sample_start(plain, np.random.default_rng(0), workspace=WS).objects["cube"].size)

    def test_object_state_json(self):
        a = ObjectState(Pose((0.1, 0.0, 0.01)), True, "red")
        self.assertNotIn("size", a.to_json(), "size is left out while it is None (§16.8)")
        b = ObjectState(Pose((0.1, 0.0, 0.01)), True, "red", (0.024, 0.024, 0.024))
        self.assertEqual(b.to_json()["size"], [0.024, 0.024, 0.024])
        self.assertEqual(ObjectState.from_json(b.to_json()), b)
        self.assertEqual(ObjectState.from_json(a.to_json()), a)
        for bad in ((0.02, 0.02), (0.02, 0.0, 0.02), (0.02, float("nan"), 0.02)):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                ObjectState(Pose((0.0, 0.0, 0.0)), True, None, bad)

    def test_episode_objects(self):
        from core.tasks.base import CheckContext, episode_objects
        from core.tasks.loader import load_task
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        task = load_task("prim_grasp_lift")
        fn = make_spec_fn(task, ARM, scene=H.SCENE)
        s = fn(Kind.SUCCESS, 0, attempt_seed(0, "prim_grasp_lift", Kind.SUCCESS, 0))
        objs = episode_objects(task.spec, s)
        self.assertEqual(objs["cube"].size, s.objects["cube"].size)
        self.assertEqual(task.spec.objects["cube"].size, (0.02, 0.02, 0.02), "the task spec itself is unchanged")
        self.assertEqual(CheckContext.from_spec(s, task.spec).objects["cube"].size, s.objects["cube"].size)

    def test_fake_env_and_expert_use_the_drawn_size(self):
        """A large cube is grasped at its own width and held at its own half height."""
        from core.generate.fake_env import FakeEnv
        from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
        from core.generate.plan import attempt_seed
        from core.tasks.loader import load_task
        from v4_support.stubs import ListWriter
        task, rig = load_task("prim_grasp_lift"), H.Rig(ARM)
        fn, fac = make_spec_fn(task, ARM, scene=H.SCENE), make_expert_factory(task, ARM)
        done = 0
        for i in range(30):
            s = fn(Kind.SUCCESS, i, attempt_seed(0, "prim_grasp_lift", Kind.SUCCESS, i))
            if s.noise is not None:
                continue
            for size in ((0.026, 0.026, 0.026), (0.018, 0.018, 0.018)):
                objects = dict(s.objects, cube=dataclasses.replace(
                    s.objects["cube"], size=size,
                    pose=Pose(s.objects["cube"].pose.position[:2] + (size[2] / 2,), s.objects["cube"].pose.quat_xyzw)))
                spec = dataclasses.replace(s, objects=objects)
                env = FakeEnv(task, rig.chain, rig.gripper)
                r, rec = run_episode(spec, task, env, fac(spec), ListWriter(), rig.mapping)
                with self.subTest(i=i, size=size):
                    self.assertEqual(r.status.value, "accepted")
                    self.assertLess(abs(env.observe().gripper_gap_m - size[0]), 0.004, "closed on the drawn width")
            done += 1
            if done == 3:
                break
        self.assertEqual(done, 3)


# ---------------------------------------------------------------------------------------------------- §16.3

class InitialPoseTests(_Sandboxed):
    """WP14 §16.3 ``initial_q = {around = "home", spread, min_tcp_height_m, min_tcp_clearance_m}``."""

    TASKS = ("prim_reach_point", "prim_grasp_lift")

    def test_validation(self):
        for bad in ({"around": "zero", "spread": 0.1}, {"around": "home", "spread": 0.0},
                    {"around": "home", "spread": 0.6}, {"around": "home"}, {"around": "home", "spread": 0.1, "x": 1},
                    {"around": "home", "spread": 0.1, "min_tcp_height_m": -0.01}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                self.spec(f"fx14_q_{abs(hash(json.dumps(bad))) % 10**8}", {"start": {"initial_q": bad}})

    def _arm(self):
        from core.arms.spec import load_arm_spec
        from core.expert.motion import expert_joint_limits
        from core.generate.loop import home_q
        spec = load_arm_spec(ARM)[0]
        rig = H.Rig(ARM)
        return (np.array(home_q(ARM)), expert_joint_limits(ARM), tuple(spec["joints"]["names"]),
                tuple(spec["expert"]["arm_joints"]), rig)

    def test_draws(self):
        from core.tasks.sampling import sample_start
        home, (lo, hi), names, arm, rig = self._arm()
        spec = self.spec("fx14_q", {"objects": {"cube": CUBE},
                                    "start": {"initial_q": {"around": "home", "spread": 0.15, "min_tcp_height_m": 0.06,
                                                            "min_tcp_clearance_m": 0.07},
                                              "place": {"cube": {"region": "workspace"}}}})
        ws = __import__("core.arms.spec", fromlist=["x"]).load_arm_spec(ARM)[0]["workspace"]
        tcp_fn = lambda q: rig.chain.fk_pose(np.asarray(q, float), "tcp")
        qs = []
        for k in range(150):
            s = sample_start(spec, np.random.default_rng(k), home_q=home, joint_limits=(lo, hi), joint_names=names,
                             arm_joints=arm, workspace=ws, tcp_fn=tcp_fn)
            q = np.array(s.initial_q)
            with self.subTest(k=k):
                self.assertTrue(np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12))
                for j, n in enumerate(names):
                    if n in arm:
                        self.assertLessEqual(abs(q[j] - home[j]), 0.15 * (hi[j] - lo[j]) + 1e-12)
                    else:
                        self.assertEqual(q[j], home[j], "the gripper starts at home")
                tcp = np.array(tcp_fn(q).position)
                self.assertGreaterEqual(tcp[2], 0.06 - 1e-12)
                cube = s.objects["cube"].pose.position
                self.assertGreaterEqual(math.hypot(tcp[0] - cube[0], tcp[1] - cube[1]), 0.07 - 1e-12)
            qs.append(q)
        qs = np.array(qs)
        for j, n in enumerate(names):
            if n in arm:
                self.assertGreater(qs[:, j].std(), 0.03 * (hi[j] - lo[j]), f"{n} barely varies")
        with self.assertRaises(ValueError):
            sample_start(spec, np.random.default_rng(0), home_q=home, joint_limits=(lo, hi), joint_names=names,
                         arm_joints=arm, workspace=ws)

    def test_build_spec_initial_q(self):
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        from core.tasks.loader import load_task
        home = np.array(self._arm()[0])
        for name in ("prim_reach_point", "prim_grasp_lift"):
            fn = make_spec_fn(load_task(name), ARM, scene=H.SCENE)
            qs = {tuple(np.round(fn(Kind.SUCCESS, i, attempt_seed(0, name, Kind.SUCCESS, i)).initial_q, 6))
                  for i in range(20)}
            with self.subTest(task=name):
                self.assertEqual(len(qs), 20, "every episode starts from its own pose")
                self.assertTrue(all(not np.allclose(q, home) for q in qs))


# ---------------------------------------------------------------------------------------------------- §16.4

class ExpertVariationTests(_Sandboxed):
    """WP14 §16.4 varied expert parameters and ``speed_scale``."""

    TASKS = ("prim_grasp_lift", "prim_reach_point")

    def _parse(self, steps, **extra):
        from core.arms.spec import load_arm_spec
        from core.expert.motion import expert_joint_limits
        from core.expert.program import parse_program
        from core.tasks.loader import load_task
        task = load_task("prim_grasp_lift")
        return parse_program(dict({"steps": steps}, **extra), task=task, arm_spec=load_arm_spec(ARM)[0],
                             joint_limits=expert_joint_limits(ARM))

    def test_parse(self):
        prog = self._parse([{"skill": "approach", "target": "@target", "offset_m": [0, 0, {"range": [0.04, 0.08]}],
                             "stage": "reached"},
                            {"skill": "lift", "height_m": {"range": [0.07, 0.1]}},
                            {"skill": "release", "dwell_steps": {"range": [5, 12]}},
                            {"skill": "retreat", "dz_m": {"choice": [0.05, 0.08]}}],
                           speed_scale={"range": [0.6, 1.0]})
        self.assertAlmostEqual(prog.speed_scale, 0.8)
        self.assertEqual(list(prog.steps[0].params["offset_m"]), [0, 0, 0.06])
        self.assertAlmostEqual(prog.steps[1].params["height_m"], 0.085)
        self.assertEqual(prog.steps[2].params["dwell_steps"], 5)
        self.assertEqual(prog.steps[3].params["dz_m"], 0.05)
        self.assertEqual(self._parse([{"skill": "lift"}]).speed_scale, 1.0)

    def test_parse_rejects(self):
        cases = {
            "lo>hi": ([{"skill": "lift", "height_m": {"range": [0.1, 0.05]}}], {}),
            "end_invalid": ([{"skill": "lift", "height_m": {"range": [-0.01, 0.05]}}], {}),
            "gap_end_outside_table": ([{"skill": "set_gripper", "gap_m": {"range": [0.01, 0.2]}}], {}),
            "choice_invalid": ([{"skill": "set_gripper", "gap_m": {"choice": [0.01, 0.5]}}], {}),
            "int_param_float_range": ([{"skill": "release", "dwell_steps": {"range": [2.5, 5]}}], {}),
            "varied_stage": ([{"skill": "lift", "stage": {"choice": ["lifted", "grasped"]}}], {}),
            "varied_budget": ([{"skill": "lift", "budget": {"range": [10, 20]}}], {}),
            "unknown_form": ([{"skill": "lift", "height_m": {"normal": [0.08, 0.01]}}], {}),
            "speed_above_1": ([{"skill": "lift"}], {"speed_scale": {"range": [0.6, 1.2]}}),
            "speed_zero": ([{"skill": "lift"}], {"speed_scale": 0.0}),
        }
        for name, (steps, extra) in cases.items():
            with self.subTest(case=name), self.assertRaises(ConfigError):
                self._parse(steps, **extra)

    def test_draw(self):
        from core.generate.specs import draw_expert_params
        table = {"speed_scale": {"range": [0.6, 1.0]},
                 "steps": [{"skill": "set_gripper", "gap_m": "open"},
                           {"skill": "approach", "target": "@target", "offset_m": [0, 0, {"range": [0.04, 0.08]}]},
                           {"skill": "release", "dwell_steps": {"range": [5, 12]}},
                           {"skill": "retreat", "dz_m": {"choice": [0.05, 0.08]}}]}
        draws = [draw_expert_params(table, np.random.default_rng(k)) for k in range(300)]
        self.assertEqual(draws[7], draw_expert_params(table, np.random.default_rng(7)))
        rng = np.random.default_rng(3)          # draw order: speed_scale, then steps in order, keys sorted
        expect = {"speed_scale": float(rng.uniform(0.6, 1.0)),
                  "steps": {"1": {"offset_m": [0, 0, float(rng.uniform(0.04, 0.08))]},
                            "2": {"dwell_steps": int(rng.integers(5, 13))},
                            "3": {"dz_m": [0.05, 0.08][int(rng.integers(2))]}}}
        self.assertEqual(draws[3], expect)
        for d in draws:
            self.assertTrue(0.6 <= d["speed_scale"] <= 1.0)
            self.assertEqual(set(d["steps"]), {"1", "2", "3"})
            self.assertTrue(0.04 <= d["steps"]["1"]["offset_m"][2] <= 0.08)
            self.assertIs(type(d["steps"]["2"]["dwell_steps"]), int)
        self.assertEqual({d["steps"]["2"]["dwell_steps"] for d in draws}, set(range(5, 13)))
        self.assertEqual(draw_expert_params({"steps": [{"skill": "lift", "height_m": 0.08}]}, np.random.default_rng(0)), {})

    def test_spec_carries_draws(self):
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        from core.generate.seeds import rng_for
        from core.generate.specs import draw_expert_params
        from core.tasks.loader import load_task
        task = load_task("prim_grasp_lift")
        fn = make_spec_fn(task, ARM, scene=H.SCENE)
        seed = attempt_seed(0, "prim_grasp_lift", Kind.SUCCESS, 4)
        s = fn(Kind.SUCCESS, 4, seed)
        self.assertEqual(json.loads(json.dumps(s.to_json()["expert_params"])),
                         json.loads(json.dumps(draw_expert_params(task.spec.raw["expert"], rng_for(seed, "expert_params")))))
        self.assertEqual(EpisodeSpec.from_json(s.to_json()), s)
        old = s.to_json()
        del old["expert_params"], old["physics"]
        self.assertEqual(EpisodeSpec.from_json(old).expert_params, {}, "earlier payloads still load")

    def _run(self, spec, task):
        from core.generate.fake_env import FakeEnv
        from core.generate.loop import make_expert_factory, run_episode
        from v4_support.stubs import ListWriter
        rig = H.Rig(ARM)
        env = FakeEnv(task, rig.chain, rig.gripper)
        r, rec = run_episode(spec, task, env, make_expert_factory(task, ARM)(spec), ListWriter(), rig.mapping)
        return r, rec, env, rig

    def test_expert_follows_draws(self):
        """A drawn lift height is the height reached; a lower speed_scale moves the TCP no faster than it allows."""
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        from core.tasks.loader import load_task
        task = load_task("prim_grasp_lift")
        fn = make_spec_fn(task, ARM, scene=H.SCENE)
        steps = task.spec.raw["expert"]["steps"]
        lift = next(i for i, st in enumerate(steps) if st["skill"] == "lift")
        base = next(s for s in (fn(Kind.SUCCESS, i, attempt_seed(0, "prim_grasp_lift", Kind.SUCCESS, i))
                                for i in range(40)) if s.noise is None)
        heights = []
        for h in (0.07, 0.1):
            params = json.loads(json.dumps(base.to_json()["expert_params"]))
            params.setdefault("steps", {}).setdefault(str(lift), {})["height_m"] = h
            params["speed_scale"] = 1.0
            r, rec, env, rig = self._run(dataclasses.replace(base, expert_params=params), task)
            self.assertEqual(r.status.value, "accepted")
            z = np.array([x.objects["cube"].position[2] for x in rec])
            heights.append(z[-1] - z[0])
        self.assertGreater(heights[1] - heights[0], 0.02, "the drawn lift height is used")
        peaks, lengths = {}, {}
        for scale in (0.5, 1.0):
            params = json.loads(json.dumps(base.to_json()["expert_params"]))
            params["speed_scale"] = scale
            spec = dataclasses.replace(base, expert_params=params, max_steps=base.max_steps * 2)
            r, rec, env, rig = self._run(spec, task)
            self.assertEqual(r.status.value, "accepted")
            tcp = np.array([rig.chain.fk(rig.mapping.to_radians(x.state_raw), "tcp")[:3, 3] for x in rec])
            peaks[scale] = float(np.percentile(np.linalg.norm(np.diff(tcp, axis=0), axis=1) * spec.fps, 99))
            lengths[scale] = len(rec)
        self.assertLessEqual(peaks[0.5], 0.7 * peaks[1.0], f"speed_scale 0.5 must slow the TCP down: {peaks}")
        self.assertGreater(lengths[0.5], lengths[1.0] * 1.3, f"a slower episode is longer: {lengths}")


# ---------------------------------------------------------------------------------------------------- §16.5

class PhysicsTests(_Sandboxed):
    """WP14 §16.5 ``[task.physics.<object|set>]`` → ``EpisodeSpec.physics`` → sidecar ``attempt.physics``."""

    TASKS = ("pick_place_color_cube",)

    def test_validate(self):
        from core.generate.specs import validate_physics
        good = self.spec("fx14_ph", {"objects": {"cube": CUBE},
                                     "physics": {"cube": {"mass_kg": [0.01, 0.03], "static_friction": [0.5, 1.0]}}})
        validate_physics(good)
        for i, bad in enumerate(({"nobody": {"mass_kg": [0.01, 0.02]}}, {"cube": {}}, {"cube": {"mass_kg": [0.0, 0.02]}},
                                 {"cube": {"mass_kg": [0.03, 0.02]}}, {"cube": {"static_friction": [-0.1, 0.5]}},
                                 {"cube": {"restitution": [0.1, 0.2]}}, {"cube": {"mass_kg": 0.02}})):
            spec = self.spec(f"fx14_ph_bad{i}", {"objects": {"cube": CUBE}, "physics": bad})
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                validate_physics(spec)

    def test_draw(self):
        from core.generate.specs import draw_physics
        from core.tasks.loader import load_task
        task = load_task("pick_place_color_cube")
        table = task.spec.raw["physics"]
        objects = {name: ObjectState(Pose((0.0, -0.3, 0.01)), name in ("cube_0", "cube_2"), None)
                   for name in task.spec.objects}
        draws = [draw_physics(task.spec, objects, np.random.default_rng(k)) for k in range(200)]
        self.assertEqual(draws[5], draw_physics(task.spec, objects, np.random.default_rng(5)))
        entry = table["cubes"]
        for d in draws:
            self.assertEqual(set(d), {"cube_0", "cube_2"}, "active objects only; set entries apply to every slot")
            for v in d.values():
                self.assertTrue(entry["mass_kg"][0] <= v["mass_kg"] <= entry["mass_kg"][1])
                self.assertTrue(entry["static_friction"][0] <= v["static_friction"] <= entry["static_friction"][1])
                self.assertLessEqual(v["dynamic_friction"], v["static_friction"])
        rng = np.random.default_rng(9)               # per object in spec order: mass, static, dynamic
        want = {}
        for name in ("cube_0", "cube_2"):
            m, st, dy = (float(rng.uniform(*entry[k])) for k in ("mass_kg", "static_friction", "dynamic_friction"))
            want[name] = {"mass_kg": m, "static_friction": st, "dynamic_friction": min(dy, st)}
        self.assertEqual(draws[9], want)

    def test_spec_rejects_inactive(self):
        s = _small_spec()
        with self.assertRaises(ValueError):
            dataclasses.replace(s, physics={"ghost": {"mass_kg": 0.01}})
        self.assertNotIn("physics", s.to_json())


def _small_spec(**kw):
    return EpisodeSpec(task="t", arm="a", scene="s", kind=Kind.SUCCESS, attempt_index=0, seed=1,
                       objects=kw.pop("objects", {"cube": ObjectState(Pose((0.2, 0.0, 0.01)), True, "red")}),
                       roles={}, goal=kw.pop("goal", {}), instruction="x", initial_q=kw.pop("initial_q", (0.0, 0.0)), **kw)


# ---------------------------------------------------------------------------------------------------- §16.6

class NearDuplicateTests(_Sandboxed):
    """WP14 §16.6 ``near_key`` and near-duplicate skips."""

    TASKS = ("prim_grasp_lift", "prim_reach_point")

    def _moved(self, s, dx=0.0, dyaw=0.0, dz=0.0, **kw):
        st = s.objects["cube"]
        p = st.pose.position
        yaw = _yaw(st.pose.quat_xyzw) + dyaw
        obj = dataclasses.replace(st, pose=Pose((p[0] + dx, p[1], p[2] + dz), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
                                  **kw)
        return dataclasses.replace(s, objects={"cube": obj})

    def test_near_key(self):
        from core.generate.fingerprint import near_key, scenario_key
        s = _small_spec(objects={"cube": ObjectState(Pose((0.2005, 0.0021, 0.01), (0.0, 0.0, math.sin(0.09), math.cos(0.09))),
                                                     True, "red")})
        k = near_key(s)
        self.assertEqual(near_key(self._moved(s, dx=0.0015)), k, "within one 5 mm cell")
        self.assertNotEqual(scenario_key(self._moved(s, dx=0.0015)), scenario_key(s), "while scenario_key differs")
        self.assertEqual(near_key(self._moved(s, dyaw=math.radians(1.0))), k)
        self.assertEqual(near_key(self._moved(s, dz=0.02)), k, "height is not part of the layout")
        self.assertNotEqual(near_key(self._moved(s, dx=0.005)), k)
        self.assertNotEqual(near_key(self._moved(s, dyaw=math.radians(5.0))), k)
        self.assertNotEqual(near_key(self._moved(s, color="blue")), k)
        self.assertNotEqual(near_key(self._moved(s, size=(0.024, 0.024, 0.024))), k)
        for same in (dataclasses.replace(s, initial_q=(0.5, -0.5)), dataclasses.replace(s, instruction="y"),
                     dataclasses.replace(s, expert_params={"speed_scale": 0.7}),
                     dataclasses.replace(s, physics={"cube": {"mass_kg": 0.02}}), dataclasses.replace(s, seed=99)):
            self.assertEqual(near_key(same), k)
        g = dataclasses.replace(s, goal={"p": [0.2, 0.1, 0.05]})
        self.assertEqual(near_key(dataclasses.replace(s, goal={"p": [0.2015, 0.1, 0.05]})), near_key(g))
        self.assertNotEqual(near_key(dataclasses.replace(s, goal={"p": [0.21, 0.1, 0.05]})), near_key(g))

    def test_scenario_key_size(self):
        from core.generate.fingerprint import scenario_key
        s = _small_spec()
        self.assertNotEqual(scenario_key(self._moved(s, size=(0.024, 0.024, 0.024))), scenario_key(s))

    def test_plan_skips_near_duplicates(self):
        """Attempt indices 1 and 3 of every kind lie within one cell of index 0 / 2: skipped only with near_duplicates."""
        from core.generate.fingerprint import near_key
        from core.generate.plan import iter_plan
        base = _small_spec()
        offsets = [0.0, 0.001, 0.02, 0.0212, 0.04]
        lane = {Kind.SUCCESS: 0.0, Kind.FAILURE: 0.1, Kind.RECOVERY: 0.2}

        def spec_fn(kind, index, seed):
            moved = self._moved(dataclasses.replace(base, attempt_index=index), dx=offsets[index] + lane[kind] - 0.0005)
            return dataclasses.replace(moved, kind=kind, perturbation=None if kind is Kind.SUCCESS else _pert())

        def run(flag):
            it = iter_plan("t", 0, spec_fn, near_duplicates=flag)
            return [next(it) for _ in range(15)]

        plain, near = run(False), run(True)
        self.assertEqual([a.skip for a in plain], [None] * 15)
        for kind in lane:
            self.assertEqual([a.skip for a in near if a.kind is kind],
                             [None, SkipReason.SCENARIO_SEEN, None, SkipReason.SCENARIO_SEEN, None], kind)
        kept = [near_key(a.spec) for a in near if a.skip is None]
        self.assertEqual(len(kept), len(set(kept)))

    def test_generation_has_no_near_duplicates(self):
        """Originals of a generated dataset have pairwise distinct near_keys (rebuilt from their attempts)."""
        from core.datasets.task_meta import list_sidecars, read_sidecar
        from core.generate.fingerprint import near_key
        from core.generate.loop import make_spec_fn
        from core.tasks.loader import load_task
        roots = H.generate_small(["prim_grasp_lift"], quotas={"success": 6, "failure": 3, "recovery": 3}, shards=2,
                                 work=self.box.root / "near", arm=ARM, max_attempt_factor=20.0)
        root = roots["prim_grasp_lift"]
        fn = make_spec_fn(load_task("prim_grasp_lift"), ARM, scene=H.SCENE)
        keys = []
        for i in list_sidecars(root):
            sc = read_sidecar(root, i)
            spec = fn(Kind(sc["kind"]), sc["attempt"]["index"], sc["attempt"]["seed"])
            self.assertEqual(sc["fingerprints"]["scenario_key"], __import__("core.generate.fingerprint",
                                                                            fromlist=["x"]).scenario_key(spec))
            keys.append(near_key(spec))
        self.assertEqual(len(keys), 12)
        self.assertEqual(len(set(keys)), 12)


def _pert():
    from core.generate.protocols import PerturbationGroup, PerturbationSpec
    return PerturbationSpec("stall", PerturbationGroup.EXPERT, None, 0, {"steps": 5})


# ---------------------------------------------------------------------------------------------------- §16.7

class CameraJitterTests(_Sandboxed):
    """WP14 §16.7 DR camera offsets: drawn, recorded, fingerprinted, audited."""

    TASKS = ("prim_grasp_lift",)

    def test_appearance_fp(self):
        from core.generate.fingerprint import appearance_fp
        golden = appearance_fp({"cube_0": "red"}, "x", 0.8, "black")
        self.assertEqual(appearance_fp({"cube_0": "red"}, "x", 0.8, "black", None), golden)
        self.assertEqual(appearance_fp({"cube_0": "red"}, "x", 0.8, "black", {}), golden)
        off = {"front": {"position_m": [0.004, -0.002, 0.001], "rotation_deg": [0.5, -1.0, 0.25]}}
        a = appearance_fp({"cube_0": "red"}, "x", 0.8, "black", off)
        self.assertNotEqual(a, golden)
        near = {"front": {"position_m": [0.004 + 4e-5, -0.002, 0.001], "rotation_deg": [0.5, -1.0, 0.25 + 4e-3]}}
        far = {"front": {"position_m": [0.004 + 1e-4, -0.002, 0.001], "rotation_deg": [0.5, -1.0, 0.25]}}
        self.assertEqual(appearance_fp({"cube_0": "red"}, "x", 0.8, "black", near), a, "position 0.1 mm, rotation 0.01°")
        self.assertNotEqual(appearance_fp({"cube_0": "red"}, "x", 0.8, "black", far), a)

    def _parent(self):
        if not hasattr(type(self), "_root"):
            roots = H.generate_small(["prim_grasp_lift"], quotas={"success": 2, "failure": 1, "recovery": 1}, shards=1,
                                     work=self.box.root / "cam", arm=ARM, max_attempt_factor=20.0)
            type(self)._root = roots["prim_grasp_lift"]
        return type(self)._root

    def test_draw_and_dataset(self):
        from core.datasets.task_meta import list_sidecars, read_sidecar, validate_sidecar
        from core.generate.appearance import draw_appearance
        from core.tasks.base import load_task_spec
        root = self._parent()
        parent = read_sidecar(root, list_sidecars(root)[0])
        spec = load_task_spec("prim_grasp_lift")
        kw = dict(task_spec=spec, terms=("light",), exposure_range=(0.5, 1.5), robot_colors=(), recolor=False,
                  dr_seed=3, copy_index=0, taken=set())
        jit = {"position_m": 0.01, "rotation_deg": 2.0}
        d = draw_appearance(parent, camera_jitter=jit, cameras=("wrist", "front"), **kw)
        self.assertEqual(d, draw_appearance(parent, camera_jitter=jit, cameras=("front", "wrist"), **kw))
        self.assertEqual(list(d.camera_offsets), ["front", "wrist"], "cameras in sorted order")
        for off in d.camera_offsets.values():
            self.assertTrue(all(abs(v) <= 0.01 for v in off["position_m"]))
            self.assertTrue(all(abs(v) <= 2.0 for v in off["rotation_deg"]))
        plain = draw_appearance(parent, **kw)
        self.assertIsNone(plain.camera_offsets)
        self.assertEqual(plain.light_exposure, d.light_exposure, "jitter is drawn after the earlier terms")
        for bad in ({"position_m": 0.01}, {"position_m": -0.01, "rotation_deg": 1.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                draw_appearance(parent, camera_jitter=bad, cameras=("front",), **kw)
        with self.assertRaises(ValueError):
            draw_appearance(parent, camera_jitter=jit, cameras=(), **kw)
        # a DR dataset with jitter: sidecars record the offsets, the audit and the independent check pass
        dr = H.fake_dr(root, self.box.root / "cam_dr_test", copies=2, camera_jitter=jit, cameras=("front", "wrist"))
        H.assert_audit_passes(self, H.audit(dr, source_root=root))
        counts = H.assert_no_duplicates(self, dr, source_root=root)
        self.assertEqual(counts["copies"], 2 * len(list_sidecars(root)))
        for i in list_sidecars(dr):
            sc = read_sidecar(dr, i)
            validate_sidecar(sc)
            if sc["source"] is not None:
                self.assertEqual(set(sc["appearance"]["camera_offsets"]), {"front", "wrist"})
            else:
                self.assertNotIn("camera_offsets", sc["appearance"])
        # tampering with a recorded offset breaks the recomputed fingerprint
        bad = read_sidecar(dr, 1)
        bad["appearance"]["camera_offsets"]["front"]["position_m"][0] += 0.003
        from core.datasets.task_meta import write_sidecar
        write_sidecar(dr, bad)
        report = H.audit(dr, source_root=root)
        self.assertFalse({c["name"]: c["passed"] for c in report.checks}["fingerprints_recomputed"])

    def test_validator(self):
        from core.datasets.task_meta import list_sidecars, read_sidecar, validate_sidecar
        from core.generate.errors import SchemaError
        root = self._parent()
        sc = read_sidecar(root, list_sidecars(root)[0])
        ok = json.loads(json.dumps(sc))
        ok["appearance"]["camera_offsets"] = {"front": {"position_m": [0.0, 0.0, 0.001], "rotation_deg": [0.0, 1.0, 0.0]}}
        validate_sidecar(ok)
        for bad in ({}, {"front": {"position_m": [0.0, 0.0]}}, {"front": {"position_m": [0.0] * 3}},
                    {"front": {"position_m": [0.0] * 3, "rotation_deg": [float("nan")] * 3}}):
            s2 = json.loads(json.dumps(sc))
            s2["appearance"]["camera_offsets"] = bad
            with self.subTest(bad=bad), self.assertRaises(SchemaError):
                validate_sidecar(s2)


# ---------------------------------------------------------------------------------------------------- sidecar

class SidecarTests(_Sandboxed):
    """WP14 §16.8 sidecars record the drawn sizes (``size_m``), physics and expert parameters."""

    TASKS = ("pick_place_color_cube",)

    def test_recorded(self):
        from core.datasets.task_meta import list_sidecars, read_sidecar, validate_sidecar
        from core.generate.loop import make_spec_fn
        from core.generate.errors import SchemaError
        from core.tasks.loader import load_task
        roots = H.generate_small(["pick_place_color_cube"], quotas={"success": 2, "failure": 1, "recovery": 1},
                                 shards=1, work=self.box.root / "sc", arm=ARM, max_attempt_factor=20.0)
        root = roots["pick_place_color_cube"]
        fn = make_spec_fn(load_task("pick_place_color_cube"), ARM, scene=H.SCENE)
        for i in list_sidecars(root):
            sc = read_sidecar(root, i)
            spec = fn(Kind(sc["kind"]), sc["attempt"]["index"], sc["attempt"]["seed"])
            with self.subTest(episode=i):
                for name, obj in sc["objects"].items():
                    size = spec.objects[name].size
                    if size is not None:
                        self.assertEqual(obj["size_m"], list(size))
                self.assertEqual(sc["attempt"]["physics"], json.loads(json.dumps(spec.to_json()["physics"])))
                self.assertEqual(sc["attempt"]["expert_params"], json.loads(json.dumps(spec.to_json()["expert_params"])))
        sc = read_sidecar(root, 0)
        for bad in ({"physics": {}}, {"physics": {"cube_0": {"mass_kg": "x"}}}, {"physics": {"cube_0": {"spin": 1.0}}},
                    {"expert_params": []}):
            s2 = json.loads(json.dumps(sc))
            s2["attempt"].update(bad)
            with self.subTest(bad=bad), self.assertRaises(SchemaError):
                validate_sidecar(s2)


# ---------------------------------------------------------------------------------------------------- §16.9

class TaskTableTests(_Sandboxed):
    """WP14 §16.9 every one of the 17 tasks uses the wide randomisation."""

    TASKS = ALL_TASKS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from core.arms.spec import load_arm_spec
        from core.tasks.loader import load_task
        cls.ws = load_arm_spec(ARM)[0]["workspace"]
        cls.tasks = {name: load_task(name) for name in ALL_TASKS}
        cls._specs = {}

    def specs(self, name, n=150, kind=Kind.SUCCESS):
        from core.generate.loop import make_spec_fn
        from core.generate.plan import attempt_seed
        if (name, kind) not in self._specs:
            fn, out = make_spec_fn(self.tasks[name], ARM, scene=H.SCENE), []
            for i in range(n * 2):
                try:
                    out.append(fn(kind, i, attempt_seed(0, name, kind, i)))
                except SamplingError:
                    continue
                if len(out) == n:
                    break
            type(self)._specs[(name, kind)] = out
        return self._specs[(name, kind)]

    def test_tables(self):
        import tomllib
        from core.arms.gripper import Gripper
        so101 = tomllib.loads((REPO_ROOT / "arms" / "so101" / "arm.toml").read_text())["arm"]
        widths = (Gripper.from_arm(ARM).max_object_width_m, so101["gripper"]["max_object_width_m"])
        for name, task in self.tasks.items():
            raw = task.spec.raw
            start, expert = raw["start"], raw["expert"]
            with self.subTest(task=name):
                q0 = start.get("initial_q")
                self.assertIsInstance(q0, dict)
                self.assertEqual(q0.get("around"), "home")
                self.assertGreaterEqual(q0["spread"], 0.1)
                self.assertIn("range", expert.get("speed_scale", {}), "speed_scale must vary")
                lo, hi = expert["speed_scale"]["range"]
                self.assertTrue(lo <= 0.7 and hi >= 0.95)
                first = expert["steps"][0]
                self.assertFalse(first["skill"] == "move_joint" and first.get("q") == "home",
                                 "episodes start moving from their own random pose")
                for obj, place in start.get("place", {}).items():
                    self.assertTrue(place["region"] == "workspace" or set(place["region"]) == {"workspace_inset_m"},
                                    f"{obj} must be placed anywhere in the workspace")
                    if place["region"] != "workspace":
                        self.assertLessEqual(place["region"]["workspace_inset_m"], 0.05)
                for goal in start.get("goal", {}).values():
                    if goal["type"] in ("point", "points"):
                        self.assertEqual(goal["region"], "workspace")
                    if goal["type"] == "region":
                        self.assertTrue(goal["center_region"] == "workspace" or "around" in goal["center_region"])
                if name in CUBE_TASKS:
                    cubes = [o for o in task.spec.objects.values() if o.size is not None]
                    self.assertTrue(cubes)
                    for o in cubes:
                        variants = o.params.get("size_variants")
                        self.assertIsNotNone(variants, f"{o.name} has no size variants")
                        self.assertGreaterEqual(len(variants), 4)
                        self.assertLessEqual(min(v[0] for v in variants), 0.019)
                        self.assertGreaterEqual(max(v[0] for v in variants), 0.025)
                        for v in variants:
                            self.assertLessEqual(min(v[:2]), min(widths))
                if name in OBJECT_TASKS:
                    target = task.spec.roles["target"]
                    self.assertTrue(target in raw.get("physics", {}) or
                                    task.spec.objects.get(target, task.spec.object_sets.get(target)) is not None
                                    and any(k in raw.get("physics", {}) for k in (target,)))
                    for key in ("mass_kg", "static_friction", "dynamic_friction"):
                        self.assertIn(key, raw["physics"][target])
                if name in CONTAINERS and name != "stack_bowls":
                    yaw = start["place"][CONTAINERS[name]].get("yaw_deg", [0, 0])
                    self.assertGreaterEqual(yaw[1] - yaw[0], 90.0, "containers turn too")

    def test_coverage(self):
        """Over 150 successes the target starts in ≥ 60 % of the 4 cm cells of its region (containers ≥ 50 %)."""
        def cells(region):
            ws = self.ws if region == "workspace" else _inset(self.ws, region["workspace_inset_m"])
            xs = np.arange(-0.42, 0.42, 0.04) + 0.02
            return {(int(round((x - 0.02) / 0.04)), int(round((y - 0.02) / 0.04))) for x in xs for y in xs
                    if _in_ws((x, y), ws)}

        def hit(points):
            return {(int(math.floor(p[0] / 0.04)), int(math.floor(p[1] / 0.04))) for p in points}

        for name in OBJECT_TASKS:
            task = self.tasks[name]
            specs = self.specs(name)
            with self.subTest(task=name):
                self.assertGreaterEqual(len(specs), 140)
                target_obj = specs[0].roles["target"]
                place_key = next(k for k in task.spec.raw["start"]["place"]
                                 if k == target_obj or k in task.spec.object_sets)
                region = task.spec.raw["start"]["place"][place_key]["region"]
                want = cells(region)
                got = hit(s.objects[s.roles["target"]].pose.position for s in specs) & want
                self.assertGreaterEqual(len(got) / len(want), 0.6, f"{name}: target covers {len(got)}/{len(want)} cells")
                if name in CONTAINERS:
                    c = CONTAINERS[name]
                    want = cells(task.spec.raw["start"]["place"][c]["region"])
                    got = hit(s.objects[c].pose.position for s in specs) & want
                    self.assertGreaterEqual(len(got) / len(want), 0.5, f"{name}: container covers {len(got)}/{len(want)}")

    def test_variety(self):
        """Within one task, starts differ in every drawn quantity."""
        for name in ALL_TASKS:
            specs = self.specs(name, n=60)
            with self.subTest(task=name):
                q = np.array([s.initial_q for s in specs])
                self.assertGreater(float(np.max(q.std(axis=0))), 0.05)
                speeds = [s.expert_params["speed_scale"] for s in specs]
                self.assertGreater(float(np.std(speeds)), 0.05)
                if name in CUBE_TASKS:
                    sizes = {s.objects[s.roles["target"]].size for s in specs}
                    self.assertGreaterEqual(len(sizes), 4)
                if name in OBJECT_TASKS:
                    masses = [s.physics[s.roles["target"]]["mass_kg"] for s in specs]
                    self.assertGreater(float(np.std(masses)), 1e-3)

    def test_relative_goals(self):
        """nudge, push_to_region and place_at put their target area around the cube, in any direction."""
        for name, (lo, hi) in (("prim_nudge", (0.025, 0.035)), ("prim_push_to_region", (0.07, 0.14)),
                               ("prim_place_at", (0.08, 0.25))):
            specs = self.specs(name, n=80)
            key = "spot" if name == "prim_place_at" else "area"
            angles = []
            for s in specs:
                cube = np.array(s.objects["cube"].pose.position[:2])
                area = s.goal[key]
                centre = (np.array(area["xy_min"]) + np.array(area["xy_max"])) / 2
                d = np.linalg.norm(centre - cube)
                with self.subTest(task=name, seed=s.seed):
                    self.assertTrue(lo - 1e-9 <= d <= hi + 1e-9, d)
                    self.assertTrue(_in_ws(centre, self.ws))
                angles.append(math.atan2(*(centre - cube)[::-1]))
            quadrants = {int((a + math.pi) // (math.pi / 2)) % 4 for a in angles}
            self.assertEqual(len(quadrants), 4, f"{name}: the area lies in every direction from the cube")


if __name__ == "__main__":
    unittest.main()
