"""WP1: start sampler, park positions and instruction rendering (contracts §1.4, §4.5, §4.6, §14.6, §14.8).

Property tests loop over ``np.random.default_rng(k)``. Reachability is an injected ``reach_fn(Pose) -> bool``
(no kinematics needed). Steps 1-3 of the draw order (count, colours, slots) are fixed exactly by §4.5 and are
recomputed here from a fresh Generator.
"""

import dataclasses
import math
import unittest

import numpy as np

from core.generate.errors import ConfigError, SamplingError
from core.generate.protocols import ObjectState
from core.tasks.base import Pose, load_task_spec
from core.tasks.generic import GenericTask
from core.tasks.sampling import StartSample, park_positions, render_instruction, sample_start
from v4_support import assert_frozen, sandbox
from v4_support.tasks import STUB_USDA, contract_color_task_table, contract_reach_task_table, write_task

BOX_FILES = {"usd/Small-Box-Physics.usda": STUB_USDA}
CUBES = tuple(f"cube_{i}" for i in range(6))
PALETTE = ("red", "green", "blue", "yellow", "orange", "purple", "pink")
TRAIN_PALETTE = tuple(c for c in PALETTE if c != "purple")
CUBE_REGION = ((0.12, -0.18), (0.30, 0.18))
BOX_REGION = ((0.18, -0.12), (0.28, 0.12))
DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)
EPS = 1e-9
SEEDS = range(150)


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rect(center, yaw, size_xy):
    hx, hy = size_xy[0] / 2, size_xy[1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return [np.array(center) + np.array([c * dx - s * dy, s * dx + c * dy]) for dx, dy in
            ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def _seg_point(p, a, b):
    ab = b - a
    t = min(1.0, max(0.0, float(np.dot(p - a, ab) / np.dot(ab, ab))))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(A, B):
    for poly in (A, B):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = [float(np.dot(p, axis)) for p in A], [float(np.dot(p, axis)) for p in B]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def footprint_distance(A, B):
    """Distance between two convex quadrilaterals (0 when they overlap)."""
    if not _separated(A, B):
        return 0.0
    return min(_seg_point(p, Q[i], Q[(i + 1) % 4]) for P, Q in ((A, B), (B, A)) for p in P for i in range(4))


def _inside(xy, region):
    (x0, y0), (x1, y1) = region
    return x0 - EPS <= xy[0] <= x1 + EPS and y0 - EPS <= xy[1] <= y1 + EPS


def _spec(box, name, table, files=BOX_FILES):
    write_task(box.tasks, name, table, files=files)
    return load_task_spec(name)


def _expected_draws(seed, *, lo=2, hi=6, holdout_counts=(4,), pal=TRAIN_PALETTE, count_max=6, distinct=True):
    """§4.5 steps 1-3 transcribed: (n, target colour, distractor colours, slots)."""
    rng = np.random.default_rng(seed)
    allowed = [c for c in range(lo, hi + 1) if c not in holdout_counts]
    n = allowed[rng.integers(len(allowed))]
    target = pal[rng.integers(len(pal))]
    distractors = [str(c) for c in rng.choice([c for c in pal if c != target], n - 1, replace=not distinct)]
    slots = [int(s) for s in rng.permutation(count_max)[:n]]
    return n, target, distractors, slots


class ParkAndRenderTests(unittest.TestCase):
    """WP1 §4.5 park_positions and render_instruction."""

    def test_park_positions_golden(self):
        """§4.5 park_positions: name_i -> (0.1 * i, 0.0, -2.0)."""
        self.assertEqual(park_positions(["a", "b", "c"]), {"a": (0.0, 0.0, -2.0), "b": (0.1, 0.0, -2.0),
                                                           "c": (0.1 * 2, 0.0, -2.0)})
        self.assertEqual(park_positions([]), {})
        self.assertEqual(list(park_positions(["z", "y"])), ["z", "y"])

    def test_render_instruction(self):
        """§4.5 placeholders {<role>.color}, {<role>.name}, {goal.<key>}."""
        roles = {"target": "cube_3", "container": "small_box"}
        colors = {"cube_3": "red", "cube_1": "blue"}
        self.assertEqual(render_instruction("Pick up the {target.color} cube and place it in the box", roles, colors, {}),
                         "Pick up the red cube and place it in the box")
        # decision: .name drops a trailing _<digits> slot suffix and replaces underscores by spaces
        self.assertEqual(render_instruction("Put the {target.name} in the {container.name}", roles, colors, {}),
                         "Put the cube in the small box")
        # decision: float goal values are formatted with 2 decimals, strings as is
        self.assertEqual(render_instruction("Open to {goal.gap} m, then {goal.mode}", {}, {}, {"gap": 0.25, "mode": "wait"}),
                         "Open to 0.25 m, then wait")
        self.assertEqual(render_instruction("No placeholders.", {}, {}, {}), "No placeholders.")


class StartSampleTypeTests(unittest.TestCase):
    """WP1 §4.5 StartSample."""

    def test_fields_and_frozen(self):
        """§4.5 StartSample(objects, roles, goal, instruction, initial_q, held_object), frozen and read-only."""
        self.assertEqual([f.name for f in dataclasses.fields(StartSample)],
                         ["objects", "roles", "goal", "instruction", "initial_q", "held_object"])
        with sandbox() as box:
            spec = _spec(box, "fx_wp1_s_frozen", contract_color_task_table())
            sample = sample_start(spec, np.random.default_rng(0), home_q=[0.0, 0.1, 0.2])
            assert_frozen(sample)
            self.assertEqual(sample.initial_q, (0.0, 0.1, 0.2))
            self.assertTrue(all(isinstance(s, ObjectState) for s in sample.objects.values()))
            reach = _spec(box, "fx_wp1_s_frozen_r", contract_reach_task_table(), files={})
            assert_frozen(sample_start(reach, np.random.default_rng(0)))


class ColorCubeSamplerTests(unittest.TestCase):
    """WP1 §4.5 sampler properties on the §4.6 colour-cube example (count 2..6, hold-outs purple and 4)."""

    @classmethod
    def setUpClass(cls):
        cls._box = sandbox()
        box = cls._box.__enter__()
        cls.spec = _spec(box, "fx_wp1_s_color", contract_color_task_table())
        cls.samples = {k: sample_start(cls.spec, np.random.default_rng(k)) for k in SEEDS}

    @classmethod
    def tearDownClass(cls):
        cls._box.__exit__(None, None, None)

    def active_cubes(self, sample):
        return [n for n in CUBES if sample.objects[n].active]

    def test_objects_cover_expanded_spec(self):
        """§3.2 / §4.5 every object of the expanded TaskSpec.objects is present; the box is active."""
        for k, sample in self.samples.items():
            with self.subTest(seed=k):
                self.assertEqual(list(sample.objects), list(self.spec.objects))
                self.assertTrue(sample.objects["small_box"].active)
                self.assertIsNone(sample.objects["small_box"].color)
                self.assertEqual(sample.goal, {})
                self.assertIsNone(sample.held_object)
                self.assertIsNone(sample.initial_q)                 # no home_q given

    def test_exact_count_colour_slot_draws(self):
        """§4.5 steps 1-3: count, target colour, distractors and slots follow the specified Generator calls."""
        for k, sample in self.samples.items():
            with self.subTest(seed=k):
                n, target, distractors, slots = _expected_draws(k)
                active = self.active_cubes(sample)
                self.assertEqual(sorted(active), sorted(f"cube_{s}" for s in slots))
                self.assertEqual(sample.roles["target"], f"cube_{slots[0]}")
                self.assertEqual(sample.roles["container"], "small_box")
                self.assertEqual(sample.objects[f"cube_{slots[0]}"].color, target)
                for slot, colour in zip(slots[1:], distractors):
                    self.assertEqual(sample.objects[f"cube_{slot}"].color, colour)

    def test_holdouts_never_drawn(self):
        """§4.5 / §14.8 held-out colour and count never appear in generation; every other value does."""
        counts, targets, all_colours = set(), set(), set()
        for k, sample in self.samples.items():
            active = self.active_cubes(sample)
            colours = [sample.objects[n].color for n in active]
            with self.subTest(seed=k):
                self.assertNotIn(len(active), (4,))
                self.assertNotIn("purple", colours)
                self.assertTrue(2 <= len(active) <= 6)
            counts.add(len(active))
            targets.add(sample.objects[sample.roles["target"]].color)
            all_colours.update(colours)
        self.assertEqual(counts, {2, 3, 5, 6})
        self.assertEqual(targets, set(TRAIN_PALETTE))
        self.assertEqual(all_colours, set(TRAIN_PALETTE))

    def test_colours_unique_and_inactive_parked(self):
        """§4.5 target colour unique (distinct_colors: all distinct); inactive slots parked, active False, no colour."""
        park = park_positions(list(self.spec.objects))
        for k, sample in self.samples.items():
            active = self.active_cubes(sample)
            colours = [sample.objects[n].color for n in active]
            target_colour = sample.objects[sample.roles["target"]].color
            with self.subTest(seed=k):
                self.assertEqual(colours.count(target_colour), 1)
                self.assertEqual(len(set(colours)), len(colours))
                for name in CUBES:
                    state = sample.objects[name]
                    if not state.active:
                        self.assertEqual(tuple(state.pose.position), park[name])
                        self.assertIsNone(state.color)
                        self.assertNotIn(name, sample.roles.values())

    def test_placement_constraints(self):
        """§4.5 inside regions, resting z, yaw only, min separation between cubes, clearance from the box."""
        box_spec = self.spec.objects["small_box"]
        for k, sample in self.samples.items():
            box_pose = sample.objects["small_box"].pose
            box_rect = _rect(box_pose.position[:2], _yaw(box_pose.quat_xyzw), box_spec.params["size_outer"][:2])
            active = self.active_cubes(sample)
            with self.subTest(seed=k):
                self.assertTrue(_inside(box_pose.position, BOX_REGION))
                self.assertAlmostEqual(box_pose.position[2], 0.0, places=12)       # bottom origin rests at table z
                self.assertAlmostEqual(_yaw(box_pose.quat_xyzw), 0.0, places=9)    # yaw_deg default [0, 0]
                for name in active:
                    pose = sample.objects[name].pose
                    self.assertTrue(_inside(pose.position, CUBE_REGION), name)
                    self.assertAlmostEqual(pose.position[2], 0.01, places=12)        # centre origin: table + h/2
                    qx, qy, qz, qw = pose.quat_xyzw
                    self.assertAlmostEqual(qx, 0.0, places=12)
                    self.assertAlmostEqual(qy, 0.0, places=12)
                    self.assertAlmostEqual(qx * qx + qy * qy + qz * qz + qw * qw, 1.0, places=9)
                    self.assertTrue(-EPS <= _yaw(pose.quat_xyzw) <= math.pi / 2 + EPS)
                    cube_rect = _rect(pose.position[:2], _yaw(pose.quat_xyzw), (0.02, 0.02))
                    self.assertGreaterEqual(footprint_distance(cube_rect, box_rect), 0.01 - EPS)
                for i, a in enumerate(active):
                    for b in active[i + 1:]:
                        pa, pb = sample.objects[a].pose.position, sample.objects[b].pose.position
                        self.assertGreaterEqual(math.hypot(pa[0] - pb[0], pa[1] - pb[1]), 0.035 - EPS)

    def test_instruction(self):
        """§4.5 the instruction is the rendered template (k = 0 without instruction_rng)."""
        for k, sample in self.samples.items():
            colour = sample.objects[sample.roles["target"]].color
            with self.subTest(seed=k):
                self.assertEqual(sample.instruction, f"Pick up the {colour} cube and place it in the box")

    def test_deterministic_and_seed_sensitive(self):
        """§1.4 same seed gives an equal StartSample; different seeds give different starts."""
        for k in range(10):
            with self.subTest(seed=k):
                self.assertEqual(sample_start(self.spec, np.random.default_rng(k)), self.samples[k])
        keys = {tuple(round(c, 9) for n in self.spec.objects for c in self.samples[k].objects[n].pose.position)
                for k in range(40)}
        self.assertEqual(len(keys), 40)

    def test_task_method_matches_function(self):
        """§4.2 Task.sample_episode_start(rng) == sample_start(task.spec, rng) for the same seed."""
        task = GenericTask(self.spec)
        for k in range(5):
            with self.subTest(seed=k):
                self.assertEqual(task.sample_episode_start(np.random.default_rng(k), home_q=[0.5]),
                                 sample_start(self.spec, np.random.default_rng(k), home_q=[0.5]))


class SamplerOptionTests(unittest.TestCase):
    """WP1 §4.5 reach_fn, instruction_rng, distinct_colors, exhaustion, goals, initial_q, held."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def test_reach_fn_pregrasp_pose(self):
        """§4.5 reach_fn receives pregrasp = object centre + [0, 0, 0.06], TCP down (decision: quat_xyzw (1, 0, 0, 0))."""
        spec = _spec(self.box, "fx_wp1_s_pre", contract_color_task_table())
        calls = []

        def reach(pose):
            calls.append(pose)
            return True
        sample = sample_start(spec, np.random.default_rng(1), reach_fn=reach)
        self.assertTrue(calls)
        for pose in calls:
            self.assertIsInstance(pose, Pose)
            self.assertEqual(tuple(pose.quat_xyzw), DOWN_XYZW)
        target = sample.objects[sample.roles["target"]].pose.position
        box = sample.objects["small_box"].pose.position
        last = {round(p.position[0], 12): p for p in calls}
        self.assertAlmostEqual(last[round(target[0], 12)].position[2], 0.01 + 0.06, places=12)
        self.assertAlmostEqual(last[round(box[0], 12)].position[2], 0.035 / 2 + 0.06, places=12)   # bottom origin

    def test_reach_fn_constrains_only_reachable(self):
        """§4.5 reach_fn must hold for objects listed in reachable (default ["@target"]); others are free."""
        def reach(pose):
            return pose.position[1] > 0.0
        table = contract_color_task_table()
        spec = _spec(self.box, "fx_wp1_s_reach_both", table)
        del table["start"]["reachable"]
        spec_default = _spec(self.box, "fx_wp1_s_reach_default", table)
        distractor_neg = box_neg = 0
        for k in range(60):
            both = sample_start(spec, np.random.default_rng(k), reach_fn=reach)
            default = sample_start(spec_default, np.random.default_rng(k), reach_fn=reach)
            with self.subTest(seed=k):
                self.assertGreater(both.objects[both.roles["target"]].pose.position[1], 0.0)
                self.assertGreater(both.objects["small_box"].pose.position[1], 0.0)
                self.assertGreater(default.objects[default.roles["target"]].pose.position[1], 0.0)
            box_neg += default.objects["small_box"].pose.position[1] < 0.0
            distractor_neg += any(both.objects[n].pose.position[1] < 0.0 for n in CUBES
                                  if both.objects[n].active and n != both.roles["target"])
        self.assertGreater(box_neg, 0)
        self.assertGreater(distractor_neg, 0)

    def test_sampling_error_on_exhaustion(self):
        """§1.5 / §4.5 exhausting max_tries raises SamplingError; reach_fn is not called more than max_tries per object."""
        table = contract_color_task_table()
        table["start"]["max_tries"] = 7
        spec = _spec(self.box, "fx_wp1_s_exhaust", table)
        calls = []

        def never(pose):
            calls.append(pose)
            return False
        with self.assertRaises(SamplingError):
            sample_start(spec, np.random.default_rng(0), reach_fn=never)
        self.assertLessEqual(len(calls), 7)
        crowded = contract_color_task_table()
        crowded["holdout"] = {"colors": ["purple"], "counts": []}
        crowded["start"]["count"] = [6, 6]
        crowded["start"]["min_separation_m"] = 0.2
        crowded["start"]["max_tries"] = 50
        spec = _spec(self.box, "fx_wp1_s_crowded", crowded)
        for k in range(5):
            with self.subTest(seed=k):
                with self.assertRaises(SamplingError):
                    sample_start(spec, np.random.default_rng(k))

    def test_instruction_rng(self):
        """§4.5 the template index is drawn from instruction_rng; the start stream is unaffected by it."""
        table = contract_color_task_table()
        texts = ["Pick up the {target.color} cube", "Put the {target.color} {target.name} in the {container.name}",
                 "Place the {target.color} one in the box"]
        table["instruction_template"] = {"texts": texts}
        spec = _spec(self.box, "fx_wp1_s_instr", table)
        seen = set()
        for k in range(30):
            plain = sample_start(spec, np.random.default_rng(k))
            ins = sample_start(spec, np.random.default_rng(k), instruction_rng=np.random.default_rng(1000 + k))
            index = int(np.random.default_rng(1000 + k).integers(len(texts)))
            colour = ins.objects[ins.roles["target"]].color
            expected = render_instruction(texts[index], ins.roles, {ins.roles["target"]: colour}, {})
            with self.subTest(seed=k):
                self.assertEqual(plain.objects, ins.objects)
                self.assertEqual(plain.roles, ins.roles)
                self.assertEqual(plain.instruction, f"Pick up the {colour} cube")
                self.assertEqual(ins.instruction, expected)
            seen.add(index)
        self.assertEqual(seen, {0, 1, 2})

    def test_repeated_distractor_colours(self):
        """§4.5 distinct_colors = false: distractors may repeat but never use the target colour."""
        table = contract_color_task_table()
        table["start"]["distinct_colors"] = False
        spec = _spec(self.box, "fx_wp1_s_rep", table)
        repeats = 0
        for k in range(80):
            sample = sample_start(spec, np.random.default_rng(k))
            colours = [sample.objects[n].color for n in CUBES if sample.objects[n].active]
            target = sample.objects[sample.roles["target"]].color
            n, exp_target, distractors, slots = _expected_draws(k, distinct=False)
            with self.subTest(seed=k):
                self.assertEqual(colours.count(target), 1)
                self.assertNotIn("purple", colours)
                self.assertEqual(target, exp_target)
            repeats += len(set(colours)) < len(colours)
        self.assertGreater(repeats, 0)

    def test_table_z_and_polar_place_region(self):
        """§4.5 resting z = table_z_m + h/2 (centre) / table_z_m (bottom); polar place regions."""
        table = contract_color_task_table()
        table["start"]["table_z_m"] = -0.05
        table["start"]["place"]["cubes"]["region"] = {"r_m": [0.15, 0.30], "bearing_deg": [-40.0, 40.0]}
        spec = _spec(self.box, "fx_wp1_s_polar", table)
        for k in range(40):
            sample = sample_start(spec, np.random.default_rng(k))
            with self.subTest(seed=k):
                self.assertAlmostEqual(sample.objects["small_box"].pose.position[2], -0.05, places=12)
                for n in CUBES:
                    if sample.objects[n].active:
                        x, y, z = sample.objects[n].pose.position
                        self.assertAlmostEqual(z, -0.04, places=12)
                        self.assertTrue(0.15 - EPS <= math.hypot(x, y) <= 0.30 + EPS)
                        self.assertTrue(-40.0 - 1e-6 <= math.degrees(math.atan2(y, x)) <= 40.0 + 1e-6)

    def test_held_and_initial_q(self):
        """§4.5 / §4.6 held ref -> held_object; initial_q "home" -> home_q, table -> home_q with overrides."""
        table = contract_color_task_table()
        table["start"]["held"] = "@target"
        table["start"]["initial_q"] = {"j2": 0.7}
        spec = _spec(self.box, "fx_wp1_s_held", table)
        names = ("j1", "j2", "j3", "grip")
        sample = sample_start(spec, np.random.default_rng(0), home_q=[0.1, 0.2, 0.3, 0.4], joint_names=names)
        self.assertEqual(sample.held_object, sample.roles["target"])
        self.assertEqual(sample.initial_q, (0.1, 0.7, 0.3, 0.4))
        with self.assertRaises(ValueError):
            sample_start(spec, np.random.default_rng(0), home_q=[0.1, 0.2, 0.3, 0.4])      # table needs joint_names
        home = _spec(self.box, "fx_wp1_s_home", contract_color_task_table())
        self.assertEqual(sample_start(home, np.random.default_rng(0), home_q=np.array([0.1, 0.2])).initial_q, (0.1, 0.2))
        self.assertIsNone(sample_start(home, np.random.default_rng(0)).initial_q)

    def test_unplaced_object_keeps_spec_position(self):
        """§4.5 decision: an active plain object without a place entry keeps ObjectSpec.position (no draws)."""
        table = contract_color_task_table()
        table["objects"]["marker"] = {"size": [0.01, 0.01, 0.01], "position": [0.4, 0.3, 0.005]}
        spec = _spec(self.box, "fx_wp1_s_unplaced", table)
        sample = sample_start(spec, np.random.default_rng(0))
        self.assertTrue(sample.objects["marker"].active)
        self.assertEqual(tuple(sample.objects["marker"].pose.position), (0.4, 0.3, 0.005))


class GoalSamplerTests(unittest.TestCase):
    """WP1 §4.5 step 5 / §4.6 start.goal entries."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def test_reach_point_goal(self):
        """§4.6 point {region, z_m} -> [x, y, z] inside the polar region and z range."""
        spec = _spec(self.box, "fx_wp1_g_point", contract_reach_task_table(), files={})
        for k in range(100):
            sample = sample_start(spec, np.random.default_rng(k))
            x, y, z = sample.goal["point"]
            with self.subTest(seed=k):
                self.assertEqual(list(sample.goal), ["point"])
                self.assertEqual(dict(sample.objects), {})
                self.assertEqual(dict(sample.roles), {})
                self.assertEqual(sample.instruction, "Move the gripper to the point")
                self.assertTrue(0.15 - EPS <= math.hypot(x, y) <= 0.32 + EPS)
                self.assertTrue(-60.0 - 1e-6 <= math.degrees(math.atan2(y, x)) <= 60.0 + 1e-6)
                self.assertTrue(0.03 - EPS <= z <= 0.20 + EPS)
                self.assertTrue(all(isinstance(v, float) for v in (x, y, z)))

    def test_reachable_goal_point(self):
        """§4.5 decision: reachable may list "@goal.<key>"; each drawn point then satisfies reach_fn (TCP down)."""
        table = contract_reach_task_table()
        table["start"]["reachable"] = ["@goal.point"]
        spec = _spec(self.box, "fx_wp1_g_reach", table, files={})
        calls = []

        def reach(pose):
            calls.append(pose)
            return pose.position[2] < 0.08
        for k in range(30):
            with self.subTest(seed=k):
                self.assertLess(sample_start(spec, np.random.default_rng(k), reach_fn=reach).goal["point"][2], 0.08)
        self.assertTrue(all(tuple(p.quat_xyzw) == DOWN_XYZW for p in calls))
        table["start"]["max_tries"] = 5
        spec = _spec(self.box, "fx_wp1_g_reach_fail", table, files={})
        with self.assertRaises(SamplingError):
            sample_start(spec, np.random.default_rng(0), reach_fn=lambda pose: False)

    def test_other_goal_types(self):
        """§4.6 points, grid, gap, region and joint goal types, in TOML order."""
        rect = {"xy_min": [0.1, -0.1], "xy_max": [0.3, 0.1]}
        grid = [[0.1, 0.0, 0.1], [0.2, 0.1, 0.1], [0.25, -0.1, 0.15]]
        table = {"variants": {"teleop": "env_cfg:apply_teleop"},
                 "start": {"goal": {
                     "path": {"type": "points", "count": 3, "region": rect, "z_m": [0.05, 0.15], "min_separation_m": 0.05},
                     "cell": {"type": "grid", "points": grid},
                     "width": {"type": "gap", "gap_m": [0.01, 0.03]},
                     "zone": {"type": "region", "center_region": rect, "half_size_m": [0.03, 0.02]},
                     "sweep": {"type": "joint", "joints": "arm", "fraction": [0.2, 0.8], "count": 3},
                     "bend": {"type": "joint", "joints": ["j3"]},
                 }},
                 "stages": {"order": ["a"]}, "checks": {"a": {"check": "signal", "name": "a"}}}
        spec = _spec(self.box, "fx_wp1_g_types", table, files={})
        names = ("j1", "j2", "j3", "grip")
        arm = ("j1", "j2", "j3")
        lower, upper = np.array([-1.0, -2.0, 0.0, 0.0]), np.array([1.0, 0.0, 3.0, 0.04])
        sweep_joints = set()
        for k in range(60):
            g = sample_start(spec, np.random.default_rng(k), joint_limits=(lower, upper), joint_names=names,
                             arm_joints=arm).goal
            with self.subTest(seed=k):
                self.assertEqual(list(g), ["path", "cell", "width", "zone", "sweep", "bend"])
                self.assertEqual(len(g["path"]), 3)
                for p in g["path"]:
                    self.assertTrue(_inside(p, ((0.1, -0.1), (0.3, 0.1))) and 0.05 - EPS <= p[2] <= 0.15 + EPS)
                for i in range(3):
                    for j in range(i + 1, 3):
                        self.assertGreaterEqual(float(np.linalg.norm(np.subtract(g["path"][i], g["path"][j]))), 0.05 - EPS)
                self.assertIn([float(v) for v in g["cell"]], grid)
                self.assertTrue(0.01 - EPS <= g["width"] <= 0.03 + EPS)
                zmin, zmax = g["zone"]["xy_min"], g["zone"]["xy_max"]
                self.assertAlmostEqual(zmax[0] - zmin[0], 0.06, places=12)
                self.assertAlmostEqual(zmax[1] - zmin[1], 0.04, places=12)
                self.assertTrue(_inside(((zmin[0] + zmax[0]) / 2, (zmin[1] + zmax[1]) / 2), ((0.1, -0.1), (0.3, 0.1))))
                sweep = g["sweep"]
                self.assertIn(sweep["joint"], arm)
                i = names.index(sweep["joint"])
                self.assertEqual(len(sweep["angles"]), 3)
                for a in sweep["angles"]:
                    f = (a - lower[i]) / (upper[i] - lower[i])
                    self.assertTrue(0.2 - EPS <= f <= 0.8 + EPS)
                self.assertEqual(g["bend"]["joint"], "j3")
                self.assertEqual(len(g["bend"]["angles"]), 2)                    # count default 2
                for a in g["bend"]["angles"]:
                    self.assertTrue(0.1 * 3.0 - EPS <= a <= 0.9 * 3.0 + EPS)          # fraction default [0.1, 0.9]
            sweep_joints.add(g["sweep"]["joint"])
        self.assertEqual(sweep_joints, set(arm))
        with self.assertRaises(ValueError):
            sample_start(spec, np.random.default_rng(0))                          # joint goal needs names and limits


if __name__ == "__main__":
    unittest.main()
