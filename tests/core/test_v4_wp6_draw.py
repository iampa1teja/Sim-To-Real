"""WP6: ``draw_perturbation`` — kinds, ranges, determinism, coverage, exact draw order, weight overrides and
per-start applicability (contracts §7.1, §1.4, §3.2 EpisodeSpec invariants, §8.5 perturbation_weights,
docs/v4/prompts/WP6.md)."""

import math
import unittest

import numpy as np

from core.generate.errors import SamplingError
from core.generate.perturbations import draw_perturbation
from core.generate.protocols import Kind, ObjectState, PerturbationGroup, PerturbationSpec
from core.generate.seeds import rng_for
from core.tasks.base import load_task_spec
from core.tasks.generic import GenericTask
from v4_support import assert_frozen, sandbox
from v4_support.factories import make_pose
from v4_support.tasks import STUB_USDA, contract_color_task_table, contract_reach_task_table, write_task

BOX_FILES = {"usd/Small-Box-Physics.usda": STUB_USDA}
GROUP = {"grasp_offset": "expert", "early_close": "expert", "partial_close": "expert", "early_release": "expert",
         "place_offset": "expert", "wrong_target": "expert", "target_offset": "expert", "stall": "expert",
         "object_shift": "world", "object_drop": "world", "container_shift": "world"}
#: Declared parameter order and defaults per type (the draw order of WP6.md); None = required.
DECLARED = {
    "grasp_offset": [("xy_m", None), ("z_m", (0.0, 0.0)), ("yaw_rad", (0.0, 0.0))],
    "early_close": [("delay_steps", None)],
    "partial_close": [("extra_gap_m", (0.004, 0.012))],
    "early_release": [("delay_steps", None)],
    "place_offset": [("xy_m", None)],
    "wrong_target": [],
    "target_offset": [("xy_m", (0.0, 0.0)), ("z_m", (0.0, 0.0)), ("joint_rad", (0.0, 0.0)), ("gap_m", (0.0, 0.0))],
    "stall": [("delay_steps", (0, 0)), ("steps", None)],
    "object_shift": [("object", "@target"), ("delay_steps", (0, 0)), ("xy_m", None), ("yaw_rad", (0.0, 0.0))],
    "object_drop": [("delay_steps", (0, 0)), ("xy_jitter_m", (0.0, 0.0))],
    "container_shift": [("object", "@container"), ("delay_steps", (0, 0)), ("xy_m", None)],
}
INT_PARAMS = ("delay_steps", "steps")
SEEDS = range(300)


def load_tasks(tables):
    out = {}
    with sandbox() as box:
        for name, table in tables.items():
            write_task(box.tasks, name, table, files=BOX_FILES)
            out[name] = GenericTask(load_task_spec(name))
    return out


def reference_draw(entries, kind, rng, weights_override=None, applicable=None):
    """Test-side transcription of the draw order fixed in WP6.md (§7.1):
    1. entry index ``rng.choice(len(entries), p=w / w.sum())`` (w = weights after the override, 0 for an
       entry not applicable to the start);
    2. ``onset_stage = str(rng.choice(stages))`` when the entry lists stages, else ``None``;
    3. every declared parameter in declared order: ints ``rng.integers(lo, hi + 1)``, floats ``rng.uniform(lo, hi)``,
       strings copied; right after ``xy_m`` comes ``dir_rad = rng.uniform(0, 2π)``. ``delay_steps`` goes to the
       ``delay_steps`` field, every other parameter into ``params``."""
    if kind is Kind.SUCCESS:
        return None
    weights = []
    for i, e in enumerate(entries):
        w = float(e.get("weight", 1.0))
        if weights_override and e["type"] in weights_override:
            w = float(weights_override[e["type"]])
        if applicable is not None and not applicable[i]:
            w = 0.0
        weights.append(w)
    w = np.array(weights, dtype=float)
    entry = entries[int(rng.choice(len(w), p=w / w.sum()))]
    onset = str(rng.choice(entry["stages"])) if "stages" in entry else None
    delay, params = 0, {}
    for key, default in DECLARED[entry["type"]]:
        value = entry.get(key, default)
        if isinstance(value, str):
            params[key] = value
            continue
        lo, hi = value
        drawn = int(rng.integers(lo, hi + 1)) if key in INT_PARAMS else float(rng.uniform(lo, hi))
        if key == "delay_steps":
            delay = drawn
        else:
            params[key] = drawn
        if key == "xy_m":
            params["dir_rad"] = float(rng.uniform(0.0, 2.0 * math.pi))
    return PerturbationSpec(type=entry["type"], group=PerturbationGroup(GROUP[entry["type"]]), onset_stage=onset,
                            delay_steps=delay, params=params)


class DrawTests(unittest.TestCase):
    """WP6 §7.1 ``draw_perturbation(task, kind, rng, weights_override=None, *, start=None)``."""

    @classmethod
    def setUpClass(cls):
        cls.color_table = contract_color_task_table()
        cls.reach_table = contract_reach_task_table()
        tasks = load_tasks({"fx6_draw_color": cls.color_table, "fx6_draw_reach": cls.reach_table})
        cls.color, cls.reach = tasks["fx6_draw_color"], tasks["fx6_draw_reach"]
        cls.cases = ((cls.color, cls.color_table), (cls.reach, cls.reach_table))

    def test_success_never_carries_a_perturbation(self):
        """§7.1 / §3.2: ``Kind.SUCCESS`` gives ``None`` and consumes nothing from ``rng``."""
        for task, _ in self.cases:
            for seed in range(20):
                rng = np.random.default_rng(seed)
                before = rng.bit_generator.state
                with self.subTest(task=task.name, seed=seed):
                    self.assertIsNone(draw_perturbation(task, Kind.SUCCESS, rng))
                    self.assertEqual(rng.bit_generator.state, before)
                    self.assertIsNone(draw_perturbation(task, "success", np.random.default_rng(seed)))

    def test_failure_and_recovery_always_carry_one(self):
        """§3.2 invariant (success ⇔ no perturbation): failure and recovery draws are never ``None``, the type is
        one listed for that kind and the group matches the §7.1 table."""
        for task, table in self.cases:
            for kind in (Kind.FAILURE, Kind.RECOVERY):
                listed = {e["type"] for e in table["perturbations"][kind.value]}
                for seed in SEEDS:
                    with self.subTest(task=task.name, kind=kind.value, seed=seed):
                        spec = draw_perturbation(task, kind, np.random.default_rng(seed))
                        self.assertIsInstance(spec, PerturbationSpec)
                        self.assertIn(spec.type, listed)
                        self.assertEqual(spec.group.value, GROUP[spec.type])
                        assert_frozen(spec)

    def test_parameters_inside_declared_ranges(self):
        """§7.1 every drawn parameter lies in its TOML range (defaults when absent); ints for ``*_steps``/``steps``;
        ``dir_rad`` in [0, 2π) exactly when ``xy_m`` is a parameter; ``onset_stage`` is one of ``stages``."""
        for task, table in self.cases:
            for kind in (Kind.FAILURE, Kind.RECOVERY):
                entries = {e["type"]: e for e in table["perturbations"][kind.value]}
                for seed in SEEDS:
                    spec = draw_perturbation(task, kind, np.random.default_rng(seed))
                    entry = entries[spec.type]
                    with self.subTest(task=task.name, kind=kind.value, seed=seed, type=spec.type):
                        expected_keys = {k for k, _ in DECLARED[spec.type] if k != "delay_steps"}
                        if any(k == "xy_m" for k, _ in DECLARED[spec.type]):
                            expected_keys.add("dir_rad")
                        self.assertEqual(set(spec.params), expected_keys)
                        if "stages" in entry:
                            self.assertIn(spec.onset_stage, entry["stages"])
                        else:
                            self.assertIsNone(spec.onset_stage)
                        for key, default in DECLARED[spec.type]:
                            value = spec.delay_steps if key == "delay_steps" else spec.params[key]
                            rng_range = entry.get(key, default)
                            if isinstance(rng_range, str):
                                self.assertEqual(value, rng_range)
                                continue
                            lo, hi = rng_range
                            self.assertLessEqual(lo, value)
                            self.assertLessEqual(value, hi)
                            self.assertIs(type(value), int if key in INT_PARAMS else float)
                        if "dir_rad" in spec.params:
                            self.assertTrue(0.0 <= spec.params["dir_rad"] < 2 * math.pi)
                        self.assertIs(type(spec.delay_steps), int)

    def test_ranges_hit_both_halves(self):
        """§7.1 a range is sampled across its width, not pinned to an end (delay_steps in [5, 40])."""
        values = [spec.delay_steps for spec in (draw_perturbation(self.color, Kind.FAILURE, np.random.default_rng(s))
                                                for s in range(400)) if spec.type == "early_release"]
        self.assertGreater(len(values), 20)
        self.assertTrue(any(v < 22 for v in values) and any(v > 23 for v in values))

    def test_deterministic_per_seed(self):
        """§1.4 the same seed gives an equal spec; different seeds give varied specs."""
        for task, _ in self.cases:
            for kind in (Kind.FAILURE, Kind.RECOVERY):
                specs = []
                for seed in range(50):
                    a = draw_perturbation(task, kind, np.random.default_rng(seed))
                    b = draw_perturbation(task, kind, np.random.default_rng(seed))
                    with self.subTest(task=task.name, kind=kind.value, seed=seed):
                        self.assertEqual(a, b)
                        self.assertEqual(a.to_json(), b.to_json())
                    specs.append(a.to_json())
                self.assertGreater(len({repr(s) for s in specs}), 20)  # stall draws are all alike

    def test_every_declared_type_is_drawn(self):
        """§7.1 choice by weight: over many seeds each listed entry of each kind appears at least once, and the
        weight-2 grasp_offset is drawn about twice as often as a weight-1 entry."""
        for task, table in self.cases:
            for kind in (Kind.FAILURE, Kind.RECOVERY):
                counts = {}
                for seed in range(600):
                    spec = draw_perturbation(task, kind, np.random.default_rng(seed))
                    counts[spec.type] = counts.get(spec.type, 0) + 1
                with self.subTest(task=task.name, kind=kind.value):
                    self.assertEqual(set(counts), {e["type"] for e in table["perturbations"][kind.value]})
        counts = {}
        for seed in range(1200):
            spec = draw_perturbation(self.color, Kind.FAILURE, np.random.default_rng(seed))
            counts[spec.type] = counts.get(spec.type, 0) + 1
        self.assertAlmostEqual(counts["grasp_offset"] / 1200, 2 / 6, delta=0.05)
        self.assertAlmostEqual(counts["place_offset"] / 1200, 1 / 6, delta=0.05)

    def test_exact_draw_order(self):
        """WP6.md: the draw sequence is fixed (entry, stage, parameters in declared order, dir_rad after xy_m), so
        specs are reproducible across implementations; the result equals the test-side transcription."""
        for task, table in self.cases:
            for kind in (Kind.FAILURE, Kind.RECOVERY):
                for seed in range(200):
                    with self.subTest(task=task.name, kind=kind.value, seed=seed):
                        self.assertEqual(draw_perturbation(task, kind, np.random.default_rng(seed)),
                                         reference_draw(table["perturbations"][kind.value], kind,
                                                        np.random.default_rng(seed)))

    def test_uses_only_the_given_rng(self):
        """§1.4 the global numpy RNG is untouched; the result depends only on ``rng`` (the perturbation stream)."""
        np.random.seed(123)
        global_before = np.random.get_state()[1].copy()
        a = draw_perturbation(self.color, Kind.FAILURE, rng_for(77, "perturbation"))
        self.assertTrue(np.array_equal(np.random.get_state()[1], global_before))
        np.random.seed(999)
        self.assertEqual(draw_perturbation(self.color, Kind.FAILURE, rng_for(77, "perturbation")), a)

    def test_weights_override(self):
        """§8.5 ``perturbation_weights`` replaces the task weight of every entry of that type; weight 0 excludes it."""
        override = {"grasp_offset": 0.0, "place_offset": 5.0}
        counts = {}
        for seed in range(300):
            spec = draw_perturbation(self.color, Kind.FAILURE, np.random.default_rng(seed), override)
            counts[spec.type] = counts.get(spec.type, 0) + 1
            with self.subTest(seed=seed):
                self.assertEqual(spec, reference_draw(self.color_table["perturbations"]["failure"], Kind.FAILURE,
                                                      np.random.default_rng(seed), override))
        self.assertNotIn("grasp_offset", counts)
        self.assertGreater(counts["place_offset"], 300 * 5 / 9 - 40)
        only = {"grasp_offset": 0, "early_release": 0, "place_offset": 0, "wrong_target": 0}
        for seed in range(30):
            self.assertEqual(draw_perturbation(self.color, Kind.FAILURE, np.random.default_rng(seed), only).type,
                             "object_drop")

    def test_reach_stall_parameters(self):
        """§4.6 primitive example: stall gets ``onset_stage="reached"``, ``steps=600``; target_offset has
        ``xy_m`` in [0.03, 0.08] with a direction and zero defaults for z, joint and gap."""
        seen = set()
        for seed in range(100):
            spec = draw_perturbation(self.reach, Kind.FAILURE, np.random.default_rng(seed))
            seen.add(spec.type)
            with self.subTest(seed=seed):
                if spec.type == "stall":
                    self.assertEqual(spec.onset_stage, "reached")
                    self.assertEqual(dict(spec.params), {"steps": 600})
                    self.assertEqual(spec.delay_steps, 0)
                else:
                    self.assertIsNone(spec.onset_stage)
                    self.assertEqual({k: spec.params[k] for k in ("z_m", "joint_rad", "gap_m")},
                                     {"z_m": 0.0, "joint_rad": 0.0, "gap_m": 0.0})
                    self.assertTrue(0.03 <= spec.params["xy_m"] <= 0.08)
        self.assertEqual(seen, {"stall", "target_offset"})
        for seed in range(30):
            self.assertEqual(draw_perturbation(self.reach, Kind.RECOVERY, np.random.default_rng(seed)).type,
                             "target_offset")


def start_with(task, active_cubes):
    """A start (anything with ``objects`` and ``roles``) with ``active_cubes`` active members of the target set."""
    objects = {}
    for i, name in enumerate(task.spec.object_sets["cubes"].slot_names()):
        active = i < active_cubes
        objects[name] = ObjectState(make_pose(0.15 + 0.03 * i, 0.0, 0.01) if active else make_pose(0.1 * i, 0, -2.0),
                                    active=active, color="red" if active else None)
    objects["small_box"] = ObjectState(make_pose(0.25, 0.0, 0.0))
    return type("Start", (), {"objects": objects, "roles": {"target": "cube_0", "container": "small_box"}})()


class ApplicabilityTests(unittest.TestCase):
    """WP6 §7.1 wrong_target 'needs a set role and count >= 2': with ``start=`` the draw never picks a type that
    is not applicable to that start (its weight becomes 0); with nothing applicable it raises ``SamplingError``."""

    @classmethod
    def setUpClass(cls):
        table = contract_color_task_table()
        table["start"]["count"] = [1, 3]
        table["holdout"]["counts"] = []
        cls.table = table
        cls.task = load_tasks({"fx6_draw_applicable": table})["fx6_draw_applicable"]
        only_wrong = contract_color_task_table()
        only_wrong["start"]["count"] = [1, 3]
        only_wrong["holdout"]["counts"] = []
        only_wrong["perturbations"]["failure"] = [{"type": "wrong_target"}]
        cls.only_wrong = load_tasks({"fx6_draw_only_wrong": only_wrong})["fx6_draw_only_wrong"]

    def test_single_cube_never_gets_wrong_target(self):
        """One active cube: wrong_target is never drawn (and the draw equals the reference with its weight 0)."""
        entries = self.table["perturbations"]["failure"]
        applicable = [e["type"] != "wrong_target" for e in entries]
        start = start_with(self.task, 1)
        for seed in range(300):
            spec = draw_perturbation(self.task, Kind.FAILURE, np.random.default_rng(seed), start=start)
            with self.subTest(seed=seed):
                self.assertNotEqual(spec.type, "wrong_target")
                self.assertEqual(spec, reference_draw(entries, Kind.FAILURE, np.random.default_rng(seed),
                                                      applicable=applicable))

    def test_two_cubes_allow_wrong_target(self):
        """Two or more active cubes (or no start given): wrong_target is drawable and the draw is unchanged."""
        entries = self.table["perturbations"]["failure"]
        for active in (2, 3):
            start = start_with(self.task, active)
            types_seen = set()
            for seed in range(200):
                spec = draw_perturbation(self.task, Kind.FAILURE, np.random.default_rng(seed), start=start)
                types_seen.add(spec.type)
                with self.subTest(active=active, seed=seed):
                    self.assertEqual(spec, draw_perturbation(self.task, Kind.FAILURE, np.random.default_rng(seed)))
                    self.assertEqual(spec, reference_draw(entries, Kind.FAILURE, np.random.default_rng(seed)))
            self.assertIn("wrong_target", types_seen)

    def test_nothing_applicable(self):
        """Only wrong_target listed and one active cube: ``SamplingError`` (build_spec propagates it, the plan
        records ``sampling_failed``)."""
        with self.assertRaises(SamplingError):
            draw_perturbation(self.only_wrong, Kind.FAILURE, np.random.default_rng(0),
                              start=start_with(self.only_wrong, 1))
        spec = draw_perturbation(self.only_wrong, Kind.FAILURE, np.random.default_rng(0),
                                 start=start_with(self.only_wrong, 2))
        self.assertEqual(spec.type, "wrong_target")
        self.assertEqual(dict(spec.params), {})
        self.assertIsNone(spec.onset_stage)


if __name__ == "__main__":
    unittest.main()
