"""WP6: ``core.generate.perturbations`` — the perturbation taxonomy and the ``[task.perturbations]`` validator
(contracts §7.1, §4.6, §1.5, §14 decisions 2 and 14, docs/v4/prompts/WP6.md).

Tasks are built from the §4.6 contract example tables (``v4_support.tasks``) and the WP1 fixture
``fx_pick`` inside a ``sandbox``; no arm, scene or Isaac module is needed.
"""

import copy
import tomllib
import unittest

import numpy as np

from core.generate.errors import ConfigError
from core.generate.perturbations import PERTURBATION_TYPES, PerturbationType, draw_perturbation
from core.generate.protocols import Kind, PerturbationGroup
from core.tasks.base import load_task_spec
from core.tasks.generic import GenericTask
from v4_support import FIXTURES_DIR, assert_frozen, sandbox
from v4_support.tasks import STUB_USDA, contract_color_task_table, contract_reach_task_table, write_task

F, R = Kind.FAILURE, Kind.RECOVERY
A, B = PerturbationGroup.EXPERT, PerturbationGroup.WORLD
BOX_FILES = {"usd/Small-Box-Physics.usda": STUB_USDA}
COLOR_STAGES = ("reached", "grasped", "lifted", "over_box", "placed", "success")

#: §7.1 table, in table order: group, kinds, parameters with their defaults (None = required). WP6 prompt fixes
#: the defaults the table leaves open and puts delay_steps into the parameter list of every type whose trigger uses it.
TABLE = {
    "grasp_offset": (A, {F, R}, {"xy_m": None, "z_m": (0.0, 0.0), "yaw_rad": (0.0, 0.0)}),
    "early_close": (A, {F, R}, {"delay_steps": None}),
    "partial_close": (A, {F, R}, {"extra_gap_m": (0.004, 0.012)}),
    "early_release": (A, {F, R}, {"delay_steps": None}),
    "place_offset": (A, {F, R}, {"xy_m": None}),
    "wrong_target": (A, {F, R}, {}),
    "target_offset": (A, {F, R}, {"xy_m": (0.0, 0.0), "z_m": (0.0, 0.0), "joint_rad": (0.0, 0.0), "gap_m": (0.0, 0.0)}),
    "stall": (A, {F}, {"delay_steps": (0, 0), "steps": None}),
    "object_shift": (B, {F, R}, {"object": "@target", "delay_steps": (0, 0), "xy_m": None, "yaw_rad": (0.0, 0.0)}),
    "object_drop": (B, {F, R}, {"delay_steps": (0, 0), "xy_jitter_m": (0.0, 0.0)}),
    "container_shift": (B, {F, R}, {"object": "@container", "delay_steps": (0, 0), "xy_m": None}),
}


def load_tasks(tables):
    """{name: task table} -> {name: GenericTask}, written into a fresh sandbox (contract example tables)."""
    out = {}
    with sandbox() as box:
        for name, table in tables.items():
            write_task(box.tasks, name, table, files=BOX_FILES)
            out[name] = GenericTask(load_task_spec(name))
    return out


def fx_pick_table():
    """``[task]`` of the WP1 fixture ``fx_pick`` (single plain cube target, box container), without ``name``.
    Write it with ``files=BOX_FILES`` (its box USD is replaced by the stub at the same relative path)."""
    with open(FIXTURES_DIR / "v4_wp1" / "tasks" / "fx_pick" / "task.toml", "rb") as fh:
        table = tomllib.load(fh)["task"]
    table.pop("name")
    table["objects"]["box"]["usd"] = "usd/Small-Box-Physics.usda"
    return table


def color_with(failure=None, recovery=None, **start):
    """The §4.6 colour-cube table with replaced perturbation lists (and optional [task.start] keys)."""
    table = contract_color_task_table()
    if failure is not None:
        table["perturbations"]["failure"] = failure
    if recovery is not None:
        table["perturbations"]["recovery"] = recovery
    table["start"].update(start)
    return table


class TaxonomyTests(unittest.TestCase):
    """WP6 §7.1: ``PERTURBATION_TYPES`` describes the eleven types of the table."""

    def test_names_in_table_order(self):
        """§7.1 the eleven type names, in the order of the table."""
        self.assertEqual(list(PERTURBATION_TYPES), list(TABLE))

    def test_each_type(self):
        """§7.1 group (A = expert, B = world), allowed kinds (stall is failure-only, §14.2) and parameters."""
        for name, (group, kinds, params) in TABLE.items():
            ptype = PERTURBATION_TYPES[name]
            with self.subTest(type=name):
                self.assertIsInstance(ptype, PerturbationType)
                self.assertEqual(ptype.name, name)
                self.assertIs(ptype.group, group)
                self.assertEqual(set(ptype.kinds), kinds)
                self.assertEqual(list(ptype.params), list(params), "parameter names in declared (draw) order")
                self.assertEqual(dict(ptype.params), params)

    def test_frozen(self):
        """§3 conventions: the registry and its entries are read-only."""
        with self.assertRaises(TypeError):
            PERTURBATION_TYPES["new"] = PERTURBATION_TYPES["stall"]
        for name, ptype in PERTURBATION_TYPES.items():
            with self.subTest(type=name):
                assert_frozen(ptype)


class AllowedStagesTests(unittest.TestCase):
    """WP6 §7.1 'Trigger' column: the stages each type may name in ``stages``."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks({"fx6_types_color": contract_color_task_table(),
                                "fx6_types_reach": contract_reach_task_table()})
        cls.color = cls.tasks["fx6_types_color"]
        cls.reach = cls.tasks["fx6_types_reach"]

    def test_colour_task(self):
        """object_shift: stages before the first grasp step's stage; object_drop: stages whose check implies
        holding; container_shift: before the first release step's stage; target_offset: stages tagged on a step
        with a target, q or gap; wrong_target: none; the others: every task stage (onset_stage informational)."""
        expected = {
            "grasp_offset": COLOR_STAGES, "early_close": COLOR_STAGES, "partial_close": COLOR_STAGES,
            "early_release": COLOR_STAGES, "place_offset": COLOR_STAGES, "wrong_target": (),
            "target_offset": ("reached", "grasped", "over_box"), "stall": COLOR_STAGES,
            "object_shift": ("reached",), "object_drop": ("grasped", "lifted", "over_box"),
            "container_shift": ("reached", "grasped", "lifted", "over_box"),
        }
        for name, stages in expected.items():
            with self.subTest(type=name):
                self.assertEqual(tuple(PERTURBATION_TYPES[name].allowed_stages(self.color)), stages)

    def test_reach_task(self):
        """Primitive without a grasp or release step: shifts may use every stage; target_offset only the stage
        of the reach step (the wait step has no target); no stage implies holding."""
        expected = {"target_offset": ("reached",), "stall": ("reached", "success"),
                    "object_shift": ("reached", "success"), "container_shift": ("reached", "success"),
                    "object_drop": (), "wrong_target": ()}
        for name, stages in expected.items():
            with self.subTest(type=name):
                self.assertEqual(tuple(PERTURBATION_TYPES[name].allowed_stages(self.reach)), stages)

    def test_holding_implication_follows_all_chains(self):
        """object_drop: a stage implies holding if its check is ``holding``, a non-negated ``held`` check, or a
        non-negated ``all`` that (transitively) contains one; ``any`` and negated checks do not."""
        table = contract_color_task_table()
        table["checks"]["lifted"] = {"check": "all", "of": ["grasped", "raised"]}           # via grasped -> holding
        table["checks"]["over_box"] = {"check": "any", "of": ["holding", "above_box"]}     # any: no
        table["checks"]["grasped"] = {"check": "held", "object": "@target"}                 # held: yes
        task = load_tasks({"fx6_types_chain": table})["fx6_types_chain"]
        self.assertEqual(tuple(PERTURBATION_TYPES["object_drop"].allowed_stages(task)), ("grasped", "lifted"))


class ValidationTests(unittest.TestCase):
    """WP6 §7.1 validator: every problem of ``[task.perturbations]`` is a ``ConfigError`` (§1.5), raised by
    ``draw_perturbation`` for any kind, including ``Kind.SUCCESS``, before any draw."""

    def assert_config_error(self, table, name):
        task = load_tasks({name: table})[name]
        for kind in Kind:
            with self.subTest(case=name, kind=kind.value):
                rng = np.random.default_rng(0)
                with self.assertRaises(ConfigError):
                    draw_perturbation(task, kind, rng)

    def test_contract_examples_are_valid(self):
        """§4.6 both example tables validate and draw for every kind."""
        tasks = load_tasks({"fx6_val_color": contract_color_task_table(), "fx6_val_reach": contract_reach_task_table()})
        for name, task in tasks.items():
            for kind in Kind:
                with self.subTest(task=name, kind=kind.value):
                    draw_perturbation(task, kind, np.random.default_rng(1))

    def test_invalid_entries(self):
        """Unknown type, kind not allowed, bad stages, unknown / missing / malformed parameters, bad weights."""
        grasp = {"type": "grasp_offset", "xy_m": [0.008, 0.02]}
        cases = {
            "unknown_type": [{"type": "teleport"}],
            "missing_type": [{"weight": 1.0}],
            "entry_not_table": ["grasp_offset"],
            "stages_not_task_stage": [dict(grasp, stages=["nowhere"])],
            "stages_empty": [dict(grasp, stages=[])],
            "stages_not_list": [dict(grasp, stages="reached")],
            "object_shift_after_grasp": [{"type": "object_shift", "stages": ["lifted"], "xy_m": [0.02, 0.05]}],
            "object_drop_not_holding": [{"type": "object_drop", "stages": ["reached"]}],
            "object_drop_placed": [{"type": "object_drop", "stages": ["placed"]}],
            "container_shift_at_placed": [{"type": "container_shift", "stages": ["placed"], "xy_m": [0.02, 0.05]}],
            "target_offset_untagged_stage": [{"type": "target_offset", "stages": ["lifted"], "xy_m": [0.01, 0.02]}],
            "wrong_target_with_stages": [{"type": "wrong_target", "stages": ["reached"]}],
            "stall_without_stages": [{"type": "stall", "steps": [10, 20]}],
            "object_shift_without_stages": [{"type": "object_shift", "xy_m": [0.02, 0.05]}],
            "missing_required_xy": [{"type": "grasp_offset"}],
            "missing_required_delay": [{"type": "early_release"}],
            "missing_required_steps": [{"type": "stall", "stages": ["reached"]}],
            "unknown_param": [dict(grasp, steps=[1, 2])],
            "delay_not_allowed": [dict(grasp, delay_steps=[1, 2])],
            "range_reversed": [{"type": "grasp_offset", "xy_m": [0.02, 0.008]}],
            "range_scalar": [{"type": "grasp_offset", "xy_m": 0.01}],
            "range_three": [{"type": "grasp_offset", "xy_m": [0.0, 0.01, 0.02]}],
            "range_string": [{"type": "grasp_offset", "xy_m": ["a", "b"]}],
            "range_bool": [{"type": "grasp_offset", "xy_m": [False, True]}],
            "range_nan": [{"type": "grasp_offset", "xy_m": [float("nan"), 0.02]}],
            "negative_magnitude": [{"type": "grasp_offset", "xy_m": [-0.01, 0.02]}],
            "float_steps": [{"type": "early_release", "delay_steps": [1.5, 3.0]}],
            "negative_delay": [{"type": "early_release", "delay_steps": [-1, 3]}],
            "zero_stall_steps": [{"type": "stall", "stages": ["reached"], "steps": [0, 5]}],
            "object_not_string": [{"type": "object_shift", "stages": ["reached"], "xy_m": [0.02, 0.05], "object": 3}],
            "object_unknown_role": [{"type": "object_shift", "stages": ["reached"], "xy_m": [0.02, 0.05],
                                     "object": "@nobody"}],
            "object_unknown_name": [{"type": "object_shift", "stages": ["reached"], "xy_m": [0.02, 0.05],
                                     "object": "no_such_object"}],
            "weight_negative": [dict(grasp, weight=-1.0)],
            "weight_string": [dict(grasp, weight="2")],
            "weight_bool": [dict(grasp, weight=True)],
        }
        for case, failure in cases.items():
            self.assert_config_error(color_with(failure=failure), f"fx6_bad_{case}")

    def test_kind_not_allowed(self):
        """§7.1 'Kinds' column: stall is failure-only."""
        self.assert_config_error(color_with(recovery=[{"type": "stall", "stages": ["reached"], "steps": [5, 5]}]),
                                 "fx6_bad_stall_recovery")

    def test_unknown_kind_key(self):
        """§4.6 ``[task.perturbations]`` has only ``failure`` and ``recovery`` lists."""
        table = contract_color_task_table()
        table["perturbations"]["success"] = [{"type": "grasp_offset", "xy_m": [0.01, 0.02]}]
        self.assert_config_error(table, "fx6_bad_success_key")

    def test_requirements_on_the_task(self):
        """A type needs what its trigger reads: the expert step it hooks into, the roles it moves, a target set
        with count >= 2 for wrong_target."""
        reach = contract_reach_task_table()
        cases = {}
        for ptype, extra in (("grasp_offset", {"xy_m": [0.01, 0.02]}), ("partial_close", {}),
                             ("early_close", {"delay_steps": [1, 2]}), ("early_release", {"delay_steps": [1, 2]}),
                             ("place_offset", {"xy_m": [0.01, 0.02]}), ("wrong_target", {}),
                             ("object_shift", {"stages": ["reached"], "xy_m": [0.01, 0.02]}),
                             ("container_shift", {"stages": ["reached"], "xy_m": [0.01, 0.02]})):
            table = copy.deepcopy(reach)
            table["perturbations"]["failure"] = [dict({"type": ptype}, **extra)]
            cases[f"fx6_req_reach_{ptype}"] = table
        no_targets = copy.deepcopy(reach)
        no_targets["expert"]["steps"] = [{"skill": "move_joint", "q": "home"}, {"skill": "wait", "steps": 5}]
        no_targets["perturbations"]["failure"] = [{"type": "target_offset", "xy_m": [0.01, 0.02]}]
        no_targets["perturbations"]["recovery"] = [{"type": "target_offset", "xy_m": [0.01, 0.02]}]
        cases["fx6_req_reach_no_target_step"] = no_targets
        single = color_with(failure=[{"type": "wrong_target"}], count=[1, 1])
        single["holdout"]["counts"] = []
        cases["fx6_req_single_cube"] = single
        for name, table in cases.items():
            self.assert_config_error(table, name)

    def test_wrong_target_needs_a_set_target(self):
        """§7.1 wrong_target 'needs a set role': fx_pick's target is a single plain cube."""
        table = fx_pick_table()
        table["perturbations"]["failure"] = [{"type": "wrong_target"}]
        self.assert_config_error(table, "fx6_bad_pick_wrong_target")

    def test_failure_kind_needs_entries(self):
        """Every task has failure and recovery (§14.2): drawing a kind with no entries is a ConfigError."""
        table = contract_color_task_table()
        table["perturbations"] = {"recovery": table["perturbations"]["recovery"]}
        task = load_tasks({"fx6_no_failure": table})["fx6_no_failure"]
        with self.assertRaises(ConfigError):
            draw_perturbation(task, Kind.FAILURE, np.random.default_rng(0))
        self.assertIsNotNone(draw_perturbation(task, Kind.RECOVERY, np.random.default_rng(0)))
        self.assertIsNone(draw_perturbation(task, Kind.SUCCESS, np.random.default_rng(0)))

    def test_all_zero_weights(self):
        """A kind whose weights are all zero cannot draw: ConfigError."""
        failure = [{"type": "grasp_offset", "weight": 0, "xy_m": [0.01, 0.02]},
                   {"type": "place_offset", "weight": 0.0, "xy_m": [0.01, 0.02]}]
        task = load_tasks({"fx6_zero_weights": color_with(failure=failure)})["fx6_zero_weights"]
        with self.assertRaises(ConfigError):
            draw_perturbation(task, Kind.FAILURE, np.random.default_rng(0))
        self.assertIsNotNone(draw_perturbation(task, Kind.RECOVERY, np.random.default_rng(0)))

    def test_weights_override_validation(self):
        """§8.5 ``perturbation_weights``: names must be perturbation types (ConfigError otherwise), weights finite
        and >= 0; a known type the task does not list is ignored; zeroing every entry of a kind is a ConfigError."""
        task = load_tasks({"fx6_override": contract_color_task_table()})["fx6_override"]
        for override in ({"teleport": 1.0}, {"grasp_offset": -1.0}, {"grasp_offset": float("inf")},
                         {"grasp_offset": "1"}):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                draw_perturbation(task, Kind.FAILURE, np.random.default_rng(0), override)
        self.assertIsNotNone(draw_perturbation(task, Kind.FAILURE, np.random.default_rng(0), {"stall": 5.0}))
        zero_all = {name: 0.0 for name in ("grasp_offset", "early_release", "place_offset", "wrong_target",
                                           "object_drop")}
        with self.assertRaises(ConfigError):
            draw_perturbation(task, Kind.FAILURE, np.random.default_rng(0), zero_all)


if __name__ == "__main__":
    unittest.main()
