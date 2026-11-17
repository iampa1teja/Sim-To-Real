"""WP3: skill registry and `[task.expert]` program parser (contracts §6.1, §4.4 ref syntax, §1.5).

Decisions pinned here (see docs/v4/prompts/WP3.md):
* ``Skill(name, required, defaults, budget)``; ``budget is None`` only for ``wait`` (its budget is ``steps``).
* ``reach`` defaults: ``offset_m=(0,0,0)``, ``orientation="down"``, ``yaw="keep"``, ``tol_m=0.005``.
* ``parse_program(table, *, task, arm_spec, joint_limits=None)``: ``arm_spec`` is the ``[arm]`` dict of
  ``load_arm_spec(arm)[0]``; ``joint_limits`` = ``(lower, upper)`` in joint-vector order (normally
  ``expert_joint_limits``) and enables the reachability checks.
* ``ProgramStep(skill, name, stage, budget, params)``: ``params`` = defaults overlaid by the step's own
  parameters (``skill``/``name``/``stage``/``budget`` removed), deep-frozen, refs left unresolved,
  ``q = "home"`` resolved to ``{arm_joint: rad}`` (``[arm.expert].home_rad`` over
  ``[arm.initial_state].joint_positions``, arm joints only).
* ``ExpertProgram(steps, max_steps, settle_steps, max_retries, objects, approach_axis, closing_axis)``.
"""

import math
import types
import unittest

import numpy as np

from core.expert.program import ExpertProgram, ProgramStep, parse_program
from core.expert.skills import SKILLS, Skill
from core.generate.errors import ConfigError
from core.tasks.loader import load_task
from v4_support import assert_frozen, clear_task_cache, copy_fixture, sandbox, temp_env
from v4_support.fx_arms import FX_ARMS, FX_OPENING_TABLE, fx_arm_table, fx_home_q, fx_joint_names, fx_limits
from v4_support.tasks import STUB_USDA, write_task

SKILL_ORDER = ("move_joint", "set_gripper", "reach", "approach", "grasp", "lift", "transport", "place", "release",
               "retreat", "push", "touch", "sweep", "wait")
STAGES = ("reached", "grasped", "lifted", "placed", "success")


def tool_task_table():
    """A GenericTask with a cube, a second cube, a container and four goal kinds; stage checks are signals."""
    return {
        "variants": {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"},
        "objects": {"cube": {"size": [0.02, 0.02, 0.02]}, "cube_b": {"size": [0.02, 0.02, 0.02]},
                    "box": {"usd": "usd/box.usda", "origin": "bottom", "size_outer": [0.05, 0.05, 0.035],
                            "inner_half_size": [0.02, 0.02], "floor_z": 0.002}},
        "roles": {"target": "cube", "container": "box"},
        "start": {"goal": {
            "point": {"type": "point", "region": {"xy_min": [0.12, -0.05], "xy_max": [0.18, 0.05]}, "z_m": [0.0, 0.05]},
            "joint": {"type": "joint", "joints": "arm"},
            "gap": {"type": "gap", "gap_m": [0.01, 0.03]},
            "area": {"type": "region", "center_region": {"xy_min": [0.14, 0.0], "xy_max": [0.16, 0.02]},
                     "half_size_m": [0.01, 0.01]}}},
        "stages": {"order": list(STAGES)},
        "checks": {s: {"check": "signal", "name": s} for s in STAGES},
        "failures": {"rules": [{"name": "failed"}]},
    }


def load_tool_task(box, name="fx_wp3_tool"):
    write_task(box.tasks, name, tool_task_table(), files={"usd/box.usda": STUB_USDA})
    clear_task_cache(name)
    return load_task(name)


def load_fixture_task(box, name):
    """Load tests/fixtures/v4_wp1/tasks/<name> from a temp copy (never import from the fixture folder)."""
    copy_fixture(f"v4_wp1/tasks/{name}", box.root)
    with temp_env(S2R_TASKS_DIR=box.root / "v4_wp1" / "tasks"):
        clear_task_cache(name)
        task = load_task(name)
    clear_task_cache(name)
    return task


class SkillRegistryTests(unittest.TestCase):
    """WP3 §6.1 skill table."""

    def test_names_and_order(self):
        """§6.1 exactly the 14 skills, in table order; each entry is a Skill named after its key."""
        self.assertEqual(tuple(SKILLS), SKILL_ORDER)
        for name, skill in SKILLS.items():
            self.assertIsInstance(skill, Skill)
            self.assertEqual(skill.name, name)
            assert_frozen(skill)

    def test_budgets(self):
        """§6.1 default budgets; wait's budget is its ``steps`` (None in the registry)."""
        expected = {"move_joint": 150, "set_gripper": 60, "reach": 200, "approach": 200, "grasp": 150, "lift": 120,
                    "transport": 250, "place": 150, "release": 60, "retreat": 120, "push": 300, "touch": 200,
                    "sweep": 300, "wait": None}
        self.assertEqual({n: s.budget for n, s in SKILLS.items()}, expected)

    def test_parameters(self):
        """§6.1 required parameters and defaults (offset_m as a tuple)."""
        expected = {
            "move_joint": (("q",), {"tol_rad": 0.01}),
            "set_gripper": (("gap_m",), {"tol_m": 0.002}),
            "reach": (("target",), {"offset_m": (0.0, 0.0, 0.0), "orientation": "down", "yaw": "keep", "tol_m": 0.005}),
            "approach": (("target",), {"offset_m": (0.0, 0.0, 0.06), "orientation": "down", "yaw": "object",
                                       "tol_m": 0.005}),
            "grasp": (("target",), {"z_offset_m": 0.0, "close_gap_m": 0.0, "dwell_steps": 10}),
            "lift": ((), {"height_m": 0.08}),
            "transport": (("target",), {"height_m": 0.06}),
            "place": (("target",), {"offset_m": (0.0, 0.0, 0.0), "clearance_m": 0.005}),
            "release": ((), {"gap_m": "open", "dwell_steps": 10}),
            "retreat": ((), {"dz_m": 0.08}),
            "push": (("target", "goal"), {"standoff_m": 0.03, "depth_m": 0.0}),
            "touch": (("target",), {"contact_tol_m": 0.004}),
            "sweep": (("joint", "angles"), {"tol_rad": 0.01}),
            "wait": (("steps",), {}),
        }
        for name, (required, defaults) in expected.items():
            with self.subTest(skill=name):
                self.assertEqual(SKILLS[name].required, required)
                self.assertEqual(dict(SKILLS[name].defaults), defaults)


class ContractProgramTests(unittest.TestCase):
    """WP3 §4.6 / §6.1 the two contract example programs (fixture tasks fx_reach, fx_color) parse."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def test_fx_reach(self):
        """§4.6 prim_reach_point program: move_joint home, reach @goal.point [reached], wait 10 [success]."""
        task = load_fixture_task(self.box, "fx_reach")
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                spec = fx_arm_table(arm)
                prog = parse_program(task.spec.raw["expert"], task=task, arm_spec=spec, joint_limits=fx_limits(arm))
                self.assertIsInstance(prog, ExpertProgram)
                self.assertEqual((prog.max_steps, prog.settle_steps, prog.max_retries), (600, 10, 2))
                self.assertEqual([s.skill for s in prog.steps], ["move_joint", "reach", "wait"])
                self.assertEqual([s.name for s in prog.steps], ["move_joint", "reach", "wait"])
                self.assertEqual([s.stage for s in prog.steps], [None, "reached", "success"])
                self.assertEqual([s.budget for s in prog.steps], [150, 200, 10])
                arm_joints = spec["expert"]["arm_joints"]
                home = dict(zip(fx_joint_names(arm), fx_home_q(arm)))
                self.assertEqual(dict(prog.steps[0].params["q"]), {j: home[j] for j in arm_joints})
                self.assertEqual(prog.steps[0].params["tol_rad"], 0.01)
                self.assertEqual(dict(prog.steps[1].params), {"target": "@goal.point", "offset_m": (0.0, 0.0, 0.0),
                                                              "orientation": "down", "yaw": "keep", "tol_m": 0.005})
                self.assertEqual(dict(prog.steps[2].params), {"steps": 10})
                self.assertEqual(prog.approach_axis, tuple(spec["kinematics"]["approach_axis"]))
                self.assertEqual(prog.closing_axis, tuple(spec["kinematics"]["closing_axis"]))
                for step in prog.steps:
                    self.assertIsInstance(step, ProgramStep)
                    assert_frozen(step)

    def test_fx_color(self):
        """§4.6 pick_place_color_cube program: 9 steps, stage tags, budgets, defaults filled, objects carried."""
        task = load_fixture_task(self.box, "fx_color")
        prog = parse_program(task.spec.raw["expert"], task=task, arm_spec=fx_arm_table("fx_arm5"))
        self.assertEqual(prog.max_steps, 600)
        self.assertEqual([s.skill for s in prog.steps],
                         ["move_joint", "set_gripper", "approach", "grasp", "lift", "transport", "place", "release",
                          "retreat"])
        self.assertEqual([s.stage for s in prog.steps],
                         [None, None, "reached", "grasped", "lifted", "over_box", None, "placed", "success"])
        self.assertEqual([s.budget for s in prog.steps], [150, 60, 200, 150, 120, 250, 150, 60, 120])
        self.assertEqual(prog.steps[1].params["gap_m"], "open")
        self.assertEqual(prog.steps[2].params["offset_m"], (0, 0, 0.06))
        self.assertEqual(prog.steps[2].params["yaw"], "object")
        self.assertEqual(dict(prog.steps[3].params), {"target": "@target", "z_offset_m": 0.0, "close_gap_m": 0.0,
                                                      "dwell_steps": 10})
        self.assertEqual(prog.steps[5].params["height_m"], 0.06)
        self.assertEqual(dict(prog.steps[7].params), {"gap_m": "open", "dwell_steps": 10})
        self.assertEqual(prog.steps[8].params["dz_m"], 0.08)
        self.assertEqual(set(prog.objects), set(task.spec.objects))
        self.assertEqual(prog.objects["small_box"], task.spec.objects["small_box"])


class ParseValidTests(unittest.TestCase):
    """WP3 §6.1 accepted forms."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()
        self.task = load_tool_task(self.box)
        self.arm = fx_arm_table("fx_arm6")
        self.limits = fx_limits("fx_arm6")

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def parse(self, steps, **top):
        return parse_program({"steps": steps, **top}, task=self.task, arm_spec=self.arm, joint_limits=self.limits)

    def test_top_level_keys(self):
        """§6.1 max_steps, settle_steps, max_retries override the defaults 600 / 10 / 2."""
        prog = self.parse([{"skill": "wait", "steps": 3}], max_steps=50, settle_steps=0, max_retries=0)
        self.assertEqual((prog.max_steps, prog.settle_steps, prog.max_retries), (50, 0, 0))

    def test_name_stage_budget(self):
        """§6.1 name (default = skill), stage, explicit budget; wait's default budget is steps."""
        prog = self.parse([{"skill": "wait", "steps": 7},
                           {"skill": "wait", "steps": 7, "budget": 3, "name": "pause", "stage": "lifted"},
                           {"skill": "lift", "budget": 5}])
        self.assertEqual([(s.name, s.stage, s.budget) for s in prog.steps],
                         [("wait", None, 7), ("pause", "lifted", 3), ("lift", None, 5)])

    def test_every_skill_and_ref_form(self):
        """§6.1/§4.4 refs: @role, @role.inner, @goal.key(.part), literal object names, literal points/regions,
        "open"/"closed", numbers, {joint = rad}, yaw as a number; parameters kept unresolved and deep-frozen."""
        names = fx_joint_names("fx_arm6")
        steps = [
            {"skill": "move_joint", "q": "home"},
            {"skill": "move_joint", "q": {names[0]: 0.3, names[2]: -0.2}, "tol_rad": 0.02},
            {"skill": "move_joint", "q": "@goal.joint"},
            {"skill": "set_gripper", "gap_m": "closed"},
            {"skill": "set_gripper", "gap_m": 0.02},
            {"skill": "set_gripper", "gap_m": "@goal.gap"},
            {"skill": "reach", "target": "@goal.point", "yaw": 0.5, "orientation": "keep"},
            {"skill": "reach", "target": [0.15, 0.0, 0.05]},
            {"skill": "reach", "target": "cube_b", "offset_m": [0.0, 0.0, 0.03]},
            {"skill": "approach", "target": "@target"},
            {"skill": "grasp", "target": "@target", "z_offset_m": -0.002, "close_gap_m": 0.004, "dwell_steps": 0},
            {"skill": "lift", "height_m": 0.05},
            {"skill": "transport", "target": "@container"},
            {"skill": "transport", "target": "@container.inner"},
            {"skill": "transport", "target": "@goal.area"},
            {"skill": "transport", "target": {"xy_min": [0.1, 0.0], "xy_max": [0.12, 0.02]}},
            {"skill": "place", "target": "@container", "offset_m": [0.0, 0.0, 0.01], "clearance_m": 0.0},
            {"skill": "release"},
            {"skill": "release", "gap_m": 0.03, "dwell_steps": 4},
            {"skill": "retreat", "dz_m": 0.05},
            {"skill": "push", "target": "@target", "goal": "@goal.area"},
            {"skill": "push", "target": "cube_b", "goal": [0.15, 0.05], "standoff_m": 0.02, "depth_m": -0.002},
            {"skill": "push", "target": "@target", "goal": "@container.inner"},
            {"skill": "touch", "target": "@target"},
            {"skill": "sweep", "joint": names[1], "angles": [0.1, -0.1]},
            {"skill": "sweep", "joint": "@goal.joint", "angles": "@goal.joint.angles"},
            {"skill": "wait", "steps": 1},
        ]
        prog = self.parse(steps)
        self.assertEqual(len(prog.steps), len(steps))
        for raw, step in zip(steps, prog.steps):
            with self.subTest(step=raw):
                self.assertEqual(step.skill, raw["skill"])
                assert_frozen(step)
                for key, value in raw.items():
                    if key in ("skill",) or (raw["skill"] == "move_joint" and key == "q"):
                        continue
                    got = step.params[key]
                    if isinstance(value, list):
                        self.assertEqual(got, tuple(value))
                    elif isinstance(value, dict):
                        self.assertIsInstance(got, types.MappingProxyType)
                        self.assertEqual({k: tuple(v) for k, v in got.items()}, {k: tuple(v) for k, v in value.items()})
                    else:
                        self.assertEqual(got, value)
        self.assertEqual(dict(prog.steps[1].params["q"]), {names[0]: 0.3, names[2]: -0.2})
        self.assertEqual(prog.steps[2].params["q"], "@goal.joint")

    def test_home_rad_overrides_initial_state(self):
        """§5.1 [arm.expert].home_rad (optional) overrides [arm.initial_state].joint_positions for "home"."""
        arm = fx_arm_table("fx_arm6")
        first = arm["expert"]["arm_joints"][0]
        arm["expert"]["home_rad"] = {first: 0.25}
        prog = parse_program({"steps": [{"skill": "move_joint", "q": "home"}]}, task=self.task, arm_spec=arm)
        q = dict(prog.steps[0].params["q"])
        self.assertEqual(q[first], 0.25)
        self.assertEqual(set(q), set(arm["expert"]["arm_joints"]))
        self.assertNotIn(arm["gripper"]["joint"], q)

    def test_input_not_mutated(self):
        """parse_program never mutates its input table."""
        steps = [{"skill": "reach", "target": [0.15, 0.0, 0.05]}]
        table = {"steps": steps}
        self.parse(steps)
        parse_program(table, task=self.task, arm_spec=self.arm)
        self.assertEqual(table, {"steps": [{"skill": "reach", "target": [0.15, 0.0, 0.05]}]})


class ParseInvalidTests(unittest.TestCase):
    """WP3 §6.1 / §1.5: every invalid program raises ConfigError at load time."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()
        self.task = load_tool_task(self.box)
        self.arm = fx_arm_table("fx_arm6")
        self.names = fx_joint_names("fx_arm6")
        lo, hi = fx_limits("fx_arm6")
        self.limits = (lo + 0.1, hi - 0.1)

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def assert_rejected(self, table, arm_spec=None, joint_limits="default"):
        limits = self.limits if joint_limits == "default" else joint_limits
        with self.assertRaises(ConfigError):
            parse_program(table, task=self.task, arm_spec=arm_spec or self.arm, joint_limits=limits)

    def test_table_shape(self):
        """§6.1 steps required, a non-empty list of tables; only the four top-level keys; typed top-level values."""
        cases = {
            "no_steps": {},
            "empty_steps": {"steps": []},
            "steps_not_list": {"steps": {"skill": "wait", "steps": 1}},
            "step_not_table": {"steps": ["wait"]},
            "unknown_top_key": {"steps": [{"skill": "wait", "steps": 1}], "speed": 2},
            "max_steps_zero": {"steps": [{"skill": "wait", "steps": 1}], "max_steps": 0},
            "max_steps_float": {"steps": [{"skill": "wait", "steps": 1}], "max_steps": 10.0},
            "settle_negative": {"steps": [{"skill": "wait", "steps": 1}], "settle_steps": -1},
            "retries_negative": {"steps": [{"skill": "wait", "steps": 1}], "max_retries": -1},
            "retries_bool": {"steps": [{"skill": "wait", "steps": 1}], "max_retries": True},
        }
        for label, table in cases.items():
            with self.subTest(case=label):
                self.assert_rejected(table)

    def test_bad_steps(self):
        """§6.1 unknown skill or key, missing required parameter (grasp without target, …), unknown stage,
        non-positive budget, bad name."""
        n0 = self.names[0]
        cases = {
            "unknown_skill": {"skill": "teleport", "target": "@target"},
            "missing_skill": {"target": "@target"},
            "unknown_key": {"skill": "lift", "height_m": 0.05, "speed": 1.0},
            "key_of_other_skill": {"skill": "lift", "dz_m": 0.05},
            "grasp_without_target": {"skill": "grasp"},
            "reach_without_target": {"skill": "reach"},
            "push_without_goal": {"skill": "push", "target": "@target"},
            "wait_without_steps": {"skill": "wait"},
            "sweep_without_angles": {"skill": "sweep", "joint": n0},
            "move_joint_without_q": {"skill": "move_joint"},
            "unknown_stage": {"skill": "lift", "stage": "flying"},
            "budget_zero": {"skill": "lift", "budget": 0},
            "budget_negative": {"skill": "lift", "budget": -3},
            "budget_float": {"skill": "lift", "budget": 10.0},
            "budget_bool": {"skill": "lift", "budget": True},
            "empty_name": {"skill": "lift", "name": ""},
            "name_not_str": {"skill": "lift", "name": 3},
        }
        for label, step in cases.items():
            with self.subTest(case=label):
                self.assert_rejected({"steps": [{"skill": "wait", "steps": 1}, step]})

    def test_bad_refs(self):
        """§4.4/§6.1 unknown role, goal key or object; bad suffix; .inner of a non-container; refs where not allowed."""
        cases = {
            "unknown_role": {"skill": "approach", "target": "@cup"},
            "unknown_goal_key": {"skill": "reach", "target": "@goal.nowhere"},
            "unknown_object": {"skill": "grasp", "target": "cube_z"},
            "bad_suffix": {"skill": "approach", "target": "@target.top"},
            "inner_of_cube": {"skill": "transport", "target": "@target.inner"},
            "grasp_goal_point": {"skill": "grasp", "target": "@goal.point"},
            "push_unknown_goal": {"skill": "push", "target": "@target", "goal": "@goal.nowhere"},
            "gap_unknown_goal": {"skill": "set_gripper", "gap_m": "@goal.width"},
            "q_unknown_goal": {"skill": "move_joint", "q": "@goal.pose"},
            "sweep_unknown_goal": {"skill": "sweep", "joint": "@goal.nowhere", "angles": [0.1]},
        }
        for label, step in cases.items():
            with self.subTest(case=label):
                self.assert_rejected({"steps": [step]})

    def test_bad_parameters(self):
        """§6.1 wrong types and ranges of skill parameters."""
        cases = {
            "offset_len2": {"skill": "approach", "target": "@target", "offset_m": [0.0, 0.06]},
            "offset_nan": {"skill": "approach", "target": "@target", "offset_m": [0.0, 0.0, math.nan]},
            "orientation": {"skill": "reach", "target": "@target", "orientation": "up"},
            "yaw_word": {"skill": "approach", "target": "@target", "yaw": "north"},
            "yaw_inf": {"skill": "approach", "target": "@target", "yaw": math.inf},
            "tol_zero": {"skill": "reach", "target": "@target", "tol_m": 0.0},
            "tol_rad_negative": {"skill": "move_joint", "q": "home", "tol_rad": -0.01},
            "contact_tol_zero": {"skill": "touch", "target": "@target", "contact_tol_m": 0.0},
            "dwell_float": {"skill": "release", "dwell_steps": 2.5},
            "dwell_negative": {"skill": "grasp", "target": "@target", "dwell_steps": -1},
            "wait_zero": {"skill": "wait", "steps": 0},
            "height_negative": {"skill": "lift", "height_m": -0.02},
            "height_str": {"skill": "transport", "target": "@container", "height_m": "high"},
            "clearance_negative": {"skill": "place", "target": "@container", "clearance_m": -0.001},
            "gap_word": {"skill": "set_gripper", "gap_m": "half"},
            "q_empty_table": {"skill": "move_joint", "q": {}},
            "q_unknown_joint": {"skill": "move_joint", "q": {"fx6_j9": 0.1}},
            "q_list": {"skill": "move_joint", "q": [0.0] * 7},
            "q_nan": {"skill": "move_joint", "q": {self.names[0]: math.nan}},
            "sweep_unknown_joint": {"skill": "sweep", "joint": "fx6_j9", "angles": [0.1]},
            "sweep_gripper": {"skill": "sweep", "joint": self.arm["gripper"]["joint"], "angles": [0.1]},
            "sweep_empty_angles": {"skill": "sweep", "joint": self.names[0], "angles": []},
            "region_bad": {"skill": "transport", "target": {"xy_min": [0.1, 0.0]}},
            "region_inverted": {"skill": "transport", "target": {"xy_min": [0.2, 0.0], "xy_max": [0.1, 0.1]}},
            "point_len4": {"skill": "reach", "target": [0.1, 0.0, 0.05, 1.0]},
        }
        for label, step in cases.items():
            with self.subTest(case=label):
                self.assert_rejected({"steps": [step]})

    def test_unreachable_configurations(self):
        """§6.1 configurations outside the arm's range: joint targets outside joint_limits (literal q, home, sweep
        angles) and gripper gaps outside the [arm.gripper].opening_table range."""
        lo, hi = self.limits
        gaps = [row[1] for row in FX_OPENING_TABLE]
        cases = {
            "q_above_limit": {"skill": "move_joint", "q": {self.names[1]: float(hi[1]) + 0.05}},
            "q_below_limit": {"skill": "move_joint", "q": {self.names[2]: float(lo[2]) - 0.05}},
            "sweep_angle_outside": {"skill": "sweep", "joint": self.names[0], "angles": [0.0, float(hi[0]) + 0.01]},
            "gap_too_wide": {"skill": "set_gripper", "gap_m": max(gaps) + 0.001},
            "gap_negative": {"skill": "set_gripper", "gap_m": min(gaps) - 0.001},
            "release_gap_too_wide": {"skill": "release", "gap_m": max(gaps) + 0.01},
            "close_gap_too_wide": {"skill": "grasp", "target": "@target", "close_gap_m": max(gaps) + 0.01},
        }
        for label, step in cases.items():
            with self.subTest(case=label):
                self.assert_rejected({"steps": [step]})
        # the same targets inside the limits are accepted
        ok = parse_program({"steps": [{"skill": "move_joint", "q": {self.names[1]: float(hi[1]) - 0.01}},
                                      {"skill": "set_gripper", "gap_m": max(gaps)}]},
                           task=self.task, arm_spec=self.arm, joint_limits=self.limits)
        self.assertEqual(len(ok.steps), 2)
        # joint_limits=None: no joint-range check (the gripper range is still checked)
        parse_program({"steps": [{"skill": "move_joint", "q": {self.names[1]: float(hi[1]) + 0.05}}]},
                      task=self.task, arm_spec=self.arm)
        self.assert_rejected({"steps": [{"skill": "set_gripper", "gap_m": max(gaps) + 0.001}]}, joint_limits=None)

    def test_home_outside_limits(self):
        """§6.1 q = "home" whose home value lies outside joint_limits is unreachable -> ConfigError."""
        home = dict(zip(self.names, fx_home_q("fx_arm6")))
        j = self.arm["expert"]["arm_joints"][2]
        lo, hi = self.limits
        lo = lo.copy()
        lo[self.names.index(j)] = home[j] + 0.01
        self.assert_rejected({"steps": [{"skill": "move_joint", "q": "home"}]}, joint_limits=(lo, hi))

    def test_arm_spec_without_axes(self):
        """§5.1 the program needs [arm.kinematics].approach_axis / closing_axis -> ConfigError when missing."""
        arm = fx_arm_table("fx_arm6")
        del arm["kinematics"]["approach_axis"]
        self.assert_rejected({"steps": [{"skill": "wait", "steps": 1}]}, arm_spec=arm)


if __name__ == "__main__":
    unittest.main()
