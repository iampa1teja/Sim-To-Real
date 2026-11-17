"""WP1: GenericTask, failure rules and outcome (contracts §4.2, §4.3, §4.4, §4.6, §14.7, §14.11).

The pick_place_cube failure names are checked against the existing ``StageTrace.finish`` of
``tasks/pick_place_cube/benchmark.py`` (extracted with ``ast`` so no arm or Isaac module is imported).
"""

import ast
import itertools
import math
import unittest

import numpy as np

from core.generate.errors import ConfigError
from core.tasks.base import (CheckContext, FailureRule, Outcome, Pose, StageTracker, StepState, Task, load_task_spec)
from core.tasks.generic import PREDICATES, GenericTask, register_predicate
from core.tasks.loader import load_task
from v4_support import REPO_ROOT, clear_task_cache, copy_fixture, copy_repo_folder, sandbox, temp_env
from v4_support.factories import make_pose
from v4_support.tasks import STUB_USDA, contract_color_task_table, contract_reach_task_table, write_task

PPC_DIR = REPO_ROOT / "tasks" / "pick_place_cube"
SIX = ("never_reached", "missed_grasp", "dropped_before_box", "dropped_outside", "placed_not_confirmed",
       "timeout_holding")
BOX_FILES = {"usd/Small-Box-Physics.usda": STUB_USDA}


def _ppc_stages():
    tree = ast.parse((PPC_DIR / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "STAGES":
            return ast.literal_eval(node.value)
    raise AssertionError("STAGES not found in tasks/pick_place_cube/__init__.py")


def _stage_trace_class():
    """Exec only ``class StageTrace`` from benchmark.py with STAGES from the task package."""
    tree = ast.parse((PPC_DIR / "benchmark.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StageTrace")
    namespace = {"STAGES": _ppc_stages()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PPC_DIR / "benchmark.py"), "exec"), namespace)
    return namespace["StageTrace"]


def _trace_failure(flags, holding, success):
    trace = _stage_trace_class()()
    trace.stages = {s: bool(flags.get(s, False)) for s in _ppc_stages()}
    trace.holding = bool(holding)
    return trace.finish(success)["failure_mode"]


def _generic(box, name, table, files=BOX_FILES):
    write_task(box.tasks, name, table, files=files)
    return GenericTask(load_task_spec(name))


def _minimal(checks, stages=("a",), **extra):
    """A [task] table with no objects: stages + checks (+ extra tables)."""
    table = {"variants": {"teleop": "env_cfg:apply_teleop"}, "stages": {"order": list(stages)}, "checks": checks}
    table.update(extra)
    return table


def _empty_state(**kw):
    return StepState({}, make_pose(0.0, 0.0, 0.5), 0.0, kw.pop("signals", {}), **kw)


def _empty_ctx(goal=None):
    return CheckContext(roles={}, start={}, goal=goal or {}, objects={}, active=frozenset())


class GenericTaskBuildTests(unittest.TestCase):
    """WP1 §4.4 GenericTask construction and validation."""

    def test_contract_examples_build(self):
        """§4.6 both contract examples build; stages, rules, failure names and hold_until come from the TOML."""
        with sandbox() as box:
            color = _generic(box, "fx_wp1_g_color", contract_color_task_table())
            self.assertEqual(color.stages, ("reached", "grasped", "lifted", "over_box", "placed", "success"))
            self.assertEqual(color.hold_until, {"reached": "grasped", "grasped": "placed", "lifted": "placed",
                                                "over_box": "placed"})
            self.assertEqual(color.failure_rules[0], FailureRule("wrong_object_placed", ("distractor_in_box",), ()))
            self.assertEqual(color.failure_rules[2], FailureRule("never_reached", (), ("reached",)))
            self.assertEqual(color.failure_rules[-1], FailureRule("dropped_outside"))
            self.assertIsInstance(color.failure_rules, tuple)
            self.assertEqual(color.failure_names, ("wrong_object_placed", "placed_not_confirmed", "never_reached",
                                                   "missed_grasp", "timeout_holding", "dropped_before_box",
                                                   "dropped_outside"))
            reach = _generic(box, "fx_wp1_g_reach", contract_reach_task_table(), files={})
            self.assertEqual(reach.stages, ("reached", "success"))
            self.assertEqual(reach.hold_until, {})
            self.assertEqual(reach.failure_names, ("never_reached", "left_target"))

    def test_failure_rule_fields(self):
        """§4.2 FailureRule(name, if_=(), unless=())."""
        import dataclasses
        self.assertEqual([f.name for f in dataclasses.fields(FailureRule)], ["name", "if_", "unless"])
        self.assertEqual(FailureRule("x"), FailureRule("x", (), ()))
        self.assertEqual([f.name for f in dataclasses.fields(Outcome)], ["success", "failure_mode", "furthest_stage"])

    def test_observe_raises_type_error(self):
        """§4.4 GenericTask.observe(state) raises TypeError (a context is required)."""
        with sandbox() as box:
            task = _generic(box, "fx_wp1_g_obs", contract_reach_task_table(), files={})
            with self.assertRaises(TypeError):
                task.observe(_empty_state())

    def test_validation_errors(self):
        """§4.4 ConfigError: missing stages, stage without check, unknown predicate, bad refs, cycles, rules."""
        sig = {"check": "signal", "name": "s"}
        color = contract_color_task_table
        cases = {
            "no_stages": {"variants": {"teleop": "env_cfg:apply_teleop"}, "checks": {"a": sig}},
            "stage_without_check": _minimal({"a": sig}, stages=("a", "b")),
            "unknown_predicate": _minimal({"a": {"check": "teleport"}}),
            "unknown_of": _minimal({"a": {"check": "all", "of": ["ghost"]}}),
            "cycle": _minimal({"a": {"check": "all", "of": ["b"]}, "b": {"check": "any", "of": ["a"]}}),
            "self_cycle": _minimal({"a": {"check": "all", "of": ["a"]}}),
            "unknown_goal_key": _minimal({"a": {"check": "ee_near", "point": "@goal.ghost"}}),
            "ee_near_no_target": _minimal({"a": {"check": "ee_near", "tol_m": 0.1}}),
            "ee_near_both": {**color(), "checks": {**color()["checks"],
                                                   "reached": {"check": "ee_near", "object": "@target",
                                                               "point": [0, 0, 0]}}},
            "unknown_role": {**color(), "checks": {**color()["checks"],
                                                   "reached": {"check": "ee_near", "object": "@ghost"}}},
            "unknown_object": {**color(), "checks": {**color()["checks"],
                                                     "reached": {"check": "ee_near", "object": "ghost_cube"}}},
            "inner_not_container": {**color(), "checks": {**color()["checks"],
                                                          "in_box": {"check": "object_in_region", "object": "@target",
                                                                     "region": "@target.inner"}}},
            "unknown_set": {**color(), "checks": {**color()["checks"],
                                                  "distractor_in_box": {"check": "objects_in_region", "set": "balls",
                                                                        "region": "@container.inner"}}},
            "bad_exclude": {**color(), "checks": {**color()["checks"],
                                                  "distractor_in_box": {"check": "objects_in_region", "set": "cubes",
                                                                        "exclude": ["@nobody"],
                                                                        "region": "@container.inner"}}},
            "nested_in_missing_max_rise": {**color(), "checks": {**color()["checks"],
                                                                 "extra": {"check": "nested_in", "object": "@target",
                                                                           "base": "small_box"}}},
            "rule_unknown_flag": _minimal({"a": sig}, failures={"rules": [{"name": "x", "if": ["ghost"]}, {"name": "y"}]}),
            "hold_unknown_stage": _minimal({"a": sig, "b": sig}, stages=("a", "b")) | {
                "stages": {"order": ["a", "b"], "hold_until": {"a": "ghost"}}},
            "hold_not_later": _minimal({"a": sig, "b": sig}, stages=("a", "b")) | {
                "stages": {"order": ["a", "b"], "hold_until": {"b": "a"}}},
        }
        with sandbox() as box:
            for i, (label, table) in enumerate(cases.items()):
                with self.subTest(label):
                    write_task(box.tasks, f"fx_wp1_bad{i}", table, files=BOX_FILES)
                    with self.assertRaises(ConfigError):
                        GenericTask(load_task_spec(f"fx_wp1_bad{i}"))

    def test_rule_may_use_stage_check_and_holding(self):
        """§4.2 rule names may be stages, helper checks or the pseudo-flag "holding"."""
        sig = {"check": "signal", "name": "s"}
        table = _minimal({"a": sig, "helper": sig}, failures={"rules": [
            {"name": "x", "if": ["holding"]}, {"name": "y", "unless": ["a", "helper"]}, {"name": "z"}]})
        with sandbox() as box:
            task = _generic(box, "fx_wp1_rule_ok", table, files={})
            self.assertEqual(task.failure_names, ("x", "y", "z"))

    def test_registered_predicate_usable(self):
        """§4.4 a predicate added with register_predicate can be named by a check."""
        @register_predicate("fx_wp1_gripper_wide")
        def _wide(state, ctx, params):
            return state.gripper >= params["min_m"]
        try:
            table = _minimal({"a": {"check": "fx_wp1_gripper_wide", "min_m": 0.03}})
            with sandbox() as box:
                task = _generic(box, "fx_wp1_custom_pred", table, files={})
                self.assertEqual(task.observe_checks(StepState({}, make_pose(), 0.04), _empty_ctx()), {"a": True})
                self.assertEqual(task.observe_checks(StepState({}, make_pose(), 0.02), _empty_ctx()), {"a": False})
        finally:
            PREDICATES.pop("fx_wp1_gripper_wide", None)


class ObserveChecksTests(unittest.TestCase):
    """WP1 §4.4 observe_checks: every check, dependency order, negate."""

    def test_dependency_order_and_negate(self):
        """§4.4 all/any are evaluated after their dependencies regardless of TOML order; negate flips."""
        checks = {"top": {"check": "all", "of": ["mid", "not_b"]},
                  "mid": {"check": "any", "of": ["a", "b"]},
                  "not_b": {"check": "signal", "name": "b", "negate": True},
                  "a": {"check": "signal", "name": "a"},
                  "b": {"check": "signal", "name": "b"},
                  "never": {"check": "all", "of": ["a", "b"], "negate": True}}
        with sandbox() as box:
            task = _generic(box, "fx_wp1_order", _minimal(checks, stages=("a", "top")), files={})
            for a, b in itertools.product((False, True), repeat=2):
                with self.subTest(a=a, b=b):
                    out = task.observe_checks(_empty_state(signals={"a": a, "b": b}), _empty_ctx())
                    self.assertEqual(set(out), set(checks))
                    self.assertTrue(all(type(v) is bool for v in out.values()))
                    self.assertEqual(out["mid"], a or b)
                    self.assertEqual(out["not_b"], not b)
                    self.assertEqual(out["top"], (a or b) and not b)
                    self.assertEqual(out["never"], not (a and b))

    def test_color_task_walkthrough(self):
        """§4.4 / §4.6 the colour-cube checks over a scripted pick-and-place in a yawed box."""
        with sandbox() as box:
            task = _generic(box, "fx_wp1_walk", contract_color_task_table())
            spec = task.spec
            box_pose = make_pose(0.25, 0.0, 0.0, yaw=math.pi / 2)
            cube0 = make_pose(0.15, 0.1, 0.01)
            start = {"small_box": box_pose, "cube_0": cube0, "cube_1": make_pose(0.15, -0.1, 0.01)}
            ctx = CheckContext(roles={"target": "cube_0", "container": "small_box"}, start=start, goal={},
                               objects=spec.objects, active=frozenset(start))

            def state(cube, tcp, gap, cube1=(0.15, -0.1, 0.01)):
                return StepState({"small_box": box_pose, "cube_0": make_pose(*cube), "cube_1": make_pose(*cube1)},
                                 make_pose(tcp[0], tcp[1], tcp[2] + 0.1), gap, {}, tcp=make_pose(*tcp))
            far = task.observe_checks(state((0.15, 0.1, 0.01), (0.3, 0.3, 0.3), 0.04), ctx)
            self.assertFalse(far["reached"] or far["holding"] or far["placed"] or far["success"])
            self.assertTrue(far["released"] and far["ee_clear"])
            grasp = task.observe_checks(state((0.15, 0.1, 0.01), (0.15, 0.1, 0.012), 0.02), ctx)
            self.assertTrue(grasp["reached"] and grasp["holding"] and grasp["grasped"])
            self.assertFalse(grasp["lifted"])
            lifted = task.observe_checks(state((0.15, 0.1, 0.05), (0.15, 0.1, 0.05), 0.02), ctx)
            self.assertTrue(lifted["lifted"] and not lifted["over_box"])
            over = task.observe_checks(state((0.25, 0.01, 0.08), (0.25, 0.01, 0.08), 0.02), ctx)
            self.assertTrue(over["over_box"] and over["above_box"] and not over["in_box"])
            dropped_in = task.observe_checks(state((0.25, 0.01, 0.012), (0.25, 0.01, 0.03), 0.04), ctx)
            self.assertTrue(dropped_in["in_box"] and dropped_in["released"] and dropped_in["placed"])
            self.assertFalse(dropped_in["success"])                         # ee still within 4 cm
            done = task.observe_checks(state((0.25, 0.01, 0.012), (0.25, 0.01, 0.1), 0.04), ctx)
            self.assertTrue(done["success"])
            self.assertFalse(done["distractor_in_box"])
            wrong = task.observe_checks(state((0.15, 0.1, 0.01), (0.3, 0.3, 0.3), 0.04, cube1=(0.25, -0.01, 0.012)), ctx)
            self.assertTrue(wrong["distractor_in_box"])
            outside_x = task.observe_checks(state((0.275, 0.0, 0.012), (0.3, 0.3, 0.3), 0.04), ctx)
            self.assertFalse(outside_x["in_box"])                            # 2.5 cm from the centre > 2 cm half size


class FailureModeTests(unittest.TestCase):
    """WP1 §4.2 failure_mode default and outcome."""

    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def _fx_pick(self):
        copy_fixture("v4_wp1/tasks/fx_pick", self.box.root)
        with temp_env(S2R_TASKS_DIR=self.box.root / "v4_wp1" / "tasks"):
            clear_task_cache("fx_pick")
            task = load_task("fx_pick")
            clear_task_cache("fx_pick")
        return task

    def test_rule_semantics(self):
        """§4.2 success -> None; first rule whose if are all true and unless all false; no rules -> None."""
        sig = {"check": "signal", "name": "s"}
        table = _minimal({"a": sig, "b": sig}, stages=("a", "b"), failures={"rules": [
            {"name": "both", "if": ["a", "b"]}, {"name": "a_not_b", "if": ["a"], "unless": ["b"]},
            {"name": "held", "if": ["holding"]}, {"name": "rest"}]})
        task = _generic(self.box, "fx_wp1_rules", table, files={})
        cases = [({"a": True, "b": True}, False, False, "both"), ({"a": True}, False, False, "a_not_b"),
                 ({"a": True, "b": False}, True, False, "a_not_b"), ({}, True, False, "held"),
                 ({"holding": True}, False, False, "rest"),            # the holding argument wins over flags
                 ({"b": True}, False, False, "rest"), ({"a": True, "b": True}, True, True, None)]
        for flags, holding, success, expected in cases:
            with self.subTest(flags=flags, holding=holding, success=success):
                self.assertEqual(task.failure_mode(flags, holding, success), expected)
        no_rules = _generic(self.box, "fx_wp1_norules", _minimal({"a": sig}), files={})
        self.assertEqual(no_rules.failure_rules, ())
        self.assertEqual(no_rules.failure_names, ())
        self.assertIsNone(no_rules.failure_mode({"a": False}, False, False))

    def test_six_pick_place_names_match_stage_trace(self):
        """§4.3 / §14.7 the base rules reproduce StageTrace.finish for all 2^6 flag sets x holding."""
        task = self._fx_pick()
        self.assertEqual(set(task.failure_names), set(SIX))
        stages = _ppc_stages()
        self.assertEqual(task.stages, stages)
        for bits in itertools.product((False, True), repeat=len(stages)):
            flags = dict(zip(stages, bits))
            for holding in (False, True):
                success = flags["success"]
                with self.subTest(flags=flags, holding=holding):
                    expected = _trace_failure(flags, holding, success)
                    self.assertEqual(task.failure_mode(flags, holding, success), expected)
                    # The same through the tracker / outcome path: one frame carrying those values.
                    tracker = task.make_tracker()
                    checks = {**flags, "holding": holding}
                    tracker.update(checks)
                    outcome = task.outcome(tracker, checks, StepState({}, make_pose(), 0.0))
                    self.assertIsInstance(outcome, Outcome)
                    self.assertEqual(outcome, Outcome(success, expected, tracker.furthest_stage))

    def test_six_names_through_multi_frame_episode(self):
        """§4.2 outcome uses ever-latched stage flags and the final-frame holding value."""
        task = self._fx_pick()
        cases = [
            ([{"reached": True}, {"reached": True, "grasped": True, "holding": True},
              {"reached": True, "grasped": True, "lifted": True, "holding": True}, {}], "dropped_before_box", "lifted"),
            ([{"reached": True}, {"reached": True, "grasped": True, "holding": True},
              {"grasped": True, "holding": True}], "timeout_holding", "grasped"),
            ([{}, {}], "never_reached", None),
            ([{"reached": True}, {}], "missed_grasp", "reached"),
            ([{"reached": True, "grasped": True, "lifted": True, "over_box": True, "holding": True}, {}],
             "dropped_outside", "over_box"),
            ([{"reached": True, "grasped": True, "lifted": True, "over_box": True, "placed": True}, {}],
             "placed_not_confirmed", "placed"),
            ([{s: True for s in _ppc_stages()}], None, "success"),
        ]
        for frames, expected, furthest in cases:
            with self.subTest(expected=expected):
                tracker = task.make_tracker()
                for checks in frames:
                    tracker.update(checks)
                outcome = task.outcome(tracker, frames[-1], StepState({}, make_pose(), 0.0))
                self.assertEqual(outcome.failure_mode, expected)
                self.assertEqual(outcome.success, expected is None)
                self.assertEqual(outcome.furthest_stage, furthest)

    def test_success_requires_final(self):
        """§4.2 success = flags[last] and (if success_requires_final, default true) final_checks[last]."""
        sig = {"check": "signal", "name": "s"}
        rules = {"rules": [{"name": "failed"}]}
        frames = [{"a": True, "b": True}, {"a": True, "b": False}]
        for requires, expected in ((None, False), (True, False), (False, True)):
            stages_table = {"order": ["a", "b"]}
            if requires is not None:
                stages_table["success_requires_final"] = requires
            table = _minimal({"a": sig, "b": sig}, stages=("a", "b"), failures=rules)
            table["stages"] = stages_table
            with self.subTest(success_requires_final=requires):
                task = _generic(self.box, f"fx_wp1_final_{requires}".lower(), table, files={})
                tracker = task.make_tracker()
                for checks in frames:
                    tracker.update(checks)
                outcome = task.outcome(tracker, frames[-1], StepState({}, make_pose(), 0.0))
                self.assertEqual(outcome.success, expected)
                self.assertEqual(outcome.failure_mode, None if expected else "failed")
                self.assertEqual(outcome.furthest_stage, "b")

    def test_outcome_holding_and_helper_flags(self):
        """§4.2 holding = final_checks["holding"], else final_state.signals["holding"]; helpers use final values."""
        sig = {"check": "signal", "name": "s"}
        table = _minimal({"a": sig, "helper": sig}, failures={"rules": [
            {"name": "helper_rule", "if": ["helper"]}, {"name": "holding_rule", "if": ["holding"]}, {"name": "other"}]})
        task = _generic(self.box, "fx_wp1_outcome", table, files={})

        def run(frames, signals):
            tracker = task.make_tracker()
            for checks in frames:
                tracker.update(checks)
            return task.outcome(tracker, frames[-1], _empty_state(signals=signals)).failure_mode
        self.assertEqual(run([{"helper": True}, {"helper": False}], {}), "other")             # helper: final value
        self.assertEqual(run([{"helper": False}, {"helper": True}], {}), "helper_rule")
        self.assertEqual(run([{}], {"holding": True}), "holding_rule")                       # signal fallback
        self.assertEqual(run([{"holding": False}], {"holding": True}), "other")              # check wins
        self.assertEqual(run([{"holding": True}], {}), "holding_rule")

    def test_wrong_object_rule_precedence(self):
        """§4.6 / §14.11 contract colour task: wrong_object_placed is checked before the six names."""
        task = _generic(self.box, "fx_wp1_wrong", contract_color_task_table())
        flags = {"reached": True, "grasped": True, "lifted": True, "over_box": True, "placed": True,
                 "distractor_in_box": True}
        self.assertEqual(task.failure_mode(flags, False, False), "wrong_object_placed")
        flags["distractor_in_box"] = False
        self.assertEqual(task.failure_mode(flags, False, False), "placed_not_confirmed")

    def test_make_tracker(self):
        """§4.2 make_tracker() = StageTracker(self.stages, self.hold_until)."""
        task = _generic(self.box, "fx_wp1_tracker", contract_color_task_table())
        tracker = task.make_tracker()
        self.assertIsInstance(tracker, StageTracker)
        tracker.update({"reached": True})
        self.assertEqual(tracker.update({}).regressed_stage, "reached")
        self.assertIsNot(task.make_tracker(), tracker)


class BaseTaskDefaultsTests(unittest.TestCase):
    """WP1 §4.2 Task defaults for non-generic subclasses and PickPlaceTask regression."""

    INIT = ("from core.tasks.base import Task, load_task_spec\n\n"
            "class _T(Task):\n"
            "    stages = ('a', 'b')\n"
            "    def observe(self, state):\n        return {'a': True, 'b': state.gripper > 0.5}\n\n"
            "TASK = _T(load_task_spec(NAME))\n")

    def test_custom_subclass_defaults(self):
        """§4.2 observe_checks defaults to observe; hold_until / failure_rules come from TOML; default failure_mode."""
        table = {"variants": {"teleop": "env_cfg:apply_teleop"},
                 "stages": {"hold_until": {"a": "b"}},
                 "failures": {"rules": [{"name": "no_b", "if": ["a"], "unless": ["b"]}, {"name": "other"}]}}
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_base", table, init_py=self.INIT.replace("NAME", repr("fx_wp1_base")))
            task = load_task("fx_wp1_base")
            self.assertEqual(task.observe_checks(StepState({}, make_pose(), 1.0), None), {"a": True, "b": True})
            self.assertEqual(task.hold_until, {"a": "b"})
            self.assertEqual(task.failure_names, ("no_b", "other"))
            self.assertEqual(task.failure_mode({"a": True}, False, False), "no_b")
            self.assertEqual(task.failure_mode({"a": True, "b": True}, False, True), None)
            tracker = task.make_tracker()
            tracker.update({"a": True})
            self.assertEqual(tracker.update({}).regressed_stage, "a")
            clear_task_cache("fx_wp1_base")

    def test_defaults_without_tables(self):
        """§4.2 without [task.stages] / [task.failures]: hold_until {}, failure_rules (), failure_mode None."""
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_base2", {"variants": {"teleop": "env_cfg:apply_teleop"}},
                       init_py=self.INIT.replace("NAME", repr("fx_wp1_base2")))
            task = load_task("fx_wp1_base2")
            self.assertEqual(task.hold_until, {})
            self.assertEqual(task.failure_rules, ())
            self.assertEqual(task.failure_names, ())
            self.assertIsNone(task.failure_mode({}, False, False))
            clear_task_cache("fx_wp1_base2")

    def test_pick_place_task_failure_mode_unchanged(self):
        """§4.2 PickPlaceTask keeps its 3-argument failure_mode, equal to StageTrace.finish."""
        with sandbox() as box:
            copy_repo_folder("tasks", "pick_place_cube", box.root)
            task = load_task("pick_place_cube")
            stages = _ppc_stages()
            for bits in itertools.product((False, True), repeat=len(stages)):
                flags = dict(zip(stages, bits))
                for holding in (False, True):
                    with self.subTest(flags=flags, holding=holding):
                        self.assertEqual(task.failure_mode(flags, holding, flags["success"]),
                                         _trace_failure(flags, holding, flags["success"]))
            self.assertEqual(task.hold_until, {})
            tracker = task.make_tracker()
            self.assertEqual(tracker.update({"reached": True, "grasped": True}).stage_index, 1)
            self.assertEqual(tracker.update({}).stage_index, 1)              # no hold_until: stays latched

    def test_generic_sample_start_is_active_poses(self):
        """§4.2 Task.sample_start returns the active-object poses of sample_episode_start(rng)."""
        with sandbox() as box:
            task = _generic(box, "fx_wp1_ss", contract_color_task_table())
            sample = task.sample_episode_start(np.random.default_rng(3))
            poses = task.sample_start(np.random.default_rng(3))
            self.assertEqual(poses, {n: s.pose for n, s in sample.objects.items() if s.active})
            self.assertTrue(all(isinstance(p, Pose) for p in poses.values()))


if __name__ == "__main__":
    unittest.main()
