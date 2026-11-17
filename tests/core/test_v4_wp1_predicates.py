"""WP1: predicate registry and reference resolution (contracts §4.4).

Truth tables call ``PREDICATES[name](state, ctx, params)`` directly. ``params`` is the check table without
``check`` / ``negate`` and with defaults omitted; refs are unresolved strings (the predicate resolves them).
Boundary cases sit at least 0.5 mm (or 0.005 rad) away from each threshold.
"""

import math
import types
import unittest
from pathlib import Path

from core.generate.errors import ConfigError
from core.tasks.base import CheckContext, ObjectSpec, StepState
from core.tasks.generic import PREDICATES, register_predicate, resolve_ref
from v4_support.factories import make_pose

BUILTIN = ("ee_near", "object_lifted", "object_in_region", "objects_in_region", "object_on_object", "nested_in",
           "held", "joint_near", "gap_near", "object_displaced", "signal", "all", "any")


def _usd(name, size_outer, **params):
    return ObjectSpec(name, usd=Path(f"/tmp/{name}.usda"), params={"size_outer": list(size_outer), **params})


OBJECTS = {
    "cube": ObjectSpec("cube", size=(0.02, 0.02, 0.02)),
    "big": ObjectSpec("big", size=(0.04, 0.04, 0.04)),
    "box": _usd("box", (0.08, 0.04, 0.035), origin="bottom", inner_half_size=[0.03, 0.01], floor_z=0.002),
    "bowl_a": _usd("bowl_a", (0.10, 0.10, 0.04), origin="bottom"),
    "bowl_b": _usd("bowl_b", (0.12, 0.12, 0.05), origin="bottom"),
    "c_0": ObjectSpec("c_0", size=(0.02, 0.02, 0.02), params={"set": "cs", "slot": 0}),
    "c_1": ObjectSpec("c_1", size=(0.02, 0.02, 0.02), params={"set": "cs", "slot": 1}),
    "c_2": ObjectSpec("c_2", size=(0.02, 0.02, 0.02), params={"set": "cs", "slot": 2}),
}
START = {"cube": make_pose(0.2, 0.0, 0.01), "big": make_pose(0.25, 0.05, 0.02), "box": make_pose(0.25, -0.05, 0.0),
         "bowl_a": make_pose(0.15, 0.1, 0.0), "bowl_b": make_pose(0.2, 0.1, 0.0), "c_0": make_pose(0.1, 0.0, 0.01),
         "c_1": make_pose(0.1, 0.05, 0.01), "c_2": make_pose(0.1, -2.0, -2.0)}
GOAL = {"point": (0.2, 0.0, 0.1), "zone": types.MappingProxyType({"xy_min": (0.1, -0.1), "xy_max": (0.2, 0.1)}),
        "joint": types.MappingProxyType({"joint": "j2", "angles": (0.3, 0.6)}), "gap": 0.02,
        "path": ((0.1, 0.0, 0.1), (0.2, 0.1, 0.15))}


def _ctx(roles=None, active=None):
    return CheckContext(roles=roles if roles is not None else {"target": "cube", "container": "box", "pick": "c_0"},
                        start=START, goal=GOAL, objects=OBJECTS,
                        active=frozenset(active if active is not None else set(OBJECTS) - {"c_2"}))


def _state(objects=None, *, tcp=(0.5, 0.5, 0.5), ee=(0.9, 0.9, 0.9), gripper=0.03, signals=None, joints=None):
    poses = dict(START)
    poses.pop("c_2")
    for name, value in (objects or {}).items():
        if value is None:
            poses.pop(name, None)
        else:
            poses[name] = make_pose(*value) if isinstance(value, tuple) else value
    return StepState(poses, make_pose(*ee), gripper, dict(signals or {}),
                     tcp=None if tcp is None else make_pose(*tcp), joints=joints)


def check(name, state, params, ctx=None):
    return PREDICATES[name](state, ctx or _ctx(), params)


class ResolveRefTests(unittest.TestCase):
    """WP1 §4.4 resolve_ref syntax."""

    def test_valid_refs(self):
        """§4.4 @role, @role.inner, @goal.key, @goal.key.i, literal object name, literal list."""
        ctx = _ctx()
        self.assertEqual(resolve_ref("@target", ctx), "cube")
        self.assertEqual(resolve_ref("@pick", ctx), "c_0")
        inner = resolve_ref("@container.inner", ctx)
        self.assertEqual(dict(inner), {"inner": "box"})
        self.assertEqual(tuple(resolve_ref("@goal.point", ctx)), (0.2, 0.0, 0.1))
        self.assertEqual(resolve_ref("@goal.point.2", ctx), 0.1)
        self.assertEqual(resolve_ref("@goal.gap", ctx), 0.02)
        self.assertEqual(tuple(resolve_ref("@goal.path.1", ctx)), (0.2, 0.1, 0.15))
        self.assertEqual(resolve_ref("@goal.joint.joint", ctx), "j2")
        self.assertEqual(resolve_ref("@goal.joint.angles.1", ctx), 0.6)
        self.assertEqual(resolve_ref("big", ctx), "big")
        self.assertEqual(resolve_ref([0.1, 0.2, 0.3], ctx), (0.1, 0.2, 0.3))

    def test_invalid_refs(self):
        """§1.5 an unresolvable reference is a ConfigError."""
        ctx = _ctx()
        for ref in ("@ghost", "ghost", "@goal.nowhere", "@target.inner", "@goal.point.7", "@target.lid"):
            with self.subTest(ref=ref):
                with self.assertRaises(ConfigError):
                    resolve_ref(ref, ctx)


class RegistryTests(unittest.TestCase):
    """WP1 §4.4 PREDICATES and register_predicate."""

    def test_builtin_names(self):
        """§4.4 every predicate of the table (nested_in included, §14.5) is registered."""
        self.assertIsInstance(PREDICATES, dict)
        for name in BUILTIN:
            with self.subTest(name):
                self.assertIn(name, PREDICATES)
                self.assertTrue(callable(PREDICATES[name]))

    def test_register_predicate(self):
        """§4.4 register_predicate(name) is a decorator that adds the function under name and returns it."""
        def always(state, ctx, params):
            return params.get("value", True)
        try:
            returned = register_predicate("fx_wp1_always")(always)
            self.assertIs(returned, always)
            self.assertIs(PREDICATES["fx_wp1_always"], always)
        finally:
            PREDICATES.pop("fx_wp1_always", None)


class PredicateTruthTableTests(unittest.TestCase):
    """WP1 §4.4 predicate truth tables (positions in metres, base frame)."""

    def table(self, name, cases):
        for label, state, params, expected in cases:
            with self.subTest(predicate=name, case=label):
                result = check(name, state, params)
                self.assertIsInstance(result, bool)
                self.assertEqual(result, expected)

    def test_ee_near(self):
        """§4.4 ee_near: ||tcp - (target + offset)|| <= tol_m (default 0.02); tcp = state.tcp or state.ee."""
        self.table("ee_near", [
            ("object_inside", _state(tcp=(0.2, 0.0, 0.029)), {"object": "@target"}, True),
            ("object_outside", _state(tcp=(0.2, 0.0, 0.031)), {"object": "@target"}, False),
            ("diagonal_outside", _state(tcp=(0.215, 0.0145, 0.01)), {"object": "cube"}, False),
            ("offset", _state(tcp=(0.2, 0.0, 0.07)), {"object": "@target", "offset_m": [0, 0, 0.06]}, True),
            ("offset_miss", _state(tcp=(0.2, 0.0, 0.01)), {"object": "@target", "offset_m": [0, 0, 0.06]}, False),
            ("tol", _state(tcp=(0.2, 0.0, 0.05)), {"object": "@target", "tol_m": 0.045}, True),
            ("point_goal", _state(tcp=(0.2, 0.0, 0.115)), {"point": "@goal.point"}, True),
            ("point_goal_tol", _state(tcp=(0.2, 0.0, 0.115)), {"point": "@goal.point", "tol_m": 0.01}, False),
            ("point_literal", _state(tcp=(0.3, 0.1, 0.2)), {"point": [0.3, 0.1, 0.21]}, True),
            ("ee_fallback", _state(tcp=None, ee=(0.2, 0.0, 0.02)), {"object": "@target"}, True),
            ("tcp_wins_over_ee", _state(tcp=(0.5, 0.5, 0.5), ee=(0.2, 0.0, 0.01)), {"object": "@target"}, False),
        ])

    def test_object_lifted(self):
        """§4.4 object_lifted: p(o).z - start(o).z >= min_lift_m (default 0.01)."""
        self.table("object_lifted", [
            ("not_lifted", _state({"cube": (0.2, 0.0, 0.0195)}), {"object": "@target"}, False),
            ("lifted", _state({"cube": (0.25, 0.1, 0.0205)}), {"object": "@target"}, True),
            ("custom_min", _state({"cube": (0.2, 0.0, 0.0205)}), {"object": "cube", "min_lift_m": 0.02}, False),
            ("custom_min_ok", _state({"cube": (0.2, 0.0, 0.0305)}), {"object": "cube", "min_lift_m": 0.02}, True),
        ])

    def test_object_in_region(self):
        """§4.4 object_in_region: rect, polar, goal rect and container inner rect (yaw frame, margin, check_z)."""
        rect = {"xy_min": [0.1, -0.1], "xy_max": [0.3, 0.1]}
        polar = {"r_m": [0.15, 0.25], "bearing_deg": [-30.0, 30.0]}
        box_yaw = make_pose(0.25, -0.05, 0.0, yaw=math.pi / 2)        # inner x half 0.03 now along base y
        floor = 0.002 + 0.01                                            # resting cube centre z in the box
        self.table("object_in_region", [
            ("rect_in", _state({"cube": (0.2, 0.05, 0.01)}), {"object": "@target", "region": rect}, True),
            ("rect_out", _state({"cube": (0.31, 0.0, 0.01)}), {"object": "@target", "region": rect}, False),
            ("rect_margin", _state({"cube": (0.29, 0.0, 0.01)}), {"object": "@target", "region": rect, "margin_m": 0.015}, False),
            ("rect_ignores_z", _state({"cube": (0.2, 0.0, 0.3)}), {"object": "@target", "region": rect}, True),
            ("polar_in", _state({"cube": (0.2, 0.05, 0.01)}), {"object": "@target", "region": polar}, True),
            ("polar_bearing_out", _state({"cube": (0.1, 0.15, 0.01)}), {"object": "@target", "region": polar}, False),
            ("polar_range_out", _state({"cube": (0.26, 0.0, 0.01)}), {"object": "@target", "region": polar}, False),
            ("goal_rect", _state({"cube": (0.15, 0.05, 0.01)}), {"object": "@target", "region": "@goal.zone"}, True),
            ("goal_rect_out", _state({"cube": (0.25, 0.05, 0.01)}), {"object": "@target", "region": "@goal.zone"}, False),
            ("inner_in", _state({"cube": (0.27, -0.045, floor)}), {"object": "@target", "region": "@container.inner"}, True),
            ("inner_out_y", _state({"cube": (0.25, -0.0385, floor)}), {"object": "@target", "region": "@container.inner"}, False),
            ("inner_yawed_in", _state({"cube": (0.25, -0.025, floor), "box": box_yaw}),
             {"object": "@target", "region": "@container.inner"}, True),
            ("inner_yawed_out", _state({"cube": (0.275, -0.05, floor), "box": box_yaw}),
             {"object": "@target", "region": "@container.inner"}, False),
            ("inner_margin", _state({"cube": (0.275, -0.05, floor)}),
             {"object": "@target", "region": "@container.inner", "margin_m": 0.01}, False),
            ("inner_above_check_z", _state({"cube": (0.25, -0.05, floor + 0.0105)}),
             {"object": "@target", "region": "@container.inner"}, False),
            ("inner_slightly_above", _state({"cube": (0.25, -0.05, floor + 0.0095)}),
             {"object": "@target", "region": "@container.inner"}, True),
            ("inner_above_no_z", _state({"cube": (0.25, -0.05, floor + 0.05)}),
             {"object": "@target", "region": "@container.inner", "check_z": False}, True),
            ("inner_z_tol", _state({"cube": (0.25, -0.05, floor + 0.015)}),
             {"object": "@target", "region": "@container.inner", "z_tol_m": 0.02}, True),
            ("inner_box_raised", _state({"cube": (0.25, -0.05, floor + 0.05), "box": (0.25, -0.05, 0.05)}),
             {"object": "@target", "region": "@container.inner"}, True),
        ])

    def test_objects_in_region(self):
        """§4.4 objects_in_region counts active members of set, not in exclude, satisfying object_in_region."""
        inner = {"set": "cs", "region": "@container.inner"}
        excl = {"set": "cs", "region": "@container.inner", "exclude": ["@pick"]}
        in_box = (0.25, -0.05, 0.012)
        self.table("objects_in_region", [
            ("none", _state(), inner, False),
            ("one", _state({"c_1": in_box}), inner, True),
            ("excluded_only", _state({"c_0": in_box}), excl, False),
            ("other_member", _state({"c_1": in_box}), excl, True),
            ("min_count_2_one", _state({"c_1": in_box}), {**inner, "min_count": 2}, False),
            ("min_count_2_two", _state({"c_0": in_box, "c_1": (0.26, -0.05, 0.012)}), {**inner, "min_count": 2}, True),
            ("inactive_ignored", _state({"c_2": in_box}), inner, False),
            ("non_member_ignored", _state({"cube": in_box}), inner, False),
            ("check_z", _state({"c_1": (0.25, -0.05, 0.05)}), inner, False),
            ("no_check_z", _state({"c_1": (0.25, -0.05, 0.05)}), {**inner, "check_z": False}, True),
        ])

    def test_object_on_object(self):
        """§4.4 object_on_object: xy within xy_tol_m and z = z(b) + top(b) + h(o)/2 within z_tol_m."""
        on_big = 0.02 + 0.02 + 0.01          # centre-origin base: z(b) + h(b)/2 + h(o)/2
        on_box = 0.0 + 0.035 + 0.01          # bottom-origin base: z(b) + h(b) + h(o)/2
        self.table("object_on_object", [
            ("stacked", _state({"cube": (0.25, 0.05, on_big)}), {"object": "@target", "base": "big"}, True),
            ("z_high", _state({"cube": (0.25, 0.05, on_big + 0.0055)}), {"object": "@target", "base": "big"}, False),
            ("z_low", _state({"cube": (0.25, 0.05, on_big - 0.0045)}), {"object": "@target", "base": "big"}, True),
            ("xy_off", _state({"cube": (0.25, 0.0605, on_big)}), {"object": "@target", "base": "big"}, False),
            ("xy_near", _state({"cube": (0.2565, 0.0565, on_big)}), {"object": "@target", "base": "big"}, True),
            ("xy_tol", _state({"cube": (0.25, 0.065, on_big)}), {"object": "cube", "base": "big", "xy_tol_m": 0.02}, True),
            ("bottom_origin_base", _state({"cube": (0.25, -0.05, on_box)}), {"object": "cube", "base": "box"}, True),
            ("bottom_origin_wrong_formula", _state({"cube": (0.25, -0.05, 0.035 / 2 + 0.01)}),
             {"object": "cube", "base": "box"}, False),
        ])

    def test_nested_in(self):
        """§4.4 nested_in: ||xy(o) - xy(b)|| <= xy_tol_m and 0 <= z(o) - z(b) <= max_rise_m."""
        p = {"object": "bowl_a", "base": "bowl_b", "max_rise_m": 0.02}
        self.table("nested_in", [
            ("nested", _state({"bowl_a": (0.2, 0.1, 0.015)}), p, True),
            ("flush", _state({"bowl_a": (0.2, 0.1, 0.0)}), p, True),
            ("too_high", _state({"bowl_a": (0.2, 0.1, 0.0205)}), p, False),
            ("below", _state({"bowl_a": (0.2, 0.1, -0.001)}), p, False),
            ("xy_off", _state({"bowl_a": (0.2105, 0.1, 0.01)}), p, False),
            ("xy_tol", _state({"bowl_a": (0.2105, 0.1, 0.01)}), {**p, "xy_tol_m": 0.02}, True),
            ("base_moved", _state({"bowl_a": (0.3, 0.0, 0.01), "bowl_b": (0.3, 0.0, 0.0)}), p, True),
        ])

    def test_held(self):
        """§4.4 held: w(o) - gap_low_m <= gripper <= w(o) + gap_high_m and ||tcp - p(o)|| <= max_dist_m."""
        at_cube = (0.2, 0.0, 0.03)          # 0.02 from the cube centre
        self.table("held", [
            ("closed_on_cube", _state(tcp=at_cube, gripper=0.02), {"object": "@target"}, True),
            ("gap_low_edge", _state(tcp=at_cube, gripper=0.0145), {"object": "@target"}, True),
            ("gap_too_small", _state(tcp=at_cube, gripper=0.0135), {"object": "@target"}, False),
            ("gap_high_edge", _state(tcp=at_cube, gripper=0.0235), {"object": "@target"}, True),
            ("gap_open", _state(tcp=at_cube, gripper=0.0245), {"object": "@target"}, False),
            ("too_far", _state(tcp=(0.2, 0.0, 0.0405), gripper=0.02), {"object": "@target"}, False),
            ("max_dist", _state(tcp=(0.2, 0.0, 0.0405), gripper=0.02), {"object": "@target", "max_dist_m": 0.035}, True),
            ("w_is_min_horizontal", _state({"box": (0.2, 0.0, 0.0)}, tcp=(0.2, 0.0, 0.0), gripper=0.04),
             {"object": "box"}, True),
            ("custom_gaps", _state(tcp=at_cube, gripper=0.012), {"object": "cube", "gap_low_m": 0.01}, True),
        ])

    def test_joint_near(self):
        """§4.4 joint_near: |state.joints[joint] - value_rad| <= tol_rad (default 0.02); joints None -> False."""
        joints = {"j1": 0.5, "j2": 0.61}
        self.table("joint_near", [
            ("near", _state(joints=joints), {"joint": "j1", "value_rad": 0.515}, True),
            ("far", _state(joints=joints), {"joint": "j1", "value_rad": 0.525}, False),
            ("tol", _state(joints=joints), {"joint": "j1", "value_rad": 0.525, "tol_rad": 0.03}, True),
            ("goal_joint", _state(joints=joints), {"joint": "@goal.joint", "value_rad": "@goal.joint.angles.1"}, True),
            ("goal_angle_0", _state(joints=joints), {"joint": "@goal.joint", "value_rad": "@goal.joint.angles.0"}, False),
            ("no_joints", _state(joints=None), {"joint": "j1", "value_rad": 0.5}, False),
            ("unknown_joint", _state(joints=joints), {"joint": "j9", "value_rad": 0.5}, False),
        ])

    def test_gap_near(self):
        """§4.4 gap_near: |state.gripper - value_m| <= tol_m (default 0.003)."""
        self.table("gap_near", [
            ("near", _state(gripper=0.0225), {"value_m": 0.02}, True),
            ("far", _state(gripper=0.0235), {"value_m": 0.02}, False),
            ("tol", _state(gripper=0.0235), {"value_m": 0.02, "tol_m": 0.004}, True),
            ("goal", _state(gripper=0.019), {"value_m": "@goal.gap"}, True),
        ])

    def test_object_displaced(self):
        """§4.4 object_displaced: min_m <= ||Δp(o)[axes]|| <= max_m (default inf), Δ from start, axes default xy."""
        moved = _state({"cube": (0.23, 0.0, 0.05)})          # Δ = (0.03, 0, 0.04)
        self.table("object_displaced", [
            ("xy", moved, {"object": "@target", "min_m": 0.02}, True),
            ("xy_too_little", moved, {"object": "@target", "min_m": 0.035}, False),
            ("max", moved, {"object": "@target", "min_m": 0.02, "max_m": 0.025}, False),
            ("xyz", moved, {"object": "@target", "min_m": 0.045, "axes": "xyz"}, True),
            ("z", moved, {"object": "@target", "min_m": 0.035, "max_m": 0.045, "axes": "z"}, True),
            ("still", _state(), {"object": "@target", "min_m": 0.001}, False),
        ])

    def test_signal_all_any(self):
        """§4.4 signal reads state.signals (missing = false); all / any over already-evaluated values."""
        self.table("signal", [
            ("on", _state(signals={"holding": True}), {"name": "holding"}, True),
            ("off", _state(signals={"holding": False}), {"name": "holding"}, False),
            ("missing", _state(), {"name": "holding"}, False),
        ])
        self.table("all", [("tt", _state(), {"of": [True, True]}, True), ("tf", _state(), {"of": [True, False]}, False)])
        self.table("any", [("ff", _state(), {"of": [False, False]}, False), ("ft", _state(), {"of": [False, True]}, True)])

    def test_missing_object_is_false(self):
        """§4.4 decision: a predicate on an object absent from state.objects (inactive) is false."""
        state = _state({"cube": None})
        self.assertFalse(check("ee_near", state, {"object": "@target", "tol_m": 10.0}))
        self.assertFalse(check("object_in_region", state, {"object": "@target", "region": {"xy_min": [-9, -9], "xy_max": [9, 9]}}))
        self.assertFalse(check("held", state, {"object": "@target", "max_dist_m": 10.0}))


class CheckContextTests(unittest.TestCase):
    """WP1 §4.2 CheckContext."""

    def test_fields_and_from_spec(self):
        """§4.2 CheckContext.from_spec: roles/goal from the spec, start = spec.objects poses, active set."""
        import dataclasses
        from v4_support.factories import make_episode_spec
        self.assertEqual([f.name for f in dataclasses.fields(CheckContext)], ["roles", "start", "goal", "objects", "active"])
        spec = make_episode_spec(goal={"point": [0.1, 0.2, 0.3]})

        class _TS:                      # only .objects is read
            objects = {"cube_0": OBJECTS["cube"], "cube_1": OBJECTS["cube"], "box": OBJECTS["box"]}
        ctx = CheckContext.from_spec(spec, _TS())
        self.assertEqual(dict(ctx.roles), {"target": "cube_0", "container": "box"})
        self.assertEqual(ctx.start["cube_0"], spec.objects["cube_0"].pose)
        self.assertEqual(set(ctx.start), {"cube_0", "cube_1", "box"})
        self.assertEqual(tuple(ctx.goal["point"]), (0.1, 0.2, 0.3))
        self.assertEqual(ctx.active, frozenset({"cube_0", "box"}))
        self.assertIsInstance(ctx.active, frozenset)
        self.assertEqual(set(ctx.objects), {"cube_0", "cube_1", "box"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.roles = {}


if __name__ == "__main__":
    unittest.main()
