"""WP1: TaskSpec / StepState additions and task.toml loading (contracts §4.1, §4.2, §4.5 rules, §4.6, §13).

Covers the new ``TaskSpec`` fields, object-set expansion, palette, roles, hold-outs, instruction templates,
load-time validation (``ConfigError``), and regression tests that existing task folders load unchanged.
"""

import dataclasses
import unittest
from pathlib import Path

import numpy as np

from core.generate.errors import ConfigError
from core.tasks.base import (HoldoutSpec, ObjectSetSpec, ObjectSpec, Pose, StepState, Task, TaskSpec, load_task_spec)
from core.tasks.loader import load_task
from v4_support import clear_task_cache, copy_fixture, copy_repo_folder, sandbox, temp_env
from v4_support.tasks import (STUB_USDA, contract_color_task_table, contract_reach_task_table, write_scene,
                              write_task)

OLD_FIELDS = ["name", "task_dir", "instruction", "variants", "default_variant", "objects", "raw"]
NEW_FIELDS = ["object_sets", "palette", "roles", "holdout", "instruction_templates"]
VARIANTS = {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"}
BOX_FILES = {"usd/Small-Box-Physics.usda": STUB_USDA}


def _color_table(**changes):
    """The §4.6 colour-cube example with top-level ``[task]`` keys replaced (``None`` deletes a key)."""
    table = contract_color_task_table()
    for key, value in changes.items():
        if value is None:
            table.pop(key, None)
        else:
            table[key] = value
    return table


def _load(box, name, table, files=BOX_FILES):
    write_task(box.tasks, name, table, files=files)
    return load_task_spec(name)


class TaskSpecFieldTests(unittest.TestCase):
    """WP1 §4.1 TaskSpec additions and §4.2 StepState additions."""

    def test_field_order(self):
        """§4.1 new fields are appended after the existing ones (positional construction unchanged)."""
        self.assertEqual([f.name for f in dataclasses.fields(TaskSpec)], OLD_FIELDS + NEW_FIELDS)

    def test_positional_construction_defaults(self):
        """§4.1 all new fields are defaulted: {} / {} / {} / empty HoldoutSpec / ()."""
        spec = TaskSpec("t", Path("."), "do it", {"teleop": "env_cfg:apply_teleop"}, "teleop", {}, {})
        self.assertEqual(dict(spec.object_sets), {})
        self.assertEqual(len(spec.palette), 0)
        self.assertEqual(dict(spec.palette), {})
        self.assertEqual(dict(spec.roles), {})
        self.assertEqual(spec.holdout, HoldoutSpec())
        self.assertEqual(spec.instruction_templates, ())
        self.assertEqual(spec.entry_point("teleop"), "s2r_tasks.t.env_cfg:apply_teleop")

    def test_holdout_spec(self):
        """§4.1 HoldoutSpec(colors=(), counts=())."""
        self.assertEqual([f.name for f in dataclasses.fields(HoldoutSpec)], ["colors", "counts"])
        empty = HoldoutSpec()
        self.assertEqual((empty.colors, empty.counts), ((), ()))
        self.assertEqual(HoldoutSpec(("purple",), (4,)).counts, (4,))

    def test_object_set_spec(self):
        """§4.1 ObjectSetSpec(name, prefix, count_max, template); slot_names = prefix_i for i < count_max."""
        self.assertEqual([f.name for f in dataclasses.fields(ObjectSetSpec)], ["name", "prefix", "count_max", "template"])
        template = ObjectSpec("cubes", size=(0.02, 0.02, 0.02))
        oset = ObjectSetSpec("cubes", "cube", 3, template)
        self.assertEqual(oset.slot_names(), ("cube_0", "cube_1", "cube_2"))
        self.assertEqual(ObjectSetSpec("s", "s", 1, template).slot_names(), ("s_0",))

    def test_step_state_fields(self):
        """§4.2 StepState gains tcp (Pose | None = None) and joints (Mapping | None = None) after signals."""
        self.assertEqual([f.name for f in dataclasses.fields(StepState)],
                         ["objects", "ee", "gripper", "signals", "tcp", "joints"])
        ee = Pose((0.1, 0.0, 0.2))
        old = StepState({}, ee, 0.01)
        self.assertEqual(old.signals, {})
        self.assertIsNone(old.tcp)
        self.assertIsNone(old.joints)
        new = StepState({}, ee, 0.01, {"holding": True}, tcp=Pose((0.1, 0.0, 0.1)), joints={"j1": 0.5})
        self.assertEqual(new.tcp.position, (0.1, 0.0, 0.1))
        self.assertEqual(new.joints["j1"], 0.5)

    def test_expert_obs_step_state_fills_tcp_and_joints(self):
        """§3.2 / §4.4 note: ExpertObs.step_state(joint_names) fills tcp always and joints only with names."""
        from v4_support.factories import make_expert_obs
        obs = make_expert_obs(q=np.arange(6) * 0.1)
        state = obs.step_state()
        self.assertEqual(state.tcp, obs.tcp)
        self.assertIsNone(state.joints)
        names = ("a", "b", "c", "d", "e", "f")
        state = obs.step_state(joint_names=names)
        self.assertEqual(dict(state.joints), {n: 0.1 * i for i, n in enumerate(names)})


class LegacyLoadTests(unittest.TestCase):
    """WP1 regression: task folders without the new tables load exactly as before (§4.1, §4.2)."""

    def test_table_without_new_tables(self):
        """§4.1 a task.toml with only the old tables gives the same TaskSpec as before plus empty new fields."""
        table = {"instruction": "Do the thing", "default_variant": "teleop", "variants": VARIANTS,
                 "objects": {"box": {"usd": "usd/b.usda", "size_outer": [0.1, 0.1, 0.05], "rigid": True},
                             "cube": {"size": [0.02, 0.02, 0.02], "mass": 0.008, "position": [0.0, 0.0, 0.05]}},
                 "success": {"min_lift": 0.01, "confirm_steps": 25}, "episode": {"decimation": 4},
                 "start": {"xy_min": [0.1, -0.1], "xy_max": [0.2, 0.1], "yaw_deg": [0.0, 90.0]}}
        with sandbox() as box:
            spec = _load(box, "fx_wp1_legacy", table, files={"usd/b.usda": STUB_USDA})
            self.assertEqual(spec.name, "fx_wp1_legacy")
            self.assertEqual(spec.instruction, "Do the thing")
            self.assertEqual(spec.variants, VARIANTS)
            self.assertEqual(spec.default_variant, "teleop")
            self.assertEqual(list(spec.objects), ["box", "cube"])
            self.assertEqual(spec.objects["cube"].size, (0.02, 0.02, 0.02))
            self.assertEqual(spec.objects["cube"].params, {"mass": 0.008})
            self.assertEqual(spec.objects["box"].params, {"size_outer": [0.1, 0.1, 0.05]})
            self.assertEqual(spec.raw["start"]["xy_min"], [0.1, -0.1])
            self.assertEqual(spec.raw["success"], {"min_lift": 0.01, "confirm_steps": 25})
            self.assertEqual(dict(spec.object_sets), {})
            self.assertEqual(dict(spec.palette), {})
            self.assertEqual(dict(spec.roles), {})
            self.assertEqual(spec.holdout, HoldoutSpec())
            self.assertEqual(spec.instruction_templates, ())

    def test_default_instruction_unchanged(self):
        """§4.1 instruction still defaults to the name with spaces when [task] has no instruction."""
        with sandbox() as box:
            spec = _load(box, "fx_wp1_noinstr", {"variants": VARIANTS}, files={})
            self.assertEqual(spec.instruction, "fx wp1 noinstr")

    def test_old_errors_still_raised(self):
        """Existing ValueError / FileNotFoundError behaviour of load_task_spec is kept."""
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_novar", {"objects": {}})
            with self.assertRaises(ValueError):
                load_task_spec("fx_wp1_novar")
            write_task(box.tasks, "fx_wp1_nousd", {"variants": VARIANTS, "objects": {"o": {"usd": "missing.usda"}}})
            with self.assertRaises(FileNotFoundError):
                load_task_spec("fx_wp1_nousd")
            with self.assertRaises(FileNotFoundError):
                load_task_spec("fx_wp1_does_not_exist")

    def test_pick_place_cube_loads_unchanged(self):
        """§4.2 the existing tasks/pick_place_cube still loads unchanged (PickPlaceTask, same objects)."""
        with sandbox() as box:
            copy_repo_folder("tasks", "pick_place_cube", box.root)
            spec = load_task_spec("pick_place_cube")
            self.assertEqual(list(spec.objects), ["white_box", "blue_cube"])
            self.assertEqual(spec.instruction, "Pick up the cube and place it in the box")
            # WP10 adds the "generate" variant; the existing ones stay.
            self.assertTrue({"teleop", "eval", "eval_dr"} <= set(spec.variants) <= {"teleop", "eval", "eval_dr", "generate"})
            self.assertEqual(spec.objects["white_box"].params["inner_half_size"], [0.065625, 0.04064896])
            self.assertEqual(dict(spec.object_sets), {})
            self.assertEqual(dict(spec.palette), {})
            self.assertEqual(spec.instruction_templates, ())
            task = load_task("pick_place_cube")
            self.assertIsInstance(task, Task)
            self.assertEqual(type(task).__name__, "PickPlaceTask")
            self.assertEqual(task.stages, ("reached", "grasped", "lifted", "over_box", "placed", "success"))
            # observe() still works with the extended StepState (holding False avoids the eval.py import).
            state = StepState({"blue_cube": Pose((0.2, 0.0, 0.01)), "white_box": Pose((0.25, 0.1, 0.0))},
                              Pose((0.2, 0.0, 0.02)), 0.03, {"grasped": False}, tcp=Pose((0.2, 0.0, 0.0)))
            flags = task.observe(state)
            self.assertEqual(flags, {"reached": True, "grasped": False, "lifted": False, "over_box": False,
                                     "placed": False, "success": False})
            self.assertEqual(task.observe_checks(state, None), flags)
            # Before WP10 PickPlaceTask.sample_start reads the old top-level [task.start] keys (ValueError without
            # them); after WP10 the base sampler places both objects from [task.start].
            if "place" in (spec.raw.get("start") or {}):
                poses = task.sample_start(np.random.default_rng(0))
                self.assertEqual(set(poses), {"white_box", "blue_cube"})
            else:
                with self.assertRaises(ValueError):
                    task.sample_start(np.random.default_rng(0))

    def test_custom_task_subclass_still_loads(self):
        """§4.2 stages/observe stay abstract; a subclass overriding sample_start loads as before."""
        init = ("from core.tasks.base import Task, load_task_spec\n\n"
                "class _T(Task):\n"
                "    stages = ('a', 'b')\n"
                "    def sample_start(self, rng):\n        return {}\n"
                "    def observe(self, state):\n        return {'a': True, 'b': False}\n\n"
                "TASK = _T(load_task_spec('fx_wp1_custom'))\n")
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_custom", {"variants": VARIANTS}, init_py=init)
            task = load_task("fx_wp1_custom")
            self.assertEqual(task.stages, ("a", "b"))
            self.assertEqual(task.sample_start(None), {})
            clear_task_cache("fx_wp1_custom")

    def test_abstract_methods(self):
        """§4.2 stages and observe stay abstract; sample_start is concrete."""
        self.assertEqual(set(Task.__abstractmethods__), {"stages", "observe"})


class ObjectSetTests(unittest.TestCase):
    """WP1 §4.1 object-set expansion."""

    def test_expansion(self):
        """§4.1 every slot is added to spec.objects after the plain objects, with params set/slot."""
        with sandbox() as box:
            spec = _load(box, "fx_wp1_sets", contract_color_task_table())
            self.assertEqual(list(spec.objects), ["small_box"] + [f"cube_{i}" for i in range(6)])
            oset = spec.object_sets["cubes"]
            self.assertIsInstance(oset, ObjectSetSpec)
            self.assertEqual((oset.name, oset.prefix, oset.count_max), ("cubes", "cube", 6))
            self.assertEqual(oset.template.size, (0.02, 0.02, 0.02))
            self.assertEqual(oset.template.params.get("mass"), 0.008)
            for i in range(6):
                slot = spec.objects[f"cube_{i}"]
                self.assertIsInstance(slot, ObjectSpec)
                self.assertEqual(slot.name, f"cube_{i}")
                self.assertEqual(slot.size, (0.02, 0.02, 0.02))
                self.assertEqual(slot.params["set"], "cubes")
                self.assertEqual(slot.params["slot"], i)
                self.assertEqual(slot.params["mass"], 0.008)
            self.assertNotIn("set", oset.template.params)

    def test_prefix_defaults_to_set_name_and_order(self):
        """§4.6 prefix defaults to the set name; sets expand in TOML order after all plain objects."""
        table = {"variants": VARIANTS,
                 "objects": {"a": {"size": [0.02, 0.02, 0.02]}},
                 "object_sets": {"pegs": {"count_max": 2, "size": [0.01, 0.01, 0.05]},
                                 "blocks": {"prefix": "blk", "count_max": 1, "size": [0.03, 0.03, 0.03]}}}
        with sandbox() as box:
            spec = _load(box, "fx_wp1_prefix", table, files={})
            self.assertEqual(list(spec.objects), ["a", "pegs_0", "pegs_1", "blk_0"])
            self.assertEqual(list(spec.object_sets), ["pegs", "blocks"])
            self.assertEqual(spec.object_sets["pegs"].prefix, "pegs")

    def test_name_clashes(self):
        """§4.1 name clashes (slot vs object, set vs object, slot vs slot) raise ConfigError."""
        cases = {
            "slot_vs_object": {"objects": {"cube_0": {"size": [0.02, 0.02, 0.02]}},
                               "object_sets": {"cubes": {"prefix": "cube", "count_max": 2, "size": [0.02] * 3}}},
            "set_vs_object": {"objects": {"cubes": {"size": [0.02, 0.02, 0.02]}},
                              "object_sets": {"cubes": {"prefix": "cube", "count_max": 2, "size": [0.02] * 3}}},
            "slot_vs_slot": {"object_sets": {"a": {"prefix": "x", "count_max": 2, "size": [0.02] * 3},
                                             "b": {"prefix": "x", "count_max": 1, "size": [0.02] * 3}}},
        }
        with sandbox() as box:
            for i, (label, extra) in enumerate(cases.items()):
                with self.subTest(label):
                    write_task(box.tasks, f"fx_wp1_clash{i}", {"variants": VARIANTS, **extra})
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_clash{i}")

    def test_bad_count_max(self):
        """§4.6 count_max is an int >= 1 (ConfigError otherwise)."""
        with sandbox() as box:
            for i, bad in enumerate([0, -1, 1.5, "2"]):
                with self.subTest(count_max=bad):
                    write_task(box.tasks, f"fx_wp1_cmax{i}",
                               {"variants": VARIANTS, "object_sets": {"s": {"count_max": bad, "size": [0.02] * 3}}})
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_cmax{i}")
            write_task(box.tasks, "fx_wp1_cmaxmiss", {"variants": VARIANTS, "object_sets": {"s": {"size": [0.02] * 3}}})
            with self.assertRaises(ConfigError):
                load_task_spec("fx_wp1_cmaxmiss")


class PaletteRolesHoldoutTests(unittest.TestCase):
    """WP1 §4.1 / §4.5 / §4.6 palette, roles, holdout and instruction templates."""

    def test_contract_example_fields(self):
        """§4.6 the colour-cube example: palette in TOML order, roles, holdout, templates."""
        with sandbox() as box:
            spec = _load(box, "fx_wp1_fields", contract_color_task_table())
            self.assertEqual(list(spec.palette), ["red", "green", "blue", "yellow", "orange", "purple", "pink"])
            self.assertEqual(tuple(spec.palette["red"]), (0.8, 0.05, 0.05))
            self.assertTrue(all(isinstance(v, float) for v in spec.palette["green"]))
            self.assertEqual(dict(spec.roles), {"target": "cubes", "container": "small_box"})
            self.assertEqual(spec.holdout, HoldoutSpec(("purple",), (4,)))
            self.assertEqual(spec.instruction_templates, ("Pick up the {target.color} cube and place it in the box",))

    def test_contract_reach_example_loads(self):
        """§4.6 the primitive example loads (no objects, no roles)."""
        with sandbox() as box:
            spec = _load(box, "fx_wp1_reach", contract_reach_task_table(), files={})
            self.assertEqual(dict(spec.objects), {})
            self.assertEqual(spec.instruction_templates, ("Move the gripper to the point",))

    def test_palette_from_setup_resolves_when_read(self):
        """§4.1 "from_setup:<key>" resolves through load_setup() when read, not at load."""
        palette = {"red": [0.8, 0.05, 0.05], "blue": "from_setup:blue_cube"}
        table = _color_table(palette=palette, holdout=None)
        table["object_sets"]["cubes"]["count_max"] = 2
        table["start"]["count"] = [1, 2]
        with sandbox() as box:
            spec = _load(box, "fx_wp1_fromsetup", table)        # no real_setup.json yet: loading must not read it
            self.assertEqual(list(spec.palette), ["red", "blue"])
            write_scene(box.scene, {"object_materials": {"blue_cube": {"color": [0.005, 0.07, 0.26], "roughness": 0.6}}})
            self.assertEqual(tuple(spec.palette["blue"]), (0.005, 0.07, 0.26))
            self.assertEqual(tuple(spec.palette["red"]), (0.8, 0.05, 0.05))

    def test_bad_palette_values(self):
        """§1.1 / §4.6 palette values are [r, g, b] in 0..1 or "from_setup:<key>" (ConfigError otherwise)."""
        with sandbox() as box:
            for i, bad in enumerate([[0.1, 0.2], [0.1, 0.2, 1.5], "blue", [0.1, 0.2, "x"]]):
                with self.subTest(value=bad):
                    write_task(box.tasks, f"fx_wp1_pal{i}", {"variants": VARIANTS, "palette": {"c": bad}})
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_pal{i}")

    def test_unknown_role_target(self):
        """§4.6 a role names an object or set; anything else is a ConfigError."""
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_role", _color_table(roles={"target": "cubes", "container": "ghost"}),
                       files=BOX_FILES)
            with self.assertRaises(ConfigError):
                load_task_spec("fx_wp1_role")

    def test_holdout_validation(self):
        """§4.5 holdout colours must be in the palette and counts in the count range (ConfigError)."""
        cases = {"colour_not_in_palette": {"colors": ["teal"], "counts": []},
                 "count_out_of_range": {"colors": [], "counts": [7]},
                 "count_below_range": {"colors": [], "counts": [1]}}
        with sandbox() as box:
            for i, (label, holdout) in enumerate(cases.items()):
                with self.subTest(label):
                    write_task(box.tasks, f"fx_wp1_hold{i}", _color_table(holdout=holdout), files=BOX_FILES)
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_hold{i}")

    def test_no_allowed_count(self):
        """§4.5 at least one allowed count must remain after hold-outs (ConfigError)."""
        table = _color_table(holdout={"colors": [], "counts": [2, 3]})
        table["start"]["count"] = [2, 3]
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_nocount", table, files=BOX_FILES)
            with self.assertRaises(ConfigError):
                load_task_spec("fx_wp1_nocount")

    def test_palette_too_small_for_distinct(self):
        """§4.5 len(palette minus held-out colours) >= count_max when distinct_colors (ConfigError)."""
        table = _color_table(holdout={"colors": ["purple", "pink"], "counts": []})     # 5 left, count_max 6
        with sandbox() as box:
            write_task(box.tasks, "fx_wp1_smallpal", table, files=BOX_FILES)
            with self.assertRaises(ConfigError):
                load_task_spec("fx_wp1_smallpal")
            table["start"]["distinct_colors"] = False
            write_task(box.tasks, "fx_wp1_smallpal_ok", table, files=BOX_FILES)
            load_task_spec("fx_wp1_smallpal_ok")        # repeated distractor colours allowed

    def test_count_range_validation(self):
        """§4.6 start.count = [lo, hi] with 1 <= lo <= hi <= count_max (ConfigError otherwise)."""
        with sandbox() as box:
            for i, bad in enumerate([[0, 3], [3, 2], [2, 7]]):
                with self.subTest(count=bad):
                    table = _color_table(holdout={"colors": [], "counts": []})
                    table["start"]["count"] = bad
                    write_task(box.tasks, f"fx_wp1_count{i}", table, files=BOX_FILES)
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_count{i}")

    def test_instruction_template_placeholders(self):
        """§4.5 placeholders are {<role>.color}, {<role>.name}, {goal.<key>}; others are a load-time ConfigError."""
        good = ["Put the {target.color} {target.name} in the {container.name}", "Plain text"]
        bad = ["Pick {target}", "Pick {target.size}", "Pick {ghost.color}", "Go to {goal.nowhere}", "Pick {}",
               "Pick {0}"]
        with sandbox() as box:
            table = _color_table(instruction_template={"texts": good})
            spec = _load(box, "fx_wp1_tmpl_ok", table)
            self.assertEqual(spec.instruction_templates, tuple(good))
            for i, text in enumerate(bad):
                with self.subTest(template=text):
                    write_task(box.tasks, f"fx_wp1_tmpl{i}", _color_table(instruction_template={"texts": [text]}),
                               files=BOX_FILES)
                    with self.assertRaises(ConfigError):
                        load_task_spec(f"fx_wp1_tmpl{i}")
            write_task(box.tasks, "fx_wp1_tmpl_empty", _color_table(instruction_template={"texts": []}),
                       files=BOX_FILES)
            with self.assertRaises(ConfigError):
                load_task_spec("fx_wp1_tmpl_empty")
            reach = contract_reach_task_table()
            reach["instruction_template"] = {"texts": ["Reach {goal.point}"]}
            self.assertEqual(_load(box, "fx_wp1_tmpl_goal", reach, files={}).instruction_templates, ("Reach {goal.point}",))


class StartTableValidationTests(unittest.TestCase):
    """WP1 §4.5 / §4.6 [task.start] validation at load time (ConfigError, never mid-episode)."""

    def _expect_error(self, box, name, table):
        write_task(box.tasks, name, table, files=BOX_FILES)
        with self.assertRaises(ConfigError):
            load_task_spec(name)

    def test_place_key_must_be_object_or_set(self):
        """§4.6 start.place.<object|set> keys name a plain object or an object set."""
        with sandbox() as box:
            table = _color_table()
            table["start"]["place"]["ghost"] = {"region": {"xy_min": [0.1, 0.1], "xy_max": [0.2, 0.2]}}
            self._expect_error(box, "fx_wp1_place_ghost", table)
            table = _color_table()
            table["start"]["place"]["cube_0"] = {"region": {"xy_min": [0.1, 0.1], "xy_max": [0.2, 0.2]}}
            self._expect_error(box, "fx_wp1_place_slot", table)

    def test_bad_regions(self):
        """§4.6 a region is rect {xy_min, xy_max} with min <= max, or polar {r_m, bearing_deg} with lo <= hi."""
        bad = [{"xy_min": [0.3, 0.0], "xy_max": [0.2, 0.1]}, {"xy_min": [0.1, 0.0]}, {"r_m": [0.3, 0.2], "bearing_deg": [0, 1]},
               {"r_m": [0.1, 0.2], "bearing_deg": [10.0, -10.0]}, {"center": [0.1, 0.1]}]
        with sandbox() as box:
            for i, region in enumerate(bad):
                with self.subTest(region=region):
                    table = _color_table()
                    table["start"]["place"]["small_box"] = {"region": region}
                    self._expect_error(box, f"fx_wp1_region{i}", table)
            table = _color_table()
            del table["start"]["place"]["small_box"]["region"]
            self._expect_error(box, "fx_wp1_region_missing", table)

    def test_reachable_and_held_refs(self):
        """§4.6 reachable entries and held are refs that must resolve (role, object or goal key)."""
        with sandbox() as box:
            table = _color_table()
            table["start"]["reachable"] = ["@target", "@ghost"]
            self._expect_error(box, "fx_wp1_reach_bad", table)
            table = _color_table()
            table["start"]["held"] = "@nobody"
            self._expect_error(box, "fx_wp1_held_bad", table)
            table = _color_table()
            table["start"]["held"] = "@target"
            _load(box, "fx_wp1_held_ok", table)

    def test_unknown_goal_type(self):
        """§4.6 start.goal.<key>.type is one of point, points, grid, joint, gap, region."""
        with sandbox() as box:
            reach = contract_reach_task_table()
            reach["start"]["goal"]["point"]["type"] = "sphere"
            self._expect_error(box, "fx_wp1_goaltype", reach)

    def test_count_without_target_set(self):
        """§4.6 start.count applies to the set of the target role; without one it is a ConfigError."""
        with sandbox() as box:
            reach = contract_reach_task_table()
            reach["start"]["count"] = [1, 2]
            self._expect_error(box, "fx_wp1_count_noset", reach)

    def test_failure_rules_last_catch_all(self):
        """§4.2 the last failure rule must have empty if and unless (ConfigError at load)."""
        with sandbox() as box:
            table = _color_table(failures={"rules": [{"name": "a", "unless": ["reached"]}, {"name": "b", "if": ["placed"]}]})
            self._expect_error(box, "fx_wp1_rules_last", table)
            table = _color_table(failures={"rules": [{"name": "a", "when": ["reached"]}, {"name": "b"}]})
            self._expect_error(box, "fx_wp1_rules_key", table)
            table = _color_table(failures={"rules": [{"if": ["reached"]}, {"name": "b"}]})
            self._expect_error(box, "fx_wp1_rules_name", table)


class FixtureTaskTests(unittest.TestCase):
    """WP1 §13 fixture tasks fx_pick, fx_reach, fx_color (tests/fixtures/v4_wp1/tasks) load as GenericTask."""

    def test_fixtures_load(self):
        """§13 each fixture folder loads through core.tasks.loader as a GenericTask."""
        from core.tasks.generic import GenericTask
        with sandbox() as box:
            for name in ("fx_pick", "fx_reach", "fx_color"):
                copy_fixture(f"v4_wp1/tasks/{name}", box.root)
            with temp_env(S2R_TASKS_DIR=box.root / "v4_wp1" / "tasks"):
                clear_task_cache()
                for name in ("fx_pick", "fx_reach", "fx_color"):
                    with self.subTest(name):
                        task = load_task(name)
                        self.assertIsInstance(task, GenericTask)
                        self.assertEqual(task.name, name)
                        sample = task.sample_episode_start(np.random.default_rng(0))
                        self.assertTrue(sample.instruction)
                clear_task_cache()


if __name__ == "__main__":
    unittest.main()
