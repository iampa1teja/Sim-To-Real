"""WP2: arm.toml v4 tables — [arm.kinematics], [arm.gripper], [arm.expert], jaw link names (contracts §5.1, §1.5).

load_arm_spec validates the v4 tables when they are present (ConfigError at load time) and returns
the file content unchanged; an arm.toml without them loads exactly as before.
"""

import copy
import json
import unittest

from core.arms.gripper import Gripper
from core.arms.spec import load_arm_spec
from core.generate.errors import ConfigError
from v4_support import FIXTURES_DIR, copy_repo_folder, sandbox
from v4_support.fx_arms import FX_ARMS, fx_arm_table, fx_kinematics, write_arm

V4_TABLES = ("kinematics", "gripper", "expert", "workspace")  # workspace: WP14 §16.1
V4_LINK_KEYS = ("moving_jaw", "moving_jaw_mesh")


def without_v4(spec):
    out = copy.deepcopy(spec)
    for key in V4_TABLES:
        out.pop(key, None)
    for key in V4_LINK_KEYS:
        out.get("links", {}).pop(key, None)
    return out


class FxArmSpecTests(unittest.TestCase):
    """WP2 §5.1 synthetic arms with and without the v4 tables."""

    def test_full_fx_table_loads_unchanged(self):
        """§5.1 a valid arm.toml with all v4 tables loads; load_arm_spec returns the file content (no injected keys)."""
        for name in FX_ARMS:
            with self.subTest(arm=name), sandbox() as box:
                folder = write_arm(box.arms, name, fx_arm_table(name), kinematics=fx_kinematics(name))
                spec, arm_dir = load_arm_spec(name)
                self.assertEqual(arm_dir, folder)
                self.assertEqual(spec, fx_arm_table(name))

    def test_table_without_v4_keys_loads_as_before(self):
        """§5.1 regression: an arm.toml without the new tables or jaw keys loads exactly as written."""
        for name in FX_ARMS:
            with self.subTest(arm=name), sandbox() as box:
                table = without_v4(fx_arm_table(name))
                table["variants"] = {"alt": {"usd_path": "usd/alt.usd"}}
                write_arm(box.arms, name, table)
                spec, _ = load_arm_spec(name)
                expected = dict(table)
                expected.pop("variants")
                self.assertEqual(spec, expected)
                alt, _ = load_arm_spec(name, "alt")
                self.assertEqual(alt["usd_path"], "usd/alt.usd")

    def test_partial_v4_tables_load(self):
        """§5.1/§13 container-measured keys may be absent until measured: partial tables still load."""
        name = "fx_arm5"
        table = fx_arm_table(name)
        for key in ("tcp_position_m", "approach_axis", "closing_axis"):
            del table["kinematics"][key]
        for key in ("open_rad", "closed_rad", "opening_table"):
            del table["gripper"][key]
        with sandbox() as box:
            write_arm(box.arms, name, table)
            spec, _ = load_arm_spec(name)
            self.assertEqual(spec["gripper"], {"joint": "fx5_grip", "max_object_width_m": 0.045})

    def test_invalid_v4_tables_raise_config_error_at_load(self):
        """§1.5/§5.1 an invalid v4 table raises ConfigError (a ValueError) from load_arm_spec."""
        name = "fx_arm6"

        def edit(path, value=None, *, delete=False):
            def apply(table):
                node = table
                for key in path[:-1]:
                    node = node[key]
                if delete:
                    del node[path[-1]]
                else:
                    node[path[-1]] = value
            return apply

        cases = {
            "moving_jaw_not_str": edit(("links", "moving_jaw"), 3),
            "moving_jaw_mesh_not_str": edit(("links", "moving_jaw_mesh"), ["a"]),
            "ee_link_differs_from_links_ee": edit(("kinematics", "ee_link"), "fx6_l5"),
            "tcp_position_len": edit(("kinematics", "tcp_position_m"), [0.0, 0.1]),
            "tcp_quat_len": edit(("kinematics", "tcp_quat_wxyz"), [1.0, 0.0, 0.0]),
            "approach_not_unit": edit(("kinematics", "approach_axis"), [0.0, 0.0, 2.0]),
            "closing_not_perpendicular": edit(("kinematics", "closing_axis"), [0.0, 0.6, 0.8]),
            "file_not_str": edit(("kinematics", "file"), 5),
            "gripper_joint_unknown": edit(("gripper", "joint"), "fx6_nojoint"),
            "opening_table_one_row": edit(("gripper", "opening_table"), [[0.0, 0.0]]),
            "opening_angles_not_increasing": edit(("gripper", "opening_table"), [[0.0, 0.0], [0.5, 0.01], [0.4, 0.02]]),
            "opening_gaps_not_monotone": edit(("gripper", "opening_table"), [[0.0, 0.0], [0.5, 0.03], [1.0, 0.02]]),
            "max_object_width_not_positive": edit(("gripper", "max_object_width_m"), 0.0),
            "arm_joints_unknown": edit(("expert", "arm_joints"), ["fx6_j1", "fx6_nojoint"]),
            "arm_joints_contains_gripper": edit(("expert", "arm_joints"), ["fx6_j1", "fx6_j2", "fx6_grip"]),
            "home_rad_unknown_joint": edit(("expert", "home_rad"), {"fx6_nojoint": 0.0}),
            "joint_speed_not_positive": edit(("expert", "max_joint_speed_rad_s"), 0.0),
            "joint_speed_unknown_joint": edit(("expert", "max_joint_speed_rad_s"), {"fx6_nojoint": 1.0}),
            "gripper_speed_not_positive": edit(("expert", "max_gripper_speed_rad_s"), -1.0),
            "ee_speed_not_positive": edit(("expert", "max_ee_speed_m_s"), 0.0),
            "ik_axis_weight_negative": edit(("expert", "ik_axis_weight"), -0.1),
            "ik_tol_pos_not_positive": edit(("expert", "ik_tol_pos_m"), 0.0),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label), sandbox() as box:
                table = fx_arm_table(name)
                mutate(table)
                write_arm(box.arms, name, table, kinematics=fx_kinematics(name))
                with self.assertRaises(ConfigError):
                    load_arm_spec(name)

    def test_valid_optional_forms(self):
        """§5.1 per-joint speed table and home_rad are accepted."""
        name = "fx_arm6"
        table = fx_arm_table(name)
        table["expert"]["max_joint_speed_rad_s"] = {"fx6_j1": 1.0, "fx6_j2": 2.0}
        table["expert"]["home_rad"] = {"fx6_j1": 0.1, "fx6_j3": -0.2}
        with sandbox() as box:
            write_arm(box.arms, name, table)
            spec, _ = load_arm_spec(name)
            self.assertEqual(spec["expert"]["max_joint_speed_rad_s"], {"fx6_j1": 1.0, "fx6_j2": 2.0})

    def test_invalid_table_in_variant_is_detected(self):
        """§1.5 validation runs on the merged spec, so a variant that breaks a v4 table raises ConfigError."""
        name = "fx_arm5"
        table = fx_arm_table(name)
        table["variants"] = {"broken": {"gripper": {"joint": "fx5_nojoint"}}}
        with sandbox() as box:
            write_arm(box.arms, name, table)
            load_arm_spec(name)
            with self.assertRaises(ConfigError):
                load_arm_spec(name, "broken")


class SO101ArmSpecTests(unittest.TestCase):
    """WP2 regression and §5.1 SO-101 values for the repo arm (copied to a temp S2R_ARMS_DIR)."""

    GOLDEN = FIXTURES_DIR / "v4_wp2" / "so101_arm_spec_before_v4.json"

    def _load(self, variant=None):
        with sandbox() as box:
            copy_repo_folder("arms", "so101", box.root)
            return load_arm_spec("so101", variant)[0]

    def test_existing_keys_unchanged(self):
        """Regression: apart from the v4 additions, the SO-101 spec (base and every variant) is byte-for-byte as before."""
        golden = json.loads(self.GOLDEN.read_text())
        for variant in (None, "contact_grasp", "no_camera"):
            with self.subTest(variant=variant):
                self.assertEqual(without_v4(self._load(variant)), golden[variant or "base"])

    def test_v4_values(self):
        """§5.1 SO-101 column: values fixed by the contract (container-measured keys are checked only if present)."""
        spec = self._load()
        self.assertEqual(spec["links"]["moving_jaw"], "jaw")
        self.assertEqual(spec["links"]["moving_jaw_mesh"], "moving_jaw_so101_v1")
        kin = spec["kinematics"]
        self.assertEqual(kin["file"], "kinematics.json")
        self.assertEqual(kin["ee_link"], "gripper")
        self.assertEqual(kin["ee_link"], spec["links"]["ee"])
        self.assertEqual(kin["tcp_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
        grip = spec["gripper"]
        self.assertEqual(grip["joint"], "Jaw")
        self.assertEqual(grip["max_object_width_m"], 0.045)
        expert = spec["expert"]
        self.assertEqual(expert["arm_joints"], [n for n in spec["joints"]["names"] if n != "Jaw"])
        self.assertEqual(len(expert["arm_joints"]), 5)
        self.assertEqual(expert["max_joint_speed_rad_s"], 1.5)
        self.assertEqual(expert["max_gripper_speed_rad_s"], 3.0)
        self.assertEqual(expert["max_ee_speed_m_s"], 0.15)
        self.assertEqual((expert["ik_position_weight"], expert["ik_axis_weight"], expert["ik_roll_weight"]),
                         (1.0, 0.3, 0.1))
        self.assertEqual((expert["ik_tol_pos_m"], expert["ik_tol_rot_rad"]), (0.0005, 0.02))
        self.assertNotIn("home_rad", expert)    # default = [arm.initial_state.joint_positions]
        if "tcp_position_m" in kin:
            self.assertEqual(len(kin["tcp_position_m"]), 3)
        if {"open_rad", "closed_rad", "opening_table"} <= set(grip):
            Gripper.from_table(grip, spec["joints"]["names"])


if __name__ == "__main__":
    unittest.main()
