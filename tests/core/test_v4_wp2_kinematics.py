"""WP2: kinematics.json format and KinematicChain FK / Jacobian (contracts §5.2, §5.3, §1.5, §13).

All checks use the synthetic fx arms of ``v4_support.fx_arms`` (no SO-101 literals) and compare with
the independent reference FK there.
"""

import copy
import json
import math
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

from core.arms.kinematics import IKResult, IKWeights, JointDef, KinematicChain, load_kinematics_json  # noqa: F401
from core.generate.errors import ConfigError, KinematicsError, SchemaError
from core.tasks.base import Pose
from v4_support import FIXTURES_DIR, sandbox
from v4_support.fx_arms import (FX_ARMS, fx_arm_table, fx_joint_names, fx_kinematics, fx_limits, fx_reference_tcp,
                                fx_setup, frame_matrix, matrix_to_quat_wxyz, random_q,
                                reference_fk, reference_link_poses, write_arm, write_fx_arm)

_RX90 = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]


def make_chain(name, *, data=None, chain_joints=None, tcp_quat_wxyz=None, q_names=None, ee_link=None):
    """KinematicChain.from_dict with the fx arm's arm.toml values (overridable)."""
    table = fx_arm_table(name)
    kin = table["kinematics"]
    return KinematicChain.from_dict(
        fx_kinematics(name) if data is None else data,
        q_names=fx_joint_names(name) if q_names is None else q_names,
        chain_joints=table["expert"]["arm_joints"] if chain_joints is None else chain_joints,
        ee_link=kin["ee_link"] if ee_link is None else ee_link,
        tcp_position_m=kin["tcp_position_m"],
        tcp_quat_wxyz=kin["tcp_quat_wxyz"] if tcp_quat_wxyz is None else tcp_quat_wxyz,
        approach_axis=kin["approach_axis"], closing_axis=kin["closing_axis"])


def pose_matrix(pose: Pose) -> np.ndarray:
    """4x4 of an in-memory Pose (XYZW quaternion)."""
    x, y, z, w = pose.quat_xyzw
    return frame_matrix(pose.position, [w, x, y, z])


def _vee(skew):
    return np.array([skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0], skew[1, 0] - skew[0, 1]]) / 2.0


class FixtureTests(unittest.TestCase):
    """WP2 §13 committed fixtures ``arms/fx_arm{5,6}`` and ``scenes/fx_scene`` match ``v4_support.fx_arms``."""

    def test_arm_fixtures_match_support_data(self):
        """§13 tests/fixtures/arms/<fx>/ hold exactly fx_arm_table and fx_kinematics."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                folder = FIXTURES_DIR / "arms" / name
                with open(folder / "arm.toml", "rb") as handle:
                    self.assertEqual(tomllib.load(handle), {"arm": fx_arm_table(name)})
                self.assertEqual(json.loads((folder / "kinematics.json").read_text()), fx_kinematics(name))

    def test_scene_fixture_has_arms_block(self):
        """§13 tests/fixtures/scenes/fx_scene/real_setup.json has an arms block with mapping and limits for both arms."""
        setup = json.loads((FIXTURES_DIR / "scenes" / "fx_scene" / "real_setup.json").read_text())
        self.assertEqual(setup, fx_setup())


class KinematicsJsonTests(unittest.TestCase):
    """WP2 §5.2 kinematics.json schema_version 1; §1.5 malformed files raise SchemaError."""

    def _write(self, tmp, payload, *, raw=None):
        path = Path(tmp) / "kinematics.json"
        path.write_text(raw if raw is not None else json.dumps(payload))
        return path

    def test_valid_file_loads_unchanged(self):
        """§5.2 load_kinematics_json returns the parsed document of a valid file."""
        for name in FX_ARMS:
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as tmp:
                path = self._write(tmp, fx_kinematics(name))
                self.assertEqual(load_kinematics_json(path), fx_kinematics(name))
                self.assertEqual(load_kinematics_json(str(path)), fx_kinematics(name))

    def test_missing_file_is_file_not_found(self):
        """§5.2 a missing file is an OSError (FileNotFoundError), not a silent default."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_kinematics_json(Path(tmp) / "nope.json")

    def test_malformed_files_raise_schema_error(self):
        """§1.5/§5.2 every malformed kinematics.json raises SchemaError (a ValueError)."""

        def joint0(mutate):
            def apply(d):
                mutate(d["joints"][0])
            return apply

        mutations = {
            "schema_version_2": lambda d: d.update(schema_version=2),
            "schema_version_missing": lambda d: d.pop("schema_version"),
            "joints_missing": lambda d: d.pop("joints"),
            "joints_empty": lambda d: d.update(joints=[]),
            "base_link_missing": lambda d: d.pop("base_link"),
            "source_missing": lambda d: d.pop("source"),
            "type_prismatic": joint0(lambda j: j.update(type="prismatic")),
            "axis_not_unit": joint0(lambda j: j.update(axis=[0, 0, 2])),
            "axis_not_basis": joint0(lambda j: j.update(axis=[0.6, 0.8, 0.0])),
            "axis_short": joint0(lambda j: j.update(axis=[0, 1])),
            "axis_string": joint0(lambda j: j.update(axis="Z")),
            "parent_position_short": joint0(lambda j: j["parent_frame"].update(position=[0.0, 0.0])),
            "child_quat_short": joint0(lambda j: j["child_frame"].update(quat_wxyz=[1.0, 0.0, 0.0])),
            "parent_frame_missing": joint0(lambda j: j.pop("parent_frame")),
            "child_missing": joint0(lambda j: j.pop("child")),
            "name_missing": joint0(lambda j: j.pop("name")),
            "limits_reversed": joint0(lambda j: j.update(limits_rad=[1.0, -1.0])),
            "limits_short": joint0(lambda j: j.update(limits_rad=[1.0])),
            "limits_nan": joint0(lambda j: j.update(limits_rad=[float("nan"), 1.0])),
            "position_inf": joint0(lambda j: j["parent_frame"].update(position=[0.0, float("inf"), 0.0])),
            "position_string": joint0(lambda j: j["parent_frame"].update(position=[0.0, "a", 0.0])),
            "duplicate_joint_name": lambda d: d["joints"].append(copy.deepcopy(d["joints"][0])),
            "joint_not_object": lambda d: d["joints"].append("fx_joint"),
        }
        for name in FX_ARMS:
            for label, mutate in mutations.items():
                with self.subTest(arm=name, case=label), tempfile.TemporaryDirectory() as tmp:
                    data = fx_kinematics(name)
                    mutate(data)
                    path = self._write(tmp, data)
                    with self.assertRaises(SchemaError):
                        load_kinematics_json(path)

    def test_non_object_and_non_json_raise_schema_error(self):
        """§1.5 a top-level list or invalid JSON text raises SchemaError."""
        for label, raw in {"list": "[1, 2]", "garbage": "{not json", "empty": ""}.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(SchemaError):
                    load_kinematics_json(self._write(tmp, None, raw=raw))


class ChainConstructionTests(unittest.TestCase):
    """WP2 §5.3 KinematicChain attributes; §1.5 construction errors raise KinematicsError."""

    def test_attributes(self):
        """§5.3 q_names / chain_joints are tuples; lower / upper are the USD limits, float64 (J,)."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                chain = make_chain(name)
                self.assertEqual(chain.q_names, fx_joint_names(name))
                self.assertIsInstance(chain.q_names, tuple)
                self.assertEqual(chain.chain_joints, tuple(fx_arm_table(name)["expert"]["arm_joints"]))
                self.assertIsInstance(chain.chain_joints, tuple)
                lower, upper = fx_limits(name)
                for got, want in ((chain.lower, lower), (chain.upper, upper)):
                    self.assertIsInstance(got, np.ndarray)
                    self.assertEqual(got.dtype, np.float64)
                    np.testing.assert_array_equal(got, want)

    def test_limits_follow_q_names_order(self):
        """§1.2/§5.3 lower/upper are indexed like q (q_names order), not like the kinematics.json joint list."""
        name = "fx_arm6"
        data = fx_kinematics(name)
        data["joints"] = list(reversed(data["joints"]))
        chain = make_chain(name, data=data)
        lower, upper = fx_limits(name)
        np.testing.assert_array_equal(chain.lower, lower)
        np.testing.assert_array_equal(chain.upper, upper)

    def test_construction_errors(self):
        """§1.5 missing joint, cycle / two parents, unconnected chain, unknown ee link and bad axis raise KinematicsError."""
        name = "fx_arm6"
        two_parents = fx_kinematics(name)
        extra = copy.deepcopy(two_parents["joints"][3])
        extra.update(name="fx6_extra", parent="fx6_l3", child="fx6_l1")
        two_parents["joints"].append(extra)
        loop = fx_kinematics(name)
        a = copy.deepcopy(loop["joints"][0])
        a.update(name="fx6_loop_a", parent="fx6_x", child="fx6_y")
        b = copy.deepcopy(loop["joints"][0])
        b.update(name="fx6_loop_b", parent="fx6_y", child="fx6_x")
        loop["joints"] += [a, b]
        unconnected = fx_kinematics(name)
        unconnected["joints"][2]["parent"] = "fx6_nowhere"
        bad_axis = fx_kinematics(name)
        bad_axis["joints"][1]["axis"] = [0.0, 0.0, 2.0]
        cases = {
            "chain_joint_not_in_file": dict(chain_joints=list(fx_joint_names(name)[:5]) + ["fx6_missing"]),
            "chain_joint_not_in_q_names": dict(q_names=fx_joint_names(name)[1:]),
            "two_parents": dict(data=two_parents),
            "loop": dict(data=loop),
            "unconnected": dict(data=unconnected),
            "unknown_ee_link": dict(ee_link="fx6_nolink"),
            "bad_axis": dict(data=bad_axis),
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(KinematicsError):
                    make_chain(name, **kwargs)


class ForwardKinematicsTests(unittest.TestCase):
    """WP2 §5.2/§5.3 FK: T_child = T_parent · F(parent_frame) · R(axis, q) · F(child_frame)⁻¹; tcp = ee ∘ tcp."""

    N = 60

    def test_fk_tcp_and_ee_match_reference(self):
        """§5.3 fk(q, "tcp") / fk(q, "ee") equal the reference FK within 1e-9 for random in-limit q."""
        for name in FX_ARMS:
            chain = make_chain(name)
            ee = fx_arm_table(name)["kinematics"]["ee_link"]
            rng = np.random.default_rng(2)
            for k in range(self.N):
                q = random_q(name, rng)
                with self.subTest(arm=name, k=k):
                    tcp = chain.fk(q)
                    self.assertEqual(tcp.shape, (4, 4))
                    self.assertEqual(tcp.dtype, np.float64)
                    np.testing.assert_allclose(tcp, fx_reference_tcp(name, q), rtol=0, atol=1e-9)
                    np.testing.assert_allclose(chain.fk(q, frame="tcp"), tcp, rtol=0, atol=0)
                    np.testing.assert_allclose(chain.fk(q, "ee"), reference_fk(fx_kinematics(name), q, ee),
                                               rtol=0, atol=1e-9)

    def test_fk_accepts_lists_and_does_not_mutate_input(self):
        """§1.3 fk accepts any array-like q of length J and leaves it unchanged."""
        name = "fx_arm5"
        chain = make_chain(name)
        q = random_q(name, np.random.default_rng(3))
        before = q.copy()
        np.testing.assert_allclose(chain.fk(list(q)), chain.fk(q), rtol=0, atol=0)
        np.testing.assert_array_equal(q, before)

    def test_fk_rejects_wrong_length(self):
        """§5.3 chain APIs take full-length q (J,): other shapes raise ValueError."""
        for name in FX_ARMS:
            chain = make_chain(name)
            n = len(fx_joint_names(name))
            for bad in (np.zeros(n - 1), np.zeros(n + 1), np.zeros((1, n))):
                with self.subTest(arm=name, shape=bad.shape):
                    with self.assertRaises(ValueError):
                        chain.fk(bad)
                    with self.assertRaises(ValueError):
                        chain.jacobian(bad)

    def test_rotated_tcp_quaternion(self):
        """§1.1 tcp = ee ∘ F(tcp_position_m, tcp_quat_wxyz) with a non-identity WXYZ quaternion."""
        for name in FX_ARMS:
            chain = make_chain(name, tcp_quat_wxyz=_RX90)
            table = fx_arm_table(name)
            rng = np.random.default_rng(4)
            for k in range(10):
                q = random_q(name, rng)
                with self.subTest(arm=name, k=k):
                    want = reference_fk(fx_kinematics(name), q, table["kinematics"]["ee_link"],
                                        tcp_position_m=table["kinematics"]["tcp_position_m"], tcp_quat_wxyz=_RX90)
                    np.testing.assert_allclose(chain.fk(q), want, rtol=0, atol=1e-9)

    def test_link_poses_match_reference(self):
        """§5.3 link_poses(q) holds every link (base included) at the reference pose; all q_names joints act."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(5)
            for k in range(10):
                q = random_q(name, rng)
                with self.subTest(arm=name, k=k):
                    got = chain.link_poses(q)
                    want = reference_link_poses(fx_kinematics(name), q, fx_joint_names(name))
                    self.assertEqual(set(got), set(want))
                    for link, pose in want.items():
                        np.testing.assert_allclose(got[link], pose, rtol=0, atol=1e-9, err_msg=link)

    def test_gripper_joint_moves_jaw_link_but_not_tcp(self):
        """§1.2 the gripper joint is in q but not in the chain: it moves the jaw link, never the TCP."""
        for name in FX_ARMS:
            chain = make_chain(name)
            table = fx_arm_table(name)
            gi = fx_joint_names(name).index(table["gripper"]["joint"])
            q = random_q(name, np.random.default_rng(6))
            q2 = q.copy()
            q2[gi] = 0.9 if q[gi] < 0.5 else 0.1
            with self.subTest(arm=name):
                np.testing.assert_allclose(chain.fk(q2), chain.fk(q), rtol=0, atol=1e-12)
                jaw = table["links"]["moving_jaw"]
                self.assertGreater(np.abs(chain.link_poses(q2)[jaw] - chain.link_poses(q)[jaw]).max(), 1e-4)

    def test_fk_pose_and_tcp_from_ee(self):
        """§5.3 fk_pose is fk as a Pose (XYZW in memory, §1.1); tcp_from_ee(ee pose) is the TCP pose."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(7)
            for k in range(20):
                q = random_q(name, rng)
                with self.subTest(arm=name, k=k):
                    for frame in ("tcp", "ee"):
                        pose = chain.fk_pose(q, frame)
                        self.assertIsInstance(pose, Pose)
                        self.assertEqual(len(pose.position), 3)
                        self.assertEqual(len(pose.quat_xyzw), 4)
                        self.assertAlmostEqual(float(np.linalg.norm(pose.quat_xyzw)), 1.0, places=9)
                        np.testing.assert_allclose(pose_matrix(pose), chain.fk(q, frame), rtol=0, atol=1e-9)
                    tcp = chain.tcp_from_ee(chain.fk_pose(q, "ee"))
                    self.assertIsInstance(tcp, Pose)
                    np.testing.assert_allclose(pose_matrix(tcp), chain.fk(q), rtol=0, atol=1e-9)

    def test_fk_pose_quaternion_is_xyzw(self):
        """§1.1 Pose.quat_xyzw is XYZW: a 90° rotation about x gives (s, 0, 0, s) up to sign."""
        name = "fx_arm6"
        chain = make_chain(name, tcp_quat_wxyz=_RX90)
        q = np.zeros(len(fx_joint_names(name)))
        want = matrix_to_quat_wxyz(chain.fk(q))   # WXYZ
        got = np.asarray(chain.fk_pose(q).quat_xyzw)
        got_wxyz = np.r_[got[3], got[:3]]
        self.assertTrue(np.allclose(got_wxyz, want, atol=1e-9) or np.allclose(got_wxyz, -want, atol=1e-9),
                        (got, want))


class JacobianTests(unittest.TestCase):
    """WP2 §5.3 jacobian: (6, J), rows [v; ω] in the base frame, non-chain columns 0; vs central differences."""

    def _numeric(self, chain, q, frame, h=1e-6):
        out = np.zeros((6, len(q)))
        for i in range(len(q)):
            dq = np.zeros(len(q))
            dq[i] = h
            plus, minus = chain.fk(q + dq, frame), chain.fk(q - dq, frame)
            out[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2 * h)
            mid = chain.fk(q, frame)[:3, :3]
            out[3:, i] = _vee(((plus[:3, :3] - minus[:3, :3]) / (2 * h)) @ mid.T)
        return out

    def test_jacobian_matches_central_differences(self):
        """§5.3 table: jacobian vs central differences (h = 1e-6) within 1e-5, for tcp and ee frames."""
        for name in FX_ARMS:
            chain = make_chain(name)
            rng = np.random.default_rng(8)
            for k in range(25):
                q = random_q(name, rng)
                for frame in ("tcp", "ee"):
                    with self.subTest(arm=name, k=k, frame=frame):
                        jac = chain.jacobian(q, frame)
                        self.assertEqual(jac.shape, (6, len(q)))
                        self.assertEqual(jac.dtype, np.float64)
                        np.testing.assert_allclose(jac, self._numeric(chain, q, frame), rtol=0, atol=1e-5)

    def test_default_frame_is_tcp(self):
        """§5.3 jacobian(q) defaults to frame="tcp"."""
        chain = make_chain("fx_arm6")
        q = random_q("fx_arm6", np.random.default_rng(9))
        np.testing.assert_array_equal(chain.jacobian(q), chain.jacobian(q, "tcp"))
        self.assertGreater(np.abs(chain.jacobian(q, "tcp") - chain.jacobian(q, "ee")).max(), 1e-3)

    def test_non_chain_columns_are_zero(self):
        """§5.3 columns of joints outside chain_joints (gripper, or an excluded arm joint) are exactly 0."""
        name = "fx_arm5"
        names = fx_joint_names(name)
        full = make_chain(name)
        arm_joints = fx_arm_table(name)["expert"]["arm_joints"]
        reduced = make_chain(name, chain_joints=[j for j in arm_joints if j != "fx5_roll"])
        rng = np.random.default_rng(10)
        for k in range(10):
            q = random_q(name, rng)
            with self.subTest(k=k):
                jf, jr = full.jacobian(q), reduced.jacobian(q)
                grip, roll = names.index("fx5_grip"), names.index("fx5_roll")
                np.testing.assert_array_equal(jf[:, grip], 0.0)
                np.testing.assert_array_equal(jr[:, grip], 0.0)
                np.testing.assert_array_equal(jr[:, roll], 0.0)
                self.assertGreater(np.abs(jf[:, roll]).max(), 1e-3)
                keep = [names.index(j) for j in arm_joints if j != "fx5_roll"]
                np.testing.assert_allclose(jr[:, keep], jf[:, keep], rtol=0, atol=1e-12)


class FromArmTests(unittest.TestCase):
    """WP2 §5.3 KinematicChain.from_arm reads arm.toml + kinematics.json (explicit name, synthetic S2R_ARMS_DIR)."""

    def test_from_arm_matches_from_dict(self):
        """§5.1/§5.3 from_arm uses [arm.joints].names, [arm.expert].arm_joints and [arm.kinematics]."""
        for name in FX_ARMS:
            with self.subTest(arm=name), sandbox() as box:
                write_fx_arm(box.arms, name)
                chain = KinematicChain.from_arm(name)
                self.assertEqual(chain.q_names, fx_joint_names(name))
                self.assertEqual(chain.chain_joints, tuple(fx_arm_table(name)["expert"]["arm_joints"]))
                rng = np.random.default_rng(11)
                for _ in range(10):
                    q = random_q(name, rng)
                    np.testing.assert_allclose(chain.fk(q), fx_reference_tcp(name, q), rtol=0, atol=1e-9)

    def test_from_arm_resolves_kinematics_file_relative_to_arm_dir(self):
        """§5.1 kinematics.file is relative to the arm folder."""
        name = "fx_arm6"
        table = fx_arm_table(name)
        table["kinematics"]["file"] = "generated/kin.json"
        with sandbox() as box:
            write_arm(box.arms, name, table, files={"generated/kin.json": json.dumps(fx_kinematics(name))})
            q = random_q(name, np.random.default_rng(12))
            np.testing.assert_allclose(KinematicChain.from_arm(name).fk(q), fx_reference_tcp(name, q), rtol=0, atol=1e-9)

    def test_from_arm_without_kinematics_table_is_config_error(self):
        """§1.5 an arm.toml without [arm.kinematics] cannot build a chain: ConfigError at from_arm."""
        name = "fx_arm5"
        table = fx_arm_table(name)
        del table["kinematics"]
        with sandbox() as box:
            write_arm(box.arms, name, table, kinematics=fx_kinematics(name))
            with self.assertRaises(ConfigError):
                KinematicChain.from_arm(name)

    def test_from_arm_with_malformed_file_is_schema_error(self):
        """§1.5 from_arm propagates kinematics.json validation as SchemaError."""
        name = "fx_arm5"
        data = fx_kinematics(name)
        data["schema_version"] = 7
        with sandbox() as box:
            write_arm(box.arms, name, fx_arm_table(name), kinematics=data)
            with self.assertRaises(SchemaError):
                KinematicChain.from_arm(name)


if __name__ == "__main__":
    unittest.main()
