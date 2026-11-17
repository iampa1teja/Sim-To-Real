"""WP2: torch-free RawMapping, JointMapping numpy methods and per-arm setup blocks (contracts §5.4, §1.6, §14.3).

The torch ``JointMapping`` is the reference: RawMapping must reproduce its formulas. JointMapping is
built directly from fx tables (its constructor), so no SO-101 data is involved except in the
explicit SO-101 regression class at the end.
"""

import copy
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from core.arms.joint_limits import joint_name_pairs, runtime_joint_limits
from core.arms.mapping import JointMapping
from core.arms.raw_mapping import RawMapping
from core.arms.spec import load_arm_spec
from core.scenes import load_setup
from v4_support import assert_frozen, sandbox
from v4_support.fx_arms import FX_ARMS, fx_joint_names, fx_lerobot_names, fx_setup, fx_setup_block, write_fx_arm
from v4_support.tasks import write_scene

RAD_TOL = 1e-5     # float32 torch vs float64 numpy, radians
RAW_TOL = 1e-4     # float32 torch vs float64 numpy, raw units (full scale 100-200)


def torch_mapping(names, mapping, runtime, gripper):
    """JointMapping built exactly as JointMapping.from_setup does (mapping = runtime overlaid by mapping)."""
    effective = dict(runtime)
    effective.update(mapping)

    def tensor(source, key):
        return torch.tensor([float(source[n][key]) for n in names], dtype=torch.float32)

    return JointMapping(names=tuple(names), mins=tensor(effective, "joint_min"), maxs=tensor(effective, "joint_max"),
                        clamp_mins=tensor(runtime, "joint_min"), clamp_maxs=tensor(runtime, "joint_max"),
                        gripper_mask=torch.tensor([n == gripper for n in names]))


def fx_tables(name):
    block = fx_setup_block(name)
    names = fx_lerobot_names(name)
    return names, block["sim_joint_mapping"], block["runtime_joint_limits"], names[-1]


def raw_samples(n, size, rng):
    """Raw vectors spanning beyond the valid ranges (arm −100…100, gripper 0…100) so clamping is exercised."""
    raw = rng.uniform(-115.0, 115.0, size=(size, n))
    raw[:, -1] = rng.uniform(-15.0, 115.0, size=size)
    return raw


class RawMappingTablesTests(unittest.TestCase):
    """WP2 §5.4 RawMapping.from_tables and its arrays."""

    def test_fields(self):
        """§5.4/§1.3 names tuple; degree arrays float64 read-only in names order; gripper_mask marks the gripper."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                m = RawMapping.from_tables(names, mapping, runtime, gripper)
                self.assertEqual(m.names, tuple(names))
                np.testing.assert_array_equal(m.mins_deg, [mapping[n]["joint_min"] for n in names])
                np.testing.assert_array_equal(m.maxs_deg, [mapping[n]["joint_max"] for n in names])
                np.testing.assert_array_equal(m.clamp_mins_deg, [runtime[n]["joint_min"] for n in names])
                np.testing.assert_array_equal(m.clamp_maxs_deg, [runtime[n]["joint_max"] for n in names])
                np.testing.assert_array_equal(m.gripper_mask, [n == gripper for n in names])
                self.assertEqual(m.gripper_mask.dtype, np.bool_)
                assert_frozen(m)
                np.testing.assert_allclose(m.lower_rad, np.deg2rad(m.clamp_mins_deg), rtol=0, atol=1e-15)
                np.testing.assert_allclose(m.upper_rad, np.deg2rad(m.clamp_maxs_deg), rtol=0, atol=1e-15)

    def test_partial_mapping_falls_back_to_runtime(self):
        """§5.4 same rule as JointMapping.from_setup: names missing from the mapping use the runtime bounds."""
        names, mapping, runtime, gripper = fx_tables("fx_arm6")
        partial = {names[0]: mapping[names[0]]}
        m = RawMapping.from_tables(names, partial, runtime, gripper)
        self.assertEqual(m.mins_deg[0], mapping[names[0]]["joint_min"])
        np.testing.assert_array_equal(m.mins_deg[1:], [runtime[n]["joint_min"] for n in names[1:]])

    def test_invalid_tables_raise_value_error(self):
        """§5.4 runtime must name exactly the joints; unknown mapping names, bad bounds or gripper raise ValueError."""
        names, mapping, runtime, gripper = fx_tables("fx_arm6")
        missing = {k: v for k, v in runtime.items() if k != names[2]}
        extra = dict(mapping, nojoint={"joint_min": 0.0, "joint_max": 1.0})
        reversed_ = copy.deepcopy(runtime)
        reversed_[names[1]] = {"joint_min": 10.0, "joint_max": -10.0}
        nonfinite = copy.deepcopy(mapping)
        nonfinite[names[0]]["joint_max"] = float("inf")
        cases = {
            "runtime_missing_joint": (names, mapping, missing, gripper),
            "mapping_unknown_name": (names, extra, runtime, gripper),
            "runtime_reversed": (names, mapping, reversed_, gripper),
            "mapping_nonfinite": (names, nonfinite, runtime, gripper),
            "gripper_not_in_names": (names, mapping, runtime, "nogripper"),
        }
        for label, args in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    RawMapping.from_tables(*args)


class RawMappingFormulaTests(unittest.TestCase):
    """WP2 §5.4 RawMapping reproduces the torch JointMapping formulas."""

    def test_agrees_with_torch_joint_mapping(self):
        """§5.4 to_radians (with clamping) and to_raw (no clamp) equal JointMapping's on the same tables, batched."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                m = RawMapping.from_tables(names, mapping, runtime, gripper)
                jm = torch_mapping(names, mapping, runtime, gripper)
                rng = np.random.default_rng(0)
                raw = raw_samples(len(names), 300, rng)
                rad = m.to_radians(raw)
                self.assertEqual(rad.shape, raw.shape)
                self.assertEqual(rad.dtype, np.float64)
                np.testing.assert_allclose(rad, jm.to_radians(raw).double().numpy(), rtol=0, atol=RAD_TOL)
                angles = rng.uniform(m.lower_rad - 0.4, m.upper_rad + 0.4, size=(300, len(names)))
                np.testing.assert_allclose(m.to_raw(angles), jm.to_raw(angles).double().numpy(), rtol=0, atol=RAW_TOL)
                for k in range(5):     # single vectors too
                    np.testing.assert_allclose(m.to_radians(raw[k]), jm.to_radians(raw[k]).double().numpy(),
                                               rtol=0, atol=RAD_TOL)

    def test_to_radians_clamps_to_runtime(self):
        """§5.4 to_radians clamps to the runtime limits: out-of-range raw lands exactly on lower_rad / upper_rad."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                m = RawMapping.from_tables(names, mapping, runtime, gripper)
                rad = m.to_radians(raw_samples(len(names), 500, np.random.default_rng(1)))
                self.assertTrue(np.all(rad >= m.lower_rad) and np.all(rad <= m.upper_rad))
                high = np.full(len(names), 1000.0)
                np.testing.assert_allclose(m.to_radians(high), m.upper_rad, rtol=0, atol=1e-12)
                np.testing.assert_allclose(m.to_radians(-high), m.lower_rad, rtol=0, atol=1e-12)

    def test_to_raw_does_not_clamp(self):
        """§5.4 to_raw has no clamp: angles beyond the mapping give raw values beyond ±100 / 0…100."""
        names, mapping, runtime, gripper = fx_tables("fx_arm6")
        m = RawMapping.from_tables(names, mapping, runtime, gripper)
        above = np.deg2rad(m.maxs_deg) + 0.2
        raw = m.to_raw(above)
        self.assertTrue(np.all(raw[~m.gripper_mask] > 100.0))
        self.assertTrue(np.all(raw[m.gripper_mask] > 100.0))
        self.assertTrue(np.all(m.to_raw(np.deg2rad(m.mins_deg) - 0.2)[~m.gripper_mask] < -100.0))

    def test_clamp_radians(self):
        """§5.4 clamp_radians clips to [lower_rad, upper_rad] elementwise (batched)."""
        names, mapping, runtime, gripper = fx_tables("fx_arm5")
        m = RawMapping.from_tables(names, mapping, runtime, gripper)
        rad = np.random.default_rng(2).uniform(-4, 4, size=(100, len(names)))
        np.testing.assert_array_equal(m.clamp_radians(rad), np.clip(rad, m.lower_rad, m.upper_rad))

    def test_round_trips(self):
        """§5.4 to_raw(to_radians(x)) == x within 1e-6 inside runtime limits; to_radians(to_raw(r)) == r within 1e-9."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                m = RawMapping.from_tables(names, mapping, runtime, gripper)
                rng = np.random.default_rng(3)
                r = rng.uniform(m.lower_rad, m.upper_rad, size=(500, len(names)))
                np.testing.assert_allclose(m.to_radians(m.to_raw(r)), r, rtol=0, atol=1e-9)
                x = m.to_raw(r)
                np.testing.assert_allclose(m.to_raw(m.to_radians(x)), x, rtol=0, atol=1e-6)

    def test_rejects_wrong_width(self):
        """§5.4 the last axis must have length J (ValueError otherwise)."""
        names, mapping, runtime, gripper = fx_tables("fx_arm6")
        m = RawMapping.from_tables(names, mapping, runtime, gripper)
        for fn in (m.to_radians, m.to_raw, m.clamp_radians):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(np.zeros(len(names) + 1))


class JointMappingNumpyTests(unittest.TestCase):
    """WP2 §5.4 JointMapping.to_raw_np / clamp_radians_np / as_raw_mapping agree with the torch methods (float32)."""

    def test_numpy_methods_agree_with_torch(self):
        """§5.4 to 1e-5 rad (radians) / 1e-4 raw units (float32 rounding of a full-scale 100 value)."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                jm = torch_mapping(names, mapping, runtime, gripper)
                rng = np.random.default_rng(4)
                rad = rng.uniform(-3.5, 3.5, size=(200, len(names)))
                out = jm.to_raw_np(rad)
                self.assertIsInstance(out, np.ndarray)
                np.testing.assert_allclose(out, jm.to_raw(rad).double().numpy(), rtol=0, atol=RAW_TOL)
                lo, hi = jm.clamp_mins.double().numpy(), jm.clamp_maxs.double().numpy()
                clamped = jm.clamp_radians_np(rad)
                self.assertIsInstance(clamped, np.ndarray)
                np.testing.assert_allclose(clamped, np.clip(rad, np.deg2rad(lo), np.deg2rad(hi)), rtol=0, atol=1e-7)

    def test_as_raw_mapping(self):
        """§5.4 as_raw_mapping() is a RawMapping with the same names, bounds (as float64) and gripper mask."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                names, mapping, runtime, gripper = fx_tables(name)
                jm = torch_mapping(names, mapping, runtime, gripper)
                m = jm.as_raw_mapping()
                self.assertIsInstance(m, RawMapping)
                self.assertEqual(m.names, tuple(names))
                for field, tensor in (("mins_deg", jm.mins), ("maxs_deg", jm.maxs), ("clamp_mins_deg", jm.clamp_mins),
                                      ("clamp_maxs_deg", jm.clamp_maxs)):
                    np.testing.assert_array_equal(getattr(m, field), tensor.double().numpy())
                np.testing.assert_array_equal(m.gripper_mask, jm.gripper_mask.numpy())
                assert_frozen(m)
                raw = raw_samples(len(names), 100, np.random.default_rng(5))
                np.testing.assert_allclose(m.to_radians(raw), jm.to_radians(raw).double().numpy(), rtol=0, atol=RAD_TOL)
                np.testing.assert_allclose(m.to_radians(raw), jm.to_radians_np(raw), rtol=0, atol=1e-12)


class SetupBlockTests(unittest.TestCase):
    """WP2 §5.4 setup["arms"][arm] blocks for RawMapping.from_setup and runtime_joint_limits; names from load_arm_spec(arm)."""

    def test_from_setup_uses_arm_block(self):
        """§5.4 with an "arms" block containing the arm, its mapping and runtime limits are used."""
        with sandbox() as box:
            for name in FX_ARMS:
                write_fx_arm(box.arms, name)
            setup = fx_setup()
            for name in FX_ARMS:
                with self.subTest(arm=name):
                    names, mapping, runtime, gripper = fx_tables(name)
                    got = RawMapping.from_setup(setup, arm=name)
                    want = RawMapping.from_tables(names, mapping, runtime, gripper)
                    self.assertEqual(got.names, want.names)
                    for field in ("mins_deg", "maxs_deg", "clamp_mins_deg", "clamp_maxs_deg", "gripper_mask"):
                        np.testing.assert_array_equal(getattr(got, field), getattr(want, field))

    def test_from_setup_reads_scene_when_setup_is_none(self):
        """§5.4 setup=None loads the selected scene's real_setup.json (S2R_SCENE_DIR)."""
        with sandbox() as box:
            write_fx_arm(box.arms, "fx_arm5")
            write_scene(box.scene, fx_setup())
            got = RawMapping.from_setup(arm="fx_arm5")
            names, mapping, runtime, gripper = fx_tables("fx_arm5")
            np.testing.assert_array_equal(got.clamp_maxs_deg, [runtime[n]["joint_max"] for n in names])
            np.testing.assert_array_equal(got.maxs_deg, [mapping[n]["joint_max"] for n in names])

    def test_top_level_keys_without_arm_block(self):
        """§5.4 without an "arms" entry for the arm, today's top-level keys are used."""
        with sandbox() as box:
            write_fx_arm(box.arms, "fx_arm6")
            block = fx_setup_block("fx_arm6")
            setup = {"sim_joint_mapping": block["sim_joint_mapping"], "runtime_joint_limits": block["runtime_joint_limits"],
                     "arms": {"fx_arm5": fx_setup_block("fx_arm5")}}
            got = RawMapping.from_setup(setup, arm="fx_arm6")
            np.testing.assert_array_equal(got.mins_deg, [block["sim_joint_mapping"][n]["joint_min"]
                                                         for n in fx_lerobot_names("fx_arm6")])
            self.assertEqual(runtime_joint_limits(setup, arm="fx_arm6"), block["runtime_joint_limits"])
            no_arms = {k: v for k, v in setup.items() if k != "arms"}
            self.assertEqual(runtime_joint_limits(no_arms, arm="fx_arm6"), block["runtime_joint_limits"])

    def test_runtime_joint_limits_per_arm(self):
        """§5.4 runtime_joint_limits(setup, arm=...) returns the arm's limits keyed and ordered by its LeRobot names."""
        with sandbox() as box:
            for name in FX_ARMS:
                write_fx_arm(box.arms, name)
            setup = fx_setup()
            for name in FX_ARMS:
                with self.subTest(arm=name):
                    got = runtime_joint_limits(setup, arm=name)
                    self.assertEqual(list(got), list(fx_lerobot_names(name)))
                    self.assertEqual(got, fx_setup_block(name)["runtime_joint_limits"])
            with sandbox() as inner:
                write_fx_arm(inner.arms, "fx_arm6")
                write_scene(inner.scene, fx_setup())
                self.assertEqual(runtime_joint_limits(arm="fx_arm6"), fx_setup_block("fx_arm6")["runtime_joint_limits"])

    def test_runtime_joint_limits_validation(self):
        """§5.4 limits must name exactly the arm's joints, be finite and increase (ValueError)."""
        name = "fx_arm6"
        names = fx_lerobot_names(name)
        with sandbox() as box:
            write_fx_arm(box.arms, name)
            bad_sets = {}
            s = fx_setup([name])
            del s["arms"][name]["runtime_joint_limits"][names[0]]
            bad_sets["missing"] = s
            s = fx_setup([name])
            s["arms"][name]["runtime_joint_limits"][names[1]] = {"joint_min": 5.0, "joint_max": 5.0}
            bad_sets["not_increasing"] = s
            s = fx_setup([name])
            s["arms"][name]["runtime_joint_limits"][names[2]]["joint_min"] = float("nan")
            bad_sets["nan"] = s
            s = fx_setup([name])
            s["arms"][name]["runtime_joint_limits"]["extra"] = {"joint_min": 0.0, "joint_max": 1.0}
            bad_sets["extra"] = s
            for label, setup in bad_sets.items():
                with self.subTest(case=label):
                    with self.assertRaises(ValueError):
                        runtime_joint_limits(setup, arm=name)

    def test_joint_name_pairs_for_fx_arm(self):
        """§1.2 joint_name_pairs pairs [arm.lerobot].names[i] with [arm.joints].names[i]."""
        with sandbox() as box:
            write_fx_arm(box.arms, "fx_arm5")
            spec, _ = load_arm_spec("fx_arm5")
            self.assertEqual(joint_name_pairs(spec), tuple(zip(fx_lerobot_names("fx_arm5"), fx_joint_names("fx_arm5"))))


class ImportHygieneTests(unittest.TestCase):
    """WP2 §1.6/§14.3 the new modules import without torch and read no arm, scene or task at import time."""

    def test_import_without_torch_or_arm_data(self):
        """§1.6 importing raw_mapping / kinematics / gripper / joint_limits pulls no torch and needs no arms/ folder."""
        code = textwrap.dedent("""
            import sys
            import core.arms.raw_mapping, core.arms.kinematics, core.arms.gripper, core.arms.joint_limits, core.arms.spec
            bad = sorted(m for m in sys.modules if m.split(".")[0] in
                         ("torch", "isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot"))
            assert not bad, bad
            print("ok")
        """)
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, S2R_ARMS_DIR=empty, S2R_SCENE_DIR=empty, S2R_TASKS_DIR=empty, S2R_ARM="fx_absent",
                       PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


class SO101RegressionTests(unittest.TestCase):
    """WP2 regression: the repo arm and scene (no per-arm block) behave exactly as before."""

    def test_runtime_joint_limits_default(self):
        """§5.4 runtime_joint_limits() with no arguments still returns the scene's top-level runtime limits."""
        setup = load_setup()
        names = tuple(load_arm_spec()[0]["lerobot"]["names"])
        got = runtime_joint_limits()
        self.assertEqual(list(got), list(names))
        self.assertEqual(got, {n: setup["runtime_joint_limits"][n] for n in names})
        self.assertEqual(runtime_joint_limits(setup), got)

    def test_raw_mapping_from_setup_matches_joint_mapping(self):
        """§5.4 RawMapping.from_setup(setup) == JointMapping.from_setup(mapping=sim_joint_mapping), as used by the sim."""
        setup = load_setup()
        jm = JointMapping.from_setup(SimpleNamespace(mapping=setup["sim_joint_mapping"]))
        m = RawMapping.from_setup(setup)
        self.assertEqual(m.names, tuple(jm.names))
        for field, tensor in (("mins_deg", jm.mins), ("maxs_deg", jm.maxs), ("clamp_mins_deg", jm.clamp_mins),
                              ("clamp_maxs_deg", jm.clamp_maxs)):
            np.testing.assert_allclose(getattr(m, field), tensor.double().numpy(), rtol=0, atol=1e-4)
        np.testing.assert_array_equal(m.gripper_mask, jm.gripper_mask.numpy())
        raw = raw_samples(len(m.names), 200, np.random.default_rng(6))
        np.testing.assert_allclose(m.to_radians(raw), jm.to_radians(raw).double().numpy(), rtol=0, atol=RAD_TOL)

    def test_joint_mapping_from_setup_unchanged(self):
        """Regression: JointMapping.from_setup() with no selection still uses the runtime limits as its mapping."""
        jm = JointMapping.from_setup()
        runtime = runtime_joint_limits()
        np.testing.assert_allclose(jm.mins.double().numpy(), [runtime[n]["joint_min"] for n in jm.names], atol=1e-4)
        np.testing.assert_allclose(jm.clamp_maxs.double().numpy(), [runtime[n]["joint_max"] for n in jm.names], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
