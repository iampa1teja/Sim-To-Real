"""Self-tests of the shared tests/v4_support helpers that need no v4 module (contracts §13). Must pass today."""

import dataclasses
import json
import os
import tempfile
import tomllib
import types
import unittest
from pathlib import Path

import numpy as np

from v4_support import (FIXTURES_DIR, REPO_ROOT, S2R_VARS, SRC_DIR, assert_frozen, clear_task_cache, copy_fixture,
                        copy_repo_folder, dump_toml, sandbox, temp_env)
from v4_support import fx_arms
from v4_support.fx_arms import (FX_ARMS, axis_rotation, frame_matrix, fx_arm_table, fx_home_q, fx_joint_names,
                                fx_kinematics, fx_lerobot_names, fx_limits, fx_reference_tcp, fx_setup,
                                matrix_to_quat_wxyz, quat_wxyz_to_matrix, random_q, reference_fk, reference_link_poses,
                                write_fx_arm)
from v4_support.stubs import ListWriter
from v4_support.tasks import (STUB_USDA, contract_color_task_table, contract_reach_task_table, write_contract_tasks,
                              write_scene, write_task)

CUSTOM_INIT = (
    "from core.tasks.base import Task, load_task_spec\n\n"
    "class _T(Task):\n"
    "    stages = ('a',)\n"
    "    def sample_start(self, rng):\n        return {}\n"
    "    def observe(self, state):\n        return {'a': True}\n\n"
    "TASK = _T(load_task_spec(NAME))\n"
)


class EnvHelperTests(unittest.TestCase):
    """v4_support.env (WP0 §13)."""

    def test_paths(self):
        """§13 path constants point into this repository."""
        self.assertTrue((REPO_ROOT / "pyproject.toml").is_file())
        self.assertTrue((SRC_DIR / "core").is_dir())
        self.assertEqual(FIXTURES_DIR, REPO_ROOT / "tests" / "fixtures")

    def test_temp_env_sets_and_restores(self):
        """§13 temp_env patches the S2R variables and restores them exactly."""
        before = {k: os.environ.get(k) for k in S2R_VARS}
        with temp_env(S2R_ARM="fx_x", S2R_TASKS_DIR=Path("/tmp/nowhere"), S2R_TASK=None):
            self.assertEqual(os.environ["S2R_ARM"], "fx_x")
            self.assertEqual(os.environ["S2R_TASKS_DIR"], "/tmp/nowhere")
            self.assertNotIn("S2R_TASK", os.environ)
        self.assertEqual({k: os.environ.get(k) for k in S2R_VARS}, before)

    def test_temp_env_restores_after_exception(self):
        """§13 temp_env restores on error."""
        before = os.environ.get("S2R_SCENE_DIR")
        with self.assertRaises(RuntimeError):
            with temp_env(S2R_SCENE_DIR="/x"):
                raise RuntimeError
        self.assertEqual(os.environ.get("S2R_SCENE_DIR"), before)

    def test_temp_env_rejects_other_names(self):
        """§13 temp_env only patches the five S2R variables."""
        with self.assertRaises(ValueError):
            with temp_env(HOME="/x"):
                pass

    def test_copy_fixture(self):
        """§13 copy_fixture keeps the relative path and fails loudly for unknown fixtures."""
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            (Path(src) / "arms" / "fx_demo").mkdir(parents=True)
            (Path(src) / "arms" / "fx_demo" / "arm.toml").write_text("[arm]\n")
            out = copy_fixture("arms/fx_demo", dst, fixtures_dir=src)
            self.assertEqual(out, Path(dst) / "arms" / "fx_demo")
            self.assertEqual((out / "arm.toml").read_text(), "[arm]\n")
            single = copy_fixture("arms/fx_demo/arm.toml", Path(dst) / "single", fixtures_dir=src)
            self.assertTrue(single.is_file())
            with self.assertRaises(FileNotFoundError):
                copy_fixture("arms/none", dst, fixtures_dir=src)

    def test_copy_repo_arm_loads_through_env(self):
        """§13 a copied repo arm (without USD binaries) loads via S2R_ARMS_DIR; the repo is untouched."""
        from core.arms.spec import load_arm_spec
        with tempfile.TemporaryDirectory() as dst:
            folder = copy_repo_folder("arms", "so101", dst)
            self.assertFalse(list(folder.rglob("*.usd")))
            with temp_env(S2R_ARMS_DIR=Path(dst) / "arms"):
                spec, arm_dir = load_arm_spec("so101")
            self.assertEqual(arm_dir, Path(dst) / "arms" / "so101")
            self.assertEqual(spec["joints"]["names"][0], "Rotation")

    def test_sandbox(self):
        """§13 sandbox creates arms/tasks/scene dirs, points the env vars there and cleans up."""
        before = {k: os.environ.get(k) for k in S2R_VARS}
        with sandbox(arm="fx_arm5", task="fx_t") as box:
            self.assertTrue(box.arms.is_dir() and box.tasks.is_dir() and box.scene.is_dir())
            self.assertEqual(os.environ["S2R_ARMS_DIR"], str(box.arms))
            self.assertEqual(os.environ["S2R_TASKS_DIR"], str(box.tasks))
            self.assertEqual(os.environ["S2R_SCENE_DIR"], str(box.scene))
            self.assertEqual(os.environ["S2R_ARM"], "fx_arm5")
            root = box.root
        self.assertFalse(root.exists())
        self.assertEqual({k: os.environ.get(k) for k in S2R_VARS}, before)


@dataclasses.dataclass(frozen=True)
class _Good:
    a: tuple
    m: types.MappingProxyType
    x: np.ndarray


@dataclasses.dataclass
class _Mutable:
    a: int


@dataclasses.dataclass(frozen=True)
class _Leaky:
    items: list


class AssertFrozenTests(unittest.TestCase):
    """v4_support.frozen (WP0 §13)."""

    def good(self, **kw):
        x = np.zeros(3)
        x.flags.writeable = False
        fields = dict(a=(1, (2, 3)), m=types.MappingProxyType({"k": (1.0,)}), x=x)
        fields.update(kw)
        return _Good(**fields)

    def test_accepts_frozen_values(self):
        """§13 assert_frozen accepts frozen dataclasses with tuples, proxies and read-only arrays."""
        assert_frozen(self.good())
        from core.tasks.base import Pose
        assert_frozen(Pose((0.0, 0.0, 0.0)))

    def test_rejects_mutable(self):
        """§13 assert_frozen rejects non-frozen classes, lists, dicts, writeable or float32 arrays."""
        bad = [_Mutable(1), _Leaky([1]), self.good(a=(1, [2])), self.good(m={"k": 1}),
               self.good(m=types.MappingProxyType({"k": [1]})), self.good(x=np.zeros(3)),
               self.good(x=np.zeros(3, np.float32)), 3]
        for value in bad:
            with self.subTest(value=repr(value)):
                with self.assertRaises(AssertionError):
                    assert_frozen(value)

    def test_shallow_mode(self):
        """§13 deep=False ignores items inside tuples / proxies."""
        assert_frozen(self.good(a=({"x": 1},)), deep=False)
        with self.assertRaises(AssertionError):
            assert_frozen(self.good(a=({"x": 1},)))


class TomlWriterTests(unittest.TestCase):
    """v4_support.toml_writer."""

    def test_round_trip(self):
        """dump_toml output reads back equal with tomllib."""
        data = {"task": {"name": "fx", "flag": True, "n": 3, "x": -0.25, "list": [1, 2], "nested_list": [[0.1, 0.2]],
                         "variants": {"teleop": "env_cfg:apply_teleop"}, "empty": {},
                         "failures": {"rules": [{"name": "a", "if": ["b"]}, {"name": "c"}]},
                         "deep": {"a": {"b": {"c": "d e"}}, "x": 1}, "q": "quote\"s"}}
        self.assertEqual(tomllib.loads(dump_toml(data)), data)

    def test_contract_examples_round_trip(self):
        """The transcribed §4.6 example tables survive dump_toml."""
        for table in (contract_reach_task_table(), contract_color_task_table(), fx_arm_table("fx_arm6")):
            self.assertEqual(tomllib.loads(dump_toml({"t": table}))["t"], table)


class FxArmTests(unittest.TestCase):
    """v4_support.fx_arms."""

    def test_kinematics_schema(self):
        """fx kinematics dicts follow contracts §5.2 (schema_version 1, revolute, unit basis axes, unit quats)."""
        for name in FX_ARMS:
            data = fx_kinematics(name)
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["arm"], name)
            self.assertEqual(set(data), {"schema_version", "arm", "base_link", "source", "joints"})
            for joint in data["joints"]:
                with self.subTest(arm=name, joint=joint["name"]):
                    self.assertEqual(joint["type"], "revolute")
                    self.assertEqual(sorted(map(abs, joint["axis"])), [0, 0, 1])
                    for key in ("parent_frame", "child_frame"):
                        self.assertAlmostEqual(np.linalg.norm(joint[key]["quat_wxyz"]), 1.0, places=12)
                    lo, hi = joint["limits_rad"]
                    self.assertLess(lo, hi)
            json.dumps(data)

    def test_names_and_limits_consistent(self):
        """fx joint names, LeRobot names, limits and home q have one entry per joint; home is in limits."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                joints = fx_joint_names(name)
                self.assertEqual(len(joints), len(fx_lerobot_names(name)))
                lower, upper = fx_limits(name)
                home = fx_home_q(name)
                self.assertEqual(lower.shape, (len(joints),))
                self.assertTrue(np.all((home >= lower) & (home <= upper)))
                table = fx_arm_table(name)
                self.assertEqual(table["joints"]["names"], list(joints))
                self.assertNotIn(table["gripper"]["joint"], table["expert"]["arm_joints"])
                self.assertEqual(table["lerobot"]["names"], list(fx_lerobot_names(name)))

    def test_no_so101_literals(self):
        """fx arms avoid the SO-101 literals of contracts §12."""
        literals = {"so101", "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw", "shoulder_pan",
                    "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper", "jaw", "moving_jaw_so101_v1"}
        def strings(value):
            if isinstance(value, dict):
                return set().union(*(strings(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(strings(v) for v in value), set())
            return {value} if isinstance(value, str) else set()

        for name in FX_ARMS:
            values = strings(fx_kinematics(name)) | strings(fx_arm_table(name))
            with self.subTest(arm=name):
                self.assertFalse(values & literals)

    def test_rotation_helpers(self):
        """Quaternion / matrix helpers are inverse and produce rotations."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            if quat[0] < 0:
                quat = -quat
            matrix = quat_wxyz_to_matrix(quat)
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=12)
            np.testing.assert_allclose(matrix_to_quat_wxyz(matrix), quat, atol=1e-12)
        np.testing.assert_allclose(axis_rotation([0, 0, 1], np.pi / 2)[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-15)

    def test_reference_fk_simple_chain(self):
        """reference_link_poses implements T_parent·F(parent)·R(axis,q)·F(child)⁻¹."""
        data = {"base_link": "b", "joints": [
            {"name": "j", "parent": "b", "child": "c", "axis": [0, 0, 1],
             "parent_frame": {"position": [1.0, 0.0, 0.0], "quat_wxyz": [1, 0, 0, 0]},
             "child_frame": {"position": [0.0, 0.0, 0.5], "quat_wxyz": [1, 0, 0, 0]}}]}
        pose = reference_fk(data, [np.pi / 2], "c")
        expected = frame_matrix([1, 0, 0], [1, 0, 0, 0]) @ axis_rotation([0, 0, 1], np.pi / 2) \
            @ np.linalg.inv(frame_matrix([0, 0, 0.5], [1, 0, 0, 0]))
        np.testing.assert_allclose(pose, expected, atol=1e-15)
        np.testing.assert_allclose(pose[:3, 3], [1.0, 0.0, -0.5], atol=1e-15)

    def test_reference_fk_fx_arms(self):
        """Reference FK of the fx arms gives proper transforms and a TCP above the base at home."""
        rng = np.random.default_rng(1)
        for name in FX_ARMS:
            with self.subTest(arm=name):
                tcp = fx_reference_tcp(name, fx_home_q(name))
                self.assertGreater(tcp[2, 3], 0.0)
                for _ in range(10):
                    q = random_q(name, rng)
                    poses = reference_link_poses(fx_kinematics(name), q)
                    self.assertEqual(len(poses), len(fx_joint_names(name)) + 1)
                    for matrix in poses.values():
                        np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-12)
                        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])

    def test_gripper_joint_does_not_move_tcp(self):
        """The gripper joint is not on the ee chain, so it does not move the TCP."""
        for name in FX_ARMS:
            q = fx_home_q(name)
            moved = q.copy()
            moved[-1] += 0.3
            np.testing.assert_allclose(fx_reference_tcp(name, q), fx_reference_tcp(name, moved), atol=1e-15)

    def test_write_fx_arm_loads_with_existing_spec_reader(self):
        """write_fx_arm writes an arm folder that core.arms.spec.load_arm_spec reads."""
        from core.arms.spec import load_arm_spec
        with sandbox() as box:
            for name in FX_ARMS:
                folder = write_fx_arm(box.arms, name)
                spec, arm_dir = load_arm_spec(name)
                self.assertEqual(arm_dir, folder)
                self.assertEqual(spec["joints"]["names"], list(fx_joint_names(name)))
                self.assertEqual(spec["gripper"]["opening_table"], fx_arms.FX_OPENING_TABLE)
                self.assertEqual(json.loads((folder / "kinematics.json").read_text()), fx_kinematics(name))

    def test_setup_block_and_scene(self):
        """fx_setup gives an arms block that write_scene / core.scenes.load_setup round-trip."""
        from core.scenes import load_setup
        with sandbox() as box:
            write_scene(box.scene, fx_setup())
            setup = load_setup()
        for name in FX_ARMS:
            block = setup["arms"][name]
            self.assertEqual(list(block["sim_joint_mapping"]), list(fx_lerobot_names(name)))
            for key, limits in block["runtime_joint_limits"].items():
                mapping = block["sim_joint_mapping"][key]
                self.assertLess(mapping["joint_min"], limits["joint_min"])
                self.assertLess(limits["joint_min"], limits["joint_max"])
                self.assertLess(limits["joint_max"], mapping["joint_max"])


class TaskFolderTests(unittest.TestCase):
    """v4_support.tasks."""

    def test_write_task_loads_with_existing_loader(self):
        """write_task output is read by core.tasks.base.load_task_spec and core.tasks.loader.load_task."""
        from core.tasks.base import load_task_spec
        from core.tasks.loader import load_task
        with sandbox() as box:
            write_task(box.tasks, "fx_selftest", {"variants": {"teleop": "env_cfg:apply_teleop"},
                                                  "objects": {"cube": {"size": [0.02, 0.02, 0.02]}}},
                       init_py=CUSTOM_INIT.replace("NAME", repr("fx_selftest")))
            spec = load_task_spec("fx_selftest")
            self.assertEqual(spec.name, "fx_selftest")
            self.assertEqual(spec.objects["cube"].size, (0.02, 0.02, 0.02))
            task = load_task("fx_selftest")
            self.assertEqual(task.stages, ("a",))
            clear_task_cache("fx_selftest")
            import core.tasks.loader as loader
            self.assertNotIn("fx_selftest", loader._loaded)

    def test_contract_tasks_load_as_specs(self):
        """The §4.6 example tasks load with the existing TaskSpec reader (new tables kept in raw)."""
        from core.tasks.base import load_task_spec
        with sandbox() as box:
            reach, color = write_contract_tasks(box.tasks)
            self.assertEqual((color / "usd" / "Small-Box-Physics.usda").read_text(), STUB_USDA)
            self.assertIn("from core.tasks.generic import GenericTask", (reach / "__init__.py").read_text())
            spec = load_task_spec("fx_contract_color")
            self.assertEqual(set(spec.objects), {"small_box"})
            self.assertEqual(spec.raw["object_sets"]["cubes"]["count_max"], 6)
            self.assertEqual(spec.raw["failures"]["rules"][0], {"name": "wrong_object_placed", "if": ["distractor_in_box"]})
            reach_spec = load_task_spec("fx_contract_reach")
            self.assertEqual(reach_spec.raw["stages"]["order"], ["reached", "success"])


class ListWriterTests(unittest.TestCase):
    """v4_support.stubs.ListWriter."""

    def test_semantics(self):
        """ListWriter commits, discards and records call order."""
        writer = ListWriter()
        writer.discard_episode()
        with self.assertRaises(RuntimeError):
            writer.save_episode()
        action = np.ones(3)
        writer.add_frame(action, np.zeros(3), {"cam": np.zeros((2, 2, 3), np.uint8)}, "t", None)
        action[0] = 5.0
        self.assertEqual(writer.save_episode(), 0)
        self.assertEqual(writer.episodes[0][0]["action_raw"][0], 1.0)
        writer.add_frame(np.ones(3), np.zeros(3), {}, "t", None)
        writer.discard_episode()
        self.assertEqual(writer.episode_count(), 1)
        self.assertEqual(writer.episode_lengths(), [1])
        writer.finalize()
        self.assertTrue(writer.finalized)
        self.assertEqual(writer.calls[:3], ["discard_episode", "save_episode", "add_frame"])


if __name__ == "__main__":
    unittest.main()
