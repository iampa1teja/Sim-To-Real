"""Recorded robot start pose for pick-place evaluation (CPU only).

python -m unittest tests.test_robot_start -v

The eval must start the arm where the demonstrations started (their first-frame
observation.state), not at the scene's calibrated reset pose, or the policy begins
off-distribution.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _fakes import CURRENT, PKG, REAL_SETUP_JSON, FakeStage, install_stubs, load_module  # noqa: E402

def _parquet_available():
    try:  # may be installed but blocked, e.g. by Windows App Control
        importlib.import_module("pyarrow.parquet")
        return True
    except ImportError:
        return False


HAVE_PYARROW = _parquet_available()
MAPPING = json.loads(REAL_SETUP_JSON.read_text()).get("sim_joint_mapping", {})
PATCHER = E = R = TERMS = None


def setUpModule():
    global PATCHER, E, R, TERMS
    mdp_pkg = SimpleNamespace(__path__=[str(PKG / "mdp")])
    TERMS = SimpleNamespace(reset_pick_place_state=MagicMock(name="reset_pick_place_state"))
    tasks_cfg = SimpleNamespace(WHITE_BOX_SIZE=(0.135, 0.085, 0.045))
    PATCHER = install_stubs({
        "sim_to_real_so101.mdp": mdp_pkg,
        "sim_to_real_so101.mdp.terms": TERMS,
        "sim_to_real_so101.tasks": SimpleNamespace(__path__=[str(PKG / "tasks")]),
        "sim_to_real_so101.tasks.pick_place_env_cfg": tasks_cfg,
    })
    E = load_module("sim_to_real_so101.utils.pick_place_eval", "source/sim_to_real_so101/utils/pick_place_eval.py")
    R = load_module("sim_to_real_so101.mdp.resets", "source/sim_to_real_so101/mdp/resets.py")


def tearDownModule():
    PATCHER.stop()


def reference_inverse(radians, mapping):
    """LeRobotSO101Interface.get_raw_actions_from_radians, re-derived independently."""
    out = []
    for i, name in enumerate(E.SO101_JOINTS):
        low, high = (mapping[name]["joint_min"], mapping[name]["joint_max"]) if name in mapping \
            else E.SO101_USD_LIMITS_DEG[name]
        normalized = (math.degrees(radians[i]) - low) / (high - low)
        out.append(normalized * 100 if name == "gripper" else normalized * 200 - 100)
    return np.array(out)


class ConversionTests(unittest.TestCase):
    def test_known_values(self):
        rad = E.dataset_state_to_radians([0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(np.degrees(rad), [0, 0, -5, 0, 0, -10], atol=1e-9)
        rad = E.dataset_state_to_radians([100, -100, 100, 100, -100, 100])
        np.testing.assert_allclose(np.degrees(rad), [110, -100, 90, 95, -160, 100], atol=1e-9)

    def test_calibrated_pan_mapping_is_used_and_clamped(self):
        if "shoulder_pan" not in MAPPING:
            self.skipTest("real_setup.json has no shoulder_pan mapping")
        low, high = MAPPING["shoulder_pan"]["joint_min"], MAPPING["shoulder_pan"]["joint_max"]
        pan = np.degrees(E.dataset_state_to_radians([0, 0, 0, 0, 0, 0], MAPPING)[0])
        self.assertAlmostEqual(pan, (low + high) / 2, places=6)
        self.assertAlmostEqual(np.degrees(E.dataset_state_to_radians([-100, 0, 0, 0, 0, 0], MAPPING)[0]),
                               max(low, -110), places=6)

    def test_inverse_of_the_recorder(self):
        rng = np.random.default_rng(0)
        low = np.radians([E.SO101_USD_LIMITS_DEG[n][0] for n in E.SO101_JOINTS])
        high = np.radians([E.SO101_USD_LIMITS_DEG[n][1] for n in E.SO101_JOINTS])
        for _ in range(500):
            q = rng.uniform(low, high)
            state = reference_inverse(q, MAPPING)            # what the recorder stored
            np.testing.assert_allclose(E.dataset_state_to_radians(state, MAPPING), q, atol=1e-9)

    def test_rejects_bad_input(self):
        for bad in ([0] * 5, [0, 0, 0, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                E.dataset_state_to_radians(bad)
        with self.assertRaises(ValueError):
            E.dataset_state_to_radians([0] * 6, {"elbow": {"joint_min": 0, "joint_max": 1}})

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "LeRobot not installed; run in the teleop container")
    def test_matches_lerobot_interface(self):
        spec = importlib.util.spec_from_file_location("iface", PKG / "utils" / "lerobot_interface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        iface = module.LeRobotSO101Interface(device="cpu", port="", id="t", cameras={}, fps=30,
                                             kind="follower", joint_mapping=MAPPING)
        rng = np.random.default_rng(1)
        for _ in range(200):
            state = np.r_[rng.uniform(-100, 100, 5), rng.uniform(0, 100)]
            expected = iface.get_mapped_actions_vectorized(torch.tensor(state, dtype=torch.float32)).numpy()
            np.testing.assert_allclose(E.dataset_state_to_radians(state, MAPPING), expected, atol=1e-5)


def write_dataset(root, episodes):
    """episodes: {episode_index: (cube_xy, [state_frame0, state_frame1, ...])}"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = root / "pick_place_meta"
    meta.mkdir(parents=True)
    rows = {"episode_index": [], "frame_index": [], "observation.state": []}
    for episode, (xy, states) in episodes.items():
        n = len(states)
        (meta / f"episode_{episode:06d}.json").write_text(json.dumps({
            "episode_index": episode, "units": "m", "reference_frame": "robot base link (base)",
            "position": [[xy[0], xy[1], 0.01]] * n, "orientation_wxyz": [[1, 0, 0, 0]] * n}))
        for frame, state in enumerate(states):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["observation.state"].append([float(v) for v in state])
    (root / "data" / "chunk-000").mkdir(parents=True)
    # Split across two files to check that every data file is read.
    half = len(rows["episode_index"]) // 2
    for i, (a, b) in enumerate(((0, half), (half, None))):
        pq.write_table(pa.table({k: v[a:b] for k, v in rows.items()}),
                       root / "data" / "chunk-000" / f"file-{i:03d}.parquet")
    return meta


STATE = {0: [-3, -99, 99, 76, -2, 3], 2: [5, -98, 98.5, 75.5, 8, 7], 5: [-9, -99.5, 99.8, 76.2, -12, 1]}


@unittest.skipUnless(HAVE_PYARROW, "pyarrow parquet unavailable; run in the teleop container")
class RecordedRobotStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = write_dataset(self.root, {
            0: ((0.20, -0.05), [STATE[0], [0] * 6, [1] * 6]),
            2: ((0.30, 0.10), [STATE[2], [9] * 6]),
            5: ((0.25, 0.00), [STATE[5], [7] * 6, [7] * 6, [7] * 6]),
        })

    def test_episode_ids_follow_filenames(self):
        self.assertEqual(E.start_episode_ids(self.meta), [0, 2, 5])

    def test_first_frames_by_episode(self):
        states = E.load_robot_starts(self.root, [0, 2, 5])
        for got, episode in zip(states, (0, 2, 5)):
            np.testing.assert_allclose(got, STATE[episode])

    def test_missing_episode(self):
        with self.assertRaisesRegex(ValueError, r"\[7\]"):
            E.load_robot_starts(self.root, [0, 7])

    def test_starts_pair_cube_and_arm_by_episode(self):
        sampler = E.RecordedStarts(self.meta, seed=0, robot_states_root=self.root)
        for index, episode in enumerate((0, 2, 5)):
            np.testing.assert_allclose(sampler.starts[index][0][:2], {0: (0.20, -0.05), 2: (0.30, 0.10),
                                                                     5: (0.25, 0.00)}[episode])
            np.testing.assert_allclose(sampler.robot_state(index), STATE[episode])
        np.testing.assert_allclose(sampler.robot_state(-1), np.mean([STATE[e] for e in (0, 2, 5)], axis=0))



def write_sidecars(root, episodes):
    meta = root / "pick_place_meta"
    meta.mkdir(parents=True)
    for episode, xy in episodes.items():
        (meta / f"episode_{episode:06d}.json").write_text(json.dumps({
            "episode_index": episode, "units": "m", "reference_frame": "robot base link (base)",
            "position": [[xy[0], xy[1], 0.01]], "orientation_wxyz": [[1, 0, 0, 0]]}))
    return meta


class ResetTermTests(unittest.TestCase):
    """reset_cube_from_recorded_starts against a fake scene; states supplied without parquet."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = write_sidecars(self.root, {0: (0.20, -0.05), 2: (0.30, 0.10), 5: (0.25, 0.00)})
        self.loads = []

        def fake_load(root, ids):
            self.loads.append((Path(root), list(ids)))
            return [np.asarray(STATE[i], dtype=float) for i in ids]

        original = E.load_robot_starts
        E.load_robot_starts = fake_load
        self.addCleanup(setattr, E, "load_robot_starts", original)

    def test_without_root(self):
        sampler = E.RecordedStarts(self.meta, seed=0)
        with self.assertRaises(ValueError):
            sampler.robot_state(0)

    def test_states_come_from_the_dataset_that_owns_the_sidecars(self):
        CURRENT.stage = FakeStage()
        env, _ = self.make_env()
        R.reset_cube_from_recorded_starts(env, torch.tensor([0]), SimpleNamespace(name="blue_cube"),
                                          str(self.meta), mode="cycle", seed=0, reset_robot=True)
        self.assertEqual(self.loads, [(self.root, [0, 2, 5])])

    def test_every_start_gets_its_own_episode_state(self):
        CURRENT.stage = FakeStage()
        env, robot = self.make_env()
        seen = set()
        for _ in range(3):  # cycle mode covers all three starts once
            R.reset_cube_from_recorded_starts(env, torch.tensor([0]), SimpleNamespace(name="blue_cube"),
                                              str(self.meta), mode="cycle", seed=0, reset_robot=True)
            episode = E.start_episode_ids(self.meta)[env._pick_place_start_index[0]]
            seen.add(episode)
            written = robot.write_joint_state_to_sim.call_args.args[0].numpy()[0]
            np.testing.assert_allclose(written, E.dataset_state_to_radians(STATE[episode], MAPPING), atol=1e-6)
        self.assertEqual(seen, {0, 2, 5})

    # ── the eval reset term, against a fake scene ──────────────────────────────

    def make_env(self):
        robot = MagicMock(name="robot")
        robot.find_bodies.return_value = ([0], ["base"])
        robot.find_joints.return_value = ([0, 1, 2, 3, 4, 5], list(E.SO101_USD_JOINTS))
        robot.data.body_pos_w = torch.zeros(1, 1, 3)
        robot.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        robot.data.joint_pos = torch.zeros(1, 6)
        cube = MagicMock(name="cube")
        cube.cfg.spawn.size = (0.02, 0.02, 0.02)
        cube.data.root_pos_w = torch.zeros(1, 3)
        box = MagicMock(name="box")
        box.data.root_pos_w = torch.tensor([[5.0, 5.0, 0.0]])   # far away: no overlap
        box.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        class Scene(dict):
            env_prim_paths = ["/World/envs/env_0"]

        scene = Scene(blue_cube=cube, robot=robot, white_box=box, contact_grasp=MagicMock())
        env = SimpleNamespace(num_envs=1, device="cpu", scene=scene, sim=MagicMock(),
                              cfg=SimpleNamespace(seed=0, sim_joint_mapping=MAPPING))
        return env, robot

    def test_reset_puts_arm_at_recorded_start(self):
        CURRENT.stage = FakeStage()
        env, robot = self.make_env()
        R.reset_cube_from_recorded_starts(env, torch.tensor([0]), SimpleNamespace(name="blue_cube"),
                                          str(self.meta), mode="cycle", seed=0, reset_robot=True)
        episode = E.start_episode_ids(self.meta)[env._pick_place_start_index[0]]
        expected = E.dataset_state_to_radians(STATE[episode], MAPPING)
        robot.find_joints.assert_called_with(list(E.SO101_USD_JOINTS), preserve_order=True)
        call = robot.write_joint_state_to_sim.call_args
        np.testing.assert_allclose(call.args[0].numpy()[0], expected, atol=1e-6)
        self.assertEqual(float(call.args[1].abs().sum()), 0.0)
        self.assertEqual(call.kwargs["joint_ids"], [0, 1, 2, 3, 4, 5])
        target = robot.set_joint_position_target.call_args
        np.testing.assert_allclose(target.args[0].numpy()[0], expected, atol=1e-6)

    def test_reset_robot_off_leaves_arm_alone(self):
        CURRENT.stage = FakeStage()
        env, robot = self.make_env()
        R.reset_cube_from_recorded_starts(env, torch.tensor([0]), SimpleNamespace(name="blue_cube"),
                                          str(self.meta), mode="cycle", seed=0, reset_robot=False)
        robot.write_joint_state_to_sim.assert_not_called()
        robot.set_joint_position_target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
