"""Step 4 - utils/replay_source.py: read recorded episodes + cube sidecars.

CPU only: python -m unittest tests.test_replay_source -v

Contract under test
- SourceEpisode(index, length, actions (N,6), states (N,6), task: str, poses_base (N,7))  dataclass
    poses_base rows are x, y, z, qw, qx, qy, qz in the robot base-link frame (from the sidecar)
- load_sidecar(root, index) -> np.ndarray (N,7)
    reads <root>/pick_place_meta/episode_{index:06d}.json (format written by episode_metadata.save_cube_trajectory)
    ValueError (message names the file) if: missing, units != "m",
    reference_frame != "robot base link (base)", episode_index mismatch, num_frames != rows,
    non-finite values, or a quaternion that is not unit length (tolerance 1e-3)
- load_source(root, repo_id, exclude, dataset=None) -> list[SourceEpisode]
    `dataset` is injectable for tests; when None, a stock LeRobotDataset(repo_id, root=root) is created
    (import LeRobot lazily inside the function, so this module imports without LeRobot).
    Reads per-episode rows via dataset.meta.episodes (dicts with episode_index, length, tasks,
    dataset_from_index, dataset_to_index) and the global columns dataset.hf_dataset["action"] /
    ["observation.state"] sliced [from:to].
    Skips `exclude`; ValueError if an excluded index does not exist or a sidecar length != episode length.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _fakes import block_imports, load_module  # noqa: E402

S = None
TASK = "Pick up the blue cube and place it in the white box"


def setUpModule():
    global S
    with block_imports("pxr", "isaaclab", "isaacsim", "omni", "lerobot"):
        S = load_module("replay_source_under_test", "source/sim_to_real_so101/utils/replay_source.py")


def sidecar(index, n, **override):
    payload = {
        "episode_index": index, "reference_frame": "robot base link (base)", "units": "m", "fps": 30,
        "num_frames": n, "cube_size_m": [0.02, 0.02, 0.02],
        "position": [[0.2 + 0.001 * i, -0.05, 0.01] for i in range(n)],
        "orientation_wxyz": [[1.0, 0.0, 0.0, 0.0] for _ in range(n)],
    }
    payload.update(override)
    return payload


class FakeDataset:
    """Minimal LeRobotDataset shape: meta.episodes rows + global hf_dataset columns."""

    def __init__(self, lengths):
        rows, start = [], 0
        for i, n in enumerate(lengths):
            rows.append({"episode_index": i, "length": n, "tasks": [TASK],
                         "dataset_from_index": start, "dataset_to_index": start + n})
            start += n
        total = start
        self.meta = SimpleNamespace(episodes=rows, total_episodes=len(lengths))
        self.hf_dataset = {
            "action": [np.full(6, k, dtype=np.float32) for k in range(total)],
            "observation.state": [np.full(6, -k, dtype=np.float32) for k in range(total)],
        }


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "pick_place_meta").mkdir()

    def write(self, index, payload):
        (self.root / "pick_place_meta" / f"episode_{index:06d}.json").write_text(json.dumps(payload))


class SidecarTests(Base):
    def test_reads_positions_and_quaternions(self):
        self.write(3, sidecar(3, 5))
        poses = S.load_sidecar(self.root, 3)
        self.assertEqual(poses.shape, (5, 7))
        np.testing.assert_allclose(poses[4], [0.204, -0.05, 0.01, 1, 0, 0, 0])

    def test_rejects_bad_sidecars(self):
        bad = {
            "units": sidecar(0, 3, units="cm"),
            "frame": sidecar(0, 3, reference_frame="world"),
            "index": sidecar(0, 3, episode_index=1),
            "count": sidecar(0, 3, num_frames=4),
            "nan": sidecar(0, 3, position=[[0, float("nan"), 0]] * 3),
            "quat": sidecar(0, 3, orientation_wxyz=[[0.5, 0, 0, 0]] * 3),
        }
        for name, payload in bad.items():
            with self.subTest(name):
                self.write(0, payload)
                with self.assertRaisesRegex(ValueError, "episode_000000"):
                    S.load_sidecar(self.root, 0)

    def test_missing_sidecar(self):
        with self.assertRaisesRegex(ValueError, "episode_000007"):
            S.load_sidecar(self.root, 7)


class LoadSourceTests(Base):
    def setUp(self):
        super().setUp()
        self.lengths = [4, 6, 5]
        for i, n in enumerate(self.lengths):
            self.write(i, sidecar(i, n))
        self.dataset = FakeDataset(self.lengths)

    def test_loads_all_episodes(self):
        episodes = S.load_source(self.root, "local/x", set(), dataset=self.dataset)
        self.assertEqual([e.index for e in episodes], [0, 1, 2])
        self.assertEqual([e.length for e in episodes], self.lengths)
        second = episodes[1]
        self.assertEqual(second.actions.shape, (6, 6))
        self.assertEqual(second.states.shape, (6, 6))
        self.assertEqual(second.poses_base.shape, (6, 7))
        # global frames 4..9 belong to episode 1
        np.testing.assert_array_equal(second.actions[:, 0], np.arange(4, 10, dtype=np.float32))
        np.testing.assert_array_equal(second.states[:, 0], -np.arange(4, 10, dtype=np.float32))
        self.assertEqual(second.task, TASK)

    def test_values_are_exact_copies(self):
        episodes = S.load_source(self.root, "local/x", set(), dataset=self.dataset)
        self.assertEqual(episodes[0].actions.dtype, np.float32)

    def test_exclude(self):
        episodes = S.load_source(self.root, "local/x", {0, 2}, dataset=self.dataset)
        self.assertEqual([e.index for e in episodes], [1])
        with self.assertRaises(ValueError):
            S.load_source(self.root, "local/x", {9}, dataset=self.dataset)

    def test_sidecar_length_mismatch(self):
        self.write(1, sidecar(1, 5))  # episode 1 has 6 frames
        with self.assertRaisesRegex(ValueError, "episode_000001"):
            S.load_source(self.root, "local/x", set(), dataset=self.dataset)

    def test_excluded_episode_sidecar_not_required(self):
        (self.root / "pick_place_meta" / "episode_000002.json").unlink()
        episodes = S.load_source(self.root, "local/x", {2}, dataset=self.dataset)
        self.assertEqual([e.index for e in episodes], [0, 1])


if __name__ == "__main__":
    unittest.main()
