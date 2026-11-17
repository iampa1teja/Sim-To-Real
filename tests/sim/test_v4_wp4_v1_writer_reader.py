"""WP4: a v1 sidecar written by the unchanged teleop writer loads through task_meta.load_episode (contracts §9.1).

Isaac-free: ``sim.recording.episode_metadata`` imports only the standard library (and, after WP4,
``core.datasets.task_meta``).
"""

import tempfile
import unittest

from core.datasets.task_meta import load_episode, validate_sidecar
from sim.recording.episode_metadata import save_cube_trajectory


class WriterToReaderTests(unittest.TestCase):
    """WP4 §9.1 a v1 file written today by teleop loads through load_episode."""

    def test_save_cube_trajectory_then_load(self):
        """§9.1 save_cube_trajectory output (v1) → load_episode → validated v2 with legacy_v1 true."""
        poses = [(0.2, 0.01 * t, 0.0125, 1.0, 0.0, 0.0, 0.0) for t in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            save_cube_trajectory(tmp, 3, poses, (0.02, 0.02, 0.02), 15)
            out = load_episode(tmp, 3, task="fx_pick")
        self.assertTrue(out["legacy_v1"])
        self.assertEqual((out["episode_index"], out["num_frames"], out["fps"]), (3, 6, 15))
        self.assertEqual(out["objects"]["blue_cube"]["position"], [[0.2, round(0.01 * t, 6), 0.0125] for t in range(6)])
        self.assertEqual(out["objects"]["blue_cube"]["orientation_wxyz"], [[1.0, 0.0, 0.0, 0.0]] * 6)
        validate_sidecar(out, num_frames=6, fps=15)


if __name__ == "__main__":
    unittest.main()
