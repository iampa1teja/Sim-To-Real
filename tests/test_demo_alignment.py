"""CPU checks for recorder coordinates and frame pairing in the demo review."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


alignment = module("demo_alignment", REPO / "scripts/check_pick_place_demo_alignment.py")
metadata = module("demo_metadata", REPO / "source/sim_to_real_so101/utils/episode_metadata.py")


class DemoAlignmentTests(unittest.TestCase):
    def sidecar(self, root):
        # Write through the actual recorder sidecar helper, with scalar-first
        # quaternion and pre-command state frame pairing.
        yaw = np.deg2rad(30)
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        poses = [[0.2, 0.1, z, *q] for z in (0.02, 0.025, 0.031, 0.06)]
        metadata.save_cube_trajectory(root, 0, poses, [0.02] * 3, 30)
        return json.loads((root / "pick_place_meta/episode_000000.json").read_text())

    def test_wxyz_projection_and_square_symmetry(self):
        yaw = np.deg2rad(30)
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.assertAlmostEqual(alignment.cube_face_yaw(q), 30)
        self.assertAlmostEqual(alignment.cube_face_yaw(-np.array(q)), 30)
        self.assertEqual(alignment.square_angle(120 - 30), 0)
        # Local X is vertical for a 90-degree Y rotation; avoid Euler-yaw
        # singularity by using a horizontal face axis.
        self.assertAlmostEqual(alignment.square_angle(alignment.cube_face_yaw(
            [np.sqrt(0.5), 0, np.sqrt(0.5), 0])), 0)

    def test_first_lift_uses_calibrated_state_not_action_or_degrees(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.sidecar(Path(directory))
        states = np.zeros((4, 6))
        states[2, 0] = 20  # normalized -100..100, not degrees
        states[2, 4] = 5
        mapping = {"shoulder_pan": {"joint_min": -116.279296875,
                                    "joint_max": 119.794921875}}
        result = alignment.review_episode(row, states, mapping, 0, 0.01)
        expected_pan = -116.279296875 + 0.6 * (119.794921875 + 116.279296875)
        self.assertEqual(result["frame"], 2)  # first lift, not the highest frame
        self.assertAlmostEqual(result["time_s"], 2 / 30)
        self.assertAlmostEqual(result["pan_deg"], expected_pan)
        self.assertAlmostEqual(result["roll_deg"], 8)
        self.assertAlmostEqual(result["face_minus_pan_deg"],
                               alignment.square_angle(result["cube_yaw_deg"] - expected_pan))

    def test_no_lift_and_invalid_timelines(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.sidecar(Path(directory))
        states = np.zeros((4, 6))
        self.assertIsNone(alignment.review_episode(row, states, {}, 0, 1))
        with self.assertRaisesRegex(ValueError, "matching frames"):
            alignment.review_episode(row, states[:3], {}, 0, 0.01)
        row["orientation_wxyz"][2] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "Zero cube quaternion"):
            alignment.review_episode(row, states, {}, 0, 0.01)

    def make_dataset(self, root):
        row = self.sidecar(root)
        (root / "meta").mkdir()
        info = {"total_episodes": 1, "fps": 30, "features": {"observation.state": {
            "names": [name + ".pos" for name in alignment.eval_utils.SO101_JOINTS]}}}
        (root / "meta/info.json").write_text(json.dumps(info))
        (root / "data/chunk-000").mkdir(parents=True)
        table = pa.table({"episode_index": [0] * 4, "frame_index": [3, 1, 0, 2],
                          "observation.state": [[0.0] * 6] * 4})
        path = root / "data/chunk-000/file-000.parquet"
        pq.write_table(table, path)
        return path, row

    def test_dataset_review_pairs_frame_ids_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            before = {str(p): hashlib.sha256(p.read_bytes()).digest()
                      for p in root.rglob("*") if p.is_file()}
            results, _ = alignment.review_dataset(root)
            self.assertEqual(results[0][1]["frame"], 2)
            after = {str(p): hashlib.sha256(p.read_bytes()).digest()
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_reject_duplicate_frames_and_wrong_joint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.make_dataset(root)
            table = pq.read_table(path)
            pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)
            with self.assertRaisesRegex(ValueError, "Duplicate episode/frame"):
                alignment.review_dataset(root)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["features"]["observation.state"]["names"].reverse()
            info_path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "joint names/order"):
                alignment.review_dataset(root)

    def test_reject_empty_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"total_episodes": 0}))
            with self.assertRaisesRegex(ValueError, "at least one saved episode"):
                alignment.review_dataset(root)


if __name__ == "__main__":
    unittest.main()
