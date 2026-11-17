"""WP4: regression tests for the v1 sidecar writer and resume check (contracts §9.1 last paragraph).

``sim.recording.episode_metadata.save_cube_trajectory`` and ``validate_cube_trajectories`` keep their
signatures and behaviour after WP4 turns them into thin wrappers (teleop still writes v1). These tests
are written against the pre-WP4 code and pass on it; they must keep passing unchanged.

The module is importable on the host without Isaac Sim / Isaac Lab, torch or lerobot (contracts §1.6);
``test_import_without_isaac`` pins that.
"""

import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sim.recording import episode_metadata
from sim.recording.episode_metadata import (SIDECAR_DIR, SIDECAR_FRAME, save_cube_trajectory,
                                            validate_cube_trajectories)
from v4_support import FIXTURES_DIR, REPO_ROOT

V1_ROOT_FIXTURE = FIXTURES_DIR / "v4_wp4" / "v1_root"
BLOCKED = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch")


def v1_poses(n, x0, y0, yaw0, lift_from):
    """Deterministic pick-like cube trajectory: rests, then rises 4 mm/frame and turns 0.05 rad/frame.

    Each pose is ``(x, y, z, qw, qx, qy, qz)`` in the base frame (v1 on-disk order). The committed
    fixture ``tests/fixtures/v4_wp4/v1_root`` was written by the pre-WP4 ``save_cube_trajectory``
    from exactly these inputs (see FIXTURE_EPISODES).
    """
    poses = []
    for t in range(n):
        k = max(0, t - lift_from)
        yaw = yaw0 + 0.05 * k
        poses.append((x0 + 0.0015 * k, y0 - 0.001 * k, 0.0125 + 0.004 * k,
                      math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
    return poses


#: (episode_index, poses, cube_size, fps) of the committed v1 fixture files.
FIXTURE_EPISODES = (
    (0, v1_poses(12, 0.2213456789, -0.0412345678, 0.3, 5), (0.025, 0.025, 0.025), 30),
    (1, v1_poses(8, 0.18, 0.052, -0.7, 3), [0.025, 0.025, 0.025], 30),
)


def sidecar_file(root, index):
    return Path(root) / "pick_place_meta" / f"episode_{index:06d}.json"


class ConstantsAndSignatureTests(unittest.TestCase):
    """WP4 §9.1 the public v1 names keep their values and signatures."""

    def test_constants(self):
        """§9.1/§1.1 SIDECAR_DIR and SIDECAR_FRAME are unchanged (v1 readers stay strict)."""
        self.assertEqual(SIDECAR_DIR, "pick_place_meta")
        self.assertEqual(SIDECAR_FRAME, "robot base link (base)")

    def test_signatures(self):
        """§9.1 save_cube_trajectory / validate_cube_trajectories keep their parameters."""
        self.assertEqual(list(inspect.signature(save_cube_trajectory).parameters),
                         ["root", "episode_index", "poses", "cube_size", "fps"])
        self.assertEqual(list(inspect.signature(validate_cube_trajectories).parameters),
                         ["root", "episode_lengths", "fps"])

    def test_import_without_isaac(self):
        """§1.6 the module imports with Isaac, torch and lerobot blocked."""
        code = ("import sys\n"
                f"for name in {BLOCKED!r}: sys.modules[name] = None\n"
                "import sim.recording.episode_metadata as m\n"
                "print(m.SIDECAR_DIR)\n")
        src = Path(episode_metadata.__file__).resolve().parents[2]      # the src tree under test
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "pick_place_meta")


class SaveCubeTrajectoryTests(unittest.TestCase):
    """WP4 §9.1 save_cube_trajectory writes the v1 format byte for byte as before."""

    def test_matches_committed_fixture_bytes(self):
        """§9.1 v1 output is unchanged: same file name and identical bytes as the pre-WP4 writer."""
        with tempfile.TemporaryDirectory() as tmp:
            for index, poses, size, fps in FIXTURE_EPISODES:
                with self.subTest(index=index):
                    self.assertIsNone(save_cube_trajectory(tmp, index, poses, size, fps))
                    written = sidecar_file(tmp, index)
                    self.assertEqual(written.read_bytes(), sidecar_file(V1_ROOT_FIXTURE, index).read_bytes())
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "pick_place_meta").iterdir()),
                             ["episode_000000.json", "episode_000001.json"])

    def test_payload_fields(self):
        """§9.1 v1 payload: 8 keys in writer order, positions/quaternions rounded to 6 decimals."""
        poses = [(0.1234567891, -0.2, 0.0300000049, 0.9999999, 0.0, 0.0, 0.00044721), (1, 2, 3, 1, 0, 0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            save_cube_trajectory(Path(tmp), 7, poses, (0.02, 0.03, 0.04), 15)
            payload = json.loads(sidecar_file(tmp, 7).read_text())
        self.assertEqual(list(payload), ["episode_index", "reference_frame", "units", "fps", "num_frames",
                                         "cube_size_m", "position", "orientation_wxyz"])
        self.assertEqual(payload["episode_index"], 7)
        self.assertEqual(payload["reference_frame"], "robot base link (base)")
        self.assertEqual(payload["units"], "m")
        self.assertEqual(payload["fps"], 15)
        self.assertEqual(payload["num_frames"], 2)
        self.assertEqual(payload["cube_size_m"], [0.02, 0.03, 0.04])
        self.assertEqual(payload["position"], [[0.123457, -0.2, 0.03], [1, 2, 3]])
        self.assertEqual(payload["orientation_wxyz"], [[1.0, 0.0, 0.0, 0.000447], [1, 0, 0, 0]])

    def test_overwrite_and_str_or_path_root(self):
        """§9.1 rewriting an index replaces the file; root may be str or Path; the folder is created."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nested" / "root"
            save_cube_trajectory(str(root), 0, [(0, 0, 0, 1, 0, 0, 0)] * 3, (0.02, 0.02, 0.02), 30)
            save_cube_trajectory(root, 0, [(0, 0, 0, 1, 0, 0, 0)] * 5, (0.02, 0.02, 0.02), 30)
            self.assertEqual(json.loads(sidecar_file(root, 0).read_text())["num_frames"], 5)
            self.assertEqual(os.listdir(root / "pick_place_meta"), ["episode_000000.json"])

    def test_failed_write_leaves_nothing(self):
        """§9.1 atomic: a non-finite pose raises ValueError and leaves no target or temp file."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_cube_trajectory(tmp, 0, [(0, 0, float("nan"), 1, 0, 0, 0)], (0.02, 0.02, 0.02), 30)
            self.assertEqual(os.listdir(Path(tmp) / "pick_place_meta"), [])

    def test_failed_replace_keeps_previous_file(self):
        """§9.1 atomic: if the final rename fails the previous file is intact and no temp file remains."""
        with tempfile.TemporaryDirectory() as tmp:
            save_cube_trajectory(tmp, 0, [(0, 0, 0, 1, 0, 0, 0)] * 2, (0.02, 0.02, 0.02), 30)
            before = sidecar_file(tmp, 0).read_bytes()
            with mock.patch("os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_cube_trajectory(tmp, 0, [(1, 1, 1, 1, 0, 0, 0)] * 4, (0.02, 0.02, 0.02), 30)
            self.assertEqual(sidecar_file(tmp, 0).read_bytes(), before)
            self.assertEqual(os.listdir(Path(tmp) / "pick_place_meta"), ["episode_000000.json"])


class ValidateCubeTrajectoriesTests(unittest.TestCase):
    """WP4 §9.1 validate_cube_trajectories accepts and rejects exactly as before."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for index, poses, size, fps in FIXTURE_EPISODES:
            save_cube_trajectory(self.root, index, poses, size, fps)
        self.lengths = [12, 8]

    def tearDown(self):
        self._tmp.cleanup()

    def edit(self, index, **changes):
        path = sidecar_file(self.root, index)
        payload = json.loads(path.read_text())
        payload.update(changes)
        path.write_text(json.dumps(payload))

    def test_accepts_matching(self):
        """§9.1 matching sidecars pass and return None."""
        self.assertIsNone(validate_cube_trajectories(self.root, self.lengths, 30))
        self.assertIsNone(validate_cube_trajectories(str(self.root), tuple(self.lengths), 30))

    def test_accepts_empty(self):
        """§9.1 no committed episodes and no sidecar folder is consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(validate_cube_trajectories(tmp, [], 30))

    def test_rejects_count_mismatch(self):
        """§9.1 a missing or extra sidecar is refused."""
        for lengths in ([12], [12, 8, 5], []):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                validate_cube_trajectories(self.root, lengths, 30)

    def test_rejects_timeline_and_frame_mismatch(self):
        """§9.1 wrong length, fps, episode_index, units or reference frame is refused."""
        cases = {"num_frames": 11, "fps": 15, "episode_index": 1, "units": "cm",
                 "reference_frame": "robot base link (fx5_base)"}
        for key, value in cases.items():
            with self.subTest(key=key):
                original = sidecar_file(self.root, 0).read_text()
                self.edit(0, **{key: value})
                with self.assertRaises(ValueError):
                    validate_cube_trajectories(self.root, self.lengths, 30)
                sidecar_file(self.root, 0).write_text(original)
        with self.assertRaises(ValueError):
            validate_cube_trajectories(self.root, self.lengths, 60)

    def test_rejects_bad_rows(self):
        """§9.1 short arrays, wrong row width and non-finite values are refused."""
        payload = json.loads(sidecar_file(self.root, 1).read_text())
        bad = {
            "short position": {"position": payload["position"][:-1]},
            "short orientation": {"orientation_wxyz": payload["orientation_wxyz"][1:]},
            "row width": {"position": [row[:2] for row in payload["position"]]},
            "nan": {"orientation_wxyz": [[float("nan"), 0, 0, 0]] + payload["orientation_wxyz"][1:]},
            "string": {"position": [["0", 0, 0]] + payload["position"][1:]},
        }
        original = sidecar_file(self.root, 1).read_text()
        for name, change in bad.items():
            with self.subTest(case=name):
                self.edit(1, **change)
                with self.assertRaises(ValueError):
                    validate_cube_trajectories(self.root, self.lengths, 30)
                sidecar_file(self.root, 1).write_text(original)

    def test_ignores_non_sidecar_files(self):
        """§9.1 only episode_*.json names count (temp files and other files are ignored)."""
        (self.root / "pick_place_meta" / "notes.txt").write_text("x")
        (self.root / "pick_place_meta" / "tmpabc.tmp").write_text("{")
        self.assertIsNone(validate_cube_trajectories(self.root, self.lengths, 30))


class ModuleSurfaceTests(unittest.TestCase):
    """WP4 §9.1 the v1 module keeps its public names."""

    def test_public_names_still_present(self):
        """§9.1 SIDECAR_DIR, SIDECAR_FRAME, save_cube_trajectory, validate_cube_trajectories remain."""
        for name in ("SIDECAR_DIR", "SIDECAR_FRAME", "save_cube_trajectory", "validate_cube_trajectories"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(episode_metadata, name))


if __name__ == "__main__":
    unittest.main()
