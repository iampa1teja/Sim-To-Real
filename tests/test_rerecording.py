"""CPU integration tests against the installed, pinned LeRobot writer."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder, SynchronizedLeRobotRecorders
from sim_to_real_so101.utils.rerecording import RerecordingSession, recover_replacement, transaction_path
from sim_to_real_so101.utils.recording_web_gui import RecordingWebGui


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class ReplacementTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        recorders = {name: LeRobotRecorder("pick", f"test/{name}", str(self.base / name), 30, "cpu",
                    cameras={"front": {"height": 64, "width": 64}}, batch_encoding_size=100,
                    image_writer_threads_per_camera=1) for name in ("sim", "real")}
        group = SynchronizedLeRobotRecorders(recorders)
        group.init_datasets()
        self.session = RerecordingSession(group)
        self.addCleanup(lambda: self.session.finalize(encode=False))
        for index, length in enumerate((3, 4, 3)):
            self.record(length, index + 1)
            self.session.save_with_sidecar(self.poses(length), (0.02,) * 3, 30)
        for recorder in group.recorders.values():
            recorder.dataset._batch_save_episode_video(0)

    def poses(self, n):
        return [[0.1 * i, 0, 0.02, 1, 0, 0, 0] for i in range(n)]

    def record(self, n, value):
        for _ in range(n):
            self.session.add_frames({name: {"action": np.full(6, value, dtype=np.float32),
                "observation.state": np.full(6, -value, dtype=np.float32), "task": "pick",
                "observation.images.front": np.full((64, 64, 3), value * 20, dtype=np.uint8)}
                for name in ("sim", "real")})

    def test_replace_middle_and_append(self):
        originals = {name: hashes(self.base / name) for name in ("sim", "real")}
        old_videos = {name: [p.read_bytes() for p in sorted((self.base / name / "videos").rglob("*.mp4"))]
                      for name in originals}
        self.session.begin(1)
        self.record(5, 8)
        for name in originals:
            self.assertEqual(hashes(self.base / name), originals[name])
        self.assertEqual(self.session.save_with_sidecar(self.poses(5), (0.02,) * 3, 30), 1)
        self.assertEqual(self.session.saved_episodes, 3)
        for name in originals:
            root = self.base / name
            data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
            self.assertEqual(data.episode_index.tolist(), [0] * 3 + [1] * 5 + [2] * 3)
            self.assertEqual(data["index"].tolist(), list(range(11)))
            self.assertEqual([a[0] for a in data.action], [1] * 3 + [8] * 5 + [3] * 3)
            stats = json.loads((root / "meta/stats.json").read_text())
            self.assertAlmostEqual(stats["action"]["mean"][0], 52 / 11, places=5)
            videos = [p.read_bytes() for p in sorted((root / "videos").rglob("*.mp4"))]
            self.assertEqual(videos[0], old_videos[name][0])
            self.assertEqual(videos[2], old_videos[name][2])
            self.assertNotEqual(videos[1], old_videos[name][1])
            dataset = self.session.recorders[name].dataset
            dataset.video_backend = "pyav"
            for frame_index in range(11):
                frame = dataset[frame_index]
                self.assertEqual(tuple(frame["observation.images.front"].shape), (3, 64, 64))
                self.assertEqual(int(frame["index"]), frame_index)
            backup, = self.base.glob(f"{name}.before-rerecord-*")
            self.assertEqual(hashes(backup), originals[name])
        sidecar = json.loads((self.base / "sim/pick_place_meta/episode_000001.json").read_text())
        self.assertEqual((sidecar["episode_index"], sidecar["num_frames"]), (1, 5))
        self.record(2, 4)
        self.assertEqual(self.session.save_with_sidecar(self.poses(2), (0.02,) * 3, 30), 3)

    def test_discard_and_invalid_indices_preserve_originals(self):
        before = {name: hashes(self.base / name) for name in ("sim", "real")}
        for invalid in (-1, 3, True, "1"):
            with self.assertRaises(ValueError):
                self.session.begin(invalid)
        self.session.begin(0)
        self.assertEqual(self.session.episode_index, 0)
        self.record(2, 8)
        self.session.cancel_episode()
        self.assertEqual(self.session.episode_index, 3)
        for name in before:
            self.assertEqual(hashes(self.base / name), before[name])

    def test_publication_failure_restores_both_originals(self):
        before = {name: hashes(self.base / name) for name in ("sim", "real")}
        self.session.begin(2)
        self.record(2, 8)
        real_rename = Path.rename

        def fail_second_stage(path, destination):
            if path.name.startswith(".real.rerecord-") and Path(destination) == self.base / "real":
                raise OSError("injected paired publication failure")
            return real_rename(path, destination)

        with patch.object(Path, "rename", fail_second_stage):
            with self.assertRaisesRegex(OSError, "injected"):
                self.session.save_with_sidecar(self.poses(2), (0.02,) * 3, 30)
        for name in before:
            self.assertEqual(hashes(self.base / name), before[name])

    def test_exit_before_recording_removes_private_copies(self):
        before = {name: hashes(self.base / name) for name in ("sim", "real")}
        self.session.begin(0)
        stages = [rec.dataset_root for rec in self.session.recorders.values()]
        self.session.finalize(encode=False)
        self.assertTrue(all(not path.exists() for path in stages))
        for name in before:
            self.assertEqual(hashes(self.base / name), before[name])


class RecoveryTest(unittest.TestCase):
    def test_restart_rolls_back_partial_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sim, real = base / "sim", base / "real"
            backup = base / "sim.backup"
            sim.mkdir(); real.mkdir(); backup.mkdir()
            (sim / "value").write_text("new")
            (backup / "value").write_text("original")
            journal = transaction_path(sim)
            journal.write_text(json.dumps([{"root": str(sim), "backup": str(backup)},
                                          {"root": str(real), "backup": str(base / "absent")}]))
            recover_replacement(sim)
            self.assertEqual((sim / "value").read_text(), "original")
            self.assertTrue(real.exists())
            self.assertFalse(journal.exists())


class GuiTest(unittest.TestCase):
    def test_http_validation(self):
        gui = RecordingWebGui([], port=0)
        self.addCleanup(gui.destroy)
        gui.set_state(False, True, "ready", saved_episodes=3)
        for value, status in (("1", 400), (True, 400), (-1, 409), (3, 409)):
            request = Request(gui.url + "command", json.dumps({"cmd": "rerecord", "episode": value}).encode(),
                              {"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, status)
        with urlopen(Request(gui.url + "command", b'{"cmd":"rerecord","episode":0}',
                             {"Content-Type": "application/json"})) as response:
            self.assertEqual(response.status, 204)
        self.assertEqual(gui.consume_requests()["rerecord_episode"], 0)

    def test_zero_and_unavailable_states(self):
        gui = RecordingWebGui([], port=0)
        self.addCleanup(gui.destroy)
        gui.set_state(False, True, "ready", saved_episodes=3)
        for invalid in (-1, 3, True, "0", None):
            self.assertFalse(gui._command("rerecord", episode=invalid))
        self.assertTrue(gui._command("rerecord", episode=0))
        self.assertEqual(gui.consume_requests()["rerecord_episode"], 0)
        for state in ({"active": True}, {"encoding": True}, {"pending_videos": 1}, {"rerecord_episode": 0}):
            params = dict(active=False, can_start=True, status="ready", saved_episodes=3)
            params.update(state)
            gui.set_state(**params)
            self.assertFalse(gui._command("rerecord", episode=1))
        gui.set_state(False, True, "ready", saved_episodes=3, rerecord_episode=0)
        self.assertTrue(gui._command("discard"))


if __name__ == "__main__":
    unittest.main()
