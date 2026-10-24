"""real_rollout script: argument checks, calibration refusal, shadow vs motion, outcome prompts, cleanup."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
spec = importlib.util.spec_from_file_location("real_rollout_script", ROOT / "source/sim_to_real_so101/scripts/real_rollout.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class FakeRobot:
    def __init__(self, calibration_file, calibrated=True):
        self.calibration_fpath, self.is_calibrated, self.sent = calibration_file, calibrated, []

    def get_observation(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return {**dict.fromkeys(JOINTS, 0.0), "room": frame, "wrist": frame}

    def send_action(self, action):
        self.sent.append(action)


class FakeInterface:
    SO101_JOINT_ORDER = JOINTS

    def __init__(self, calibration_file, calibrated=True):
        self.id, self.cameras = "rpd_robot", {"external": {}, "gripper": {}}
        self.robot = FakeRobot(calibration_file, calibrated)
        self.events = []

    def init_device(self):
        self.events.append("init")

    def connect(self, calibrate=True):
        self.events.append(f"connect(calibrate={calibrate})")

    def disconnect(self):
        self.events.append("disconnect")


class FakePolicy:
    inference_calls = 0

    def __init__(self, error=None):
        self.error, self.closed, self.resets = error, False, 0

    def connect(self):
        pass

    def reset(self):
        self.resets += 1

    def get_raw_action(self, state, images):
        if self.error:
            raise self.error
        self.inference_calls += 1
        return dict.fromkeys(JOINTS, 1.0)

    def close_timing(self):
        self.closed = True


class ScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.calibration = self.dir / "rpd_robot.json"
        self.calibration.write_text("{}", encoding="utf-8")

    def run_script(self, extra=(), answers=(), calibrated=True, calibration_exists=True, policy=None):
        if not calibration_exists:
            self.calibration.unlink()
        interface = FakeInterface(self.calibration, calibrated)
        policy = policy or FakePolicy()
        answers = iter(answers)
        argv = ["--policy", "demo", "--lang_instruction", "pick it", "--follower_port", "/dev/ttyACM1",
                "--follower_id", "rpd_robot", "--camera_gripper", "/dev/g", "--camera_external", "/dev/e",
                "--fps", "10", "--episode_seconds", "0.3", "--output", str(self.dir / "out"), *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = script.main(argv, interface_factory=lambda args: interface,
                                   policy_factory=lambda args, iface: policy, input_fn=lambda prompt: next(answers))
            except SystemExit as exit_:
                code = exit_.code
        return code, interface, policy, out.getvalue() + err.getvalue()

    def report(self):
        return json.loads((self.dir / "out" / "report.json").read_text(encoding="utf-8"))

    def test_default_is_shadow_mode_and_never_moves_the_arm(self):
        code, interface, policy, text = self.run_script(["--no_prompt"])
        self.assertEqual(code, 0)
        self.assertIn("SHADOW MODE", text)
        self.assertEqual(interface.robot.sent, [])
        self.assertEqual(policy.inference_calls, 3)
        self.assertEqual(self.report()["episodes"][0]["outcome"], "unlabeled")
        self.assertFalse(self.report()["motion"])
        self.assertEqual(interface.events, ["init", "connect(calibrate=False)", "disconnect"])
        self.assertTrue(policy.closed)

    def test_motion_sends_commands_and_asks_for_the_outcome(self):
        code, interface, _, text = self.run_script(["--enable-motion"], answers=["", "y"])
        self.assertEqual(code, 0)
        self.assertIn("MOTION ENABLED", text)
        self.assertEqual(len(interface.robot.sent), 3)
        episode = self.report()["episodes"][0]
        self.assertEqual((episode["status"], episode["outcome"]), ("completed", "success"))
        self.assertEqual(self.report()["summary"]["successes"], 1)
        self.assertEqual(len((self.dir / "out" / "episode_000.jsonl").read_text(encoding="utf-8").splitlines()), 3)

    def test_missing_calibration_file_refuses_before_connecting(self):
        code, interface, _, _ = self.run_script(["--no_prompt"], calibration_exists=False)
        self.assertIn("lerobot-calibrate", str(code))
        self.assertNotIn("connect(calibrate=False)", interface.events)
        self.assertEqual(interface.events[-1], "disconnect")

    def test_calibration_mismatch_on_the_arm_refuses_to_run(self):
        code, interface, policy, _ = self.run_script(["--no_prompt", "--enable-motion"], calibrated=False)
        self.assertIn("does not match", str(code))
        self.assertEqual(interface.robot.sent, [])
        self.assertEqual(policy.inference_calls, 0)

    def test_policy_failure_stops_and_returns_nonzero_with_a_report(self):
        code, interface, _, _ = self.run_script(["--no_prompt", "--enable-motion", "--episodes", "3"],
                                                policy=FakePolicy(error=TimeoutError("no reply")))
        self.assertEqual(code, 1)
        self.assertEqual([e["status"] for e in self.report()["episodes"]], ["policy_error"])
        self.assertEqual(interface.robot.sent, [])

    def test_ctrl_c_still_writes_the_report_and_disconnects(self):
        code, interface, policy, text = self.run_script(["--enable-motion"], answers=["", "y"],
                                                        policy=FakePolicy(error=KeyboardInterrupt()))
        self.assertEqual(code, 1)
        self.assertIn("holds its last goal", text)
        self.assertEqual(interface.events[-1], "disconnect")
        self.assertTrue(policy.closed)

    def test_argument_errors_are_reported_before_any_hardware(self):
        for extra, message in ((["--rename_map", '{"external": "room"}'], "no policy name"),
                               (["--episodes", "0"], "must be positive"),
                               (["--policy_option", "nokey"], "KEY=VALUE")):
            with self.subTest(extra=extra):
                code, interface, _, text = self.run_script(extra)
                self.assertEqual(code, 2)
                self.assertIn(message, text)
                self.assertEqual(interface.events, [])

    def test_missing_port_names_the_variable(self):
        code, _, _, text = self.run_script(["--follower_port", ""])
        self.assertEqual(code, 2)
        self.assertIn("ROBOT_PORT", text)


if __name__ == "__main__":
    unittest.main()
