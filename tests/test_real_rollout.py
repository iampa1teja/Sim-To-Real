"""Real-arm rollout loop: shadow mode, clamping, start check, failures, stopping and the report."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_rollout", ROOT / "source/sim_to_real_so101/utils/real_rollout.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAMERAS = ["room", "wrist"]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeRobot:
    def __init__(self, pose=(0, 0, 0, 0, 0, 10), fail_observation_at=None, fail_send=False):
        self.pose, self.sent, self.reads = list(pose), [], 0
        self.fail_observation_at, self.fail_send = fail_observation_at, fail_send

    def get_observation(self):
        self.reads += 1
        if self.reads == self.fail_observation_at:
            raise OSError("serial bus lost")
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return {**dict(zip(JOINTS, self.pose)), **{camera: frame for camera in CAMERAS}}

    def send_action(self, action):
        if self.fail_send:
            raise OSError("write failed")
        self.sent.append(action)
        return action


class FakePolicy:
    def __init__(self, action=None, fail_at=None, error=TimeoutError("server timeout")):
        self.action = action if action is not None else dict(zip(JOINTS, (1, 2, 3, 4, 5, 20)))
        self.fail_at, self.error, self.calls = fail_at, error, []

    def get_raw_action(self, state, images):
        self.calls.append((np.array(state), dict(images)))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise self.error
        return self.action


def run(robot, policy, **overrides):
    clock = FakeClock()
    settings = rr.RolloutSettings(**{"fps": 10, "episode_seconds": 0.5, "enable_motion": True, **overrides})
    logged = []
    result = rr.run_episode(robot, policy, JOINTS, CAMERAS, settings, log=logged.append, clock=clock, sleep=clock.sleep)
    return result, logged, clock


class RolloutLoopTest(unittest.TestCase):
    def test_runs_for_the_time_limit_at_the_requested_rate(self):
        robot = FakeRobot()
        result, logged, clock = run(robot, FakePolicy())
        self.assertEqual((result.status, result.steps), ("completed", 5))
        self.assertEqual(len(robot.sent), 5)
        self.assertTrue(all(abs(s - 0.1) < 1e-9 for s in clock.sleeps))
        self.assertEqual(logged[0]["state"], [0, 0, 0, 0, 0, 10])

    def test_shadow_mode_reads_and_queries_but_never_sends(self):
        robot, policy = FakeRobot(), FakePolicy()
        result, logged, _ = run(robot, policy, enable_motion=False)
        self.assertEqual((result.status, result.steps), ("completed", 5))
        self.assertEqual(robot.sent, [])
        self.assertEqual(len(policy.calls), 5)
        self.assertTrue(all(entry["sent"] is False for entry in logged))

    def test_policy_receives_joint_order_state_and_named_cameras(self):
        policy = FakePolicy()
        run(FakeRobot(pose=(5, -5, 1, 2, 3, 40)), policy)
        state, images = policy.calls[0]
        self.assertEqual(state.tolist(), [5, -5, 1, 2, 3, 40])
        self.assertEqual(sorted(images), sorted(CAMERAS))

    def test_targets_are_clamped_per_joint_range(self):
        wild = dict(zip(JOINTS, (150, -150, 90, 0, -101, -7)))
        robot = FakeRobot(pose=(95, -95, 90, 0, -95, 5))
        result, _, _ = run(robot, FakePolicy(action=wild), max_start_error=500)
        self.assertEqual(result.status, "completed")
        self.assertEqual(list(robot.sent[0].values()), [100, -100, 90, 0, -100, 0])

    def test_far_first_target_is_refused_before_anything_is_sent(self):
        robot = FakeRobot(pose=(0, 0, 0, 0, 0, 10))
        far = dict(zip(JOINTS, (60, 0, 0, 0, 0, 10)))
        result, _, _ = run(robot, FakePolicy(action=far))
        self.assertEqual((result.status, result.steps), ("start_too_far", 0))
        self.assertIn("60.0 raw units", result.detail)
        self.assertEqual(robot.sent, [])

    def test_policy_timeout_stops_sending_and_reports(self):
        robot = FakeRobot()
        result, _, _ = run(robot, FakePolicy(fail_at=3))
        self.assertEqual((result.status, result.steps), ("policy_error", 2))
        self.assertIn("server timeout", result.detail)
        self.assertEqual(len(robot.sent), 2)

    def test_non_finite_or_incomplete_actions_are_never_sent(self):
        for action in (dict(zip(JOINTS, (float("nan"), 0, 0, 0, 0, 10))), {"shoulder_pan.pos": 1.0}):
            robot = FakeRobot()
            result, _, _ = run(robot, FakePolicy(action=action))
            self.assertEqual((result.status, result.steps), ("policy_error", 0))
            self.assertEqual(robot.sent, [])

    def test_robot_failures_stop_the_episode(self):
        result, _, _ = run(FakeRobot(fail_observation_at=2), FakePolicy())
        self.assertEqual((result.status, result.steps), ("robot_error", 1))
        result, _, _ = run(FakeRobot(fail_send=True), FakePolicy())
        self.assertEqual((result.status, result.steps), ("robot_error", 0))

    def test_stop_request_ends_the_episode_cleanly(self):
        robot, clock = FakeRobot(), FakeClock()
        counter = iter(range(100))
        result = rr.run_episode(robot, FakePolicy(), JOINTS, CAMERAS, rr.RolloutSettings(fps=10, enable_motion=True),
                                should_stop=lambda: next(counter) >= 2, clock=clock, sleep=clock.sleep)
        self.assertEqual((result.status, result.steps), ("stopped", 2))

    def test_overrun_does_not_burst_to_catch_up(self):
        robot, clock = FakeRobot(), FakeClock()
        original = robot.get_observation

        def slow():
            clock.now += 0.35  # three ticks late
            return original()

        robot.get_observation = slow
        settings = rr.RolloutSettings(fps=10, episode_seconds=1.0, enable_motion=True)
        result = rr.run_episode(robot, FakePolicy(), JOINTS, CAMERAS, settings, clock=clock, sleep=clock.sleep)
        self.assertEqual(clock.sleeps, [])
        self.assertLessEqual(result.steps, 10)

    def test_keyboard_interrupt_is_not_swallowed(self):
        with self.assertRaises(KeyboardInterrupt):
            run(FakeRobot(), FakePolicy(fail_at=2, error=KeyboardInterrupt()))


class ReportTest(unittest.TestCase):
    def test_only_labelled_episodes_count(self):
        episodes = [{"outcome": "success"}, {"outcome": "failure"}, {"outcome": "unlabeled"}, {"outcome": "success"}]
        self.assertEqual(rr.summarize(episodes), {"episodes": 4, "labelled": 3, "successes": 2, "success_rate": 2 / 3})
        self.assertIsNone(rr.summarize([{"outcome": "unlabeled"}])["success_rate"])

    def test_report_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            rr.write_report(path, {"policy": "demo", "motion": False}, [{"outcome": "success"}])
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((data["schema"], data["policy"], data["summary"]["successes"]), ("so101_real_rollout_v1", "demo", 1))


if __name__ == "__main__":
    unittest.main()
