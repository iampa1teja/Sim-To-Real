"""End to end: launcher + hold_position server in a subprocess, RemotePolicy as the client."""
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
os.environ["POLICIES_DIR"] = str(ROOT / "policies")

from sim_to_real_so101.policy_client.remote_policy import RemotePolicy  # noqa: E402

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

# The server container imports `server_core` and `policy_client` as top-level packages.
# Outside the container they live under policy_servers/core and sim_to_real_so101/policy_client.
SERVER_BOOT = (
    "import importlib, sys\n"
    f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'source')!r}]\n"
    "sys.modules['server_core'] = importlib.import_module('policy_servers.core')\n"
    "sys.modules['policy_client'] = importlib.import_module('sim_to_real_so101.policy_client')\n"
    "from server_core.launcher import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeRobot:
    joint_order = JOINTS


def observation(state):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"joint_positions": np.asarray(state, dtype=np.float32), "images": {"room": img, "wrist": img}}


class HoldPositionEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        env = dict(os.environ, POLICIES_DIR=str(ROOT / "policies"))
        cls.proc = subprocess.Popen(
            [sys.executable, "-c", SERVER_BOOT, "--policy-dir", str(ROOT / "policies" / "hold_position"),
             "--port", str(cls.port), "--host", "127.0.0.1", "--device", "cpu"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        cls.policy = RemotePolicy(FakeRobot(), "hold_position", port=cls.port, timeout_ms=3000,
                                  lang_instruction="pick the cube")
        for _ in range(50):
            try:
                cls.policy.connect()
                return
            except (ConnectionError, TimeoutError):
                time.sleep(0.2)
        cls.tearDownClass()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.policy._client.kill_server()
        except Exception:
            pass
        try:
            cls.proc.wait(timeout=10)
        except Exception:
            cls.proc.kill()

    def test_actions_repeat_state_for_the_whole_chunk(self):
        self.policy.reset(seed=1)
        state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for step in range(8):
            self.assertEqual(self.policy.steps_until_query(), 0 if step == 0 else 8 - step)
            action = self.policy.get_action(observation(state))
            self.assertEqual(list(action), JOINTS)
            np.testing.assert_allclose(list(action.values()), state, rtol=1e-6)

    def test_new_query_after_chunk_is_used(self):
        self.policy.reset()
        for _ in range(8):
            self.policy.get_action(observation([0.0] * 6))
        action = self.policy.get_action(observation([1.0] * 6))
        np.testing.assert_allclose(list(action.values()), [1.0] * 6)
        self.assertGreaterEqual(self.policy.close_timing()["count"], 2)

    def test_missing_camera_is_reported(self):
        self.policy.reset()
        bad = {"joint_positions": np.zeros(6, dtype=np.float32), "images": {"room": np.zeros((4, 4, 3), np.uint8)}}
        with self.assertRaises(KeyError):
            self.policy.get_action(bad)


if __name__ == "__main__":
    unittest.main()
