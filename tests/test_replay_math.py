"""Step 5 - utils/replay_math.py (pure torch) + the joint conversion the replay relies on.

python -m unittest tests.test_replay_math -v
  * pose tests: CPU, torch only
  * joint round-trip test: needs LeRobot importable (run it inside the teleop container);
    it is skipped elsewhere.

Contract under test (utils/replay_math.py; all wxyz quaternions, batch-friendly (..., 3)/(..., 4))
- base_to_world(base_pos, base_quat, pos_b, quat_b) -> (pos_w, quat_w)
      pos_w = base_pos + R(base_quat) @ pos_b ;  quat_w = base_quat * quat_b
- world_to_base(base_pos, base_quat, pos_w, quat_w) -> (pos_b, quat_b)
      exact inverse; identical to isaaclab.utils.math.subtract_frame_transforms, which the
      recorder used to write the sidecars (pick_place_agent._cube_pose_in_base)

Joint conversion (existing LeRobotSO101Interface; nothing new to write):
  the replay turns a recorded observation.state into sim radians with
  get_mapped_actions_vectorized(). Every recorded state came from
  get_raw_actions_from_radians(q) of a real sim angle q inside the USD limits, so the property
  that must hold is   forward(inverse(q)) == q   for all q within the USD joint limits.
  (The other direction does NOT hold near the ends: shoulder pan maps across -116..+120 deg but
  is clamped to the USD +-110 deg.)
"""
import importlib.util
import json
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _fakes import PKG, REAL_SETUP_JSON, block_imports, load_module, quat_mul, yaw_quat  # noqa: E402

M = None


def setUpModule():
    global M
    with block_imports("pxr", "isaaclab", "isaacsim", "omni", "lerobot"):
        M = load_module("replay_math_under_test", "source/sim_to_real_so101/utils/replay_math.py")


def T(*values):
    return torch.tensor(values, dtype=torch.float64)


def random_quat(generator, n):
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def rotate(q, v):
    """Reference rotation v' = q v q* (independent of the module under test)."""
    zeros = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype)
    conj = q * T(1, -1, -1, -1)
    return quat_mul(quat_mul(q, torch.cat([zeros, v], dim=-1)), conj)[..., 1:]


def same_rotation(a, b):
    # q and -q are the same rotation
    return torch.minimum((a - b).abs().max(), (a + b).abs().max()).item()


class PoseTests(unittest.TestCase):
    def test_identity_base(self):
        pos, quat = M.base_to_world(T(0, 0, 0), T(1, 0, 0, 0), T(0.2, -0.1, 0.01), T(*yaw_quat(0.4)))
        torch.testing.assert_close(pos, T(0.2, -0.1, 0.01))
        self.assertLess(same_rotation(quat, T(*yaw_quat(0.4))), 1e-12)

    def test_translated_and_yawed_base(self):
        base_pos, base_quat = T(1, 2, 3), T(*yaw_quat(math.pi / 2))
        # +x in the base frame is +y in the world after a 90 deg yaw
        pos, quat = M.base_to_world(base_pos, base_quat, T(1, 0, 0), T(1, 0, 0, 0))
        torch.testing.assert_close(pos, T(1, 3, 3), atol=1e-12, rtol=0)
        self.assertLess(same_rotation(quat, base_quat), 1e-12)
        back_pos, back_quat = M.world_to_base(base_pos, base_quat, T(1, 3, 3), base_quat)
        torch.testing.assert_close(back_pos, T(1, 0, 0), atol=1e-12, rtol=0)
        self.assertLess(same_rotation(back_quat, T(1, 0, 0, 0)), 1e-12)

    def test_matches_reference_formula(self):
        g = torch.Generator().manual_seed(0)
        base_pos, base_quat = torch.randn(64, 3, generator=g, dtype=torch.float64), random_quat(g, 64)
        pos_b, quat_b = torch.randn(64, 3, generator=g, dtype=torch.float64), random_quat(g, 64)
        pos_w, quat_w = M.base_to_world(base_pos, base_quat, pos_b, quat_b)
        torch.testing.assert_close(pos_w, base_pos + rotate(base_quat, pos_b), atol=1e-12, rtol=0)
        torch.testing.assert_close(quat_w, quat_mul(base_quat, quat_b), atol=1e-12, rtol=0)

    def test_round_trip(self):
        g = torch.Generator().manual_seed(1)
        base_pos, base_quat = torch.randn(256, 3, generator=g, dtype=torch.float64), random_quat(g, 256)
        pos_b, quat_b = torch.randn(256, 3, generator=g, dtype=torch.float64), random_quat(g, 256)
        pos_w, quat_w = M.base_to_world(base_pos, base_quat, pos_b, quat_b)
        pos_b2, quat_b2 = M.world_to_base(base_pos, base_quat, pos_w, quat_w)
        torch.testing.assert_close(pos_b2, pos_b, atol=1e-9, rtol=0)
        for a, b in zip(quat_b2, quat_b):
            self.assertLess(same_rotation(a, b), 1e-9)

    def test_float32_and_broadcast(self):
        pos, quat = M.base_to_world(torch.zeros(3), torch.tensor([1.0, 0, 0, 0]),
                                    torch.zeros(5, 3), torch.tensor([[1.0, 0, 0, 0]] * 5))
        self.assertEqual(pos.shape, (5, 3))
        self.assertEqual(quat.shape, (5, 4))


def lerobot_available():
    return importlib.util.find_spec("lerobot") is not None


@unittest.skipUnless(lerobot_available(), "LeRobot not installed; run inside the teleop container")
class JointRoundTripTests(unittest.TestCase):
    """Uses the real LeRobotSO101Interface (CPU; no robot connected)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("lerobot_interface_under_test",
                                                      PKG / "utils" / "lerobot_interface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.Interface = module.LeRobotSO101Interface
        setup = json.loads(REAL_SETUP_JSON.read_text())
        cls.iface = cls.Interface(device="cpu", port="/dev/null", id="test", cameras={}, fps=30,
                                  kind="follower", joint_mapping=setup.get("sim_joint_mapping", {}))

    def test_state_to_radians_recovers_sim_angle(self):
        lo = self.iface.joint_mins * math.pi / 180
        hi = self.iface.joint_maxs * math.pi / 180
        g = torch.Generator().manual_seed(0)
        for _ in range(2000):
            q = lo + (hi - lo) * torch.rand(6, generator=g)
            state = self.iface.get_raw_actions_from_radians(q.clone())   # what the recorder stored
            q2 = self.iface.get_mapped_actions_vectorized(state.clone())  # what the replay applies
            torch.testing.assert_close(q2, q, atol=1e-4, rtol=0)

    def test_limits_exactly(self):
        for q in (self.iface.joint_mins, self.iface.joint_maxs):
            q = q * math.pi / 180
            q2 = self.iface.get_mapped_actions_vectorized(self.iface.get_raw_actions_from_radians(q.clone()))
            torch.testing.assert_close(q2, q, atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
