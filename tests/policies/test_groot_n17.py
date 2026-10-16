"""groot_n17 plugin: request/observation conversion matches the old GR00TRemotePolicy exactly."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]
sys.modules.setdefault("policy_client", importlib.import_module("sim_to_real_so101.policy_client"))
sys.modules.setdefault("server_core", importlib.import_module("policy_servers.core"))

from sim_to_real_so101.policy_client.contract import load_client_policy  # noqa: E402
from sim_to_real_so101.policy_client.discovery import load_manifest  # noqa: E402
from server_core.contract import load_server_policy  # noqa: E402

POLICY_DIR = ROOT / "policies" / "groot_n17"


def old_add_batch_time_dims(obs):
    """Verbatim logic of GR00TRemotePolicy._add_batch_time_dims."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = old_add_batch_time_dims(val)
        else:
            obs[key] = [val]
    return obs


def old_inputs(state_np, images, instruction):
    """Verbatim logic of GR00TRemotePolicy._sim_obs_to_groot_inputs after the state/cameras are chosen."""
    obs = {
        "video": dict(images),
        "state": {"single_arm": state_np[:5], "gripper": state_np[5:6]},
        "language": {"annotation.human.task_description": instruction},
    }
    return old_add_batch_time_dims(old_add_batch_time_dims(obs))


class FakeGr00tPolicy:
    def __init__(self):
        self.seen = []
        self.resets = 0

    def get_action(self, observation):
        self.seen.append(observation)
        steps = 16
        arm = np.tile(np.arange(5, dtype=np.float32), (1, steps, 1))
        grip = np.full((1, steps, 1), 7.0, dtype=np.float32)
        return {"single_arm": arm, "gripper": grip}, {}

    def reset(self):
        self.resets += 1


class Gr00tN17Test(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(POLICY_DIR)
        self.server = load_server_policy(POLICY_DIR)
        self.client = load_client_policy("groot_n17", self.manifest, {})
        self.state = np.arange(6, dtype=np.float32)
        self.images = {c: np.full((4, 5, 3), i, dtype=np.uint8) for i, c in enumerate(self.manifest.cameras)}

    def test_request_converts_to_the_exact_old_groot_input(self):
        request = self.client.build_request(self.state, self.images, "pick the cube")
        from_plugin = sys.modules[type(self.server).__module__].build_observation(request)
        expected = old_inputs(self.state.copy(), {k: v.copy() for k, v in self.images.items()}, "pick the cube")
        self.assertEqual(from_plugin["language"], expected["language"])
        for name in self.manifest.cameras:
            self.assertEqual(from_plugin["video"][name].shape, (1, 1, 4, 5, 3))
            np.testing.assert_array_equal(from_plugin["video"][name], expected["video"][name])
        for key in ("single_arm", "gripper"):
            np.testing.assert_array_equal(from_plugin["state"][key], expected["state"][key])
            self.assertEqual(from_plugin["state"][key].dtype, np.float32)

    def test_infer_returns_full_chunk_as_T_by_6(self):
        self.server.policy = FakeGr00tPolicy()
        request = self.client.build_request(self.state, self.images, "go")
        response = self.server.infer(request)
        actions = self.client.parse_response(response, [f"j{i}" for i in range(6)])
        self.assertEqual(actions.shape, (16, 6))
        np.testing.assert_array_equal(actions[0], [0, 1, 2, 3, 4, 7])
        self.assertEqual(len(self.server.policy.seen), 1)

    def test_missing_camera_and_bad_state_are_rejected(self):
        with self.assertRaises(KeyError):
            self.client.build_request(self.state, {}, "x")
        self.server.policy = FakeGr00tPolicy()
        request = self.client.build_request(self.state, self.images, "x")
        request["state"] = np.zeros(5, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.server.infer(request)

    def test_info_names_the_policy_for_connect(self):
        self.assertEqual(self.server.info()["name"], "groot_n17")

    def test_reset_without_seed_resets_the_model(self):
        self.server.policy = FakeGr00tPolicy()
        self.assertEqual(self.server.reset(), {"seed": None})
        self.assertEqual(self.server.policy.resets, 1)

    def test_bad_seed_is_rejected(self):
        for bad in (-1, 2**32, True, 1.5):
            with self.assertRaises(ValueError):
                self.server.reset(seed=bad)


if __name__ == "__main__":
    unittest.main()
