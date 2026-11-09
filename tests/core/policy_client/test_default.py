"""Shared client plugins preserve the NumPy wire contract for any joint count."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.policy_client.contract import load_client_policy
from core.policy_client.default import DefaultClientPolicy
from core.policy_client.discovery import load_manifest
from core.policy_client.transport import MsgSerializer


POLICIES = Path(__file__).resolve().parents[3] / "policies"


@pytest.mark.parametrize("name", ["_template", "example", "groot_n17"])
def test_plugin_request_and_response_round_trip(name):
    manifest = replace(load_manifest(POLICIES / "example"), path=POLICIES / name)
    client = load_client_policy(name, manifest, {"custom": "value"})
    assert isinstance(client, DefaultClientPolicy)
    joints = np.asarray([1.25, -12.5, 0.0, 42.0], dtype=np.float64)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    request = client.build_request(joints, {"room": image, "wrist": image, "unused": image}, "Move the cube")
    request = MsgSerializer.from_bytes(MsgSerializer.to_bytes(request))
    assert request["state"].dtype == np.float32
    np.testing.assert_array_equal(request["state"], joints)
    assert list(request["images"]) == list(manifest.cameras)
    np.testing.assert_array_equal(request["images"]["room"], image)
    assert request["instruction"] == "Move the cube"
    response = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"actions": np.stack([joints, joints + 1])}))
    actions = client.parse_response(response, ["a", "b", "c", "d"])
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, response["actions"])
    client.reset()


def test_request_requires_manifest_cameras_and_respects_language_flag():
    client = DefaultClientPolicy(("room",), language=False)
    with pytest.raises(KeyError, match="missing cameras"):
        client.build_request(np.zeros(4), {}, "ignored")
    request = client.build_request(np.zeros(4), {"room": np.zeros((2, 2, 3))}, "ignored")
    assert request["instruction"] == ""
    assert request["images"]["room"].dtype == np.uint8


@pytest.mark.parametrize("actions", [np.zeros(4), np.zeros((2, 3)), np.zeros((1, 2, 4))])
def test_rejects_action_chunks_incompatible_with_joint_order(actions):
    with pytest.raises(ValueError, match="actions"):
        DefaultClientPolicy(("room",), False).parse_response({"actions": actions}, ["a", "b", "c", "d"])
