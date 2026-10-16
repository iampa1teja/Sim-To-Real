"""Glue between a robot-side control loop and a policy plugin.

The simulator wrapper (utils/lerobot_interface.py) and the real-robot adapter (so101_eval.py)
both need the same three steps: build the plugin's request, call the server, parse the reply into
an (T, J) action array. PluginClient does exactly that and keeps the small client interface
(ping / reset / call_endpoint / get_action) the control loops already use.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .contract import ClientPolicy, load_client_policy
from .discovery import PolicyManifest, discover
from .transport import PolicyClient

log = logging.getLogger(__name__)


class PluginClient:
    """A PolicyClient that speaks one plugin's request/response format."""

    def __init__(self, client: PolicyClient, policy: ClientPolicy, manifest: PolicyManifest,
                 joint_order: Sequence[str]) -> None:
        self.client = client
        self.policy = policy
        self.manifest = manifest
        self.joint_order = list(joint_order)

    def ping(self) -> bool:
        return self.client.ping()

    def describe(self) -> dict[str, Any]:
        return self.client.describe()

    def reset(self) -> Any:
        self.policy.reset()
        return self.client.reset()

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        return self.client.call_endpoint(endpoint, data, requires_input)

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """observation = {"state": (J,), "images": {camera: array}, "instruction": str} -> (T, J)."""
        request = self.policy.build_request(
            np.asarray(observation["state"], dtype=np.float32),
            observation["images"],
            observation.get("instruction", ""),
        )
        reply = self.client.call_endpoint("get_action", request)
        actions = np.asarray(self.policy.parse_response(reply, self.joint_order), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(self.joint_order):
            raise ValueError(f"expected (T, {len(self.joint_order)}) actions, got {actions.shape}")
        return actions


def build_plugin_client(
    policy_name: str,
    joint_order: Sequence[str],
    host: str = "localhost",
    port: int | None = None,
    timeout_ms: int = 15000,
    api_token: str | None = None,
    policy_options: dict[str, str] | None = None,
) -> PluginClient:
    """Look up the plugin, build its ClientPolicy and a transport client for the server."""
    policies = discover()
    if policy_name not in policies:
        raise ValueError(f"Unknown policy {policy_name!r}; available: {', '.join(sorted(policies))}")
    manifest = policies[policy_name]
    policy = load_client_policy(policy_name, manifest, dict(policy_options or {}))
    client = PolicyClient(host, manifest.default_port if port is None else port, timeout_ms, api_token)
    return PluginClient(client, policy, manifest, joint_order)


def check_server(client: PluginClient) -> dict[str, Any]:
    """Ping the server and confirm it serves the policy this client expects; return its info."""
    name = client.manifest.name
    try:
        alive = client.ping()
    except TimeoutError as exc:
        raise ConnectionError(f"Timed out connecting to the {name!r} policy server") from exc
    if not alive:
        raise ConnectionError(f"The {name!r} policy server did not answer ping")
    info = client.describe()
    if not isinstance(info, dict) or info.get("name") != name:
        got = info.get("name") if isinstance(info, dict) else info
        raise RuntimeError(f"Policy mismatch: expected {name!r}, the server reports {got!r}")
    log.info("Connected to policy server %s: %s", name, info)
    return info
