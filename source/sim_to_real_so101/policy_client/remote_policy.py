"""Generic client-side policy used by every plugin: talks to a policy server over ZeroMQ.

Lives in source/sim_to_real_so101/policy_client/. Imports no torch. It does not know which
model is on the other side: the plugin's ClientPolicy decides what goes into the request and
how the reply is read, this class only moves data and feeds the existing smoothing/grasp code.
"""
from __future__ import annotations
import logging
import numpy as np
from collections.abc import Mapping 
import time 

from .contract import ClientPolicy, load_client_policy
from .discovery import PolicyManifest, discover
from .transport import PolicyClient
from .types import ActionChunk

log = logging.getLogger(__name__)


class RemotePolicy:
    """Drop-in replacement for GR00TRemotePolicy that works for any policy plugin."""

    def __init__(
        self,
        robot_iface,
        policy_name: str,
        host: str = "localhost",
        port: int | None = None,
        action_horizon: int | None = None,
        lang_instruction: str = "",
        timeout_ms: int = 15000,
        token: str | None = None,
        policy_options: dict | None = None,
    ) -> None:
        policies = discover()
        policy_names = sorted(policies.keys())
        if policy_name not in policy_names:
            raise ValueError(f"Unknown policy {policy_name!r}; available: {', '.join(policy_names)}")
        manifest = policies[policy_name]
        port = manifest.default_port if port is None else port
        action_horizon = manifest.default_horizon if action_horizon is None else action_horizon

        self._robot = robot_iface
        self._manifest = manifest
        self._action_horizon = action_horizon
        self._lang_instruction = lang_instruction
        self._policy = load_client_policy(policy_name, manifest, policy_options or {})
        self._client = PolicyClient(host, port, timeout_ms, token)
        self._buffer: list = []        # remaining rows of the current chunk
        self._latencies: list[float] = []

    def connect(self) -> None:
        try:
            alive = self._client.ping()
        except TimeoutError as exc:
            raise ConnectionError(f"Timed out connecting to policy server for {self._manifest.name!r}") from exc

        if not alive:
            raise ConnectionError(f"Policy server did not respond to ping for {self._manifest.name!r}")

        info = self._client.describe()

        if not isinstance(info, dict):
            raise RuntimeError(f"Invalid describe response: expected dict, received {type(info).__name__}")

        server_name = info.get("name")

        if server_name != self._manifest.name:
            raise RuntimeError(f"Policy mismatch: expected {self._manifest.name!r}, server reports {server_name!r}")

        log.info("Connected to policy server: %s; info=%s", self._manifest.name, info,)

        
    def reset(self, seed: int | None = None) -> None:
        self._buffer.clear()
        self._policy.reset() 
        self._client.reset() 

        if seed is not None:
            self._client.seed(seed) 

        log.info("Reset policy %s with seed %s", self._manifest.name, seed)

    def steps_until_query(self) -> int:
        return len(self._buffer) 

    def close_timing(self) -> dict:
        if not self._latencies:
            return {
                "count" : 0, 
                "mean_ms": 0.0, 
                "max_ms": 0.0, 
            }
        return {
            "count": len(self._latencies), 
            "mean_ms": float(np.mean(self._latencies)),
            "max_ms": float(max(self._latencies)),
        }

    def _get_joint_order(self, observation: dict):
        for attr in ("joint_order", "joint_names"):
            value = getattr(self._robot, attr, None)

            if callable(value):
                value = value() 

            if value is not None:
                joint_order = list(value)
                if joint_order:
                    return joint_order 
        value = observation.get("joint_order")

        if value is not None:
            joint_order = list(value) 
            if joint_order:
                return joint_order 

        raise ValueError("Cannot determine robot joint order. Provide robot._iface.joint_order, robot_iface.joint_names, or observation['joint_order']")

    def get_action(self, observation: dict) -> dict:
        if not isinstance(observation, dict): 
            raise TypeError("observation must be dictionary")

        if not self._buffer:
            if "joint_positions" not in observation:
                raise KeyError("observation must contain 'joint_positions'")

            joint_positions = np.asarray(observation["joint_positions"], dtype=np.float32)

            if joint_positions.ndim != 1:
                raise ValueError("'joint_positions' must be a one dimensional array")
            images = observation.get("images", observation) 

            if not isinstance(images, Mapping): 
                raise TypeError("Observation images must be a dictionary")

            missing_cameras = [camera for camera in self._manifest.cameras if camera not in images]
            if missing_cameras:
                raise KeyError(f"observation is missing cameras {missing_cameras}; got {sorted(images)}")

            camera_images = {
                camera: np.asarray(images[camera])
                for camera in self._manifest.cameras
            }

            joint_order = self._get_joint_order(observation) 

            request = self._policy.build_request(joint_positions, camera_images, self._lang_instruction)

            start = time.perf_counter()

            try:
                reply = self._client.call_endpoint("get_action", request)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000 
                self._latencies.append(elapsed_ms)

            parsed = self._policy.parse_response(reply, joint_order) 

            if isinstance(parsed, ActionChunk): 
                chunk = parsed 
            else:
                chunk = ActionChunk(actions=np.asarray(parsed, dtype=np.float32))
            actions = chunk.actions[:self._action_horizon]

            if actions.shape[1] != len(joint_order):
                raise ValueError(f"Action width {actions.shape[1]} does not match joint count {len(joint_order)}")

            self._buffer = [row.copy() for row in actions]

        row = self._buffer.pop(0) 

        joint_order = self._get_joint_order(observation) 

        if len(row) != len(joint_order):
            raise ValueError(f"Action contains {len(row)} values, but the robot has {len(joint_order)} joints")

        return {
            name: float(value)
            for name, value in zip(joint_order, row)
        }