# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
import io
import json
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from sim_to_real_so101.gr00t_client.types import ModalityConfig
from sim_to_real_so101.gr00t_client.utils import to_json_serializable

from sim_to_real_so101.gr00t_client.policy import BasePolicy


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def _numpy_dtype(description):
        """Restore msgpack's list-form structured dtype descriptors safely."""
        if isinstance(description, (list, tuple)):
            fields = []
            for name, subtype, *shape in description:
                if isinstance(name, bytes):
                    name = name.decode()
                elif isinstance(name, (list, tuple)):
                    name = tuple(part.decode() if isinstance(part, bytes) else part for part in name)
                fields.append((name, MsgSerializer._numpy_dtype(subtype),
                               *(tuple(value) if isinstance(value, list) else value for value in shape)))
            description = fields
        elif isinstance(description, bytes):
            description = description.decode()
        return np.dtype(description)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj

        def field(name):
            for key in (name, name.encode()):
                if key in obj:
                    return obj[key]
            raise ValueError(f"Malformed serialized payload: '{name}' missing")

        if any(marker in obj for marker in ("__ModalityConfig__", b"__ModalityConfig__",
                                             "__ModalityConfig_class__", b"__ModalityConfig_class__")):
            payload = field("as_json")
            if isinstance(payload, bytes):
                payload = payload.decode()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError("Malformed ModalityConfig payload: 'as_json' must contain an object")
            return ModalityConfig(**payload)
        if "__ndarray_class__" in obj or b"__ndarray_class__" in obj:
            return np.load(io.BytesIO(field("as_npy")), allow_pickle=False)

        # N1.7 responses use msgpack_numpy's wire format. Decode numeric buffers
        # directly so Isaac Sim needs no msgpack_numpy dependency or pickle path.
        if "nd" in obj or b"nd" in obj:
            kind = obj.get("kind", obj.get(b"kind"))
            if kind in ("O", b"O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload")
            try:
                dtype = MsgSerializer._numpy_dtype(field("type"))
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("Malformed NumPy dtype payload") from exc
            if dtype.hasobject:
                raise ValueError("Refusing to decode object-bearing NumPy dtype")
            try:
                nd = field("nd")
                if not isinstance(nd, bool):
                    raise ValueError("'nd' must be a boolean")
                values = np.frombuffer(field("data"), dtype=dtype)
                if nd:
                    shape = field("shape")
                    if not isinstance(shape, (list, tuple)) or any(
                            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape):
                        raise ValueError("'shape' must contain nonnegative integers")
                    return values.reshape(shape)
                if values.size != 1:
                    raise ValueError("scalar payload must contain exactly one value")
                return values[0]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Malformed NumPy payload: {exc}") from exc
        if "complex" in obj or b"complex" in obj:
            value = field("data")
            return complex(value.decode() if isinstance(value, bytes) else value)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(
        self, policy: BasePolicy, host: str = "*", port: int = 5555, api_token: str = None
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = PolicyServer(policy, port=port, api_token=api_token)
        server.run()


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _reset_socket(self):
        """Drop a socket stuck awaiting a reply and open a fresh one."""
        self.socket.close(linger=0)
        self._init_socket()

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except TimeoutError:
            return False
        except zmq.error.ZMQError:
            self._reset_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        try:
            message = self.socket.recv()
        except zmq.error.Again:
            self._reset_socket()  # REQ socket is stuck awaiting the lost reply
            raise TimeoutError(
                f"No reply from policy server at {self.host}:{self.port} within {self.timeout_ms} ms"
            ) from None
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        socket = getattr(self, "socket", None)
        if socket is not None:
            socket.close(linger=0)
        context = getattr(self, "context", None)
        if context is not None:
            context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
