import io
from typing import Any

import msgpack
import numpy as np
import zmq

class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default = MsgSerializer.encode_custom_classes , use_bin_type = True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        try:
            return msgpack.unpackb(data, object_hook = MsgSerializer.decode_custom_classes, raw = False)
        except Exception as exc:
            raise ValueError(f"Invalid Input: {exc}") from exc 

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if obj.get("__nd__") is True and "npy" in obj:
            return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
        return obj 

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise ValueError("Object arrays are not allowed")
            
            buffer = io.BytesIO() 
            np.save(buffer, obj, allow_pickle=False)
            
            return {
                "__nd__": True, 
                "npy" : buffer.getvalue() 
            }
        
        if isinstance(obj, np.generic): 
            return obj.item() 

        raise TypeError(f"Cannot Seialize {type(obj).__name__}")
        

class PolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000, api_token: str | None = None,) -> None:
        self.host = host 
        self.port = port 
        self.timeout_ms = timeout_ms 
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = None 
        self._init_socket() 

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ) 
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms) 
        self.socket.setsockopt(zmq.LINGER, 0) 
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _reset_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0) 

        self._init_socket() 

    def ping(self) -> bool:
        try:
            response = self.call_endpoint(
                "ping",
                requires_input=False,
            )
            return isinstance(response, dict) and response.get("status") == "ok"
        except (TimeoutError, zmq.ZMQError):
            return False

    def kill_server(self) -> None:
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request = {
            "endpoint": endpoint
        }

        if requires_input:
            request["data"] = data if data is not None else {} 

        if self.api_token is not None: 
            request["api_token"] = self.api_token

        try:
            encoded_request = MsgSerializer.to_bytes(request) 
            self.socket.send(encoded_request) 
            encoded_response = self.socket.recv() 
            response = MsgSerializer.from_bytes(encoded_response)
        except zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"Timeout waiting for response from endpoint '{endpoint}'") from exc 

        except zmq.ZMQError:
            self._reset_socket() 
            raise 

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])

        return response

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        return self.call_endpoint("get_action", data={"observation": observation, "options": options}, requires_input=True)

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        return self.call_endpoint("reset", data=options or {}, requires_input=options is not None,)

    def describe(self) -> dict[str, Any]:
        return self.call_endpoint("describe", requires_input=False) 

    def seed(self, seed: int) -> Any:
        return self.call_endpoint("seed_policy", data = {"seed": seed}, requires_input=True,)

    def __del__(self) -> None:
        try:
            socket = getattr(self, "socket", None) 
            if socket is not None:
                socket.close(linger=0)
                self.socket = None

            context = getattr(self, "context", None) 
            if context is not None:
                context.term() 
                self.context = None 
        except Exception:
            pass 