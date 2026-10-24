import logging
import traceback
from dataclasses import dataclass
from typing import Callable

import zmq 

from policy_client.transport import MsgSerializer # type: ignore , in the docker container both server and client will be present next to each other 

from .contract import ServerPolicy

log = logging.getLogger(__name__)


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    def __init__(
        self,
        policy: ServerPolicy,
        host: str = "*",
        port: int = 5555,
        timeout_ms: int = 15000, 
        api_token: str | None = None,
    ) -> None:
        super().__init__()
        self.context = zmq.Context() 
        self.host = host 
        self.port = port 
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.policy = policy 
        self.running = False 
        self.endpoints: dict[str, EndpointHandler] = {} 

        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0) 
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms) 

        self.port = self.socket.bind_to_random_port(f"tcp://{host}") if port == 0 else port 

        if port != 0:
            self.socket.bind(f"tcp://{host}:{port}")

        self.register_endpoint("get_action", lambda **data: self.policy.infer(data))
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("reset", self.policy.reset, requires_input=False)
        self.register_endpoint("describe", self.policy.info, requires_input=False) 
        self.register_endpoint("seed_policy", self._seed_policy)

        log.info("Policy server listening on tcp://%s:%s", host, self.port)
        
    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        if not name:
            raise ValueError("Endpoint name cannot be empty")

        self.endpoints[name] = EndpointHandler(handler=handler,  requires_input=requires_input)

    def run(self) -> None:
        self.running = True
        poller = zmq.Poller() 
        poller.register(self.socket, zmq.POLLIN)

        try:
            while self.running:
                try:
                    events = dict(poller.poll(self.timeout_ms))
                except zmq.ZMQError:
                    if not self.running:
                        break 
                    log.exception("Error pulling socket")
                    continue 

                if self.socket not in events: 
                    continue 

                if not events[self.socket] & zmq.POLLIN:
                    continue 

                try:
                    request = MsgSerializer.from_bytes(self.socket.recv())

                    if not isinstance(request, dict): 
                        raise ValueError("Request must be a dictionary")

                    if not self._validate_token(request):
                        response = {"error": "Unauthorized"}
                    else:
                        endpoint_name = request.get("endpoint", "get_action")
                        endpoint = self.endpoints.get(endpoint_name)

                        if endpoint is None:
                            raise ValueError(f"Unkown endpoitn {endpoint_name}")

                        data = request.get("data", {})

                        if endpoint.requires_input:
                            if not isinstance(data, dict):
                                raise ValueError("Request data must be a dictionary")

                            result = endpoint.handler(**data) 
                        else:
                            result = endpoint.handler() 
                        response = result 
                    self.socket.send(MsgSerializer.to_bytes(response))
                except Exception as exc:
                    log.error("Error handling policy request: \n%s",traceback.format_exc(),)

                    try:
                        self.socket.send(
                            MsgSerializer.to_bytes({"error": str(exc)})
                        )
                    except Exception:
                        log.exception("Failed to send error response")
        finally:
            self.running = False
            log.info("Policy server stopped")

    def _validate_token(self, request: dict) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def _handle_ping(self) -> dict:
        return {
            "status": "ok", 
            "message": "Policy server is running"
        }

    def _kill_server(self) -> None:
        self.running = False
        return None

    def _seed_policy(self, seed: int) -> dict:
        self.policy.reset(seed = seed) 
        return {"seed": seed}

    def close(self) -> None:
        self.running = False
        self.socket.close(linger=0) 
        self.context.term() 