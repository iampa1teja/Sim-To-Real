import abc
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np

log = logging.getLogger(__name__)

class ServerPolicy(abc.ABC):

    @abc.abstractmethod
    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None: ...

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]: ...

    @abc.abstractmethod
    def infer(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abc.abstractmethod
    def info(self) -> dict[str, Any]: ...


def load_server_policy(policy_dir: str | Path) -> ServerPolicy:
    policy_dir = Path(policy_dir)
    server_path = policy_dir / "server.py"
    if not server_path.is_file():
        raise FileNotFoundError(f"{server_path} not found")
    
    path_key = str(server_path.resolve()).encode("utf-8")
    module_name = f"_policy_server_{hashlib.sha256(path_key).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {server_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    factory = getattr(module, "create")
    policy = factory()
    if not isinstance(policy, ServerPolicy):
        raise TypeError("create() did not return a ServerPolicy instance")
    return policy
