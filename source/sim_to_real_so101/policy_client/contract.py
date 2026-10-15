import abc
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
from .discovery import PolicyManifest

log = logging.getLogger(__name__)


class ClientPolicy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> "ClientPolicy": ...

    @abc.abstractmethod
    def build_request(self, joint_positions: np.ndarray, images: dict[str, np.ndarray], instruction: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    def parse_response(self, response: dict[str, Any], joint_order: list[str]) -> np.ndarray: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


def load_client_policy(name: str, manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    client_path = Path(manifest.path) / "client.py"
    if not client_path.is_file():
        raise FileNotFoundError(f"{client_path} not found")

    path_key = str(client_path.resolve()).encode("utf-8")
    module_name = f"_policy_client_{name}_{hashlib.sha256(path_key).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, client_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {client_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, "create")
        policy = factory(manifest, options)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not isinstance(policy, ClientPolicy):
        sys.modules.pop(module_name, None)
        raise TypeError("create() did not return a ClientPolicy instance")
    return policy
