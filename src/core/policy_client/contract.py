import abc
import hashlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .discovery import PolicyManifest

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmInfo:
    """Policy-facing joint order and device codec; no simulator or torch imports."""

    name: str
    joint_names: tuple[str, ...]
    roles: tuple[str, ...]
    types: tuple[str, ...]
    joint_mapping: dict[str, dict[str, float]]

    def __post_init__(self):
        count = len(self.joint_names)
        if not count or len(set(self.joint_names)) != count or len(self.roles) != count or len(self.types) != count:
            raise ValueError('ArmInfo requires unique joint names with one role and type per joint')
        if any(role not in ('arm', 'gripper') for role in self.roles):
            raise ValueError('ArmInfo roles must be arm or gripper')
        if any(kind not in ('revolute', 'prismatic') for kind in self.types):
            raise ValueError('ArmInfo joint types must be revolute or prismatic')
        if set(self.joint_mapping) != set(self.joint_names):
            raise ValueError('ArmInfo requires a complete joint mapping')
        for name, role in zip(self.joint_names, self.roles):
            span = self.joint_mapping[name]
            if ('raw_min' in span) != ('raw_max' in span):
                raise ValueError(f'ArmInfo requires both raw codec endpoints for {name!r}')
            low, high = span['joint_min'], span['joint_max']
            raw_low, raw_high = span.get('raw_min', 0 if role == 'gripper' else -100), span.get('raw_max', 100)
            if not np.isfinite([low, high, raw_low, raw_high]).all() or high <= low or raw_low == raw_high:
                raise ValueError(f'ArmInfo has invalid physical or device bounds for {name!r}')

    @property
    def feature_names(self):
        return tuple(name + '.pos' for name in self.joint_names)

    @property
    def arm_indices(self):
        return tuple(index for index, role in enumerate(self.roles) if role == 'arm')

    @property
    def gripper_indices(self):
        return tuple(index for index, role in enumerate(self.roles) if role == 'gripper')

    @classmethod
    def from_arm(cls, arm, mapping=None):
        if mapping is not None and hasattr(mapping, 'mins'):
            result = cls.from_mapping(mapping, name=arm.name)
            if result.joint_names != arm.joint_names or result.roles != tuple(joint.role for joint in arm.action_joints):
                raise ValueError('Mapping joint order or roles differ from the selected arm')
            return result
        spans = {name: dict(span) for name, span in arm.runtime_joint_limits.items()}
        spans.update({name: dict(span) for name, span in getattr(mapping, 'mapping', mapping or {}).items()})
        if arm.lerobot.use_degrees:
            missing = [joint.name for joint in arm.action_joints if joint.role == 'arm'
                       and not all(key in spans[joint.name] for key in ('raw_min', 'raw_max'))]
            if missing:
                raise ValueError(f'Degree mode needs an explicit calibrated device codec for {missing}')
        return cls(arm.name, arm.joint_names, tuple(joint.role for joint in arm.action_joints),
                   tuple(joint.type for joint in arm.action_joints), spans)

    @classmethod
    def from_mapping(cls, mapping, *, name=''):
        def values(array):
            return np.asarray(array.detach().cpu() if hasattr(array, 'detach') else array)

        names = tuple(mapping.names)
        gripper = values(mapping.gripper_mask).astype(bool)
        revolute = values(mapping.revolute_mask).astype(bool)
        low, high = values(mapping.mins), values(mapping.maxs)
        raw_low = getattr(mapping, 'raw_mins', None)
        raw_high = getattr(mapping, 'raw_maxs', None)
        raw_low = values(raw_low) if raw_low is not None else np.where(gripper, 0, -100)
        raw_high = values(raw_high) if raw_high is not None else np.full(len(names), 100)
        spans = {joint: dict(joint_min=float(low[index]), joint_max=float(high[index]),
                             raw_min=float(raw_low[index]), raw_max=float(raw_high[index]))
                 for index, joint in enumerate(names)}
        return cls(name, names, tuple('gripper' if value else 'arm' for value in gripper),
                   tuple('revolute' if value else 'prismatic' for value in revolute), spans)


class ClientPolicy(abc.ABC):
    def configure(self, arm: ArmInfo) -> None:
        """Optional: configure the selected arm's joint order, roles and raw codec."""

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
