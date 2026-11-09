"""Client side of the example policy: holds the current joint state (no torch).

Wire format: request {"state": (J,), "images": {camera: array}, "instruction": str},
response {"actions": (H, J)} absolute joint targets in joint_order.
"""
from core.policy_client.contract import ClientPolicy
from core.policy_client.default import DefaultClientPolicy
from core.policy_client.discovery import PolicyManifest


class ExampleClientPolicy(DefaultClientPolicy):
    """Use the standard client request/response contract."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return ExampleClientPolicy.from_manifest(manifest, options)
