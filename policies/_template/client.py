"""Client side of the policy plugin (runs in the teleop container, no torch allowed).

Wire format agreed with server.py in this folder:
  request  = {"state": float32 (J,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (horizon, J)}   absolute joint targets, in joint_order
Change both files together if your model needs something else.
"""
from core.policy_client.contract import ClientPolicy
from core.policy_client.default import DefaultClientPolicy
from core.policy_client.discovery import PolicyManifest


class TemplateClientPolicy(DefaultClientPolicy):
    """Use the standard client request/response contract."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return TemplateClientPolicy.from_manifest(manifest, options)
