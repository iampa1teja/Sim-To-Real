"""Client side of the GR00T N1.7 policy (runs in the teleop / real-robot containers, no torch).

The server (server.py) owns GR00T's input format, so this side only picks the manifest cameras and
passes the joint state through. Wire format:
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)} absolute joint targets in joint_order
"""
from core.policy_client.contract import ClientPolicy
from core.policy_client.default import DefaultClientPolicy
from core.policy_client.discovery import PolicyManifest


class Gr00tN17ClientPolicy(DefaultClientPolicy):
    """Use the standard client request/response contract."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return Gr00tN17ClientPolicy.from_manifest(manifest, options)
