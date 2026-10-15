"""Shared command line options so every eval script can pick a policy the same way.

Lives in source/sim_to_real_so101/policy_client/. Imports no torch.
"""
from __future__ import annotations

import argparse

from .discovery import PolicyManifest, discover


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    policies = discover() 
    policy_names = sorted(policies.keys()) 
    default_policy = "groot_n17" if "groot_n17" in policies else None
    parser.add_argument("--policy", type=str, choices=policy_names,default=default_policy,help="Policy to use")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server hostname")
    parser.add_argument("--port", type=int, default=None, help="Policy server port. Defaults to the manifest's default_port.")
    parser.add_argument("--policy_option", dest="policy_option",action="append",default=None,metavar="KEY-VALUE",help="Policy-specific option; may be specified multiple times.")

def policy_from_args(args: argparse.Namespace) -> tuple[PolicyManifest, dict]:
    policies = discover() 
    if not policies: 
        raise ValueError("No enabled policies were discovered.")
    policy_name = args.policy 

    if policy_name is None: 
        raise ValueError("No default policy is available. Specify one using --policy")

    if policy_name not in policies: 
        raise ValueError(
            f"Unkown policy {policy_name!r} "
            f"Available policies: {', '.join(sorted(policies))}"
        )

    manifest = policies[policy_name]
    options: dict[str, str] = {} 

    pairs = getattr(args, "policy_option", None) 

    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Invalid policy option {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid policy option {item!r}: key cannot be empty")
        options[key] = value 

    return manifest, options 