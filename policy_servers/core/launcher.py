from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .contract import load_server_policy
from .discovery import load_manifest
from .server import PolicyServer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a policy server.")

    parser.add_argument("--policy-dir", type=Path, required=True, help="Path to policies/<name>.",)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint passed to policy.load().",)
    parser.add_argument("--host", type=str, default="*", help="Host address to bind to.",)
    parser.add_argument("--port", type=int, default=None, help="Port to bind to; defaults to the manifest's default_port.",)
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to load the policy.",)
    parser.add_argument("--option", action="append", default=None, metavar="KEY=VALUE", help="Policy option; may be specified multiple times.", )
    parser.add_argument("--api-token", type=str, default=None, help="Optional API authentication token.",)
    return parser


def parse_options(pairs: list[str] | None) -> dict[str, str]:
    options: dict[str, str]  = {} 
    if not pairs:
        return options 

    for item in pairs:
        if "=" not in item: 
            raise ValueError(f"Invalid option {item!r}: expected KEY-VALUE")
        key, value = item.split("=",1)

        if not key: 
            raise ValueError(f"Invalid option {item!r}: key cannot be empty")

        options[key] = value 

    return options            

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try :
        mainfest = load_manifest(args.policy_dir) 
    except Exception:
        log.exception("failed to load manifest form %s", args.policy_dir)
        return 1 

    try: 
        options = parse_options(args.option) 
    except Exception as exc: 
        log.error("%s", exc) 
        return 1 

    server = None 
    exit_code = 0 
    try:
        policy = load_server_policy(args.policy_dir) 
        policy.load(checkpoint=args.checkpoint, device=args.device, options=options) 

        port = (args.port if args.port is not None else mainfest.default_port)

        server = PolicyServer(policy=policy, host = args.host, port = port, api_token=args.api_token) 

        log.info("Starting policy '%s' on %s:%s", mainfest.name, args.host, server.port)
        server.run() 

    except KeyboardInterrupt:
        log.info("Policy server interrupted")

    except Exception:
        log.exception("Failed to start the policy server")
        exit_code = 1 
    finally:
        if server is not None:
            server.close() 
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
