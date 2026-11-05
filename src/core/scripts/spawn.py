# `spawn [--object NAME]`: ask a running lerobot_agent to put a task object beside the sim gripper's
# fixed jaw tip (sim.generate.object_spawner). Run it in the same container as the agent.

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Send one spawn request and print where the object landed.
def main():
    parser = argparse.ArgumentParser(description="Spawn a task object beside the simulated fixed jaw tip.")
    parser.add_argument("--object", default=None, help="Scene object name (default: the task's first spawnable object).")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--authkey", default="so101-spawn")
    args = parser.parse_args()
    request = Request(
        f"http://{args.host}:{args.port}/spawn",
        data=json.dumps({"cmd": "spawn", "object": args.object}).encode(),
        headers={"Content-Type": "application/json", "X-Spawn-Key": args.authkey},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.load(response)
    except HTTPError as exc:
        parser.exit(1, f"[spawn] {exc.read().decode()}\n")
    except (URLError, TimeoutError, OSError) as exc:
        parser.exit(1, f"[spawn] Cannot reach lerobot_agent at {args.host}:{args.port}: {exc}. "
                    "Start lerobot_agent first and run this in the same container.\n")
    position = ", ".join(f"{v:.3f}" for v in result["position"])
    print(f"[spawn] {result['object']} placed beside the fixed jaw tip at ({position}).")


if __name__ == "__main__":
    main()
