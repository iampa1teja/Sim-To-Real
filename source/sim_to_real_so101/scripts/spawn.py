"""Spawn one blue 2 cm cube just beside the fixed jaw tip."""
import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=6001)
    parser.add_argument('--authkey', default='so101-spawn')
    args = parser.parse_args()
    request = Request(
        f'http://{args.host}:{args.port}/spawn',
        data=b'{"cmd":"spawn_blue_cube"}',
        headers={'Content-Type': 'application/json', 'X-Spawn-Key': args.authkey},
        method='POST',
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.load(response)
    except HTTPError as exc:
        parser.exit(1, f'[spawn] {exc.read().decode()}\n')
    except (URLError, TimeoutError, OSError) as exc:
        parser.exit(1, f'[spawn] Cannot reach lerobot_agent at {args.host}:{args.port}: {exc}. '
                    'Start/restart lerobot_agent first and run this in the same container.\n')
    print(f"[spawn] Created {result['prim_path']}: blue 2 cm cube just beside the fixed jaw tip.")


if __name__ == '__main__':
    main()