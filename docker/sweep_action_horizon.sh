#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_dir/source${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$repo_dir/scripts/sweep_action_horizon.py" "$@"
