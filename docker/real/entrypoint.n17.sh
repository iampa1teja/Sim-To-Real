#!/usr/bin/env bash
set -euo pipefail
# Preserve standard server flags and every other image command.
if [[ ${1:-} == python && ${2:-} == /Isaac-GR00T/gr00t/eval/run_gr00t_server.py ]]; then
    shift 2
    exec python /opt/so101/launch_gr00t_server_n17.py --groot /Isaac-GR00T "$@"
fi
exec "$@"
