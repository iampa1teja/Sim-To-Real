#!/usr/bin/env bash
# `so101 prepare --policy groot_n17 --generic`: turn the SO101_* contract into the script's own flags.
# Extra arguments (for example --exclude_episodes 3,7 or --resume) are passed through.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SO101_PYTHON:?run this through ./so101 prepare (the train container sets SO101_PYTHON)}"
"$SO101_PYTHON" "$here/prepare_pick_place_gr00t.py" --groot "$SO101_PROJECT_DIR" --model_profile so_arm_n17 \
    --source "$SO101_DATASET" --output "$SO101_OUTPUT" "$@"
# The preparer refuses existing destinations; only its newly completed copy changes ownership.
if [[ -n ${SO101_HOST_UID:-} && -n ${SO101_HOST_GID:-} ]]; then
    chown -R -- "$SO101_HOST_UID:$SO101_HOST_GID" "$SO101_OUTPUT"
fi
