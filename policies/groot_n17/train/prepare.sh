#!/usr/bin/env bash
# `s2r prepare --policy groot_n17`: turn the S2R_* contract into the script's own flags.
# Extra arguments (for example --exclude_episodes 3,7 or --resume) are passed through.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${S2R_PYTHON:?run this through s2r prepare (the train container sets S2R_PYTHON)}"
"$S2R_PYTHON" "$here/prepare_pick_place_gr00t.py" --groot "$S2R_PROJECT_DIR" --model_profile so_arm_n17 \
    --source "$S2R_DATASET" --output "$S2R_OUTPUT" "$@"
# The preparer refuses existing destinations; only its newly completed copy changes ownership.
if [[ -n ${S2R_HOST_UID:-} && -n ${S2R_HOST_GID:-} ]]; then
    chown -R -- "$S2R_HOST_UID:$S2R_HOST_GID" "$S2R_OUTPUT"
fi
