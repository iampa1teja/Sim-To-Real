#!/usr/bin/env bash
# `s2r verify --policy groot_n17 --generic`: check a prepared dataset with the pinned GR00T reader.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${S2R_PYTHON:?run this through s2r verify (the train container sets S2R_PYTHON)}"
exec env PYTHONPATH="$S2R_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$S2R_PYTHON" "$here/verify_pick_place_training.py" \
    --dataset "$S2R_DATASET" "$@"
