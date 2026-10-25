#!/usr/bin/env bash
# `so101 verify --policy groot_n17 --generic`: check a prepared dataset with the pinned GR00T reader.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SO101_PYTHON:?run this through ./so101 verify (the train container sets SO101_PYTHON)}"
exec env PYTHONPATH="$SO101_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$SO101_PYTHON" "$here/verify_pick_place_training.py" \
    --dataset "$SO101_DATASET" "$@"
