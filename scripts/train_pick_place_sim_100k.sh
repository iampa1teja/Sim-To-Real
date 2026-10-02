#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GROOT="${GROOT:-$HOME/Isaac-GR00T}"
export BASE_MODEL="${BASE_MODEL:-$HOME/sim2real/models/nv_so_arm_n17}"
export DATASET="${DATASET:-$REPO/datasets/pick_place_sim}"
export OUT="${OUT:-$HOME/sim2real/models/so101_pick_place_sim_n17_100k}"
export BATCH="${BATCH:-16}" MAX_STEPS="${MAX_STEPS:-100000}"
export SAVE_STEPS="${SAVE_STEPS:-10000}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
exec bash "$REPO/scripts/train_pick_place.sh" "$@"
