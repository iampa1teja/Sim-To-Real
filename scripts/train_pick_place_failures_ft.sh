#!/usr/bin/env bash
# Fine-tune a trained pick-place checkpoint for FT_STEPS more steps on clean demos, oversampling the demos whose
# train-split eval start failed.
#
#   bash scripts/train_pick_place_failures_ft.sh            # build datasets (first run), then train
#   RESUME=1 bash scripts/train_pick_place_failures_ft.sh   # continue after a crash
#
# Two prepared copies of datasets/pick_place_v1 are trained together (EXTRA_DATASETS):
#   CLEAN  = every source demo except BAD_EPISODES
#   FAILED = the clean demos whose start failed in RESULTS (the train split replays each demo's own start)
# GR00T samples each copy in proportion to its frames, so failed demos are seen ~2x as often as the others.
# Both copies are built once and reused; delete them to rebuild from other RESULTS or BAD_EPISODES.
# Env: BASE_MODEL, RESULTS, BAD_EPISODES (3,51,72), FT_STEPS (50000), LEARNING_RATE (5e-5), OUT, MILESTONE_DIR,
#      CLEAN, FAILED, GROOT, plus anything scripts/train_pick_place.sh accepts.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=so101_pick_place_sim_n17_200k_failures_ft
export GROOT="${GROOT:-$HOME/Isaac-GR00T}"
export BASE_MODEL="${BASE_MODEL:-/external_storage/models/so101_pick_place_sim_n17_200k/milestones/checkpoint-200000}"
RESULTS="${RESULTS:-$(ls -d "$REPO"/datasets/hyperparameters/so101_pick_place_sim_n17_200k_milestones_checkpoint-200000_*/ah_8/results.json 2>/dev/null | tail -1)}"
BAD_EPISODES="${BAD_EPISODES:-3,51,72}"
SOURCE="${SOURCE:-$REPO/datasets/pick_place_v1}"
CLEAN="${CLEAN:-$REPO/datasets/pick_place_sim_clean}"
FAILED="${FAILED:-$REPO/datasets/pick_place_sim_failed200k}"
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
[[ -f "$RESULTS" ]] || die "Eval results not found: ${RESULTS:-<none>} (set RESULTS=.../ah_8/results.json)"
[[ "$BAD_EPISODES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'BAD_EPISODES must be comma-separated episode indices.'

# Episodes to exclude from each copy. FAILED keeps only failed, non-bad demos.
read -r exclude_clean exclude_failed failed_list < <(python3 - "$RESULTS" "$SOURCE" "$BAD_EPISODES" <<'PY'
import json, sys
from pathlib import Path
results, source, bad = json.load(open(sys.argv[1])), Path(sys.argv[2]), {int(x) for x in sys.argv[3].split(',')}
if not results.get('complete'):
    sys.exit(f'Incomplete eval results: {sys.argv[1]}')
starts = {s['id']: s for s in results['eval_set']['starts']}
failed = sorted({starts[e['start_id']]['source_episode'] for e in results['episodes']
                 if e['split'] == 'train' and not e['success']} - bad)
total = json.loads((source / 'meta/info.json').read_text())['total_episodes']
if not failed or max(failed) >= total or max(bad) >= total:
    sys.exit('Failed/bad episode indices do not match the source dataset')
csv = lambda xs: ','.join(map(str, sorted(xs)))
print(csv(bad), csv(set(range(total)) - set(failed)), csv(failed))
PY
)
printf 'Base checkpoint: %s\nEval results: %s\nBad demos (dropped): %s\nFailed demos (oversampled): %s\n' \
    "$BASE_MODEL" "$RESULTS" "$exclude_clean" "$failed_list"

prepare() {  # prepare <output> <exclude list>
    local output=$1 exclude=$2
    if [[ -f "$output/preparation_report.json" ]]; then
        local had
        had=$(python3 -c 'import json,sys,os; p=sys.argv[1]; print(",".join(map(str, json.load(open(p)) if os.path.exists(p) else [])))' \
              "$output/excluded_episodes.json")
        [[ "$had" == "$exclude" ]] || die "$output was built with other exclusions (${had:-none}); delete it to rebuild."
        printf 'Reusing %s\n' "$output"
        return
    fi
    printf 'Preparing %s ...\n' "$output"
    (cd "$REPO" && env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 "$GROOT/.venv/bin/python" \
        scripts/prepare_pick_place_gr00t.py --model_profile so_arm_n17 --groot "$GROOT" \
        --source "$SOURCE" --output "$output" --exclude_episodes "$exclude")
}
prepare "$CLEAN" "$exclude_clean"
prepare "$FAILED" "$exclude_failed"

export DATASET="$CLEAN" EXTRA_DATASETS="$FAILED"
export OUT="${OUT:-$HOME/sim2real/models/$RUN}"
export MILESTONE_DIR="${MILESTONE_DIR:-/external_storage/models/$RUN/milestones}"
export MAX_STEPS="${FT_STEPS:-50000}" LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export BATCH="${BATCH:-16}" SAVE_STEPS="${SAVE_STEPS:-1000}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export MILESTONE_STEPS="${MILESTONE_STEPS:-10000}" MILESTONE_KEEP="${MILESTONE_KEEP:-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}" RESUME="${RESUME:-0}"
exec env -u LD_LIBRARY_PATH -u PYTHONPATH bash "$REPO/scripts/train_pick_place.sh" "$@"
