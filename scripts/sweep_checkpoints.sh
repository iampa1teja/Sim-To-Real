#!/usr/bin/env bash
# Run sweep_action_horizon.py for several checkpoints at once, sharing the GPU.
# Each job gets its own GR00T server, Isaac client and port; results land in the usual
# datasets/hyperparameters/<model>_<checkpoint>_<eval set>_<UTC>/ folders.
#
#   MODEL=so101_pick_place_sim_n17_100k PARALLEL=2 bash scripts/sweep_checkpoints.sh 50000 60000 70000 ...
#
# Env: MODEL (required), MODELS_DIR (~/sim2real/models), PARALLEL (2), HORIZONS (8),
#      EVAL_SET (eval_sets/pick_place_v1_75ep_seed1984.json), BASE_PORT (5555),
#      MIN_FREE_MIB (9000: free VRAM needed before starting another job),
#      START_GAP_S (120: wait after each launch so its VRAM shows up before the next check),
#      EXTRA_ARGS (passed through, e.g. "--episode_length_s 30 --resume").
# Shows live tqdm bars per checkpoint (scripts/sweep_progress.py); job messages go to <logs>/driver.log.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL:?set MODEL to the run directory under MODELS_DIR}"
MODELS_DIR="${MODELS_DIR:-$HOME/sim2real/models}"
PARALLEL="${PARALLEL:-2}"
HORIZONS="${HORIZONS:-8}"
EVAL_SET="${EVAL_SET:-$REPO/eval_sets/pick_place_v1_75ep_seed1984.json}"
BASE_PORT="${BASE_PORT:-5555}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
START_GAP_S="${START_GAP_S:-120}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
(($# > 0)) || { echo "Usage: MODEL=<run> [PARALLEL=2] bash $0 <step> [<step> ...]" >&2; exit 2; }
[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "PARALLEL must be a positive integer" >&2; exit 2; }

if pgrep -f launch_pick_place_n17.py >/dev/null; then
    echo "Training is still running; stop it or wait before sweeping." >&2; exit 1
fi
for step in "$@"; do
    [[ -f "$MODELS_DIR/$MODEL/checkpoint-$step/config.json" ]] || echo "WARNING: $MODELS_DIR/$MODEL/checkpoint-$step missing; that job will fail" >&2
done

LOGS="$REPO/datasets/hyperparameters/parallel_logs/${MODEL}_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS"
echo "Logs: $LOGS (driver messages: $LOGS/driver.log)"
exec 3>&1 >>"$LOGS/driver.log" 2>&1
# Live tqdm bars on the terminal; job start/finish lines go to driver.log.
SWEEP_DRIVER=1 python3 "$REPO/scripts/sweep_progress.py" "$LOGS" "$@" >&3 2>&3 &
monitor=$!
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

declare -A pid_step
job_pids=()
# Count sweep jobs only (the progress monitor is also a background job).
running() { local n=0 p; for p in "${job_pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
index=0
for step in "$@"; do
    # Wait for a free slot, then for enough free VRAM.
    while (( $(running) >= PARALLEL )); do sleep 10; done
    while (( $(running) > 0 && $(free_mib) < MIN_FREE_MIB )); do sleep 10; done
    port=$((BASE_PORT + index)); index=$((index + 1))
    # shellcheck disable=SC2086
    python3 "$REPO/scripts/sweep_action_horizon.py" --models_dir "$MODELS_DIR" --model "$MODEL" \
        --checkpoint "checkpoint-$step" --horizons "$HORIZONS" --eval_set "$EVAL_SET" \
        --port "$port" --shared_gpu $EXTRA_ARGS > "$LOGS/checkpoint-$step.log" 2>&1 &
    pid_step[$!]=$step; job_pids+=($!)
    echo "$(date +%T) started checkpoint-$step (port $port, pid $!, free VRAM $(free_mib) MiB before load)"
    (( index < $# )) && sleep "$START_GAP_S"
done

status=0
for pid in "${job_pids[@]}"; do
    if wait "$pid"; then echo "checkpoint-${pid_step[$pid]}: complete"
    else echo "checkpoint-${pid_step[$pid]}: FAILED (see $LOGS/checkpoint-${pid_step[$pid]}.log)"; status=1; fi
done
touch "$LOGS/.done"; wait "$monitor" 2>/dev/null
cat "$LOGS/driver.log" >&3
exit $status
