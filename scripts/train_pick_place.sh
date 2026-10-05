#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT="${GROOT:-$HOME/Isaac-GR00T-N1.7}"
BASE_MODEL="${BASE_MODEL:-$HOME/sim2real/models/nv_so_arm_n17}"
DATASET="${DATASET:-$REPO/datasets/pick_place_v2_gr00t}"
# Optional extra prepared datasets (colon-separated), trained together with DATASET. GR00T samples every dataset
# in proportion to its frame count, so episodes present in two datasets are seen about twice as often.
EXTRA_DATASETS="${EXTRA_DATASETS:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
OUT="${OUT:-$HOME/sim2real/models/so101_pick_place_v2_nvinit}"
BATCH="${BATCH:-16}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
# Optional milestones: hard-link every MILESTONE_STEPS checkpoint into $OUT/milestones, keep the newest
# MILESTONE_KEEP. Pair with frequent SAVE_STEPS and SAVE_TOTAL_LIMIT=1 for a rolling "last" checkpoint.
# MILESTONE_DIR (optional, e.g. on a big slow disk while OUT is on a fast SSD): copy milestones there in a
# detached background process instead of hard-linking into $OUT/milestones.
MILESTONE_STEPS="${MILESTONE_STEPS:-0}"
MILESTONE_KEEP="${MILESTONE_KEEP:-3}"
MILESTONE_DIR="${MILESTONE_DIR:-}"
RESUME="${RESUME:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MEASURED_S_PER_STEP="${MEASURED_S_PER_STEP:-0.3902}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Downloads are explicit setup steps; a missing cached backbone must fail locally.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
# GR00T's inherited test hooks can bypass pretrained weights; real runs clear them.
unset GROOT_SKIP_HF_MODEL_WEIGHTS GROOT_HF_LOCAL_FIRST PYTEST_CURRENT_TEST

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
case "${1:-}" in
    --dry_run) DRY_RUN=1; shift ;;
    --preflight_only) PREFLIGHT_ONLY=1; shift ;;
    --help|-h) printf 'Usage: bash scripts/train_pick_place.sh [--dry_run|--preflight_only]\nConfigure BASE_MODEL, GROOT, DATASET, OUT, BATCH, MAX_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, MILESTONE_STEPS, MILESTONE_KEEP, MILESTONE_DIR, EXTRA_DATASETS, LEARNING_RATE, RESUME and DATALOADER_NUM_WORKERS as environment variables.\n'; exit 0 ;;
esac
[[ $# == 0 ]] || die 'Unknown argument; use --help.'
for setting in BATCH MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || die "$setting must be a positive integer."
done
for setting in RESUME DRY_RUN PREFLIGHT_ONLY; do
    [[ "${!setting}" == 0 || "${!setting}" == 1 ]] || die "$setting must be 0 or 1."
done
[[ "$DATALOADER_NUM_WORKERS" =~ ^(0|[1-9][0-9]*)$ ]] || die 'DATALOADER_NUM_WORKERS must be a non-negative integer.'
[[ "$MILESTONE_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] || die 'MILESTONE_STEPS must be a non-negative integer.'
[[ "$MILESTONE_KEEP" =~ ^[1-9][0-9]*$ ]] || die 'MILESTONE_KEEP must be a positive integer.'
(( MILESTONE_STEPS == 0 || MILESTONE_STEPS % SAVE_STEPS == 0 )) || die 'MILESTONE_STEPS must be a multiple of SAVE_STEPS.'
[[ -z "$MILESTONE_DIR" ]] || (( MILESTONE_STEPS > 0 )) || die 'MILESTONE_DIR needs MILESTONE_STEPS > 0.'
[[ -z "$MILESTONE_DIR" ]] || MILESTONE_DIR="$(realpath -m -- "$MILESTONE_DIR")"
[[ "$LEARNING_RATE" =~ ^[0-9]*\.?[0-9]+([eE]-?[0-9]+)?$ ]] || die 'LEARNING_RATE must be a positive number.'
for setting in GROOT BASE_MODEL OUT DATASET; do
    printf -v "$setting" '%s' "$(realpath -m -- "${!setting}")"
done
datasets=("$DATASET")
IFS=: read -r -a extra <<< "$EXTRA_DATASETS"
for path in "${extra[@]}"; do
    [[ -n "$path" ]] && datasets+=("$(realpath -m -- "$path")")
done
dataset_paths="$(IFS=:; printf '%s' "${datasets[*]}")"
command=("$GROOT/.venv/bin/python" -u "$REPO/scripts/launch_pick_place_n17.py" --groot "$GROOT"
    --milestone-steps "$MILESTONE_STEPS" --milestone-keep "$MILESTONE_KEEP"
    ${MILESTONE_DIR:+--milestone-dir "$MILESTONE_DIR"}
    --base-model-path "$BASE_MODEL" --no-tune-llm --no-tune-visual --no-tune-diffusion-model
    --dataset-path "$dataset_paths" --modality-config-path "$DATASET/so100_config.py"
    --embodiment-tag NEW_EMBODIMENT --num-gpus 1 --output-dir "$OUT"
    --save-steps "$SAVE_STEPS" --save-total-limit "$SAVE_TOTAL_LIMIT" --max-steps "$MAX_STEPS"
    --warmup-ratio 0.05 --weight-decay 1e-5 --learning-rate "$LEARNING_RATE"
    --global-batch-size "$BATCH" --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.02)
[[ "$RESUME" == 0 ]] || command+=(--resume-from-checkpoint)
printf 'Launch command: '; printf '%q ' "${command[@]}"; printf '\n'
python3 - "$MAX_STEPS" "$MEASURED_S_PER_STEP" <<'PY_TIME'
import math, sys
try:
    seconds = float(sys.argv[2])
    assert math.isfinite(seconds) and seconds > 0
except (ValueError, AssertionError):
    sys.exit('MEASURED_S_PER_STEP must be finite and positive.')
print(f'Estimate: {int(sys.argv[1]) * seconds / 3600:.2f} h at {seconds:g} s/step. '
      'Default timing comes from the N1.7 batch-16 smoke; excludes loading and checkpoint saving.')
PY_TIME
printf 'Allocator: %s; log: %s/train.log\n' "$PYTORCH_CUDA_ALLOC_CONF" "$OUT"
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Dry run only; checkpoint, dataset, runtime, disk and GPU checks were not executed.\n'
    exit 0
fi
if [[ ! -d "$BASE_MODEL" ]]; then
    python3 - "$REPO/scripts" "$BASE_MODEL" <<'PY_DOWNLOAD'
import sys
sys.path.insert(0, sys.argv[1])
from gr00t_model_profiles import download_command
print('Official NVIDIA checkpoint missing. Download its root model files manually:')
print(download_command(sys.argv[2]))
print('Cosmos-Reason2-2B also requires access approval on Hugging Face and hf auth login; '
      'cache it before training with: hf download nvidia/Cosmos-Reason2-2B')
PY_DOWNLOAD
    die "Missing BASE_MODEL: $BASE_MODEL"
fi
if compgen -G "$OUT/checkpoint-*" >/dev/null && [[ "$RESUME" != 1 ]]; then
    die "Checkpoints already exist in $OUT. Set RESUME=1 to resume the latest checkpoint."
fi
[[ -x "$GROOT/.venv/bin/python" ]] || die 'N1.7 runtime missing; run bash scripts/setup_gr00t_n17.sh.'
[[ "$(git -C "$GROOT" rev-parse HEAD)" == 51d4c89f72fda44cbf77285c6a8114b52676b8a1 ]] || die 'GR00T N1.7 revision differs from the verified pin.'
python3 "$REPO/scripts/gr00t_model_profiles.py" --checkpoint "$BASE_MODEL"
# Keep relative/empty legacy cache settings anchored to this startup directory.
TRANSFORMERS_CACHE="$(python3 "$REPO/scripts/gr00t_model_profiles.py" --cache-path)"
export TRANSFORMERS_CACHE
python3 "$REPO/scripts/gr00t_model_profiles.py" --cosmos-cache
python3 - "$OUT" "$REPO/scripts" "${datasets[@]}" <<'PY_CHECK'
import hashlib, json, pathlib, shutil, sys
out, scripts, *roots = map(pathlib.Path, sys.argv[1:])
for root in roots:
    report = json.loads((root / 'preparation_report.json').read_text())
    if (report.get('status') != 'passed' or report.get('model_profile') != 'so_arm_n17'
        or report.get('groot_revision') != '51d4c89f72fda44cbf77285c6a8114b52676b8a1'):
        sys.exit(f'Prepare a NEW dataset with --model_profile so_arm_n17 before N1.7 training: {root}')
    if (root / 'so100_config.py').read_bytes() != (scripts / 'so_arm_n17_config.py').read_bytes():
        sys.exit(f'Prepared modality configuration differs from the NVIDIA absolute-joint profile: {root}')
    manifest = json.loads((root / 'prepared_sha256.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            sys.exit('Prepared checksum path leaves dataset root.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            sys.exit(f'Prepared checksum mismatch: {name}' + (f' ({root})' if len(roots) > 1 else ''))
probe = out
while not probe.exists():
    probe = probe.parent
available = shutil.disk_usage(probe).free
# Measured N1.7 checkpoints are 15.66 GiB; rotation saves a third before pruning.
if available < 50 * 1024 ** 3:
    sys.exit(f'Need at least 50 GiB free on the output disk; available {available / 1024**3:.1f} GiB.')
print(f'Output disk available: {available / 1024**3:.1f} GiB', flush=True)
PY_CHECK
# The venv activation file is generated by the separately installed GR00T runtime.
# shellcheck source=/dev/null
source "$GROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="${CUDA_HOME:-$HOME/.local/cuda-12.8}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA compiler missing: $CUDA_HOME/bin/nvcc"
# Permit the small desktop baseline, but reject compute jobs and substantial use.
python - <<'PY'
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET

def query(fields, kind='gpu'):
    result = subprocess.run(
        ['nvidia-smi', '--id=0', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    )
    return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))

memory, utilization = map(int, query('memory.used,utilization.gpu')[0])
busy = []
process_info = subprocess.run(
    ['nvidia-smi', '--id=0', '-q', '-x'], check=True, capture_output=True, text=True,
)
for process in ET.fromstring(process_info.stdout).findall('.//process_info'):
    pid = process.findtext('pid', 'unknown')
    name = process.findtext('process_name', 'unknown')
    used = process.findtext('used_memory', 'unknown').split()[0]
    desktop = name in ('/usr/lib/xorg/Xorg', '/usr/bin/xfce4-session')
    if not desktop or not used.isdigit() or int(used) > 128:
        busy.append(f'PID {pid}: {name} ({used} MiB)')
if busy or memory > 512 or utilization > 10:
    sys.exit('GPU 0 is busy: ' + '; '.join(busy + [f'{memory} MiB used, {utilization}% utilization']))
print(f'GPU 0 available: {memory} MiB used, {utilization}% utilization', flush=True)
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    printf 'PASS: training preflight complete; no training launched.\n'
    exit 0
fi
mkdir -p -- "$OUT"
cd -- "$GROOT"
# N1.7 requires an explicit resume flag; optimizer state stays in checkpoints.
"${command[@]}" 2>&1 | tee -a "$OUT/train.log"
