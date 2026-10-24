#!/usr/bin/env bash
# Run only after the memory gate and production trainer/save checks pass.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
storage=${MOLMO_STORAGE:-/external_storage/molmoact2}
run_dir=${MOLMO_PIPELINE_DIR:-$storage/lora_pipeline_20261024}
export MOLMO_OUT=${MOLMO_OUT:-$storage/so101_pick_place_v1_lora_as_real_20261024}
base=${MOLMO_BASE:-$storage/molmoact2_so101}
export_dir=$storage/molmoact2_so101_workshop_lora
mkdir -p "$run_dir"
stage=starting
status() {
    python3 - "$run_dir/status.json" "$1" "$stage" "${2:-0}" <<'PY'
import datetime,json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status":sys.argv[2],"stage":sys.argv[3],
    "exit_code":int(sys.argv[4]),
    "updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
}
resume_args=()
if [[ ${MOLMO_RESUME:-0} == 1 ]]; then
    resume_config=$MOLMO_OUT/checkpoints/last/pretrained_model/train_config.json
    [[ -f $resume_config && ! -e $export_dir ]] || { echo 'Missing resume checkpoint or export already exists' >&2; exit 1; }
    resume_args=(--config_path="$resume_config" --resume=true)
else
    [[ ! -e $MOLMO_OUT && ! -e $export_dir ]] || { echo 'Refusing to overwrite model output' >&2; exit 1; }
fi
python3 - "$storage/lora_probe_20261024/report.json" <<'PY'
import json,os,sys
report=json.load(open(sys.argv[1]))
assert report['status']=='passed' and report['optimizer_updates']>=2
assert report['expert_weights_changed'] and report['gradients_finite']
assert int(os.environ.get('MOLMO_BATCH','1'))==report['batch_size']
assert int(os.environ.get('MOLMO_ACCUMULATION','16'))==report['gradient_accumulation']
PY
# Snapshot the as-real evaluation contract and set an explicit benchmark report.
results_dir=$repo/datasets/eval_results/molmo_lora_20261024
if [[ ! -d $results_dir ]]; then
    if [[ -w $repo/datasets/eval_results ]]; then
        mkdir -p "$results_dir"
    else
        # Earlier simulator runs created the parent as root. Create only this
        # run's new subdirectory; leave previous evaluation artifacts untouched.
        docker run --rm --entrypoint bash -v "$repo/datasets/eval_results:/results" \
            policy-server:latest -c 'mkdir /results/molmo_lora_20261024; chown "$1" /results/molmo_lora_20261024' \
            bash "$(id -u):$(id -g)"
    fi
fi
[[ ! -e $results_dir/benchmark.json && ! -e $results_dir/nine_zones.json ]] || {
    echo 'Refusing to overwrite evaluation results' >&2; exit 1;
}
fix_result_ownership() {
    # The simulator atomically creates reports with mode 0600 as container root.
    docker run --rm --entrypoint bash -v "$results_dir:/results" policy-server:latest -c '
        for name in benchmark.json benchmark.png; do
            if [[ -f /results/$name ]]; then chown "$1" "/results/$name"; fi
        done' bash "$(id -u):$(id -g)"
}
trap 'code=$?; if (( code != 0 )); then fix_result_ownership || true; status failed "$code"; fi' EXIT
python3 - "$repo/configs/eval/pick_place_molmo_so101_lora_as_real.yaml" \
    "$run_dir" "$storage" <<'PY'
import sys,yaml
from pathlib import Path
config=yaml.safe_load(Path(sys.argv[1]).read_text())
config['policy']['models_dir']=sys.argv[3]
config['eval']['results_json']='/workspace/Sim-to-Real-SO-101-Workshop/datasets/eval_results/molmo_lora_20261024/benchmark.json'
root=Path(sys.argv[2])
(root/'nine_zones.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
config.pop('smoke_test',None)
(root/'benchmark.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
PY
stage=training
status running
./docker/molmo_train.sh train "${resume_args[@]}" > "$run_dir/train.log" 2>&1
stage=export
status running
./docker/molmo_train.sh python scripts/export_molmo_lora.py \
    --checkpoint "$MOLMO_OUT/checkpoints/last/pretrained_model" \
    --base "$base" --output "$export_dir" > "$run_dir/export.log" 2>&1
stage=nine_zone_evaluation
status running
EVAL_VERBOSE=1 ./docker/eval.sh "$run_dir/nine_zones.yaml" --smoke-test > "$run_dir/nine_zones.log" 2>&1
python3 - "$run_dir/nine_zones.log" "$results_dir/nine_zones.json" <<'PY'
import json,sys
from pathlib import Path
prefix='[SMOKE_RESULT] '
lines=Path(sys.argv[1]).read_text().splitlines()
report=json.loads(next(line.split(prefix,1)[1] for line in reversed(lines) if prefix in line))
assert report['complete'] and report['overall']['episodes']==9
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
PY
stage=full_150_evaluation
status running
./docker/eval.sh "$run_dir/benchmark.yaml" > "$run_dir/benchmark.log" 2>&1
fix_result_ownership
python3 - "$results_dir/benchmark.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['complete'] and report['overall']['episodes']==150
print(json.dumps({'overall':report['overall'],'by_zone':report['by_zone']},indent=2))
PY
stage=complete
status complete
