#!/usr/bin/env bash
# Isolated LeRobot MolmoAct2 runtime, sharing existing read-only Torch/CUDA wheels.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
storage=${MOLMO_STORAGE:-/external_storage/molmoact2}
runtime=${MOLMO_RUNTIME:-$storage/lerobot_lora_env}
python_root=${MOLMO_PYTHON_ROOT:-}
if [[ ${1:-probe} != setup && -f $storage/validated_training_runtime.json ]]; then
    mapfile -t validated_runtime < <(python3 - "$storage/validated_training_runtime.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
assert value['status']=='passed'
print(value['runtime'])
print(value['python_root'])
PY
    )
    runtime=${MOLMO_RUNTIME:-${validated_runtime[0]}}
    python_root=${MOLMO_PYTHON_ROOT:-${validated_runtime[1]}}
fi
source_dir=$storage/lerobot_lora_20261024
revision=d3793b7e36522321c1c455f0dc4826db3793dceb
image=${MOLMO_TRAIN_IMAGE:-policy-server:latest}
dataset=${MOLMO_DATASET:-$storage/pick_place_v1_lora_as_real_20261024}
checkpoint=${MOLMO_BASE:-$storage/molmoact2_so101}
command=${1:-probe}
[[ $# == 0 ]] || shift
mounts=(-v "$repo:$repo" -v "$storage:$storage" -v policy-cache:/opt/cache:ro -w "$repo")
if [[ -n $python_root ]]; then
    mounts+=(-v "$python_root:$python_root:ro")
fi
if [[ $command == setup ]]; then
    mkdir -p "$storage"
    if [[ ! -d $source_dir/.git ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "$source_dir"
        git -C "$source_dir" checkout "$revision"
    fi
    [[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'Wrong LeRobot revision' >&2; exit 1; }
    docker run --rm --network host --entrypoint bash "${mounts[@]}" \
        -e UV_CACHE_DIR="$storage/uv-cache" -e UV_NATIVE_TLS=true -e UV_HTTP_TIMEOUT=600 "$image" -c '
        set -euo pipefail
        runtime=$1
        source_dir=$2
        if [[ ! -f $runtime/pyvenv.cfg ]]; then
            uv venv --python /opt/cache/venvs/molmoact2/bin/python --system-site-packages "$runtime"
            echo /opt/cache/venvs/molmoact2/lib/python3.12/site-packages > "$runtime/lib/python3.12/site-packages/molmo_base.pth"
        fi
        uv pip install --python "$runtime/bin/python" --no-deps --index-url https://pypi.org/simple \
            -r "$3/docker/molmo/requirements.txt"
        uv pip install --python "$runtime/bin/python" --no-deps --no-build-isolation -e "$source_dir"
        uv pip freeze --python "$runtime/bin/python" > "${runtime%/*}/lora_runtime_freeze.txt"
        ' bash "$runtime" "$source_dir" "$repo"
    exit
fi
[[ -x $runtime/bin/python ]] || { echo "Run $0 setup first" >&2; exit 1; }
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'Wrong LeRobot revision' >&2; exit 1; }
case $command in
    probe) args=(scripts/probe_molmo_lora.py --dataset "$dataset" --checkpoint "$checkpoint"
                 --report "$storage/lora_probe_20261024/report.json" "$@") ;;
    train)
        batch=${MOLMO_BATCH:-1}
        accumulation=${MOLMO_ACCUMULATION:-16}
        updates=${MOLMO_UPDATES:-10000}
        for value in "$batch" "$accumulation" "$updates"; do
            [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'Batch, accumulation and updates must be positive integers' >&2; exit 1; }
        done
        # LeRobot counts microbatches and steps its scheduler on each microbatch.
        # Express the requested optimizer-update budget and LR schedule in that clock.
        args=(scripts/train_molmo_lora.py
            --dataset.repo_id=local/pick_place_v1_lora_as_real --dataset.root="$dataset"
            --dataset.video_backend=pyav --dataset.image_transforms.enable=true
            --dataset.image_transforms.max_num_transforms=3 --dataset.image_transforms.random_order=false
            '--dataset.image_transforms.tfs={"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[256,256],"scale":[0.9025,0.9025],"ratio":[1.0,1.0]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":"bilinear"}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}'
            --policy.type=molmoact2 --policy.checkpoint_path="$checkpoint"
            --policy.device=cuda --policy.dtype=bfloat16 --policy.train_mode_vlm=lora
            --policy.gradient_checkpointing=true --policy.compile_model=false
            --policy.chunk_size=30 --policy.n_action_steps=30 --policy.action_mode=continuous
            '--policy.joint_signs=[1,-1,1,1,1,1]' '--policy.joint_offsets=[0,90,90,0,0,0]'
            --policy.norm_tag=workshop_pick_place_v1_as_real
            --policy.norm_stats_path="$dataset/meta/molmo_norm_stats.json"
            --policy.normalize_gripper=true
            '--policy.image_keys=["observation.images.realsense_rgb","observation.images.wrist_cam"]'
            --policy.scheduler_warmup_steps="$((200 * accumulation))"
            --policy.scheduler_decay_steps="$((24000 * accumulation))"
            --policy.push_to_hub=false --wandb.enable=false --seed=1984
            --batch_size="$batch" --accelerator.gradient_accumulation.steps="$accumulation"
            --num_workers=0 --steps="$((updates * accumulation))"
            --save_freq="$((2000 * accumulation))" --log_freq="$((10 * accumulation))"
            --env_eval_freq=0 --save_checkpoint=true
            --output_dir="${MOLMO_OUT:-$storage/so101_pick_place_v1_lora_as_real_20261024}"
            "$@") ;;
    python) args=("$@") ;;
    *) echo "Usage: $0 setup | probe [options] | train [LeRobot options] | python <script> [options]" >&2; exit 2 ;;
esac
if [[ $command == train || $command == probe ]]; then
    exec 9>"$storage/.molmo-train.lock"
    flock -n 9 || { echo 'A Molmo training/probe process is already running.' >&2; exit 1; }
fi
mkdir -p "$storage/runtime-cache/hf" "$storage/runtime-cache/torch" "$storage/runtime-cache/triton" "$storage/runtime-cache/inductor" "$storage/runtime-cache/kernels"
exec docker run --rm --gpus all --ipc=host --network host --user "$(id -u):$(id -g)" --entrypoint "$runtime/bin/python" \
    "${mounts[@]}" -e PYTHONUNBUFFERED=1 -e WANDB_MODE=disabled \
    -e PYTHONFAULTHANDLER=1 -e TORCH_SHOW_CPP_STACKTRACES=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME="$storage/runtime-cache/hf" -e TORCH_HOME="$storage/runtime-cache/torch" \
    -e TRITON_CACHE_DIR="$storage/runtime-cache/triton" \
    -e TORCHINDUCTOR_CACHE_DIR="$storage/runtime-cache/inductor" \
    -e PYTORCH_KERNEL_CACHE_PATH="$storage/runtime-cache/kernels" \
    -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
    "$image" "${args[@]}"
