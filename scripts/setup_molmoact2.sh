#!/usr/bin/env bash
# One-time setup for configs/eval/pick_place_molmo_base.yaml. Run from anywhere on the GPU host:
#   ./scripts/setup_molmoact2.sh [models_dir]
# 1. builds the policy-server image if it is missing (./policy_servers/build.sh)
# 2. downloads the official allenai/MolmoAct2 weights into <models_dir>/molmoact2_base
# 3. builds the molmoact2 uv environment on the policy-cache volume, so the first eval starts fast
# models_dir defaults to /external_storage/molmoact2 (the YAML's policy.models_dir). Set HF_TOKEN if the
# Hub asks for it; SKIP_ENV=1 skips step 3. Re-running resumes or skips finished steps.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_id=${MOLMOACT2_REPO:-allenai/MolmoAct2}
models_dir=${1:-/external_storage/molmoact2}
target=$models_dir/molmoact2_base
image=${SERVER_IMAGE:-policy-server:latest}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }
[[ $# -le 1 && ${1:-} != -* ]] || fail 'Usage: setup_molmoact2.sh [models_dir]'
command -v docker >/dev/null || fail 'Install Docker and the NVIDIA Container Toolkit first.'
docker info >/dev/null 2>&1 || fail 'Start Docker and grant this user Docker access (no sudo is used).'

step "1/3 Policy server image ($image)"
if docker image inspect "$image" >/dev/null 2>&1; then
    echo 'Image already built.'
else
    "$repo_dir/policy_servers/build.sh" "$image"
fi

step "2/3 Weights: $repo_id -> $target"
mkdir -p "$target" 2>/dev/null || fail "Cannot create $target (check permissions on its parent)."
if [[ -f $target/config.json ]] && compgen -G "$target/*.safetensors" >/dev/null && [[ ! -d $target/.cache/huggingface/download ]]; then
    echo 'Weights already present.'
else
    if command -v hf >/dev/null; then
        hf download "$repo_id" --local-dir "$target"
    elif command -v huggingface-cli >/dev/null; then
        huggingface-cli download "$repo_id" --local-dir "$target"
    elif command -v uvx >/dev/null; then
        uvx --from huggingface_hub hf download "$repo_id" --local-dir "$target"
    else
        command -v python3 >/dev/null || fail 'Install huggingface_hub (pip install huggingface_hub) or uv.'
        python3 -m pip install --user -q huggingface_hub
        python3 - "$repo_id" "$target" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
    fi
fi
[[ -f $target/config.json ]] || fail "$target/config.json is missing after the download."
compgen -G "$target/*.safetensors" >/dev/null || fail "No *.safetensors weights in $target."

step '3/3 molmoact2 Python environment'
if [[ ${SKIP_ENV:-0} == 1 ]]; then
    echo 'Skipped (SKIP_ENV=1); the first eval will build it.'
else
    # Same commands as policy_servers/entrypoint.sh, without starting the server.
    docker run --rm --network host \
        -v "$repo_dir/policies:/policies" -v policy-cache:/opt/cache \
        -e UV_PROJECT_ENVIRONMENT=/opt/cache/venvs/molmoact2 \
        --entrypoint bash "$image" -c '
            set -euo pipefail
            if [[ -f /policies/molmoact2/uv.lock ]]; then
                uv sync --frozen --project /policies/molmoact2 --python 3.12
            else
                echo "WARNING: policies/molmoact2/uv.lock missing; resolving fresh (run uv lock there and commit it)." >&2
                uv sync --project /policies/molmoact2 --python 3.12
            fi'
fi

printf '\nMolmoAct2 is ready. Run:\n  ./docker/eval.sh pick_place_molmo_base.yaml --dry_run\n  ./docker/eval.sh pick_place_molmo_base.yaml\n'
