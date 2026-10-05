#!/usr/bin/env bash
# Continue the saved 200k run with only its diffusion transformer trainable.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROOT="${GROOT:-$HOME/Isaac-GR00T-N1.7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="${CUDA_HOME:-$HOME/.local/cuda-12.8}"
unset GROOT_SKIP_HF_MODEL_WEIGHTS GROOT_HF_LOCAL_FIRST PYTEST_CURRENT_TEST
[[ -x "$GROOT/.venv/bin/python" ]] || { printf 'Missing pinned GR00T runtime: %s\n' "$GROOT" >&2; exit 1; }
# The Python launcher owns preflight checks and the remaining defaults. It prints
# the resolved flags, official --help and command, then checks GPU/disk/data
# before calling the untouched upstream fine-tune entry point.
# Memory-probe defaults are passed first so explicit user flags take precedence.
# First passing 50-update rung on the 4090: 21.06 GiB sampled process peak,
# 0.9284 s/update. All 32 DiT blocks remain trainable; effective batch is 32.
defaults=(--batch 16 --grad_accum 2 --grad_ckpt --optim adamw8bit)
exec "$GROOT/.venv/bin/python" -u "$REPO/scripts/train_dit_only.py" --groot "$GROOT" "${defaults[@]}" "$@"
