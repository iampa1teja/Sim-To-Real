#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-$HOME/sim2real/models/nv_so_arm_n17}"
command -v uv >/dev/null || { printf 'uv is required.\n' >&2; exit 1; }
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_XET_HIGH_PERFORMANCE=1
uv tool run --from 'huggingface_hub[hf_xet]' hf download \
    nvidia/SO_ARM_Starter_Gr00tN17 \
    --revision 93a5a88f78a395939f784a5fe3685184913dcc60 \
    --include config.json --include processor_config.json \
    --include embodiment_id.json --include statistics.json \
    --include 'model-*.safetensors' --include model.safetensors.index.json \
    --local-dir "$BASE_MODEL"
python3 "$REPO/scripts/gr00t_model_profiles.py" --checkpoint "$BASE_MODEL"
python3 "$REPO/scripts/gr00t_model_profiles.py" --cosmos-cache
