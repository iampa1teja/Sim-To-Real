#!/usr/bin/env bash
# Explicit dependency installation; checkpoint weights are a separate manual step.
set -euo pipefail
GROOT=${GROOT:-$HOME/Isaac-GR00T-N1.7}
SOURCE_GROOT=${SOURCE_GROOT:-$HOME/Isaac-GR00T}
PIN=51d4c89f72fda44cbf77285c6a8114b52676b8a1
LEROBOT_PIN=c75455a6de5c818fa1bb69fb2d92423e86c70475
dry_run=false
case ${1:-} in
    --dry_run) dry_run=true; shift ;;
    --help|-h) printf 'Usage: bash scripts/setup_gr00t_n17.sh [--dry_run]\nInstalls the pinned N1.7 dependencies into a separate checkout. No model downloads.\n'; exit 0 ;;
esac
[[ $# == 0 ]] || { printf 'Unknown argument; use --help.\n' >&2; exit 1; }
run() { printf '+ '; printf '%q ' "$@"; printf '\n'; if ! "$dry_run"; then "$@"; fi; }
command -v uv >/dev/null || { printf 'Install uv before running this helper.\n' >&2; exit 1; }
export CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda-12.8}
export GIT_LFS_SKIP_SMUDGE=1 UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1
if ! "$dry_run"; then
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || { printf 'CUDA compiler missing: %s/bin/nvcc\n' "$CUDA_HOME" >&2; exit 1; }
    command -v python3.12 >/dev/null || { printf 'A local Python 3.12 interpreter is required.\n' >&2; exit 1; }
    command -v ffmpeg >/dev/null || { printf 'Install an FFmpeg 4–7 runtime before setup.\n' >&2; exit 1; }
    ffmpeg_major=$(ffmpeg -version | sed -n '1s/.*version \([0-9]*\).*/\1/p')
    [[ $ffmpeg_major =~ ^[4-7]$ ]] || { printf 'TorchCodec 0.8 requires FFmpeg 4–7.\n' >&2; exit 1; }
fi
if [[ ! -e "$GROOT" ]]; then
    if git -C "$SOURCE_GROOT" cat-file -e "$PIN^{commit}" 2>/dev/null; then
        run git -C "$SOURCE_GROOT" worktree add --detach "$GROOT" "$PIN"
    else
        run git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT"
        run git -C "$GROOT" checkout --detach "$PIN"
    fi
fi
if ! "$dry_run"; then
    [[ $(git -C "$GROOT" rev-parse HEAD) == "$PIN" ]] || { printf 'N1.7 checkout differs from the required pin; refusing to change it.\n' >&2; exit 1; }
    git -C "$GROOT" diff --quiet HEAD -- pyproject.toml uv.lock || { printf 'Pinned dependency files have local changes; refusing to install a different runtime.\n' >&2; exit 1; }
    cd -- "$GROOT"
else
    printf '# Working directory: %q\n' "$GROOT"
fi
# Use the verified pinned lock without resolving metadata for other platforms.
# --locked inspects the unused aarch64 TorchCodec LFS pointer on this x86 host.
run uv sync --frozen --python 3.12 --project "$GROOT"
# The converter subproject requires Python <3.12, while N1.7 requires 3.12.
# Install its pinned LeRobot utilities without resolving/replacing GR00T's Torch.
run uv pip install --python "$GROOT/.venv/bin/python" --no-deps \
    "lerobot @ git+https://github.com/huggingface/lerobot.git@$LEROBOT_PIN"
run uv pip install --python "$GROOT/.venv/bin/python" 'av==16.0.1'
if ! "$dry_run"; then
    "$GROOT/.venv/bin/python" - <<'PY'
import importlib.util
import pathlib
import torch, torchcodec, transformers, msgpack_numpy, av
from lerobot.datasets.utils import load_info
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
converter = pathlib.Path('scripts/lerobot_conversion/convert_v3_to_v2.py')
spec = importlib.util.spec_from_file_location('conversion_import_check', converter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert torch.__version__.split('+')[0] == '2.9.0', torch.__version__
assert transformers.__version__ == '4.57.3', transformers.__version__
assert torchcodec.__version__ == '0.8.0', torchcodec.__version__
print('PASS: N1.7 runtime, converter and TorchCodec imports; Torch pins preserved.')
PY
fi
printf 'Activate with: source %q/.venv/bin/activate\n' "$GROOT"
printf 'Checkpoint download and Cosmos access are separate steps in docs/05-training.md.\n'
