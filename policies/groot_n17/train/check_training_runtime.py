#!/usr/bin/env python3
"""Verify the container's pinned training stack without loading model weights."""
import importlib.util
from pathlib import Path
import subprocess

import av
import bitsandbytes
import torch
import torchcodec
import transformers
from lerobot.datasets.utils import load_info  # noqa: F401  (import check only)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: F401  (import check only)

root = Path('/Isaac-GR00T')
revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '51d4c89f72fda44cbf77285c6a8114b52676b8a1', revision
assert torch.__version__.split('+')[0] == '2.9.0', torch.__version__
assert transformers.__version__ == '4.57.3', transformers.__version__
assert torchcodec.__version__ == '0.8.0', torchcodec.__version__
assert torch.cuda.is_available(), 'CUDA is unavailable inside the training container'
converter = root / 'scripts/lerobot_conversion/convert_v3_to_v2.py'
spec = importlib.util.spec_from_file_location('training_converter_check', converter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f'PASS: GR00T {revision}; Torch {torch.__version__}; Transformers {transformers.__version__}; '
      f'TorchCodec {torchcodec.__version__}; PyAV {av.__version__}; bitsandbytes {bitsandbytes.__version__}; '
      f'GPU {torch.cuda.get_device_name(0)}; converter and dataset reader imported.')
