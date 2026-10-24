"""Check an exported checkpoint in the unchanged inference runtime (no rollout)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from policies.molmoact2.server import create

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
policy = create()
tag = "workshop_pick_place_v1_as_real"
policy.load(str(args.checkpoint), "cuda", {
    "norm_tag": tag, "dtype": "bfloat16", "action_dtype": "float32", "num_steps": 10,
})
policy.reset(seed=1984)
response = policy.infer({
    "state": np.array([0, 90, 90, 0, 0, 40], dtype=np.float32),
    "images": {camera: np.zeros((480, 640, 3), dtype=np.uint8) for camera in ("room", "wrist")},
    "instruction": "Pick up the blue cube and place it in the white box",
})
actions = np.asarray(response["actions"])
assert actions.shape == (30, 6) and np.isfinite(actions).all()
stats = json.loads((args.checkpoint / "norm_stats.json").read_text())["metadata_by_tag"][tag]["action_stats"]
assert np.all(actions >= np.array(stats["q01"]) - 1e-3)
assert np.all(actions <= np.array(stats["q99"]) + 1e-3)
report = {
    "status": "passed", "checkpoint": str(args.checkpoint), "server_info": policy.info(),
    "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype),
    "two_stub_images": True, "task_performance_tested": False,
    "fresh_quantile_bounds_verified": True,
    "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
}
args.report.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
