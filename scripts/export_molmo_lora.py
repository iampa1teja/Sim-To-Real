"""Merge trained LoRA into an original-HF artifact for the existing Molmo server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = MolmoAct2Config.from_pretrained(args.checkpoint)
    if config.train_mode_vlm != "lora" or config.chunk_size != 30 or config.n_action_steps != 30:
        raise ValueError("Expected the workshop H30 LoRA checkpoint")
    if config.joint_signs != [1, -1, 1, 1, 1, 1] or config.joint_offsets != [0, 90, 90, 0, 0, 0]:
        raise ValueError("Unexpected joint frame")
    # LeRobot clears norm_tag/norm_stats_path when saving: saved processor tensors
    # become authoritative. Export those, independently of mutable dataset files.
    pipeline = json.loads((args.checkpoint / "policy_preprocessor.json").read_text())
    assert any(step["registry_name"] == "workshop_molmo_joint_frame" for step in pipeline["steps"])
    normalizer = next(step for step in pipeline["steps"]
                      if step["registry_name"] == "molmoact2_masked_normalizer")
    stats_path = args.checkpoint / normalizer["state_file"]
    feature_stats = {}
    for name, tensor in load_file(stats_path).items():
        feature, statistic = name.rsplit(".", 1)
        value = tensor.bool().tolist() if statistic == "mask" else tensor.tolist()
        feature_stats.setdefault(feature, {})[statistic] = value
    tag = "workshop_pick_place_v1_as_real"
    stats = {"format": "molmoact2_norm_stats.v1", "norm_mode": "q01_q99", "metadata_by_tag": {tag: {
        "action_key": "action", "state_key": "observation.state",
        "normalize_gripper": config.normalize_gripper,
        "camera_keys": config.image_keys, "action_horizon": 30, "n_action_steps": 30,
        "setup_type": config.setup_type, "control_mode": config.control_mode,
        "action_stats": feature_stats["action"], "state_stats": feature_stats["observation.state"],
    }}}
    config.checkpoint_path = str(args.base.resolve())
    config.device = "cpu"
    config.compile_model = False
    config.enable_inference_cuda_graph = False
    policy = MolmoAct2Policy.from_pretrained(args.checkpoint, config=config).cpu().eval()
    # Keep expert weights exact while merging only the VLM adapter deltas.
    expert_before = {name: tensor.detach().clone() for name, tensor in policy.model.named_parameters()
                     if "action_expert" in name}
    model = policy.model.merge_and_unload(safe_merge=True)
    for name, tensor in model.named_parameters():
        if "action_expert" in name:
            original = next(value for key, value in expert_before.items() if key.endswith(name))
            assert torch.equal(original, tensor), f"Expert changed while merging: {name}"
    del expert_before
    original_index = json.loads((args.base / "model.safetensors.index.json").read_text())
    expected_keys = set(original_index["weight_map"])
    assert set(model.state_dict()) == expected_keys, "Merged keys do not match original HF architecture"
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    # The existing server uses Transformers 4.57 and the original remote-code modules.
    # Retain their original config/processor/tokenizer schema alongside the merged tensors.
    for source in args.base.iterdir():
        if source.is_file() and source.suffix in (".py", ".json", ".jinja"):
            if source.name not in ("model.safetensors.index.json", "norm_stats.json"):
                shutil.copy2(source, args.output / source.name)
    (args.output / "norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    report = {
        "source_checkpoint": str(args.checkpoint.resolve()), "base": str(args.base.resolve()),
        "norm_tag": tag, "saved_normalizer_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "joint_signs": config.joint_signs, "joint_offsets": config.joint_offsets,
        "chunk_size": 30, "n_action_steps": 30,
        "action_expert_preserved_exactly": True, "original_hf_tensor_keys_verified": True,
        "vlm_lora_merged": True, "torch": torch.__version__,
    }
    (args.output / "workshop_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
