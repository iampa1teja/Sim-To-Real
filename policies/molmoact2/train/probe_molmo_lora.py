"""Run real MolmoAct2 LoRA forward/backward/AdamW updates before full training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2 import processor_molmoact2

from molmo_lora_support import install_training_correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "batch_size": args.batch_size,
              "gradient_accumulation": args.accumulation, "optimizer_updates": 0,
              "losses": [], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "checkpoint": str(args.checkpoint), "dataset": str(args.dataset)}
    report["runtime_versions"] = {name: importlib.metadata.version(name) for name in
                                  ("lerobot", "transformers", "peft", "numpy", "av", "accelerate")}
    report["norm_stats_sha256"] = hashlib.sha256(
        (args.dataset / "meta/molmo_norm_stats.json").read_bytes()).hexdigest()
    report["policy_settings"] = {
        "dtype": "bfloat16", "action_expert_dtype": "float32", "train_mode_vlm": "lora",
        "lora_rank": 64, "lora_alpha": 16, "gradient_checkpointing": True,
        "chunk_size": 30, "n_action_steps": 30, "action_mode": "continuous",
        "joint_signs": [1, -1, 1, 1, 1, 1], "joint_offsets": [0, 90, 90, 0, 0, 0],
        "cameras": ["realsense_rgb", "wrist_cam"], "compile_model": False,
    }
    started = time.perf_counter()
    stage = "processors"
    torch.manual_seed(1984)
    torch.cuda.reset_peak_memory_stats()
    try:
        install_training_correction()
        cameras = ["observation.images.realsense_rgb", "observation.images.wrist_cam"]
        config = MolmoAct2Config(
            checkpoint_path=str(args.checkpoint), device="cuda", dtype=torch.bfloat16,
            train_mode_vlm="lora", gradient_checkpointing=True, compile_model=False,
            chunk_size=30, n_action_steps=30, action_mode="continuous",
            joint_signs=[1, -1, 1, 1, 1, 1], joint_offsets=[0, 90, 90, 0, 0, 0],
            norm_tag="workshop_pick_place_v1_as_real",
            norm_stats_path=str(args.dataset / "meta/molmo_norm_stats.json"),
            normalize_gripper=True, image_keys=cameras,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,)),
                            **{key: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for key in cameras}},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        )
        pre, post = processor_molmoact2.make_molmoact2_pre_post_processors(config)
        dataset = LeRobotDataset("local/pick_place_v1_lora_as_real", root=args.dataset,
                                 delta_timestamps={"action": [i / 30 for i in range(30)]},
                                 video_backend="pyav")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        iterator = iter(loader)
        # Verify the actual frame conversion and normalization before allocating the model.
        raw = next(iterator)
        original_action = raw["action"].clone()
        batch = pre(raw)
        tagged = json.loads((args.dataset / "meta/molmo_norm_stats.json").read_text())
        tagged = tagged["metadata_by_tag"][config.norm_tag]["action_stats"]
        lower, upper = (torch.tensor(tagged[key]) for key in ("q01", "q99"))
        signs = torch.tensor(config.joint_signs)
        offsets = torch.tensor(config.joint_offsets)
        expected = (2 * (original_action * signs + offsets - lower) / (upper - lower) - 1).clamp(-1, 1)
        torch.testing.assert_close(batch["action"][..., :6].cpu(), expected, atol=2e-5, rtol=2e-5)
        decoded = post(batch["action"][..., :6]).cpu()
        expected_degrees = (((expected + 1) / 2) * (upper - lower) + lower - offsets) * signs
        torch.testing.assert_close(decoded, expected_degrees, atol=2e-4, rtol=2e-5)
        report["preprocessing_checks"] = ["action frame before quantile normalization", "inverse action frame round-trip"]
        report["processed_shapes"] = {key: list(value.shape) for key, value in batch.items()
                                      if torch.is_tensor(value)}
        print("PREPROCESSOR_OK", json.dumps(report["processed_shapes"]), flush=True)
        stage = "model_load"
        policy = MolmoAct2Policy(config).cuda().train()
        expert = [(name, p) for name, p in policy.named_parameters() if "action_expert" in name]
        assert expert and all(p.requires_grad for _, p in expert), "Action expert must be fully trainable"
        assert all(p.dtype == torch.float32 for _, p in expert), "Action expert must retain FP32 storage"
        trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
        assert any("lora_" in name for name, _ in trainable), "LoRA adapters missing"
        assert all("lora_" in name or "action_expert" in name for name, _ in trainable), trainable[:2]
        report["parameters"] = {
            "total": sum(p.numel() for p in policy.parameters()),
            "trainable": sum(p.numel() for _, p in trainable),
            "action_expert": sum(p.numel() for _, p in expert),
            "trainable_by_dtype": dict(Counter({str(dtype): sum(p.numel() for _, p in trainable if p.dtype == dtype)
                                               for dtype in {p.dtype for _, p in trainable}})),
        }
        print("PARAMETERS", json.dumps(report["parameters"]), flush=True)
        optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
        optimizer.zero_grad(set_to_none=True)
        tracked = next(p for name, p in expert if p.ndim > 1)
        initial = tracked.detach().cpu().clone()
        stage = "forward_backward"
        for update in range(args.updates):
            tick = time.perf_counter()
            for micro in range(args.accumulation):
                if update or micro:
                    raw = next(iterator)
                    batch = pre(raw)
                loss, metrics = policy(batch)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite loss: {value}")
                report["losses"].append(value)
                (loss / args.accumulation).backward()
                del loss, metrics, batch, raw
                print(f"MICROBATCH {len(report['losses'])} loss={value:.6f} "
                      f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}", flush=True)
            stage = "optimizer_step"
            assert any(p.grad is not None for _, p in expert)
            assert any(p.grad is not None for name, p in trainable if "lora_" in name)
            for name, parameter in trainable:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"Nonfinite gradient: {name}")
            report["gradients_finite"] = True
            optimizer.step()
            torch.cuda.synchronize()
            optimizer.zero_grad(set_to_none=True)
            report["optimizer_updates"] += 1
            print(f"UPDATE {update + 1} seconds={time.perf_counter() - tick:.2f} "
                  f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}", flush=True)
            stage = "forward_backward"
        report["expert_weights_changed"] = not torch.equal(initial, tracked.detach().cpu())
        assert report["expert_weights_changed"], "No expert weight update"
        report["status"] = "passed"
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="cuda_out_of_memory", stage=stage, error=str(exc))
        raise
    except BaseException as exc:
        report.update(status="failed", stage=stage, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("REPORT", args.report, report["status"], flush=True)


if __name__ == "__main__":
    main()
