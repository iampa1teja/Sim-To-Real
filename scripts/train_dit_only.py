#!/usr/bin/env python3
"""Continue a pinned GR00T N1.7 checkpoint with only action_head.model trainable."""

import argparse
import copy
import csv
import functools
import hashlib
import importlib
import json
import math
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

PIN = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
DEFAULT_INIT = "/external_storage/models/so101_pick_place_sim_n17_200k/milestones/checkpoint-200000"
DEFAULT_OUTPUT = "/external_storage/models/so101_pick_place_sim_n17_200k_dit_only"
DIT = "action_head.model"
GIB = 1024**3


def assert_dit_only(model, allowed=None):
    """Fail closed if any trainable parameter is outside the selected DiT scope."""
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    outside = sorted(name for name in names if not name.startswith(DIT + "."))
    if outside or (allowed is not None and names != set(allowed)):
        raise RuntimeError(
            f"Invalid DiT-only trainable parameters: {outside or sorted(names ^ set(allowed))}"
        )
    if not names:
        raise RuntimeError("No DiT parameters are trainable")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_dit_only(model, last_n=0):
    """Enable the whole DiT, or strictly the last N blocks (other DiT parts frozen)."""
    dit = model.get_submodule(DIT)
    blocks = dit.transformer_blocks
    if last_n < 0 or last_n > len(blocks):
        raise ValueError(f"--train_last_n_blocks must be 0..{len(blocks)} (0 = whole DiT)")
    model.requires_grad_(False)
    selected = (
        list(range(len(blocks) - last_n, len(blocks))) if last_n else list(range(len(blocks)))
    )
    if last_n:
        for index in selected:
            blocks[index].requires_grad_(True)
    else:
        dit.requires_grad_(True)
    allowed = {name for name, p in model.named_parameters() if p.requires_grad}
    assert_dit_only(model, allowed)
    # These flags also keep frozen adapters/encoders in eval mode in upstream forward().
    for target in (model.config, model.action_head.config, model.action_head):
        target.tune_projector = False
        target.tune_diffusion_model = True
        target.tune_vlln = False
    model.config.tune_llm = model.config.tune_visual = False
    print(f"Selected DiT blocks ({len(selected)}/{len(blocks)}): {selected}", flush=True)
    return allowed


def assert_complete_loading(info):
    """Even optional missing tensors must abort this continuation."""
    print(
        f"Checkpoint loading: missing={len(info.get('missing_keys', []))}, "
        f"unexpected={len(info.get('unexpected_keys', []))}, "
        f"mismatched={len(info.get('mismatched_keys', []))}",
        flush=True,
    )
    errors = {
        name: info.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(name)
    }
    if errors:
        raise RuntimeError(f"Incomplete checkpoint weights: {errors}")


def parameter_table(model):
    groups = defaultdict(lambda: {True: 0, False: 0})
    for name, parameter in model.named_parameters():
        pieces = name.split(".")
        depth = (
            4
            if name.startswith(DIT + ".transformer_blocks.")
            else (3 if name.startswith(DIT + ".") else 2)
        )
        groups[".".join(pieces[:depth])][parameter.requires_grad] += parameter.numel()
    print("\nModule path | Parameters | Trainable", flush=True)
    rows = []
    for module, counts in sorted(groups.items()):
        for trainable, count in counts.items():
            if count:
                row = {"module": module, "parameters": count, "trainable": trainable}
                rows.append(row)
                print(f"{module} | {count:,} | {'yes' if trainable else 'no'}", flush=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = assert_dit_only(model)
    print(f"Trainable total: {trainable:,} / {total:,} ({trainable / total:.2%})\n", flush=True)
    return rows


def checkpoint_blocks(dit):
    """The pinned DiT declares checkpoint support but its forward ignores the flag.

    Wrap its existing blocks in-process, using non-reentrant recomputation. No
    upstream files or forward equations change; frozen backbones stay frozen.
    """
    from torch.utils.checkpoint import checkpoint

    for block in dit.transformer_blocks:
        original = block.forward

        @functools.wraps(original)
        def forward(*args, _forward=original, _block=block, **kwargs):
            import torch

            if (
                _block.training
                and torch.is_grad_enabled()
                and any(p.requires_grad for p in _block.parameters())
            ):
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        block.forward = forward
    print("Gradient checkpointing: non-reentrant DiT block recomputation enabled", flush=True)


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def gpu_query(fields, kind="gpu"):
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return list(csv.reader(output.strip().splitlines(), skipinitialspace=True))


def preflight(args, source):
    import yaml

    if git_revision(args.groot) != PIN:
        raise RuntimeError(f"GR00T must remain pinned to {PIN}")
    if subprocess.check_output(
        ["git", "-C", str(args.groot), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(
            "Pinned GR00T checkout has local modifications; use its unchanged checkout"
        )
    if not (source / "config.json").is_file():
        raise RuntimeError(f"Missing checkpoint config: {source}")
    weights = list(source.glob("*.safetensors")) + list(source.glob("pytorch_model*.bin"))
    if not weights:
        raise RuntimeError(f"Missing checkpoint weights: {source}")
    index = source / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (source / shard).is_file() for shard in shards):
            raise RuntimeError("Checkpoint index references missing shards")
    saved_path = args.init_ckpt / "experiment_cfg/config.yaml"
    saved = yaml.safe_load(saved_path.read_text())
    init_step = json.loads((args.init_ckpt / "trainer_state.json").read_text())["global_step"]
    if init_step != 200000:
        raise RuntimeError(f"Expected the 200k init checkpoint, found step {init_step}")
    for dataset in saved["data"]["datasets"]:
        if dataset["dataset_type"] != "physical_embodiment":
            raise RuntimeError("Continuation requires readable local physical-embodiment data")
        for name in dataset["dataset_paths"]:
            root = Path(name)
            if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
                raise RuntimeError(f"Dataset missing or unreadable: {root}")
            json.loads((root / "meta/info.json").read_text())
            if not (root / "so100_config.py").is_file():
                raise RuntimeError(f"Saved dataset modality registration missing: {root}")
            for subtree in (root / "meta", root / "data", root / "videos"):
                if not subtree.is_dir():
                    raise RuntimeError(f"Dataset subtree missing: {subtree}")
                for path in subtree.rglob("*"):
                    if path.is_file() and not os.access(path, os.R_OK):
                        raise RuntimeError(f"Dataset file unreadable: {path}")
    size = sum(path.stat().st_size for path in args.init_ckpt.rglob("*") if path.is_file())
    probe = args.output_dir
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = size * args.save_total_limit + 20 * GIB
    print(
        f"Disk: checkpoint={size / GIB:.2f} GiB, required={required / GIB:.2f} GiB, "
        f"free={free / GIB:.2f} GiB",
        flush=True,
    )
    if free < required:
        raise RuntimeError("Insufficient output disk space")
    gpu, used, utilization = gpu_query("name,memory.used,utilization.gpu")[0]
    processes = gpu_query("pid,process_name,used_gpu_memory", "compute-apps")
    if processes or int(used) > 512 or int(utilization) > 10:
        for process in processes:
            print(f"GPU process: {process}", flush=True)
        raise RuntimeError(
            f"GPU 0 is busy: {used} MiB used, {utilization}% utilization; stop its owner first"
        )
    print(f"GPU available: {gpu}, {used} MiB used", flush=True)
    return saved, init_step, size, gpu


def official_command(args, saved):
    dataset = saved["data"]["datasets"][0]
    command = [
        str(args.groot / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(args.init_ckpt),
        "--dataset-path",
        os.pathsep.join(dataset["dataset_paths"]),
        "--embodiment-tag",
        dataset["embodiment_tag"],
        "--modality-config-path",
        str(Path(dataset["dataset_paths"][0]) / "so100_config.py"),
        "--no-tune-llm",
        "--no-tune-visual",
        "--no-tune-projector",
        "--tune-diffusion-model",
        "--output-dir",
        str(args.output_dir),
        "--global-batch-size",
        str(args.batch),
        "--gradient-accumulation-steps",
        str(args.grad_accum),
        "--max-steps",
        str(args.smoke or args.steps),
        "--learning-rate",
        str(args.lr),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--num-gpus",
        "1",
    ]
    if args.resume:
        command.append("--resume-from-checkpoint")
    return command


def resolved_config(args, saved):
    """Reconstruct the complete saved data/model config; only training controls change."""
    from gr00t.configs.base_config import Config
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
    )
    from gr00t.data.utils import parse_modality_configs

    config = Config().load_dict(copy.deepcopy(saved))
    # Config.save() serializes enum VALUES; ModalityConfig's generic dict parser
    # expects enum NAMES. Rebuild typed actions without changing their meaning.
    for modalities in config.data.modality_configs.values():
        for modality in modalities.values():
            if modality.get("action_configs") is not None:
                modality["action_configs"] = [
                    ActionConfig(
                        **dict(
                            action,
                            rep=ActionRepresentation(action["rep"]),
                            type=ActionType(action["type"]),
                            format=ActionFormat(action["format"]),
                        )
                    )
                    for action in modality["action_configs"]
                ]
    config.data.modality_configs = parse_modality_configs(config.data.modality_configs)
    config.load_config_path = None
    config.model.tune_llm = config.model.tune_visual = config.model.tune_projector = (
        config.model.tune_vlln
    ) = False
    config.model.tune_diffusion_model = True
    training = config.training
    training.start_from_checkpoint = str(args.load_ckpt)
    training.output_dir = str(args.output_dir)
    training.experiment_name = None
    training.max_steps = args.smoke or args.steps
    training.global_batch_size, training.per_gpu_batch_size = args.batch, None
    training.gradient_accumulation_steps = args.grad_accum
    training.learning_rate, training.lr_scheduler_type = args.lr, args.schedule
    training.warmup_steps, training.warmup_ratio = args.warmup, 0.0
    training.save_steps, training.save_total_limit = args.save_steps, args.save_total_limit
    training.resume_from_checkpoint = args.resume
    training.skip_weight_loading = False
    training.num_gpus, training.use_wandb = 1, False
    training.optim = "adamw_torch" if args.optim == "adamw" else "adamw_bnb_8bit"
    training.gradient_checkpointing = args.grad_ckpt
    training.transformers_local_files_only = True
    training.upload_checkpoints = False
    return config


def run_training(args, saved, record, command):
    import torch
    import transformers
    from gr00t.experiment import experiment
    from gr00t.model.gr00t_n1d7 import setup
    from gr00t_offline_cosmos import offline_cosmos_processor

    original_run = experiment.run
    original_create = setup.Gr00tN1d7Pipeline._create_model
    original_load = setup.AutoModel.from_pretrained
    original_trainer = experiment.Gr00tTrainer
    allowed = set()

    def load(*positional, **kwargs):
        kwargs["output_loading_info"] = True
        model, info = original_load(*positional, **kwargs)
        assert_complete_loading(info)
        return model, info

    def create(pipeline):
        model = original_create(pipeline)
        allowed.update(freeze_dit_only(model, args.train_last_n_blocks))
        record["parameter_groups"] = parameter_table(model)
        record["trainable_parameter_count"] = assert_dit_only(model, allowed)
        record["dit_blocks"] = len(model.get_submodule(DIT).transformer_blocks)
        record["checkpoint_loading"] = {"missing": 0, "unexpected": 0, "mismatched": 0}
        record["torch_version"], record["transformers_version"] = (
            torch.__version__,
            transformers.__version__,
        )
        write_json(args.output_dir / "run_config.json", record)
        if args.grad_ckpt:
            checkpoint_blocks(model.get_submodule(DIT))
        return model

    class AuditedTrainer(original_trainer):
        def create_optimizer(self):
            assert_dit_only(self.model, allowed)
            result = super().create_optimizer()
            actual = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            expected = {id(p) for p in self.model.parameters() if p.requires_grad}
            if actual != expected:
                raise RuntimeError(
                    "Optimizer parameter groups differ from the audited DiT parameters"
                )
            print(
                f"Optimizer audited: {record['trainable_parameter_count']:,} DiT parameters",
                flush=True,
            )
            return result

        def train(self, *positional, **kwargs):
            self.add_callback(metrics)
            return super().train(*positional, **kwargs)

    class Metrics(transformers.TrainerCallback):
        def __init__(self):
            self.durations = []
            self.started = None
            self.peak_process_mib = 0
            self.stop = threading.Event()

        def sample(self):
            while not self.stop.wait(0.2):
                try:
                    for pid, _, used in gpu_query(
                        "pid,process_name,used_gpu_memory", "compute-apps"
                    ):
                        if int(pid) == os.getpid() and used.isdigit():
                            self.peak_process_mib = max(self.peak_process_mib, int(used))
                except (ValueError, subprocess.SubprocessError):
                    pass

        def on_train_begin(self, *positional, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            self.worker = threading.Thread(target=self.sample, daemon=True)
            self.worker.start()

        def on_step_begin(self, *positional, **kwargs):
            torch.cuda.synchronize()
            if self.started is None:
                self.started = time.perf_counter()

        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.durations.append(now - self.started)
            self.started = now
            if not args.smoke and len(self.durations) == 50:
                self.report(state.global_step)

        def on_train_end(self, arguments, state, control, **kwargs):
            self.report(state.global_step)

        def report(self, step):
            measured = self.durations[5:] or self.durations
            seconds = sum(measured) / len(measured)
            result = {
                "completed_steps": step,
                "seconds_per_step": seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "peak_process_gib": self.peak_process_mib / 1024,
                "remaining_compute_hours": max(0, args.steps - step) * seconds / 3600,
            }
            write_json(
                args.output_dir / ("smoke_metrics.json" if args.smoke else "training_metrics.json"),
                result,
            )
            print("TRAINING_METRICS " + json.dumps(result), flush=True)

    metrics = Metrics()

    def run(_official_config):
        config = resolved_config(args, saved)
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

        for tag, modalities in config.data.modality_configs.items():
            if MODALITY_CONFIGS.get(tag) != modalities:
                raise RuntimeError(
                    f"Dataset modality registration differs from the saved run: {tag}"
                )
        print(
            "Reusing the full saved data/preprocessing config; launcher overrides training controls only",
            flush=True,
        )
        return original_run(config)

    try:
        with ExitStack() as stack:
            stack.enter_context(offline_cosmos_processor())
            stack.enter_context(patch.object(experiment, "run", run))
            stack.enter_context(patch.object(experiment, "Gr00tTrainer", AuditedTrainer))
            stack.enter_context(patch.object(setup.Gr00tN1d7Pipeline, "_create_model", create))
            stack.enter_context(patch.object(setup.AutoModel, "from_pretrained", load))
            sys.argv = command
            runpy.run_path(command[0], run_name="__main__")
    except torch.cuda.OutOfMemoryError:
        result = {
            "status": "oom",
            "completed_steps": len(metrics.durations),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "peak_process_gib": metrics.peak_process_mib / 1024,
        }
        if args.smoke:
            write_json(args.output_dir / "smoke_metrics.json", result)
        print("TRAINING_METRICS " + json.dumps(result), flush=True)
        raise
    finally:
        metrics.stop.set()
        if hasattr(metrics, "worker"):
            metrics.worker.join(timeout=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot", type=Path, default=Path.home() / "Isaac-GR00T-N1.7")
    parser.add_argument("--init_ckpt", type=Path, default=Path(DEFAULT_INIT))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT))
    for flag, default in (
        ("steps", 20000),
        ("warmup", 1000),
        ("save_steps", 5000),
        ("save_total_limit", 4),
        ("batch", 32),
        ("grad_accum", 1),
        ("train_last_n_blocks", 0),
        ("smoke", 0),
    ):
        parser.add_argument("--" + flag, type=int, default=default)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--schedule", default="cosine", choices=("cosine", "linear", "constant"))
    parser.add_argument("--optim", default="adamw", choices=("adamw", "adamw8bit"))
    for flag in ("grad_ckpt", "dry_run", "resume", "preflight_only"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument(
        "--no_grad_ckpt",
        action="store_false",
        dest="grad_ckpt",
        help="Disable a checkpointing default from train_dit.sh",
    )
    args = parser.parse_args(argv)
    for name in ("groot", "init_ckpt", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (
        any(
            getattr(args, name) <= 0
            for name in ("steps", "save_steps", "save_total_limit", "batch", "grad_accum")
        )
        or any(getattr(args, name) < 0 for name in ("smoke", "warmup", "train_last_n_blocks"))
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error("Invalid steps, batch, warmup, block count or learning rate")
    if args.smoke and args.resume:
        parser.error("Smoke tests use a fresh output; --smoke and --resume cannot be combined")
    if (
        args.output_dir == args.init_ckpt
        or args.output_dir in args.init_ckpt.parents
        or args.init_ckpt in args.output_dir.parents
    ):
        parser.error("Output must be separate from the init checkpoint")
    for name in ("GROOT_SKIP_HF_MODEL_WEIGHTS", "GROOT_HF_LOCAL_FIRST", "PYTEST_CURRENT_TEST"):
        os.environ.pop(name, None)
    os.environ.update(
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.groot))
    os.environ["PYTHONPATH"] = str(args.groot)
    official = args.groot / "gr00t/experiment/launch_finetune.py"
    print(
        "Resolved flags: "
        + json.dumps(
            {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        f"Effective batch: {args.batch} × {args.grad_accum} × 1 GPU = {args.batch * args.grad_accum}",
        flush=True,
    )
    print("\nPinned official fine-tune --help:", flush=True)
    help_text = subprocess.check_output([sys.executable, str(official), "--help"], text=True)
    print(help_text, flush=True)
    import yaml

    saved = yaml.safe_load((args.init_ckpt / "experiment_cfg/config.yaml").read_text())
    command = official_command(args, saved)
    missing_flags = [
        flag for flag in command[1:] if flag.startswith("--") and flag not in help_text
    ]
    if missing_flags:
        raise RuntimeError(f"Flags absent from the pinned launcher --help: {missing_flags}")
    print("Official command: " + shlex.join([sys.executable, *command]), flush=True)
    print("Saved data config: " + json.dumps(saved["data"], indent=2), flush=True)
    print("Saved model/preprocessing config: " + json.dumps(saved["model"], indent=2), flush=True)
    print("Saved training config: " + json.dumps(saved["training"], indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: no model loaded, no training started, no files written", flush=True)
        return
    checkpoints = sorted(
        (
            p
            for p in args.output_dir.glob("checkpoint-*")
            if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints and not args.resume:
        raise RuntimeError("Output already has checkpoints; choose another output or use --resume")
    if not args.resume and (
        (args.output_dir / "config.json").exists()
        or any(args.output_dir.glob("*.safetensors"))
        or any(args.output_dir.glob("pytorch_model*.bin"))
    ):
        raise RuntimeError("Output already contains a model; choose another output")
    if args.resume and not checkpoints:
        raise RuntimeError("--resume needs a checkpoint in output_dir")
    args.load_ckpt = checkpoints[-1] if args.resume else args.init_ckpt
    saved, init_step, size, gpu = preflight(args, args.load_ckpt)
    record = {
        "flags": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "init_checkpoint": str(args.init_ckpt),
        "init_step": init_step,
        "checkpoint_bytes": size,
        "data_mixture": saved["data"]["datasets"],
        "saved_data_sha256": hashlib.sha256(
            json.dumps(saved["data"], sort_keys=True).encode()
        ).hexdigest(),
        "repo_commit": git_revision(Path(__file__).resolve().parents[1]),
        "GROOT_REF": PIN,
        "gpu_name": gpu,
        "pid": os.getpid(),
        "trainable_parameter_count": None,
    }
    if args.resume:
        previous = json.loads((args.output_dir / "run_config.json").read_text())
        for name in ("init_checkpoint", "init_step", "saved_data_sha256", "GROOT_REF"):
            if previous[name] != record[name]:
                raise RuntimeError(f"Resume config differs: {name}")
        for name in (
            "batch",
            "grad_accum",
            "optim",
            "grad_ckpt",
            "train_last_n_blocks",
            "lr",
            "warmup",
            "schedule",
            "steps",
        ):
            if previous["flags"][name] != record["flags"][name]:
                raise RuntimeError(f"Resume training setting differs: {name}")
    if args.optim == "adamw8bit" and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes missing; install with uv pip install --python <pinned venv python> --no-deps bitsandbytes, preserving torch"
        )
    if args.preflight_only:
        print("PASS: preflight only; no training started", flush=True)
        return
    from gr00t_model_profiles import inspect_cosmos_cache, transformers_hub_cache

    os.environ["TRANSFORMERS_CACHE"] = str(transformers_hub_cache())
    record["cosmos_cache"] = inspect_cosmos_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_config.json", record)
    run_training(args, saved, record, command)


if __name__ == "__main__":
    main()
