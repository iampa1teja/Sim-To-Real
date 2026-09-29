# 05 — Prepare data and train GR00T

[Guide hub](../README.md) · Previous: [Recording](04-recording.md) · Next: [Evaluation](06-evaluation.md)

## Pin the training environment

The supported preparation/server revision is
`ead52833afbbf4243f8cd5e7664f48a94de03b19` of NVIDIA/Isaac-GR00T. The Dockerfiles
pin this revision too. Consult its [installation guide](https://github.com/NVIDIA/Isaac-GR00T/blob/ead52833afbbf4243f8cd5e7664f48a94de03b19/README.md)
for platform dependencies. Training needs a CUDA GPU; preparation/verification
are CPU checks. They require LeRobot's v3 converter dependencies, PyArrow, PyAV,
NumPy, FFmpeg and the pinned GR00T reader/config dependencies in the executing
Python environment. The GR00T training uv environment alone should not be assumed
to provide the converter's separate LeRobot environment.

**Training machine**, with Git and uv installed, create a fresh GR00T checkout:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout ead52833afbbf4243f8cd5e7664f48a94de03b19
git submodule update --init --recursive
uv sync
uv pip install -e .
source .venv/bin/activate
```

Use a full checkout of this workshop and your complete dataset on this machine.
The teleop Docker mount recipe exposes `source/`, not the top-level preparation
scripts. Do not replace Isaac's pinned Torch/NumPy packages with training packages
inside its simulation environment to try to combine incompatible installations.

## Prepare an independent copy

[prepare_pick_place_gr00t.py](../scripts/prepare_pick_place_gr00t.py) reads a v3
recording and creates a v2.1 training copy without the upstream converter's
source-directory swap. It requires the two RGB features
`observation.images.realsense_rgb` and `observation.images.wrist_cam`, six state
and six action values, the recorder's complete per-episode videos and cube
sidecars. This is not a general converter for the paired real camera names.
Encode all saved episodes first.

**Training machine**, workshop root, in a Python environment with both conversion
and reader dependencies. Replace the checkout/dataset paths:

```bash
python scripts/prepare_pick_place_gr00t.py \
  --groot '<path_to_Isaac-GR00T>' \
  --source '<datasets_dir>/pick_place_v1' \
  --output '<datasets_dir>/pick_place_v1_gr00t'
PYTHONPATH='<path_to_Isaac-GR00T>' python scripts/verify_pick_place_training.py \
  --dataset '<datasets_dir>/pick_place_v1_gr00t'
```

To exclude failed source episodes, use a **different output path**. The following
is an example only; use the failure IDs you actually reviewed.

**Training machine**, workshop root:

```bash
python scripts/prepare_pick_place_gr00t.py \
  --groot '<path_to_Isaac-GR00T>' \
  --source '<datasets_dir>/pick_place_v1' \
  --output '<datasets_dir>/pick_place_v1_gr00t_filtered' \
  --exclude_episodes 0,6
PYTHONPATH='<path_to_Isaac-GR00T>' python scripts/verify_pick_place_training.py \
  --dataset '<datasets_dir>/pick_place_v1_gr00t_filtered'
```

| Preparation flag | Default / behavior |
| --- | --- |
| `--groot` | Required checkout; HEAD must equal the pin |
| `--source` | `datasets/pick_place_v1` |
| `--output` | `datasets/pick_place_v1_gr00t`; refuses completed output overwrite |
| `--exclude_episodes` | None; comma-separated nonnegative source IDs; duplicates deduplicated; unknown/all-excluded rejected |
| `--resume` | Retry an interrupted output `.building` directory with unchanged source hashes and the same exclusion selection |

The verifier's only dataset argument is `--dataset`, defaulting to
`datasets/pick_place_v1_gr00t`. Both scripts also accept argparse help.

Retained episodes stay in original order and are renumbered contiguously. The
copy rewrites parquet `episode_index` and global `index`, metadata, totals/split
boundaries and sidecar filenames/IDs. Videos remain byte-identical under their
new numbers; other parquet values are checked against the source. Unused tasks
are removed without changing retained task IDs. The filtered report's `excluded`
and `index_map` map new IDs back to original recordings.

GR00T reads episodes in JSONL order and uses IDs with `info.json` path templates.
Filtered preparation never copies source global statistics and removes existing
stats on resume before generating `meta/stats.json` and `meta/relative_stats.json`
from retained parquet. Per-episode statistics remain valid apart from shifted
index statistics. With no exclusions, the original preparation behavior is kept,
including copying valid source global stats. The copied SO100 config registers
`NEW_EMBODIMENT`, with relative single-arm actions and absolute gripper actions.

Preparation preserves source SHA-256 snapshots, compares parquet values, hashes
video copies, decodes every frame and checks timestamps/dimensions, runs the real
GR00T loader for every episode, and writes `preparation_report.json`,
`source_sha256.json`, `prepared_sha256.json`. The independent verifier checks that
manifest and compares first/middle/last images in each camera/episode between
GR00T FFmpeg decoding and PyAV (maximum permitted RGB difference 2). Do not modify
the prepared output after signing its manifest; make a new preparation instead.

## Fine-tune

The command below uses the actual Tyro fields from the pinned
[FinetuneConfig](https://github.com/NVIDIA/Isaac-GR00T/blob/ead52833afbbf4243f8cd5e7664f48a94de03b19/gr00t/configs/finetune_config.py).
The launcher's underscore spellings are accepted by Tyro. The output name is an
example. Choose the prepared dataset's copied config, not an unrelated modality
file. Authenticate for model downloads if necessary; never paste tokens into Git.

**Training machine**, activated GR00T environment, GR00T checkout root:

```bash
hf auth login
export CUDA_VISIBLE_DEVICES=0
python gr00t/experiment/launch_finetune.py \
  --base_model_path nvidia/GR00T-N1.6-3B \
  --dataset_path '<datasets_dir>/pick_place_v1_gr00t_filtered' \
  --modality_config_path '<datasets_dir>/pick_place_v1_gr00t_filtered/so100_config.py' \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir ./outputs/so101_pick_place_v1 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --max_steps 10000 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --learning_rate 1e-4 \
  --global_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 2
```

| Flag in the command | Meaning; code default when omitted |
| --- | --- |
| `--base_model_path` | Required pretrained model/repo path |
| `--dataset_path` | Required prepared dataset root |
| `--modality_config_path` | Python config registration; default `None` uses registered config |
| `--embodiment_tag` | Required embodiment enum; must match config registration |
| `--num_gpus` | Number of training GPUs; 1 |
| `--output_dir` | Checkpoint/log directory; `./outputs` |
| `--save_steps` | Checkpoint interval; 1000 |
| `--save_total_limit` | Retained checkpoint count; 5; older checkpoints may be removed |
| `--max_steps` | Training optimizer-step budget; 10000 |
| `--warmup_ratio` | Warmup fraction; 0.05 |
| `--weight_decay` | Optimizer decay; `1e-5` |
| `--learning_rate` | Initial learning rate; `1e-4` |
| `--global_batch_size` | 64 by default; example lowers to 4 for initial memory testing |
| `--gradient_accumulation_steps` | 1; accumulation passed separately to Trainer |
| `--dataloader_num_workers` | 2; data-loader workers |

**Batch-size implementation detail:** the config description calls the global
batch effective across accumulation, but the pinned `experiment.py` sets
per-device batch to `global_batch_size / num_gpus` and passes accumulation
separately. Thus increasing accumulation without lowering the global batch does
not reduce per-forward memory, and increases effective samples per update.
Global batch must be divisible by the GPU count. Start with a short measured
run before estimating a full job.

### Other supported fine-tuning options

| Flags | Defaults and purpose |
| --- | --- |
| `--tune_llm`, `--tune_visual` | false; backbone tuning switches |
| `--tune_projector`, `--tune_diffusion_model` | true; projector/action decoder tuning |
| `--state_dropout_prob` | 0.0; input state dropout |
| `--random_rotation_angle` | `None`; optional image rotation |
| `--color_jitter_params` | `None`; optional brightness/contrast/saturation/hue dictionary; see config for inherited augmentation behavior |
| `--extra_augmentation_config` | `None`; JSON augmentation config, including masks when supplied |
| `--experiment_name` | `None`; optional run name |
| `--wandb_project`, `--use_wandb` | `finetune-gr00t-n1d6`, false; experiment logging |
| `--shard_size` | 1024; dataset preloading shard size |
| `--episode_sampling_rate` | 0.1; episode sampling rate |
| `--num_shards_per_epoch` | 100000; preloading/sampling configuration |

Consult the pinned launcher's help for Boolean Tyro negation syntax rather than
using argparse-style `true`/`false` values from another program.

### Small datasets and a 24 GB GPU

Ten thousand steps is the launcher default, not a promise of convergence. Evaluate
several saved checkpoints with the same starts, seed and instructions; the lowest
training loss need not give the best placement rate. Keep held-out starts/scenes
for generalization checks. With 50 demos, repeated epochs can overfit appearance
and trajectory habits; collect failures rather than only extending training.

For 24 GB, batch 4 above is a starting memory probe, **not a validated fit**. Reduce
`--global_batch_size` further on OOM, preserve divisibility, and keep default
frozen language/visual backbones. Do not train beside Isaac simulation on an
already full GPU. Distinguish GPU OOM from host RAM pressure caused by data loading.
The course gives multi-hour training expectations, but this repo has no measured
24 GB pick-place throughput; time a run on your actual GPU and estimate from
observed steps/second. See [NVIDIA training guidance](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html).

**Training machine**, after the chosen checkpoint exists, copy it to a model
folder for transfer/mounting (example selects step 10000):

```bash
mkdir -p ~/sim2real/models/so101_pick_place_v1
cp -a outputs/so101_pick_place_v1/checkpoint-10000 ~/sim2real/models/so101_pick_place_v1/
```

On a separate robot host, transfer this complete checkpoint directory to the
host models directory. It must contain `config.json`, weights and the other
checkpoint artifacts, not only a single weight file. The real-container mount
uses `~/sim2real/models`; the host evaluator defaults to `~/models`, so pass
`--models_dir` explicitly as shown in [Evaluation](06-evaluation.md).

## Optional Hub upload

`lerobot_push_dataset` is declared in this fork's pyproject but points to a
nonexistent `main` function. The existing module's `__main__` block works around
that entry-point mismatch; do not present the console command as working.
Its flags are `--repo-id`, `--root` (both default `None`), `--private` (false), and
`--tags` (one or more strings, default `None`). This differs from recorder
`--repo_id` spelling. Inspect the upload scope before publishing any data.

**Teleop container**, after encoding and reviewing the source dataset:

```bash
hf auth login
python -m sim_to_real_so101.scripts.lerobot_push_dataset \
  --repo-id '<hf_user>/pick_place_v1' \
  --root /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --private --tags so101 pick-place
```

The upload helper catches dataset initialization errors and returns after printing
an error; an exit code alone is not proof of upload. Source v3 and prepared v2.1
copies should have distinct identities if shared. Use `<your_hf_token>` only in
your private environment if authentication requires it; do not put tokens in docs.

## Optional extensions and future work

Paired sim/real recording exists. A turnkey merger/GR00T preparation path for
those two differently named camera datasets is not supplied. The pinned launcher
accepts a single dataset path; a custom multi-dataset configuration would be an
additional implementation. See the course's
[co-training section](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html)
for the broader workflow, not a claim that this fork already automates it.

Training-time colour jitter exists via `--color_jitter_params`. Regenerating
pick-place episodes by replaying states/actions through recoloured simulation
is future work: this fork has no tracked dataset-producing replay/augmentation
tool. External LeRobot replay and GR00T ReplayPolicy are not that pipeline.

## CPU preparation regression tests

**Training machine**, workshop root with the preparation dependencies:

```bash
GROOT_CHECKOUT='<path_to_Isaac-GR00T>' \
  python -m unittest discover -s tests -p test_prepare_pick_place_gr00t.py -v
```

The synthetic five-episode suite excludes 0 and 3 and tests renumbering, values,
video bytes, sidecars, stats, unchanged source hashes, actual GR00T reads and
resume. `PREPARATION_BASELINE` optionally names a saved pre-change script for a
default-output byte comparison; only provenance/variable report fields differ.
