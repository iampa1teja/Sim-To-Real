# 05 — Training

Train from an independent prepared dataset, verify it with the model's real reader, then serve a saved checkpoint.
Run these commands from the repository root. Complete [getting started](01-getting-started.md) first; use
[data generation](08-data-generation.md) or recording to create the source dataset.

The working recipe below is **GR00T N1.7 for six-channel SO-101 pick-place data**, with the recorded
`realsense_rgb` and `wrist_cam` cameras. Its model profile splits five arm values and one gripper value.
The GR00T and Molmo training scripts still contain embodiment/task assumptions. A generic arm or task manifest
does not make those scripts train a different embodiment; write that preparation/modality recipe inside the
policy folder first. Serving and policy-client dimensions are separate contracts.

## Pin the training environment

`policies/groot_n17/policy.toml` pins Isaac-GR00T to
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`. The policy image installs that checkout's frozen lockfile, then
`train/requirements.txt` with `--no-deps` so those additions do not resolve transitive dependencies. Serving and training use separate environments
in the shared `policy-cache` volume: `/opt/cache/venvs/groot_n17` and `groot_n17-train`.
The image supplies CUDA 12.8 build tools; the host needs a compatible NVIDIA driver and Container Toolkit.

```bash
./s2r build train --tag local
./s2r build serve --tag local
```

Weights are a separate setup step. Install the CLI version used to check these commands, then choose host paths
and download the pinned starter checkpoint: `uv tool install 'huggingface_hub[cli]==0.36.2'`. `hf auth login` and approved access to `nvidia/Cosmos-Reason2-2B` are required before
caching that gated backbone. Never place an access token in a manifest or committed config.

```bash
export S2R_DOC_HF_HOME="$PWD/models/hf-cache"
export S2R_DOC_BASE="$PWD/models/so_arm_n17"
mkdir -p "$S2R_DOC_HF_HOME" "$S2R_DOC_BASE"
HF_HOME="$S2R_DOC_HF_HOME" hf auth login
HF_HOME="$S2R_DOC_HF_HOME" hf download nvidia/SO_ARM_Starter_Gr00tN17 \
  --revision 93a5a88f78a395939f784a5fe3685184913dcc60 \
  --local-dir "$S2R_DOC_BASE" \
  --include config.json processor_config.json embodiment_id.json statistics.json \
  'model-*.safetensors' model.safetensors.index.json
HF_HOME="$S2R_DOC_HF_HOME" hf download nvidia/Cosmos-Reason2-2B \
  --revision 9ce19a195e423419c349abfc86fd07178b230561
```

The current offline Cosmos reader resolves the cache's `refs/main`. A commit-only download creates the immutable
snapshot but not that named reference. In the dedicated cache selected above, point it at the downloaded revision:

```bash
python3 - <<'PY_CACHE'
import os
from pathlib import Path
revision = "9ce19a195e423419c349abfc86fd07178b230561"
repository = Path(os.environ["S2R_DOC_HF_HOME"]) / "hub/models--nvidia--Cosmos-Reason2-2B"
assert (repository / "snapshots" / revision).is_dir(), "Download the pinned snapshot first"
(repository / "refs").mkdir(exist_ok=True)
(repository / "refs/main").write_text(revision + "\n")
PY_CACHE
HF_HOME="$S2R_DOC_HF_HOME" python3 policies/groot_n17/train/gr00t_model_profiles.py --cosmos-cache
```

The training recipe runs offline and checks the pinned starter profile and cached Cosmos files. A missing model
fails locally; the training run does not download weights. Keep the cache intact while a run uses it.
The optional `train/setup_gr00t_n17.sh` helper installs a separate **host** checkout/environment; the container
workflow above does not require running that host installer.

## Prepare an independent copy

Choose an existing recorded/generated source and a **new** destination. These are host paths:

```bash
export S2R_DOC_SOURCE="$PWD/datasets/pick_place_scripted"
export S2R_DOC_PREPARED="$PWD/datasets/pick_place_gr00t"
export S2R_DOC_RUN="$PWD/outputs/train/pick_place_gr00t"

./s2r prepare --policy groot_n17 \
  --dataset "$S2R_DOC_SOURCE" --output "$S2R_DOC_PREPARED" \
  --image s2r-train:local

./s2r verify --policy groot_n17 --dataset "$S2R_DOC_PREPARED" \
  --hf-home "$S2R_DOC_HF_HOME" --image s2r-train:local
```

`prepare` mounts the source read-only and builds a new LeRobot v2.1 copy for `NEW_EMBODIMENT`, using absolute
joint actions and the `room`/`wrist` model camera names. It checks all parquet values, preserves video bytes,
decodes every video frame/timestamp, and exercises the pinned GR00T episode loader. It also preserves generic
schema-2 `task_meta/` and legacy `pick_place_meta/` sidecars, including their episode renumbering.

Inspect `preparation_report.json` (`status: passed`, `model_profile: so_arm_n17`) and
`prepared_sha256.json` in the new dataset. `verify` rechecks every prepared checksum and every episode, and
compares first/middle/last frames from both cameras through the real reader. Preparation and verification run
without requesting a GPU.

To omit source episode IDs, choose a different new output directory and pass the preparer's own options after `--`:

```bash
./s2r prepare --policy groot_n17 \
  --dataset "$S2R_DOC_SOURCE" --output "$PWD/datasets/pick_place_gr00t_filtered" \
  --image s2r-train:local -- --exclude_episodes 3,7
```

Only list IDs that exist in your source. The retained episodes are renumbered and their statistics recomputed;
the report retains the source-to-output mapping. An interrupted preparation can use the same command plus
`--resume`, with unchanged source hashes and exactly the same exclusions. A completed output is never a target
for a second preparation.

## Fine-tune

The training container mounts the prepared dataset and base model read-only, and the run output writable.
The recipe checks prepared hashes again, requires at least 50 GiB free on the output filesystem, and refuses
a GPU with another compute workload. Coordinate GPU availability before starting. A two-step pipeline check is:

```bash
./s2r train --policy groot_n17 \
  --dataset "$S2R_DOC_PREPARED" --output "$S2R_DOC_RUN" \
  --base-model "$S2R_DOC_BASE" --hf-home "$S2R_DOC_HF_HOME" \
  --image s2r-train:local -- \
  --batch 1 --max-steps 2 --save-steps 2 --save-total-limit 1
```

This preserves the recipe's 5% warmup: with two total steps, the first optimizer step has zero learning rate
and the second can change weights. The recipe freezes the language model, vision backbone and diffusion model;
its remaining trainable modules follow the pinned GR00T configuration. Two steps still load the full model and
save a full checkpoint. They establish an optimizer/pipeline check, not learned task performance.

To inspect the exact Docker command without executing it, add `--dry-run` before `--`. To print the inner
recipe's resolved launch command without loading weights, checking the GPU or training, pass `-- --dry-run`
after the image/options instead. The latter still starts the train container and installs/checks its environment.

For longer runs, choose the batch/step/save settings deliberately; the recipe defaults are batch 16,
10,000 steps and a checkpoint every 1,000 steps. It refuses an output already containing checkpoints unless
its explicit `RESUME=1` recipe environment is configured. Use a new run directory for the commands in this guide.

A completed run has `train.log` and `checkpoint-<step>/trainer_state.json`, `optimizer.pt`, model weights,
and copied processor/experiment configuration. Confirm finite loss, a nonzero completed global step and a real
weight update before accepting a training smoke check. The repository's Phase 5 check verified step 2,
finite/nonzero Adam moments and a trained projector tensor changed from the immutable base after accounting for
dtype conversion. Evaluate any quality claim on the fixed benchmark in [evaluation](06-evaluation.md).

### An existing Hugging Face snapshot cache

Snapshot weight files may be relative symlinks into `../../blobs`. Mount the **whole model-cache folder** and
select its snapshot inside the container, rather than mounting a snapshot directory alone:

```bash
export S2R_DOC_MODEL_CACHE="$S2R_DOC_HF_HOME/hub/models--nvidia--SO_ARM_Starter_Gr00tN17"
./s2r train --policy groot_n17 \
  --dataset "$S2R_DOC_PREPARED" --output "$S2R_DOC_RUN" \
  --base-model "$S2R_DOC_MODEL_CACHE" --hf-home "$S2R_DOC_HF_HOME" \
  --image s2r-train:local -- \
  --base-model-path /data/base_model/snapshots/93a5a88f78a395939f784a5fe3685184913dcc60 \
  --batch 1 --max-steps 2 --save-steps 2 --save-total-limit 1
```

Use this form **instead of** the flat `--local-dir` model form, with a new run output. The immutable revision is
the same in both cases.

## Serve a saved checkpoint

Select a complete `checkpoint-<step>` directory, which includes the processor and experiment metadata copied
by the checkpoint callback. The serving CLI takes a checkpoint **relative to the host models directory**:

```bash
./s2r serve --policy groot_n17 \
  --models-dir "$S2R_DOC_RUN" --checkpoint checkpoint-2 \
  --hf-home "$S2R_DOC_HF_HOME" --image s2r-serve:local \
  --port 5555 --option embodiment_tag=NEW_EMBODIMENT --foreground
```

In another terminal, use [evaluation](06-evaluation.md) with the same policy/port and camera mapping:
`realsense_rgb` → `room`, `wrist_cam` → `wrist`. Keep the fixed eval set and its geometry/calibration checks.
Stop the foreground server with Ctrl-C when finished. The server is a separate process; evaluation does not load
its checkpoint. `--option` configures the server; evaluator `--policy_option` configures the client.

## Paths and the training plugin contract

| Host CLI input | Container path | Access |
| --- | --- | --- |
| repository | `/workspace/repo` | read-only |
| `policies/` | `/policies` | writable for dependency locks |
| `--dataset` | `/data/dataset` | read-only |
| `--output` for train | `/data/output` | writable |
| `--output` for prepare | `/data/output/<new-folder>` | writable parent permits atomic publication |
| `--base-model` | `/data/base_model` | read-only |
| `--hf-home` for prepare/verify/train | `/root/.cache/huggingface` | read-only |

Paths passed **after `--`** are script/container paths; `s2r` does not translate them. The script receives
`S2R_POLICY_DIR`, `S2R_DATASET`, `S2R_OUTPUT`, `S2R_BASE_MODEL`, `S2R_PYTHON` and `S2R_PROJECT_DIR`.
Declare optional `prepare`, required `train`, and optional `verify` script paths under `[train.commands]`;
training-only dependencies belong in `train/requirements.txt`. See [policy plugins](07-policy-plugins.md#training-commands).

## Retained MolmoAct2 recipe

Molmo preparation remains specific to normalized six-joint SO-101 data. It reads
the source's saved mapping; an explicit `--unit` is required for legacy data
without that mapping and must agree when both are supplied. It no longer reads
the removed `real_setup.json` or substitutes default joint spans.

```bash
./s2r prepare --policy molmoact2 \
  --dataset "$PWD/datasets/pick_place_scripted" \
  --output "$PWD/datasets/pick_place_molmo" --image s2r-train:local
```

This converts body values to the recipe's centred-degree frame, preserves the
gripper values, and computes model-frame quantiles with the existing signs and
offsets. Videos and named-object/annotation sidecars are independent byte-identical
copies. The output alignment contains the converted codec; original alignment
and generation reports are retained under `meta/source_*.json`. Do not treat the
converted values as ordinary normalized LeRobot actions.

The train wrapper forwards explicit LeRobot arguments unchanged. Without an
explicit `--config_path`, mounted `S2R_DATASET`, `S2R_OUTPUT` and `S2R_BASE_MODEL`
provide defaults for dataset root, output directory and policy checkpoint;
explicit flags win. The prepared normalization path/tag are supplied from the
selected dataset. Configure the model's LoRA mode, joint-frame signs/offsets,
camera inputs and training hyperparameters for your checkpoint. A complete
explicit LeRobot config owns its defaults, including resume, instead of being
overridden by these environment values.

Molmo's `verify` is an actual forward/backward optimizer probe, unlike GR00T's
CPU data verification. It requires a compatible Molmo checkpoint and a coordinated
GPU window. `--gpu` explicitly requests GPU access for this command:

```bash
./s2r verify --gpu --policy molmoact2 \
  --dataset "$PWD/datasets/pick_place_molmo" \
  --base-model /absolute/host/path/to/molmo-checkpoint \
  --output "$PWD/outputs/molmo-probe" --image s2r-train:local -- \
  --batch-size 1 --accumulation 1 --updates 2
```

Replace the checkpoint placeholder with actual compatible weights. The report
defaults to `verification.json` inside the output. Ordinary `s2r verify` remains
CPU-only unless `--gpu` is supplied. The restructure validated Molmo preparation
and argument contracts; the completed Phase 5 optimizer acceptance used GR00T.
It does not establish Molmo fine-tuning convergence or broaden either recipe to
arbitrary arms/tasks.

## CPU preparation regression tests

With the host test environment installed, run:

```bash
PYTHONPATH=src uv run --group host python -m pytest \
  tests/policies/test_groot_preparation.py -q
```

These check schema-2/legacy sidecars and bounded training-flag forwarding without loading a model. The real
prepare/verify commands above remain the data-reader acceptance checks. See [phase status](restructure-status.md)
for the recorded physical generation and two-step training evidence.
