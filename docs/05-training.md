# 05 — Prepare data and train GR00T

[Guide hub](../README.md) · Previous: [Recording](04-recording.md) · Next: [Evaluation](06-evaluation.md)

## Pin the training environment

The v2 workflow uses NVIDIA's official
[SO_ARM_Starter_Gr00tN17](https://huggingface.co/nvidia/SO_ARM_Starter_Gr00tN17)
checkpoint at revision `93a5a88f78a395939f784a5fe3685184913dcc60`, with GR00T N1.7
GA code pinned to `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. Keep a separate
checkout at `~/Isaac-GR00T-N1.7`; its Python 3.12, Torch, Transformers and video
reader differ from the old N1.6 environment. The setup helper installs the locked
N1.7 environment and the converter's extra dependencies without replacing its
Torch version.

**Training machine**, workshop root, with Git and uv installed:

```bash
bash scripts/setup_gr00t_n17.sh
source ~/Isaac-GR00T-N1.7/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
```

Training needs a CUDA GPU. Preparation and verification are CPU checks requiring
NumPy, PyArrow, PyAV, FFmpeg, LeRobot's converter and the pinned GR00T reader.
N1.7 uses TorchCodec and requires an FFmpeg 4–7 runtime; the verification command
checks actual video decoding. Do not replace Isaac's simulation packages with
training packages. Use a complete workshop checkout on the training machine:
the standard teleop mount exposes `source/`, not these top-level scripts.

The legacy `so100_n16` preparation profile still uses the original
`ead52833afbbf4243f8cd5e7664f48a94de03b19` pin and front/wrist, relative-arm
configuration. Keep its existing checkout separate when preparing old runs.

## Prepare an independent copy

[prepare_pick_place_gr00t.py](../scripts/prepare_pick_place_gr00t.py) reads a v3
recording and creates a v2.1 training copy without the upstream converter's
source-directory swap. It requires the two RGB features
`observation.images.realsense_rgb` and `observation.images.wrist_cam`, six state
and six action values, the recorder's complete per-episode videos and cube
sidecars. This is not a general converter for the paired real camera names.
Encode all saved episodes first.

**Training machine**, workshop root, after encoding the new yaw-aligned v2
recordings and activating the N1.7 environment above:

```bash
python scripts/check_pick_place_demo_alignment.py --dataset datasets/pick_place_v2
python scripts/prepare_pick_place_gr00t.py \
  --model_profile so_arm_n17 \
  --groot ~/Isaac-GR00T-N1.7 \
  --source datasets/pick_place_v2 \
  --output datasets/pick_place_v2_gr00t
PYTHONPATH="$HOME/Isaac-GR00T-N1.7" python scripts/verify_pick_place_training.py \
  --dataset datasets/pick_place_v2_gr00t
```

The selected N1.7 profile maps the external view to `room` and the wrist view to
`wrist`; state is `single_arm` (five normalized SO101 joints) plus `gripper`
(one value). Both action groups are **absolute**, matching the starter checkpoint.
It registers `NEW_EMBODIMENT` (saved ID 10) and supervises 16 future actions; the
model's maximum horizon of 40 is a padded capacity, not 40 supervised SO-arm
steps. The verifier obtains the model profile from the preparation report.
Never train this copy with the legacy relative-arm config.

To exclude failed source episodes, use a **different output path**. The following
is an example only; use the failure IDs you actually reviewed.

**Training machine**, workshop root:

```bash
python scripts/prepare_pick_place_gr00t.py \
  --model_profile so_arm_n17 \
  --groot ~/Isaac-GR00T-N1.7 \
  --source datasets/pick_place_v2 \
  --output datasets/pick_place_v2_gr00t_filtered \
  --exclude_episodes 0,6
PYTHONPATH="$HOME/Isaac-GR00T-N1.7" python scripts/verify_pick_place_training.py \
  --dataset datasets/pick_place_v2_gr00t_filtered
```

| Preparation flag | Default / behavior |
| --- | --- |
| `--groot` | Required checkout; HEAD must equal the selected profile pin |
| `--model_profile` | `so100_n16` for compatibility; select `so_arm_n17` explicitly for the v2 starter workflow |
| `--source` | `datasets/pick_place_v1` |
| `--output` | `datasets/pick_place_v1_gr00t`; refuses completed output overwrite |
| `--exclude_episodes` | None; comma-separated nonnegative source IDs; duplicates deduplicated; unknown/all-excluded rejected |
| `--resume` | Retry an interrupted output `.building` directory with unchanged source hashes, the same model profile and the same exclusion selection |

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
stats on resume before generating `meta/stats.json` from retained parquet.
The N1.7 generator also recomputes copied global statistics when their schema
fingerprints are absent or stale. Its absolute arm and gripper profile does not
generate `meta/relative_stats.json`. The legacy N1.6 relative-arm profile generates
relative-action statistics and can reuse valid source global statistics when
there are no exclusions. Per-episode statistics remain valid apart from shifted
index statistics. The copied modality config registers `NEW_EMBODIMENT`; action
representation and camera keys follow the selected profile.

Preparation preserves source SHA-256 snapshots, compares parquet values, hashes
video copies, decodes every frame and checks timestamps/dimensions, runs the real
GR00T loader for every episode, and writes `preparation_report.json`,
`source_sha256.json`, `prepared_sha256.json`. The independent verifier checks that
manifest and compares first/middle/last images in each camera/episode between
the selected GR00T reader and PyAV (maximum permitted RGB difference 2). Do not
modify the prepared output after signing its manifest; make a new preparation
instead.

## Fine-tune

The starter model lives directly at `~/sim2real/models/nv_so_arm_n17`, with
`config.json`, processor/statistics/embodiment files, a safetensors index and model
weight shards at that root. It is **not** a `checkpoint-10000` subdirectory.
Training writes new `checkpoint-*` directories under a separate output root.
The saved model is N1.7, with room/wrist views, five arm joints plus gripper,
absolute actions and a 16-step SO-arm modality. Its config must pass the training
launcher's local compatibility check; serving uses the same tag and camera keys
with the N1.7 image.

**Training machine**, manual download after inspecting available disk space.
If authentication is needed, run `hf auth login` yourself. N1.7 also loads the
gated [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) backbone;
accept its access conditions on Hugging Face before downloading it. The targeted
starter download below copies model artifacts, without optimizer states. The
Cosmos command fills the Hugging Face cache used by the same training user:

```bash
hf auth login
hf download nvidia/SO_ARM_Starter_Gr00tN17 \
  --revision 93a5a88f78a395939f784a5fe3685184913dcc60 \
  --include config.json --include processor_config.json \
  --include embodiment_id.json --include statistics.json \
  --include 'model-*.safetensors' --include model.safetensors.index.json \
  --local-dir ~/sim2real/models/nv_so_arm_n17
hf download nvidia/Cosmos-Reason2-2B
```

The launcher prints the download command and exits when the base model is
missing. It sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so both the local
starter files and the Cosmos Hugging Face cache must be ready before launching;
training never downloads a missing backbone implicitly. If you set `HF_HOME`,
use the same value for the cache download and training.
It also sets `NO_ALBUMENTATIONS_UPDATE=1` to disable the optional version check.
The launcher validates the offline Cosmos snapshot, processor files and complete
weight payloads before checking the GPU or creating its output directory. The
N1.7 serving image also operates offline and needs this complete cache mounted.
Modern Hub paths expand variables such as `$HOME`. Legacy Transformers cache
variables take precedence even when empty: an empty value means the startup
directory, and a relative value resolves from there. The launcher exports the
effective cache as an absolute `TRANSFORMERS_CACHE` before entering GR00T, so
preflight and model loading use the same location. Cosmos also needs its
processor chat-template file; a tokenizer-only template does not supply it.
The project training and serving wrappers resolve the Cosmos processor to its
local cached snapshot. This avoids Transformers 4.57.3's repository metadata
request during processor loading, even in offline mode. The change is scoped to
processor creation; saved repository IDs, model weights and the official GR00T
sources are preserved.

**Training machine**, workshop root, inspect the launch, then run it after
preparation, compatibility and the memory probe have passed:

```bash
DRY_RUN=1 bash scripts/train_pick_place.sh
bash scripts/train_pick_place.sh
```

[train_pick_place.sh](../scripts/train_pick_place.sh) prints its complete command,
checks for a busy GPU and at least **50 GiB free** on the output filesystem,
refuses an output with existing checkpoints unless `RESUME=1`, and logs through
`tee` to `$OUT/train.log`. It trains **10,000 steps**, saves every **1,000 steps**,
and retains two checkpoints. Resume explicitly requests the latest checkpoint;
use a new output directory when changing the model, dataset or modality profile.

The project launcher scopes modality overrides to the selected dataset embodiment
before calling the pinned official N1.7 launcher. This keeps an unused default
robot's 50-step modality out of the starter's 40-step processor validation while
preserving saved checkpoint modality tags and the SO-arm's 16 absolute actions.
It checks the GR00T pin/import path and clears inherited test hooks that can skip
pretrained weights; it does not rebuild the model architecture.

| Environment variable | Default |
| --- | --- |
| `GROOT` | `$HOME/Isaac-GR00T-N1.7` at the N1.7 GA pin |
| `BASE_MODEL` | `$HOME/sim2real/models/nv_so_arm_n17` |
| `DATASET` | `$REPO/datasets/pick_place_v2_gr00t` |
| `OUT` | `$HOME/sim2real/models/so101_pick_place_v2_nvinit` |
| `MAX_STEPS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT` | `10000`, `1000`, `2` |
| `BATCH` | `16`; validated by the N1.7 30-step memory probe |
| `MEASURED_S_PER_STEP` | `0.3902`; measured N1.7 batch-16 compute timing |
| `RESUME` | `0`; set to `1` only to resume an existing run |
| `DATALOADER_NUM_WORKERS` | `0` |

The diffusion transformer stays frozen via `--no-tune-diffusion-model`, with
language and visual backbones frozen. The actual smoke parameter list confirmed
that state/action heads, projections and vision-language adaptation layers
remain trainable, including four vision-language self-attention layers:
528,793,728 of 3,144,016,000 parameters (16.82%) are trainable; 2,615,222,272
are frozen. The launcher sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and lowers hue jitter to `0.02`
to preserve useful blue/red colour cues.

The real N1.7 probe on the RTX 4090 loaded the official checkpoint and completed
30 optimizer updates at **batch 16**, with finite losses, and saved its checkpoint
and final model successfully. Parameters loaded as FP32. Peak Torch allocation
was **20.38 GiB**, and peak reservation was **20.47 GiB**; process-memory sampling
every 0.2 s observed **20.94 GiB**. Batch 32 ran out of memory after its first
update. Repeat a separate **at most 30-step** probe into
`/tmp` when changing hardware, the base model or the training configuration.

The printed estimate uses the batch-16 synchronized mean for steps 6–30,
`0.3902 s/step`: 10,000 × 0.3902 is about **1.08 h of training compute**.
Model loading, dataset caching and checkpoint saving add time. The smoke's
resumable checkpoint occupied **15.66 GiB**, and the final model and metadata
occupied another **11.72 GiB**: two retained checkpoints plus the final model
require about **43.04 GiB** before working space. Checkpoint rotation writes a
third checkpoint before pruning the oldest, temporarily requiring about
**46.98 GiB**; the launcher requires at least **50 GiB free** to cover that save
order and margin. Global batch is per forward/backward across GPUs before
accumulation, so increasing accumulation alone does not reduce per-forward
memory. Reduce `BATCH` when a changed configuration requires it.

Evaluate checkpoints 9000 and 10000 on the same recorded v2 starts. Keep
`--random_fraction 0` for option B; use `--episode_length_s 30 --action_horizon 8`
to report both 15 s and 30 s success rates. See [Evaluation](06-evaluation.md).
A longer episode or training run does not compensate for inconsistent grasp
orientation in the demonstrations.

## Current v1 baseline

The earlier v1 recordings have been prepared into the **new**
`datasets/pick_place_v1_gr00t_n17` directory using the N1.7 absolute profile,
excluding reviewed failures 0 and 6. The completed copy contains 48 demonstrations,
10,632 frames and 96 videos; all 21,264 camera frames decoded, the real N1.7
reader matched PyAV with maximum RGB difference 0, and source hashes stayed
unchanged. This baseline uses earlier demonstrations. Record a new yaw-aligned
v2 dataset to apply the grasp-orientation correction described above.

The copy was created with this command; preparation refuses to overwrite the
completed directory:

```bash
python scripts/prepare_pick_place_gr00t.py \
  --model_profile so_arm_n17 \
  --groot ~/Isaac-GR00T-N1.7 \
  --source datasets/pick_place_v1 \
  --output datasets/pick_place_v1_gr00t_n17 \
  --exclude_episodes 0,6
```

**Training machine**, workshop root, verify the existing baseline, then use this
command for the requested 10k run. The Cosmos cache and batch-16 N1.7 probe have
been verified; the full training run has not been started.

```bash
PYTHONPATH="$HOME/Isaac-GR00T-N1.7" python scripts/verify_pick_place_training.py \
  --dataset datasets/pick_place_v1_gr00t_n17
DATASET="$PWD/datasets/pick_place_v1_gr00t_n17" \
OUT="$HOME/sim2real/models/so101_pick_place_v1_n17_10k" \
BATCH=16 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=2 \
bash scripts/train_pick_place.sh
```

The launcher defaults continue to point at the recommended v2 dataset and output;
these explicit paths identify the v1 baseline separately.

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
those two differently named camera datasets is not supplied. The N1.7 launcher
accepts multiple dataset paths joined by the platform path separator (`:` on
Linux); each still needs compatible camera keys and modality configuration.
See the course's
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
