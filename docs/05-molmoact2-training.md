# MolmoAct2 LoRA on the workshop demonstrations

This path starts from the released `allenai/MolmoAct2-SO100_101` weights in
`/external_storage/molmoact2/molmoact2_so101`. It uses a separate LeRobot
runtime; the existing GR00T images, checkpoints and completed evaluations are
preserved. LeRobot is pinned to `d3793b7e36522321c1c455f0dc4826db3793dceb`.
The local released checkpoint's download metadata records revision
`152569fe57914d97be91055800035f54e250d009`.

## Data and joint frame

Run from the workshop root:

```bash
./docker/molmo_train.sh setup
./docker/molmo_train.sh python scripts/prepare_molmo_lora.py \
  --output /external_storage/molmoact2/pick_place_v1_lora_as_real_20261024
```

Preparation refuses to overwrite an existing output. It creates independent
numeric data and metadata and references the unchanged source videos with a
symlink. Keep the source dataset available and mounted when training. On another
machine, prepare the copy there so video paths resolve locally.

The source contains 75 episodes and 16,448 frames. The raw recordings use
normalized simulator joints. Convert the first five joints to range-midpoint
degrees using the same per-joint scales as
`configs/eval/pick_place_molmo_so101_as_real.yaml`; leave gripper units unchanged.
Do not add USD zero offsets or the experimental pan adaptation. Apply
`joint_signs=[1,-1,1,1,1,1]` and `joint_offsets=[0,90,90,0,0,0]` to both
observed state and supervised actions before normalization. The standard output
processor inverts this frame after action denormalization.

The pinned LeRobot preprocessor applies its joint correction only to state.
`scripts/molmo_lora_support.py` supplies a registered processor that also
transforms training actions. Only the workshop training entrypoint replaces the
factory; upstream source files are unchanged. Import that module when reloading
the saved LeRobot processor.

Preparation recomputes numeric statistics, including per-episode summaries.
`meta/molmo_norm_stats.json` contains exact whole-dataset model-frame quantiles
from every frame, selected by `norm_tag=workshop_pick_place_v1_as_real`. These are
used instead of the released checkpoint's quantiles. Gripper is normalized too,
because these recordings store its values on a 0–100 scale. Camera order is front
(`observation.images.realsense_rgb`) then wrist (`observation.images.wrist_cam`).

Action pan q01/q99 are **−94.91994° / +90.08438°**, compared with the released
checkpoint's **−42.13002° / +48.55350°**. No clamp is removed. The new quantiles
define the learned action range. `preparation_report.json` records source parquet
SHA-256 hashes, scales and all statistics.

## Check memory before training

```bash
./docker/molmo_train.sh probe \
  > /external_storage/molmoact2/lora_probe_20261024.log 2>&1
```

The default probe uses real decoded front/wrist frames, BF16 VLM storage, FP32
LoRA and action-expert storage, gradient checkpointing, batch 1, accumulation 16,
and two AdamW updates. It checks action preprocessing and inverse-frame
round-tripping, asserts that the full action expert is trainable, checks finite
losses and gradients, and verifies an expert weight change. Its report includes
CUDA allocation/reservation peaks and the stage of any OOM:

`/external_storage/molmoact2/lora_probe_20261024/report.json`

An OOM requires a memory fix and sustained revalidation or a larger training GPU.
Do not freeze the expert, narrow the action range, or change the cameras to make
training fit. Return externally trained weights for evaluation locally.

On 2026-10-24 the RTX 4090 probe **passed** both optimizer updates and all
preprocessing/gradient/weight-change checks. Peak Torch allocation was
**22.0925 GiB**, with **22.3965 GiB** reserved. Total parameters were
5,601,988,080; 737,356,320 were trainable in FP32, including all 577,564,448
action-expert parameters. The 32 real microbatches took 13.34 seconds after
initialization. This establishes the tested batch-1/accumulation-16 configuration;
it is not a benchmark score.

The production LeRobot trainer also completed 32 microbatches, saved a resumable
checkpoint, and recorded exactly two Adam updates in all 1,186 allocated optimizer
states. The saved normalizer's q01/q99 tensors matched the prepared quantiles.
The resulting export passed exact expert-tensor and original-HF key checks, then
loaded in the unchanged Transformers 4.57 policy runtime and returned finite
30×6 FP32 actions within the new bounds. That serving check used two blank images
and establishes compatibility, not task performance. Its report is
`/external_storage/molmoact2/lora_smoke_server_check_20261024.json`.

## Train after sustained GPU validation

The 4090 passed the short memory probe but **failed sustained validation with
GPU-resident Adam states**.
After the native-crash mitigations below, the resumed run reached microbatch
1518 and raised a CUDA OOM: 22.83 GiB allocated by Torch and 23.38 GiB used by
the process. It was stopped in accordance with the request to use another GPU
for training if the 24 GB card did not fit. The step-512 checkpoint is complete
and retains model weights, all Adam states, RNG, processors and schedule.

Do not treat the two-update probe as proof that the 10,000-update job fits.
The original GPU-state configuration needs an alternate training GPU; the CPU
optimizer-state mode below reduces memory without changing the policy recipe.
`/external_storage/molmoact2/training_handoff_20261024.json` records the exact
checkpoint, data paths, runtime and policy settings from the failed run.

### CPU optimizer-state offloading

`MOLMO_OPTIMIZER_OFFLOAD=cpu` keeps Adam's FP32 first and second moments in pinned
host RAM and stages one parameter's moments on CUDA for its update. Parameters,
gradients, update arithmetic, learning rates, group clipping and nonfinite-step
handling stay as configured. No state is quantized. This frees approximately
5.40 GiB of persistent CUDA storage for this model, at the cost of PCIe transfers
per optimizer update and the corresponding host RAM.

`scripts/check_molmo_optimizer_offload.py` checks eight CUDA updates against the
native Molmo AdamW optimizer, including two groups with different learning rates,
weight decay, a missing gradient, a nonfinite skipped update, loading native
Adam state and resuming serialized CPU moments. Every parameter and moment was
exactly equal in the test. Saved optimizer states retain the standard PyTorch
AdamW schema.

The real-model CPU-offload recovery logs and status live under
`/external_storage/molmoact2/lora_pipeline_20261024_cpu_optimizer/`. It retains
single-threaded autograd and resumed the step-512 checkpoint. It passed the old
OOM location at microbatch 1518 with 16.12 GiB peak Torch allocation and about
17.7 GiB total GPU usage. It then **segfaulted at microbatch 1720**, exit 139;
the kernel placed this fault in `libcuda.so.580.173.02` on CPU 5. There was no
CUDA OOM in this run. The fault's underlying cause is unproven.

CPU offloading therefore addresses the observed memory pressure, but sustained
training is **not validated**. The job is stopped and the last completed
checkpoint remains step 512. `validation_report.json`, `kernel_crashes.log` and
the full training log preserve the evidence. No runtime manifest promotes this
configuration to the default. If a future run passes sustained validation,
`validated_training_runtime.json` can record the runtime and default offload mode.
Enable offloading explicitly with `MOLMO_OPTIMIZER_OFFLOAD=cpu`; it is a memory
option, not a demonstrated fix for the native crashes.

```bash
./docker/molmo_train.sh train \
  > /external_storage/molmoact2/lora_train_20261024.log 2>&1
```

The default is 10,000 optimizer updates at effective batch 16, using VLM LoRA
(rank 64, alpha 16), a fully trainable action expert, absolute joint actions,
30-step chunks, and all 30 steps executed. The action expert and LoRA learning
rates are 5e-5. Training enables LeRobot's documented crop/rotation/color recipe.
The memory probe decodes the original camera frames; training augmentations
resize to 256×256 before model processing.

LeRobot counts training iterations and steps its LR scheduler per microbatch.
The launcher converts optimizer-update counts, warmup, decay and checkpoint
intervals to this clock: 10,000 updates with accumulation 16 require 160,000
iterations. Checkpoints are saved every 2,000 optimizer updates. Output defaults
to `/external_storage/molmoact2/so101_pick_place_v1_lora_as_real_20261024`.

`MOLMO_BATCH`, `MOLMO_ACCUMULATION`, `MOLMO_UPDATES`, `MOLMO_OUT`,
`MOLMO_DATASET`, `MOLMO_BASE`, `MOLMO_STORAGE` and `MOLMO_TRAIN_IMAGE` override
the corresponding launch settings. Repeat the probe when changing the hardware
or memory-relevant settings. Keep batch × accumulation at 16–32 for this dataset.

### Native-crash investigation

The first long run stopped at microbatch 813 with container exit code 139.
The kernel reported `pt_autograd_0` faulting at `0xb1` in
`_PyEval_EvalFrameDefault` under the image's CPython 3.12.3. There was no CUDA
OOM. A second launch also exited 139 at the same native instruction. The earlier
32-microbatch smoke verified memory and checkpoint serialization but did not
establish long-run stability.

The replacement runtime is isolated at
`/external_storage/molmoact2/lerobot_lora_env_py31213`, using the already installed
standalone CPython 3.12.13 and the exact same Torch 2.11.0+cu128, LeRobot,
Transformers, NumPy and PyAV packages. It changes no weights, dataset values,
LoRA settings, action-expert trainability, normalization or camera settings.
`MOLMO_RUNTIME` and `MOLMO_PYTHON_ROOT` select this runtime explicitly. After
validation, `validated_training_runtime.json` selects it for normal launcher use.
Python 3.12.13 alone also segfaulted, at microbatch 954. Its all-thread traceback
identified `torch.utils.checkpoint.unpack_hook` on `pt_autograd_0`, while the main
thread waited in `Accelerator.backward`. Changing the Python patch release alone
is therefore insufficient.

The current recovery changes autograd scheduling with
`torch.autograd.set_multithreading_enabled(False)`. A CUDA checkpoint test
verified that recomputation moves from the autograd worker onto the calling
thread and that every input and parameter gradient is exactly equal between
thread modes. Gradient checkpointing, CUDA kernels and policy settings remain
enabled as before. This workaround is under sustained-run validation; it is not
a proven root-cause diagnosis. Recovery resumes the saved step-512 checkpoint
without resetting its weights, Adam state, RNG or scheduler.

Native faulthandler and C++ stack traces are now enabled. A process lock rejects
duplicate training launches. Extra recovery checkpoints at microbatches 512 and
2048 protect the first periodic-save interval. Failed pipeline status now records
the actual exit code. Old logs remain under `lora_pipeline_20261024`; replacement
logs and status are under `lora_pipeline_20261024_py31213`.
The resumed single-threaded-autograd run uses `lora_pipeline_20261024_recovery`.
It passed both prior native-crash locations but later failed with the CUDA OOM
described above. This does not establish that the native issue is permanently
fixed. Its captured traceback and `validation_report.json` are preserved there.
Set `MOLMO_RESUME=1` on the pipeline runner to explicitly resume its last completed
checkpoint; fresh-run launches still refuse existing model output.

## Evaluate the trained policy

Use the existing as-real scene and joint mapping, front + wrist only, H30/R1,
the same nine zone starts and root seed 1984. Then run all 150 fixed starts from
`eval_sets/pick_place_v1_seed1984.json`, including `id,ood,yaw,train`, with one
repeat. A nine-start smoke score must be reported separately from the benchmark.
The current generic Molmo server loads original-HF artifacts, so a saved LeRobot
checkpoint is exported with `scripts/export_molmo_lora.py`. That merges the VLM
LoRA, checks exact equality of the action-expert tensors and that tensor keys match
the original HF model, and carries the fresh normalization file into the export.
The existing client then performs the same as-real joint conversion as before.

After the memory gate and production trainer/checkpoint-save smoke have passed,
the complete sequence can run unattended:

```bash
./scripts/run_molmo_lora_pipeline.sh \
  > /external_storage/molmoact2/lora_pipeline_20261024.log 2>&1
```

The runner trains, exports, executes the nine-zone screen and then all 150
starts. It snapshots separate smoke and benchmark YAML files because the
evaluation wrapper intentionally rejects smoke deadlines in a full benchmark.
Logs and `status.json` live under
`/external_storage/molmoact2/lora_pipeline_20261024/`; evaluation JSONs live under
`datasets/eval_results/molmo_lora_20261024/`. Completion is recorded only after a
complete 9-start smoke and complete 150-start report. A failed stage stops the
sequence and is recorded in `status.json`.

If far-zone results remain weak after fine-tuning, record more demonstrations
in **FR, FL, FC and OUT** before changing the policy, action representation,
cameras, or scene.

References: [LeRobot MolmoAct2 training](https://huggingface.co/docs/lerobot/main/molmoact2),
[released SO-100/101 model](https://huggingface.co/allenai/MolmoAct2-SO100_101).
