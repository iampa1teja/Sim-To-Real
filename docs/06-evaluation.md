# 06 — Evaluate in simulation and with DR

[Guide hub](../README.md) · Previous: [Training](05-training.md) · [Troubleshooting](troubleshooting.md)

## Data flow and counting

```mermaid
sequenceDiagram
  participant S as Isaac Lab / lerobot_eval
  participant P as GR00T policy server
  S->>P: room/wrist images + joint state + instruction
  P-->>S: action chunk
  S->>S: Execute action_horizon actions
  S->>P: Updated observation
  S->>S: Count success or timeout; auto-reset once
```

`lerobot_eval` uses `SimRemotePolicy` and the SO101 interface to convert between
normalized LeRobot joints and simulation radians, including `sim_joint_mapping`.
Its `rename_map` translates sim camera names to checkpoint keys. For the prepared
v2 N1.7 starter profile use `{"realsense_rgb":"room","wrist_cam":"wrist"}`.
Both RGB views and language must match training. Legacy N1.6 front/wrist
checkpoints need their original rename map. For GR00T, `--action_horizon` defaults to 16
executed actions before another query and accepts integers **1–16**. Setting it
to 8 executes the first eight actions and replans; it does not change the
checkpoint's 16-step SO-arm training horizon or its padded model capacity.
MolmoAct2 instead defaults to **30** and accepts **1–30**. `action_repeat`
controls how many control steps hold each predicted target (default **1**).
The SO-101 profile uses the complete 30-step prediction with **action_repeat=1**:
30 control steps at the simulator's 30 Hz control rate give **one simulated second between queries**,
preserving the predicted motion and gripper timing. For an explicit slower
comparison, `eval.action_repeat: 5` stretches the same 30 targets over five
simulated seconds. It slows the motion fivefold and reduces visual feedback;
it does not make the checkpoint predict 150 distinct future actions. Lowering
`action_horizon` executes fewer predicted targets before another observation.

The SO-101 Molmo profile sets `policy.options.dtype: bfloat16` and
`policy.options.action_dtype: float32`. Large backbone matrices stay in BF16;
the action expert, normalization modules, and RoPE retain FP32 storage with
BF16 autocast for eligible operations. Original checkpoint tensors are restored
before GPU transfer, avoiding a full-FP32 backbone or widening already rounded
weights. The expert adds approximately **1.1 GiB** over all-BF16 storage; simulator
and activation memory are additional. Omitting `action_dtype` uses `dtype`,
preserving the base profile's default. Physical action denormalization always
uses FP32, preventing the extra whole-degree rounding that BF16 absolute angles
near 180 degrees would introduce. This addresses numerical precision, without
establishing that grasp accuracy is solved.

The released SO-101 normalizer also clips normalized actions to `[-1, 1]`
before converting them to joint units. Its pan output is consequently limited
to approximately **-42.13 to 48.55 degrees** in the model frame, which is narrower
than the arm's physical range. To inspect whether a prediction exceeds these
bounds, set `policy.options.action_range_diagnostics: true`. The server's
`describe` response then includes `last_raw_action_range`: per-joint minima,
maxima and out-of-range counts before clipping. This option defaults to false;
enabling it preserves the returned actions, checkpoint statistics and clipping.

For the installed checkpoint **without pan adaptation**, unchanged `ARM_new.usd`
benchmark layout and CAD-aligned simulator conversion described below, an offline
check of the actual joint anchors and jaw collision surfaces found a remaining action-range
constraint. A grasp-centre placement inside the walls needs pan approximately
**88.86 degrees or more**, or the opposite branch at **-73.83 degrees or less**.
Allowing contact anywhere on the moving jaw still leaves a conservative
**22.25 mm** gap between a held 20 mm cube and the box at clipped joint targets.
These bounds assume static tracking and physical surface contact; they do not
cover tossing or large dynamic overshoot. Before the simulator conversion was
corrected, three controlled replays of a captured held-cube state produced
pre-clipping pan values below **48.91 degrees**. Removing clipping alone would
not resolve that captured prediction. The added camera enabled a grasp and lift
in both the two-image and three-image trials, but each completed attempt timed
out before placement. Those results predate the conversion correction and do
not establish the corrected policy's task performance. With the corrected
conversion alone, the same first start also grasped and lifted the cube but
timed out after 300 simulated seconds while pan tracked its clipped target.

The SO-101 evaluation profile opts into an **experimental pan adaptation**:
`policy.options.pan_model_scale: 0.49016972984379564` and
`policy.options.pan_model_offset: 3.5352321757436442`. Together they apply
`model_pan = scale * physical_USD_pan + offset` to observations and the exact
inverse to actions. The released model pan limits then correspond to about
**-93.16 to 91.84 physical degrees**. All other joints retain the corrected
conversion. Omit **both** options for the released policy's native pan range;
supplying one alone, nonfinite values, a nonpositive scale, or using a real-client
unit mode is rejected. Enabled runs record the adaptation in
`policy_joint_transform.pan_adaptation`.

These values match the pan action q01/q99 of all 75 local training demonstrations
(16,448 frames, `datasets/pick_place_v1`, declared `train: 0:75`) to the
checkpoint's pan q01/q99. Quantiles are frame-weighted with NumPy's linear
method after the existing simulator joint conversion. Evaluation target
positions were not used for the fit. The checkpoint weights, normalization
statistics, USD geometry and scene remain unchanged. This is a change to policy
action semantics, not a physical calibration correction or an official
zero-shot deployment recipe. In an earlier H30/R1 three-camera run,
start `id-000` and policy seed `2177438227`, adaptation completed the task in
**15.37 simulated seconds / 23.79 wall seconds**. The unadapted corrected
baseline timed out at 300 simulated seconds on the same start and seed. Both
runs seeded the server once at episode reset, allowing inference noise
to advance between queries. That three-camera smoke run completed one episode;
five-episode validation used the original front and wrist cameras instead.
The front/wrist SO-101 profile used for the following historical results kept
the same pan adaptation and H30/R1 timing. An earlier five-episode smoke test of
that adapted profile completed normally with
**2/5 successes (40%)** and a median time to success of **19.85 simulated
seconds**. Starts `id-000` (MC) and `id-002` (MR) succeeded; `id-001` (FR),
`id-003` (NL) and `id-004` (FL) each timed out after 300 simulated seconds.
This verifies successful pick-and-place on two starts; five attempts and three
timeouts do not establish reliability across the full benchmark. A later
180-second full run was initially interrupted with 11 completed attempts and
4 successes. That report was subsequently resumed to completion on 2026-10-22:
**25/150 successes (16.7%)**, with a mean time to success of **31.51 simulated
seconds** and median **24.50 seconds**, considering successful attempts only.
The completed report retains the original front/wrist, H30/R1 profile and
config fingerprint `a2d507ad51ca74cbc5d33fb64e29aaac2b4c37d02d78d12476fd6d5285db23df`.
Its 300 listed camera-video files were present and nonempty at verification.

The 125 failures comprised 88 `never_reached`, 31 `missed_grasp`, five
`dropped_before_box` and one `dropped_outside`: 119 failed before a confirmed
grasp. Successes by zone were FC 1/18, FL 0/10, FR 0/20, MC 14/27, ML 2/18,
MR 5/15, NC 2/10, NL 0/12 and OUT 1/20. These results describe this resumed
benchmark; they neither prove that a whole zone is physically unreachable nor
establish a ceiling on the released model. The stricter per-zone smoke protocol
below uses different time limits and a different sample, so compare candidate
changes against its own matched control rather than this 150-start percentage.

The native stage flags recorded 62 reaches, 31 confirmed grasps and 25 successes.
Half of the reached attempts confirmed a grasp; 25/31 grasps (80.6%) succeeded.
Thus 119/125 failures (95.2%) ended before a confirmed grasp. This prioritizes
approach and grasp accuracy for further comparisons. A lifted-cube flag alone
is insufficient: 33 attempts registered a lift, including two without a
confirmed grasp.

A read-only audit decoded and hashed **all 1,397,268 frames across the 300
listed videos**. Every frame count matched its episode's step count; all streams
were 640×480 at 30 Hz. No decode error, non-increasing timestamp, near-black
span of at least 50 ms after the ten settling frames, entirely static
post-settling stream, or identical
room/wrist stream was found. Synthetic black, static, mismatched-length and
corrupt clips verified the detectors. This checks recording health, not semantic
understanding of every frame or the exact live model-input tensors.

Repeat this benchmark-specific audit inside the container; it reads the
completed report and videos without altering them and writes evidence only to
a fresh RAM directory printed on completion:

```bash
docker exec teleop python3 \
  /workspace/Sim-to-Real-SO-101-Workshop/scripts/audit_molmo_baseline_videos.py
```

An isolated replay of failing start `id-005` retained policy seed `1832710734`,
H30/R1 and the 180-second limit. It again failed before reaching the cube.
The closest measured tool error was 64.32 mm, primarily along local gripper Z, with a
49.97 mm jaw gap around a 20 mm cube. Requested and applied simulator joint
targets matched throughout the replay. At closest approach, tracking error was
at most 1.67 degrees; forward kinematics at the requested target still missed
by 74.56 mm. Shoulder and elbow commands were near the released normalizer's
learned bounds. This evidence points to the predicted approach targets for that
attempt; it does not prove a physical reachability or calibration defect, nor
that bypassing the model's clipping would fix the grasp.

An additional 2026-10-22 offline check loaded all 5.44 billion released parameters
in FP32 on the CPU. Two reconstructed observations produced finite 30-by-6
action chunks, using about 21.3 GiB peak resident memory and 9.7 seconds per query
with 16 CPU threads. This establishes a full-precision reference option without
using GPU memory; it does not establish episode success.

The saved stalled observation at query 29 (reconstructed video frame 880) was
also tested with matched CPU sampling noise, seed `1832710763`, while changing
the instruction and image preprocessing:

| Input variant | Closest commanded tool point to cube | Query wall time |
| --- | --- | --- |
| Released resize, original instruction | 79.89 mm | 9.81 s |
| Released resize, explicit step-by-step instruction | 79.86 mm | 10.09 s |
| Supported high-resolution tiling, original instruction | 88.59 mm | 34.39 s |
| High-resolution tiling, explicit instruction | 89.03 mm | 34.58 s |

Five noise seeds at that same observation gave closest commanded distances of
75.29–79.89 mm with the released resize. These are static forward-kinematics
measurements of predicted targets, not executed robot approaches. The source
images are lossy video reconstructions; CPU noise and kernels do not reproduce
the saved GPU stream. They cannot select a production setting or replace a
closed-loop comparison.

For that CPU baseline chunk, the observed pre-clipping overflow could move the
fixed diagnostic tool point by at most 3.4 mm if clipping alone were removed.
The bound sums each joint's maximum rigid displacement,
`2 * downstream_radius * sin(abs(angle_change) / 2)`, over the five arm joints.
Angles are in radians and each radius is bounded by downstream segment lengths.
Thus unclipping those same 30 targets still leaves at least 76.4 mm of static
error. This local bound does not rule out different closed-loop predictions if
clipping changes earlier in an episode, and is not a limit on all possible
Molmo tuning. Those offline probes did not change camera poses, preprocessing
or inference defaults.

Reconditioning those same saved images on the native physical pan angle
(35.65° rather than the experimental warp's 21.01°), while keeping the other
state values fixed, also left a predominantly vertical commanded miss. The
closest static errors were 79.89/80.31 mm with the existing warp and
78.71/78.85 mm with native pan, respectively with/without output clipping.
The clipped control exactly reproduced the previous CPU reference array.
This tests input conditioning as well as output interpretation; it provides no
evidence for a new pan correction and does not exclude a closed-loop effect.

The checkpoint's native discrete decoder was also checked offline with the
released FAST tokenizer and the same two reconstructed observations. Both
queries produced valid 30-by-6 chunks, with 180 decoded DCT coefficients rather
than the tokenizer's malformed-sequence zero fallback. On the stalled query,
the closest commanded point missed by **75.18 mm**, versus **79.89 mm** for
continuous decoding. The closest targets were respectively **65.85 mm** and
**76.50 mm above the cube** in the robot-base frame. Optimizing pan alone for
those same continuous targets offline reduced the miss only to **79.65 mm**.
This diagnoses a predominantly vertical miss in that observation; it does not
select a new pan correction or establish closed-loop performance. The
[model card](https://huggingface.co/allenai/MolmoAct2-SO100_101#discrete-actions)
exposes discrete decoding for parity/debugging and recommends continuous
inference. The adapter's inference mode remains continuous.

The 2026-10-21 Rerun smoke check then completed five 180-second rollouts on
`id-000` through `id-004` with **0/5 successes**: two `missed_grasp` and three
`never_reached`. It confirmed live milestone reporting and quiet default logs,
not improved grasp performance. The rollout finished, but its host launcher
exited with a parse error after the script was edited while Bash was executing
it; fresh-launch lifecycle checks pass. Avoid editing `docker/eval.sh` during a
run. The full benchmark was stopped at that point with its previous results preserved;
it was later resumed to the completed result reported above.

Separate helper trials enabled `--seed_per_episode`, which resets the same
noise seed before **every inference query**. In that mode, H30/R1 succeeded in
**161.23 simulated seconds / 237.21 wall seconds**, while H16/R1 missed the grasp
and timed out at 300 simulated seconds / 475.89 wall seconds (562 queries).
Those two helper trials compare horizons within that fixed-noise mode; their
timings cannot be compared directly with the standard profile's advancing
noise. These limited results do not establish reliability or a globally optimal
horizon.

`--episode_length_s` overrides the task config before the environment is created.
When omitted, it uses the task's value (**15 s for Pick-Place-Eval**). Explicit
values must be finite and greater than zero. A 30 s run reports success within
15 s as well as success within the full 30 s window.

For recorded PickPlace starts, the first ten control steps hold the actual reset
joint pose from the same demo's first frame. This settling interval is refreshed
after each automatic or manual reset; policy actions begin on step 11. With
`--robot_start default`, or other tasks, the existing calibrated warmup pose is
used instead. Success timing includes these ten settling steps.

For PickPlace, success is termination **without** timeout. Timeout counts as
failure even if success also fires on that step. Isaac's done step already resets
the environment; the client reuses the returned observation without another reset.
It snapshots start/zone/colour before stepping, so results describe the completed
episode. Manual R resets are excluded. Closing the simulator early saves an
incomplete report and raises an error. See [success logic](03-define-a-task.md).

## Host runner

Start the teleop container using [Setup](01-setup.md), and build the N1.7 serving
image for the RTX 4090:

```bash
./docker/real/build.sh ada n17
```

This creates `real-robot:n1.7` from GR00T GA revision
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`, in a separate serving environment.
N1.7 requires access to the Cosmos-Reason2 backbone; complete the
[training setup and authentication](05-training.md#fine-tune) on the host so the
mounted Hugging Face cache is complete and available to serving. The N1.7 image
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; it requires prefetched Cosmos
files and does not download missing backbone artifacts. Put
the checkpoint under your **host** models directory. `--dataset` is an absolute
path **inside teleop**, pointing at the original dataset with saved cube sidecars;
model paths are relative to the host models directory.

**Host**, repository root. Every setting lives in one YAML file
([configs/eval/pick_place.yaml](../configs/eval/pick_place.yaml) is the template: policy, checkpoint,
dataset, episodes, smoothing-free grasp options and so on). Copy it, edit it, then inspect the launch
without starting Docker jobs:

```bash
./docker/eval.sh pick_place.yaml --dry_run
```

Run the evaluation (a bare file name is looked up in `configs/eval/`):

```bash
./docker/eval.sh pick_place.yaml
```

To compare checkpoints, copy the YAML, change `policy.checkpoint`, and run the copied
file. Keep the dataset, benchmark, seed and other settings identical. The launcher
reads settings from YAML; its optional flags are `--dry_run`, `--external_server`,
`--server_only`, `--gui`, `--rerun`, `--save_videos`, `--resume [RESULTS.json]`, `--smoke-test`,
`--molmo-overhead-camera`, and `--help`.

When `eval.eval_set` is configured, the selected benchmark starts and repeats
determine the episode count; `eval.episodes` is used for runs without a fixed
benchmark. The SO-101 Molmo profile selects 150 starts across its four splits.

For a five-episode check with Rerun and no saved evaluation artifacts:

```bash
./docker/eval.sh pick_place_molmo_so101.yaml --smoke-test
```

`--smoke-test` forces five episodes and enables Rerun, overriding the benchmark
count and any report, video or CSV saving options. It uses the first five
scheduled benchmark starts; selections shorter than five are repeated to reach
five attempts. The configured episode timeout stays in effect (180 simulated
seconds per attempt in the SO-101 Molmo profile). Progress, success timing and
the final summary appear in the terminal. No results JSON, heatmap, MP4,
gripper/timing CSV or launcher log directory is written, even on failure or
interruption; `EVAL_LOG_DIR` is ignored. Model/viewer caches and simulator-owned
diagnostic logs follow their usual behavior. Smoke mode works with
`--external_server` and an existing `RERUN_CONNECT_URL`; it cannot be combined
with `--server_only`.

Molmo pick-place smoke tests also print one `[GRASP N]` diagnostic per episode.
It reports the closest tool-to-cube error (XYZ in the fixed gripper frame,
millimetres), the live gap between opposing CAD fingertips versus cube width,
gripper commands, and joint tracking error. The same measurements are reported
at the first and closest closing commands. These measurements run before the
environment's automatic reset and stay in memory; they do not change actions,
contacts or success criteria. A small tool error with an insufficient jaw gap
shows whether the closing command leaves enough clearance for the cube;
joint tracking error distinguishes a missed predicted target from a target the
arm did not execute. These measurements do not establish a calibration defect.

The policy is a plugin under `policies/<name>/` (see [07 policy plugins](07-policy-plugins.md)); the
server runs in the generic `policy-server` image (`./policy_servers/build.sh`) and the first start of a
policy builds its environment, so allow for `READY_TIMEOUT` (default 900 s). The teleop container needs
`./policies` mounted (see the README launch block).

The launcher prints startup status, episode progress and a compact results summary.
Outside smoke mode, it retains complete output under `/tmp/so101-eval/<run>/`: `evaluation.log`,
`server.log`, `rerun.log` when enabled, and a copy of `results.json`.
Set `EVAL_LOG_DIR=/path/to/run` to choose a directory, or `EVAL_VERBOSE=1` to
also print routine Isaac initialization details and warnings. Default console
output hides warnings and checkpoint-loading progress; full non-smoke logs retain
them unchanged, and errors remain visible. Failed commands show their diagnostic
log path and error context. The repeated Molmo model-state dump is removed.
Ctrl+C cleans up this run without dumping an unrelated server-log tail; a second
interrupt cannot abort cleanup. Completed attempts remain available to `--resume`.

For fixed benchmarks, the progress bar shows the highest observed milestone:
`approaching`, `reached`, `grasped`, `lifted`, `over box`, `placed`, then `success`.
After a grasp, `held=yes/no` reports the current holding state, so a drop remains
visible even while the highest milestone stays `lifted`. These labels use the
existing benchmark measurements and do not affect control or success criteria.
The episode summary includes the final milestone and a failure category.

`--rerun` starts a native host viewer before loading the model. The launcher
caches the exact viewer binary shipped with the simulator's Rerun SDK, so an
unrelated host Rerun installation cannot introduce a protocol mismatch. It
uses the host's `DISPLAY`, falling back to the teleop container's display;
start teleop from your desktop terminal using the README's X11 mounts.
`RERUN_PORT` changes the default viewer port of 9876. Set `RERUN_CONNECT_URL`
to connect to an existing viewer instead. The launcher cleans up its own
viewer and policy server when the evaluation ends or is interrupted.
Camera frames are timestamped and JPEG-compressed in memory, so scrubbing the
recording replays the selected camera views alongside the joint values.

Without an X11 desktop, run the matching SDK's web viewer in a separate terminal
instead of changing display permissions. With `teleop` running:

```bash
docker exec teleop /isaac-sim/python.sh -m rerun --serve-web \
  --bind 127.0.0.1 --port 9876 --web-viewer-port 9090 --server-memory-limit 4GB
```

Then connect the evaluator to that server:

```bash
RERUN_CONNECT_URL=rerun+http://127.0.0.1:9876/proxy \
  ./docker/eval.sh pick_place_molmo_so101_smoke.yaml --smoke-test
```

Open `http://localhost:9090` on the workstation. For remote viewing, forward both
9090 and 9876 over SSH. The server buffers recordings in RAM; these commands do
not save an RRD. An explicitly supplied viewer remains externally owned, so stop
it separately when finished.

YAML keys (old `eval_pick_place.sh` flag in parentheses):

| Key | Default / interpretation |
| --- | --- |
| `task` | `pick_place`; `pick_place_dr` selects `Lerobot-So101-Teleop-Pick-Place-DR-Eval` (`--dr`) |
| `policy.name` | `groot_n17`; folder under `policies/` |
| `policy.checkpoint` | Required relative checkpoint path under `policy.models_dir` (`--model`) |
| `policy.models_dir` | `~/models` on the host (`--models_dir`) |
| `policy.port` | The plugin manifest's `default_port` (`--port`) |
| `policy.server_image` | `policy-server:latest` (`--server_image`) |
| `policy.options` | Mapping given to the server (`--option`) and client (`--policy_option`), e.g. `embodiment_tag` (`--embodiment_tag`) |
| `eval.dataset` | Required absolute teleop dataset root; expects `pick_place_meta/` (`--dataset`) |
| `eval.episodes` | 20 (`--episodes`) |
| `eval.random_fraction`, `eval.start_mode`, `eval.robot_start` | 0.0, `cycle`, `recorded` |
| `eval.robot_usd` | Task asset; optional absolute USD path inside teleop (`--robot_usd`), saved in results |
| `eval.episode_length_s` | Task default (15 s); finite positive seconds |
| `eval.action_horizon` | MolmoAct2: 30, integer in 1-30; other policies: 16, integer in 1-16 |
| `eval.action_repeat` | 1; positive integer control steps per target. SO-101 Molmo profile: 1 (one-second query interval with horizon 30 at 30 Hz) |
| `eval.molmo_overhead_camera` | Generic default false; true in the SO-101 full profile. Adds a camera behind/above the arm as `room`, retaining the original front and wrist sensors |
| `eval.eval_set`, `eval.splits`, `eval.repeats` | Fixed benchmark JSON, list of `id/ood/yaw/train`, repeats; splits and repeats need `eval_set` |
| `eval.lang`, `eval.lang_by_color` | Instruction, per-colour map (blue, plus red for DR) |
| `eval.rename_map` | `realsense_rgb: room`, `wrist_cam: wrist` (`--rename_map`) |
| `eval.render_warmup`, `eval.results_json` | 0; absolute teleop path |
| `eval.save_videos`, `eval.video_dir` | false (`--save_videos`); optional absolute video directory inside teleop |
| `eval.video_encoder` | `libx264` (background CPU encoding); optional `h264_nvenc` (NVIDIA hardware encoder) |
| `grasp.*` | `gripper_latch`, `log_gripper`, `seed_per_episode`, `latch_*`, `gripper_csv`, `gripper_empty_closed` |
| `run.gui`, `run.rerun` | false (`--gui`, `--rerun`) |

Molmo's `pick_place_molmo_so101.yaml` selects the SO-100/101 fine-tuned checkpoint;
prepare it with `./scripts/setup_molmoact2.sh --variant so101`, then evaluate with
`./docker/eval.sh pick_place_molmo_so101.yaml`. SO-101 attempts allow up to 180
simulation seconds (3 minutes); successful attempts end early. Each episode
logs simulation elapsed time and actual wall time, with time to success reported
for successful attempts. These clocks include settling steps; wall time includes
inference, rendering and optional recording, and excludes model startup.

The SO-101 full profile selects all 150 original starts, H30/R1, a 180-second
limit per attempt, three model inputs, and video recording. Rerun is off.
Start a fresh three-camera run:

```bash
./docker/eval.sh pick_place_molmo_so101.yaml --save_videos
```

It saves top (`room`), wrist (`wrist`) and front (`original_room`) videos.
The camera configuration differs from the earlier two-camera report: start
without `--resume` for this first three-camera run. The completed two-camera
results and videos are retained. The existing pan adaptation and all other
inference settings stay in effect; no controller assistance is enabled.

After an interruption, continue the latest matching three-camera report:

```bash
./docker/eval.sh pick_place_molmo_so101.yaml --save_videos --resume
```

`--resume` uses `eval.results_json` when configured; otherwise it finds the most
recent matching report in the default results directory. To select a run
explicitly, use `--resume /workspace/Sim-to-Real-SO-101-Workshop/datasets/eval_results/NAME.json`
(the absolute path inside teleop). It requires `eval.eval_set` and cannot be
combined with `--smoke-test` or `--server_only`.

Results are written atomically before the first attempt and after every completed
episode. Resume retains successful and failed completed attempts, restores their
aggregate statistics, and restarts the unfinished attempt from its scheduled
start with the original policy seed and configured timeout. It validates the
checkpoint, benchmark, selected schedule, policy options, camera mapping, joint
transform and action settings before modifying saved results. Older reports use
their recorded settings and schedule for validation. Already complete reports
execute no further attempts. Start a fresh run by omitting `--resume`; an explicit
existing `eval.results_json` is protected against accidental replacement.

Recording resumes in the saved video directory, even if `--save_videos` is omitted
on restart. Existing completed and partial MP4 files are preserved; new recordings
use the next unused attempt number, which may differ from the logical episode
number after an interruption. Each result row retains its actual video paths.
An abruptly killed episode's partial MP4 may be unplayable; that episode is retried.

Run this on the host with `teleop` running. Add `--rerun` from a desktop terminal
to watch live. If a retained container is stopped, use `docker start teleop`
first; a container created with
`--rm` must be created again using the README launch block. Close the evaluation
with Ctrl+C to clean up its evaluator, policy server and owned viewer. Stop the
remaining idle container with `docker stop teleop` when finished.

Videos are streamed to separate MP4 files per camera and attempt at the simulation
frame rate, so playback duration follows simulation time. By default they are
saved beside the results JSON in `<results_stem>_videos/`. Set `eval.video_dir`
to an absolute container path to choose another location; recording requires
`--save_videos` or `eval.save_videos: true`. The results record each attempt's
video paths. Existing video files are preserved, so use a fresh output directory
for a new run.

Recording copies each camera frame before the next simulation step and encodes
it in a background worker. The queue holds at most eight frame sets; if encoding
falls behind, the producer waits instead of dropping frames. Episode completion
waits for encoding and MP4 finalization before saving the result row. Encoding
errors surface in the evaluator and stop the run.

The SO-101 Molmo profile keeps `eval.video_encoder: libx264`: background CPU
encoding was faster overall in the isolated recording comparison. Optional
`h264_nvenc` encoding was also verified in the simulator container; it requires
NVIDIA video driver support. Direct evaluator invocations accept the same choices
through `--video_encoder`. Encoder selection affects saved videos and is recorded
per attempt; it does not change policy images, rendering, physics or actions,
and can change when resuming a benchmark. A running evaluator keeps its loaded
recorder; recording optimizations take effect on its next launch.

For example, a successful attempt prints
`time_to_success=12.50s sim (18.20s wall)`. An unsuccessful attempt prints
`timeout_elapsed=180.00s sim (...)` at its configured limit. Both elapsed times
are stored in JSON; success-time fields are null on failure. Completed videos
include `episode_000000_room.mp4`, `episode_000000_wrist.mp4` and, when the
original front view is retained separately, `episode_000000_original_room.mp4`.
Paths are stored in each episode's `videos` mapping. Ordinary Python exceptions
finalize the current attempt's files; forced process termination can interrupt
MP4 finalization. Frames are captured before environment stepping so automatic
resets do not add the next attempt's initial pose to the preceding video.

Both Molmo configs select `assets/usd/ARM_new.usd`, which layers
over `SO-ARM101-USD.usd` and restores the original wrist-roll joint frame and the
matching gripper/jaw zero-pose transforms. The existing USD and other policy
configs retain their calibration. The wrist camera follows the gripper; the
base pose, joint command mapping and benchmark starts are unchanged. Results
record `robot_usd` so runs with different wrist frames can be identified.
The camera and replacement mount use gripper-local coordinates. Original camera
extrinsics and jaw geometry are preserved; the user has checked the simulated
camera views against the real setup.

The SO-101 full profile uses three cameras. `molmo_room` (top) maps to `room`,
`wrist_cam` maps to `wrist`, and `realsense_rgb` (front) maps to `original_room`.
Both `eval.molmo_overhead_camera` and `policy.options.include_original_room`
are `true`. Each model query receives all three RGB images; results record
`policy_cameras: [room, wrist, original_room]`. The separate nine-zone baseline
smoke config retains the original front/wrist pair, with both options `false`.

For other Molmo configs, `--molmo-overhead-camera` adds a separate
view behind and above the arm, with image roll aligned to the table's front
edge. It leaves both original cameras in place. The new `molmo_room` sensor
then supplies `room`, the wrist remains `wrist`, and the original front view
remains available as `original_room` in Rerun and videos. To send that original
front image as a third model input, after `room` and `wrist`, set
`policy.options.include_original_room: true` in the YAML. Both options are
already enabled in the SO-101 full profile; to view that run in Rerun:

```bash
./docker/eval.sh pick_place_molmo_so101.yaml --rerun
```

The added camera is restricted to Molmo pick-place tasks and changes only the
requested evaluation configuration instance. Default scenes and other policy
profiles retain their existing cameras. Enabled runs record the added camera's
pose and settings under `molmo_overhead_camera` in their results. Disabling both
options restores the original front/wrist input pair.

Molmo also needs a joint-unit conversion in both directions. Simulation selects
`input_joint_units=usd_normalized` and reconstructs each actual USD arm angle
as `normalized * (maximum - minimum) / 200 + (maximum + minimum) / 2`, using
`sim_joint_mapping`. Matching the canonical arm CAD to `ARM_new.usd` establishes
the model angles as `[pan, 90 - lift, 90 + elbow, wrist_flex, wrist_roll]` in
degrees. This uses the signs and offsets from the
[official SO-100/101 frame correction](https://huggingface.co/docs/lerobot/main/en/molmoact2#joint-frame-transform-so-100101-zero-shot),
applied to the reconstructed USD angles. The gripper stays in 0–100 units.
For example, zero normalized pan reconstructs `1.7578125` USD/model degrees;
zero normalized elbow reconstructs `-5` USD degrees and `85` model degrees.
Discarding these range midpoints previously introduced pan and elbow offsets.

The original LeRobot
[degree calibration](https://github.com/huggingface/lerobot/blob/b536f47e3ff8c3b340fc5efa52f0ece0a7212a57/lerobot/common/robot_devices/robots/feetech_calibration.py#L421)
uses a fixed physical zero pose. Current LeRobot hardware
[`DEGREES`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py#L874)
instead centres on the recorded range midpoint. The simulator conversion is therefore
explicitly separate from the existing real-client conversion; saved hardware
calibration is unchanged. Direct plugin clients default to
`input_joint_units=lerobot_degrees`; real normalized callers use
`input_joint_units=lerobot_normalized` with five comma-separated
`arm_degrees_per_unit` scales. Simulator callers additionally supply the five
`arm_zero_degrees` range midpoints with `usd_normalized`.
`input_joint_units=model` accepts model-frame values directly.

Returned actions undergo the exact inverse conversion before the simulator's
normal joint-limit handling. The full startup log and results field
`policy_joint_transform` record the actual scales and offsets. Restart an
existing rollout to load client changes.

#### Official MolmoAct2 inference contract

The adapter uses the released checkpoint's `predict_action()` implementation.
The [SO-100/101 model card](https://huggingface.co/allenai/MolmoAct2-SO100_101)
and [LeRobot frame documentation](https://huggingface.co/docs/lerobot/main/en/molmoact2#joint-frame-transform-so-100101-zero-shot)
specify the following contract, checked on 2026-10-21 against downloaded model
revision `152569fe57914d97be91055800035f54e250d009`:

| Item | Workshop behavior |
| --- | --- |
| Weights | `allenai/MolmoAct2-SO100_101`, stored as `molmoact2_so101`; not the base checkpoint |
| State and actions | Six absolute joint positions/targets in the model's robot frame; arm degrees and gripper 0–100, with the inverse frame conversion on output |
| Normalization | Original `norm_stats.json`, tag `so100_so101_molmoact2`; the model performs normalization and denormalization |
| Images | RGB front and wrist images; the model card permits arbitrary camera order |
| Prediction | Continuous actions, depth reasoning disabled, 10 flow solver steps |
| Language and CUDA graphs | Use the installed checkpoint's defaults (both enabled), matching the saved 4/11 run |
| Chunk | Checkpoint maximum 30 targets; H30/R1 executes them at 30 Hz |

The paper's [SO-100/101 deployment recipe](https://arxiv.org/html/2605.02881v1#S4.SS3.SSS1.Px4)
also specifies 30 targets at 30 Hz, so the baseline preserves the trained timing.
Its [SO-100 evaluation table](https://arxiv.org/html/2605.02881v1#S6.T7)
reports a 33.3% mean score for block-in-box over 15 trials, awarding partial
credit for reaching and pickup. That is not a 33.3% completed-placement rate,
and it cannot be directly compared with this workshop's placement-only success
rate or taken as a guarantee of five successful smoke episodes.

All five installed weight-shard SHA-256 hashes were checked against that
revision's Hugging Face LFS metadata. The configuration, normalization statistics,
model source and shard index also matched their published Git blob hashes.

The released quantile normalizer clips its normalized action inputs to `[-1, 1]`
before mapping them into physical targets. Those learned output bounds are
different from the simulator's physical joint limits. Removing this clipping or
replacing the checkpoint statistics changes the released inference behavior.

The RTX 4090 profile uses a BF16 backbone with FP32 action-expert and sensitive
modules, restoring their original FP32 checkpoint tensors. The
[LeRobot mixed-precision path](https://huggingface.co/docs/lerobot/main/en/molmoact2#installation-requirements)
uses this module split; the model card's reference evaluation uses full FP32 and
also offers BF16 for lower-memory GPUs. This adapter additionally keeps action
denormalization in FP32 to avoid rounding absolute angles near 180 degrees.
Consequently this profile is not a bit-for-bit reproduction of full-FP32 evaluation.
FP32 weight storage alone also does not prevent the outer BF16 autocast from
evaluating expert projections in BF16. The optional research setting
`policy.options.action_compute_dtype: float32` additionally promotes incoming
VLM KV tensors and disables autocast around expert context projection,
modulation and flow prediction. It requires `action_dtype: float32`; the default
`autocast` retains the saved benchmark's computation. A CPU regression using a
tiny instance of the released expert implementation verifies unchanged weights and exact output
agreement with a separate FP32 expert reference. This does not establish better
rollout performance. Neither setting changes model normalization or clipping.

The existing `pan_model_scale` and `pan_model_offset` in the workshop YAML are
**experimental local policy adaptation**, not an official Molmo setting or a
physical calibration correction. They transform observations and actions
consistently, but alter the policy's effective pan range. To compare the native
released policy, copy the YAML and remove both options; start a separate report
because resume checks the transform. Camera geometry, simulator control frequency,
timeout and benchmark starts are also local evaluation choices. No visual-servo
or gripper override is enabled by this profile.

#### Progress-gated, per-zone Molmo smoke test

```bash
./docker/eval.sh pick_place_molmo_so101_smoke.yaml --smoke-test
```

This separate config keeps the released SO-101 weights, original front/wrist
inputs, H30/R1 and the existing experimental pan adaptation. It selects the
first original scheduled example in each available zone: MC, FR, MR, NL, FL,
NC, FC, ML and OUT. OUT is the benchmark's OOD label; no example is invented for
an unrepresented table zone. Original start IDs, poses, recorded robot starts
and seeds are retained. Selection happens after the full schedule's seed
assignment, so reducing coverage cannot accidentally reassign seeds.

The evaluation clock allows 60 **simulated** seconds to reach the cube, then
30 seconds from each new ordered milestone: reached, grasped, lifted, over box,
placed. For example, a first reach at 20 seconds sets the next deadline to
50 seconds. Reaching or grasping again cannot refresh an earlier milestone;
knocking the cube upward without grasping it does not earn more time. Progress
on the deadline is accepted, but progress after expiration cannot revive the
attempt. The configured 210-second ceiling is only the mathematical maximum
when every new milestone arrives just in time; an unreached cube still stops
the attempt at 60 seconds. Native task success and physics are unchanged.
The native grasp detector already requires a lift, so grasp and lift often
register on the same frame; simultaneous milestones earn one 30-second window,
not stacked extensions.

The deadline adds evaluation truncation, not a controller: model observations
and actions are untouched. The profile has no visual servo, gripper latch,
action smoothing, modified weights, or task-specific action generator. Rerun
shows each attempt. No evaluator report, recording, CSV or run log is saved.
The final `[SMOKE_RESULT]` console record supports automated comparisons; normal
console filtering hides that machine record, while `EVAL_VERBOSE=1` displays it.
The record retains the in-memory grasp measurements, including error direction,
jaw gap, closing events and tracking error, for failure analysis.
Legacy `--smoke-test` configs without this section still run five episodes with
their existing fixed timeout. These progress deadlines do not apply to the
ordinary 180-second full evaluation. Camera settings are included in resume
compatibility checks, so changing them requires a new run.

For research comparisons, hold these starts, seeds, deadlines and rendering
settings fixed. Compare the existing pan-adapted baseline separately from the
native unwarped frame; the latter retains the learned pan-range constraint
described above. Test one inference change at a time, repeat promising results
with additional seeds, and report reach, grasp, placement, time and regressions.
A nine-zone smoke score is a diagnostic sample, not a 150-start benchmark score.
Failed candidates cannot prove that every possible tuning is exhausted.

After a host restart on 2026-10-23, the public smoke command above completed all
nine zones with the unchanged front/wrist H30 profile: 2/9 reached, 1/9 grasped
and 1/9 placed (MR, 14.53 s sim / 20.71 s wall). MC reached but missed its grasp
and stopped at 30.03 s; the other seven failures stopped unreached at 60 s.
The browser Rerun view showed both camera streams and joint/action traces.
This verifies the public launcher path, rather than only the private research
runner. It is a fresh diagnostic run, not a replacement for the completed
150-start report or evidence that the model's possible improvements are exhausted.

A fresh matched camera comparison then completed all nine starts at root 1984,
using the same frozen source, seeds, H30/R1, released weights and deadlines:

| Model inputs | Reached / grasped / placed | MC success time (sim / wall) | MR success time (sim / wall) |
| --- | --- | --- | --- |
| Top + wrist | 2/9 / 2/9 / 2/9 | 35.20 / 61.64 s | 16.90 / 28.97 s |
| Top + wrist + front | 2/9 / 2/9 / 2/9 | 14.43 / 26.65 s | 40.33 / 66.36 s |

Both placed the same two starts; all seven other starts stopped unreached at
60 simulated seconds. Three images sped up MC but slowed MR. Mean successful
episode time was 26.05 s sim with two images and 27.38 s with three; the screen
does not establish a success-rate or overall speed improvement. FR's closest
tool error remained about 69 mm in both settings. The requested 150-start run
uses three inputs to measure that configuration on the full benchmark; this is
not a claim that it solved the remaining approach failures. Only the Molmo full
YAML changes camera selection; original camera poses and default scene configs
are preserved.

The subsequent isolated three-camera shoulder-range screen retained that
control's frozen source, nine starts, root 1984, H30/R1 and 60/30-second deadlines.
It compressed positive physical shoulder angles by 0.75 on model input and
applied the inverse to returned actions; the other five joint conversions were unchanged.
The recorded transform confirms the candidate was active. This is an
experimental state/action adaptation, not an official calibration correction.

| Three-camera setting | Reached | Confirmed grasps | Successful placements |
| --- | --- | --- | --- |
| Existing H30 control | 2/9 | 2/9 | 2/9 (MC, MR) |
| Shoulder gain 0.75, H30 | 5/9 | 2/9 | 1/9 (MR) |

FR, FL and FC newly reached at 4.67, 4.83 and 8.90 simulated seconds, respectively,
but all missed their grasps within the next 30 seconds. Their closest tool errors
fell from 69.47/53.35/64.50 mm to 11.29/15.29/12.32 mm. At those closest frames,
maximum joint tracking errors were 1.73/1.42/1.73 degrees and jaw gaps were
42.61/58.61/30.64 mm around the 20 mm cube. The remaining miss therefore cannot
be described as a solved grasp simply because the reach milestone fired.
NL's smaller closest error of 33.16 mm coincided with a 132.10-degree maximum
tracking error; this transient does not demonstrate accurate target tracking.

MR succeeded at 19.73 simulated seconds. MC registered a transient placement at
12.67 seconds but did not satisfy the unchanged 25-step confirmation rule, and
stopped at 42.67 seconds. The aggregate flags do not establish whether recontact
or leaving the container caused that confirmation failure. NL, NC, ML and OUT
never reached. Other GPU workloads were active late in the run, so wall times
are not an isolated throughput comparison. No shoulder adaptation was promoted
to the prepared full profile. More reaches without more grasps, plus the lost MC
placement, falsify the claim that this range change alone solved the task.
A follow-up can hold the candidate fixed while shortening execution to H10,
testing whether more frequent observations improve final alignment and closure;
its premortem is increased hesitation, recontact or loss of useful later targets
from the released 30-target chunk. Success must still be measured as placements.

The first eight completed nine-zone comparisons on 2026-10-22 used root seed
1984 and exactly the same original per-start seeds, recorded starts, 60/30-second
progress deadlines, continuous rendering and released weights:

| Inference setting | Reached | Grasped | Successful placements | Successful placement time |
| --- | --- | --- | --- | --- |
| Existing mixed-precision control, H30/R1 | 2/9 | 1/9 | 1/9 (MC) | 25.50 s sim |
| Existing mixed-precision computation, H10/R1 | 2/9 | 1/9 | 1/9 (MR) | 11.97 s sim |
| FP32 expert computation, H30/R1 | 2/9 | 0/9 | 0/9 | — |
| FP32 expert computation, H10/R1 | 2/9 | 1/9 | 1/9 (MR) | 26.97 s sim |
| FP32 expert computation, 20 solver steps, H30/R1 | 2/9 | 1/9 | 0/9 | — |
| FP32 expert computation, native pan, H30/R1 | 2/9 | 1/9 | 0/9 | — |
| FP32 expert computation, three cameras, H30/R1 | 2/9 | 0/9 | 0/9 | — |
| FP32 expert computation, longer instruction, H30/R1 | 2/9 | 1/9 | 1/9 (MC) | 23.43 s sim |

All eight retain a BF16 vision/language backbone; none is a full-FP32 model
evaluation. All reached MC and MR; the other seven attempts expired before a
first reach. Both H10 variants gained MR but lost MC, so this screen shows no
net placement gain and does not justify changing the default horizon. The
11.97-second and 26.97-second MR placements used the same start and seed with
different expert computation; timing successful attempts on different starts
does not establish a general speed advantage. FP32 expert computation
alone also showed no improvement. With 20 flow-solver steps, MR was lifted and
then dropped before the box. Removing both pan-adaptation options instead left
MR held until its progress deadline at 55.73 simulated seconds, without reaching
the box; it added no reached zone. This agrees with the native pan-range
constraint discussed above, without proving a universal model limit.

Adding the overhead image alongside front and wrist scored 0/9, matching the
FP32-compute two-camera control; it added no reached zone or grasp. This does
not substitute for the separate three-camera comparison with baseline
computation or for a two-external-view comparison. The longer instruction,
“Move the arm towards the blue cube, grasp it, lift it up, and drop it into the
white box,” completed MC in 23.43 seconds with FP32 expert computation, while
the shorter-instruction FP32 screen had no placement. It still missed MR and
the other seven starts, so its total did not exceed the baseline's 1/9.

These are initial screening results, without a claim of repeatability or a limit
on model capacity; further candidates and paired validation are separate
evidence. No production inference defaults were promoted from these results.

Matched H30/H10 verification subsequently completed roots 9182 and 4173, keeping
the same nine original positions and each root's original per-start seeds.
Across roots 1984/9182/4173, H30 placed 1/9, 0/9 and 2/9; H10 placed 1/9, 1/9
and 1/9. Both therefore placed **3/27**. H10 placed MR in all three seed sets
but never MC; H30 placed MR once and MC twice. H10 also reached NC once without
grasping it. This is a tradeoff, not a net placement improvement, so H30 remains
the default. These are repeated trials on nine positions, not 27 newly sampled
positions or a full-benchmark estimate. A new camera/precision batch uses its
own matched control and records returned chunks in RAM for failure analysis.

That batch completed these additional comparisons at root 1984, with the same
nine starts and deadlines. All four below use the existing mixed-precision
computation, rather than the FP32-expert variant above:

| Setting | Reached | Grasped | Successful placements | Successful placement time |
| --- | --- | --- | --- | --- |
| Front/wrist, H30/R1 control | 2/9 | 1/9 | 1/9 (MC) | 29.30 s sim |
| Front/wrist, H10/R1 | 3/9 | 1/9 | 1/9 (MR) | 18.23 s sim |
| Overhead/wrist/front, H30/R1 | 2/9 | 1/9 | 1/9 (MC) | 14.40 s sim |
| Overhead/front, H30/R1 | 2/9 | 1/9 | 1/9 (MR) | 26.53 s sim |

The three-image setting completed MC faster but missed MR; the two external
views gained MR and lost MC. Neither added a reached zone or a placement to
the H30 control. These single-root screens do not establish repeatability or
exclude a benefit on other starts. The H10 setting reached NC without grasping.
Episode results and grasp diagnostics are complete for all four settings;
the separate query trace for the three-image setting is partial (218 of 474
queries), so it cannot support claims about every returned chunk.

The remaining overhead/wrist pair was then compared against a fresh front/wrist
control with the same frozen source, root 1984, original starts and deadlines:

| Model inputs, H30/R1 | Reached | Grasped | Successful placements | Placement times |
| --- | --- | --- | --- | --- |
| Front/wrist control | 2/9 | 1/9 | 1/9 | MC 11.47 s sim |
| Overhead/wrist | 3/9 | 2/9 | 2/9 | MC 29.13 s, MR 39.20 s sim |

The original front sensor remained in the scene but was excluded from the
overhead/wrist model input. MR's closest measured approach before grasping was
2.03 mm, versus 22.44 mm in the failed control. FL newly reached at 33.03 seconds,
then missed its grasp and stopped at 63.03 seconds under the agreed progress
deadline. The camera setting retained MC's placement but took longer on that
start. FL's closest measured tool error was 27.53 mm. In its last 20 queries,
62.0% of shoulder targets and 46.5% of elbow targets equalled the learned lower
quantiles; median query-boundary arm tracking errors remained below 1.51 model
degrees. This motivates a separate range experiment, not a claim of incorrect
physical calibration. This is an initial placement gain on nine positions, not a verified
benchmark improvement. No production defaults were promoted from this result.

Root 9182 subsequently completed the same matched comparison: front/wrist placed
0/9; overhead/wrist placed 1/9 (MC, 17.40 s sim). MR was grasped, lifted and brought
over the box but did not confirm placement; the native classifier recorded
`dropped_outside`. FL's new reach from root 1984 did not repeat. Across these two
completed seed sets, the control has 1/18 placements and 1/18 confirmed grasps;
overhead/wrist has 3/18 placements and 4/18 grasps. These are repeated attempts on
the same nine positions, not 18 distinct positions. Two additional placements
are still too little evidence to establish a reliable population improvement.
Root 4173 and broader validation remain pending.

The full-FP32 attempt in the preceding batch failed on its first inference, before any
episode could be scored. With Isaac sharing the RTX 4090, the released text
embedding lookup temporarily concatenated its weight tables and requested
another 1.48 GiB while only 1.13 GiB was free. This is a runtime failure,
not a 0/9 policy score. A private retry uses separate lookups into the unchanged
tables to remove that temporary allocation. On an exact captured input and
CUDA RNG state, its 30-by-6 action targets matched the original lookup bit for
bit, and peak PyTorch allocation fell from 18.69 to 17.64 GiB. This comparison
used FP32 CPU vision and GPU language/action modules in both cases. It validates
the lookup's numerical behavior in that setup, not better task performance;
the retry remained separate from production defaults.

The repaired full-FP32 retry then completed all nine starts without another
memory failure: **2/9 reached, 0/9 grasped, 0/9 placed**. MC and MR missed their
grasps; the other seven expired before reaching. All 491 returned query chunks
and four captured input packets were retained and checked after server shutdown.
Aggregate rollout throughput was 8.03 control steps per wall second, versus
23.68 for the fresh mixed-precision control, including inference and rendering
but excluding model startup. This single-root result provides no reason to
promote that precision/device setup. It neither reproduces the model card's
all-GPU execution nor proves that precision never matters on other inputs.

That fresh H30 control again placed 1/9 (MC, 29.30 s sim). Its FR and FL closest
errors were 68.93 and 41.84 mm, predominantly along local gripper Z. In the last
20 queries, 42.8%/51.0% of FR shoulder/elbow targets and 83.8%/88.0% of FL
targets equalled the released lower quantiles. Median arm tracking errors at
query boundaries were below 1.63 model degrees. These are learned output bounds,
not physical actuator stops; the boundary measurements exclude gripper error
and transient tracking during motion.

A separate conservative geometric bound used the original USD anchors,
a covering pan/lift grid with an analytic rotation-displacement allowance,
and the sum of downstream segment lengths. For the first OUT start, the
requested fixed diagnostic point remains at least 67.6 mm from the stored cube
position under the baseline's learned pan/lift bounds, even allowing unrestricted
downstream joint rotations. This constrains requested static poses; it does not
bound every possible finger contact, actuator overshoot, pushing or moving cube,
and cannot prove that all tuning or eventual dynamic success is impossible.

A CPU-only expansion of that audit on 2026-10-23 used the current USD anchors,
the checkpoint's action quantiles, the existing pan adaptation and the stored
benchmark scene. It distinguishes a conservative **lower bound** on error from
the error of a **numerically found pose**. The latter is an example within the
joint ranges, not a global minimum or a collision-free grasp. For the seven
starts that failed in the fresh three-camera smoke comparison:

| Zone / start | Static error lower bound | Found static tool error | Observed smoke closest tool error |
| --- | --- | --- | --- |
| FR / id-001 | 13.11 mm | 17.58 mm | 69.47 mm |
| NL / id-003 | 0 mm | <0.001 mm | 98.95 mm |
| FL / id-004 | 0 mm | <0.001 mm | 53.35 mm |
| NC / id-006 | 0 mm | <0.001 mm | 45.03 mm |
| FC / id-011 | 5.92 mm | 10.38 mm | 64.50 mm |
| ML / id-013 | 0 mm | <0.001 mm | 144.02 mm |
| OUT / ood-000 | 67.96 mm | 72.49 mm | 107.95 mm |

These are different measurements: static calculations target the **stored**
cube centre, while smoke diagnostics measure the live cube. The fixed tool
point is the same CAD-derived point used by grasp diagnostics. The numerical
poses do not constrain gripper orientation, table/self collisions, or the path
from the reset pose. A zero lower bound alone proves nothing; the near-zero
found errors show point-position solutions for NL, FL, NC and ML. Thus output
range alone does not exclude those point positions. Conversely, camera or
sampling changes cannot make an unchanged bounded static target put that same
point exactly at FR, FC or OUT's stored centre. This does not exclude other
finger contact, pushing, moving the cube or dynamic overshoot.

Across all 150 stored starts, 30 have this conservative lower bound above
1 mm and 11 above 30 mm. All 30 were classified `never_reached` in the completed
front/wrist baseline. This association does not establish causality, a success
ceiling, or a reason to remove benchmark starts. Even full physical joint ranges
leave a 23.28 mm lower bound for the selected OUT diagnostic point; that is
still not a bound on every possible contact or successful manipulation.

A hypothetical 0.75 scale on positive physical shoulder angles expands the
permitted shoulder maximum from 44.82° to 59.76°. The same static solver then
finds point-position solutions for FR and FC, while OUT retains a 44.94 mm lower
bound. **This range change is not enabled.** It changes state/action semantics,
not physical calibration. Its premortem is that wider output range may still
produce the same inaccurate predictions, distort state conditioning, cause
collisions or lose MC/MR grasps. It would need a matched nine-zone smoke and
additional seeds before any promotion; geometry is insufficient evidence of a
performance improvement. The prepared three-camera 150-start profile is unchanged.

An orientation follow-up also searched four cube-face alignments, requiring the
gripper's Z axis to follow the cube's upright Z axis. Under the baseline ranges,
it found poses within 1 mm and 5° for MC, MR, NL, NC and ML. NL and NC were within
0.001 mm and 0.001°; ML's best residual was 0.43 mm and 1.52°. These still lack
collision and path checks, but rule out a simple absence of any cube-aligned
static pose for those selected starts. FL's point-only solution did not satisfy
this extra orientation test. The shoulder-range candidate did not add a start
meeting the orientation tolerances. Even the full physical-range search found
no such pose for FR, FL, FC or OUT. Failed numerical searches are not proofs of
infeasibility, and a strictly upright grasp is **not required** for task success:
angled grasps remain possible. This check therefore challenges the assumption
that extending shoulder range alone guarantees a useful grasp; it does not
justify adding an orientation controller or changing evaluation criteria.

Reproduce the CPU diagnostic with NumPy, SciPy, PyYAML and the USD Python bindings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_molmo_target_reach.py
```

Add `--check-orientation` to reproduce the optional cube-aligned pose search.
It uses a 50 mm/radian orientation residual and reports position and angle errors
separately; the search's weighting and tolerances are not simulator success rules.

It uses a covering 0.125° pan/lift grid and subtracts an analytic rotation
displacement allowance (about 0.324 mm), rather than treating sampled poses as
an impossibility proof. Downstream rotations are relaxed to a containing ball.
The zero-pose chain matches authored link transforms within 0.02 micrometres;
10,000 sampled physical poses also passed the ball-inclusion check. Tests cover
an exact pose between grid points, conservative bounds and the recorded-training
orientation rule. Source hashes, bounds and found joint vectors are written only
to a fresh `/dev/shm/molmo-target-reach-*/report.json`; no model or simulator runs.

An offline projection of the 150 stored cube poses through the configured
camera intrinsics/extrinsics found eight OUT starts partly below the front
image's bottom edge; none was wholly outside. The original `ood-000` room video
confirms that clipping at frame 10. All 150 cube boxes fit the added overhead
camera's image bounds before accounting for robot occlusion. Projection was
checked against captured MC/FR inputs from both external cameras. This identifies
a specific framing limitation, not a general sim-to-real mismatch or a sole
explanation for failure: the other starts fit the front view, and adding the
overhead image in the completed three-camera screen did not improve placements.
Original camera poses and benchmark starts remain unchanged.

`scripts/molmo_capability_sweep.py` automates the documented comparison with
RAM-backed state under a required `--state-dir /dev/shm/NAME`. An optional
`--wait-pid PID` waits for an existing evaluation launcher without signalling it.
The runner also waits for the policy port and sufficient GPU memory, freezes
client/server source in RAM, and runs the baseline, FP32-compute, shorter-chunk,
more-flow-step and native-frame comparisons. A separate camera-input comparison
adds the existing Molmo-only overhead sensor and feeds all three images to the
model; it does not run a vision-based controller or alter the original cameras.
If those candidates do not improve the screening score, it also tests a fixed,
more explicit instruction describing the same task. It verifies the baseline and the
leading candidate with two additional matched seeds. Existing reports and
videos are never used as output paths. Its `planned_trials_complete` status
means only that the planned tests ended: findings still require a premortem,
validation and review before any default configuration is changed.
Incomplete results are rejected even if Isaac exits with status zero. Failure
reports preserve typed exceptions such as `torch.OutOfMemoryError` and the
owned policy server's recent output before cleanup, so shutdown warnings cannot
hide the original cause. Failed attempts are never added as completed scores.
Policy containers are retained until the runner reads their exit state and logs,
then removed by its owned-container cleanup. This covers failures that occur
before the readiness port opens, including a startup exit 139 observed before
any H10 follow-up episode ran. That infrastructure failure has no policy score.
Episode summaries are retained even when tqdm appends them to a progress line.

Programmatic follow-ups can construct `Investigation(args,
camera_inputs=('room', 'original_room'))` to compare the added overhead view
and the original front view, excluding the wrist from model input. Pass
`overhead=True` to that instance's trial so both external sensors are available.
This edits only the new RAM snapshot's manifest before hashing it; the normal
manifest, original camera poses and task stay unchanged. Trial records and
validation use the actual selected camera names rather than relabelling a
three-image result as a two-image comparison.

To inspect the initial pose in Isaac without loading a policy, run:

```bash
docker exec -it -w /workspace/Sim-to-Real-SO-101-Workshop teleop \
  /workspace/isaaclab/_isaac_sim/python.sh scripts/preview_eval_pose.py \
  --config pick_place_molmo_so101.yaml
```

The preview uses the YAML's USD and first benchmark start, holds the reset pose
for the evaluator's ten settling steps, then pauses physics. The Isaac window
stays open for inspection. External/wrist images and `pose.json` are saved under
`outputs/preview_molmo/` (`--out_dir` overrides this). Close Isaac to end it.

The table maps former `eval_pick_place.sh` options to YAML keys. For example,
set `eval.episodes: 5` for a five-attempt run without a fixed benchmark, and
`grasp.gripper_latch: true` to enable the latch. This launcher has no `--set`
override flag.

Preflight checks Docker access without sudo, running `teleop`, the server image, checkpoint
containment/existence, `config.json` and weight files, dataset and policies folders inside teleop,
valid values, and a free host port. Dry-run validates the YAML but skips the Docker/filesystem checks.
The missing-teleop diagnostic extracts the launch block from the hub README.

The script starts a uniquely named detached, auto-removed policy server, waits for TCP readiness,
runs `lerobot_eval` inside teleop with Isaac's Python wrapper, and prints a compact summary.
It retains server and evaluator logs so failure details survive automatic container removal.
Exit/interrupt traps clean up this run's server, evaluator and owned viewer.
The teleop container stays available. A readiness
connection is not a complete inference health test.

### Results JSON

Results go to `<dataset>/../eval_results/`, inside teleop. Filenames combine the
model path (slashes replaced by underscores), `nodr`/`dr` and a UTC timestamp.
The example's directory is inside the host-mounted `datasets/`; arbitrary input
paths need their own persistence. The console prints the exact output path.

| Field | Contents |
| --- | --- |
| `schema_version` | 1 |
| `task`, `checkpoint`, `seed`, `time` | Run identity; `time` is UTC ISO text |
| `requested_episodes`, `complete` | Requested count and whether it was reached |
| `random_fraction` | Actual probability of random cube starts used by this run |
| `episode_length_s`, `action_horizon`, `action_repeat`, `step_dt` | Configured episode seconds, predicted targets per query, control steps per target and simulation seconds per control step |
| `replan_interval_s` | Nominal simulation seconds per query: `action_horizon * action_repeat * step_dt` |
| `molmo_overhead_camera` | Added camera pose/settings when the optional Molmo view is enabled; absent otherwise |
| `policy_cameras` | Actual Molmo image inputs in request order, also printed once at connection |
| `episodes` | Per-attempt rows, including `success_step`, `elapsed_sim_s`, `time_to_success_s`, `wall_time_s`, `wall_time_to_success_s`, and optional `videos` paths |
| `overall` | `episodes`, `successes`, `success_rate`, `successes_within_15s`, `success_rate_within_15s`, `median_time_to_success_s` |
| `by_start` | Aggregates for `recorded` and `random` (not individual ID groups) |
| `by_zone`, `by_cube_color` | Aggregates for observed zones/colours |
| `save_videos`, `video_dir` | Present when recording is enabled; the absolute video directory inside teleop |

Rates are in [0,1], or null for an empty aggregate. `start_index=-1` means a random
start; recorded indices refer to the sorted input sidecar list. If you pass a
filtered copy, use its preparation mapping to recover source IDs.

The 15 s rate counts successes whose `success_step × step_dt` is at most 15 s,
using all completed episodes as the denominator. The ordinary success rate uses
the configured full episode window. Median time-to-success uses successful
episodes only and is null when there are none. Start/zone/colour aggregates carry
the same added metrics, and the saved JSON report includes the run options.
`time_to_success_s` and `wall_time_to_success_s` are null for unsuccessful attempts;
`elapsed_sim_s` and `wall_time_s` still record the time spent on those attempts.
These fields are additive, so `schema_version` remains 1.

## Recorded starts and DR

`--cube_starts` is a **client** flag; the host runner derives it from
`--dataset`. `PICK_PLACE_EVAL_STARTS` is the reset function's fallback when no
explicit path is supplied. Each sidecar must contain finite metre-valued frame-0
position and a nonzero wxyz quaternion in the robot base-link frame. The reader
uses only frame zero for starts, not later trajectory frames.

Cycle mode reshuffles all recorded starts using the seed and consumes each once
before reshuffling. Random mode samples recorded starts uniformly with replacement.
With the default `random_fraction=0`, only saved cube poses are used. Keep this
setting for the yaw-aligned option B protocol. An explicit nonzero
`random_fraction` chooses random XY within recorded bounds and yaw in [0,π/2),
standing upright on the live table plane. This is probabilistic, not an exact
fraction of a small run. Recorded Z is discarded: the live table support is solved
along base Z while preserving recorded XY/orientation. Overlap with the box fails
explicitly for recorded starts; random candidates are rejected (limit 10,000 tries).

**Arm start pose.** By default (`--robot_start recorded`) the arm is also reset to the
same episode's recorded first-frame `observation.state`, read from the dataset that owns
`pick_place_meta/` (its `data/*/*.parquet`) and converted to sim radians with the same
calibrated joint mapping the recorder used; random starts use the mean recorded state.
This matters: teleoperated demos start wherever the leader arm rested, which can be far
from the scene's calibrated reset pose, and a policy started off its training
distribution moves erratically. `--robot_start default` restores the calibrated reset
pose. On the real robot, start the arm from the same rest pose as the demonstrations.

Zones divide recorded XY bounds into thirds. N/M/F means near/middle/far from the
base; L/C/R is left/centre/right when looking away from the base. The away axis is
derived from the bounding-box centre, not assumed to be world X. Degenerate axes
use the middle third. Compare counts as well as rates; one success in one trial
is weak evidence of coverage.

**Host**, repository root, DR with matching colour instructions:

```bash
cat > /tmp/pick_place_dr.yaml <<'YAML'
task: pick_place_dr
policy:
  name: groot_n17
  checkpoint: so101_pick_place_v2_nvinit/checkpoint-10000
  models_dir: ~/sim2real/models
  options:
    embodiment_tag: NEW_EMBODIMENT
eval:
  dataset: /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2
  episodes: 20
  robot_start: recorded
  action_horizon: 16
  episode_length_s: 30
  render_warmup: 2
  lang_by_color:
    blue: Pick up the blue cube and place it in the white box
    red: Pick up the red cube and place it in the white box
YAML
./docker/eval.sh /tmp/pick_place_dr.yaml
```

DR randomizes robot colour, blue/red cube colour and the measured room tube-light
exposure when that light exists. Cameras are **not** randomized: the sim cameras are
calibrated to the real ones (`real_setup.json`), so the external and wrist views
stay identical to the real robot's. The domain-randomized training replay
(`lerobot_dr_replay`) follows the same rule. MyRoom has no LightStudio/sky-light HDRI/mat;
those DR targets are skipped. A custom room without the measured tube light also
skips that exposure term. Inspect the source ranges for your scene rather than
assuming they model your hardware variation.

Read `by_cube_color` with the actual instructions and training distribution.
Poor red performance after blue-only demonstrations suggests a colour/language
coverage gap, not necessarily a geometry failure. Do not interpret DR success as
a guarantee on the real robot. The current fork has no PickPlace DR teleop ID.

## Direct client and renderer checks

**Teleop container**, no policy server needed, preview recorded starts and DR:

```bash
export PICK_PLACE_EVAL_STARTS=/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2/pick_place_meta
zero_agent --task Lerobot-So101-Teleop-Pick-Place-Eval --num_steps 2250
zero_agent --task Lerobot-So101-Teleop-Pick-Place-DR-Eval --num_steps 4500
```

Confirm startup event order, cube support on the actual tabletop, no box overlap,
and varying camera/light/robot/cube appearance. These checks require a GPU and
renderer. Passing CPU tests is not proof
that rendering was tested on your hardware.

**Teleop container**, with a server already listening, direct evaluation:

```bash
lerobot_eval --task Lerobot-So101-Teleop-Pick-Place-Eval --num_envs 1 \
  --cube_starts /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2/pick_place_meta \
  --rename_map '{"realsense_rgb":"room","wrist_cam":"wrist"}' \
  --policy_host localhost --policy_port 5555 --episode_length_s 30 --action_horizon 8 \
  --lang_instruction 'Pick up the blue cube and place it in the white box' \
  --num_episodes 20 --seed 1984 \
  --checkpoint so101_pick_place_v2_nvinit/checkpoint-10000 \
  --results_json /workspace/Sim-to-Real-SO-101-Workshop/datasets/eval_results/manual.json \
  --headless
```

The direct parser additionally accepts `--disable_fabric`, `--rerun`, `--action_repeat`,
`--smoke-test`, `--molmo-overhead-camera`,
`--start_mode` (cycle), `--random_fraction` (0.0), `--robot_start` (recorded), `--lang_instruction_by_color`
(JSON), and the inherited [Isaac launcher flags](04-recording.md#inherited-isaac-launcher-options).
Its defaults differ from the host runner: MyRoom task, 10 episodes, seed 1984,
no rename map/results path/cube-start directory, `num_envs=None`, and language
`Pick up the vial and place it in the rack`. Set an Eval task explicitly: default
MyRoom has no success/timeout terminations. Checkpoint metadata falls back to
`PICK_PLACE_EVAL_CHECKPOINT`, then `MODEL`, then `unknown`.

## Smooth control

The simulation client uses NumPy-only smoothing in
[`policy_client/smoothing.py`](../source/sim_to_real_so101/policy_client/smoothing.py).
All options default to off; the original simulation action queue remains the
default path.
Smoothing operates in the existing LeRobot joint units before the unchanged
simulation joint mapping. It does not change the checkpoint, server, or success
criteria. EMA and prediction history reset on every simulation episode, including
manual resets.

| Option | Default / behavior |
| --- | --- |
| `--prefetch_steps P` | 0, off; require `0 <= P < action_horizon` (simulation: `action_horizon * action_repeat`) |
| `--blend_steps K` | 0, off; linearly cross-fade old unexecuted predictions into a replacement over K steps, with new weights `1/K, 2/K, ..., 1` |
| `--ema_alpha A` | 1.0, off; `sent = A * action + (1-A) * previous_sent`, finite `0 < A <= 1`; first action passes through |
| `--smooth_gripper` | False; include gripper in EMA. Otherwise gripper passes through EMA; blend and ensemble still include it |
| `--temporal_ensemble M` | 0, off; query every M control steps and average every prediction that covers the current step |
| `--te_decay k` | 0.01; ensemble weights `exp(-k*i)`, `i=0` oldest, normalized over the current overlaps; finite `k >= 0` |
| `--log_timing` | False; per-action CSV and exit summary |
| `--timing_csv PATH` | Empty; optional CSV destination, otherwise `outputs/timing/sim_<timestamp>.csv` relative to the working directory |

`--temporal_ensemble` and `--blend_steps` are mutually exclusive. EMA may follow
either. Argparse booleans such as `--log_timing` and `--smooth_gripper` are bare flags.

**Simulation timing.** There is no wall-clock scheduler or inference thread.
The first query delivers immediately. In chunk mode, query synchronously from
the observation **P steps before the current executable chunk ends**, keep
executing the old chunk for those P simulation steps, then deliver the new chunk
and drop exactly its first P actions. Execute at most
`min(action_horizon, chunk_length-P)` remaining actions; thus a 16-action model
with the preset below executes 12 actions per replacement. A synchronous query
can take real time, but that time never advances simulation steps or changes the
selected actions. In ensemble mode, queries occur at steps `0, M, 2M, ...`;
all queries after startup deliver P steps later, even when several are pending.
Weights include only delivered predictions that cover the current step.
`--render_warmup` uses this adjusted query schedule to refresh camera frames.
When simulation uses `action_repeat > 1`, each native target is expanded into
that many control ticks before smoothing. Its effective execution horizon is
`action_horizon * action_repeat`, and chunk length, prefetch, blending and
ensemble intervals all count those expanded control ticks. The simulation
formula above then uses the effective horizon and expanded chunk length.

The CSV contains control step, active/latest contributing chunk ID, that chunk's
inference latency in seconds, per-tick overrun, hold indicator, and raw/sent values
for all six joints. Raw values precede blend/EMA; for an ensemble they are its
weighted prediction. The exit summary reports mean/p95 latency **per completed
inference**, total overruns (zero for sim), and mean joint jerk: the mean over
steps of `sum(abs(a[t] - 2*a[t-1] + a[t-2]))` on the five arm joints, in LeRobot
units per step squared. Jerk excludes episode boundaries. Sim CSV steps count
policy actions across episodes, excluding the existing ten settling steps.

Recommended starting preset:
`--action_horizon 12 --prefetch_steps 4 --blend_steps 4 --ema_alpha 0.6`.
Compare against the saved unsmoothed rollout with the same checkpoint and starts.

With the [N1.7 200k server below](#gr00t-n17-200k-simulation-server) already serving on port
5560, run the smooth **simulation** client inside teleop:

```bash
lerobot_eval --task Lerobot-So101-Teleop-Pick-Place-Eval --num_envs 1 \
  --cube_starts /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta \
  --rename_map '{"realsense_rgb":"room","wrist_cam":"wrist"}' \
  --policy_host localhost --policy_port 5560 --episode_length_s 30 \
  --action_horizon 12 --prefetch_steps 4 --blend_steps 4 --ema_alpha 0.6 \
  --lang_instruction 'Pick up the blue cube and place it in the white box' \
  --num_episodes 20 --seed 1984 \
  --checkpoint so101_pick_place_sim_n17_200k/milestones/checkpoint-200000 \
  --log_timing --timing_csv /tmp/so101_sim_smooth.csv --headless
```

To try ensemble instead, replace `--blend_steps 4` with
`--temporal_ensemble 4 --te_decay 0.01`.

## GR00T N1.7 200k simulation server

This serves the external-storage milestone checkpoint on port 5560 for the
simulation client. Start it in a host terminal and keep it running after
`SERVER_READY port=5560`:

```bash
cd ~/Sim-to-Real-SO-101-Workshop
cat > /tmp/groot_n17_200k.yaml <<'YAML'
task: pick_place
policy:
  name: groot_n17
  checkpoint: so101_pick_place_sim_n17_200k/milestones/checkpoint-200000
  models_dir: /external_storage/models
  port: 5560
  options:
    embodiment_tag: NEW_EMBODIMENT
eval:
  dataset: /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1
  robot_start: recorded
  action_horizon: 8
  episode_length_s: 30
YAML
./docker/eval.sh /tmp/groot_n17_200k.yaml --server_only
```

### Optional grasp latch and diagnostics

The simulation client uses `policy_client/grasp.py`. Everything below is off by default;
no model, training, joint mapping, or success/grasp detection logic changes.
The configured action horizon stays fixed; there is no `--h_grasp` option.

`--gripper_latch` enables the following sequence, after any arm/gripper smoothing:

- Derive closed/open action units from the named `gripper.pos` column's minimum
  and maximum in the **training** dataset's `meta/stats.json`. The default dataset
  is `datasets/pick_place_sim`; use `--latch_dataset` for another training set.
- Engage below `closed + 0.50 * (open - closed)` for 3 consecutive predictions.
  Require the maximum commanded arm change to be at most 2 units per step across
  the last 10 changes (11 command samples). Unsteady steps reset close confirmation.
- While latched, command the training-data fully-closed value. Normal release
  needs predictions above `closed + 0.80 * (open - closed)` for 5 consecutive
  steps, with at least 30 steps since engagement. Held/stale commands do not count
  as new close/release evidence. Only the gripper is overridden.
- Read measured gripper position each step by reversing the existing simulation
  joint mapping. The empty-closed reference defaults to the minimum **state** gripper
  position in training metadata. Override with `--gripper_empty_closed`.
- If latched and measured position is at most `empty_closed + 3` for 15 consecutive
  steps, log `missed grasp`, release, command fully open for 15 steps, then require
  a fresh close confirmation. This measured-empty escape is an exception to the
  normal minimum hold. A measured stall above that threshold never causes an
  automatic timed release, however long the latch has been held.

All limits are configurable: `--latch_close_below` and `--latch_open_above` accept
**absolute gripper units**, not percentages; `--latch_confirm_steps`,
`--latch_release_confirm`, `--latch_min_hold`, `--latch_steady_steps`,
`--latch_steady_max_delta`, `--latch_empty_margin`, `--latch_empty_confirm`, and
`--latch_reopen_steps` expose the other defaults above. Startup prints the actual
closed/open range, both thresholds, and the empty-closed reference. The 200k
training metadata currently resolves C≈44.376, O≈70.955, closed≈0.07704,
open≈88.67488, empty-closed≈1.22932 (empty threshold≈4.22932).

`--seed_per_episode` reissues the same acknowledged server noise seed **before
every inference query** within an episode. The benchmark supplies its existing
per-start seed; other sim runs use `--seed` (1984 by default). The generic policy server exposes
`seed_policy` and the GR00T plugin resets its RNGs through its reset endpoint.
An external server must expose `seed_policy`; an absent acknowledgement fails
explicitly. Keep one client per server when using this option.

`--log_gripper` writes a per-step CSV, optionally at `--gripper_csv`. It records
raw predicted, measured, and final commanded gripper; latch state/events;
steadiness; chunk ID; per-joint and norm changes between successive **full chunk
arm endpoints**, in LeRobot arm units (not Cartesian distance); thresholds;
and commanded arm positions. With temporal ensembling, “raw” means the newest
applicable chunk's prediction. The first endpoint delta and missing predictions
on held ticks are NaN. CSV episodes flush on completion/reset or Ctrl+C.

At episode end, the client prints the observed raw predicted gripper min/max
in the final two seconds before the first observed grasp, or the episode end if
there was no grasp. This helps move C into a hover region. The dither score counts
arm direction reversals, summed over the five **commanded** arm joints in that
same window; changes ≤1e-6 units are treated as zero. It ignores gripper movement
and velocities crossing the window boundary. The benchmark uses its unchanged
StageTrace grasp signal, not latch engagement. Grasp time includes the sim's ten settling
steps; diagnostic command samples start after settling.

Run this 200-start on/off comparison **after training releases the GPU**. Each
variant runs all 200 starts (400 episodes total at horizon 8):

```bash
cd ~/Sim-to-Real-SO-101-Workshop
python3 scripts/sweep_action_horizon.py \
  --model so101_pick_place_sim_n17_200k/milestones/checkpoint-200000 \
  --models_dir /external_storage/models \
  --eval_set eval_sets/pick_place_v1_75ep_seed1984.json \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --latch_dataset datasets/pick_place_sim \
  --horizons 8 --gripper_latch off,on --seed_per_episode \
  --episode_length_s 20 --repeats 1 --port 5560 \
  --out_root outputs/grasp_sweeps
```

The sweep automatically logs both variants in separate `ah_8_latch_off` and
`ah_8_latch_on` directories. `summary.csv`, `summary.md`, and `summary.json`
compare success, grasp rate, mean time to observed grasp (grasped episodes only),
and mean dither score (all episodes). JSON also includes paired on-only/off-only
success counts and metric differences. Incomplete schedules cannot qualify for
comparison/resume. Add `--dry_run` to inspect commands without GPU work.

## Iterate deliberately

Evaluate plain and DR with fixed checkpoint/start lists and seeds. Inspect low
success zones and colours; review trajectories/videos to separate miscalibration,
missed grasps, failed transport and failed release. Record targeted demonstrations
for the failing starts, preserve scene versions, exclude known failed saved demos
from the training copy, then retrain and compare the same evaluation protocol.

## Benchmark protocol

Use the committed `eval_sets/pick_place_v1_seed1984.json` for checkpoint
comparisons. It contains **100 held-out trials** (60 `id`, 20 `ood`, 20 `yaw`)
and **50 recorded `train` sanity trials**. With all splits selected, one repeat
is 150 episodes; `--splits id,ood,yaw` is 100. The yaw trials reuse the first
20 ID positions, so there are 80 distinct held-out XY positions. `--episodes`
is ignored when `--eval_set` is supplied: every selected start runs exactly once
per repeat, in file order. Do not regenerate the set separately for each model.

The fixed set's zone counts (zero means no sampled starts in that zone):

| Split | NR | NC | NL | MR | MC | ML | FR | FC | FL | OUT | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 0 | 7 | 4 | 6 | 17 | 11 | 6 | 6 | 3 | 0 | 60 |
| ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 |
| yaw | 0 | 3 | 1 | 3 | 5 | 1 | 4 | 1 | 2 | 0 | 20 |
| train | 0 | 0 | 7 | 6 | 5 | 6 | 10 | 11 | 5 | 0 | 50 |

All positions and yaws are in the robot base-link frame, in metres and radians.
ID positions are uniform rejection samples inside the **convex hull** of the
recorded starts, at least 2 cm from every training start. OOD positions are
2–5 cm outside that hull and within the recorded cube-start polar/pan sweep
expanded by 10 degrees. This is the protocol's angular reachability gate, not
an inverse-kinematics guarantee. New cube footprints must also fit on the
measured tabletop, so OOD sampling cannot create unsupported off-table starts.
First-frame arm poses are rest poses; they
must not be used to infer the workspace's pan sweep. All starts use the mean
recorded first-frame `observation.state` with `--robot_start recorded` (the
default), including the sanity split. `--robot_start default` instead uses the
existing calibrated reset pose and is a different comparison condition.

`--yaw_mode matched` (the default) keeps the recorded relative-yaw distribution
for `pick_place_v1`. For each training start, `bearing = atan2(y, x)` and
`rel_yaw = wrap(cube_yaw - bearing)` in [-45°,45°). A new ID/OOD start copies
`rel_yaw` from its nearest recorded XY neighbour and sets `yaw = bearing +
rel_yaw`, without jitter. `--yaw_mode radial` sets `yaw = bearing`, intended for
future datasets recorded with the yaw-aligned protocol. `--yaw_mode random`
draws uniform yaw in [0°,90°). The `yaw` split **always** uses uniform random yaw,
regardless of this option, at its paired ID positions. The set stores the yaw
mode, recorded relative-yaw mean/population-standard-deviation/min/max in radians,
and each start's relative yaw and `source_start_id` (`null` for radial/random
trials, self for recorded sanity trials). Absolute yaw is stored modulo 90°;
recorded sanity trials also retain their original full orientation.
New trials stand upright on the measured tabletop using the
existing reset helpers. Cube/box overlap is rejected by the same projected
cuboid footprint test as recorded-start evaluation. Paired ID/yaw candidates
are accepted together only when both clear the box.

Zones use the existing 3×3 training-region convention; OOD is `OUT`. Every trial
has a stable ID, split, XY, yaw, zone and six-element arm state. The JSON embeds
the captured live box/base poses, measured table outline and plane, existing
cube/box dimensions, input parameters, dataset path, creation time, generator
commit and a content hash. At runtime the benchmark checks live geometry against
the snapshot. A different scene requires a new set; do not silently reuse it.

Regenerate on the **host**, repository root (NumPy and PyArrow required):

```bash
PYTHONPATH=source python3 -m sim_to_real_so101.scripts.make_eval_set \
  --dataset datasets/pick_place_v1 \
  --out /tmp/pick_place_v1_seed1984.json --seed 1984 \
  --n_id 60 --n_ood 20 --n_yaw 20 --min_dist_m 0.02 \
  --ood_margin_m 0.02 0.05 --yaw_mode matched
```

After reinstalling the package, `make_pick_place_eval_set` is the equivalent
console command. It uses `eval_sets/pick_place_scene.json` by default;
`--scene_json` selects another live capture. To intentionally capture a changed
scene, with a free GPU and the teleop container running:

```bash
docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/Sim-to-Real-SO-101-Workshop/scripts/capture_eval_scene.py \
  --headless --out /tmp/pick_place_scene.json
```

That helper resets the scene once and reads the geometry; it runs no policy.
Copy the capture out of teleop before passing it to a host-side generator.

Run **one checkpoint**, on the host from the repository root:

```bash
cat > /tmp/pick_place_benchmark.yaml <<'YAML'
task: pick_place
policy:
  name: groot_n17
  checkpoint: so101_pick_place_v1_n17_10k/checkpoint-50000
  models_dir: ~/sim2real/models
  options:
    embodiment_tag: NEW_EMBODIMENT
eval:
  dataset: /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1
  eval_set: /workspace/Sim-to-Real-SO-101-Workshop/eval_sets/pick_place_v1_seed1984.json
  splits: [id, ood, yaw, train]
  repeats: 1
  robot_start: recorded
  episode_length_s: 30
  action_horizon: 8
  render_warmup: 2
YAML
./docker/eval.sh /tmp/pick_place_benchmark.yaml
```

Add `--dry_run` to inspect commands without starting jobs. Set `task: pick_place_dr`
for DR and use the same task and language settings for both checkpoints. The
launcher uses the generic policy-server image with `policies/groot_n17/` mounted;
its reset endpoint accepts the benchmark's policy seed. Direct clients must
connect to that same policy server. Each (start ID, repeat) receives a
deterministic policy seed derived from the run seed and original file index,
so split selection does not change its seed. The client requires server
acknowledgement; setting only a local seed is insufficient for remote inference.
Seeding improves repeatability but does not guarantee bitwise GPU determinism.

### Benchmark results and stages

Benchmark reports use **schema version 2**; ordinary recorded/random evaluation
retains schema version 1. Version 2 includes the fixed set and its hash,
`by_split`, `by_zone`, `overall` and `per_start_consistency`. Each aggregate gives
successes/episodes, success rate, a **95% Wilson interval**, stage rates,
failure-mode counts and median simulated time to success. The median excludes
failures and includes the existing ten settling steps. Per-start consistency is
the fraction of its completed repeats that succeeded. Reports record completion
and requested episode count; comparisons reject incomplete runs.

Stages are latched over the episode, sampled on every control step **before
automatic reset**, including the terminal frame:

- `reached`: the grasp point was within 3 cm of the cube centre. The grasp point is where a held cube's centre sits: the
  fixed-jaw tip offset measured from the gripper USD (`gripper_tip_local_offset`, zero clearance), applied to the live
  `ee_frame`. The `ee_frame` origin itself is the gripper link origin, about 9.4 cm behind that point. Reports made before
  2026-10-03 measured from the origin, so their `reached` stage is always false and every failure is `never_reached`.
- `grasped`: the existing `object_grasped` observation fired.
- `lifted`: cube rise from the existing resting-height tracker reached `min_lift`.
- `over_box`: cube centre XY entered the live outer box footprint while held.
- `placed`: the existing placement observation fired.
- `success`: existing confirmed termination, with timeout still taking precedence.

Failure classification uses `placed_not_confirmed` if placement occurred without
success; otherwise `never_reached`, `missed_grasp`, `timeout_holding` if still
held, `dropped_before_box` if released before ever crossing the box, or
`dropped_outside` if released after crossing without placement. These labels
summarize observed stages, not a new success predicate. The observer does not
change task geometry, contacts, thresholds or confirmation logic.

A PNG beside each JSON shows the measured table and box outlines, grey training
starts, and split-specific markers coloured from red (failure) to green
(success). With repeats, colour represents each start's success fraction.
ID/yaw markers share positions intentionally.

### Compare checkpoints

Run the same evaluation command for the second checkpoint, changing only
`--model` to `so101_pick_place_v1_n17_10k/checkpoint-49000`. Then use the two result paths printed by the runner (on the host,
under `datasets/eval_results/`):

```bash
python3 scripts/compare_eval_results.py \
  datasets/eval_results/checkpoint_A_results.json \
  datasets/eval_results/checkpoint_B_results.json
```

Substitute the actual timestamped filenames. The table reports each split's
success rate and Wilson interval. The **exact two-sided paired McNemar test**
compares starts shared by each pair, overall and per split. With repeats, it
uses repeat 0 by default; `--repeat 1` selects the second repeat. This avoids
treating repeated trials of the same start as independent McNemar observations.
Policy seeds and task, timeout, action horizon, robot-start mode and environment
seed must match. P-values are unadjusted when comparing multiple pairs/splits.

Wilson intervals describe the tested episode sample. Repeats of the same start,
and the paired ID/yaw positions, are correlated; their pooled episode-level
interval is descriptive, not evidence of equally many independent workspace
positions. Report per-split rates, denominators and consistency alongside it.
No long benchmark run is performed by the implementation's smoke checks.

## Action-horizon sweep

The sweep wrapper currently emits `--set` overrides that the YAML-only host
runner does not accept. Its launch examples below describe the earlier sweep
workflow and require updating before use. Individual checkpoint evaluations
can use the YAML profile shown above; the comparison and summary utilities
can still inspect existing result files.

`action_horizon` is the number of actions the client executes from a returned
chunk before querying again. GR00T's SO-arm processor returns 16 actions; a
horizon of 8 executes its first eight and replans. The horizon is client-side,
so the sweep loads the checkpoint **once** and reuses the acknowledged benchmark
server across all horizons. Each horizon starts a fresh simulator evaluation
through `eval.sh`; success logic and benchmark geometry are unchanged.

From the host repository root:

```bash
./docker/sweep_action_horizon.sh \
  --model so101_pick_place_v1_n17_10k/checkpoint-50000 \
  --models_dir ~/sim2real/models \
  --eval_set eval_sets/pick_place_v1_seed1984.json \
  --splits all --horizons 1,2,4,6,8,12,16 \
  --repeats 1 --episode_length_s 20
```

Add `--dry_run` to print the commands without launching Docker jobs. `--horizons
1-16` accepts an inclusive range; commas and ranges can be combined. Values must
be positive and fit the checkpoint's selected embodiment action chunk. The
processor's `action.delta_indices` takes precedence over the model config's
padded capacity (40 in N1.7); without a configured length the fallback is 16.
The current SO101 evaluator also requires horizons at most 16.

`--model` accepts the same relative checkpoint path as the evaluation runner.
Alternatively use `--model so101_pick_place_v1_n17_10k --checkpoint
checkpoint-50000`. Defaults are the committed benchmark, all splits, horizons
`1,2,4,6,8,12,16`, one repeat, 20 simulation seconds, and
`--out_root datasets/hyperparameters`. `--dataset` optionally overrides the
benchmark's dataset path. `--dr`, `--lang`, `--lang_by_color`, and `--gui` pass
through to the existing runner; `--port` defaults to 5555. Existing
`SERVER_IMAGE`, `EMBODIMENT_TAG`, and `RENAME_MAP` environment defaults are retained.
Model files live on the host. Eval-set, dataset and output paths must be mounted
in teleop; both host paths and repository paths inside teleop are accepted.
A real sweep refuses to start while another compute process occupies the GPU.

At about **25 minutes per horizon for 150 episodes**, seven horizons take roughly
**2 hours 55 minutes**, plus startup overhead. Shorter horizons make more
inference calls and can take longer. Each horizon has a tqdm episode progress bar showing its position in the sweep,
completed episodes, wall elapsed time, episode rate and ETA estimated from completed
trials. The estimate becomes useful once episodes finish.

The output layout is:

```text
datasets/hyperparameters/<model>_<checkpoint>_<eval_set_name>_<UTC timestamp>/
  sweep_config.json       # flags, checkpoint, git revision, benchmark/hash, host/GPU
  server_cmd.txt
  server.log
  ah_8/
    cmd.txt              # exact shell-quoted runner command
    run.log              # evaluation stdout and stderr
    status.json
    results.json
    results.png          # existing benchmark heatmap
  ah_16/ ...
  summary.csv
  summary.json
  summary.md
  plots/
    success_vs_horizon.png
    stages_vs_horizon.png
    time_to_success_vs_horizon.png
    failure_modes_vs_horizon.png
```

The direct evaluator can save videos with `--save_videos` under the corresponding
results directory. JSON rows add measured
`inference_calls` and `wall_time_s`; older reports without this telemetry show
missing values, not zeros.

Use `--resume` with the same options to reuse the latest matching folder. The
stored flags and benchmark hash must agree. A horizon is skipped only when its
report is complete and contains the exact unique selected `(start_id, repeat)`
schedule, the requested count, horizon and run settings. Incomplete artifacts
are retained as `results.json.previous` and `results.png.previous` before retry.
A failed horizon receives a failed status and remains visible in the summary;
subsequent horizons still run. Exit and Ctrl+C cleanup stop the owned server.

Open `summary.md` for successes/episodes and 95% Wilson intervals by split,
stage rates and failure counts. `summary.csv`/`summary.json` also contain mean
and median simulated time to success, the fraction successful within 15 seconds,
mean inference calls and wall time per episode. For repeats, per-start
consistency is stored as each start's success fraction. The best **complete**
horizon has the highest overall success rate; ties prefer `id_near` when present
(or `id`), then shorter median success time. Final ties use the smaller horizon
for a deterministic verdict. Failed or incomplete horizons cannot win.

The summary reuses the comparison helper's exact paired McNemar test on shared
starts in repeat 0, comparing the best horizon to the runner-up and horizon 16.
A statistical-better verdict requires p < 0.05 and more first-only than
second-only successes. Missing horizon 16 is reported as unavailable. A
self-comparison against 16 has p=1. These unadjusted tests and descriptive Wilson
intervals do not account for all correlations or multiple comparisons; two
smoke-test trials are useful for checking execution, not choosing a model.

Regenerate a summary without running the simulator:

```bash
python3 scripts/summarize_sweep.py datasets/hyperparameters/<sweep-folder>
# After installing the package:
summarize_eval_sweep datasets/hyperparameters/<sweep-folder>
```
