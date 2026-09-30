# 06 — Evaluate in simulation, with DR, and on the robot

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

`lerobot_eval` uses `GR00TRemotePolicy` and the SO101 interface to convert between
normalized LeRobot joints and simulation radians, including `sim_joint_mapping`.
Its `rename_map` translates sim camera names to checkpoint keys. For the prepared
v2 N1.7 starter profile use `{"realsense_rgb":"room","wrist_cam":"wrist"}`.
Both RGB views and language must match training. Legacy N1.6 front/wrist
checkpoints need their original rename map. `--action_horizon` defaults to 16
executed actions before another query and accepts integers **1–16**. Setting it
to 8 executes the first eight actions and replans; it does not change the
checkpoint's 16-step SO-arm training horizon or its padded model capacity.

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

**Host**, repository root, inspect the launch without starting Docker jobs:

```bash
./docker/eval_pick_place.sh \
  --model so101_pick_place_v2_nvinit/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2 \
  --episodes 20 --episode_length_s 30 --action_horizon 8 --dry_run
```

**Host**, repository root, run the plain evaluation:

```bash
./docker/eval_pick_place.sh \
  --model so101_pick_place_v2_nvinit/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2 \
  --episodes 20 --episode_length_s 30 --action_horizon 8
```

Run the same command with `checkpoint-9000` to compare both saved checkpoints.
These commands use recorded cube starts and the corresponding recorded arm
poses. Keep the dataset, seed and other options identical for the comparison.

Every flag from [eval_pick_place.sh](../docker/eval_pick_place.sh):

| Flag | Default / interpretation |
| --- | --- |
| `--model` | Required relative checkpoint path; no absolute, `.` or `..` path components |
| `--dataset` | Required absolute teleop dataset root; expects `pick_place_meta/` |
| `--models_dir` | `MODELS_DIR` or `$HOME/models` on host |
| `--episodes` | 20, positive integer |
| `--dr` | Off; choose `Lerobot-So101-Teleop-Pick-Place-DR-Eval` instead of plain Eval |
| `--random_fraction` | 0.0; recorded cube starts only; `0.25` explicitly opts into random XY/yaws |
| `--start_mode` | `cycle`; alternative `random` for recorded-start selection |
| `--episode_length_s` | Task default (15 s); override with finite positive seconds |
| `--action_horizon` | 16; integer in 1–16, executed actions before replanning |
| `--embodiment_tag` | `EMBODIMENT_TAG` or `NEW_EMBODIMENT`; passed to the server as `--embodiment-tag` |
| `--server_image` | `SERVER_IMAGE` or `real-robot:n1.7` |
| `--rename_map` | `RENAME_MAP` or `{"realsense_rgb":"room","wrist_cam":"wrist"}` |
| `--lang` | `Pick up the blue cube and place it in the white box` |
| `--lang_by_color` | Empty; JSON with blue instruction, and red too with DR |
| `--port` | 5555; host TCP port, 1–65535 |
| `--gui` | Off; omit the client's headless flag when enabled |
| `--rerun` | Off; pass the client's Rerun visualization flag |
| `--dry_run` | Off; print Docker commands/preflight rather than execute Docker |
| `--help`, `-h` | Usage |

For a legacy N1.6 checkpoint, build with `./docker/real/build.sh ada n16` and add
`--server_image real-robot --rename_map '{"realsense_rgb":"front","wrist_cam":"wrist"}'`.
Keep the checkpoint's own action/modality configuration; changing camera names
does not convert a model between N1.6 and N1.7.

Preflight checks Docker access without sudo, running `teleop`, the chosen server
image, checkpoint containment/existence, `config.json` and weight files, sidecar
directory existence, valid arguments/colour JSON, and a free host port. Dry-run
still validates arguments and needs host Python 3, but does not perform those
Docker/filesystem readiness checks. The missing-teleop diagnostic extracts the
launch block from the hub README; that block is intentionally retained there.

The script starts a uniquely named detached, auto-removed policy server with the
existing GPU/network/device/display/model mounts, waits up to 300 seconds for TCP
readiness, runs `lerobot_eval` inside teleop with Isaac's Python wrapper, and
prints the JSON. It streams server logs to a temporary host file so failure logs
survive automatic container removal. Exit/interrupt traps stop/remove only this
run's server, stop log following and delete its temporary log. It does not stop
the user's teleop container. Save terminal output if you need persistent server
failure logs. A readiness connection is not a complete inference health test.

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
| `episode_length_s`, `action_horizon`, `step_dt` | Configured episode seconds, executed chunk length and simulation seconds per control step |
| `episodes` | Existing row fields plus `success_step`: 1-based control step where success fired, or null for timeout |
| `overall` | `episodes`, `successes`, `success_rate`, `successes_within_15s`, `success_rate_within_15s`, `median_time_to_success_s` |
| `by_start` | Aggregates for `recorded` and `random` (not individual ID groups) |
| `by_zone`, `by_cube_color` | Aggregates for observed zones/colours |

Rates are in [0,1], or null for an empty aggregate. `start_index=-1` means a random
start; recorded indices refer to the sorted input sidecar list. If you pass a
filtered copy, use its preparation mapping to recover source IDs.

The 15 s rate counts successes whose `success_step × step_dt` is at most 15 s,
using all completed episodes as the denominator. The ordinary success rate uses
the configured full episode window. Median time-to-success uses successful
episodes only and is null when there are none. Start/zone/colour aggregates carry
the same added metrics, and the printed JSON summary includes the run options.
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
./docker/eval_pick_place.sh \
  --model so101_pick_place_v2_nvinit/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2 \
  --episodes 20 --dr --random_fraction 0 --start_mode cycle \
  --lang_by_color '{"blue":"Pick up the blue cube and place it in the white box","red":"Pick up the red cube and place it in the white box"}'
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
renderer. The [validation history](../docker/eval_pick_place_validation.md)
records earlier CPU tests/dry runs and retains the GPU checklist; it is not proof
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

The direct parser additionally accepts `--disable_fabric`, `--rerun`,
`--start_mode` (cycle), `--random_fraction` (0.0), `--robot_start` (recorded), `--lang_instruction_by_color`
(JSON), and the inherited [Isaac launcher flags](04-recording.md#inherited-isaac-launcher-options).
Its defaults differ from the host runner: MyRoom task, 10 episodes, seed 1984,
no rename map/results path/cube-start directory, `num_envs=None`, and language
`Pick up the vial and place it in the rack`. Set an Eval task explicitly: default
MyRoom has no success/timeout terminations. Checkpoint metadata falls back to
`PICK_PLACE_EVAL_CHECKPOINT`, then `MODEL`, then `unknown`.

## Real-robot rollout

The commands below retain the legacy N1.6 real-robot workflow. Build its image
with `./docker/real/build.sh ada n16`, then use [Setup](01-setup.md) to start the
real-robot container and calibrate hardware. The new N1.7 image supplies policy
serving for the simulation runner; it does not include this physical LeRobot
client environment.
Validate the script's `SO101Control.initial_pose`/`home_pose` against your actual
setup before running; connection/disconnection can move the arm. Stop any other
process that owns its serial port or cameras. The real rollout does not inherit
Isaac's success monitor or simulation collision constraints.

**Real-robot container**, first shell, serve the checkpoint:

```bash
cd /Isaac-GR00T
export MODEL=so101_pick_place_v1/checkpoint-10000
python gr00t/eval/run_gr00t_server.py --model-path "/workspace/models/$MODEL" --port 5555
```

The pinned Tyro `ServerConfig` supports `--model-path` (default `None`),
`--embodiment-tag` (`NEW_EMBODIMENT`), `--device` (`cuda`), `--host` (`0.0.0.0`),
`--port` (5555), `--strict` (true), `--use-sim-policy-wrapper` (false),
`--dataset-path`, `--modality-config-path` and `--execution-horizon` (all `None`).
The latter dataset options support upstream replay use; they are not needed for
the trained model command. Do not run this extra server alongside the host runner
on the same port.

**Host**, second terminal, attach:

```bash
docker exec -it real-robot bash
```

**Real-robot container**, second shell, after checking environment and cameras:

```bash
cd /Isaac-GR00T
python gr00t/eval/real_robot/SO100/so101_eval.py \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID" \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $CAMERA_GRIPPER, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: $CAMERA_EXTERNAL, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost --policy_port=5555 --action_horizon=16 \
  --lang_instruction='Pick up the blue cube and place it in the white box'
```

The local Draccus `EvalConfig` has `robot=None`, `policy_host=0.0.0.0`,
`policy_port=5555`, `action_horizon=16`, language `Grab markers and place into pen
holder.`, and `play_sounds`, `rerun`, `passive_mode`, `plot` all false. Boolean
options use Draccus values, e.g. `--plot=true`. It also defines `--timeout` with
60, but **the loop never reads it**: rollout runs until interrupted, not for a
bounded episode count. The loop targets 30 Hz action timing, queries again after
each executed chunk, and has no automatic success scoring. With plotting enabled,
current code writes a PNG under `outputs/plots/` relative to the working directory;
its docstring promises JSON too, but the implementation does not write JSON.
Copy desired plots out before the ephemeral container is removed.

## Iterate deliberately

Evaluate plain and DR with fixed checkpoint/start lists and seeds. Inspect low
success zones and colours; review trajectories/videos to separate miscalibration,
missed grasps, failed transport and failed release. Record targeted demonstrations
for the failing starts, preserve scene versions, exclude known failed saved demos
from the training copy, then retrain and compare the same evaluation protocol.
Keep real outcomes as an explicit human-scored log until a real success sensor
or scorer is implemented.

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
./docker/eval_pick_place.sh \
  --model so101_pick_place_v1_n17_10k/checkpoint-50000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --eval_set /workspace/Sim-to-Real-SO-101-Workshop/eval_sets/pick_place_v1_seed1984.json \
  --splits id,ood,yaw,train --repeats 1 \
  --episode_length_s 30 --action_horizon 8
```

Add `--dry_run` to inspect commands without starting jobs. `--dr` retains the
existing DR behavior; use the same DR setting and language configuration for
both checkpoints. The wrapper uses a mounted `benchmark_server.py` entrypoint
for benchmark runs; existing images need no rebuild. It preserves the pinned
N1.7 offline launcher and adds an acknowledged server RNG seed endpoint. Direct
clients must connect to this entrypoint too. Each (start ID, repeat) receives a
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

- `reached`: `ee_frame` was within 3 cm of the cube.
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

`action_horizon` is the number of actions the client executes from a returned
chunk before querying again. GR00T's SO-arm processor returns 16 actions; a
horizon of 8 executes its first eight and replans. The horizon is client-side,
so the sweep loads the checkpoint **once** and reuses the acknowledged benchmark
server across all horizons. Each horizon starts a fresh simulator evaluation
through `eval_pick_place.sh`; success logic and benchmark geometry are unchanged.

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
inference calls and can take longer. Progress reports show the current horizon,
completed episodes, wall elapsed time and an ETA estimated from completed
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

The evaluator currently saves no video files. Any future video recording should
write under the corresponding horizon directory. JSON rows add measured
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
