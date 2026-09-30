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
