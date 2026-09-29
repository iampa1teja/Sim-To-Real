# 06 — Evaluate in simulation, with DR, and on the robot

[Guide hub](../README.md) · Previous: [Training](05-training.md) · [Troubleshooting](troubleshooting.md)

## Data flow and counting

```mermaid
sequenceDiagram
  participant S as Isaac Lab / lerobot_eval
  participant P as GR00T policy server
  S->>P: front/wrist images + joint state + instruction
  P-->>S: action chunk
  S->>S: Execute action_horizon actions
  S->>P: Updated observation
  S->>S: Count success or timeout; auto-reset once
```

`lerobot_eval` uses `GR00TRemotePolicy` and the SO101 interface to convert between
normalized LeRobot joints and simulation radians, including `sim_joint_mapping`.
Its `rename_map` translates sim camera names to checkpoint keys. For the prepared
MyRoom dataset use `{"realsense_rgb":"front","wrist_cam":"wrist"}`. Both RGB
views and language must match training. `--action_horizon` defaults to 16 executed
actions before another query; it does not change the checkpoint's training horizon.

The first ten steps of each rollout command a hard-coded `initial_action` in
[lerobot_eval.py](../source/sim_to_real_so101/scripts/lerobot_eval.py). This is a
setup-specific pose, not automatically taken from your demonstration. Check it
when adapting a scene. After that the policy provides actions.

For PickPlace, success is termination **without** timeout. Timeout counts as
failure even if success also fires on that step. Isaac's done step already resets
the environment; the client reuses the returned observation without another reset.
It snapshots start/zone/colour before stepping, so results describe the completed
episode. Manual R resets are excluded. Closing the simulator early saves an
incomplete report and raises an error. See [success logic](03-define-a-task.md).

## Host runner

Start the teleop container and build `real-robot` using [Setup](01-setup.md). Put
the checkpoint under your **host** models directory. `--dataset` is an absolute
path **inside teleop**, pointing at the original dataset with saved cube sidecars;
model paths are relative to the host models directory.

**Host**, repository root, inspect the launch without starting Docker jobs:

```bash
./docker/eval_pick_place.sh \
  --model so101_pick_place_v1/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --episodes 20 --dry_run
```

**Host**, repository root, run the plain evaluation:

```bash
./docker/eval_pick_place.sh \
  --model so101_pick_place_v1/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --episodes 20
```

Every flag from [eval_pick_place.sh](../docker/eval_pick_place.sh):

| Flag | Default / interpretation |
| --- | --- |
| `--model` | Required relative checkpoint path; no absolute, `.` or `..` path components |
| `--dataset` | Required absolute teleop dataset root; expects `pick_place_meta/` |
| `--models_dir` | `MODELS_DIR` or `$HOME/models` on host |
| `--episodes` | 20, positive integer |
| `--dr` | Off; choose `Lerobot-So101-Teleop-Pick-Place-DR-Eval` instead of plain Eval |
| `--random_fraction` | 0.0; fraction/probability of held-out random starts, in [0,1] |
| `--start_mode` | `cycle`; alternative `random` for recorded-start selection |
| `--lang` | `Pick up the blue cube and place it in the white box` |
| `--lang_by_color` | Empty; JSON with blue instruction, and red too with DR |
| `--port` | 5555; host TCP port, 1–65535 |
| `--gui` | Off; omit the client's headless flag when enabled |
| `--dry_run` | Off; print Docker commands/preflight rather than execute Docker |
| `--help`, `-h` | Usage |

Preflight checks Docker access without sudo, running `teleop`, the `real-robot`
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
| `episodes` | Rows with `episode`, `steps`, `success`, `start_index`, `start_kind`, `zone`, `cube_color`, `instruction` |
| `overall` | `episodes`, `successes`, `success_rate` |
| `by_start` | Aggregates for `recorded` and `random` (not individual ID groups) |
| `by_zone`, `by_cube_color` | Aggregates for observed zones/colours |

Rates are in [0,1], or null for an empty aggregate. `start_index=-1` means a random
start; recorded indices refer to the sorted input sidecar list. If you pass a
filtered copy, use its preparation mapping to recover source IDs.

## Recorded starts and DR

`--cube_starts` is a **client** flag; the host runner derives it from
`--dataset`. `PICK_PLACE_EVAL_STARTS` is the reset function's fallback when no
explicit path is supplied. Each sidecar must contain finite metre-valued frame-0
position and a nonzero wxyz quaternion in the robot base-link frame. The reader
uses only frame zero for starts, not later trajectory frames.

Cycle mode reshuffles all recorded starts using the seed and consumes each once
before reshuffling. Random mode samples recorded starts uniformly with replacement.
`random_fraction` chooses random XY within recorded bounds and yaw in [0,π/2),
standing upright on the live table plane. This is probabilistic, not an exact
fraction of a small run. Recorded Z is discarded: the live table support is solved
along base Z while preserving recorded XY/orientation. Overlap with the box fails
explicitly for recorded starts; random candidates are rejected (limit 10,000 tries).

Zones divide recorded XY bounds into thirds. N/M/F means near/middle/far from the
base; L/C/R is left/centre/right when looking away from the base. The away axis is
derived from the bounding-box centre, not assumed to be world X. Degenerate axes
use the middle third. Compare counts as well as rates; one success in one trial
is weak evidence of coverage.

**Host**, repository root, DR with matching colour instructions:

```bash
./docker/eval_pick_place.sh \
  --model so101_pick_place_v1/checkpoint-10000 \
  --models_dir ~/sim2real/models \
  --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --episodes 20 --dr --random_fraction 0.25 --start_mode cycle \
  --lang_by_color '{"blue":"Pick up the blue cube and place it in the white box","red":"Pick up the red cube and place it in the white box"}'
```

DR reuses the workshop camera focal/pose ranges from `TaskEventCfg` on external
RGB, aligned depth and wrist, then synchronizes calibrated lens intrinsics. It
randomizes robot colour, blue/red cube colour and the measured room tube-light
exposure when that light exists. MyRoom has no LightStudio/sky-light HDRI/mat;
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
export PICK_PLACE_EVAL_STARTS=/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta
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
  --cube_starts /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta \
  --rename_map '{"realsense_rgb":"front","wrist_cam":"wrist"}' \
  --policy_host localhost --policy_port 5555 --action_horizon 16 \
  --lang_instruction 'Pick up the blue cube and place it in the white box' \
  --num_episodes 20 --seed 1984 \
  --checkpoint so101_pick_place_v1/checkpoint-10000 \
  --results_json /workspace/Sim-to-Real-SO-101-Workshop/datasets/eval_results/manual.json \
  --headless
```

The direct parser additionally accepts `--disable_fabric`, `--rerun`,
`--start_mode` (cycle), `--random_fraction` (0.0), `--lang_instruction_by_color`
(JSON), and the inherited [Isaac launcher flags](04-recording.md#inherited-isaac-launcher-options).
Its defaults differ from the host runner: MyRoom task, 10 episodes, seed 1984,
no rename map/results path/cube-start directory, `num_envs=None`, and language
`Pick up the vial and place it in the rack`. Set an Eval task explicitly: default
MyRoom has no success/timeout terminations. Checkpoint metadata falls back to
`PICK_PLACE_EVAL_CHECKPOINT`, then `MODEL`, then `unknown`.

## Real-robot rollout

Use [Setup](01-setup.md) to start the real-robot container and calibrate hardware.
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
