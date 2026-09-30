# 04 — Record demonstrations

[Guide hub](../README.md) · Previous: [Task](03-define-a-task.md) · Next: [Training](05-training.md)

## Choose the recorder

`pick_place_agent` owns cube spawning, countdowns, the four-pane browser GUI,
trajectory sidecars and background encoding. It defaults to
`Lerobot-So101-Teleop-Pick-Place`. Use it for this guide's pick-place training and
recorded-start evaluation. `lerobot_agent` is generic teleoperation/recording,
defaults to `Lerobot-So101-Teleop-MyRoom`, uses terminal or Isaac keyboard controls,
and exposes a separate spawn service. It does not provide the pick-place GUI or
write cube-trajectory sidecars.

**Sim-only means no physical follower, not no hardware:** both recorders still
need the leader. With `--enable_real_follower`, the same raw leader actions drive
the physical follower and paired datasets use synchronized episode/frame counts.
Only one environment and a teleop task without automatic terminations are allowed.
Supply `--repo_id`, `--repo_root` and `--task_name` together; without all three,
teleoperation can run without dataset recording.

**Teleop container**, sim-only recording after [Setup](01-setup.md):

```bash
pick_place_agent --task Lerobot-So101-Teleop-Pick-Place \
  --repo_id '<hf_user>/pick_place_v1' \
  --repo_root /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 \
  --task_name 'Pick up the blue cube and place it in the white box'
```

**Teleop container**, paired recording instead (do not run simultaneously with
the first command; use a new root when changing the configuration):

```bash
pick_place_agent --enable_real_follower \
  --real_gripper_camera "$CAMERA_GRIPPER" --real_external_camera "$CAMERA_EXTERNAL" \
  --repo_id '<hf_user>/pick_place_paired_v1' \
  --repo_root /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_paired_v1 \
  --task_name 'Pick up the blue cube and place it in the white box'
```

Default real repo/root names append `-real` to the sim name; optional real
arguments override them. Physical camera keys are `gripper` and `external`,
whereas simulation keys are `wrist_cam` and `realsense_rgb`. They are not silently
interchangeable in GR00T preparation. Real camera feeds are opened through the
follower; supplying camera arguments without enabling it does not open real panes.

## All recorder flags

Defaults below are from the two parsers. Environment values in [Setup](01-setup.md)
override the listed fallback. Both parsers also accept standard `--help`.

| Shared flag | Default | Meaning |
| --- | --- | --- |
| `--task` | PickPlace for `pick_place_agent`, MyRoom for `lerobot_agent` | Exact Gym ID |
| `--num_envs` | `None` (config supplies 1) | Must resolve to 1 for manual recording |
| `--disable_fabric` | false | USD I/O mode; keep off for this camera workflow |
| `--port`, `--robot_id` | `TELEOP_PORT` / `/dev/ttyACM0`; `TELEOP_ID` / `leader_arm_1` | Physical leader connection/calibration |
| `--enable_real_follower` | false | Connect and command physical follower |
| `--follower_port`, `--follower_id` | `ROBOT_PORT` / `/dev/ttyACM1`; `ROBOT_ID` / `follower_arm_1` | Follower connection/calibration |
| `--repo_id`, `--repo_root`, `--task_name` | `None` | Dataset identity, local root and instruction; required together |
| `--real_repo_id`, `--real_repo_root` | `None` | Override derived `-real` identity/root; base dataset triple still required |
| `--save_mp4` | false | Extra auxiliary videos; normal dataset RGB videos are written regardless |
| `--depth`, `--instance_id_seg` | false | Include respective auxiliary streams when `--save_mp4` is set; not extra GR00T modalities |
| `--image_writer_threads_per_camera` | 4 | Async PNG writer threads; nonnegative |
| `--real_gripper_camera`, `--real_external_camera` | corresponding `CAMERA_*` or unset | OpenCV index or path |
| `--real_camera_width`, `--real_camera_height` | 640, 480 | Real capture resolution |
| `--real_camera_fourcc` | `CAMERA_FOURCC` or `None` | Optional capture codec |
| `--fps` | `CONTROL_FPS` or 30 | Master control/recording rate; positive and must divide physics rate |
| `--seed` | 101 | Simulation seed |

| `pick_place_agent` only | Default | Meaning |
| --- | --- | --- |
| `--control_input` | `lerobot` | `lerobot`, `terminal`, or `isaac`; unavailable LeRobot listener falls back to terminal |
| `--encode_every` | 0 | Background encoding every N saved episodes; 0 means request/exit; recovered episodes encode automatically |
| `--rerun` | false | Optional Rerun camera/action visualization |
| `--reset_wait_s` | 10 | Preparation countdown; it does not reset the scene |
| `--gui_host` | `127.0.0.1` | HTTP bind address |
| `--gui_port` | 8765 | Browser recorder port |
| `--gui_fps` | 15.0 | GUI refresh limit; not dataset FPS |

| `lerobot_agent` only | Default | Meaning |
| --- | --- | --- |
| `--control_input` | `terminal` | `terminal` or `keyboard` |
| `--spawn_port` | 6001 | Local spawn listener; 0 disables it |
| `--spawn_authkey` | built-in `so101-spawn` | Shared spawn-service header value, not secure network authentication |
| `--sim_reset_object_name` | `None` | Existing scene entity to place near gripper on reset |
| `--sim_reset_object_offset` | `0 0 0` | Three world-frame metre offsets |

### Inherited Isaac launcher options

The recorders, zero/random agents and evaluator call Isaac Lab 2.3.2's
`AppLauncher.add_app_launcher_args`. Relevant options are `--headless` (false),
`--device` (`cuda:0`), `--enable_cameras` (false in launcher, forced on by these
scripts), `--livestream` (default −1 uses environment, explicit choices 0/1/2),
`--xr` (false), `--verbose`, `--info`, `--experience` (empty = automatic),
`--rendering_mode` (`performance`, `balanced`, `quality`; balanced launcher fallback), `--kit_args` (empty),
`--anim_recording_enabled` (false), `--anim_recording_start_time` (0) and
`--anim_recording_stop_time` (10). Animation options concern time-sampled USD,
not the LeRobot dataset. `--cpu` is a deprecated launcher option that raises an
error; do not use it as a CPU-rendering recipe. Environment/task render settings
can further configure the selected mode. `list_envs` does not expose this parser.

## Browser GUI and controls

The page at `http://localhost:8765/` has four panes: **Sim: realsense**, **Sim:
wrist**, **Real: external**, **Real: gripper**. Real panes show no signal without
configured follower cameras. The top toolbar provides Start recording, Stop
recording, Discard recording and Spawn the cube. Stop saves; it does not exit.
The bottom bar has Save, Encode videos and Exit. Exit offers Encode & Exit,
Exit without encoding and Cancel when videos are pending. The status line shows
countdown, current episode/duration, save/discard messages and encoding progress.

The web server has no authentication and can trigger real motion through the
active recorder. Keep its default localhost bind; host networking makes it
available on the host's loopback. The UI currently downloads React/Babel from a
CDN, so an offline browser may show an empty page even when the server is alive.
GUI images are downsampled previews; this does not change dataset resolution.

| Control mode | Actions |
| --- | --- |
| PickPlace LeRobot keyboard | S spawn; Right start/save; Left discard; R reset; Escape quit |
| PickPlace Isaac keyboard | Same principal keys; focus Isaac viewport, not browser |
| PickPlace terminal | `s`/`spawn` spawn; `start` start; `right` toggle; `save` save; `left`/`c` discard; `r` reset; `q` quit; Enter submits |
| Generic terminal | `s` toggles start/save; `save`, `c`, `r`, `q`; Enter submits |
| Generic Isaac keyboard | S start/save; Right save; Left/C discard; R reset; Escape quit |

**Teleop container**, generic recorder example without dataset writing:

```bash
lerobot_agent --task Lerobot-So101-Teleop-MyRoom --control_input terminal
```

**Teleop container**, separate shell while that generic agent runs:

```bash
python -m sim_to_real_so101.scripts.spawn
```

`spawn.py` is a module, not a console entry point. Its `--host`, `--port`,
`--authkey` default to `localhost`, 6001 and the agent's built-in key. This service
creates a practice cube; it does not implement PickPlace sidecars or success.

## One episode

1. Align sim and real views. Return the leader/follower to a consistent starting
   pose. Spawn near the gripper at a measured, reachable position.
2. Press Start. The countdown preserves the current scene; only reset controls
   reset it. Starting without a visible cube is refused.
3. At recording start, perform a steady grasp, lift, move and release. Hold long
   enough after release to observe a stable placement. This is demonstration
   guidance, not an automatic stopping rule.
4. Save/Stop commits the episode and sidecar. Discard clears the active episode;
   it does not delete an already saved one. During countdown, finishing controls
   cancel preparation without recording frames.
5. The cube hides between episodes. Spawn again for the next demonstration.
   Encode videos when convenient, or encode on exit. Do not train on pending PNGs.

### Option B: yaw-aligned demos

For `pick_place_v2`, keep wrist roll neutral **before pressing Spawn** and inspect
that the cube's faces line up with the arm's approach direction. Spawn copies the
live gripper orientation; it has no independent yaw control. Let the cube settle
and check the alignment in the external view. Vary cube **position** across the
reachable 3×3 grid, keeping this approach-relative alignment consistent; do not
introduce independent cube-yaw or wrist-roll variation.

Approach, grasp, lift, place, and hold for **0.5–1 s after release** before saving.
Use a **new** root, `datasets/pick_place_v2`, and never mix these demos with v1.
Discard a visibly misaligned or failed attempt during collection. Encode every
saved episode before preparation.

**Teleop container**, sim-only v2 collection:

```bash
pick_place_agent --task Lerobot-So101-Teleop-Pick-Place \
  --repo_id 'iampa1teja/pick_place_v2' \
  --repo_root /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v2 \
  --task_name 'Pick up the blue cube and place it in the white box'
```

**Host**, workshop root with NumPy and PyArrow, after recording:

```bash
python scripts/check_pick_place_demo_alignment.py --dataset datasets/pick_place_v2
```

This read-only review pairs sidecar frames with recorded `observation.state` and
prints cube-face yaw, calibrated shoulder pan and wrist roll at the first base-Z
rise over 1 cm. It accounts for wxyz quaternions, cubes resting on different
faces, 90° face symmetry, and the recorder's normalized joint mapping. Review
large changes in face-minus-pan or wrist roll, and episodes without a lift
candidate, alongside the videos. Pan and roll are joint angles, so this comparison
does not reconstruct the gripper's yaw or certify grasp alignment or success.

Then run [prepare + verify](05-training.md#prepare-an-independent-copy) with
`--model_profile so_arm_n17`, from `datasets/pick_place_v2` into the new
`datasets/pick_place_v2_gr00t` root. Evaluate v2 on its recorded cube and arm
starts with `--random_fraction 0`.

## Storage and recovery

The primary recording is LeRobot v3: `meta/info.json`, `meta/tasks.parquet`,
`meta/episodes/chunk-*/file-*.parquet`, `data/chunk-*/file-*.parquet`, and video
paths given by `info.json`. Pending frames live under the dataset image layout.
The workshop encoder writes one episode per video, closes parquet metadata at
save, and publishes encoded video metadata after successful encoding. Additional
MP4 outputs are separate from the training dataset's required RGB videos.

PickPlace writes `pick_place_meta/episode_000000.json` and so on under the **sim**
root. Each payload has `episode_index`, `reference_frame="robot base link (base)"`,
`units="m"`, `fps`, `num_frames`, `cube_size_m`, `position` and
`orientation_wxyz`. Position and quaternion lists have one entry per saved frame,
with scalar-first quaternion order. They do not contain a box pose or success
label. The paired `-real` dataset does not gain physical cube trajectories.

PickPlace defaults to deferred/background encoding. Relaunching the same dataset
recovers a contiguous suffix of saved episodes with missing video metadata,
requires every expected PNG, cleans abandoned encoding staging, and starts the
background encoder. It does not recover an unsaved demonstration as a committed
episode. Encoding failures preserve recoverable frames and expose a retry message;
press Encode videos after resolving disk/encoder errors. Generic `lerobot_agent`
uses immediate episode encoding rather than the PickPlace GUI scheduler.

Resume is automatic, not a recorder flag. FPS, feature names/shapes, episode/frame
counts, sim/real counts and sidecar timelines must agree. Refusal is intentional:
keep the source intact and diagnose it before retrying. A newly created dataset
with `meta/info.json` but **zero saved episodes** takes the existing-dataset load
path; the underlying loader may fail without parquet/episode files. Use a new
root after preserving the empty attempt; do not assume empty-dataset relaunch is
supported. See [Troubleshooting](troubleshooting.md).

## Plan coverage and review quality

The NVIDIA course's [dataset inventory](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/datasets-and-models.html)
uses 75 sim demonstrations for its original vial task. For your cube task, 50
successful, varied demos are a reasonable initial experiment, not a success
threshold. For option B, aim for a 3×3 reachable position grid plus edges, keep
wrist roll neutral when spawning and preserve cube-face alignment with the arm's
approach. Keep the arm start pose consistent and avoid long idle sections or
abrupt motion.
The recorder spawns beside the gripper; it has no grid/yaw CLI. Position starts
through the actual workflow and inspect achieved coverage instead of claiming
uniform coverage from a requested count.

**Training machine or teleop container**, repository root with NumPy installed:
this read-only check uses the same zone definition as evaluation and reports
sidecar episode lengths. Replace the dataset path if needed.

```bash
python - <<'PYCODE'
from collections import Counter
import importlib.util
import json
from pathlib import Path
root = Path('datasets/pick_place_v2')
path = Path('source/sim_to_real_so101/utils/pick_place_eval.py')
spec = importlib.util.spec_from_file_location('starts', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
starts = module.RecordedStarts(root / 'pick_place_meta')
print('base XY bounds:', starts.low.tolist(), starts.high.tolist())
print('zone counts:', dict(Counter(starts.zone(p[:2]) for p, q in starts.starts)))
for path in sorted((root / 'pick_place_meta').glob('episode_*.json')):
    row = json.loads(path.read_text())
    assert len(row['position']) == len(row['orientation_wxyz']) == row['num_frames']
    print(path.name, row['num_frames'], 'frames,', round(row['num_frames'] / row['fps'], 2), 'seconds')
PYCODE
```

### Sidecar success review — not automatic success certification

The stored fields cannot establish earlier jaw contact, release, the live box
transform or the termination confirmation counter. There is no truthful command
that reproduces the full simulator success condition from these files alone.
Use this read-only trajectory review alongside the videos and your episode
notes; do not turn its output into a claimed success rate.

**Training machine or teleop container**, repository root:

```bash
python - <<'PYCODE'
import json
from pathlib import Path
for path in sorted(Path('datasets/pick_place_v1/pick_place_meta').glob('episode_*.json')):
    row = json.loads(path.read_text())
    poses = row['position']
    print(path.name, 'start=', poses[0], 'final=', poses[-1],
          'base-Z excursion=', round(max(p[2] for p in poses) - min(p[2] for p in poses), 6),
          'success=UNDETERMINED (review video/contact history)')
PYCODE
```

Keep a list of failed saved **source episode indices**. Omit them from the training
copy with `--exclude_episodes` in [Training](05-training.md), leaving the original
recordings byte-identical. Use Discard during collection when failure is already
clear. Keep evaluation starts and the training exclusion mapping explicit so
renumbered training IDs are not mistaken for original recorded IDs.
