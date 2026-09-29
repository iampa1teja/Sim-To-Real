# 01 — Containers and hardware

[Guide hub](../README.md) · Next: [Digital twin](02-digital-twin.md)

## Prerequisites

Use a Linux host with an NVIDIA GPU, a driver compatible with the selected CUDA
image, Docker, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and a working X11 display. The original workshop lists Ubuntu newer than 22.04,
RTX 6000 Ada, RTX 5090 and RTX 6000 Pro Blackwell as tested configurations; these
are not guarantees for every driver/VRAM combination. No minimum driver version
is pinned in this repository. Verify the image/vendor requirements for your GPU.

For sim teleop you need an SO-101 leader with USB serial access and its calibration.
For paired collection or real evaluation, add a powered SO-101 follower, a fixed
external RGB camera (the packaged scene models a RealSense), and a wrist RGB
camera. Match their resolution, mounting and orientation to your measured scene.
Mount the arms securely and keep their travel area clear.

**Host**, initial checks:

```bash
nvidia-smi
docker info
```

## Build and start

The [hub quick start](../README.md#quick-start-record-one-simulation-episode)
contains clone/build/run commands for `teleop-docker:latest` and the complete
mount list. The [simulation Dockerfile](../docker/sim/Dockerfile) pulls
`nvcr.io/nvidia/isaac-lab:2.3.2` as its base and pins LeRobot to `e670ac5daf9b76`.
There is no published fork image/tag in this repo to substitute for the build.

Create `docker/env` as a file before starting containers; it is ignored by Git.
The teleop entrypoint sources it, configures Isaac's libraries and installs the
extension editable. The mount of `source/` also exposes `source/gui/`. `datasets/`
and `outputs/` persist on the host. `scripts/` and the repository README are **not**
mounted by the existing launch recipe: run preparation from a full checkout on a
training machine, rather than assuming those paths exist inside teleop.

**Host**, repository root, teleop launch (same mounts as the hub):

```bash
xhost +
docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  -e "PRIVACY_CONSENT=Y" -e DISPLAY \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v "$HOME/.Xauthority:/root/.Xauthority" \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
  -v "$(pwd)/docker/env:/root/env" \
  -v "$(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source" \
  -v "$(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs" \
  -v "$(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets" \
  -v "$(pwd)/docker/real/scripts:/workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts" \
  teleop-docker:latest
```

**Host**, repository root, build the real-robot image for your architecture;
replace `ada` with `blackwell` for that build:

```bash
./docker/real/build.sh ada
```

The build script accepts only positional `ada` or `blackwell`. Both Dockerfiles
pin GR00T to `ead52833afbbf4243f8cd5e7664f48a94de03b19`. Ada uses CUDA 12.8 and
uv; Blackwell uses CUDA 13.0, nightly PyTorch and a source FlashAttention build.
Do not assume the two Python environments are interchangeable.

**Host**, repository root, start the real-robot container:

```bash
export MODELS_DIR=~/sim2real/models
mkdir -p "$MODELS_DIR"
docker run -it --rm --name real-robot --network host --privileged --gpus all \
  -e DISPLAY \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v "$HOME/.Xauthority:/root/.Xauthority" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
  -v "$(pwd)/docker/env:/root/env" \
  -v "$MODELS_DIR:/workspace/models" \
  -v "$(pwd)/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100" \
  real-robot /bin/bash
```

That script mount replaces the SO100 directory at runtime. Python dependencies
were installed at build time. Models persist in the host directory; the real
container's other generated files disappear when the `--rm` container exits.

**Host**, extra teleop shell:

```bash
docker exec -it teleop bash
```

**Host**, extra real-robot shell (a different terminal):

```bash
docker exec -it real-robot bash
```

An interactive shell reads the sourced environment helper from `.bashrc`.
Noninteractive `docker exec` does not inherit the entrypoint's exported Isaac
library environment: use `/workspace/isaaclab/_isaac_sim/python.sh` for Python
modules in such executions, as the evaluation runner does.

## Persistent environment

Both images append [docker/utils.sh](../docker/utils.sh) to `.bashrc`. Its actual
function name is `setenv` (some usage text mistakenly says `envset`). It writes
`export KEY=value` lines to `/root/env`, which is the host's ignored `docker/env`.
Use simple values without shell metacharacters; the helper sources that file.
Do not commit credentials or copy an existing environment file into documentation.
New shells read saved settings; existing shells must reload or export them.

| Variable | Reader and default / purpose |
| --- | --- |
| `TELEOP_PORT`, `TELEOP_ID` | Both recorder parsers; `/dev/ttyACM0`, `leader_arm_1` |
| `ROBOT_PORT`, `ROBOT_ID` | Recorder follower arguments; `/dev/ttyACM1`, `follower_arm_1`; calibration checker independently defaults to `/dev/ttyACM0` |
| `CAMERA_GRIPPER`, `CAMERA_EXTERNAL` | Both recorder parsers; unset means no corresponding real camera |
| `CAMERA_WIDTH`, `CAMERA_HEIGHT` | Both recorders; 640, 480 |
| `CAMERA_FOURCC` | Both recorders; unset/empty means backend default |
| `CONTROL_FPS` | Both recorder parsers; 30; does not set real-eval rate |
| `ROOM_USD_PATH` | `tasks/my_room_env_cfg.py`; unset uses packaged room |
| `STATS_JSON` | `so101_check_calibration.py`; defaults to adjacent `calibration_stats.json` |
| `PICK_PLACE_EVAL_STARTS` | `mdp/resets.py`; fallback sidecar directory for Eval resets |
| `PICK_PLACE_EVAL_CHECKPOINT`, `MODEL` | `lerobot_eval.py`; report checkpoint fallback, in that order, then `unknown` |
| `MODELS_DIR` | Host evaluation shell script; `$HOME/models`; also a shell variable in our real-container launch |
| `DISPLAY` | Docker/X11; inherited from host |
| `RERUN_FLUSH_NUM_BYTES`, `RERUN_MEMORY_LIMIT` | Real `so101_control.py`; `8000`, `10%` |

`ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` are passed to the Isaac base image by the
launch recipe. Read the vendor terms before using them. Camera variables do not
automatically configure `so101_eval.py`; its explicit camera dictionary is in
[Evaluation](06-evaluation.md).

## Discovery, permissions and calibration

**Teleop container**, identify devices individually and inspect captured camera
images before assigning indices:

```bash
lerobot-find-port
lerobot-find-cameras opencv
```

These are external LeRobot entry points. `lerobot-find-cameras` accepts the
optional positional `opencv` or `realsense`, `--output-dir` (default
`outputs/captured_images`) and `--record-time-s` (6.0). RealSense support depends
on its optional backend being installed. Prefer stable `/dev/serial/by-id/` paths
for arms; never infer leader/follower identity from enumeration order.

**Teleop container**, replace all device placeholders, then persist settings:

```bash
setenv TELEOP_PORT '/dev/serial/by-id/<leader-device>'
setenv ROBOT_PORT '/dev/serial/by-id/<follower-device>'
setenv TELEOP_ID leader_arm_1
setenv ROBOT_ID follower_arm_1
setenv CAMERA_GRIPPER '<wrist-camera-index>'
setenv CAMERA_EXTERNAL '<external-camera-index>'
setenv CAMERA_WIDTH 640
setenv CAMERA_HEIGHT 480
setenv CONTROL_FPS 30
```

The recorder checks that serial paths exist, are character devices, are readable
and writable, and are distinct when a follower is enabled. For native host
permission errors, use appropriate device groups and log out/in after changes.
The mounted `/dev` and privileged container are the existing workshop recipe.

**Host**, if your user lacks serial/camera group access:

```bash
sudo usermod -a -G dialout "$USER"
sudo usermod -a -G video "$USER"
```

**Teleop container**, with correct IDs/ports, calibrate both arms and smoke-test
physical teleoperation. Omit the follower steps for a leader-only setup:

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id="$TELEOP_ID"
lerobot-calibrate --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID"
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID" \
  --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id="$TELEOP_ID"
```

Follow LeRobot's prompts and reach real mechanical limits without trapping cables.
Calibration files persist through the shared Hugging Face calibration mount.
Stop the smoke test before starting another process that opens the same arm.

## Calibration diagnostics and manual control

**Teleop container**, repository root, compare against the supplied baseline
and attempt a live encoder read:

```bash
cd /workspace/Sim-to-Real-SO-101-Workshop
python docker/real/scripts/so101_check_calibration.py
```

This script has no argparse flags. It uses `ROBOT_ID`, `ROBOT_PORT`, `STATS_JSON`,
checks motion ranges against baseline mean ±2 standard deviations and warns for
large homing offsets or live positions outside range. It can report file-only
results when connection fails; printed PASS is not proof that a live check ran.

**Teleop container**, to build a baseline from your own known-good calibration
files (replace the input directory; requires matplotlib):

```bash
python docker/real/scripts/so101_calibration_stats.py \
  --calib-dir '<calibration_samples_dir>' \
  --output-json docker/real/scripts/calibration_stats.json \
  --output-plot outputs/calibration_stats.png
```

Its four options are `--calib-dir` (default `real_robot/sample_callibrations`),
`--output-json` (`real_robot/calibration_stats.json`), `--output-plot`
(`real_robot/calibration_stats.png`) and `--exclude` (zero or more file stems).
The defaults are historical paths; supply paths that actually exist. Rebuilding
this baseline changes a mounted file; review it before saving it to Git.

**Teleop container**, passive control (review the source's initial/home poses
before using active mode on your own hardware):

```bash
python docker/real/scripts/so101_control.py \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID" \
  --passive_mode=true
```

The Draccus `SetupConfig` supports `robot`, `play_sounds=false`,
`passive_mode=false`, `rerun=false`. Active mode moves to the script's initial
pose on connection and home pose on disconnection. Passive mode disables torque
and prints the final pose. These poses are setup-specific; inspect
[so101_control.py](../docker/real/scripts/so101_control.py) before commanding motion.

**Teleop container**, alternative per-joint keyboard controller:

```bash
python docker/real/scripts/so101_manual_control.py \
  --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID" \
  --step-size 1.0
```

`--robot.type` and `--robot.id` are required; `--robot.port` defaults to
`/dev/ttyUSB0`. `--step-size` defaults to 1.0; `--use-degrees` is a Boolean switch.
Left/right adjust, up/down select joints, `z` commands all joints to zero,
`q`/Escape exits. Zero is a motion command, not a stop command.

## Updating the extension

**Teleop container**, after updating the host checkout's mounted `source/`:

```bash
cd /workspace/Sim-to-Real-SO-101-Workshop
python -m pip install -e source/sim_to_real_so101
list_envs
```

The entrypoint also does this install on each container launch. Restart agent
processes to load changed Python/config/JSON. `list_envs` launches Isaac with a
window; it has no argparse CLI and no headless option. Expect the ten Gym IDs
listed in [Task definition](03-define-a-task.md).

## Git hygiene across machines

**Host or training machine**, in each clone, set your own identity and use
fast-forward-only updates. Save local work before pulling:

```bash
git config user.name '<your_name>'
git config user.email '<your_email>'
git pull --ff-only
```

After an explicitly coordinated history rewrite, save local changes and local
commits elsewhere first. The following discards tracked local work and points
the current branch at remote main; it is not the normal update command.

**Host or training machine**, clean clone on `main` after saving local work:

```bash
git fetch && git reset --hard origin/main
```
