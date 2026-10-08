# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of NVIDIA's SO-101 sim-to-real workshop. It adds a photo-matched **MyRoom** digital twin and a
cube pick-and-place task to the original vial-to-rack workflow. Isaac Lab handles simulation, LeRobot
handles teleop and datasets, and GR00T (N1.6 legacy, N1.7 current) handles training and policy serving.
The `docs/01..06-*.md` guides are the maintained reference for every workflow; `README.md` is the hub
and command index.

For the user's GR00T N1.7 200k real-arm rollout, use the saved commands in
[docs/06-evaluation.md](docs/06-evaluation.md#n17-200k-real-arm). The server uses port 5560 and the
external-storage milestone checkpoint (`/external_storage/models`). The client uses action horizon 8,
room/wrist YUYV cameras and the saved start pose. See also `AGENTS.md`.

## Execution environments (important)

Code runs in four separate places. Check which one a command belongs to before you run it:

1. **Teleop/sim container** (`teleop-docker`, built from `docker/sim/Dockerfile`, Isaac Sim Python 3.11).
   `source/`, `outputs/`, `datasets/` and `docker/real/scripts` are bind-mounted at
   `/workspace/Sim-to-Real-SO-101-Workshop`. The entrypoint runs `pip install -e source/sim_to_real_so101`
   on every launch. Console commands such as `pick_place_agent`, `lerobot_eval` and `list_envs` only
   exist here. `setenv` is a helper sourced from `docker/utils.sh`.
2. **Policy-server container** (`real-robot:n1.7`, built with `./docker/real/build.sh ada n17`).
   `docker/eval_pick_place.sh` starts and stops it from the host.
3. **Host uv env**: `uv run --project source/sim_to_real_so101`. The `host` dependency group holds
   torch, lerobot 0.4.2, pytest and similar packages. They live in a dependency group rather than
   `[project]` deps so the container's editable install doesn't pull them into Isaac Sim's Python. The
   real-arm client `docker/real/scripts/so101_eval.py` runs here, with `--no-sync --with feetech-servo-sdk ...`.
4. **Pinned GR00T checkout** at `~/Isaac-GR00T-N1.7/.venv/bin/python`, with pins in
   `scripts/gr00t_model_profiles.py`. Training, dataset preparation and the GPU-free training tests use it.

Isaac-dependent modules (`tasks/`, `mdp/`, `assets/real_setup.py`, anything that imports `pxr`, `isaaclab`
or `isaacsim`) can't be imported on the host.

## Tests

The tests are CPU-only. Most of them fake Isaac/USD through `tests/_fakes.py`: `install_stubs()` patches
`sys.modules` with fake `pxr`, `isaaclab` and `isaacsim`, and `load_module(name, path)` imports a repo
file by path. The test files mix `unittest` and plain pytest functions, so use pytest to run everything.
The host has ROS Jazzy on its path, and ROS's pytest plugins break collection. Disable plugin autoload:

```bash
# full suite (host uv env)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project source/sim_to_real_so101 --no-sync python -m pytest tests -q
# single file / single test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project source/sim_to_real_so101 --no-sync python -m pytest tests/test_smoothing.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project source/sim_to_real_so101 --no-sync python -m pytest tests/test_grasp.py -k latch -q
# GR00T-dependent suites use the pinned GR00T venv (as the docs do)
~/Isaac-GR00T-N1.7/.venv/bin/python -m unittest discover -s tests -p test_train_dit_only.py -v
GROOT_CHECKOUT=<path_to_Isaac-GR00T> python -m unittest discover -s tests -p test_prepare_pick_place_gr00t.py -v
```

Some tests skip or fail on the host because `isaaclab` or the GR00T checkout is missing. Read the error
before you treat a failure as a regression.

Lint: `flake8` with the repo `.flake8` (max line 120, Google docstrings; E402/E501/W503/E203 ignored).

## Architecture

**Package** `source/sim_to_real_so101/` is an Isaac Lab extension installed as the namespace package
`sim_to_real_so101`. Console entry points are listed in its `pyproject.toml` `[project.scripts]`.

- **Tasks**: `tasks/__init__.py` registers the Gym IDs (`Lerobot-So101-Teleop-*`). The configs form an
  inheritance chain:
  `so101_env_cfg.SO101TeleopEnvCfg` → `my_room_env_cfg.MyRoomEnvCfg` (digital-twin room, calibrated
  room/wrist cameras) → `pick_place_env_cfg.PickPlaceEnvCfg` → `PickPlaceEvalEnvCfg` →
  `PickPlaceEvalDREnvCfg`. Each layer subclasses the parent's Scene/Events/Observations configs. The
  vials-to-rack task is a separate branch from the base. Reset, observation and success functions
  live in `mdp/`.
- **Measured setup**: `assets/real_setup.json` holds one physical room's measured geometry and camera
  calibration. `assets/real_setup.py` (`SETUP`, `robot_rotation`, `camera_rotation`) turns it into
  poses that every MyRoom launcher uses. If you change the JSON, sim placement changes everywhere.
- **Recording**: `pick_place_agent` drives a LeRobot leader and the sim, and serves the browser recorder
  (`source/gui/app.jsx`, served by `utils/recording_web_gui.py`, http://localhost:8765). The
  `utils/recording_*`, `episode_*`, `rerecording.py` and `replay_*` modules handle dataset writing,
  sidecar metadata, resume and replay.
- **Policy client**: `gr00t_client/` is a vendored, Isaac-free GR00T client. `server_client.py` is a ZMQ
  and msgpack `PolicyClient`. `smoothing.py` covers action chunk blending, EMA, temporal ensembling,
  async inference and fixed-rate scheduling. `grasp.py` holds the optional gripper latch and grasp
  diagnostics. Sim eval (`scripts/lerobot_eval.py` via `utils/lerobot_interface.GR00TRemotePolicy`) and
  the real client (`docker/real/scripts/so101_eval.py`) both use this one module, so a change there
  affects both paths.
- **Evaluation**: `docker/eval_pick_place.sh` is the host-side orchestrator. It starts the GR00T server
  container (or uses `--external_server`/`--server_only`), runs `lerobot_eval` in the teleop container
  and writes results JSON. Fixed start sets live in `eval_sets/*.json`, built with
  `make_pick_place_eval_set`. Keep the same set (and its hash) across checkpoints you compare. Benchmark
  stages and success logic are in `utils/pick_place_benchmark*.py` and `utils/pick_place_eval.py`.
  `scripts/compare_eval_results.py` runs paired McNemar comparisons.
- **Training pipeline** (`scripts/`, run in the GR00T venv): `prepare_pick_place_gr00t.py` makes an
  independent v2.1 copy of a recorded dataset and never modifies the source. `verify_pick_place_training.py`
  checks it. `launch_pick_place_n17.py` and `train_pick_place*.sh` wrap the official, unmodified GR00T
  fine-tune launcher. `train_dit_only.py` and `train_dit.sh` continue a checkpoint with only
  `action_head.model` trainable. `so_arm_n17_config.py` defines the embodiment modality config with
  `room`/`wrist` cameras and absolute joint actions.

## Conventions and gotchas

- N1.7 models expect camera keys `room`/`wrist`. Older N1.6 checkpoints use `front`/`wrist` and the
  `real-robot` image (see `eval_pick_place.sh --help`).
- `datasets/` and `outputs/` are local and unversioned. Preparation and audit scripts check source hashes;
  never edit recorded datasets in place.
- Upstream GR00T sources stay untouched. Wrappers patch behavior in-process and pin exact revisions.
- `lerobot_push_dataset` is a known-broken console entry. Use the module workaround in `docs/05-training.md`.
- Keep the SPDX/Apache headers in files that came from upstream NVIDIA or Isaac Lab.
