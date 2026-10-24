# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of NVIDIA's SO-101 sim-to-real workshop. It adds a photo-matched **MyRoom** digital twin and a
cube pick-and-place task to the original vial-to-rack workflow. Isaac Lab handles simulation, LeRobot
handles teleop and datasets, and GR00T (N1.6 legacy, N1.7 current) handles training and policy serving.
The `docs/01..06-*.md` guides are the maintained reference for every workflow; `README.md` is the hub
and command index.

Training and preparation use `./docker/train.sh`; simulation policy rollouts use
`./docker/eval.sh`. The host GR00T checkout, host real-arm environment and physical
rollout client were removed to reclaim space. See `AGENTS.md` and the maintained
training/evaluation guides. Model weights and recorded datasets remain in storage.

## Execution environments (important)

Code runs in three container images (teleop, serve, train), plus two legacy GR00T ones, and an optional host test environment. Check which one a command belongs to before you run it:

1. **Teleop/sim container** (`teleop-docker`, built from `docker/sim/Dockerfile`, Isaac Sim Python 3.11).
   `source/`, `outputs/`, `datasets/` and `docker/real/scripts` are bind-mounted at
   `/workspace/Sim-to-Real-SO-101-Workshop`. The entrypoint runs `pip install -e source/sim_to_real_so101`
   on every launch. Console commands such as `pick_place_agent`, `lerobot_eval` and `list_envs` only
   exist here. `setenv` is a helper sourced from `docker/utils.sh`.
2. **Serve/eval container** (`so101-serve`, alias `policy-server:latest`; `--target serve` of
   `policy_servers/Dockerfile`, built with `./so101 build serve`). One image serves any policy under `policies/`
   in its own uv venv. `docker/eval.sh <config>.yaml` starts and stops it from the host.
3. **Train container** (`so101-train`, `--target train` of the same Dockerfile, `./so101 build train`). Runs a
   policy's `[train.commands]` through `entrypoint.sh --env <policy> -- <cmd>` in a separate `<policy>-train`
   venv. Not yet validated on a GPU machine.

`./so101 build [teleop|serve|train|all]` builds all three; `--registry NAME --push` publishes them (the teleop push
also needs `--allow-isaac-push`); `./so101 pull` fetches them and keeps the old local names working.

**Legacy images, still needed until `./so101 train --generic` is validated for GR00T:** `real-robot:n1.7`
(`./docker/real/build.sh ada n17`) and `so101-groot-train:n1.7` (`./docker/train/build.sh`, used by
`./docker/train.sh`, which runs preparation, verification and training against the pinned GR00T checkout at
`/Isaac-GR00T`). Checkpoints default to `/external_storage/models`. MolmoAct2 trains through
`docker/molmo_train.sh`, which uses the serve image.

CPU-only tests can use an existing host Python environment with their dependencies;
do not recreate the deleted real-arm environment just to run them.

Isaac-dependent modules (`tasks/`, `mdp/`, `assets/real_setup.py`, anything that imports `pxr`, `isaaclab`
or `isaacsim`) can't be imported on the host.

## Tests

The tests are CPU-only. Most of them fake Isaac/USD through `tests/_fakes.py`: `install_stubs()` patches
`sys.modules` with fake `pxr`, `isaaclab` and `isaacsim`, and `load_module(name, path)` imports a repo
file by path. The test files mix `unittest` and plain pytest functions, so use pytest to run everything.
The host has ROS Jazzy on its path, and ROS's pytest plugins break collection. Disable plugin autoload:

```bash
# CPU-only suites in an environment with their dependencies installed
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
# single file / single test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_smoothing.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_grasp.py -k latch -q
# GR00T-dependent suites run inside the training image
./docker/train.sh python -m unittest discover -s tests -p test_train_dit_only.py -v
./docker/train.sh python -m unittest discover -s tests -p test_prepare_pick_place_gr00t.py -v
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
  Reset, observation and success functions live in `mdp/`.
- **Measured setup**: `assets/real_setup.json` holds one physical room's measured geometry and camera
  calibration. `assets/real_setup.py` (`SETUP`, `robot_rotation`, `camera_rotation`) turns it into
  poses that every MyRoom launcher uses. If you change the JSON, sim placement changes everywhere.
- **Recording**: `pick_place_agent` drives a LeRobot leader and the sim, and serves the browser recorder
  (`source/gui/app.jsx`, served by `utils/recording_web_gui.py`, http://localhost:8765). The
  `utils/recording_*`, `episode_*`, `rerecording.py` and `replay_*` modules handle dataset writing,
  sidecar metadata, resume and replay.
- **Policy client**: `policy_client/` is an Isaac-free plugin client with ZMQ/msgpack
  transport, joint transforms and plugin discovery. `smoothing.py` covers blending,
  EMA and temporal ensembling; `grasp.py` holds optional gripper latching and diagnostics.
  Simulation evaluation uses it through `utils/lerobot_interface.SimRemotePolicy`.
- **Evaluation**: `docker/eval.sh <config>.yaml` is the host-side orchestrator. It starts the GR00T server
  container (or uses `--external_server`/`--server_only`), runs `lerobot_eval` in the teleop container
  and writes results JSON. Fixed start sets live in `eval_sets/*.json`, built with
  `make_pick_place_eval_set`. Keep the same set (and its hash) across checkpoints you compare. Benchmark
  stages and success logic are in `utils/pick_place_benchmark*.py` and `utils/pick_place_eval.py`.
  `scripts/compare_eval_results.py` runs paired McNemar comparisons.
- **Training pipeline** (`policies/groot_n17/train/`, run through `./docker/train.sh`; declared in the policy's
  `[train]` table): `prepare_pick_place_gr00t.py` makes an
  independent v2.1 copy of a recorded dataset and never modifies the source. `verify_pick_place_training.py`
  checks it. `launch_pick_place_n17.py` and `train_pick_place*.sh` wrap the official, unmodified GR00T
  fine-tune launcher. `train_dit_only.py` and `train_dit.sh` continue a checkpoint with only
  `action_head.model` trainable. `so_arm_n17_config.py` defines the embodiment modality config with
  `room`/`wrist` cameras and absolute joint actions.

- **Platform CLI**: `./so101` (`cli/so101_cli/`) is the front door for prepare/train/verify/serve/eval and the
  remote helpers. `plan.py` holds pure planners returning exact commands; GR00T and MolmoAct2 train through
  the wrappers `docker/train.py` and `docker/molmo_train.sh` until the shared train image exists.

## Conventions and gotchas

- N1.7 models expect camera keys `room`/`wrist`. Older N1.6 checkpoints use `front`/`wrist` and the
  `real-robot` image (see `eval.sh --help`).
- `datasets/` and `outputs/` are local and unversioned. Preparation and audit scripts check source hashes;
  never edit recorded datasets in place.
- Upstream GR00T sources stay untouched. Wrappers patch behavior in-process and pin exact revisions.
- `lerobot_push_dataset` is a known-broken console entry. Use the module workaround in `docs/05-training.md`.
- Keep the SPDX/Apache headers in files that came from upstream NVIDIA or Isaac Lab.
