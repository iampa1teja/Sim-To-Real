# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`s2r`: an SO-101 sim-to-real platform. Isaac Lab 3.0 (Newton physics, MJWarp solver, RTX cameras) simulates a
photo-matched digital twin of a real room; LeRobot v0.6.1 drives the real leader/follower arms and owns the dataset
format; policies (GR00T N1.7, MolmoAct2) run as separate ZMQ servers. The same leader arm teleoperates sim and real
together, producing paired datasets. The pre-restructure code lives on branch `v1`; many comments say "ported from v1".

## Commands

Almost nothing runs on a bare host: sim code needs the `s2r-sim` container (Isaac Lab + GPU), policy code needs the
`s2r-serve` / `s2r-train` containers. `./s2r` (= `PYTHONPATH=src python3 -m cli`, Python >= 3.11) only *plans* a
`docker` / `ssh` / `rsync` command, prints it, and runs it.

```bash
./s2r --dry-run record --config_path configs/teleop/sim_only.yaml   # print the docker exec without running it
./s2r build [sim|serve|train|all]
docker compose up -d sim                                            # compose.hardware.yaml adds USB/X11, compose.remote.yaml the server
./s2r record --config_path configs/teleop/paired_real.yaml          # lerobot_agent inside s2r-sim
./s2r serve --policy groot_n17 --checkpoint <rel path> --models-dir <root> [--option k=v]
./s2r eval --config_path configs/eval/pick_place.yaml               # lerobot_eval inside s2r-sim; flags override the YAML
./s2r prepare|train|verify --policy <name> --dataset <dir> --output <dir> [-- script args]
./s2r policies | envs
```

Sim commands (`SIM_PROGRAMS` in `src/cli/paths.py`) bypass argparse and forward every argument to the console script
inside the container, so `--dry-run` must come *before* the command name. The full command table is in `README.md`.

### Tests

CPU-only, `unittest`-style, run with pytest from the repo root (`src` must be on the path):

```bash
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python -m pytest tests/policy_client/test_smoothing.py -q            # one file
PYTHONPATH=src python -m pytest tests/policy_client/test_grasp.py -k latch -q       # one test
```

Host deps are the `host` dependency group in `pyproject.toml` (`uv sync --group host`); several tests skip themselves
when `torch`/`pxr` are missing. Only `tests/**/test_*.py` are tests; the other scripts in `tests/sim/`
(`check_*`, `audit_*`, `replay_*`, `profile_*`, ...) are diagnostics that need the running sim container.

Three test files currently fail at collection because their targets are not in the repo: `tests/sim/test_app.py`
(`src/sim/app.py`) and `tests/test_code_explorer_*.py` (`tools/code_explorer/`, and `tools` is git-ignored). Add
`--ignore` for them or expect 3 collection errors.

Lint config is `.flake8` (max line length 120, google docstrings); there is no formatter or CI config.

## Architecture

### Import boundary

- `src/core/` must import without Isaac, `pxr`, or (in `policy_client`) torch. It runs on the host, in the sim
  container, and partly in the policy container.
- `src/sim/` is Isaac-only. Anything touching `isaaclab` / `pxr` must be imported *after* `AppLauncher` has started the
  app, so entry points (`lerobot_agent.main`, `lerobot_eval.main`, `dr_replay`, `calibration_replay`) parse config
  first and do Isaac imports inside functions.
- `src/cli/` is stdlib-only planning code: every `plan_*` in `cli/paths.py` returns a `Plan` and never executes, so
  commands can be asserted exactly.

### Arm, scene and task are data, selected by environment variables

`arms/<arm>/arm.toml`, `scenes/<scene>/{scene.toml,real_setup.json}`, `tasks/<task>/task.toml` are read by
`core.arms.spec`, `core.scenes`, `core.tasks.base`. Selection is `S2R_ARM` / `S2R_SCENE` / `S2R_TASK`
(defaults `so101`, `rpd_meeting_room`, `pick_place_cube`), and these are read **at import time** into module constants
(`ARM_NAME`, `SCENE_NAME`, `TASK_NAME`, plus derived ones such as `FOLLOWER_JOINTS`, `SETUP`). YAML `arm:/scene:/task:`
keys are exported by `apply_selection()` before any `core`/`sim` import; keep that ordering when adding entry points.

`tasks/` is not a Python package on `sys.path`: `core.tasks.loader` mounts it as the synthetic package `s2r_tasks`, and
each `tasks/<task>/__init__.py` must define `TASK` (a `core.tasks.base.Task`). Use `load_task_module("benchmark")` to
import a task helper from core code.

### Env composition

Importing `sim.envs` registers gym ids: `S2R-base`, `S2R-<scene>` and one `S2R-<task>-<variant>` per
`[task.variants]` entry. A variant is a function `apply_<variant>(cfg, task)` in `tasks/<task>/env_cfg.py` that mutates
the scene's env cfg (from `scene.toml` `env_cfg`), layered as `BaseEnvCfg` -> `RpdMeetingRoomEnvCfg` -> task variant.
`apply_eval` calls `apply_teleop`, `apply_eval_dr` calls `apply_eval`. Teleop variants must have no terminations;
eval variants add the success termination and recorded-start resets. Reset event *order* matters and is asserted at
startup (`check_pick_place_eval_event_order`).

Newton builds its model once, so objects cannot be created at runtime: spawnable task objects are declared up front,
parked below the scene on reset (`park_object`), and teleported to the jaw by `sim.generate.object_spawner`
(triggered by `s2r spawn`, the GUI, or `sim_reset_object_name`).

### Units and conventions (the usual source of bugs)

- **Joint units.** LeRobot raw units are -100..100 for arm joints and 0..100 for the gripper, keyed `<joint>.pos`.
  `core.arms.mapping.JointMapping` converts raw <-> sim radians using the calibrated span, then clamps to
  `runtime_joint_limits`. Datasets, policies and the real follower all use raw units; only the sim uses radians.
  `tasks/pick_place_cube/eval.py::dataset_state_to_radians` deliberately duplicates the maths because frozen eval sets
  carry their own `joint_mapping`.
- **Joint names.** `arm.toml` pairs LeRobot names (`shoulder_pan`, ...) with USD names (`Rotation`, ...) by position.
- **Quaternions.** Isaac Lab 3.0 tensors and configs are XYZW. Everything on disk (object sidecars, eval sets) and the
  numpy geometry helpers in `tasks/pick_place_cube/eval.py` are WXYZ. Convert at the boundary with `_wxyz` / `_xyzw`
  (`sim/envs/mdp/resets.py`).
- **Frames.** Recorded object poses and eval-set starts are in the robot base-link frame, metres/radians.
- **Isaac Lab 3.0 data** is a ProxyArray; wrap reads in `as_torch(...)`.

### Calibration provenance

`real_setup.json` holds the measured scene (robot pose, cameras, lighting, materials) plus `sim_joint_mapping`,
`sim_joint_mapping_source` and `runtime_joint_limits`. `core.arms.joint_calibration.resolve_calibration` picks the
follower/leader calibration files (`HF_LEROBOT_CALIBRATION`, `SO101_FOLLOWER_CALIBRATION`, config ids) and returns a
`CalibrationSelection` with SHA-256 provenance; connecting a device whose loaded calibration differs from the sim
mapping is refused. `core.utils.recording_alignment` writes a contract to `<dataset>/meta/recording_alignment.json`;
resuming a dataset recorded under a different calibration fails, or (`on_alignment_mismatch: new`) goes to a suffixed
sibling folder. Much of this code is intentionally fail-closed; do not loosen checks to make an error go away.

### Recording (`sim.teleop.lerobot_agent`)

One control loop at `fps`: read leader -> map to radians (wrist roll is applied relative to its episode-start value) ->
optionally read the real follower -> add frames -> send the *unmodified* leader action to the follower -> `env.step`.
`SynchronizedLeRobotRecorders` keeps the `sim` and `real` datasets frame-aligned; `RerecordingSession` splices a
re-recorded episode into private copies. Each sim episode also gets a per-frame object-pose sidecar,
`<sim root>/pick_place_meta/episode_*.json`, which eval starts, `make_eval_set` and `dr_replay` depend on.
Controls come from the terminal, Isaac keyboard, or the browser GUI (`sim/recording/web_gui.py` serving `gui/`, SSE +
`POST /command`); all of them only set request flags that the loop consumes once per tick.

### Policies

Each `policies/<name>/` folder is a plugin; folders starting with `_` and manifests with `enabled = false` are skipped.

- `policy.toml` is validated by `core.policy_client.discovery` (and re-parsed by `docker/policy/entrypoint.sh`).
- `client.py` exposes `create(manifest, options) -> ClientPolicy`; no torch; imports `core.policy_client...`.
- `server.py` exposes `create() -> ServerPolicy`; runs in a per-policy uv venv in the container, where
  `src/core/policy_server` and `src/core/policy_client` are copied flat to `/opt/src`, so server-side code imports
  `policy_server...` / `policy_client...` (no `core.` prefix).
- Transport is ZMQ REQ/REP with msgpack + `.npy`-encoded arrays; endpoints `get_action`, `ping`, `reset`, `describe`,
  `seed_policy`, `kill`. Wire format: `{"state", "images", "instruction"}` -> `{"actions": (T, J)}` absolute joints.
- `core.policy_client.remote_policy.RemotePolicy` owns the action queue, smoothing (`smoothing.py`), the optional
  gripper latch (`grasp.py`) and per-episode seeding in raw units; `sim.eval.sim_remote_policy.SimRemotePolicy` wraps
  it with radians <-> raw conversion and the camera `rename_map`; `core.scripts.real_rollout` uses it directly.
- Training: `[train.commands]` scripts run in the train image with `SO101_POLICY_DIR`, `SO101_DATASET`,
  `SO101_OUTPUT`, `SO101_BASE_MODEL`, `SO101_PYTHON`, `SO101_PROJECT_DIR` set (`plan_train` and the entrypoint share
  this contract).

`new_policy <name>` copies `policies/_template/`. `policies/README.md` has the plugin walkthrough.

### Evaluation

`sim.eval.lerobot_eval` flattens a `configs/eval/*.yaml` into argparse defaults (CLI flags override), then runs one env
against a served policy. With `eval_set`, starts come from a frozen benchmark JSON
(`tasks/pick_place_cube/eval_sets/`, splits `id`/`ood`/`yaw`/`train`), validated by content hash (`eval_set_id`);
results are schema-2 reports with Wilson intervals, stage rates (`reached` ... `success`) and failure modes, written
atomically after each episode so `--resume` can continue. Compare checkpoints only on the same eval set
(`compare_eval_results`, paired McNemar). Success is defined in `tasks/pick_place_cube/mdp.py`: jaw-contact grasp plus
lift, release inside the box's measured inner volume, held for `confirm_steps`.

## Known stale references

Docs and tests mention things that are not in this checkout: `docs/` (e.g. `docs/05-training.md`,
`docs/07-policy-plugins.md`), `tests/policies/test_conformance.py`, `scripts/update_sim_calibration.py`,
`src/sim/app.py`, `tools/code_explorer/`. `tasks/pick_place_cube/benchmark.py` imports `one_per_zone` from
`.eval_smoke`, but that module is `core.results.eval_smoke`. `src/s2r.egg-info/` and `.env` (device ports/ids, read by
`compose.hardware.yaml`) are tracked.

## Planned restructure

`docs/restructure-plan.md` is the agreed plan for making arms, scenes, tasks and policies pluggable (target layout,
file formats, file-by-file changes, phases). Check it before moving files or adding abstractions.
