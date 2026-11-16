# Sim-to-Real · v3

Tabletop manipulation with Isaac Lab 3.0 / Newton, LeRobot datasets, and pluggable policy servers.
The included arms are SO-101 and SP7; the tasks are pick-and-place, stacking two cubes, and stacking bowls.

On `v3`, run programs as Python modules. The `s2r` wrapper and `src/cli` are removed until the CLI is rebuilt.
The previous layouts remain in Git history. [CLAUDE.md](CLAUDE.md) defines the repository's units,
import boundaries, physics and benchmark invariants.

## Layout

```text
workspace/
  arms/<arm>/         arm.toml, utils.py, calibration/, usd/, urdf/
  scenes/<scene>/     scene.toml, utils.py, mounts/<arm>.toml, usd/, textures/
  tasks/<task>/       task.toml, task.py, assets/ or usd/, starts/, eval_sets/
  policies/<policy>/  policy.toml, client.py, server.py, optional train/
  configs/            record/, eval/, rollout/, datagen/
src/
  utils/              manifests, geometry, calibration, codecs, datasets, atomic I/O
  core/tasks/         explicit task loading, stage latching and failure reporting
  core/eval/          fixed starts, eval sets, reports/comparisons, resume and sweeps
  core/policy_client/ self-contained policy clients
  core/policy_server/ self-contained policy servers
  core/rollout/       real-arm rollout and HTTP/GUI bridges
  sim/actions/        checks, expert moves and unified IK
  sim/workspace/      arm/scene composition and generic task environments
  sim/teleop/         devices, command queue, recording loop and object spawner
  sim/recording/      shared setup, dataset writers, replay and recorder GUI backend
  sim/datagen/        scripted generation, grid starts, Mimic and replay
  sim/eval/           simulator evaluation, scene capture, video and paired trials
gui/                  recorder and real/simulation consoles
tests/                utils/, core/, sim/, workspace/, policies/, frozen oracles
```

Arm, scene and task loaders resolve selection at call time. `load_task(name)` requires an explicit name.
After selecting and launching the simulator, entry points call `sim.workspace.envs.register(scene, task)`.
Gym IDs retain `S2R-<task>-<variant>`; the scene-only ID is `S2R-<scene>`.

## Environment and commands

Copy `.env.example` to `.env` if needed, then set local serial ports, cameras, calibration IDs and dataset paths.
Keep hardware-specific values out of committed files. The simulator requires Docker, NVIDIA GPU support,
and the Isaac image's dependencies.

Build images from the repository root:

```bash
docker build -f docker/sim/Dockerfile -t s2r-sim:latest .
docker build -f docker/policy/Dockerfile --target serve -t s2r-serve:latest .
```

For example, generate with the retained random-start reference recipe:

```bash
S2R_GIT_COMMIT=$(git rev-parse HEAD) docker compose run --rm sim \
  s2r-entrypoint python -m sim.datagen.scripted \
  --config_path workspace/configs/datagen/scripted.yaml
```

Pass `S2R_GIT_COMMIT` so container outputs retain source provenance. Paths in YAML files are container paths.
Choose a fresh generation destination when repeating a run. Existing datasets are preserved. Compose exports
`S2R_ARM`, `S2R_SCENE` and `S2R_TASK`; pass explicit `--arm/--scene/--task` flags when switching to another
recipe selection (for example `--task stack_bowls`).

| Workflow | Module | Configuration or input |
| --- | --- | --- |
| Teleop and recording | `sim.teleop.agent` | `workspace/configs/record/sim_only.yaml` or `paired_real.yaml` |
| Simulator evaluation | `sim.eval.run` | `workspace/configs/eval/pick_place.yaml` |
| Scripted generation | `sim.datagen.scripted` | `workspace/configs/datagen/scripted.yaml`, `stack_two_cubes.yaml`, `stack_bowls.yaml` |
| Mimic generation | `sim.datagen.mimic` | `workspace/configs/datagen/mimic.yaml` |
| Visual DR replay | `sim.datagen.dr_replay` | `workspace/configs/datagen/dr_replay.yaml` |
| Calibration replay | `sim.datagen.calibration_replay` | Use the module's `--help` for source and mapping options |
| Real-arm testing GUI | `core.rollout.testing_gui` | `workspace/configs/rollout/gui.yaml` |
| Matched real/sim trials | `sim.eval.sim_real.trials` | `workspace/configs/eval/sim_real.yaml` |
| Real follower rollout | `core.rollout.real_rollout` | `workspace/configs/rollout/` |
| Generate/convert eval sets | `core.eval.eval_set` | Recorded dataset and captured scene geometry |
| Compare result JSONs | `core.eval.report` | Two or more complete benchmark reports |
| Summarize horizon sweeps | `core.eval.sweep` | Use the module's `--help` |
| Calibration | `utils.calibration.calibrate`, `.check`, `.stats`, `.report` | Selected arm/unit and explicit hardware settings |

Use `python -m <module> --help` for arguments. YAML/CLI entry points let explicit flags override YAML.
For recording with USB devices, add the hardware override:

```bash
S2R_GIT_COMMIT=$(git rev-parse HEAD) \
  docker compose -f compose.yaml -f compose.hardware.yaml run --rm sim \
  s2r-entrypoint python -m sim.teleop.agent \
  --config_path workspace/configs/record/sim_only.yaml
```

The sim-only recording example starts with dataset recording disabled. Set `dataset.repo_id`, `dataset.root`
and `dataset.task_name` together to enable it. `follower.enabled` separately enables the physical follower.

Policy servers use the Compose `serve` profile, with a separate environment per policy:

```bash
S2R_POLICY=groot_n17 S2R_CHECKPOINT=checkpoint S2R_MODELS_DIR=/path/to/models \
  docker compose --profile serve up -d serve
```

Match the policy client, server endpoint, checkpoint options and camera rename map. See
[policy contracts and training commands](workspace/policies/README.md) and the selected policy's manifest.
The `example` policy repeats current joint positions and tests transport only.

## Recording and generation

Recording uses one command queue for GUI buttons and typed terminal commands: `start`, `save`, `discard`,
`reset`, `quit`; `spawn <object>` selects a named object. `control_input: gui` disables terminal input and
requires `gui.enabled: true`. Keyboard listeners are removed. Both GUIs obtain spawnable choices from the
selected task; choose an object in the dropdown before spawning it beside the open jaw. Newton bodies are
created before simulation and repositioned when spawned.

`sim.recording.setup` owns calibration/root preparation, environment configuration, cameras and trajectory
validation. Alignment markers retain the selected device codec and wrist-roll behavior. A mismatching or
legacy alignment requires a separate root; `dataset.on_alignment_mismatch: new` selects that flow.

Scripted generation uses the task's expert moves and the arm's IK settings. `[arm.ik].solver` selects `dls`
or `bounded`; `[task.expert]` may override the solver, control offset and orientation constraints. Mimic uses
the same task expert settings. Calibration command replay retains its arm settings.

Scripted defaults to a seeded grid over task start ranges. `--cells_per_axis`, `--yaw_steps` and `--max_starts`
control its size; reports include per-object coverage maps. It skips invalid combinations, then cycles its
selected starts. An explicit `--starts_dir` selects recorded starts. The retained `scripted.yaml` explicitly
sets `start_mode: random`; use it to reproduce that recipe. Mimic and fixed eval sets never use grid starts.
`--debug` writes optional rollout diagnostics and boundary images.

Generated frames capture observations before their action, then execute that action and observe task signals.
Only success that remains valid at the final frame is committed. Actions are absolute simulator targets encoded
through the device mapping. Annotations retain FK of the executed target separately from requested waypoints.

Schema-2 `task_meta/episode_<index>.json` sidecars contain one metre/WXYZ pose per tracked object per dataset
frame. Legacy `pick_place_meta/` remains readable. The shared readers and alignment utilities are in
`utils.datasets`.

## Adding an arm, scene or task

Anything specific to an arm, scene or task belongs in its `workspace/` folder. Optional arm/scene hooks are
named `utils.py`. Scenes own anchors, lighting, cameras and physics; `mounts/<arm>.toml` owns the arm/scene
combination. Arm joint tables define action order, names, roles and units. No generic code assumes a motor
index or joint count.

Copy [the task template](workspace/tasks/_template/) to `workspace/tasks/<name>/`, set `[task].name` to the
folder name, and edit both files. `task.toml` declares objects, starts, variants and episode settings.
`task.py` composes stages from `sim.actions` checks and an expert from moves; it must import without Torch
or Isaac. Its stage order must match the manifest. Templates are starting points for validation, not validated
hardware/task demonstrations.

The stage engine latches progress and uses the final declared stage as success. Failures are
`failed_at_<stage after the furthest latched stage>`; `timeout_holding` is a separate boolean. Stage, waypoint,
object, sensor and event names are part of the stored-data contract.

Live quaternions use **XYZW**; sidecars and benchmark JSON use **WXYZ**. Simulator joint values are radians
or metres. Calibration spans are physical degrees/metres; raw device values may use normalized or driver
encoder-degree coordinates. Unit files live under `workspace/arms/<arm>/calibration/units/`; recording selects
`follower.unit`, and evaluation/replay/rollout select `--unit`. `JointMapping` owns these conversions.

## Evaluation and verification

Keep eval-set content IDs, seeds and schedules fixed when comparing checkpoints. Schema conversion retains
`legacy_eval_set_id`; saved robot states use the set's mapping. Live geometry validation remains mandatory.
A geometry mismatch requires an explicit new geometry/set decision. The pinned rejection of `train-059`
is retained. Use `--smoke-test --smoke-selection one_per_zone` with a fixed set for a short wiring check;
the example policy's outcomes are not a learned-policy success measurement.

Run the local baseline comparator and required regressions:

```bash
PYTHONPATH=src python tests/fixtures/v3_baseline/compare_outcomes.py
PYTHONPATH=src python -m pytest tests/test_import_graph.py tests/core/test_eval_set_regression.py tests/workspace -q
python -B -m compileall -q src workspace tests
```

`uv sync --group host` provides host tools; it is not a full Isaac/LeRobot environment. Run the complete suite
in the simulator image:

```bash
S2R_GIT_COMMIT=$(git rev-parse HEAD) docker compose run --rm sim \
  s2r-entrypoint python -m pytest tests -q
```

The restructure retains old-vs-new task/IK oracles and report byte-equality tests. The approved Phase 12
hash-gate exception and fresh control results are recorded in
[phase12_gate.json](tests/fixtures/v3_baseline/phase12_gate.json): unchanged Phase 11 repeats differed, so exact
simulator frame-hash equivalence remains unproven. Intermittent native test crashes are recorded with their
runtime versions; production physics and installed SDK files were not changed to obtain a pass.

Training and visual DR recipes retain their documented SO-101/pick-place scope. Integration tests, finite
commands or a scripted success do not establish learned task performance. SP7 simulation/manifest support
does not establish hardware support; its real driver remains unconfigured.

Branch `feat/async-policy-worker`, which touches only `core/policy_client`, **must be rebased onto `v3`**.
