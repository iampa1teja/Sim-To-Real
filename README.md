# sim-to-real (s2r)

A sim-to-real platform for tabletop manipulation arms: Isaac Lab 3.0 (Newton / MJWarp) for simulation,
LeRobot v0.6.1 for teleop and datasets, and pluggable policy servers (GR00T N1.7, MolmoAct2).
Arms, scenes, tasks and policies are folders: `s2r new` scaffolds one and `s2r check` validates it.
The included arms are **SO101** and **SP7**. The `_template` folder is scaffolding
for the importer, not a selectable robot.
The previous code, with its full history, is on branch [`v1`](../../tree/v1).

Copy `.env.example` to `.env` for local device settings. [Repository conventions](CLAUDE.md) covers units,
quaternion order and import boundaries.

What has been verified, and what has not:

- The SO-101 in the meeting-room and blank-table scenes on the pick-place and stacking tasks: recording,
  paired real/sim motion and cameras, named-object benchmarks and evaluation.
- Scripted generation produced a physically successful pick-place episode that passes dataset validation and
  GR00T preparation/verification. Mimic episodes pass at the source start and a nearby changed start. These
  show physical success and valid datasets, not a measured success rate over every start.
- GR00T and MolmoAct2 serve and evaluate with real inference and finite joint commands, and a two-step GR00T
  run verifies a real optimizer update. That establishes integration, not learned-policy success.
- The GR00T and MolmoAct2 training recipes and the visual DR replay recipe still assume the SO-101 and
  pick-place.
- No second arm has been validated on hardware. The template arm establishes composition only.
- SP7 retains its URDF, meshes, calibration mapping and self-contained simulation USD.
  Its real hardware driver remains unconfigured.
- The three-bowl task uses the supplied bowl USD with separate yellow, red and blue
  instances. Its [generation and replay recipe](tasks/stack_bowls/README.md) produces
  a simulation demonstration; Cosmos augmentation has not passed visual validation.

The [verification record](docs/restructure-status.md) holds the evidence and the renderer limitations.

## Guides

| Workflow | Guide |
| --- | --- |
| Install, launch, create a first dataset | [Getting started](docs/01-getting-started.md) |
| Arm manifest, USD, mapping and calibration | [Add an arm](docs/02-add-an-arm.md) |
| Room, anchors, cameras and mounts | [Add a scene](docs/03-add-a-scene.md) |
| Objects, starts, stages and success | [Add a task](docs/04-add-a-task.md) |
| Prepare, verify and train a policy | [Training](docs/05-training.md) |
| Capture geometry, freeze starts and compare results | [Evaluation](docs/06-evaluation.md) |
| Implement a client/server plugin | [Policy plugins](docs/07-policy-plugins.md) |
| Scripted experts, Mimic and replay | [Data generation](docs/08-data-generation.md) |

## Layout

```
s2r                     CLI launcher (PYTHONPATH=src python3 -m cli)
pyproject.toml          packages cli, core, sim (src/) and the single `s2r` console entrypoint
arms/<arm>/             typed arm.toml, optional <arm>_utils.py, usd/, calibration/units/<id>.json
scenes/<scene>/         declarative scene.toml, optional appearance.json, mounts/<arm>.toml, scene.py, usd/
tasks/<task>/           task.toml, __init__.py (TASK), env_cfg.py (variants), mdp.py, eval_sets/*.json
configs/record/         recording YAMLs (sim_only, paired_real)
configs/eval/           evaluation YAMLs (one per benchmark run)
configs/rollout/        real-rollout YAMLs (shadow mode by default)
configs/datagen/        source-specific generation YAMLs
policies/<name>/        policy.toml, client.py, server.py, train/   (see policies/README.md)
docker/sim/             s2r-sim image: Isaac Lab 3.0 + LeRobot (teleop, record, eval, replay)
docker/policy/          s2r-serve / s2r-train images: one uv venv per policy
src/core/               no Isaac: arms, scenes, calibration, tasks (spec + stages), benchmark, datasets,
                        policy_client, policy_server, rollout, utils
src/sim/                Isaac only: envs (+ mdp), scene, teleop, recording, datagen, eval
src/cli/                the s2r CLI (plans docker commands)
tests/                  CPU-only tests
```

Arm, scene and task are data. `S2R_ARM`, `S2R_SCENE` and `S2R_TASK` (or `--arm/--scene/--task`) select them;
`sim.envs` registers `S2R-<scene>` and one `S2R-<task>-<variant>` gym id per `[task.variants]` entry
(`teleop`, `eval`, `eval_dr` for pick_place_cube).

## Arm and calibration selection

`core.arms.spec.load_arm_spec()` returns `(ArmSpec, arm_directory)`. The arm's ordered `[[arm.joints]]` tables
own LeRobot names, USD names, joint types and roles, initial values, and runtime limits. Optional gripper,
wrist-roll, and wrist-camera settings live in the arm folder. CAD-specific collision and appearance hooks live
in `<arm>_utils.py`. The SO101 camera variants share one CAD asset through a small USD override layer.

An arm with `lerobot.calibration = "per_unit"` selects an explicit file under
`arms/<arm>/calibration/units/<unit_id>.json`. For recording, set the unit in the existing teleop YAML:

```yaml
follower:
  unit: rpd_robot       # sim-only example; the paired SO101 example uses rpd_follower
```

Evaluation, rollout, and replay accept `--unit <unit_id>`; the eval examples also set `unit:` in YAML.
A unit identifies measured physical spans and anchors; a follower `id` identifies its LeRobot device calibration
file. These are separate settings. `s2r calibrate` writes the raw LeRobot calibration into the folders declared
by `arm.lerobot` under `$HF_LEROBOT_CALIBRATION` (default `~/.cache/huggingface/lerobot/calibration`). Unit files
also retain the mapping source and anchor evidence. The selected unit and loaded device file are checked for
agreement before a paired session proceeds.

Factory-calibrated arms need no unit file. Their revolute mapping uses the hardware's full encoder range;
prismatic joints use their declared full stroke in metres. Joint count, gripper positions in the action vector,
and the presence of a wrist-roll joint are derived from the manifest.

Raw normalized body joints use `-100..100`; gripper joints use `0..100`. Driver degree mode is an encoded motor
coordinate: LeRobot centres it on the measured encoder-range midpoint, which can differ from the arm's physical
zero. Grippers retain their normalized mode. Selected mappings preserve optional `raw_min`/`raw_max` endpoints,
including reversed directions, alongside physical spans. When deriving reversed normalized encoders from raw
calibration, the unit source declares the bus's `apply_drive_mode` capability. Explicit raw endpoints can describe
other driver conventions. Calibration spans and runtime limits are physical degrees for revolute joints and metres
for prismatic joints. `JointMapping.to_sim()` applies calibration, clamps
to runtime limits, and converts revolute joints to radians; `to_raw()` reverses the unit conversion without a
runtime clamp. Recorded actions and policy requests retain device units in manifest joint order.

## Scene selection

`core.scenes.load_scene_spec()` reads the selected scene at call time. A scene manifest declares its room USD,
anchors, fixed cameras, lighting, renderer settings and physics overrides. Arm-mounted cameras remain in
`arm.toml`. Fitted room materials and camera response live in the scene's optional `appearance.json`.
The meeting room uses an optional `scene.py:customize` hook for its measured material and table-photo treatment.

An optional `mounts/<arm>.toml` selects the mount anchor, a world-space translation correction, and initial
joint positions in radians/metres. Without that file, the builder uses the scene's mount anchor and the arm's
declared initial joints. `scenes/blank_table/` exercises this path with a flat table and one fixed camera.
Choose it with `S2R_SCENE=blank_table` or `scene: blank_table` in a recording YAML. Explicit YAML values take
precedence over environment defaults. The production recorder has been exercised on this scene without a mount
file; see the status file for the reference-image comparison and recording evidence.

## Tasks and object metadata

`tasks/<task>/task.toml` declares ordered stages, tracked objects, object placements and the start distribution.
The task's `signals()` reads simulator signals and `observe()` turns a step's poses/signals into stage flags.
The shared engine latches those flags and derives success from the last declared stage. Task-specific grasp,
placement and failure rules stay in the task folder. `stack_two_cubes` provides a second task with two tracked
objects; select it through `task:` in a recording/evaluation YAML or `--task stack_two_cubes`.

New recordings write schema-2 `task_meta/episode_<index>.json` sidecars with task/arm identity, metre positions,
WXYZ orientations and a size/trajectory entry for every `task.tracked_objects` name. Each trajectory has exactly
one pose per dataset frame. Dataset sources and alignment live in `core.datasets`; benchmark geometry, starts,
statistics and comparison live in `core.benchmark`. Existing single-object `pick_place_meta/` datasets remain
readable through the same production source/start readers. Use `s2r migrate <dataset> --task <task> --arm <arm>` to write schema-2 sidecars explicitly.

## Commands

Install the CPU dependencies for host validation with `uv sync --group host`. The `./s2r` launcher uses that environment; `uv run s2r` also works.

`s2r new arm|scene|task|policy <name>` creates a folder from its template. `s2r check --static` validates every
extension without Isaac imports. Static checking needs NumPy; it does not import Torch or the simulator.
`s2r check` then runs a small set of headless compositions covering every available arm, scene and task;
explicit selectors check one composition. `--policy` checks the selected policy's action dimension against
the selected arm, without requiring every fixed-embodiment policy to support every arm.

Sim commands run inside the `s2r-sim` container (`S2R_SIM_CONTAINER` overrides the name):

| Command | Program | What it does |
| --- | --- | --- |
| `s2r record --config_path configs/record/paired_real.yaml` | `python -m sim.teleop.agent` | Leader-driven teleop, sim-only or paired sim+real recording |
| `s2r spawn --object blue_cube` | `python -m cli.spawn` | Put a parked task object beside the sim jaw while recording |
| `s2r eval --config_path configs/eval/pick_place.yaml` | `python -m sim.eval.run` | Evaluate a served policy on the fixed benchmark |
| `s2r test-gui --config_path configs/rollout/gui.yaml` | `python -m core.rollout.testing_gui` | Open the real-arm prompt and policy testing GUI |
| `s2r rollout ...` | `python -m core.rollout.real_rollout` | Run a served policy on the real follower |
| `s2r eval-set --dataset <root> --out <json> --task <task> --arm <arm> --scene_json <geometry.json>` | `python -m core.benchmark.eval_set` | Build a fixed benchmark from recorded starts and captured live geometry |
| `s2r datagen --source dr_replay ...` | `python -m sim.datagen.dr_replay` | Re-render recorded episodes with colour / light / robot-colour DR |
| `s2r datagen --config_path configs/datagen/scripted.yaml` | `cli.datagen` selects `sim.datagen.scripted` | Task-owned waypoints, bounded IK and successful episodes only |
| `s2r datagen --config_path configs/datagen/mimic.yaml` | `cli.datagen` selects `sim.datagen.mimic` | Isaac Lab Mimic retargets annotated source demonstrations; successful episodes only |
| `s2r calibration-replay ...` | `python -m sim.datagen.calibration_replay` | Method B calibration replay (physics, keeps successes) |
| `s2r calibration-report --gap ...` | `python -m core.calibration.report` | Tracking-gap report |
| `s2r calibrate leader\|follower --port ...` | `python -m core.calibration.calibrate` | LeRobot calibration saved where `arm.toml` expects it |
| `s2r calibration-check --id follower_arm_1` | `python -m core.calibration.check` | Follower calibration vs `arms/so101/calibration/stats.json`, then live encoders (read only) |
| `s2r calibration-stats` | `python -m core.calibration.stats` | Rebuild stats from `arms/so101/calibration/samples/` |
| `s2r envs` | `python -m cli.list_envs` | List arms, scenes, tasks and gym ids |

Policy commands run on the host (or a remote GPU with `--host user@box --remote-root /abs/path`):

```
s2r build [sim|serve|train|all]          build the images from this checkout
s2r serve --policy groot_n17 --checkpoint <dir> --models-dir <root> [--option k=v ...]
s2r prepare|train|verify --policy <name> --dataset <dir> --output <dir>
s2r policies                             list policies and their train commands
s2r tunnel --host user@box --port N      reach a remote policy server from the local sim
```

Use `s2r compare` for paired results, `s2r sweep` for sweep summaries, and `s2r new policy <name>` for a policy scaffold.

## Recording

`s2r record` drives the sim (and, with `follower.enabled`, the real follower) from the leader arm.
Newton builds its model once, so spawnable task objects are declared up front and parked off the table;
`spawn --object <name>` (or `sim_reset_object_name` in the config, at each episode start) teleports one beside
the fixed-jaw tip. Every object declared in `task.tracked_objects` is written per frame to
`<sim root>/task_meta/episode_<index>.json` with task/arm identity and WXYZ orientations. Recorded starts,
eval-set generation and replay also read legacy single-object `pick_place_meta/` sidecars.
New recording-alignment markers save the arm, joint order and units, selected physical mapping, and
relative-roll behavior. Resuming with different or legacy alignment metadata uses a new root when
`dataset.on_alignment_mismatch: new` is configured. Legacy recordings remain readable for replay; a per-unit
recording without a saved mapping needs an explicit `--unit` to establish its action units.

## Evaluation

`s2r eval` reads a `configs/eval` YAML (flags override it). Keep the same eval set, and its hash, across
the checkpoints you compare; `compare_eval_results` runs paired McNemar tests on the results JSONs.
Schema-2 eval sets declare the task, arm, joint order and named object poses for each start. Saved robot states
remain raw LeRobot values interpreted with the set's saved joint mapping. The three approved frozen-set
conversions retain their former hashes in `legacy_eval_set_id`; comparison accepts the shared identity.
That conversion preserves geometry. The frozen pick-place sets still predate the existing mount correction,
so their geometry guard rejects the corrected live mount until an explicit geometry decision is made.

Compare the same policy on matched physical/simulation starts with `./s2r eval-sim-real`.
The console at http://localhost:8765 has a prompt, policy dropdown, duration per side,
Spawn, Run real → sim and two independently selectable real/simulation camera views.
See [the comparison guide](docs/09-sim-real-comparison.md).


## Real-arm testing GUI

```bash
./s2r test-gui --config_path configs/rollout/gui.yaml
```

Open **http://localhost:8766**. Click **Connect real arm** to see its cameras, select a policy,
enter a task prompt, then click **Run on real arm**. **Stop** or **Esc** cancels the run;
a late policy reply cannot issue another command. Closing the controlling browser stops the run
when its heartbeat expires. Disconnect retains motor torque and never starts calibration.
The GUI starts no simulator and opens no device until Connect is clicked.

Hardware and camera defaults come from the current local `.env`; explicit CLI flags override them.
Select another calibrated arm with `--arm` and, when needed, `--unit`. The policy dropdown checks
joint counts and required camera inputs against that arm. The `example` policy holds the current
position and is only a wiring test.

The selected policy must already be served using `s2r serve`, locally or remotely. Changing the dropdown
selects its client protocol; the server owns the loaded checkpoint. Set **Policy server** and **Port**
under **Connection settings** to match it. For example:

```bash
./s2r serve --policy molmoact2 --checkpoint <checkpoint-relative-to-models-dir> --models-dir <models-dir> --port 5556
```

For the wider pan deployment, append `--option action_extrapolation_joints=shoulder_pan`
and `--option action_range_diagnostics=true` to that serve command. This experimental option
decodes pan predictions beyond the checkpoint's 1st/99th-percentile envelope, preserving the
original in-range values, other joints, state normalization and saved model statistics.
The real client still clamps targets to the calibrated range and limits each motor command.
Allowing a wider range does not establish learned task performance in that wider workspace.

The current GUI profile also enables experimental pan workspace retargeting:
`pan_model_scale: 0.5` and `pan_model_offset: 0.0`. Observed pan angles are halved
for the policy, and returned pan targets use the exact inverse, so the model's
usual pan envelope covers approximately -71 to +82 driver units instead of
-36 to +41. Other joints retain their original conversion. This is a deployment
adaptation, not a calibration correction or evidence of successful box placement.
Set the scale to `1.0` to restore the original pan workspace. Action horizon is unchanged.

The GUI config pins the `rpd_follower` unit and its requested `7705cc34…` file hash;
Connect refuses a different file. Run reports include the loaded server options and joint
transform, and action traces retain the requested target alongside the bounded/sent targets.

Per-policy client options are in `configs/rollout/gui.yaml`. Physical commands use the selected
calibration, per-joint runtime bounds, the existing first-target distance check and the follower's
relative-target limit. A run ends at its configured duration or when Stop is pressed; a finished
execution does not automatically label the physical task as successful. Action traces and downloadable
run reports are saved under `outputs/real_testing/`.
