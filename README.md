# sim-to-real (s2r)

The s2r restructure of the SO-101 sim-to-real platform: Isaac Lab 3.0 (Newton / MJWarp) for simulation,
LeRobot v0.6.1 for teleop and datasets, and pluggable policy servers (GR00T N1.7, MolmoAct2).
The previous code, with its full history, is on branch [`v1`](../../tree/v1).

The `v2` restructure is in progress. [Phase status and verification](docs/restructure-status.md) records the
acceptance gates; [repository conventions](CLAUDE.md) covers units, quaternion order and import boundaries.
Copy `.env.example` to `.env` for local device settings. The arm contract passed its Phase 1 CPU, recording, replay-target and simulator acceptance checks;
the evidence is tracked in the status file. Declarative scenes, the generic task engine, `s2r check`,
module-only commands, data generation, and the second-arm hardware checks remain pending.

## Layout

```
s2r                     CLI launcher (PYTHONPATH=src python3 -m cli)
pyproject.toml          packages cli, core, sim (src/) and the console scripts
arms/<arm>/             typed arm.toml, optional <arm>_utils.py, usd/, calibration/units/<id>.json
scenes/<scene>/         scene.toml (env_cfg entry point), real_setup.json (measured room + cameras), usd/
tasks/<task>/           task.toml, __init__.py (TASK), env_cfg.py (variants), mdp.py, eval.py, benchmark.py,
                        eval_sets/*.json
configs/teleop/         lerobot_agent YAMLs (sim_only, paired_real)
configs/eval/           lerobot_eval YAMLs (one per benchmark run)
policies/<name>/        policy.toml, client.py, server.py, train/   (see policies/README.md)
docker/sim/             s2r-sim image: Isaac Lab 3.0 + LeRobot (teleop, record, eval, replay)
docker/policy/          s2r-serve / s2r-train images: one uv venv per policy
src/core/               no Isaac: arms, calibration, tasks (spec + loader), policy_client, policy_server,
                        results, utils, scripts
src/sim/                Isaac only: envs (+ mdp), scene, teleop, recording, generate, eval
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

## Commands

Sim commands run inside the `s2r-sim` container (`S2R_SIM_CONTAINER` overrides the name):

| Command | Program | What it does |
| --- | --- | --- |
| `s2r record --config_path configs/teleop/paired_real.yaml` | `lerobot_agent` | Leader-driven teleop, sim-only or paired sim+real recording |
| `s2r spawn --object blue_cube` | `spawn` | Put a parked task object beside the sim jaw while recording |
| `s2r eval --config_path configs/eval/pick_place.yaml` | `lerobot_eval` | Evaluate a served policy on the fixed benchmark |
| `s2r rollout ...` | `real_rollout` | Run a served policy on the real follower |
| `s2r eval-set --dataset <root> --out <json>` | `make_eval_set` | Build a fixed benchmark from a recorded dataset |
| `s2r dr-replay ...` | `dr_replay` | Re-render recorded episodes with colour / light / robot-colour DR |
| `s2r calibration-replay ...` | `calibration_replay` | Method B calibration replay (physics, keeps successes) |
| `s2r calibration-report --gap ...` | `calibration_report` | Tracking-gap report |
| `s2r calibrate leader\|follower --port ...` | `calibrate_arm` | LeRobot calibration saved where `arm.toml` expects it |
| `s2r calibration-check --id follower_arm_1` | `check_calibration` | Follower calibration vs `arms/so101/calibration/stats.json`, then live encoders (read only) |
| `s2r calibration-stats` | `calibration_stats` | Rebuild stats from `arms/so101/calibration/samples/` |
| `s2r envs` | `list_envs` | List arms, scenes, tasks and gym ids |

Policy commands run on the host (or a remote GPU with `--host user@box --remote-root /abs/path`):

```
s2r build [sim|serve|train|all]          build the images from this checkout
s2r serve --policy groot_n17 --checkpoint <dir> --models-dir <root> [--option k=v ...]
s2r prepare|train|verify --policy <name> --dataset <dir> --output <dir>
s2r policies                             list policies and their train commands
s2r tunnel --host user@box --port N      reach a remote policy server from the local sim
```

Host-side helpers: `compare_eval_results`, `summarize_eval_sweep`, `new_policy`.

## Recording

`lerobot_agent` drives the sim (and, with `follower.enabled`, the real follower) from the leader arm.
Newton builds its model once, so spawnable task objects are declared up front and parked off the table;
`spawn --object <name>` (or `sim_reset_object_name` in the config, at each episode start) teleports one beside
the fixed-jaw tip. The single spawnable object's pose is written per frame to `<sim root>/pick_place_meta/`,
which eval recorded starts, `make_eval_set` and `dr_replay` read. The generic `task_meta/` layout is scheduled for
Phase 3. New recording-alignment markers save the arm, joint order and units, selected physical mapping, and
relative-roll behavior. Resuming with different or legacy alignment metadata uses a new root when
`dataset.on_alignment_mismatch: new` is configured. Legacy recordings remain readable for replay; a per-unit
recording without a saved mapping needs an explicit `--unit` to establish its action units.

## Evaluation

`lerobot_eval` reads a `configs/eval` YAML (flags override it). Keep the same eval set, and its hash, across
the checkpoints you compare; `compare_eval_results` runs paired McNemar tests on the results JSONs.
