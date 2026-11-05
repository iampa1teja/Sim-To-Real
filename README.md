# sim-to-real (s2r)

The s2r restructure of the SO-101 sim-to-real platform: Isaac Lab 3.0 (Newton / MJWarp) for simulation,
LeRobot v0.6.1 for teleop and datasets, and pluggable policy servers (GR00T N1.7, MolmoAct2).
The previous code, with its full history, is on branch [`v1`](../../tree/v1).

## Layout

```
s2r                     CLI launcher (PYTHONPATH=src python3 -m cli)
pyproject.toml          packages cli, core, sim (src/) and the console scripts
arms/<arm>/             arm.toml (links, joints, actuators, LeRobot names), usd/
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
which eval recorded starts, `make_eval_set` and `dr_replay` read.

## Evaluation

`lerobot_eval` reads a `configs/eval` YAML (flags override it). Keep the same eval set, and its hash, across
the checkpoints you compare; `compare_eval_results` runs paired McNemar tests on the results JSONs.
