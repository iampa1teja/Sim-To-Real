# s2r restructure plan

Goal: any arm, any scene, any task, any policy. Generate a dataset for a task (teleop or entirely in sim), train any
policy on it, and judge policies on a fixed benchmark in sim and on the real robot. Adding an arm, scene, task or
policy must be "add a folder, run `s2r check`", with no edits under `src/`.

Status of this document: a plan written from reading the code. Nothing in it has been run in the simulator; every
phase that touches `src/sim/` must be verified inside the `s2r-sim` container before the next one starts.
The browser GUI (`gui/`, `src/sim/recording/web_gui.py`) is out of scope: it is being replaced by React + Django.

Decisions already taken:

| # | Decision |
| --- | --- |
| 1 | Scenes are declarative (`scene.toml` + a generic builder); `scene.py` is an optional escape hatch. |
| 2 | Arm-specific Python lives in the arm folder as `arms/<arm>/<arm>_utils.py`. |
| 3 | Git LFS for USD / textures: undecided, not part of this plan. |
| 4 | `SO101_*` variables become `S2R_*`; programs are reached only through `s2r <command>`. |
| 5 | Object sidecars move from `pick_place_meta/` to a generic `task_meta/`. |

Contents: [1 Background reading](#1-background-reading) · [2 Cleanup](#2-cleanup-of-the-current-repository) ·
[3 Target structure](#3-target-structure) · [4 Contracts and file formats](#4-contracts-and-file-formats) ·
[5 File-by-file changes](#5-file-by-file-changes) · [6 Phases](#6-phases-and-acceptance-checks) ·
[7 Risks and open questions](#7-risks-and-open-questions)

---

## 1. Background reading

| Area | What it is used for here | Link |
| --- | --- | --- |
| Isaac Lab 3.0 | Manager-based envs, scene/asset/sensor configs, event and termination managers | https://isaac-sim.github.io/IsaacLab |
| Isaac Lab Mimic | Demo multiplication from sub-task signals (phase 5) | https://isaac-sim.github.io/IsaacLab (Mimic / imitation learning section) |
| Isaac Sim | Kit app, RTX renderer, viewport | https://docs.isaacsim.omniverse.nvidia.com |
| Newton | Physics backend; builds its model once (no runtime object creation) | https://newton-physics.github.io/newton |
| MuJoCo Warp | The solver Newton runs here (`MJWarpSolverCfg`) | https://github.com/google-deepmind/mujoco_warp |
| OpenUSD | Stage, prims, `UsdPhysics`, `UsdShade`; all scene and arm assets | https://openusd.org |
| LeRobot v0.6.1 | Robot / teleoperator devices, calibration, dataset format (v2.1 and v3) | https://github.com/huggingface/lerobot · https://huggingface.co/docs/lerobot |
| Gymnasium | `gym.register` / `gym.make` for the `S2R-*` ids | https://gymnasium.farama.org |
| Isaac GR00T N1.7 | Policy plugin `groot_n17` | https://github.com/NVIDIA/Isaac-GR00T |
| MolmoAct2 | Policy plugin `molmoact2` | https://github.com/allenai/molmoact2 |
| uv | One venv per policy inside the policy image | https://docs.astral.sh/uv |
| ZeroMQ / pyzmq | REQ/REP transport between client and policy server | https://pyzmq.readthedocs.io |
| msgpack | Wire encoding (arrays as embedded `.npy`) | https://msgpack.org |
| draccus | YAML + CLI parsing for the recorder config | https://github.com/dlwh/draccus |
| Rerun | Live camera / joint visualisation | https://rerun.io/docs |
| TOML / `tomllib` | Every manifest (`arm.toml`, `scene.toml`, `task.toml`, `policy.toml`) | https://toml.io · https://docs.python.org/3/library/tomllib.html |
| Docker Compose, NVIDIA Container Toolkit | The `s2r-sim`, `s2r-serve`, `s2r-train` containers | https://docs.docker.com/compose · https://github.com/NVIDIA/nvidia-container-toolkit |
| pytest, flake8 | CPU tests, lint (`.flake8`, line length 120) | https://docs.pytest.org · https://flake8.pycqa.org |

Repository conventions that are easy to get wrong are in `CLAUDE.md` (units, quaternion order, import boundary).

---

## 2. Cleanup of the current repository

Phase 0. No behaviour change; the existing CPU tests must still pass.

### 2.1 Delete

| Path | Reason |
| --- | --- |
| `src/core/policy_client/types.py` | `ActionChunk` has no users. |
| `smoothing.py`: `InferenceWorker`, `InferenceResult`, `FixedRateScheduler`, `log_late` | No users (leftovers of the v1 threaded real-robot loop). |
| `transport.py`: `PolicyClient.kill_server` | No users. |
| `recording_alignment.py`: `synced_start_pose` | No users. |
| `calibration_check.py`: `reference_poses`, `pose_measurement` | No users; the report docstring says the pose half was removed. |
| `joint_angle_check.py`: `summarize_teleop`, `observation_only_session`, `compare_readback` | No users. Confirm `make_observation_only_robot` / `guard_observation_only_robot` are still reached before removing them too. |
| `joint_calibration.py`: `sim_gripper_closed_degrees`, `sim_wrist_roll_zero_offset_degrees` | No users. |
| `policies/groot_n17/embodiment_tags.py`, `policies/groot_n17/uv.lock` | Zero-byte files. |
| Every `.gitkeep` in a folder that has other files (11 of them) | Not needed. |
| `src/s2r.egg-info/` | Build artefact; add `*.egg-info/` to `.gitignore`. |
| `compose.remote.yaml` | Duplicates `s2r serve`. |
| `tests/test_code_explorer_indexer.py`, `tests/test_code_explorer_server.py` | Target `tools/code_explorer/`, which is not in the repo. (Restore the tool instead if it is wanted.) |

### 2.2 Fix

| Path | Change |
| --- | --- |
| `.env` | Rename to `.env.example`; add `.env` to `.gitignore`. |
| `.gitignore` | Remove the `tools` entry (section 3 adds `tools/diagnostics/`). |
| `tasks/pick_place_cube/benchmark.py:255` | `from .eval_smoke import one_per_zone` -> `from core.results.eval_smoke import one_per_zone`. |
| `scenes/rpd_meeting_room/scene.toml` | Remove the "Not read by code yet" comment. |
| `joint_calibration.py` error text | Remove the reference to `scripts/update_sim_calibration.py`. |
| `policies/README.md`, `policies/groot_n17/train/README.md` | Dead links to `docs/` and `tests/policies/test_conformance.py`; fixed when phases 4 and 7 create them. |

### 2.3 De-duplicate

| Duplicated | Single home |
| --- | --- |
| `rendering_kit_args` (4 copies) and `apply_selection` (2 copies) | New `src/sim/app.py`. This also makes `tests/sim/test_app.py` pass. |
| `as_torch` (`obs.py`, `resets.py`) | `src/sim/envs/mdp/_common.py`. |
| `file_sha256` (`hashing.py`, `joint_calibration.py`) | `core/utils/hashing.py`. |
| Raw units <-> degrees (`JointMapping`, task `eval.py::dataset_state_to_radians`, `calibration_replay._units_to_degrees`, `calibration_check.units_to_degrees`) | `core/arms/mapping.py`: pure functions taking an explicit mapping, used by all four. |
| `git_commit` / `git_identity` | `core/utils/provenance.py`. |
| `TRAIN_COMMANDS` (`cli/paths.py`, `discovery.py`) | `discovery.py`; the CLI imports it. |
| `client.py` in `_template`, `example`, `groot_n17` (same request/response code) | `core/policy_client/default.py::DefaultClientPolicy`; the three plugins subclass or re-export it. |
| `seed_everything` in both servers | `policy_server/seeding.py` (copied flat into the image like the rest of `policy_server`). |
| Stage and failure logic in `PickPlaceTask` and `benchmark.StageTrace` | One implementation in `core/tasks/stages.py` (phase 3). |

### 2.4 Optimizations

| Where | Change |
| --- | --- |
| `lerobot_eval._evaluate` (about 550 lines) | Split into argument resolution, env + policy setup, the episode loop, and report building (`sim/eval/{run,loop,report}.py`). |
| `lerobot_agent.run` (about 290 lines) | Split setup from the tick; move request handling into a small state machine (`sim/teleop/{agent,loop}.py`). |
| `reset_cube_from_recorded_starts` | Calls `table_plane()` (a full mesh read and least-squares fit) per env per reset. Fit once per env at startup and cache. |
| `reset_object_pose` | Same `table_plane()` call and a jaw-mesh vertex read on every spawn; cache both. |
| `set_cube_color` | Walks the prim range and resolves bound materials on every reset; cache the shader inputs. |
| `load_robot_starts` | Reads all parquet files into Python lists to find frame 0; filter with a pyarrow predicate. |
| `EvalSetStarts.__init__` | `self.data['starts'].index(s)` inside the schedule comprehension is quadratic; enumerate once. |
| `arms/so101/usd/` | Two 23 MB USDs differ only by the camera. Keep one and make the camera variant a USD variant or an override layer. |

---

## 3. Target structure

```
s2r                         launcher
pyproject.toml  README.md  CLAUDE.md  .env.example
compose.yaml  compose.hardware.yaml

arms/<arm>/
    arm.toml                joints, gripper, links, actuators, wrist camera, lerobot, hardware
    <arm>_utils.py          optional: arm-specific spawn fixes
    usd/
    calibration/            samples/, stats.json, units/<unit_id>.json
arms/_template/

scenes/<scene>/
    scene.toml              room USD, fixed cameras, lighting, anchors, physics
    appearance.json         fitted materials and camera response
    mounts/<arm>.toml       mount pose, correction, start joint pose for one arm in this scene
    scene.py                optional: custom env cfg hook
    usd/  textures/
scenes/blank_table/
scenes/_template/

tasks/<task>/
    task.toml               objects, stages, start distribution, success, DR ranges
    __init__.py             TASK
    env_cfg.py  mdp.py      variants and task-specific terms
    expert.py               optional: scripted waypoints for data generation
    usd/  eval_sets/
tasks/_template/

policies/<name>/            policy.toml, client.py, server.py, pyproject.toml, uv.lock, train/
policies/_template/

configs/  record/  eval/  datagen/  train/  rollout/

src/
    cli/                    main.py, plan.py, scaffold.py, check.py
    core/                   no Isaac
        arms/               spec.py, mapping.py, calibration.py
        scenes/             spec.py, anchors.py
        tasks/              base.py, loader.py, stages.py
        benchmark/          eval_set.py, starts.py, geometry.py, report.py, stats.py, compare.py, resume.py, smoke.py, sweep.py
        datasets/           task_meta.py, alignment.py, sources.py
        calibration/        calibrate.py, check.py, stats.py, report.py, gap.py, telemetry.py, display.py
        policy_client/  policy_server/
        rollout/            real_rollout.py
        utils/              hashing.py, provenance.py
    sim/                    Isaac only
        app.py
        envs/               __init__.py (registry), base_env_cfg.py, scene_builder.py, mdp/
        scene/              arm.py, cameras.py, renderer.py, viewport.py, camera_response.py, contact_materials.py, geometry.py, appearance.py
        teleop/             agent.py, loop.py, config.py, devices.py, controls/
        recording/
        eval/               run.py, loop.py, report.py, stage_tracking.py, video.py, sim_remote_policy.py
        datagen/            scripted.py, ik.py, mimic.py, dr_replay.py, dr_variants.py, calibration_replay.py, spawner.py

tests/                      cli/ core/ sim/ policies/ (mirrors src/, plus plugin conformance)
tools/diagnostics/          check_*, audit_*, replay_*, profile_* scripts
docker/sim/  docker/policy/
docs/
gui/                        untouched
```

---

## 4. Contracts and file formats

### 4.1 Arm: `arms/<arm>/arm.toml`

The arm folder owns everything that is true of the arm regardless of where it stands.

```toml
[arm]
name = "so101"                       # must equal the folder name
usd_path = "usd/SO-ARM101-USD.usd"
utils = "so101_utils.py"             # optional; see 4.2

[arm.links]
base = "base"
ee = "gripper"

# One table per joint, in action order. Joint count and order are derived from this list everywhere.
[[arm.joints]]
name = "shoulder_pan"                # LeRobot name; dataset key is "<name>.pos"
usd = "Rotation"                     # joint name in the USD articulation
type = "revolute"                    # revolute | prismatic
role = "arm"                         # arm | gripper
limits = [-110.0, 110.0]             # runtime limits: degrees (revolute) or metres (prismatic)
initial = -0.2736                    # radians or metres

[arm.gripper]
kind = "jaw"                         # jaw (one moving finger) | parallel (two)
joints = ["gripper"]                 # names from [[arm.joints]] with role = "gripper"
contact_bodies = ["jaw"]             # bodies the grasp contact sensor reads
friction_bodies = ["gripper", "jaw"] # bodies that get the grasp contact material
tip_mesh = "wrist_roll_follower_so101_v1"   # mesh measured to place objects beside the fixed finger

[arm.wrist_roll]                     # optional; omit for arms without a roll joint
joint = "wrist_roll"
relative_in_teleop = true            # sim follows the leader's change from episode start

[arm.cameras.wrist]                  # optional; arm-mounted cameras belong to the arm
parent_link = "gripper"
position = [0.0, 0.0, 0.0]
euler_degrees = [0.0, 0.0, 0.0]
width = 640
height = 480
focal_length = 0.0
horizontal_aperture = 0.0
vertical_aperture = 0.0
# calibration = { fx = ..., fy = ..., cx = ..., cy = ... }
# response = { ... }

[arm.appearance]                     # optional: named materials DR and fitted looks may edit
body_material = "Looks/material_a_3d_printed"
motor_material = "Looks/material_sts3215"

[arm.actuators.<group>]              # unchanged from today
joint_names_expr = ["Rotation"]
effort_limit_sim = 30.0
stiffness = 55.0
damping = 0.7

[arm.spawn]                          # unchanged from today
[arm.variants.<name>]                # unchanged from today (deep-merged overrides)

[arm.lerobot]
follower_type = "so101_follower"
leader_type = "so101_leader"
modules = ["lerobot.robots.so_follower", "lerobot.teleoperators.so_leader"]   # imported to register the types
use_degrees = false
calibration = "per_unit"             # per_unit (measured file per robot) | factory (fixed full range)
follower_calibration_dir = "robots/so101_follower"
leader_calibration_dir = "teleoperators/so101_leader"

[arm.hardware]
motor = "ST3215"
encoder_counts_per_turn = 4096
zero_count = 2048
```

Removed keys: `[arm.joints] names`, `[arm.lerobot] names`, `roll_joint`, `gripper_joint`, `[arm.links] fixed_jaw_mesh`,
`[arm.initial_state.joint_positions]` (all folded into the tables above). `[arm.initial_state] position` and
`rotation_euler_degrees` stay as the default pose for the bare `S2R-base` env.

Per physical robot: `arms/<arm>/calibration/units/<unit_id>.json` holds what `real_setup.json` keeps today under
`sim_joint_mapping` and `sim_joint_mapping_source` (calibrated span per joint, gripper closed anchor, roll zero offset,
source hash). A record or rollout config selects a unit by id. For `calibration = "factory"` no unit file is needed;
the mapping is derived from `[arm.hardware]`.

### 4.2 Arm utilities: `arms/<arm>/<arm>_utils.py`

Optional. Loaded by path (same mechanism as policy plugins). Every function is optional.

```python
def prepare_collisions(root_prim) -> None:
    """Runs after the USD is spawned and before contact materials are bound (Newton imports the stage after this)."""

def apply_appearance(stage, robot_prim_path, look: dict) -> None:
    """Apply a fitted look from the scene's appearance.json to this arm."""
```

`so101_utils.py` receives today's SO-101-specific code: `_split_fixed_finger`, `_split_moving_finger`,
`_split_link_mesh` (from `contact_materials.py`), `apply_primitive_pads` (all of `gripper_collisions.py`),
`apply_robot_appearance` and `apply_real_wrist_mount_visual` (from `real_setup.py`).

### 4.3 Scene: `scenes/<scene>/scene.toml`

The scene owns the room. It never names an arm joint, link or material.

```toml
[scene]
name = "rpd_meeting_room"
description = "..."
room_usd = "usd/digital_twin_room.usda"
appearance = "appearance.json"       # optional
hook = "scene.py:customize"          # optional: customize(env_cfg, scene_spec) -> None

[scene.anchors.table]
prim_name = "Tabletop"               # mesh the table plane is fitted to
contact_material = true

[scene.anchors.mount]                # where an arm base goes
position = [0.6972, -0.6076, 0.7635]
yaw_degrees = 184.583662
up = [-0.001159, -0.018505, 0.999828]

[scene.anchors.workspace]            # optional, table frame, metres
xy_min = [0.0, 0.0]
xy_max = [0.0, 0.0]

[scene.cameras.realsense_rgb]        # one table per fixed camera; key becomes camera_<key> in the env
position = [0.0, 0.0, 0.0]
target = [0.0, 0.0, 0.0]
width = 640
height = 480
focal_length = 0.0
horizontal_aperture = 0.0
vertical_aperture = 0.0
data_types = ["rgb"]
depth_sensor = true                  # adds the aligned depth-only sensor
# calibration = { fx = ..., fy = ..., cx = ..., cy = ... }

[scene.lighting]                     # fitted light parameters
[scene.viewer]
eye = [0.0, 0.0, 0.0]
lookat = [0.0, 0.0, 0.0]
camera = "realsense_rgb"             # optional: show this camera in the viewport

[scene.physics]                      # solver overrides on top of the base env
use_mujoco_contacts = false
cone = "elliptic"
impratio = 1000.0
tolerance = 1e-8
num_substeps = 4
```

`appearance.json` holds today's `appearance`, `object_materials` and per-camera `response` blocks.

### 4.4 Mount: `scenes/<scene>/mounts/<arm>.toml`

The only place that knows about one arm in one scene.

```toml
[mount]
anchor = "mount"
translation_correction_m = [-0.0012, -0.0052, 0.0082]
source = "Contact alignment inferred from paired episode 1, 2026-11-06"

[mount.start_joint_positions]        # LeRobot joint names; radians or metres
shoulder_pan = -0.2736
```

A missing mount file is not an error: the arm goes on the anchor with its `arm.toml` initial pose. That is what lets a
new arm run in an existing scene before anything is measured.

### 4.5 Task: `tasks/<task>/`

`task.toml` (changes from today in comments):

```toml
[task]
name = "pick_place_cube"
instruction = "Pick up the cube and place it in the box"
default_variant = "teleop"
stages = ["reached", "grasped", "lifted", "over_box", "placed", "success"]   # new: was a Python constant
tracked_objects = ["blue_cube"]      # new: written to task_meta every frame

[task.variants]                      # unchanged
teleop = "env_cfg:apply_teleop"

[task.objects.white_box]
usd = "usd/White-Box-Physics.usda"
frame = "table"                      # new: table | arm_base | world
position = [0.0, 0.0, 0.0]           # new: in that frame; replaces _TABLE_RIGHT / _TABLE_FRONT_EDGE in env_cfg.py

[task.objects.blue_cube]
size = [0.02, 0.02, 0.02]
mass = 0.008
spawnable = true                     # new: parked on reset, placed by the spawner

[task.start]                         # distribution for random and generated starts
frame = "arm_base"
xy_min = [0.0, 0.0]
xy_max = [0.0, 0.0]
yaw_deg = [0.0, 90.0]

[task.success]   [task.episode]   [task.dr]     # unchanged
```

`core/tasks/base.py`:

```python
class Task(ABC):
    spec: TaskSpec

    @abstractmethod
    def signals(self, env) -> dict[str, torch.Tensor]:
        """Sim side: per-env booleans the stages need (holding, inside container, ...). Called once per step."""

    @abstractmethod
    def observe(self, state: StepState) -> dict[str, bool]:
        """Isaac-free: stage flags for one env from poses + signals. Keys are spec.stages."""

    def failure_mode(self, flags, state) -> str | None: ...
    def sample_start(self, rng) -> dict[str, Pose]: ...          # default: [task.start]
    def start_is_valid(self, start, scene) -> bool: ...          # default: on the table, clear of other objects
    def expert(self): ...                                        # optional: tasks/<task>/expert.py
```

The engine (`core/tasks/stages.py` + `sim/eval/stage_tracking.py`) calls `signals` then `observe` every step, latches
stages, derives success from the last stage and builds the report. No engine code may reference a task by name, an
`_pick_place_*` attribute, or a reset function by `__name__`.

### 4.6 Object sidecars: `<dataset>/task_meta/episode_<index>.json`

```json
{
  "schema_version": 2,
  "task": "pick_place_cube",
  "arm": "so101",
  "reference_frame": "robot base link (base)",
  "units": "m",
  "fps": 30,
  "objects": {
    "blue_cube": { "size": [0.02, 0.02, 0.02], "position": [[x, y, z]], "orientation_wxyz": [[w, x, y, z]] }
  }
}
```

One entry per frame in each array. Readers accept the old `pick_place_meta/` layout (schema 1, single object) so
existing datasets keep loading; `s2r migrate` rewrites a dataset in place to schema 2.

### 4.7 Eval set: `tasks/<task>/eval_sets/*.json` schema 2

Schema 1 plus: `task`, `arm`, `joint_names`, and per start an `objects` map (`{name: {position, orientation_wxyz}}`)
in place of top-level `x`, `y`, `yaw`. `arm_start_state` keeps raw LeRobot units with the saved `joint_mapping`.
`eval_set_id` is a content hash, so converting the three frozen sets changes it: the converter stores the old value as
`legacy_eval_set_id`, and `compare` treats results as paired when either id matches.

### 4.8 Policy

`policy.toml` is unchanged. Two additions to the client contract in `core/policy_client/contract.py`:

```python
class ClientPolicy(ABC):
    def configure(self, arm: ArmInfo) -> None:
        """Optional. Joint names, roles and the raw<->degree mapping of the arm in use."""
```

`ArmInfo` replaces the `policy_name == "molmoact2"` branches in `remote_policy.py`; `molmo_joint_options` moves into
`policies/molmoact2/client.py`. `[action]` gains `dim = "arm"` (default: the arm's joint count) so a plugin can declare
a fixed embodiment and be refused for a mismatching arm with a clear error.

### 4.9 Configs and commands

One YAML per run under `configs/<command>/`, all starting with the same four selectors:

```yaml
arm: so101
scene: rpd_meeting_room
task: pick_place_cube
variant: teleop
```

| Command | Program (`python -m` inside the container) | Config folder |
| --- | --- | --- |
| `s2r record` | `sim.teleop.agent` | `configs/record/` |
| `s2r datagen` (new) | `sim.datagen.scripted` / `mimic` / `dr_replay` (chosen by `source:`) | `configs/datagen/` |
| `s2r prepare` / `train` / `verify` | policy `train/` scripts | `configs/train/` |
| `s2r serve` | `policy_server.launcher` | - |
| `s2r eval` | `sim.eval.run` | `configs/eval/` |
| `s2r rollout` | `core.rollout.real_rollout` | `configs/rollout/` |
| `s2r new arm\|scene\|task\|policy <name>` (new) | `cli.scaffold` (host) | - |
| `s2r check [arm\|scene\|task\|policy] [<name>]` (new) | `cli.check` (static, host) then `sim.check` (headless spawn, container) | - |
| `s2r migrate <dataset>` (new) | `core.datasets.task_meta` | - |

Unchanged: `spawn`, `envs`, `eval-set`, `calibrate`, `calibration-check`, `calibration-stats`, `calibration-report`,
`calibration-replay`, `build`, `pull`, `sync`, `tunnel`, `policies`. `dr-replay` becomes `s2r datagen` with
`source: dr_replay`.

`pyproject.toml [project.scripts]` keeps only `s2r`; `SIM_PROGRAMS` maps commands to module paths.

### 4.10 Environment variables

| Old | New |
| --- | --- |
| `SO101_POLICY`, `SO101_POLICY_DIR`, `SO101_DATASET`, `SO101_OUTPUT`, `SO101_BASE_MODEL` | `S2R_POLICY`, `S2R_POLICY_DIR`, `S2R_DATASET`, `S2R_OUTPUT`, `S2R_BASE_MODEL` |
| `SO101_PYTHON`, `SO101_PROJECT_DIR`, `SO101_ROLE`, `SO101_HOST_UID`, `SO101_HOST_GID` | `S2R_PYTHON`, `S2R_PROJECT_DIR`, `S2R_ROLE`, `S2R_HOST_UID`, `S2R_HOST_GID` |
| `SO101_FOLLOWER_CALIBRATION`, `SO101_LEADER_CALIBRATION` | `S2R_FOLLOWER_CALIBRATION`, `S2R_LEADER_CALIBRATION` |
| `PICK_PLACE_EVAL_STARTS` | `S2R_EVAL_STARTS` |
| `ROOM_USD_PATH` | `S2R_ROOM_USD` |

Touches `cli/paths.py`, `docker/policy/Dockerfile`, `docker/policy/entrypoint.sh`, `joint_calibration.py`, the task
`mdp.py` / `eval.py`, the room env cfg, and every script under `policies/*/train/`. No compatibility aliases.

---

## 5. File-by-file changes

Actions: **keep**, **edit**, **move** (new path given), **split**, **delete**, **new**.

### 5.1 Data folders

| Path | Action | Detail |
| --- | --- | --- |
| `arms/so101/arm.toml` | edit | Rewrite to 4.1. Runtime limits come from `real_setup.json::runtime_joint_limits`. |
| `arms/so101/so101_utils.py` | new | 4.2. |
| `arms/so101/calibration/units/<id>.json` | new | From `real_setup.json::sim_joint_mapping` + `sim_joint_mapping_source`. |
| `arms/_template/` | new | Commented `arm.toml`, empty `usd/`, stub `_utils.py`. |
| `scenes/rpd_meeting_room/real_setup.json` | split, then delete | -> `scene.toml` (robot mount, cameras, lighting, viewer), `appearance.json`, `mounts/so101.toml`, arm files. |
| `scenes/rpd_meeting_room/scene.toml` | edit | Rewrite to 4.3; drop `env_cfg`, `setup`, `room_usd_runtime`. |
| `scenes/rpd_meeting_room/scene.py` | new, only if needed | Anything the builder cannot express (table photo projection is the likely candidate). |
| `scenes/blank_table/` | new | Flat table, one room camera, default light; no measurements. Default scene for new users. |
| `scenes/_template/` | new | |
| `tasks/pick_place_cube/task.toml` | edit | 4.5: `stages`, `tracked_objects`, object `frame`/`position`, `spawnable`, `[task.start]`. |
| `tasks/pick_place_cube/__init__.py` | edit | Implement `signals`; drop `STAGES`/`FAILURES` constants and `_inside_convex` (moves to core geometry). |
| `tasks/pick_place_cube/env_cfg.py` | edit | Remove `_TABLE_*` constants and `_box_position` (anchors do it); contact sensor bodies from `arm.gripper.contact_bodies`; no import of `rpd_meeting_room_env_cfg`. |
| `tasks/pick_place_cube/mdp.py` | edit | Keep grasp/place terms and colour DR. `reset_cube_from_recorded_starts` and `check_pick_place_eval_event_order` become the generic `reset_from_starts` in `sim/envs/mdp/starts.py`. Light DR moves to the scene (no `Room/Lighting/Tubelight` path in a task). |
| `tasks/pick_place_cube/eval.py` | split, then delete | Geometry -> `core/benchmark/geometry.py`; `RecordedStarts`, `load_starts`, `load_robot_starts` -> `core/benchmark/starts.py`; `results_report` -> `core/benchmark/report.py`; `dataset_state_to_radians` -> `core/arms/mapping.py`. |
| `tasks/pick_place_cube/benchmark.py` | split, then delete | `generate_eval_set`, `validate_eval_set`, `EvalSetStarts` -> `core/benchmark/eval_set.py`; `wilson_interval`, `mcnemar` -> `stats.py`; `summarize`, `benchmark_report`, `save_heatmap` -> `report.py`; `StageTrace` -> `core/tasks/stages.py`. Task-specific validity (`clear_of_box`) becomes `Task.start_is_valid`. |
| `tasks/pick_place_cube/eval_sets/*.json` | edit | Convert to schema 2 (4.7). |
| `tasks/pick_place_cube/expert.py` | new | Phase 5. |
| `tasks/_template/` | new | |
| `configs/teleop/*.yaml` | move | -> `configs/record/`; calibration path becomes `follower.unit: <id>`. |
| `configs/eval/*.yaml` | edit | `--smoke-test` options no longer require `molmoact2`. |
| `configs/datagen/`, `configs/rollout/` | new | One example each. |
| `policies/groot_n17/train/*`, `policies/molmoact2/train/*` | edit | `SO101_*` -> `S2R_*`. The pick-place-specific preparation scripts stay as they are for now (see 7). |
| `policies/molmoact2/client.py` | edit | Receives `molmo_joint_options`; implements `configure`. |
| `policies/_template/client.py`, `example/client.py`, `groot_n17/client.py` | edit | Use `DefaultClientPolicy`. |

### 5.2 `src/cli`

| Path | Action | Detail |
| --- | --- | --- |
| `cli/paths.py` | move | -> `cli/plan.py`. `SIM_PROGRAMS` maps to modules; `plan_sim` runs `python -m <module>`; env var renames; `SYNC_PATHS` adds `arms`, `scenes`. |
| `cli/main.py` | edit | Add `new`, `check`, `datagen`, `migrate`; import `TRAIN_COMMANDS` from discovery. |
| `cli/scaffold.py` | new | Copies a `_template/`, renames, validates the name. Absorbs `core/scripts/new_policy.py`. |
| `cli/check.py` | new | Static checks per plugin type: manifest parses, referenced files exist, names match folders, joints consistent. |

### 5.3 `src/core`

| Path | Action | Detail |
| --- | --- | --- |
| `core/paths.py` | keep | |
| `core/scenes.py` | split | -> `core/scenes/spec.py` (typed `SceneSpec`, `MountSpec`) and `core/scenes/anchors.py` (frame maths, Isaac-free). `load_setup` is removed. |
| `core/arms/spec.py` | edit | Typed `ArmSpec` with `joints`, `gripper`, `wrist_roll`, `cameras`; validation with clear errors; `load_arm_utils()`. |
| `core/arms/joint_limits.py` | delete | Limits and name pairs come from `ArmSpec`. |
| `core/arms/mapping.py` | edit | Built from `ArmSpec` + a unit mapping; per-joint unit (deg or m); gripper mask from `role`; no `[:5]`, no `"gripper"` literal; hosts the shared pure conversion functions. |
| `core/arms/joint_calibration.py` | move | -> `core/arms/calibration.py`. Reads unit files instead of `real_setup.json`; `calibration = "factory"` path; roll and gripper anchors optional; no import-time constants. |
| `core/tasks/base.py` | edit | 4.5. |
| `core/tasks/loader.py` | keep | |
| `core/tasks/stages.py` | new | Stage latch, failure-mode fallback, progress label. |
| `core/results/compare.py`, `eval_resume.py`, `eval_smoke.py` | move | -> `core/benchmark/{compare,resume,smoke}.py`. |
| `core/scripts/make_eval_set.py`, `summarize_sweep.py` | move | -> `core/benchmark/{eval_set,sweep}.py` entry points. |
| `core/scripts/list_envs.py`, `spawn.py` | move | -> `cli/` (they are thin). |
| `core/scripts/real_rollout.py` | move | -> `core/rollout/real_rollout.py`; uses `ArmSpec` (no roll assumption), reads `configs/rollout/`. |
| `core/scripts/new_policy.py` | delete | Replaced by `cli/scaffold.py`. |
| `core/utils/recording_alignment.py` | move | -> `core/datasets/alignment.py`; joint count from the arm. |
| `core/utils/hashing.py` | keep | |
| `core/utils/provenance.py` | new | Git identity. |
| `core/datasets/task_meta.py` | new | Sidecar read/write/validate/migrate (from `sim/recording/episode_metadata.py`, which is Isaac-free already). |
| `core/datasets/sources.py` | new | From `sim/recording/replay_source.py` (pure Python). |
| `core/calibration/*` | move/rename | `calibrate_arm.py` -> `calibrate.py`; `check_calibration.py` -> `check.py`; `calibration_stats.py` -> `stats.py`; `calibration_report.py` -> `report.py`; `calibration_check.py` -> `gap.py`; `joint_angle_check.py` -> `telemetry.py`; `joint_angle_display.py` -> `display.py`. Replace `"gripper"` literals with the arm's gripper role. |
| `core/policy_client/contract.py` | edit | `configure(arm)`. |
| `core/policy_client/default.py` | new | `DefaultClientPolicy`. |
| `core/policy_client/remote_policy.py` | edit | Remove `MOLMO` branches; pass `ArmInfo`. |
| `core/policy_client/grasp.py`, `smoothing.py` | edit | Gripper index from `ArmInfo`, not the key `"gripper.pos"`; default latch dataset path removed (must be configured). |
| `core/policy_client/adapter.py`, `discovery.py`, `transport.py` | keep | Apart from the deletions in 2.1. |
| `core/policy_server/*` | keep | Plus `seeding.py`. |

### 5.4 `src/sim`

| Path | Action | Detail |
| --- | --- | --- |
| `sim/app.py` | new | `rendering_kit_args`, `apply_selection`, a single `launch()` used by every entry point. |
| `sim/envs/__init__.py` | edit | Scene entry is `scene_builder.build`; registers variants as today. |
| `sim/envs/scene_builder.py` | new | `scene.toml` + `mounts/<arm>.toml` + `ArmSpec` -> env cfg: room, cameras (fixed and arm-mounted), lighting, physics, robot pose, then the optional `scene.py` hook. |
| `sim/envs/base_env_cfg.py` | edit | Joint names from `ArmSpec`. |
| `sim/envs/rpd_meeting_room_env_cfg.py` | delete | Generic parts go to `scene_builder.py` and `sim/scene/cameras.py`; `drop_unused_camera_outputs` becomes generic (any camera, by data type). |
| `sim/envs/mdp/obs.py` | keep | |
| `sim/envs/mdp/resets.py` | split | `_common.py` (`as_torch`, `env_indices`, `_wxyz`, `_xyzw`, `_env_prim_path`), `placement.py` (`reset_object_pose`, `park_object`), `cameras.py` (setters), `appearance.py` (robot colour; material from `arm.appearance`), `limits.py`. |
| `sim/envs/mdp/starts.py` | new | Generic `reset_from_starts` for any tracked objects. |
| `sim/scene/arm.py` | edit | Calls `prepare_collisions` from the arm's utils; friction bodies from `arm.gripper`. |
| `sim/scene/contact_materials.py` | edit | Keep `contact_material`, `contact_properties`, `enable_collision_meshes`, `bind_contact_material`; SO-101 splitting leaves. |
| `sim/scene/gripper_collisions.py` | move | -> `arms/so101/so101_utils.py`. |
| `sim/scene/gripper_geometry.py` | move | -> `sim/scene/geometry.py`; tip mesh from `arm.gripper.tip_mesh`; `table_plane` takes the anchor's `prim_name` and caches. |
| `sim/scene/real_setup.py` | split, then delete | Room lighting/appearance -> `sim/scene/appearance.py`; `apply_opencv_intrinsics` -> `sim/scene/cameras.py`; robot appearance -> arm utils; module-level `SETUP` disappears. |
| `sim/scene/calibrated_camera.py`, `calibrated_renderer.py`, `calibrated_viewport.py`, `calibrated_viewport_cfg.py` | move | -> `cameras.py`, `renderer.py`, `viewport.py` (cfg merged). |
| `sim/scene/camera_response.py` | keep | |
| `sim/teleop/lerobot_agent.py` | split | -> `agent.py` (setup, teardown) and `loop.py` (tick + request state machine). Roll handling only when `arm.wrist_roll` exists; tracks every `tracked_objects` entry. |
| `sim/teleop/config.py` | edit | `follower.unit`; uses `sim/app.py`. |
| `sim/teleop/devices.py` | edit | Import `arm.lerobot.modules` instead of hard-coding the SO modules. |
| `sim/teleop/keyboard.py`, `terminal_control.py` | move | -> `sim/teleop/controls/`. |
| `sim/teleop/profiling.py` | move | -> `tools/diagnostics/` support module. |
| `sim/teleop/rerun_logging.py` | keep | |
| `sim/recording/episode_metadata.py` | move | -> `core/datasets/task_meta.py`. |
| `sim/recording/replay_source.py` | move | -> `core/datasets/sources.py`. |
| `sim/recording/replay_scene.py` | edit | Camera names from the scene, not `camera_realsense_rgb` / `camera_wrist_cam`. |
| `sim/recording/lerobot_recorder.py`, `recording_dataset.py`, `recording_setup.py`, `rerecording.py` | keep | Feature names from `ArmSpec`. |
| `sim/recording/web_gui.py` | keep | Out of scope. |
| `sim/eval/lerobot_eval.py` | split | -> `run.py`, `loop.py`, `report.py`. Remove the `spawn_cube` / `__name__` detection, `_pick_place_*` attributes and `molmoact2` conditions. |
| `sim/eval/pick_place_benchmark_runtime.py` | move | -> `stage_tracking.py`, driven by `Task.signals` / `observe`. |
| `sim/eval/grasp_diagnostics.py` | move | -> `tools/diagnostics/` (SO-101 mesh names), or generalise through `arm.gripper`. |
| `sim/eval/eval_video.py` | move | -> `video.py`; default rename map removed. |
| `sim/eval/sim_remote_policy.py` | keep | |
| `sim/generate/` | move | -> `sim/datagen/`; `object_spawner.py` -> `spawner.py`. |
| `sim/datagen/ik.py`, `scripted.py`, `mimic.py` | new | Phase 5. |
| `sim/check.py` | new | Headless: build the env, reset, step, render each camera, assert joint and body names resolve. |

### 5.5 Tests, tools, docker, docs

| Path | Action | Detail |
| --- | --- | --- |
| `tests/policy_client/` | move | -> `tests/core/policy_client/`. |
| `tests/sim/test_*.py` | keep | Update imports. |
| `tests/sim/{check_,audit_,replay_,render_,profile_,summarize_,grasp_collision_}*.py` | move | -> `tools/diagnostics/`. |
| `tests/core/test_arm_spec.py`, `test_scene_spec.py`, `test_task_spec.py`, `test_mapping.py`, `test_task_meta.py`, `test_eval_set.py` | new | Include a synthetic 4-joint + parallel-gripper arm so layout assumptions are tested without hardware. |
| `tests/cli/test_plan.py` | new | The planners were written to be asserted exactly; nothing asserts them today. |
| `tests/policies/test_conformance.py` | new | Every enabled plugin: manifest valid, `create()` returns the right type, request/response round-trips. |
| `tests/_fakes.py` | keep | |
| `docker/sim/Dockerfile`, `entrypoint.sh` | edit | Copy `arms/ scenes/ tasks/` are bind-mounted already; only the module-based programs change. |
| `docker/policy/Dockerfile`, `entrypoint.sh` | edit | Env var renames. |
| `docs/` | new | `01-getting-started`, `02-add-an-arm`, `03-add-a-scene`, `04-add-a-task`, `05-training`, `06-evaluation`, `07-policy-plugins`, `08-data-generation`. |
| `README.md`, `CLAUDE.md`, `policies/README.md` | edit | After each phase. |

---

## 6. Phases and acceptance checks

Each phase ends green on `PYTHONPATH=src python -m pytest tests -q` and on the listed sim check, on the SO-101 in the
meeting room, before the next begins.

| Phase | Work | Acceptance |
| --- | --- | --- |
| 0 Cleanup | Section 2. | CPU tests pass with no `--ignore`. `s2r --dry-run` output unchanged for record, eval, serve, train. |
| 1 Arm contract | 4.1, 4.2; `ArmSpec`, mapping, calibration units; SO-101 code into `so101_utils.py`. `real_setup.json` still read for scene data. | Record one sim-only episode and run the smoke eval; joint targets identical to before for a replayed episode. Synthetic 4-joint arm passes the core tests. |
| 2 Scenes | 4.3, 4.4; `scene_builder.py`; split `real_setup.json`; add `blank_table`. | Meeting-room camera images match a reference frame captured before the change. `S2R_SCENE=blank_table s2r record` works with the SO-101 and no mount file. |
| 3 Task engine | 4.5 to 4.7; generic stages, starts, `task_meta`, eval set schema 2; add a second task (stack two cubes). | Re-running a previous results JSON's schedule gives the same per-episode stages. Old datasets load. The second task records, builds an eval set and evaluates without edits under `src/`. |
| 4 Entry points and CLI | Split agent and eval; `sim/app.py`; `s2r new`, `s2r check`, `s2r migrate`; env var and command renames; policy `configure`. | `s2r check` passes for every arm, scene, task and policy. `tests/cli` asserts every planner. Both policies serve and evaluate. |
| 5 Data generation | `sim/datagen/ik.py`, `scripted.py`, task `expert.py`; one rollout -> success filter -> recorder path shared with `calibration_replay`; then `mimic.py`. | `s2r datagen` produces a LeRobot dataset for pick-place whose episodes all pass the task's success check, and a policy can be prepared and trained on it. |
| 6 Second arm | `arms/omx/` with `omx_utils.py`, a mount file, `calibration = "factory"`. | Sim-only record and eval on `blank_table`, then paired recording on hardware. No edits under `src/`; any that prove necessary are fixed generically and listed. |
| 7 Docs and conformance | `docs/`, templates, conformance tests. | A new arm, scene, task and policy can each be added from the docs alone. |

Phase 6 is the test of the whole plan: if adding the arm needs a change under `src/`, the contract was wrong.

---

## 7. Risks and open questions

- **Not verified in the simulator.** The blockers and the file moves come from reading the code. Files read only at
  outline level (signatures and docstrings): calibration tooling, `lerobot_recorder.py`, `rerecording.py`,
  `dr_replay.py`, `calibration_replay.py`, most of `sim/scene/`, both policy servers and their training scripts.
  Their rows in section 5 should be re-checked when each phase reaches them.
- **Camera look regression.** Moving the fitted lighting, materials and response out of `real_setup.py` risks subtle
  image changes that affect policies. Phase 2 needs a pixel-level comparison against frames captured beforehand.
- **Reset event order.** Eval correctness depends on reset events running in a fixed order, asserted today by
  `check_pick_place_eval_event_order`. The generic version must keep an equivalent assertion.
- **Frozen eval sets.** Converting to schema 2 changes `eval_set_id`. Old results stay comparable only through
  `legacy_eval_set_id`; confirm that is acceptable before converting.
- **Isaac Lab Mimic on Newton.** Whether Mimic runs on the Newton backend in 3.0 is unconfirmed. The scripted IK
  expert does not depend on it and comes first.
- **IK on low-DOF arms.** A 5-joint arm cannot reach arbitrary orientations; the IK layer solves position plus the
  orientation axes the arm has, declared per arm.
- **Policy training is still task-specific.** `prepare_pick_place_gr00t.py` and the MolmoAct2 LoRA scripts assume
  pick-place and a 6-value SO-101 action. Making training arm- and task-agnostic is not covered here and needs its own
  pass once phase 3 fixes the dataset side.
- **Open decision.** Git LFS for `arms/` (46 MB) and `scenes/` (109 MB).
