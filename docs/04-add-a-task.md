# Add a task

[Guide index](../README.md) · [Add a scene](03-add-a-scene.md) · [Evaluation](06-evaluation.md)

A task folder owns its objects, simulator composition, start distribution, sensor measurements,
milestones and success/failure rules. The recorder, evaluator and stage engine operate through that
contract. Add a folder without adding a task-name branch to `src/`.

Run commands from the repository root after [getting started](01-getting-started.md).

## Create a complete two-object example

```bash
./s2r new task guide_task
./s2r check --static --arm so101 --scene blank_table --task guide_task
```

The [task template](../tasks/_template/task.toml) is a complete two-cube stacking task. The scaffold
renames its manifest and `TASK` loader. It supplies all four required files:

```text
tasks/guide_task/
  task.toml
  __init__.py
  env_cfg.py
  mdp.py
```

`__init__.py` imports no simulator packages; it exports `TASK`, an instance of a `core.tasks.base.Task`
subclass. `env_cfg.py` and `mdp.py` may import Isaac Lab and Torch. Keep their imports lazy from the pure
task class so static tools and benchmark generation can load the task on a CPU host.

Inspect sampling and the pure observation contract:

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
import numpy as np
from core.tasks.base import Pose, StepState
from core.tasks.loader import load_task
from core.tasks.stages import StageTrace

task = load_task("guide_task")
objects = task.sample_start(np.random.default_rng(7))
assert set(objects) == set(task.spec.tracked_objects) == {"cube_a", "cube_b"}
state = StepState(objects, Pose((1.0, 0.0, 0.5)), 0.0, {})
trace = StageTrace(task)
flags = trace.observe(state)
assert set(flags) == set(task.stages)
assert not task.is_success(trace.stages)
print(task.name, task.stages, objects, trace.finish())
PY
```

This is a CPU contract check using synthetic poses, not evidence that an episode succeeded.

## Declare objects, stages and starts

`[task].name` equals the folder name. `instruction` is the default language instruction. `stages` is
an ordered nonempty list; its last stage defines success. `tracked_objects` selects the object trajectories
written to datasets. Every tracked name must exist in `[task.objects]`. Static objects can remain observable
without a trajectory: pick-place tracks the cube but also observes the box.

Each `[task.objects.<name>]` supplies either `size` for a primitive or a relative `usd` path. It can also
declare `position`, XYZ `rotation_euler_degrees`, `rigid` and task-owned parameters such as mass/color.
The task's `env_cfg.py` consumes these declarations and creates the corresponding scene entities; the
manifest parser does not spawn arbitrary object types by itself. When changing a primitive into a USD
asset, update that task-folder composition code too. Preserve useful size/footprint information for starts
and benchmark validation; bottom-origin assets can declare `bottom_origin = true`.

`frame` is a task-owned object parameter consumed by the composition code. The provided tasks support
`world`, `table`, `arm_base`, or another named scene anchor through `core.scenes.transform_pose`. All
positions are metres. Pass the actual arm mount pose for `arm_base` placements. Put room geometry and
mount corrections in the scene rather than embedding measured room coordinates in the task.

Set `spawnable = true` only for objects that should be parked and placed through the recorder's spawn
workflow. Newton imports rigid bodies once; declare the object before startup, then reposition it.
The template parks `cube_a` in teleop mode while leaving `cube_b` as the target. Task `reset_target`
restores that target at each teleop reset.

`[task.start]` declares the primary `object`, `frame`, increasing `xy_min`/`xy_max`, and `yaw_deg` range.
The default `sample_start(rng)` samples its primary object and returns `{name: Pose}` in that declared
frame. Override it for multiobject sampling, as the template does to maintain cube separation. Return
all tracked objects needed for a reset. Use only the supplied RNG to preserve seed reproducibility.

`sample_start_in_base(task, rng, scene)` converts the plugin result to the arm-base frame. Callers then
use `start_is_valid(start, scene)`: the default checks table support/overhang and object overlap using
captured live geometry. Override it for additional clearances or constraints. Record any task-specific
sampling assumptions, and keep source poses within the actual chosen scene's support surface. Random
starts are placed upright on the table unless an object's `rest_on_table = false` preserves its 3D pose.

## Compose variants and order resets

`[task.variants]` maps names to task-local `module:function` entries. The template exports
`env_cfg:apply_teleop` and `env_cfg:apply_eval`. Each receives `(cfg, task)` and modifies the generic
scene config. The registry creates `S2R-<task>-<variant>` automatically. `default_variant` must name a
declared variant; set selectors before importing the simulator registry.

The template's composition uses `cfg.arm_spec` and `cfg.scene_spec`, declared contact bodies and shared
contact materials. It explicitly rejects an arm without the gripper contacts it needs. For a new task,
declare only genuine compatibility requirements; obtain links, joints and roles from the selected arm.

For evaluation, use the shared `sim.envs.mdp.starts.reset_from_starts` event. It supports recorded,
fixed-eval-set and task-distribution starts without knowing object names. Required reset order is:

1. `reset_robot_position` restores robot state.
2. `reset_from_starts` restores the selected object/robot start.
3. Task `reset_tracking` clears contact sensors, confirmation counters and grasp history.

Register `assert_reset_event_order` as a startup event as the template does. A task-specific target-reset
event must not overwrite a scheduled start later in the reset. Material randomization must precede color
randomization when both are used. Configure episode decimation, render interval, timeout and success
termination in the task variant; the template has complete implementations to adapt.

## Implement measurements and pure stage rules

The required task methods have these responsibilities:

| Method | Contract |
| --- | --- |
| `signals(env)` | Read live sensors/geometry; return named values, normally one value per environment. Simulator imports stay in task-folder helpers. |
| `observe(state)` | Pure conversion of one `StepState` into `{stage_name: bool}` for the declared stages. |
| `sample_start(rng)` | Optional override returning named `Pose` objects in `task.start.frame`. |
| `start_is_valid(start, scene)` | Optional extra checks on arm-base poses against captured scene geometry. |
| `failure_mode(flags, state)` | Classify an unsuccessful final state using latched milestone flags; return `None` for success. |
| `episode_metadata(env, index=0)` | Optional report/instruction metadata, such as a randomized color. |

`StepState.objects` contains every declared object, including untracked static objects; `ee` is the
declared EE-link pose; `gripper` is the mean simulator coordinate of declared gripper-role joints, or
zero when absent. Live poses are in world metres with XYZW quaternions. `core.tasks.base.Pose` exposes
`quat_xyzw`; the scene-transform `Pose` exposes `orientation_xyzw`. Convert explicitly when crossing
those APIs. For a grasp-point distance, derive the control point from arm geometry/IK declarations or
an optional arm hook rather than assuming that the EE-link origin is a fingertip.

`StageTrace` latches each declared milestone. The final stage determines task success; flags for earlier
stages remain true after a drop. Failure classification therefore also needs the final live state, such
as whether the object is still held. The evaluator observes the physical terminal frame before automatic
reset. Do not read the next episode's poses to classify the previous episode.

Confirmation counters belong in the task. Make `signals()` idempotent within a `common_step_counter`:
termination checks, stage tracking and diagnostics can read the same physical frame. Reset interrupted
confirmation and contact/grasp history correctly. The template demonstrates contact-based grasp history,
relative-pose stacking geometry, speed limits and consecutive placement confirmation. Keep those tests
physical: elapsed time or reaching a waypoint alone cannot establish a grasp or successful placement.

Declare a `failure_modes` tuple when reports should retain zero-count categories. Keep stage/failure
names within the task; generic engine code must not dispatch on those names or task-specific attributes.

## Record, evaluate and optionally generate demonstrations

With a coordinated simulator/GPU window, first check both variants:

```bash
./s2r check --arm so101 --scene blank_table --task guide_task --variant teleop
./s2r check --arm so101 --scene blank_table --task guide_task --variant eval \
  --scene-json /workspace/s2r/outputs/guide-task-scene.json
```

These commands need no hardware. They establish composition, finite state and camera output, not stacking
success. For a leader-driven demonstration, start from the complete recording YAML and select the task:

```bash
./s2r record --config_path configs/record/sim_only.yaml \
  --arm so101 --scene blank_table --task guide_task --variant teleop \
  --dataset.repo_id local/guide_task --dataset.root /workspace/datasets/guide_task \
  --dataset.task_name "Stack the blue cube on top of the red cube"
```

This config selects SO101 unit `rpd_robot`, opens a physical leader, and leaves the real follower disabled.
Use your verified leader port/ID and matching calibration. Type `start`, perform the demonstration, type
`save`, then `q`; use `s2r spawn --object cube_a` from another terminal when the parked source is needed.
Open the jaw sufficiently before spawning. Terminal `help` shows the active bindings.

The recorder writes one schema-2 `task_meta/episode_<index>.json` trajectory per episode for every tracked
object, with task/arm identity, metre positions and WXYZ orientations aligned to dataset frames. Raw joint
states/actions remain in selected device units and manifest order. Readers retain legacy single-object
`pick_place_meta` support; `s2r migrate` writes explicit schema-2 copies without changing the old sidecars.
Do not manufacture missing second-object poses when converting an old single-object dataset.

Use the captured `guide-task-scene.json` and recorded dataset with the commands in
[evaluation](06-evaluation.md) to generate a fixed set, run a served policy and compare results. Retain
its content hash, seeds and geometry. A hold-state/example policy can exercise the pipeline while timing
out; that does not demonstrate task completion.

Scripted generation is optional. Add `expert.py` exporting `create(task)` whose result has
`plan(ExpertContext) -> sequence[Waypoint]`; all waypoint poses are world metres/XYZW, with a declared
gripper-open fraction and positive timing/tolerances. See the complete
[stack expert](../tasks/stack_two_cubes/expert.py) and [data-generation guide](08-data-generation.md).
No expert is needed for recording or evaluation. A missing optional expert raises a clear error.

Mimic additionally needs task-owned `[[task.mimic.subtasks]]`, valid object references and unique
non-final termination signals from the declared stages; the final subtask has no termination signal.
Its source must contain genuine successful, aligned annotations using the same TCP. See
[pick-place's complete declarations](../tasks/pick_place_cube/task.toml). A new task does not inherit
Mimic semantics automatically from its stage names.

Before accepting a new task, exercise reset order, seed repeatability, stage latching, interrupted
confirmation, failures and successful physical execution. Inspect frame counts and both tracked-object
trajectories in a saved episode. Static checks and short headless steps do not replace those checks,
policy training, or paired hardware validation.
