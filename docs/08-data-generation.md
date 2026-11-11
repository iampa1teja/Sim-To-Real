# Data generation

Scripted generation runs a task-folder expert through physical IK, and Mimic
retargets successful annotated demonstrations. Neither derives a manipulation
strategy from an English instruction alone. A new task supplies its objects,
success/stage logic and, for scripted generation, `expert.py`.

## Generate pick-place or stacking

Start `s2r-sim` as described in [getting started](01-getting-started.md). These
examples use the selected SO-101 unit and meeting-room scene. Destinations must
not already exist; flags override YAML defaults.

```bash
./s2r datagen --config_path configs/datagen/scripted.yaml \
  --seed 101 --attempts 1 --episodes 1 \
  --destination /workspace/datasets/pick_place_scripted \
  --report /workspace/s2r/outputs/datagen/scripted-report.json

./s2r datagen --config_path configs/datagen/stack_two_cubes.yaml \
  --attempts 1 --episodes 1 \
  --destination /workspace/datasets/stack_two_cubes_scripted \
  --report /workspace/s2r/outputs/datagen/stack-report.json
```

The stacking config uses seed 182482. These demonstrated starts passed the
physical checks; grasp reliability across the entire random distribution is not
established. `attempts` is the maximum number of tries, `episodes` is the accepted
target, and `episodes: 0` runs all attempts. `max_steps` bounds each rollout.
`warmup_steps` settles the arm before tracking begins. An episode is saved only
when the task's success stage has latched **and the final state is successful**.
Failures remain in the report, not as successful training episodes.

The pick-place expert keeps one requested orientation through transport. It does
not deliberately roll the wrist after pickup. Its short transport arc avoids the
arm base; its release is intentional, followed by a settled success check.

## Add a task expert

The [task guide](04-add-a-task.md) defines `Task` and object contracts. Add
`tasks/<task>/expert.py` exporting `create(task)`. The returned object implements
`plan(context)` and returns an iterable of `core.tasks.expert.Waypoint` values.
Use [pick-place](../tasks/pick_place_cube/expert.py) and
[stacking](../tasks/stack_two_cubes/expert.py) as complete physical examples.

`ExpertContext` contains:

| Field | Meaning |
| --- | --- |
| `state` | Initial `StepState`, with named object poses, EE pose and signals |
| `control_pose` | Actual initial TCP pose in world metres / XYZW |
| `table_normal` | World-frame fitted table normal |
| `arm_base` | Actual declared base-link pose in the world |
| `options` | The run's task-specific `expert_options` dictionary |

A minimal API illustration below holds the current TCP open. It demonstrates the
interface; it will not satisfy a manipulation task's success conditions. Replace
its plan with approach, grasp, lift, transport, release and settle poses suitable
for your task, including collision clearance and reachable orientation axes.

```python
from core.tasks.expert import Waypoint

class Expert:
    def plan(self, context):
        return (Waypoint(
            name="hold", pose=context.control_pose, gripper_open_fraction=1.0,
            duration_s=1.0, tolerance_m=0.004, timeout_s=4.0,
        ),)

def create(task):
    return Expert()
```

Waypoint positions are world metres; orientations are XYZW. The waypoint targets
the arm's declared TCP, not necessarily its EE link origin. Durations, tolerances
and timeouts must be positive, with timeout at least duration. Gripper fractions
are in `[0,1]`, interpolated through each gripper joint's physical codec endpoints.
The engine derives gripper indices from roles and records absolute device-unit
joint commands in manifest order. It applies no teleop wrist rezeroing.

Declare `[arm.ik] position_offset_m` in the EE link frame and `orientation_axes`
as supported arm-base rotation-error components. A low-DOF arm cannot enforce
arbitrary orientation. Isaac Lab `DifferentialIKController` supplies every step;
Newton FK supplies command poses. The controller uses the articulation Jacobian
and bounded load-error compensation. Failed reachability/grasping must be fixed in the expert,
TCP/arm contract or real calibration; do not relax task success or change physics
to label failures as successes.

## Inspect the output

The report records arm/task/scene, calibration and mapping provenance, seed,
controller settings, attempt outcomes, stages, step counts, final signals and
accepted episode indices. Check both `complete` and `accepted_episodes`: a partial
run may contain some valid episodes while failing the requested count.

Saved observations, images and object poses describe the pre-action frame; task
success is observed after executing that frame's action. Every tracked object has
one position and WXYZ orientation per frame in `task_meta/episode_*.json` (schema
2, robot-base frame, metres). `datagen_meta/episode_*.json` adds TCP poses, named
object poses and subtask stage annotations for Mimic. Its target pose is FK of the
executed joint command; nominal requested poses are recorded separately.

Review the two camera videos as well as task results. Run the selected policy's
`s2r prepare` and `s2r verify` workflow before training; see
[training](05-training.md). Format validation alone does not prove physical task
success, and a short training check does not prove a learned policy succeeds.

## Multiply demonstrations with Mimic

The source must contain successful aligned actions and `datagen_meta` annotations
for the same task/arm and TCP offset. Ordinary teleop data without these annotations
cannot be passed directly to this adapter. The current recorder runs one
environment. Task manifests declare `[[task.mimic.subtasks]]` with named object
references, unique stage termination signals, and a final subtask without a
termination signal. The installed Isaac Lab `DataGenerator` handles source
selection, rigid retargeting, interpolation and sequencing. See the complete
[pick-place declaration](../tasks/pick_place_cube/task.toml).

```bash
./s2r datagen --config_path configs/datagen/mimic.yaml \
  --source_dataset /workspace/datasets/pick_place_scripted \
  --destination /workspace/datasets/pick_place_mimic \
  --report /workspace/s2r/outputs/datagen/mimic-report.json \
  --seed 24716 --attempts 1 --episodes 1
```

This nearby start was validated against the seed-101 source above. It changed the
cube start by about 4.42 mm and 0.25 degrees and produced a 402-frame physical
success, including SDK interpolation frames. A different nearby start failed
placement and was discarded. This demonstrates retargeting, not broad coverage.
Use additional source demonstrations for larger workspace/orientation changes.

Mimic recovers source command poses through a bounded local kinematic inverse;
it does not add the scripted controller's load compensation a second time.
`--diagnostic_trace` writes attempted trajectories beside the report, outside
the dataset, including failures. Keep those diagnostics separate from accepted
training data. It does not change physics or call task signals an extra time.

## Physical calibration replay and visual replay

`s2r calibration-replay --help` describes physical replay. It requires an audit
JSON containing `source_hashes` (the production `core.utils.hashing.tree_sha256`
mapping), `old_mapping` (the source's saved per-joint physical mapping), and
optionally `expected_source_episodes`. Capture that audit before calibration
changes and keep it outside the source tree. The replay checks exact source
hashes and agreement with embedded source calibration, restores only initial
poses, then executes the original device commands under the selected target unit.

For the generated source above, this host-side metadata snippet constructs the
audit from its saved report and hashes every source file. For a legacy recording,
obtain `old_mapping` from that recording's original calibration evidence instead.

```python
import hashlib
import json
from pathlib import Path

source = Path("datasets/pick_place_scripted")
report = json.loads((source / "meta/datagen.json").read_text())
hashes = {}
for path in sorted(source.glob("**/*")):
    if path.is_file():
        with path.open("rb") as stream:
            hashes[path.relative_to(source).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
audit = dict(source_hashes=hashes, old_mapping=report["joint_mapping"],
             expected_source_episodes=report["accepted_episodes"])
destination = Path("outputs/source-calibration-audit.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
```

```bash
./s2r calibration-replay \
  --source /workspace/datasets/pick_place_scripted \
  --destination /workspace/datasets/pick_place_recalibrated \
  --audit /workspace/s2r/outputs/source-calibration-audit.json \
  --report /workspace/s2r/outputs/calibration-replay.json \
  --arm so101 --scene rpd_meeting_room --task pick_place_cube \
  --unit rpd_robot --expected_source_episodes 1 --seed 101 --headless
```

The audit path above is an input created from your actual unchanged source.
Do not invent or replace its mapping with the
target calibration. Physical replay uses the same success-filtered recorder as
scripted and Mimic generation.

`s2r datagen --source dr_replay` instead re-renders recorded joint/object poses
kinematically, preserving actions. Its shipped color/language options implement
the pick-place augmentation recipe; it does not prove new physical successes.
Inspect a plan with `--dry_run` and select the task's supported appearance terms.
This differs from the global `s2r --dry-run ...`, which only prints the Docker
command. `configs/datagen/` contains the available source examples.


## Imported arms and physical gripper openings

Set `S2R_ARM` to select an imported arm when reusing a generation recipe; an explicit
`--arm` flag takes precedence over that selector. The existing SO-101 fractions and
recipe values are unchanged. Simulation-only manifests encode actions from their
declared physical limits, without a hardware calibration unit.

For a gripper with measured gap fields, experts open to the smaller of its maximum
gap and object width plus 20 mm, and close toward zero. Add
`gripper_open_gap_m` / `gripper_closed_gap_m` to `expert_options` to override those
metre values. Physical openings take precedence over the fraction options. Arms
without measured gaps retain the existing fraction defaults. Jaw arms may supply
an optional `gripper_q_for_gap(gap_m)` hook in their arm utility file; without that
hook the fraction codec remains the fallback.

Each waypoint declares exactly one of `gripper_open_fraction` or
`gripper_open_gap_m`. Mimic followers are resolved at the articulation boundary and
never appear in saved actions, states or policy contracts. For arms declaring `z`
orientation control and a measured closing axis, the experts align that axis with
the nearest cube face about the table normal, then retain the pickup orientation
through transport.

Live importer acceptance tests run in isolated temporary repositories:

```bash
docker exec -e S2R_RUN_IMPORT_TESTS=1 -w /workspace/s2r s2r-sim \
  s2r-entrypoint python -m pytest tests/sim/test_urdf_import.py \
  tests/sim/test_mimic_joints.py tests/sim/test_parallel_gripper_geometry.py -q
```

Run those tests on an idle GPU. The ordinary CPU suite skips the live imports but
checks parsing, packages, manifests, codecs, expert options and command planning.

For yaw-controlled stacking, object-centre goals retain the measured TCP-to-object
translation acquired after the lift. This compensates small offsets introduced by
contact while preserving the pickup orientation. The existing fraction-based,
yaw-free reference path is unchanged.
