# Add an arm

[Guide index](../README.md) · [Getting started](01-getting-started.md) · [Add a scene](03-add-a-scene.md)

An arm is a folder under `arms/`. Its manifest owns joint order, units, hardware registration,
calibration selection and optional cameras. CAD-specific Python stays in that folder. Adding an arm
does not require an entry in a source-code registry.

Run commands from the repository root after the [host/container setup](01-getting-started.md).

## Import from URDF

With the simulator container running, import a ROS description package directly:

```bash
./s2r new arm my_arm --urdf /path/to/description/robot.urdf.xacro \
  --ee_link Tool --gripper FingerDrive --camera CameraLink=wrist_cam
S2R_ARM=my_arm ./s2r check
S2R_ARM=my_arm ./s2r datagen --config_path configs/datagen/scripted.yaml \
  --destination /workspace/datasets/my_arm_pick
```

Plain URDF and split xacro are supported. Xacro handles includes, properties and
`ros2_control`; package names resolve through nearby `package.xml` files or explicit
`--package_path pkg=/path/to/pkg` mappings. Mesh origins, scales, arbitrary joint axes,
fixed brackets, uppercase names, and prismatic mimic fingers are preserved. Joint and
link names do not determine their roles: supply the ee link and gripper source joints.
The importer automatically includes each selected source's mimic followers.

The command creates `arms/my_arm/source/`, `usd/my_arm.usd`, `arm.toml`, an optional
hooks file, and `import-report.json`. Source files are copied unchanged and hashed.
The expanded URDF and a resolved converter input are retained separately. Inputs
outside the checkout receive read-only container mounts; repository inputs use the
running container's existing mount. `--dry-run` prints that executable command.

| Flag | Meaning |
| --- | --- |
| `--urdf FILE` | Expanded URDF or xacro entry point. Omit it for the manual template path below. |
| `--ee_link LINK` | Required tool body; its base-to-tool chain defines arm joints. |
| `--gripper JOINT` | Required, repeatable source joints below the ee; followers are added automatically. |
| `--base_link LINK` | Explicit base; otherwise the fixed-root child is proposed for confirmation. |
| `--camera LINK=NAME` | Repeatable fixed wrist camera declarations. |
| `--package_path PKG=DIR` | Repeatable explicit ROS package directories. |
| `--ros2_control FILE` | Additional control XML for hardware placeholders. |
| `--rename URDF_NAME=NAME` | Repeatable public action-name changes; USD names remain the URDF names. |
| `--grasp_depth_m NUMBER` | Distance back from collision fingertips; default 0.010 metres. |
| `--continuous_limit_deg NUMBER` | Symmetric bounds for continuous joints; default ±180 degrees, marked for confirmation. |
| `--force` | Overwrite an existing imported folder only; a manual arm without `[arm.import]` is protected. |
| `--dry-run` | Print the actual host/container plan. |

Read every field in the report before accepting an imported arm. Each generated field
has a value, reason and one of `derived`, `from-flag`, `measured`, `confirm`, or `TODO`.
Measured grasp geometry also requests confirmation. Review the base proposal, contact
bodies, orientation axes, grasp centre, source opening direction, camera optics and
forward yaw. The report includes numeric FK errors, gravity-hold drift, mimic tracking,
workspace coverage/unreachable cells, bilateral cube contact and lift, and camera PNGs.
A simulator-step failure still leaves the folder and its evidence; source/flag failures
publish no arm. Fix the reported cause and repeat with `--force`.

Conversion uses Isaac Lab's URDF converter, with the Isaac Sim extension as fallback.
Named fixed links remain separate. Newton supplies FK, dynamics and the library IK
adapter. A placement failure caused by the imported base intersecting the tabletop may
produce a scene mount fitted from its collision support; that translation is reported
as derived and requires confirmation. Production physics is unchanged.

Mimic followers are physical joints, with source/multiplier/offset metadata, but do not
add action or dataset dimensions. A seven-axis arm with one source and one follower
therefore has eight actions. Converted Newton mimic constraints are preferred; their
followers have passive drives. Manual follower targets cover missing constraints.

Hardware output is a placeholder. Motor IDs and assumed encoder parameters require
confirmation. A prismatic joint without `counts_per_metre` has a transmission TODO
and is excluded from the factory unit. The driver section stays commented out. A
simulation-only arm uses its declared limits for its dataset codec and never reads
these hardware placeholders. Real rollout requires an owner-supplied driver and
calibration.

Closed loops, floating/planar joints, arm-joint mimics and multiple movable end-effector
branches fail with the offending element named. The importer does not promise that an
arbitrary arm can reach an arbitrary task: inspect the reach report and accepted episode
before expanding start coverage or attempting stacking.

## Add an arm manually

### Create and inspect a complete example

```bash
./s2r new arm guide_arm
./s2r check --static --arm guide_arm --scene blank_table --task stack_two_cubes
```

This copies the complete [arm template](../arms/_template/arm.toml), renames its utility file, and
creates this layout:

```text
arms/guide_arm/
  arm.toml
  guide_arm_utils.py
  usd/arm.usda
```

The supplied USD has one revolute arm joint and one prismatic finger. It is a small simulator example;
the hardware driver names in its manifest are placeholders. Keep it for the extension exercise, or
replace its asset and manifest together for your robot. Static validation checks every public extension,
so an error in another extension must also be resolved.

Inspect the parsed order and factory mapping without importing the simulator:

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
from core.arms.spec import load_arm_spec
from core.arms.calibration import resolve_calibration

arm, directory = load_arm_spec("guide_arm")
selection = resolve_calibration(arm, environ={})
print(directory)
print(arm.joint_name_pairs)
print(selection.mapping)
assert arm.joint_names == ("shoulder", "finger")
assert selection.unit is None
PY
```

## Fill the arm contract

Edit the generated `arm.toml`; use [SO101](../arms/so101/arm.toml) for a complete measured example.
Paths such as `usd_path` and `utils` are relative to the arm folder and must exist.

| Declaration | Meaning |
| --- | --- |
| `[arm]` | `name` equals the folder name; `usd_path` selects the articulation; `utils` is optional. |
| `[arm.links]` | `base` and `ee` are articulation body names. They must resolve uniquely at runtime. |
| `[[arm.joints]]` | One ordered entry per controlled joint: device `name`, USD `usd`, `type`, `role`, `limits`, `initial`. |
| `[arm.gripper]` | Optional `kind = "jaw"` or `"parallel"`; `joints` exactly matches the joints with `role = "gripper"`. |
| `[arm.wrist_roll]` | Optional independently controllable revolute arm joint; `relative_in_teleop` selects teleop rezeroing. |
| `[arm.actuators.<group>]` | Isaac Lab `ImplicitActuatorCfg` parameters; `joint_names_expr` matches USD joint names. |
| `[arm.spawn]` | Supported Newton rigid/articulation spawn properties. |
| `[arm.initial_state]` | Bare-scene position in metres and XYZ Euler rotation in degrees. Joint defaults come from the joint entries. |
| `[arm.lerobot]` | Follower/leader config type names, registration import modules, raw-unit mode and calibration mode/directories. |
| `[arm.hardware]` | Motor description, encoder counts per turn and physical zero count used to derive mappings. |

`[[arm.joints]]` order is the action, observation and dataset order. It can differ from USD traversal order.
Gripper axes may appear anywhere in that list. Do not add an assumed joint count or gripper index to `src/`.

| Value | Revolute joint | Prismatic joint |
| --- | --- | --- |
| Manifest `limits`, calibrated `joint_min`/`joint_max` | Physical degrees | Metres |
| Manifest `initial`, simulator state/action | Radians | Metres |
| Device/dataset value | Declared driver codec | Declared driver codec |

Conventional normalized device values are `-100..100` for arm-role joints and `0..100` for gripper-role
joints. `use_degrees = true` changes arm-role values to the driver's encoder-degree coordinate; its
midpoint need not equal physical zero. Grippers remain normalized. Explicit `raw_min`/`raw_max` mapping
endpoints preserve other or reversed conventions. Use `JointMapping` at simulator boundaries instead of
converting every axis with a degree function. Runtime limits clamp simulator targets; calibration spans
describe the device-to-physical conversion.

The USD must contain a valid articulation, rigid bodies with sensible masses/inertias, joints matching
the manifest, and usable collision geometry. The template demonstrates a fixed base. Package referenced
meshes/materials with the arm, retain metre scaling, and check joint axes, local frames and zero poses.
Newton imports topology once, so collision edits belong in asset preparation or the pre-import hook.

For a gripper, `contact_bodies` identifies bodies available to task contact sensors; `friction_bodies`
identifies subtrees receiving the shared grasp material. `tip_mesh` is optional geometry used by tasks
that measure a mesh-based grasp point. Tasks may require gripper contacts even though the arm contract
allows an arm without a gripper. Declare those requirements in the task folder.

## Keep optional geometry and cameras with the arm

The generated utility file contains two optional hooks: `prepare_collisions(root_prim)` authors
CAD-specific collision fixes before Newton imports the articulation;
`apply_appearance(stage, robot_prim_path, look)` applies the supplied fitted appearance.
The template's no-op hooks already work. Implement only
the hooks your asset needs, or remove `utils` from the manifest. Import simulator/USD packages in the
utility module, not in `core`. Keep mesh and material names local to the arm. The generic loader imports
the file by path. `[arm.appearance]` can supply body/motor shader paths consumed by the arm's appearance
hook; an unfitted scene may supply no appearance payload.

Each `[arm.cameras.<key>]` declares a mounted camera. `<key>` becomes its sensor/observation camera key;
use a key distinct from every fixed camera in the selected scene. Supply `parent_link`, metre `position`,
XYZ `euler_degrees`, image dimensions, and positive `focal_length`, `horizontal_aperture`,
`vertical_aperture` in the USD camera optical units. The parent link must exist. Optional `data_types`
defaults to RGB. `calibration` contains measured OpenCV `fx`, `fy`, `cx`, `cy` and optional distortion
coefficients/model; `response` contains fitted image-response parameters. Omit calibration/response
when you have no measurement rather than copying another arm's fit. See the complete wrist camera in
[SO101's manifest](../arms/so101/arm.toml) and update policy camera mappings accordingly.

For scripted/Mimic generation, optional `[arm.ik]` declares `position_offset_m`, the control point in
the EE-link frame, and `orientation_axes`, selected `x`, `y`, `z` rotation-error components in the arm-base
frame. Omitted axes request position-only IK. `method` selects `"dls"` (the default) or `"adaptive_dls"`.
`joint_limit_avoidance_gain` defaults to `0.0`; a positive gain enables null-space limit avoidance and
requires more arm joints than `3 + len(orientation_axes)`. Gripper joints do not count toward that total.
`joint_limit_avoidance_margin` defaults to `0.3` and must be positive. It is one shared distance from a
joint limit, in radians for revolute joints and metres for prismatic joints, including mixed chains.

Typical settings for five, six and seven arm joints are:

```toml
# Five arm joints: position plus tilt; yaw stays free.
[arm.ik]
orientation_axes = ["x", "y"]
method = "dls"

# Six arm joints: full pose (use this table instead of the one above).
# [arm.ik]
# orientation_axes = ["x", "y", "z"]
# method = "dls"

# Seven arm joints: full pose with a spare joint for limit avoidance.
# [arm.ik]
# orientation_axes = ["x", "y", "z"]
# method = "adaptive_dls"
# joint_limit_avoidance_gain = 0.1
# joint_limit_avoidance_margin = 0.3
```

Select constraints the arm can control and measure the TCP used by your task; an EE origin and an
object-centre grasp point are different points. Isaac Lab's `DifferentialIKController` supplies the IK
steps, and Newton supplies offline FK and Jacobians. For a different library backend, the optional
`<arm>_utils.py` module can define `make_ik(robot, arm, **kwargs)` and `make_kinematics(arm, **kwargs)`.
These hooks must provide the same public surface as `ArticulationIK` and `ArmKinematics`, respectively.
Keep their arm-specific settings in `arm.toml`.

Newton 1.6.1 requires each USD joint's `physics:body0` to name the body toward the base and
`physics:body1` to name the child toward the tool. A rotated joint axis is supported. If the importer
reports a reversed joint, swap body0/body1 and the local frames in the USD; the error names the affected
joint. The adapter preserves the import error and provides this correction.

Newton 1.6.1 retains rigid USD distances in authored stage units. The adapter reads the importer's
`linear_unit` metadata and converts poses, prismatic coordinates and Jacobian rows/columns at the
library boundary. The public action coordinates, TCP offsets and IK settings remain radians/metres,
including arms authored with `metersPerUnit = 0.001`.

## Choose calibration deliberately

With `calibration = "factory"`, no unit file is used. Revolute mappings derive the full encoder range
from `encoder_counts_per_turn` and `zero_count`; prismatic mappings use the joint's declared full stroke.
These declarations must match the actual driver and factory convention. A successful synthetic mapping
does not establish a physical robot's calibration.

With `calibration = "per_unit"`, create `calibration/units/<unit_id>.json`. A unit is distinct from the
LeRobot follower `id`: the unit stores the measured simulation mapping and its evidence; the device ID
selects the raw calibration file. The complete [rpd_robot unit](../arms/so101/calibration/units/rpd_robot.json)
shows the version-1 structure: `schema_version`, `arm`, `unit_id`, a `mapping` entry for every joint, and
`source` with raw `joints`, source hash and measured anchors. Keep the raw source file unchanged.

Per-joint parameters support physical `offset`, gripper `closed`/`contact_units`, and positive
`metres_per_count` for prismatic axes. State whether the bus applies drive-mode inversion with
`source.apply_drive_mode`, or provide explicit raw endpoints; do not infer direction from the motor name.
Degree-mode mappings also need the calibrated raw codec. The loader rejects stale spans that disagree
with the retained source.

For real hardware, first install/register the LeRobot driver named in `[arm.lerobot]` inside `s2r-sim`,
including its motor SDK. `modules` contains Python import paths that register the config classes.
Choose verified ports and IDs in `.env` or recording YAML. For a per-unit arm, these interactive commands
write raw calibration through that driver's lifecycle:

```bash
S2R_ARM=guide_arm ./s2r calibrate leader
S2R_ARM=guide_arm ./s2r calibrate follower
```

Run them only after replacing the template's driver placeholders and verifying its hardware contract.
The explicit host selector is forwarded into the command's container environment. These commands use
the hardware container's `TELEOP_PORT`/`TELEOP_ID` and
`ROBOT_PORT`/`ROBOT_ID`, and save under the arm-declared folders in
`$HF_LEROBOT_CALIBRATION`. Factory preset arms do not require this per-unit workflow. A calibrated device
file alone does not supply a measured contact anchor or validate the USD.

After collecting a raw follower calibration and physical anchor measurements, this complete CPU helper
creates a new unit without modifying the raw file. Save a JSON object of per-joint parameters first:
each key is a manifest joint name, and its value contains the measured parameter fields listed above.
Use `{}` only when no additional physical anchors are needed; prismatic axes always need a measured
`metres_per_count`. Enter `true`/`false` for drive-mode handling from the driver's actual convention.

```bash
read -r -p "Arm folder name: " ARM_NAME
read -r -p "New unit ID: " UNIT_ID
read -r -p "Raw follower calibration JSON path: " RAW_CALIBRATION
read -r -p "Measured per-joint parameters JSON path: " UNIT_PARAMETERS
read -r -p "Does the bus apply drive-mode inversion (true/false)? " APPLY_DRIVE_MODE
export ARM_NAME UNIT_ID RAW_CALIBRATION UNIT_PARAMETERS APPLY_DRIVE_MODE
PYTHONPATH=src uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from core.arms.spec import load_arm_spec
from core.arms.calibration import follower_mapping_from_calibration, load_unit, with_device_codec

arm, directory = load_arm_spec(os.environ["ARM_NAME"])
if arm.lerobot.calibration != "per_unit":
    raise ValueError("Set this completed arm's calibration mode to per_unit first")
unit_id = os.environ["UNIT_ID"]
if not unit_id or Path(unit_id).name != unit_id or unit_id in (".", ".."):
    raise ValueError("Unit ID must be one filename stem")
mode = os.environ["APPLY_DRIVE_MODE"].lower()
if mode not in ("true", "false"):
    raise ValueError("Drive-mode capability must be explicitly true or false")
apply_drive_mode = mode == "true"
raw_bytes = Path(os.environ["RAW_CALIBRATION"]).expanduser().read_bytes()
joints = json.loads(raw_bytes)
parameters = json.loads(Path(os.environ["UNIT_PARAMETERS"]).expanduser().read_text())
mapping = follower_mapping_from_calibration(joints, arm, parameters=parameters,
                                             apply_drive_mode=apply_drive_mode)
mapping = with_device_codec(arm, mapping, joints, apply_drive_mode=apply_drive_mode)
payload = {"schema_version": 1, "arm": arm.name, "unit_id": unit_id, "mapping": mapping,
           "source": {"joints": joints, "parameters": parameters,
                      "apply_drive_mode": apply_drive_mode,
                      "calibration_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                      "encoder_counts_per_turn": arm.hardware.encoder_counts_per_turn,
                      "zero_count": arm.hardware.zero_count}}
path = directory / "calibration" / "units" / (unit_id + ".json")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x") as handle:
    json.dump(payload, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(load_unit(arm, unit_id).path)
PY
```

Retain measurement provenance alongside `source.parameters`, especially the physical closed-gripper
anchor, roll zero and transmission scale. The helper computes mappings; it does not measure them.

Set recording `follower.unit` for a per-unit arm; pass `--unit` to evaluation, rollout and generation.
Do not pass a unit for a factory arm. See [getting started](01-getting-started.md) for the recording
config and hardware compose overlay. Sim-only leader-driven recording still needs a physical leader;
`s2r check` needs neither leader nor follower.

## Validate the simulator boundary

With the simulator container running and a coordinated GPU window, run:

```bash
./s2r check --arm guide_arm --scene blank_table --task stack_two_cubes --variant teleop
```

This resolves declared joints/bodies, steps finite joint targets and checks rendered cameras and task
object state. The two-joint example is not expected to solve stacking. Test your completed arm with the
actual intended scene/task and a compatible policy; fixed-embodiment policies must declare a matching
action dimension. Use [evaluation](06-evaluation.md) and [data generation](08-data-generation.md) for
longer execution, successful episodes and dataset validation.

Static checks do not inspect USD physics or open a device. Headless checks do not prove calibration,
grasp quality, policy success or hardware fidelity. Validate paired recording separately with the real
leader/follower and compare joint order, units, poses and synchronized camera frames. No second arm has
been validated on hardware; this guide does not claim OMX or other untested hardware acceptance.
