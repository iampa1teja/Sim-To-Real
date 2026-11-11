# Spec: import any arm from its URDF and run scripted pick-place with it

Audience: the implementer (Codex). Work on `v2` at its current head, including the IK library commits
(`48ec9c3`, `942f291`). Commit in logical commits; do not push.

## 1. Goal

A user with only a ROS description package (URDF or xacro files, meshes, optionally a `ros2_control` block)
runs one command and gets a working arm folder:

```bash
./s2r new arm <name> --urdf <path/to/robot.urdf.xacro> --ee_link <link> --gripper <joint>
```

After that, with no hand edits:

```bash
S2R_ARM=<name> ./s2r check
S2R_ARM=<name> ./s2r datagen --config_path configs/datagen/scripted.yaml --destination <dir>
```

spawns the arm in the default scene and produces an accepted scripted pick-place episode that passes dataset
validation. Anything the importer could not decide is listed in an import report for the owner to confirm. It
is not a silent default.

Scope: simulation only. Real hardware gets placeholders (section 6.6), not a working driver. No policy
training. SO-101 behaviour must not change.

## 2. Rules

- Follow `CLAUDE.md`. `src/core/` imports no Isaac Sim, Isaac Lab, Kit, `pxr` or Newton. Simulator work lives in
  `src/sim/`. `s2r` planning stays SDK-free and `--dry-run` prints the real command.
- No arm names in generic code. Never infer roles from joint or link names. Roles come from flags; structure only
  proposes values, and the report tags them `confirm`.
- Library math only (as in `docs/ik-library-spec.md`): URDF conversion through Isaac Lab, FK/IK through the
  existing `sim.datagen.ik`, geometry through `pxr`/NumPy.
- Never change production physics or installed SDK files. SO-101's seed-101 pick reference, stacking and Mimic
  results must stay identical to `v2` head.
- Never stop or mark the goal blocked. When a step fails, record it, take the listed fallback and continue.
  Report every fallback at the end.

## 3. URDF inputs the importer must handle

Write test fixtures for every item below. Real description packages commonly contain all of them.

1. xacro with `xacro:include`, `xacro:property`, `$(find <pkg>)` and `package://<pkg>/...` paths. Packages are
   resolved from `--package_path pkg=dir` flags. By default the directory holding `package.xml` with that
   `<name>` is searched upward and in sibling folders of the input file.
2. A `ros2_control` block inside the xacro: it must not break expansion and is parsed for joint parameters such
   as `id`.
3. A fixed `world` link joined to the real base by a fixed joint, and fixed joints between movable joints
   (motor housings, brackets). Named links must survive conversion.
4. Visual and collision meshes (STL, DAE, OBJ) with `origin xyz/rpy` and optional `scale`. Visual and collision
   may share one mesh file. Inertial origins with non-zero `rpy`.
5. Revolute joints on arbitrary axes, including negative axes and asymmetric limits such as `[-pi, 1.05]`.
   Continuous joints are read as revolute with the limits from `--continuous_limit_deg` (default ±180).
6. Prismatic gripper fingers on opposite axes, with one finger a `<mimic>` of the other, multiplier and offset
   honoured.
7. A gripper whose URDF joint is prismatic while its real motor is rotary (the conversion is unknown, so it
   becomes a TODO).
8. 5, 6 or 7 arm joints, one or two gripper joints, and an optional camera link fixed under the end effector.
9. Link and joint names containing uppercase letters, digits and underscores.

Unsupported input fails with a clear error naming the element: closed loops, floating joints inside the chain,
planar joints, more than one end effector.

## 4. Files

| File | Change |
| --- | --- |
| `src/core/arms/urdf.py` (new) | Pure URDF model: parse, chain, mimic, units. No simulator imports |
| `src/core/arms/arm_toml.py` (new) | Build the `arm.toml` text and the import report from a parsed model and flags |
| `src/core/arms/spec.py` | `mimic` joints, parallel-gripper gap fields, `[arm.import]` provenance table |
| `src/core/arms/mapping.py` | Mimic joints are excluded from codecs; no other change |
| `src/sim/arms/__init__.py`, `src/sim/arms/urdf_import.py` (new) | Expansion, USD conversion, geometry measurements, validation, writing the folder |
| `src/sim/arms/ready_pose.py` (new) | Ready pose and base yaw from IK and reach |
| `src/cli/main.py`, `src/core/paths.py` (`plan_new`), `src/cli/scaffold.py` | `s2r new arm <name> --urdf ...` plans the import |
| `src/sim/envs/...` action and reset terms that write joint targets/state | Write mimic joints from their source |
| `src/sim/scene/geometry.py`, `src/sim/envs/mdp/placement.py` | Fingertip geometry for `kind = "parallel"` grippers from finger contact bodies |
| `src/core/tasks/expert.py`, `src/sim/datagen/scripted.py`, `tasks/pick_place_cube/expert.py`, `tasks/stack_two_cubes/expert.py` | Physical gripper opening in metres; grasp yaw alignment for arms that control yaw |
| `configs/datagen/scripted.yaml`, `configs/datagen/stack_two_cubes.yaml` | Keep the current SO-101 values; add the metre option only as documented |
| `docker/` sim image, `pyproject.toml` host group | Add `xacro` (pure Python) where the import runs |
| `tests/core/test_urdf.py`, `tests/core/test_arm_toml.py`, `tests/core/test_arm_spec.py`, `tests/cli/...` | CPU tests |
| `tests/sim/test_urdf_import.py`, `tests/sim/test_mimic_joints.py`, `tests/sim/test_parallel_gripper_geometry.py` | Container tests |
| `tests/fixtures/urdf/` | Synthetic description packages (section 9) |
| `docs/02-add-an-arm.md`, `docs/08-data-generation.md`, `arms/_template/arm.toml`, `CLAUDE.md` | Docs |

## 5. Core (pure Python)

### 5.1 `src/core/arms/urdf.py`

Imports: `dataclasses`, `math`, `pathlib`, `xml.etree.ElementTree`, `numpy`. No simulator imports.

- `UrdfJoint` (dataclass): name, type, parent, child, origin (4x4), axis (unit 3-vector), lower/upper (radians or
  metres), effort, velocity, mimic `(source, multiplier, offset)` or `None`, and the `ros2_control` parameters
  (a dict).
- `UrdfLink` (dataclass): name, inertial (mass, COM origin, inertia tensor in link frame), visuals and
  collisions (mesh path resolved, scale, origin).
- `UrdfModel` (dataclass): name, links, joints, root link.
  - `from_xml(text, base_dir, package_paths)`: parse an expanded URDF. Resolve `package://` and relative mesh
    paths and fail on a missing mesh file. Attach `ros2_control` joint parameters when the block is present.
  - `chain(base_link, ee_link)`: ordered joints from base to ee. Errors on branching ambiguity.
  - `movable_chain(base_link, ee_link)`: the revolute/prismatic joints of `chain`, in order.
  - `default_base()`: the child of a fixed joint from a link with no parent (the `world` pattern), else the root.
  - `subtree_joints(link)`: movable joints below a link. Used to propose gripper joints and to check the
    `--gripper` flag.
  - `fk(q, base, tip)`: forward kinematics from URDF origins and axes. **Test oracle and report only.** It is
    used to cross-check the converted USD, never at runtime.
- `expand_xacro(path, package_paths) -> str`: wraps the `xacro` package (imported inside the function).
  Substitutes `$(find pkg)` using `package_paths`. The error message names the missing package.
- `find_packages(start) -> dict`: name to directory, from `package.xml` files near the input.

### 5.2 `src/core/arms/arm_toml.py`

Imports: `math`, `json`, `numpy`, `core.arms.urdf`.

- `ImportChoices` (dataclass): every flag value plus the measurements produced by the simulator step (section
  6.3). Measurements are optional so the CPU tests can supply them directly.
- `build_arm_toml(model, choices) -> str`: produces the full `arm.toml`, in the field order of
  `arms/so101/arm.toml`, with comments that say where each value came from:
  - `[arm]`: name, `usd_path = "usd/<name>.usd"`, `utils = "<name>_utils.py"`.
  - `[arm.links]`: base from `--base_link` or `default_base()`; ee from `--ee_link`.
  - `[[arm.joints]]`: arm joints in chain order, then gripper joints. `name` is the URDF name (`--rename a=b`
    optional), `usd` is the converted USD joint name, `type`, `role`, and `limits` in degrees (revolute) or metres
    (prismatic). `initial` comes from the ready pose (section 6.4) or 0 clamped into limits. Mimic joints get a
    `mimic` table and `role = "gripper"`.
  - `[arm.gripper]`: `kind = "parallel"` when every gripper joint is prismatic, else `"jaw"`; `joints` lists
    the non-mimic gripper joints. `contact_bodies` are the child links of all gripper joints, mimic included.
    `friction_bodies` are `contact_bodies` plus the ee link. Gap fields come from section 6.3.
  - `[arm.ik]`: `orientation_axes` is `["x","y"]` for 5 arm joints and `["x","y","z"]` for 6 or more.
    `method = "dls"`. `joint_limit_avoidance_gain` is 0.5 when the arm has more than 6 joints, else 0.
    `position_offset_m` is the measured grasp centre (section 6.3).
  - `[arm.cameras.<name>]` for each `--camera link=name`: `parent_link`, position and euler from the link's
    fixed transform to its parent body after conversion, and default intrinsics (640x480, horizontal FOV 60°),
    tagged `confirm`.
  - `[arm.actuators]`: one group per joint, from section 6.5.
  - `[arm.spawn]`, `[arm.initial_state]`: the spawn table copies the template defaults. `initial_state.rotation`
    uses the base yaw from section 6.4.
  - `[arm.hardware]` and calibration only when `--ros2_control` data exists (section 6.6).
  - `[arm.import]`: source file hashes, the command line, importer version, conversion path used, and the
    report path.
- `build_report(model, choices, checks) -> dict`: one entry per generated field with
  `{value, tag: derived|from-flag|measured|confirm|TODO, reason}`, plus each validation check
  (section 6.7) with pass/fail and numbers.

### 5.3 `src/core/arms/spec.py`

- `[[arm.joints]]` accepts `mimic = {joint, multiplier = 1.0, offset = 0.0}`. Validate:
  - the source exists and is not itself a mimic;
  - the joint types are the same;
  - the role is gripper (arm-joint mimics are rejected for now with a clear message).
- `ArmSpec`: `joints` keeps every declared joint. New properties:
  - `action_joints`: the non-mimic joints. `joint_names`, `feature_names` and the action dimension use these.
  - `mimic_joints`: tuples of `(index, source_index, multiplier, offset)`.
  - `usd_joint_names` keeps all joints.

  Audit every use of `arm.joints` and `len(arm.joints)` in `src/`, `tasks/` and `policies/`. Each one that
  means "action order" switches to `action_joints`. Record the list in the commit message.
- `GripperSpec`: optional `closed_gap_m` and `open_gap_m` (floats). For `kind = "parallel"` with both present,
  `gap_m(q)` and `q_for_gap(gap)` are linear maps over the source joint's limits. For `jaw`, the existing arm
  hook provides them (section 7.2). `tip_mesh` stays optional.
- `[arm.import]` is parsed and kept as provenance; it has no runtime effect.

### 5.4 `src/core/arms/mapping.py`

Codecs, `JointMapping.from_arm` and `raw_joint_limits` iterate `action_joints`. Mimic joints never appear in
raw/normalized units.

## 6. Simulator side: `src/sim/arms/urdf_import.py`

Imports at module level: `argparse`, `json`, `hashlib`, `pathlib`, `shutil`, `numpy`. Simulator imports happen
inside functions, after `sim.app.launch`.

Entry point `main(argv)`, run in `s2r-sim` through `s2r-entrypoint python -m sim.arms.urdf_import`. It does the
steps below in order. Every step writes its result into the report even when it fails. The arm folder is
written in all cases unless step 6.1 fails.

### 6.1 `stage_sources(args) -> (UrdfModel, paths)`

Copy the input description package(s) into `arms/<name>/source/` unchanged (provenance; hashes recorded). Expand
xacro to `arms/<name>/source/<name>.urdf` and parse it with `core.arms.urdf`. Validate the flags:
- `--ee_link` exists and is below the base.
- Each `--gripper` joint is in the ee subtree and is not on the base-to-ee chain.
- Mimic joints of a `--gripper` joint are picked up automatically.

### 6.2 `convert_to_usd(model, paths) -> usd_path`

Use Isaac Lab's URDF converter (`isaaclab.sim.converters.UrdfConverter` / `UrdfConverterCfg`, or the 3.0
equivalent; confirm the API at start and note it):
- fixed base; `merge_fixed_joints = False`; joint drive `position`;
- collision approximation convex hull for arm links and convex decomposition for gripper contact bodies;
- keep mimic joints when the importer supports them.

Output `arms/<name>/usd/<name>.usd`.

Fallback, if the converter is missing or fails: the Isaac Sim URDF importer extension through `sim.app.launch`.
Record which path was used.

After conversion, check:
- every URDF link and movable joint exists by name;
- USD units: metres-per-unit recorded;
- the USD FK from Newton (`ArmKinematics.from_arm` on a temporary spec) matches `UrdfModel.fk` at zero and
  at 20 random in-limit poses within 1 mm / 0.01 rad.

A mismatch is a failed check in the report, not a stop.

### 6.3 `measure_gripper(stage, spec) -> dict`

From the converted USD collision geometry of the contact bodies, in the ee link frame:
- **Closing axis:** for parallel grippers, from the finger joint axes; for jaw grippers, from the jaw motion
  between its limits.
- **`closed_gap_m` / `open_gap_m`:** finger inner-surface separation along the closing axis at the source
  joint's lower and upper limits. Sample the joint, sweep the finger collision points and take the smallest
  separation. Which limit is "open" is decided by the larger gap.
- **`position_offset_m`:** the grasp centre in the ee frame. This is the midpoint between the fingertips'
  inner surfaces at the closed limit, moved back from the tips by `--grasp_depth_m` (default 10 mm) along the
  approach axis. The approach axis is perpendicular to the closing axis and points from the ee origin toward
  the fingertip centroid. Tag it `measured`, with `confirm`.
- **Fingertip points per contact body,** used by section 7.1.

### 6.4 `src/sim/arms/ready_pose.py`

- `ready_pose(spec, kinematics) -> (q, info)`: solve with `ArmKinematics.inverse`, iterating, for a TCP
  pointing straight down (approach axis = −z of the base) at a point in front of the base. The point's height
  and distance are chosen by sampling a grid of heights 0.10–0.30 m and distances 30–70% of the maximum
  horizontal reach. Pick the pose with the largest margin to joint limits that converges (position < 1 mm,
  declared orientation axes < 0.01 rad). Fallback: zeros clamped into limits, tagged `TODO`.
- `forward_yaw(spec, q)`: the horizontal direction from the base origin to the TCP at the ready pose, in the
  base frame.
- `base_yaw_for_scene(spec, reference_spec)`: the `initial_state.rotation_euler_degrees` yaw that makes this
  arm's forward direction point the same world direction as the reference arm's (the existing default arm, run
  through the same rule). Tag it `derived`, with `confirm`.

### 6.5 `derive_actuators(spec, kinematics, model) -> dict`

Per joint, documented in the report:
- **`effort_limit_sim`:** the URDF `effort`.
- **`stiffness`:** chosen so the worst-case gravity torque over the reach grid gives at most 0.5° of steady sag
  (revolute) or 0.5 mm (prismatic). The gravity torque comes from the converted model's masses and COMs via
  Newton at the sampled poses. Clamped to at most `effort / 0.02 rad`.
- **`damping`:** critical damping with the joint's effective inertia from the Newton mass matrix at the ready
  pose, `2 * sqrt(stiffness * inertia)`.
- **Gripper joints:** stiffness so the URDF effort closes from open to closed within 0.5 s; `effort_limit_sim`
  from URDF.

### 6.6 Hardware placeholders, when `--ros2_control` or an in-xacro `ros2_control` block exists

`[arm.hardware]` holds the per-joint motor `id` parameters, `encoder_counts_per_turn = 4096` and
`zero_count = 2048`, all tagged `confirm`. It also writes `calibration/units/factory.json` with full-range
mappings: revolute joints over the full encoder range, prismatic joints over their declared stroke.

A prismatic URDF joint whose `ros2_control` joint has no conversion parameter gets
`counts_per_metre = "TODO"` in the report and is excluded from the factory unit. `[arm.lerobot]` is written
commented out with a `TODO` that real rollout needs a driver. Nothing here may be read by simulation.

### 6.7 Validation (`validate_import(name) -> list[check]`)

Each check is recorded with its numbers:
1. The spec loads, and the action dimension equals non-mimic joints.
2. FK cross-check (6.2).
3. Spawn in the default scene. Hold the ready pose for 2 s under gravity: drift per joint ≤ 1° / 1 mm.
4. Mimic: command the gripper source joint across its range; the mimic joint tracks within 1 mm.
5. Reach sweep: IK convergence over a 5 cm grid in the default scene's task workspace (the task's
   start-sampling region and the container region). Report the fraction reached and the unreachable cells.
6. Gripper: closing on the default task cube from the measured open gap gives contact on both contact
   bodies and holds the cube through a 5 cm lift.
7. Wrist camera: one rendered frame per declared camera, saved next to the report.

A failed check still writes the folder. The final summary prints every `confirm` / `TODO` field and every
failed check.

## 7. Making a new arm work in scripted pick-place

### 7.1 Fingertip geometry for parallel grippers

`src/sim/scene/geometry.py` and `src/sim/envs/mdp/placement.py` read fingertip points from `tip_mesh` today.
Add a parallel-gripper path: fingertip points come from the collision geometry of each contact body,
transformed into the ee frame (cached per stage like the jaw points). Use them wherever jaw points are used:
- jaw-relative start placement and clearance checks;
- contact/overlap guards.

`tip_mesh` remains the SO-101 path.

### 7.2 Physical gripper opening

- `Waypoint` gains `gripper_open_gap_m: float | None`. Exactly one of fraction and gap is set. Fraction stays,
  for compatibility with existing configs and the SO-101 reference.
- Expert options `gripper_open_gap_m` and `gripper_closed_gap_m` (metres) take priority over the fraction
  options when present. With neither present, the defaults are "open to `min(open_gap_m, object width +
  20 mm)`, close to 0 m" for arms whose `GripperSpec` has gap fields. Arms without gap fields use the fraction
  defaults as today, so SO-101 is unchanged.
- `GenerationContext.gripper_target`: converts a gap with `GripperSpec.q_for_gap`. For `jaw` grippers it uses
  an optional `<arm>_utils.gripper_q_for_gap` hook if present, else the fraction path. Mimic joints are set from
  the source.

### 7.3 Grasp yaw for arms that control yaw

When `"z"` is in `arm.ik.orientation_axes`, the pick-place and stack experts rotate the grasp orientation about
the table normal so the closing axis aligns with the nearest cube face normal, choosing the smallest rotation
from the current TCP yaw. That orientation is kept through transport (the no-roll-after-pickup rule still
holds). Arms without `"z"` are unchanged.

### 7.4 Mimic joints in the environment

The action terms and reset terms that write joint position targets or state write each mimic joint as
`multiplier * source + offset`. Prefer a converted mimic constraint when present and honoured by Newton; then
skip the manual write, and record which path is active. Datasets, policies and teleop never see mimic joints.

### 7.5 Missing mount

A missing `scenes/<scene>/mounts/<arm>.toml` already falls back to the anchor and the arm's initial joints. No
mount file is generated. The importer's base yaw and ready pose must be enough. If pick-place fails only
because of workspace placement, the importer may write a mount with a world translation that puts the reach
sweep's best region over the task workspace, tagged `derived`, with `confirm`.

## 8. CLI

- `s2r new arm <name> --urdf <file> [flags]`: `plan_new` returns a container plan running
  `python -m sim.arms.urdf_import` with the flags passed through and the input paths mounted. Without
  `--urdf`, the existing template copy is unchanged. `--dry-run` prints the command. No SDK imports in
  planning.
- Flags:
  - `--urdf`
  - `--ee_link`
  - `--gripper` (repeatable)
  - `--base_link`
  - `--camera link=name` (repeatable)
  - `--package_path pkg=dir` (repeatable)
  - `--ros2_control file`
  - `--rename urdf_name=name` (repeatable)
  - `--grasp_depth_m`
  - `--continuous_limit_deg`
  - `--force` (overwrite an existing arm folder that has an `[arm.import]` table only)
- The command `s2r sync` already includes `arms/`; source meshes are included. Use Git LFS rules only if the
  repo already tracks meshes with LFS; otherwise commit them normally.

## 9. Tests and fixtures

### Fixtures in `tests/fixtures/urdf/`

Generate small STL meshes in the fixture build script (boxes and cylinders, written as ASCII STL) so nothing
large is committed.

1. `five_jaw_pkg/`: plain URDF, 5 revolute joints, one revolute jaw, mesh `scale` attribute.
2. `seven_parallel_pkg/`: a description package with `package.xml`.
   - Split xacro: robot, arm, gripper and `ros2_control` files, using `$(find)` and `package://`.
   - A `world` link and fixed joints between all movable joints.
   - 7 revolute joints with mixed axes, one asymmetric limit (`-pi..1.05`) and a negative axis.
   - Two prismatic fingers on opposite axes, one a mimic of the other.
   - A fixed camera link under the gripper base; inertial origins with rpy.
3. `bad_*`: missing mesh, unknown package, closed loop, `--gripper` on the chain, a mimic of a mimic.

### CPU tests

- **`test_urdf.py`:** expansion, package resolution, chain/subtree, mimic parsing, `ros2_control` parameters,
  units, oracle FK against hand-computed poses, and every bad fixture's error message.
- **`test_arm_toml.py`:** generated toml for fixtures 1 and 2, checked with injected measurements:
  - joint order, units, roles and mimic table;
  - IK settings by joint count;
  - gripper kind and bodies;
  - report tags;
  - the toml loads through `load_arm_spec` from a temporary repo.
- **`test_arm_spec.py`:** mimic validation, `action_joints`, gap maps.
- **`tests/cli`:** `s2r new arm --urdf ... --dry-run` prints the module command; template path unchanged.

### Container tests

- **`test_urdf_import.py`:** full import of fixtures 1 and 2 into a temporary repo copy, then all section 6.7
  checks.
- **`test_mimic_joints.py`:** fingers move together through a reset and 50 action steps.
- **`test_parallel_gripper_geometry.py`:** fingertip points, measured gaps and grasp centre are within 1 mm of
  the fixture's construction values.

### Real-world import check

Import the official SO-101 URDF from the SO-ARM100 repository (`TheRobotStudio/SO-ARM100`, the
`Simulation/SO101` description) as `so101_urdf`. Do not touch `arms/so101`. Compare with `arms/so101`:
- same joint count, order (after `--rename`) and types;
- limits reported side by side;
- FK of both arms' USDs at 20 shared poses: TCP distance reported.

Differences are reported, not fixed. If the download is not possible, record that and continue.

## 10. Acceptance

1. Full CPU suite, `s2r check --static`, and `s2r check` with the default arm: same pass count plus new tests,
   no failures.
2. SO-101 regression: seed-101 scripted pick-place, stack, and Mimic numbers identical to `v2` head (steps,
   success, per-waypoint errors).
3. Fixture 2 imported with one command and no hand edits:
   - `S2R_ARM=<fixture> s2r check` passes;
   - `S2R_ARM=<fixture> s2r datagen --config_path configs/datagen/scripted.yaml` produces at least one
     accepted episode, task success at the final frame, and dataset validation passes with action
     dimension 8 (7 joints + 1 gripper);
   - stack_two_cubes likewise if reachable, else the reach report explains why.

   If pick-place fails, tune only the fixture's generated values through the importer's rules (never per-arm
   hand edits, never physics), with up to 5 iterations. Then commit with the failure analysis and images.
4. Fixture 1 (5-joint jaw) imports and passes section 6.7 checks 1–5 and 7.
5. `so101_urdf` comparison table produced.
6. Docs: `docs/02-add-an-arm.md` leads with "Import from URDF" (command, flags, how to read the report, what to
   confirm). The manual path stays. `CLAUDE.md` gets one bullet: arms can be imported from URDF; mimic joints
   are not action dimensions; gripper commands may be physical gaps.

## 11. Commits and final report

Suggested commits:
1. Spec, mapping and mimic support.
2. Parallel-gripper geometry and the physical gripper gap.
3. The URDF importer and CLI.
4. Fixtures and tests.
5. Docs.

Final report:
- changes per file;
- the list of `arms/_template` / generic code paths that switched to `action_joints`;
- the fixture 2 import report summary;
- pick-place and stack results with per-waypoint errors;
- the SO-101 regression table;
- the `so101_urdf` comparison;
- every fallback taken and every open `confirm`/`TODO` the importer produces.
