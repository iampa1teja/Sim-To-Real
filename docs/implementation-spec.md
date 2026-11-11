# Implementation spec: v2 audit changes and Isaac Lab IK

## How to use this file

1. Apply the sections strictly in the order of the table of contents. Skip a step only when a GATE says so.
2. Run every command from the repository root on the host, unless the command starts with `docker exec`.
3. For each edit, open the named file, find the "Remove" text (it occurs exactly once) and replace it with the "Insert" text.
4. The quoted text is authoritative. "Locator" line numbers give where the text starts after all earlier steps are applied.
5. Fenced blocks are exact: copy them byte for byte, including blank lines and indentation. Edit blocks use four backticks.
6. A GATE is a command with an expected output. Do exactly what the GATE says for the output you get.
7. Each part ends with a verification block. Commit only after every check in that block gives the expected result.
8. If a check fails, do not commit and do not continue; follow the failure instruction of that block.
9. Simulator checks and data generation use the GPU. Coordinate GPU use before running them.
10. Do not edit any file that this spec does not name. Commits use `git commit -a`, so untracked files are never added.

## What was verified and what was not

- Step A1 is the content of the two saved patches `0001-Remove-v1-leftovers-that-nothing-in-v2-can-reach.patch` and
  `0002-Describe-verified-scope-instead-of-restructure-phase.patch` (also local branch `audit-fixes-wip`). That
  content was applied once before and the host suite was run on it.
- While writing this spec, all of Part A (A1 to A8, and A9 in its outcome-A text) was applied to an extracted copy
  of commit `472efd1` in a scratch directory, not to this repository. Host results with Isaac Lab absent and `av`,
  `warp-lang`, `usd-core`, `gymnasium`, `mujoco-warp`, `lerobot[dataset]` 0.6.1 and `rerun-sdk` installed:
  377 passed, 36 skipped, 0 failed (untouched baseline: 377 passed, 11 skipped, 25 failed). `s2r check --static`
  passed 8 of 8. `s2r --dry-run train --policy groot_n17 ...` and `S2R_EVAL_CHECKPOINT=ck s2r --dry-run eval ...`
  printed the expected commands.
- Nothing in Part A has been run inside `s2r-sim`. The expected container count (424) is arithmetic.
- The A9 export command (audit 4.4) has not been run. The smoke evaluation that the audit suggests after section 1
  has not been run (see "Not in this spec").
- Nothing in Part B has been run in the simulator container, against an installed Isaac Lab, or with physics.
  The B1 and B3 Python code and their `tests/core/test_ik.py` changes were executed on a CPU host against the
  Isaac Lab `v3.0.0` source files `controllers/differential_ik.py`, `controllers/differential_ik_cfg.py` and
  `utils/math.py`, loaded through a minimal stand-in package: 20 of 21 tests passed; the remaining test needs
  `isaaclab.envs`, which the stand-in lacks. The new controller path matched the retained reference step to
  within 2e-7 rad/m in float32 and float64. This shows numerical equivalence of one step only, not physical
  behaviour. No Newton code was written or run.

## Table of contents

1. Part A: audit changes
   1. A0. Preconditions and baseline
   2. A1. Content of the saved patches (audit 1, 2, 3, 4.2 two rows, 5.1, 5.2)
   3. A2. Section 2 unused imports not covered by the patches
   4. A3. Default selectors defined once (audit 3)
   5. A4. Redundant `sys.path.insert` in `load_policies` (audit 3)
   6. A5. Host test environment (audit 4.1)
   7. A6. Stale v1 comments (audit 4.2)
   8. A7. One forwarded evaluation checkpoint variable (audit 4.3)
   9. A8. Phase references and the CLAUDE.md status diary (audit 5.3, 5.4)
   10. A9. Document the MolmoAct2 LoRA export (audit 4.4), with GATE A9.0
   11. A10. Part A verification and commit
2. Part B: replace the custom IK
   1. B0. Preconditions
   2. B1. Differential step through `DifferentialIKController`, with GATE B1.0
   3. B1-V. B1 verification and commit
   4. B2. `ArmKinematics` through `NewtonIKSolver`: GATE B2.0 (decides to skip)
   5. B3. Optional removal of `_compensate_load`, with GATE B3.E (experiment)
   6. B3-V. B3 verification and commit
3. Not in this spec

---

# Part A: audit changes

## A0. Preconditions and baseline

Run:

```bash
git rev-parse --short HEAD
git status --short
docker exec -w /workspace/s2r s2r-sim ls /workspace/s2r/docs/audit-v2.md
```

Expected: `472efd1`, then no output from `git status --short`, then `/workspace/s2r/docs/audit-v2.md`.
If any output differs, stop: this spec is written against commit `472efd1` mounted at `/workspace/s2r`.

Create the working branch:

```bash
git switch -c v2-audit-and-ik
```

Record the baseline:

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -m pytest tests -q -rs 2>&1 | tail -40
```

Expected: the last line contains `426 passed` and `22 subtests passed` and no `failed` or `error`. Write down the
`skipped` number from that line, if any, as SKIP0 (SKIP0 = 0 when the line has no `skipped`).

## A1. Content of the saved patches

Covers: audit section 1 (all rows, including `docs/restructure-plan.md`), section 2 (all rows except the test-file
imports and `tools/diagnostics/check_grasp_stability.py`, which are in A2), section 3 rows
`train_pick_place.sh`, `src/cli/main.py` (end of `main`), `src/cli/datagen.py` and `so101_utils.py` imports,
4.2 rows `requirements.txt` and `calibration.py:334`, 5.1 and 5.2.

### Whole-file deletions

```bash
git rm -q \
  configs/train/.gitkeep \
  docs/restructure-plan.md \
  policies/groot_n17/train/check_training_runtime.py \
  policies/groot_n17/train/download_pick_place_n17.sh \
  policies/groot_n17/train/launch_gr00t_server_n17.py \
  policies/groot_n17/train/train_dit.sh \
  policies/groot_n17/train/train_dit_only.py \
  policies/groot_n17/train/train_pick_place_failures_ft.sh \
  policies/groot_n17/train/train_pick_place_sim_100k.sh \
  policies/molmoact2/eval/audit_molmo_target_reach.py \
  src/__init__.py \
  tests/_fakes.py \
  tests/sim/test_grasp_diagnostics.py \
  tools/diagnostics/grasp_diagnostics.py
```

### `arms/so101/so101_utils.py`

#### module imports

Locator: line 7.

Remove:

````python

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def prepare_collisions(root_prim):
````

Insert:

````python

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def prepare_collisions(root_prim):
````

#### function `_split_link_mesh`

Locator: line 44.

Remove:

````python

def _split_link_mesh(root, *, link_name, mesh_name, axis, coordinate):
    """Clip the source CAD at a link-local plane and cook two convex pieces."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Gf, Vt

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    if not link:
````

Insert:

````python

def _split_link_mesh(root, *, link_name, mesh_name, axis, coordinate):
    """Clip the source CAD at a link-local plane and cook two convex pieces."""
    from scipy.spatial import ConvexHull

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    if not link:
````

#### function `_mesh_triangles`

Locator: line 102.

Remove:

````python


def _mesh_triangles(root, link_name, mesh_name):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    mesh = next(prim for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
````

Insert:

````python


def _mesh_triangles(root, link_name, mesh_name):

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    mesh = next(prim for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
````

#### function `_covered_by_triangles`

Locator: line 121.

Remove:

````python

def _covered_by_triangles(points, triangles):
    """Membership in the actual planar CAD triangles, including their boundaries."""
    import numpy as np

    covered = np.zeros(len(points), dtype=bool)
    for triangle in triangles:
````

Insert:

````python

def _covered_by_triangles(points, triangles):
    """Membership in the actual planar CAD triangles, including their boundaries."""

    covered = np.zeros(len(points), dtype=bool)
    for triangle in triangles:
````

#### function `_rectangle_covered_area`

Locator: line 138.

Remove:

````python

def _rectangle_covered_area(rectangle, triangles):
    """Exact polygon clipping verifies that a proposed pad face adds no area."""
    import numpy as np

    total = 0.0
    for triangle in triangles:
````

Insert:

````python

def _rectangle_covered_area(rectangle, triangles):
    """Exact polygon clipping verifies that a proposed pad face adds no area."""

    total = 0.0
    for triangle in triangles:
````

#### function `_inscribed_pad_face`

Locator: line 165.

Remove:

````python

def _inscribed_pad_face(triangles, plane_x, long_axis):
    """Find a rectangle strictly inside one measured CAD step, at its actual X."""
    import numpy as np

    plane_triangles = triangles[np.max(np.abs(triangles[:, :, 0] - plane_x), axis=1) < 2e-6]
    if len(plane_triangles) < 2:
````

Insert:

````python

def _inscribed_pad_face(triangles, plane_x, long_axis):
    """Find a rectangle strictly inside one measured CAD step, at its actual X."""

    plane_triangles = triangles[np.max(np.abs(triangles[:, :, 0] - plane_x), axis=1) < 2e-6]
    if len(plane_triangles) < 2:
````

#### function `_clipped_cad_points`

Locator: line 220.

Remove:

````python

def _clipped_cad_points(triangles, constraints):
    """Clip CAD triangles to axis halfspaces, retaining exact cut intersections."""
    import numpy as np

    polygons = triangles
    for axis, coordinate, sign in constraints:
````

Insert:

````python

def _clipped_cad_points(triangles, constraints):
    """Clip CAD triangles to axis halfspaces, retaining exact cut intersections."""

    polygons = triangles
    for axis, coordinate, sign in constraints:
````

#### function `_author_cad_support`

Locator: line 247.

Remove:

````python


def _author_cad_support(root, link, name, points):
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
````

Insert:

````python


def _author_cad_support(root, link, name, points):
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
````

#### function `apply_primitive_pads`

Locator: line 279.

Remove:

````python
    Returns ``(disabled_hull_paths, created_collider_paths)`` for material binding.
    """
    import json
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = root.GetStage()
    metre = UsdGeom.GetStageMetersPerUnit(stage)
````

Insert:

````python
    Returns ``(disabled_hull_paths, created_collider_paths)`` for material binding.
    """
    import json

    stage = root.GetStage()
    metre = UsdGeom.GetStageMetersPerUnit(stage)
````

#### functions `_rotation_matrix`, `measure_grasp` and `grasp_tip_offsets` at the end of the file (all three are deleted)

Locator: line 395.

Remove:

````python
        UsdShade.MaterialBindingAPI.Apply(replacement).Bind(
            UsdShade.Material(material), bindingStrength=UsdShade.Tokens.strongerThanDescendants)




def _rotation_matrix(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def measure_grasp(fixed_position, fixed_quaternion, moving_position, moving_quaternion,
                  cube_position, fixed_tip, moving_tip, cube_size_m):
    """Measure the opposing CAD fingertip centres using live rigid-body poses (quaternions WXYZ).

    Gap is projected on the fixed jaw's closing axis, not the distance between
    link origins. Tool error is expressed in the fixed gripper frame.
    """
    rotation = _rotation_matrix(fixed_quaternion)
    fixed = np.asarray(fixed_position) + rotation @ fixed_tip
    moving = np.asarray(moving_position) + _rotation_matrix(moving_quaternion) @ moving_tip
    gap = float(((moving - fixed) @ rotation)[0])
    tool = fixed + rotation[:, 0] * (cube_size_m / 2)
    error = (tool - cube_position) @ rotation
    return dict(tool_error_xyz_mm=(error * 1000).tolist(),
                tool_error_mm=float(np.linalg.norm(error) * 1000),
                jaw_gap_mm=gap * 1000, cube_width_mm=cube_size_m * 1000)


def grasp_tip_offsets(stage, root, arm, half_size):
    """Read mesh vertices once, then retain only two body-local contact points."""
    from pxr import Gf, Usd, UsdGeom

    fixed_body = arm.links.ee
    moving_body = arm.gripper.contact_bodies[0]
    cache = UsdGeom.XformCache()
    gripper = stage.GetPrimAtPath(f"{root}/{fixed_body}")
    jaw = stage.GetPrimAtPath(f"{root}/{moving_body}")
    gripper_matrix = cache.GetLocalToWorldTransform(gripper)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    result = []
    for body, fragment in ((gripper, f'/visuals/{arm.gripper.tip_mesh}/'),
                           (jaw, f'/visuals/moving_jaw_so101_v1/')):
        mesh = next((p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.IsA(UsdGeom.Mesh) and fragment in str(p.GetPath())), None)
        if mesh is None:
            raise RuntimeError(f'Cannot measure grasp: missing fingertip mesh under {body.GetPath()}')
        transform = cache.GetLocalToWorldTransform(mesh) * gripper_matrix.GetInverse()
        points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, p)))
                             for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]) * scale
        tip_z = points[:, 2].min()
        band = points[points[:, 2] <= tip_z + 2 * half_size]
        # The fixed jaw is left of the moving jaw in the authored zero pose.
        x = band[:, 0].max() if body == gripper else band[:, 0].min()
        point = np.array([x, (band[:, 1].min() + band[:, 1].max()) / 2, tip_z + half_size])
        body_from_gripper = gripper_matrix * cache.GetLocalToWorldTransform(body).GetInverse()
        local = body_from_gripper.Transform(Gf.Vec3d(*(point / scale)))
        result.append(np.asarray(local) * scale)
    return result


````

Insert:

````python
        UsdShade.MaterialBindingAPI.Apply(replacement).Bind(
            UsdShade.Material(material), bindingStrength=UsdShade.Tokens.strongerThanDescendants)

````

### `policies/groot_n17/policy.toml`

#### comment above the `[train]` table

Locator: line 26.

Remove:

````toml
TRANSFORMERS_OFFLINE = "1"
PYTHONDONTWRITEBYTECODE = "1"

# Other recipes in train/ (train_pick_place_sim_100k.sh, train_pick_place_failures_ft.sh, train_dit.sh)
# are variants of the commands below and are selected by the training CLI.
[train]
dataset_format = "lerobot_v2.1"

````

Insert:

````toml
TRANSFORMERS_OFFLINE = "1"
PYTHONDONTWRITEBYTECODE = "1"

[train]
dataset_format = "lerobot_v2.1"

````

### `policies/groot_n17/train/requirements.txt`

#### header comment and the `bitsandbytes` pin

Locator: line 1.

Remove:

````text
# Training-only packages on top of the pinned Isaac-GR00T environment, installed with --no-deps by the
# train image's --env mode. Pins copied from docker/train/Dockerfile and docker/train/download_wheels.py.
# UNVALIDATED here: GR00T still trains through docker/train.py until one `s2r train --generic` run succeeds.
av==16.0.1
bitsandbytes==0.50.2
lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475
````

Insert:

````text
# Training-only packages on top of the pinned Isaac-GR00T environment, installed with --no-deps by the
# train image's --env mode.
av==16.0.1
lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475
````

### `policies/groot_n17/train/train_pick_place.sh`

#### script body: the `S2R_PYTHON` block and the relaunch block after it

Locator: line 3.

Remove:

````bash
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Started by `s2r train` in the train container (entrypoint --env mode): use its variables and its venv.
if [[ -n ${S2R_PYTHON:-} ]]; then
    export S2R_TRAIN_IN_CONTAINER=1
    GROOT="${GROOT:-${S2R_PROJECT_DIR:-}}"
    DATASET="${DATASET:-${S2R_DATASET:-}}"
    OUT="${OUT:-${S2R_OUTPUT:-}}"
    BASE_MODEL="${BASE_MODEL:-${S2R_BASE_MODEL:-}}"
    GROOT_PYTHON="$S2R_PYTHON"
fi
if [[ ${S2R_TRAIN_IN_CONTAINER:-0} != 1 ]]; then
    exec bash "$REPO/docker/train.sh" train "$@"
fi
GROOT="${GROOT:-$HOME/Isaac-GR00T-N1.7}"
BASE_MODEL="${BASE_MODEL:-$HOME/sim2real/models/nv_so_arm_n17}"
DATASET="${DATASET:-$REPO/datasets/pick_place_v2_gr00t}"
````

Insert:

````bash
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Started by `s2r train` in the train container (entrypoint --env mode): use its variables and its venv.
if [[ -n ${S2R_PYTHON:-} ]]; then
    GROOT="${GROOT:-${S2R_PROJECT_DIR:-}}"
    DATASET="${DATASET:-${S2R_DATASET:-}}"
    OUT="${OUT:-${S2R_OUTPUT:-}}"
    BASE_MODEL="${BASE_MODEL:-${S2R_BASE_MODEL:-}}"
    GROOT_PYTHON="$S2R_PYTHON"
fi
GROOT="${GROOT:-$HOME/Isaac-GR00T-N1.7}"
BASE_MODEL="${BASE_MODEL:-$HOME/sim2real/models/nv_so_arm_n17}"
DATASET="${DATASET:-$REPO/datasets/pick_place_v2_gr00t}"
````

### `src/cli/main.py`

#### function `main` (the unreachable "not implemented yet" branch)

Locator: line 240.

Remove:

````python
    except CliError as error:
        print(f"s2r: error: {error}", file=sys.stderr)
        return 2
    if args.command == "datagen" and args.source is not None and not leading_dry_run:
        module = Path(repo) / "src" / "sim" / "datagen" / (args.source + ".py")
        if not module.is_file():
            print(f"s2r: data source {args.source!r} is not implemented yet ({module.name}).", file=sys.stderr)
            return 2
    return execute(plan, getattr(args, "dry_run", False) or leading_dry_run)
````

Insert:

````python
    except CliError as error:
        print(f"s2r: error: {error}", file=sys.stderr)
        return 2
    return execute(plan, getattr(args, "dry_run", False) or leading_dry_run)
````

### `src/core/arms/calibration.py`

#### function `resolve_calibration`

Locator: line 331.

Remove:

````python
    arm = load_arm_spec()[0] if arm is None else arm
    environ = os.environ if environ is None else environ
    root = Path(environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()
    # Environment names are migrated with all commands in Phase 4.
    explicit_follower = follower_path or environ.get("S2R_FOLLOWER_CALIBRATION")
    explicit_leader = leader_path or environ.get("S2R_LEADER_CALIBRATION")
    selected_unit = None
````

Insert:

````python
    arm = load_arm_spec()[0] if arm is None else arm
    environ = os.environ if environ is None else environ
    root = Path(environ.get("HF_LEROBOT_CALIBRATION") or "~/.cache/huggingface/lerobot/calibration").expanduser()
    explicit_follower = follower_path or environ.get("S2R_FOLLOWER_CALIBRATION")
    explicit_leader = leader_path or environ.get("S2R_LEADER_CALIBRATION")
    selected_unit = None
````

### `src/core/arms/mapping.py`

#### module constants `ARM_SCALE`, `GRIPPER_SCALE` (deleted)

Locator: line 6.

Remove:

````python
from core.arms.spec import ArmSpec, load_arm_spec
from core.arms.calibration import raw_endpoints

ARM_SCALE, GRIPPER_SCALE = 200.0, 100.0


def _raw_endpoints(mapping, joint_order, gripper_joints):
    low, high = raw_endpoints(mapping, joint_order, gripper_joints)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
````

Insert:

````python
from core.arms.spec import ArmSpec, load_arm_spec
from core.arms.calibration import raw_endpoints

def _raw_endpoints(mapping, joint_order, gripper_joints):
    low, high = raw_endpoints(mapping, joint_order, gripper_joints)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
````

#### class `JointMapping`, property `raw_limits` (deleted)

Locator: line 168.

Remove:

````python
            return np.deg2rad(physical)
        return np.where(self.revolute_mask.detach().cpu().numpy(), np.deg2rad(physical), physical)

    @property
    def raw_limits(self):
        low, high = self.raw_mins, self.raw_maxs
        if low is None:
            low = torch.where(self.gripper_mask, 0.0, -100.0)
            high = torch.full_like(self.mins, 100.0)
        return {name: (float(min(a, b)), float(max(a, b)))
                for name, a, b in zip(self.names, low.detach().cpu(), high.detach().cpu())}


def usd_limits_deg(arm=None):
````

Insert:

````python
            return np.deg2rad(physical)
        return np.where(self.revolute_mask.detach().cpu().numpy(), np.deg2rad(physical), physical)



def usd_limits_deg(arm=None):
````

### `src/core/arms/spec.py`

#### function `_parse_spec` (unused local `names`)

Locator: line 253.

Remove:

````python
        values = [getattr(joint, field_name) for joint in joints]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate joint {field_name}")
    names = {joint.name for joint in joints}
    gripper_names = {joint.name for joint in joints if joint.role == "gripper"}
    grip = raw.get("gripper")
    gripper = None
````

Insert:

````python
        values = [getattr(joint, field_name) for joint in joints]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate joint {field_name}")
    gripper_names = {joint.name for joint in joints if joint.role == "gripper"}
    grip = raw.get("gripper")
    gripper = None
````

### `src/core/benchmark/eval_set.py`

#### module imports

Locator: line 14.

Remove:

````python
from core.arms.mapping import mapping_for_arm
from core.arms.calibration import validate_mapping
from core.utils.hashing import json_sha256
from .geometry import (circular_mean, face_yaw, footprint, hull, hull_distance, inside_polygon,
                       pose_for_xy, rotation_matrix, sample_yaw, supported_on_table, wrap_yaw)
from .starts import (RecordedStarts, copy_objects, objects_to_poses, poses_to_objects, primary_object, task_instance)

SPLITS = ('id', 'ood', 'yaw', 'train')
````

Insert:

````python
from core.arms.mapping import mapping_for_arm
from core.arms.calibration import validate_mapping
from core.utils.hashing import json_sha256
from .geometry import (circular_mean, face_yaw, hull, hull_distance, inside_polygon,
                       rotation_matrix, sample_yaw, supported_on_table, wrap_yaw)
from .starts import (RecordedStarts, copy_objects, objects_to_poses, poses_to_objects, primary_object, task_instance)

SPLITS = ('id', 'ood', 'yaw', 'train')
````

### `src/core/benchmark/geometry.py`

#### function `inside_convex` (deleted)

Locator: line 171.

Remove:

````python
    projected = footprint(centre, base_rotation.T @ rotation, scene['objects'][object_name]['size'])
    return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)


def inside_convex(point, corners):
    """Test a projected point against the ordered convex hull of cuboid corners."""
    centre = corners.mean(axis=0)
    ordered = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
    edges = np.roll(ordered, -1, axis=0) - ordered
    rel = point - ordered
    return bool(np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] >= 0))

````

Insert:

````python
    projected = footprint(centre, base_rotation.T @ rotation, scene['objects'][object_name]['size'])
    return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)

````

### `src/core/benchmark/report.py`

#### function `select_instruction` (deleted)

Locator: line 5.

Remove:

````python
import numpy as np
from .stats import wilson_interval

def select_instruction(color, default, by_color=None):
    if by_color is None:
        return default
    if not isinstance(by_color, dict) or color not in by_color or not isinstance(by_color[color], str) or not by_color[color].strip():
        raise ValueError(f'--lang_instruction_by_color requires a non-empty instruction for {color!r}')
    return by_color[color]


def results_report(episodes, task, checkpoint, seed, *, random_fraction=0.0,
                   episode_length_s=None, action_horizon=16, step_dt=None):
````

Insert:

````python
import numpy as np
from .stats import wilson_interval


def results_report(episodes, task, checkpoint, seed, *, random_fraction=0.0,
                   episode_length_s=None, action_horizon=16, step_dt=None):
````

### `src/core/calibration/telemetry.py`

#### module imports

Locator: line 2.

Remove:

````python
from pathlib import Path
import time

import numpy as np

from core.arms.calibration import resolve_calibration
from core.utils.provenance import git_identity
from .gap import CalibrationGapLogger
````

Insert:

````python
from pathlib import Path
import time

from core.arms.calibration import resolve_calibration
from core.utils.provenance import git_identity
from .gap import CalibrationGapLogger
````

### `src/core/policy_client/smoothing.py`

#### class `ActionSmoother`, method `reset`

Locator: line 119.

Remove:

````python
        self.last_sent = None if initial_target is None else np.asarray(initial_target).copy()
        self.last_chunk_id = -1
        self.last_latency = 0.0
        self.stale_actions_dropped = 0
        self.ema.reset()

    def remaining(self, step):
````

Insert:

````python
        self.last_sent = None if initial_target is None else np.asarray(initial_target).copy()
        self.last_chunk_id = -1
        self.last_latency = 0.0
        self.ema.reset()

    def remaining(self, step):
````

#### class `ActionSmoother`, method `add_chunk`

Locator: line 131.

Remove:

````python
        if self.grasp is not None:
            self.grasp.observe_chunk(actions, chunk_id)
        stale = max(0, step - observation_step)
        self.stale_actions_dropped += min(stale, len(values))
        values = values[stale:]
        if len(values) == 0:
            return False
````

Insert:

````python
        if self.grasp is not None:
            self.grasp.observe_chunk(actions, chunk_id)
        stale = max(0, step - observation_step)
        values = values[stale:]
        if len(values) == 0:
            return False
````

### `src/core/rollout/real_rollout.py`

#### module constants `BODY_RANGE`, `GRIPPER_RANGE`, `OUTCOMES` (deleted)

Locator: line 27.

Remove:

````python
import numpy as np

DEFAULT_RENAME_MAP = '{"external": "room", "gripper": "wrist"}'
BODY_RANGE = (-100.0, 100.0)
GRIPPER_RANGE = (0.0, 100.0)
OUTCOMES = ("success", "failure", "unlabeled")


# ---------------------------------------------------------------------------
````

Insert:

````python
import numpy as np

DEFAULT_RENAME_MAP = '{"external": "room", "gripper": "wrist"}'


# ---------------------------------------------------------------------------
````

### `src/core/tasks/base.py`

#### class `Task`, method `start_from_row` (deleted)

Locator: line 278.

Remove:

````python
        stage = self.furthest_stage(flags) or "approaching"
        return "stage=" + stage.replace("_", " ")

    def start_from_row(self, row: dict[str, Any]) -> dict[str, Pose]:
        objects = row.get("objects", {})
        unknown = set(objects) - set(self.spec.objects)
        if unknown:
            raise KeyError(f"Eval-set row has unknown objects {sorted(unknown)} for task {self.name!r}")
        return {name: Pose.from_lists(value["position"], value.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0)))
                for name, value in objects.items()}

    def is_success(self, flags: dict[str, bool]) -> bool:
        return bool(flags.get(self.stages[-1], False))

````

Insert:

````python
        stage = self.furthest_stage(flags) or "approaching"
        return "stage=" + stage.replace("_", " ")

    def is_success(self, flags: dict[str, bool]) -> bool:
        return bool(flags.get(self.stages[-1], False))

````

### `src/sim/datagen/dr_variants.py`

#### module imports

Locator: line 4.

Remove:

````python
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

````

Insert:

````python
import hashlib
import json
import os
import random
import tempfile

````

#### module constant `DEFAULT_DR` and its comment (deleted)

Locator: line 18.

Remove:

````python


DR_TERMS = ("light", "robot_color", "camera_external", "camera_wrist")
# Cameras are calibrated to the real setup, so the default keeps camera views fixed.
DEFAULT_DR = ("light", "robot_color")


@dataclass
````

Insert:

````python


DR_TERMS = ("light", "robot_color", "camera_external", "camera_wrist")


@dataclass
````

### `src/sim/envs/scene_builder.py`

#### module imports

Locator: line 16.

Remove:

````python
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

from core.arms.spec import ArmSpec, load_arm_spec
from core.scenes import SceneSpec, load_mount_spec, load_scene_spec, mount_joint_positions, mount_pose
from sim.envs.base_env_cfg import BaseEnvCfg, _config_to_dict
from sim.envs.mdp import env_indices, image_raw, image_rgb_raw
````

Insert:

````python
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

from core.arms.spec import ArmSpec
from core.scenes import SceneSpec, load_mount_spec, load_scene_spec, mount_joint_positions, mount_pose
from sim.envs.base_env_cfg import BaseEnvCfg, _config_to_dict
from sim.envs.mdp import env_indices, image_raw, image_rgb_raw
````

### `src/sim/eval/report.py`

#### function `finish_report` (smoke-result loop)

Locator: line 190.

Remove:

````python
            }
            for row in state.episodes
        ]
        for item, row in zip(smoke_result["episodes"], state.episodes):
            diagnostics = row.get("grasp_diagnostics") or {}
            item["grasp_diagnostics"] = diagnostics
            item["closest_tool_error_mm"] = (diagnostics.get("closest") or {}).get("tool_error_mm")
            item["closest_closing_tool_error_mm"] = (diagnostics.get("closest_closing") or {}).get("tool_error_mm")
        print("[SMOKE_RESULT] " + json.dumps(smoke_result, separators=(",", ":"), allow_nan=False), flush=True)
    elif not context.smoke_test:
        print(json.dumps(report, indent=2, allow_nan=False))
````

Insert:

````python
            }
            for row in state.episodes
        ]
        print("[SMOKE_RESULT] " + json.dumps(smoke_result, separators=(",", ":"), allow_nan=False), flush=True)
    elif not context.smoke_test:
        print(json.dumps(report, indent=2, allow_nan=False))
````

### `tasks/pick_place_cube/mdp.py`

#### module imports (`numpy`)

Locator: line 13.

Remove:

````python
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from pxr import Gf, Usd, UsdShade

````

Insert:

````python
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pxr import Gf, Usd, UsdShade

````

#### module imports (`_wxyz`, `_xyzw`)

Locator: line 23.

Remove:

````python
from isaaclab.sensors import ContactSensor
from isaaclab.sim import get_current_stage

from sim.envs.mdp._common import _env_prim_path, _wxyz, _xyzw, as_torch, env_indices


# ---------------------------------------------------------------------------
````

Insert:

````python
from isaaclab.sensors import ContactSensor
from isaaclab.sim import get_current_stage

from sim.envs.mdp._common import _env_prim_path, as_torch, env_indices


# ---------------------------------------------------------------------------
````

### `tests/policies/test_groot_preparation.py`

#### function `test_small_training_flags_reach_the_existing_recipe_without_starting_cuda`

Locator: line 51.

Remove:

````python


def test_small_training_flags_reach_the_existing_recipe_without_starting_cuda(tmp_path):
    environment = dict(os.environ, S2R_TRAIN_IN_CONTAINER='1', GROOT=str(tmp_path),
                       DATASET=str(tmp_path / 'data'), OUT=str(tmp_path / 'out'),
                       BASE_MODEL=str(tmp_path / 'base'))
    result = subprocess.run(['bash', str(SCRIPTS / 'train_pick_place.sh'), '--batch', '1',
````

Insert:

````python


def test_small_training_flags_reach_the_existing_recipe_without_starting_cuda(tmp_path):
    environment = dict(os.environ, GROOT=str(tmp_path),
                       DATASET=str(tmp_path / 'data'), OUT=str(tmp_path / 'out'),
                       BASE_MODEL=str(tmp_path / 'base'))
    result = subprocess.run(['bash', str(SCRIPTS / 'train_pick_place.sh'), '--batch', '1',
````

### `README.md`

#### introduction (the text above `## Guides`)

Locator: line 1.

Remove:

````markdown
# sim-to-real (s2r)

The s2r restructure of the SO-101 sim-to-real platform: Isaac Lab 3.0 (Newton / MJWarp) for simulation,
LeRobot v0.6.1 for teleop and datasets, and pluggable policy servers (GR00T N1.7, MolmoAct2).
The previous code, with its full history, is on branch [`v1`](../../tree/v1).

The `v2` restructure has passed Phases 0–5 and 7; Phase 6 is deferred. The contact-solver fix passes motion replay and CPU regressions, and paired real/sim motion and camera
updates have been confirmed on hardware. [Phase status and verification](docs/restructure-status.md) records the
acceptance gates; [repository conventions](CLAUDE.md) covers units, quaternion order and import boundaries.
Copy `.env.example` to `.env` for local device settings. The arm, declarative scene and generic task phases passed
their CPU, recording and simulator checks; the evidence and renderer limitations are tracked in the status file.
Generic stages, named-object benchmarks and multiobject recording work for pick-place and stacking.
The CLI uses module entrypoints, provides `s2r new` scaffolds and `s2r check` validation.
Phase 4's full CPU suite, host checks, three simulator compositions and production recording pass.
The renamed evaluator reproduced the example policy's nine-episode schedule; live GR00T and Molmo also
served and evaluated with finite joint commands. Phase 4 is complete. Scripted generation has produced a
physically successful pick-place episode without a post-pickup wrist roll; dataset validation and real GR00T
preparation/verification pass. Mimic episodes pass physical and dataset validation at the source start and a
nearby changed start. A two-step GR00T training check verifies a real optimizer update on the generated data.
Eight workflow guides and their folder-extension conformance checks are complete.
Second-arm hardware work is deferred at the user's request.

## Guides

````

Insert:

````markdown
# sim-to-real (s2r)

A sim-to-real platform for tabletop manipulation arms: Isaac Lab 3.0 (Newton / MJWarp) for simulation,
LeRobot v0.6.1 for teleop and datasets, and pluggable policy servers (GR00T N1.7, MolmoAct2).
Arms, scenes, tasks and policies are folders: `s2r new` scaffolds one and `s2r check` validates it.
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

The [verification record](docs/restructure-status.md) holds the evidence and the renderer limitations.

## Guides

````

### `policies/README.md`

#### text above the heading `## Training a policy`

Locator: line 35.

Remove:

````markdown
tracked object's poses in `task_meta/`; legacy `pick_place_meta/` datasets remain readable. Fixed eval sets carry
task/arm identity, joint order and their saved raw-unit mapping. Their final declared stage determines success.
The three approved frozen-set conversions retain `legacy_eval_set_id` for paired comparisons; geometry checks
still apply. See the phase status for hold-state smoke evidence, which does not establish learned-policy success.

Clients receive `configure(ArmInfo)` with the selected joint names, roles, types and raw-unit mapping.
The default/example client derives its action dimension from this metadata. A fixed-embodiment plugin declares
`[action].dim = 6` (as GR00T and Molmo do); incompatible arms are rejected before inference.
`s2r check --static --arm <arm> --policy <name>` checks that selected pairing without rejecting unrelated policies.

The Phase 4 serving image `s2r-serve:v2-phase4` is built, and the renamed evaluation entrypoint completed the
example hold-state policy's nine-episode schedule. Actual GR00T and Molmo each completed a short simulator
evaluation with real model inference and finite joint commands. These checks establish serving/evaluation
integration, not learned-policy task success or benchmark performance. Final documentation/conformance is pending.
The current GR00T and Molmo training recipes retain their SO101/pick-place assumptions;
a generic arm loader alone does not make those model recipes embodiment-independent. See
[phase status](../docs/restructure-status.md) for the acceptance gates still being verified.

## Training a policy

````

Insert:

````markdown
tracked object's poses in `task_meta/`; legacy `pick_place_meta/` datasets remain readable. Fixed eval sets carry
task/arm identity, joint order and their saved raw-unit mapping. Their final declared stage determines success.
The three approved frozen-set conversions retain `legacy_eval_set_id` for paired comparisons; geometry checks
still apply. Hold-state smoke runs do not establish learned-policy success.

Clients receive `configure(ArmInfo)` with the selected joint names, roles, types and raw-unit mapping.
The default/example client derives its action dimension from this metadata. A fixed-embodiment plugin declares
`[action].dim = 6` (as GR00T and Molmo do); incompatible arms are rejected before inference.
`s2r check --static --arm <arm> --policy <name>` checks that selected pairing without rejecting unrelated policies.

GR00T and Molmo each completed a short simulator evaluation with real model inference and finite joint
commands. That establishes serving/evaluation integration, not learned-policy task success or benchmark
performance. The current GR00T and Molmo training recipes retain their SO101/pick-place assumptions;
a generic arm loader alone does not make those model recipes embodiment-independent. See the
[verification record](../docs/restructure-status.md) for the evidence.

## Training a policy

````

### `src/cli/datagen.py`

#### module imports

Locator: line 1.

Remove:

````python
"""Dispatch data generation to a source-specific module."""
import argparse
import importlib.util
from pathlib import Path
import runpy

````

Insert:

````python
"""Dispatch data generation to a source-specific module."""
import argparse
from pathlib import Path
import runpy

````

#### function `main`

Locator: line 33.

Remove:

````python
def main(argv=None):
    source, extra = resolve_source(argv)
    module = DATAGEN_MODULES[source]
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Data source {source!r} is not implemented yet ({module}).")
    import sys
    previous_argv = sys.argv
    try:
````

Insert:

````python
def main(argv=None):
    source, extra = resolve_source(argv)
    module = DATAGEN_MODULES[source]
    import sys
    previous_argv = sys.argv
    try:
````

### GATE A1.1: compare with the saved branch

```bash
git rev-parse --verify -q audit-fixes-wip
```

- If this prints a commit hash, run `git diff --stat audit-fixes-wip -- . ':!docs/audit-v2.md'`. Expected: no
  output. If it prints any file, redo the edits of A1 for the printed files until the command prints nothing.
- If it prints nothing, the branch is not in this clone; continue with A2.

## A2. Section 2 unused imports that the saved patches do not cover

### `tools/diagnostics/check_grasp_stability.py`

#### function `main`

Locator: line 109.

Remove:

````python
        from scipy.spatial.transform import Rotation
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaaclab_tasks.utils import parse_env_cfg
````

Insert:

````python
        from scipy.spatial.transform import Rotation
        from isaaclab_tasks.utils import parse_env_cfg
````

### `tests/cli/test_plan.py`

#### module imports

Locator: line 7.

Remove:

````python
from cli.plan import (CliError, DATAGEN_MODULES, DOCKERFILES, SIM_PROGRAMS, SYNC_PATHS, Plan,
````

Insert:

````python
from cli.plan import (CliError, DATAGEN_MODULES, DOCKERFILES, SIM_PROGRAMS, SYNC_PATHS,
````

### `tests/core/conftest.py`

#### module imports

Locator: line 1.

Remove:

````python
"""CPU fixtures for arms with layouts different from the production SO101."""
from pathlib import Path

import pytest
````

Insert:

````python
"""CPU fixtures for arms with layouts different from the production SO101."""

import pytest
````

### `tests/core/test_anchors.py`

#### module imports

Locator: line 1.

Remove:

````python
"""World/table/arm-base frames preserve the measured mount and placement."""
from dataclasses import replace

import numpy as np
````

Insert:

````python
"""World/table/arm-base frames preserve the measured mount and placement."""

import numpy as np
````

#### module imports

Locator: line 8.

Remove:

````python
    AnchorSpec, MountSpec, Pose, anchor_basis, anchor_position, anchor_quaternion, camera_quaternion,
````

Insert:

````python
    AnchorSpec, MountSpec, Pose, anchor_basis, anchor_position, camera_quaternion,
````

### `tests/core/test_calibration_units.py`

#### module imports

Locator: line 1.

Remove:

````python
"""Mixed-joint telemetry keeps metre fields and thresholds separate from degrees."""
import json

import pytest
````

Insert:

````python
"""Mixed-joint telemetry keeps metre fields and thresholds separate from degrees."""

import pytest
````

### `tests/core/test_rollout.py`

#### function `test_degree_mode_wrist_start_is_encoded_with_calibrated_zero`

Locator: line 46.

Remove:

````python
def test_degree_mode_wrist_start_is_encoded_with_calibrated_zero(monkeypatch):
    import math
    from dataclasses import replace
````

Insert:

````python
def test_degree_mode_wrist_start_is_encoded_with_calibrated_zero(monkeypatch):
    from dataclasses import replace
````

### `tests/core/test_teleop_prepare.py`

#### module imports

Locator: line 1.

Remove:

````python
"""Factory recordings still validate and retain their arm alignment marker."""
from types import SimpleNamespace

import pytest
````

Insert:

````python
"""Factory recordings still validate and retain their arm alignment marker."""

import pytest
````

### `tests/policies/test_groot_preparation.py`

#### module imports

Locator: line 10.

Remove:

````python
import numpy as np
import pytest

from core.datasets.task_meta import read_task_meta, save_task_trajectory
````

Insert:

````python
import numpy as np

from core.datasets.task_meta import read_task_meta, save_task_trajectory
````

### `tests/sim/test_scene_builder.py`

#### module imports

Locator: line 9.

Remove:

````python
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
````

Insert:

````python
from pxr import Usd, UsdPhysics, UsdShade
````

### `tests/sim/test_stage_tracking.py`

#### module imports

Locator: line 3.

Remove:

````python
from types import SimpleNamespace
import numpy as np
import pytest
````

Insert:

````python
from types import SimpleNamespace
import pytest
````

## A3. Define the default arm, scene and task once (audit section 3, "Default selectors")

The constants go in `src/core/paths.py`, which imports only `pathlib`, so every caller can import it before
Kit starts and before the S2R_* selectors are exported. Placement rule used below: a module that already imports
from `core` at module level gets a module-level import; every other entry point imports inside the function that
uses the constant, next to its existing lazy imports. `src/sim/teleop/config.py` needs a module-level import
because the defaults are dataclass field defaults. `src/cli/` and `src/core/` import only `core.paths`, never `sim`.

Occurrences left unchanged, with the reason:

- `src/sim/check.py` `--scene` default `"blank_table"`: a different value, deliberately not the default scene.
- `src/sim/recording/web_gui.py` and `gui/`: outside the restructure (CLAUDE.md).
- Test files that assert the literal values (for example `tests/sim/test_datagen_rollout.py`): they check the
  defaults and must keep the literals.
- `"so101-spawn"` authkey and `"so101_real_rollout_v1"` schema string: audit 7.7, owner decision.

Two occurrences not named in the audit row are included: `src/cli/list_envs.py` and `src/core/scenes/spec.py`.

### `src/core/paths.py`

#### module header

Locator: line 1.

Remove:

````python
"""Repository data folders (arms/, scenes/, tasks/) without any Isaac imports."""

from pathlib import Path

````

Insert:

````python
"""Repository data folders (arms/, scenes/, tasks/) and default selectors, without any Isaac imports."""

from pathlib import Path

DEFAULT_ARM = "so101"
DEFAULT_SCENE = "rpd_meeting_room"
DEFAULT_TASK = "pick_place_cube"

````

### `src/core/arms/spec.py`

#### module imports

Locator: line 17.

Remove:

````python
from core.paths import repo_dir
````

Insert:

````python
from core.paths import DEFAULT_ARM, repo_dir
````

#### function `selected_arm_name`

Locator: line 27.

Remove:

````python
    return os.environ.get("S2R_ARM", "so101")
````

Insert:

````python
    return os.environ.get("S2R_ARM", DEFAULT_ARM)
````

### `src/core/scenes/spec.py`

#### module imports

Locator: line 14.

Remove:

````python
from core.paths import repo_dir
````

Insert:

````python
from core.paths import DEFAULT_SCENE, repo_dir
````

#### function `selected_scene_name`

Locator: line 18.

Remove:

````python
    return os.environ.get("S2R_SCENE", "rpd_meeting_room")
````

Insert:

````python
    return os.environ.get("S2R_SCENE", DEFAULT_SCENE)
````

### `src/core/tasks/base.py`

#### module imports and module constant `TASK_NAME`

Locator: line 12.

Remove:

````python
from core.paths import repo_dir

TASK_NAME = os.environ.get("S2R_TASK", "pick_place_cube")
````

Insert:

````python
from core.paths import DEFAULT_TASK, repo_dir

TASK_NAME = os.environ.get("S2R_TASK", DEFAULT_TASK)
````

### `src/core/rollout/real_rollout.py`

#### function `build_parser`

Locator: line 177.

Remove:

````python
def build_parser():
    env = os.environ.get
````

Insert:

````python
def build_parser():
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    env = os.environ.get
````

#### function `build_parser`

Locator: line 183.

Remove:

````python
    parser.add_argument("--arm", default=env("S2R_ARM", "so101"))
    parser.add_argument("--scene", default=env("S2R_SCENE", "rpd_meeting_room"))
    parser.add_argument("--task", default=env("S2R_TASK", "pick_place_cube"))
````

Insert:

````python
    parser.add_argument("--arm", default=env("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=env("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=env("S2R_TASK", DEFAULT_TASK))
````

### `src/sim/check.py`

#### function `main`

Locator: line 76.

Remove:

````python
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", "so101"))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "blank_table"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", "pick_place_cube"))
````

Insert:

````python
def main(argv=None):
    from core.paths import DEFAULT_ARM, DEFAULT_TASK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "blank_table"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
````

### `src/sim/datagen/calibration_replay.py`

#### function `build_parser`

Locator: line 15.

Remove:

````python
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
````

Insert:

````python
def build_parser():
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
````

#### function `build_parser`

Locator: line 30.

Remove:

````python
    for key, default in (("arm", "so101"), ("scene", "rpd_meeting_room"), ("task", "pick_place_cube")):
````

Insert:

````python
    for key, default in (("arm", DEFAULT_ARM), ("scene", DEFAULT_SCENE), ("task", DEFAULT_TASK)):
````

### `src/sim/datagen/dr_replay.py`

#### module imports

Locator: line 25.

Remove:

````python
from core.utils.provenance import git_commit
````

Insert:

````python
from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from core.utils.provenance import git_commit
````

#### function `build_parser`

Locator: line 54.

Remove:

````python
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", "so101"))
````

Insert:

````python
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
````

#### function `build_parser`

Locator: line 56.

Remove:

````python
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "rpd_meeting_room"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", "pick_place_cube"))
````

Insert:

````python
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
````

### `src/sim/datagen/scripted.py`

#### module imports

Locator: line 13.

Remove:

````python
from core.tasks.base import Pose
````

Insert:

````python
from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from core.tasks.base import Pose
````

#### function `build_parser`

Locator: line 32.

Remove:

````python
    for key, default in (("arm", "so101"), ("scene", "rpd_meeting_room"), ("task", "pick_place_cube")):
````

Insert:

````python
    for key, default in (("arm", DEFAULT_ARM), ("scene", DEFAULT_SCENE), ("task", DEFAULT_TASK)):
````

### `src/sim/envs/__init__.py`

#### module imports

Locator: line 6.

Remove:

````python
from core.scenes import load_scene_spec, selected_scene_name
````

Insert:

````python
from core.paths import DEFAULT_TASK
from core.scenes import load_scene_spec, selected_scene_name
````

#### function `register_envs`

Locator: line 61.

Remove:

````python
    task_name = task if task is not None else os.environ.get("S2R_TASK", "pick_place_cube")
````

Insert:

````python
    task_name = task if task is not None else os.environ.get("S2R_TASK", DEFAULT_TASK)
````

### `src/sim/eval/run.py`

#### function `_build_parser`

Locator: line 137.

Remove:

````python
def _build_parser():
    parser = argparse.ArgumentParser(description="S2R policy evaluation in simulation (remote policy server).")
    parser.add_argument("--config_path", type=Path, default=None, help="configs/eval YAML; flags override it")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", "so101"))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "rpd_meeting_room"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", "pick_place_cube"))
````

Insert:

````python
def _build_parser():
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    parser = argparse.ArgumentParser(description="S2R policy evaluation in simulation (remote policy server).")
    parser.add_argument("--config_path", type=Path, default=None, help="configs/eval YAML; flags override it")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
````

### `src/sim/teleop/config.py`

#### module imports

Locator: line 15.

Remove:

````python
from sim.app import add_launch_arguments
````

Insert:

````python
from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from sim.app import add_launch_arguments
````

#### class `AgentConfig`

Locator: line 86.

Remove:

````python
    arm: str = _env("S2R_ARM", "so101")
    scene: str = _env("S2R_SCENE", "rpd_meeting_room")
    task: str = _env("S2R_TASK", "pick_place_cube")
````

Insert:

````python
    arm: str = _env("S2R_ARM", DEFAULT_ARM)
    scene: str = _env("S2R_SCENE", DEFAULT_SCENE)
    task: str = _env("S2R_TASK", DEFAULT_TASK)
````

### `src/cli/main.py`

#### function `check_compositions`

Locator: line 170.

Remove:

````python
def check_compositions(repo, arm=None, scene=None, task=None):
    base = (arm or os.environ.get("S2R_ARM", "so101"), scene or os.environ.get("S2R_SCENE", "rpd_meeting_room"),
            task or os.environ.get("S2R_TASK", "pick_place_cube"))
````

Insert:

````python
def check_compositions(repo, arm=None, scene=None, task=None):
    from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK

    base = (arm or os.environ.get("S2R_ARM", DEFAULT_ARM), scene or os.environ.get("S2R_SCENE", DEFAULT_SCENE),
            task or os.environ.get("S2R_TASK", DEFAULT_TASK))
````

### `src/cli/list_envs.py`

#### function `main`

Locator: line 34.

Remove:

````python
    from core.paths import repo_dir
    from core.tasks.base import available_tasks

    parser = argparse.ArgumentParser(description="List arms, scenes, tasks and their S2R gym ids.")
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "rpd_meeting_room"))
````

Insert:

````python
    from core.paths import DEFAULT_SCENE, repo_dir
    from core.tasks.base import available_tasks

    parser = argparse.ArgumentParser(description="List arms, scenes, tasks and their S2R gym ids.")
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
````

## A4. Remove the redundant `sys.path.insert` in `load_policies` (audit section 3)

The line is redundant in every call path. Every way to import `cli.main` makes `core` importable from the same
place: `./s2r` sets `PYTHONPATH=<repo>/src`; the installed console script `s2r = cli.main:main` installs the
packages `cli*`, `core*` and `sim*` from `src` (`pyproject.toml`); tests run with `PYTHONPATH=src`. In addition,
`main` imports `cli.plan` before it calls `load_policies`, and `cli.plan` imports `core.policy_client.discovery`
at module load, so the module is already cached. The only test that touches `load_policies`
(`tests/cli/test_plan.py`) replaces it with `monkeypatch`. `sys` stays used in `src/cli/main.py`.

### `src/cli/main.py`

#### function `load_policies`

Locator: line 101.

Remove:

````python
def load_policies(repo):
    sys.path.insert(0, str(repo / "src"))
    from core.policy_client.discovery import discover
````

Insert:

````python
def load_policies(repo):
    from core.policy_client.discovery import discover
````

## A5. Host test environment (audit 4.1)

Adds the two packages whose absence stopped collection, and makes the seven listed test files skip without
Isaac Lab instead of failing. Each `pytest.importorskip` line is the first statement of the named test function
(for `tests/sim/test_check.py`, of the helper `check_environment` that both failing tests call). In the container
the modules exist, so nothing is skipped there and the count does not change.

The `host` group still lacks `usd-core`, `lerobot`, `gymnasium` and `mujoco-warp`. Measured on a host with only
the documented group: tests in `tests/core/test_device_codec.py`, `tests/core/test_ik.py` (USD FK tests),
`tests/core/test_teleop_loop.py`, `tests/sim/recording/test_arm_data_contract.py` and
`tests/sim/test_solver_numerics.py` still fail there. A8 corrects the CLAUDE.md claim accordingly.

### `pyproject.toml`

#### `[dependency-groups] host`

Locator: line 23.

Remove:

````toml
    "pytest",
    "scipy",
]
````

Insert:

````toml
    "pytest",
    "scipy",
    "av",
    "warp-lang",
]
````

### `tests/cli/test_extensions.py`

#### function `test_scaffolded_composition_builds_and_exports_sdk_config_without_source_edits`

Locator: line 101.

Remove:

````python
def test_scaffolded_composition_builds_and_exports_sdk_config_without_source_edits(guide_extensions):
    from isaaclab_tasks.utils import parse_env_cfg
````

Insert:

````python
def test_scaffolded_composition_builds_and_exports_sdk_config_without_source_edits(guide_extensions):
    pytest.importorskip("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg
````

### `tests/core/test_ik.py`

#### function `test_mimic_actual_inverse_handles_native_permutation_and_two_gripper_roles`

Locator: line 293.

Remove:

````python
def test_mimic_actual_inverse_handles_native_permutation_and_two_gripper_roles(synthetic_arm):
    from sim.datagen.mimic import S2RMimicEnv
````

Insert:

````python
def test_mimic_actual_inverse_handles_native_permutation_and_two_gripper_roles(synthetic_arm):
    pytest.importorskip("isaaclab")
    from sim.datagen.mimic import S2RMimicEnv
````

### `tests/sim/test_check.py`

#### helper function `check_environment` (used by both failing tests)

Locator: line 9.

Remove:

````python
def check_environment(monkeypatch, *, bad_camera=False):
    import gymnasium
````

Insert:

````python
def check_environment(monkeypatch, *, bad_camera=False):
    pytest.importorskip("isaaclab_tasks")
    import gymnasium
````

### `tests/sim/test_datagen_rollout.py`

#### function `test_supplied_scripted_yaml_parses_without_launching_kit`

Locator: line 156.

Remove:

````python
def test_supplied_scripted_yaml_parses_without_launching_kit(tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs/datagen/scripted.yaml"
````

Insert:

````python
def test_supplied_scripted_yaml_parses_without_launching_kit(tmp_path):
    pytest.importorskip("isaaclab")
    path = Path(__file__).resolve().parents[2] / "configs/datagen/scripted.yaml"
````

#### function `test_live_step_records_before_action_and_observes_success_after_it`

Locator: line 197.

Remove:

````python
def test_live_step_records_before_action_and_observes_success_after_it(tmp_path):
    from core.arms.mapping import JointMapping
````

Insert:

````python
def test_live_step_records_before_action_and_observes_success_after_it(tmp_path):
    pytest.importorskip("isaaclab")
    from core.arms.mapping import JointMapping
````

#### function `test_controller_bias_resets_after_warmup_and_for_audit_replay`

Locator: line 276.

Remove:

````python
def test_controller_bias_resets_after_warmup_and_for_audit_replay(tmp_path, monkeypatch):
    from core.arms.mapping import JointMapping
````

Insert:

````python
def test_controller_bias_resets_after_warmup_and_for_audit_replay(tmp_path, monkeypatch):
    pytest.importorskip("isaaclab")
    from core.arms.mapping import JointMapping
````

### `tests/sim/test_datagen_task_signals.py`

#### function `test_placement_confirmation_counts_physics_steps_and_requires_grasp_history`

Locator: line 12.

Remove:

````python
def test_placement_confirmation_counts_physics_steps_and_requires_grasp_history(with_termination):
    task = load_task("pick_place_cube")
````

Insert:

````python
def test_placement_confirmation_counts_physics_steps_and_requires_grasp_history(with_termination):
    pytest.importorskip("isaaclab")
    task = load_task("pick_place_cube")
````

### `tests/sim/test_mimic.py`

#### function `test_actual_mimic_pool_segments_then_retargets_reference_object`

Locator: line 30.

Remove:

````python
def test_actual_mimic_pool_segments_then_retargets_reference_object():
````

Insert:

````python
def test_actual_mimic_pool_segments_then_retargets_reference_object():
    pytest.importorskip("isaaclab")
````

#### function `test_source_annotation_requires_latched_aligned_signals`

Locator: line 66.

Remove:

````python
def test_source_annotation_requires_latched_aligned_signals():
````

Insert:

````python
def test_source_annotation_requires_latched_aligned_signals():
    pytest.importorskip("isaaclab")
````

#### function `test_mimic_recovers_command_pose_without_reapplying_feedback_bias`

Locator: line 88.

Remove:

````python
def test_mimic_recovers_command_pose_without_reapplying_feedback_bias():
````

Insert:

````python
def test_mimic_recovers_command_pose_without_reapplying_feedback_bias():
    pytest.importorskip("isaaclab")
````

#### function `test_actual_sdk_queue_execution_keeps_physics_outside_asyncio`

Locator: line 127.

Remove:

````python
def test_actual_sdk_queue_execution_keeps_physics_outside_asyncio(
        max_steps, accepted, reason, step_error, diagnostic_trace, tmp_path, monkeypatch):
````

Insert:

````python
def test_actual_sdk_queue_execution_keeps_physics_outside_asyncio(
        max_steps, accepted, reason, step_error, diagnostic_trace, tmp_path, monkeypatch):
    pytest.importorskip("isaaclab")
````

#### function `test_mimic_attempt_refuses_physics_from_a_running_coroutine`

Locator: line 260.

Remove:

````python
def test_mimic_attempt_refuses_physics_from_a_running_coroutine():
````

Insert:

````python
def test_mimic_attempt_refuses_physics_from_a_running_coroutine():
    pytest.importorskip("isaaclab")
````

#### function `test_source_raw_units_are_decoded_before_sdk_gripper_extraction`

Locator: line 271.

Remove:

````python
def test_source_raw_units_are_decoded_before_sdk_gripper_extraction():
````

Insert:

````python
def test_source_raw_units_are_decoded_before_sdk_gripper_extraction():
    pytest.importorskip("isaaclab")
````

#### function `test_source_reflections_are_rejected`

Locator: line 285.

Remove:

````python
def test_source_reflections_are_rejected():
````

Insert:

````python
def test_source_reflections_are_rejected():
    pytest.importorskip("isaaclab")
````

#### function `test_mimic_entrypoint_uses_shared_selectors_and_source_validation`

Locator: line 293.

Remove:

````python
def test_mimic_entrypoint_uses_shared_selectors_and_source_validation(tmp_path):
````

Insert:

````python
def test_mimic_entrypoint_uses_shared_selectors_and_source_validation(tmp_path):
    pytest.importorskip("isaaclab")
````

### `tests/sim/test_mimic_config.py`

#### function `test_registered_scene_task_becomes_valid_sdk_mimic_cfg_without_launch`

Locator: line 13.

Remove:

````python
def test_registered_scene_task_becomes_valid_sdk_mimic_cfg_without_launch(scene_name, monkeypatch):
    from isaaclab.envs import MimicEnvCfg
````

Insert:

````python
def test_registered_scene_task_becomes_valid_sdk_mimic_cfg_without_launch(scene_name, monkeypatch):
    pytest.importorskip("isaaclab")
    from isaaclab.envs import MimicEnvCfg
````

#### function `test_mimic_adapter_matches_installed_constructor_and_generator_interface`

Locator: line 55.

Remove:

````python
def test_mimic_adapter_matches_installed_constructor_and_generator_interface():
    from isaaclab.envs import ManagerBasedRLMimicEnv
````

Insert:

````python
def test_mimic_adapter_matches_installed_constructor_and_generator_interface():
    pytest.importorskip("isaaclab")
    from isaaclab.envs import ManagerBasedRLMimicEnv
````

## A6. Stale comments that describe v1 (audit 4.2, rows not covered by the saved patches)

Module headers only. Each replacement keeps the sentences that state a current constraint and drops the history.
Left unchanged: v1 mentions inside function bodies and function docstrings (`src/sim/teleop/config.py`,
`src/sim/teleop/agent.py`, `src/sim/envs/mdp/obs.py`, `src/sim/envs/mdp/placement.py`, `src/sim/envs/mdp/_common.py`,
`src/sim/recording/recording_setup.py` lines 111 and 199, `src/sim/eval/run.py` smoke comment); the
"ported from the workshop" headers of `src/core/calibration/check.py` and `stats.py` (not v1); and
`src/sim/recording/web_gui.py` (outside the restructure). The `local/pick_place_v1` text in `dr_replay.py` is a
dataset id, not history.

### `policies/groot_n17/train/prepare.sh`

#### file header comment

Locator: line 2.

Remove:

````bash
# `s2r prepare --policy groot_n17 --generic`: turn the S2R_* contract into the script's own flags.
````

Insert:

````bash
# `s2r prepare --policy groot_n17`: turn the S2R_* contract into the script's own flags.
````

### `policies/groot_n17/train/verify.sh`

#### file header comment

Locator: line 2.

Remove:

````bash
# `s2r verify --policy groot_n17 --generic`: check a prepared dataset with the pinned GR00T reader.
````

Insert:

````bash
# `s2r verify --policy groot_n17`: check a prepared dataset with the pinned GR00T reader.
````

### `src/cli/plan.py`

#### module docstring

Locator: line 1.

Remove:

````python
"""Pure command planners for the s2r CLI (v1 cli/so101_cli/plan.py, renamed).
````

Insert:

````python
"""Pure command planners for the s2r CLI.
````

### `src/sim/eval/run.py`

#### module docstring and header comment

Locator: line 15.

Remove:

````python
"""Evaluate a served policy in simulation (ported from v1 scripts/lerobot_eval.py).

    s2r eval --config_path configs/eval/pick_place.yaml [--flag value ...]

The env is S2R-<task>-<variant> (default variant: eval). Flags override the YAML.
"""

# Changes from v1:
# - --arm/--scene/--task/--variant pick the env through sim.envs (S2R_* exported before any core/sim import);
#   v1's --task gym id and --robot_usd are gone (the arm USD comes from arms/<arm>/arm.toml).
# - --config_path reads the configs/eval YAML directly (v1 docker/eval_config.py turned it into flags).
# - Joint units go through core.arms.mapping.JointMapping and sim.eval.sim_remote_policy.SimRemotePolicy
#   (v1 LeRobotSO101Interface is gone). Settling holds the reset pose from the first observation.
# - Task plugins supply signals/stages; benchmark and dataset handling are shared core modules.
````

Insert:

````python
"""Evaluate a served policy in simulation.

    s2r eval --config_path configs/eval/pick_place.yaml [--flag value ...]

The env is S2R-<task>-<variant> (default variant: eval). Flags override the YAML.
"""

# - --arm/--scene/--task/--variant pick the env through sim.envs. S2R_* are exported before importing any
#   core/sim module that reads them at import. The arm USD comes from arms/<arm>/arm.toml.
# - --config_path reads the configs/eval YAML directly.
# - Joint units go through core.arms.mapping.JointMapping and sim.eval.sim_remote_policy.SimRemotePolicy.
#   Settling holds the reset pose from the first observation.
# - Task plugins supply signals/stages; benchmark and dataset handling are shared core modules.
````

#### function `parse_args`

Locator: line 292.

Remove:

````python
    # Arm/scene/task must be exported before anything imports core or sim (they read S2R_* at import).
````

Insert:

````python
    # Arm/scene/task must be exported before importing any core or sim module that reads S2R_* at import.
````

### `src/sim/datagen/dr_replay.py`

#### file header comment and module docstring

Locator: line 1.

Remove:

````python
# sim/datagen/dr_replay.py  (ported from v1 scripts/dr_replay.py)
"""Re-render recorded pick-place episodes with blue/red cubes and domain randomization.
````

Insert:

````python
"""Re-render recorded pick-place episodes with blue/red cubes and domain randomization.
````

#### header comment after the module docstring

Locator: line 14.

Remove:

````python
# Changes from v1:
# - --arm/--scene/--task/--variant select the env (S2R_* exported before core/sim imports); v1's --task gym id is gone.
# - DR ranges come from task.toml [task.dr] (the same values apply_eval_dr uses) instead of PickPlaceEvalDREnvCfg.
# - Joint units go through JointMapping; the recorder is sim.recording.lerobot_recorder.LeRobotRecorder.
````

Insert:

````python
# - --arm/--scene/--task/--variant select the env. S2R_* are exported before importing any core/sim module
#   that reads them at import.
# - DR ranges come from task.toml [task.dr] (the same values apply_eval_dr uses).
# - Joint units go through JointMapping; the recorder is sim.recording.lerobot_recorder.LeRobotRecorder.
````

### `src/sim/recording/replay_scene.py`

#### file header comment

Locator: line 1.

Remove:

````python
# sim/recording/replay_scene.py  (ported from v1 utils/replay_scene.py)
"""Pose recorded task frames in the live scene and render them with a chosen appearance.
````

Insert:

````python
"""Pose recorded task frames in the live scene and render them with a chosen appearance.
````

#### header comment after the module docstring

Locator: line 6.

Remove:

````python
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Joints: raw dataset units -> radians through core.arms.mapping.JointMapping (v1 LeRobotSO101Interface is gone),
#   written to the arm.toml joints in order with the *_index write APIs.
# - Sidecar poses are WXYZ (v1 format); converted to XYZW and composed with Isaac Lab's combine_frame_transforms
#   (v1 replay_math.base_to_world is gone). Base body comes from arm.toml.
# - Object colour comes from the task; declared scene lighting is independent of the task.
````

Insert:

````python
# - Joints: raw dataset units -> radians through core.arms.mapping.JointMapping,
#   written to the arm.toml joints in order with the *_index write APIs.
# - Sidecar poses are WXYZ; they are converted to XYZW and composed with Isaac Lab's combine_frame_transforms.
#   The base body comes from arm.toml.
# - Object colour comes from the task; declared scene lighting is independent of the task.
````

### `src/sim/recording/rerecording.py`

#### file header comment

Locator: line 1.

Remove:

````python
# sim/recording/rerecording.py  (ported from v1 utils/rerecording.py for LeRobot 0.6.1; used by the browser GUI)
"""Replace a recorded episode in private copies, keeping recoverable originals.
````

Insert:

````python
"""Replace a recorded episode in private copies, keeping recoverable originals.
````

#### module docstring

Locator: line 6.

Remove:

````python
LeRobot 0.6.1 differences from v1: ``finalize()`` is final (it stops the image writer and the
streaming encoder), so the live datasets are copied without finalizing them. That is safe between
````

Insert:

````python
In LeRobot 0.6.1 ``finalize()`` is final (it stops the image writer and the
streaming encoder), so the live datasets are copied without finalizing them. That is safe between
````

### `src/sim/datagen/spawner.py`

#### file header comment

Locator: line 1.

Remove:

````python
# Object spawner for teleop / paired sim+real recording (replaces v1's cube_spawner).
#
# Newton builds its physics model once, so objects cannot be created while the sim runs (v1 added a new
# USD cube prim per request; that only worked on PhysX). Instead, spawnable task objects are declared up
# front in the task's env cfg with a `park_object` reset event: they sit parked out of the scene, this
# module re-parks them after substantial drift, and a spawn request teleports one next to the fixed jaw tip (same
# placement as v1: jaw-tip offset, oriented like the end effector, snapped onto the table plane).
````

Insert:

````python
# Object spawner for teleop / paired sim+real recording.
#
# Newton builds its physics model once, so objects cannot be created while the sim runs. Instead, spawnable
# task objects are declared up front in the task's env cfg with a `park_object` reset event: they sit parked
# out of the scene, this module re-parks them after substantial drift, and a spawn request teleports one next
# to the fixed jaw tip (jaw-tip offset, oriented like the end effector, snapped onto the table plane).
````

### `src/core/calibration/report.py`

#### module docstring

Locator: line 3.

Remove:

````python
Port of v1 utils/calibration_report.py without the reference-pose half: the pose capture
(calibration_poses) and the hash-guarded lock were removed, so this report is tracking evidence only
(sim vs real vs commanded on still frames) and never a calibration verdict. Writes report.json,
report.md and tracking-gaps.png into --output.
````

Insert:

````python
This report is tracking evidence only (sim vs real vs commanded on still frames) and never a
calibration verdict. Writes report.json, report.md and tracking-gaps.png into --output.
````

### `src/sim/eval/sim_remote_policy.py`

#### module docstring

Locator: line 1.

Remove:

````python
"""Drive the simulated arm from a served policy (v1 lerobot_interface.SimRemotePolicy, split in two).
````

Insert:

````python
"""Drive the simulated arm from a served policy.
````

### `src/core/policy_client/remote_policy.py`

#### module docstring

Locator: line 1.

Remove:

````python
"""Robot-agnostic driver for a served policy (port of v1 SimRemotePolicy's core).
````

Insert:

````python
"""Robot-agnostic driver for a served policy.
````

### `src/cli/list_envs.py`

#### file header comment

Locator: line 4.

Remove:

````python
# (TASK), so this runs on the host or in the sim image without launching Isaac Sim (v1 launched the app
# just to read the gym registry). Env cfg classes are not built here.
````

Insert:

````python
# (TASK), so this runs on the host or in the sim image without launching Isaac Sim. Env cfg classes are
# not built here.
````

### `src/sim/recording/recording_setup.py`

#### module docstring

Locator: line 3.

Remove:

````python
Takes plain values instead of v1's argparse namespace so the same code serves CLI and
YAML configs. Device connection lives in ``sim.teleop.devices``. LeRobot imports are
````

Insert:

````python
Takes plain values so the same code serves CLI and
YAML configs. Device connection lives in ``sim.teleop.devices``. LeRobot imports are
````

## A7. One forwarded evaluation checkpoint variable (audit 4.3)

`PICK_PLACE_EVAL_CHECKPOINT` and `MODEL` are replaced by one `S2R_EVAL_CHECKPOINT`, which `s2r` now forwards into
the container like the other runtime variables. The YAML `policy.checkpoint` and the `--checkpoint` flag still
take precedence over it.

### `src/sim/eval/run.py`

#### function `_build_parser`

Locator: line 277.

Remove:

````python
    parser.add_argument(
        "--checkpoint", default=os.environ.get("PICK_PLACE_EVAL_CHECKPOINT", os.environ.get("MODEL", "unknown"))
    )
````

Insert:

````python
    parser.add_argument("--checkpoint", default=os.environ.get("S2R_EVAL_CHECKPOINT", "unknown"))
````

### `src/cli/plan.py`

#### module constant `SIM_RUNTIME_ENV`

Locator: line 26.

Remove:

````python
SIM_RUNTIME_ENV = (*SIM_SELECTION_ENV, "S2R_ROOM_USD", "S2R_EVAL_STARTS",
                   "S2R_FOLLOWER_CALIBRATION", "S2R_LEADER_CALIBRATION")
````

Insert:

````python
SIM_RUNTIME_ENV = (*SIM_SELECTION_ENV, "S2R_ROOM_USD", "S2R_EVAL_STARTS",
                   "S2R_FOLLOWER_CALIBRATION", "S2R_LEADER_CALIBRATION", "S2R_EVAL_CHECKPOINT")
````

### `tests/cli/test_plan.py`

#### function `test_runtime_file_overrides_are_explicit_container_environment`

Locator: line 45.

Remove:

````python
                 'S2R_LEADER_CALIBRATION': '/calibration/leader.json'}
    plan = plan_sim('sim.eval.run', (), selection_env=overrides)
    assert plan.argv[:11] == ('docker', 'exec', '-it', '-e', 'S2R_ROOM_USD=' + overrides['S2R_ROOM_USD'],
                              '-e', 'S2R_EVAL_STARTS=' + overrides['S2R_EVAL_STARTS'],
                              '-e', 'S2R_FOLLOWER_CALIBRATION=' + overrides['S2R_FOLLOWER_CALIBRATION'],
                              '-e', 'S2R_LEADER_CALIBRATION=' + overrides['S2R_LEADER_CALIBRATION'])
````

Insert:

````python
                 'S2R_LEADER_CALIBRATION': '/calibration/leader.json',
                 'S2R_EVAL_CHECKPOINT': 'checkpoint-2'}
    plan = plan_sim('sim.eval.run', (), selection_env=overrides)
    assert plan.argv[:13] == ('docker', 'exec', '-it', '-e', 'S2R_ROOM_USD=' + overrides['S2R_ROOM_USD'],
                              '-e', 'S2R_EVAL_STARTS=' + overrides['S2R_EVAL_STARTS'],
                              '-e', 'S2R_FOLLOWER_CALIBRATION=' + overrides['S2R_FOLLOWER_CALIBRATION'],
                              '-e', 'S2R_LEADER_CALIBRATION=' + overrides['S2R_LEADER_CALIBRATION'],
                              '-e', 'S2R_EVAL_CHECKPOINT=' + overrides['S2R_EVAL_CHECKPOINT'])
````

## A8. Phase references in user guides and the status diary in CLAUDE.md (audit 5.3, 5.4)

### `docs/05-training.md`

#### section "Fine-tune"

Locator: line 134.

Remove:

````markdown
weight update before accepting a training smoke check. The repository's Phase 5 check verified step 2,
````

Insert:

````markdown
weight update before accepting a training smoke check. The repository's recorded two-step check verified step 2,
````

#### section "Retained MolmoAct2 recipe"

Locator: line 233.

Remove:

````markdown
CPU-only unless `--gpu` is supplied. The restructure validated Molmo preparation
and argument contracts; the completed Phase 5 optimizer acceptance used GR00T.
````

Insert:

````markdown
CPU-only unless `--gpu` is supplied. Molmo preparation and argument contracts
are validated; the recorded optimizer-update check used GR00T.
````

#### section "CPU preparation regression tests"

Locator: line 248.

Remove:

````markdown
prepare/verify commands above remain the data-reader acceptance checks. See [phase status](restructure-status.md)
````

Insert:

````markdown
prepare/verify commands above remain the data-reader acceptance checks. See the [verification record](restructure-status.md)
````

### `docs/02-add-an-arm.md`

#### last section

Locator: line 231.

Remove:

````markdown
leader/follower and compare joint order, units, poses and synchronized camera frames. Phase 6 second-arm
hardware validation is deferred; this guide does not claim OMX or other untested hardware acceptance.
````

Insert:

````markdown
leader/follower and compare joint order, units, poses and synchronized camera frames. No second arm has
been validated on hardware; this guide does not claim OMX or other untested hardware acceptance.
````

### `docs/01-getting-started.md`

#### recording section

Locator: line 97.

Remove:

````markdown
the terminal controls and display/GUI mode for your session. The browser GUI is
outside this restructure. For paired recording use the paired-real config and
````

Insert:

````markdown
the terminal controls and display/GUI mode for your session. The browser GUI is
not covered by these guides. For paired recording use the paired-real config and
````

### `CLAUDE.md`

#### bullet on `s2r check --static`

Locator: line 82.

Remove:

````markdown
  only external dependency. The complete host test/tool environment is available with `uv sync --group host`.
````

Insert:

````markdown
  only external dependency. `uv sync --group host` installs the host tools; it is not a complete test
  environment, and tests that need Isaac Lab, USD, LeRobot or MJWarp fail or skip there. Run the full suite
  inside `s2r-sim`.
````

#### closing paragraphs

Locator: line 90.

Remove:

````markdown
The arm, declarative scene and generic task phases have passed their acceptance checks; consult the status file
for evidence and renderer limitations. Phase 4's full CPU suite, static/plugin checks, production recording,
example evaluation and live GR00T/Molmo serving/evaluation pass. Phase 5 CPU implementation passes, including
real SDK Mimic configuration/retargeting and schema-2 GR00T preparation. A no-roll scripted episode now passes
physical success, dataset validation and real GR00T preparation/verification. Mimic passes physical success and
dataset validation at the source start and a nearby changed start. A two-step GR00T optimizer update and source
integrity check pass; Phase 5 is complete. Phase 7 guides were exercised through new folders, headless
composition, policy serving/evaluation and paired comparison. The user deferred Phase 6 on 2026-11-10;
second-arm hardware validation remains outstanding. Training recipes and the retained visual DR recipe still
have their documented SO-101/pick-place scope; do not claim arbitrary-model training or learned task performance.


The paired-motion follow-up isolated a float32 MJWarp Cholesky failure during table contact. The same replay
fails identically on an isolated copy of main in the current container. `sim.solver_numerics` repairs only
failed Newton search directions: double-precision factorization, then positive-diagonal descent if needed.
The existing contact objective/line search and every physical parameter remain unchanged. Install before
CUDA graph capture; do not edit installed SDK files. Captured-matrix CPU tests and a 1,140-step contact/retreat
replay pass. Teleop rejects invalid simulation joints immediately, including outside active recording.

The user confirmed the restarted paired real/sim recorder now keeps motion and both simulation cameras working.
Phases 0–5 and 7 are complete; Phase 6 remains deferred. Do not infer a saved hardware episode from this motion check.
````

Insert:

````markdown
Training recipes and the retained visual DR recipe have their documented SO-101/pick-place scope; do not claim
arbitrary-model training or learned task performance.

`sim.solver_numerics` repairs only failed Newton search directions of the MJWarp contact solve: double-precision
factorization, then positive-diagonal descent if needed. The contact objective, line search and every physical
parameter stay unchanged. Install it before CUDA graph capture; do not edit installed SDK files.
````

## A9. Document the MolmoAct2 LoRA export (audit 4.4)

### GATE A9.0: does the train image's MolmoAct2 environment import what the script needs?

Requires the image `s2r-train:local` (`./s2r build train --tag local`, docs/05). Run:

```bash
docker run --rm -v "$PWD/policies:/policies" -v policy-cache:/opt/cache s2r-train:local --env molmoact2 -- \
  sh -c 'exec "$S2R_PYTHON" -c "import lerobot.policies.molmoact2.modeling_molmoact2, peft, safetensors.torch; print(\"export-env-ok\")"'
```

- Outcome A: the last line is `export-env-ok`. Apply the edit below with Insert text A.
- Outcome B: any other last line (for example `ModuleNotFoundError`). Apply the edit below with Insert text B.

### `docs/05-training.md`

#### section "Retained MolmoAct2 recipe", after its last paragraph

Locator: line 235.

Remove:
````markdown
It does not establish Molmo fine-tuning convergence or broaden either recipe to
arbitrary arms/tasks.
````

Insert text A (outcome A):
````markdown
It does not establish Molmo fine-tuning convergence or broaden either recipe to
arbitrary arms/tasks.

### Export a trained LoRA checkpoint for serving

The Molmo server loads an original-layout Hugging Face checkpoint. `train/export_molmo_lora.py` turns a
trained LeRobot MolmoAct2 LoRA checkpoint into that layout: it merges only the VLM LoRA deltas into the base
weights on the CPU, keeps the action-expert weights bit-identical, and checks that the merged tensor names
equal the base checkpoint's. It takes three paths:

- `--checkpoint`: the saved LeRobot policy folder, containing `config.json` and `policy_preprocessor.json`.
- `--base`: the original MolmoAct2 Hugging Face checkpoint folder, containing `model.safetensors.index.json`.
- `--output`: a new folder. The script refuses to overwrite an existing path.

It accepts only the workshop H30 LoRA configuration: `train_mode_vlm = "lora"`, `chunk_size = 30`,
`n_action_steps = 30`, joint signs `[1, -1, 1, 1, 1, 1]` and joint offsets `[0, 90, 90, 0, 0, 0]`. The
checkpoint's preprocessor must contain the `workshop_molmo_joint_frame` and `molmoact2_masked_normalizer`
steps. The output holds the merged safetensors shards, the base folder's `.py`, `.json` and `.jinja` files
except `model.safetensors.index.json` and `norm_stats.json`, a `norm_stats.json` built from the checkpoint's
saved normalizer under the tag `workshop_pick_place_v1_as_real`, and `workshop_export_report.json`.
Run it in the MolmoAct2 training environment of the train image:

```bash
export S2R_DOC_MOLMO_CHECKPOINT=/absolute/host/path/to/pretrained_model
export S2R_DOC_MOLMO_BASE=/absolute/host/path/to/molmoact2-base
export S2R_DOC_MOLMO_MODELS=/absolute/host/path/to/molmo-models
docker run --rm --init \
  -v "$PWD/policies:/policies" -v policy-cache:/opt/cache \
  -v "$S2R_DOC_MOLMO_CHECKPOINT:/data/checkpoint:ro" \
  -v "$S2R_DOC_MOLMO_BASE:/data/base_model:ro" \
  -v "$S2R_DOC_MOLMO_MODELS:/data/output" \
  s2r-train:local --env molmoact2 -- \
  sh -c 'exec "$S2R_PYTHON" "$@"' s2r /policies/molmoact2/train/export_molmo_lora.py \
  --checkpoint /data/checkpoint --base /data/base_model \
  --output /data/output/molmoact2_so101_workshop_lora
```

`configs/eval/pick_place_molmo_so101_lora_as_real.yaml` names this output folder,
`molmoact2_so101_workshop_lora`, as `policy.checkpoint` and selects `norm_tag: workshop_pick_place_v1_as_real`.
````

Insert text B (outcome B):
````markdown
It does not establish Molmo fine-tuning convergence or broaden either recipe to
arbitrary arms/tasks.

### Export a trained LoRA checkpoint for serving

The Molmo server loads an original-layout Hugging Face checkpoint. `train/export_molmo_lora.py` turns a
trained LeRobot MolmoAct2 LoRA checkpoint into that layout: it merges only the VLM LoRA deltas into the base
weights on the CPU, keeps the action-expert weights bit-identical, and checks that the merged tensor names
equal the base checkpoint's. It takes three paths:

- `--checkpoint`: the saved LeRobot policy folder, containing `config.json` and `policy_preprocessor.json`.
- `--base`: the original MolmoAct2 Hugging Face checkpoint folder, containing `model.safetensors.index.json`.
- `--output`: a new folder. The script refuses to overwrite an existing path.

It accepts only the workshop H30 LoRA configuration: `train_mode_vlm = "lora"`, `chunk_size = 30`,
`n_action_steps = 30`, joint signs `[1, -1, 1, 1, 1, 1]` and joint offsets `[0, 90, 90, 0, 0, 0]`. The
checkpoint's preprocessor must contain the `workshop_molmo_joint_frame` and `molmoact2_masked_normalizer`
steps. The output holds the merged safetensors shards, the base folder's `.py`, `.json` and `.jinja` files
except `model.safetensors.index.json` and `norm_stats.json`, a `norm_stats.json` built from the checkpoint's
saved normalizer under the tag `workshop_pick_place_v1_as_real`, and `workshop_export_report.json`.
Run it with a Python environment that provides LeRobot's MolmoAct2 policy package
(`lerobot.policies.molmoact2`), PEFT, `safetensors` and Torch.

`configs/eval/pick_place_molmo_so101_lora_as_real.yaml` names this output folder,
`molmoact2_so101_workshop_lora`, as `policy.checkpoint` and selects `norm_tag: workshop_pick_place_v1_as_real`.
````

The decision whether `check_molmo_export_server.py` and `check_molmo_optimizer_offload.py` belong to this
workflow is an owner decision and is not in this spec.

## A10. Part A verification and commit

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -m pytest tests -q -rs 2>&1 | tail -40
```

Expected: `424 passed`, `22 subtests passed`, no `failed`, no `error`, and the skipped count equal to SKIP0.
Arithmetic: 426 baseline - 2 (`tests/sim/test_grasp_diagnostics.py` deleted in A1) = 424. A2 to A9 add no test
functions; the A7 test change keeps the same test.

```bash
mkdir -p outputs
./s2r check > outputs/spec-check-a.log 2>&1; echo "exit=$?"; tail -30 outputs/spec-check-a.log
```

Expected: `exit=0`; the log shows 8 `PASS` lines (`arms/so101`, `scenes/blank_table`, `scenes/rpd_meeting_room`,
`tasks/pick_place_cube`, `tasks/stack_two_cubes`, `policies/example`, `policies/groot_n17`, `policies/molmoact2`)
and the 3 headless simulator checks complete as in the baseline.

```bash
./s2r --dry-run train --policy groot_n17 --dataset "$PWD/datasets" --output "$PWD/outputs/spec-dry-run"; echo "exit=$?"
S2R_EVAL_CHECKPOINT=spec-check ./s2r --dry-run eval --config_path configs/eval/pick_place.yaml
git grep -n -e '"so101"' -e '"rpd_meeting_room"' -e '"pick_place_cube"' -- src ':!src/sim/recording/web_gui.py'
git grep -n -e 'Changes from v1' -e 'ported from v1' -- src ':!src/sim/recording/web_gui.py'
```

Expected, in order: a line starting `+ docker run` and `exit=0`; a line starting `+ docker exec -it -e S2R_EVAL_CHECKPOINT=spec-check`;
exactly three lines, all in `src/core/paths.py`; no output.

If every check matches, commit:

```bash
git commit -a -m "Apply v2 audit sections 1 to 5 (without 5.5)"
```

If any check fails: do not commit, do not start Part B, and report the failing output to the owner.

---

# Part B: replace the custom IK

## B0. Preconditions

> Part B (B0 to B3-V) is superseded by `docs/ik-library-spec.md`. Do not apply it.

Part A is committed. B1, B2 and B3 are applied in that order; each ends in its own verification. B3 requires B1.
Shared helpers for the verification blocks (run once per shell):

```bash
export SPEC_DATA="$(docker inspect s2r-sim --format '{{range .Mounts}}{{if eq .Destination "/workspace/datasets"}}{{.Source}}{{end}}{{end}}')"
echo "$SPEC_DATA"
spec_report() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['accepted_episodes'], r['complete'])" "$1"; }
spec_prepared() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['status'], r['model_profile'])" "$1"; }
docker image inspect s2r-train:local --format ok
test -d "$S2R_DOC_HF_HOME/hub" && echo hf-ok
```

Expected: an absolute host path, then `ok`, then `hf-ok`. If the path is empty, stop. If `ok` or `hf-ok` is
missing, complete "Pin the training environment" in `docs/05-training.md` (it sets `S2R_DOC_HF_HOME`) first.

## B1. Differential step through `DifferentialIKController`

### What changes and what stays

- `ArticulationIK.solve` keeps its signature and `IKResult` meaning; its step now runs through
  `isaaclab.controllers.differential_ik.DifferentialIKController`, imported lazily inside the new method
  `_controller` (and `isaaclab.utils.math` inside the new method `_step`), so CPU tests can still import the module.
- Unchanged: `IKOptions`, `IKResult`, `ArticulationIK.current_pose`, `reset`, `integral_bias`, `_compensate_load`,
  the attributes `joint_ids`, `robot`, `options`, `position_offset_m`, and all of `ArmKinematics`.
- Deleted in B1: nothing. `solve_damped_step` stays because `ArmKinematics.inverse` calls it on every iteration
  and the tests use it as the reference step. `_batch`, `_quaternion` and `_bounded_norm` stay because the new
  `_step` uses them; `_rotation_error` and `_inverse_rotate` stay because `solve_damped_step` and
  `ArmKinematics.inverse` use them.

### Library facts used (Isaac Lab `v3.0.0`)

- `DifferentialIKController(cfg, num_envs, device)` (`controllers/differential_ik.py:55`). Its command and
  desired-pose buffers are float32 (`:69-73`).
- `set_command(command, ee_pos=None, ee_quat=None)` (`:119`). With `command_type="pose"`, `use_relative_mode=False`
  the command is `(x, y, z, qx, qy, qz, qw)`, quaternion XYZW, normalized by the controller. With
  `command_type="position"` the command is `(x, y, z)` and `ee_quat` is required.
- `compute(ee_pos, ee_quat, jacobian, joint_pos)` returns `joint_pos + delta` (`:192`). The controller assumes
  nothing about frames (`:41-42`); pose and Jacobian only need to share one frame.
- `dls`: `delta = J^T (J J^T + lambda_val^2 I)^-1 e` (`:253-254`), `lambda_val` from `ik_params`.
- Orientation error is `axis_angle(q_des * q_cur^-1)` (`utils/math.py:928-929`), shortest arc (`:492`).
  `orientation_weight=(wx, wy, wz)` multiplies the three angular Jacobian rows and the three error components
  (`differential_ik.py:296-299`); weight 0 drops an axis.
- `joint_limit_avoidance_gain` defaults to 0 (`differential_ik_cfg.py:75`); B1 sets it to 0 explicitly.
- The action term expresses pose with `subtract_frame_transforms` and rotates both Jacobian halves by the inverse
  base rotation (`envs/mdp/actions/task_space_actions.py:150-155`); B1 does the same with the declared base body.

### Mapping of `IKOptions` to the controller or to retained wrapper code

| `IKOptions` field | Where it goes in B1 |
| --- | --- |
| `damping` | `ik_params={"lambda_val": damping}` with `ik_method="dls"` |
| `orientation_weight` | `orientation_weight=(w if axis declared else 0.)` per base-frame axis x, y, z |
| `max_position_step_m` | wrapper: the commanded position is at most this far from the current TCP |
| `max_orientation_step_rad` | wrapper: the commanded rotation is scaled so its declared components have at most this norm |
| `max_joint_step_rad`, `max_joint_step_m` | wrapper: Jacobian columns are scaled by each joint's cap (solve in step-normalized joint coordinates), then the step is shrunk uniformly so no joint exceeds its cap |
| `position_tolerance_m`, `orientation_tolerance_rad` | wrapper: `IKResult.converged` |
| `integral_gain`, `max_integral_bias_rad`, `max_integral_bias_m` | unchanged `_compensate_load` |

Revolute limits are converted from degrees to radians and prismatic limits stay metres, then the arm joints are
clamped; gripper columns never enter the solve and keep their values. With no target orientation, or no declared
axes, the position controller is used, exactly as the old step dropped the angular rows. Column scaling and the
weighted rows reproduce the old step; the only numeric difference is the controller's float32 command buffer.

### GATE B1.0: installed controller API

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -c "import hashlib, inspect; import isaaclab.controllers.differential_ik as d, isaaclab.controllers.differential_ik_cfg as c, isaaclab.utils.math as m; print([hashlib.sha256(open(x.__file__,'rb').read()).hexdigest()[:12] for x in (d, c, m)]); print(inspect.signature(d.DifferentialIKController.compute)); print(inspect.signature(d.DifferentialIKController.set_command)); print(sorted(k for k in ('orientation_weight','joint_limit_avoidance_gain','ik_params') if k in c.DifferentialIKControllerCfg.__dataclass_fields__)); print(all(hasattr(m, n) for n in ('subtract_frame_transforms','quat_apply_inverse','matrix_from_quat','quat_inv','quat_mul','compute_pose_error','quat_from_angle_axis')))"
```

Expected last five lines:

```text
['2f8ab9d73dae', '9ca4e558d762', 'be896b173490']
(self, ee_pos: 'torch.Tensor', ee_quat: 'torch.Tensor', jacobian: 'torch.Tensor', joint_pos: 'torch.Tensor') -> 'torch.Tensor'
(self, command: 'torch.Tensor', ee_pos: 'torch.Tensor | None' = None, ee_quat: 'torch.Tensor | None' = None)
['ik_params', 'joint_limit_avoidance_gain', 'orientation_weight']
True
```

- Lines 2 to 5 match (line 1 matching or not): apply B1. When line 1 differs, the installed files differ from
  `v3.0.0`; the B1 tests compare the new path against the retained reference step and decide.
- Any of lines 2 to 5 differs, or the command fails: do not apply B1, B2 or B3. Part B ends; report the output.

### B1 edits

### `src/sim/datagen/ik.py`

#### module docstring

Locator: line 1.

Remove:

````python
"""Bounded native-Jacobian IK and side-effect-free USD forward kinematics.

Targets use world metres/XYZW. Orientation constraints are rotation-error components
in the declared arm-base frame, so a low-DOF arm need not promise a full 6-D pose.
Tracking updates return one limited joint target per physical step. Command-pose
inversion iterates cached kinematics locally before returning a bounded command.
"""
````

Insert:

````python
"""Bounded native-Jacobian IK and side-effect-free USD forward kinematics.

Targets use world metres/XYZW. Orientation constraints are rotation-error components
in the declared arm-base frame, so a low-DOF arm need not promise a full 6-D pose.
Tracking updates return one limited joint target per physical step; each step is
Isaac Lab's DifferentialIKController on the base-frame pose and Jacobian. Command-pose
inversion iterates cached kinematics locally before returning a bounded command.
"""
````

#### class `ArticulationIK`, class docstring

Locator: line 162.

Remove:

````python
class ArticulationIK:
    """Read Newton/Isaac Lab's native link-origin Jacobian without changing simulation state."""
````

Insert:

````python
class ArticulationIK:
    """Track world TCP targets with Isaac Lab's DifferentialIKController on the native link Jacobian."""
````

#### class `ArticulationIK`, method `__init__`

Locator: line 170.

Remove:

````python
        self.control_dt = control_dt
        self._integral_bias = None
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
````

Insert:

````python
        self.control_dt = control_dt
        self._integral_bias = None
        self._controllers = {}
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
````

#### class `ArticulationIK`, method `solve` (replace the whole method)

Locator: line 203.

Remove:

````python
    def solve(self, position_world, orientation_xyzw=None, *, advance=True):
        """Produce a bounded position command, retaining a slow integral load bias.

        ``advance=False`` previews a command/error without updating controller state.
        Only a command that will be physically dispatched should advance the integral.
        Bias is encoded in the returned joint target, so recorded actions replay it
        without an additional hidden effort controller.
        """
        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = _rotate(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        solved = solve_damped_step(self.arm, joint_positions, position, orientation, jacobian,
                                   position_world, orientation_xyzw, base_orientation=base_orientation,
                                   options=self.options)
        if self.options.integral_gain == 0:
            return solved
        return self._compensate_load(solved, joint_positions, advance=advance)
````

Insert:

````python
    def solve(self, position_world, orientation_xyzw=None, *, advance=True):
        """Produce a bounded position command, retaining a slow integral load bias.

        ``advance=False`` previews a command/error without updating controller state.
        Only a command that will be physically dispatched should advance the integral.
        Bias is encoded in the returned joint target, so recorded actions replay it
        without an additional hidden effort controller.
        """
        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = _rotate(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_position = _tensor(self.robot.data.body_link_pos_w)[:, self.base_id]
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        solved = self._step(joint_positions, position, orientation, jacobian, base_position, base_orientation,
                            position_world, orientation_xyzw)
        if self.options.integral_gain == 0:
            return solved
        return self._compensate_load(solved, joint_positions, advance=advance)

    def _controller(self, command_type, count, device):
        """One Isaac Lab controller per command type, batch size and device, built on first use."""
        key = (command_type, count, str(device))
        if key not in self._controllers:
            from isaaclab.controllers.differential_ik import DifferentialIKController
            from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

            weight = None
            if command_type == 'pose':
                weight = tuple(self.options.orientation_weight if axis in self.arm.ik.orientation_axes else 0.
                               for axis in ('x', 'y', 'z'))
            cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method='dls',
                                              ik_params={'lambda_val': self.options.damping},
                                              orientation_weight=weight, joint_limit_avoidance_gain=0.)
            self._controllers[key] = DifferentialIKController(cfg, num_envs=count, device=str(device))
        return self._controllers[key]

    def _step(self, joint_positions, position, orientation, jacobian, base_position, base_orientation,
              target_position, target_orientation):
        """One DifferentialIKController step in the arm-base frame, bounded per tick and per joint.

        Inputs are world-frame metres/XYZW and a world-frame TCP Jacobian whose columns follow
        manifest action order. Revolute columns act on radians, prismatic columns on metres.
        Gripper columns never enter the solve and their values stay exact.
        """
        import isaaclab.utils.math as math_utils

        q = joint_positions if joint_positions.is_floating_point() else joint_positions.to(torch.float64)
        count = len(self.arm.joints)
        if q.ndim != 2 or q.shape[1] != count or not torch.isfinite(q).all():
            raise ValueError(f'IK joint_positions must have finite shape (N,{count})')
        position, orientation, jacobian, base_position, base_orientation = (
            value.to(q) for value in (position, orientation, jacobian, base_position, base_orientation))
        if jacobian.shape != (q.shape[0], 6, count) or not torch.isfinite(jacobian).all():
            raise ValueError(f'IK jacobian must have finite shape (N,6,{count})')
        arm_ids = [index for index, joint in enumerate(self.arm.joints) if joint.role == 'arm']
        if not arm_ids:
            raise ValueError('IK requires at least one joint with role arm')
        options = self.options
        # The controller only assumes that pose and Jacobian share one frame: use the declared arm base.
        position_b, orientation_b = math_utils.subtract_frame_transforms(
            base_position, base_orientation, position, orientation)
        target_b = math_utils.quat_apply_inverse(
            base_orientation, _batch(target_position, 3, q, 'target_position') - base_position)
        rotation = math_utils.matrix_from_quat(math_utils.quat_inv(base_orientation))
        jacobian_b = torch.cat((rotation @ jacobian[:, :3], rotation @ jacobian[:, 3:]), dim=1)
        error = target_b - position_b
        position_error = error.norm(dim=-1)
        # Per-tick lead limit: the controller receives a target no farther than one bounded step.
        command = position_b + _bounded_norm(error, options.max_position_step_m)
        orientation_error = torch.zeros_like(position_error)
        axes = [{'x': 0, 'y': 1, 'z': 2}[axis] for axis in self.arm.ik.orientation_axes]
        if target_orientation is not None and axes:
            target_orientation_b = math_utils.quat_mul(
                math_utils.quat_inv(base_orientation), _quaternion(target_orientation, q, 'target_orientation'))
            _, rotation_error = math_utils.compute_pose_error(position_b, orientation_b, target_b,
                                                              target_orientation_b, rot_error_type='axis_angle')
            orientation_error = rotation_error[:, axes].norm(dim=-1)
            # Shrink the whole rotation so its declared components have at most the per-tick norm;
            # undeclared components carry zero controller weight.
            limit = options.max_orientation_step_rad
            bounded = rotation_error * (limit / orientation_error.clamp_min(limit)).unsqueeze(-1)
            lead = math_utils.quat_from_angle_axis(bounded.norm(dim=-1), bounded)
            command = torch.cat((command, math_utils.quat_mul(lead, orientation_b)), dim=-1)
            controller = self._controller('pose', q.shape[0], q.device)
            controller.set_command(command)
        else:
            controller = self._controller('position', q.shape[0], q.device)
            controller.set_command(command, ee_quat=orientation_b)
        step_limits = q.new_tensor([options.max_joint_step_rad if self.arm.joints[index].type == 'revolute'
                                    else options.max_joint_step_m for index in arm_ids])
        # Solve in joint coordinates normalized by each joint's step cap, then keep the direction
        # and shrink uniformly so no joint exceeds its cap.
        normalized_step = controller.compute(position_b, orientation_b, jacobian_b[:, :, arm_ids] * step_limits,
                                             torch.zeros_like(q[:, arm_ids]))
        scale = normalized_step.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
        delta = normalized_step / scale * step_limits
        lower = q.new_tensor([math.radians(self.arm.joints[index].limits[0])
                              if self.arm.joints[index].type == 'revolute'
                              else self.arm.joints[index].limits[0] for index in arm_ids])
        upper = q.new_tensor([math.radians(self.arm.joints[index].limits[1])
                              if self.arm.joints[index].type == 'revolute'
                              else self.arm.joints[index].limits[1] for index in arm_ids])
        proposed = q[:, arm_ids] + delta
        clipped = proposed.clamp(lower, upper)
        result = q.clone()
        result[:, arm_ids] = clipped.to(q.dtype)
        limited = (scale[:, 0] > 1) | (clipped != proposed).any(dim=-1)
        return IKResult(result, position_error, orientation_error,
                        (position_error <= options.position_tolerance_m)
                        & (orientation_error <= options.orientation_tolerance_rad), limited)
````

### `tests/core/test_ik.py`

#### function `test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs`

Locator: line 83.

Remove:

````python
def test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs(synthetic_arm):
    q = torch.tensor([0., .013, .1, .027], dtype=torch.float64)
````

Insert:

````python
def test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs(synthetic_arm):
    pytest.importorskip("isaaclab")
    q = torch.tensor([0., .013, .1, .027], dtype=torch.float64)
````

#### function `test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs`, and a new test after it

Locator: line 108.

Remove:

````python
    torch.testing.assert_close(result.joint_positions, expected.joint_positions, atol=1e-12, rtol=0)
    torch.testing.assert_close(robot.data.joint_pos, native_q, atol=0, rtol=0)
````

Insert:

````python
    # The controller stores its commanded pose in float32; the reference step is float64.
    torch.testing.assert_close(result.joint_positions, expected.joint_positions, atol=1e-6, rtol=0)
    torch.testing.assert_close(robot.data.joint_pos, native_q, atol=0, rtol=0)


def test_controller_pose_step_matches_reference_step_in_rotated_base_frame(synthetic_arm):
    pytest.importorskip("isaaclab")
    arm = replace(synthetic_arm, ik=IKSpec(('x',)))
    q = torch.tensor([.1, .013, .05, .027], dtype=torch.float64)
    jacobian = torch.randn((6, 4), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    jacobian[:, [1, 3]] = 0
    base_position = torch.tensor([.2, -.1, .05], dtype=torch.float64)
    base_orientation = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)], dtype=torch.float64)
    position = torch.tensor([.3, .1, .2], dtype=torch.float64)
    orientation = torch.tensor([0., math.sin(.1), 0., math.cos(.1)], dtype=torch.float64)
    native_jacobian = torch.zeros((1, 1, 6, 4), dtype=torch.float64)
    native_jacobian[0, 0] = jacobian
    data = SimpleNamespace(body_link_pos_w=torch.stack((base_position, position)).unsqueeze(0),
                           body_link_quat_w=torch.stack((base_orientation, orientation)).unsqueeze(0),
                           joint_pos=q.unsqueeze(0), body_link_jacobian_w=native_jacobian)
    robot = SimpleNamespace(data=data, is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(4)), list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, arm, options=IKOptions(integral_gain=0))
    target_position = position + position.new_tensor([.004, -.003, .002])
    target_orientation = [math.sin(.15), 0., 0., math.cos(.15)]  # 0.3 rad exceeds the 0.1 rad lead limit.
    result = solver.solve(target_position, target_orientation)
    expected = solve_damped_step(arm, q, position, orientation, jacobian, target_position, target_orientation,
                                 base_orientation=base_orientation)
    torch.testing.assert_close(result.joint_positions, expected.joint_positions, atol=1e-6, rtol=0)
    torch.testing.assert_close(result.position_error, expected.position_error, atol=1e-12, rtol=0)
    torch.testing.assert_close(result.orientation_error, expected.orientation_error, atol=1e-9, rtol=0)
    torch.testing.assert_close(result.limited, expected.limited)
    torch.testing.assert_close(result.joint_positions[0, [1, 3]], q[[1, 3]], atol=0, rtol=0)
````

#### function `test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers`

Locator: line 204.

Remove:

````python
def test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers(synthetic_arm):
    errors = []
````

Insert:

````python
def test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers(synthetic_arm):
    pytest.importorskip("isaaclab")
    errors = []
````

#### function `test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits`

Locator: line 222.

Remove:

````python
def test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits(synthetic_arm):
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
````

Insert:

````python
def test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits(synthetic_arm):
    pytest.importorskip("isaaclab")
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
````

### `CLAUDE.md`

#### bullet "Data generation uses ..."

Locator: line 46.

Remove:

````markdown
- Data generation uses task-folder `expert.py` waypoints and arm-declared `ik` TCP/orientation constraints.
````

Insert:

````markdown
- Data generation uses task-folder `expert.py` waypoints and arm-declared `ik` TCP/orientation constraints.
  Each scripted tracking step is Isaac Lab's `DifferentialIKController` on the arm-base-frame pose and
  Jacobian; `sim.datagen.ik` adds only the per-tick lead limits, per-joint step caps, runtime joint clamps
  and load compensation around it.
````

### `docs/08-data-generation.md`

#### section on `[arm.ik]`

Locator: line 84.

Remove:

````markdown
arbitrary orientation. The controller uses the articulation Jacobian and bounded
load-error compensation. Failed reachability/grasping must be fixed in the expert,
````

Insert:

````markdown
arbitrary orientation. Each tracking step is Isaac Lab's `DifferentialIKController`
(damped least squares) on the articulation Jacobian in the arm-base frame, with
bounded load-error compensation. Failed reachability/grasping must be fixed in the expert,
````

### B1 test summary (`tests/core/test_ik.py`)

- Unchanged: `test_four_joint_ik_converges_with_interleaved_parallel_gripper_and_prismatic`,
  `test_unreachable_prismatic_target_stops_at_runtime_limit`,
  `test_orientation_constraints_ignore_uncontrolled_yaw_and_use_base_frame`,
  `test_zero_jacobian_is_finite_and_does_not_report_unreachable_target_success`, all `ArmKinematics` tests
  (`test_usd_fk_*`, `test_so101_fk_*`, `test_fk_jacobian_*`, `test_local_inverse_*`), and
  `test_mimic_actual_inverse_handles_native_permutation_and_two_gripper_roles` (already skip-guarded in A5).
- `pytest.importorskip("isaaclab")` added to: `test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs`,
  `test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers`,
  `test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits`, and the new test.
- Tolerance in `test_native_adapter_...` changes from 1e-12 to 1e-6 (float32 controller buffer; measured 2.3e-8).
- New: `test_controller_pose_step_matches_reference_step_in_rotated_base_frame`.
- Deleted: none.

## B1-V. B1 verification and commit

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -m pytest tests -q -rs 2>&1 | tail -40
```

Expected: `425 passed` (424 + 1 new test), `22 subtests passed`, no `failed`, no `error`, skipped equal to SKIP0.

```bash
./s2r check > outputs/spec-check-b1.log 2>&1; echo "exit=$?"; tail -30 outputs/spec-check-b1.log
./s2r datagen --config_path configs/datagen/scripted.yaml --destination /workspace/datasets/spec_b1_scripted --report /workspace/s2r/outputs/datagen/spec_b1_scripted.json; echo "exit=$?"
spec_report outputs/datagen/spec_b1_scripted.json
./s2r datagen --config_path configs/datagen/stack_two_cubes.yaml --destination /workspace/datasets/spec_b1_stack --report /workspace/s2r/outputs/datagen/spec_b1_stack.json; echo "exit=$?"
spec_report outputs/datagen/spec_b1_stack.json
./s2r datagen --config_path configs/datagen/mimic.yaml --source_dataset /workspace/datasets/spec_b1_scripted --destination /workspace/datasets/spec_b1_mimic --report /workspace/s2r/outputs/datagen/spec_b1_mimic.json; echo "exit=$?"
spec_report outputs/datagen/spec_b1_mimic.json
./s2r prepare --policy groot_n17 --dataset "$SPEC_DATA/spec_b1_scripted" --output "$SPEC_DATA/spec_b1_gr00t" --image s2r-train:local; echo "exit=$?"
spec_prepared "$SPEC_DATA/spec_b1_gr00t/preparation_report.json"
./s2r verify --policy groot_n17 --dataset "$SPEC_DATA/spec_b1_gr00t" --hf-home "$S2R_DOC_HF_HOME" --image s2r-train:local; echo "exit=$?"
```

Expected: `exit=0` after every command; each `spec_report` prints `1 True`; `spec_prepared` prints
`passed so_arm_n17`; the check log matches A10. A committed scripted episode means every waypoint met its position
tolerance and the 0.03 rad declared-orientation tolerance (`execute_expert` only advances on that) and the task
succeeded at the final frame.

If every check matches: `git commit -a -m "Use Isaac Lab DifferentialIKController for scripted IK steps"`.

If any check fails: run
`git checkout -- src/sim/datagen/ik.py tests/core/test_ik.py CLAUDE.md docs/08-data-generation.md`, skip B2 and
B3, and report the failing output to the owner.

## B2. `ArmKinematics` through `NewtonIKSolver`

The Isaac Lab `v3.0.0` wrapper builds the rotation objective with one scalar weight:
`NewtonIKPoseObjective.__init__` calls `ik.IKObjectiveRotation(link_index=..., link_offset_rotation=...,
target_rotations=..., weight=cfg.rotation_weight)` (`isaaclab_newton/controllers/ik/newton_ik_objectives.py`), and
`NewtonIKPoseObjectiveCfg.rotation_weight` is a `float`. A solver driven by that objective weights all three
rotation axes equally, so it cannot leave yaw free for the five-joint SO-101 (`orientation_axes = ["x", "y"]`).
The gate confirms that the installed copy has the same limitation.

### GATE B2.0

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -c "import inspect; from isaaclab_newton.controllers.ik import newton_ik_objectives as o, newton_ik_objectives_cfg as c; print('weight=cfg.rotation_weight' in inspect.getsource(o.NewtonIKPoseObjective.__init__)); print(c.NewtonIKPoseObjectiveCfg.__dataclass_fields__['rotation_weight'].type)"
```

Expected last two lines: `True` and `float`.

- Outcome A (`True` and `float`): skip B2 entirely; `ArmKinematics` stays. Continue with B3.
- Any other output, including an import error: skip B2 entirely as well; `ArmKinematics` stays. Report the output
  to the owner, because a different installed wrapper needs a new spec. Continue with B3.

No B2 code is given: under both outcomes the step is skipped, and the Newton functions it would call
(`newton.ik` objectives, model building from the arm USD, FK) cannot be checked against source here.

## B3. Optional removal of `_compensate_load`

The standard alternative to the integral bias: each step starts from the previous dispatched command instead of
the measured joints, so a steady PD load error keeps moving the command until the measured TCP reaches the target.
The command never leads the measured joints by more than one joint step cap. B3.1 is the experiment; B3.2 removes
what the experiment makes unused. Do not run the test suite between B3.1 and B3.2: the two old integral tests fail
by design in that state.

### B3.1 edits (experiment)

### `src/sim/datagen/ik.py`

#### class `ArticulationIK`, method `__init__`

Locator: line 171.

Remove:

````python
        self._integral_bias = None
        self._controllers = {}
````

Insert:

````python
        self._integral_bias = None
        self._previous_command = None
        self._controllers = {}
````

#### class `ArticulationIK`, method `reset`

Locator: line 196.

Remove:

````python
    def reset(self):
        """Clear learned command bias after an episode reset/warmup."""
        self._integral_bias = None
````

Insert:

````python
    def reset(self):
        """Clear the stored command and its lead after an episode reset/warmup."""
        self._integral_bias = None
        self._previous_command = None
````

#### class `ArticulationIK`, method `solve` (replace the whole method)

Locator: line 205.

Remove:

````python
    def solve(self, position_world, orientation_xyzw=None, *, advance=True):
        """Produce a bounded position command, retaining a slow integral load bias.

        ``advance=False`` previews a command/error without updating controller state.
        Only a command that will be physically dispatched should advance the integral.
        Bias is encoded in the returned joint target, so recorded actions replay it
        without an additional hidden effort controller.
        """
        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = _rotate(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_position = _tensor(self.robot.data.body_link_pos_w)[:, self.base_id]
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        solved = self._step(joint_positions, position, orientation, jacobian, base_position, base_orientation,
                            position_world, orientation_xyzw)
        if self.options.integral_gain == 0:
            return solved
        return self._compensate_load(solved, joint_positions, advance=advance)
````

Insert:

````python
    def solve(self, position_world, orientation_xyzw=None, *, advance=True):
        """Produce a bounded position command that integrates on the previous dispatched command.

        Each step starts from the last dispatched command instead of the measured joints, so a steady
        PD load error keeps moving the command until the measured TCP reaches the target. The command
        never leads the measured joints by more than one joint step cap and stays inside the runtime
        joint limits. ``advance=False`` previews a command/error without storing it; only a command
        that will be physically dispatched should advance.
        """
        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = _rotate(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_position = _tensor(self.robot.data.body_link_pos_w)[:, self.base_id]
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        start = joint_positions if self._previous_command is None else self._previous_command.to(joint_positions)
        solved = self._step(start, position, orientation, jacobian, base_position, base_orientation,
                            position_world, orientation_xyzw)
        return self._bound_lead(solved, joint_positions, advance=advance)
````

#### class `ArticulationIK`, method `_compensate_load` (replace the whole method with `_bound_lead`)

Locator: line 320.

Remove:

````python
    def _compensate_load(self, solved, measured, *, advance):
        ids = [index for index, joint in enumerate(self.arm.joints) if joint.role == 'arm']
        delta = solved.joint_positions[:, ids] - measured[:, ids]
        previous = (torch.zeros_like(measured) if self._integral_bias is None
                    else self._integral_bias.to(measured))
        bias = previous[:, ids]
        step_caps = measured.new_tensor([self.options.max_joint_step_rad if self.arm.joints[index].type == 'revolute'
                                        else self.options.max_joint_step_m for index in ids])
        bias_caps = measured.new_tensor([self.options.max_integral_bias_rad if self.arm.joints[index].type == 'revolute'
                                        else self.options.max_integral_bias_m for index in ids])
        lower = measured.new_tensor([math.radians(self.arm.joints[index].limits[0])
                                     if self.arm.joints[index].type == 'revolute'
                                     else self.arm.joints[index].limits[0] for index in ids])
        upper = measured.new_tensor([math.radians(self.arm.joints[index].limits[1])
                                     if self.arm.joints[index].type == 'revolute'
                                     else self.arm.joints[index].limits[1] for index in ids])
        if advance:
            proposed = (bias + self.options.integral_gain * self.control_dt * delta).clamp(-bias_caps, bias_caps)
            increment = proposed - bias
            total = delta + proposed
            # Freeze integration that would drive an already lead-limited command farther outward.
            blocked = ((total > step_caps) & (increment > 0)) | ((total < -step_caps) & (increment < 0))
            bias = torch.where(blocked, bias, proposed)
            # Back-project at hard runtime limits; retained bias must not wind up beyond a stop.
            bias = bias.clamp(lower - measured[:, ids] - delta, upper - measured[:, ids] - delta)
            stored = torch.zeros_like(measured)
            stored[:, ids] = bias
            self._integral_bias = stored.detach().clone()
        total = delta + bias
        bounded = total.clamp(-step_caps, step_caps)
        command = (measured[:, ids] + bounded).clamp(lower, upper)
        result = measured.clone()
        result[:, ids] = command
        limited = solved.limited | (bounded != total).any(dim=-1) | (command != measured[:, ids] + bounded).any(dim=-1)
        return IKResult(result, solved.position_error, solved.orientation_error, solved.converged, limited)
````

Insert:

````python
    def _bound_lead(self, solved, measured, *, advance):
        ids = [index for index, joint in enumerate(self.arm.joints) if joint.role == 'arm']
        step_caps = measured.new_tensor([self.options.max_joint_step_rad if self.arm.joints[index].type == 'revolute'
                                        else self.options.max_joint_step_m for index in ids])
        lower = measured.new_tensor([math.radians(self.arm.joints[index].limits[0])
                                     if self.arm.joints[index].type == 'revolute'
                                     else self.arm.joints[index].limits[0] for index in ids])
        upper = measured.new_tensor([math.radians(self.arm.joints[index].limits[1])
                                     if self.arm.joints[index].type == 'revolute'
                                     else self.arm.joints[index].limits[1] for index in ids])
        lead = solved.joint_positions[:, ids].to(measured) - measured[:, ids]
        bounded = lead.clamp(-step_caps, step_caps)
        command = (measured[:, ids] + bounded).clamp(lower, upper)
        result = measured.clone()
        result[:, ids] = command
        if advance:
            self._previous_command = result.detach().clone()
            stored = torch.zeros_like(measured)
            stored[:, ids] = command - measured[:, ids]
            self._integral_bias = stored.detach().clone()
        limited = solved.limited | (bounded != lead).any(dim=-1) | (command != measured[:, ids] + bounded).any(dim=-1)
        return IKResult(result, solved.position_error, solved.orientation_error, solved.converged, limited)
````

### GATE B3.E: physical experiment

```bash
./s2r datagen --config_path configs/datagen/scripted.yaml --destination /workspace/datasets/spec_b3x_scripted --report /workspace/s2r/outputs/datagen/spec_b3x_scripted.json; echo "exit=$?"
spec_report outputs/datagen/spec_b3x_scripted.json
./s2r datagen --config_path configs/datagen/stack_two_cubes.yaml --destination /workspace/datasets/spec_b3x_stack --report /workspace/s2r/outputs/datagen/spec_b3x_stack.json; echo "exit=$?"
spec_report outputs/datagen/spec_b3x_stack.json
```

Pass criterion (report fields written by `make_report`/`publish_report` in `src/sim/datagen/scripted.py`): both
commands print `exit=0` and both `spec_report` lines print `1 True` (`accepted_episodes` 1 of the requested 1,
`complete` true).

- Pass: apply B3.2.
- Fail (any other output): run `git checkout -- src/sim/datagen/ik.py`. Leave `_compensate_load` as is. B3 ends;
  nothing is committed for B3.

### B3.2 edits (only after GATE B3.E passed)

Deletion list: `IKOptions.integral_gain`, `IKOptions.max_integral_bias_rad`, `IKOptions.max_integral_bias_m`;
the `control_dt` parameter and attribute of `ArticulationIK`; `ArticulationIK._compensate_load` (already replaced
by `_bound_lead` in B3.1); the `--ik_integral_gain`, `--ik_max_integral_bias_rad` and `--ik_max_integral_bias_m`
flags of `src/sim/datagen/scripted.py` and their validation; the report key `integral_feedback_applied` (replaced by
`command_integration_applied`); the tests `test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers`
and `test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits` (replaced by two tests of the new
behaviour). The report field `integral_bias_sim` keeps its name and now holds the lead of the last dispatched
command over the measured joints.

### `src/sim/datagen/ik.py`

#### class `IKOptions`

Locator: line 25.

Remove:

````python
    orientation_tolerance_rad: float = 0.03
    integral_gain: float = 1.0  # per second; zero disables physical PD-load compensation
    max_integral_bias_rad: float = 0.08
    max_integral_bias_m: float = 0.005

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for key, value in vars(self).items()
               if key != 'integral_gain') or not math.isfinite(self.integral_gain) or self.integral_gain < 0:
            raise ValueError('IK options must be finite and positive')
````

Insert:

````python
    orientation_tolerance_rad: float = 0.03

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError('IK options must be finite and positive')
````

#### class `ArticulationIK`, method `__init__`

Locator: line 161.

Remove:

````python
    def __init__(self, robot, arm, *, position_offset_m=None, options=None, control_dt=1 / 30):
        self.robot, self.arm = robot, arm
        self.options = options or IKOptions()
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError('IK control_dt must be finite and positive')
        self.control_dt = control_dt
        self._integral_bias = None
````

Insert:

````python
    def __init__(self, robot, arm, *, position_offset_m=None, options=None):
        self.robot, self.arm = robot, arm
        self.options = options or IKOptions()
        self._integral_bias = None
````

#### class `ArticulationIK`, property `integral_bias`

Locator: line 194.

Remove:

````python
    @property
    def integral_bias(self):
        return None if self._integral_bias is None else self._integral_bias.clone()
````

Insert:

````python
    @property
    def integral_bias(self):
        """Lead of the last dispatched command over the measured joints (radians/metres), or None."""
        return None if self._integral_bias is None else self._integral_bias.clone()
````

### `src/sim/datagen/scripted.py`

#### function `build_parser`

Locator: line 48.

Remove:

````python
    parser.add_argument("--ik_integral_gain", type=float, default=1.0, help="Bounded servo-error correction per second; zero disables")
    parser.add_argument("--ik_max_integral_bias_rad", type=float, default=.08)
    parser.add_argument("--ik_max_integral_bias_m", type=float, default=.005)
````

Insert: nothing. Delete the removed text.

#### function `parse_args`

Locator: line 65.

Remove:

````python
        parser.error("control_offset must contain three finite metres")
    correction = (args.ik_integral_gain, args.ik_max_integral_bias_rad, args.ik_max_integral_bias_m)
    if not np.isfinite(correction).all() or correction[0] < 0 or min(correction[1:]) <= 0:
        parser.error("IK correction gain must be nonnegative and its radian/metre caps positive and finite")
    return args
````

Insert:

````python
        parser.error("control_offset must contain three finite metres")
    return args
````

#### function `setup_context`

Locator: line 169.

Remove:

````python
    from sim.datagen.ik import ArticulationIK, ArmKinematics, IKOptions
````

Insert:

````python
    from sim.datagen.ik import ArticulationIK, ArmKinematics
````

#### function `setup_context`

Locator: line 206.

Remove:

````python
        options = IKOptions(integral_gain=getattr(args, "ik_integral_gain", 1.0),
                            max_integral_bias_rad=getattr(args, "ik_max_integral_bias_rad", .08),
                            max_integral_bias_m=getattr(args, "ik_max_integral_bias_m", .005))
        ik = ArticulationIK(raw.scene["robot"], arm, position_offset_m=offset,
                            control_dt=cfg.decimation * cfg.sim.dt, options=options)
````

Insert:

````python
        ik = ArticulationIK(raw.scene["robot"], arm, position_offset_m=offset)
````

#### function `make_report`

Locator: line 231.

Remove:

````python
                "integral_feedback_applied": context.source == "scripted" and context.ik.options.integral_gain > 0,
````

Insert:

````python
                "command_integration_applied": context.source == "scripted",
````

### `tests/core/test_ik.py`

#### helper `biased_prismatic_servo` and the two tests after it (replace the whole block)

Locator: line 188.

Remove:

````python
def biased_prismatic_servo(arm, *, gain=1.):
    """A CPU position servo with constant load: equilibrium is command minus 2 mm."""
    measured = torch.tensor([[0., .013, 0., .027]], dtype=torch.float64)
    positions = torch.zeros((1, 2, 3), dtype=torch.float64)
    rotations = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]], dtype=torch.float64)
    jacobian = torch.zeros((1, 1, 6, len(arm.joints)), dtype=torch.float64)
    jacobian[0, 0, 0, 2] = 1.
    data = SimpleNamespace(joint_pos=measured, body_link_pos_w=positions,
                            body_link_quat_w=rotations, body_link_jacobian_w=jacobian)
    robot = SimpleNamespace(data=data, is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(len(arm.joints))), list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, arm, options=IKOptions(integral_gain=gain), control_dt=1 / 30)
    return solver, measured, positions


def test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers(synthetic_arm):
    pytest.importorskip("isaaclab")
    errors = []
    for gain in (0., 1.):
        solver, measured, positions = biased_prismatic_servo(synthetic_arm, gain=gain)
        for _ in range(360):
            positions[0, 1, 0] = measured[0, 2]
            result = solver.solve([.05, 0., 0.])
            torch.testing.assert_close(result.joint_positions[:, [1, 3]], measured[:, [1, 3]], atol=0, rtol=0)
            assert abs(result.joint_positions[0, 2] - measured[0, 2]) <= .005 + 1e-12
            # Same fixed servo/load model in both runs. Only the encoded joint target changes.
            measured[0, 2] += .35 * (result.joint_positions[0, 2] - measured[0, 2] - .002)
        errors.append(abs(measured[0, 2].item() - .05))
    assert .002 < errors[0] < .0022
    assert errors[1] < 1e-5 and errors[1] < errors[0] / 100
    assert solver.integral_bias[0, 2] == pytest.approx(.002, abs=1e-5)


def test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits(synthetic_arm):
    pytest.importorskip("isaaclab")
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
    for _ in range(30):
        solver.solve([1., 0., 0.])  # A blocked servo must not integrate beyond the allowed command lead.
    torch.testing.assert_close(solver.integral_bias, torch.zeros_like(measured), atol=0, rtol=0)
    for _ in range(20):
        solver.solve([.001, 0., 0.])
    before = solver.integral_bias
    assert before[0, 2] > 0
    for _ in range(10):
        solver.solve([.001, 0., 0.], advance=False)
    torch.testing.assert_close(solver.integral_bias, before, atol=0, rtol=0)
    measured[0, 2] = .2
    positions[0, 1, 0] = .2
    command = solver.solve([1., 0., 0.])
    assert command.joint_positions[0, 2] == .2
    assert solver.integral_bias[0, 2] == 0  # Hard-stop projection unwinds retained outward bias.
    solver.reset()
    assert solver.integral_bias is None
    solver.solve([.1, 0., 0.], advance=False)
    assert solver.integral_bias is None


````

Insert:

````python
def biased_prismatic_servo(arm):
    """A CPU position servo with constant load: equilibrium is command minus 2 mm."""
    measured = torch.tensor([[0., .013, 0., .027]], dtype=torch.float64)
    positions = torch.zeros((1, 2, 3), dtype=torch.float64)
    rotations = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]], dtype=torch.float64)
    jacobian = torch.zeros((1, 1, 6, len(arm.joints)), dtype=torch.float64)
    jacobian[0, 0, 0, 2] = 1.
    data = SimpleNamespace(joint_pos=measured, body_link_pos_w=positions,
                            body_link_quat_w=rotations, body_link_jacobian_w=jacobian)
    robot = SimpleNamespace(data=data, is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(len(arm.joints))), list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, arm)
    return solver, measured, positions


def test_command_integration_removes_pd_load_residual_without_changing_grippers(synthetic_arm):
    pytest.importorskip("isaaclab")
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
    for _ in range(360):
        positions[0, 1, 0] = measured[0, 2]
        result = solver.solve([.05, 0., 0.])
        torch.testing.assert_close(result.joint_positions[:, [1, 3]], measured[:, [1, 3]], atol=0, rtol=0)
        assert abs(result.joint_positions[0, 2] - measured[0, 2]) <= .005 + 1e-12
        # Fixed servo/load model: only the encoded joint target compensates the 2 mm droop.
        measured[0, 2] += .35 * (result.joint_positions[0, 2] - measured[0, 2] - .002)
    assert abs(measured[0, 2].item() - .05) < 1e-5
    assert solver.integral_bias[0, 2] == pytest.approx(.002, abs=1e-5)


def test_command_preview_reset_and_lead_limit_at_joint_limits(synthetic_arm):
    pytest.importorskip("isaaclab")
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
    for _ in range(30):
        command = solver.solve([1., 0., 0.])  # A blocked servo never receives more than one step of lead.
        assert command.joint_positions[0, 2] - measured[0, 2] <= .005 + 1e-12
    before = solver.integral_bias
    assert before[0, 2] > 0
    for _ in range(10):
        solver.solve([.001, 0., 0.], advance=False)
    torch.testing.assert_close(solver.integral_bias, before, atol=0, rtol=0)
    measured[0, 2] = .2
    positions[0, 1, 0] = .2
    solver.reset()
    command = solver.solve([1., 0., 0.])
    assert command.joint_positions[0, 2] == .2
    assert solver.integral_bias[0, 2] == 0  # The runtime limit bounds the stored command.
    solver.reset()
    assert solver.integral_bias is None
    solver.solve([.1, 0., 0.], advance=False)
    assert solver.integral_bias is None


````

#### function `test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs`

Locator: line 101.

Remove:

````python
    solver = ArticulationIK(robot, synthetic_arm, position_offset_m=[0., .1, 0.], options=IKOptions(integral_gain=0))
````

Insert:

````python
    solver = ArticulationIK(robot, synthetic_arm, position_offset_m=[0., .1, 0.])
````

#### function `test_controller_pose_step_matches_reference_step_in_rotated_base_frame`

Locator: line 131.

Remove:

````python
    solver = ArticulationIK(robot, arm, options=IKOptions(integral_gain=0))
````

Insert:

````python
    solver = ArticulationIK(robot, arm)
````

### `CLAUDE.md`

#### bullet "Data generation uses ..."

Locator: line 48.

Remove:

````markdown
  Jacobian; `sim.datagen.ik` adds only the per-tick lead limits, per-joint step caps, runtime joint clamps
  and load compensation around it.
````

Insert:

````markdown
  Jacobian; `sim.datagen.ik` adds only the per-tick lead limits, per-joint step caps and runtime joint clamps
  around it, and starts each step from the previous dispatched command so steady load error is removed.
````

### `docs/08-data-generation.md`

#### section on `[arm.ik]`

Locator: line 85.

Remove:

````markdown
(damped least squares) on the articulation Jacobian in the arm-base frame, with
bounded load-error compensation. Failed reachability/grasping must be fixed in the expert,
````

Insert:

````markdown
(damped least squares) on the articulation Jacobian in the arm-base frame. Each step
starts from the previous dispatched command, so steady load error is removed, and the
command never leads the measured joints by more than one joint step. Failed reachability/grasping must be fixed in the expert,
````

#### Mimic section

Locator: line 135.

Remove:

````markdown
Mimic recovers source command poses through a bounded local kinematic inverse;
it does not add the scripted controller's load compensation a second time.
````

Insert:

````markdown
Mimic recovers source command poses through a bounded local kinematic inverse;
it does not repeat the scripted controller's command integration.
````

## B3-V. B3 verification and commit

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -m pytest tests -q -rs 2>&1 | tail -40
```

Expected: `425 passed` (425 - 2 deleted + 2 new tests), `22 subtests passed`, no `failed`, no `error`, skipped
equal to SKIP0.

```bash
./s2r check > outputs/spec-check-b3.log 2>&1; echo "exit=$?"; tail -30 outputs/spec-check-b3.log
./s2r datagen --config_path configs/datagen/scripted.yaml --destination /workspace/datasets/spec_b3_scripted --report /workspace/s2r/outputs/datagen/spec_b3_scripted.json; echo "exit=$?"
spec_report outputs/datagen/spec_b3_scripted.json
./s2r datagen --config_path configs/datagen/stack_two_cubes.yaml --destination /workspace/datasets/spec_b3_stack --report /workspace/s2r/outputs/datagen/spec_b3_stack.json; echo "exit=$?"
spec_report outputs/datagen/spec_b3_stack.json
./s2r datagen --config_path configs/datagen/mimic.yaml --source_dataset /workspace/datasets/spec_b3_scripted --destination /workspace/datasets/spec_b3_mimic --report /workspace/s2r/outputs/datagen/spec_b3_mimic.json; echo "exit=$?"
spec_report outputs/datagen/spec_b3_mimic.json
./s2r prepare --policy groot_n17 --dataset "$SPEC_DATA/spec_b3_scripted" --output "$SPEC_DATA/spec_b3_gr00t" --image s2r-train:local; echo "exit=$?"
spec_prepared "$SPEC_DATA/spec_b3_gr00t/preparation_report.json"
./s2r verify --policy groot_n17 --dataset "$SPEC_DATA/spec_b3_gr00t" --hf-home "$S2R_DOC_HF_HOME" --image s2r-train:local; echo "exit=$?"
```

Expected: as in B1-V (`exit=0` everywhere, `1 True` three times, `passed so_arm_n17`).

If every check matches: `git commit -a -m "Integrate scripted IK on the previous command instead of an integral bias"`.

If any check fails: run
`git checkout -- src/sim/datagen/ik.py src/sim/datagen/scripted.py tests/core/test_ik.py CLAUDE.md docs/08-data-generation.md`
and report the failing output to the owner. B1 stays committed.

---

# Not in this spec

Owner decisions from audit section 7, not applied:

- 7.1 `tools/diagnostics/` (remaining 10 files): keep, document or delete.
- 7.2 `tasks/_template/` duplicating `tasks/stack_two_cubes/`.
- 7.3 `policies/groot_n17/offline_cosmos.py` duplicating training-side code.
- 7.4 `src/sim/solver_numerics.py` version guard.
- 7.5 `[scene.anchors.workspace] xy_min / xy_max` unused.
- 7.6 Site-specific paths in shipped configs and eval sets.
- 7.7 Arm and task names in generic code (`"so101-spawn"`, `"so101_real_rollout_v1"`, `"blue_cube"` fallbacks).
- 7.8 Tests that execute AST-extracted source.
- 7.9 MolmoAct2 eval config duplication.
- 7.10 Git LFS.

Also not applied:

- 5.5 Keep or split `docs/restructure-status.md`.
- 4.4 second half: whether `check_molmo_export_server.py` and `check_molmo_optimizer_offload.py` stay.
- The smoke evaluation suggested in audit section 9. It needs a served model. The `[SMOKE_RESULT]` line loses
  `grasp_diagnostics`, `closest_tool_error_mm` and `closest_closing_tool_error_mm`; no file in the repository
  parses those keys; parsers outside the repository are not known.
- Adding `usd-core`, `lerobot`, `gymnasium` and `mujoco-warp` to the `host` group (the brief named only `av` and
  `warp-lang`).
- The phase sentence at the top of `CLAUDE.md` (lines 3 and 4); the audit's 5.4 row names only the closing paragraphs.
- A Newton-based `ArmKinematics`: it needs either a custom `NewtonIKObjective` with a per-axis rotation residual or
  an iterated `NewtonDifferentialIKController` (6-value `axis_weight`) with Newton FK. Both need a decision and an
  API check inside the container.
