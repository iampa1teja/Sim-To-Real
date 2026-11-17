# WP2 — arm kinematics, gripper, torch-free raw mapping (implementation prompt)

## Goal

Make every arm-dependent number the v4 expert needs come from `arms/<arm>/` files instead of SO-101 literals: a serial-chain FK / Jacobian / damped-least-squares IK loaded from `arms/<arm>/kinematics.json`, the gripper gap <-> jaw angle table, a numpy-only version of the raw-unit joint mapping, the new `arm.toml` tables (validated at load time), the optional per-arm block in `real_setup.json`, and the pure part of the USD -> `kinematics.json` extractor. Everything must work for any n-DoF arm; the tests run on two synthetic arms (`fx_arm6`: 6 + gripper, `fx_arm5`: SO-101-like 5 + jaw) and never on SO-101 numbers, except in explicit SO-101 regression tests.

Prerequisite: WP0 (`src/core/generate/errors.py`) is implemented. WP2 imports `ConfigError`, `SchemaError`, `KinematicsError` from there and `Pose` from `core.tasks.base`.

## Files

Create:

| File | Content |
|---|---|
| `src/core/arms/kinematics.py` | `JointDef`, `KinematicChain`, `IKWeights`, `IKResult`, `load_kinematics_json` (§5.2, §5.3) |
| `src/core/arms/gripper.py` | `Gripper` (§5.4) |
| `src/core/arms/raw_mapping.py` | `RawMapping` (§5.4), numpy only |
| `src/sim/generate/extract_kinematics.py` | `build_kinematics`, `main` (§5.2) |

Change:

| File | Change |
|---|---|
| `src/core/arms/mapping.py` | add `JointMapping.to_raw_np`, `.clamp_radians_np`, `.as_raw_mapping`; nothing else changes |
| `src/core/arms/joint_limits.py` | `runtime_joint_limits(setup=None, *, arm=None)` with the per-arm block; names from `load_arm_spec(arm)`; no module-level import of `core.arms.joint_calibration` |
| `src/core/arms/spec.py` | `load_arm_spec` validates the v4 tables (private helper) and raises `ConfigError` |
| `arms/so101/arm.toml` | add the v4 keys listed below |

Do NOT touch: anything under `tests/` (test files, `tests/v4_support/`, `tests/conftest.py`, and the fixtures `tests/fixtures/arms/fx_arm{5,6}/`, `tests/fixtures/scenes/fx_scene/`, `tests/fixtures/v4_wp2/`, which are already committed), `docs/`, `scenes/`, `tasks/`, `pyproject.toml`, `src/core/generate/*` (WP0/WP5), `src/core/tasks/*` (WP1), `src/sim/generate/dr_variants.py`, `src/core/arms/joint_calibration.py`. Add no public names beyond §2's WP2 row (helpers get a leading underscore).

## Read first

`docs/v4/contracts.md`: §1.1 (quaternion order: XYZW in memory, WXYZ on disk; ee vs TCP), §1.2 (joint order), §1.3 (float64, read-only arrays), §1.5 (errors), §1.6 (imports: no torch in new modules, no Isaac at module level), §2 (WP2 row), §5 (all), §13. Also `tests/sim/check_grasp_stability.py` lines 14–85 (the FK formula and USD reading this package generalises) and `tests/v4_support/fx_arms.py` (the synthetic arms and the independent reference FK the tests compare with).

## Interfaces and the decisions the tests pin

### `load_kinematics_json(path) -> dict` (§5.2)

Reads the file and returns the parsed dict unchanged. Missing file: `FileNotFoundError`. Everything else wrong raises `SchemaError`: invalid JSON or empty file; top level not an object; missing `schema_version` / `arm` / `base_link` / `source` / `joints`; `schema_version != 1`; `source` not an object of strings (`usd`, `usd_sha256`, `git_commit`); `joints` not a non-empty list; a joint that is not an object or misses any of `name, type, parent, child, axis, parent_frame, child_frame, limits_rad`; duplicate joint names; `type != "revolute"`; `axis` not exactly one of `[1,0,0]`, `[0,1,0]`, `[0,0,1]` (numbers; a string, a length-2 list, `[0,0,2]` or `[0.6,0.8,0]` are errors); a frame without `position` (3 finite numbers) and `quat_wxyz` (4 finite numbers); `limits_rad` not 2 finite numbers with `lo < hi`. Any NaN/inf anywhere numeric is an error. (Unknown extra keys are not tested; accepting them is fine.)

### `KinematicChain` (§5.3)

- `from_dict(data, *, q_names, chain_joints, ee_link, tcp_position_m, tcp_quat_wxyz, approach_axis, closing_axis)`. All keywords required. `data` is a kinematics.json dict. Construction problems raise `KinematicsError` and **never `SchemaError`** (do not run the file validator inside `from_dict`, or wrap it): a chain joint missing from `data` or from `q_names`; a link that is the child of two joints, or a loop; joints not connected to `base_link`; `ee_link` not a link of the tree; an axis that is not a unit basis vector.
- `from_arm(name=None)`: `spec, arm_dir = load_arm_spec(name)`; `q_names = [arm.joints].names`; `chain_joints = [arm.expert].arm_joints`; `[arm.kinematics]`: `file` (relative to `arm_dir`, default `"kinematics.json"`), `ee_link` (default `links.ee`), `tcp_position_m`, `tcp_quat_wxyz` (default `[1,0,0,0]`), `approach_axis`, `closing_axis`. No `[arm.kinematics]` table → `ConfigError`. The file is read with `load_kinematics_json` (so a malformed file → `SchemaError`).
- Attributes: `q_names`, `chain_joints` (tuples); `lower`, `upper`: float64 `(J,)` USD limits from `limits_rad`, indexed by `q_names` (not by the file's joint order). A `q_names` joint absent from the file gets `-inf` / `+inf` (not tested).
- FK: `T_child = T_parent · F(parent_frame) · R(axis, q) · F(child_frame)⁻¹`, base = identity. **Every** joint named in `q_names` uses its `q` value (so the gripper joint moves the jaw link in `link_poses`); joints in the file but not in `q_names` use 0. Normalise quaternions before use.
- `fk(q, frame="tcp")`: `(4,4)` float64. `"ee"` = pose of `ee_link`; `"tcp"` = `ee · F(tcp_position_m, tcp_quat_wxyz)`. Any other frame → `ValueError`. `q` must have shape `(J,)` (lists accepted) else `ValueError`; never mutate the input. Tolerance vs the reference: 1e-9.
- `link_poses(q) -> dict[str, (4,4)]`: every link of the tree including `base_link`, exactly the reference's key set.
- `fk_pose(q, frame="tcp") -> Pose`: position tuple of 3 floats, `quat_xyzw` unit **XYZW** (sign free). `tcp_from_ee(ee: Pose) -> Pose` = `ee ∘ tcp`.
- `jacobian(q, frame="tcp")`: `(6, J)` float64, rows `[v; ω]` in the base frame, `v` the velocity of the chosen frame's origin. Columns of joints not in `chain_joints` are exactly 0 (also for chain joints that are not ancestors of the ee link). Must match central differences (h = 1e-6) within 1e-5. Wrong `q` shape → `ValueError`.
- `JointDef`: a public frozen dataclass describing one parsed joint (suggested fields: `name, parent, child, axis, parent_frame (4x4), child_frame (4x4), lower, upper`); only its existence is tested.

### IK (§5.3)

`ik(target: Pose, seed_q, *, weights=None, lower=None, upper=None, max_iters=200, tol_pos_m=5e-4, tol_rot_rad=0.02) -> IKResult`

- `IKWeights(position=1.0, axis=0.3, roll=0.1)`: frozen; a negative or non-finite weight → `ValueError`.
- `IKResult(q, success, position_error_m, orientation_error_rad, axis_error_rad, roll_error_rad, iterations)` — exactly these fields in this order; frozen; `q` float64 `(J,)`, C-contiguous, read-only; errors are Python `float`, `success` a Python `bool`, `iterations` a Python `int`. The tests apply `v4_support.assert_frozen` (use `eq=False` or a custom `__eq__` because of the array).
- Definitions (the tests recompute them from `fk(result.q)` and require the reported values to match: position 1e-9, orientation/axis 1e-5, roll 1e-4 when the axis error < 1e-3). With `R` the TCP rotation at `result.q`, `R*` the target rotation, `a = R·approach`, `a* = R*·approach`, `c = R·closing`, `c* = R*·closing`:
  - `position_error_m = ‖p − p*‖`
  - `orientation_error_rad` = rotation angle of `Rᵀ R*` in `[0, π]`
  - `axis_error_rad = arccos(clip(a·a*))`
  - `roll = atan2(a*·(c* × c_p), c*·c_p)` with `c_p = c − (c·a*) a*`; `roll_error_rad = |roll|`
  - residual for the solver: `[w_p (p − p*); w_a (a × a*); w_r · roll]`.
- `success = position_error_m ≤ tol_pos_m and (w_a == 0 or axis_error_rad ≤ tol_rot_rad) and (w_r == 0 or roll_error_rad ≤ tol_rot_rad)` — a check is on for **any** positive weight, even `1e-9`.
- Bounds: effective box `[max(lower, self.lower), min(upper, self.upper)]` (`lower`/`upper` are `(J,)` arrays that may hold ±inf; `None` = chain limits). Every chain joint of the result lies inside the box (exactly, `>=`/`<=`), also when the seed lies outside it (clip the seed). An empty box → `ValueError`. Joints not in `chain_joints` are copied from `seed_q` **verbatim** (even if outside limits).
- Unreachable targets (out of workspace, or reachable only outside the user box): return `success=False` with the honest errors of the returned (in-box) `q`. Never raise, never report pre-clip errors.
- NaN (or inf) in target position, target quaternion or seed → `ValueError`. Normalise the target quaternion.
- **Convergence is tighter than the success tolerance.** With the default tolerances the tests require, for `ik(fk(q))` from `seed = q + U(±0.3)` (q at least 0.35 rad inside the limits): `success`, position ≤ 2e-4 m on both arms, orientation ≤ 1e-3 rad on `fx_arm6`. So do not stop when `success` first becomes true: iterate until the residual stops decreasing (the reference stops at cost < 1e-20 or step < 1e-13).
- Deterministic: same inputs → bit-identical outputs, also across chain instances (no randomness).
- `seed_q` is never mutated.
- Algorithm that passes the tests (reference implementation): DLS `dq = −Jᵀ(J Jᵀ + λ²I)⁻¹ r` with λ = 1e-3 on the chain joints, step clamped to 0.3 rad (max-abs), result clipped to the box, keep the best-cost iterate; if the best cost is not ≈ 0 (> 1e-16), run `scipy.optimize.least_squares` (TRF, `bounds=box`, analytic Jacobian, `xtol=ftol=gtol=1e-15`) from the best DLS iterate and from the clipped seed, and keep the lowest cost. Joints whose box has `lo == hi` are held fixed (TRF needs `lo < hi`). Analytic residual Jacobian: position rows `w_p J_v`; axis rows `w_a [a*]× [a]× J_ω`; roll row `w_r (k·ds − s·dk)/(s² + k²)` with `s = a*·(c* × c_p)`, `k = c*·c_p`, `dc_i = ω_i × c`, `ds_i = a*·(c* × dc_i)`, `dk_i = c*·dc_i`. The whole IK test file runs in about 10 s with this; keep a single solve well under 0.2 s.

### `Gripper` (§5.4)

- `from_table(table, q_names)`: required keys `joint, open_rad, closed_rad, opening_table, max_object_width_m`; missing / wrong type → `ConfigError`. `joint` must be in `q_names` (`index = q_names.index(joint)`), else `ConfigError`. `opening_table`: ≥ 2 `[angle_rad, gap_m]` rows, angles strictly increasing, gaps strictly monotone (**increasing or decreasing**), else `ConfigError`. `angles_rad` / `gaps_m` are tuples of Python floats.
- `gap_from_angle(rad)`: piecewise linear, clamped to the end rows. `angle_from_gap(gap)`: the inverse (also for a decreasing table), clamped; round trips ≤ 1e-9; return Python `float`.
- `from_arm(name=None)` = `from_table([arm.gripper], [arm.joints].names)`; no `[arm.gripper]` → `ConfigError`. Plain frozen dataclass (default `==`).

### `RawMapping` (§5.4, numpy only, no torch)

- Same semantics as `JointMapping.from_setup`: `from_tables(names, mapping, runtime, gripper_joint)` — `names` are the LeRobot names (joint-vector order); `gripper_joint` is the **LeRobot** gripper name (`[arm.lerobot].gripper_joint`) and must be in `names`; `runtime` must name exactly `names`; the effective calibration mapping is `runtime` overlaid by `mapping` (names missing from `mapping` use the runtime bounds); names in `mapping` not in `names` → `ValueError`; all bounds finite with `max > min` → else `ValueError`.
- Fields: `names` tuple; `mins_deg`, `maxs_deg` (effective mapping), `clamp_mins_deg`, `clamp_maxs_deg` (runtime): float64 read-only; `gripper_mask` bool read-only. Frozen (tests run `assert_frozen`); use `eq=False`.
- `to_radians(raw)`: arm `norm = (raw + 100)/200`, gripper `norm = raw/100`, `deg = mins + norm·(maxs − mins)`, clipped to the runtime bounds, then radians. `to_raw(rad)`: inverse formula, **no clamp**. `clamp_radians(rad)`: clip to `[lower_rad, upper_rad]`. All accept `(..., J)` (batched) and raise `ValueError` if the last axis is not J. `lower_rad`/`upper_rad` = `deg2rad(clamp_*_deg)`.
- `from_setup(setup=None, *, arm=None)`: `setup = load_setup()` when `None`; `spec = load_arm_spec(arm)`; `use_degrees = true` → `ValueError`; block = `setup["arms"][name]` if `setup` has an `"arms"` mapping containing `name = arm or core.arms.spec.ARM_NAME`, else `setup` itself; then `from_tables(spec["lerobot"]["names"], block.get("sim_joint_mapping", {}), block["runtime_joint_limits"], spec["lerobot"]["gripper_joint"])`. So for the repo scene it equals `JointMapping.from_setup(SimpleNamespace(mapping=setup["sim_joint_mapping"]))`, which is how the sim builds its mapping.
- Agreement with torch (float32) is checked to 1e-5 rad and 1e-4 raw units (float32 rounding of values of magnitude 100).

### `JointMapping` additions (`mapping.py`)

`to_raw_np(radians)`, `clamp_radians_np(radians)` (numpy in, numpy float64 out) and `as_raw_mapping() -> RawMapping` built from the tensors cast to float64 (`names`, bounds, `gripper_mask` exactly equal to the tensors). Import `RawMapping` inside the method or at the top; `raw_mapping.py` itself must never import torch or `mapping.py`.

### `joint_limits.py`

`runtime_joint_limits(setup=None, *, arm=None)`: `setup = load_setup()` when `None`; names = `load_arm_spec(arm)[0]["lerobot"]["names"]`; bounds from the per-arm block when present (same selection rule as `RawMapping.from_setup`), else the top-level `runtime_joint_limits`; same validation as today (exact name set, finite, increasing → `ValueError`); returns `{name: dict}` in `names` order. `joint_name_pairs` is unchanged. Remove the module-level `from .joint_calibration import FOLLOWER_JOINTS`: a test imports `core.arms.{raw_mapping,kinematics,gripper,joint_limits,spec}` in a fresh interpreter with `S2R_ARMS_DIR` / `S2R_SCENE_DIR` pointing at an empty folder and requires success with no `torch`/Isaac module loaded — so nothing may read an arm, scene or task at import time.

### `spec.py` — v4 table validation

`load_arm_spec(name, variant)` keeps its signature and return value; after the variant merge it validates the v4 keys **that are present** and raises `ConfigError` (message names the file). It injects no defaults: the returned dict equals the file content (tests compare it with `==`). Absent keys/tables are fine (container-measured values arrive later; consumers like `from_arm` raise when they need a missing key). Checks the tests exercise:

- `links.moving_jaw`, `links.moving_jaw_mesh`: strings.
- `[arm.kinematics]`: `file` string; `ee_link` must equal `links.ee`; `tcp_position_m` 3 finite numbers; `tcp_quat_wxyz` 4 finite numbers (unit within 1e-6); `approach_axis`, `closing_axis` 3 finite numbers, unit within 1e-6, mutually perpendicular within 1e-6.
- `[arm.gripper]`: `joint` in `joints.names`; `open_rad`/`closed_rad` finite; `max_object_width_m > 0`; `opening_table` as for `Gripper`.
- `[arm.expert]`: `arm_joints` distinct names from `joints.names`, not containing `gripper.joint`; `home_rad` a table of `joints.names` → finite; `max_joint_speed_rad_s` a positive float or a table of `joints.names` → positive floats; `max_gripper_speed_rad_s`, `max_ee_speed_m_s`, `ik_tol_pos_m`, `ik_tol_rot_rad` > 0; `ik_position_weight`, `ik_axis_weight`, `ik_roll_weight` ≥ 0.

Import `ConfigError` from `core.generate.errors` (stdlib-only module, safe at top level).

### `arms/so101/arm.toml` additions

Add to `[arm.links]`: `moving_jaw = "jaw"`, `moving_jaw_mesh = "moving_jaw_so101_v1"`. Add the tables (keep every existing line and comment unchanged; the regression test compares the parsed spec minus these additions with `tests/fixtures/v4_wp2/so101_arm_spec_before_v4.json`, for the base spec and both variants):

```toml
[arm.kinematics]
file = "kinematics.json"
ee_link = "gripper"
tcp_quat_wxyz = [1.0, 0.0, 0.0, 0.0]
# tcp_position_m, approach_axis, closing_axis: container-measured (extract_kinematics run), not yet committed

[arm.gripper]
joint = "Jaw"
max_object_width_m = 0.045
# open_rad, closed_rad, opening_table: container-measured, not yet committed

[arm.expert]
arm_joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
max_joint_speed_rad_s = 1.5
max_gripper_speed_rad_s = 3.0
max_ee_speed_m_s = 0.15
ik_position_weight = 1.0
ik_axis_weight = 0.3
ik_roll_weight = 0.1
ik_tol_pos_m = 0.0005
ik_tol_rot_rad = 0.02
```

No `home_rad` (the default is `[arm.initial_state.joint_positions]`). Do not commit `arms/so101/kinematics.json` or the measured values from the host: they come from the container run (`s2r arm-kinematics`, WP8 adds the CLI row) and a later commit. If you add them, the test only checks they are well-formed.

### `sim/generate/extract_kinematics.py`

- `build_kinematics(joints, base_link, arm, source) -> dict`, pure (inputs not mutated). Input rows: `name, parent, child, axis ("X"|"Y"|"Z"), pos0, rot0_wxyz, pos1, rot1_wxyz, lower_deg, upper_deg`. Output: exactly the §5.2 document — keys `schema_version` (1), `arm`, `base_link`, `source` (a copy), `joints`; each joint exactly `name, type ("revolute"), parent, child, axis ([1.0,0.0,0.0] etc.), parent_frame {position, quat_wxyz}, child_frame {position, quat_wxyz}, limits_rad [math.radians(lower), math.radians(upper)]`, all numbers plain Python floats (JSON-serialisable with `allow_nan=False`). It must pass `load_kinematics_json`.
- Joint order: topological from `base_link`, parents before children; an input already in chain order is kept as is (the reference does repeated passes over the pending list, taking ready rows in input order).
- `KinematicsError` for an axis letter other than X/Y/Z, a row not connected to `base_link`, or a child with two parents; non-finite or non-increasing limits are also errors (not tested).
- `main()`: argparse (`--arm`, optional `--usd`, `--out`), opens the USD lazily (all `pxr`/Isaac imports inside the function, as in `tests/sim/check_grasp_stability.py::_forward_kinematics`), builds rows from `UsdPhysics.RevoluteJoint` (`localPos0/1`, `localRot0/1` → WXYZ, `axis`, `lowerLimit/upperLimit` in degrees), fills `source` (`usd` relative to the arm dir, `usd_sha256` via `core.utils.hashing.file_sha256`, `git_commit` or `"unknown"`), and writes `arms/<arm>/kinematics.json` atomically (a private temp-file + `os.replace` helper; do not import `sim.generate.dr_variants`, which reads the scene at import). It is container-only and untested. No module-level import of `isaaclab`, `isaacsim`, `omni`, `carb`, `pxr`, `warp`, `lerobot` or `torch` (checked by AST and in a fresh interpreter).

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp2_kinematics.py      # kinematics.json validation, construction errors, FK, link poses, Jacobian, from_arm, fixtures
tests/core/test_v4_wp2_ik.py              # IK round trip, limits, unreachable, NaN, determinism, 5-DoF reduced orientation
tests/core/test_v4_wp2_gripper.py         # Gripper table, round trip, monotonicity, errors, from_arm
tests/core/test_v4_wp2_raw_mapping.py     # RawMapping vs torch JointMapping, setup blocks, import hygiene, SO-101 regression
tests/core/test_v4_wp2_arm_spec.py        # arm.toml v4 tables: validation, unchanged loading, SO-101 values and regression
tests/sim/test_v4_wp2_extract_kinematics.py
```

Run from the repo root (host group, Python 3.11):

```
uv run --group host pytest tests/core/test_v4_wp2_*.py tests/sim/test_v4_wp2_*.py -q
```

Without uv: `PYTHONPATH=src python -m pytest tests/core/test_v4_wp2_*.py tests/sim/test_v4_wp2_*.py -q` (needs numpy, scipy, torch, pytest).

Before implementation every file fails at collection with `ModuleNotFoundError` (`core.arms.kinematics`, `core.arms.gripper`, `core.arms.raw_mapping`, `sim.generate.extract_kinematics`). They are never skipped.

## What must keep working

- The full suite: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors` — no test that passed before may fail (pre-existing collection errors: `tests/sim/test_app.py`, `tests/test_code_explorer_*.py`; other WPs' test files may error until those WPs land).
- `JointMapping.from_setup` and its torch methods (`to_radians`, `to_raw`, `to_radians_np`, `raw_from_actions`) are unchanged; `runtime_joint_limits()` / `runtime_joint_limits(setup)` without an `"arms"` block return exactly today's result; `joint_name_pairs` unchanged; existing callers (`sim/envs/mdp/resets.py`, `core/calibration/joint_angle_check.py`, `core/scripts/real_rollout.py`) work without changes.
- `load_arm_spec("so101")` and its variants return exactly the old dict plus the new keys.

## Rules

- Do not edit, skip, weaken or delete any test, fixture or `tests/v4_support/` file.
- New `src/core` modules import only the standard library, numpy, scipy and existing `core` modules; never torch (only `mapping.py` keeps torch). Python 3.11 syntax only.
- No SO-101 literal in `kinematics.py`, `gripper.py`, `raw_mapping.py`, `joint_limits.py`, `spec.py` or `extract_kinematics.py`; read names from `load_arm_spec(arm)` at call time, never from `FOLLOWER_JOINTS` / `ARM_NAME` constants captured at import (resolving `arm=None` through `core.arms.spec.ARM_NAME` at call time is fine).
- If a test seems to contradict `docs/v4/contracts.md` or another test, stop and report the test name, the contract clause and your reasoning instead of working around it.

## Definition of done

1. The six test files pass (no skips or xfails).
2. The full suite is no worse than before.
3. Only the files listed under "Files" changed.
