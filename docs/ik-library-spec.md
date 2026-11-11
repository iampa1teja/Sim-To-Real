# Spec: replace the custom IK math with library IK, for any arm

Audience: the implementer (Codex). Branch: work on `v2`, starting from `6624e2a` (Part A applied).
This file replaces Part B of `docs/implementation-spec.md`. Part B there is superseded; do not apply it.

## 1. Goal and rules

Goal: `src/sim/datagen/ik.py` keeps no inverse-kinematics or forward-kinematics math of its own. All IK and FK
math comes from libraries already in the `s2r-sim` image. The same code must serve any single serial arm with
one end effector: 5, 6 or 7 actuated arm joints, revolute or prismatic, with any number of gripper joints.

Rules:

- Library math only. Allowed in our code: frame bookkeeping (base-frame expression, TCP offset), reading
  `arm.toml`, per-step caps, clamps to declared limits, convergence flags. Not allowed: our own least squares,
  rotation-error formulas, Jacobians, USD joint-chain parsing or FK.
- Public surface used by callers stays: `IKOptions`, `IKResult`, `ArticulationIK.current_pose/solve/reset/
  integral_bias/position_offset_m/joint_ids/robot/options`, `ArmKinematics.from_arm/fk/inverse`. `mimic.py`
  needs no changes; `scripted.py` changes only its construction lines.
- `arm.toml` stays the only place an arm declares IK settings. No arm names in generic code.
- Do not change production physics, expert waypoints, task files, the dataset action contract (absolute joint
  targets in device units) or `_compensate_load` behaviour (its removal is a separate, later experiment).
- Out of scope: bimanual, Newton's IK solver, dataset regeneration, GR00T training.

## 2. Libraries used (read at Isaac Lab `v3.0.0`, Newton `1.7.0`)

| Need | Library call | Notes |
| --- | --- | --- |
| One IK step, any joint count | `isaaclab.controllers.DifferentialIKController` + `DifferentialIKControllerCfg` | `command_type="pose"` or `"position"`, `use_relative_mode=False`, `ik_method` `"dls"` or `"adaptive_dls"`, `ik_params`, per-axis `orientation_weight=(wx, wy, wz)` (weight 0 drops an axis), `joint_limit_avoidance_gain/margin` (null-space, for redundant arms) with `set_joint_pos_limits(lower, upper)`. `compute(ee_pos, ee_quat, jacobian, joint_pos)` returns `joint_pos + delta`. Quaternions XYZW. Frame-agnostic: pose and Jacobian only need to share a frame |
| Frame helpers | `isaaclab.utils.math`: `subtract_frame_transforms`, `quat_apply`, `quat_apply_inverse`, `quat_mul`, `quat_inv`, `compute_pose_error`, `axis_angle_from_quat`, `quat_from_angle_axis`, `matrix_from_quat`, `quat_from_matrix`, `make_pose` | Use these instead of the hand-written `_rotate`, `_rotation_error`, `_numpy_rotation`, `_numpy_quaternion` |
| Live Jacobian | `robot.data.body_link_jacobian_w` | Already used today; link-origin linear rows, world frame |
| Offline FK (no simulator) | `newton.ModelBuilder().add_usd(path, floating=False)`, `builder.finalize(device="cpu")`, `model.state()`, `newton.eval_fk(model, joint_q, joint_qd, state)` | Reads the arm USD; `state.body_q[i]` is the body transform (position, XYZW). Bodies and joints are named by `model.body_label` / `model.joint_label` (prim paths) |
| Offline Jacobian | `newton.eval_jacobian(model, state)` | Shape `(articulations, links*6, dofs)`; rows per link are `(v_com_world, omega_world)`. Shift linear rows to the TCP point: `v_tcp = v_com + omega x (p_tcp - p_com)` (frame bookkeeping, allowed) |

Newton's own IK solver (`NewtonIKSolver`) is not used: its rotation objective has one scalar weight, so it cannot
leave a single axis free for a 5-joint arm.

## 3. Step 0: gates before any edit

Run in the container. Stop and report if a gate fails.

### G0.1 Library versions and API

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -c "import importlib.metadata as md, inspect, newton; from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg as C; from isaaclab.controllers.differential_ik import DifferentialIKController as D; print(md.version('isaaclab'), newton.__version__); print([k for k in ('orientation_weight','joint_limit_avoidance_gain','joint_limit_avoidance_margin','ik_params') if k in C.__dataclass_fields__]); print(hasattr(D,'set_joint_pos_limits'), inspect.signature(D.compute)); print('floating' in inspect.signature(newton.ModelBuilder.add_usd).parameters, hasattr(newton,'eval_fk'), hasattr(newton,'eval_jacobian'))"
```

Pass: all four config fields listed, `True` for `set_joint_pos_limits`, and `True True True` on the last line.
Record the printed versions in the commit message.

### G0.2 Newton reads the arm USD

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -c "import newton; from core.arms.spec import load_arm_spec; a,_=load_arm_spec('so101'); b=newton.ModelBuilder(); b.add_usd(str(a.usd_path), floating=False); m=b.finalize(device='cpu'); print(m.articulation_count, m.joint_coord_count, m.joint_dof_count); print([l.rsplit('/',1)[-1] for l in m.joint_label]); print([l.rsplit('/',1)[-1] for l in m.body_label])"
```

Pass: one articulation; every name in `arm.usd_joint_names` appears in the joint list; `arm.links.base` and
`arm.links.ee` appear in the body list. If `add_usd` fails on the arm USD, stop: section 6.3 cannot proceed and
`ArmKinematics` stays as it is (report the error).

### G0.3 Crash triage (the earlier B1 recheck crashed in `tests/sim/test_gripper_collisions.py` setup)

On unchanged `v2`:

```bash
docker exec -w /workspace/s2r s2r-sim bash -lc 'PYTHONPATH=src s2r-entrypoint python -m pytest tests/sim/test_gripper_collisions.py -q; echo exit=$?'
docker exec -w /workspace/s2r s2r-sim bash -lc 'PYTHONPATH=src s2r-entrypoint python -c "import isaaclab.controllers.differential_ik" && PYTHONPATH=src s2r-entrypoint python -m pytest -p no:cacheprovider tests/sim/test_gripper_collisions.py -q; echo exit=$?'
docker exec -w /workspace/s2r s2r-sim bash -lc 'PYTHONPATH=src s2r-entrypoint python - <<EOF
import isaaclab.controllers.differential_ik, pytest, sys
sys.exit(pytest.main(["-q", "tests/sim/test_gripper_collisions.py"]))
EOF
echo exit=$?'
```

Decide:

- First command crashes: the crash is not caused by IK. Record it in the commit message and continue.
- Only the third command crashes (controller imported in the same process before the USD tests): keep every
  `isaaclab.controllers` import inside functions in `ik.py` (already required below), and put the tests that
  call the controller in `tests/sim/test_ik_controller.py` (section 7), not in `tests/core/`. Re-run the full
  suite after the change; it must not crash.
- None crash: continue as written.

### G0.4 Baseline reports (unchanged code, same seed)

```bash
./s2r datagen --config_path configs/datagen/scripted.yaml --destination <datasets>/ik_base_pick --report <datasets>/ik_base_pick.report.json
./s2r datagen --config_path configs/datagen/stack_two_cubes.yaml --destination <datasets>/ik_base_stack --report <datasets>/ik_base_stack.report.json
```

Keep both reports. They are the comparison for section 8.

## 4. Files touched

| File | Change |
| --- | --- |
| `src/core/arms/spec.py` | `IKSpec` gains three optional fields; parser validates them |
| `arms/_template/arm.toml` | Commented examples for 5-, 6- and 7-joint arms |
| `arms/so101/arm.toml` | No change (defaults reproduce current behaviour) |
| `src/sim/datagen/ik.py` | Rewritten internals; same public surface; optional arm hook |
| `src/sim/datagen/scripted.py` | Construct IK through `make_ik` / `make_kinematics`; report fields |
| `src/sim/datagen/mimic.py` | No change |
| `tests/core/test_arm_spec.py` | New validation cases |
| `tests/core/test_ik.py` | Reduced to pure helpers and spec mapping |
| `tests/sim/test_ik_controller.py` | New: controller, FK and inverse tests on synthetic 5/6/7-joint USD chains |
| `tests/sim/test_mimic.py` | No change expected (imports `IKOptions` only); run it |
| `docs/02-add-an-arm.md`, `docs/08-data-generation.md`, `CLAUDE.md` | Text updates (section 9) |

## 5. `src/core/arms/spec.py`

Imports: no change.

### `IKSpec` (dataclass)

Keep `orientation_axes` and `position_offset_m`. Add:

| Field | Type, default | Meaning |
| --- | --- | --- |
| `method` | `str = "dls"` | Passed as `ik_method`. Allowed: `"dls"`, `"adaptive_dls"` |
| `joint_limit_avoidance_gain` | `float = 0.0` | Null-space joint-limit avoidance. Only has an effect when the arm has more arm joints than constrained task rows (for example 7 joints with full pose) |
| `joint_limit_avoidance_margin` | `float = 0.3` | Distance from a limit where avoidance starts; radians for revolute, metres for prismatic (the library uses one value for all joints) |

### arm parser (the block that builds `ik = IKSpec(...)`)

Read the three keys from `[arm.ik]` with the defaults above. Validate:
- `method` in `{"dls", "adaptive_dls"}`; otherwise `ValueError("ik.method must be dls or adaptive_dls")`.
- gain finite and `>= 0`; margin finite and `> 0`.
- If `joint_limit_avoidance_gain > 0` and the number of `role == "arm"` joints is not greater than
  `3 + len(orientation_axes)`, raise `ValueError` saying the arm has no spare joint for limit avoidance.

No Isaac Lab import here (rule: `src/core/` imports no simulator code).

## 6. `src/sim/datagen/ik.py`

### Imports

| Remove | Keep | Add |
| --- | --- | --- |
| `from dataclasses import replace` (only if unused after the rewrite) | `dataclass`, `math`, `numpy as np`, `torch` | none at module level |

Simulator imports stay inside functions so `import sim.datagen.ik` works without Kit:
`isaaclab.controllers` and `isaaclab.utils.math` inside `_JointStep.__init__` / `_JointStep.solve` and
`ArticulationIK` methods; `newton` and `warp` inside `ArmKinematics.from_arm` and its methods;
`core.arms.spec.load_arm_utils` inside `make_ik` / `make_kinematics`.

### Delete

`_batch`, `_quaternion`, `_rotate`, `_inverse_rotate`, `_rotation_error`, `_bounded_norm`, `solve_damped_step`,
`_numpy_rotation`, `_numpy_quaternion`, and the whole USD chain parser / finite-difference code inside
`ArmKinematics` (`__init__` graph search, `fk`, `jacobian`, `inverse` bodies). The four repeated
"caps and limits per arm joint" list comprehensions are replaced by one helper.

Keep `_tensor` (converts warp/torch-like arrays to torch).

### Module docstring

Say: library IK (`DifferentialIKController`) for every arm; Newton FK/Jacobian for offline (command) kinematics;
this module only expresses frames, applies the TCP offset, caps each step and clamps to declared limits.

### `IKOptions` (unchanged fields)

Keep all fields and validation. `damping` now feeds `ik_params` (`lambda_val` for `dls`, `lambda_min` for
`adaptive_dls`); `orientation_weight` is the weight on each declared axis.

### `IKResult` (unchanged)

### New helper `_arm_bounds(arm, options) -> dict`

Single source for per-joint numbers. Returns, for the joints with `role == "arm"` in manifest order:
`ids` (indices into the action vector), `step_caps` (`max_joint_step_rad` or `max_joint_step_m` by joint type),
`bias_caps` (`max_integral_bias_rad` / `_m`), `lower`, `upper` (revolute limits converted degrees to radians,
prismatic kept in metres). Raises `ValueError("IK requires at least one joint with role arm")` when empty.
Used by `_JointStep`, `ArticulationIK._compensate_load` and `ArmKinematics.inverse`.

### New helper `_controller_cfg(arm, options, *, use_orientation) -> DifferentialIKControllerCfg`

Builds the library config from `arm.ik` and `options`:
- `command_type="pose"` when `use_orientation` (a target orientation is given and `orientation_axes` is not
  empty), else `"position"`; `use_relative_mode=False`.
- `ik_method=arm.ik.method`; `ik_params` from `options.damping` as above.
- `orientation_weight`: per base-frame axis x, y, z, `options.orientation_weight` if the axis is in
  `arm.ik.orientation_axes`, else `0.0`.
- `joint_limit_avoidance_gain/margin` from `arm.ik`.

### New class `_JointStep`

One bounded IK step on tensors already expressed in the arm-base frame. Shared by the live and offline paths.

- `__init__(arm, options, num_envs, device)`: stores `_arm_bounds`; builds two controllers lazily
  (pose and position variants of `_controller_cfg`); when the gain is positive calls
  `set_joint_pos_limits(lower, upper)` on both.
- `solve(q, ee_pos, ee_quat, jacobian, target_pos, target_quat=None) -> IKResult`
  - Inputs: `q` `(N, J)` full action order; `ee_pos` `(N,3)`, `ee_quat` `(N,4)` XYZW, `jacobian` `(N,6,J)`, all in
    base frame; targets in base frame.
  - Validates shapes and finiteness (same error messages as today's `solve_damped_step` so existing callers'
    error handling is unchanged).
  - Limits the commanded target before calling the library: position moved at most `max_position_step_m`
    from `ee_pos`; orientation moved at most `max_orientation_step_rad` from `ee_quat` (use
    `compute_pose_error` / `quat_from_angle_axis` from `isaaclab.utils.math`).
  - Calls `set_command` then `compute` on the arm-joint columns only (`jacobian[:, :, ids]`, `q[:, ids]`).
  - Caps the step: `delta = new - q`; shrink `delta` uniformly so every `|delta_i| <= step_caps_i`.
  - Clamps to `lower/upper`. Gripper columns are copied unchanged from `q`.
  - Errors for `IKResult`: position error norm before the command; orientation error from
    `compute_pose_error`, expressed in base frame, restricted to declared axes, norm (0 when none declared).
  - `converged` uses `position_tolerance_m` and `orientation_tolerance_rad`; `limited` is true when the step was
    shrunk or a clamp moved any joint.
  - Works in the input dtype for our tensors; the library computes in float32 (its buffers). Cast its output
    back to `q.dtype`.

### `ArticulationIK` (live robot, one end effector)

- `__init__(robot, arm, *, position_offset_m=None, options=None, control_dt=1/30)`: unchanged checks and
  attributes (`joint_ids`, `ee_id`, `base_id`, `jacobian_body_id`, `position_offset_m`); add
  `self._step = _JointStep(arm, self.options, num_envs=robot.data.joint_pos.shape[0], device=robot.device)`.
- `current_pose()`: unchanged meaning (world TCP position, ee link XYZW). Use `quat_apply` for the offset.
- `reset()`, `integral_bias`: unchanged.
- `solve(position_world, orientation_xyzw=None, *, advance=True) -> IKResult`
  1. Read joint positions (`robot.data.joint_pos[:, joint_ids]`), the Jacobian columns
     (`body_link_jacobian_w[:, jacobian_body_id, :, joint_ids + num_base_dofs]`), ee and base link poses.
  2. Shift the Jacobian linear rows to the TCP (`v += omega x offset_world`), as today.
  3. Express TCP pose, target and Jacobian in the base frame (`subtract_frame_transforms`; rotate both Jacobian
     halves with `quat_apply_inverse(base_quat, ...)`), as Isaac Lab's `DifferentialInverseKinematicsAction`
     does.
  4. `solved = self._step.solve(...)`.
  5. `_compensate_load(solved, joint_positions, advance=advance)` when `options.integral_gain > 0`, else return
     `solved`.
- `_compensate_load`: behaviour unchanged; replace its inline cap/limit lists with `_arm_bounds`.

### `ArmKinematics` (offline, no simulator stepping)

Same public methods, now backed by a Newton model of the arm USD.

- `from_arm(arm, *, position_offset_m=None)` (classmethod): build `newton.ModelBuilder()`, `add_usd(arm.usd_path,
  floating=False)`, `finalize(device="cpu")`; return `cls(model, arm, position_offset_m=...)`.
- `__init__(model, arm, *, position_offset_m=None)`:
  - Validate one articulation.
  - Map each `arm.usd_joint_names` entry to its Newton coordinate index (`model.joint_q_start` of the joint whose
    label ends with that name) and dof index (`model.joint_qd_start`). Every declared joint must be found, and
    every non-fixed joint on the base-to-ee path must be declared; otherwise `ValueError`.
  - Find `arm.links.base` and `arm.links.ee` body indices by label suffix; exactly one each.
  - Allocate one `model.state()` and the default `joint_q` copy (undeclared coordinates stay at their default).
  - Self-check: `fk(zeros)` equals the base-to-ee transform of the authored USD at zero (same check as today,
    tolerance `2e-5` m). Keeps the existing guard against wrong USD frames.
- `fk(joint_positions) -> 4x4 ndarray`: write the arm-order values into `joint_q`, `newton.eval_fk`, read
  `state.body_q` for base and ee, return `inv(T_base) @ T_ee` with the TCP offset applied. Same input validation
  and error text as today.
- `jacobian(joint_positions) -> (6, J) ndarray`: run FK, `newton.eval_jacobian(model, state)`, take the ee rows,
  shift linear rows from the body COM to the TCP, rotate into the base frame, reorder columns to manifest order,
  zero gripper columns. Docstring as today (base-frame linear then angular rows).
- `inverse(target_pose_base, measured_joint_positions, *, options=None, max_iterations=12,
  position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4) -> IKResult`: same contract and validation as
  today. Loop up to `max_iterations`: `fk` and `jacobian` at the current `q`, then one `_JointStep.solve` (N=1,
  CPU) with tolerances replaced by the call's; stop when converged. Then clamp to the final dispatch window
  (`measured ± step_caps`, inside limits) and report errors of the clamped command. Gripper values stay exactly
  measured. No integral bias.

### New functions `make_ik(robot, arm, **kwargs)` and `make_kinematics(arm, **kwargs)`

Arm plug-in point for arms that need a different library backend.
- Load the arm's optional utils module with `core.arms.spec.load_arm_utils(arm)`.
- If it defines `make_ik` (resp. `make_kinematics`), call it with the same arguments and return the result;
  it must provide the same public surface as `ArticulationIK` (resp. `ArmKinematics`).
- Otherwise return `ArticulationIK(robot, arm, **kwargs)` (resp. `ArmKinematics.from_arm(arm, **kwargs)`).

## 7. Other source and test files

### `src/sim/datagen/scripted.py`

- Import in `setup_context`: replace `from sim.datagen.ik import ArticulationIK, ArmKinematics, IKOptions` with
  `from sim.datagen.ik import IKOptions, make_ik, make_kinematics`.
- `setup_context`: `ik = make_ik(raw.scene["robot"], arm, position_offset_m=offset, control_dt=..., options=options)`;
  `kinematics = make_kinematics(arm, position_offset_m=ik.position_offset_m)`.
- `make_report`: inside `"ik_controller"`, add `"library": "isaaclab.controllers.DifferentialIKController"`,
  `"method": arm.ik.method`, `"joint_limit_avoidance_gain": arm.ik.joint_limit_avoidance_gain`. Keep
  `"options": asdict(context.ik.options)` and all existing keys.
- Nothing else changes (`target_pose`, `execute_expert`, `failure_diagnostics` use the unchanged surface).

### `src/sim/datagen/mimic.py`

No change. It calls `context.ik.current_pose`, `s2r_fk.inverse(..., options=..., **COMMAND_INVERSE_OPTIONS)` and
`s2r_fk.fk`, which keep their signatures.

### `tests/core/test_arm_spec.py`

Add parametrized cases to the existing invalid-`[arm.ik]` test: unknown `method`, negative gain, zero margin,
positive gain on an arm without a spare joint. Add one valid case reading all three keys.

### `tests/core/test_ik.py`

Delete every test that calls `solve_damped_step`, the USD-FK tests and the integral tests that construct
`ArticulationIK` with a fake robot (they move to `tests/sim/test_ik_controller.py`). Keep or add only tests that
need no Isaac Lab or Newton:
- `IKOptions` validation.
- `_arm_bounds`: degrees to radians for revolute, metres for prismatic, gripper joints excluded, empty arm raises.
- `make_ik` / `make_kinematics` pick the arm hook when present and the default otherwise (use a fake utils module
  and monkeypatch `load_arm_utils`).

### `tests/sim/test_ik_controller.py` (new)

Module-level `skipUnless` like `tests/sim/test_gripper_collisions.py` (needs `pxr`, `isaaclab`, `newton`).
Synthetic arms: write USD files to `tmp_path` with the existing helper pattern from the old
`tests/core/test_ik.py::stage_for_arm` (move that helper here). Build 5-, 6- and 7-arm-joint chains, each with a
gripper joint, one with a prismatic joint and a reversed axis.

Tests (each states its tolerance):
1. `test_fk_matches_authored_frames_at_zero` for each synthetic arm and for `so101`.
2. `test_fk_units_and_axis_direction` (metres-per-unit `1` and `0.001`, reversed prismatic).
3. `test_jacobian_matches_fk_differences` (finite differences used only as a test oracle, tolerance `1e-4`).
4. `test_step_respects_caps_limits_and_grippers` (step caps, clamps, gripper values unchanged, `limited` flag).
5. `test_five_joint_arm_leaves_undeclared_axis_free` (`orientation_axes=["x","y"]`: yaw error does not move
   joints, tilt error does).
6. `test_six_and_seven_joint_arms_reach_full_pose` via `ArmKinematics.inverse` with
   `orientation_axes=["x","y","z"]`, position error `< 1e-4` m, orientation `< 1e-3` rad.
7. `test_seven_joint_limit_avoidance_keeps_pose` (gain `> 0`: joints near a limit move away, TCP pose change
   `< 1e-4` m).
8. `test_live_adapter_offset_permutation_and_base_dofs` (fake robot data as in the old adapter test; controller is
   real).
9. `test_integral_bias_unchanged` (port the two old integral tests unchanged in intent).
10. `test_inverse_recovers_saved_source_commands` (port the old `so101` saved-command case).

If G0.3 found an import-order crash, run this file in its own process in the suite command:
`pytest tests --ignore=tests/sim/test_ik_controller.py` then `pytest tests/sim/test_ik_controller.py`.

## 8. Verification (after all edits)

1. CPU suite: `PYTHONPATH=src python -m pytest tests -q` in the container. Same number of passed tests minus
   removed, plus added; no failures, no crash.
2. `./s2r check --static` and `./s2r check`.
3. Scripted generation with the commands from G0.4 into new destinations (`ik_new_pick`, `ik_new_stack`).
   Pass when, for each task:
   - one accepted episode, task success at the final frame;
   - for every waypoint, the final position error is at most `max(baseline + 0.002 m, tolerance)` and
     orientation error at most `baseline + 0.01 rad`;
   - total steps at most `1.2 x` baseline.
4. Mimic: `./s2r datagen --config_path configs/datagen/mimic.yaml` at the source start. Mimic already fails at
   normal seeds on unchanged code, so the gate is "no worse than a baseline Mimic run on unchanged code with the
   same seed" (run the baseline first if you do not have one).
5. `./s2r prepare` and `./s2r verify --policy groot_n17` on `ik_new_pick`.
6. Line count: `ik.py` has no function containing matrix inversions, `atan2`, quaternion products or joint-chain
   traversal (grep for `linalg`, `atan2`, `cross(` must only hit the TCP-offset shift).

If step 3 fails, compare one waypoint trace against the baseline and adjust only `IKOptions` defaults for the
controller path (damping, step caps); do not reintroduce custom math. Report before committing if it still fails.

## 9. Docs

- `docs/02-add-an-arm.md` `[arm.ik]` paragraph: list `method`, `joint_limit_avoidance_gain`,
  `joint_limit_avoidance_margin`; examples: 5-joint `orientation_axes=["x","y"]`; 6-joint `["x","y","z"]`;
  7-joint `["x","y","z"]` plus a positive gain. Mention the optional `make_ik` / `make_kinematics` hooks in
  `<arm>_utils.py`.
- `arms/_template/arm.toml`: the same three examples as comments.
- `docs/08-data-generation.md`: replace the description of the custom solver with "Isaac Lab
  `DifferentialIKController` for every step; Newton FK for command poses".
- `CLAUDE.md` data-generation bullet: add "IK and FK math come from Isaac Lab and Newton; `ik.py` only handles
  frames, caps and limits."

## 10. Commits

One commit on `v2` for sections 5 to 9, after section 8 passes, message
"Use Isaac Lab and Newton for IK and FK", with the G0.1 versions, the G0.3 outcome and the section 8 numbers
(baseline vs new) in the body. Do not push until the owner reviews the numbers.
