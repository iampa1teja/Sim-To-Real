# WP3 — scripted expert: motion, skills, state machine, program parser (implementation prompt)

## Goal

Create the Isaac-free package `src/core/expert/`: speed-limited motion helpers, the skill library, the `[task.expert] steps = [...]` parser/validator and `ScriptedExpert`, the deterministic state machine that implements the WP0 `Expert` protocol. It turns a task's program into absolute joint targets (radians, joint-vector order) from measured observations, reads object poses every step in closed-loop mode, freezes them in blind mode, re-enters the earliest unsatisfied stage after a detected regression, and executes the group-A (expert-error) perturbations of §7.1. Everything is arm-agnostic; the tests run on the synthetic arms `fx_arm5` / `fx_arm6` only. Follow `docs/v4/contracts.md` §6 exactly; where the contract leaves a choice open, this prompt fixes it and the tests pin it.

Prerequisites: WP0 (`core.generate.{errors,seeds,protocols}`), WP1 (`core.tasks.base` v2, `core.tasks.generic`, `core.tasks.sampling`) and WP2 (`core.arms.{kinematics,gripper,raw_mapping}`, the v4 `arm.toml` tables) are implemented.

## Files

Create exactly these files:

| File | Public names |
|---|---|
| `src/core/expert/__init__.py` | none (docstring only) |
| `src/core/expert/motion.py` | `MotionLimits`, `step_toward_joint`, `cartesian_waypoint`, `expert_joint_limits` |
| `src/core/expert/skills.py` | `SKILLS`, `Skill` |
| `src/core/expert/program.py` | `ProgramStep`, `ExpertProgram`, `parse_program` |
| `src/core/expert/machine.py` | `ScriptedExpert` (it may import `MotionLimits` from `motion.py`, so `from core.expert.machine import MotionLimits` also works) |

Every other module-level name starts with `_` (a test checks the defined names of each file). Do NOT touch anything under `tests/` (test files, `tests/v4_support/`, `tests/conftest.py`, fixtures), `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`, `pyproject.toml`, or any other package's modules (`core/generate/*`, `core/tasks/*`, `core/arms/*`, `core/datasets/*`, `sim/*`).

## Read first

`docs/v4/contracts.md`: §1.1 (frames, XYZW in memory, ee vs TCP, object origins), §1.2 (joint order), §1.3, §1.4 (RNG, sub-seed purposes), §1.5, §1.6 (imports), §2 (WP3 row), §3.2–3.3 (`EpisodeSpec.expert_mode`, `StageProgress`, `ExpertObs`, `ExpertAction`, the `Expert` protocol and call order), §4.3 (StageTracker: `stage_index`, `furthest_index`, `regressed_stage`), §4.4 (ref syntax, `p(o)`, `h(o)`, `w(o)`), §4.6 (both example programs), §5.1, §5.3, §5.4, **§6 (the main spec)**, §7.1 (group A rows), §11 (FakeEnv dynamics: the expert must work on them), §11.1 (the loop calls `act` once per frame), §12 (no SO-101 literals in `src/core/expert`), §13, §15. Also `docs/v4/prompts/WP1.md` (predicates, `resolve_ref`, `StageTracker`), `WP2.md` (`KinematicChain`, `Gripper`, `RawMapping`) and `WP6.md` (the `PerturbationSpec.params` keys the expert reads).

## Interfaces and the decisions the tests pin

### `motion.py`

- **`MotionLimits(max_joint_speed_rad_s, max_gripper_speed_rad_s, max_ee_speed_m_s)`** — frozen dataclass (`eq=False` + an `__eq__` comparing the array, `__hash__ = None`). The speed array is stored as a read-only float64 C-contiguous copy (the tests run `assert_frozen`); the scalars as `float`. Non-positive / non-finite values → `ValueError`.
  - `from_arm(name=None)`: from `load_arm_spec(name)`'s `[arm.expert]`. The array has length J in `[arm.joints].names` order: `max_joint_speed_rad_s` (a float, or a `{joint = float}` table that must name every non-gripper joint) for the non-gripper joints and **`max_gripper_speed_rad_s` at the gripper index**. A missing `[arm.expert]` table, a missing speed key or an incomplete table → `ConfigError`.
- **`step_toward_joint(current, target, max_step) -> np.ndarray`** = `current + clip(target − current, −max_step, max_step)`; `max_step` is a scalar or a `(J,)` array; inputs are not mutated; mismatched or non-1-D shapes → `ValueError`.
- **`cartesian_waypoint(current, goal, max_step_m) -> np.ndarray`**: the point at most `max_step_m` from `current` on the segment to `goal` (the goal itself once within reach). Positions are `(3,)`, else `ValueError`.
- **`expert_joint_limits(arm=None, *, setup=None) -> (lower, upper)`** — **this resolves the open point of contracts §15 (WP2)**: the public name of the "expert joint limits" of §5.4 is `core.expert.motion.expert_joint_limits`. `lower = max(RawMapping.from_setup(setup, arm=arm).lower_rad, KinematicChain.from_arm(arm).lower)`, `upper = min(…upper_rad, ….upper)`, per joint, float64 `(J,)`, joint-vector order (`setup=None` reads the scene's `real_setup.json`, i.e. the per-arm block when present). Any joint with `lower > upper` → `ConfigError`. WP7's `make_expert_factory` passes these arrays as `ScriptedExpert(lower=…, upper=…)` and `parse_program(joint_limits=…)`.

### `skills.py`

`Skill(name, required, defaults, budget)` — frozen; `required` a tuple, `defaults` a `MappingProxyType` (tuples for vectors), `budget` an `int`, or `None` for `wait` (whose budget is its `steps`). `SKILLS` is a dict from name to `Skill` in exactly the §6.1 table order: `move_joint, set_gripper, reach, approach, grasp, lift, transport, place, release, retreat, push, touch, sweep, wait`. Required parameters and defaults are those of the §6.1 table, with these decisions:

- `reach` defaults: `offset_m=(0,0,0)`, `orientation="down"`, **`yaw="keep"`**, `tol_m=0.005` (approach: `offset_m=(0,0,0.06)`, `"down"`, `"object"`, `0.005`).
- `push` requires `target` and `goal`; `sweep` requires `joint` and `angles`; `wait` requires `steps`; `release`, `lift`, `retreat` require nothing.

The skill *behaviour* lives in `machine.py` (private per-skill handlers); `SKILLS` is the descriptor table the parser and the machine share.

### `program.py`

- **`ProgramStep(skill, name, stage, budget, params)`** — frozen; `params` = the skill defaults overlaid by the step's own parameters (without `skill`/`name`/`stage`/`budget`), deep-frozen (lists → tuples, tables → `MappingProxyType`), refs left as strings (resolved per episode), and `q = "home"` resolved at parse time to `{arm_joint: rad}` for `[arm.expert].arm_joints` only (values from `[arm.initial_state].joint_positions`, overridden by `[arm.expert].home_rad`). A literal `{joint = rad}` table is kept as a mapping of floats. Must pass `assert_frozen`.
- **`ExpertProgram(steps, max_steps, settle_steps, max_retries, objects, approach_axis, closing_axis)`** — frozen; `steps` a tuple; `objects` = `task.spec.objects` (the expanded `ObjectSpec`s; the expert needs sizes, origins and container parameters, and the contract constructor of `ScriptedExpert` has no other way to get them); `approach_axis` / `closing_axis` = tuples from `[arm.kinematics]` (the expert builds "TCP pointing down" orientations from them; `KinematicChain` does not expose them publicly).
- **`parse_program(table, *, task, arm_spec, joint_limits=None) -> ExpertProgram`**. `table` is the `[task.expert]` dict, `task` a WP1 `Task` (read `task.stages`, `task.spec.roles`, `.objects`, `.object_sets`, `.raw["start"]["goal"]`), `arm_spec` the `[arm]` dict (`load_arm_spec(arm)[0]`), `joint_limits` an optional `(lower, upper)` pair in joint-vector order. The input is never mutated. Defaults: `max_steps=600`, `settle_steps=10`, `max_retries=2`; a step's `name` defaults to its skill and its `budget` to the skill budget (`wait`: `steps`). Everything below is `ConfigError`:
  - table shape: not a table; a top-level key other than the four; `steps` missing, not a list, empty, or holding a non-table; `max_steps` not an int ≥ 1, `settle_steps` / `max_retries` not an int ≥ 0 (bools are not ints).
  - steps: missing or unknown `skill`; a key that is neither common (`skill, name, stage, budget`) nor a parameter of *that* skill; a missing required parameter (e.g. `grasp` without `target`); `stage` not in `task.stages`; `budget` not an int ≥ 1; `name` not a non-empty string.
  - refs (§4.4 syntax, checked statically): `@role` must be a role of the task; `@role.inner` only where a region is allowed (`transport`/`place` target, `push` goal) and only for a container (object or set template with `inner_half_size`); any other `.suffix` is an error; `@goal.<key>(…)` needs `<key>` in `[task.start.goal]`; a literal string must be an object of `task.spec.objects`. `grasp` and `push` targets must be objects (`@role` or a name, no goal ref); `reach`/`approach`/`touch` targets may also be a literal `[x, y, z]`; `transport`/`place` targets may also be a goal ref or a literal rect `{xy_min, xy_max}` (min ≤ max); a `push` goal is a goal ref, `@role.inner`, a rect, or a 2-/3-point.
  - parameter types: `offset_m` 3 finite numbers; `tol_m`, `tol_rad`, `contact_tol_m`, `height_m` > 0; `clearance_m`, `standoff_m` ≥ 0; `z_offset_m`, `depth_m`, `dz_m` finite; `dwell_steps` int ≥ 0; `wait.steps` int ≥ 1; `orientation` ∈ {down, keep}; `yaw` ∈ {object, keep} or a finite number; `gap_m` a number, `"open"`, `"closed"` or a goal ref (`close_gap_m`: a number or a goal ref); `q` = `"home"`, a non-empty `{joint = finite rad}` table of `[arm.joints].names`, or a `@goal` ref; `sweep.joint` an arm joint (`[arm.expert].arm_joints`, so never the gripper) or a goal ref; `angles` a non-empty list of numbers or a goal ref.
  - **unreachable configurations**: a gap (`gap_m`, `close_gap_m`) outside `[min, max]` of the `opening_table` gaps; with `joint_limits`, a literal `q` value, a `home` value or a literal `sweep` angle outside `[lower, upper]` of its joint. With `joint_limits=None` only the joint-range checks are skipped.
  - `arm_spec` without `kinematics.approach_axis` / `closing_axis`.

### `machine.py` — `ScriptedExpert`

Constructor exactly as §6.2: `ScriptedExpert(program, chain, gripper, *, lower, upper, limits, fps, stages, weights=None)`; `weights=None` means `IKWeights()`. `lower`/`upper` not `(J,)` or `lower > upper`, a speed array whose length is not J, or a non-positive int `fps` → `ValueError`. Attributes/properties: `reentries: list[tuple[int, str]]` (`(obs.frame, resumed_stage)`, a fresh list at every `reset`), `phase`, `done`.

`reset(spec, obs)`: `obs.q` must be `(J,)` (else `ValueError`); `q_target[−1] := obs.q`; mode = `spec.expert_mode`; blind mode snapshots the active object poses of this `obs`; the group-A perturbation (if `spec.perturbation.group is EXPERT`) is prepared; step 0 starts. Every piece of state is per instance and fully reset (two instances never interact; a reused instance behaves like a new one). `act` before the first `reset` → `RuntimeError`.

**Output rules (tested on random starts on both arms):** `q_target` is a fresh float64 `(J,)` finite array, `q_t = clip(step_toward_joint(q_{t−1}, desired, max_step), lower, upper)` where `max_step = max_joint_speed_rad_s / fps` per joint **except the gripper index, which uses `max_gripper_speed_rad_s / fps`** (the gripper entry of the array is ignored). Every Cartesian skill moves a private carrot with `cartesian_waypoint(carrot, goal, max_ee_speed_m_s / fps)` (the carrot starts at the measured TCP when the step / sub-phase starts) and solves IK to the carrot. `phase` is the current step's `name` (never empty); `ExpertAction.phase` equals `expert.phase` after the call, `ExpertAction.done` equals `expert.done`.

**IK.** Do **not** call `chain.ik` every frame: with a 5-DoF arm or a not-yet-feasible orientation it falls back to `scipy.optimize.least_squares` and costs 30–200 ms per call, which makes the tests take minutes. Use a private damped-least-squares solver on `chain.fk` / `chain.jacobian` with the §5.3 weighted residual `[w_p (p − p*); w_a (a × a*); w_r · roll]`, seeded at the previous target, a few iterations per frame (the reference does ≤ 12, λ² = 1e-4, step clamp 0.2 rad), clipped to `[lower, upper]` on the chain joints. Non-chain joints come from the skill (the gripper) or the previous target. Target: about 1 ms per `act`; the whole WP3 test set runs in ≈ 22 s with the reference.

**Orientation.** `"down"` = the rotation `R` with `R·approach_axis = (0, 0, −1)` and `R·closing_axis = (cos ψ, sin ψ, 0)`. Yaw `ψ`: `"keep"` = the measured TCP's closing-axis yaw `atan2(c_y, c_x)` at step start; a number = that yaw; `"object"` = the object yaw `+ k·π/2` closest to the current reference (the step-start TCP yaw, then the previous choice), and `"keep"` when the target is not an object. `orientation = "keep"` = the measured TCP rotation at step start. `lift`, `transport`, `place`, `retreat` keep the step-start orientation; `grasp` and `touch` use down + object yaw; `push` uses down + keep.

**Step lifecycle (pinned with exact frame counts).** A step emits its first action on the frame it starts. From the next frame on, its termination condition is evaluated on that frame's observation *before* acting; a step that has emitted `budget` actions ends (timeout). On the frame a step ends, the next step starts and emits that frame's action (no idle frame). So every step emits between 1 and `budget` actions, a timed-out step emits exactly `budget`, and "done" is reported on the frame the last step ends: `done=True`, `q_target` = the previous target, `phase` = the last step's name; afterwards every call returns the same held target with `done=True`.

**Skills** (termination on the measured state; references per mode, see below):

| Skill | Behaviour |
|---|---|
| `move_joint` | desired = previous target with the commanded joints set to `q` (clamped to `[lower, upper]`); ends when `max |obs.q − q|` over the commanded joints ≤ `tol_rad`. A goal ref may give `{joint: rad}` or a full-length list. |
| `set_gripper` | gripper → `gripper.angle_from_gap(gap)` (`"open"`/`"closed"` = the gap of `open_rad`/`closed_rad`); ends when `|obs gap − gap| ≤ tol_m` **or** the measured gap changed by < 1e-4 m on 5 consecutive frames of the step (blocked). |
| `reach` / `approach` | goal = target point (object position, goal point or literal) + `offset_m`; ends when `‖tcp − goal‖ ≤ tol_m`. Gripper held. |
| `grasp` | sub-phases **up** (to `max(start z, grasp z + h(o))` at the start xy), **over** (grasp xy at that height), **down** (grasp point = object centre + `(0, 0, z_offset_m)`; bottom-origin objects use origin + h/2), each left within 0.005 m; then **close** to `angle_from_gap(close_gap_m)` until the gap is within 0.002 m of it or blocked (5 frames < 1e-4 m), then **dwell** `dwell_steps` frames. The arm holds during close/dwell. The gripper is held during the motion, **except in closed-loop mode on a grasp started by a re-entry, which commands `open_rad` while it moves** (so a re-grasp after a failed close works; blind keeps the error). When it ends, the grasped object is "held" (for `place`). |
| `lift` | goal = TCP at step start + `(0, 0, height_m)`; within 0.005. |
| `transport` | object or `@role.inner` target: goal = (target xy, top(target) + `height_m`), `top` = z + h/2 (centre origin) or z + h (bottom origin); region target: (region centre, support + `height_m`) where support = the held object's reset bottom (or the step-start TCP z when nothing is held); within 0.005. |
| `place` | goal = (target xy + offset xy, top(target) + h(held)/2 + `clearance_m` + offset z) (region: support as above); within 0.003. |
| `release` | gripper → `gap_m` (default open) until within 0.002 or blocked, then `dwell_steps`; arm held; clears "held". |
| `retreat` | goal = TCP at step start + `(0, 0, dz_m)`; within 0.005. |
| `push` | direction d = unit(goal centre − object xy) at step start; sub-goals: above the pre-push point (object xy − d·(w/2 + `standoff_m`), push height + 0.04), the pre-push point at push height (= object centre z + `depth_m`), then the goal centre − d·w/2; ends as soon as the object is in the goal (rect, container inner rect in its yaw frame, or within 0.01 m of a point), checked every frame. |
| `touch` | goal = (object xy, top(o)), down + object yaw; ends when `‖tcp − goal‖ ≤ contact_tol_m`. |
| `sweep` | the joint visits each angle in turn (others held), each reached within `tol_rad`; ends after the last. A goal ref resolving to a `{joint, angles}` mapping uses its `"joint"` / `"angles"` entries. |
| `wait` | holds the previous target; ends after `steps` actions. |

**Modes** (`spec.expert_mode`): `closed_loop` re-reads object poses from every `obs` (a moved object is followed); `blind` resolves every object reference from the `obs` given to `reset` and never re-reads it, and never re-enters. Both always read `obs.q`, `obs.tcp` and `obs.gripper_gap_m` for termination. An object missing from an observation keeps its last seen pose.

**Detection** (closed loop only, while fewer than `max_retries` re-entries have happened; afterwards no rule fires any more but references stay closed-loop): (1) `progress.regressed_stage is not None` (checked first, on any frame); (2) a step with `stage = s` ends (condition or budget) while `progress.stage_index < stages.index(s)` — **not applied to `approach` steps** (their goal is a pre-grasp offset, so the stage they are tagged with — `reached` in the §4.6 example — is only reached by the following grasp; without this the nominal example program would re-enter every episode); (3) a `wrong_target` perturbation is active and a `grasp` step ends. At most one re-entry per frame.

**Re-entry.** `resumed_stage = stages[progress.stage_index + 1]` (the tracker's `regressed_stage` does not choose it; if `stage_index + 1 == len(stages)` nothing happens). Step index = 1 + the index of the last stage-tagged step before the first step whose stage is **at or after** `resumed_stage` in stage order (the contract's "first step tagged resumed_stage", generalised for stages no step carries), or 0. The re-entry is reported on the detection frame (`reentered=True`, `resumed_stage` set; both `False`/`None` on every other frame) and **that frame's action is already the re-entered step's first action** (its phase is that step's name), and `reentries` gets `(obs.frame, resumed_stage)`. Ending a `wrong_target` binding happens at the first re-entry.

**Group-A perturbations** (§7.1; `spec.perturbation.group is EXPERT`; parameter keys as WP6 draws them: `params` plus `PerturbationSpec.delay_steps`). Only the **first execution** of the triggering step is perturbed (re-executions after a re-entry are clean). `ExpertAction.injected` is True on exactly these frames:

| Type | Window and effect |
|---|---|
| `grasp_offset` | the first grasp execution: grasp point + `(xy_m·cos dir_rad, xy_m·sin dir_rad, z_m)`, yaw + `yaw_rad` |
| `partial_close` | the first grasp execution: close gap = w(object) + `extra_gap_m` |
| `early_close` | first approach execution, from its action number `delay_steps` (0-based) to its end: gripper → `closed_rad` (the closed gripper persists afterwards because later skills hold it) |
| `early_release` | first transport execution, from its action number `delay_steps` until the measured gap ≥ open gap − 0.002 (that frame is not injected) or the step ends: gripper → `open_rad` |
| `place_offset` | the first place execution (goal xy + `xy_m·(cos, sin)`) and the release step that directly follows it |
| `target_offset` | the first execution of the first step (program order) tagged `onset_stage` — any stage tag when it is `None` — that has a `target`, `q` or `gap_m` parameter: goal + `(xy_m·(cos, sin), z_m)` for target skills, `+ joint_rad` on every commanded `move_joint` joint, `+ gap_m` on the `set_gripper`/`release` gap |
| `wrong_target` | at `reset`: candidates = active objects of the episode (other than the target) whose `params["set"]` equals the target's; if any, `@target` is bound to `sorted(candidates)[rng.integers(len)]` with **`rng = rng_for(spec.seed, "expert")`** (the §1.4 `"expert"` purpose; the expert factory has no other generator); window from frame 0 until the first re-entry (exclusive) or the episode end, including done frames. Without candidates nothing is applied. |
| `stall` | onset = the first frame with `progress.furthest_index ≥ stages.index(onset_stage)` (frame `delay_steps` when `onset_stage` is None) + `delay_steps`; for `params["steps"]` frames the previous target is returned, nothing else advances (no termination checks, no budget use, no detection) |

Group-B types are the `WorldPerturber`'s job: the expert never marks them injected.

## Contract deviations / additions to report back into contracts.md

1. **Expert joint limits** (open point of §15 WP2): `core.expert.motion.expert_joint_limits(arm=None, *, setup=None) -> (lower, upper)`.
2. `MotionLimits` lives in `motion.py` (§2) although §6.2 shows it under `machine.py`; its array has the gripper speed at the gripper index; the expert uses `max_gripper_speed_rad_s` for the gripper whatever the array says.
3. `ExpertProgram` fields (`objects`, `approach_axis`, `closing_axis`) and `ProgramStep` fields; `Skill` fields; `parse_program(..., joint_limits=None)`; `arm_spec` = the `[arm]` dict; "home" = arm joints only.
4. `reach` default `yaw = "keep"` (the contract gives no reach defaults).
5. The step lifecycle and the same-frame re-entry described above; the "≥ resumed_stage" generalisation of the step index; rule 2 is not applied to `approach`; after `max_retries` "acts as blind" means "no more detection".
6. `grasp` sub-phases and the closed-loop re-open on a re-entered grasp; region support height for `transport` / `place`; `push` sub-goals; blocked = 5 consecutive frames with |Δgap| < 1e-4 m.
7. Group A: first-execution-only; `wrong_target` uses `rng_for(seed, "expert")` (the contract says "the perturbation rng", which the expert does not have); `target_offset` step choice requires a target/q/gap parameter (consistent with WP6's `allowed_stages`); `stall` uses `furthest_index` ("latched", as WP6 decided for world perturbations).
8. Known limitation to review with WP6/WP10: with the contract's colour-cube program, `early_close` and `partial_close` recoveries rely on rule 2 at the grasp end plus the re-entered grasp re-opening; the §4.4 `held` predicate can latch `grasped` while a misaligned close passes through `[w − 0.006, w + 0.004]`, so a recovery may use two re-entries.

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp3_motion.py     # MotionLimits, step_toward_joint, cartesian_waypoint, expert_joint_limits
tests/core/test_v4_wp3_program.py    # SKILLS table, both contract programs parse, valid forms, every ConfigError case,
                                     # unreachable configurations, module hygiene (public names, imports, 3.11, no arm literals)
tests/core/test_v4_wp3_machine.py    # protocol, output rules (random starts, both arms), Cartesian speed, determinism,
                                     # closed loop vs blind, re-entry table, rules 1-2, approach exemption, max_retries, done,
                                     # target_offset step selection
tests/core/test_v4_wp3_episodes.py   # kinematic stand-in: every skill finishes within budget / times out after exactly budget,
                                     # fx_reach + fx_color complete all stages (both arms, closed loop and blind), group A
```

`test_v4_wp3_machine.py` feeds synthetic observations (perfect tracking, no object physics). `test_v4_wp3_episodes.py` defines a private kinematic stand-in (perfect tracking, attach on close, drop on open, a point pusher; deliberately close to §11 but it is not FakeEnv) and places the table at z = −0.05 m because the fx arms reach a down-pointing TCP only for z ≈ −0.06…0.10 m and r ≈ 0.10…0.20 m. Both use the fixture tasks `tests/fixtures/v4_wp1/tasks/{fx_reach,fx_color}` (the §4.6 programs), a synthetic tool task written into a `sandbox`, and the arm fixtures `tests/fixtures/arms/fx_arm{5,6}` with `tests/fixtures/scenes/fx_scene`.

Run from the repo root (Python 3.11, host group):

```
uv run --group host pytest tests/core/test_v4_wp3_*.py -q
# without uv:
PYTHONPATH=src python -m pytest tests/core/test_v4_wp3_*.py -q
```

Before implementation every file fails at collection with `ModuleNotFoundError: core.expert`. The set takes about 22 s with the reference implementation; keep it under ~30 s.

## What must keep working

- All WP0, WP1, WP2 tests (`tests/core/test_v4_wp{0,1,2}_*.py`, `tests/sim/test_v4_wp2_*.py`) and `tests/core/test_v4_support_selftest.py` (apart from its known WP1 conflict).
- The full suite must be no worse than before: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`. Other packages' tests that fail only because their modules do not exist yet are not your concern.

## Rules

- Do not edit, skip, xfail or weaken any test, and do not change `tests/v4_support/`, `tests/conftest.py` or the fixtures.
- Create only the five files listed under "Files".
- Imports: standard library, `numpy`, `scipy` (optional) and `core.*` (WP0, WP1, WP2 and existing modules). No `torch`, `yaml`, `pyarrow`, `random`, Isaac, `pxr` or LeRobot; no global `np.random.*` (use `rng_for` / `np.random.default_rng`). Python 3.11 syntax.
- No literal of a repo arm in `src/core/expert` (`"so101"`, its USD and LeRobot joint names, its ee / jaw link names); the string `"gripper"` may appear only as the `[arm.gripper]` table key. Read every arm-specific value from `load_arm_spec`, the chain, the gripper and the program.
- If a test appears to contradict `docs/v4/contracts.md` or this prompt, or two tests cannot both pass, **stop and report** the test name, the clause and your reasoning instead of working around it.

## Definition of done

1. The four WP3 test files pass (no skips, no xfails) in about 30 s or less.
2. The full suite is no worse than before.
3. Only the five files listed under "Files" were created.
