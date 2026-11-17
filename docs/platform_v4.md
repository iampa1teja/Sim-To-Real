# v4 plan: automated dataset generation for all 17 tasks

## Context

The current repository layout (v3: `arms/ scenes/ tasks/ configs/ policies/ src/{core,sim,cli}`) is kept unchanged. v4 has one deliverable: **dataset generation ready for every task**, with no teleoperation needed. The model lives in a separate repository and is out of scope (no policy, no policy-server changes, no policy rollouts, no retry evaluation).

Today the repo has no scripted expert, no IK, no physics-based generator, a single-cube sidecar (`pick_place_meta/`), one task, and `num_envs = 1` everywhere. The only non-teleop writers are `calibration_replay.py` and `dr_replay.py`.

Workflow: **Codex writes all code; Opus writes the tests and the Codex prompts, tests first, per work package**; I coordinate and review. Codex and CI have no Isaac Sim, so the design puts almost all logic in CPU-testable code and leaves a thin Isaac adapter.

## Owner decisions (fixed)

- **Tasks (17):** 13 primitives (`prim_joint_sweep`, `prim_gripper_gap`, `prim_reach_point`, `prim_test_reach`, `prim_calibration_reach`, `prim_waypoint_path`, `prim_touch_object`, `prim_nudge`, `prim_push_to_region`, `prim_lift_test`, `prim_grasp_lift`, `prim_place_at`, `prim_regrasp_after_drop`), `pick_place_cube` (exists), `stack_two_cubes`, `stack_bowls`, and the new precise task `pick_place_color_cube` (2 cm cube into a 4 cm box, 2–6 cubes of different colours, instruction names the colour, box position randomised, one colour and one cube count held out).
- **Arm:** SO-101 now; code must be arm-agnostic (a new arm folder with `arm.toml`, USD, kinematics and calibration works with no code change).
- **Episode kinds, for all 17 tasks:** scripted successes, failures, recoveries; all doubled by visual DR re-render. Not in scope: hidden physical-factor series, probe-then-act series, policy rollouts, Mimic.
- **Generator:** scripted IK expert only.
- **Hard requirement:** no duplicate episodes; this is a main test.
- **Per-frame labels** (stage index, end-effector pose, injected-error flag) are dataset columns as well as sidecar fields, so they stream with the frames.
- **Storage target:** decided later; upload tooling is written against an interface that supports both the Hugging Face Hub and a bucket. Datasets stay in the standard LeRobot v3 layout so they can be streamed.

Defaults I chose (change on request): a failure and a success never share a start scenario; one dataset per task with the kind stored per episode; `pick_place_cube` keeps its contact-sensor success signals, new tasks use geometric checks; kinematics are a generated `arms/<arm>/kinematics.json` next to `arm.toml`; small bounded action noise on success episodes; LeRobot's default video codec.

## Architecture

Three narrow interfaces in `src/core/generate/protocols.py` are the seam: `GenEnv` (reset to a spec, step with joint targets, read joint state / end-effector / object poses / images), `EpisodeWriter`, `Expert`. The whole generation loop is pure Python against them.

- **CPU side (`src/core`, no Isaac):** kinematics, expert, checks and stage tracking, start sampling, perturbations, labels, plan/seeds/quotas, fingerprints, manifest, sidecar, audit, upload planning, and a kinematic `FakeEnv` + `MemoryWriter` so the full loop runs in tests.
- **Isaac side (`src/sim`):** `IsaacGenEnv` (implements `GenEnv`), a generic env config that builds any task's objects, and a runner under ~200 lines with no decisions in it.
- **Experts are data:** `[task.expert] steps = [...]` in `task.toml` composes a shared skill library. The expert is closed-loop (reads measured poses every step), outputs absolute joint targets in radians clamped to runtime limits, so the recorded action equals the executed one.
- **What CPU tests cannot prove:** grasp physics, pushing, stacking stability, rendering. Those are covered by container check scripts the owner runs (`tests/sim/check_*.py` pattern, already used in the repo).

## Files (additions follow existing patterns; nothing moved or renamed)

**`src/core` (new unless marked)**
- `tasks/base.py` (change): object sets and roles in `TaskSpec`; default `sample_start` from `[task.start]`; `failure_mode` and a `StageTracker` in the base class (replaces the duplicate in `tasks/pick_place_cube/benchmark.py` `StageTrace.finish`).
- `tasks/generic.py`: `GenericTask` driven by `[task.stages]`, `[task.checks]`, `[task.failures]`; predicate registry (ee near, lifted, in region, on object, nested in, held, joint near, gap near, displaced).
- `tasks/sampling.py`: start sampler (regions, yaw, min separation, clear of container, IK reachability, cube count and colours, hold-outs).
- `arms/kinematics.py`, `arms/gripper.py`: FK, Jacobian, damped-least-squares IK for any n-DoF chain; gap ↔ jaw angle. Logic lifted from `tests/sim/check_grasp_stability.py:14-85`.
- `arms/mapping.py`, `arms/joint_limits.py` (small change): numpy `to_raw`/clamp; optional per-arm block in `real_setup.json`.
- `expert/{motion,skills,machine,program}.py`: interpolation with speed limits; skills (move_joint, set_gripper, reach, approach, grasp, lift, transport, place, release, retreat, push, touch, sweep, wait); deterministic state machine with recovery re-entry; program parser.
- `generate/{protocols,plan,fingerprint,manifest,perturbations,labels,loop,fake_env,config}.py`.
- `datasets/{task_meta,audit,consolidate,merge,upload,streaming}.py`: sidecar v2 with a v1 (`pick_place_meta`) reader; audit; per-episode JSON → one parquet; shard merge; upload behind a storage interface; streaming reader helper.
- `scripts/{plan_generation,audit_dataset,merge_shards,upload_dataset}.py`.

**`src/sim`**
- `envs/generic_task_cfg.py` (new): `apply_teleop`/`apply_generate` building every declared object with distinct park offsets; jaw sensor path from `arm.toml`.
- `envs/mdp/objects.py` (new): set object pose and colour, read object poses in the base frame (generalised from `tasks/pick_place_cube/mdp.py` and `lerobot_agent._object_pose_in_base`).
- `generate/isaac_gen_env.py`, `generate/scripted_generate.py`, `generate/extract_kinematics.py` (new). Reuse `dr_replay.make_env`, `discover_sim_cameras`, `make_mapping`, `make_writer`, `rendering_kit_args`, `git_commit`, and `LeRobotRecorder`.
- `generate/dr_replay.py`, `dr_variants.py`, `recording/replay_scene.py`, `replay_source.py` (change): multi-object poses, per-task colour draws from the palette, instruction re-rendered from the template, v2 sidecar out; v1 sources behave exactly as today.
- `recording/episode_metadata.py` (change): old functions kept as thin wrappers, so teleop is untouched. `recording/lerobot_recorder.py` (small change): optional extra per-frame columns.

**Data and config**
- `arms/so101/arm.toml` (add): `[arm.kinematics]`, `[arm.gripper]`, `[arm.expert]`, moving-jaw link names (removes SO-101 literals from `scene/arm.py`, `contact_materials.py`, `grasp_diagnostics.py`, `pick_place_cube/env_cfg.py`). `arms/so101/kinematics.json` generated once in the container and committed.
- `tasks/prim_*/` (13) and the three new target tasks: `task.toml`, `__init__.py` (`TASK = GenericTask(...)`), a three-line `env_cfg.py`; `usd/` only where needed (hand-written 4 cm box, procedural bowl). `tasks/pick_place_cube/task.toml` gains `[task.start]`, `[task.expert]`, `[task.perturbations]` and a `generate` variant.
- New `task.toml` tables: `[task.object_sets.*]`, `[task.palette]`, `[task.roles]`, `[task.start]`, `[task.instruction_template]`, `[task.stages]`, `[task.checks.*]`, `[task.failures]`, `[task.expert]`, `[task.perturbations]`.
- `configs/generate/*.yaml` (new folder beside `teleop/ eval/ train/`): one per task group plus `smoke.yaml`.
- `src/cli/paths.py` `SIM_PROGRAMS` rows + `pyproject.toml` console scripts: `generate`, `generate-plan`, `arm-kinematics`, `dataset-audit`, `dataset-merge`, `dataset-upload`.
- `docs/platform_v4.md` (this plan) and `docs/v4/prompts/WP*.md` (Codex prompts).

## Key mechanisms

- **Multi-object scenes:** all `count_max` cubes are built once and parked at distinct offsets (Newton builds the model once); each episode teleports the active ones and sets colours. Roles (`@target`, `@container`) bind the instruction and the success check to the target cube; a distractor in the box is the failure `wrong_object_placed`.
- **Failures and recoveries:** perturbations are either expert errors (grasp offset, early close/release, place offset, wrong target, stall) or world disturbances (object shift, drop, container shift). Failure = perturbation with a blind expert; recovery = same with the closed-loop expert re-entering the earliest unsatisfied stage. **Labels come from the observed outcome**, not the intent (a "failure" that succeeds is rejected). Each episode records kind, failure name, furthest stage, per-frame stage, segments (nominal / perturbation / recovery) and an `imitate` flag that is false on injected-error frames.
- **No duplicates:** each attempt has a deterministic seed from (master seed, task, kind, index). Fingerprints (SHA-256 via `core/utils/hashing.py`): `scenario_key` (quantised start situation), `traj_fp` (quantised actions and states), `appearance_fp` (colours, light, robot colour, instruction), `episode_fp = hash(traj_fp + appearance_fp)`. Repeated scenarios are skipped identically on every shard and resume; a trajectory collision is rejected after the fact; a DR copy must differ from its parent and siblings in appearance. Every attempt is one line in an append-only `meta/generation_manifest.jsonl`. `dataset-audit` recomputes fingerprints from the stored data; merge and upload refuse without a passing audit.
- **Throughput:** `num_envs` stays 1 (many `env_0` assumptions); scale comes from shards run in several containers or GPUs. `generate-plan` prints an hours estimate from a smoke-run timing.

## Work packages (tests first; same wave = parallel Codex sessions)

| Wave | WP | Scope | Tests pin down |
|---|---|---|---|
| 0 | WP0 | `generate/protocols.py` dataclasses | construction, immutability, JSON round-trip |
| 1 | WP1 | task contract v2, generic task, sampler | predicate truth tables; six pick-place failure names reproduced; sampler properties (region, separation, unique target colour, hold-outs never drawn, same seed same result); existing task still loads |
| 1 | WP2 | kinematics, gripper, mapping additions, `arm.toml` additions, extractor | `ik(fk(q))` within 0.2 mm; Jacobian vs finite differences; limits respected; unreachable target flagged; synthetic 6-DoF arm passes |
| 1 | WP4 | sidecar v2 + v1 reader | round-trip; atomic write; validator rejects NaN, bad lengths, segment gaps; v1 fixture loads; old validators unchanged |
| 1 | WP5 | plan, fingerprints, manifest, config | golden fingerprints; quantisation sensitivity; shards partition the stream; deterministic skips; exhaustion error; torn manifest line; reconcile cases |
| 2 | WP3 | expert | skills terminate in budget; speed and limits respected; determinism; blind never re-enters, closed-loop re-enters the right stage |
| 2 | WP6 | perturbations, labels | parameters in range; segments cover `[0, N)`; `imitate` false exactly on error frames; outcome-based acceptance table |
| 2 | WP12 | consolidate, merge, upload, streaming | parquet equals JSON; merge refuses collisions; upload refuses without audit, resumable, works against both storage back-ends through an injected client |
| 3 | WP7 | `FakeEnv`, loop, audit | **main no-duplicates test:** all 17 tasks × 3 kinds, random kill and resume, 3 shards merged, fake DR ×2, audit passes; resumed run equals uninterrupted run; an injected copy makes the audit fail |
| 4 | WP8 | Isaac adapter, generic env config, runner, CLI | CLI argv, config parsing, dry run without Isaac, no SO-101 literals |
| 4 | WP9 | 13 primitive task folders | each loads, program validates, ≥95% success in `FakeEnv` over 50 seeds, all three kinds produced |
| 5 | WP10 | `pick_place_cube` additions, `stack_two_cubes`, `pick_place_color_cube` | 4 cm box USDA parses; colour binding; `wrong_object_placed`; existing eval/benchmark tests still pass |
| 5 | WP11 | `dr_replay` v2 | deterministic planning; unique appearance per parent; target colour unique and not held out; v1 source plans exactly as today |
| 5 | WP13 | `stack_bowls` + bowl asset | nested-in predicate; `FakeEnv` suite (isolated last: highest physics risk) |

Process per package: an Opus agent writes the test files and `docs/v4/prompts/WPn.md` (scope, files, interfaces, acceptance = the tests) → I review both → Codex implements until the tests pass → I check the result and the owner runs the container checks.

## Verification

- **Host (no Isaac):** `pytest tests/core tests/sim -q` with the existing `host` dependency group (numpy, scipy, torch, pyarrow, pytest). The no-duplicates test in WP7 is the gate for "generation logic is ready".
- **Container (owner runs):**
  - `check_kinematics_vs_sim.py`: FK vs the sim's `ee_frame_state`, < 1 mm and < 0.5°.
  - `check_multi_object_scene.py`: parking, teleport and recolour of N cubes.
  - `check_expert_task.py --task X --n 20`: expert success ≥ 80% on target tasks, ≥ 90% on primitives.
  - `check_generate_smoke.py`: 10 episodes per kind per task, audit passes, dataset opens with LeRobot.
  - `check_dr_replay_v2.py`, `check_streaming.py`.
- **Done means:** `s2r generate --config_path configs/generate/<task>.yaml` produces an audited dataset for each of the 17 tasks, `s2r dr-replay` doubles it, and `s2r dataset-audit` passes on both.

## Risks

- **Throughput:** about 34,000 source episodes at one environment, plus the same again for DR, is likely hundreds of GPU-hours; not measured. Shards are the answer.
- **Grasp reliability under Newton:** the expert will need per-task tuning in the container; all tuning values are in `task.toml`.
- **`stack_bowls`:** no bowl asset exists and concave shapes need convex pieces; isolated as the last package.
- **Precise placement:** ±1 cm clearance needs closed-loop correction before release and depends on actuator tracking in sim.
- **LeRobot v0.6.1 merge and streaming APIs** are unverified here; checked in the container before WP12 is written.
- **Primitive meanings** are my reading of the names (for example `prim_calibration_reach` = reaches to a fixed calibration grid, now that hidden offsets are out of scope); each `task.toml` is reviewed with the owner in WP9.
