# WP7 — FakeEnv, lite writer, generation loop, audit and the no-duplicates gate (Codex prompt)

## Goal

Make the whole dataset-generation pipeline run on the CPU: a kinematic `GenEnv` (`FakeEnv`), a writer that persists a lite LeRobot-v3 dataset (`LiteDatasetWriter` / `MemoryWriter`), the generation loop (`run_episode`, `run_generation`), the dataset audit, and the two scripts `dataset-audit` and `generate-plan`. The acceptance gate of the whole project is `tests/core/test_no_duplicates.py`: generation in 3 shards with crashes and resumes, merge, fake DR ×2, and an audit that proves no episode is duplicated.

WP7 is the integration package. It only wires together WP0–WP6 and WP12; it must not change their modules.

## Files

Create exactly these files:

| File | Public names |
|---|---|
| `src/core/generate/fake_env.py` | `FakeEnv`, `SimulatedCrash` |
| `src/core/datasets/frames.py` | `LiteDatasetWriter`, `MemoryWriter`, `read_frames` (optional constant `LITE_VERSION = "v3.0-lite"`) |
| `src/core/generate/loop.py` | `run_episode`, `run_generation`, `GenerationSummary`, `make_expert_factory`, `make_spec_fn`, `home_q` |
| `src/core/datasets/audit.py` | `AuditReport`, `audit_dataset`, `write_audit_report`, `read_audit_report`, `content_digest`, `CHECKS` |
| `src/core/scripts/audit_dataset.py` | `main` (optional `build_parser`) |
| `src/core/scripts/plan_generation.py` | `main` (optional `build_parser`) |

Everything else is private (leading underscore). The tests check the public names with the AST.

Already written for you (do not edit): `tests/v4_support/harness.py` (shared harness, also used by WP9/WP10/WP13), `tests/core/test_v4_wp7_*.py`, `tests/core/test_no_duplicates.py`, and the fixture tasks in `tests/fixtures/v4_wp7/tasks/` (`fx7_grip`, `fx7_reach`, `fx7_pick`, `fx7_color`).

Do NOT touch: anything under `tests/`, `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`, `pyproject.toml`, or any module of another work package (`core/generate/{protocols,errors,seeds,plan,fingerprint,manifest,config,appearance,perturbations,labels,specs}.py`, `core/tasks/*`, `core/arms/*`, `core/expert/*`, `core/datasets/{task_meta,consolidate,merge,upload,streaming}.py`). If a test seems to need a change there, stop and report.

## Read first

`docs/v4/contracts.md` §1 (conventions), §2 (WP7 row), §3 (protocols, call order), §7.2 (acceptance table), §8 (plan, fingerprints, manifest, commit order, reconcile, plan json, config), §9.1 (sidecar v2), §9.3 (label columns), §10.1 (audit), §11 and §11.1 (FakeEnv, MemoryWriter, loop), §13, and **§15** (binding clarifications, including the audit-record format WP12 reads). Then the prompts of the packages you call: `WP3.md` (expert), `WP5.md` (plan/manifest/fingerprints), `WP6.md` (perturbations, labels, `build_spec`), `WP12.md` (merge/upload read your audit report).

## Acceptance

```
uv run --group host pytest tests/core/test_v4_wp7_*.py tests/core/test_no_duplicates.py -q
```

all pass (about 30 s on a laptop), and every earlier test file still passes. Do not edit, skip or weaken any test, and do not change `tests/v4_support/`. If a test contradicts the contract, stop and report the test name, the clause and your reasoning.

## Decisions you must follow (the tests pin all of them)

### FakeEnv (§11)

- `FakeEnv(task, chain, gripper, *, fps=30, alpha=0.5, cameras={}, push_radius_m=0.01, crash_at_step=None)`; `joint_names = chain.q_names`; `camera_names = tuple(cameras)`; `cameras` maps name → `(h, w)`; `render()` returns `np.zeros((h, w, 3), np.uint8)` per camera.
- `reset(spec)`: every object of `task.spec.objects` exists; objects not in the spec are inactive. Unknown object names in the spec or a wrong `initial_q` length raise `EnvError`. Frame 0. `held_object` is attached at the TCP (object centre at the TCP position) with the gripper set to gap `w`.
- Geometry: `w` = min footprint side, `h` = height (`size`, or `params.size_outer` for USD objects); centre = position (+ h/2 for `origin = "bottom"`); a container is an object with `params.inner_half_size`.
- `step`: exactly the §11 order. Tracking `q ← q + alpha·(clip(target, chain.lower, chain.upper) − q)`. Jaw box in the TCP frame: `d = R_tcpᵀ(c − p_tcp)`, attach iff `hypot(d_x, d_y) ≤ w/2` and `|d_z| ≤ h/2` (and the gap decreased this step and `gap ≤ w + 0.004`); the nearest qualifying object wins; then the gripper angle is set to `angle_from_gap(w)`. Detach when `gap > w + 0.002`; drop to the highest support (table at `[task.start].table_z_m`, a container floor `z_c + floor_z` inside its inner rectangle, otherwise the top of any active object whose footprint contains the xy), yaw kept, roll/pitch 0. Push as in §11 (containers are never pushed).
- `signals` always contain `holding`, `grasped`, `lifted`, `placed`, `success`. `success` is true on a frame when `placed` has been true on the last `[task.success].confirm_steps` observed frames, counting the current one (the reset frame counts).
- `set_object_pose(name, pose)`: exact pose, no drop, detaches if held; unknown name → `EnvError`.
- `crash_at_step = k`: the k-th `step` call (0-based, counted over the object's life, across resets) raises `SimulatedCrash` before anything moves. `SimulatedCrash` subclasses `Exception` but **not** `GenerationError`.
- Deterministic; no randomness at all.

### Lite writer (§11) — `core/datasets/frames.py`

- `LiteDatasetWriter(root, *, fps, joint_names)`; `MemoryWriter(root, *, fps, joint_names, crash=None, crash_on_episode=None)` is the same writer plus crash hooks.
- `meta/info.json` = `{"codebase_version": "v3.0-lite", "fps", "total_episodes", "features"}`, written atomically (tmp + fsync + `os.replace`); that replace is the commit point. `features`: `action` and `observation.state` → `{"dtype": "float32", "shape": [J], "names": list(joint_names)}`, plus the three `LABEL_FEATURES` with `shape` as a list. No camera features: images are validated (uint8, `(H, W, 3)`) and dropped.
- One file per episode, `data/chunk-000/file-{index:06d}.parquet`, written to a temp file and renamed before `info.json` is updated. Columns and arrow types: `action`, `observation.state` `list<float32>`; `episode_index`, `frame_index`, `index` (global running index) `int64`; `timestamp` `float32` = frame / fps; `task` `string` (the instruction); `label.stage`, `label.imitate` `list<int64>` of length 1; `label.ee_pose` `list<float32>` of length 7.
- Opening an existing root: check `codebase_version`, `fps` and `features` (`ValueError` if different); delete data files with index ≥ `total_episodes` and any `*.tmp`; `episode_lengths()` from the committed files.
- `add_frame`: `ValueError` for a wrong action/state shape or a bad image; `TypeError` if `labels` is not `FrameLabels`. `save_episode` on an empty buffer: `RuntimeError`. `discard_episode` on an empty buffer is a no-op.
- Crash hooks: `crash_on_episode=None` arms the first save; otherwise the save of episode index `crash_on_episode`. `"before_save"` raises `SimulatedCrash` before writing anything (the buffer stays); `"after_save"` commits, then raises. A hook fires once per writer object.
- `read_frames(root)` reads every `data/**/*.parquet` (lite and real LeRobot v3: list or fixed-size-list vectors, scalar or length-1-list label columns; rows in any order, several episodes per file) and returns `{episode_index: {"action" (N, J) float32, "observation.state" (N, J) float32, "label.stage" (N,) int64, "label.imitate" (N,) int64, "label.ee_pose" (N, 7) float32, "frame_index" (N,) int64}}` sorted by episode and frame.

### `run_episode` (§11.1)

- Only adds frames: never calls `save_episode` or `discard_episode`. An attempt that passes `accept` returns `status=accepted`, `reason=None`, `episode_index=None`.
- Setup: `env.reset(spec)`, `expert.reset(spec, obs)`, `tracker = task.make_tracker()`, `ctx = CheckContext.from_spec(spec, task.spec)`, `WorldPerturber(spec, task.spec, perturber_rng or rng_for(spec.seed, "perturbation"))`, noise = `ou_noise(spec.noise, ~mapping.gripper_mask, spec.max_steps, noise_rng or rng_for(spec.seed, "noise"))` when `spec.noise` is set. `settle = max(1, [task.expert].settle_steps or 10)`.
- Per frame t (t < `spec.max_steps`): non-finite observation → `invalid_state` (frame not added); `checks = task.observe_checks(obs.step_state(env.joint_names), ctx)`; `prog = tracker.update(checks)`; `act = expert.act(obs, prog)`; `target = mapping.clamp_radians(act.q_target + noise[t])` (non-finite → `invalid_state`); `images = env.render()`; `labels = frame_labels(prog.stage_index, act.injected, obs.ee)`; `writer.add_frame(mapping.to_raw(target), mapping.to_raw(obs.q), images, spec.instruction, labels)`; append a `StepRecord` (objects = poses of the active objects of `obs`); stop after this frame if the expert was `done` on `settle` consecutive frames (counting t) or t = max_steps − 1; otherwise `obs = env.step(target)`, then if `perturber.due(prog, t)`: `perturber.apply(env, obs)` and `obs = env.observe()`.
- Any exception from a GenEnv call (reset, render, step, apply, observe) except `SimulatedCrash` ends the attempt with `env_error`; `SimulatedCrash` propagates. Errors in the expert or task are bugs and propagate.
- Perturbation bookkeeping: group A → onset = first injected frame, window end = last injected frame + 1, applied = any injected frame. Group B → `perturber.applied` / `perturber.onset_frame`.
- Outcome = `task.outcome(tracker, checks of the last frame, obs.step_state(...) of the last frame)`; reason = `accept(kind, success=…, applied=…, reentered=bool(reentries), reentered_before_onset=any re-entry frame < onset)`, with `env_error` / `invalid_state` taking precedence. Re-entries are `(t, act.resumed_stage)` for frames with `act.reentered`. `recovery = build_recovery(...)` only for `Kind.RECOVERY` with ≥ 1 re-entry. `segments = build_segments(N, kind, onset=…, injected_end=…, detected=…, back_on_plan=…, phases=[act.phase …])`.
- Fingerprints: `scenario_key(spec)`, `spec_fp(spec)`; for N ≥ 1 frames and no error: `traj_fp(stacked action_raw, stacked state_raw, objects0)` with **objects0 = frame-0 positions of the active objects rounded to 6 decimals** (`round(float(c), 6)`, exactly what the sidecar stores, so the audit recomputes the same value), `appearance_fp({active name: spec colour}, spec.instruction, None, None)`, `episode_fp(traj, appearance)`. Otherwise those three are `None`.

### `run_generation` (§8, §11.1)

- `task = load_task(task_name)`; root = `writer.root`. Write `generation_plan.json` with `write_plan_json` (§8.5): `config_sha256 = cfg.sha256`, `config = cfg.to_json()`, `git_commit`, `shard`, `task`, `task_toml_sha256` (sha256 of `task.spec.task_dir / "task.toml"`), `arm`, `arm_files_sha256` = sha256 of `arm.toml` and `kinematics.json` in the arm folder, `scene`, `quotas` (totals by kind value), `holdout` (`task.spec.holdout`), `created_at` (UTC ISO-8601). Then `reconcile(root, writer, manifest)`.
- Walk `shard_attempts(iter_plan(task_name, cfg.master_seed, spec_fn, unique_across_kinds=cfg.scenario_unique_across_kinds), i, n)` **from the start**. Shard quota per kind = `shard_quota(total, i, n)`. Stop when every kind's quota is met. Pass over a position of a quota-met kind without writing anything.
- A position that already has a manifest line is not executed again; its status updates the counters (accepted, executed non-skipped, consecutive skips) exactly as when it was first run. This is what makes a resumed run equal to an uninterrupted one.
- Skip (`planned.skip` set): append a `skipped` line (`scenario_key` and `spec_fp` from the plan; both null for `sampling_failed`); after `max_consecutive_skips` consecutive skips of one kind raise `ScenarioSpaceExhausted` (the line is written first).
- Before executing: if the kind already has `ceil(max_attempt_factor × shard quota)` executed non-skipped attempts, raise `AttemptBudgetExceeded`.
- Execute with `expert_factory(spec)`. If accepted but its `traj_fp` equals the `traj_fp` of an accepted line of this dataset (rebuilt from the manifest on resume), it becomes `rejected / duplicate_trajectory`.
- Accepted: `index = writer.episode_count()`; `write_sidecar(root, sidecar)`; `writer.save_episode()` must return `index`; then the manifest line (`ManifestLine.from_result(result with episode_index, shard=…, perturbation=spec.perturbation.type or None, wall_s=elapsed)`). Rejected: `writer.discard_episode()`, then the line.
- Sidecar (§9.1): `objects` = active objects with `set` (the object set whose slot names contain it, else null), `size_m`, `color`, `active: true`, `role` (alphabetically smallest role naming it, else null), per-frame `position` and `orientation_wxyz` (6 decimals); `ee_pose` from the frame labels; `joint_names = env.joint_names`; `reference_frame = f"robot base link ({[arm.links].base})"`; `stages = task.stages`; `phases = build_phases(...)`; `perturbation = spec.perturbation.to_json() + onset_frame + applied`; `attempt = {index, seed, shard, spec_fp}`; `appearance = {null, null}`; `source = null`; `generator = {"name": "scripted_expert", "git_commit": git_commit}`; `legacy_v1 = false`.
- `GenerationSummary` is computed from the whole manifest after the run: `accepted` {Kind: n} for all three kinds, `rejected` {reason value: n}, `skipped`, `attempts` (all lines).
- `make_expert_factory(task, arm, *, setup=None)`: parse the program once (`parse_program(..., joint_limits=expert_joint_limits(arm, setup=setup))`) and return `spec -> ScriptedExpert(program, chain, gripper, lower, upper, limits=MotionLimits.from_arm(arm), fps=spec.fps, stages=task.stages)`.
- `make_spec_fn(task, arm, *, scene, noise=None, weights_override=None, fps=30, setup=None, reach_fn=None)` returns `(kind, index, seed) -> build_spec(...)` with `home_q=home_q(arm)`, `joint_names`, `joint_limits=expert_joint_limits(arm, setup=setup)` and `arm_joints` from `[arm.expert]`. `home_q(arm, *, setup=None)` = `[arm.initial_state.joint_positions]` in `[arm.joints].names` order, overridden by `[arm.expert].home_rad` when present.

### Audit (§10.1, §15 WP12)

- `CHECKS` = the ten names of §10.1 in that order; `AuditReport.checks` is a tuple of `{"name", "passed", "details"}` dicts in that order, at most 50 details each, at least one detail for a failed check.
- `counts_match`: `info.json` `total_episodes` = data episodes = sidecars; indices 0..E−1; per-episode frame counts = `num_frames`; and when the manifest exists and `check_manifest`, the accepted lines are exactly the originals (`source` null) with equal `scenario_key`, `spec_fp`, `traj_fp`, `appearance_fp`, `episode_fp`, kind, attempt index, seed and `num_frames`. An unreadable manifest fails this check.
- `labels_match`: stage and imitate exact, ee pose within 1e-5 (sidecars with null `stage` / `ee_pose` are skipped).
- `fingerprints_recomputed`: `traj_fp(data action, data state, {name: sidecar position[0]})`, `appearance_fp(sidecar colours, instruction, appearance.light_exposure, appearance.robot_color)`, `episode_fp(...)`; all three must equal the sidecar.
- `episode_fp_unique` over all episodes; `traj_fp_unique_originals` and `scenario_key_unique_originals` over originals only.
- `holdout_absent`: hold-outs from `meta/generation_plan.json` if present, else from `load_task_spec(task).holdout`; fail on a held-out colour on any object or a held-out count of the target's object set.
- `dr_sources_exist`: when `source_root` is given, each copy's `source.traj_fp` and `source.episode_fp` must exist in the source; without `source_root` the check passes with a detail containing "skipped".
- `dr_appearance_distinct`: copies grouped by `source.episode_fp` differ from each other and from the parent (in this dataset or in `source_root`) in `appearance_fp`, and keep `traj_fp == source.traj_fp`.
- `content_digest(root)` = `json_sha256({posix path relative to root: file sha256})` over every regular file under `meta/`, `data/`, `videos/`, `task_meta/` except `meta/audit_report.json` (the same formula WP12 checks). `fingerprints_digest = json_sha256(sorted episode_fp)`.
- `write_audit_report` writes the eight fields as JSON (`checks` as a list) atomically to `meta/audit_report.json`; `read_audit_report` raises `SchemaError` on bad JSON, wrong keys or types, or non-hex digests, and returns an `AuditReport` equal to the one written.

### Scripts

- `audit_dataset.main(argv=None) -> int`: positional `root`, `--source_root`, `--no_manifest`; writes the report; prints one line per check (`PASS <name>` or `FAIL <name>: <details>`) in `CHECKS` order, then a summary line; returns 0 if passed else 1.
- `plan_generation.main(argv=None) -> int`: `--config_path` (required), `--task`, `--shard i/n` (override); for each task prints one JSON object with keys `task`, `shard`, `quotas`, `shard_quotas`, `non_skipped`, `horizon` (kinds by value) from `check_capacity(..., unique_across_kinds=cfg.scenario_unique_across_kinds)` with `make_spec_fn`; on any `GenerationError` prints `<ErrorClass>: <message>` to stderr and returns 1. No Isaac imports.

### Imports

Only stdlib, numpy, pyarrow (only in `core/datasets/*`) and existing / earlier-package `core` modules. No `torch`, `lerobot`, `huggingface_hub`, `fsspec`, Isaac, `pxr` or `sim` imports (a fresh-interpreter test checks this). No SO-101 string literals. Python 3.11 syntax.

## Notes

- The fixture tasks put the table at z = −0.05 m and keep start regions inside the fx arms' reachable workspace; per-frame cost is dominated by the expert's IK step (WP3), so keep `run_episode` free of extra FK calls.
- WP9, WP10 and WP13 will call `v4_support.harness.run_no_duplicates_suite` for the 17 real tasks; keep `run_generation` free of fixture-specific behaviour.
