<!-- Detailed design from the planning pass. docs/platform_v4.md holds the approved plan and wins on any conflict. -->

# v4 plan: automated dataset generation for all 17 tasks

Nothing was written to disk and no code was run. The plan is checked against the repo by reading only. LeRobot API names (`aggregate_datasets`, `StreamingLeRobotDataset`) are from memory, because LeRobot is not installed on this host.

## 0. Repo facts that shape the plan

- **Per-task `env_cfg.py` is unavoidable.** `TaskSpec.entry_point` (`src/core/tasks/base.py`) always prefixes `s2r_tasks.<task>.`, so each task folder needs its own (thin) `env_cfg.py`.
- **One task per process.** `register_envs` (`src/sim/envs/__init__.py`) registers only the `S2R_TASK` task.
- **Single-cube assumptions in replay.** `SceneReplayer` has one `self.cube`, `SourceEpisode.poses_base` is one `(N,7)` array, `dr_replay.make_env` hard-codes `env_cfg.events.spawn_cube = None`, and `dr_variants.task_text` hard-codes `lang_blue`/`lang_red`.
- **SO-101 names are hard-coded outside `arm.toml`.**
  - `{ENV_REGEX_NS}/Robot/jaw` in `tasks/pick_place_cube/env_cfg.py`
  - `("gripper", "jaw")` in `src/sim/scene/arm.py:30`
  - `MOVING_JAW_BODY`/`MOVING_JAW_MESH` in `src/sim/eval/grasp_diagnostics.py:16-17`
  - `moving_jaw_so101_v1` in `src/sim/scene/contact_materials.py:95`
  - the five joint names in `tests/sim/check_grasp_stability.py`
- **Joint limits are per scene, not per arm.** `sim_joint_mapping` and `runtime_joint_limits` sit in `scenes/rpd_meeting_room/real_setup.json`, keyed by LeRobot joint names, with one arm per scene.
- **`JointMapping.to_radians` clamps to runtime limits.** A recorded action only equals the executed action if the expert clamps first.
- **Parked objects overlap.** `park_object` defaults every object to `(0,0,-2)`; with N cubes they need distinct offsets.
- **The box can be hand-written.** `tasks/pick_place_cube/usd/White-Box-Physics.usda` is text USDA, so a 4 cm box can be authored and parsed in CPU tests.
- **Reusable as-is:** `core.utils.hashing.canonical_json/json_sha256`, `dr_variants.episode_seed/write_json_atomic`, and `lerobot_eval.load_yaml_config` (the pattern for `configs/generate`).
- **Host test deps:** the `host` group has numpy, scipy, torch, pyarrow and pytest, but no lerobot and no usd-core.

## 1. Architecture

Your proposed split is right. I would refine it in six places.

1. **Three narrow protocols are the seam.** `GenEnv`, `EpisodeWriter` and `Expert` live in `src/core/generate/protocols.py`. The whole loop (`core.generate.loop.run_generation`) is pure Python against them. Isaac provides `IsaacGenEnv` plus a `LeRobotRecorder` adapter; tests provide `FakeEnv` plus `MemoryWriter`. The Isaac runner should stay under about 200 lines with no decisions in it.
2. **The expert is closed-loop.** It reads measured ee and object poses every step and outputs absolute joint targets in radians, USD order, clamped to runtime limits. This is needed for the 2 cm-in-4 cm placement and for recoveries.
3. **Experts are data.** `[task.expert] steps = [...]` composes a shared skill library in `src/core/expert/skills.py`. A task may add `tasks/<task>/expert.py` only if it needs a custom skill. This keeps the 13 primitive folders tiny.
4. **Success checks for new tasks are geometric and live in core.** `[task.checks]` holds predicates over `StepState` (ee near, object lifted, object in region, object on object, held, joint near, gap near, object displaced). They are CPU-testable and arm-agnostic. `pick_place_cube` keeps its contact-sensor signals from `tasks/pick_place_cube/mdp.py`, passed through as `StepState.signals`.
5. **Kinematics go in a generated file, not hand-edited `arm.toml`.** `arms/<arm>/kinematics.json` is referenced from `[arm.kinematics] file = ...`. A script should not rewrite a commented TOML; `arms/so101/calibration/stats.json` is the precedent.
6. **Attempts are not episodes.** Quotas are counted by observed outcome. The plan is an infinite deterministic stream of attempts per (task, kind), and every attempt, including rejected ones, is recorded in the manifest.

**The fake env cannot validate physics.** `FakeEnv` has first-order joint tracking, an "attach when the jaw closes around the object, detach on open, drop to support" rule, and teleports. It validates logic, labels, resumability and uniqueness. It does not validate grasp success, pushing, stacking stability or bowl nesting.

**Container-only checks** follow the existing `tests/sim/check_*.py` pattern, and the owner runs them:

| Script | What it proves | Pass threshold |
|---|---|---|
| `check_kinematics_vs_sim.py` | FK matches `ee_frame_state` over 200 random joint vectors | < 1 mm, < 0.5° |
| `check_multi_object_scene.py` | parking, teleport and recolour of N cubes render correctly | manual image check |
| `check_expert_task.py --task X --n 20` | nominal expert success rate in real physics | ≥ 80% target tasks, ≥ 90% primitives |
| `check_generate_smoke.py` | 10 episodes per kind; audit passes; `LeRobotDataset` opens | all pass |
| `check_dr_replay_v2.py` | DR re-render of a multi-object dataset | audit passes |
| `check_streaming.py` | streamed read of an uploaded shard | frames decode |

## 2. Files

### `src/core` (no Isaac)

- `src/core/tasks/base.py` (change)
  - Add `object_sets` expansion and `roles` to `TaskSpec`.
  - Add concrete defaults for `failure_mode(flags, state)` and `sample_start` from `[task.start]`, and add `StageTracker`.
  - Keep abstract methods so `PickPlaceTask` is untouched, except that it drops its own `failure_mode` in favour of an override hook.
- `src/core/tasks/generic.py` (new): `GenericTask(Task)` driven by `[task.stages]`/`[task.checks]`, plus the predicate registry.
- `src/core/tasks/sampling.py` (new): start sampler with regions, yaw, minimum separation, clear-of-container, reachability filter via IK, count and colour sampling.
- `src/core/arms/kinematics.py` (new): `KinematicChain.from_arm(name)`, `fk(q)`, `jacobian(q)`, `ik(target, seed, weights)`.
  - IK is damped least squares with a scipy `least_squares` fallback.
  - It works for any n-DoF arm, with weighted position, tool axis and roll about the tool axis.
- `src/core/arms/gripper.py` (new): opening in metres ↔ jaw angle, from the `[arm.gripper]` table.
- `src/core/arms/mapping.py` (small change): add numpy `to_raw_np` and `clamp_radians_np`; torch stays.
- `src/core/arms/joint_limits.py` (small change): accept `setup["arms"][ARM_NAME]` when present, else today's keys.
- `src/core/expert/` (new)
  - `motion.py`: joint and Cartesian interpolation with speed limits.
  - `skills.py`: move_joint, set_gripper, reach, approach, grasp, lift, transport, place, release, retreat, push, touch, sweep, wait.
  - `machine.py`: `ScriptedExpert`, a deterministic state machine with phase names and a re-entry rule for recovery.
  - `program.py`: parse and validate `[task.expert]`.
- `src/core/generate/` (new)
  - `protocols.py`: `GenEnv`, `EpisodeWriter`, `Expert`, `ExpertObs`, `EpisodeSpec`, `EpisodeResult`.
  - `plan.py`: seed schedule, attempt stream, shards, quotas.
  - `fingerprint.py`, `manifest.py`, `perturbations.py`, `labels.py` (segments and imitate mask).
  - `loop.py`: `run_generation`.
  - `fake_env.py`: `FakeEnv`, `MemoryWriter`.
  - `config.py`: YAML → dataclass.
- `src/core/datasets/` (new)
  - `task_meta.py`: sidecar v2 writer, reader and validator, and the v1 reader.
  - `audit.py`: recompute fingerprints from parquet and assert uniqueness and consistency.
  - `consolidate.py`: per-episode JSON → `task_meta/episodes.parquet`.
  - `merge.py`: shard merge index maps.
  - `upload.py`, `streaming.py`.
- `src/core/scripts/` (new files): `plan_generation.py` (dry run, prints quotas and estimated hours), `audit_dataset.py`, `upload_dataset.py`, `merge_shards.py`.

### `src/sim` (Isaac)

- `src/sim/envs/generic_task_cfg.py` (new): `apply_teleop` and `apply_generate`.
  - Builds every object in `spec.objects` (cuboid or USD) with distinct park offsets.
  - Jaw contact sensor path comes from `arm.toml`.
  - No terminations.
- `src/sim/envs/mdp/objects.py` (new): `set_object_pose_base`, `set_object_color` (generalised from `set_cube_color` in `tasks/pick_place_cube/mdp.py`), `object_poses_in_base` (generalised from `lerobot_agent._object_pose_in_base`).
- `src/sim/generate/isaac_gen_env.py` (new): `IsaacGenEnv`, implementing `GenEnv`. It reuses `dr_replay.make_env`, `discover_sim_cameras` and `make_mapping`.
- `src/sim/generate/scripted_generate.py` (new): `main()` parses the config, builds the env and writer, and calls `core.generate.loop.run_generation`.
- `src/sim/generate/extract_kinematics.py` (new): USD → `arms/<arm>/kinematics.json`, with the logic lifted from `tests/sim/check_grasp_stability.py:14-52`.
- `src/sim/generate/dr_replay.py`, `dr_variants.py`, `src/sim/recording/replay_scene.py`, `replay_source.py` (change): see section 4.
- `src/sim/recording/episode_metadata.py` (change): `save_cube_trajectory` and `validate_cube_trajectories` stay as thin wrappers, so teleop is untouched.
- `src/sim/recording/lerobot_recorder.py` (small change): optional extra low-dimensional features (see open question 3).

### CLI, configs, deps

- `src/cli/paths.py` `SIM_PROGRAMS` rows:

  | CLI name | Program |
  |---|---|
  | `generate` | `scripted_generate` |
  | `generate-plan` | `plan_generation` |
  | `arm-kinematics` | `extract_kinematics` |
  | `dataset-audit` | `audit_dataset` |
  | `dataset-merge` | `merge_shards` |
  | `dataset-upload` | `upload_dataset` |

- `pyproject.toml`: matching console scripts.
- `configs/generate/`: `primitives.yaml`, `pick_place_cube.yaml`, `stack_two_cubes.yaml`, `stack_bowls.yaml`, `pick_place_color_cube.yaml`, `smoke.yaml`. Keys: arm, scene, task(s), master_seed, quotas per kind, shard `i/n`, dst_root, repo_id, perturbation mix, `max_attempt_factor`, dr block.

### Arms and tasks

- `arms/so101/arm.toml` (add): `[arm.kinematics]`, `[arm.gripper]`, `[arm.expert]`, `links.moving_jaw`, `links.moving_jaw_mesh` (see section 6).
- `arms/so101/kinematics.json` (generated once in the container, then committed).
- `tasks/prim_*/` (13 folders), each with:
  - `task.toml`;
  - `__init__.py`: `TASK = GenericTask(load_task_spec("<name>"))`;
  - `env_cfg.py`: `from sim.envs.generic_task_cfg import apply_teleop, apply_generate`.
  - No `mdp.py`, `eval.py` or `benchmark.py`.
- `tasks/pick_place_cube/` (change): add `[task.start]`, `[task.expert]`, `[task.perturbations]` and a `generate` variant to `task.toml`. `mdp.py` is unchanged. `benchmark.py` `StageTrace.finish` delegates to `Task.failure_mode`.
- `tasks/stack_two_cubes/`, `tasks/stack_bowls/`, `tasks/pick_place_color_cube/`: same thin trio, plus `usd/` where needed (hand-written `Small-Box-Physics.usda`, bowl asset).

### New `task.toml` tables

```toml
[task.object_sets.cubes]        # expands to cube_0..cube_5 in spec.objects
count_max = 6
size = [0.02, 0.02, 0.02]
mass = 0.008
[task.palette]                  # name -> rgb
red = [0.8, 0.05, 0.05]
[task.roles]                    # bound per episode
target = "cubes"                # sampled slot of this set
container = "small_box"
[task.start]
count = [2, 6]
region = { xy_min = [..], xy_max = [..] }
yaw_deg = [0, 90]
min_separation = 0.035
distinct_colors = true
[task.instruction_template]
text = "Pick up the {target.color} cube and place it in the box"
[task.stages]
order = ["reached", "grasped", "lifted", "over_box", "placed", "success"]
[task.checks.placed]
check = "object_in_region"
object = "@target"
region = "@container.inner"
[task.failures]                 # ordered rules -> failure name
[task.expert]
steps = [{ skill = "reach", target = "@target", offset = [0, 0, 0.05] }]
[task.perturbations]
failure = [{ type = "grasp_offset", weight = 2, xy = [0.008, 0.02] }]
recovery = [{ type = "object_shift", stages = ["reached"], xy = [0.02, 0.05] }]
```

### Primitive semantics

These are my reading of the task names; please confirm them against your catalogue.

| Primitive | Meaning |
|---|---|
| `prim_joint_sweep` | one joint moves between two sampled angles |
| `prim_gripper_gap` | open the jaw to a sampled gap |
| `prim_reach_point`, `prim_test_reach`, `prim_calibration_reach` | ee to a sampled point, a workspace-boundary point, or a fixed calibration grid |
| `prim_waypoint_path` | visit 2–4 waypoints |
| `prim_touch_object`, `prim_nudge` | contact an object; small displacement |
| `prim_push_to_region` | push an object into a region |
| `prim_lift_test`, `prim_grasp_lift` | grasp, lift a little, set down; grasp, lift and hold |
| `prim_place_at` | starts holding the object, places it at a pose |
| `prim_regrasp_after_drop` | starts with a scripted drop, then regrasps |

## 3. Multi-object, multi-colour scenes

- **Build.** `generic_task_cfg` creates all `count_max` cubes once, each parked at `(0.1·i, 0, -2)`. The reset event parks all of them.
- **Per episode.** `IsaacGenEnv.reset(spec)` teleports the active slots to their sampled base-frame poses, leaves the rest parked, and sets each active cube's material colour. This reuses the cached-material approach of `set_cube_color`.
- **Sampling** (core, CPU-tested):
  - draw the count n from `[count_min, count_max]`;
  - draw n colours from the palette, with the target colour unique;
  - draw a random slot permutation, so slot index never correlates with colour or with being the target;
  - draw poses with minimum separation and clear of the box.
- **Binding.** `EpisodeSpec.roles = {"target": "cube_3", "container": "small_box"}` and `EpisodeSpec.colors = {"cube_3": "red", ...}`. The instruction is rendered from the template. Checks reference `@target`, so success means the target slot is in the box.
- **Wrong-object failure.** A distractor in the box gives `wrong_object_placed`, which is also a perturbation type (`wrong_target`).
- **Contact sensor.** The filter is a list of all cube prims. Whether Newton returns per-filter forces must be checked in the container. The geometric `held` check is the fallback, so this is not blocking.
- **Hold-out.** Config keys `holdout_colors` and `holdout_counts` are excluded from generation and listed in the manifest. A test asserts they never appear.

## 4. Failures, recoveries and `dr_replay`

**Perturbation taxonomy** (`core/generate/perturbations.py`, each a pure function of spec and RNG):

| Group | Types |
|---|---|
| A, expert error (bad robot action) | `grasp_offset` (xy/z/yaw), `early_close`, `partial_close`, `early_release`, `place_offset`, `wrong_target`, `stall` |
| B, world disturbance (not a robot action) | `object_shift` before grasp, `object_drop` (held object teleported to the table), `container_shift` |
| C, noise | bounded Ornstein–Uhlenbeck joint-target noise, used on success episodes |

**Modes:**
- **Failure:** A or B, with the expert in `blind` mode. It keeps its stale plan and performs no re-entry.
- **Recovery:** A or B, with the closed-loop expert. Detection is a stage regression or target displacement above a threshold. The expert then re-enters the earliest unsatisfied stage, with at most `max_retries` re-entries.

**Labels come from the observed outcome, not the intent:**
- A "failure" attempt that succeeds is rejected as `perturbation_ineffective`.
- A "recovery" attempt is accepted only if a regression was observed and the final result is success.
- The failure name comes from `Task.failure_mode`.

**Sidecar v2** at `<root>/task_meta/episode_NNNNNN.json` contains:
- `schema_version: 2`, `task`, `arm`, `scene`, `fps`, `num_frames`, `reference_frame`, `units`;
- `objects: {name: {size_m, color, active, role, position[N,3], orientation_wxyz[N,4]}}`;
- `ee_pose[N,7]`, `instruction`, `roles`;
- `kind` (success | failure | recovery), `success`, `failure_mode`, `furthest_stage`;
- `stage[N]`, a per-frame stage index;
- `segments: [{label: nominal | perturbation | failure | recovery, start, end, phase}]`;
- `imitate[N]`, false on injected-error frames;
- `perturbation: {type, params, onset_frame}`, `recovery: {detected_frame, resumed_stage, retries}`;
- `attempt: {attempt_index, seed, spec_fp}`, `fingerprints`, `source` (for DR copies).

**Recovery segmentation:** `nominal [0, onset)`, `perturbation [onset, detected)`, `recovery [detected, back_on_plan)`, `nominal [back_on_plan, N)`. A property test asserts the segments are contiguous, non-overlapping and cover `[0, N)`.

**Reader compatibility.** `task_meta.load_episode` falls back to `pick_place_meta/` and maps v1 to v2, with one object `blue_cube`. All four current readers go through it:
- `replay_source.load_sidecar`
- `tasks/pick_place_cube/eval.py` `load_starts` / `start_episode_ids`
- `tasks/pick_place_cube/benchmark.py` `generate_eval_set`
- the GR00T prepare script

**`dr_replay` changes:**
- `SourceEpisode` gains `objects: {name: (N,7)}`, `colors`, `roles` and `meta`; `poses_base` stays as an alias for the target.
- `SceneReplayer.pose_frame(state_raw, poses_by_name)` poses every active object and parks inactive ones.
- `make_env` disables all park and spawn events generically, not only `spawn_cube`.
- `dr_variants.plan_episodes` replaces blue/red with a per-task draw: light, robot colour, and optionally a colour re-assignment from the palette that keeps the target colour unique and inside the non-held-out set.
- The instruction is re-rendered from the template. `--lang_blue` and `--lang_red` remain only for v1 sources.
- Output is a v2 sidecar copied from the source with new colours, `source` and `fingerprints.appearance`.
- Failures and recoveries are re-rendered too, because the replay is kinematic.

## 5. Duplicate prevention

**Definitions** (`core/generate/fingerprint.py`, all SHA-256 over `canonical_json`):

| Fingerprint | Covers | Purpose |
|---|---|---|
| `scenario_key` | task, arm, scene, sampled start and goal on a coarse grid (1 mm, 1°, joint 0.5°), active slots with roles | no two attempts start from the "same" situation |
| `spec_fp` | the full `EpisodeSpec` at fine quantisation (0.1 mm, 0.01°): kind, perturbation params, colours, instruction; seed and indices excluded | identity of what was asked |
| `traj_fp` | `round(action·100)` and `round(state·100)` as int32 bytes plus shapes plus frame-0 object poses at 0.1 mm | identity of what happened |
| `appearance_fp` | object colours, light exposure, robot colour, instruction; `"nominal"` for originals | identity of how it looks |
| `episode_fp` | `sha256(traj_fp + appearance_fp)` | **the dataset invariant: globally unique** |

**Guarantees:**

1. **Deterministic attempt stream.** `seed(master, task, kind, i)` follows the `dr_variants.episode_seed` pattern, and the spec is a pure function of that seed. Each shard computes the whole stream on CPU and takes `i % n == shard`.
2. **Deterministic skips.** If `scenario_key` was already seen earlier in the stream, attempt `i` is skipped identically in every shard and on every resume. By default the key is unique across kinds within a task (config `scenario_unique_across_kinds = true`).
3. **Exhaustion fails early.** If a discrete task cannot supply its quota (for example `prim_calibration_reach` on a fixed grid), `plan_generation` fails with `ScenarioSpaceExhausted`.
4. **Post-hoc check.** After an episode, a `traj_fp` collision with any accepted episode causes rejection as `duplicate_trajectory`.
5. **DR copies.** A copy shares `traj_fp` with its parent by design. `appearance_fp` must differ from the parent's and from its siblings', otherwise the draw is redone with the next sub-seed. `source.traj_fp` must exist in the source dataset.
6. **Storage.**
   - `meta/generation_manifest.jsonl`: append-only with fsync, one line per attempt (accepted, rejected with reason, or skipped).
   - `meta/generation_plan.json`: config hash, master seed, git commit via `dr_replay.git_commit`, shard, hold-outs.
   - The same fingerprints in each sidecar.
7. **Resume.** The commit order is sidecar, then `save_episode`, then the manifest line. On start, the manifest is reconciled with `LeRobotRecorder.episode_lengths()` and the sidecars: a missing line is rebuilt from the sidecar, an orphan sidecar is deleted, and the stream continues at the first unrecorded attempt. An attempt index is never executed twice.
8. **Merge.** `merge_shards` refuses any `episode_fp` or `scenario_key` collision.

**Main test** (`tests/core/test_no_duplicates.py`, CPU, FakeEnv):
- Generate all 17 tasks × 3 kinds at small quotas.
- Kill and resume at random points (property test over kill positions).
- Run 3 shards and merge.
- Run fake DR ×2.
- Run `audit` and assert:
  - all `episode_fp` are unique;
  - `traj_fp` is unique among originals;
  - `scenario_key` is unique;
  - manifest accepted count = dataset episodes = sidecar count;
  - fingerprints recomputed from the stored arrays equal the stored ones;
  - a resumed run is identical to an uninterrupted run;
  - injecting a copied episode makes the audit fail.

The same `dataset-audit` command runs on the real dataset before upload, and upload refuses without a passing audit report.

## 6. Arm-agnosticism

Code reads nothing arm-specific except through `load_arm_spec`. Additions to `arm.toml`:

- `[arm.links] moving_jaw`, `moving_jaw_mesh`: replaces the hard-coded names listed in section 0.
- `[arm.kinematics] file`, `base`, `ee`, `tcp_offset` (position and quaternion in the ee frame), `approach_axis`, `closing_axis`.
- `[arm.gripper] joint`, `open`, `closed`, `opening_table` (angle → gap in metres), `max_object_width`.
- `[arm.expert] arm_joints`, `home`, `max_joint_speed`, `ik_orientation_weight`. A 5-DoF arm uses a low roll weight; a 6-DoF arm such as YAM uses full weights.

Other requirements:
- `kinematics.json` holds the joints (name, parent, child, axis, the two local frames, USD limits) and comes from `s2r arm-kinematics`.
- The joint count comes from `FOLLOWER_JOINTS` everywhere, as `replay_source` already does.
- Mapping and limits allow an optional `arms.<arm>` block in `real_setup.json`.

**Proof of agnosticism (CPU test):** a synthetic 6-DoF-plus-gripper arm folder in a temp `S2R_ARMS_DIR`, with a hand-written chain, runs the whole FakeEnv generation for three tasks with zero code changes. A static test also greps `src/core/expert`, `src/core/generate` and `src/sim/generate` for SO-101 joint and link literals.

**A new arm still needs** its USD, its `kinematics.json`, mapping and limits, and a container run of the expert smoke checks. Grasp tuning values may differ, but they are TOML.

## 7. Work packages

For every package, Opus writes the tests first and Codex makes them pass. Packages in the same wave can run as separate Codex sessions.

**Wave 0**

- **WP0: contracts.** Files: `core/generate/protocols.py`, dataclasses only.
  - Tests: construction, immutability, JSON round-trip of `EpisodeSpec` and `EpisodeResult`.
  - Acceptance: every later WP imports only these types across package boundaries.

**Wave 1 (parallel: WP1, WP2, WP4, WP5)**

- **WP1: task contract v2.** Files: `core/tasks/{base,generic,sampling}.py`.
  - `object_sets` expand to N `ObjectSpec`s.
  - Role resolution `@target` works.
  - Each predicate is covered by truth-table cases.
  - `furthest_stage` is monotone.
  - `failure_mode` reproduces all six pick-place failure names from `StageTrace.finish` on recorded flag sequences.
  - Property tests on the sampler: inside region, separation ≥ minimum, clear of container, target colour unique, same seed gives the same result, hold-outs never drawn.
  - The existing `tasks/pick_place_cube` still loads.
- **WP2: kinematics.** Files: `core/arms/{kinematics,gripper}.py`, the `mapping.py` numpy functions, `arm.toml` additions, `sim/generate/extract_kinematics.py`.
  - Property: `ik(fk(q))` reproduces the pose within 0.2 mm for random in-limit q.
  - The analytic Jacobian matches finite differences.
  - IK never leaves limits, and an unreachable target returns a flagged failure rather than an exception.
  - `to_raw_np(to_radians_np(x)) ≈ x` inside limits.
  - Gripper table is monotone and invertible.
  - Synthetic 6-DoF chain passes the same tests.
  - Tests use a fixture chain until the container extraction is committed; the pure builder is tested on a dict mimicking USD joint data.
  - Container: `check_kinematics_vs_sim.py`.
- **WP4: sidecar v2.** Files: `core/datasets/task_meta.py`, the `episode_metadata.py` wrappers, `replay_source.py`.
  - Write and read round-trip.
  - Atomic write: a simulated crash leaves no partial file.
  - Validator rejects NaN, wrong lengths, non-unit quaternions, and segment gaps or overlaps.
  - A v1 fixture loads as v2.
  - `validate_cube_trajectories` behaviour is unchanged on v1 roots.
  - `load_source` returns multi-object episodes.
- **WP5: plan, fingerprint, manifest.** Files: `core/generate/{plan,fingerprint,manifest,config}.py`, `core/scripts/plan_generation.py`.
  - Fingerprints are stable across processes and platforms (golden values), insensitive below the quantisation step and sensitive above it.
  - The stream is identical regardless of shard count, and shards partition it.
  - A duplicate `scenario_key` is skipped deterministically.
  - Exhaustion raises.
  - Manifest append and reload work, a torn last line is tolerated, and the reconcile cases of section 5 item 7 pass.

**Wave 2 (parallel: WP3, WP6, WP12)**

- **WP3: expert.** Depends on WP1 and WP2. Files: `core/expert/*`.
  - Each skill terminates within its step budget on a kinematic stub.
  - Joint steps never exceed `max_joint_speed/fps`, and outputs stay within limits.
  - Deterministic given the same observations.
  - The program parser rejects unknown skills and roles.
  - The blind expert never re-enters; the closed-loop expert re-enters the correct stage after a stage regression.
  - Phase names are emitted on every step.
- **WP6: perturbations and labels.** Depends on WP0 and WP1. Files: `core/generate/{perturbations,labels}.py`.
  - Each type is a pure function of (spec, rng) with parameters inside declared ranges.
  - Onset happens only in allowed stages.
  - Property: segments cover `[0, N)`, and `imitate` is false exactly on group-A error frames.
  - Outcome-based acceptance rules are covered as a table of cases.
- **WP12: upload and streaming.** Depends on WP4. Files: `core/datasets/{consolidate,upload,streaming,merge}.py` and scripts.
  - Consolidated parquet equals the per-episode JSON.
  - Merge index maps are correct and refuse collisions.
  - The upload planner (pure) refuses without an audit report, is resumable, and uses `upload_large_folder` semantics behind an injected client.
  - Container or host with network: `check_streaming.py`.

**Wave 3**

- **WP7: FakeEnv, loop and audit.** Depends on WP3–WP6. Files: `core/generate/{fake_env,loop}.py`, `core/datasets/audit.py`.
  - The main no-duplicates test of section 5.
  - Quotas met by outcome, and `max_attempt_factor` aborts cleanly.
  - Recorded action equals the executed action after mapping round-trip.
  - Failed and recovery episodes carry correct labels.
  - A writer exception mid-episode leaves a consistent state.

**Wave 4 (parallel: WP8, WP9)**

- **WP8: Isaac adapter and runner.** Depends on WP7. Files: `sim/envs/generic_task_cfg.py`, `sim/envs/mdp/objects.py`, `sim/generate/{isaac_gen_env,scripted_generate}.py`, CLI rows, `pyproject.toml`, `configs/generate/smoke.yaml`.
  - CPU, in the existing AST and `sys.modules`-patch style: CLI plan argv, config parsing, `--dry_run` without Isaac, no SO-101 literals, the runner calls `run_generation` with the adapters.
  - Container: multi-object check and generate smoke.
- **WP9: 13 primitives.** Depends on WP7. Files: `tasks/prim_*`, `configs/generate/primitives.yaml`.
  - Parametrised over all 13: loads, expert program validates, FakeEnv success ≥ 95% over 50 seeds, stages reached in order, instruction non-empty.
  - Container: `check_expert_task.py` per primitive.

**Wave 5 (parallel: WP10, WP11, WP13)**

- **WP10: cube target tasks.** Depends on WP8. Files: `tasks/pick_place_cube` additions, `tasks/stack_two_cubes`, `tasks/pick_place_color_cube` (including the USDA box), configs.
  - Same parametrised suite, plus:
  - a USDA parse test (inner 4 cm, 5 colliders);
  - colour-binding test (success only when the target slot is in the box; distractor in box gives `wrong_object_placed`);
  - the three kinds are produced in FakeEnv;
  - existing pick-place eval and benchmark tests still pass.
  - Container: success-rate gates, then tuning by TOML only.
- **WP11: `dr_replay` v2.** Depends on WP4, WP5 and WP8. Files as in section 4.
  - Planning is deterministic.
  - `appearance_fp` is unique per parent.
  - Target colour stays unique and non-held-out.
  - Instruction is re-rendered.
  - A v1 source still plans exactly as today (golden manifest).
  - Multi-object pose dispatch works against a stub scene.
  - Container: `check_dr_replay_v2.py`.
- **WP13: `stack_bowls`.** Depends on WP10. Files: `tasks/stack_bowls`, the bowl asset, and a `nested_in` check.
  - Predicate truth table and FakeEnv suite.
  - Container: nesting stability. This is the highest-risk package and is isolated last so it cannot block the rest.

## 8. Risks and open questions

**Risks**

1. **Throughput.** About 34k source episodes at `num_envs = 1` is likely hundreds of GPU-hours, plus the same again for DR. The plan answers this with shards (several containers or GPUs), not with `num_envs > 1`, which would touch many hard-coded `env_0` paths. I have not measured steps per second; `generate-plan` should print an estimate from a smoke-run timing.
2. **Grasp reliability under Newton.** The scripted expert may need per-task tuning (approach height, close angle, dwell). All of it is in TOML, but budget owner time in the container.
3. **Bowls.** No bowl asset exists, and a concave mesh under MJWarp needs convex decomposition. I suggest a procedural USDA bowl made of convex wedges.
4. **Precise placement.** ±1 cm clearance is feasible only with closed-loop correction before release; it also depends on actuator tracking error in sim.
5. **Per-filter contact forces with N cubes under Newton** are unverified. The geometric fallback is planned.
6. **Small-file counts on the Hub.** About 68k JSON sidecars per dataset is too many, hence the consolidated parquet at finalise.
7. **LeRobot v0.6.1 APIs** for merge and streaming need a container check before WP12 is written.

**Questions for the owner**

1. Are the primitive semantics in section 2 right? Do primitives get failures and recoveries, or successes only? (I assumed successes only.)
2. May a failure and a success share the same start scenario? (Default: no.)
3. Should per-frame labels (`stage`, `imitate`, `ee_pose`) also be dataset feature columns, so they stream with the frames? This needs the small recorder change; otherwise they live only in the sidecar parquet.
4. One dataset repo per task, or one per task and kind? Hub or bucket, and which organisation and visibility?
5. Which colours and counts are held out for generalisation testing, if any?
6. Is `pick_place_color_cube` an acceptable name for the precise task? Is the 4 cm box fixed in place or randomised? What is the palette?
7. Should generated `pick_place_cube` labels use the existing contact-sensor signals (my assumption, for continuity with eval) or the new geometric checks?
8. Is `kinematics.json` beside `arm.toml` acceptable instead of inline TOML?
9. For a second arm in the same scene, is an `arms.<arm>` block in `real_setup.json` acceptable, or should mapping and limits move into the arm folder?
10. Is OU action noise on success episodes wanted, and should those be counted as successes?
11. Is the video codec the LeRobot default, or a faster encoder given the volume?

### Critical Files for Implementation
- /home/user/Sim-To-Real/src/core/tasks/base.py
- /home/user/Sim-To-Real/src/sim/generate/dr_replay.py
- /home/user/Sim-To-Real/src/sim/recording/replay_source.py
- /home/user/Sim-To-Real/arms/so101/arm.toml
- /home/user/Sim-To-Real/tasks/pick_place_cube/env_cfg.py
