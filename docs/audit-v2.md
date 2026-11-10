# v2 audit: what to delete, simplify and fix

Audit of branch `v2` at `83dea9a`, compared against `v1` to find what was carried over without its launcher.
This file is a work list only. Nothing in it has been applied to this branch.

Order of preference used throughout: delete, then simplify, then optimize.

## 0. How this was checked

| Check | Result on untouched `v2` |
| --- | --- |
| Container suite (run by the owner) | 426 tests + 22 subtests pass, 8 static checks pass, 3 simulator checks pass |
| Host suite, only the documented `host` dependency group | **Fails at collection**: 3 errors (see 4.1) |
| Host suite with `av`, `warp-lang`, `usd-core`, `gymnasium`, `mujoco-warp`, `lerobot[dataset]` v0.6.1, `rerun-sdk` added | 377 pass, 12 skip, 25 fail; all 25 need `isaaclab` |
| `./s2r check --static` on host | 8 of 8 pass |
| Dead-code scan (vulture), unused-name lint (flake8 `F`), import-resolution scan, duplicate-body scan | Findings below |

Method limits:

- Isaac Lab is not installed where this audit ran. Anything under `src/sim/`, `tasks/*/mdp.py`,
  `tasks/*/env_cfg.py` and `tools/diagnostics/` was checked statically only.
- Policy training and serving code was not executed.
- Sections 1 to 3 and items 4.2, 4.3, 5.1 and 5.2 were applied on a scratch copy before this file was written.
  The host suite gave the same result as the baseline (377 pass, same 25 `isaaclab` failures, one skipped test
  file removed) and `s2r check --static` still passed 8 of 8. They have **not** been run in the container.

After sections 1 to 3 the container suite should report **424** tests: `tests/sim/test_grasp_diagnostics.py`
(2 tests) is deleted with the module it tests.

---

## 1. Delete entirely

### 1.1 v1 recipes whose launcher no longer exists

`v1` selected these through `docker/train.sh` and `docker/train.py`. Neither file exists in `v2`, and
`s2r train` can only run the one script named in `policy.toml`, so nothing can start them.

| Path | Lines | Why it cannot run |
| --- | --- | --- |
| `policies/groot_n17/train/train_dit.sh` | 22 | `exec bash "$REPO/docker/train.sh" dit`; file missing |
| `policies/groot_n17/train/train_dit_only.py` | 640 | Only caller is `train_dit.sh`; its v1 test (`tests/test_train_dit_only.py`) is already gone |
| `policies/groot_n17/train/train_pick_place_sim_100k.sh` | 13 | `exec bash "$REPO/docker/train.sh" sim`; file missing |
| `policies/groot_n17/train/train_pick_place_failures_ft.sh` | 77 | `exec bash "$REPO/docker/train.sh" failures`; defaults point at `/external_storage/...` and `datasets/hyperparameters/...` |
| `policies/groot_n17/train/launch_gr00t_server_n17.py` | 54 | No references; replaced by `policies/groot_n17/server.py` |
| `policies/groot_n17/train/check_training_runtime.py` | 28 | No references; asserts a checkout at `/Isaac-GR00T`, the v1 train image layout. The v2 image uses `/opt/cache/checkouts/` |
| `policies/groot_n17/train/download_pick_place_n17.sh` | 16 | No references; `docs/05-training.md` documents the download with explicit `hf download` commands |

Edits these deletions require:

- `policies/groot_n17/policy.toml`: remove the two comment lines starting "Other recipes in train/ ...
  selected by the training CLI".
- `policies/groot_n17/train/requirements.txt`: remove `bitsandbytes==0.50.2` (imported only by
  `train_dit_only.py` and `check_training_runtime.py`).

### 1.2 Grasp diagnostics that nothing installs

`install_grasp_diagnostics` was called by the v1 smoke evaluation. `v2` never calls it, so the whole chain is
dead.

| Path | What to remove |
| --- | --- |
| `tools/diagnostics/grasp_diagnostics.py` | Whole file (128 lines) |
| `tests/sim/test_grasp_diagnostics.py` | Whole file (33 lines, 2 tests) |
| `policies/molmoact2/eval/audit_molmo_target_reach.py` | Whole file (291 lines). No references; it hashes the file above and uses the helpers below. Its sibling `audit_molmo_baseline_videos.py` was already dropped from v1 |
| `arms/so101/so101_utils.py` | `_rotation_matrix`, `measure_grasp`, `grasp_tip_offsets` (the last 62 lines). Only the two files above call them |
| `src/sim/eval/report.py` | The five-line loop that adds `grasp_diagnostics`, `closest_tool_error_mm` and `closest_closing_tool_error_mm` to each smoke episode |

The last row changes console output: those three keys are always `{}`, `null`, `null` today and would
disappear from the `[SMOKE_RESULT]` line. Anything parsing that line with strict key access must be checked.

### 1.3 Files with no content or no purpose

| Path | Reason |
| --- | --- |
| `docs/restructure-plan.md` | The plan has been executed; the guides and `restructure-status.md` supersede it. Present on this branch only, not on `v2` |
| `tests/_fakes.py` | 0 bytes, no references |
| `configs/train/.gitkeep` | Placeholder for an empty folder nothing reads |
| `src/__init__.py` | 0 bytes; `src` is a path root, not a package, and nothing imports `src.*` |

---

## 2. Dead symbols to remove

Zero references in code, tests, manifests, configs or docs.

| File (line on `v2`) | Symbol |
| --- | --- |
| `src/core/arms/mapping.py:9` | `ARM_SCALE`, `GRIPPER_SCALE` |
| `src/core/arms/mapping.py:174` | `JointMapping.raw_limits` property |
| `src/core/benchmark/geometry.py:175` | `inside_convex` |
| `src/core/benchmark/report.py:8` | `select_instruction` (the logic is inlined in `src/sim/eval/report.py:16`) |
| `src/core/rollout/real_rollout.py:30-32` | `BODY_RANGE`, `GRIPPER_RANGE`, `OUTCOMES` |
| `src/core/tasks/base.py:281` | `Task.start_from_row` |
| `src/sim/datagen/dr_variants.py:22-23` | `DEFAULT_DR` and its comment (the default lives in `dr_replay.py:42`) |
| `src/core/policy_client/smoothing.py:122,135` | `stale_actions_dropped`: written, never read |
| `src/core/arms/spec.py:256` | Local `names`, assigned and never used |

Unused imports (flake8 `F401`):

| File | Import |
| --- | --- |
| `src/core/benchmark/eval_set.py:17` | `footprint`, `pose_for_xy` |
| `src/core/calibration/telemetry.py:5` | `numpy` |
| `src/sim/datagen/dr_variants.py:7` | `Path` |
| `src/sim/envs/scene_builder.py:19` | `load_arm_spec` |
| `tasks/pick_place_cube/mdp.py:16,27` | `numpy`, `_wxyz`, `_xyzw` |
| `tools/diagnostics/check_grasp_stability.py:110` | `Usd`, `UsdGeom`, `UsdPhysics` |
| `tests/` (10 files) | `Plan`, `Path`, `replace`, `anchor_quaternion`, `json`, `math`, `SimpleNamespace`, `pytest`, `UsdGeom`, `numpy`; run `flake8 --select=F tests` for the list |

Not dead, despite what a scanner reports. Leave these alone:

- `Thresholds.limit_tolerance_deg` / `limit_tolerance_m` in `core/calibration/gap.py`: read through
  `_gates(..., "limit_tolerance")`.
- `meta.latest_episode` / `writer._latest_episode` in `sim/recording/recording_dataset.py`: resets LeRobot
  internals.
- `apply_eval_dr` in `tasks/pick_place_cube/env_cfg.py`: referenced as a string in `task.toml`.
- `TRAIN_COMMANDS` import in `cli/plan.py`: re-exported to `cli/main.py`.
- Everything flagged in `policies/*/train/`, `sim/datagen/mimic.py`, `sim/scene/renderer.py`,
  `sim/scene/viewport.py`, `web_gui.py` and `spawner.py`: framework callbacks and lazy module attributes.

---

## 3. Simplify

| Where | Change |
| --- | --- |
| `policies/groot_n17/train/train_pick_place.sh:6,13-15` | Remove `export S2R_TRAIN_IN_CONTAINER=1` and the `if [[ ${S2R_TRAIN_IN_CONTAINER:-0} != 1 ]]; then exec bash "$REPO/docker/train.sh" train "$@"; fi` block. The relaunch target is missing, so a host run dies with "No such file". Without the block the script runs directly, which is what `setup_gr00t_n17.sh` and the test already assume. Drop `S2R_TRAIN_IN_CONTAINER='1'` from `tests/policies/test_groot_preparation.py:54` |
| `src/cli/main.py` (end of `main`) | Remove the `if args.command == "datagen" and args.source is not None ...: "is not implemented yet"` block. All three sources exist |
| `src/cli/datagen.py:37-38` | Remove the `importlib.util.find_spec(module) is None` / "not implemented yet" guard and the `importlib.util` import, for the same reason |
| `arms/so101/so101_utils.py` | `numpy` is imported at module top and again inside 8 functions; `pxr` names likewise in 4 functions. Keep the top-level imports (add `Vt`), delete the local ones. Leave the lazy `scipy` imports |
| Default selectors | `"so101"`, `"rpd_meeting_room"`, `"pick_place_cube"` are repeated as fallbacks in 11 files: `cli/main.py:171`, `core/arms/spec.py:27`, `core/tasks/base.py:14`, `core/rollout/real_rollout.py:184-186`, `sim/check.py:78-80`, `sim/datagen/calibration_replay.py:28`, `sim/datagen/dr_replay.py:53-56`, `sim/datagen/scripted.py:31`, `sim/envs/__init__.py:60`, `sim/eval/run.py:140-142`, `sim/teleop/config.py:85-87`. Define them once (for example in `core/paths.py`) and import |
| `src/cli/main.py:102` | `sys.path.insert(0, repo / "src")` in `load_policies` is redundant: `cli.plan` already imports `core` at module load |

---

## 4. Fix

### 4.1 The documented host test environment does not work

`CLAUDE.md` and `docs/05-training.md` say `uv sync --group host` gives the complete host test environment.
With exactly that group, `pytest tests` stops at collection:

```
ERROR tests/sim/test_eval_loop.py        (imports av through sim.eval)
ERROR tests/sim/test_solver_numerics.py  (import warp)
ERROR tests/sim/test_video.py            (import av)
```

Two parts to the fix:

1. Add `av` and `warp-lang` to `[dependency-groups] host` in `pyproject.toml`, or guard those three modules
   with `pytest.importorskip`.
2. Decide what the host suite is meant to be. Seven test files hard-fail on a host instead of skipping,
   because they import `isaaclab`, `isaaclab_tasks`, `gymnasium`, `mujoco_warp`, `pxr` or `lerobot` inside test
   bodies: `tests/cli/test_extensions.py` (1 test), `tests/core/test_ik.py` (1), `tests/sim/test_check.py` (2),
   `tests/sim/test_datagen_rollout.py` (3), `tests/sim/test_datagen_task_signals.py` (2),
   `tests/sim/test_mimic.py` (13), `tests/sim/test_mimic_config.py` (3). Either make them skip cleanly
   without the SDK, or state in the docs that the full suite only runs inside `s2r-sim`.

### 4.2 Stale comments that describe v1

| File | Text to remove or correct |
| --- | --- |
| `policies/groot_n17/train/requirements.txt` | "Pins copied from docker/train/Dockerfile and docker/train/download_wheels.py" and "UNVALIDATED here: GR00T still trains through docker/train.py until one `s2r train --generic` run succeeds". Neither file exists, and the status record says the `s2r train` optimizer check passed |
| `policies/groot_n17/train/prepare.sh:2`, `verify.sh:2` | Mention `--generic`; the CLI has no such flag |
| `src/core/arms/calibration.py:334` | "Environment names are migrated with all commands in Phase 4" (already migrated) |
| `src/cli/plan.py:1` | "(v1 cli/so101_cli/plan.py, renamed)" |
| Many module headers under `src/sim/` and `src/core/` | "ported from v1 ...", "Changes from v1: ..." blocks (`sim/eval/run.py:15-27`, `sim/datagen/dr_replay.py:1-16`, `sim/recording/replay_scene.py:1-11`, `sim/recording/rerecording.py:1-7`, `sim/datagen/spawner.py:1-7`, `core/calibration/report.py:3`, others). They explain history to readers who never saw v1. Keep any sentence that states a current constraint; drop the rest |

### 4.3 Old environment variable names left in the evaluator

`src/sim/eval/run.py:277`: `--checkpoint` defaults to `$PICK_PLACE_EVAL_CHECKPOINT`, then `$MODEL`. Every other
variable was renamed to `S2R_*`, these are task-named and undocumented, and `cli/plan.py` does not forward them
into the container (`SIM_RUNTIME_ENV`), so they only work when set inside it. Rename to one `S2R_*` variable and
forward it, or drop the environment default.

### 4.4 Undocumented step in the MolmoAct2 workflow

`policies/molmoact2/train/export_molmo_lora.py` is the only way to turn a trained LoRA checkpoint into what the
server loads (the `molmoact2_so101_workshop_lora` checkpoint in `configs/eval/pick_place_molmo_so101_lora_as_real.yaml`).
v1 chained it from `run_molmo_lora_pipeline.sh`, which is gone. Nothing in `docs/05-training.md` mentions it.
Document the export step. Then decide whether `check_molmo_export_server.py` and
`check_molmo_optimizer_offload.py` (both unreferenced, GPU-only manual checks) are part of that documented
workflow or should be deleted.

---

## 5. Documentation

| # | Where | Change |
| --- | --- | --- |
| 5.1 | `README.md:3-20` | The introduction narrates the restructure phase by phase ("Phase 4 is complete", "deferred at the user's request"). Replace with a description of the product and a short list of what is verified and what is not. Carry over these facts unchanged: scripted and Mimic generation show physical success, not a success rate; GR00T and MolmoAct2 serving shows integration, not learned-policy success; training recipes and the DR replay recipe assume SO-101 and pick-place; no second arm is validated on hardware |
| 5.2 | `policies/README.md:38,45-51` | Remove "The Phase 4 serving image `s2r-serve:v2-phase4` is built ..." and "Final documentation/conformance is pending" (stale: the guides are done). Keep the scope statements |
| 5.3 | `docs/05-training.md:134,233-234,248`, `docs/02-add-an-arm.md:231`, `docs/01-getting-started.md:98,108` | Same: phase references in user guides |
| 5.4 | `CLAUDE.md` | The last three paragraphs are a status diary ("The user confirmed ..."). Reduce them to the durable rules they contain: the `sim.solver_numerics` install order and "do not edit installed SDK files"; "do not claim arbitrary-model training or learned task performance". Also correct the host-environment claim (4.1) |
| 5.5 | `docs/restructure-status.md` | Decide: keep as the engineering record, or move the "Remaining boundaries" section into a short limitations page and leave the rest to Git history. It is linked from `README.md`, `CLAUDE.md`, `docs/01`, `docs/05` and `policies/README.md` |

---

## 6. Challenged and kept

Looked at as deletion candidates and left in place, with the reason.

| Item | Reason to keep |
| --- | --- |
| Readers for `pick_place_meta/`, schema-1 eval sets, legacy alignment markers, degree-only telemetry logs | They keep existing customer datasets and results loading |
| `convert_eval_set` and `tests/fixtures/legacy_eval_set_v1.json` | Test-only today and has no CLI, but it is what guards the three converted frozen sets against drift |
| `euler_xyz_degrees_to_xyzw` in both `core/scenes/anchors.py` and `sim/scene/cameras.py` | Deliberately separate: a test freezes the camera path to the `Gf` composition. Compared over 20,000 random angles: same convention, no sign flips, largest difference 1.3e-14. Do not merge without re-verifying camera images |
| `JointMapping.to_sim_np`, `ArmSpec.joint_name_pairs`, `anchor_position`, `camera_quaternion`, `world_to_frame` | Only tests and docs use them, but they are documented or exported plugin API |
| `policies/groot_n17/train/setup_gr00t_n17.sh`, `checkpoint_milestones.py`, `gr00t_offline_cosmos.py`, `so_arm_n17_config.py`, `dataset_sidecars.py` | All reachable from the live `train_pick_place.sh` / `prepare.sh` path |
| `arms/so101/so101_utils.py::apply_primitive_pads` | Its docstring says "Candidate", but `prepare_collisions` calls it in production |
| Three frozen eval sets plus `pick_place_scene.json` | Only `pick_place_v1_seed1984.json` is referenced by a config, but the others define benchmarks that published results may cite |
| Both calibration units (`rpd_robot`, `rpd_follower`) | Both referenced by shipped configs |

---

## 7. Weak assumptions and decisions needed

No change proposed without a decision from the owner.

| # | Item | Concern |
| --- | --- | --- |
| 7.1 | `tools/diagnostics/` (remaining 10 files, about 2,200 lines) | Undocumented, untested, hard-coded to SO-101, the meeting room and pick-place, with defaults tied to one recorded episode (`--start-frame 145 --end-frame 211`, `--frames 736`). `replay_grasp_episode.py` is a contact-parameter sweep harness and may be worth keeping for the next arm; `render_grasp_replay.py`, `summarize_grasp_replays.py`, `audit_grasp_enclosure.py` and `grasp_collision_candidates.py` supported a finished investigation. `check_teleop_runtime.py` overlaps `s2r check`. Keep, document, or delete |
| 7.2 | `tasks/_template/` | Byte-for-byte copy of `tasks/stack_two_cubes/` except the name (4 files, 246 lines, including the class name `StackTwoCubesTask`). The two will drift. Either scaffold new tasks from `stack_two_cubes`, or make the template a real minimal skeleton |
| 7.3 | `policies/groot_n17/offline_cosmos.py` | Duplicates `train/gr00t_offline_cosmos.py` and part of `train/gr00t_model_profiles.py` (5 identical function bodies). Serving and training load from different folders, so merging needs a GR00T run to verify |
| 7.4 | `src/sim/solver_numerics.py` | Patches MJWarp's internal factorization at import. Correct for the pinned image; a silent no-op or a break on any SDK update. Needs a version guard or an upstream fix |
| 7.5 | `[scene.anchors.workspace] xy_min / xy_max` | Parsed, validated and documented, set in `blank_table`, and read by nothing. Either use it (for example to bound random starts) or remove it from the schema; removing breaks manifests that set it |
| 7.6 | Site-specific values in shipped files | `configs/eval/*.yaml` point at `/external_storage/...` and `/workspace/datasets/pick_place_v1_calibrated_A`; `train_pick_place.sh` defaults to `$HOME/sim2real/models/...`; the eval sets embed `/home/rpuser1/...` in `dataset_path` |
| 7.7 | Arm and task names in generic code | `"so101-spawn"` default authkey (`cli/spawn.py:16`, `sim/datagen/spawner.py:27,128`, `sim/teleop/config.py:98`); report schema `"so101_real_rollout_v1"` (`core/rollout/real_rollout.py:169`); `legacy_object` falls back to `"blue_cube"` (`core/datasets/task_meta.py:103`); `dr_variants.cube_colors` reads `object_materials.blue_cube`. Renaming the schema string or authkey changes what existing consumers see |
| 7.8 | Tests that execute source extracted by AST | `tests/sim/test_color_cache.py` and `tests/sim/test_scene_cameras.py` compile single functions out of `tasks/pick_place_cube/mdp.py` and `sim/scene/cameras.py` with a hand-built namespace. They break on any refactor that adds a name, and pass even if the module itself fails to import |
| 7.9 | `configs/eval/` | Five MolmoAct2 configs that differ in a handful of keys. Not wrong, but a base file plus overrides would be shorter |
| 7.10 | Git LFS | Still undecided: `arms/` 46 MB, `scenes/` 109 MB of binary assets in plain Git |

---

## 8. Not audited

Read only at signature level or not at all. No claim is made about these:

- Logic inside `src/sim/eval/run.py`, `loop.py`, `scene.py`; `src/sim/datagen/ik.py`, `mimic.py`, `scripted.py`,
  `rollout.py`; `src/sim/teleop/loop.py`, `agent.py`, `devices.py`; `src/sim/recording/*`.
- `src/core/calibration/*`, `src/core/arms/calibration.py`, `src/core/scenes/spec.py`,
  `src/core/benchmark/eval_set.py` generation paths, `sweep.py`, `starts.py`.
- Both policy servers and all remaining training scripts.
- `docker/`, `compose*.yaml`, `gui/`.
- Whether the tests assert the right things; only whether they pass.

---

## 9. Suggested order and verification

1. Sections 1, 2 and 3 in one change. They only remove unreachable code.
2. 4.1 (host environment), since it decides how every later change is checked.
3. 4.2, 4.3, 4.4 and section 5.
4. Section 7, one decision at a time.

After each step:

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python -m pytest tests -q -rs 2>&1 | tail -40
```

```bash
./s2r check 2>&1 | tail -30
```

Expected after step 1: 424 tests pass, 8 static checks pass, 3 simulator checks pass. Also run one
`./s2r --dry-run train --policy groot_n17 --dataset <dir> --output <dir>` and one smoke evaluation, since
`train_pick_place.sh` and the smoke report line are the two behaviour changes in that step.
