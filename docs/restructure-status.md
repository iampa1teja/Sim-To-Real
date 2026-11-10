# v2 restructure status

The complete user-supplied restructure plan (sections 1–7) governs this work. This file records progress and evidence;
it does not replace that plan or reduce its scope. Work stays on branch `v2`.

## Phase gates

Every phase must pass the complete CPU suite without `--ignore`. Simulator changes must also be exercised inside
`s2r-sim` on the SO-101 in the meeting room before the next phase begins.

| Phase | Required acceptance | Status |
| --- | --- | --- |
| 0 Cleanup | Section 2 cleanup; CPU tests; record/eval/serve/train dry runs unchanged | Complete |
| 1 Arm | Typed arm contract and per-unit calibration; SO-101 utilities; synthetic four-joint arm; sim-only recorded episode, smoke eval, identical replay joint targets | Complete |
| 2 Scene | Declarative builder, appearance/mount split, blank table; reference-image comparison; SO-101 recording without a mount file | Complete |
| 3 Task | Generic stages/starts, task metadata/schema 2, stack task; same scheduled stages; legacy dataset reads; stack record/eval-set/eval without src edits | Complete |
| 4 CLI | Split agent/eval, module commands, new/check/migrate, environment renames, policy configure; all planner tests, all plugin checks, both policies served and evaluated | Complete |
| 5 Data generation | IK scripted expert, shared success-filtered recorder and calibration replay, then Mimic; successful generated LeRobot episodes and policy preparation/training | In progress — scripted and Mimic episodes pass; training pending |
| 6 Second arm | OMX factory calibration/utilities/mount; sim record/eval on blank table, then paired hardware recording; no arm-specific src edits | Pending |
| 7 Docs/conformance | Eight guides, templates and conformance; each plugin type addable from the docs alone | Pending |

The stage-logic deduplication in section 2.3 is explicitly assigned to Phase 3 by the plan. The large agent/eval
splits listed in section 2.4 were completed in Phase 4.

## Phase 0 evidence and remaining work

- Restored `sim.app` and consolidated prelaunch helpers; restored full test collection without ignoring a test.
- Removed the two code-explorer tests as requested. Their target tool was untracked, locally ignored, and outside
  the repository. No local explorer files were removed.
- Shared policy client and seeding tests exercise actual msgpack serialization and RNG repeatability. The live
  image lacked msgpack; the dependency is now explicitly installed by `docker/sim/Dockerfile` too.
- Local `.env` remains available for hardware configuration but is removed from version control; use `.env.example`.
- Geometry/material caches have CPU tests for numerical results, per-environment/stage isolation and repeat updates.
- Pre-cleanup CLI output was captured at `/tmp/s2r-v2-phase0-dry-run-baseline.json`; the four representative dry runs
  must still match after the entire cleanup.
- `outputs/v2/phase0/teleop-final.json` records a successful 30-step full runtime check: both cameras rendered,
  both observed joint errors were zero, and images changed with motion.
- The original checker spawned a 20 mm cube into the closed moving jaw, producing up to 12.27 mm penetration and
  NaNs. The corrected fixture opens the jaw through normal simulation steps before spawning. Camera/observation
  assertions remain intact. Production spawn behavior and solver settings are unchanged; robust placement for a
  closed gripper remains an issue to address when generalizing placement.
- The no-camera CAD variant is now a small override of the shared USD. The original comparison found exactly
  11 authored differences and no resolved differences across 4,016 paths; joint-frame differences are preserved,
  not discarded as if the variant only hid a camera.
- Repeated scalar NumPy cross products in collision preparation exposed an intermittent native crash during the
  expanded USD tests. The two loops now batch cross products, retaining the same dot/sign/orientation logic.
  `outputs/v2/phase0/collider-vectorization-comparison.json` confirms identical data for all 59 enabled colliders
  in both arm variants, including 1,426 attribute hashes. The final full suite and simulator check both passed.
- `outputs/v2/phase0/recording-baseline/profile.json` reports a successfully saved/decoded sim-only episode:
  45 parquet frames and 45 frames in each of two camera videos. Its fixture also opens the jaw before spawning.
- Final CPU gate: **83 passed, 22 subtests passed, zero skips/ignores**; see
  `outputs/v2/phase0/cpu-tests-final.log`. The final 30-step simulator check passed with zero joint-observation error.
- The four CLI dry-run outputs still match the saved baseline exactly. Phase 0 is complete.
- Before Phase 1, a 403-frame recorded-episode target/state baseline was captured under
  `outputs/v2/phase1-baseline/`, including both embedded `rpd_robot` and recorded `rpd_follower` calibrations.

## Phase 1 evidence

- Arm manifests now declare joints in action order, including roles and revolute/prismatic units. The core mapping
  and calibration tools use that contract; synthetic four-joint arms include interleaved parallel grippers and slides.
- Simulator actions and joint observations preserve that manifest order explicitly; Isaac Lab otherwise defaults
  to articulation order. The runtime checker also supports a permuted manifest order to exercise that boundary.
- Measured `rpd_robot` and `rpd_follower` mappings live under `arms/so101/calibration/units/`. Recording configs
  select `follower.unit`; evaluation and rollout accept `--unit`. Scene data remains in `real_setup.json` until Phase 2.
- SO-101 collision splitting, collision pads, fitted robot appearance and wrist mount visuals moved to
  `arms/so101/so101_utils.py`. Optional arm hooks keep the simulator independent of those mesh names.
- `outputs/v2/phase1/joint-target-comparison.json` verifies all six saved arrays bitwise across 403 frames,
  under both measured mappings, including float32 actions/states and float64 action conversion.
- `outputs/v2/phase1/teleop-runtime-final.json` passed 30 steps with zero observation error. Camera attributes,
  robot state samples, spawn placement and integrator match Phase 0 exactly.
- `outputs/v2/phase1/recording-final/profile.json` reports a saved 45-frame episode, both decoded 45-frame RGB videos,
  and passing fidelity. The new version-4 recording contract records arm, unit, joint names/roles/units and mapping.
  This short profiled run averaged 28.18 FPS with seven missed deadlines; sustained 30 Hz headroom is not established.
- Final CPU gate: **190 passed, 22 subtests passed, zero skips/ignores**; see
  `outputs/v2/phase1/cpu-tests-final.log`. This includes SDK differential codec checks, mixed units, device cleanup,
  and manifest-order telemetry. The 35 warnings come from dependencies, including deprecated Torch JIT APIs.
- The first live smoke evaluation exposed a pre-existing reset selector assumption: Isaac Lab supplies a slice for
  all environments. Task reset/color/light events now use the shared environment-index normalizer.
- `outputs/v2/phase1/smoke-result.json` records all nine one-per-zone attempts completed through the example
  hold-state policy server. Each timed out as expected; this verifies execution and reporting, not learned-policy
  task success. The separately hashed `smoke-eval-set.json` uses captured current geometry and validated starts.
- The frozen benchmark geometry predates the existing November 6 mount correction. Live base translation differs
  by approximately `[-1.2, -5.2, +8.2]` mm while the box pose is identical. The original frozen sets and their
  geometry guard remain unchanged. They cannot be evaluated against the corrected mount without an explicit
  geometry decision; schema migration alone will not repair that difference.
- `outputs/v2/phase1/eval-report-baseline.json` is a complete nine-episode report after the final runtime fixes,
  retained for the task-engine schedule comparison. `permuted-joints.json` passes 30 simulator steps with the
  gripper interleaved in action order and zero observation error. `runtime-baseline-comparison.json` confirms the
  normal-order runtime still matches every checked Phase 0 state and camera field exactly.
- The simulator emits transient RTX CUDA memcpy warnings at its first step, also present in the Phase 0 runtime
  log. The runtime and recording checks still produce valid camera images and finite state; these warnings remain
  an observed renderer limitation rather than evidence of a new arm-contract regression.
- Git metadata lookup inside this existing container cannot follow the host worktree's absolute `.git` pointer.
  The CLI provenance path must be handled during Phase 4; the local verification fixture records its source hash.

`outputs/` contains local verification artifacts and is ignored. Passing a helper or smoke test alone is not evidence
that later-phase record, training, policy evaluation, image matching or real-hardware gates have passed.

## Scene reference captured before Phase 2

`outputs/v2/phase2-baseline/frozen-before-a/` and `frozen-before-b/` contain three raw room and wrist frames each.
Both launches use the same exact frozen robot/object poses, optics and renderer settings. Repeated renders have
mean absolute differences of 0.156–0.172 uint8 levels and p99 of 1. The pixel tolerances were recorded before any
scene edits in `acceptance.json`; `frozen-repeat-noise.json` records the measured baseline. Actual fitted response
parameters are also captured, with a deterministic CPU response replay for checking policy-facing images.

## Phase 2 integration

- Scene and mount specifications now drive the generic builder; fitted meeting-room data is split into
  `scene.toml`, `appearance.json`, and `mounts/so101.toml`. The old setup JSON and scene-specific source builder
  are removed. The meeting-room hook owns its material/photo treatment.
- Core tests preserve the measured data hashes and exact float32 mount/box poses. Contact checks select the
  same 49 meeting-room colliders and the blank table's single solid collider. Both scenes construct every
  existing task variant through the actual Isaac Lab config classes.
- Final CPU suite: **234 passed, 22 subtests passed**, with 89 dependency deprecation warnings and no skips.
  `outputs/v2/phase2/cpu-tests-final.log` records this run after the runtime fixes.
- The first post-change simulator reference capture completed. Exact physics state, authored camera optics,
  renderer settings and room-camera pixels pass the predeclared comparison. Wrist pixels exceed the measured
  noise bound slightly. A fresh isolated copy of unchanged Phase 1 reproduced the same residual, establishing
  that it was not introduced by this restructure.
- `outputs/v2/phase2/blank-record-3/validation.json` passes the production record entry point with
  `S2R_SCENE=blank_table` and no mount file: 45 parquet frames, 45 decoded frames in each of the `room` and
  `wrist_cam` videos, finite joint values, and matching object sidecar poses. Only leader input and UI requests
  were deterministic test fixtures; no physical device was opened.
- `reference-control-acceptance-a.json` passes all 110 checks against that fresh Phase 1 control, using the
  original thresholds for raw and response-processed RGB, exact frozen physics, optics and renderer settings.
  `runtime-stage-diff.json` finds zero composed differences across all 512 runtime prims, including task objects.
  Separate authoring comparisons also preserve 5,337 room and 4,898 arm prim/property entries.
- Renderer repeatability is limited: another repeat had one processed room-channel error of 21 against a
  maximum bound of 20, and capture with a USD snapshot read changed the wrist residual. These diagnostic failures
  are retained in `reference-control-acceptance-b.json` and `reference-control-acceptance-stage.json`; no limits
  were relaxed and no production camera parameters were changed to fit them. The direct controlled comparison
  passes, while exact pixel repeatability across launches/timing remains unestablished.

## Phase 3 integration

- Task manifests declare stages, tracked objects, placement frames and start distributions. The shared stage
  engine calls task-owned signals/observation rules, latches stages and derives success from the last stage.
  Generic reset events preserve robot/object/tracking order. Benchmark geometry, starts, reports, statistics
  and comparison moved to `core.benchmark`.
- New `core.datasets` readers/writers handle schema-2 named-object sidecars in `task_meta/`, retain schema-1
  `pick_place_meta/` compatibility, and keep mapping provenance with recorded sources. Recording, replay and
  rerecording preserve every declared tracked object on the dataset timeline.
- CPU gate: **267 passed, 22 subtests passed**, with 105 dependency deprecation warnings and no skips/ignores;
  see `outputs/v2/phase3/cpu-tests-final.log`.
- `schedule-comparison.json` passes all nine episodes of the saved Phase 1 schedule, including start ids, splits,
  repeats, policy seeds, step counts, stages, success fields and failure modes. The example hold-state policy
  produces the same `never_reached` outcomes. This proves schedule/stage preservation, not learned-policy success.
- The three frozen pick-place eval sets were converted to the approved schema-2 copies exactly.
  `approved-eval-set-conversions.json` records their new content hashes, preserved `legacy_eval_set_id` values
  and matching approved-copy SHA-256 hashes. The existing mount mismatch remains: converting schema does not
  update frozen geometry or bypass its guard.
- `pick-record/validation.json` passes a production meeting-room recording: 45 finite joint/action frames,
  45 schema-2 `blue_cube` poses and 45 decoded frames in each of four primary/auxiliary camera videos.
- `legacy-dataset-validation.json` reads the untouched Phase 2 schema-1 recording through production
  `core.datasets.sources.load_source` and `RecordedStarts`. All 45 actions, states and object poses match the
  saved data; first-frame starts and the saved mapping match. All 11 source-file hashes remain unchanged.
- `tasks/stack_two_cubes/` supplies the second task through its folder. `stack-record-2/validation.json` passes
  45 frames for both cubes, all joint/action values and all four videos in the meeting room. The first diagnostic
  run exposed a target pose beyond the table edge; the task's declared/reset target was corrected inside its folder.
- The actual `s2r eval-set` command generated five stack starts (three ID, one yaw, one recorded train start) from
  that recording and captured live geometry; see `stack-eval-set.log` and `stack-eval-set.json`.
- The actual `s2r eval` stack check completed all five generated starts, preserving both named objects and all
  six declared stages in every episode report. Each 30-step attempt ended `never_reached` through the example
  hold-state policy. `stack-acceptance.json` and `stack-source-comparison.json` confirm this extension workflow
  while all 108 `src/` files stayed unchanged from the pre-stack check. This verifies recording/benchmark/eval
  integration, not learned stacking success. Phase 3 is complete.

## Phase 4 integration

- CLI plans now invoke `python -m` entrypoints. Only `s2r` remains an installed console command; recording,
  evaluation, rollout, calibration and data-generation modules use their new package boundaries. Shared app
  launch and YAML/argparse helpers preserve selector-before-import ordering and CLI-over-YAML precedence.
- `s2r new arm|scene|task|policy <name>` creates a folder from its template. All four generated scaffolds pass
  typed manifest and entrypoint checks. This is static validation, not proof that a template is a calibrated
  production robot or a trained model.
- CLI planner/scaffold tests: **21 passed**; see `outputs/v2/phase4-cli-tests.log`. Dry runs print module commands
  and pass the host revision through `S2R_GIT_COMMIT`, avoiding the container's unresolved host worktree pointer.
- Final full CPU suite: **320 passed, 22 subtests passed**, with 105 dependency deprecation warnings and no
  skips/ignores; see `outputs/v2/phase4/cpu-tests-yaml-final.log`.
- Actual host `./s2r check --static` passed all eight available extensions in an isolated temporary uv environment.
  Only `numpy==2.5.1` was installed. `phase4-host-static-check.log` records the command/output;
  `phase4-host-static-import-boundary.log` confirms no Isaac Lab, Isaac Sim, USD, Omni or Torch modules were
  imported and lists the environment's sole installed package. No global Python packages were changed.
- Static checks validate every extension; fixed policy action dimensions are checked against the selected arm
  only. Without `--static`, the default plan adds three deduplicated runtime compositions covering the current
  arm, both scenes and both tasks. Explicit selectors request one composition.
- All three actual CLI runtime checks pass, with rendered RGB and three finite simulation steps for each:
  SO-101/meeting-room/pick-place, SO-101/blank-table/pick-place, and SO-101/meeting-room/stacking.
  `outputs/v2/phase4/composition-checks.json` records their joints, bodies, cameras and objects.
- Recording configs moved to `configs/record/`; rollout and data-generation examples use common selectors.
  Sync includes arms/scenes, and training/calibration environment variables use the `S2R_` prefix without
  legacy-variable fallbacks. Shell syntax, Compose configuration and whitespace checks pass.
- Policy clients now receive `configure(ArmInfo)` with joint order, roles, types and mapping. Generic clients
  derive the action dimension; the fixed six-axis model manifests declare it explicitly.
- `outputs/v2/phase4/pick-record/validation.json` passes recording through the renamed production
  `sim.teleop.agent.main`: 45 finite joint/action frames, 45 matching object poses, and 45 decoded frames in
  each of four primary/auxiliary RGB videos. Only leader input and control requests are test fixtures.
- `sim.eval.run` completed the example hold-state policy's nine-episode saved schedule; its report is
  `outputs/v2/phase4/example-eval.json`. `eval-entrypoint-comparison.json` confirms identical start IDs,
  seeds, steps, stage flags, success fields and failure modes. This does not establish learned-policy success.
- The first evaluation startup exited with a native MuJoCo crash (status 139), retained in
  `outputs/v2/phase4/example-eval-attempt1.log`. An identical retry completed the schedule. No production
  physics settings were changed between attempts; native startup repeatability remains an observed limitation.
- The actual host `s2r migrate` command migrated a copied schema-1, 45-frame episode to schema 2 while retaining
  its legacy sidecar. `outputs/v2/phase4/migrate-cli.log` records the direct `core.datasets.task_meta` invocation;
  the disposable copy is under `migrate-fixture/`. `migration-validation.json` confirms every saved pose and
  all original files are preserved in the migrated copy.
- The isolated image `s2r-serve:v2-phase4` built successfully as
  `sha256:4dad52d630649aa183d0b7e332cbcbc5002a8c78ae0f66a1cb3a188c6b0474dc`;
  see `outputs/v2/phase4/serve-build.log`. The existing `latest` tag was unchanged.
- Both live model gates pass: GR00T made three inference calls and Molmo made one during their respective
  completed 30-step episodes. Each logged 20 policy action ticks with all 12 raw/sent joint columns finite.
  `live-policy-gates.json`, the individual `*-live-validation.json` files, reports, server descriptions, action
  CSVs and logs are retained under `outputs/v2/phase4/`. These are integration checks, not success benchmarks.
- The first live container launch exposed a blank line before the shell entrypoint's shebang; it was removed
  and both isolated serving/training images rebuilt. Their entrypoints now execute successfully. The live
  evaluations emitted the same RTX memcpy warning class present in Phase 3; no model-server errors occurred.
- Both owned policy servers were stopped after verification. Phase 4 is complete.

- Evaluation YAML `run.gui` is accepted with the Lab 3 launcher, which no longer registers a headless parser flag.
  The full suite includes this regression and CLI-over-YAML precedence. `yaml-example-acceptance.json` records
  one complete simulator episode with actual example-policy inference through `s2r eval --config_path`.
  Prepared GR00T/Molmo acceptance YAMLs both parse through the production parser; their live runs also passed as recorded above.

## Phase 5 integration (not yet accepted)

- Updated user requirement: no deliberate wrist roll after pickup. Earlier rolled demonstrations are retained
  as diagnostic/format evidence only and must not be used as the final training dataset. The task expert now
  keeps one requested pickup orientation throughout transport and refuses the removed roll overrides.
  `scripted-noroll-fast1.json` and its independent validation now pass: 391 frames, held through transport,
  deliberate gripper opening over the box, and final settled task success. The corresponding two-camera
  preview is `outputs/v2/phase5/no-roll-episode.mp4`.
- Latest full CPU suite: **394 passed, 22 subtests passed**, with 118 dependency deprecation warnings and no skips or
  ignores; see `outputs/v2/phase5/cpu-tests-mimic-precise.log`. All eight host static plugin checks pass in `static-check.log`.
  SDK Mimic tests exercise actual source segmentation, retargeting, async action/reset queues and bounded
  cancellation. Its config wrapper preserves measured mounts and task fields through copying/export.
- Arm manifests can declare a TCP offset and the orientation components their IK controls. The native Jacobian
  solver preserves gripper roles and bounds revolute/prismatic targets in their own units. USD forward kinematics
  supplies the actual executed target pose for Mimic annotations without changing simulator state.
- `outputs/v2/phase5/ik-reachability.json` records CPU kinematic solutions for all 36 expert waypoints across a
  recorded scene and three sampled starts. This does not establish collision clearance, grasping or task success.
  The live native-Jacobian/FK comparison now passes across five physical states: maximum TCP position error
  0.286 micrometres, linear-Jacobian error 2.94e-7 and angular-Jacobian error 1.85e-6 (`live-kinematics.json`).
- Scripted generation and calibration replay share one physical rollout and success-filtered LeRobot recorder.
  Failed attempts and episodes that lose success before their final frame are excluded. Recorded observations,
  actions, cameras and named-object poses describe the same pre-action frame; success observes the resulting
  physical state. Calibration replay retains its source-hash and saved-mapping checks.
- The real `s2r prepare --policy groot_n17` path passed on the existing schema-2 Phase 4 recording fixture:
  45 frames, two videos, all 90 decoded video frames, real GR00T loader validation and unchanged source hashes.
  `outputs/v2/phase5/prepare-fixture-validation.json` independently verifies the object sidecar and all 11 source
  files. This is a format-compatibility check, not the required generated-data training gate.
  The actual `s2r verify` command also passed prepared checksums and first/middle/last images from both cameras
  against independent PyAV decoding, with zero RGB difference (`verify-fixture.log`).
- The user authorized the Phase 5 GPU session. Physical trials exposed steady load-induced servo error; bounded
  integral compensation now reaches the pickup without changing production physics. The task expert routes
  transport around the arm base and targets the container centre. Two-second arc segments allowed the cube
  to slip; shortening them to half a second retains the original pickup orientation and reaches the box while
  still holding the cube. The successful 391-frame no-roll demonstration opens deliberately at step 363 and
  reaches settled success at step 391. Every saved video frame, feature/timestamp, object sidecar, device codec
  and Mimic annotation passes the independent validator (`scripted-noroll-fast1-validation.json`).
- Earlier rolled prototypes remain in ignored output folders but are superseded by the user's no-roll constraint.
  A seven-attempt batch using that older trajectory had one success; the remaining failures involved unreachable
  fixed-orientation starts or grasp loss. It was stopped deliberately and is not a completed acceptance batch.
  General success rates across the full start distribution are not established.
- Real GR00T preparation and verification pass on the no-roll source: 391 frames, all 782 camera frames decoded,
  unchanged parquet/video content and all 11 source hashes, preserved schema-2 object sidecars, and zero RGB
  difference at sampled frames. Evidence: `prepare-noroll-fast1-validation.json`, `prepare-noroll-fast1.log` and
  `verify-noroll-fast1.log`. `noroll-fast1-orientation-proof.json` confirms all 28 waypoint goal quaternions are
  identical; after grasp, commanded wrist-roll span is 0.943 degrees and measured span is 0.908 degrees.
- The requested stacking demonstration also passes: `stack/stack2.json` records all six stacking stages and
  one accepted 304-frame episode at seed 182482. Both camera videos and both named-object trajectories pass
  independent validation. The expert lives in `tasks/stack_two_cubes/expert.py`; all 123 generic Python files
  remain unchanged from the pre-extension baseline. `configs/datagen/stack_two_cubes.yaml` reproduces the
  setup. A first attempt at seed 1184 pinched the cube diagonally and failed; it was not saved. The successful
  arrangement aligns the cube with the fingers before pickup and keeps one requested orientation throughout.
  The preview is `outputs/v2/phase5/stack/stack-two-cubes-episode.mp4`; additional final-contact geometry and
  25-frame stability checks are recorded in `stack/stack2-geometry-and-extension-proof.json`.
- The live Mimic path exposed two integration issues: config dispatch must preserve Python module identity, and
  Kit rendering must occur outside the private asyncio coroutine. Both now have CPU regression coverage;
  the latter follows the installed SDK's synchronous environment loop around asynchronous generator queues.
  Source annotations encode executed joint commands, so Mimic now recovers them with a bounded local kinematic
  inverse instead of reapplying feedback compensation. Contact-sensitive grasps require tighter inversion:
  32 iterations with 1e-8 m / 1e-7 rad tolerances recover all 391 float32 source targets within 0.000028 degrees
  in the CPU check (`mimic-tight-inverse-fidelity.json`). The adapter's 14 tests pass after this precision change.
- The actual SDK Mimic path now produces a successful 402-frame episode at source seed 101, including its
  subtask interpolation frames. `mimic-precise1-validation.json` independently passes final physical success,
  all six stages, both decoded camera streams, object poses, action codec and annotation alignment. Dataset:
  `datasets/v2_phase5_mimic_precise1`; preview: `outputs/v2/phase5/mimic-episode.mp4`.
  Changed-start retargeting and a generated-data optimizer update remain unverified; Phase 5 is not complete.
