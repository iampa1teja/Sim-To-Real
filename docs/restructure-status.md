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
| 2 Scene | Declarative builder, appearance/mount split, blank table; reference-image comparison; SO-101 recording without a mount file | Pending |
| 3 Task | Generic stages/starts, task metadata/schema 2, stack task; same scheduled stages; legacy dataset reads; stack record/eval-set/eval without src edits | Pending |
| 4 CLI | Split agent/eval, module commands, new/check/migrate, environment renames, policy configure; all planner tests, all plugin checks, both policies served and evaluated | Pending |
| 5 Data generation | IK scripted expert, shared success-filtered recorder and calibration replay, then Mimic; successful generated LeRobot episodes and policy preparation/training | Pending |
| 6 Second arm | OMX factory calibration/utilities/mount; sim record/eval on blank table, then paired hardware recording; no arm-specific src edits | Pending |
| 7 Docs/conformance | Eight guides, templates and conformance; each plugin type addable from the docs alone | Pending |

The stage-logic deduplication in section 2.3 is explicitly assigned to Phase 3 by the plan. The large agent/eval
splits listed in section 2.4 are assigned to Phase 4. Their work remains required.

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
