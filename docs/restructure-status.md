# v2 restructure status

The complete user-supplied restructure plan (sections 1–7) governs this work. This file records progress and evidence;
it does not replace that plan or reduce its scope. Work stays on branch `v2`.

## Phase gates

Every phase must pass the complete CPU suite without `--ignore`. Simulator changes must also be exercised inside
`s2r-sim` on the SO-101 in the meeting room before the next phase begins.

| Phase | Required acceptance | Status |
| --- | --- | --- |
| 0 Cleanup | Section 2 cleanup; CPU tests; record/eval/serve/train dry runs unchanged | Complete |
| 1 Arm | Typed arm contract and per-unit calibration; SO-101 utilities; synthetic four-joint arm; sim-only recorded episode, smoke eval, identical replay joint targets | Pending |
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

`outputs/` contains local verification artifacts and is ignored. Passing a helper or smoke test alone is not evidence
that later-phase record, training, policy evaluation, image matching or real-hardware gates have passed.
