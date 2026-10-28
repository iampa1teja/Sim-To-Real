# Prompt: build the sim-vs-real joint calibration check

You are working in the Sim-To-Real SO-101 repository on the GPU machine. Read `CLAUDE.md`/`AGENTS.md`
first; they describe the containers (teleop/sim, serve, train) and the rules below. Hardware commands run in
the **teleop container**. CPU-only tests run on the host with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Goal

The leader and follower arms get calibrated with `lerobot-calibrate`. The simulated arm must use the
calibration that is actually in use, chosen through environment variables. Then, during an ordinary
teleoperation run with the real follower attached, we log the difference between the real arm's joint angles
and the sim's joint angles every tick. At the end of the run we report the **average gap per joint**,
plus enough breakdown to tell what is causing the gap.

## Background you must understand before writing code

- Leader encoder → LeRobot normalized units (−100..100 for the five body joints, 0..100 gripper) using the
  **leader's** calibrated range. The real follower and the sim both receive that same normalized value.
- The sim converts it to degrees with the follower's span: `utils/lerobot_interface.py`
  `get_mapped_actions_vectorized`, which is mathematically `(count − 2048) × 360 / 4096` for body joints, then
  clamps to `runtime_joint_limits` from `assets/real_setup.json`.
- The span comes from `assets/real_setup.json` → `sim_joint_mapping`, derived by
  `utils/joint_calibration.follower_mapping_from_calibration` from the embedded follower file
  (`rpd_robot`). `assets/real_setup.py` raises if the stored mapping is stale. `scripts/update_sim_calibration.py`
  regenerates it. There is **no environment-variable override yet**.
- `real_rollout` (`scripts/real_rollout.py`) already loads the follower file from
  `$HF_LEROBOT_CALIBRATION/robots/so101_follower/$ROBOT_ID.json` (default root
  `~/.cache/huggingface/lerobot/calibration`). Leader files live under `.../teleoperators/so101_leader/$TELEOP_ID.json`.
  Env vars already used: `ROBOT_PORT`, `ROBOT_ID`, `TELEOP_PORT`, `TELEOP_ID`.
- **Key limitation, which must appear in the report:** "real angle" here is computed from the follower's
  encoder through the same calibration the sim uses. So this check can measure mapping mismatches, clamping,
  sign errors, wrist-frame offsets and tracking lag (real or sim). It **cannot** detect a zero-pose error
  made during `lerobot-calibrate`, where the arm held during "move to middle of range" was not the USD zero
  pose. That needs an external reference (operator visually matching a sim pose, inclinometer, or camera).
  Phase 2 below adds the operator-pose mode for that.

### Existing code to reuse, not duplicate

- `utils/joint_angle_check.py`:
  - `JointAngleTelemetry` is already a passive per-tick JSONL tap in `scripts/lerobot_agent.py`
    (`--joint_angle_log`, requires `--enable_real_follower`). It logs follower units, mapped degrees, the
    PhysX read-back (`root_physx_view.get_dof_positions`, not the target) and their difference.
  - `summarize_teleop` reports median and p95 gaps.
  - `make_observation_only_robot` + `observation_only_session` give a torque-off, write-guarded follower.
- `utils/joint_angle_check_physx.py` (`--physx-check`): offline zero-step replay. It passed 468/468 within 0.05°.
- `scripts/joint_angle_check.py` is the CLI (`--watch`, summarize, `--physx-check`).
- `docs/reports/2026-10-27-joint-angle-check/report.md` covers the previous run. In it, elbow_flex was at a
  runtime limit in 442/468 frames with a +192.6° median sim−mapped gap, wrist_roll had +88.0°, and
  shoulder_pan was at a limit in 459/468. **Explain these before trusting new numbers.** Likely suspects are
  clamping against `runtime_joint_limits`, a full-turn branch, and the USD-vs-arm 90° wrist roll frame
  (`wrist_roll_model_offset`, `wrist_roll_zero_offset_degrees`).
- `pick_place_agent` has `--enable_real_follower` but no joint-angle log.

## Work to do

### 1. Calibration selection through environment variables

- Add `SO101_FOLLOWER_CALIBRATION` (path to a follower JSON). If it is unset but `ROBOT_ID` is set, fall back to
  the `real_rollout` default path. Add `SO101_LEADER_CALIBRATION` the same way with `TELEOP_ID`; it is used for
  provenance and to verify LeRobot loaded the same file.
- When a follower file is selected, the sim mapping (`MyRoomEnvCfg.sim_joint_mapping` and anything that reads
  `SETUP["sim_joint_mapping"]`) must come from `follower_mapping_from_calibration(file,
  **sim_mapping_parameters())`. Put the resolution in one function in `utils/joint_calibration.py`, used by
  `real_setup.py`, the agents and `real_rollout`, so there is one code path.
- Print the follower and leader file paths and SHA-256 at startup. Fail clearly if the file the robot actually
  loaded (`robot.calibration_fpath`) differs from the one the sim mapping used.
- With no env var set, behavior must be byte-identical to today: the `real_setup.json` mapping and its stale check.
- Recording: `utils/recording_alignment.recording_contract` must include the selected calibration hash, so
  datasets recorded under different calibrations are refused when resumed into the same root.

### 2. New module `source/sim_to_real_so101/utils/calibration_check.py`

Keep it Isaac-free and hardware-free (numpy only) so it is unit-testable on the host. Callers pass arrays in.

- `CalibrationGapLogger(path, *, joint_order, runtime_limits, provenance)`:
  - Open the JSONL with `'x'` and never overwrite. The first line is a header: follower/leader calibration
    paths and SHA-256, the `real_setup.json` SHA-256, the mapping in use, the runtime limits, the git commit,
    and the interpretation/limitation text.
  - `record(...)` writes one row per tick with a timestamp, the frame index, and for all six joints:
    - `commanded_deg`: the leader action converted with the follower mapping, i.e. what both arms were told
    - `real_deg`: the follower observation converted with the same mapping
    - `sim_deg`: the PhysX read-back
    - `sim_target_deg`: the clamped target sent to the sim
    - gaps: `sim − real` (the headline), `real − commanded` (real tracking), `sim − sim_target` (sim tracking)
    - `at_limit` flags (within 0.05° of a runtime limit) and a per-joint `static` flag: velocity of both
      `real_deg` and `sim_deg` below a threshold over a short window
  - Never wrap differences modulo 360. Flag any |gap| > 180° as `suspected_branch_error` instead.
- `summarize(path, thresholds) -> dict`, plus a markdown table writer. Per joint:
  - mean signed gap, mean |gap|, median, std, p95 |gap|, max |gap|
  - frame counts: used, excluded at limit, excluded moving
  - computed for **all frames** and for **static, not-at-limit frames**
  - the same stats for the real-tracking and sim-tracking gaps, so a reader can tell lag from a constant offset
  - correlation/slope of the gap against `real_deg`: a slope near 0 means a constant offset, a slope near −2
    means an inverted sign, anything else points to a span/scale error
  - a verdict per joint on static frames: PASS if mean |gap| ≤ 1°, WARN ≤ 3°, FAIL otherwise, with the
    thresholds configurable. Report the gripper separately (its closed anchor is marked unresolved in
    `real_setup.json`).
  - the limitation paragraph from "Background" above, verbatim in meaning
- The summary must be written automatically when the run ends, including on Ctrl-C and on exceptions
  (`<log>.summary.json` and `.summary.md`). It must also be reproducible offline from the JSONL.
- Refactor `JointAngleTelemetry` to delegate to this module. Keep `--joint_angle_log` and its existing
  JSONL fields working; existing tests must still pass.

### 3. Wiring

- `scripts/lerobot_agent.py` and `scripts/pick_place_agent.py`: add `--calibration_check_log PATH` (requires
  `--enable_real_follower`). Observe the previous completed tick **before** sending the next command, exactly
  like the existing tap. The tap must never command either arm and never call calibration.
- A live line once per second: `joint: real / sim / gap` for the five body joints.
- CLI `scripts/calibration_check.py`: `summarize --log X [--output Y] [--static-velocity-deg-s ...]
  [--pass-deg 1 --warn-deg 3]` and `watch --log X`.

### 4. Phase 2: operator reference poses (catches zero-pose errors)

- `scripts/calibration_check.py poses`: the sim shows a named target pose: all-zero, then 4–6 further poses
  spread across each joint's range, away from limits. The operator, with the real follower torque-off and
  supported, physically matches the real arm to the rendered pose and presses Enter. Use
  `observation_only_session`; this mode must issue no motor writes except torque-off.
- Record 30 samples per pose, reject the pose if the std exceeds 0.3 units, and compute real − target per
  joint. Use `summarize_candidates`' consistency rule (≥5 accepted poses, spread ≤ 1°) to label each joint's
  offset CONSISTENT or UNRESOLVED.
- Output **candidate** per-joint zero offsets only. Apply nothing, and edit no file.

### 5. Tests (CPU-only, `tests/test_calibration_check.py`)

Use synthetic data:
- a known constant offset per joint is recovered
- an inverted sign is detected through the slope
- at-limit and moving frames are excluded
- a gap above 180° is flagged, not wrapped
- the summary is written on an exception path
- an empty log raises
- the existing file cannot be overwritten
- env-var resolution: unset gives the current mapping, set gives the mapping from that file, and a mismatch
  with the loaded file raises

Also run the existing `tests/test_joint_angle_check.py`, `tests/test_joint_calibration.py` and
`tests/test_real_rollout*.py` and keep them green. Run `flake8` with the repo config.

### 6. Docs

Add a "Sim-vs-real calibration check" section to `docs/08-real-arm-rollout.md`, or a new
`docs/09-calibration-check.md` linked from `README.md`. It must include the exact commands for:
calibrating the leader and follower, exporting the env vars, running teleop with `--calibration_check_log`,
reading the summary, and running the pose mode. Add one line about this to `CLAUDE.md`.

## Hard rules

- Never edit, regenerate or overwrite real calibration files, recorded datasets (`datasets/`), or upstream
  GR00T sources. Never call `calibrate()`, `write_calibration`, or anything that sets homing.
- Do not launch hardware runs yourself. The operator starts every teleop or pose session manually. Give them
  the exact commands and stop.
- Do not change `assets/real_setup.json` as part of this work. A changed mapping makes before/after evaluation
  results incomparable (see `docs/reports/2026-10-27-calibration-fixes/report.md`).
- Default behavior with no new env vars and no new flags must not change.

## Done means

- The code, tests and docs are committed on a branch, all CPU tests are green and flake8 is clean.
- You have written an explanation of the earlier +192° elbow and +88° wrist-roll gaps.
- You have handed the operator a short run sheet:
  1. calibrate both arms
  2. `setenv` the variables
  3. teleop for about 2 minutes, sweeping each joint slowly through its range and holding still at about
     5 poses per joint
  4. read `<log>.summary.md`
  5. run the pose mode

  Then report back the per-joint average gap table and verdicts.
