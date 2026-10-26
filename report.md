# Sim vs. real SO-101 calibration audit

Date: 2026-10-26. Read-only audit: no files edited, no calibration changed, no motion, no torque commanded.

**Verdict: the sim and the real arm agree only on shoulder_pan, and even that joint is cut off near both ends.**
The other five joints still use the generic `SO101_USD_MAPPING` defaults, and four of those differ from the real arm
by more than the 3° span limit. There is no live check-5 reading: no arm was plugged in (`/dev/ttyACM*` did not exist).

## Method

- Four one-shot `teleop-docker:latest` containers with the repo and
  `~/.cache/huggingface/lerobot/calibration` mounted read-only and the entrypoint skipped (its `pip install -e` writes
  into the repo). USD was read through `omni.usd.libs` from the image's `extscache`.
- Calibration IDs come from `docker/env`: `ROBOT_ID=rpd_robot`, `TELEOP_ID=rpd_teleop`. The `orange_*` files in the same
  folder belong to another arm pair and were not used.
- Container LeRobot version: 0.4.3. Normalization: body `((v - min) / (max - min)) * 200 - 100`, gripper
  `((v - min) / (max - min)) * 100`, input clipped to [min, max].
- Sim side: `LeRobotSO101Interface(..., joint_mapping=SETUP["sim_joint_mapping"])`, functions
  `get_mapped_actions_vectorized` and `get_raw_actions_from_radians`.

## 1. Joint ranges (follower `rpd_robot` vs. the mapping the sim actually uses)

Real degrees = (count − 2048) × 360/4096.

| joint | counts | real lo..hi (°) | real span / centre | sim lo..hi (°) | sim span / centre | Δspan | Δcentre | where the sim value comes from | result |
|---|---|---|---|---|---|---|---|---|---|
| shoulder_pan | 725–3411 | −116.28..119.79 | 236.07 / 1.76 | −116.28..119.79 | 236.07 / 1.76 | 0.00 | 0.00 | override | **PASS** (matches the worked example) |
| shoulder_lift | 831–3326 | −106.96..112.32 | 219.29 / 2.68 | −100..100 | 200 / 0 | −19.29 | −2.68 | default | **FAIL** |
| elbow_flex | 877–3109 | −102.92..93.25 | 196.17 / −4.83 | −100..90 | 190 / −5 | −6.17 | −0.17 | default | **FAIL** (span only) |
| wrist_flex | 888–3245 | −101.95..105.21 | 207.16 / 1.63 | −95..95 | 190 / 0 | −17.16 | −1.63 | default | **FAIL** |
| wrist_roll | 0–4095 | −180..179.91 | 359.91 / −0.04 | −160..160 | 320 / 0 | −39.91 | 0.04 | default | **FAIL** |
| gripper | 2039–3514 | −0.79..128.85 | 129.64 / 64.03 | −10..100 | 110 / 45 | −19.64 | −19.03 | default | **FAIL** |

Degrees per raw unit: follower body [1.1804, 1.0964, 0.9809, 1.0358, 1.7996], gripper 1.2964; sim body
[1.1804, 1.0, 0.95, 0.95, 1.6], gripper 1.1.

Notes:
- **The real wrist_roll range wasn't measured.** LeRobot sets wrist_roll to 0..4095 automatically during calibration.
- **LeRobot's zero point is count 2047, not 2048** (`int(4095/2)` in `_get_half_turn_homings`). That shifts everything
  by 0.09°, which doesn't matter.
- **The real assumption behind the centres is unchecked for every joint.** Each centre assumes the pose the operator
  held at "move to the middle of the range" equals the USD zero pose. Only shoulder_pan has a recorded spot check
  ("verified with first encoded pick/place episode" in `sim_joint_mapping_source`).
- Timing is consistent: `rpd_robot.json` was last modified 2026-09-29 10:03, the pan override was committed in
  `b9a360e` the same day, and `pick_place_v1-old` / `-real` were recorded on 2026-10-01.

## 2. USD limits (`assets/usd/SO-ARM101-USD.usd`, `UsdPhysics.RevoluteJoint`, all Z axis)

| joint | USD limits (°) | sim clamp | clamp inside USD | sim mapping inside clamp | real range inside USD |
|---|---|---|---|---|---|
| shoulder_pan (Rotation) | ±110 | ±110 | PASS | **FAIL** (−116.3..119.8) | **FAIL** |
| shoulder_lift (Pitch) | ±100 | ±100 | PASS | PASS | **FAIL** (−107.0..112.3) |
| elbow_flex (Elbow) | −100..90 | same | PASS | PASS | **FAIL** (−102.9..93.3) |
| wrist_flex (Wrist_Pitch) | ±95 | same | PASS | PASS | **FAIL** (−102.0..105.2) |
| wrist_roll (Wrist_Roll) | ±160 | same | PASS | PASS | **FAIL** (±180, auto range) |
| gripper (Jaw) | −10..100 | same | PASS | PASS | **FAIL** (max 128.8) |

On every joint, the real arm's calibrated range is wider than what the USD model allows.

## 3. Leader `rpd_teleop` vs. follower `rpd_robot` (drive_mode is 0/0 on all joints)

| joint | leader counts | follower counts | leader span | follower span | Δspan | Δcentre | angle error at −100 / +100 for the same raw value | result |
|---|---|---|---|---|---|---|---|---|
| shoulder_pan | 746–3457 | 725–3411 | 238.27 | 236.07 | 2.20 | 2.94 | +1.9 / +4.0 | WARN |
| shoulder_lift | 794–3179 | 831–3326 | 209.62 | 219.29 | −9.67 | −8.09 | −3.3 / **−12.9** | **FAIL** |
| elbow_flex | 827–3069 | 877–3109 | 197.05 | 196.17 | 0.88 | −3.96 | −4.4 / −3.5 | WARN |
| wrist_flex | 927–3253 | 888–3245 | 204.43 | 207.16 | −2.72 | 2.07 | +3.4 / +0.7 | WARN |
| wrist_roll | 0–4095 | 0–4095 | 359.91 | 359.91 | 0 | 0 | 0 / 0 | PASS (both auto ranges) |
| gripper | 2028–3326 | 2039–3514 | 114.08 | 129.64 | −15.56 | −8.75 | −1.0 / **−16.5** | **FAIL** |

Homing offsets differ between the two arms (leader 588 / −642 / 2022 / 1887 / 683 / −1570, follower −1078 / 1811 /
1458 / 1152 / 1578 / −458), as expected for different hardware. During recording the sim maps the *leader's* raw
values and the follower runs the same raw values (`pick_place_agent.py`), so what matters for sim-to-real agreement is
check 1. This table only shows how far the follower's angles drift from the leader's.

## 4. Round trip (raw → radians → raw)

| joint | worst error over grid | −100 / +100 (gripper 0 / 100) map exactly to mapping min/max | raw range that survives the clamp | result |
|---|---|---|---|---|
| shoulder_pan | **8.30** | no: lands on −110 / +110 instead of −116.28 / 119.79 | [−94.68, 91.70] | **FAIL** |
| shoulder_lift | 0 | yes | full | PASS |
| elbow_flex | 1e-5 | yes | full | PASS |
| wrist_flex | 1e-5 | yes | full | PASS |
| wrist_roll | 0 | yes | full | PASS |
| gripper | 0 | yes | full | PASS |

Grid: body −100, −75, −50, −25, −10, 0, 10, 25, 50, 75, 100; gripper 0, 10, 25, 50, 75, 90, 100.
The clamp silently caps values: an input of raw 150 lands on the USD limit with no warning. Shoulder_pan's ±100 comes
back as −94.68 and +91.70. The other joints pass only because their default mapping equals the clamp.

## 5. Rest pose: live reading not done; estimated from recorded data

The repo script `docker/real/scripts/so101_check_calibration.py` has no read-only mode. It calls
`SO101Follower.connect()`, which runs `configure()`. That writes motor registers and turns torque back on
(`bus.torque_disabled()` re-enables torque on exit), so it was not run.

Stand-in: the first frame of each of the 75 real episodes in `pick_place_v1-real`.

| joint | sim start pose (°) | real first-frame raw (median, min..max) | real angle by follower calibration | sim angle by mapping | sim − start | real − start |
|---|---|---|---|---|---|---|
| shoulder_pan | −15.68 | −3.50 (−12.29..3.80) | −2.37 | −2.37 | 13.30 | 13.30 |
| shoulder_lift | −89.95 | −99.28 (−99.68..−89.98) | −106.17 | −99.28 | −9.32 | −16.22 |
| elbow_flex | 83.08 | 98.48 (97.22..98.57) | 91.76 | 88.55 | 5.47 | 8.68 |
| wrist_flex | 54.40 | 75.05 (61.14..76.24) | 79.37 | 71.30 | 16.90 | 24.97 |
| wrist_roll | −1.87 | −5.54 (−14.38..2.37) | −10.02 | −8.87 | −7.00 | −8.15 |
| gripper | −8.39 | 1.69 (1.56..18.51) | 1.41 | −8.14 | 0.26 | 9.80 |

- **The sim start pose (`robot.joint_positions`) doesn't match the real resting pose.** The joints differ by 5–25°.
- **The real arm rests outside the USD limits.** Its shoulder_lift rest angle (≈ −106°) is past the USD limit of −100°.
- **A safe live reader is ready** (not in the repo):
  `/tmp/claude-1116/-home-rpuser1-Sim-to-Real-SO-101-Workshop/6ff28e50-0bed-4049-afc5-b19c6607b235/scratchpad/read_rest_pose.py`.
  It uses only `FeetechMotorsBus.connect` (ping and firmware read), `sync_read`, and `disconnect(disable_torque=False)`.
  It also checks that the homing offset and position limits stored on the motors match the JSON. It has only been
  confirmed to load in the container; with no arm attached it stopped at "could not connect".

## 6. Recorded data

`datasets/pick_place_v1` is not the real dataset. Its `meta/dr_replay_manifest.json` says it is a domain-randomized
re-render of `pick_place_v1-old`, the sim side. The real follower data is `pick_place_v1-real`. Both have 16,448
frames recorded side by side from the same leader stream (episode and frame indices align exactly).

| joint | real state min / max / mean | sim state min / max / mean | real − sim per frame (p95 / max) |
|---|---|---|---|
| shoulder_pan | −96.28 / 79.97 / 3.81 | **−94.68** / 80.49 / 3.83 | 6.15 / 17.60 |
| shoulder_lift | −99.76 / 88.70 / −30.23 | −99.67 / 90.67 / −30.43 | 5.98 / 12.86 |
| elbow_flex | −70.97 / 99.10 / 39.83 | −70.46 / 100.0 / 39.73 | 5.11 / 15.06 |
| wrist_flex | −20.07 / 98.64 / 59.20 | −20.65 / 99.37 / 59.42 | 2.45 / 9.83 |
| wrist_roll | −65.03 / 22.00 / −7.16 | −60.70 / 23.21 / −3.32 | 6.95 / 8.62 (sim re-zeroes this per episode by design) |
| gripper | 1.42 / 88.27 / 19.68 | 1.23 / 88.41 / 20.43 | 3.68 / 24.95 |

- **The clamp shows up in the data.** The sim's shoulder_pan minimum sits exactly at the −94.68 clamp limit.
  Episode 59 has 53 frames where the real pan went to −96.9 (−112.6°) and the sim stopped at −110°.
- **These numbers can't reveal a wrong scale.** Both datasets store raw units, and the sim converts back through the
  same mapping, so matching stats are expected even when the angles disagree.
- With the leader commanding a closed gripper (9,155 frames), the real and sim gripper states are 2.24 and 2.70 raw,
  but 2.1° on the real arm vs. −7.0° in the sim.

The real gap appears when you ask what the same raw value means physically. Over all real frames, sim angle minus
real angle (degrees) is:

| joint | mean | p95 | max |
|---|---|---|---|
| shoulder_pan | 0.0 | 0.0 | 1.9 |
| shoulder_lift | 0.2 | 7.1 | 11.2 |
| elbow_flex | −1.4 | 3.2 | 3.2 |
| wrist_flex | −6.7 | 8.3 | 10.1 |
| wrist_roll | 1.5 | 4.8 | 13.0 |
| gripper | **−13.1** | **19.3** | **26.5** |

## 7. Cameras

Resolution is 640×480 in `real_setup.json`, in `docker/env` (`CAMERA_WIDTH` / `CAMERA_HEIGHT`) and in both datasets:
**PASS**.

| camera | fx / fy from aperture vs. measured | cx / cy, image centre vs. measured | measured values written by `apply_opencv_intrinsics` | how the values were obtained | result |
|---|---|---|---|---|---|
| realsense | 605.87 vs 606.07 / 605.67 (−0.03% / +0.03%) | 320 / 240 vs 330.64 / 253.33 (−10.6 px / −13.3 px) | yes: fx 606.067, fy 605.669, cx 331.142, cy 253.827 (includes the +0.5 pixel offset), zero distortion | factory SDK values from the live stream, 2026-09-24 | **PASS** |
| wrist | 554.26 vs 554.26 (0%) | 320 / 240 vs 315.6 / 230.2 (+4.4 px / +9.8 px) | yes: cx 316.1, cy 230.7, k1 0.164 | hand-tuned visually (FOV set to exactly 60°), not a measured calibration | **unverified** |

- **The off-centre principal point exists only in the lens-model attributes.** A plain USD aperture would put it at
  the image centre, off by 10–13 px (realsense) or 4–10 px (wrist). The attributes were confirmed on a test camera
  prim, and `CalibratedTiledCamera` reports K from them, but no frame was rendered to prove the RTX renderer uses them.
- The realsense extrinsics are a data fit ("edge-snapped real arm masks ... real-state FK + separate joint-offset
  nuisance params"). The fit's outputs (`visual_matching/work/ext_cam2`) are not in the repo, so its joint-offset
  estimates could not be compared with this audit.

## 8. Base pose: no recorded provenance, unverified

- `robot.position` = [0.697253387506893, −0.6076356057211214, 0.7635429149057084]
- `yaw_degrees` = 184.583662
- `table_normal` = [−0.001159014, −0.01850508, 0.9998281] (≈ 1.06° tilt)

The yaw and table normal date from `ee434bd` (2026-09-23), whose file description then said "visual estimates, not
measured calibration". The x position changed from 0.7721 to 0.6973 in `b9a360e` (2026-09-29), with no measurement
note. The camera fits only record "robot pose unchanged". Nothing in the docs, commit messages or changelogs says how
these values were measured.

## Summary

| item | status |
|---|---|
| shoulder_pan mapping | matches calibration, but clipped by the ±110° clamp outside raw [−94.68, 91.70] |
| shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper mappings | **mismatched** (generic defaults; see section 1) |
| USD joint limits | narrower than the real range on every joint |
| leader vs. follower | shoulder_lift and gripper differ materially |
| round trip | exact except shoulder_pan (clamp) |
| rest pose | not measured live; recorded first frames differ from the sim start pose by 5–25° |
| realsense intrinsics | verified (factory values, applied in code) |
| wrist intrinsics | unverified (visual tuning) |
| joint zero = USD zero | unverified for all joints except a single-episode spot check on shoulder_pan |
| base pose | unverified (no provenance) |

## Proposed edits (not applied)

Add to `source/sim_to_real_so101/assets/real_setup.json` under `sim_joint_mapping`, with entries added to
`sim_joint_mapping_source` to match:

```json
"shoulder_lift": {"joint_min": -106.962890625, "joint_max": 112.32421875},
"elbow_flex":    {"joint_min": -102.919921875, "joint_max": 93.251953125},
"wrist_flex":    {"joint_min": -101.953125,    "joint_max": 105.205078125},
"wrist_roll":    {"joint_min": -180.0,         "joint_max": 179.912109375},
"gripper":       {"joint_min": -0.791015625,   "joint_max": 128.84765625}
```

Decisions for the owner before applying any of them:

- **The JSON change alone makes the sim cut off more.** The clamp would hold shoulder_lift to raw [−93.65, 88.76],
  wrist_flex to [−93.29, 90.15], elbow_flex to [−97.02, 96.68], wrist_roll to [−88.89, 88.94] and the gripper to
  [0, 77.75]. The recorded data goes past the shoulder_lift, wrist_flex, elbow_flex and gripper limits, and
  shoulder_pan is already cut off at both ends. To avoid that, widen the clamp (`SO101_USD_MAPPING` in
  `utils/lerobot_interface.py`) and the USD joint limits together. That's a code and asset change, not a JSON one.
- **Check the zero first, especially for the gripper.** Take at least one physical reference pose per joint and
  compare it with the sim; `read_rest_pose.py` gives the real side. The gripper entry is only right if the USD Jaw = 0
  is the real jaw's calibration pose. The wrist-camera calibration note already mentions a "gripper sim/real
  calibration mismatch".
- **Changing the mapping changes the sim's rendered poses.** The existing sim datasets (`pick_place_v1`,
  `pick_place_v1-old`), and checkpoints trained on them, used the current mapping, and eval baselines will shift.

The optional hand-moved range test (each joint to its physical limits with torque off) was not done. It needs the
operator's written agreement and the operator turning torque off themselves.


## 9. Follow-up implementation and simulator experiments — 2026-10-26

This section records subsequent user-authorized changes and experiments. The original read-only audit above is
preserved. No real arm was connected or moved; all real-arm implementation tests used fakes.

### Real rollout mapping fix

`real_rollout` now reads all six follower ranges from
`$HF_LEROBOT_CALIBRATION/robots/so101_follower/$ROBOT_ID.json` (or the standard Hugging Face calibration cache).
`--calibration_file` overrides the mapping file. Encoder counts convert using `(count - 2048) * 360 / 4096`.
Missing or invalid mapping files return exit code 2 before interface initialization. The calibrated mapping is
passed to the follower interface. Shadow mode remains the default. The user subsequently requested that the
real-rollout default action horizon increase from 8 to 30; an explicit override is still supported.

Validation: **30 targeted tests passed**. Before/after per-test results for smoothing, grasp, evaluator rollout,
and GR00T transport were identical: 108 passed, 4 pre-existing failures, 14 skipped. Full-suite before/after
FAILED/ERROR sets were also identical: before 509 passed; after 518 passed, with the same 89 failures,
53 skipped tests and 14 collection errors. System Python was used because the default Conda Python lacks pytest.
The simulator mapping, source, limits and rest-pose settings were preserved. Detailed validation and full tracked
diff: `/tmp/real_rollout_report.txt` and `/tmp/real_rollout_changes.diff`.

### Frame convention and shoulder-lift findings

The user confirmed that the real follower was homed with **wrist roll at the camera-up, 90-degree orientation**.
`SO-ARM101-USD.usd` retains a baked 90-degree wrist-roll frame; `ARM_new.usd` layers over it and explicitly
removes that offset, with matching gripper/jaw zero-pose transforms. The Molmo client applies joint-frame offsets
`[0, 90, 90, 0, 0, 0]`, with no explicit wrist-roll correction. Whether the physical calibrated zero matches the
checkpoint's joint frame remains unverified. A signed wrist offset is a hypothesis to test, not a verified
calibration correction; rotating an image or changing a range midpoint alone does not establish agreement.

For the required 2048-count convention, the calibrated range midpoint is the correct affine offset when converting
normalized observations back to angles. It is not the physical zero. The midpoint arithmetic therefore does not
resolve the physical/model zero-frame question.

The first five fixed benchmark starts use shoulder_lift = **-99.19216049194335 degrees** under the unchanged
simulator mapping. The simulator command clamp is **-100 degrees**, leaving approximately **0.81 degrees** of
additional downward command travel from the start. The real calibrated lower bound is **-106.962890625 degrees**.
A different mapping alone would not remove the simulator's separate clamp. The live joint target was not sampled,
so this identifies the available range rather than proving saturation of a particular policy output.

### Initial five-episode comparison

Both configurations use exactly **front/wrist** inputs, H30/R1 (one query per simulated second), the released
SO-101 checkpoint with BF16 backbone/FP32 action storage, existing pan adaptation, the first five fixed starts,
and a 180-second simulated episode deadline. The user explicitly confirmed that real calibration is used only by
`real_rollout`; simulator mapping and rest-pose settings stay unchanged. Only `robot_usd` differs between trials.
These are small stochastic smoke trials, not a population success estimate or proof of real-arm correctness.

| Trial | Asset | Completed | Placements | Outcome |
|---|---|---:|---:|---|
| Front/wrist control | `ARM_new.usd` | 5/5 | **2/5** | MC and MR succeeded; FR, NL and FL timed out without reaching |
| Camera-up wrist frame | `SO-ARM101-USD.usd` | 5/5 | **2/5** | MC and MR succeeded; FR, NL and FL timed out without reaching |

The control exited 0 with no reported runtime errors. Its successful placement times were 11.47s (MC) and 87.37s
(MR), median 49.42s simulated. Logs/configs are under `/tmp/so101-mapping-smoke/`: `smoke-front-wrist.log`,
`smoke-old-usd.log`, `pick_place_molmo_two_cameras.yaml`, and `pick_place_molmo_old_usd.yaml`.
The old-USD run completed all five episodes, exited 0, and reported no runtime errors. Its successful placement times were 40.30s (MC) and 14.30s (MR), median 27.30s simulated. The asset change alone did not improve the five-start placement count: both assets placed MC and MR and failed to reach FR, NL and FL. Different placement times in this small comparison do not establish a repeatable speed improvement.

The earlier three-camera run and an intermediate run using `observations.front`/`observations.wrist` were
intentionally interrupted when the user corrected the camera selection/names. They are excluded from this
comparison. Their Rerun shutdown disconnect messages are not inference failures in the completed control.

Further experiments will be appended here with their exact changed setting, fixed conditions, terminal outcome,
and limits on the conclusions. Candidate results will be checked against matched controls before treating an
apparent placement gain as repeatable.


### Next experiment protocol

The next batch uses the repository's existing **one original scheduled start per zone** smoke protocol, with
60 simulated seconds to reach and 30 seconds from each new milestone (210-second hard ceiling). The two cameras,
recorded rest-pose settings, simulator joint mapping, H30/R1, checkpoint, precision and task success rule remain
fixed. Benchmark file: `eval_sets/pick_place_v1_seed1984.json`, SHA-256 `42c5054fa93a548dd85bf57cb1bf62331076a3384401927ed2261bf33bcdd259`. Each start retains its recorded
policy seed. Nine-zone scores will be compared with controls under this same protocol, not with the five-start,
180-second scores above.

Planned configuration comparisons are: old-USD control; old USD with -90-degree or +90-degree model wrist-roll
offset; old USD without the existing experimental pan adaptation; and a matched `ARM_new.usd` control. The signed
wrist offsets use the already-supported `arm_zero_degrees` client option, which converts observations and inverse
actions consistently. They do not change the simulator mapping, rest pose, camera extrinsics or saved physical
calibration. A better score would motivate replication, not establish a physical calibration correction.

The serialized runner `/tmp/so101-mapping-smoke/run_trials.py` retains full logs/configs/results, verifies the
terminal nine-episode report and actual runtime wrist offset/camera inputs, and appends completed findings here.
Incomplete or failed runs receive no validated score. Trial artifacts are in `/tmp/so101-mapping-smoke/trials/`.


Protocol caveat observed in the live old-USD control: MC reached immediately but was stopped at **30.03s**
by the progress deadline before grasping. The preceding full-length old-USD trial placed MC at **40.30s**.
The nine-zone protocol therefore screens prompt milestone progress and can discard late eventual successes.
Its scores must not be interpreted as unrestricted task capability. Matched screening comparisons remain useful,
but candidate/default decisions require full-length confirmation against a full-length control.


### Tighter shoulder-lift restriction: checkpoint output quantiles

Read the deployed checkpoint's `norm_stats.json` and `modeling_molmoact2.py`, rather than inferring a bound from
rendered motion. The checkpoint uses `norm_mode=q01_q99`; its action normalizer clips normalized outputs to
[-1,1] before unnormalizing. The shoulder-lift model-angle quantiles are **45.182583581650..186.106466800778 degrees**.
The unchanged client uses `physical_lift = 90 - model_lift`, so returned physical targets are restricted to
**-96.106466800778..44.817416418350 degrees** under the default transform.

The benchmark start is -99.19216049194335 degrees, below this learned-output lower bound. Thus the default policy
cannot request a lower shoulder-lift target than its starting angle, independently of the separate -100-degree
simulator clamp. This is a stronger explanation of the observed limited downward motion than the clamp alone.
It does not prove that this one joint explains every failed approach. Existing pan adaptation and the signed
wrist-only experiments do not change this shoulder bound. The normalizer and checkpoint weights remain unchanged.

A subsequent configuration-only experiment can change the model-coordinate scale/offset consistently in both
state and inverse action conversion through supported client options. That is an experimental adaptation of the
policy's learned range, not evidence of a different physical calibration or permission to widen simulator limits.


### Prepared range-adaptation candidates (not yet scored)

Read all 75 simulation action parquet chunks, **16,448 action frames**, without reading or modifying videos.
Converted body actions using the unchanged simulator spans and canonical Molmo signs/offsets, then fitted one
positive affine model-coordinate transform per joint so local action q01/q99 match the released checkpoint's
q01/q99. The independently calculated pan gain and shift reproduce the existing pan adaptation to numerical
precision. This checks the method against the implemented pan fit; it does not establish performance gains for
other joints.

| Joint | Model-coordinate gain | Model-coordinate shift (degrees) |
|---|---:|---:|
| shoulder_pan | 0.490169730 | 3.535232176 |
| shoulder_lift | 0.891048328 | 17.255595856 |
| elbow_flex | 0.985999651 | -2.869560768 |
| wrist_flex | 0.913314392 | 15.896622225 |
| wrist_roll | 1.088649662 | 17.150841474 |

Prepared supported-option candidates isolate shoulder lift, wrist flex, wrist roll, and all five body joints.
State and inverse action transforms use the same affine mapping; the gripper mapping, checkpoint statistics,
weights, simulator joint mapping, joint limits and rest-pose settings stay fixed. These are empirical policy
adaptations, not physical calibrations. Configurations and exact parameters are in
`/tmp/so101-mapping-smoke/trials/range_candidates.json` and `quantile_fit.json`. They have not yet been run, and
no success claim is made for them.


### Nine-zone trial: old_control

Old USD, existing model transform. Completed 2026-10-26T07:22:53.522181+00:00; exit code 0. All nine original scheduled zones completed, retaining their benchmark policy seeds.

Reached **2/9**, grasped **1/9**, confirmed placements **1/9**. The runtime transform reports wrist-roll offset **0.0 degrees**; camera inputs are exactly `front`, `wrist`; H30/R1 is unchanged.

| Zone | Terminal outcome | Simulated seconds | Closest tool error (mm) |
|---|---|---:|---:|
| MC | missed_grasp | 30.03 | 29.61 |
| FR | never_reached | 60.00 | 68.47 |
| MR | success | 38.20 | 6.89 |
| NL | never_reached | 60.00 | 97.69 |
| FL | never_reached | 60.00 | 40.62 |
| NC | never_reached | 60.00 | 41.68 |
| FC | never_reached | 60.00 | 64.47 |
| ML | never_reached | 60.00 | 135.93 |
| OUT | never_reached | 60.00 | 108.92 |

Config: [old_control.yaml](docs/reports/2026-10-26-real-arm-calibration/old_control.yaml) (SHA-256 `b254b1579fae3eee91f4d8f98311ad2c2235bd60a366a49d5363f27e4f26796d`). Full log: [old_control-console.txt](docs/reports/2026-10-26-real-arm-calibration/old_control-console.txt). Machine-readable result: [old_control.result.json](docs/reports/2026-10-26-real-arm-calibration/old_control.result.json). This is a configuration experiment, not a real-arm calibration validation.


### Predicted target support under candidate transforms

Used the actual client implementation with each candidate's options and the checkpoint's clipped action
quantiles to propagate available targets into the unchanged simulator joint coordinates and clamps. These are
mathematical target bounds, not measurements of a live target or estimates of task success. The gap columns show
how far the first fixed start lies outside each candidate's available returned target interval.

| Candidate | Shoulder-lift target degrees | Wrist-roll target degrees | Initial lift gap (degrees) | Initial elbow gap (degrees) | Initial roll gap (degrees) |
|---|---|---|---:|---:|---:|
| old_control | -96.106..44.817 | -65.576..43.531 | 3.086 | 5.645 | 0.000 |
| old_wrist_minus90 | -96.106..44.817 | 24.424..133.531 | 3.086 | 5.645 | 27.017 |
| old_wrist_plus90 | -96.106..44.817 | -155.576..-46.469 | 3.086 | 5.645 | 43.876 |
| old_body_quantile_fit | -99.497..58.658 | -75.990..24.232 | 0.000 | 0.270 | 0.000 |

The default transform also places the initial elbow approximately **5.645 degrees above** its largest available
returned target. Signed wrist offsets change the available wrist target interval and therefore the initial
motion; they are not a harmless relabeling of an otherwise identical trajectory. The empirical body-range fit
reduces these initial lift/elbow gaps, but that alone does not establish grasping or placement performance.
The full analysis and actual client matrices are saved in
[reachable_output_bounds.json](docs/reports/2026-10-26-real-arm-calibration/reachable_output_bounds.json).


### Nine-zone trial: old_wrist_minus90

Old USD, model wrist-roll zero -90 degrees. Completed 2026-10-26T07:32:14.736694+00:00; exit code 0. All nine original scheduled zones completed, retaining their benchmark policy seeds.

Reached **4/9**, grasped **2/9**, confirmed placements **2/9**. The runtime transform reports wrist-roll offset **-90.0 degrees**; camera inputs are exactly `front`, `wrist`; H30/R1 is unchanged.

| Zone | Terminal outcome | Simulated seconds | Closest tool error (mm) |
|---|---|---:|---:|
| MC | success | 8.40 | 10.15 |
| FR | never_reached | 60.00 | 103.78 |
| MR | success | 30.30 | 9.18 |
| NL | never_reached | 60.00 | 97.06 |
| FL | missed_grasp | 36.20 | 26.28 |
| NC | missed_grasp | 53.17 | 19.43 |
| FC | never_reached | 60.00 | 64.65 |
| ML | never_reached | 60.00 | 144.53 |
| OUT | never_reached | 60.00 | 109.35 |

Config: [old_wrist_minus90.yaml](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90.yaml) (SHA-256 `e0855f201fcafbfd791c7f8690b67231885ba5ee2322d90d6ad483246489d5b8`). Full log: [old_wrist_minus90-console.txt](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90-console.txt). Machine-readable result: [old_wrist_minus90.result.json](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90.result.json). This is a configuration experiment, not a real-arm calibration validation.


### Paired control vs. -90-degree wrist-offset interpretation

Verified exact equality of all nine `(start_id, zone, policy_seed)` tuples between the control and -90-degree
trial. The runtime joint transform confirms the signed offset was active. The control reached 2/9, grasped 1/9
and placed 1/9; the -90-degree trial reached **4/9**, grasped **2/9** and placed **2/9**. MC became a timely
placement at 8.40s simulated. FL and NC newly crossed the unchanged 30 mm reach threshold, then missed their
30-second grasp-progress deadlines. This is an approach gain as well as a timely-placement gain, not a solved
grasp in those new zones. An interim chat update missed these two reached milestones; the completed machine
report and this report contain the corrected counts.

| Zone without a placement | Control closest error (mm) | -90-degree closest error (mm) |
|---|---:|---:|
| FR | 68.47 | 103.78 |
| NL | 97.69 | 97.06 |
| FL | 40.62 | 26.28 |
| NC | 41.68 | 19.43 |
| FC | 64.47 | 64.65 |
| ML | 135.93 | 144.53 |
| OUT | 108.92 | 109.35 |

The effect is not uniformly beneficial across all positions. One paired root does not establish repeatability,
and the control's known later MC placement under the full 180-second protocol means the 1/9-to-2/9 screening
score cannot alone establish a full-deadline placement gain. The candidate warrants full-length and independent
seed confirmation. This is no proof of the real arm's physical zero or safe hardware behavior.


Following the completed -90-degree result, additional unscored candidates combine that wrist transform with
isolated shoulder-lift or wrist-flex quantile fits, and a combined four-joint fit retaining the tested wrist
transform. Full 180-second nine-zone control/-90 confirmation configurations have also been prepared without
milestone deadlines. They remain future experiments; prepared files are not counted as results. The batch subsequently completed +90, then stopped on a startup crash in the no-pan trial; the new-USD screen was not started. See the publication checkpoint below.


### Nine-zone trial: old_wrist_plus90

Old USD, model wrist-roll zero +90 degrees. Completed 2026-10-26T07:41:22.175526+00:00; exit code 0. All nine original scheduled zones completed, retaining their benchmark policy seeds.

Reached **3/9**, grasped **2/9**, confirmed placements **2/9**. The runtime transform reports wrist-roll offset **90.0 degrees**; camera inputs are exactly `front`, `wrist`; H30/R1 is unchanged.

| Zone | Terminal outcome | Simulated seconds | Closest tool error (mm) |
|---|---|---:|---:|
| MC | missed_grasp | 30.03 | 30.53 |
| FR | never_reached | 60.00 | 69.21 |
| MR | success | 16.50 | 11.80 |
| NL | never_reached | 60.00 | 100.22 |
| FL | never_reached | 60.00 | 44.48 |
| NC | success | 11.20 | 1.78 |
| FC | never_reached | 60.00 | 64.51 |
| ML | never_reached | 60.00 | 147.81 |
| OUT | never_reached | 60.00 | 109.09 |

Config: [old_wrist_plus90.yaml](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90.yaml) (SHA-256 `939075f8d1fcc86a1df3c0c976abe23e3e525d4013471a5f4902e1be3491e564`). Full log: [old_wrist_plus90-console.txt](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90-console.txt). Machine-readable result: [old_wrist_plus90.result.json](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90.result.json). This is a configuration experiment, not a real-arm calibration validation.


### Trial without a validated score: old_no_pan_adaptation

evaluator exited 1. No success-rate claim is made. Log: [old_no_pan_adaptation-console.txt](docs/reports/2026-10-26-real-arm-calibration/old_no_pan_adaptation-console.txt).


## 10. Publication checkpoint — experiments paused

The user paused automatic experimentation and requested that this report and the completed implementation be
committed and pushed. No experiment was resumed after that pause. The simulator evaluator, policy server and
viewer have exited; the task-created idle teleop container was stopped. No real arm was connected or moved.

### Completed outcomes

The five-start, 180-second tests placed **2/5** with either USD: MC and MR succeeded; FR, NL and FL never reached.
The matched nine-zone milestone screen produced the following results at root seed 1984. All three completed
records have identical start IDs, zones and policy seeds. Runtime transforms and camera inputs were checked.

| Old-USD setting | Reached | Grasped | Confirmed placements | Successful zones |
|---|---:|---:|---:|---|
| Existing transform (control) | 2/9 | 1/9 | 1/9 | MR |
| Wrist model offset -90 degrees | 4/9 | 2/9 | 2/9 | MC, MR |
| Wrist model offset +90 degrees | 3/9 | 2/9 | 2/9 | MR, NC |

The -90-degree candidate newly reached FL and NC but missed both grasps. The +90-degree candidate placed NC at
11.20 simulated seconds, while MC missed its progress deadline. Neither sign is established as the correct
physical calibration. The changes trade success between locations; a single root and short milestone deadlines
are insufficient to promote either offset. In particular, the full-length control placed MC at 40.30 seconds,
after the screening protocol's 30-second grasp deadline. Full-length and independent-seed confirmation remain
**not run**. No experimental offset, fitted range, pan change or asset choice was promoted to a default.

### Startup failure and unrun candidates

The first no-pan-adaptation attempt exited **1** during simulator startup with a segmentation fault. Its stack
runs through `libneuray.so`, `librtx.mdltranslator.plugin.so` and RTX material initialization. No episode completed,
so it has **no task score**, not 0/9. The cause was not established, and no retry was made after the pause.
Its full startup log and terminal process record are archived below.

The matched nine-zone `ARM_new.usd` control, fitted range candidates, combined range/wrist candidates,
full-180-second confirmations and independent-seed replications were prepared but **not executed**. The second
seed launcher was checked only in dry-run mode; that is not an experiment result. The prior five-start
`ARM_new.usd` result remains valid and is distinct from the unrun nine-zone control.

### Portable evidence

The completed configuration and result files are archived in
[docs/reports/2026-10-26-real-arm-calibration](docs/reports/2026-10-26-real-arm-calibration/manifest.json).
The manifest records file SHA-256 hashes and the experiment/publication base commits. Archived console text has LF line endings and trailing spaces removed; original console SHA-256 values are also retained. Original temporary paths
inside terminal records are retained as provenance; adjacent archived files contain the corresponding inputs
and output. No model weights, datasets or camera recordings were copied into Git.

| Trial | Configuration | Terminal record | Console output |
|---|---|---|---|
| Old-USD control | [YAML](docs/reports/2026-10-26-real-arm-calibration/old_control.yaml) | [JSON](docs/reports/2026-10-26-real-arm-calibration/old_control.result.json) | [Log](docs/reports/2026-10-26-real-arm-calibration/old_control-console.txt) |
| Wrist -90 | [YAML](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90.yaml) | [JSON](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90.result.json) | [Log](docs/reports/2026-10-26-real-arm-calibration/old_wrist_minus90-console.txt) |
| Wrist +90 | [YAML](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90.yaml) | [JSON](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90.result.json) | [Log](docs/reports/2026-10-26-real-arm-calibration/old_wrist_plus90-console.txt) |
| No-pan startup failure | [YAML](docs/reports/2026-10-26-real-arm-calibration/old_no_pan_adaptation.yaml) | [JSON](docs/reports/2026-10-26-real-arm-calibration/old_no_pan_adaptation.result.json) | [Log](docs/reports/2026-10-26-real-arm-calibration/old_no_pan_adaptation-console.txt) |

The archive also contains the two full-length five-start console logs/configs, the exact two-camera manifest,
local action quantiles, fitted transform parameters and computed output bounds. Reproduction uses the canonical
`policies/molmoact2/client.py` with the archived `front_wrist.policy.toml` in a temporary `molmoact2` policy folder,
mounted/copied into teleop and selected through `TELEOP_POLICIES_DIR`. Run `./so101 eval <archived YAML> --smoke-test`
with the README teleop container and the matching checkpoint. The physical calibration cache is not needed for
these simulator trials.

Remote main advanced to `7da176c` during this work, adding a reach-envelope evaluation set. That commit is
preserved in the publication history. The newly added feasible set was **not used in these experiments**; their
scores refer to the original `pick_place_v1_seed1984.json` starts and cannot be transferred to the new set.


### Final checks on the publication base

Repeated verification after incorporating remote base `7da176c`: **30 targeted tests passed**. The four simulator
suite per-test status files still have an **empty before/after diff** (108 passed, 4 existing failures,
14 skipped). The full suite reports **518 passed, 89 failed, 53 skipped, 14 collection errors**, with an **empty
diff of FAILED/ERROR sets** against the recorded baseline. The full suite therefore remains non-green for the
same pre-existing failures; no new failed test or collection error was introduced.

[Verification metadata](docs/reports/2026-10-26-real-arm-calibration/publication-validation.json),
[targeted output](docs/reports/2026-10-26-real-arm-calibration/publication-targeted.txt),
[simulator baseline](docs/reports/2026-10-26-real-arm-calibration/sim-before.txt),
[simulator publication statuses](docs/reports/2026-10-26-real-arm-calibration/sim-publication.txt),
[baseline FAILED/ERROR set](docs/reports/2026-10-26-real-arm-calibration/full-failures-before.txt), and
[publication FAILED/ERROR set](docs/reports/2026-10-26-real-arm-calibration/full-failures-publication.txt)
are archived for review. All relative report links, evidence-file hashes, copied config fingerprints and the
three completed nine-zone runtime protocols/transforms/seed schedules were checked. Simulator source, mapping,
limits, rest-pose settings and Molmo client code remain unchanged by this implementation.
