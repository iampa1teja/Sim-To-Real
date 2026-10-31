# Joint-angle check — 2026-10-27

**Physical comparison is incomplete. No physical joint-offset candidate is established or applied.** The separate software mapping/read-back check passes all 468 stored states. The requested new operator-started teleoperation capture is not present yet; physical base tilt, link readings and pan reference are also missing.

The operator selected ordinary `lerobot_agent` teleoperation instead of the phone-inclinometer procedure, then stopped the run and required subsequent agent launches to be manual. The observer and offline tool issue no motor commands; the explicitly requested ordinary teleoperation run used the existing leader-to-follower control loop. No further teleoperation launch or serial connection is part of this offline follow-up.

## Task 1 — Current tooling and software result

Software commit: `5f256b83ae69d8504404f8628af170648955103b` on `calibration-fixes`. The fetched remote branch contains the all-six-joint start sync and no-wrist-re-zero fix in `pick_place_agent`. The generic `lerobot_agent` still contains its older episode-start wrist execution re-zero; this investigation does not modify that behavior.

`scripts/joint_angle_check.py` watches or summarizes passive telemetry and offers `--physx-check` for separate simulation-only replay. Each live row stores six follower `.pos` values, their production-mapped/clamped angles, actual PhysX joint angles, the previous sim target, timestamps and limit flags. The comparison is taken before the next existing teleoperation command. The observer adds no motion path, torque command or calibration call.

Thirty-frame, non-overlapping blocks are rejected when **any of six follower standard deviations exceeds 0.3 units**. Fourteen of fifteen complete blocks in the stopped log are stable, but their medians represent one repeated pose. Eighteen trailing frames do not form another 30-frame block. These blocks were not operator-keyed pose captures.

### Part A — Production mapping into PhysX

The offline worker has no robot devices mounted. It uses the actual `LeRobotSO101Interface.get_raw_actions_tensor`, `get_mapped_actions_vectorized`, `synced_start_pose`, and `apply_runtime_joint_limits` functions used by recorder start synchronization. It writes a separate articulation and reads `root_physx_view.get_dof_positions()`, rather than echoing a target or cached joint buffer.

**468/468 states pass the 0.05° read-back gate; maximum difference = 0.000000°.** After Isaac initialization, simulation time remains `0.033333335072` seconds before and after all checks: **zero physics steps during state checks**. Runtime limits are independently read back from PhysX and match the shared JSON limits within 0.00005°. The JSON contains every per-state/per-joint result.

The stopped log is suitable as a collection of numerical mapping inputs. This replay does not validate which real arm supplied those inputs, or establish real physical angles.

First stored state (gripper is logged for schema/stability checks only):

| Joint | Follower units | Mapped ° | PhysX read-back ° | Difference ° |
| --- | ---: | ---: | ---: | ---: |
| shoulder_pan | 0.074460 | 1.845703 | 1.845703 | 0.000000 |
| shoulder_lift | -99.919838 | -106.875000 | -106.875000 | 0.000000 |
| elbow_flex | 97.491043 | 90.791016 | 90.791016 | 0.000000 |
| wrist_flex | 72.252861 | 76.464844 | 76.464844 | 0.000000 |
| wrist_roll | -4.029304 | -7.294922 | -7.294922 | 0.000000 |
| gripper | 0.542373 | -9.296878 | -9.296878 | 0.000000 |

Saturation in the stopped log makes its steady tracking gaps unsuitable as physical offsets:

| Joint | States at a runtime limit / 468 | Median sim minus mapped follower ° (diagnostic only) |
| --- | ---: | ---: |
| shoulder_pan | 459 | +0.000000 |
| shoulder_lift | 0 | -1.320631 |
| elbow_flex | 442 | +192.596055 |
| wrist_flex | 445 | -0.248230 |
| wrist_roll | 0 | +87.962361 |

### Part B — Reference surfaces and missing physical readings

Expected values are gravity-horizontal longitudinal elevations in the configured world frame. The direction is proximal to distal; positive means the distal end rises. The folded angle is in [-90°, +90°], so vertical positions are singular for angle inversion. The sim base/table orientation is retained exactly as configured. A measured real base orientation is needed before assigning any discrepancy to joints.

| Surface | Selected mesh under `/so101_new_calib` | Link-local datum | Phone placement |
| --- | --- | --- | --- |
| upper_arm | `/upper_arm/visuals/upper_arm_so101_v1/mesh` | normal `[0.0, 1.0, 0.0]`, plane offset `0.012` m; longitudinal `[-1.0, 0.0, 0.0]` | Broad flat printed cheek between shoulder and elbow; place the phone long edge toward the elbow, avoiding raised screws and the round servo horn. |
| forearm | `/lower_arm/visuals/under_arm_so101_v1/mesh` | normal `[0.0, -1.0, 0.0]`, plane offset `0.012` m; longitudinal `[-1.0, 0.0, 0.0]` | Broad flat printed cheek between elbow and wrist; place the phone long edge from elbow toward wrist, avoiding the motor and raised fasteners. |
| housing | `/gripper/visuals/wrist_roll_follower_so101_v1/mesh` | normal `[1.0, 0.0, 0.0]`, plane offset `-0.015` m; longitudinal `[0.0, 0.0, -1.0]` | Flat stationary printed housing wall opposite the moving-jaw hinge, behind the fingers; place the long edge from wrist toward fingertips. For roll, turn the phone edge across the housing toward the moving-jaw hinge. Do not use the movable jaw, finger pad or a protruding part. |

The selected planes were inspected directly in the immutable USD mesh and transformed into their rigid-link frames; the geometry evidence retains plane areas, triangle indices and local bounds. Parallel opposite cheeks have the same longitudinal elevation if the phone is aligned identically. A phone must seat on a genuinely flat surface; physical access, seating and correspondence have not been confirmed. No phone/fixture angle is assumed.

| Surface | Expected first-frame world pitch ° | Expected last-frame world pitch ° | Measured world pitch |
| --- | ---: | ---: | --- |
| upper_arm | -17.931106 | 43.367335 | Not supplied |
| forearm | 17.139889 | 53.363350 | Not supplied |
| housing | -59.324412 | -51.410636 | Not supplied |

Zero of 468 states have a housing longitudinal/roll axis within 5° of horizontal. Therefore the requested horizontal-axis roll check has no qualifying data. The transverse gravity tilt alone also has a branch ambiguity; it is not treated as an independently measured full roll angle.

Candidate sign convention: an offset would be **added to the current mapped USD angle**. In a verified planar inclinometer procedure, base-corrected upper-arm tilt error determines lift; the lower-arm minus upper-arm error determines elbow; housing minus lower-arm error determines wrist flex, each converted using the corresponding joint-axis sign from FK. No such measured errors or physical sign checks are available here, so no numerical inversion is claimed. A joint can be CONSISTENT only with a max-minus-min spread <= 1° across at least five distinct accepted physical poses; repeated encoder windows cannot satisfy that gate.

The requested rest / three lift / three elbow / three flex / five horizontal-axis roll / mixed-pose schedule, exact manually held targets within two follower units, keypress sampling, measured base tilt, per-pose phone readings and cumulative-angle inversion remain unperformed under the selected teleoperation method. The current script does not implement that replacement physical measurement session.

## Connection, torque and port evidence

Installed LeRobot `SO101Follower.connect(calibrate=False)` still calls `configure()`. Its `torque_disabled()` context unconditionally calls `enable_torque()` on exit, and configuration writes motor modes/gains. The read-only helper skips configuration and blocks motion, torque-enable, calibration and other motor writes. The only allowed Feetech writes are `Torque_Enable=0` and the `Lock=0` write used by `disable_torque()`. Calibration is disabled at connect; all six torque registers are then read back.

The operator confirmed support before the read-only preflight. All six `Torque_Enable` values were **0** after connection and at finalization. A later read-only motor-fingerprint check also read all six as 0. These observations are timestamped evidence from those checks, not a claim about an unobserved later manual run.

`rpd_robot.json` SHA-256 before and after: `86c6eee0eb97435082d5d6b820dafc3c4ad774b424ad84105c32cad96365be03`. The leader file also remains `baaf77e9fc25d93b6d5d3710bfd899be77cb9865b89d80340c12ef2c33f10518`. No calibration file was accepted, regenerated or written. `--no_calibration` in ordinary teleoperation disables calibration calls/writes; it does not make ordinary teleoperation torque-off or motion-free.

The motor-fingerprint evidence on `/dev/ttyACM0` matches the values saved in `rpd_robot.json` rather than `rpd_teleop.json`. This identifies a stored-calibration match, not an independently confirmed physical arm label. Port-role provenance for the stopped run was questioned, so its samples are excluded from physical candidates. The operator subsequently specified **leader `/dev/ttyACM0`, follower `/dev/ttyACM1`**, and those exact values are in `docker/env`; no automatic USB role reassignment is active. Only one USB serial device was visible at the last offline inspection, and the requested new `teleop-manual.jsonl` has not appeared.

## Task 2 — Candidates, tests and invariants

| Joint | Offset ° | Spread ° | Accepted physical poses | Status |
| --- | --- | --- | ---: | --- |
| shoulder_pan | Not measured | Not measured | 0 | NOT MEASURED |
| shoulder_lift | Not measured | Not measured | 0 | UNRESOLVED |
| elbow_flex | Not measured | Not measured | 0 | UNRESOLVED |
| wrist_flex | Not measured | Not measured | 0 | UNRESOLVED |
| wrist_roll | Not measured | Not measured | 0 | UNRESOLVED |

`candidates.json` SHA-256: `00d340519f25d4c51fb23dc914a53e28b69e084f554696f902d1af37d20c7516`. All numerical physical offset/spread fields are JSON `null`, never guessed zero. Gripper contact and angle calibration are out of scope.

| Verification | Result |
| --- | --- |
| Host targeted tests | 24 passed, 2 skipped; host lacks Torch and installed LeRobot |
| Container targeted joint/mapping/limit tests | 50 passed |
| Container full suite | 825 passed, 7 failed, 27 skipped, 279 subtests passed |
| Previous container baseline | 799 passed, same 7 failures, 27 skipped |
| Introduced failures | 0; failure IDs compared exactly |
| PhysX mapping/read-back | 468/468 pass, maximum 0.000000°, zero physics steps during checks |
| Protected input recheck | 1022 original inputs match their saved hashes; zero changed |
| Images/camera work | No new images, image processing, tags or camera-value changes |

Tests include a fake robot that raises on motion, torque-enable and calibration calls; the actual installed LeRobot connect method is also executed against fake motors. They verify torque-off read-back, hash preservation on exceptions, rejection of calibration mismatch, the six-joint stability gate, limit flags, 0.05° comparison tolerance, actual PhysX getter use instead of target/cache, and refusal to promote a constant tracking error into a physical offset. No hardware is opened by tests.

The separate CPU PhysX worker initially completed calculation but blocked during full Isaac 5.1 stage teardown. The final run uses the installed documented `SimulationApp.close(wait_for_replicator=False, skip_cleanup=True)` API after writing its JSON; the final worker exits successfully. It owns no hardware or image output.

Unchanged full-suite failures (details in the raw test log):

- `FAILED tests/test_docker_training.py::test_existing_entrypoints_route_to_docker_without_starting_training[policies/groot_n17/train/train_dit.sh-dit]`
- `FAILED tests/test_eval_script.py::test_bare_config_name_is_looked_up_in_configs_eval`
- `FAILED tests/test_launch_pick_place_n17.py::WrapperTests::test_official_flags_and_selected_config_are_forwarded_unchanged`
- `FAILED tests/test_launch_pick_place_n17.py::WrapperTests::test_runpy_entry_adds_project_helper_path_and_restores_it`
- `FAILED tests/test_lerobot_eval_rollout.py::EvaluatorRolloutTests::test_process_exit_without_finally_retains_completed_episode`
- `SUBFAILED(error='RuntimeError') tests/test_launch_pick_place_n17.py::WrapperTests::test_training_exception_and_system_exit_are_preserved_and_patch_restored`
- `SUBFAILED(error='SystemExit') tests/test_launch_pick_place_n17.py::WrapperTests::test_training_exception_and_system_exit_are_preserved_and_patch_restored`

## Evidence SHA-256

Raw telemetry and all detailed evidence are unversioned under `outputs/calibration-evidence/2026-10-27-joint-angle-check/`. Original datasets, USD assets, hardware calibration files, `real_setup.json`, its joint mapping, and camera values are unchanged. The source protection check covers the saved original dataset/asset/calibration manifest and records its own baseline digest.

| Evidence file | SHA-256 |
| --- | --- |
| `connection_preflight.json` | `9f270f18a1c21ff352bc1a4c5573e8b30b86f51aa93247283ce330cd57e3ba7e` |
| `port_identification.json` | `058034e1d38b2ca29aefb70be7f11dee01283ba042e2cfcfc41a7ded3d81e6ce` |
| `operator_decisions.json` | `aa09bea08f2038ef3ff9a7fbadbe25006a47ccc7bef4ba93ab5e3976fa60b624` |
| `teleop.jsonl` | `bad771f5a06f71c321548fc0d5ba99b22da4dce71e82a56840fa4e9929ce847b` |
| `stopped_teleop_summary.json` | `0329202b1ff12600a3a3fc55c1881e271684109c060d7ed99f7e9df6befecb23` |
| `physx_zero_step.json` | `7ffa2adec5b3b4769429be2aa55e150f68ec75fde63a6b256106ac794e443cc1` |
| `physx-runtime.log` | `a3043b75484fbd83e5ccee601dd6330d4f457ff7f068b9795e930cea514ebafa` |
| `host_pytest.log` | `eb5cb3ec77f630013c9f6d267a3bc44f7c6dac8b849c57d36fc9a759c7072c46` |
| `container_pytest.log` | `13e02c46762de925847c67de45184e90dcdbdec3ee4ef2eaeab9cb02641bcc53` |
| `test_comparison.json` | `c3dc84392d60f6c0d44f1ff63dadcb02961a7aefdba6a91adeb34d5fe4dcfe69` |
| `protected_before.json` | `b3247f62258748bf7c74d1e4cb0bf30a819e39f81770ac5920c4aaa2ece4738a` |
| `protected_after.json` | `b3247f62258748bf7c74d1e4cb0bf30a819e39f81770ac5920c4aaa2ece4738a` |
| `original_inputs_recheck.json` | `0be412e1a71522a860d007d583e60334876f5d69ef3088c2814ffcc181a91ff9` |
| `surface_geometry.json` | `80a2f766f149899185184cac90356db4861389549bb1377e03e37be887e32db2` |
| `inspect_surfaces.py` | `d544b00e95fb5f7e0cbe1d9a797e6bff849d738972bf5f2368f8e088282e4156` |

## Remaining verification

- A new operator-started capture with confirmed physical leader/follower identity and adequate unsaturated coverage.
- Independent base tilt and link inclinometer readings if physical joint-zero offsets remain the objective.
- At least five distinct accepted physical poses for each offset, including five roll poses with the roll axis horizontal.
- An independently supplied base-forward reference for pan; absent that, pan remains NOT MEASURED.
- Physical confirmation of the reference faces and measurement signs.

No candidate has been applied. Teleoperation launch remains under operator control.

## Follow-up — decomposing the large gaps

An offline re-read of the preserved `teleop.jsonl` (SHA-256
`bad771f5a06f71c321548fc0d5ba99b22da4dce71e82a56840fa4e9929ce847b`)
separates the existing sim target from the observed follower. No new hardware
run or calibration change was made. The first frame has no previous target;
target statistics below use the remaining 467 rows. Each column is its own
median, so the medians need not add exactly.

| Joint | Median sim − mapped follower ° | Median previous target − mapped follower ° | Median sim − previous target ° |
| --- | ---: | ---: | ---: |
| elbow_flex | +192.596055 | +193.251965 | −0.655910 |
| wrist_roll | +87.962361 | +87.956178 | +0.006183 |

The sim followed its targets closely. Most of the discrepancy already existed
between those targets and the follower encoder reading. In particular, the
elbow's mapped follower ranged from −100.000007° to +90.878891°, while the sim
ranged from +36.599671° to +92.600747°. In 442/468 frames the mapped elbow was
at a runtime limit; the dominant comparison spans opposite ends of the allowed
range. Clamping hides how far the raw mapped reading might lie beyond that
bound. **Clamping alone cannot explain why sim and real received different
effective angles when both use the same normalized command and mapping.**
The log lacks the previous raw leader command, unclipped real degrees, and
loaded-file hashes needed to distinguish a role/calibration mismatch, a real
tracking problem or an encoder branch. The disputed port-role provenance is an
additional reason to withhold a physical offset. A +192.6° linear gap remains
+192.6°: it must be flagged for branch investigation, never converted to −167.4°
by modulo arithmetic. The separate 468/468 zero-step test proves the numerical
mapping can be written/read through PhysX; it does not resolve this live mismatch.

For wrist roll, the generic `lerobot_agent` still computes
`sim_wrist_start + mapped_leader_roll - leader_wrist_start`. That introduces a
session/start-dependent constant relative to absolute follower mapping. The
trace's +87.956° **target−follower** median and +0.006° sim-tracking median are
consistent with that mechanism. They do not prove an independently measured
physical +90° zero error. The original run did not record the leader start
angle, so its exact contribution cannot be recovered. Current setup's
`wrist_roll_zero_offset_degrees` is 0°; the separate
`wrist_roll_model_offset=-90` belongs to policy inference and is not executed
by ordinary leader teleop. A near-90° number by itself cannot identify a policy
frame offset. `pick_place_agent` already uses absolute wrist targets; this
tooling preserves each agent's existing execution behavior and labels it in
the provenance header.

The [new calibration-check run sheet](../../09-calibration-check.md) records
unclipped commanded/real degrees, actual PhysX degrees, previous clamped target,
all three linear gaps, limit/static flags, and selected file hashes. Its
still/unsaturated statistics separate tracking from a persistent mapping gap.
Encoder-based checks still cannot detect a zero-pose error made during
calibration: real degrees use the same encoder calibration as sim. The
operator-matched torque-off pose mode supplies the independent reference;
it prints candidates and applies nothing. Fresh per-joint average gaps and
verdicts remain pending an operator-started run.
