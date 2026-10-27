# Camera, gripper and recording audit — 2026-10-27

Work is on `calibration-fixes`; source datasets and USD assets are protected. Calibration gates concern numerical alignment, not whether a recorded video succeeded.

## Part 1 — static room-camera fit: gate not met

No camera values were applied. Used 21 episodes: 11 fit, 10 held out. Static landmarks only: five box corners, up to four base-foot corners, and two back wall/brown-boundary line samples. No arm points entered the camera objective. Dark right-side box vertex and occluded/ambiguous contours were excluded before the split. RGB corner localisation and the small fixed viewpoint limit confidence; the fit is an image measurement, not a new hardware calibration.

| Fit | Fit median / max (px) | Held-out median / max (px) |
|---|---:|---:|
| Existing camera | 5.904 / 17.065 | 5.135 / 17.065 |
| Extrinsics, measured K fixed | 2.753 / 10.574 | 2.796 / 10.574 |
| Extrinsics + common focal scale | 2.727 / 9.269 | 3.018 / 9.304 |

Gate: held-out median ≤2 px. Both candidates miss it. Free focal was attempted only after the fixed-K fit missed; it drifted to 0.7273× measured focal, illustrating focal/distance ambiguity. Neither candidate is a replacement calibration. Old values remain intact in the setup and the before snapshot. The contact sheet labels the rejected candidate explicitly; production “after” equals “before”.

The room camera reports inverse Brown–Conrady coefficients `[0,0,0,0,0]`. `CalibratedPinholeCameraCfg` authors the RTX native OpenCV pinhole model, so these measured zero coefficients are represented exactly: distortion omission error **0 px**. Nonzero inverse coefficients would require conversion; the existing code rejects them rather than interpreting them as a forward model. This does not establish that the physical lens has no residual distortion beyond its reported calibration.

![Room camera comparisons](room_camera_contact_sheet.png)

Median signed edge displacements (real minus sim, pixels; normals for landmark edges):

| Edge | Before | Rejected candidate |
|---|---:|---:|
| front_bottom_real_minus_sim_px | 6.000 | 4.000 |
| back_edge_median_real_minus_sim_px | -3.500 | -4.000 |
| box_back_rim | 2.369 | -0.312 |
| box_front_bottom | -5.504 | -2.678 |
| box_left_vertical | 6.030 | -1.481 |
| box_left_top_depth | 8.160 | 0.101 |
| base_lower_span | 1.061 | -0.401 |

The joint check used five quiet, original-USD-in-limit follower frames, two coordinate-descent passes, offsets −10…+10° in 1° steps, with only pan/lift/elbow/flex varied. Because the camera gate failed, this diagnostic uses the unchanged source camera. Arm masks use plastic segmentation and filled components near the rendered arm, excluding the base; mask and camera uncertainty make these flags unsuitable as calibration estimates. No joint mapping was changed.

| Episode / frame | Pan / lift / elbow / flex best offsets (deg) | IoU before → after |
|---|---|---:|
| 2 / 142 | -10, -10, -10, -10 | 0.295 → 0.435 |
| 10 / 182 | +1, -1, +0, -2 | 0.617 → 0.636 |
| 11 / 115 | +2, +0, -3, -2 | 0.583 → 0.621 |
| 26 / 77 | +5, -2, +1, -2 | 0.421 → 0.639 |
| 32 / 44 | +6, -4, +0, +0 | 0.557 → 0.715 |

Offsets above 2° and solutions on the sweep boundary are flagged, not applied. The user's physical arm/box checks remain the established geometry reference.

![Joint offset IoU curves](joint_offset_iou.png)

Part 1 blocks only the camera image rerender in 5a, the paired measurement dataset in 5b, and the live cube guide in 6. Independent parts continue.

## Evidence files

Bulk data stay outside git. SHA-256 below fixes the exact evidence used:

| File | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/source_hashes_before.json` | `3097b681816a0852f44adbd2776452e395b1867915a4871358b090d4be5346e8` |
| `/tmp/so101-camera-gripper-recording/real_setup_before.json` | `c518cf0ac9509b83d51af944817e3d24162de20b843255e08165edee403790f5` |
| `/tmp/so101-camera-gripper-recording/camera_fit.json` | `0e71890d4d08cbf7d23c8ef7d70c0ef1e1edac386072dc7d49983d2da44f492e` |
| `/tmp/so101-camera-gripper-recording/camera_edges_bulk.json` | `671a7ce526c68e744e406c3a25424d2ec882e17c56b0bdfcdd84f4507c76875e` |
| `/tmp/so101-camera-gripper-recording/joint_check.json` | `578aee78bbba6a8ea5e4df697c36cddc874ba7c1879f21754aa6f714418f9048` |
| `/tmp/so101-camera-gripper-recording/tests_before_host.txt` | `b15c01bdc1dcf075ead4d299ad7a7c469f6e0ab9cea67314b004c2172757372b` |
| `/tmp/so101-camera-gripper-recording/tests_before_container.txt` | `4f6750fffc4dcf705fae011b3f641e3759cbe0f7dafe468746216f336948c7c2` |

## Baseline tests

Host (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, temporary analysis venv, excluding the Isaac/LeRobot codec integration file): 761 passed, 33 skipped, 5 reported failures including two failing subtests. Existing failures: three `test_launch_pick_place_n17` wrapper tests (missing `gr00t.data`) and `test_benchmark_progress_updates_stages_and_retains_terminal_milestone` (missing host `rerun`).

Teleop container: 772 passed, 27 skipped, 7 reported failures including two failing subtests. Existing failures: the `train_dit.sh` entrypoint storage-path test; bare config lookup under a read-only repo bind; the same three `test_launch_pick_place_n17` wrapper tests; and `test_process_exit_without_finally_retains_completed_episode` (bare child Python lacks NumPy). Full logs above contain tracebacks. Final failure-set comparison is recorded in Part 7.

## Part 2A — wrist roll: UNRESOLVED

Posed the simulator from **real follower observation.state**, without action-based episode re-zero correction, on 55 quiet frames from 55 episodes. The pool has 630 original-USD-in-limit quiet frames from 55 episodes. Quiet means all-six-joint maximum adjacent step ≤0.25° and five-frame excursion ≤0.5°. Selected roll extremes are −116.982° and +23.643°. The global +39.551° extreme has no frame satisfying the original all-joint in-limit/quiet gate; it is included in the limit audit, not mislabelled as a quiet roll calibration point.

Undistorted background-edge candidates were matched to projected static box edges, using FK of the gripper and the verified camera mount. A +1° roll render provides the rotation axis and camera orbit, so fitting is in joint degrees rather than assuming image rotation equals wrist rotation. Camera sensor pose metadata is stale when Isaac Lab's `update_latest_camera_pose` is false; analysis therefore used live body FK plus the mount and checked the CAD projection against the rendered silhouette. The corrected capture worker also enables current camera metadata.

The generous automatic matcher found **11 candidate frames/episodes**, median offset **−4.188°**, standard deviation **3.921°**, range **−12.780…+1.289°**. These include ambiguous edges; they are an upper bound on usable matches, not 11 independently verified correspondences. The narrower matcher finds only two clear matches. Neither set meets ≥15 cross-episode frames with spread about 1°. Background edges are absent from most still frames; wood grain, cube edges and specular highlights were not accepted as established static geometry.

The −4.19° trial is shown for diagnosis and **was not applied**. Its scatter does not establish any multiple-of-90° zero change. The room overlay provides a jaw-direction cross-check, but room camera error and overlapping silhouettes cannot validate a roughly 4° roll adjustment; it supplies no independent numeric confirmation. The physical arm-vs-USD roll zero remains unresolved. The existing sim-only wrist re-zero must therefore remain for future recordings.

![Wrist roll before/trial overlays](roll_contact_sheet.png)

![Roll spread and room cross-check](roll_spread_room_check.png)

## Part 2B — gripper: no contact-anchor change

Stillness was evaluated on **gripper only**. The requested episode 23/frame 187 is usable despite body motion: state **1.4237288 units**, five-frame jaw excursion **0.35156°**. Mid-open episode 18/frame 21 is 44.94915 units with 0° jaw excursion. Fully-open episode 30/frame 28 is 88.27119 units with 0.08789° excursion. No source frame reaches the calibrated zero-unit closed endpoint.

Real and simulated **fixed white jaw faces** were registered first (affine image fit; moving pixels excluded). Fixed-jaw IoUs are **0.951, 0.940, 0.942** for closed/mid/open. The real moving-jaw outer-edge image angles are **52.993°** and **88.766°** for closed and mid-open. The fully-open moving jaw leaves the wrist image and cannot supply an angle.

Direct moving-edge scans after this registration do **not** yield a consistent contact anchor. The closed diagnostic reaches a best angle match at −21.5° USD (implied unit-zero anchor −23.35°), already beyond the CAD contact proxy. The mid-open scan has a large residual (26.57° at its best accepted line) and an incompatible anchor. These are rejected image-fit outputs, not motor calibration measurements: a partial fixed-jaw silhouette permits nonphysical affine shear, and side faces/occlusion change the measured moving edge. High registration IoU alone does not validate a 3D angle. The lowest recorded frame also does not establish that the empty pads are touching.

Thus these images cannot defensibly decide between a true touching angle near −10° and −13°. The CAD proxy at −12.93098° remains a proxy. The previous −10.0000038° at unit 0 is retained explicitly with **unresolved** status, independent of the jaw's runtime lower limit. No offset was inferred from the jaw's moving body or forced to the proxy value.

![Fixed-jaw registration and moving jaw](gripper_registration_contact_sheet.png)

![Rejected moving-angle fits](gripper_angle_diagnostic.png)

Live capture needed to resolve contact:

1. Keep the wrist camera rigidly mounted, remove the cube, and put a contrasting background behind both fingertips. Use a pose where both fixed and moving white faces, their holes, and the touching pads are visible. Hold the arm supported and stationary.
2. Close the empty gripper slowly until the pads just touch, without squeezing further. Verify contact directly from the side; the lowest dataset value alone is not evidence of contact.
3. Hold for at least one second. Save a lossless wrist image and the **same-tick follower `get_observation()` six `.pos` values**, timestamp, follower ID, and calibration-file SHA-256. Save 30 consecutive observations/images so jaw-only stillness can be checked. Do not substitute leader actions.
4. Repeat at mid-open and fully open, and repeat empty contact three times. Take one room/side image confirming pad contact. Use fixed-jaw hole correspondences to recover a physical camera/plane registration; then fit the moving edge and require the same anchor across openings before changing it.

Additional bulk evidence:

| File | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/wrist_selection.json` | `8898c7bbd21361de25cb184b3bfb7b647a6503a763ea6fb623595d2bcc7e3a2f` |
| `/tmp/so101-camera-gripper-recording/roll_fit.json` | `8dcb61e2aa50b0e795e96dd3588d6cb65173e44cc55af19a9abdf2345f00363b` |
| `/tmp/so101-camera-gripper-recording/gripper_measurements.json` | `790c0402b631458c1f0a86961a880732eb0d366a9602852c7b37248c3d20c714` |
| `/tmp/so101-camera-gripper-recording/gripper_fit.json` | `34b37adcbf954b40ffeee1e25ae6c422ca84c32edacde8064c211b7155af68c2` |

## Part 3 — one shared mapping derivation

`utils/joint_calibration.py` now accepts and validates explicit persisted `wrist_roll_zero_offset_degrees`, `gripper_closed_degrees`, and `gripper_contact_units`. The gripper anchor is the USD angle **at the stated contact sample**, so a future measured nonzero contact-unit value does not get mistaken for the unit-zero angle. Encoder spans are retained. Joint limits are independent of all three anchors.

The unresolved roll fit retains offset **0°**. The unresolved contact fit retains **−10.000003814697266° at 0 units**. Both statuses and evidence references are saved in `sim_joint_mapping_source`; the CAD contact proxy is separately labelled. These are retained assumptions, not newly verified physical zeros.

Regenerated `sim_joint_mapping` with `scripts/update_sim_calibration.py` from the unchanged rpd_robot calibration inside the teleop image. No mapping numbers were hand edited. The generator preserves anchor/provenance fields, accepts explicit anchor overrides, and no longer derives contact from the USD jaw lower limit. `assets/real_setup.py` recomputes with all anchors and still rejects stale persisted mappings. Real rollout passes the same complete anchor dictionary into the shared function.

Molmo's **−90° policy convention is retained separately**. The unresolved arm fit cannot establish a replacement or independently prove −90° on the physical arm. There is no extra 90° in the shared mapping. Mapping midpoints still give wrist `arm_zero_degrees = −0.0439453125°`; a synthetic +90° arm-zero regression moves that policy zero to 89.9560546875° without changing its encoder span or Molmo option.

Validation: 43 host tests passed (21 subtests), including nonzero contact samples, roll-offset midpoint consistency, invalid anchors, generator provenance preservation and real-rollout entry points. A fresh teleop-image import passed the runtime stale-map check (`PART3_SHARED_SETUP_IMPORT_OK`).

| Evidence | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/part3-tests.txt` | `c84b94f19321f17dc22bf833a32dd20279b48839c79679ba855e94f9c5ce6d2a` |
| `/tmp/so101-camera-gripper-recording/part3-container-import.txt` | `233aacf17c929f913d25e81ee318eabf46fe5eba91bd61211ef6422af88be77b` |

## Part 4 — runtime limits

Both clamps now read `runtime_joint_limits` in `real_setup.json`: the interface's code clamp and the startup PhysX articulation limits. The installed Isaac Lab call is `Articulation.write_joint_position_limit_to_sim(limits, joint_ids=..., env_ids=...)`, confirmed in `/workspace/isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py:710`. Limits are radians at that API and degrees in JSON. A fresh Isaac environment read them back from `robot.data.joint_pos_limits`; all six match the code clamp within 0.00002°. Twelve full recorded poses at per-joint extrema pass the production mapping without clipping, maximum float32 conversion error 0.0000214°.

Bounds are the union of previous bounds and observed real extrema plus 1°; the jaw lower is 5° below the retained unit-zero anchor. The wrist roll remains **−160…+160°**, which already covers −117.070…+39.551°; it was not expanded to ±180°.

| Joint | Observed min / max (deg) | Runtime min / max (deg) |
|---|---:|---:|
| shoulder_pan | -111.88477 / 96.15234 | -112.88477 / 110.00000 |
| shoulder_lift | -106.69922 / 99.93164 | -107.69922 / 100.93164 |
| elbow_flex | -74.44336 / 92.37305 | -100.00000 / 93.37305 |
| wrist_flex | -19.16016 / 103.79883 | -95.00000 / 104.79883 |
| wrist_roll | -117.07031 / 39.55078 | -160.00000 / 160.00000 |
| gripper | -8.15430 / 104.43359 | -15.00000 / 105.43359 |

The base USD and its layers were not modified. Self-collision remains off. Checked the 12 recorded joint-extreme poses, configured start, and simultaneous all-lower/all-upper limit corners. Using conservative convex hulls of each authored collision mesh, FCL found **zero intersections** between the base and upper arm/distal links; minimum tested clearance is **5.229 mm**. The directly adjacent shoulder/base joint is excluded. These discrete checks do not prove clearance for every combined pose or intermediate trajectory.

![Extreme pose checks](joint_limit_extremes.png)

Tests cover observed-extrema coverage, the restricted wrist range, jaw anchor/limit separation, ordered radians passed to the actual API name, malformed limits and existing interface round trips. Host: 17 passed. Teleop container: 37 passed (8 subtests), including actual LeRobot interface round trips. Readback evidence:

| Evidence | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/runtime_limits_verified.json` | `00bbbdc89f065afee75f8345e5961f41d503c6f895eb94ab5ca168c214956ee4` |
| `/tmp/so101-camera-gripper-recording/base_clearance.json` | `57343e18cabc4d78074e0a79c7f2227aecf33c1bc95ed365a58e0dae35a43b3c` |
| `/tmp/so101-camera-gripper-recording/part4-host-tests.txt` | `743f4053f98d4adcfe4425e7dd10d69ebe01468bb9b733050b1f10788958a6ff` |
| `/tmp/so101-camera-gripper-recording/part4-container-tests.txt` | `55e284b3e2ee5a5e787bf92b1976d50301bef8baa065234f4926010651e2ed2c` |
| `/tmp/so101-camera-gripper-recording/collision_meshes.npz` | `60ffd4bedc951fd3d7cceec2dd41dc1fd830814803ca2864a8dbeb1dc371ddb8` |

## Part 5 — calibrated_A2, sources unchanged

Created **`datasets/pick_place_v1_calibrated_A2` directly from the ORIGINAL `pick_place_v1`**, never from calibrated_A. Method A applies `inverse_new_mapping(old_mapping(units))` to both actions and observation.state, without clamping. Metadata joint statistics are transformed too. Maximum preserved physical-angle error is **0.00000488311°** (float32 relabel rounding). The explicit roll/contact values retain the Part 3 mapping; A2 is a fresh original-source derivation with current provenance, not a second relabel.

A2 has **75 episodes / 16,448 frames**. All **150 original video files are byte-identical**. Fully decoded 16,448 frames in each camera, RGB 640×480. The official container `LeRobotDataset` reader reports 16,448 frames / 75 episodes and decodes both cameras at indices 0, 8,224 and 16,447. There are no failed video files in this decode check. Demonstration cube trajectories and source images remain unchanged.

Part 5a's camera image rerender is **skipped** because Part 1 missed its gate. Part 5b's entire `pick_place_v1_realpaired` measurement dataset is **skipped for the same dependency**, including cube detection, hand-checked detection error and held-phase FK. No substitute paired dataset or cube-placement measurement is claimed. The Molmo smoke config now points to A2.

Every file in the two source datasets, all four source hardware calibration files and all eight USD assets matches the before hash. These source dataset manifest digests use SHA-256 of the sorted path→file-SHA dictionary:

| Source | Files | Before digest | After digest |
|---|---:|---|---|
| pick_place_v1 | 379 | `361878a6d44e6c2a6aea2d33ea89b0624fe848e2a37f9d0b9deaf53ba5bda1b3` | `361878a6d44e6c2a6aea2d33ea89b0624fe848e2a37f9d0b9deaf53ba5bda1b3` |
| pick_place_v1-real | 249 | `7e8dfde4b3bc5a0d2647fe59063ef163a67ccacc3ff7751c346b43bf27e6fc05` | `7e8dfde4b3bc5a0d2647fe59063ef163a67ccacc3ff7751c346b43bf27e6fc05` |

Hardware calibration files (before = after):

| File | Matching SHA-256 |
|---|---|
| `/home/rpuser1/.cache/huggingface/lerobot/calibration/robots/so101_follower/orange_robot.json` | `54c01e5a649f7396b5d0b6c4d8b6e89a88b95b1305c82922ef1c42847e99a3ad` |
| `/home/rpuser1/.cache/huggingface/lerobot/calibration/robots/so101_follower/rpd_robot.json` | `86c6eee0eb97435082d5d6b820dafc3c4ad774b424ad84105c32cad96365be03` |
| `/home/rpuser1/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/orange_teleop.json` | `1e99f3fc1d9c15b11249393559f0661a17152ea6f1a3b45d37ecdf2af7eff6b3` |
| `/home/rpuser1/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/rpd_teleop.json` | `baaf77e9fc25d93b6d5d3710bfd899be77cb9865b89d80340c12ef2c33f10518` |

Evidence and all per-file hashes:

| File | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/a2_relabel_manifest.json` | `b92b8198ab1e56821c6a87134749fae3dcfdc4c0ae11cfdccc902608c8171de4` |
| `/tmp/so101-camera-gripper-recording/a2_lerobot_reader.json` | `58f011880fe01e3cec0dc415842e108be5dd722d2c56e3bef840a585c06fc2fc` |
| `/tmp/so101-camera-gripper-recording/a2_video_decode.json` | `5547b9de620f31cef1471a928493f0b331e496e04dd8de3f1eb66c6623eced0c` |
| `/tmp/so101-camera-gripper-recording/source_hashes_after.json` | `b5a3459eb6269e8f71fe91632b277b3bbc5d089f023fdb3b3088a75bc0a7b398` |
| `/tmp/so101-camera-gripper-recording/source_hash_verification.json` | `32d0f09e76b6bd6c5d61687f0bb25e31e5ca28501abdf9956d7acdb15179ddfa` |

## Part 6 — future recording alignment

At each episode start, read the real follower once, map its measured state with the production mapping/clamp, write the sim joint state and zero velocity, set its hold target, and refresh both images without stepping physics. Because Part 2A is unresolved, sync **pan/lift/elbow/flex/gripper** and preserve the current sim roll. Reset the leader wrist reference at each episode start and keep the sim-only wrist execution re-zero. When `wrist_roll_fit_status` becomes `verified`, the same path syncs all six joints and uses direct mapped leader targets.

Both dataset `action` fields now store the **same unchanged leader action**. During the unresolved fallback the executed simulated roll target can still differ from that nominal action; this exception is explicit in `meta/recording_alignment.json`. Removal of that execution fallback remains gated by a consistent physical roll fit. The leader dictionary sent to the real follower is never modified.

Recording roots must carry the exact new version-2 contract, including calibration/limits fingerprint and arm IDs. Nonempty legacy roots or changed contracts are refused **before opening serial hardware**. Markers are written only after successful dataset initialization. The guide now uses `pick_place_v2_start_synced` / `pick_place_paired_v2_start_synced`; derived real roots append `-real`. No new physical recording was attempted here. The live cube placement guide is skipped solely because Part 1 failed its gate.

Real rollout now checks the observed wrist against the configured **sim start roll through the shared mapping**, before inference/motion. It retains the separate all-joint first-policy-target/current-pose check. For this retained mapping, sim start roll is **-1.868052871°**, or **-1.013640559 raw units**. A future physical roll-zero offset shifts this raw reference consistently. This numerical check does not resolve the unknown physical roll zero. Molmo real rollout now defaults its separate model option to −90°, matching the sim smoke config; an explicit policy option still overrides it.

Relevant source lines (quoted from the implementation):

`source/sim_to_real_so101/utils/real_rollout.py:50`

```python
degrees = math.degrees(setup["robot"]["joint_positions"]["Wrist_Roll"])
bounds = mapping["wrist_roll"]
return (degrees - bounds["joint_min"]) / (bounds["joint_max"] - bounds["joint_min"]) * 200 - 100
```

`source/sim_to_real_so101/utils/real_rollout.py:89`

```python
if step == 0 and settings.expected_start_roll is not None:
    measured_roll = float(state[joint_order.index("wrist_roll.pos")])
    roll_gap = abs(measured_roll - settings.expected_start_roll)
    if roll_gap > settings.max_start_error:
        status = "start_too_far"
        detail = (f"wrist roll is {roll_gap:.1f} raw units from the mapped sim start "
                  f"({measured_roll:.3f} measured, {settings.expected_start_roll:.3f} expected; "
                  f"limit {settings.max_start_error})")
        break
```

`source/sim_to_real_so101/scripts/real_rollout.py:170`

```python
expected_roll = rr.sim_start_roll_units(args.joint_mapping)
header["sim_start_roll_units"] = expected_roll
header["mapping_parameters"] = rr.sim_mapping_parameters()
settings = rr.RolloutSettings(args.fps, args.episode_seconds, args.enable_motion, args.max_start_error,
                              expected_start_roll=expected_roll)
```


Validation: **40 host tests and 40 container tests passed** (21 subtests in each). The regression executes the actual recorder control-tick source with fake devices for resolved/unresolved roll, checks measured start pose and zero velocities, asserts image refresh precedes frame 0, proves action equality while the fallback target differs, and checks legacy/changed roots cannot resume. Rollout tests verify a mismatched wrist stops before inference/motion and a synthetic shared-zero change moves the start reference. Live hardware timing and pad contact remain unverified.

| Evidence | SHA-256 |
|---|---|
| `/tmp/so101-camera-gripper-recording/part6-tests.txt` | `6b22ffbb14a545d4fabd34af32c64be1ad2e39b108df543a384840e5a6f0a5c9` |
| `/tmp/so101-camera-gripper-recording/part6-container-tests.txt` | `ac03cbf699b6877c3dad43299d40ad23b5aa97f20f4751747c13a798c65ba883` |
