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
