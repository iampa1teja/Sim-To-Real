# 02 — Make your own digital twin

[Guide hub](../README.md) · Previous: [Setup](01-setup.md) · Next: [Task](03-define-a-task.md)

## What MyRoom loads

[MyRoomEnvCfg](../source/sim_to_real_so101/tasks/my_room_env_cfg.py) subclasses the
base SO-101 configuration directly. It is a sibling of the original LightStudio
task, not a subclass of it. It loads the packaged
[digital_twin.usdz](../source/sim_to_real_so101/assets/usd/digital_twin.usdz) through
[digital_twin_room.usda](../source/sim_to_real_so101/assets/usd/digital_twin_room.usda),
which excludes the preview robot/cameras. Isaac Lab then owns the live robot and
sensors. This avoids duplicate articulations.

[real_setup.json](../source/sim_to_real_so101/assets/real_setup.json) is the shared
parameter source; [real_setup.py](../source/sim_to_real_so101/assets/real_setup.py)
turns it into transforms, appearance and lights. Values are a fit for the author's
room, **not physical dimensions or calibration valid for your room**.

| Configuration | Meaning and coordinate convention |
| --- | --- |
| `robot.position`, `yaw_degrees`, `table_normal` | World placement, heading and table tilt; `robot_rotation()` combines yaw and tilt |
| `robot.joint_positions` | Initial USD joint pose in radians; not a universal home pose |
| `realsense.position`, `euler_degrees` | World camera pose; USD/OpenGL, −Z forward and +Y up; XYZ Euler degrees |
| `realsense.target` | Used by `camera_rotation()` only when Euler angles are absent |
| `realsense.calibration` | Measured `fx`, `fy`, `cx`, `cy`, distortion `coeffs` and provenance |
| `realsense.width/height`, apertures, focal length | Sensor projection/resolution; values must agree with the selected real stream |
| `wrist.position`, `euler_degrees` | Camera mount relative to the gripper; current mount is an approximation |
| `appearance`, `lighting`, `object_materials` | Fitted textures, roughness, robot/object colours and lights |
| `sim_joint_mapping` | Optional per-joint normalized-command to simulation-angle mapping |
| `sim_joint_mapping_source` | Calibration provenance; explanatory metadata, not runtime calibration loading |

Live RGB scene entities are `camera_realsense_rgb` and `camera_wrist_cam`.
The aligned metric-depth entity is `realsense_depth`, deliberately not a
`camera_*` recording entry. The visual group exposes `rgb_realsense_rgb`,
`rgb_wrist_cam`, their `depth_*` and `instance_id_seg_*` counterparts. The USD
camera display names are `realsense_camera-rgb`, `realsense_camera-depth` and
`wrist_cam`. The viewer selects the calibrated external sensor at startup.

## Measure, update, compare

1. **Measure your scene.** Record table surface/normal, robot base position and
   orientation, external-camera pose, wrist mount, cube and container geometry.
   Keep a consistent world frame and units. Base origin is not necessarily at
   tabletop height. The live reset uses `table_plane()` in
   [gripper_geometry.py](../source/sim_to_real_so101/utils/gripper_geometry.py).
   Pick-place box placement also has setup-specific `_TABLE_POINT`,
   `_TABLE_RIGHT`, `_TABLE_FRONT_EDGE`, `WHITE_BOX_RIGHT_OFFSET` and geometry
   constants in [pick_place_env_cfg.py](../source/sim_to_real_so101/tasks/pick_place_env_cfg.py).
   Updating only the JSON does not update those measurements.
2. **Measure intrinsics at the actual stream resolution.** Obtain the RGB
   intrinsic matrix from the device calibration/API or a calibration procedure.
   Put your `fx`, `fy`, `cx`, `cy` and coefficients in `realsense.calibration`.
   [calibrated_camera.py](../source/sim_to_real_so101/utils/calibrated_camera.py)
   applies the RTX OpenCV lens model. Do not copy the packaged camera's numbers
   or silently scale resolution without updating intrinsics. This repository
   has no tracked camera-calibration CLI.
3. **Edit your setup JSON and task constants.** Use the fields above, preserving
   units/conventions. Log how each value was measured. The current external
   extrinsics include visual fitting and the light intensities are appearance
   estimates, not surveyed geometry or photometry. Restart the process after
   editing because `SETUP` is loaded at import time.
4. **Use your own room USD when needed.** Put the room and all referenced assets
   somewhere mounted inside the container. Custom rooms disable embedded prims
   named `Robot` and camera prims. The packaged room's material/light edits are
   skipped for custom overrides; supply suitable collision geometry/materials
   and lighting yourself. A custom USD does not automatically recalibrate the
   robot, cameras, table detector or pick-place constants.

**Teleop container**, example custom-room selection (replace with an existing
container-visible file):

```bash
setenv ROOM_USD_PATH '<container_path_to_your_room.usd>'
zero_agent --task Lerobot-So101-Teleop-MyRoom --num_steps 300
```

The placeholder must name your existing mounted USD, not a shipped asset. To return to the
packaged room, run in the **teleop container**:

```bash
setenv -d ROOM_USD_PATH
```

5. **Compare live views.** With both arms calibrated and camera variables set,
   use the recorder without dataset arguments. This opens four panes without
   saving demonstrations. Inspect base alignment, arm silhouette, box corners
   and camera framing across several poses, not just one screenshot.

**Teleop container**:

```bash
pick_place_agent --enable_real_follower --control_input terminal
```

Open `http://localhost:8765/` in the host browser. Sim external/wrist are the top
row; real external/gripper are the bottom row. A sim-only run has no real feeds.
The leader still drives motion; starting without dataset arguments is not a
passive hardware inspection.

## Calibrate the simulation joint mapping

The physical leader/follower values use LeRobot normalization. The adapter
[lerobot_interface.py](../source/sim_to_real_so101/utils/lerobot_interface.py)
maps the first five joints from −100..100 and the gripper from 0..100 to configured
degree ranges, clamps to the existing USD mapping limits, then converts to radians.
It also implements the inverse mapping for observations. Raw leader commands sent
to the physical follower are not changed by the simulation override.

Calibrate the follower first and inspect its matching ID JSON under the shared
calibration directory. For the current shoulder-pan approach, derive angular
bounds from your encoder `range_min`, `range_max`, zero reference and counts per
turn: `(count - zero_count) * 360 / encoder_counts_per_turn`. The current values
and provenance are in `sim_joint_mapping.shoulder_pan` and
`sim_joint_mapping_source`; the stored zero/counts are not universal constants
for another actuator/calibration. Confirm direction and zero against several
physical poses before replacing `joint_min`/`joint_max`. Do not change collision
geometry to hide an actuation mismatch. Recheck limits after calibration changes.

## Keep scene versions separate

Keep Fabric enabled: the recorded local motion check reported stale wrist-camera
rendering with `--disable_fabric`. The flag exists, but is unsuitable for this
camera-training workflow until validated on your Isaac build.

Re-aiming a real or simulated camera changes the visual distribution. Treat old
data as belonging to the previous scene version, not as aligned recordings for
the new scene. Use a new dataset root/repo ID for each calibrated scene version;
record its code revision and calibration provenance. Dataset schema checks do
not detect a camera that physically moved while keeping the same resolution.

The local `env setup/` builder and validation tools mentioned by older notes are
not tracked in this fork. There is no reproducible room-rebuild command shipped
here; see [open questions](troubleshooting.md#open-questions-and-code-findings).
