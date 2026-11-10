# Add a scene

[Guide index](../README.md) · [Add an arm](02-add-an-arm.md) · [Add a task](04-add-a-task.md)

A scene folder describes the room, anchors, fixed cameras, lighting and renderer/physics settings.
An arm/scene mount belongs under that scene's `mounts/`; arm cameras and calibration remain in the arm.
The generic scene builder discovers the folder without a scene-name branch in `src/`.

Run commands from the repository root after [getting started](01-getting-started.md).

## Create a working table scene

```bash
./s2r new scene guide_scene
./s2r check --static --arm so101 --scene guide_scene --task stack_two_cubes
```

The complete [scene template](../scenes/_template/scene.toml) supplies a metre-scaled table USD, one fixed
RGB camera, a dome light and table/mount anchors. No appearance fit, custom hook or mount file is needed.

```text
scenes/guide_scene/
  scene.toml
  usd/room.usda
  mounts/<arm>.toml       optional
  appearance.json        optional
  scene.py               optional
```

Inspect the loader and the absent-mount default on the CPU:

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
from core.arms.spec import load_arm_spec
from core.scenes import load_scene_spec, load_mount_spec, mount_pose

arm, _ = load_arm_spec("so101")
scene = load_scene_spec("guide_scene")
mount = load_mount_spec(scene, arm)
assert not mount.exists
print(scene.room_usd)
print(mount_pose(scene, mount, arm))
print(tuple(scene.cameras))
PY
```

`[scene].name` must match the folder. `room_usd`, optional `appearance`, and the optional hook file
must exist relative to it. `room_root` selects a subtree relative to the imported room root; leave it
empty for the template. Package referenced textures/assets in the scene folder.

For a run-specific room asset, set `S2R_ROOM_USD` before `./s2r`. It is read at load time and forwarded
to simulator commands. Absolute paths refer to the container filesystem; relative paths resolve from
the simulator working directory `/workspace/s2r`, unlike the manifest's scene-relative `room_usd`.
The override must name an existing file with the scene's expected anchors and geometry. A full `s2r check`
validates both the manifest's normal asset on the host and the overridden composition in the simulator.
`--static` alone checks the manifest asset, not a container-only override. `ROOM_USD_PATH` is no longer
accepted. Recapture benchmark geometry after intentionally changing the room asset.

## Define coherent anchors and contacts

Each `[scene.anchors.<name>]` defines a rigid frame through metre `position`, `yaw_degrees` and `up`.
Optional `right` supplies the zero-yaw right axis; position and orientation transforms use the same
resulting rigid basis. Keep raw survey vectors in `metadata` if they are provenance rather than a frame.

The built-in tasks use `table` for tabletop-relative placements and support geometry. The default mount
uses `mount`. A `workspace` anchor may declare increasing `xy_min`/`xy_max` bounds. Named anchors can
also be used as task placement/start frames. Keep these values in the coordinate system of the scene
as imported by the builder.

`prim_name` identifies the USD prim used for measured anchor geometry. A tabletop anchor should select
the actual top surface, not a closed slab whose lower vertices would contaminate the plane fit. The
template uses a planar visual `Tabletop` mesh with a separate solid child collider. This gives Newton
full-dimensional collision geometry while preserving an exact top plane and outline.

Set `contact_material = true` to bind the shared contact material before Newton imports the room.
Normally the builder binds enabled collider descendants of that anchor prim. If collision meshes live
elsewhere in the imported room, `contact_prim_pattern` can select them by a full match against their
prim names. Verify the pattern selects only the intended colliders. The meeting room's
[scene.toml](../scenes/rpd_meeting_room/scene.toml) demonstrates this separate visual/collision layout.

USD transforms and authored anchor coordinates must describe the same surface. Changing a mesh or
table transform after static geometry was cached requires `sim.scene.geometry.clear_geometry_cache(stage)`.
Topology/collision changes belong before model construction.

## Add an optional mount

Without `mounts/<arm>.toml`, the builder uses the `mount` anchor and the arm's default joint positions.
If no mount anchor exists, the bare arm pose is used. A mount file can select another anchor, add a
world-space translation correction, and override selected initial joints.

For the two-joint `guide_arm` from [the arm guide](02-add-an-arm.md), this is a complete optional file:

```bash
mkdir -p scenes/guide_scene/mounts
cat > scenes/guide_scene/mounts/guide_arm.toml <<'TOML'
[mount]
anchor = "mount"
translation_correction_m = [0.0, 0.0, 0.0]

[mount.start_joint_positions]
shoulder = 0.0
finger = 0.02

[mount.source]
method = "Template exercise; no physical mount survey"
TOML
```

Joint keys are device/manifest joint names; values are radians for revolute joints and metres for
prismatic joints. The anchor determines orientation; `translation_correction_m` adds in world axes,
not rotated anchor axes. Record the measurement/source behind a real correction. The complete measured
[SO101 meeting-room mount](../scenes/rpd_meeting_room/mounts/so101.toml) is an example, not a fit to copy
to another scene.

## Declare cameras, lighting and optional appearance

`[scene.cameras.<key>]` becomes a fixed-camera key. Supply metre `position`, either XYZ `euler_degrees`
or `target` plus a nonparallel `up`, dimensions and positive optical values. Explicit Euler angles take
precedence if both forms are present. Focal length and aperture use USD camera optical units. Camera
poses must agree with the convention of the supplied calibration; the template's target/up form avoids
guessing a quaternion convention. Mounted cameras remain in `arm.toml`, and their keys must be distinct.

`data_types = ["rgb"]` is sufficient for ordinary recording/evaluation. Other supported sensor outputs
can be declared for depth/segmentation recording. `depth_sensor = true` adds a separate depth sibling;
optional `prim_name`, `depth_prim_name`, `depth_sensor_name` preserve authored path/entity names when
needed. They are not required for a new scene. `viewer.camera` selects a fixed camera key.

Optional `calibration` stores OpenCV `fx`, `fy`, `cx`, `cy` and distortion coefficients/model for that
image resolution. RTX's half-pixel adjustment is applied by the camera helper; retain measured principal
points unchanged. Scene camera response fits live in `appearance.json` under `camera_response.<key>`.
Do not copy another room's intrinsics or response without measurement. Policy-facing camera names are
mapped separately in evaluation/rollout configuration.

`[scene.lighting]` declares a root, optional replacement of imported lights, and named light rows.
The template's dome light works without a hook. Supported types are `rect`, `dome`, `distant` and
`sphere`. Rows supply `intensity`, `color` and optional `exposure`; orient a light with
`position`/`target`/`up` or `euler_degrees` (plus optional `position`). A row with
`exposure_dr = true` participates in generic light-exposure randomization. See
[appearance.py](../src/sim/scene/appearance.py) for the normalized light fields.

Optional `[scene.renderer]` settings include antialiasing, ambient occlusion/global illumination, DLSSG,
ambient intensity and explicit `carb_settings`. Optional `[scene.physics]` keys override Newton solver
settings such as `num_substeps`, `collision_decimation`, `cone`, `impratio` and `tolerance`. Start from the
template defaults and justify changes with physical evidence. Solver tuning is not a substitute for
correct geometry or a way to force an acceptance result.

An optional `appearance = "appearance.json"` can hold fitted room/robot looks, camera response and
task material suggestions. Scene-specific material or texture names belong in a scene-folder hook:
set `hook = "scene.py:customize"` and export `customize(env_cfg, scene_spec)`. This receives the composed
config; it can register named startup/spawn functions from that module. Keep collision/contact work
before Newton construction and visual-only treatments in their intended startup order. The complete
[meeting-room hook](../scenes/rpd_meeting_room/scene.py) demonstrates this boundary. Omit the hook for
a declarative scene such as the template or [blank_table](../scenes/blank_table/scene.toml).

## Check the composed scene and capture geometry

With `s2r-sim` running and GPU use coordinated:

```bash
./s2r check --arm so101 --scene guide_scene --task stack_two_cubes --variant teleop
./s2r check --arm so101 --scene guide_scene --task stack_two_cubes --variant eval \
  --scene-json /workspace/s2r/outputs/guide-scene.json
```

`--scene-json` requires a runtime check of one explicitly selected arm/scene/task composition; it cannot
be combined with `--static`. The path is inside the container. With the normal compose mount, that file
is `outputs/guide-scene.json` on the host. It contains the live base frame, table geometry, named object
poses and frames needed by [eval-set generation](06-evaluation.md).

Static checking validates declarations and files without loading USD physics. Headless checking loads
and steps the composition, resolves bodies/joints and checks camera output. Neither establishes visual
matching to a real room, robust grasps or hardware alignment. For a measured scene, retain reference
images, optics, response parameters and object/robot poses; compare renders under controlled settings.
Record and inspect a complete episode before using the scene for training or benchmark comparisons.
