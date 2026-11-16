# Repository conventions

This is the `v3` layout. Programs run as `python -m ...`; the `s2r` wrapper and `src/cli` are absent.
Keep changes within the requested scope, preserve user datasets/local tools, and keep existing license headers
attached to their code. Never rewrite pushed history or add local hardware settings from ignored `.env`.

## Architecture and selection

- `src/utils` contains shared geometry, manifest loaders, codecs, calibration, dataset readers and atomic I/O.
  It imports neither `core` nor `sim`.
- `src/core` must never import Isaac Sim, Isaac Lab, Kit or `pxr`. Simulator integration belongs in `src/sim`.
  Existing task runtime hooks into `sim` remain lazy. `core/policy_client` and `core/policy_server` are
  self-contained packages; change them only for explicitly requested work, not shared-code consolidation.
- `workspace/{arms,scenes,tasks,policies}` contains item-specific data/code. Anything that differs per arm,
  scene or task belongs there. Optional arm/scene hooks are `utils.py`.
- Arm/scene loaders resolve selectors at call time. `load_arm_spec()` returns `(ArmSpec, Path)`.
  `core.tasks.task.load_task(name)` requires an explicit name and caches it; imports select no task.
  Pass an explicit spec to components that keep one selection for their lifetime.
- Entry points apply selection, launch through `sim.launch`, then call
  `sim.workspace.envs.register(scene, task)`. Gym IDs are `S2R-<task>-<variant>` and `S2R-<scene>`.
  Shared launch/YAML helpers are in `sim` and `utils.cli`; explicit flags override YAML.
- Tasks are `task.toml` plus `task.py`. The manifest owns objects, starts, reset/contact policy, DR, minimum
  solver capacities and episode settings. The module composes stages/checks and expert moves from
  `sim.actions`; descriptors/checks/moves must import without Torch/Isaac. Runtime imports stay lazy;
  `ik.py` is the explicit heavy-import exception checked by `tests/test_import_graph.py`.
- `core.tasks.stages` latches stages; the last stage means success. Failure names are
  `failed_at_<stage after the furthest latched stage>`, plus a separate `timeout_holding` boolean.
  Task modules own signal semantics and optional geometry/progress hooks, not failure-name maps.
- `core.eval.report` owns statistics, paired comparisons and pure report assembly; live metadata/progress
  reads stay in `sim.eval.loop`. Shared manual recording setup is `sim.recording.setup`; generation setup
  and execution are in `sim.datagen.rollout`. Event parameters carry `TaskSpec` data, not code-bearing
  `Task` instances; reconstruct the sampling task at reset. Grid samplers also keep only the spec so SDK
  configuration validation cannot traverse imported module graphs.

## Physics, frames and units

- Never change production physics to make a diagnostic pass. Never edit installed SDK files. Newton imports
  scene topology once: create bodies before simulation, then park/reposition them instead of adding bodies.
- Preserve production gym IDs, object/stage/waypoint names, `/CUBE`, `/WhiteBox`, `contact_grasp`, the
  `success` termination, event names and ordering (`reset_robot_position`, `reset_from_starts`,
  `reset_tracking`), contact-filter ordering and DR draw/event ordering.
- Live Isaac Lab 3.0 / Newton quaternions are XYZW. Sidecars and benchmark JSON use WXYZ. Convert explicitly
  at boundaries using `utils.geometry`. Keep the existing Gf and float32 Euler paths numerically unchanged.
- Non-mimic `[[arm.joints]]` order defines action/dataset order. Mimic joints are not action dimensions.
  Derive roles, names, units, gripper indices and wrist cameras from the arm; never guess by motor name.
- Simulator positions are radians/metres. Calibration spans/runtime limits are physical degrees/metres.
  Raw normalized body joints use -100..100 and grippers 0..100. Driver degree mode is an encoder coordinate
  centred/scaled by the measured calibration; it is not physical degree zero. Grippers remain normalized.
- `JointMapping.from_arm(arm, selection)` and `to_sim`/`to_sim_np`/`to_raw` own the device/sim boundary.
  `to_physical`/`sim_to_physical` own physical telemetry. Preserve raw endpoints and reversed directions;
  deriving drive mode requires the unit source's declared `apply_drive_mode` capability or stored endpoints.
  Pure `units_to_physical`/`physical_to_units` interpolate without runtime clamps. Use `raw_joint_limits` for
  raw action limits. Never convert prismatic axes to degrees or pool degree/metre errors in diagnostics.
- Per-unit files are `workspace/arms/<arm>/calibration/units/<id>.json`, selected with recording
  `follower.unit` or `--unit`. Device IDs/raw LeRobot files under `$HF_LEROBOT_CALIBRATION` are separate;
  preserve their hashes, source evidence and arm-declared folders. Factory mappings use full encoder range
  for revolute joints and declared full stroke for prismatic joints.
- Scenes own room anchors, cameras, lighting, renderer and physics. `mounts/<arm>.toml` owns mount corrections
  and initial joints; absent mounts use the anchor/arm defaults. Translation corrections are world-space metres.
  Anchor positions and orientations use the same rigid transform; surveyed metadata must not replace it.
- Scene visual/contact hooks bind materials before Newton imports topology and retain startup ordering.
  Grasp materials apply to cuboids or explicitly flagged USDs only. Table/jaw geometry caches are per stage;
  call `clear_geometry_cache(stage)` after explicitly editing those meshes/transforms.
- `sim.workspace.solver_numerics` repairs only failed MJWarp search directions with the existing double
  factorization/positive-diagonal fallback. Preserve the objective, line search, parameters and installation
  before CUDA graph capture. Diagnostic object spawns need sufficient physical jaw clearance.

## Recording, generation and evaluation

- `sim.teleop.controls.CommandQueue` accepts GUI and typed terminal commands, one transition per tick.
  Teleop/eval share it; keyboard listeners are removed. GUI-only recording requires `control_input: gui`
  and `gui.enabled: true`. Both GUIs expose task spawnables and carry the chosen name to `spawn(name)`.
  Keep GUI changes within the requested scope.
- Alignment markers preserve arm, joint order/units/roles, device mode, mapping and relative-roll behavior.
  Legacy markers cannot prove equivalent resume units: retain the separate-root mismatch flow.
- Generated frames capture pre-action observations, execute the physical action, then observe task signals.
  Commit only latched success still valid at the final frame; discard failed buffers. Actions are absolute
  simulator targets encoded through the device mapping, without teleop wrist rezeroing. Mimic annotations
  use FK of executed targets and the same TCP offset; nominal requested waypoints remain separate.
- `[arm.ik].solver` selects `dls` or `bounded`; `[task.expert]` may override solver, orientation tolerance/weight
  and control offset for scripted/Mimic generation. Calibration command replay keeps arm defaults.
  Do not add deliberate pickup wrist roll to the pick-place expert; preserve pickup orientation in transport.
- Scripted defaults to `GridStarts`; `cells_per_axis`, `yaw_steps`, `max_starts` control it and reports retain
  per-object coverage. Explicit `starts_dir` selects recorded starts. Reference `scripted.yaml` stays random.
  Mimic/eval never use grid. `--debug` enables optional generation diagnostics/boundary images.
- Schema-2 `task_meta/` stores task/arm identity, reference frame, metre units and one WXYZ object pose per
  declared tracked object per dataset frame. Readers accept legacy `pick_place_meta/`. Pure readers,
  validation, migration and source handling stay in `utils.datasets`.
- Preserve fixed eval-set content IDs, seeds and every scheduled start. Schema conversion preserves
  `legacy_eval_set_id`; saved robot states use the set's mapping, not the active unit's mapping. Never bypass
  the live geometry guard. `train-059` remains the pinned pre-existing rejection among the 500 stored starts.
- Policy clients receive ordered names, roles, joint types and the device codec through `configure(ArmInfo)`.
  Fixed embodiments declare action dimensions; model-specific coordinates stay in their policy folders.
  Training/visual-DR recipes retain their SO-101/pick-place scope; do not claim arbitrary-model training or
  learned task performance from wiring tests.
- Known issues outside this restructure remain: paired evaluation's MolmoAct2 hardcode, the pinned eval-start
  rejection, and `sim.apply_selection` ignoring the `S2R_*_DIR` overrides. Do not silently fix them.

## Verification

Run the CPU baseline comparator, import/eval-set/workspace tests, compileall and stale-import scans after
structural changes; run the full suite in `s2r-sim`. The host dependency group is not a complete SDK environment.
Pass `S2R_GIT_COMMIT=$(git rev-parse HEAD)` through Compose to record provenance. See README for commands.

Retain frozen task/IK and report-byte oracles. Phase 15 intentionally replaces legacy failure-label assertions;
those replacements are named in the baseline comparator. Phase 12 has a user-approved exact frame-hash gate
exception in `tests/fixtures/v3_baseline/phase12_gate.json`: unchanged controls were nondeterministic.
Record native crashes, retries and dependency versions honestly when using an isolated test overlay;
never alter SDK installations or production physics to hide them. Preserve the local untracked code explorer.
