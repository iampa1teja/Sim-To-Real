# Repository conventions

The restructure is on branch `v2`; Phases 0–5 and 7 are complete, with Phase 6 deferred. See `docs/restructure-status.md` for phase gates and verified evidence.
Do not start the next phase until the current phase's CPU suite and simulator checks pass.

- `src/core/` must not import Isaac Sim, Isaac Lab, Kit, or `pxr`. Simulator-specific code belongs in `src/sim/`.
- `s2r` is the only installed console command. Plans invoke Python modules; keep CLI planning free of SDK
  imports and side effects so `--dry-run` prints the actual command. Sync includes arms, scenes, tasks, policies
  and configs. Host plans pass `S2R_GIT_COMMIT` into containers whose worktree pointer cannot resolve there.
- The typed arm loader reads `S2R_ARM` when called. Pass an explicit `ArmSpec` to components that must keep one arm
  for their lifetime. Scene loaders also resolve selectors at call time; the Gym registry and task modules still
  select registrations at import time, so choose selectors before importing them. Shared selection and Kit
  argument helpers live in `sim.app` and are safe before Kit startup.
- Launch simulator programs through `sim.app.launch` after choosing arm/scene/task. Recording lives in
  `sim.teleop.agent`, evaluation in `sim.eval.run`, generation in `sim.datagen`, and real rollout in
  `core.rollout.real_rollout`. Shared argparse/YAML handling lives in `core.utils.config`; flags override YAML.
- `load_arm_spec()` returns `(ArmSpec, Path)`. Use `.joint_names`, `.usd_joint_names`, `.feature_names`, joint roles,
  optional `.gripper`/`.wrist_roll`, and the declared wrist cameras. No arm-specific names or motor indices belong
  in generic source. Optional spawn/appearance hooks are loaded from the arm's `<arm>_utils.py`.
- Isaac Lab 3.0 / Newton live tensors use XYZW quaternions. Dataset sidecars and benchmark JSON use WXYZ.
  Convert explicitly at those boundaries. Positions and translations are metres; articulation angles are radians.
- `[[arm.joints]]` order is action/dataset order. Joint `limits` and calibrated physical spans use degrees for
  revolute joints and metres for prismatic joints; `initial` and simulator state use radians/metres. Raw normalized
  body joints use -100..100 and gripper joints 0..100. LeRobot `use_degrees = true` changes body joints to a
  driver encoder-degree coordinate, centred on the calibration midpoint and scaled by resolution-1; this is not
  the arm's physical degree zero. Grippers remain normalized. Selected mapping entries preserve calibrated
  `raw_min`/`raw_max` endpoints; reverse directions are explicit. Deriving nonzero drive_mode requires an explicit
  unit-source `apply_drive_mode` capability or stored endpoints; never guess the bus family from a motor name.
- Calibration spans and runtime clamps are distinct. Use `JointMapping.from_arm(arm, selection)` and its
  `to_sim`/`to_sim_np`/`to_raw` methods. Pure interpolation lives in `units_to_physical`/`physical_to_units`;
  these pure functions interpret stored raw endpoints (or conventional normalized defaults) and do not apply runtime clamps.
  `mapping_for_arm` validates required degree-mode codecs and canonicalizes redundant normalized defaults.
  Use `raw_joint_limits(arm, mapping)` for action limits; physical runtime limits are not raw driver bounds. `to_physical` and
  `sim_to_physical` are the calibrated device/simulator boundaries for telemetry. Never apply degree conversion
  to prismatic axes or infer gripper roles from motor names.
- Per-unit calibration is in `arms/<arm>/calibration/units/<id>.json`, selected by recording `follower.unit` or
  command `--unit`. It is independent of scene JSON. Raw LeRobot device files live in the arm-declared
  calibration folders under `$HF_LEROBOT_CALIBRATION`; preserve their hashes and the unit's source evidence.
  Factory revolute mappings use full encoder range; factory prismatic mappings use the declared full stroke.
- New alignment markers retain arm, joint order/units/roles, device-unit mode, mapping, and relative-roll behavior.
  A legacy marker cannot prove equivalent action units for resuming, so use the existing separate-root flow.
  Mixed-axis telemetry uses physical-value fields with per-joint unit metadata and separate metre/degree gates;
  degree-labelled fields are reserved for revolute-only logs. Never pool degree and metre errors.
- Newton imports scene topology once. Declare objects before model construction, then park/reposition them;
  do not add rigid bodies during a running episode.
- Data generation uses task-folder `expert.py` waypoints and arm-declared `ik` TCP/orientation constraints.
  The pick-place expert must not add a deliberate wrist roll after pickup. Preserve its pickup orientation
  through transport; earlier rolled diagnostic datasets are not approved final training sources.
  Scripted generation, Mimic and calibration replay share `sim.datagen.rollout`: capture pre-action frames,
  execute physical actions, then observe task signals. Commit only confirmed successes still valid at the final
  frame. Generated actions are absolute simulator targets encoded through the selected device mapping; they
  do not use teleop wrist rezeroing. Mimic annotations record FK of the executed target, the same TCP offset,
  named object poses and latched task stages. Nominal requested waypoints are separate from executed targets.
- Table planes and link-local jaw geometry are static across resets and cached per USD stage. Call
  `clear_geometry_cache(stage)` after explicitly editing those meshes or table transforms.
- Scene manifests own room anchors, fixed cameras, lights, renderer and physics settings. `mounts/<arm>.toml`
  owns the arm/scene combination; missing mounts use the anchor and arm initial joints. Mount translation
  corrections are world-space metres. Anchor positions and rotations use the same rigid frame transform;
  surveyed vectors kept as metadata must not silently replace that frame's basis.
- Scene-specific visual treatments belong in the optional scene-folder hook. A contact anchor may declare
  `contact_prim_pattern` when its collision meshes are separate from its measured visual mesh. Bind those
  static materials before Newton imports the stage; visual-only lighting retains startup event order.
- Preserve fixed evaluation schedules, seeds and content hashes. Schema conversion must retain the legacy hash.
- Policy clients receive ordered joint names, roles, joint types and the saved device codec through
  `configure(ArmInfo)`. Generic clients derive their dimension from that contract; a fixed embodiment declares
  `[action].dim`. Keep model-specific coordinate conventions inside its policy folder.
- Task plugins own signals, per-step stage flags and failure rules. The generic stage engine latches the declared
  stages and treats the last as success. Reset order is robot pose, object starts, then task tracking state.
  Use the generic reset-order assertion; engine code must not dispatch on task names, reset function names or
  task-specific environment attributes. Object placement/start frames are declared in the task manifest.
- Schema-2 sidecars store every declared tracked object in `task_meta/`; readers accept legacy `pick_place_meta/`.
  Write task/arm identity, reference frame and metre units, with one WXYZ object pose per committed dataset frame.
  Pure readers, validation, migration and source handling belong in `core.datasets`.
  Saved eval-set arm states are raw units interpreted through that set's mapping, not the active unit's mapping.
  Approved schema-2 eval-set conversions retain `legacy_eval_set_id` for comparison; preserve every scheduled
  start/seed. Schema migration does not change or bypass the live geometry guard.
- `gui/` and `src/sim/recording/web_gui.py` are outside this restructure. The local untracked code explorer is also
  outside the repository; its files are preserved.
- Run CPU checks with `PYTHONPATH=src python -m pytest tests -q`. The existing `s2r-sim` container includes the
  required Python/Isaac dependencies; run simulator scripts through `s2r-entrypoint python` there.
- `s2r check --static` runs host manifest/entrypoint validation without Isaac/USD or Torch imports; NumPy is its
  only external dependency. The complete host test/tool environment is available with `uv sync --group host`.
  Without `--static`, the command additionally launches selected headless simulator checks; coordinate GPU use.
- Keep local hardware settings in ignored `.env`; `.env.example` contains portable example values.
- Do not change production physics to make a diagnostic pass. Diagnostic spawn fixtures must start with enough
  jaw clearance for their object. Closed-jaw object overlap is a known placement issue, not a safe simulator input.

The arm, declarative scene and generic task phases have passed their acceptance checks; consult the status file
for evidence and renderer limitations. Phase 4's full CPU suite, static/plugin checks, production recording,
example evaluation and live GR00T/Molmo serving/evaluation pass. Phase 5 CPU implementation passes, including
real SDK Mimic configuration/retargeting and schema-2 GR00T preparation. A no-roll scripted episode now passes
physical success, dataset validation and real GR00T preparation/verification. Mimic passes physical success and
dataset validation at the source start and a nearby changed start. A two-step GR00T optimizer update and source
integrity check pass; Phase 5 is complete. Phase 7 guides were exercised through new folders, headless
composition, policy serving/evaluation and paired comparison. The user deferred Phase 6 on 2026-11-10;
second-arm hardware validation remains outstanding. Training recipes and the retained visual DR recipe still
have their documented SO-101/pick-place scope; do not claim arbitrary-model training or learned task performance.


The paired-motion follow-up isolated a float32 MJWarp Cholesky failure during table contact. The same replay
fails identically on an isolated copy of main in the current container. `sim.solver_numerics` repairs only
failed Newton search directions: double-precision factorization, then positive-diagonal descent if needed.
The existing contact objective/line search and every physical parameter remain unchanged. Install before
CUDA graph capture; do not edit installed SDK files. Captured-matrix CPU tests and a 1,140-step contact/retreat
replay pass. Teleop rejects invalid simulation joints immediately, including outside active recording.
