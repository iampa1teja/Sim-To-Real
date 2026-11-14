# v3 restructure plan

## Context

`v2` works but is hard to read: a task is spread over five files, the same helpers exist in several
places (two `Pose` classes, three sets of quaternion helpers, copied TOML validators, five atomic
writers), per-arm and per-task exceptions hide in library code, and user-editable folders are mixed
with source. The structure for `v3` was agreed folder by folder with the user (decision log: appendix below).
A planning subagent then checked the decisions against the code; where the code contradicted a
decision, the ruling is listed below. Goal: same behaviour, about 14% less `src/` code, one obvious
place for everything.

Out of scope for this plan: raw serializer, interactive mode, HTML docs (`final` branch), CLI rebuild.
The async policy worker (`feat/async-policy-worker`) touches only `core/policy_client`, which v3
leaves untouched, so it rebases cleanly at the end.

## Target structure

```
workspace/   arms/<arm>/{arm.toml, utils.py, calibration/, usd/, urdf/}
             scenes/<scene>/{scene.toml, utils.py, mounts/, usd/, textures/}
             tasks/<task>/{task.toml, task.py, usd/|assets/|eval_sets/|starts/}
             policies/ (unchanged)   configs/{record, eval, rollout, datagen}/
src/utils/   cli.py hashing.py provenance.py paths.py io.py geometry.py workspace.py scene.py
             arms/{spec, arms, mapping, calibration, urdf, arm_toml}.py
             calibration/{calibrate, check, stats, gap, report, telemetry, display}.py
             datasets/{alignment, sources, task_meta}.py
src/core/    tasks/{task.py, stages.py}
             policy_client/  policy_server/        (physically untouched)
             rollout/                              (unchanged backend)
             eval/{eval_set, starts, resume, smoke, report, sweep}.py   (was benchmark/)
src/sim/     __init__.py (sim.launch, was app.py)
             actions/{action.py, ik.py, checks/{reached, grasped, lifted, placed}.py,
                      moves/{approach, pick, place}.py}
             workspace/{arms/, scene/, envs/{base_env_cfg, solver_numerics, scene_builder,
                        task_env, frame}.py, envs/mdp/}
             teleop/{agent, loop, devices, config, controls, spawner, rerun_logging}.py
             recording/{recorder, setup, replay_scene, rerecording, web_gui}.py
             datagen/{scripted, mimic, rollout, calibration_replay, dr_replay, dr_variants}.py
             eval/{run, loop, scene, video, sim_remote_policy}.py, eval/sim_real/{paired, trials}.py
tests/       utils/ core/ sim/ workspace/tasks/ policies/ fixtures/
```

Not in v3: `docs/`, `tools/`, `src/cli/`, `s2r`, `sim/check.py`, per-task
`__init__/env_cfg/mdp/expert.py`.

## Rulings where the code contradicted a decision

| Decision | Ruling |
|---|---|
| Shared policy code to `utils/policy` | Dropped. The policy image ships `policy_client` and `policy_server` as self-contained top-level packages; the two `contract.py` are different classes. Both folders stay untouched (user confirmed). |
| IK solver only in `arm.toml` | New key `[arm.ik] solver = "dls" \| "bounded"` (sp7 bounded, so101 dls); existing `method` untouched. One per-task override `[task.expert] ik_solver` in `task.toml` (stack_bowls uses bounded on so101), with its tolerance, weight and `control_offset` moved there from the datagen YAML. `--ik_backend` and the two IK flags removed. |
| `solver_numerics` folded into `base_env_cfg.py` | Stays its own file, `sim/workspace/envs/solver_numerics.py`, so its test still runs without Isaac. |
| `stage_tracking` merged into core | Pure latching goes to `core/tasks/stages.py`; the env hook goes into the generic env builder (core must not import sim). |
| `failed_at_<first unlatched stage>` | Rule is `failed_at_<stage after the furthest latched stage>` plus a separate `timeout_holding` flag. No legacy-name map; old and new report columns simply differ. |
| `dr_replay`/`dr_variants` untouched | Import rewrites only, plus a generic object-colour term driven by `[task.dr]` replacing the pick-place `set_cube_color` they call. |
| Delete `tools/` | Deleted, along with the tests that only exercise tools scripts. |
| Keyboard and terminal controls removed | `teleop/controls.py` is one command queue fed by the GUI backend and by typed terminal commands (user chose to keep terminal). Keyboard removed. |
| Grid start coverage | Default for scripted datagen only; `start_mode: random` stays for reproducing reference datasets and is set explicitly in `scripted.yaml`; Mimic and eval sets never use grid. |

| `s2r` wrapper kept at root | Removed with the CLI (it only runs `python -m cli`); returns when the CLI is rebuilt. Programs run as `python -m sim...` / `python -m core...` meanwhile. |
| `sim/eval/report.py` merged into core | Only its pure report assembly merges; `episode_metadata` reads the live env and stays in `sim/eval/loop.py`. |
| `eval/run.py` uses `KeyboardControl` | Switches to the same `teleop/controls.py` queue (GUI backend or terminal). |
| `control_offset` in the datagen YAML | Scripted and Mimic both read it from `[task.expert]`; otherwise existing bowls source datasets stop validating (`mimic.py:271-273`). |
| `reached(a, b, dist)` and `placed` with stillness as shown in the discussion | **Ports reproduce today's rules exactly.** Stage names stay (`over_box`, `over_target`, `placed`, ...) because Mimic `subtask_term_signal` and report `stage_rates` use them. Today `over_box` is "held and footprint inside", `over_target` is "held and aligned", and pick-place `placed` has no stillness test; these become parameters of `reached`/`placed` with defaults equal to today's values. Switching any task to a plain "within x cm" rule changes eval success, so it would be a separate, explicit behaviour-change commit after Phase 18, only if the user asks for it. |

License headers: eight files carry NVIDIA SPDX/Apache headers (`core/calibration/{calibrate,check,stats}.py`,
`sim/eval/run.py`, `sim/recording/{lerobot_recorder,web_gui}.py`, `sim/teleop/controls/keyboard.py`,
`tasks/pick_place_cube/mdp.py`). The header travels with the code wherever it lands
(`recording/recorder.py`, `sim/actions/checks/*`). `dr_replay.py` and `dr_variants.py` carry no such
header; they still only get import rewrites, as decided.

Frozen names (changing them alters physics ordering or breaks stored results): gym id
`S2R-<task>-<variant>` with `teleop`/`eval`/`eval_dr`; events `reset_robot_position`,
`reset_from_starts`, `reset_tracking`; termination `success`; pick-place prims `/CUBE` and
`/WhiteBox`; contact filter order; DR event order; eval-set object names (`blue_cube`).

## Key interfaces

- **`utils/workspace.py`**: `kind_dir`, `available`, `item_dir`, `read_manifest`,
  `load_file_module` (used for `utils.py` hooks and `task.py`), validators `number/vector/text/table/boolean/file`.
  Reuses today's `core/scenes/spec.py:33-74` and `core/arms/spec.py:448-462`.
- **`sim/actions/action.py`** (module top: stdlib and numpy only; torch imported inside methods):
  `Frame` (per-step bundle of object poses, velocities, TCP point, contact forces, filled once per
  step by `sim/workspace/envs/frame.py`); `Check` with `reset`, `evaluate(frame) -> bool tensor`,
  `&` and `~`; `hold(check, steps)`; `Move.plan(ctx) -> tuple[Waypoint, ...]`.
  Check parameters default to today's per-task values (reach point and threshold, holding
  hysteresis, warmup gating, release rule, stillness).
- **`core/tasks/task.py`**: concrete `Task(spec, module)`; `load_task(name)` with a required name,
  cached, no environment reads at import. `task.py` exposes `stages`, `expert`, optional custom checks.
- **`core/tasks/stages.py`**: `StageTrace` (API unchanged) and `failure_mode(stages, latched, holding)`.
- **`sim/workspace/envs/task_env.py`**: `build_task_cfg(scene, task, variant)` builds objects,
  materials, contact sensor, resets, terminations and DR from `task.toml`; `register(scene, task)`
  is called by entry points after selection (replaces import-time `TASK`).
- `stack_bowls` keeps its order-free nesting check (about 40 lines) inline in its `task.py`.

## Phases

Every phase is one or more commits on `v3` and must leave the suite at the Phase 0 baseline.

**How a phase is gated.** This container cannot run Isaac, so the gate has two halves:
1. I run the local checks below, commit, and push the phase to `v3`.
2. You run one command in `s2r-sim` (`PYTHONPATH=src python -m pytest tests -q`, plus the named
   oracle script for phases 7-12) and report the result. **The next phase does not start until that
   passes**, as `CLAUDE.md` already requires.
Phase 0 also tries `uv sync --group host` (or installing pytest and CPU torch) here; if it works, the
Isaac-free tests become a local gate too and fewer round trips are needed.

**Nothing lives only in this session.** Phase 0 commits to `v3`: the decision log and this plan
(`RESTRUCTURE.md` at the root, removed in closeout), the baseline artefacts under `tests/fixtures/v3_baseline/`,
and the two regression checks as real tests (`tests/test_import_graph.py`,
`tests/core/test_eval_set_regression.py`). `CLAUDE.md` is updated in every phase that invalidates one
of its rules (first in Phase 1, when `s2r` and `sim.app` references change), not only at the end.

**Checks that run in this container** (no pytest or torch here): `python -B -m compileall`,
`rg` for stale module names and paths, an AST import-graph script (`utils` imports nothing from
`core`/`sim`; `core` has no module-top `sim`/Isaac imports; `sim/actions` module tops are stdlib and
numpy only), a numpy-only regression that rehashes the three stored eval sets and re-checks all 500 stored
starts (one, train-059, is already rejected on v2 and is pinned), and loading every task without torch.
**Checks that need `s2r-sim`**: full `PYTHONPATH=src python -m pytest tests -q`, oracle
comparisons, dataset frame hashes.

| # | Phase | What happens | Proof |
|---|---|---|---|
| 0 | Baseline | Branch `v3` from `v2`. Record pytest outcomes, eval-set digests, `to_dict()` cfg dumps per task and variant, per-step signal traces and frame hashes for one scripted episode per task, and `find_spec('utils')` in both images. | Recorded artefacts |
| 1 | Remove | Delete `docs/`, `tools/`, `src/cli`, `s2r`, `sim/check.py`, their tests; drop `core.paths.plan_new`; Dockerfile import check; add a compose `serve` profile and pass `S2R_GIT_COMMIT` through compose. | `rg` clean, suite green |
| 2 | `workspace/` | `git mv arms scenes tasks policies configs workspace/`; fix `paths.repo_dir`, config paths, test paths, `policies.molmoact2` imports (path-based loading); rename hooks to `utils.py`. | Hash regression, suite green |
| 3 | `src/utils` | Gate on the `utils` name check (fallback name `shared`). Pure `git mv` plus one mechanical import rewrite: `core/{utils,paths,arms,calibration,datasets,scenes}` to `utils/`; `core/benchmark` to `core/eval`. | Import graph, suite green |
| 4 | `sim` reshuffle | Moves only: `app.py` to `sim/__init__.py`; `arms/scene/envs` under `sim/workspace/`; spawner to teleop; IK files to `sim/actions/`; `eval/sim_real/`; recorder merge. | Suite green |
| 5 | De-duplicate | One `Pose` and explicit XYZW/WXYZ helpers in `geometry.py`; `workspace.py` validators; `io.atomic_write` keeping each caller's byte format; `hashing.legacy_json_sha256` for the two non-canonical stored hashes; split `arms/spec.py` into `spec.py` and `arms.py`. Gf and float32 Euler helpers are left alone. | Byte-equality tests, suite green |
| 6 | Task library | Add `core/tasks/{task,stages}.py`, `sim/actions/*`, `frame.py`, `task_env.py`. Loader accepts both the new `task.py` and the legacy layout. Reuses the existing signal math from the pick-place and stack `mdp.py`. | New CPU tests on synthetic frames |
| 7 | Port pick_place_cube | New `task.py` and extended `task.toml`; old `mdp.py` runs beside it as an oracle. | Signals equal per step, waypoints identical for 50 seeds, cfg diff only intended keys, eval-set starts valid |
| 8 | Port stack_two_cubes | Same method. | Same |
| 9 | IK unification | `[arm.ik] solver`, `[task.expert]` override, merge `bounded_ik` into `ik.py`, delete the SP7 hook swap and CLI flags. Done before bowls so differences are attributable. | sp7 and so101-bowls actions match Phase 0 |
| 10 | Port stack_bowls | Same method as 7, nesting check inline. | Same |
| 11 | Delete legacy task plumbing | Remove per-task `__init__/env_cfg/mdp/expert/geometry.py`, `core/tasks/{base,loader,expert}.py`, `stage_tracking.py`, import-time `TASK`. | `rg` clean, suite green |
| 12 | Datagen slimming | `scripted.py` to about 200 lines, `mimic.py` slimmed, diagnostics behind `--debug` in `rollout.py`. | Phase 0 dataset frame hashes reproduce (random mode) |
| 13 | Eval report merge | `core/eval/report.py` absorbs `stats`, `compare` and the pure part of `sim/eval/report.py`. | Old vs new JSON identical |
| 14 | Teleop/recording dedupe | Fold duplicated setup into `recording/setup.py`. | Suite green |
| 15 | Failure names | Auto `failed_at_*` rule. **Behaviour change.** | Updated `test_task_stages` |
| 16 | Grid starts | `GridStarts` sampler, default for scripted only; `cells_per_axis`, `yaw_steps`, `max_starts`; coverage map in the report. **Behaviour change.** | New sampler tests |
| 17 | Controls | `teleop/controls.py` queue fed by GUI backend and terminal. **Behaviour change.** | Recording smoke |
| 18 | Spawn dropdown | State exposes spawnable objects; spawn command carries the object name through `web_gui`, `paired_gui`, `testing_http` to `spawner.spawn(name)`; both GUIs get a dropdown. **Behaviour change.** | GUI smoke |
| 19 | Closeout | Rewrite `CLAUDE.md` and `README.md` for the new layout; new-format `_template`s; rebase the async worker. | Final full suite |

Phases 0-14 preserve behaviour; 15-18 are the only intended behaviour changes, one commit each.

## Risks to watch

- **Eval comparability**: digests hash content, not paths, so moves are safe; the generic start
  validity rule must give the same verdict on all 500 stored starts (checked every phase).
- **Physics**: prim names and contact filter order are frozen; grasp material only on cuboids and
  objects flagged `grasp_material = true`; bowls' solver capacity moves to `[task.physics]` unchanged.
- **Hidden numeric drift** when call sites switch between XYZW and WXYZ helpers (Phase 5): covered
  by the per-step oracle and dataset frame hashes.
- **Top-level `utils` name** shadowed by site-packages in the containers: gated in Phase 3.

## Verification

1. Here, after every phase: compileall, stale-name `rg`, import-graph script, eval-set hash and
   start regression, tasks load without torch.
2. In `s2r-sim`, after every phase: `PYTHONPATH=src python -m pytest tests -q` matches the Phase 0
   pass/skip set (minus deliberately deleted tests).
3. Phases 7-10: old-vs-new signal oracle and waypoint equality.
4. Phase 12: scripted reference recipe (seed 101) reproduces the Phase 0 frame hashes.
5. End: one scripted episode per task, one `one_per_zone` eval smoke, one recording session with the
   spawn dropdown.

Expected size: `src/` about 20,700 to about 17,900 lines; `docs/` and `tools/` (about 5,800 lines) removed.

---

# Appendix: decision log



Branches: v2 = current; v3 = restructure; final = after restructure (HTML docs).

### Task design
- Per task: task.toml (objects, per-object start ranges, placement, DR, episode) + task.py (stages, success, expert; custom checks inline).
- task.py imports only core building blocks (Isaac-free descriptions); sim evaluates them.
- Expert = primitives (pick, place_on, place_in...). Task never names IK; arm.toml [ik] picks solver.
- Failure modes auto-generated: failed_at_<first unlatched stage> (+ timeout_holding flag).
- env_cfg.py replaced by library building env from task.toml.

### Top level
- workspace/{arms, scenes, tasks, policies, configs}
- docs/: removed in v3; HTML docs later on branch `final`.
- tools/: removed in v3 (kept in v2 for reference).
- s2r, compose.yaml, compose.hardware.yaml stay at root.

### core/arms, scenes, utils (round 2)
- Common TOML loader code shared by all manifests (arm, scene, task).
- arms: spec.py split -> spec.py (dataclasses only) + arms.py (loaders/imports/hooks).
- Scenes: no scenes folder; anchors math -> utils/geometry.py; scene spec -> scene.py (like arms).
- utils/geometry.py: single Pose (XYZW), quaternion helpers, frame transforms; dupes deleted.
- utils root: cli.py (was config.py), hashing.py, provenance.py, paths.py (-> workspace/), io.py (atomic writes, JSON).
- utils/arms/{spec.py (dataclasses), arms.py (loaders+hooks), mapping.py, calibration.py, urdf.py, arm_toml.py}
- utils/scene.py: scene specs + loader (no scenes folder) [confirm single file vs utils/scenes/]
- utils/workspace.py: common TOML/manifest helpers + workspace folder discovery

### Round 3
- No core/workspace. Task library = core/tasks/ (low-level stage checks + expert moves written once; task.py calls them).
- utils/datasets/{alignment, sources, task_meta}
- utils/calibration/{calibrate, check, stats, gap, report, telemetry, display}
- OPEN: where check math lives (Isaac-free core vs sim), library file split, sim-side env builder.
- RULE: anything that varies per arm / per scene / per task goes in workspace/<arms|scenes|tasks>/<name>/ (toml or hook file), never in library code.
- Tentative (user can override): check math once in core/tasks, Isaac-free torch on a state bundle; core/tasks/{task, checks, moves, stages}.py; sim side = one generic env builder fills the state bundle. Arm-specific inputs (gripper tip offset, contact bodies, IK) come from workspace/arms/<arm>/.

### Round 4
- Low-level stage checks + expert moves -> sim/actions/ (overrides tentative "core/tasks" choice). core/tasks keeps Task/loader/stage engine only.
  Caveat: task.py imports sim.actions -> keep sim/actions free of Isaac imports at module import time (lazy), so `s2r check --static` can still load tasks.
- core/policy_client, core/policy_server keep files; shared contract+transport -> utils/policy/{contract, transport}.py
- grasp.py, smoothing.py stay in policy_client.
- core/rollout: frontend parts (HTML/UI) -> gui/; backend (HTTP/control bridges, real_rollout) stays in core/rollout.
- core/benchmark renamed core/eval; stats+compare merged into report.py; geometry -> utils/geometry.py.

### Round 5 (sim)
- core/rollout unchanged (all backend; frontend already in gui/).
- sim/workspace/{arms, scene, envs}; envs/mdp kept for generic env terms.
- IK -> sim/actions/ik.py; bounded merged in; solver from workspace/arms/<arm>/arm.toml; SP7 hook swap removed.
- sim/app.py, sim/check.py, sim/solver_numerics.py stay at sim root; FLAGGED for usefulness review.

### Round 6
- datagen: shrink + modularize; dr_replay.py, dr_variants.py unchanged (NVIDIA code). Details being discussed.
- sim/eval/report.py merged into core/eval/report.py; stage_tracking latching merged into core/tasks/stages.py.
- sim/eval/sim_real/{paired, trials}.py
- teleop = control {agent, loop, devices, config, controls/, rerun_logging}; recording = data {recorder (lerobot_recorder+recording_dataset), setup, replay_scene, rerecording, web_gui}; setup duplication folded into recording/setup.py.

### Round 7 (datagen)
- scripted.py slimmed to ~200 lines: planning -> sim/actions/moves, IK -> sim/actions/ik, argparse -> YAML + utils/cli; failure diagnostics -> optional --debug dump in rollout.py.
- mimic.py kept, slimmed (shared rollout + YAML, drop own diagnostic trace).
- spawner.py -> sim/teleop/spawner.py
- calibration_replay.py stays in datagen.
- rollout.py: write_json -> utils/io, pose helpers -> utils/geometry.

### Round 8
- Per-item hook file = utils.py everywhere (workspace/arms/<arm>/utils.py, workspace/scenes/<scene>/utils.py).
- workspace/configs/{record, eval, rollout, datagen}/ by command; task-specific values move into task.toml.
- tests/ mirrors new src: tests/{utils, core, sim, cli, workspace}; per-task tests -> tests/workspace/tasks/.
- CLI: same structure, new paths; review subcommands and flag rarely used ones (test-gui, envs, migrate, spawn...) for removal.

### Draft v3 tree
```
s2r  compose.yaml  compose.hardware.yaml  pyproject.toml  README.md  CLAUDE.md
docker/  gui/
workspace/
  arms/<arm>/        arm.toml  utils.py  calibration/  usd/  urdf/
  scenes/<scene>/    scene.toml  utils.py  mounts/  usd/  textures/
  tasks/<task>/      task.toml  task.py  (usd/, eval_sets/ if needed)
  policies/<policy>/ (unchanged)
  configs/{record, eval, rollout, datagen}/
src/
  utils/
    cli.py  hashing.py  provenance.py  paths.py  io.py  geometry.py  workspace.py  scene.py
    arms/        spec.py  arms.py  mapping.py  calibration.py  urdf.py  arm_toml.py
    calibration/ calibrate.py  check.py  stats.py  gap.py  report.py  telemetry.py  display.py
    datasets/    alignment.py  sources.py  task_meta.py
    policy/      contract.py  transport.py
  core/
    tasks/          task.py (Task + loader)  stages.py (latching, success hold, auto failure names, stage tracking)
    policy_client/  adapter  default  discovery  grasp  remote_policy  smoothing
    policy_server/  launcher  seeding  server
    rollout/        real_rollout  testing_gui  testing_http  paired_gui   (backend only)
    eval/           eval_set  starts  resume  smoke  report (+stats, compare, sim report)  sweep
  sim/
    app.py  check.py  solver_numerics.py        (FLAGGED: review usefulness)
    actions/        checks.py  moves.py  ik.py   (Isaac imports lazy)
    workspace/
      arms/   urdf_import  clearance  ready_pose  validation
      scene/  appearance  arm  camera_response  cameras  contact_materials  geometry  renderer  viewport
      envs/   base_env_cfg  scene_builder (+ generic task env from task.toml)  mdp/
    teleop/         agent  loop  devices  config  controls/  rerun_logging  spawner
    recording/      recorder  setup  replay_scene  rerecording  web_gui
    datagen/        scripted (~200)  mimic (slim)  rollout  calibration_replay  dr_replay  dr_variants
    eval/           run  loop  scene  video  sim_remote_policy  sim_real/{paired, trials}
  cli/              (same; trim list pending)
tests/  utils/ core/ sim/ cli/ workspace/
```
Removed in v3: docs/, tools/, per-task __init__/env_cfg/mdp/expert.py, core/scenes, core/benchmark/geometry.py, sim/datagen/bounded_ik.py, sim/eval/report.py, sim/eval/stage_tracking.py.

### Round 9
- v3 starts WITHOUT the CLI (src/cli + s2r planning added back later). Entry points run as `python -m sim....` meanwhile.
- No standalone sim/app.py, check.py, solver_numerics.py; code folded into the files that use them:
  - app.py (Kit launch + arm/scene/task selection, used by every sim entry point) -> sim/launch.py? [pending: user said fold into a sim file]
  - check.py (only used by CLI `s2r check`) -> dropped with the CLI for now.
  - solver_numerics.py (only used by envs/base_env_cfg.py; MJWarp contact fix) -> folded into sim/workspace/envs/base_env_cfg.py.
- app.py -> folded into sim/__init__.py as sim.launch(...) (user accepted).

### Round 10 (names, draft)
- Checks: approach(obj, dist), grasped(obj, force), lifted(obj, min_lift), reached(obj, target, relation=on_top|inside|above, tol, released?, max_speed?)
- Moves: pick(obj, approach, lift), place(obj, target, relation=on_top|inside, clearance)

### Round 10 (names, FINAL)
- Checks (sim/actions/checks.py):
  - reached(a, b, dist): a within dist of b's target range; a may be "gripper" (gripper near object) or an object (object over target).
  - grasped(obj, force)
  - lifted(obj, min_lift)
  - placed(obj, target, relation=on_top|inside, xy_tol, z_tol, max_speed) -> success; includes released + still.
- Moves (sim/actions/moves.py): approach(obj, height), pick(obj, lift), place(obj, target, relation, clearance)

### Round 11
- sim/actions/: action.py (abstract Check + Move base classes) + checks/{reached, grasped, lifted, placed}.py + moves/{approach, pick, place}.py + ik.py. New action = new file in subfolder.
- teleop/controls/{keyboard, terminal}.py removed -> teleop/controls.py: GUI-backend command queue (start/save/discard/reset/quit). eval/run.py also uses it (was KeyboardControl).
- datagen/scripted.py: deterministic even coverage of task.toml start ranges (stratified grid over each object's x/y/yaw range, all object combinations, respecting min_gap); today it is random per seed.

### Round 12 (GUI spawn)
- Spawn button gets an object dropdown filled from the task's spawnable objects (task.toml `spawnable = true`); no hardcoded names or "cube" text.
- Backend already supports names (spawner.spawn(name)); missing pieces: state exposes `spawnable` list; /api/spawn + "spawn" command carry {object}; paired_gui.request_spawn(name); web_gui COMMANDS pass object; help text uses the object name.
- Both GUIs: gui/app.jsx (recording) and gui/real/ (paired real/sim console).
