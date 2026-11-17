# WP8 — Isaac adapter, generic env config, generation runner and CLI (Codex prompt)

## Goal

Connect the CPU generation loop (WP7) to Isaac Sim: an `IsaacGenEnv` that implements `GenEnv` over an Isaac Lab env, a `RecorderEpisodeWriter` that implements `EpisodeWriter` over the existing `LeRobotRecorder` (with the label columns), a generic env config that builds any task's objects, the `generate` runner, and the CLI wiring for all v4 programs. Also remove the remaining SO-101 string literals from four sim files.

Only part of this can be tested without Isaac Sim. The CPU tests pin the wiring, the dry run, the recorder change, the writer adapter and the literals; the Isaac behaviour is specified below and is checked in the container by the owner (container check scripts are written after this package lands).

## Files

| File | What |
|---|---|
| `src/sim/generate/isaac_gen_env.py` (new) | `IsaacGenEnv`, `RecorderEpisodeWriter` |
| `src/sim/generate/scripted_generate.py` (new) | `build_parser`, `main` |
| `src/sim/envs/generic_task_cfg.py` (new) | `apply_teleop(cfg, task)`, `apply_generate(cfg, task)` |
| `src/sim/envs/mdp/objects.py` (new) | `set_object_pose_base`, `set_object_color`, `object_poses_in_base`, `park_all` |
| `src/sim/recording/lerobot_recorder.py` (change) | `extra_features` argument, `make_sim_frame(..., extra=None)` |
| `src/cli/paths.py` (change) | six `SIM_PROGRAMS` rows |
| `pyproject.toml` (change) | six console scripts |
| `configs/generate/smoke.yaml` (new) | smoke config |
| `src/sim/scene/arm.py`, `src/sim/scene/contact_materials.py`, `src/sim/eval/grasp_diagnostics.py`, `tasks/pick_place_cube/env_cfg.py` (change) | link and mesh names from `[arm.links]` |

Do NOT touch anything under `tests/`, `docs/`, the `core/` modules of other packages, or other tasks. `arms/so101/arm.toml` gets `links.moving_jaw` / `links.moving_jaw_mesh` in WP2; use them.

## Read first

`docs/v4/contracts.md` §1.2 (joint order; read names from `load_arm_spec(arm)` at call time), §1.6 (imports), §2 (WP8 row), §3.3 (protocols and call order), §9.3 (label columns), §11.1 (loop), §12 (Isaac side), §15. Prompts `WP7.md` (loop, `make_spec_fn`, `make_expert_factory`, plan script) and `WP2.md` (`RawMapping`, `KinematicChain`, `Gripper`, arm.toml links). Existing code to reuse: `sim/generate/dr_replay.py` (`make_env`, `discover_sim_cameras`, `rendering_kit_args`, `git_commit`, the AppLauncher pattern), `sim/recording/lerobot_recorder.py`, `tasks/pick_place_cube/{env_cfg,mdp}.py`, `sim/teleop/lerobot_agent.py::_object_pose_in_base`.

## Acceptance

```
uv run --group host pytest tests/sim/test_v4_wp8_*.py -q
```

pass, all earlier tests still pass, and the owner's container smoke run (`s2r generate --config_path configs/generate/smoke.yaml`, then `s2r dataset-audit <root>`) succeeds. Do not edit, skip or weaken tests; if one contradicts the contract, stop and report.

## Decisions pinned by the CPU tests

**CLI.** `SIM_PROGRAMS` gains `generate → scripted_generate`, `generate-plan → plan_generation`, `arm-kinematics → extract_kinematics`, `dataset-audit → audit_dataset`, `dataset-merge → merge_shards`, `dataset-upload → upload_dataset`; existing rows unchanged. `[project.scripts]` gains `scripted_generate = "sim.generate.scripted_generate:main"`, `plan_generation = "core.scripts.plan_generation:main"`, `extract_kinematics = "sim.generate.extract_kinematics:main"`, `audit_dataset = "core.scripts.audit_dataset:main"`, `merge_shards = "core.scripts.merge_shards:main"`, `upload_dataset = "core.scripts.upload_dataset:main"`; existing scripts unchanged.

**`configs/generate/smoke.yaml`.** A valid generate config (§8.5): `arm: so101`, `scene: rpd_meeting_room`, `tasks` = every task that exists (today `[pick_place_cube]`), quotas between 1 and 20 per kind, `shard: "0/1"`.

**`scripted_generate.build_parser()`.** `--config_path` (required), `--task`, `--shard`, `--dst_root`, `--master_seed` (int), `--dry_run`, `--limit_attempts` (int, default 0 = no limit), `--disable_fabric`. `main(argv=None)` calls `parse_known_args` first (AppLauncher arguments may follow), applies `--shard` / `--dst_root` / `--master_seed` as config overrides, and:
- `--dry_run`: prints exactly what `generate-plan` prints for the resolved config (one JSON object per task, same keys and order, same exit code and stderr format), importing neither Isaac, LeRobot nor gymnasium. Share the code with `core.scripts.plan_generation` through a private helper there or here; do not change its CLI.
- otherwise: add the AppLauncher arguments, start the app (cameras enabled when `cfg.cameras`, `rendering_kit_args`), and for each task build the env with the task's `generate` variant, an `IsaacGenEnv`, a `LeRobotRecorder(extra_features=LABEL_FEATURES, ...)` wrapped in `RecorderEpisodeWriter` (dataset root `cfg.dataset_root(task)`, repo id `cfg.dataset_repo_id(task)`), and call `run_generation(cfg, task, env, writer, expert_factory=make_expert_factory(task, cfg.arm), mapping=RawMapping.from_setup(arm=cfg.arm), spec_fn=make_spec_fn(task, cfg.arm, scene=cfg.scene, noise=cfg.noise, weights_override=cfg.perturbation_weights, fps=cfg.fps), git_commit=git_commit())`. `--limit_attempts N > 0` stops after N executed attempts (use a wrapper around `spec_fn`/the writer; do not change WP7). Print the `GenerationSummary`. Close the env, finalize the writer and close the app in `finally`.
- No Isaac / LeRobot / torch import at module level (tests block those imports and import the module).

**`LeRobotRecorder`.** New keyword `extra_features: dict | None = None`. `None` keeps `dataset_features` exactly as today (same dict, same key order). A dict is appended after the default features. `make_sim_frame(action, observation, visual_buffers, extra=None)`: `None` keeps today's frame; each extra column is copied as a numpy array with the feature's dtype; a key that is not a dataset feature raises `KeyError`.

**`RecorderEpisodeWriter(recorder)`.** `root = Path(recorder.dataset_root)`; `episode_count() = len(recorder.episode_lengths())`; `episode_lengths()` passes through; `add_frame(action_raw, state_raw, images, task, labels)` = `frame = recorder.make_sim_frame(action_raw, state_raw, images, extra=label_columns(labels))`, `frame["task"] = task`, `recorder.add_frame(frame)`; `discard_episode()` = `recorder.clear_episode_buffer()`; `save_episode()` returns `recorder.save_episode()`; `finalize()` = `recorder.finalize()`. `sim.generate.isaac_gen_env` imports no Isaac / LeRobot module at top level.

**No SO-101 literals** (AST scan, docstrings and mapping keys such as `spec["gripper"]` excluded): in `src/core/expert/`, `src/core/generate/`, `src/sim/generate/{isaac_gen_env,scripted_generate}.py`, `src/sim/envs/generic_task_cfg.py`, `src/sim/envs/mdp/objects.py`, `src/sim/scene/arm.py`, `src/sim/scene/contact_materials.py`, `src/sim/eval/grasp_diagnostics.py` and `tasks/pick_place_cube/env_cfg.py` no string constant may equal `"so101"`, `"jaw"`, `"gripper"`, `"moving_jaw_so101_v1"`, `"wrist_roll_follower_so101_v1"`, an SO-101 USD joint name or LeRobot name, contain `so101`, or end with `/jaw` or `/gripper`. Read `[arm.links]` (`ee`, `moving_jaw`, `fixed_jaw_mesh`, `moving_jaw_mesh`) with `load_arm_spec(...)`. The SO-101-specific split-plane numbers in `contact_materials.py` stay; apply them only when the named mesh exists.

## Isaac-side behaviour (container-checked)

**`IsaacGenEnv(env, task, *, arm, mapping, chain, gripper, cameras, palette_rgb)`** implements `GenEnv` (§12):
- `fps = round(1 / (env.step_dt))` (30 for the current config); `joint_names = [arm.joints].names`; `camera_names = tuple(cameras)`.
- At construction: map USD joint names to articulation indices with `find_joints(names, preserve_order=True)`; assert the joint-position action term has scale 1 and offset 0 (so the action equals the target in radians).
- `reset(spec)`: `env.reset()`; `park_all`; teleport active objects with `set_object_pose_base`; set colours with `set_object_color` from `palette_rgb`; write `initial_q` as the joint state (positions and targets, zero velocity); if `held_object`, close the gripper on it at the TCP and settle for a few physics steps; return frame 0.
- `step(q_target)`: one env step with the absolute joint-position action in `[arm.joints].names` order; returns the new observation. `observe()` reads without stepping.
- Observation: `q` from the articulation (radians, mapped order); `ee` from `ee_frame_state` in the base frame; `tcp = chain.tcp_from_ee(ee)`; `gripper_gap_m = gripper.gap_from_angle(q[gripper.index])`; objects from `object_poses_in_base` (inactive ones at their park pose, `active=False`); `signals` from the task's `subtask_terms` observation group when present (`pick_place_cube` keeps its contact-sensor signals), otherwise `{}`.
- `set_object_pose(name, pose)`: teleport with zero velocity (detaches by physics).
- `render()`: `{camera: uint8 (H, W, 3)}` from the scene cameras.

**`generic_task_cfg.apply_teleop(cfg, task)` / `apply_generate(cfg, task)`:** build every object of `task.spec.objects` (`CuboidCfg` for `size`, `UsdFileCfg` for `usd`, mass and contact properties as in `pick_place_cube`), at distinct park offsets from `park_positions(names)` with a park reset event each; contact sensor at `{ENV_REGEX_NS}/Robot/<links.moving_jaw>` filtered to all movable objects; `apply_generate` has no terminations and `episode_length_s` covers `[task.expert].max_steps`. A new task's `env_cfg.py` is three lines that call these.

**`mdp/objects.py`:** `set_object_pose_base(env, name, pose)`, `set_object_color(env, name, rgb)`, `object_poses_in_base(env, names) -> {name: Pose}`, `park_all(env, names)`, generalised from `tasks/pick_place_cube/mdp.py` and `lerobot_agent._object_pose_in_base` (base frame = `[arm.links].base`).

## Notes

- `pick_place_cube` keeps working for teleop and eval exactly as before; its `env_cfg.py` only stops spelling the jaw link.
- Container checks to add after this lands (owner runs): FK vs `ee_frame_state` (< 1 mm, < 0.5°), multi-object parking/teleport/recolour, expert success rate per task, smoke generation + audit + LeRobot load, DR replay v2, streaming.
