# Pick-place evaluation validation

This validation snapshot was captured before commits were authorized.
`CLAUDE.md` was not present in the checkout.
Existing geometry, WHITE_BOX constants, grasp/success terms, teleop/recording,
MyRoom, vial configs and `so101_env_cfg.py` are unchanged.

## Behaviour and implementation checks

- `utils/pick_place_eval.py` loads frame zero from each sorted `episode_*.json`.
  Reported recorded indices are zero-based indices into that sorted list.
  Loading validates metres, position and wxyz quaternion data. Each cycle
  reshuffles all starts with the evaluation seed; random mode samples uniformly.
- Recorded XY and orientation remain in the live robot base-link frame. Recorded
  Z is discarded. Placement solves the measured `table_plane` support equation
  along base Z, retaining base XY even when the base is tilted. It uses the same
  oriented cuboid support and 0.2 mm vertical clearance as `reset_object_pose`.
- Held-out XY comes from the recorded bounds; yaw is uniform in `[0, pi/2)` and
  the cube stands upright on the live table plane. Projected cuboid footprints
  are tested against the live white box, with no added footprint margin.
  Impossible rejection sampling fails clearly after 10,000 attempts. A recorded
  start overlapping the current box fails explicitly instead of moving that start.
- Zones split base XY bounds into thirds. The axis with the largest absolute
  bounding-box-centre coordinate points away from the base (ties choose X), with
  sign toward that centre. N/M/F are near/middle/far along that direction;
  L/C/R are left/centre/right looking away from the base (+Z up). Degenerate
  axes use the middle third. For starts predominantly along +X, near is low X
  and left is high Y. All bounds and the away direction come from the data.
- Isaac Lab 2.3.2's event manager enumerates `cfg.__dict__.items()` and applies
  terms in insertion order. The Eval override retains `spawn_cube`'s inherited
  position: `reset_robot_position < spawn_cube < reset_tracking`. A startup
  callback checks the actual manager order, validates/loads the starts before
  environment creation finishes, and prints the reset order. DR appends robot
  colour after `reset_set_robot_visual_material` and verifies that order too.
- The measured room tube light is created by MyRoom's startup event. Its live
  scene path is used with the existing light-exposure term, range `(-3, 1)`.
  There is no LightStudio, sky-light/HDRI asset, or mat in this scene: those
  targets are skipped, with comments. Custom room overrides lacking the
  measured tube light skip its exposure randomization.
- All three actual cameras (RGB, depth, wrist) reuse the vial task's inherited
  camera DR terms/ranges: focal length `(12, 15)` mm, position X/Y `(-.02, .02)`,
  Z `(-.01, .01)`, and roll/pitch/yaw `(-.05, .05)` radians. The calibrated
  OpenCV lens's fx/fy are synchronized so focal-length DR affects RTX projection.
  Robot DR reuses `ROBOT_COLORS`; live cube shaders get the SETUP blue or
  `(0.8, 0.05, 0.05)` red, with the colour name saved for instruction selection.
- Eval snapshots start/zone/colour before stepping. Isaac auto-resets on done;
  the client uses the returned next observation without another reset, so it
  does not skip starts or report the next episode's colour. Timeout is failure,
  including when success and timeout occur together. Manual resets are excluded.
- JSON schema version 1 includes task, checkpoint, seed, UTC time, episodes,
  overall, by_start, by_zone and by_cube_color. Each aggregate gives episode and
  success counts plus a rate in `[0,1]` (null for empty groups). `complete` and
  `requested_episodes` identify interrupted evaluations. The checkpoint comes
  from `--checkpoint`, `PICK_PLACE_EVAL_CHECKPOINT`, or `MODEL`.

## Container checks to run

These are GPU/renderer checks for the robot machine; they were not run here.
Use your own dataset/checkpoint paths. Start `teleop` with the README command
first, so source and datasets are mounted.

1. Inspect recorded starts, with no policy server:

   ```bash
   docker exec -it teleop bash
   export PICK_PLACE_EVAL_STARTS=/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta
   zero_agent --task Lerobot-So101-Teleop-Pick-Place-Eval --num_steps 2250
   ```

   Pass: the startup order check succeeds; over five 450-step timeout resets,
   the cube appears at recorded starting spots, resting on the tabletop and
   outside the white box. Cube and box sizes and success behaviour are unchanged.
   Missing metadata must produce the explicit starts-directory error.

2. In the same container shell (with the environment variable still set):

   ```bash
   zero_agent --task Lerobot-So101-Teleop-Pick-Place-DR-Eval --num_steps 4500
   ```

   Pass: inspect several resets; tube-light exposure, camera pose/FOV and robot
   colour vary, and blue/red cubes both appear. Positions still come from the
   recorded starts. Selection is random, so adjacent resets may repeat a colour.
   The event-order output must place `eval_robot_color` after the material reset.
   HDRI and mat changes are not expected because those assets are absent.

3. Back on the host, run the complete five-episode evaluation:

   ```bash
   ./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --episodes 5
   docker ps -a --filter name=groot-eval-server --format '{{.Names}}'
   ```

   Pass: exactly five episode lines and the grouped report print; the reported
   JSON path exists, `complete` is true, and `overall.episodes` is 5. The Docker
   command lists no server left by this run. Repeat with `--dr --random_fraction
   0.25` and `--lang_by_color` from the README to check mixed starts and matching
   instructions. Ctrl+C or a failed eval must also remove this run's server.

## Local automated checks

Commands run from the repo root:

```bash
python -m unittest discover -s tests -v
python -m py_compile source/sim_to_real_so101/mdp/resets.py source/sim_to_real_so101/tasks/pick_place_env_cfg.py source/sim_to_real_so101/tasks/__init__.py source/sim_to_real_so101/scripts/lerobot_eval.py source/sim_to_real_so101/utils/pick_place_eval.py tests/test_pick_place_eval.py tests/test_eval_pick_place_script.py tests/test_lerobot_eval_rollout.py
python -m pyflakes source/sim_to_real_so101/mdp/resets.py source/sim_to_real_so101/tasks/pick_place_env_cfg.py source/sim_to_real_so101/tasks/__init__.py source/sim_to_real_so101/scripts/lerobot_eval.py source/sim_to_real_so101/utils/pick_place_eval.py tests/test_pick_place_eval.py tests/test_eval_pick_place_script.py tests/test_lerobot_eval_rollout.py
bash -n docker/eval_pick_place.sh
git diff --check
git update-index --add --chmod=+x docker/eval_pick_place.sh
```

Test output:

```text
test_auto_removed_server_uses_streamed_logs (test_eval_pick_place_script.HostScriptTests.test_auto_removed_server_uses_streamed_logs) ... ok
test_ctrl_c_cleans_up_server (test_eval_pick_place_script.HostScriptTests.test_ctrl_c_cleans_up_server) ... ok
test_dr_gui_language_is_literal (test_eval_pick_place_script.HostScriptTests.test_dr_gui_language_is_literal) ... ok
test_dry_runs_never_execute_docker (test_eval_pick_place_script.HostScriptTests.test_dry_runs_never_execute_docker) ... ok
test_eval_failure_cleans_up_and_preserves_status (test_eval_pick_place_script.HostScriptTests.test_eval_failure_cleans_up_and_preserves_status) ... ok
test_failed_server_logs_and_cleanup (test_eval_pick_place_script.HostScriptTests.test_failed_server_logs_and_cleanup) ... ok
test_path_escape_rejected (test_eval_pick_place_script.HostScriptTests.test_path_escape_rejected) ... ok
test_plain_success_cleanup_and_argv (test_eval_pick_place_script.HostScriptTests.test_plain_success_cleanup_and_argv) ... ok
test_autoreset_preserves_episode_metadata_and_exact_count (test_lerobot_eval_rollout.EvaluatorRolloutTests.test_autoreset_preserves_episode_metadata_and_exact_count) ... ok
test_closed_app_writes_partial_report_and_fails (test_lerobot_eval_rollout.EvaluatorRolloutTests.test_closed_app_writes_partial_report_and_fails) ... ok
test_base_world_roundtrip_and_table_support (test_pick_place_eval.RecordedStartTests.test_base_world_roundtrip_and_table_support) ... ok
test_color_instructions (test_pick_place_eval.RecordedStartTests.test_color_instructions) ... ok
test_cycle_coverage_and_reproducibility (test_pick_place_eval.RecordedStartTests.test_cycle_coverage_and_reproducibility) ... ok
test_frame_zero_only (test_pick_place_eval.RecordedStartTests.test_frame_zero_only) ... ok
test_invalid_sampling_options (test_pick_place_eval.RecordedStartTests.test_invalid_sampling_options) ... ok
test_missing_empty_malformed_and_units (test_pick_place_eval.RecordedStartTests.test_missing_empty_malformed_and_units) ... ok
test_overlap_known_cases (test_pick_place_eval.RecordedStartTests.test_overlap_known_cases) ... ok
test_random_cube_is_upright_on_live_table (test_pick_place_eval.RecordedStartTests.test_random_cube_is_upright_on_live_table) ... ok
test_random_inside_bounds_and_outside_rotated_box (test_pick_place_eval.RecordedStartTests.test_random_inside_bounds_and_outside_rotated_box) ... ok
test_random_recorded_uniform (test_pick_place_eval.RecordedStartTests.test_random_recorded_uniform) ... ok
test_results_json_schema_and_rates (test_pick_place_eval.RecordedStartTests.test_results_json_schema_and_rates) ... ok
test_zones_and_away_axis (test_pick_place_eval.RecordedStartTests.test_zones_and_away_axis) ... ok

----------------------------------------------------------------------
Ran 22 tests in 2.523s

OK
```

`py_compile`, `pyflakes`, `bash -n`, LF and whitespace checks passed. ShellCheck was not installed, so that optional check was skipped. The executable shell script was recorded with Git mode `100755`.

Additional CPU checks passed: live bound USD shader colour/metadata; the installed Isaac configclass and EventManager preparation code with simulator dependencies stubbed produced `reset_robot_position, reset_set_robot_visual_material, spawn_cube, reset_tracking, eval_robot_color`. AST/byte comparisons against HEAD confirm protected constants, geometry, non-Eval configs, original reset functions and task logic are unchanged. These checks do not validate GPU rendering.

## Dry run: plain

```bash
./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --dry_run
```

```text
# Preflight (checks printed, not executed)
+ docker info
+ docker inspect -f \{\{.State.Running\}\} teleop
+ docker image inspect real-robot
# Check checkpoint files under /root/models/so101_pick_place_v1/checkpoint-10000
+ docker exec teleop test -d /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta
# Verify host port 5555 is free; wait up to 300 seconds for server readiness
+ docker run -d --rm --name groot-eval-server-20260929T112318869224486-2421 --network host --privileged --gpus all -e DISPLAY -v /dev:/dev -v /run/udev:/run/udev:ro -v /root/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v /root/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration -v ./docker/env:/root/env -v /root/models:/workspace/models -v /workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100 real-robot python /Isaac-GR00T/gr00t/eval/run_gr00t_server.py --model-path /workspace/models/so101_pick_place_v1/checkpoint-10000 --port 5555
+ docker logs --follow groot-eval-server-20260929T112318869224486-2421
+ docker inspect -f \{\{.State.Running\}\} groot-eval-server-20260929T112318869224486-2421
# On failure/timeout: + docker logs groot-eval-server-20260929T112318869224486-2421
+ docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.scripts.lerobot_eval --task Lerobot-So101-Teleop-Pick-Place-Eval --num_envs 1 --cube_starts /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta --start_mode cycle --random_fraction 0.0 --rename_map \{\"realsense_rgb\":\ \"front\"\,\ \"wrist_cam\":\ \"wrist\"\} --action_horizon 16 --policy_host localhost --policy_port 5555 --lang_instruction Pick\ up\ the\ blue\ cube\ and\ place\ it\ in\ the\ white\ box --num_episodes 20 --checkpoint so101_pick_place_v1/checkpoint-10000 --results_json /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_nodr_20260929T112318869224486.json --headless
+ docker exec teleop cat /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_nodr_20260929T112318869224486.json
Results JSON (inside teleop): /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_nodr_20260929T112318869224486.json
+ docker stop groot-eval-server-20260929T112318869224486-2421
+ docker rm -f groot-eval-server-20260929T112318869224486-2421
```

## Dry run: dr

```bash
./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --dr --dry_run
```

```text
# Preflight (checks printed, not executed)
+ docker info
+ docker inspect -f \{\{.State.Running\}\} teleop
+ docker image inspect real-robot
# Check checkpoint files under /root/models/so101_pick_place_v1/checkpoint-10000
+ docker exec teleop test -d /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta
# Verify host port 5555 is free; wait up to 300 seconds for server readiness
+ docker run -d --rm --name groot-eval-server-20260929T112318909176202-2427 --network host --privileged --gpus all -e DISPLAY -v /dev:/dev -v /run/udev:/run/udev:ro -v /root/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v /root/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration -v ./docker/env:/root/env -v /root/models:/workspace/models -v /workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100 real-robot python /Isaac-GR00T/gr00t/eval/run_gr00t_server.py --model-path /workspace/models/so101_pick_place_v1/checkpoint-10000 --port 5555
+ docker logs --follow groot-eval-server-20260929T112318909176202-2427
+ docker inspect -f \{\{.State.Running\}\} groot-eval-server-20260929T112318909176202-2427
# On failure/timeout: + docker logs groot-eval-server-20260929T112318909176202-2427
+ docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.scripts.lerobot_eval --task Lerobot-So101-Teleop-Pick-Place-DR-Eval --num_envs 1 --cube_starts /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta --start_mode cycle --random_fraction 0.0 --rename_map \{\"realsense_rgb\":\ \"front\"\,\ \"wrist_cam\":\ \"wrist\"\} --action_horizon 16 --policy_host localhost --policy_port 5555 --lang_instruction Pick\ up\ the\ blue\ cube\ and\ place\ it\ in\ the\ white\ box --num_episodes 20 --checkpoint so101_pick_place_v1/checkpoint-10000 --results_json /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_dr_20260929T112318909176202.json --headless
+ docker exec teleop cat /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_dr_20260929T112318909176202.json
Results JSON (inside teleop): /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/../eval_results/so101_pick_place_v1_checkpoint-10000_dr_20260929T112318909176202.json
+ docker stop groot-eval-server-20260929T112318909176202-2427
+ docker rm -f groot-eval-server-20260929T112318909176202-2427
```

## Changed files

- `.gitignore`
- `docker/README.md`
- `docker/eval_pick_place.sh`
- `docker/eval_pick_place_validation.md`
- `source/sim_to_real_so101/mdp/resets.py`
- `source/sim_to_real_so101/tasks/pick_place_env_cfg.py`
- `source/sim_to_real_so101/tasks/__init__.py`
- `source/sim_to_real_so101/scripts/lerobot_eval.py`
- `source/sim_to_real_so101/utils/pick_place_eval.py`
- `tests/test_pick_place_eval.py`
- `tests/test_eval_pick_place_script.py`
- `tests/test_lerobot_eval_rollout.py`
