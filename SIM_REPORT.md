# v3 simulator verification

Verified on 2026-11-14. **Overall: FAIL — the v3 equivalence gate is not satisfied.**

- **Six new test failures:** stale `from sim.datagen import spawner` imports at `tests/sim/test_sim_real.py:212` and `:274`; all six passed in Phase 0.
- **Eval smoke cannot complete:** live geometry differs from the stored eval set at `base_position_world` (`src/sim/eval/scene.py:95`), aborting before the first of nine selected episodes. Stage rates are unavailable; no geometry was recaptured or bypassed.
- **Scripted generation differs between commits.** Per-attempt outcomes and available same-seed array comparisons are below. Several recipes did not produce the requested two accepted episodes; equivalence is unverified wherever a matching accepted array is absent.
- **One shared baseline/v3 test failure:** the provenance fallback test expects `unknown` despite the required `S2R_GIT_COMMIT` environment variable (`tests/utils/test_provenance.py:36`; baseline `tests/core/test_provenance.py:36`).
- **Cfg textual differences:** every changed key is listed below; all differences are strings (paths, module references and hook names/IDs). No numeric cfg value differs. Changes beyond a simple `workspace/` prefix are explicitly retained as findings.
- Runtime construction/reset/20-step probes passed for all seven SO-101 variants and SP7 pick-place teleop. `utils` resolves to the repository before and after Kit. The seven requested help/import checks passed.

## Provenance and execution conditions

Tested v3 commit: `f423b29b8752590308cebf7716a6b40e203ecdbb`. Phase 0 commit: `45b07310ee8f18e6c6f85a2d36e1993082e8a1a1`. Report branch: `v3-sim-checks`, branched directly from the tested v3 commit.

Simulation image: `sha256:cfb85d8696a20d44965c561f63b49c0f5c7d9ab8dc5c4af4536d75710cb2a9a7`. Serving image: `sha256:39b295c24e22e9ae4f054bdc26e2bf268c65306e8392280c95e0f35b34d4563c`. **No image was rebuilt.** Compose selected its existing local `s2r-sim:latest`; the previously running sim container used image `sha256:42d5e49d762a74db44db29780b0a4f08452fe6ffa3c05429bc01830c8e03a7d8`. Every baseline and v3 test here used the same selected `cfb85d…` image.

GPU: RTX 4090; driver 580.173.02. Python 3.12.14, Torch 2.11.0+cu130, Isaac Lab 28.0.0, NumPy 2.5.1, pytest 9.1.1. Container working directory: `/workspace/s2r`. The existing container used the hardware overlay, so it was retained for sim startup. Simulator jobs ran headless and sequentially. The example policy runs on CPU and remained available during checks.

The user authorized stopping `s2r-serve-molmoact2-5556`; Docker removed it when it stopped. The user subsequently said **no need to restart it**. No source, physics or asset file was edited. The Phase 0 capture script ran unchanged; there is no fixed version or change list to supply. Scratch harnesses and recipe copies are preserved as JSON text in `verification_harness.json` and `recipes.json`; no dataset, video or image is committed.

## Setup — PASS

Initial commands were `git fetch origin`, then `git checkout v3 && git pull && git rev-parse HEAD` (already up to date; commit above). Read `CLAUDE.md` and `RESTRUCTURE.md` on v3. The initial worktree had no tracked changes; local benchmark outputs were preserved.

```bash
S2R_GIT_COMMIT=$(git rev-parse HEAD) docker compose -f compose.yaml -f compose.hardware.yaml up -d sim
```
**PASS** (command exit 0); full output: [commands/compose_v3.json](tests/fixtures/v3_baseline/sim/commands/compose_v3.json).

```bash
docker inspect s2r-sim --format '{{.Image}} {{json .Config.Env}}'
```
**PASS** (command exit 0); full output: [commands/container_identity.json](tests/fixtures/v3_baseline/sim/commands/container_identity.json).

```bash
docker stop s2r-serve-molmoact2-5556
```
**PASS** (command exit 0); full output: [commands/stop_existing_gpu_server.json](tests/fixtures/v3_baseline/sim/commands/stop_existing_gpu_server.json).

```bash
git checkout 45b0731 && git rev-parse HEAD
```
**PASS** (command exit 0); full output: [commands/checkout_phase0.json](tests/fixtures/v3_baseline/sim/commands/checkout_phase0.json).

## Part A: Phase 0 baseline

### Capture — PASS; baseline suite — FAIL

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 s2r-sim s2r-entrypoint python tests/fixtures/v3_baseline/capture_sim_baseline.py --out /workspace/s2r/_sim_baseline
```
**PASS** (command exit 0); full output: [commands/baseline_capture.json](tests/fixtures/v3_baseline/sim/commands/baseline_capture.json).

All seven cfg dumps were written without capture errors. Baseline suite: **636 passed, 1 failed, 8 skipped**, plus 22 passed subtests. Its internal exact pytest command is recorded in `pytest_outcomes.json` (the temporary JUnit path belongs to that invocation). The failed test was rerun once to retain the full failure text:

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 s2r-sim s2r-entrypoint env PYTHONPATH=src python -m pytest tests/core/test_provenance.py::test_missing_repository_has_fallback_or_explicitly_required_commit -q -p no:cacheprovider -rA
```
**FAIL** (command exit 1); full output: [commands/baseline_failure_detail.json](tests/fixtures/v3_baseline/sim/commands/baseline_failure_detail.json).

```text
F                                                                        [100%]
=================================== FAILURES ===================================
______ test_missing_repository_has_fallback_or_explicitly_required_commit ______

tmp_path = PosixPath('/tmp/pytest-of-root/pytest-1/test_missing_repository_has_fa0')

    def test_missing_repository_has_fallback_or_explicitly_required_commit(tmp_path):
>       assert git_commit(tmp_path) == "unknown"
E       AssertionError: assert '45b07310ee8f...1993082e8a1a1' == 'unknown'
E         
E         - unknown
E         + 45b07310ee8f18e6c6f85a2d36e1993082e8a1a1

tests/core/test_provenance.py:36: AssertionError
=========================== short test summary info ============================
FAILED tests/core/test_provenance.py::test_missing_repository_has_fallback_or_explicitly_required_commit
1 failed in 0.04s
```

### Reference recipes — FAIL (insufficient accepted episodes)

All recipes used seed 101, random starts, attempt limit 10, target 2 accepted episodes, and headless mode. Settings otherwise come from the named repository recipe. For bowls, `starts_dir` was removed only in a scratch YAML copy to meet the requested random-start condition. Baseline used scratch YAML copies for all three recipes; v3 used the stock pick-place and cube recipes directly and the equivalent scratch bowls recipe. Repository YAMLs were unchanged. Full recipe text is in `recipes.json`.

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path outputs/v3-sim-checks/baseline/scripted-random.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/baseline/datasets/scripted --repo_id local/v3_sim_scripted --report /workspace/s2r/outputs/v3-sim-checks/baseline/scripted-report.json
```
**FAIL** (command exit 1); full output: [commands/baseline_datagen_scripted.json](tests/fixtures/v3_baseline/sim/commands/baseline_datagen_scripted.json).

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_two_cubes s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path outputs/v3-sim-checks/baseline/stack_two_cubes-random.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/baseline/datasets/stack_two_cubes --repo_id local/v3_sim_stack_two_cubes --report /workspace/s2r/outputs/v3-sim-checks/baseline/stack_two_cubes-report.json
```
**FAIL** (command exit 1); full output: [commands/baseline_datagen_stack_two_cubes.json](tests/fixtures/v3_baseline/sim/commands/baseline_datagen_stack_two_cubes.json).

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_bowls s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path outputs/v3-sim-checks/baseline/stack_bowls-random.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/baseline/datasets/stack_bowls --repo_id local/v3_sim_stack_bowls --report /workspace/s2r/outputs/v3-sim-checks/baseline/stack_bowls-report.json
```
**FAIL** (command exit 1); full output: [commands/baseline_datagen_stack_bowls.json](tests/fixtures/v3_baseline/sim/commands/baseline_datagen_stack_bowls.json).

| Recipe | Baseline attempts | Baseline accepted | Requested |
| --- | --- | --- | --- |
| scripted | 10 | 0 | 2 |
| stack_two_cubes | 10 | 0 | 2 |
| stack_bowls | 10 | 1 | 2 |

No Python traceback accompanied these recipe exit codes: they return 1 when the requested accepted-episode target is not met (Phase 0 `src/sim/datagen/scripted.py:609`; v3 criterion at `src/sim/datagen/scripted.py:608`, return at `:614`). The exact rejection strings for every attempt are listed with Part B; full JSON reports and command outputs are retained. No failures were fixed.

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 s2r-sim s2r-entrypoint python /tmp/v3_hash_arrays.py --phase baseline
```
**PASS** (command exit 0); full output: [commands/baseline_array_hashes.json](tests/fixtures/v3_baseline/sim/commands/baseline_array_hashes.json).

PyArrow read the recorded `action` and `observation.state` columns, grouped by `episode_index`, ordered by `frame_index`, flattened in their stored primitive dtype, and SHA-256 hashed as contiguous C-order bytes. Videos are excluded. Every accepted array found is listed below; zero accepted episodes means no recorded accepted array exists to hash.

### Return to v3 / baseline copy — PASS

```bash
git checkout v3 && git rev-parse HEAD
```
**PASS** (command exit 0); full output: [commands/checkout_v3.json](tests/fixtures/v3_baseline/sim/commands/checkout_v3.json).

```bash
git checkout -b v3-sim-checks
```
**PASS** (command exit 0); full output: [commands/create_report_branch.json](tests/fixtures/v3_baseline/sim/commands/create_report_branch.json).

```bash
S2R_GIT_COMMIT=$(git rev-parse HEAD) docker compose -f compose.yaml -f compose.hardware.yaml up -d sim
```
**PASS** (command exit 0); full output: [commands/compose_v3_recheck.json](tests/fixtures/v3_baseline/sim/commands/compose_v3_recheck.json).

Copied `_sim_baseline/.` to `tests/fixtures/v3_baseline/sim/` with `mkdir -p tests/fixtures/v3_baseline/sim` and `cp -r _sim_baseline/. tests/fixtures/v3_baseline/sim/`.

## Part B: v3 verification

### B1: utils shadowing — PASS

```bash
docker exec s2r-sim s2r-entrypoint python -c "import importlib.util, utils; print(utils.__file__); print(importlib.util.find_spec(\"utils\").origin)"
```
**PASS** (command exit 0); full output: [commands/shadow_before_kit.json](tests/fixtures/v3_baseline/sim/commands/shadow_before_kit.json).

Both pre-Kit lookups printed `/workspace/s2r/src/utils/__init__.py`. Every runtime probe printed `UTILS_AFTER_KIT=/workspace/s2r/src/utils/__init__.py` and saved the same origin. There was no observed shadowing; this check does not require a package rename.

### B2: full suite — FAIL

```bash
docker exec s2r-sim s2r-entrypoint env PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider -rA --junitxml=/workspace/s2r/_v3_pytest.xml
```
**FAIL** (command exit 1); full output: [commands/v3_pytest.json](tests/fixtures/v3_baseline/sim/commands/v3_pytest.json).

| Run | Passed | Failed | Skipped | Total |
| --- | --- | --- | --- | --- |
| Phase 0 | 636 | 1 | 8 | 645 |
| v3 | 585 | 7 | 6 | 598 |

Matched by **file name plus complete class/test ID**, including parameter IDs. Excluded deliberate removals: all baseline `tests/cli/`, `test_check.py`, `test_sweep.py::test_gap_*`, and `test_stack_bowls.py::test_terminal_stability_*`. **62 removed IDs** were excluded. There are **6 previously passing tests that no longer pass** and **6 newly failing tests**; both lists contain the same six IDs:

- `test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[False-False]` — passed → failed
- `test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[False-True]` — passed → failed
- `test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[True-False]` — passed → failed
- `test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[False-False-False]` — passed → failed
- `test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[False-True-True]` — passed → failed
- `test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[True-False-True]` — passed → failed

The 15 additional IDs in `test_geometry.py`, `test_hashing.py`, `test_io.py`, and `test_workspace.py` passed. `test_import_graph.py` and `test_eval_set_regression.py` already existed at Phase 0; their v3 tests passed. Skips remain the opt-in live URDF/mimic/parallel-gripper acceptance tests (they require `S2R_RUN_IMPORT_TESTS=1`); they were not silently counted as passes.

Full failure text for all seven v3 failures follows. Each block includes its exact file:line:

#### tests/sim/test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[False-False-False]

```text
no_object = False, spawn = False, connects = False
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_trial_loop_requires_spawn0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdbaf860>

    @pytest.mark.parametrize('no_object,spawn,connects', [(False, False, False), (False, True, True), (True, False, True)])
    def test_trial_loop_requires_spawn_unless_opted_out(no_object, spawn, connects, tmp_path, monkeypatch):
        """Exercise actual Start gating with fake hardware and no simulator launch."""
        import sys
        import torch
        import gymnasium
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from utils.arms.arms import load_arm_spec
        from sim.eval.sim_real.trials import parse_args, run
        from core.rollout import real_rollout
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:212: ImportError
```

#### tests/sim/test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[False-True-True]

```text
no_object = False, spawn = True, connects = True
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_trial_loop_requires_spawn1')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdbada60>

    @pytest.mark.parametrize('no_object,spawn,connects', [(False, False, False), (False, True, True), (True, False, True)])
    def test_trial_loop_requires_spawn_unless_opted_out(no_object, spawn, connects, tmp_path, monkeypatch):
        """Exercise actual Start gating with fake hardware and no simulator launch."""
        import sys
        import torch
        import gymnasium
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from utils.arms.arms import load_arm_spec
        from sim.eval.sim_real.trials import parse_args, run
        from core.rollout import real_rollout
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:212: ImportError
```

#### tests/sim/test_sim_real.py::test_trial_loop_requires_spawn_unless_opted_out[True-False-True]

```text
no_object = True, spawn = False, connects = True
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_trial_loop_requires_spawn2')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdbadbb0>

    @pytest.mark.parametrize('no_object,spawn,connects', [(False, False, False), (False, True, True), (True, False, True)])
    def test_trial_loop_requires_spawn_unless_opted_out(no_object, spawn, connects, tmp_path, monkeypatch):
        """Exercise actual Start gating with fake hardware and no simulator launch."""
        import sys
        import torch
        import gymnasium
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from utils.arms.arms import load_arm_spec
        from sim.eval.sim_real.trials import parse_args, run
        from core.rollout import real_rollout
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:212: ImportError
```

#### tests/sim/test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[False-False]

```text
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_both_trials_use_the_submi0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdbacbc0>
cancel_inference = False, operator_end = False

    @pytest.mark.parametrize('cancel_inference,operator_end', [(False,False),(True,False),(False,True)])
    def test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration(tmp_path, monkeypatch, cancel_inference,operator_end):
        """Drive a complete real -> sim comparison with injected devices and policies."""
        import sys
        import threading
        import torch
        import gymnasium
        from utils.arms.arms import load_arm_spec
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from core.rollout import real_rollout, paired_gui
        from sim.eval import sim_real
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:274: ImportError
```

#### tests/sim/test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[True-False]

```text
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_both_trials_use_the_submi1')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdbac380>
cancel_inference = True, operator_end = False

    @pytest.mark.parametrize('cancel_inference,operator_end', [(False,False),(True,False),(False,True)])
    def test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration(tmp_path, monkeypatch, cancel_inference,operator_end):
        """Drive a complete real -> sim comparison with injected devices and policies."""
        import sys
        import threading
        import torch
        import gymnasium
        from utils.arms.arms import load_arm_spec
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from core.rollout import real_rollout, paired_gui
        from sim.eval import sim_real
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:274: ImportError
```

#### tests/sim/test_sim_real.py::test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration[False-True]

```text
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_both_trials_use_the_submi2')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d0ecdb626f0>
cancel_inference = False, operator_end = True

    @pytest.mark.parametrize('cancel_inference,operator_end', [(False,False),(True,False),(False,True)])
    def test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration(tmp_path, monkeypatch, cancel_inference,operator_end):
        """Drive a complete real -> sim comparison with injected devices and policies."""
        import sys
        import threading
        import torch
        import gymnasium
        from utils.arms.arms import load_arm_spec
        from utils.arms.calibration import resolve_calibration
        from utils.arms.mapping import JointMapping
        from core.rollout import real_rollout, paired_gui
        from sim.eval import sim_real
        from sim.teleop import agent, loop
>       from sim.datagen import spawner
E       ImportError: cannot import name 'spawner' from 'sim.datagen' (/workspace/s2r/src/sim/datagen/__init__.py)

tests/sim/test_sim_real.py:274: ImportError
```

#### tests/utils/test_provenance.py::test_missing_repository_has_fallback_or_explicitly_required_commit

```text
tmp_path = PosixPath('/tmp/pytest-of-root/pytest-2/test_missing_repository_has_fa0')

    def test_missing_repository_has_fallback_or_explicitly_required_commit(tmp_path):
>       assert git_commit(tmp_path) == "unknown"
E       AssertionError: assert 'f423b29b8752...6b40e203ecdbb' == 'unknown'
E         
E         - unknown
E         + f423b29b8752590308cebf7716a6b40e203ecdbb

tests/utils/test_provenance.py:36: AssertionError
```

### B3: construction/reset/20 zero-action steps — PASS; cfg literal comparison — FAIL beyond permitted prefix-only changes

Each row ran in a fresh headless process through `sim.launch`, selected SO-101 / rpd_meeting_room / the named task, used `parse_env_cfg`, called `gym.make`, reset once, stepped 20 times with zero actions, then closed. SP7 repeated pick-place teleop. The full executable harness is saved in `verification_harness.json` (`env_probe.py`).

| Arm | Task | Variant | Result | Reset | Steps | utils after Kit |
| --- | --- | --- | --- | --- | --- | --- |
| so101 | pick_place_cube | eval | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | pick_place_cube | eval_dr | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | pick_place_cube | teleop | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | stack_bowls | eval | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | stack_bowls | teleop | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | stack_two_cubes | eval | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| so101 | stack_two_cubes | teleop | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |
| sp7 | pick_place_cube | teleop | PASS | True | 20 | /workspace/s2r/src/utils/__init__.py |

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task pick_place_cube --variant eval --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__pick_place_cube__eval.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_pick_place_cube_eval.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_pick_place_cube_eval.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task pick_place_cube --variant eval_dr --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__pick_place_cube__eval_dr.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_pick_place_cube_eval_dr.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_pick_place_cube_eval_dr.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task pick_place_cube --variant teleop --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__pick_place_cube__teleop.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_pick_place_cube_teleop.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_pick_place_cube_teleop.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_bowls s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task stack_bowls --variant eval --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__stack_bowls__eval.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_stack_bowls_eval.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_stack_bowls_eval.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_bowls s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task stack_bowls --variant teleop --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__stack_bowls__teleop.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_stack_bowls_teleop.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_stack_bowls_teleop.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_two_cubes s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task stack_two_cubes --variant eval --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__stack_two_cubes__eval.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_stack_two_cubes_eval.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_stack_two_cubes_eval.json).

```bash
docker exec -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_two_cubes s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task stack_two_cubes --variant teleop --arm so101 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/so101__stack_two_cubes__teleop.json
```
**PASS** (command exit 0); full output: [commands/v3_env_so101_stack_two_cubes_teleop.json](tests/fixtures/v3_baseline/sim/commands/v3_env_so101_stack_two_cubes_teleop.json).

```bash
docker exec -e S2R_ARM=sp7 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python /tmp/v3_env_probe.py --task pick_place_cube --variant teleop --arm sp7 --out /workspace/s2r/outputs/v3-sim-checks/v3/env_cfg/sp7__pick_place_cube__teleop.json
```
**PASS** (command exit 0); full output: [commands/v3_env_sp7_pick_place_cube_teleop.json](tests/fixtures/v3_baseline/sim/commands/v3_env_sp7_pick_place_cube_teleop.json).

All seven SO-101 cfgs were compared to their matching Phase 0 dump. **366 differing leaf keys: 156 simple `workspace/` path-prefix changes and 210 other string changes**. The latter are module/class/callback references, hook filenames and path-derived hook module IDs; they are listed as findings under the requested strict comparison rule. There are no changed numeric/non-string values. SP7 had no Phase 0 cfg reference and was checked only for construction/reset/steps.

#### Cfg differences: pick_place_cube__eval

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.check_eval_order.func | `"sim.envs.mdp.starts:assert_reset_event_order"` | `"sim.workspace.envs.mdp.starts:assert_reset_event_order"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.starts:reset_from_starts"` | `"sim.workspace.envs.mdp.starts:reset_from_starts"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.white_box.spawn.usd_path | `"/workspace/s2r/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | `"/workspace/s2r/workspace/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: pick_place_cube__eval_dr

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.check_eval_order.func | `"sim.envs.mdp.starts:assert_reset_event_order"` | `"sim.workspace.envs.mdp.starts:assert_reset_event_order"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.eval_robot_color.func | `"sim.envs.mdp.appearance:randomize_robot_color"` | `"sim.workspace.envs.mdp.appearance:randomize_robot_color"` | False |
| events.eval_room_light.func | `"sim.scene.appearance:randomize_scene_light_exposure"` | `"sim.workspace.scene.appearance:randomize_scene_light_exposure"` | False |
| events.reset_from_starts.func | `"sim.envs.mdp.starts:reset_from_starts"` | `"sim.workspace.envs.mdp.starts:reset_from_starts"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.white_box.spawn.usd_path | `"/workspace/s2r/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | `"/workspace/s2r/workspace/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: pick_place_cube__teleop

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.placement:park_object"` | `"sim.workspace.envs.mdp.placement:park_object"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.depth_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.depth_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.realsense_depth.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.realsense_depth.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.realsense_depth.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.white_box.spawn.usd_path | `"/workspace/s2r/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | `"/workspace/s2r/workspace/tasks/pick_place_cube/usd/White-Box-Physics.usda"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: stack_bowls__eval

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.check_eval_order.func | `"sim.envs.mdp.starts:assert_reset_event_order"` | `"sim.workspace.envs.mdp.starts:assert_reset_event_order"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.starts:reset_from_starts"` | `"sim.workspace.envs.mdp.starts:reset_from_starts"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.blue_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.red_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.yellow_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: stack_bowls__teleop

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.check_eval_order.func | `"sim.envs.mdp.starts:assert_reset_event_order"` | `"sim.workspace.envs.mdp.starts:assert_reset_event_order"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.starts:reset_from_starts"` | `"sim.workspace.envs.mdp.starts:reset_from_starts"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.depth_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.depth_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.blue_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.realsense_depth.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.realsense_depth.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.realsense_depth.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.red_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.yellow_bowl.spawn.usd_path | `"/workspace/s2r/tasks/stack_bowls/assets/bowl.usd"` | `"/workspace/s2r/workspace/tasks/stack_bowls/assets/bowl.usd"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: stack_two_cubes__eval

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.check_eval_order.func | `"sim.envs.mdp.starts:assert_reset_event_order"` | `"sim.workspace.envs.mdp.starts:assert_reset_event_order"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.starts:reset_from_starts"` | `"sim.workspace.envs.mdp.starts:reset_from_starts"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

#### Cfg differences: stack_two_cubes__teleop

| Key | Baseline JSON value | v3 JSON value | Simple prefix only |
| --- | --- | --- | --- |
| arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| events.cache_static_geometry.func | `"sim.scene.geometry:cache_scene_geometry"` | `"sim.workspace.scene.geometry:cache_scene_geometry"` | False |
| events.configure_scene_updates.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:configure_scene_updates"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:configure_scene_updates"` | False |
| events.configure_scene_updates.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.configure_scene_updates.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.configure_scene_updates.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.configure_scene_updates.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.reset_from_starts.func | `"sim.envs.mdp.placement:park_object"` | `"sim.workspace.envs.mdp.placement:park_object"` | False |
| events.reset_set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.reset_set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.reset_set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| events.runtime_joint_limits.func | `"sim.envs.mdp.limits:apply_runtime_joint_limits"` | `"sim.workspace.envs.mdp.limits:apply_runtime_joint_limits"` | False |
| events.set_robot_visual_material.func | `"sim.envs.scene_builder:set_robot_materials"` | `"sim.workspace.envs.scene_builder:set_robot_materials"` | False |
| events.set_robot_visual_material.params.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| events.set_robot_visual_material.params.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| events.set_robot_visual_material.params.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| events.set_robot_visual_material.params.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| observations.policy.ee_frame_state.func | `"sim.envs.mdp.obs:ee_frame_state"` | `"sim.workspace.envs.mdp.obs:ee_frame_state"` | False |
| observations.visual.depth_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.depth_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_realsense_rgb.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.instance_id_seg_wrist_cam.func | `"sim.envs.mdp.obs:image_raw"` | `"sim.workspace.envs.mdp.obs:image_raw"` | False |
| observations.visual.rgb_realsense_rgb.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| observations.visual.rgb_wrist_cam.func | `"sim.envs.mdp.obs:image_rgb_raw"` | `"sim.workspace.envs.mdp.obs:image_rgb_raw"` | False |
| scene.camera_realsense_rgb.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_realsense_rgb.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_realsense_rgb.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.camera_wrist_cam.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.camera_wrist_cam.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.camera_wrist_cam.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.realsense_depth.class_type | `"sim.scene.cameras:CalibratedTiledCamera"` | `"sim.workspace.scene.cameras:CalibratedTiledCamera"` | False |
| scene.realsense_depth.renderer_cfg.class_type | `"sim.scene.renderer:CalibratedRtxRenderer"` | `"sim.workspace.scene.renderer:CalibratedRtxRenderer"` | False |
| scene.realsense_depth.spawn.func | `"sim.scene.cameras:_spawn_named_camera"` | `"sim.workspace.scene.cameras:_spawn_named_camera"` | False |
| scene.robot.spawn.arm_spec.directory | `"/workspace/s2r/arms/so101"` | `"/workspace/s2r/workspace/arms/so101"` | True |
| scene.robot.spawn.arm_spec.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.robot.spawn.arm_spec.utils | `"/workspace/s2r/arms/so101/so101_utils.py"` | `"/workspace/s2r/workspace/arms/so101/utils.py"` | False |
| scene.robot.spawn.func | `"sim.scene.arm:_spawn_arm"` | `"sim.workspace.scene.arm:_spawn_arm"` | False |
| scene.robot.spawn.usd_path | `"/workspace/s2r/arms/so101/usd/SO-ARM101-USD.usd"` | `"/workspace/s2r/workspace/arms/so101/usd/SO-ARM101-USD.usd"` | True |
| scene.room.spawn.func | `"_s2r_scene_hook_f9414b32f8e2f9819b7a43ced370f854a35532d67f4d413a2709e541fd65d734:spawn_room"` | `"_s2r_scene_hook_a14569d14b349f72d8dffda3bab985d3c1d6b1532683abbb633a6b71976afe93:spawn_room"` | False |
| scene.room.spawn.scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene.room.spawn.scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene.room.spawn.scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene.room.spawn.scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene.room.spawn.usd_path | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| scene_spec.appearance_path | `"/workspace/s2r/scenes/rpd_meeting_room/appearance.json"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/appearance.json"` | True |
| scene_spec.directory | `"/workspace/s2r/scenes/rpd_meeting_room"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room"` | True |
| scene_spec.hook | `"scene.py:customize"` | `"utils.py:customize"` | False |
| scene_spec.room_usd | `"/workspace/s2r/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | `"/workspace/s2r/workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"` | True |
| sim.default_visualizer_cfg.class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |
| sim.visualizer_cfgs[0].class_type | `"sim.scene.viewport:CalibratedKitVisualizer"` | `"sim.workspace.scene.viewport:CalibratedKitVisualizer"` | False |

### B4: scripted equivalence — FAIL

```bash
docker exec -e S2R_GIT_COMMIT=f423b29b8752590308cebf7716a6b40e203ecdbb -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=pick_place_cube s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path workspace/configs/datagen/scripted.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/v3/datasets/scripted --repo_id local/v3_sim_scripted --report /workspace/s2r/outputs/v3-sim-checks/v3/scripted-report.json
```
**FAIL** (command exit 1); full output: [commands/v3_datagen_scripted.json](tests/fixtures/v3_baseline/sim/commands/v3_datagen_scripted.json).

```bash
docker exec -e S2R_GIT_COMMIT=f423b29b8752590308cebf7716a6b40e203ecdbb -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_two_cubes s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path workspace/configs/datagen/stack_two_cubes.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/v3/datasets/stack_two_cubes --repo_id local/v3_sim_stack_two_cubes --report /workspace/s2r/outputs/v3-sim-checks/v3/stack_two_cubes-report.json
```
**FAIL** (command exit 1); full output: [commands/v3_datagen_stack_two_cubes.json](tests/fixtures/v3_baseline/sim/commands/v3_datagen_stack_two_cubes.json).

```bash
docker exec -e S2R_GIT_COMMIT=f423b29b8752590308cebf7716a6b40e203ecdbb -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_bowls s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path outputs/v3-sim-checks/v3/stack_bowls-random.yaml --seed 101 --attempts 10 --episodes 2 --headless --destination /workspace/s2r/outputs/v3-sim-checks/v3/datasets/stack_bowls --repo_id local/v3_sim_stack_bowls --report /workspace/s2r/outputs/v3-sim-checks/v3/stack_bowls-report.json
```
**PASS** (command exit 0); full output: [commands/v3_datagen_stack_bowls.json](tests/fixtures/v3_baseline/sim/commands/v3_datagen_stack_bowls.json).

```bash
docker exec -e S2R_GIT_COMMIT=f423b29b8752590308cebf7716a6b40e203ecdbb -e S2R_ARM=so101 -e S2R_SCENE=rpd_meeting_room -e S2R_TASK=stack_bowls s2r-sim s2r-entrypoint python -m sim.datagen.scripted --config_path outputs/v3-sim-checks/v3/stack_bowls-random.yaml --seed 101 --attempts 10 --episodes 0 --headless --destination /workspace/s2r/outputs/v3-sim-checks/v3_all10/datasets/stack_bowls --repo_id local/v3_sim_stack_bowls --report /workspace/s2r/outputs/v3-sim-checks/v3_all10/stack_bowls-report.json
```
**PASS** (command exit 0); full output: [commands/v3_datagen_stack_bowls_all10.json](tests/fixtures/v3_baseline/sim/commands/v3_datagen_stack_bowls_all10.json).

The initial v3 bowls run reached two accepted episodes and stopped after seven attempts. It is retained in `datagen/v3/`. A separate unchanged-recipe run with `--episodes 0` completed all ten seed-101-through-110 attempts in `datagen/v3_all10/`, matching the baseline attempt coverage. The bowls outcome comparison below uses that full ten-attempt run. `--episodes 0` changes only the stopping rule, and no source or physical parameter was changed.

| Recipe | Baseline attempts / accepted | v3 attempts / accepted | Outcome differences | Array comparison |
| --- | --- | --- | --- | --- |
| scripted | 10 / 0 | 10 / 1 | 10 | UNVERIFIED: no accepted episodes in one or both runs |
| stack_two_cubes | 10 / 0 | 10 / 0 | 17 | UNVERIFIED: no accepted episodes in one or both runs |
| stack_bowls | 10 / 1 | 10 / 2 | 33 | FAIL: array hashes differ |

The initial commands retain the same seed, attempt limit and accepted-episode target; the supplemental all-ten bowls run disables early stopping. Random-start bowls uses a scratch copy with `starts_dir` removed in both runs. Per-attempt outcomes, including every exact rejection/failure string:

#### Attempts: scripted

| Run | Attempt (0-based) | Seed | Accepted | Frames | Exact reason | Exact failure mode |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 101 | False | 492 | expert_finished_without_success | dropped_before_box |
| baseline | 1 | 102 | False | 240 | waypoint_timeout:approach | never_reached |
| baseline | 2 | 103 | False | 300 | waypoint_timeout:descend | missed_grasp |
| baseline | 3 | 104 | False | 506 | expert_finished_without_success | dropped_before_box |
| baseline | 4 | 105 | False | 240 | waypoint_timeout:approach | missed_grasp |
| baseline | 5 | 106 | False | 484 | expert_finished_without_success | dropped_before_box |
| baseline | 6 | 107 | False | 300 | waypoint_timeout:descend | missed_grasp |
| baseline | 7 | 108 | False | 499 | expert_finished_without_success | dropped_before_box |
| baseline | 8 | 109 | False | 240 | waypoint_timeout:approach | missed_grasp |
| baseline | 9 | 110 | False | 496 | expert_finished_without_success | dropped_before_box |
| v3 | 0 | 101 | True | 391 | task_success | None |
| v3 | 1 | 102 | False | 240 | waypoint_timeout:approach | never_reached |
| v3 | 2 | 103 | False | 300 | waypoint_timeout:descend | missed_grasp |
| v3 | 3 | 104 | False | 502 | expert_finished_without_success | dropped_before_box |
| v3 | 4 | 105 | False | 240 | waypoint_timeout:approach | missed_grasp |
| v3 | 5 | 106 | False | 484 | expert_finished_without_success | dropped_before_box |
| v3 | 6 | 107 | False | 300 | waypoint_timeout:descend | missed_grasp |
| v3 | 7 | 108 | False | 499 | expert_finished_without_success | dropped_before_box |
| v3 | 8 | 109 | False | 240 | waypoint_timeout:approach | missed_grasp |
| v3 | 9 | 110 | False | 496 | expert_finished_without_success | dropped_before_box |

Outcome differences:

```json
[
  {
    "attempt": 0,
    "field": "accepted",
    "baseline": false,
    "v3": true
  },
  {
    "attempt": 0,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "task_success"
  },
  {
    "attempt": 0,
    "field": "failure_mode",
    "baseline": "dropped_before_box",
    "v3": null
  },
  {
    "attempt": 0,
    "field": "frames",
    "baseline": 492,
    "v3": 391
  },
  {
    "attempt": 0,
    "field": "steps",
    "baseline": 492,
    "v3": 391
  },
  {
    "attempt": 0,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_box": false,
      "placed": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_box": true,
      "placed": true,
      "success": true
    }
  },
  {
    "attempt": 0,
    "field": "latched_success",
    "baseline": false,
    "v3": true
  },
  {
    "attempt": 0,
    "field": "final_success",
    "baseline": false,
    "v3": true
  },
  {
    "attempt": 3,
    "field": "frames",
    "baseline": 506,
    "v3": 502
  },
  {
    "attempt": 3,
    "field": "steps",
    "baseline": 506,
    "v3": 502
  }
]
```

Same-seed accepted-array comparisons (step indices are zero-based):

```json
[]
```

#### Attempts: stack_two_cubes

| Run | Attempt (0-based) | Seed | Accepted | Frames | Exact reason | Exact failure mode |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 101 | False | 444 | waypoint_timeout:transport | timeout_holding |
| baseline | 1 | 102 | False | 300 | waypoint_timeout:descend | missed_grasp |
| baseline | 2 | 103 | False | 399 | expert_finished_without_success | dropped_before_target |
| baseline | 3 | 104 | False | 462 | waypoint_timeout:lower | timeout_holding |
| baseline | 4 | 105 | False | 399 | expert_finished_without_success | dropped_before_target |
| baseline | 5 | 106 | False | 492 | waypoint_timeout:lower | dropped_before_target |
| baseline | 6 | 107 | False | 466 | waypoint_timeout:lower | timeout_holding |
| baseline | 7 | 108 | False | 459 | waypoint_timeout:lower | timeout_holding |
| baseline | 8 | 109 | False | 300 | waypoint_timeout:descend | missed_grasp |
| baseline | 9 | 110 | False | 459 | waypoint_timeout:lower | timeout_holding |
| v3 | 0 | 101 | False | 444 | waypoint_timeout:transport | timeout_holding |
| v3 | 1 | 102 | False | 300 | waypoint_timeout:descend | missed_grasp |
| v3 | 2 | 103 | False | 408 | expert_finished_without_success | dropped_before_target |
| v3 | 3 | 104 | False | 461 | waypoint_timeout:lower | dropped_before_target |
| v3 | 4 | 105 | False | 459 | waypoint_timeout:lower | timeout_holding |
| v3 | 5 | 106 | False | 492 | waypoint_timeout:lower | timeout_holding |
| v3 | 6 | 107 | False | 466 | waypoint_timeout:lower | timeout_holding |
| v3 | 7 | 108 | False | 459 | waypoint_timeout:lower | stacked_not_confirmed |
| v3 | 8 | 109 | False | 300 | waypoint_timeout:descend | missed_grasp |
| v3 | 9 | 110 | False | 399 | expert_finished_without_success | dropped_before_target |

Outcome differences:

```json
[
  {
    "attempt": 0,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_target": true,
      "stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_target": false,
      "stacked": false,
      "success": false
    }
  },
  {
    "attempt": 2,
    "field": "frames",
    "baseline": 399,
    "v3": 408
  },
  {
    "attempt": 2,
    "field": "steps",
    "baseline": 399,
    "v3": 408
  },
  {
    "attempt": 3,
    "field": "failure_mode",
    "baseline": "timeout_holding",
    "v3": "dropped_before_target"
  },
  {
    "attempt": 3,
    "field": "frames",
    "baseline": 462,
    "v3": 461
  },
  {
    "attempt": 3,
    "field": "steps",
    "baseline": 462,
    "v3": 461
  },
  {
    "attempt": 4,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "waypoint_timeout:lower"
  },
  {
    "attempt": 4,
    "field": "failure_mode",
    "baseline": "dropped_before_target",
    "v3": "timeout_holding"
  },
  {
    "attempt": 4,
    "field": "frames",
    "baseline": 399,
    "v3": 459
  },
  {
    "attempt": 4,
    "field": "steps",
    "baseline": 399,
    "v3": 459
  },
  {
    "attempt": 5,
    "field": "failure_mode",
    "baseline": "dropped_before_target",
    "v3": "timeout_holding"
  },
  {
    "attempt": 7,
    "field": "failure_mode",
    "baseline": "timeout_holding",
    "v3": "stacked_not_confirmed"
  },
  {
    "attempt": 7,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_target": false,
      "stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "over_target": true,
      "stacked": true,
      "success": false
    }
  },
  {
    "attempt": 9,
    "field": "reason",
    "baseline": "waypoint_timeout:lower",
    "v3": "expert_finished_without_success"
  },
  {
    "attempt": 9,
    "field": "failure_mode",
    "baseline": "timeout_holding",
    "v3": "dropped_before_target"
  },
  {
    "attempt": 9,
    "field": "frames",
    "baseline": 459,
    "v3": 399
  },
  {
    "attempt": 9,
    "field": "steps",
    "baseline": 459,
    "v3": 399
  }
]
```

Same-seed accepted-array comparisons (step indices are zero-based):

```json
[]
```

#### Attempts: stack_bowls

| Run | Attempt (0-based) | Seed | Accepted | Frames | Exact reason | Exact failure mode |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 101 | False | 988 | waypoint_timeout:blue_bowl_lower | incomplete_stack |
| baseline | 1 | 102 | False | 1079 | waypoint_timeout:blue_bowl_retreat | incomplete_stack |
| baseline | 2 | 103 | False | 972 | waypoint_timeout:blue_bowl_transport | incomplete_stack |
| baseline | 3 | 104 | False | 991 | expert_finished_without_success | incomplete_stack |
| baseline | 4 | 105 | True | 915 | task_success | None |
| baseline | 5 | 106 | False | 972 | expert_finished_without_success | incomplete_stack |
| baseline | 6 | 107 | False | 972 | expert_finished_without_success | incomplete_stack |
| baseline | 7 | 108 | False | 276 | waypoint_timeout:yellow_bowl_descend | never_grasped |
| baseline | 8 | 109 | False | 975 | expert_finished_without_success | incomplete_stack |
| baseline | 9 | 110 | False | 972 | expert_finished_without_success | incomplete_stack |
| v3 | 0 | 101 | False | 972 | expert_finished_without_success | incomplete_stack |
| v3 | 1 | 102 | False | 974 | expert_finished_without_success | incomplete_stack |
| v3 | 2 | 103 | False | 977 | expert_finished_without_success | incomplete_stack |
| v3 | 3 | 104 | False | 368 | waypoint_timeout:yellow_bowl_lift | one_bowl_only |
| v3 | 4 | 105 | True | 908 | task_success | None |
| v3 | 5 | 106 | False | 432 | waypoint_timeout:yellow_bowl_transport | one_bowl_only |
| v3 | 6 | 107 | False | 1051 | waypoint_timeout:blue_bowl_release | incomplete_stack |
| v3 | 7 | 108 | False | 276 | waypoint_timeout:yellow_bowl_descend | never_grasped |
| v3 | 8 | 109 | True | 901 | task_success | None |
| v3 | 9 | 110 | False | 972 | expert_finished_without_success | incomplete_stack |

Outcome differences:

```json
[
  {
    "attempt": 0,
    "field": "reason",
    "baseline": "waypoint_timeout:blue_bowl_lower",
    "v3": "expert_finished_without_success"
  },
  {
    "attempt": 0,
    "field": "frames",
    "baseline": 988,
    "v3": 972
  },
  {
    "attempt": 0,
    "field": "steps",
    "baseline": 988,
    "v3": 972
  },
  {
    "attempt": 1,
    "field": "reason",
    "baseline": "waypoint_timeout:blue_bowl_retreat",
    "v3": "expert_finished_without_success"
  },
  {
    "attempt": 1,
    "field": "frames",
    "baseline": 1079,
    "v3": 974
  },
  {
    "attempt": 1,
    "field": "steps",
    "baseline": 1079,
    "v3": 974
  },
  {
    "attempt": 2,
    "field": "reason",
    "baseline": "waypoint_timeout:blue_bowl_transport",
    "v3": "expert_finished_without_success"
  },
  {
    "attempt": 2,
    "field": "frames",
    "baseline": 972,
    "v3": 977
  },
  {
    "attempt": 2,
    "field": "steps",
    "baseline": 972,
    "v3": 977
  },
  {
    "attempt": 2,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": false,
      "second_bowl_lifted": true,
      "all_bowls_stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": true,
      "second_bowl_lifted": true,
      "all_bowls_stacked": false,
      "success": false
    }
  },
  {
    "attempt": 3,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "waypoint_timeout:yellow_bowl_lift"
  },
  {
    "attempt": 3,
    "field": "failure_mode",
    "baseline": "incomplete_stack",
    "v3": "one_bowl_only"
  },
  {
    "attempt": 3,
    "field": "frames",
    "baseline": 991,
    "v3": 368
  },
  {
    "attempt": 3,
    "field": "steps",
    "baseline": 991,
    "v3": 368
  },
  {
    "attempt": 3,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": true,
      "second_bowl_lifted": true,
      "all_bowls_stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": false,
      "second_bowl_lifted": false,
      "all_bowls_stacked": false,
      "success": false
    }
  },
  {
    "attempt": 4,
    "field": "frames",
    "baseline": 915,
    "v3": 908
  },
  {
    "attempt": 4,
    "field": "steps",
    "baseline": 915,
    "v3": 908
  },
  {
    "attempt": 5,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "waypoint_timeout:yellow_bowl_transport"
  },
  {
    "attempt": 5,
    "field": "failure_mode",
    "baseline": "incomplete_stack",
    "v3": "one_bowl_only"
  },
  {
    "attempt": 5,
    "field": "frames",
    "baseline": 972,
    "v3": 432
  },
  {
    "attempt": 5,
    "field": "steps",
    "baseline": 972,
    "v3": 432
  },
  {
    "attempt": 5,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": true,
      "second_bowl_lifted": true,
      "all_bowls_stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": false,
      "second_bowl_lifted": false,
      "all_bowls_stacked": false,
      "success": false
    }
  },
  {
    "attempt": 6,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "waypoint_timeout:blue_bowl_release"
  },
  {
    "attempt": 6,
    "field": "frames",
    "baseline": 972,
    "v3": 1051
  },
  {
    "attempt": 6,
    "field": "steps",
    "baseline": 972,
    "v3": 1051
  },
  {
    "attempt": 8,
    "field": "accepted",
    "baseline": false,
    "v3": true
  },
  {
    "attempt": 8,
    "field": "reason",
    "baseline": "expert_finished_without_success",
    "v3": "task_success"
  },
  {
    "attempt": 8,
    "field": "failure_mode",
    "baseline": "incomplete_stack",
    "v3": null
  },
  {
    "attempt": 8,
    "field": "frames",
    "baseline": 975,
    "v3": 901
  },
  {
    "attempt": 8,
    "field": "steps",
    "baseline": 975,
    "v3": 901
  },
  {
    "attempt": 8,
    "field": "stages",
    "baseline": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": true,
      "second_bowl_lifted": true,
      "all_bowls_stacked": false,
      "success": false
    },
    "v3": {
      "reached": true,
      "grasped": true,
      "lifted": true,
      "first_bowl_stacked": true,
      "second_bowl_lifted": true,
      "all_bowls_stacked": true,
      "success": true
    }
  },
  {
    "attempt": 8,
    "field": "latched_success",
    "baseline": false,
    "v3": true
  },
  {
    "attempt": 8,
    "field": "final_success",
    "baseline": false,
    "v3": true
  }
]
```

Same-seed accepted-array comparisons (step indices are zero-based):

```json
[
  {
    "baseline_episode_index": 0,
    "v3_episode_index": 0,
    "seed": 105,
    "column": "action",
    "baseline_sha256": "daf21fdb6d52cda2cb5ddc83fea0a82d574d26c6e7fe4b35243152412f945231",
    "v3_sha256": "4ff6f7c745edce5f819fd6d149e5bee21d55da7dcee1ef7da9c6d3ff39451292",
    "baseline_shape": [
      915,
      6
    ],
    "v3_shape": [
      908,
      6
    ],
    "first_differing_step": 0,
    "first_differing_joint": 1,
    "max_absolute_difference_shared_extent": 19.292896270751953
  },
  {
    "baseline_episode_index": 0,
    "v3_episode_index": 0,
    "seed": 105,
    "column": "observation.state",
    "baseline_sha256": "9972f8c22936c02214dc91e7bf25b33ae4592c394afb56ccab20b908bf9625c1",
    "v3_sha256": "6be8f9fc54079a9f4c1eb3032e10943f45d564d00cae074fbff9eada4d2e89c3",
    "baseline_shape": [
      915,
      6
    ],
    "v3_shape": [
      908,
      6
    ],
    "first_differing_step": 2,
    "first_differing_joint": 2,
    "max_absolute_difference_shared_extent": 19.616626739501953
  }
]
```

The initial target-two bowls run also had a same-seed accepted pair (seed 105). Its exact array comparison is retained independently of the subsequent full-coverage run:

```json
[
  {
    "baseline_episode_index": 0,
    "v3_episode_index": 0,
    "seed": 105,
    "column": "action",
    "baseline_sha256": "daf21fdb6d52cda2cb5ddc83fea0a82d574d26c6e7fe4b35243152412f945231",
    "v3_sha256": "2bcbe3724dde0baf58c005c219dfa57987260d4c948cc5c845b84b84e430c6f7",
    "baseline_shape": [
      915,
      6
    ],
    "v3_shape": [
      797,
      6
    ],
    "first_differing_step": 0,
    "first_differing_joint": 1,
    "max_absolute_difference_shared_extent": 16.206417083740234
  },
  {
    "baseline_episode_index": 0,
    "v3_episode_index": 0,
    "seed": 105,
    "column": "observation.state",
    "baseline_sha256": "9972f8c22936c02214dc91e7bf25b33ae4592c394afb56ccab20b908bf9625c1",
    "v3_sha256": "d872b41e9bd6d64041fa5a062e2fb22fccd8eb516b9c04e2643b39845fe1412e",
    "baseline_shape": [
      915,
      6
    ],
    "v3_shape": [
      797,
      6
    ],
    "first_differing_step": 2,
    "first_differing_joint": 2,
    "max_absolute_difference_shared_extent": 16.473655700683594
  }
]
```

Same-commit repeat observations (the full-ten stopping rule affects only whether later attempts execute):

| Seed | Initial v3 accepted frames | Full-ten v3 accepted frames |
| --- | --- | --- |
| 105 | 797 | 908 |

Different results at the same v3 commit demonstrate run-to-run variability; the cross-commit differences above are findings and do not by themselves identify a restructure-induced cause.

Available SHA-256 digests for **every** recorded accepted action/state array:

| Run | Recipe | Seed | Episode | Column | Shape / dtype | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | stack_bowls | 105 | 0 | action | [915, 6] float32 | daf21fdb6d52cda2cb5ddc83fea0a82d574d26c6e7fe4b35243152412f945231 |
| baseline | stack_bowls | 105 | 0 | observation.state | [915, 6] float32 | 9972f8c22936c02214dc91e7bf25b33ae4592c394afb56ccab20b908bf9625c1 |
| v3 | scripted | 101 | 0 | action | [391, 6] float32 | 9c7a59236827d2c90318dac097051943b4001c6470f48349b4859f3985faf548 |
| v3 | scripted | 101 | 0 | observation.state | [391, 6] float32 | 71da6e29f8ccc96cece7eb2c1003a21c7801f6fa982a1e6926560bcc2f7975e5 |
| v3 | stack_bowls | 105 | 0 | action | [797, 6] float32 | 2bcbe3724dde0baf58c005c219dfa57987260d4c948cc5c845b84b84e430c6f7 |
| v3 | stack_bowls | 105 | 0 | observation.state | [797, 6] float32 | d872b41e9bd6d64041fa5a062e2fb22fccd8eb516b9c04e2643b39845fe1412e |
| v3 | stack_bowls | 107 | 1 | action | [901, 6] float32 | 79e5835b0f8f99e276a0bee11c2dce728c9d2b71b05fc2e67575a2ad9cc04676 |
| v3 | stack_bowls | 107 | 1 | observation.state | [901, 6] float32 | 8432e671626977e3adf99f715a359641c4955097803d6f43c3ade52ba08d1281 |
| v3_all10 | stack_bowls | 105 | 0 | action | [908, 6] float32 | 4ff6f7c745edce5f819fd6d149e5bee21d55da7dcee1ef7da9c6d3ff39451292 |
| v3_all10 | stack_bowls | 105 | 0 | observation.state | [908, 6] float32 | 6be8f9fc54079a9f4c1eb3032e10943f45d564d00cae074fbff9eada4d2e89c3 |
| v3_all10 | stack_bowls | 109 | 1 | action | [901, 6] float32 | 3428b3a04ae18ac75da910ca31d00906e2d4c6c73d6c37b8e411c0b5aa66831d |
| v3_all10 | stack_bowls | 109 | 1 | observation.state | [901, 6] float32 | df05591c2dedd665c0a2e25867815e81faeb2d2a8d2340c987dc3d6583ca284e |

Raw array comparisons match accepted episodes by seed, then by frame and joint index, and record the first differing step and maximum absolute difference across the shared extent. When an episode was rejected in one run, its action/state arrays were discarded by the existing success-only writer: a raw-array first-difference step and maximum difference are **unavailable**, not zero. Outcome divergence is still reported. These single runs do not distinguish implementation drift from simulator nondeterminism.

### B5: example-policy eval smoke

```bash
docker image inspect s2r-serve:latest --format "{{.Id}}"
```
**PASS** (command exit 0); full output: [commands/serve_image_identity.json](tests/fixtures/v3_baseline/sim/commands/serve_image_identity.json).

```bash
S2R_POLICY=example docker compose --profile serve up -d serve
```
**PASS** (command exit 0); full output: [commands/start_example_policy.json](tests/fixtures/v3_baseline/sim/commands/start_example_policy.json).

```bash
docker logs s2r-serve-example
```
**PASS** (command exit 0); full output: [commands/example_policy_startup.json](tests/fixtures/v3_baseline/sim/commands/example_policy_startup.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.eval.run --config_path workspace/configs/eval/pick_place.yaml --policy example --policy_host 127.0.0.1 --policy_port 5600 --action_horizon 8 --num_envs 1 --smoke-test --smoke-selection one_per_zone --headless
```
**FAIL** (command exit 2); full output: [commands/eval_smoke.json](tests/fixtures/v3_baseline/sim/commands/eval_smoke.json).

The first invocation supplied an unsupported `--headless` CLI option and exited during argument parsing (`src/sim/eval/run.py:311`). The source already supports headless selection through YAML `run.gui: false` (`:127`, `:305`), so the actual smoke invocation below uses that existing configuration without the invalid flag. This corrects the invocation only; no source/config file was changed.

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.eval.run --config_path workspace/configs/eval/pick_place.yaml --policy example --policy_host 127.0.0.1 --policy_port 5600 --action_horizon 8 --num_envs 1 --smoke-test --smoke-selection one_per_zone
```
**FAIL** (command exit 1); full output: [commands/eval_smoke_yaml_headless.json](tests/fixtures/v3_baseline/sim/commands/eval_smoke_yaml_headless.json).

Result: **FAIL** for smoke completion. The example policy repeats the current joint state, so task success is not a learned-policy quality test.

```json
{
  "complete": false,
  "requested_episodes": 9,
  "checkpoint": "stage_2_main/checkpoint-60000",
  "seed": 1984,
  "step_dt": 0.03333333333333333,
  "episode_length_s": 30.0,
  "action_horizon": 8,
  "action_repeat": 1,
  "policy_cameras": [
    "room",
    "wrist"
  ],
  "policy_joint_transform": null,
  "resume_settings": {
    "policy": "example",
    "policy_options": {
      "embodiment_tag": "NEW_EMBODIMENT"
    },
    "cube_starts": "/workspace/datasets/pick_place_v1_calibrated_A",
    "rename_map": {
      "realsense_rgb": "room",
      "wrist_cam": "wrist"
    },
    "lang": "Pick up the blue cube and place it in the white box",
    "lang_by_color": null,
    "render_warmup": 2,
    "smoothing": {
      "prefetch_steps": 0,
      "blend_steps": 0,
      "ema_alpha": 1.0,
      "smooth_gripper": false,
      "temporal_ensemble": 0,
      "te_decay": 0.01
    },
    "grasp": {
      "gripper_latch": false,
      "latch_close_below": null,
      "latch_confirm_steps": 3,
      "latch_open_above": null,
      "latch_release_confirm": 5,
      "latch_min_hold": 30,
      "latch_dataset": "",
      "latch_steady_steps": 10,
      "latch_steady_max_delta": 2.0,
      "gripper_empty_closed": null,
      "latch_empty_margin": 3.0,
      "latch_empty_confirm": 15,
      "latch_reopen_steps": 15
    },
    "seed_per_episode": false
  },
  "smoke_test": {
    "selection": "one_per_zone",
    "reach_timeout_s": null,
    "progress_timeout_s": null
  },
  "overall": {
    "episodes": 0,
    "successes": 0,
    "success_rate": null,
    "success_rate_ci95": [
      null,
      null
    ],
    "stage_rates": {
      "reached": null,
      "grasped": null,
      "lifted": null,
      "over_box": null,
      "placed": null,
      "success": null
    },
    "failure_modes": {},
    "median_time_to_success_s": null
  },
  "episodes": []
}
```

Full error text for eval_smoke:

```text
usage: run.py [-h] [--config_path CONFIG_PATH] [--arm ARM] [--scene SCENE]
              [--task TASK] [--unit UNIT] [--variant VARIANT]
              [--disable_fabric] [--num_envs NUM_ENVS] [--seed SEED]
              [--num_episodes NUM_EPISODES]
              [--policy_timeout_ms POLICY_TIMEOUT_MS]
              [--rename_map RENAME_MAP] [--policy POLICY]
              [--policy_option KEY=VALUE] [--policy_host POLICY_HOST]
              [--policy_port POLICY_PORT]
              [--policy_api_token POLICY_API_TOKEN]
              [--action_horizon ACTION_HORIZON]
              [--action_repeat ACTION_REPEAT]
              [--prefetch_steps PREFETCH_STEPS] [--blend_steps BLEND_STEPS]
              [--ema_alpha EMA_ALPHA] [--smooth_gripper]
              [--temporal_ensemble TEMPORAL_ENSEMBLE] [--te_decay TE_DECAY]
              [--log_timing] [--timing_csv TIMING_CSV]
              [--episode_length_s EPISODE_LENGTH_S]
              [--lang_instruction LANG_INSTRUCTION] [--rerun] [--smoke-test]
              [--save_videos] [--video_dir VIDEO_DIR]
              [--video_encoder {libx264,h264_nvenc}]
              [--render_warmup RENDER_WARMUP] [--eval_set EVAL_SET]
              [--splits SPLITS] [--repeats REPEATS]
              [--smoke-selection {first_five,one_per_zone}]
              [--smoke-reach-timeout-s SMOKE_REACH_TIMEOUT_S]
              [--smoke-progress-timeout-s SMOKE_PROGRESS_TIMEOUT_S]
              [--cube_starts CUBE_STARTS] [--start_mode {cycle,random}]
              [--random_fraction RANDOM_FRACTION]
              [--robot_start {recorded,default}]
              [--lang_instruction_by_color LANG_INSTRUCTION_BY_COLOR]
              [--results_json RESULTS_JSON] [--resume]
              [--checkpoint CHECKPOINT] [--gripper_latch]
              [--latch_close_below LATCH_CLOSE_BELOW]
              [--latch_open_above LATCH_OPEN_ABOVE]
              [--latch_confirm_steps LATCH_CONFIRM_STEPS]
              [--latch_release_confirm LATCH_RELEASE_CONFIRM]
              [--latch_min_hold LATCH_MIN_HOLD]
              [--latch_dataset LATCH_DATASET]
              [--latch_steady_steps LATCH_STEADY_STEPS]
              [--latch_steady_max_delta LATCH_STEADY_MAX_DELTA]
              [--gripper_empty_closed GRIPPER_EMPTY_CLOSED]
              [--latch_empty_margin LATCH_EMPTY_MARGIN]
              [--latch_empty_confirm LATCH_EMPTY_CONFIRM]
              [--latch_reopen_steps LATCH_REOPEN_STEPS] [--log_gripper]
              [--gripper_csv GRIPPER_CSV] [--seed_per_episode]
              [--livestream {0,1,2}] [--xr] [--device DEVICE]
              [--visualizer VISUALIZER] [--verbose] [--info]
              [--experience EXPERIENCE] [--deterministic]
              [--kit_args KIT_ARGS] [--anim_recording_enabled]
              [--anim_recording_start_time ANIM_RECORDING_START_TIME]
              [--anim_recording_stop_time ANIM_RECORDING_STOP_TIME]
              [--max_visible_envs MAX_VISIBLE_ENVS]
run.py: error: unrecognized arguments: --headless
```

Full error text for eval_smoke_yaml_headless:

```text
Traceback (most recent call last):
  File "/workspace/s2r/src/sim/eval/loop.py", line 133, in run_loop
    validate_scene_geometry(live, context.benchmark.data["scene"], context.task)
  File "/workspace/s2r/src/sim/eval/scene.py", line 95, in validate_scene_geometry
    raise ValueError(f"Live scene differs from eval set: {key}; recapture geometry")
ValueError: Live scene differs from eval set: base_position_world; recapture geometry
[INFO]: Connecting to the example policy server at 127.0.0.1:5600...
[INFO]: Policy server connected
[RESULT] incomplete: 0/9 episodes; success 0/0 (n/a)
[SMOKE_RESULT] {"complete":false,"requested_episodes":9,"checkpoint":"stage_2_main/checkpoint-60000","seed":1984,"step_dt":0.03333333333333333,"episode_length_s":30.0,"action_horizon":8,"action_repeat":1,"policy_cameras":["room","wrist"],"policy_joint_transform":null,"resume_settings":{"policy":"example","policy_options":{"embodiment_tag":"NEW_EMBODIMENT"},"cube_starts":"/workspace/datasets/pick_place_v1_calibrated_A","rename_map":{"realsense_rgb":"room","wrist_cam":"wrist"},"lang":"Pick up the blue cube and place it in the white box","lang_by_color":null,"render_warmup":2,"smoothing":{"prefetch_steps":0,"blend_steps":0,"ema_alpha":1.0,"smooth_gripper":false,"temporal_ensemble":0,"te_decay":0.01},"grasp":{"gripper_latch":false,"latch_close_below":null,"latch_confirm_steps":3,"latch_open_above":null,"latch_release_confirm":5,"latch_min_hold":30,"latch_dataset":"","latch_steady_steps":10,"latch_steady_max_delta":2.0,"gripper_empty_closed":null,"latch_empty_margin":3.0,"latch_empty_confirm":15,"latch_reopen_steps":15},"seed_per_episode":false},"smoke_test":{"selection":"one_per_zone","reach_timeout_s":null,"progress_timeout_s":null},"overall":{"episodes":0,"successes":0,"success_rate":null,"success_rate_ci95":[null,null],"stage_rates":{"reached":null,"grasped":null,"lifted":null,"over_box":null,"placed":null,"success":null},"failure_modes":{},"median_time_to_success_s":null},"episodes":[]}
[INFO]: SimulationContext cleared
 10de      | 0          |
|     |                                  |        |     |            | 2684      | 0aa900f2.. |
|     |                                  |        |     |            | 1         |            |
|=============================================================================================|
| OS: 24.04.3 LTS (Noble Numbat) ubuntu, Version: 24.04.3, Kernel: 6.8.0-142-generic
| XServer Vendor: The X.Org Foundation, XServer Version: 12101011 (1.21.1.11)
| Processor: Intel(R) Core(TM) i9-14900K
| Bare Metal Cores: 24 | Bare Metal Logical Cores: 48
| Available Cores:  32 
|---------------------------------------------------------------------------------------------|
| Total Memory (MB): 63040 | Free Memory: 56992
| Total Page/Swap (MB): 2047 | Free Page/Swap: 0
|---------------------------------------------------------------------------------------------|
2026-11-14T14:57:48Z [751ms] [Warning] [omni.rtx] CPU performance profile is set to powersave. This profile sets the CPU to the lowest frequency reducing performance.
2026-11-14T14:57:48Z [753ms] [Warning] [omni.rtx] IOMMU is enabled.
2026-11-14T14:57:48Z [1,260ms] [Warning] [omni.log] Source: omni.hydra was already registered.
2026-11-14T14:57:49Z [1,550ms] [Warning] [pxr.Semantics] pxr.Semantics is deprecated - please use Semantics instead
2026-11-14T14:57:49Z [1,602ms] [Warning] [omni.replicator.core.scripts.extension] No material configuration file, adding configuration to material settings directly.
2026-11-14T14:57:50Z [2,442ms] [Warning] [rtx.hydra] RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together but this can cause issues with dynamic objects not streaming correctly due to problems in the transform update.
2026-11-14T14:58:01Z [13,918ms] [Warning] [omni.usd] Encountered USD Warnings but USD Diagnostics are currently muted. To view USD Warnings, please set the '/persistent/app/usd/muteUsdDiagnostics' setting to false in Preferences page.
2026-11-14T14:58:01Z [13,919ms] [Warning] [omni.fabric.plugin] Warning: attribute material:binding not found for path /World/envs/env_0/WhiteBox/node_

2026-11-14T14:58:01Z [13,958ms] [Warning] [isaaclab_newton.assets.articulation.articulation_data] Failed to get root com velocity. If the articulation is fixed, this is expected. Setting root com velocity to zeros.
2026-11-14T14:58:01Z [14,093ms] [Warning] [omni.replicator.core.scripts.annotators] omni.hydra.engine.stats is not installed, assuming single GPU
```

#### All stored starts — PASS for loading; known validity rejections retained

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 s2r-sim s2r-entrypoint python /tmp/v3_start_loader_probe.py --phase baseline
```
**PASS** (command exit 0); full output: [commands/baseline_stored_starts.json](tests/fixtures/v3_baseline/sim/commands/baseline_stored_starts.json).

```bash
docker exec s2r-sim s2r-entrypoint python /tmp/v3_start_loader_probe.py --phase v3
```
**PASS** (command exit 0); full output: [commands/v3_stored_starts.json](tests/fixtures/v3_baseline/sim/commands/v3_stored_starts.json).

| Run | Eval set | Stored | Loaded through EvalSetStarts.sample | Rejected by stored-scene guard |
| --- | --- | --- | --- | --- |
| baseline | pick_place_v1_75ep_reach150_seed1984.json | 150 | 150 | train-059 |
| baseline | pick_place_v1_75ep_seed1984.json | 200 | 200 | train-059 |
| baseline | pick_place_v1_seed1984.json | 150 | 150 | none |
| v3 | pick_place_v1_75ep_reach150_seed1984.json | 150 | 150 | train-059 |
| v3 | pick_place_v1_75ep_seed1984.json | 200 | 200 | train-059 |
| v3 | pick_place_v1_seed1984.json | 150 | 150 | none |

All 500 stored entries were read through the real loader/sample path, including robot state, and checked against both stored and normalized stored scene geometry. `train-059` loads as data but fails start validity in the 75ep and 75ep_reach150 sets on both commits; the 150-start default set has no rejected starts. These loader checks do not claim that all 500 starts were live-reset in the smoke run; smoke uses its selected zone schedule.

### B6: mimic/replay/teleop/real/eval imports — PASS

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.datagen.mimic --help
```
**PASS** (command exit 0); full output: [commands/help_sim_datagen_mimic.json](tests/fixtures/v3_baseline/sim/commands/help_sim_datagen_mimic.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.datagen.dr_replay --help
```
**PASS** (command exit 0); full output: [commands/help_sim_datagen_dr_replay.json](tests/fixtures/v3_baseline/sim/commands/help_sim_datagen_dr_replay.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.datagen.calibration_replay --help
```
**PASS** (command exit 0); full output: [commands/help_sim_datagen_calibration_replay.json](tests/fixtures/v3_baseline/sim/commands/help_sim_datagen_calibration_replay.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.teleop.agent --help
```
**PASS** (command exit 0); full output: [commands/help_sim_teleop_agent.json](tests/fixtures/v3_baseline/sim/commands/help_sim_teleop_agent.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m sim.workspace.arms.urdf_import --help
```
**PASS** (command exit 0); full output: [commands/help_sim_workspace_arms_urdf_import.json](tests/fixtures/v3_baseline/sim/commands/help_sim_workspace_arms_urdf_import.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m core.rollout.real_rollout --help
```
**PASS** (command exit 0); full output: [commands/help_core_rollout_real_rollout.json](tests/fixtures/v3_baseline/sim/commands/help_core_rollout_real_rollout.json).

```bash
docker exec s2r-sim s2r-entrypoint python -m core.eval.eval_set --help
```
**PASS** (command exit 0); full output: [commands/help_core_eval_eval_set.json](tests/fixtures/v3_baseline/sim/commands/help_core_eval_eval_set.json).

All seven returned 0 with help text and no import error. Scripted generation and eval help were also inspected before choosing runtime flags.

## Limits and steps that could not be completed

- The baseline recipes did not produce two accepted episodes each in the selected ten-attempt limit (0 pick-place, 0 cube stack, 1 bowl stack). The command failures and all per-attempt failure modes were retained and no production fixes or recipe tuning were applied.
- Cross-commit raw-array equivalence cannot be proved for episodes without accepted data in both runs. Available same-seed arrays, hashes, first differing steps and maximum differences are reported above; absent comparisons remain unverified.
- The live eval smoke completed 0/9 selected episodes because the geometry guard rejected `base_position_world`. All stage rates are null and per-episode failure-mode counts are empty; these are unavailable measurements, not zero failure/success rates. The example policy server connected successfully. No guard bypass or geometry edit was attempted.
- The full suite retains six opt-in live-import skips. Separate runtime construction probes covered all requested variants; the skips themselves were not enabled or reclassified.

## Diagnostic harness failures (not source fixes)

Two scratch diagnostic commands initially failed and were retained. The start-loader probe initially tried the scene-snapshot JSON as if it were an eval set, then was scoped to JSONs containing `eval_set_id` and `starts`. The first host pytest comparison could not write into the root-owned scratch output directory; scratch directory permissions were adjusted and the same comparison was saved. No repository source/test/asset was changed.

```bash
docker exec -e S2R_GIT_COMMIT=45b07310ee8f18e6c6f85a2d36e1993082e8a1a1 s2r-sim s2r-entrypoint python /tmp/v3_start_loader_probe.py --phase baseline
```
**FAIL** (command exit 1); full output: [commands/baseline_start_loads.json](tests/fixtures/v3_baseline/sim/commands/baseline_start_loads.json).

```text
Traceback (most recent call last):
  File "/tmp/v3_start_loader_probe.py", line 14, in <module>
    task=load_task('pick_place_cube'); source=module.EvalSetStarts(path,task=task)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/s2r/src/core/benchmark/eval_set.py", line 160, in __init__
    self.data = load_eval_set(path, arm=arm, task=self.task)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/s2r/src/core/benchmark/eval_set.py", line 119, in load_eval_set
    return validate_eval_set(json.loads(Path(path).read_text()), arm=arm, task=task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/s2r/src/core/benchmark/eval_set.py", line 72, in validate_eval_set
    raise ValueError('Unsupported eval set schema or coordinate units')
ValueError: Unsupported eval set schema or coordinate units
```

```bash
python3 /tmp/v3-sim-checks/compare_pytest.py
```
**FAIL** (command exit 1); full output: [commands/pytest_comparison.json](tests/fixtures/v3_baseline/sim/commands/pytest_comparison.json).

```text
Traceback (most recent call last):
  File "/tmp/v3-sim-checks/compare_pytest.py", line 28, in <module>
    output=root/'outputs/v3-sim-checks/pytest_comparison.json';output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
                                                                                                               ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rpuser1/miniforge3/lib/python3.13/pathlib/_local.py", line 555, in write_text
    return PathBase.write_text(self, data, encoding, errors, newline)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rpuser1/miniforge3/lib/python3.13/pathlib/_abc.py", line 651, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rpuser1/miniforge3/lib/python3.13/pathlib/_local.py", line 537, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: '/home/rpuser1/Isaac-Sim-to-Real/outputs/v3-sim-checks/pytest_comparison.json'
```

## Evidence, cleanup and publication

`tests/fixtures/v3_baseline/sim/artifact_manifest.json` inventories JSON artifacts and SHA-256 hashes. `commands/*.json` retains exact commands, exit codes and full stdout/stderr; timed runner records also include UTC start times. The original JUnit XML is retained as JSON text in `v3_pytest_junit.json`. Reports, cfg dumps, hashes, per-start results, recipe texts and harness source are JSON only. Datasets, images and videos remain in ignored scratch outputs and are not staged.

```bash
docker stop s2r-serve-example && docker rm s2r-serve-example
```
**PASS** (command exit 0); full output: [commands/stop_example_policy.json](tests/fixtures/v3_baseline/sim/commands/stop_example_policy.json).

Publication commands: `git add SIM_REPORT.md tests/fixtures/v3_baseline/sim`, `git commit -m "Record v3 simulator verification against Phase 0"`, and `git push -u origin v3-sim-checks`. No pull request is opened. The committed diff is restricted to this report and JSON fixtures.
