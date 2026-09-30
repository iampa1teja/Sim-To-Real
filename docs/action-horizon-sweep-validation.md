# Action-horizon sweep validation

Validated on 2026-10-01 IST, RTX 4090, checkpoint
`so101_pick_place_v1_n17_10k/checkpoint-50000`. No full sweep was executed.

## Files changed

- `.gitignore`: track the new sweep regression tests; existing local edits excluded.
- `docker/sweep_action_horizon.sh`: shell entry point.
- `scripts/sweep_action_horizon.py`: host argument/configuration handling, one-server lifecycle, horizon runs, logging, progress and resume.
- `docker/eval_pick_place.sh`: shared-server ownership modes and explicit results destination.
- `scripts/summarize_sweep.py` and `source/sim_to_real_so101/scripts/summarize_sweep.py`: repository and installed summary entry points, CSV/JSON/Markdown and plots.
- `scripts/compare_eval_results.py` and `source/sim_to_real_so101/utils/eval_comparison.py`: one shared comparison implementation, allowing horizon differences only for same-checkpoint sweeps.
- `source/sim_to_real_so101/utils/lerobot_interface.py` and `source/sim_to_real_so101/scripts/lerobot_eval.py`: per-episode inference calls, measured wall time and flushed episode progress.
- `source/sim_to_real_so101/pyproject.toml` and `source/sim_to_real_so101/setup.py`: console registration and namespace package discovery.
- `tests/test_summarize_sweep.py`: synthetic statistics, selection, comparisons, completeness, failure/resume, plots and server lifecycle tests.
- `docs/06-evaluation.md` and this report: commands, layout, interpretation and validation.

Success predicates, geometry, stage thresholds and the eval-set generator were unchanged.

## CPU tests

```text
python3 -m unittest discover -s tests -p test_summarize_sweep.py -v
Ran 13 tests in 6.198s — OK
python3 -m unittest discover -s tests -p test_pick_place_benchmark.py -v
Ran 13 tests in 2.280s — OK
python3 -m unittest discover -s tests -p test_eval_pick_place_script.py -v
Ran 19 tests in 3.549s — OK
python3 -m unittest discover -s tests -p test_lerobot_eval_rollout.py -v
Ran 16 tests in 0.155s — OK
```

All **61 tests passed** in the working tree. The staged evaluator was also tested
with the committed baseline rollout tests independently of existing local
changes: **11 tests passed**. Shell syntax, Python compilation and git whitespace
checks passed. The installed `summarize_eval_sweep` console command was exercised
from a wheel built in a temporary copy, without relying on repository imports.
Temporary packaging avoided the existing root-owned source egg-info directory.

## Dry run

```bash
./docker/sweep_action_horizon.sh \
  --model so101_pick_place_v1_n17_10k/checkpoint-50000 \
  --models_dir ~/sim2real/models \
  --horizons 1,2,4,6,8,12,16 --dry_run
```

Output excerpt (the full command output is in
[`action_horizon_dry_run.log`](../datasets/hyperparameters/action_horizon_dry_run.log)):

```text
# Check checkpoint files under .../so101_pick_place_v1_n17_10k/checkpoint-50000
+ docker run -d --rm ... real-robot:n1.7 python .../benchmark_server.py ... --port 5555
SERVER_READY port=5555
+ docker stop groot-eval-server-<timestamp>-<pid>
+ docker rm -f groot-eval-server-<timestamp>-<pid>
Horizon 1/7 (AH 1): episodes 0/150, elapsed 0s, ETA pending
Horizon 2/7 (AH 2): episodes 0/150, elapsed 1s, ETA pending
Horizon 3/7 (AH 4): episodes 0/150, elapsed 1s, ETA pending
Horizon 4/7 (AH 6): episodes 0/150, elapsed 2s, ETA pending
Horizon 5/7 (AH 8): episodes 0/150, elapsed 2s, ETA pending
Horizon 6/7 (AH 12): episodes 0/150, elapsed 3s, ETA pending
Horizon 7/7 (AH 16): episodes 0/150, elapsed 3s, ETA pending
```

Each horizon printed the existing runner's client command with its own
`--action_horizon`, `--external_server`, and `ah_<N>/results.json` destination.
The stop/remove lines are printed cleanup commands, not executed Docker calls.
Only one server launch was printed; no GPU work occurred in the dry run.

## Four-episode GPU smoke test

A private benchmark copy kept the first two committed ID starts and recalculated
its content hash using the existing helper. The committed benchmark and generator
were unchanged. Smoke command:

```bash
./docker/sweep_action_horizon.sh \
  --model so101_pick_place_v1_n17_10k/checkpoint-50000 \
  --models_dir ~/sim2real/models \
  --eval_set datasets/hyperparameters/smoke_eval_set.json \
  --horizons 8,16 --episode_length_s 20
```

Both reports are complete with exactly two episodes. The one shared server was
removed on exit, and no GPU compute processes remained. Inference counts match
actual action-queue behavior: AH 8 reports 74 calls for 600 control steps and 14
for 121 steps; AH 16 reports 37 calls for each 600-step episode. Measured wall
times are 35.41/10.64 seconds for AH 8 and 32.73/33.76 seconds for AH 16.
Both heatmaps and all four summary plots were created; the success plot was
visually inspected. No videos are saved by the current evaluator.

Repeating the command with `--resume` printed:

```text
AH 8: skipping complete 2 episodes
AH 16: skipping complete 2 episodes
```

No additional episodes or server launches occurred in that resume check.

The full [smoke summary](../datasets/hyperparameters/so101_pick_place_v1_n17_10k_checkpoint-50000_smoke_eval_set_20260930-211949/summary.md) follows:

### Action-horizon sweep

95% Wilson intervals are descriptive episode-level intervals; repeats are correlated. McNemar uses repeat 0; p-values are unadjusted.

Best horizon: **8**.

runner_up: AH 8 vs 16; statistically better: **False**.
- all: shared starts=2, discordant=1/0, p=1.
- id: shared starts=2, discordant=1/0, p=1.
horizon_16: AH 8 vs 16; statistically better: **False**.
- all: shared starts=2, discordant=1/0, p=1.
- id: shared starts=2, discordant=1/0, p=1.

Time to success includes settling and uses successes only. Missing telemetry is shown as —.

| AH | Split | Status | Successes/episodes | Rate [95% CI] | Median / mean success s | ≤15 s |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | id | complete | 1/2 | 50.0% [9.5%, 90.5%] | 4.03 / 4.03 | 50.0% |
| 8 | all | complete | 1/2 | 50.0% [9.5%, 90.5%] | 4.03 / 4.03 | 50.0% |
| 16 | id | complete | 0/2 | 0.0% [0.0%, 65.8%] | — / — | 0.0% |
| 16 | all | complete | 0/2 | 0.0% [0.0%, 65.8%] | — / — | 0.0% |

| AH | Split | Reached | Grasped | Lifted | Over box | Placed |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | id | 0.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| 8 | all | 0.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| 16 | id | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 16 | all | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

| AH | Split | never_reached | missed_grasp | dropped_before_box | dropped_outside | placed_not_confirmed | timeout_holding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | id | 1 | 0 | 0 | 0 | 0 | 0 |
| 8 | all | 1 | 0 | 0 | 0 | 0 | 0 |
| 16 | id | 2 | 0 | 0 | 0 | 0 | 0 |
| 16 | all | 2 | 0 | 0 | 0 | 0 | 0 |

## Full sweep command

This command is provided for the user and was **not executed**:

```bash
./docker/sweep_action_horizon.sh \
  --model so101_pick_place_v1_n17_10k/checkpoint-50000 \
  --models_dir ~/sim2real/models \
  --eval_set eval_sets/pick_place_v1_seed1984.json \
  --splits all --horizons 1,2,4,6,8,12,16 \
  --repeats 1 --episode_length_s 20
```

Approximate budget: 25 minutes per horizon for 150 episodes, or 2 hours 55 minutes
for seven horizons, plus initialization. The four-episode smoke result establishes
execution and output behavior; it does not establish a statistically best horizon.
