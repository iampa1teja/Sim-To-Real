# Background policy inference worker

See [ASYNC_RESULTS.md](ASYNC_RESULTS.md) for all new and reference control-loop tables, [async_summary.csv](async_summary.csv) for the 36 new summaries, and [async_audit.json](async_audit.json) for raw-evidence checks. Reference B summaries are pinned to benchmark commit `e1b941727dbe51c234c136df72f6c1a0e3723aee` in [reference_control.csv](reference_control.csv). Raw reference results are not copied into this branch.

## Using the worker

Real rollout and the testing GUI accept `--async_inference`, or `async_inference: true` in rollout YAML. The default is false. The paired real/sim console uses the worker on its real side only; its simulation side stays on the existing synchronous/step-driven path. Fixed simulator evaluation, schedules, seeds and hashes are unchanged.

```bash
./s2r rollout --config_path configs/rollout/example.yaml --async_inference
# Retains shadow mode until the existing explicit motion option is selected.
# For an already configured real-arm testing GUI:
PYTHONPATH=src python -m core.rollout.testing_gui \
  --config_path configs/rollout/gui.yaml --async_inference
```

The real-device entrypoints need the existing simulator/LeRobot environment; run the Python command through `docker exec -it -w /workspace/s2r s2r-sim s2r-entrypoint python ...` there. These commands retain their existing device/calibration requirements. No hardware validation was performed by this task.

`refill_seconds` defaults to 0.5. A request starts when remaining control slots cover at most `max(refill_seconds, recent turnaround p95 + control_dt)`. Only one inference request can be outstanding. `starvation_grace` defaults to 1 second of continuous held ticks, including startup; increase it explicitly when a real policy's startup/inference needs more time. `policy_timeout_ms` defaults to 15000 and faults a run when a request reaches its deadline. Tick calls poll completed results and never wait for network I/O. Connect/reset/close are lifecycle operations and can wait for the worker.

The worker creates, pings, describes, resets, seeds, queries and closes its own socket. Inputs are copied before crossing threads. Replies carry a generation, monotonic capture time and control-step index. Capture is stamped when the client takes ownership of an observation; this does not recover a camera's physical exposure timestamp. Arrival drops the elapsed prefix after `action_repeat` expansion. Buffered targets also expire when the control clock advances during a paused tick. Optional `blend_steps` reuse `blend_actions`; gripper joints bypass blending and EMA unless `smooth_gripper` is enabled. `ema_alpha` defaults to 1. Grasp hooks run on the control thread: observe an accepted chunk before processing its first target, with measured gripper feedback on every tick. `seed_per_episode` retains the per-query seed acknowledgement.

`set_instruction(text)` discards buffered/old-generation replies, holds the last applied target and starts fresh inference when the old socket request has drained. It does not restart a stopped policy. Reset, cancellation, instruction change and faults invalidate generations. `cancel()` returns without waiting on the network; `reset()` is required to resume. `close()` drains a pending request up to the existing transport timeout, then closes the socket on its owning thread. `temporal_ensemble` is explicitly unsupported in async mode.

Callers apply the existing range clamp, first-fresh-target gap guard and motion lock before sending. Startup holds retain that first-target guard. `record_applied_target()` records the bounded target actually dispatched so starvation repeats that target exactly. Stop is still checked inside the existing dispatch lock, and no reply is sent after Stop acknowledgement. Shadow mode and torque behavior retain their existing contracts.

`TimingLog` adds `action_age_ms` and `turnaround_s` alongside the existing held/chunk-id fields. Age includes holds and conservatively uses the oldest observation contributing through overlap blending or EMA. Startup holds have no policy-observation age. EMA's oldest-contributor age can grow over an episode; benchmark runs use EMA=1 and blend=0.

## Install the isolated CPU benchmark environment

From the repository root with `uv` and network access:

```bash
uv venv --python 3.12 tools/bench_transport/.venv
uv pip install --python tools/bench_transport/.venv/bin/python \
  'torch==2.11.0' 'torchvision==0.26.0' \
  --index https://download.pytorch.org/whl/cpu
uv pip install --python tools/bench_transport/.venv/bin/python \
  -r tools/bench_transport/requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
```

This is the same pinned CPU environment as the reference, including LeRobot 0.7.0/Zenoh for reproducing its reference harness paths. It does not upgrade the simulator or serving containers. The async benchmark itself uses the existing ZMQ/numpy protocol, with a benchmark-only variable camera list on the example plugin instance. The dummy server advertises the example contract and returns fixed zero chunks after the configured delay. No production policy folder, serializer or transport is edited.

## Tests and the full control matrix

```bash
PYTHONPATH=src tools/bench_transport/.venv/bin/python -m pytest \
  tests/core/policy_client -q -p no:cacheprovider

# Full CPU acceptance, including existing live-import cases; do not overlap with measurement.
docker exec -e S2R_RUN_IMPORT_TESTS=1 -w /workspace/s2r s2r-sim \
  bash -c 'PYTHONPATH=src s2r-entrypoint python -m pytest tests -q -rs' \
  > tools/bench_transport/cpu-suite.txt 2>&1
async_test_exit=$?
printf '%s\n' "$async_test_exit" > tools/bench_transport/cpu-suite.exitcode
test "$async_test_exit" -eq 0

# Functional smoke; its data is never accepted as a 60-second result.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/bench.py \
  --mode B --async-only --seconds 3 --limit 2 \
  --output tools/bench_transport/raw_async/smoke-new

# 36 serial, fresh-process, 60-second measurements (~38 minutes).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/bench.py \
  --mode B --async-only --seconds 60 \
  --output tools/bench_transport/raw_async/control
OPENBLAS_NUM_THREADS=1 tools/bench_transport/.venv/bin/python \
  tools/bench_transport/async_report.py --require-complete
```

Completed JSON files are resume checkpoints. To remeasure everything, first move `raw_async/control` to an ignored archive (only after its process exits), then run the same command. For alternate refill settings use `--refill-seconds <value>` and a separate output directory. The accepted report reads only `raw_async/control` and requires exactly the complete 60-second matrix. Each cell retains all 1,800 tick rows, chunk-acceptance events and request turnarounds; their hashes are recorded in the audit. Raw files and logs are ignored; only summaries, the report, the small CPU-suite log and source scripts are committed.

The full suite has two existing skips for parallel-gripper checks parametrized against the inapplicable jaw fixture. These tests are unchanged; no new skip, ignore or deselection is introduced. The only existing test modified is `tests/core/policy_client/test_transport.py`, which gained an additional real-ZMQ worker test without changing existing assertions. New tests in `test_async_worker.py` cover scheduling, fake-clock refill boundaries, deadlines, generation changes, repeat, blending/EMA, grasp order, seed ACKs, socket ownership and the real-arm dispatch guards.

## Real GPU server follow-up

On the GPU machine, start the actual trained server with its normal checkpoint/options (not the CPU dummy). Example using this repository's CLI:

```bash
./s2r serve --policy molmoact2 --models-dir /absolute/path/to/models \
  --checkpoint checkpoint-directory --port 5556
nvidia-smi

# Use the CPU client venv above; model/GPU dependencies stay in the serving container.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/gpu_client.py \
  --policy molmoact2 --host 127.0.0.1 --port 5556 \
  --arm so101 --unit rpd_follower --action-horizon 30 \
  --instruction 'Pick up the blue cube and place it in the white box' \
  --policy-option input_joint_units=lerobot_normalized \
  --height 480 --width 640 --seconds 60 --fps 30 \
  --output tools/bench_transport/raw_async/gpu-sync.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/gpu_client.py \
  --policy molmoact2 --host 127.0.0.1 --port 5556 \
  --arm so101 --unit rpd_follower --action-horizon 30 \
  --instruction 'Pick up the blue cube and place it in the white box' \
  --policy-option input_joint_units=lerobot_normalized \
  --height 480 --width 640 --seconds 60 --fps 30 \
  --async-inference --refill-seconds 0.5 --starvation-grace 5 \
  --output tools/bench_transport/raw_async/gpu-async.json
```

Use the correct arm/unit, camera dimensions and model/client options for that trained checkpoint. This client opens no robot and sends no motor commands. It sends synthetic RGB and calibrated midpoint state through the actual policy wrapper; returned predictions include real model execution and device-to-host work in the serving process. Record checkpoint/precision, GPU/driver and serving package versions with the outputs. Repeat for the other resolutions/horizons and for GR00T using its configured server and client options. These commands are for the separate GPU budget; real camera/arm timing, reset/fault behavior and physical smoothness still require hardware validation.
