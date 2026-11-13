# Policy transport benchmark

Compare the production ZMQ transport and client with **LeRobot v0.7.0**, pinned to commit `410c095eacc0011d998a73383eb94abed2715af0`. Benchmark implementation lives here; the approved test-only fixture correction is under `tests/sim/`. Production `src/`, `policies/`, `docker/`, and `configs/` stay unchanged. See [REPORT.md](REPORT.md), [summary CSV](results/summary.csv), and [coverage/raw-data audit](results/audit.json).

## Install an isolated CPU environment

From the repository root on Linux x86_64, with `uv`, Git and network access:

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

The lock includes the `remote` extra and all resolved versions. CPU Torch is intentional: these numbers exclude GPU model execution. The venv and generated checkpoints are ignored. No existing Python or simulator environment is upgraded.

The tiny ACT checkpoint has seed 20261114, a randomly initialized ResNet18 (no downloaded backbone weights), transformer width 32, feedforward width 64, two heads, one encoder/decoder layer, no VAE, identity normalization and a 50-step prediction horizon. Saved processors and a real `load_deployment` warmup are mandatory. One checkpoint per resolution/camera count is cached; the server owns the requested 16/30/50 execution slice. Files are reproducibly generated rather than committing hundreds of megabytes of random weights.

## Quick checks and individual runs

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python -m pytest \
  tools/bench_transport/test_benchmark.py -q -p no:cacheprovider

# Four transport cells for one payload; smoke data is excluded from REPORT.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/bench.py \
  --mode A --iterations 10 --warmup 2 --limit 4 \
  --output tools/bench_transport/results/smoke-new

# One actual 60 s control cell; filter matches a case name.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv/bin/python tools/bench_transport/bench.py \
  --mode B --filter 480x640-c1-h30-lerobot-raw-d150-tuned \
  --output tools/bench_transport/results/control-example
```

Standalone dummy servers, using the real production server or LeRobot server respectively:

```bash
tools/bench_transport/.venv/bin/python tools/bench_transport/servers.py \
  --port 5558 --chunk 30 --infer-ms 50 --serializer npy
tools/bench_transport/.venv/bin/python tools/bench_transport/servers.py \
  --stack lerobot --port 7447 --chunk 30 --infer-ms 50 \
  --cameras 1 --height 480 --width 640
```

For the unmodified ACT forward path and upstream CLI, generate the checkpoint and use the reference YAML:

```bash
PYTHONPATH=tools/bench_transport tools/bench_transport/.venv/bin/python -c \
  'from servers import make_checkpoint; make_checkpoint(1,480,640,"tools/bench_transport/checkpoints")'
tools/bench_transport/.venv/bin/lerobot-policy-server \
  --config_path=tools/bench_transport/server.yaml
```

The standalone benchmark LeRobot server emits `standalone.server.json`. Matrix runs emit a `.server.json` per LeRobot cell with exact absolute checkpoint path and feature contract. Both JSON and the reference YAML follow upstream `ServerConfig`. Artificial delay is injected only in the benchmark process's bound `predict_action_chunk` method; the installed LeRobot source and checkpoint files are unchanged. The benchmark wraps the complete prediction entry point so the CPU results include real processor overhead, but exclude neural inference. The reference upstream CLI runs the actual ACT model without that hook.

## Full A/B/C matrix with packet accounting and netem

`run_matrix.sh` must run **inside its own network namespace**. It applies delay to that namespace's `lo`; no host interface is shaped. The following uses a separate disposable container based on the existing simulator image only for its Linux runtime. The benchmark interpreter and all its packages still come from the isolated venv.

```bash
bench_repo="$(pwd)"
bench_image="$(docker inspect -f '{{.Image}}' s2r-sim)"
docker run -d --name s2r-transport-netns \
  --cap-add NET_ADMIN --cap-add NET_RAW \
  -v "$HOME:$HOME" -v /tmp:/tmp \
  -w "$bench_repo" --entrypoint bash "$bench_image" \
  -c 'sleep infinity'
docker exec s2r-transport-netns bash -c \
  'apt-get update -qq && apt-get install -y -qq iproute2 tcpdump'
docker network disconnect bridge s2r-transport-netns
docker exec s2r-transport-netns bash tools/bench_transport/run_matrix.sh \
  > tools/bench_transport/results/matrix.log 2>&1
```

Keep the checkout/venv accessible at the same absolute paths inside the container (the command above covers a repository and `uv` installation under the user's home). Run the production CPU suite separately from the measured matrix to avoid load interference. The live-import suite requires the existing Isaac environment:

```bash
docker exec -e S2R_RUN_IMPORT_TESTS=1 -w /workspace/s2r s2r-sim \
  bash -c 'PYTHONPATH=src s2r-entrypoint python -m pytest tests -q -rs' \
  > tools/bench_transport/results/cpu-tests-final.txt 2>&1
bench_test_exit=$?
printf '%s\n' "$bench_test_exit" > tools/bench_transport/results/cpu-tests-final.exitcode
test "$bench_test_exit" -eq 0
```

The original test fixture generated two inapplicable jaw cases that skipped even with imports enabled (`cpu-tests-imports.txt` records that baseline). The approved test-only correction shares cached live imports, retains both embodiments in the generic import test, and gives the two parallel-specific tests only the parallel fixture. No skip or deselection flags are used. The full rerun is recorded in `cpu-tests-final.txt`.

The matrix is **432 cells**: 72 transport payload/stack combinations × three latency settings, plus 216 control cells. Every A/C cell has 50 warmups then 1,000 measured requests. Every B cell runs 1,800 scheduled ticks over 60 wall-clock seconds (3.6 hours of B runs alone). Cases run serially in fresh client/server processes. Completed result files are resume checkpoints: rerunning the same command skips them. Use a new output directory for different iteration counts, code, hardware, inference modes or tuning. Never place a smoke or GPU run in the accepted CPU result directories.

Without packet privileges, both A and B still run through `bench.py` (omit `--capture`), but wire fields are absent and the final audit fails; do not claim complete acceptance. If netem cannot be used on another host, record C as skipped with the actual error rather than fabricating latency results. On this host netem is available in the dedicated namespace.

After all jobs and tests finish:

```bash
# Restore user ownership of files created in the helper container.
docker exec s2r-transport-netns chown -R "$(id -u):$(id -g)" \
  "$bench_repo/tools/bench_transport/results"
OPENBLAS_NUM_THREADS=1 tools/bench_transport/.venv/bin/python \
  tools/bench_transport/report.py --require-complete
git diff --exit-code origin/v2 -- src policies docker configs
# Only after the benchmark process has exited:
docker rm -f s2r-transport-netns
```

The shell trap removes its netem qdisc on normal exit/failure. The final audit checks coverage, sample counts/statistics, request/response byte equality, nonempty loss-free captures, packet-length totals, all 1,800 tick indices, held counts/ages, production invariance and a zero-skip CPU-suite result. Inspect its evidence before accepting the report; it does not infer successful physical behavior.

## Payloads, metrics and raw files

- Six float32 state values, the same instruction, seeded uint8 RGB with 1/2/3 cameras at 480×640 or 224×224, and 16/30/50 × 6 float32 actions. Images are fixed high-entropy noise to avoid an unrealistically JPEG-friendly black frame. Encodings are ZMQ np.save/msgpack, benchmark-only raw bytes/dtype/shape/msgpack, LeRobot raw and LeRobot JPEG q90.
- ZMQ calls the real `PolicyClient`/`PolicyServer`. The raw ablation patches the serializer reference only inside benchmark processes and retains the production transport. LeRobot uses the real loader, saved processors, exclusive session worker, Zenoh transport, admission, `RemoteClient.infer` and aligned `RemoteInferenceEngine`/`dispatch_tick`.
- A RTT includes observation snapshot/client preparation, codec, transport, server processors and validation; payload generation, connection, admission and warmup are outside measurement. Codec time is measured at the actual encode/decode calls on both ends. Do not subtract these percentiles to estimate other stages.
- CPU% is process user+system CPU time / elapsed wall time ×100; 100% means one fully used core. Short run sampling has OS CPU-time quantization. Server CPU includes all its worker/Zenoh threads; the packet sniffer is separate.
- Encoded application bytes exclude ZMTP/Zenoh/TCP headers. Wire fields count captured loopback Ethernet/IP/TCP packet lengths in each direction, including ACKs and protocol framing, divided by measured exchanges. The capture starts after warmup and stops before close. Raw packet files retain header lengths and ports, not image bodies. Loopback GRO/GSO segmentation differs from physical Ethernet; no physical preamble/FCS is invented. TCPdump must report zero kernel drops.
- B uses a fixed wall-clock schedule and retains the previously applied target during missed deadlines. If a call blocks, all scheduled slots before completion are held; the next completed dispatch occupies its completion slot. Startup holds have no action age until the first target. Age includes held targets and takes the oldest source contributing to a blended action. Counts cover startup and steady state; raw traces allow separating them. Replacements count accepted chunks after the first within the 60 s window; executed source transitions are also retained. A final blocking query may return after the window and extend the elapsed CPU sampling interval; that late arrival is recorded separately and excluded from replacement counts.
- Our tuned setup uses the **existing** `RemotePolicy`/`SimulationSmoother`, with P=ceil(delay×30)+2 and blend=2. This helper still blocks for each query and then emulates a P-step delivery delay; it is not a new async implementation. LeRobot uses default refill=.5 s or tuned refill=delay+2/30 s, aligned merging and no blending, with the runtime's adaptive refill floor intact. The raw ZMQ codec is an A/C ablation; B tests the existing production serializer/client against both LeRobot encodings.
- `.json`: per-cell settings and distributions. `.samples.json.gz`: all RTT and four codec samples/byte counts. `.packets.csv.gz`: packet lengths/ports/timestamps. `.ticks.csv.gz`: every scheduled B tick, hold/missed flags, source identity, age and callback duration. `.chunks.csv.gz`: accepted chunk identities and arrival times, including any late arrival. `summary.csv`: flattened complete matrix. `environment.json`: hardware/OS/Python/package versions. `sim-environment.json`/`gpu-driver.txt`: actual simulator/driver inspection. Logs and smoke data are excluded from the report.

## Separate GPU end-to-end follow-up

On the GPU machine, create another isolated environment with matching CUDA wheels **before** installing the locked non-Torch packages; do not install CPU Torch there. Check its driver and the simulator's actual Torch/CUDA build first. Example matching the inspected Isaac image:

```bash
uv venv --python 3.12 tools/bench_transport/.venv-gpu
uv pip install --python tools/bench_transport/.venv-gpu/bin/python \
  'torch==2.11.0' 'torchvision==0.26.0' \
  --index https://download.pytorch.org/whl/cu130
# Install the pinned remote implementation without the CPU-Torch lock entries.
awk '!/^torch==/ && !/^torchvision==/' tools/bench_transport/requirements.lock \
  > /tmp/s2r-bench-gpu-requirements.txt
uv pip install --python tools/bench_transport/.venv-gpu/bin/python \
  -r /tmp/s2r-bench-gpu-requirements.txt
nvidia-smi
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv-gpu/bin/python tools/bench_transport/bench.py \
  --mode A --device cuda --real-forward \
  --output tools/bench_transport/results/gpu-act-transport
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/bench_transport/.venv-gpu/bin/python tools/bench_transport/bench.py \
  --mode B --device cuda --real-forward \
  --output tools/bench_transport/results/gpu-act-control
```

`--real-forward` runs the same actual ACT checkpoint/processor pair on **both** transports, and validates finite outputs rather than fixed zeros. Model compute and GPU↔CPU synchronization are therefore included in RTT. In B, 50/150 ms sleeps are added around the actual prediction on both transports. The ACT follow-up is a reproducible model-inclusive harness check, not a GR00T/MolmoAct2 budget: those actual checkpoints need the adapters/exports and parity checks described in [compatibility](COMPATIBILITY.md). Record model/checkpoint/precision, image preparation, actual inference time, host/device copies and full request-to-dispatch p95/p99 separately on that machine. CPU transport numbers must not be added to GPU medians to claim an end-to-end percentile.
