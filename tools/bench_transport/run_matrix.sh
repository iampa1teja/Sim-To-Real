#!/usr/bin/env bash
# Run INSIDE a dedicated namespace with CAP_NET_ADMIN and CAP_NET_RAW.
# Do not apply this script on the workstation's network namespace.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bench_python=tools/bench_transport/.venv/bin/python
result_root=tools/bench_transport/results
trap 'tc qdisc del dev lo root 2>/dev/null || true' EXIT
ip link set lo up
# Refuse to replace pre-existing shaping.
if [[ $(tc qdisc show dev lo) != 'qdisc noqueue '* ]]; then
    echo 'Expected an unshaped loopback interface in the benchmark namespace.' >&2
    exit 1
fi
tc qdisc show dev lo > "$result_root/netem-before.txt"
"$bench_python" tools/bench_transport/bench.py --mode A --capture --output "$result_root/localhost"
for delay in 1 5; do
    tc qdisc add dev lo root netem delay "${delay}ms"
    tc qdisc show dev lo > "$result_root/netem-${delay}ms.txt"
    "$bench_python" tools/bench_transport/bench.py --mode A --capture --netem-ms "$delay" --output "$result_root/netem-${delay}ms"
    tc qdisc del dev lo root
done
"$bench_python" tools/bench_transport/bench.py --mode B --output "$result_root/control"
"$bench_python" tools/bench_transport/report.py --require-complete
