#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() { printf 'Usage: ./policy_servers/build.sh [image tag]\nBuilds policy-server:latest (or the given tag) from the repo root.\n'; }
[[ ${1:-} != --help && ${1:-} != -h ]] || { usage; exit 0; }
[[ $# -le 1 ]] || { usage >&2; exit 1; }
image=${1:-policy-server:latest}
cd "$repo_dir"
docker build -t "$image" -f policy_servers/Dockerfile .
