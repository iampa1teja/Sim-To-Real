#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
usage() { printf 'Usage: ./docker/real/build.sh <ada|blackwell> [n17|n16]\nDefaults to n16 (legacy real-robot); n17 builds real-robot:n1.7 for policy serving.\n'; }
[[ ${1:-} != --help && ${1:-} != -h ]] || { usage; exit 0; }
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 1; }
arch=$1
generation=${2:-n16}
[[ $arch == ada || $arch == blackwell ]] || { usage >&2; exit 1; }
case "$generation" in
    n17) image=real-robot:n1.7; dockerfile=docker/real/Dockerfile.n17 ;;
    n16) image=real-robot; dockerfile="docker/real/Dockerfile.${arch}" ;;
    *) usage >&2; exit 1 ;;
esac
cd "$repo_dir"
docker build -t "$image" -f "$dockerfile" .
