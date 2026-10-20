#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
image=${TRAIN_IMAGE:-so101-groot-train:n1.7}
base_image=${GROOT_IMAGE:-real-robot:n1.7}
[[ $# == 0 ]] || { printf 'Usage: ./docker/train/build.sh\nConfigure TRAIN_IMAGE or GROOT_IMAGE with environment variables.\n' >&2; exit 1; }
if ! docker image inspect "$base_image" >/dev/null 2>&1; then
    [[ $base_image == real-robot:n1.7 ]] || { printf 'Missing base image: %s\n' "$base_image" >&2; exit 1; }
    bash "$repo_dir/docker/real/build.sh" ada n17
fi
if [[ -n ${TRAIN_WHEEL_DIR:-} ]]; then
    wheel_dir=$(realpath -- "$TRAIN_WHEEL_DIR")
else
    wheel_dir=$(mktemp -d -t so101-training-wheels-XXXXXXXX)
    trap 'rm -rf -- "$wheel_dir"' EXIT
fi
python3 "$repo_dir/docker/train/download_wheels.py" "$wheel_dir"
docker build --network "${TRAIN_BUILD_NETWORK:-host}" --build-context "training_wheels=$wheel_dir" \
    --build-arg "GROOT_IMAGE=$base_image" -t "$image" \
    -f "$repo_dir/docker/train/Dockerfile" "$repo_dir/docker/train"
