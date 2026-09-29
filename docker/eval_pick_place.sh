#!/usr/bin/env bash
# Host-side runner; mount list matches docker/README.md and the NVIDIA course.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
model='' dataset='' dr=nodr episodes=20 random_fraction=0.0 start_mode=cycle
lang='Pick up the blue cube and place it in the white box'
lang_by_color='' models_dir=${MODELS_DIR:-$HOME/models} port=5555 gui=false dry_run=false
usage() {
    cat <<'EOF'
Usage: ./docker/eval_pick_place.sh --model <relative checkpoint> --dataset <container dataset root>
  [--dr] [--episodes 20] [--random_fraction 0.25] [--start_mode cycle|random]
  [--lang <instruction>] [--lang_by_color <json>] [--models_dir ~/models]
  [--port 5555] [--gui] [--dry_run]
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --dr) dr=dr; shift ;;
        --gui) gui=true; shift ;;
        --dry_run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --model|--dataset|--episodes|--random_fraction|--start_mode|--lang|--lang_by_color|--models_dir|--port)
            (($# >= 2)) || fail "Missing value for $1; see --help."
            key=${1#--}; printf -v "$key" '%s' "$2"; shift 2 ;;
        *) fail "Unknown option $1; see --help." ;;
    esac
done
[[ -n $model && -n $dataset ]] || { usage; fail 'Provide --model and --dataset.'; }
[[ $model != /* && /$model/ != */../* && /$model/ != */./* ]] || fail '--model must be a relative path under --models_dir, without . or .. components.'
[[ $dataset == /* ]] || fail '--dataset must be an absolute path inside teleop.'
[[ $episodes =~ ^[1-9][0-9]*$ ]] || fail '--episodes must be a positive integer.'
[[ $port =~ ^[0-9]+$ && ${#port} -le 5 ]] || fail '--port must be an integer from 1 to 65535.'
port=$((10#$port))
((port > 0 && port <= 65535)) || fail '--port must be from 1 to 65535.'
[[ $start_mode == cycle || $start_mode == random ]] || fail '--start_mode must be cycle or random.'
command -v python3 >/dev/null || fail 'Install python3 on the host for path, JSON and TCP checks.'
python3 - "$random_fraction" "$lang_by_color" "$dr" <<'PY'
import json, sys
try:
    assert 0 <= float(sys.argv[1]) <= 1
    if sys.argv[2]:
        mapping = json.loads(sys.argv[2])
        assert isinstance(mapping, dict)
        required = ('blue', 'red') if sys.argv[3] == 'dr' else ('blue',)
        assert all(isinstance(mapping.get(c), str) and mapping[c].strip() for c in required)
except (ValueError, AssertionError):
    sys.exit('Use --random_fraction in [0,1] and --lang_by_color JSON with a blue instruction (and red with --dr).')
PY
models_dir=$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$models_dir")
model=${model%/}
dataset=${dataset%/}
timestamp=$(date -u +%Y%m%dT%H%M%S%N)
server_name="groot-eval-server-${timestamp}-$$"
results="${dataset}/../eval_results/${model//\//_}_${dr}_${timestamp}.json"
print_command() { printf '+ '; printf '%q ' "$@"; printf '\n'; }
run() { if "$dry_run"; then print_command "$@"; else "$@"; fi; }
server_started=false
server_log=''
server_log_pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if "$server_started"; then
        run docker stop "$server_name" || true
        # --rm may already have removed it after stop or a server failure.
        run docker rm -f "$server_name" 2>/dev/null || true
    fi
    if [[ -n $server_log_pid ]]; then
        kill "$server_log_pid" 2>/dev/null || true
        wait "$server_log_pid" 2>/dev/null || true
    fi
    [[ -z $server_log ]] || rm -f -- "$server_log"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if "$dry_run"; then
    printf '# Preflight (checks printed, not executed)\n'
    print_command docker info
    print_command docker inspect -f '{{.State.Running}}' teleop
    print_command docker image inspect real-robot
    printf '# Check checkpoint files under %q\n' "$models_dir/$model"
else
    command -v docker >/dev/null || fail 'Install Docker and NVIDIA Container Toolkit; see docker/README.md.'
    docker info >/dev/null 2>&1 || fail 'Start Docker and grant this user Docker access (no sudo is used).'
    if [[ $(docker inspect -f '{{.State.Running}}' teleop 2>/dev/null || true) != true ]]; then
        printf 'Start teleop from the repo root using the README command:\n' >&2
        python3 - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
start = text.index('xhost +')
print(text[start:text.index('```', start)])
PY
        fail 'The teleop container must be running before evaluation.'
    fi
    docker image inspect real-robot >/dev/null 2>&1 || fail 'Build the real-robot image: ./docker/real/build.sh ada (or blackwell).'
    python3 - "$models_dir" "$model" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
checkpoint = (root / sys.argv[2]).resolve()
if not checkpoint.is_relative_to(root) or not checkpoint.is_dir():
    sys.exit('Checkpoint missing or outside MODELS_DIR; fix --models_dir/--model.')
if not (checkpoint / 'config.json').is_file() or not any(checkpoint.glob('*.safetensors')) and not any(checkpoint.glob('pytorch_model*.bin')):
    sys.exit('Checkpoint needs config.json and model weights (*.safetensors or pytorch_model*.bin); pass the checkpoint directory.')
PY
fi
run docker exec teleop test -d "$dataset/pick_place_meta" || fail 'Dataset metadata missing inside teleop; use its mounted /workspace/Sim-to-Real-SO-101-Workshop/datasets path.'
if "$dry_run"; then
    printf '# Verify host port %s is free; wait up to 300 seconds for server readiness\n' "$port"
else
    python3 - "$port" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', int(sys.argv[1])))
except OSError as error:
    sys.exit(f'Port {sys.argv[1]} is unavailable ({error}); stop the listener or choose --port.')
PY
fi
# Pinned GR00T ead52833 uses tyro.cli(ServerConfig); its port field supports --port.
server_cmd=(docker run -d --rm --name "$server_name" --network host --privileged --gpus all
    -e DISPLAY
    -v /dev:/dev
    -v /run/udev:/run/udev:ro
    -v "$HOME/.Xauthority:/root/.Xauthority"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "$HOME/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration"
    -v ./docker/env:/root/env
    -v "$models_dir:/workspace/models"
    -v "$repo_dir/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100"
    real-robot python /Isaac-GR00T/gr00t/eval/run_gr00t_server.py
    --model-path "/workspace/models/$model" --port "$port")
# Arm cleanup before launching so a signal during docker run cannot orphan it.
server_started=true
run "${server_cmd[@]}"
# Stream logs onto the host immediately: --rm removes container logs on exit.
# No new Docker mount is needed, and the trap removes this temporary file.
if "$dry_run"; then
    print_command docker logs --follow "$server_name"
else
    server_log=$(mktemp "${TMPDIR:-/tmp}/groot-eval-log.XXXXXX")
    docker logs --follow "$server_name" >"$server_log" 2>&1 &
    server_log_pid=$!
fi
show_server_logs() {
    if ! docker logs "$server_name" >&2; then
        [[ -z $server_log_pid ]] || wait "$server_log_pid" 2>/dev/null || true
        [[ -z $server_log ]] || cat "$server_log" >&2
    fi
}
if "$dry_run"; then
    print_command docker inspect -f '{{.State.Running}}' "$server_name"
    printf '# On failure/timeout: '
    print_command docker logs "$server_name"
else
    ready=false
    deadline=$((SECONDS + 300))
    while ((SECONDS < deadline)); do
        if [[ $(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true) != true ]]; then
            show_server_logs
            fail 'GR00T server exited; check the model, GPU and real-robot image.'
        fi
        if python3 - "$port" <<'PY'
import socket, sys
try:
    with socket.create_connection(('localhost', int(sys.argv[1])), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
        then ready=true; break; fi
        sleep 2
    done
    if ! "$ready"; then
        show_server_logs
        fail 'GR00T readiness timed out after 300 seconds; inspect the server logs above.'
    fi
fi
task=Lerobot-So101-Teleop-Pick-Place-Eval
[[ $dr == nodr ]] || task=Lerobot-So101-Teleop-Pick-Place-DR-Eval
# Use the README's Isaac Python wrapper: docker exec does not inherit the
# environment exported by the container entrypoint. Run the lerobot_eval module.
eval_cmd=(docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.scripts.lerobot_eval --task "$task" --num_envs 1
    --cube_starts "$dataset/pick_place_meta" --start_mode "$start_mode" --random_fraction "$random_fraction"
    --rename_map '{"realsense_rgb": "front", "wrist_cam": "wrist"}' --action_horizon 16
    --policy_host localhost --policy_port "$port" --lang_instruction "$lang"
    --num_episodes "$episodes" --checkpoint "$model" --results_json "$results")
[[ -z $lang_by_color ]] || eval_cmd+=(--lang_instruction_by_color "$lang_by_color")
"$gui" || eval_cmd+=(--headless)
run "${eval_cmd[@]}"
# The client prints the grouped summary; also retrieve the saved report.
run docker exec teleop cat "$results"
printf 'Results JSON (inside teleop): %s\n' "$results"
