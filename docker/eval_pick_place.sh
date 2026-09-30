#!/usr/bin/env bash
# Host-side runner; mount list matches docker/README.md and the NVIDIA course.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
model='' dataset='' dr=nodr episodes=20 random_fraction=0.0 start_mode=cycle
eval_set='' splits='' repeats=1 robot_start=recorded
results_json='' external_server=false server_only=false
episode_length_s='' action_horizon=16 embodiment_tag=${EMBODIMENT_TAG:-NEW_EMBODIMENT}
server_image=${SERVER_IMAGE:-real-robot:n1.7}
rename_map=${RENAME_MAP:-'{"realsense_rgb":"room","wrist_cam":"wrist"}'}
lang='Pick up the blue cube and place it in the white box'
lang_by_color='' models_dir=${MODELS_DIR:-$HOME/models} port=5555 gui=false rerun=false dry_run=false
usage() {
    cat <<'EOF'
Usage: ./docker/eval_pick_place.sh --model <relative checkpoint> --dataset <container dataset root>
  [--dr] [--episodes 20] [--random_fraction 0.0] [--start_mode cycle|random]
  [--episode_length_s <seconds>] [--action_horizon 16] [--embodiment_tag NEW_EMBODIMENT]
  [--server_image real-robot:n1.7] [--rename_map <camera-name JSON>]
  [--lang <instruction>] [--lang_by_color <json>] [--models_dir ~/models]
  [--eval_set <container JSON>] [--splits id,ood,yaw,train] [--repeats 1]
  [--robot_start recorded|default]
  [--port 5555] [--gui] [--rerun] [--dry_run]
  [--results_json <absolute container path>] [--external_server | --server_only]
  --external_server uses an existing server; --server_only owns one until interrupted.

Defaults: recorded cube and arm starts only; --random_fraction 0.25 opts into random starts.
Episode length uses the task's configured value (15 s for Pick-Place-Eval).
--episode_length_s must be finite and positive; --action_horizon must be 1..16.
The serving tag uses EMBODIMENT_TAG or NEW_EMBODIMENT; match the training checkpoint.
Serving uses SERVER_IMAGE or real-robot:n1.7 (build: ./docker/real/build.sh ada n17).
Camera names use RENAME_MAP or realsense_rgb->room, wrist_cam->wrist for the SO-arm N1.7 model.
For older N1.6 checkpoints use --server_image real-robot and
  --rename_map '{"realsense_rgb":"front","wrist_cam":"wrist"}'.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --external_server) external_server=true; shift ;;
        --server_only) server_only=true; shift ;;
        --dr) dr=dr; shift ;;
        --gui) gui=true; shift ;;
        --rerun) rerun=true; shift ;;
        --dry_run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --results_json|--model|--dataset|--episodes|--random_fraction|--start_mode|--lang|--lang_by_color|--models_dir|--port|--episode_length_s|--action_horizon|--embodiment_tag|--server_image|--rename_map|--eval_set|--splits|--repeats|--robot_start)
            (($# >= 2)) || fail "Missing value for $1; see --help."
            [[ $1 != --episode_length_s || -n $2 ]] || fail '--episode_length_s must be finite and positive.'
            key=${1#--}; printf -v "$key" '%s' "$2"; shift 2 ;;
        *) fail "Unknown option $1; see --help." ;;
    esac
done
[[ -n $model && -n $dataset ]] || { usage; fail 'Provide --model and --dataset.'; }
[[ $model != /* && /$model/ != */../* && /$model/ != */./* ]] || fail '--model must be a relative path under --models_dir, without . or .. components.'
[[ $dataset == /* ]] || fail '--dataset must be an absolute path inside teleop.'
[[ $repeats =~ ^[1-9][0-9]*$ ]] || fail '--repeats must be a positive integer.'
[[ $robot_start == recorded || $robot_start == default ]] || fail '--robot_start must be recorded or default.'
[[ -n $eval_set || ( -z $splits && $repeats == 1 ) ]] || fail '--splits and --repeats require --eval_set.'
[[ -z $eval_set || $eval_set == /* ]] || fail '--eval_set must be an absolute path inside teleop.'
if [[ -n $splits ]]; then
    [[ $splits =~ ^(id|ood|yaw|train)(,(id|ood|yaw|train))*$ ]] || fail 'Invalid --splits.'
fi
[[ $episodes =~ ^[1-9][0-9]*$ ]] || fail '--episodes must be a positive integer.'
[[ $port =~ ^[0-9]+$ && ${#port} -le 5 ]] || fail '--port must be an integer from 1 to 65535.'
port=$((10#$port))
((port > 0 && port <= 65535)) || fail '--port must be from 1 to 65535.'
[[ $start_mode == cycle || $start_mode == random ]] || fail '--start_mode must be cycle or random.'
[[ -n $embodiment_tag ]] || fail "--embodiment_tag must be the checkpoint's embodiment tag."
[[ -n $server_image && $server_image != -* && $server_image != *[[:space:]]* ]] || fail '--server_image must be a Docker image name.'
command -v python3 >/dev/null || fail 'Install python3 on the host for path, JSON and TCP checks.'
python3 - "$random_fraction" "$lang_by_color" "$dr" "$episode_length_s" "$action_horizon" "$rename_map" <<'PY'
import json, math, sys
try:
    assert 0 <= float(sys.argv[1]) <= 1
    if sys.argv[2]:
        mapping = json.loads(sys.argv[2])
        assert isinstance(mapping, dict)
        required = ('blue', 'red') if sys.argv[3] == 'dr' else ('blue',)
        assert all(isinstance(mapping.get(c), str) and mapping[c].strip() for c in required)
except (ValueError, AssertionError):
    sys.exit('Use --random_fraction in [0,1] and --lang_by_color JSON with a blue instruction (and red with --dr).')
try:
    if sys.argv[4]:
        length = float(sys.argv[4])
        assert math.isfinite(length) and length > 0
except (ValueError, AssertionError):
    sys.exit('--episode_length_s must be finite and positive.')
try:
    assert 1 <= int(sys.argv[5]) <= 16
except (ValueError, AssertionError):
    sys.exit('--action_horizon must be an integer in 1..16 (the model chunk length).')
try:
    mapping = json.loads(sys.argv[6])
    assert isinstance(mapping, dict)
    assert all(isinstance(mapping.get(camera), str) and mapping[camera].strip()
               for camera in ('realsense_rgb', 'wrist_cam'))
    assert mapping['realsense_rgb'] != mapping['wrist_cam']
except (ValueError, AssertionError):
    sys.exit('--rename_map must map realsense_rgb and wrist_cam to distinct nonempty camera names.')
PY
models_dir=$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$models_dir")
model=${model%/}
dataset=${dataset%/}
timestamp=$(date -u +%Y%m%dT%H%M%S%N)
server_name="groot-eval-server-${timestamp}-$$"
results="${dataset}/../eval_results/${model//\//_}_${dr}_${timestamp}.json"
[[ -z $results_json || $results_json == /* ]] || fail "--results_json must be absolute inside teleop."
[[ -z $results_json ]] || results=$results_json
! "$external_server" || ! "$server_only" || fail "Choose only one server mode."
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
    printf '# Evaluation: random_fraction=%s, episode_length_s=%s, action_horizon=%s, robot_start=recorded\n' \
        "$random_fraction" "${episode_length_s:-task default (15 s)}" "$action_horizon"
    printf '# Preflight (checks printed, not executed)\n'
    print_command docker info
    print_command docker inspect -f '{{.State.Running}}' teleop
    print_command docker image inspect "$server_image"
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
    docker image inspect "$server_image" >/dev/null 2>&1 || fail "Build the serving image ($server_image): ./docker/real/build.sh ada n17 (or blackwell n17); use n16 for legacy real-robot."
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
if "$external_server"; then
    : # The shared server owns the port.
elif "$dry_run"; then
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
if [[ -n $eval_set ]]; then
    run docker exec teleop test -f "$eval_set" || fail 'Eval set JSON missing inside teleop.'
fi
# Pinned GR00T uses tyro.cli(ServerConfig): --embodiment-tag accepts a tag name/value.
server_cmd=(docker run -d --rm --name "$server_name" --network host --privileged --gpus all
    -e DISPLAY
    -v /dev:/dev
    -v /run/udev:/run/udev:ro
    -v "$HOME/.Xauthority:/root/.Xauthority"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    -v ./docker/env:/root/env
    -v "$models_dir:/workspace/models"
    -v "$repo_dir/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100"
    "$server_image" python /Isaac-GR00T/gr00t/eval/run_gr00t_server.py
    --model-path "/workspace/models/$model" --embodiment-tag "$embodiment_tag" --port "$port")
if [[ -n $eval_set ]]; then
    for index in "${!server_cmd[@]}"; do
        if [[ ${server_cmd[index]} == /Isaac-GR00T/gr00t/eval/run_gr00t_server.py ]]; then
            server_cmd[index]=/Isaac-GR00T/gr00t/eval/real_robot/SO100/benchmark_server.py
        fi
    done
fi
if ! "$external_server"; then
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
fi # server ownership
if "$server_only"; then
    printf 'SERVER_READY port=%s\n' "$port"
    if ! "$dry_run"; then
        while [[ $(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true) == true ]]; do
            sleep 1 & wait $!
        done
        fail 'Shared GR00T server exited.'
    fi
    exit 0
fi
task=Lerobot-So101-Teleop-Pick-Place-Eval
[[ $dr == nodr ]] || task=Lerobot-So101-Teleop-Pick-Place-DR-Eval
# Use the README's Isaac Python wrapper: docker exec does not inherit the
# environment exported by the container entrypoint. Run the lerobot_eval module.
eval_cmd=(docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.scripts.lerobot_eval --task "$task" --num_envs 1
    --cube_starts "$dataset/pick_place_meta" --start_mode "$start_mode" --random_fraction "$random_fraction"
    --robot_start "$robot_start"
    --rename_map "$rename_map" --action_horizon "$action_horizon"
    --policy_host localhost --policy_port "$port" --lang_instruction "$lang"
    --num_episodes "$episodes" --checkpoint "$model" --results_json "$results")
if [[ -n $eval_set ]]; then
    eval_cmd+=(--eval_set "$eval_set" --repeats "$repeats")
    [[ -z $splits ]] || eval_cmd+=(--splits "$splits")
fi
[[ -z $episode_length_s ]] || eval_cmd+=(--episode_length_s "$episode_length_s")
[[ -z $lang_by_color ]] || eval_cmd+=(--lang_instruction_by_color "$lang_by_color")
"$gui" || eval_cmd+=(--headless)
"$rerun" && eval_cmd+=(--rerun)
run "${eval_cmd[@]}"
# The client prints the grouped summary; also retrieve the saved report.
run docker exec teleop cat "$results"
printf 'Results JSON (inside teleop): %s\n' "$results"
