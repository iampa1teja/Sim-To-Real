#!/usr/bin/env bash
# Host-side evaluation runner. Every setting comes from one YAML file:
#   ./docker/eval.sh pick_place.yaml [--dry_run] [--external_server | --server_only] [--gui] [--rerun]
# A bare file name is looked up under configs/eval/. See configs/eval/pick_place.yaml for the schema.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
usage() {
    cat <<'EOF'
Usage: ./docker/eval.sh <config>.yaml [--dry_run] [--external_server | --server_only] [--gui] [--rerun]

  <config>.yaml      Evaluation settings (policy, checkpoint, dataset, episodes, ...). A bare file
                     name is looked up in configs/eval/.
  --dry_run          Print the docker commands instead of running them.
  --external_server  Use a policy server that is already running on the configured port.
  --server_only      Start the policy server and wait until interrupted; no evaluation.
  --gui, --rerun     Show the simulator window / enable Rerun (also settable under run: in the YAML).

Needs the teleop container (README) running, python3 with PyYAML on the host, and the policy-server
image (./policy_servers/build.sh). READY_TIMEOUT (seconds, default 900) bounds the server start-up,
which includes the first-time uv environment build.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
config='' external_server=false server_only=false dry_run=false gui_flag=false rerun_flag=false
while (($#)); do
    case "$1" in
        --external_server) external_server=true; shift ;;
        --server_only) server_only=true; shift ;;
        --dry_run) dry_run=true; shift ;;
        --gui) gui_flag=true; shift ;;
        --rerun) rerun_flag=true; shift ;;
        --help|-h) usage; exit 0 ;;
        -*) fail "Unknown option $1; see --help." ;;
        *) [[ -z $config ]] || fail 'Give exactly one config file.'; config=$1; shift ;;
    esac
done
[[ -n $config ]] || { usage >&2; fail 'Provide a config YAML, e.g. ./docker/eval.sh pick_place.yaml.'; }
! "$external_server" || ! "$server_only" || fail 'Choose only one server mode.'
command -v python3 >/dev/null || fail 'Install python3 on the host for config, path and TCP checks.'
if [[ ! -f $config && $config != */* && -f configs/eval/$config ]]; then config=configs/eval/$config; fi
[[ -f $config ]] || fail "Config file not found: $config"

# Variables defined by the helper: dr task policy_name model models_dir port server_image dataset
# episodes repeats start_mode random_fraction robot_start robot_usd render_warmup episode_length_s action_horizon
# eval_set splits lang lang_by_color rename_map results_json seed_per_episode gui rerun
# and the arrays policy_options (KEY=VALUE) and grasp_args.
config_vars=$(python3 "$repo_dir/docker/eval_config.py" "$config") || exit 1
eval "$config_vars"
"$gui_flag" && gui=true
"$rerun_flag" && rerun=true

if [[ -z $port ]]; then
    port=$(python3 - "$repo_dir/policies/$policy_name/policy.toml" <<'PY'
import re, sys
try:
    text = open(sys.argv[1]).read()
except OSError:
    sys.exit(f'No manifest at {sys.argv[1]}; check policy.name.')
match = re.search(r'^\s*default_port\s*=\s*(\d+)', text, re.M)
if not match:
    sys.exit(f'{sys.argv[1]} has no default_port; set policy.port in the config.')
print(match.group(1))
PY
    ) || exit 1
fi
[[ -f policies/$policy_name/policy.toml ]] || fail "Unknown policy '$policy_name': no policies/$policy_name/policy.toml."
models_dir=$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$models_dir")
timestamp=$(date -u +%Y%m%dT%H%M%S%N)
server_name="policy-eval-server-${timestamp}-$$"
results="$dataset/../eval_results/${model//\//_}_${dr}_${timestamp}.json"
[[ -z $results_json ]] || results=$results_json
teleop_policies=${TELEOP_POLICIES_DIR:-/workspace/Sim-to-Real-SO-101-Workshop/policies}

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
    printf '# Config %s: policy=%s checkpoint=%s port=%s\n' "$config" "$policy_name" "$model" "$port"
    printf '# Evaluation: random_fraction=%s, episode_length_s=%s, action_horizon=%s, robot_start=%s\n' \
        "$random_fraction" "${episode_length_s:-task default (15 s)}" "$action_horizon" "$robot_start"
    printf '# Preflight (checks printed, not executed)\n'
    print_command docker info
    print_command docker inspect -f '{{.State.Running}}' teleop
    "$external_server" || print_command docker image inspect "$server_image"
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
    if ! "$external_server"; then
        docker image inspect "$server_image" >/dev/null 2>&1 \
            || fail "Build the policy server image ($server_image): ./policy_servers/build.sh"
        python3 - "$models_dir" "$model" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
checkpoint = (root / sys.argv[2]).resolve()
if not checkpoint.is_relative_to(root) or not checkpoint.is_dir():
    sys.exit('Checkpoint missing or outside policy.models_dir; fix policy.models_dir / policy.checkpoint.')
if not (checkpoint / 'config.json').is_file() or not any(checkpoint.glob('*.safetensors')) and not any(checkpoint.glob('pytorch_model*.bin')):
    sys.exit('Checkpoint needs config.json and model weights (*.safetensors or pytorch_model*.bin); point at the checkpoint directory.')
PY
    fi
fi
run docker exec teleop test -d "$dataset/pick_place_meta" || fail 'Dataset metadata missing inside teleop; use its mounted /workspace/Sim-to-Real-SO-101-Workshop/datasets path.'
run docker exec teleop test -d "$teleop_policies" || fail "Policies folder missing inside teleop ($teleop_policies); mount ./policies there (README) or set TELEOP_POLICIES_DIR."
if [[ -n $eval_set ]]; then
    run docker exec teleop test -f "$eval_set" || fail 'Eval set JSON missing inside teleop.'
fi
if [[ -n $robot_usd ]]; then
    run docker exec teleop test -f "$robot_usd" || fail 'Robot USD missing inside teleop; check eval.robot_usd.'
fi
if "$external_server"; then
    : # The shared server owns the port.
elif "$dry_run"; then
    printf '# Verify host port %s is free; wait up to %s seconds for server readiness\n' "$port" "${READY_TIMEOUT:-900}"
else
    python3 - "$port" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', int(sys.argv[1])))
except OSError as error:
    sys.exit(f'Port {sys.argv[1]} is unavailable ({error}); stop the listener or change policy.port.')
PY
fi

# One generic image serves every policy: policies/ is mounted, and each policy builds its own uv
# environment on the policy-cache volume the first time it starts.
server_cmd=(docker run -d --rm --name "$server_name" --network host --gpus all
    -v "$repo_dir/policies:/policies"
    -v policy-cache:/opt/cache
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    -v "$models_dir:/workspace/models"
    "$server_image" --policy "$policy_name" --checkpoint "/workspace/models/$model" --port "$port")
client_options=()
for item in "${policy_options[@]}"; do
    server_cmd+=(--option "$item")
    client_options+=(--policy_option "$item")
done

if ! "$external_server"; then
    # Arm cleanup before launching so a signal during docker run cannot orphan the container.
    server_started=true
    run "${server_cmd[@]}"
    # Stream logs onto the host immediately: --rm removes container logs on exit.
    if "$dry_run"; then
        print_command docker logs --follow "$server_name"
    else
        server_log=$(mktemp "${TMPDIR:-/tmp}/policy-eval-log.XXXXXX")
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
        ready_timeout=${READY_TIMEOUT:-900}
        deadline=$((SECONDS + ready_timeout))
        while ((SECONDS < deadline)); do
            if [[ $(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true) != true ]]; then
                show_server_logs
                fail "The $policy_name policy server exited; check the checkpoint, GPU and server logs above."
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
            fail "Server readiness timed out after $ready_timeout seconds; inspect the logs above."
        fi
    fi
fi
if "$server_only"; then
    printf 'SERVER_READY port=%s\n' "$port"
    if ! "$dry_run"; then
        while [[ $(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true) == true ]]; do
            sleep 1 & wait $!
        done
        fail 'Policy server exited.'
    fi
    exit 0
fi

# Use the README's Isaac Python wrapper: docker exec does not inherit the environment exported by the
# container entrypoint. Run the lerobot_eval module.
run docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m pip install tqdm
eval_cmd=(docker exec -e "POLICIES_DIR=$teleop_policies" teleop /workspace/isaaclab/_isaac_sim/python.sh
    -m sim_to_real_so101.scripts.lerobot_eval --task "$task" --num_envs 1
    --cube_starts "$dataset/pick_place_meta" --start_mode "$start_mode" --random_fraction "$random_fraction"
    --robot_start "$robot_start"
    --rename_map "$rename_map" --action_horizon "$action_horizon"
    --policy "$policy_name" "${client_options[@]}"
    --policy_host localhost --policy_port "$port" --lang_instruction "$lang"
    --num_episodes "$episodes" --checkpoint "$model" --results_json "$results")
if [[ -n $eval_set ]]; then
    eval_cmd+=(--eval_set "$eval_set" --repeats "$repeats")
    [[ -z $splits ]] || eval_cmd+=(--splits "$splits")
fi
eval_cmd+=("${grasp_args[@]}")
[[ -z $episode_length_s ]] || eval_cmd+=(--episode_length_s "$episode_length_s")
[[ -z $robot_usd ]] || eval_cmd+=(--robot_usd "$robot_usd")
[[ $render_warmup == 0 ]] || eval_cmd+=(--render_warmup "$render_warmup")
[[ -z $lang_by_color ]] || eval_cmd+=(--lang_instruction_by_color "$lang_by_color")
"$gui" || eval_cmd+=(--headless)
"$rerun" && eval_cmd+=(--rerun)
run "${eval_cmd[@]}"
# The client prints the grouped summary; also retrieve the saved report.
run docker exec teleop cat "$results"
printf 'Results JSON (inside teleop): %s\n' "$results"
