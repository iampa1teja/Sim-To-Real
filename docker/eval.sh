#!/usr/bin/env bash
# Host-side evaluation runner. Every setting comes from one YAML file:
#   ./docker/eval.sh pick_place.yaml [--dry_run] [--external_server | --server_only] [--gui] [--rerun] [--save_videos] [--resume [RESULTS.json]] [--smoke-test]
# A bare file name is looked up under configs/eval/. See configs/eval/pick_place.yaml for the schema.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
usage() {
    cat <<'EOF'
Usage: ./docker/eval.sh <config>.yaml [--dry_run] [--external_server | --server_only] [--gui] [--rerun] [--save_videos] [--resume [RESULTS.json]] [--smoke-test]

  <config>.yaml      Evaluation settings (policy, checkpoint, dataset, episodes, ...). A bare file
                     name is looked up in configs/eval/.
  --dry_run          Print the docker commands instead of running them.
  --external_server  Use a policy server that is already running on the configured port.
  --server_only      Start the policy server and wait until interrupted; no evaluation.
  --gui, --rerun     Show the simulator window / enable Rerun (also settable under run: in the YAML).
  --save_videos      Save room/wrist MP4 videos for each attempt (also eval.save_videos in YAML).
  --resume [JSON]    Continue the latest matching fixed-benchmark report, or an explicit absolute
                     results JSON path inside teleop. Completed attempts and videos are retained.
  --smoke-test       Run five episodes, or one per zone when smoke_test.selection=one_per_zone.
                     Rerun is enabled; no reports, videos, CSVs or run logs are saved.
                     Optional smoke_test deadlines stop stalled attempts. Overrides output settings.

Needs the teleop container (README) running, python3 with PyYAML on the host, and the policy-server
image (./policy_servers/build.sh). READY_TIMEOUT (seconds, default 900) bounds the server start-up,
which includes the first-time uv environment build.
Complete logs are retained under /tmp/so101-eval (override with EVAL_LOG_DIR), except in smoke mode.
Warnings and routine startup details are hidden from the default console; errors remain visible.
Set EVAL_VERBOSE=1 to print the complete output, including warnings.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
config='' external_server=false server_only=false dry_run=false gui_flag=false rerun_flag=false save_videos_flag=false smoke_test=false
resume=false resume_path=''
while (($#)); do
    case "$1" in
        --external_server) external_server=true; shift ;;
        --server_only) server_only=true; shift ;;
        --dry_run) dry_run=true; shift ;;
        --gui) gui_flag=true; shift ;;
        --rerun) rerun_flag=true; shift ;;
        --save_videos) save_videos_flag=true; shift ;;
        --resume)
            resume=true; shift
            if (($#)) && [[ $1 == *.json ]]; then resume_path=$1; shift; fi ;;
        --smoke-test) smoke_test=true; shift ;;
        --help|-h) usage; exit 0 ;;
        -*) fail "Unknown option $1; see --help." ;;
        *) [[ -z $config ]] || fail 'Give exactly one config file.'; config=$1; shift ;;
    esac
done
[[ -n $config ]] || { usage >&2; fail 'Provide a config YAML, e.g. ./docker/eval.sh pick_place.yaml.'; }
! "$external_server" || ! "$server_only" || fail 'Choose only one server mode.'
! "$smoke_test" || ! "$server_only" || fail '--smoke-test cannot be combined with --server_only.'
! "$resume" || { ! "$smoke_test" && ! "$server_only"; } || fail '--resume cannot be combined with --smoke-test or --server_only.'
[[ -z $resume_path || $resume_path == /* ]] || fail '--resume JSON must be an absolute path inside teleop.'
command -v python3 >/dev/null || fail 'Install python3 on the host for config, path and TCP checks.'
if [[ ! -f $config && $config != */* && -f configs/eval/$config ]]; then config=configs/eval/$config; fi
[[ -f $config ]] || fail "Config file not found: $config"

# Variables defined by the helper: dr task policy_name model models_dir port server_image dataset
# episodes repeats start_mode random_fraction robot_start robot_usd render_warmup episode_length_s action_horizon action_repeat
# eval_set splits lang lang_by_color rename_map results_json save_videos video_dir video_encoder seed_per_episode gui rerun
# and the arrays policy_options (KEY=VALUE) and grasp_args.
config_args=("$config")
config_vars=$(python3 "$repo_dir/docker/eval_config.py" "${config_args[@]}") || exit 1
eval "$config_vars"
if ! "$smoke_test" && [[ $smoke_selection != first_five || -n $smoke_reach_timeout_s ]]; then
    fail 'This smoke protocol requires --smoke-test; it cannot change a normal evaluation.'
fi
! "$resume" || [[ -n $eval_set ]] || fail '--resume requires eval.eval_set (a fixed benchmark schedule).'
"$gui_flag" && gui=true
"$rerun_flag" && rerun=true
"$save_videos_flag" && save_videos=true
if "$smoke_test"; then
    episodes=5
    rerun=true
    save_videos=false
    video_dir=''
    results_json=''
fi

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
run_logged() {
    local logfile=$1
    shift
    if "$dry_run"; then print_command "$@"
    else
        # A background command plus the wait builtin lets Bash handle TERM immediately.
        local log_args=(--log "$logfile")
        if "$smoke_test"; then log_args=(--no-log); fi
        python3 "$repo_dir/docker/eval_output.py" "${log_args[@]}" -- "$@" &
        active_log_pid=$!
        local command_status=0
        wait "$active_log_pid" || command_status=$?
        active_log_pid=''
        return "$command_status"
    fi
}
show_log() {
    if ! "$smoke_test"; then python3 "$repo_dir/docker/eval_output.py" --tail "$1"; fi
}
check_rerun_viewer() {
    if [[ -n $rerun_log_pid ]] && ! kill -0 "$rerun_log_pid" 2>/dev/null; then
        show_log "$rerun_log"
        fail 'Rerun viewer exited; inspect the diagnostics above.'
    fi
}
server_started=false
server_log=''
server_log_pid=''
server_failure_shown=false
evaluator_started=false
active_log_pid=''
eval_run_id=${timestamp}-$$
rerun_log_pid=''
rerun_log=''
rerun_connect_url=${RERUN_CONNECT_URL:-}
log_dir=${EVAL_LOG_DIR:-${TMPDIR:-/tmp}/so101-eval/${timestamp}-$$}
cleanup() {
    local status=$?
    trap - EXIT
    # A second Ctrl+C must not interrupt cleanup and leave the policy port occupied.
    trap '' INT TERM
    # Reap the host CLI before inspecting /proc, so an in-flight exec cannot race the first scan.
    if [[ -n $active_log_pid ]]; then
        kill "$active_log_pid" 2>/dev/null || true
        wait "$active_log_pid" 2>/dev/null || true
    fi
    if "$evaluator_started" && ! "$dry_run"; then
        # Stopping docker exec alone leaves Isaac running. Match only this invocation's tag.
        local cleanup_code cleanup_output
        cleanup_code=$(<"$repo_dir/docker/eval_cleanup.py")
        if cleanup_output=$(docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -c "$cleanup_code" "$eval_run_id" 2>&1); then
            if ! "$smoke_test"; then printf '%s\n' "$cleanup_output" >"$log_dir/evaluator-cleanup.log"; fi
        else
            printf '%s\nWARNING: Evaluation cleanup failed.\n' "$cleanup_output" >&2
            if ! "$smoke_test"; then printf '%s\n' "$cleanup_output" >"$log_dir/evaluator-cleanup.log"; fi
        fi
    fi
    if "$server_started"; then
        if "$dry_run"; then
            print_command docker stop "$server_name"
            print_command docker rm -f "$server_name"
        elif "$smoke_test"; then
            docker stop "$server_name" >/dev/null 2>&1 || true
            docker rm -f "$server_name" >/dev/null 2>&1 || true
        else
            docker stop "$server_name" >>"$log_dir/cleanup.log" 2>&1 || true
            # --rm may already have removed it after stop or a server failure.
            docker rm -f "$server_name" >>"$log_dir/cleanup.log" 2>&1 || true
        fi
    fi
    if [[ -n $server_log_pid ]]; then
        kill "$server_log_pid" 2>/dev/null || true
        wait "$server_log_pid" 2>/dev/null || true
    fi
    if [[ -n $rerun_log_pid ]]; then
        kill "$rerun_log_pid" 2>/dev/null || true
        wait "$rerun_log_pid" 2>/dev/null || true
    fi
    if ((status != 0 && status != 130 && status != 143)) && [[ -n $server_log && -f $server_log ]] && ! "$server_failure_shown"; then
        python3 "$repo_dir/docker/eval_output.py" --tail "$server_log"
    fi
    if ((status == 130 || status == 143)) && ! "$smoke_test"; then
        printf 'Evaluation interrupted; completed episodes and existing videos are retained. Use --resume to continue.\n'
    fi
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
if "$resume"; then
    resume_target=${resume_path:-${results_json:-${results%/*}}}
    if "$dry_run"; then
        print_command docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.utils.eval_resume \
            "$resume_target" "${model//\//_}_${dr}_*.json" "$config_fingerprint" "$model" "$task"
        results=${resume_path:-${results_json:-${results%/*}/LATEST_MATCHING_RESULTS.json}}
    else
        results=$(docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -m sim_to_real_so101.utils.eval_resume \
            "$resume_target" "${model//\//_}_${dr}_*.json" "$config_fingerprint" "$model" "$task") || exit 1
        printf 'Resuming results: %s\n' "$results"
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
if ! "$dry_run"; then
    printf 'Evaluation: policy=%s checkpoint=%s port=%s\n' "$policy_name" "$model" "$port"
    if ! "$smoke_test"; then
        mkdir -p -- "$log_dir"
        printf 'Logs: %s\n' "$log_dir"
    fi
fi
if "$smoke_test"; then
    coverage='5 episodes'
    [[ $smoke_selection != one_per_zone ]] || coverage='one original example per zone'
    printf 'Smoke test: %s with Rerun; no evaluation artifacts will be saved.\n' "$coverage"
fi
if "$rerun" && ! "$server_only" && [[ -z $rerun_connect_url ]]; then
    # Use a native viewer matching teleop's SDK; fail before loading policy weights if unavailable.
    rerun_port=${RERUN_PORT:-9876}
    rerun_connect_url="rerun+http://127.0.0.1:$rerun_port/proxy"
    if "$dry_run"; then
        print_command python3 "$repo_dir/docker/rerun_viewer.py" --container teleop --port "$rerun_port"
    else
        python3 - "$rerun_port" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', int(sys.argv[1])))
except (OSError, ValueError, OverflowError) as error:
    sys.exit(f'Rerun port {sys.argv[1]} is unavailable ({error}); set RERUN_PORT or RERUN_CONNECT_URL.')
PY
        rerun_log=$log_dir/rerun.log
        log_args=(--log "$rerun_log")
        if "$smoke_test"; then log_args=(--no-log); fi
        python3 "$repo_dir/docker/eval_output.py" "${log_args[@]}" -- \
            python3 "$repo_dir/docker/rerun_viewer.py" --container teleop --port "$rerun_port" &
        rerun_log_pid=$!
        rerun_ready=false
        rerun_deadline=$((SECONDS + ${RERUN_READY_TIMEOUT:-60}))
        while ((SECONDS < rerun_deadline)); do
            if ! kill -0 "$rerun_log_pid" 2>/dev/null; then
                wait "$rerun_log_pid" 2>/dev/null || true
                show_log "$rerun_log"
                fail 'Rerun viewer exited before startup; evaluation was not started.'
            fi
            if python3 - "$rerun_port" <<'PY'
import socket, sys
try:
    with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
            then rerun_ready=true; break; fi
            sleep 1
        done
        if ! "$rerun_ready"; then
            show_log "$rerun_log"
            fail 'Rerun viewer readiness timed out; evaluation was not started.'
        fi
        printf 'Rerun viewer ready: %s\n' "$rerun_connect_url"
    fi
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
    if "$dry_run"; then print_command "${server_cmd[@]}"
    else
        printf 'Starting policy server; loading %s...\n' "$model"
        if "$smoke_test"; then "${server_cmd[@]}" >/dev/null
        else "${server_cmd[@]}" >"$log_dir/server-container.txt"; fi
    fi
    # Stream logs onto the host immediately: --rm removes container logs on exit.
    if "$dry_run"; then
        print_command docker logs --follow "$server_name"
    else
        log_args=(--no-log)
        if ! "$smoke_test"; then
            server_log=$log_dir/server.log
            log_args=(--log "$server_log")
        fi
        python3 "$repo_dir/docker/eval_output.py" "${log_args[@]}" -- docker logs --follow "$server_name" &
        server_log_pid=$!
    fi
    show_server_logs() {
        server_failure_shown=true
        if "$smoke_test"; then docker logs --tail 50 "$server_name" >&2 || true
        else show_log "$server_log"; fi
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
            check_rerun_viewer
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
if ! "$dry_run"; then printf 'Using policy server on port %s.\n' "$port"; fi
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
# Avoid installing (and warning about root pip) on every evaluation in the prebuilt simulator.
dependency_output=$log_dir/dependencies.log
if "$smoke_test"; then dependency_output=/dev/null; fi
if "$dry_run"; then
    print_command docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -c 'import tqdm'
elif ! docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh -c 'import tqdm' >"$dependency_output" 2>&1; then
    printf 'Installing missing evaluator dependency: tqdm...\n'
    run_logged "$log_dir/dependencies-install.log" docker exec teleop /workspace/isaaclab/_isaac_sim/python.sh \
        -m pip install --quiet --disable-pip-version-check --root-user-action=ignore tqdm
fi
check_rerun_viewer
rerun_env=()
if "$rerun"; then rerun_env=(-e "RERUN_CONNECT_URL=$rerun_connect_url"); fi
eval_cmd=(docker exec -e "POLICIES_DIR=$teleop_policies" -e "SO101_EVAL_RUN_ID=$eval_run_id" -e PYTHONUNBUFFERED=1 "${rerun_env[@]}" teleop /workspace/isaaclab/_isaac_sim/python.sh
    -m sim_to_real_so101.scripts.lerobot_eval --task "$task" --num_envs 1
    --cube_starts "$dataset/pick_place_meta" --start_mode "$start_mode" --random_fraction "$random_fraction"
    --robot_start "$robot_start"
    --rename_map "$rename_map" --action_horizon "$action_horizon" --action_repeat "$action_repeat"
    --policy "$policy_name" "${client_options[@]}"
    --policy_host localhost --policy_port "$port" --lang_instruction "$lang"
    --num_episodes "$episodes" --checkpoint "$model" --config_fingerprint "$config_fingerprint")
"$resume" && eval_cmd+=(--resume)
if "$smoke_test"; then
    eval_cmd+=(--smoke-test --smoke-selection "$smoke_selection")
    if [[ -n $smoke_reach_timeout_s ]]; then
        eval_cmd+=(--smoke-reach-timeout-s "$smoke_reach_timeout_s"
                   --smoke-progress-timeout-s "$smoke_progress_timeout_s")
    fi
else eval_cmd+=(--results_json "$results"); fi
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
"$save_videos" && eval_cmd+=(--save_videos)
if ! "$smoke_test"; then eval_cmd+=(--video_encoder "$video_encoder"); fi
[[ -z $video_dir ]] || eval_cmd+=(--video_dir "$video_dir")
evaluator_started=true
run_logged "$log_dir/evaluation.log" "${eval_cmd[@]}"
# The client prints the grouped summary; keep the full report on disk without dumping it twice.
if ! "$smoke_test"; then
    if "$dry_run"; then print_command docker exec teleop cat "$results"
    else docker exec teleop cat "$results" >"$log_dir/results.json"; fi
    printf 'Results JSON (inside teleop): %s\n' "$results"
fi
