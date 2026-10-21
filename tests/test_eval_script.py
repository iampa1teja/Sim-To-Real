"""docker/eval.sh and docker/eval_config.py: YAML validation and the commands --dry_run prints."""
import os
import json
import signal
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker/eval.sh"
DATASET = "/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1"

BASE = f"""
policy:
  checkpoint: example/checkpoint-200000
eval:
  dataset: {DATASET}
"""


def run(tmp_path, yaml_text, *flags, name="case.yaml"):
    config = tmp_path / name
    config.write_text(textwrap.dedent(yaml_text))
    return subprocess.run(["bash", str(SCRIPT), str(config), "--dry_run", *flags],
                          capture_output=True, text=True, cwd=ROOT)


def commands(result):
    return [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]


def find(result, prefix):
    return next(line for line in commands(result) if line.startswith(prefix))


def test_defaults_start_the_policy_server_and_the_evaluator(tmp_path):
    result = run(tmp_path, BASE)
    assert result.returncode == 0, result.stderr
    server = find(result, "docker run")
    assert "policy-server:latest --policy groot_n17 --checkpoint /workspace/models/example/checkpoint-200000 --port 5555" in server
    assert "-v policy-cache:/opt/cache" in server and "--gpus all" in server
    client = find(result, "docker exec -e POLICIES_DIR")
    for expected in ("--task Lerobot-So101-Teleop-Pick-Place-Eval", "--policy groot_n17", "--policy_port 5555",
                     "--num_episodes 20", "--action_horizon 16", "--action_repeat 1", "--headless",
                     f"--cube_starts {DATASET}/pick_place_meta", "--checkpoint example/checkpoint-200000"):
        assert expected in client
    assert "--eval_set" not in client and "--gripper_latch" not in client
    assert "--robot_usd" not in client
    assert "--molmo-overhead-camera" not in client
    assert "-e SO101_EVAL_RUN_ID=" in client
    assert "-e PYTHONUNBUFFERED=1" in client
    assert "--save_videos" not in client and "--video_dir" not in client


def test_save_videos_cli_and_yaml_options_reach_evaluator(tmp_path):
    result = run(tmp_path, BASE, "--save_videos")
    assert result.returncode == 0, result.stderr
    client = find(result, "docker exec -e POLICIES_DIR")
    assert "--save_videos" in client and "--video_dir" not in client
    result = run(tmp_path, BASE + "  save_videos: true\n  video_dir: /workspace/outputs/custom_videos\n")
    assert result.returncode == 0, result.stderr
    client = find(result, "docker exec -e POLICIES_DIR")
    assert "--save_videos" in client and "--video_dir /workspace/outputs/custom_videos" in client


def test_resume_selects_latest_matching_or_explicit_report_and_preserves_video_flag(tmp_path):
    config = BASE + "  eval_set: /workspace/starts.json\n"
    for path in (None, "/workspace/results/previous.json"):
        flags = ["--resume"] + ([path] if path else [])
        result = run(tmp_path, config, "--save_videos", *flags)
        assert result.returncode == 0, result.stderr
        client = find(result, "docker exec -e POLICIES_DIR")
        assert "--resume" in client and "--save_videos" in client
        assert "--config_fingerprint" in client
        assert (path or "LATEST_MATCHING_RESULTS.json") in client
        assert "sim_to_real_so101.utils.eval_resume" in result.stdout


@pytest.mark.parametrize("extra", [["--smoke-test"], ["--server_only"]])
def test_resume_rejects_incompatible_modes(tmp_path, extra):
    result = run(tmp_path, BASE + "  eval_set: /workspace/starts.json\n", "--resume", *extra)
    assert result.returncode != 0 and "--resume" in result.stderr


def test_resume_requires_fixed_benchmark_and_absolute_explicit_path(tmp_path):
    result = run(tmp_path, BASE, "--resume")
    assert result.returncode != 0 and "eval.eval_set" in result.stderr
    result = run(tmp_path, BASE + "  eval_set: /workspace/starts.json\n", "--resume", "relative.json")
    assert result.returncode != 0 and "absolute path" in result.stderr


def test_smoke_test_forces_five_rerun_episodes_and_overrides_saved_outputs(tmp_path):
    result = run(tmp_path, BASE + """  episodes: 99
  eval_set: /workspace/fixed_eval_set.json
  splits: [id, train]
  save_videos: true
  video_dir: /workspace/outputs/ignored_videos
  results_json: /workspace/outputs/ignored_report.json
run:
  rerun: false
""", "--smoke-test", "--save_videos")
    assert result.returncode == 0, result.stderr
    client = find(result, "docker exec -e POLICIES_DIR")
    for expected in ("--num_episodes 5", "--smoke-test", "--rerun", "--eval_set /workspace/fixed_eval_set.json"):
        assert expected in client
    for forbidden in ("--results_json", "--save_videos", "--video_dir"):
        assert forbidden not in client
    assert not any(line.startswith("docker exec teleop cat ") for line in commands(result))


def test_smoke_test_rejects_server_only(tmp_path):
    result = run(tmp_path, BASE, "--smoke-test", "--server_only")
    assert result.returncode != 0
    assert "smoke" in result.stderr.lower() and "server_only" in result.stderr


@pytest.mark.parametrize("mode, existing_log_dir", [("success", False), ("failure", True), ("interrupt", False)])
def test_smoke_lifecycle_creates_no_output_files_even_on_failure_or_interrupt(tmp_path, mode, existing_log_dir):
    # The fake CLI writes harness markers only; evaluator output directories must remain untouched.
    docker = tmp_path / "docker"
    calls, ready, cleaned = (tmp_path / name for name in ("calls.jsonl", "ready", "cleaned"))
    docker.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
with open({str(calls)!r}, "a") as output:
    output.write(json.dumps(args) + "\\n")
if args[0] == "inspect":
    print("true")
elif "sim_to_real_so101.scripts.lerobot_eval" in args:
    assert args[args.index("--num_episodes")+1] == "5", args
    assert "--smoke-test" in args and "--rerun" in args, args
    assert not any(flag in args for flag in ["--results_json", "--save_videos", "--video_dir"]), args
    print("WARNING: smoke diagnostic remains visible", flush=True)
    if {mode!r} == "interrupt":
        pathlib.Path({str(ready)!r}).write_text("started")
        time.sleep(60)
    if {mode!r} == "failure":
        print("ERROR: smoke inference failed", flush=True)
        sys.exit(7)
    print("[RESULT] Smoke episodes: 5", flush=True)
elif args[:3] == ["exec", "teleop", "cat"]:
    sys.exit("Smoke test must not retrieve a saved report")
elif args[:4] == ["exec", "teleop", "/workspace/isaaclab/_isaac_sim/python.sh", "-c"] and "SO101_EVAL_RUN_ID=" in args[4]:
    pathlib.Path({str(cleaned)!r}).write_text(args[5])
''')
    docker.chmod(0o755)
    config = tmp_path / "case.yaml"
    config.write_text(textwrap.dedent(BASE + "  save_videos: true\n  video_dir: /workspace/ignored_videos\n"
                                      "  results_json: /workspace/ignored_report.json\n"))
    log_dir, runtime_dir = tmp_path / "must_not_write_logs", tmp_path / "runtime"
    runtime_dir.mkdir()
    if existing_log_dir:
        log_dir.mkdir()
        (log_dir / "sentinel").write_bytes(b"keep existing log directory untouched")
    before = {path.name: path.read_bytes() for path in log_dir.iterdir()} if log_dir.exists() else None
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}", "EVAL_LOG_DIR": str(log_dir),
                        "TMPDIR": str(runtime_dir), "RERUN_CONNECT_URL": "rerun+http://127.0.0.1:9999/proxy"}
    process = subprocess.Popen(["bash", str(SCRIPT), str(config), "--external_server", "--smoke-test", "--save_videos"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
                               env=env, start_new_session=True)
    try:
        if mode == "interrupt":
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == {"success": 0, "failure": 7, "interrupt": 143}[mode], stderr
        assert "smoke diagnostic remains visible" not in stdout
        if mode == "failure":
            assert "smoke inference failed" in stdout + stderr
        if mode == "success":
            assert "Smoke episodes: 5" in stdout
        assert "Diagnostic log:" not in stderr
        assert cleaned.read_text()
        after = {path.name: path.read_bytes() for path in log_dir.iterdir()} if log_dir.exists() else None
        assert after == before
        assert list(runtime_dir.iterdir()) == []
        recorded_calls = [json.loads(line) for line in calls.read_text().splitlines()]
        assert not any(args[:3] == ["exec", "teleop", "cat"] for args in recorded_calls)
        assert not any(args[0] == "run" for args in recorded_calls)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


@pytest.mark.parametrize("value", ['"yes"', "1", "null"])
def test_video_save_option_requires_yaml_boolean(tmp_path, value):
    result = run(tmp_path, BASE + f"  save_videos: {value}\n")
    assert result.returncode != 0
    assert "eval.save_videos" in result.stderr


@pytest.mark.parametrize("value", ["relative/videos", "true", "123"])
def test_video_directory_requires_absolute_container_path(tmp_path, value):
    result = run(tmp_path, BASE + f"  video_dir: {value}\n")
    assert result.returncode != 0
    assert "eval.video_dir" in result.stderr


@pytest.mark.parametrize("encoder", ["libx264", "h264_nvenc"])
def test_video_encoder_reaches_evaluator_without_changing_resume_identity(tmp_path, encoder):
    base = run(tmp_path, BASE, "--save_videos")
    selected = run(tmp_path, BASE + f"  video_encoder: {encoder}\n", "--save_videos")
    assert selected.returncode == 0, selected.stderr
    client = shlex.split(find(selected, "docker exec -e POLICIES_DIR"))
    original = shlex.split(find(base, "docker exec -e POLICIES_DIR"))
    assert client[client.index("--video_encoder") + 1] == encoder
    assert client[client.index("--config_fingerprint") + 1] == original[original.index("--config_fingerprint") + 1]
    smoke = run(tmp_path, BASE + f"  video_encoder: {encoder}\n", "--smoke-test")
    assert smoke.returncode == 0, smoke.stderr
    assert "--video_encoder" not in find(smoke, "docker exec -e POLICIES_DIR")


@pytest.mark.parametrize("value", ["unknown", "true", "null", "42"])
def test_video_encoder_rejects_unsupported_values_before_launch(tmp_path, value):
    result = run(tmp_path, BASE + f"  video_encoder: {value}\n")
    assert result.returncode != 0 and "eval.video_encoder" in result.stderr


def test_policy_options_reach_server_and_client(tmp_path):
    result = run(tmp_path, f"""
policy:
  checkpoint: example/checkpoint-200000
  port: 6001
  options:
    embodiment_tag: OTHER_TAG
eval:
  dataset: {DATASET}
""")
    assert result.returncode == 0, result.stderr
    assert "--option embodiment_tag=OTHER_TAG" in find(result, "docker run")
    assert "--port 6001" in find(result, "docker run")
    client = find(result, "docker exec -e POLICIES_DIR")
    assert "--policy_option embodiment_tag=OTHER_TAG" in client and "--policy_port 6001" in client


@pytest.mark.parametrize("horizon", [None, 1, 30])
def test_molmo_horizon_uses_full_chunk_default_and_accepts_boundaries(tmp_path, horizon):
    config = BASE.replace("policy:\n", "policy:\n  name: molmoact2\n", 1)
    if horizon is not None:
        config += f"  action_horizon: {horizon}\n"
    result = run(tmp_path, config)
    assert result.returncode == 0, result.stderr
    expected = 30 if horizon is None else horizon
    assert f"--action_horizon {expected}" in find(result, "docker exec -e POLICIES_DIR")


def test_molmo_horizon_above_chunk_length_is_rejected_before_launch(tmp_path):
    config = BASE.replace("policy:\n", "policy:\n  name: molmoact2\n", 1) + "  action_horizon: 31\n"
    result = run(tmp_path, config)
    assert result.returncode != 0
    assert "1..30" in result.stderr and "molmoact2" in result.stderr
    assert not commands(result)


@pytest.mark.parametrize("setting, flag, enabled", [(None, False, False), (False, False, False),
                                                   (True, False, True), (False, True, True)])
def test_molmo_overhead_camera_is_opt_in_and_routes_only_the_new_room_view(tmp_path, setting, flag, enabled):
    config = BASE.replace("policy:\n", "policy:\n  name: molmoact2\n", 1)
    if setting is not None:
        config += f"  molmo_overhead_camera: {str(setting).lower()}\n"
    result = run(tmp_path, config, *(["--molmo-overhead-camera"] if flag else []))
    assert result.returncode == 0, result.stderr
    client = shlex.split(find(result, "docker exec -e POLICIES_DIR"))
    assert ("--molmo-overhead-camera" in client) is enabled
    rename = json.loads(client[client.index("--rename_map") + 1])
    assert rename == ({"realsense_rgb": "original_room", "molmo_room": "room", "wrist_cam": "wrist"}
                      if enabled else {"realsense_rgb": "room", "wrist_cam": "wrist"})


@pytest.mark.parametrize("flag", [False, True])
def test_overhead_camera_rejects_other_policies_before_launch(tmp_path, flag):
    result = run(tmp_path, BASE + ("" if flag else "  molmo_overhead_camera: true\n"),
                 *(["--molmo-overhead-camera"] if flag else []))
    assert result.returncode != 0
    assert "requires molmoact2" in result.stderr
    assert commands(result) == []


@pytest.mark.parametrize("value", ['"true"', '1', '0', 'null'])
def test_overhead_camera_rejects_non_boolean_yaml_values(tmp_path, value):
    config = BASE.replace("policy:\n", "policy:\n  name: molmoact2\n", 1)
    result = run(tmp_path, config + f"  molmo_overhead_camera: {value}\n")
    assert result.returncode != 0
    assert "molmo_overhead_camera must be true or false" in result.stderr


def test_disabling_overhead_camera_restores_the_original_room_policy_input(tmp_path):
    config = BASE.replace("policy:\n", "policy:\n  name: molmoact2\n", 1) + """  molmo_overhead_camera: false
  rename_map:
    realsense_rgb: original_room
    molmo_room: room
    wrist_cam: wrist
"""
    result = run(tmp_path, config)
    assert result.returncode == 0, result.stderr
    client = shlex.split(find(result, "docker exec -e POLICIES_DIR"))
    assert "--molmo-overhead-camera" not in client
    assert json.loads(client[client.index("--rename_map") + 1]) == {"realsense_rgb": "room", "wrist_cam": "wrist"}


def test_action_repeat_reaches_the_evaluator(tmp_path):
    result = run(tmp_path, BASE + "  action_repeat: 5\n")
    assert result.returncode == 0, result.stderr
    assert "--action_repeat 5" in find(result, "docker exec -e POLICIES_DIR")


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5"])
def test_action_repeat_requires_positive_integer_before_launch(tmp_path, value):
    result = run(tmp_path, BASE + f"  action_repeat: {value}\n")
    assert result.returncode != 0
    assert "eval.action_repeat" in result.stderr
    assert not commands(result)


def test_eval_set_grasp_dr_and_rerun_flags(tmp_path):
    result = run(tmp_path, f"""
task: pick_place_dr
policy:
  checkpoint: example/checkpoint-200000
eval:
  dataset: {DATASET}
  eval_set: {DATASET}/eval_set.json
  splits: [id, ood]
  repeats: 2
  episode_length_s: 12.5
  render_warmup: 3
  lang_by_color:
    blue: Pick up the blue cube
    red: Pick up the red cube
grasp:
  gripper_latch: true
  latch_close_below: 20
  seed_per_episode: true
run:
  gui: true
""", "--rerun")
    assert result.returncode == 0, result.stderr
    client = find(result, "docker exec -e POLICIES_DIR")
    for expected in ("Pick-Place-DR-Eval", f"--eval_set {DATASET}/eval_set.json --repeats 2", "--splits id\\,ood",
                     "--episode_length_s 12.5", "--render_warmup 3", "--gripper_latch",
                     "--latch_close_below 20", "--seed_per_episode", "--lang_instruction_by_color", "--rerun"):
        assert expected in client
    assert "--headless" not in client


def test_external_server_skips_docker_run(tmp_path):
    result = run(tmp_path, BASE, "--external_server")
    assert result.returncode == 0, result.stderr
    assert not any(line.startswith("docker run") for line in commands(result))
    assert "docker exec -e POLICIES_DIR" in result.stdout


def test_rerun_starts_matched_native_viewer_before_loading_policy(tmp_path):
    result = run(tmp_path, BASE, "--rerun")
    assert result.returncode == 0, result.stderr
    printed = commands(result)
    viewer = next(line for line in printed if "rerun_viewer.py" in line)
    assert "--container teleop --port 9876" in viewer
    assert printed.index(viewer) < printed.index(find(result, "docker run"))
    assert "RERUN_CONNECT_URL=rerun+http://127.0.0.1:9876/proxy" in find(result, "docker exec -e POLICIES_DIR")


def test_existing_rerun_endpoint_does_not_spawn_owned_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("RERUN_CONNECT_URL", "rerun+http://127.0.0.1:9999/proxy")
    result = run(tmp_path, BASE, "--rerun")
    assert result.returncode == 0, result.stderr
    assert "rerun_viewer.py" not in result.stdout
    assert "RERUN_CONNECT_URL=rerun+http://127.0.0.1:9999/proxy" in find(result, "docker exec -e POLICIES_DIR")


def test_dependency_is_checked_without_installing_each_run(tmp_path):
    result = run(tmp_path, BASE)
    assert result.returncode == 0, result.stderr
    assert "import\\ tqdm" in result.stdout
    assert "pip install" not in result.stdout


@pytest.mark.parametrize("eval_exit", [0, 7])
def test_real_launcher_keeps_full_log_and_result_without_verbose_dump(tmp_path, eval_exit):
    # Exercise the actual logging path with a fake Docker CLI; no simulator/server is launched.
    docker = tmp_path / "docker"
    docker.write_text(f'''#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_DOCKER_CALLS"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "inspect":
    print("true")
elif "sim_to_real_so101.scripts.lerobot_eval" in sys.argv:
    print("[INFO] Event Manager: full init detail")
    print("WARNING: new warning remains visible")
    print("Success rate: 0.00% (0/1)")
    print("Results JSON: /outputs/result.json")
    if {eval_exit}:
        print("ERROR: real inference failure")
        sys.exit({eval_exit})
elif sys.argv[1:4] == ["exec", "teleop", "cat"]:
    print('{{"complete": true, "raw_report_detail": "saved"}}')
''')
    docker.chmod(0o755)
    config = tmp_path / "case.yaml"
    config.write_text(textwrap.dedent(BASE))
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}", "EVAL_LOG_DIR": str(tmp_path / "logs"),
                        "FAKE_DOCKER_CALLS": str(tmp_path / "calls.jsonl")}
    result = subprocess.run(["bash", str(SCRIPT), str(config), "--external_server"],
                            capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.returncode == eval_exit, result.stderr
    assert "full init detail" not in result.stdout
    assert "new warning remains visible" not in result.stdout
    assert "Success rate" in result.stdout
    assert "raw_report_detail" not in result.stdout
    assert "full init detail" in (tmp_path / "logs/evaluation.log").read_text()
    assert "new warning remains visible" in (tmp_path / "logs/evaluation.log").read_text()
    if eval_exit:
        assert "real inference failure" in result.stderr
        assert str(tmp_path / "logs/evaluation.log") in result.stderr
        assert not (tmp_path / "logs/results.json").exists()
    else:
        assert "raw_report_detail" in (tmp_path / "logs/results.json").read_text()
    assert str(tmp_path / "logs") in result.stdout
    assert "pip" not in (tmp_path / "calls.jsonl").read_text()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_repeated_interrupt_cannot_abort_owned_container_cleanup(tmp_path, signum):
    docker = tmp_path / "docker"
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    cleaning = tmp_path / "cleaning"
    docker.write_text(f'''#!{sys.executable}
import os, pathlib, sys, time
if sys.argv[1] == "inspect":
    print("true")
elif "sim_to_real_so101.scripts.lerobot_eval" in sys.argv:
    pathlib.Path({str(ready)!r}).write_text("started")
    time.sleep(60)
elif sys.argv[1:5] == ["exec", "teleop", "/workspace/isaaclab/_isaac_sim/python.sh", "-c"]:
    assert "SO101_EVAL_RUN_ID=" in sys.argv[5]
    pathlib.Path({str(cleaning)!r}).write_text("cleanup started")
    time.sleep(.2)
    pathlib.Path({str(cleaned)!r}).write_text(sys.argv[6])
''')
    docker.chmod(0o755)
    config = tmp_path / "case.yaml"
    config.write_text(textwrap.dedent(BASE))
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}", "EVAL_LOG_DIR": str(tmp_path / "logs")}
    process = subprocess.Popen(["bash", str(SCRIPT), str(config), "--external_server"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
                               env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        process.send_signal(signum)  # Signal Bash alone, as a supervising launcher does.
        deadline = time.monotonic() + 5
        while not cleaning.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert cleaning.exists()
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 128 + signum, stderr
        assert cleaned.read_text()
        assert "Logs:" in stdout
        assert "Evaluation interrupted" in stdout
        assert "Diagnostic log:" not in stderr
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def test_server_only_never_runs_the_evaluator(tmp_path):
    result = run(tmp_path, BASE, "--server_only")
    assert result.returncode == 0, result.stderr
    assert "SERVER_READY port=5555" in result.stdout
    assert "lerobot_eval" not in result.stdout


def test_bare_config_name_is_looked_up_in_configs_eval(tmp_path):
    name = "zz_temp_case.yaml"
    path = ROOT / "configs/eval" / name
    path.write_text(textwrap.dedent(BASE))
    try:
        result = subprocess.run(["bash", str(SCRIPT), name, "--dry_run"], capture_output=True, text=True, cwd=ROOT)
    finally:
        path.unlink()
    assert result.returncode == 0, result.stderr


def test_shipped_pick_place_config_is_valid():
    result = subprocess.run(["bash", str(SCRIPT), "pick_place.yaml", "--dry_run"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("config_name, checkpoint", [
    ("pick_place_molmo_base.yaml", "molmoact2_base"),
    ("pick_place_molmo_so101.yaml", "molmoact2_so101"),
])
def test_molmo_config_selects_original_wrist_asset(config_name, checkpoint):
    result = subprocess.run(["bash", str(SCRIPT), config_name, "--dry_run"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    asset = "/workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/ARM_new.usd"
    client = find(result, "docker exec -e POLICIES_DIR")
    assert f"--checkpoint /workspace/models/{checkpoint}" in find(result, "docker run")
    assert f"--robot_usd {asset}" in client
    assert f"docker exec teleop test -f {asset}" in [line.rstrip() for line in commands(result)]
    if checkpoint == "molmoact2_so101":
        assert "--option action_dtype=float32" in find(result, "docker run")
        assert "--option include_original_room=False" in find(result, "docker run")
        assert "--policy_option include_original_room=False" in client
        assert "--action_horizon 30" in client
        assert "--action_repeat 1" in client
        assert "--video_encoder libx264" in client
        assert "--render_warmup" not in client
        assert "--molmo-overhead-camera" not in client
        client_args = shlex.split(client)
        assert json.loads(client_args[client_args.index("--rename_map") + 1]) == {
            "realsense_rgb": "room", "wrist_cam": "wrist",
        }
    else:
        assert "--molmo-overhead-camera" not in client


def test_original_wrist_asset_composes_with_consistent_joint_anchors():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    assets = ROOT / "source/sim_to_real_so101/assets/usd"
    source = Usd.Stage.Open(str(assets / "SO-ARM101-USD.usd"))
    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim("/Robot")
    robot.GetReferences().AddReference(str(assets / "ARM_new.usd"))
    assert stage.GetPrimAtPath("/Robot/gripper/visuals").IsValid()
    assert stage.GetPrimAtPath("/Robot/jaw/collisions").IsValid()
    previous = source.GetPrimAtPath("/so101_new_calib/gripper").GetAttribute("xformOp:orient").Get()
    restored = stage.GetPrimAtPath("/Robot/gripper").GetAttribute("xformOp:orient").Get()
    angle = (Gf.Rotation(previous) * Gf.Rotation(restored).GetInverse()).GetAngle()
    assert abs((angle + 180) % 360 - 180) == pytest.approx(90, abs=.001)
    cache = UsdGeom.XformCache()
    joints = [UsdPhysics.Joint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert len(joints) == 6
    for joint in joints:
        transforms = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(rel.GetTargets()[0]))
                      for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
        positions = [transform.Transform(Gf.Vec3d(attr.Get())) for transform, attr in zip(
            transforms, (joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()))]
        rotations = [Gf.Rotation(attr.Get()) * transform.ExtractRotation() for transform, attr in zip(
            transforms, (joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr()))]
        assert (positions[0] - positions[1]).GetLength() < 1e-5, joint.GetPath()
        angle = (rotations[0] * rotations[1].GetInverse()).GetAngle()
        assert abs((angle + 180) % 360 - 180) < .01, joint.GetPath()


@pytest.mark.parametrize("yaml_text, message", [
    ("eval:\n  dataset: /d\n", "policy.checkpoint is required"),
    ("policy:\n  checkpoint: /abs/ckpt\neval:\n  dataset: /d\n", "relative path"),
    ("policy:\n  checkpoint: a/../b\neval:\n  dataset: /d\n", "relative path"),
    ("policy:\n  checkpoint: a/b\n", "eval.dataset is required"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: relative\n", "absolute path"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  episodes: 0\n", "eval.episodes"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  action_horizon: 17\n", "1..16"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  repeats: 2\n", "require eval.eval_set"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  eval_set: /e.json\n  splits: [nope]\n", "Invalid eval.splits"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  random_fraction: 2\n", "random_fraction"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  episode_length_s: -1\n", "finite and positive"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  robot_usd: relative.usd\n", "eval.robot_usd"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  robot_usd: true\n", "eval.robot_usd"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  start_mode: sideways\n", "start_mode"),
    ("policy:\n  checkpoint: a/b\n  port: 70000\neval:\n  dataset: /d\n", "policy.port"),
    ("policy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  lang_by_color:\n    red: x\n", "lang_by_color"),
    ("task: pick_place_dr\npolicy:\n  checkpoint: a/b\neval:\n  dataset: /d\n  lang_by_color:\n    blue: x\n", "lang_by_color"),
    ("policy:\n  checkpoint: a/b\n  bogus: 1\neval:\n  dataset: /d\n", "Unknown key"),
    ("surprise: 1\n", "Unknown top-level key"),
    ("task: nope\n", "task must be"),
    ("policy:\n  name: no_such_policy\n  checkpoint: a/b\neval:\n  dataset: /d\n", "no_such_policy"),
])
def test_invalid_configs_are_rejected(tmp_path, yaml_text, message):
    result = run(tmp_path, yaml_text)
    assert result.returncode != 0
    assert message in result.stderr


def test_missing_config_and_unknown_flag():
    missing = subprocess.run(["bash", str(SCRIPT), "does_not_exist.yaml", "--dry_run"],
                             capture_output=True, text=True, cwd=ROOT)
    assert missing.returncode != 0 and "not found" in missing.stderr
    flag = subprocess.run(["bash", str(SCRIPT), "x.yaml", "--bogus"], capture_output=True, text=True, cwd=ROOT)
    assert flag.returncode != 0 and "Unknown option" in flag.stderr
    both = subprocess.run(["bash", str(SCRIPT), "x.yaml", "--external_server", "--server_only"],
                          capture_output=True, text=True, cwd=ROOT)
    assert both.returncode != 0 and "only one server mode" in both.stderr
