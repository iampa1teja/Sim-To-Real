"""docker/eval.sh and docker/eval_config.py: YAML validation and the commands --dry_run prints."""
import os
import signal
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
                     "--num_episodes 20", "--action_horizon 16", "--headless",
                     f"--cube_starts {DATASET}/pick_place_meta", "--checkpoint example/checkpoint-200000"):
        assert expected in client
    assert "--eval_set" not in client and "--gripper_latch" not in client
    assert "--robot_usd" not in client
    assert "-e SO101_EVAL_RUN_ID=" in client
    assert "-e PYTHONUNBUFFERED=1" in client








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
    assert "new warning remains visible" in result.stdout and "Success rate" in result.stdout
    assert "raw_report_detail" not in result.stdout
    assert "full init detail" in (tmp_path / "logs/evaluation.log").read_text()
    if eval_exit:
        assert "real inference failure" in result.stderr
        assert str(tmp_path / "logs/evaluation.log") in result.stderr
        assert not (tmp_path / "logs/results.json").exists()
    else:
        assert "raw_report_detail" in (tmp_path / "logs/results.json").read_text()
    assert str(tmp_path / "logs") in result.stdout
    assert "pip" not in (tmp_path / "calls.jsonl").read_text()


def test_term_interrupts_host_wait_and_runs_owned_container_cleanup(tmp_path):
    docker = tmp_path / "docker"
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    docker.write_text(f'''#!{sys.executable}
import os, pathlib, sys, time
if sys.argv[1] == "inspect":
    print("true")
elif "sim_to_real_so101.scripts.lerobot_eval" in sys.argv:
    pathlib.Path({str(ready)!r}).write_text("started")
    time.sleep(60)
elif sys.argv[1:5] == ["exec", "teleop", "/workspace/isaaclab/_isaac_sim/python.sh", "-c"]:
    assert "SO101_EVAL_RUN_ID=" in sys.argv[5]
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
        process.terminate()  # Signal Bash alone, as a supervising launcher does.
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        assert cleaned.read_text()
        assert "Logs:" in stdout
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
