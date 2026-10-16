"""docker/eval.sh and docker/eval_config.py: YAML validation and the commands --dry_run prints."""
import shutil
import subprocess
import sys
import textwrap
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


def test_molmo_config_selects_original_wrist_asset():
    result = subprocess.run(["bash", str(SCRIPT), "pick_place_molmo_base.yaml", "--dry_run"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    asset = "/workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/ARM_new.usd"
    client = find(result, "docker exec -e POLICIES_DIR")
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
