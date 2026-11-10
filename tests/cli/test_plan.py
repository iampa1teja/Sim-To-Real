"""The front door plans exact module commands without importing the simulator."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from cli.plan import (CliError, DATAGEN_MODULES, DOCKERFILES, SIM_PROGRAMS, SYNC_PATHS, Plan,
    plan_build, plan_check, plan_datagen, plan_host, plan_new, plan_pull, plan_serve,
    plan_sim, plan_sync, plan_train, plan_tunnel)


def manifest():
    return SimpleNamespace(name="demo", default_port=5555,
        train=SimpleNamespace(commands={"prepare": "train/prepare.sh", "train": "train/train.py", "verify": "train/verify.sh"}))


@pytest.mark.parametrize("command,module", SIM_PROGRAMS.items())
def test_every_sim_command_uses_its_final_python_module(command, module):
    assert plan_sim(module, ("--arm", "synthetic"), git_commit="abc").argv == (
        "docker", "exec", "-it", "-e", "S2R_GIT_COMMIT=abc", "-w", "/workspace/s2r", "s2r-sim",
        "s2r-entrypoint", "python", "-m", module, "--arm", "synthetic")
    assert command != "dr-replay"


@pytest.mark.parametrize("source,module", DATAGEN_MODULES.items())
def test_every_data_source_maps_to_a_module(source, module):
    assert plan_datagen(source, ("--seed", "9")).argv[-5:] == ("python", "-m", module, "--seed", "9")


def test_datagen_yaml_source_is_selected_inside_container(tmp_path, capsys):
    from cli.datagen import resolve_source
    from cli.main import main

    path = tmp_path / "generation.yaml"
    path.write_text("source: mimic\nseed: 7\n")
    argv = ["--config_path", str(path), "--seed", "9"]
    assert resolve_source(argv) == ("mimic", argv)
    assert resolve_source(["--source", "scripted", *argv]) == ("scripted", argv)
    assert plan_datagen(None, argv).argv[-7:] == ("python", "-m", "cli.datagen", *argv)
    assert main(["--dry-run", "datagen", *argv], repo=tmp_path) == 0
    assert "python -m cli.datagen --config_path" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["source: missing\n", "source: [scripted]\n", "seed: 7\n", "- scripted\n"])
def test_datagen_rejects_invalid_yaml_source(tmp_path, content):
    from cli.datagen import resolve_source

    path = tmp_path / "generation.yaml"
    path.write_text(content)
    with pytest.raises(SystemExit) as error:
        resolve_source(["--config_path", str(path)])
    assert error.value.code == 2


def test_host_new_check_migrate_plan_paths(tmp_path):
    assert plan_host("core.datasets.task_meta", ("dataset",), tmp_path).argv == (sys.executable, "-m", "core.datasets.task_meta", "dataset")
    assert plan_new("arm", "sample", tmp_path).argv == (sys.executable, "-m", "cli.scaffold", "arm", "sample", "--repo", str(tmp_path))
    plans = plan_check(tmp_path, arm="sample", scene="blank_table", task="stack_two_cubes", policy="example", unit="one")
    assert plans[0].argv == (sys.executable, "-m", "cli.check", "--repo", str(tmp_path), "--static-only", "--arm", "sample",
                             "--scene", "blank_table", "--task", "stack_two_cubes", "--policy", "example", "--unit", "one")
    assert plans[1].argv[-9:] == ("sim.check", "--arm", "sample", "--scene", "blank_table", "--task", "stack_two_cubes", "--variant", "teleop")
    assert len(plan_check(tmp_path, static_only=True)) == 1


def test_build_pull_tunnel_and_sync_plans(tmp_path):
    for role, (dockerfile, target) in DOCKERFILES.items():
        expected = ("docker", "build", *(("--target", target) if target else ()), "-t", f"s2r-{role}:latest", "-f", dockerfile, ".")
        assert plan_build([role], tmp_path)[0].argv == expected
    assert [p.argv for p in plan_pull(["serve"], registry="org", tag="v2")] == [
        ("docker", "pull", "org/s2r-serve:v2"), ("docker", "tag", "org/s2r-serve:v2", "s2r-serve:v2")]
    assert plan_tunnel("user@gpu", 5555).argv == ("ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", "5555:localhost:5555", "user@gpu")
    assert SYNC_PATHS == ("arms", "scenes", "tasks", "policies", "configs")
    assert plan_sync(tmp_path, "user@gpu", "/repo").argv == ("rsync", "-azR", "--info=progress2", "--exclude=__pycache__", "--exclude=.git",
        "./arms", "./scenes", "./tasks", "./policies", "./configs", "user@gpu:/repo/")


def test_serve_plan_and_train_environment_contract(tmp_path):
    serve = plan_serve(manifest(), tmp_path, checkpoint="model/checkpoint", models_dir="/models", hf_home="/cache", foreground=True)
    assert serve.argv == ("docker", "run", "--rm", "--name", "s2r-serve-demo-5555", "--network", "host", "--gpus", "all",
                          "-v", f"{tmp_path}/policies:/policies", "-v", "policy-cache:/opt/cache", "-v", "/cache:/root/.cache/huggingface",
                          "-v", "/models:/workspace/models", "s2r-serve:latest", "--policy", "demo", "--checkpoint", "/workspace/models/model/checkpoint", "--port", "5555")
    for command in ("prepare", "train", "verify"):
        plan = plan_train(command, manifest(), tmp_path, dataset="/dataset", output="/output", host_uid=1000, host_gid=1000, hf_home="/cache")
        assert "SO101_" not in " ".join(plan.argv)
        assert "S2R_POLICY=demo" in plan.argv
        assert "S2R_DATASET=/data/dataset" in plan.argv
        if command == "train":
            assert plan.argv[-3:] == ('exec "${S2R_PYTHON:-python3}" "$@"', "s2r", "/policies/demo/train/train.py")


def test_invalid_plans_fail_before_execution(tmp_path):
    with pytest.raises(CliError):
        plan_datagen("missing")
    with pytest.raises(CliError):
        plan_sync(tmp_path, "user@gpu", "relative")
    with pytest.raises(CliError):
        plan_serve(manifest(), tmp_path, checkpoint="../secret")
