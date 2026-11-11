"""The front door plans exact module commands without importing the simulator."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from cli.plan import (CliError, DATAGEN_MODULES, DOCKERFILES, SIM_PROGRAMS, SYNC_PATHS,
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


def test_sim_planner_uses_only_explicit_selector_mapping(monkeypatch):
    monkeypatch.setenv('S2R_ARM', 'ambient_arm')
    assert plan_sim('core.calibration.calibrate', ('leader',)).argv == (
        'docker', 'exec', '-it', '-w', '/workspace/s2r', 's2r-sim',
        's2r-entrypoint', 'python', '-m', 'core.calibration.calibrate', 'leader')
    selected = {'S2R_UNIT': 'robot_1', 'S2R_TASK': '', 'S2R_SCENE': 'blank_table', 'S2R_ARM': 'guide_arm'}
    assert plan_sim('sim.teleop.agent', ('--arm', 'cli_arm'), git_commit='abc', selection_env=selected).argv == (
        'docker', 'exec', '-it', '-e', 'S2R_GIT_COMMIT=abc', '-e', 'S2R_ARM=guide_arm',
        '-e', 'S2R_SCENE=blank_table', '-e', 'S2R_TASK=', '-e', 'S2R_UNIT=robot_1',
        '-w', '/workspace/s2r', 's2r-sim', 's2r-entrypoint', 'python', '-m', 'sim.teleop.agent', '--arm', 'cli_arm')
    with pytest.raises(CliError, match='runtime environment key'):
        plan_sim('sim.teleop.agent', (), selection_env={'UNRELATED_SECRET': 'not forwarded'})


def test_runtime_file_overrides_are_explicit_container_environment(tmp_path, monkeypatch, capsys):
    from cli.main import main

    overrides = {'S2R_ROOM_USD': '/workspace/s2r/scenes/custom/room.usda',
                 'S2R_EVAL_STARTS': '/workspace/datasets/starts',
                 'S2R_FOLLOWER_CALIBRATION': '/calibration/follower.json',
                 'S2R_LEADER_CALIBRATION': '/calibration/leader.json',
                 'S2R_EVAL_CHECKPOINT': 'checkpoint-2'}
    plan = plan_sim('sim.eval.run', (), selection_env=overrides)
    assert plan.argv[:13] == ('docker', 'exec', '-it', '-e', 'S2R_ROOM_USD=' + overrides['S2R_ROOM_USD'],
                              '-e', 'S2R_EVAL_STARTS=' + overrides['S2R_EVAL_STARTS'],
                              '-e', 'S2R_FOLLOWER_CALIBRATION=' + overrides['S2R_FOLLOWER_CALIBRATION'],
                              '-e', 'S2R_LEADER_CALIBRATION=' + overrides['S2R_LEADER_CALIBRATION'],
                              '-e', 'S2R_EVAL_CHECKPOINT=' + overrides['S2R_EVAL_CHECKPOINT'])
    checks = plan_check(tmp_path, arm='guide_arm', scene='guide_scene', task='guide_task', selection_env=overrides)
    assert checks[0].env['S2R_ROOM_USD'] == ''  # Host validates manifest assets, not container-only overrides.
    assert 'S2R_ROOM_USD=' + overrides['S2R_ROOM_USD'] in checks[1].argv
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    assert main(['--dry-run', 'eval', '--config_path', 'configs/eval/pick_place.yaml'], repo=tmp_path) == 0
    output = capsys.readouterr().out
    for key, value in overrides.items():
        assert key + '=' + value in output
    assert 'ROOM_USD_PATH=' not in output


@pytest.mark.parametrize('command', [
    ['calibrate', 'leader'], ['record', '--config_path', 'configs/record/sim_only.yaml', '--arm', 'cli_arm'],
    ['datagen', '--source', 'scripted', '--arm', 'cli_arm'],
    ['datagen', '--config_path', 'configs/datagen/mimic.yaml'],
    ['check', '--arm', 'cli_arm', '--scene', 'cli_scene', '--task', 'cli_task'],
])
def test_cli_forwards_host_selectors_for_every_sim_dispatch_branch(command, tmp_path, monkeypatch, capsys):
    from cli.main import main

    for key, value in {'S2R_ARM': 'guide_arm', 'S2R_SCENE': 'blank_table',
                       'S2R_TASK': 'guide_task', 'S2R_UNIT': 'robot_1'}.items():
        monkeypatch.setenv(key, value)
    assert main(['--dry-run', *command], repo=tmp_path) == 0
    output = capsys.readouterr().out
    assert ('-e S2R_ARM=guide_arm -e S2R_SCENE=blank_table -e S2R_TASK=guide_task '
            '-e S2R_UNIT=robot_1') in output
    if command[0] == 'calibrate':
        assert 'python -m core.calibration.calibrate leader' in output
    if '--arm' in command:
        assert '--arm cli_arm' in output  # Explicit module flags remain intact and take precedence.


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


def test_datagen_dispatch_preserves_python_m_module_identity(tmp_path, monkeypatch, capsys):
    from cli import datagen

    (tmp_path / "lazy_source.py").write_text('''import sys
def __getattr__(name):
    if name == "RuntimeEnv":
        return "resolved lazily"
    raise AttributeError(name)
assert getattr(sys.modules[__name__], "RuntimeEnv") == "resolved lazily"
assert sys.modules[__name__].__dict__ is globals()
assert sys.argv[1:] == ["--seed", "7"]
print("source module identity passed")
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(datagen, "DATAGEN_MODULES", {"example": "lazy_source"})
    original_main, original_argv = sys.modules["__main__"], sys.argv
    datagen.main(["--source", "example", "--seed", "7"])
    assert sys.modules["__main__"] is original_main and sys.argv is original_argv
    assert "source module identity passed" in capsys.readouterr().out


def test_host_new_check_migrate_plan_paths(tmp_path):
    assert plan_host("core.datasets.task_meta", ("dataset",), tmp_path).argv == (sys.executable, "-m", "core.datasets.task_meta", "dataset")
    assert plan_new("arm", "sample", tmp_path).argv == (sys.executable, "-m", "cli.scaffold", "arm", "sample", "--repo", str(tmp_path))
    plans = plan_check(tmp_path, arm="sample", scene="blank_table", task="stack_two_cubes", policy="example", unit="one")
    assert plans[0].argv == (sys.executable, "-m", "cli.check", "--repo", str(tmp_path), "--static-only", "--arm", "sample",
                             "--scene", "blank_table", "--task", "stack_two_cubes", "--policy", "example", "--unit", "one")
    assert plans[1].argv[-9:] == ("sim.check", "--arm", "sample", "--scene", "blank_table", "--task", "stack_two_cubes", "--variant", "teleop")
    assert len(plan_check(tmp_path, static_only=True)) == 1


def test_scene_export_plan_targets_one_runtime_composition(tmp_path):
    output = Path('/workspace/s2r/outputs/guide-scene.json')
    plans = plan_check(tmp_path, arm='guide_arm', scene='guide_scene', task='guide_task', scene_json=output)
    assert '--scene-json' not in plans[0].argv
    assert plans[1].argv[-4:] == ('--variant', 'teleop', '--scene-json', str(output))
    with pytest.raises(CliError, match='omit --static'):
        plan_check(tmp_path, static_only=True, scene_json=output)
    with pytest.raises(CliError, match='one composition'):
        plan_check(tmp_path, compositions=[('a', 's', 't'), ('b', 's', 't')], scene_json=output)
    with pytest.raises(CliError, match='one composition'):
        plan_check(tmp_path, scene_json=output)


def test_compare_front_door_preserves_results_and_comparison_options(tmp_path, capsys):
    from cli.main import main

    assert SIM_PROGRAMS['compare'] == 'core.benchmark.compare'
    assert main(['--dry-run', 'compare', 'outputs/first.json', 'outputs/second.json',
                 '--task', 'guide_task', '--repeat', '0'], repo=tmp_path) == 0
    assert ('python -m core.benchmark.compare outputs/first.json outputs/second.json '
            '--task guide_task --repeat 0') in capsys.readouterr().out


def test_sweep_front_door_uses_existing_summary_entrypoint(tmp_path, capsys):
    from cli.main import main

    assert SIM_PROGRAMS['sweep'] == 'core.benchmark.sweep'
    assert main(['--dry-run', 'sweep', 'outputs/policy-sweep'], repo=tmp_path) == 0
    assert 'python -m core.benchmark.sweep outputs/policy-sweep' in capsys.readouterr().out


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


@pytest.mark.parametrize('gpu', [False, True])
def test_verification_gpu_access_requires_explicit_opt_in(gpu, tmp_path, monkeypatch, capsys):
    import cli.main as cli

    plan = plan_train('verify', manifest(), tmp_path, dataset='/dataset', verify_gpu=gpu)
    assert ('--gpus' in plan.argv) is gpu
    assert ('--pid' in plan.argv) is gpu
    if gpu:
        index = plan.argv.index('--gpus')
        assert plan.argv[index:index + 4] == ('--gpus', 'all', '--pid', 'host')
    monkeypatch.setattr(cli, 'load_policies', lambda repo: {'demo': manifest()})
    assert cli.main(['--dry-run', 'verify', '--policy', 'demo', '--dataset', '/dataset',
                     *(['--gpu'] if gpu else [])], repo=tmp_path) == 0
    assert ('--gpus all' in capsys.readouterr().out) is gpu
    with pytest.raises(CliError, match='only supported for verify'):
        plan_train('prepare', manifest(), tmp_path, dataset='/dataset', output='/output', verify_gpu=True)


def test_invalid_plans_fail_before_execution(tmp_path):
    with pytest.raises(CliError):
        plan_datagen("missing")
    with pytest.raises(CliError):
        plan_sync(tmp_path, "user@gpu", "relative")
    with pytest.raises(CliError):
        plan_serve(manifest(), tmp_path, checkpoint="../secret")
