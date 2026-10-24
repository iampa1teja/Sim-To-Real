"""Check container boundaries without launching a GPU job or a real Docker daemon."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('docker_training', ROOT / 'docker/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def fixture_env(tmp_path):
    paths = {name: tmp_path / name.lower() for name in ('BASE_MODEL', 'DATASET', 'OUT', 'HF_HOME', 'TRAIN_CACHE')}
    for name in ('BASE_MODEL', 'DATASET', 'HF_HOME'):
        paths[name].mkdir()
    (paths['HF_HOME'] / 'hub').mkdir()
    return {name: str(path) for name, path in paths.items()}


def options(command, flag):
    return [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == flag]


def test_training_preserves_host_paths_and_protects_inputs(tmp_path):
    env = fixture_env(tmp_path)
    env.update(BATCH='8', RESUME='1', PYTHONPATH='/host-only-packages', GROOT='/obsolete/checkout',
               HF_TOKEN='must-not-be-forwarded', PYTEST_CURRENT_TEST='test-only')
    command, cache = training.build_command('train', ['--preflight_only'], env)
    settings = dict(item.split('=', 1) for item in options(command, '--env'))
    assert settings['GROOT'] == '/Isaac-GR00T'
    assert settings['CUDA_HOME'] == '/usr/local/cuda'
    assert settings['PYTHONPATH'] == '/Isaac-GR00T'
    assert settings['BASE_MODEL'] == env['BASE_MODEL']
    assert settings['DATASET'] == env['DATASET']
    assert settings['OUT'] == env['OUT']
    assert settings['BATCH'] == '8' and settings['RESUME'] == '1'
    assert 'HF_TOKEN' not in settings and 'PYTEST_CURRENT_TEST' not in settings
    mounts = options(command, '--mount')
    for key in ('BASE_MODEL', 'DATASET'):
        assert f'type=bind,source={env[key]},target={env[key]},readonly' in mounts
    assert options(command, '--user') == [f'{os.getuid()}:{os.getgid()}']
    assert options(command, '--network') == ['none']
    assert options(command, '--pid') == ['host']
    assert '--privileged' not in command and '--rm' in command and '--init' in command
    assert command[-1] == '--preflight_only'
    assert not cache.exists() and not Path(env['OUT']).exists()


@pytest.mark.parametrize('workflow', ['prepare', 'verify'])
def test_data_tools_do_not_reserve_gpu_and_keep_preparation_output_writable(tmp_path, workflow):
    env = fixture_env(tmp_path)
    command, _ = training.build_command(workflow, [], env)
    assert '--gpus' not in command
    mounts = options(command, '--mount')
    repo_mount = next(m for m in mounts if m.startswith(f'type=bind,source={ROOT},'))
    assert repo_mount.endswith(',readonly') is (workflow == 'verify')
    assert not any(f'source={ROOT / "datasets"},' in m for m in mounts)
    assert not any(f'source={env["DATASET"]},' in m for m in mounts)
    if workflow == 'prepare':
        assert command[-4:] == ['--groot', '/Isaac-GR00T', '--model_profile', 'so_arm_n17']


def test_explicit_paths_with_spaces_are_individual_arguments(tmp_path):
    env = fixture_env(tmp_path)
    source = tmp_path / 'recorded demos'
    source.mkdir()
    output = tmp_path / 'prepared demos'
    command, _ = training.build_command('prepare', ['--source', str(source), '--output=' + str(output)], env)
    assert str(source) in command and '--output=' + str(output) in command
    assert f'type=bind,source={source},target={source},readonly' in options(command, '--mount')
    assert not output.exists()


def test_legacy_cache_override_is_resolved_before_container_workdir_changes(tmp_path, monkeypatch):
    env = fixture_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'legacy-cache').mkdir()
    env['TRANSFORMERS_CACHE'] = 'legacy-cache'
    command, _ = training.build_command('train', [], env)
    settings = dict(item.split('=', 1) for item in options(command, '--env'))
    assert settings['HF_HUB_CACHE'] == str(tmp_path / 'legacy-cache')
    assert 'TRANSFORMERS_CACHE' not in settings


def test_rejects_host_groot_override_instead_of_hiding_pinned_runtime(tmp_path):
    with pytest.raises(ValueError, match='pinned inside the image'):
        training.build_command('dit', ['--groot', '/old-checkout'], fixture_env(tmp_path))


@pytest.mark.parametrize('script,workflow', [
    ('policies/groot_n17/train/train_pick_place.sh', 'train'),
    ('policies/groot_n17/train/train_pick_place_sim_100k.sh', 'sim'),
    ('policies/groot_n17/train/train_pick_place_failures_ft.sh', 'failures'),
    ('policies/groot_n17/train/train_dit.sh', 'dit'),
])
def test_existing_entrypoints_route_to_docker_without_starting_training(tmp_path, script, workflow):
    env = {**os.environ, **fixture_env(tmp_path)}
    env.pop('SO101_TRAIN_IN_CONTAINER', None)
    env['DRY_RUN'] = '1'
    result = subprocess.run(['bash', str(ROOT / script)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.removeprefix('Docker command: ').strip())
    assert command[:3] == ['docker', 'run', '--rm']
    assert 'so101-groot-train:n1.7' in command
    assert str(ROOT / script) == command[-1]
    assert '/Isaac-GR00T' in options(command, '--env')[options(command, '--env').index('GROOT=/Isaac-GR00T')]
    assert not Path(env['OUT']).exists() and not Path(env['TRAIN_CACHE']).exists()


def test_container_failure_is_propagated_and_created_cache_is_user_owned(tmp_path):
    env = {**os.environ, **fixture_env(tmp_path)}
    env.pop('DRY_RUN', None)
    binary = tmp_path / 'bin'
    binary.mkdir()
    record = tmp_path / 'docker-run.json'
    docker = binary / 'docker'
    docker.write_text('#!/usr/bin/env python3\nimport json,sys\n'
                      f'from pathlib import Path\np=Path({str(record)!r})\n'
                      'if sys.argv[1:3] == ["image","inspect"]: sys.exit(0)\n'
                      'p.write_text(json.dumps(sys.argv[1:]))\nsys.exit(47)\n')
    docker.chmod(0o755)
    env['PATH'] = str(binary) + os.pathsep + env['PATH']
    result = subprocess.run(['bash', str(ROOT / 'docker/train.sh'), '--preflight_only'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 47, result.stderr
    assert json.loads(record.read_text())[-1] == '--preflight_only'
    assert Path(env['TRAIN_CACHE']).stat().st_uid == os.getuid()
    assert not Path(env['OUT']).exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_build_reuses_only_wheels_matching_pypi_sha256(tmp_path, monkeypatch, corrupt):
    import urllib.request
    payload = b'verified wheel payload'
    calls = []
    for package, tag in [('av', 'cp312-cp312'), ('bitsandbytes', 'py3-none')]:
        (tmp_path / f'{package}-{tag}-manylinux_x86_64.whl').write_bytes(b'corrupt' if corrupt else payload)

    def metadata_only(url, **kwargs):
        calls.append(url)
        assert url.startswith('https://pypi.org/pypi/') and url.endswith('/json')
        package = url.split('/')[4]
        tag = 'cp312-cp312' if package == 'av' else 'py3-none'
        return io.BytesIO(json.dumps({'urls': [{
            'filename': f'{package}-{tag}-manylinux_x86_64.whl',
            'url': 'https://files.pythonhosted.org/must-not-download',
            'digests': {'sha256': hashlib.sha256(payload).hexdigest()},
        }]}).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', metadata_only)
    monkeypatch.setattr('sys.argv', ['download_wheels.py', str(tmp_path)])
    if corrupt:
        with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
            runpy.run_path(str(ROOT / 'docker/train/download_wheels.py'))
    else:
        runpy.run_path(str(ROOT / 'docker/train/download_wheels.py'))
        assert len(calls) == 2
