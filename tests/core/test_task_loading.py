"""Tasks and gym registration require explicit selection and stay safe before Kit."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.tasks.task import load_task


ROOT = Path(__file__).resolve().parents[2]


def test_task_loader_requires_a_name_and_caches_explicit_selection(monkeypatch):
    monkeypatch.setenv('S2R_TASK', 'stack_bowls')
    first = load_task('pick_place_cube')
    monkeypatch.setenv('S2R_TASK', 'stack_two_cubes')
    assert load_task('pick_place_cube') is first
    assert first.name == 'pick_place_cube'
    with pytest.raises(TypeError):
        load_task()
    with pytest.raises(ValueError, match='folder name'):
        load_task(None)


def test_legacy_package_is_not_a_loader_fallback(tmp_path, monkeypatch):
    folder = tmp_path / 'legacy_only'
    folder.mkdir()
    (folder / 'task.toml').write_text('[task]\nname = "legacy_only"\n')
    (folder / '__init__.py').write_text('raise AssertionError("Legacy package executed")\n')
    monkeypatch.setenv('S2R_TASKS_DIR', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='needs task.py'):
        load_task('legacy_only')


def test_prelaunch_imports_and_task_loading_need_no_torch_sdk_or_gym():
    script = '''
import os, sys
os.environ['S2R_TASK'] = 'a_task_that_does_not_exist'
import sim.workspace.envs as envs
assert not hasattr(envs, 'TASK')
assert 'gymnasium' not in sys.modules
from core.tasks.task import load_task
for name in ('pick_place_cube', 'stack_two_cubes', 'stack_bowls'):
    assert load_task(name).name == name
assert not any(name in sys.modules for name in ('torch', 'isaaclab', 'pxr', 'omni'))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_registration_uses_the_selected_task_without_global_state(monkeypatch):
    from sim.workspace import envs

    registry = {}
    def register(**kwargs):
        registry[kwargs['id']] = SimpleNamespace(entry_point=kwargs['entry_point'], kwargs=kwargs['kwargs'])
    monkeypatch.setitem(sys.modules, 'gymnasium', SimpleNamespace(registry=registry, register=register))
    first = envs.register('blank_table', 'pick_place_cube')
    second = envs.register('blank_table', 'stack_two_cubes')
    assert (first.name, second.name) == ('pick_place_cube', 'stack_two_cubes')
    assert not hasattr(envs, 'TASK')
    assert set(registry) == {'S2R-base', 'S2R-blank_table', 'S2R-pick_place_cube-teleop',
                             'S2R-pick_place_cube-eval', 'S2R-pick_place_cube-eval_dr',
                             'S2R-stack_two_cubes-teleop', 'S2R-stack_two_cubes-eval'}
