"""Scaffolds, static checking and dry runs work without the simulator SDK."""
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

from cli.check import check_static, extension_roots
from cli.main import check_compositions, main
from cli.scaffold import create_extension

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def guide_extensions(tmp_path, monkeypatch):
    """Follow the folder-only guide; only the new policy needs an implementation."""
    from core.tasks import loader
    from core.tasks.base import TASK_PACKAGE

    monkeypatch.setattr(loader, '_loaded', {})
    for name in list(sys.modules):
        if name == TASK_PACKAGE or name.startswith(TASK_PACKAGE + '.'):
            monkeypatch.delitem(sys.modules, name)
    for kind, folder in (('arm', 'arms'), ('scene', 'scenes'), ('task', 'tasks'), ('policy', 'policies')):
        shutil.copytree(REPO / folder / '_template', tmp_path / folder / '_template')
        create_extension(kind, 'guide_' + kind, tmp_path)
        monkeypatch.setenv('S2R_' + kind.upper(), 'guide_' + kind)
    create_extension('policy', 'guide_fixed', tmp_path)
    for name in ('guide_policy', 'guide_fixed'):
        folder = tmp_path / 'policies' / name
        manifest = (folder / 'policy.toml').read_text().replace('cameras = ["room", "wrist"]', 'cameras = ["room"]')
        if name == 'guide_fixed':
            manifest = manifest.replace('[action]\n', '[action]\ndim = 2\n')
        (folder / 'policy.toml').write_text(manifest)
        (folder / 'server.py').write_text('''from pathlib import Path
import numpy as np
from policy_server.contract import ServerPolicy

class EchoPolicy(ServerPolicy):
    def load(self, checkpoint, device, options):
        self.horizon = int(options.get("horizon", 2))
    def reset(self, seed=None):
        return {"seed": seed}
    def infer(self, request):
        return {"actions": np.tile(request["state"], (self.horizon, 1))}
    def info(self):
        return {"name": Path(__file__).parent.name}

def create():
    return EchoPolicy()
''')
    return tmp_path


def test_each_scaffold_has_a_matching_manifest_and_passes_static_checks(tmp_path):
    for kind, folder, marker in (("arm", "arms", "arm.toml"), ("scene", "scenes", "scene.toml"),
                                  ("task", "tasks", "task.toml"), ("policy", "policies", "policy.toml")):
        shutil.copytree(REPO / folder / "_template", tmp_path / folder / "_template")
        created = create_extension(kind, "test_" + kind, tmp_path)
        assert created == tmp_path / folder / ("test_" + kind)
        assert f'name = "test_{kind}"' in (created / marker).read_text()
        with pytest.raises(FileExistsError):
            create_extension(kind, "test_" + kind, tmp_path)
    rows, errors = check_static(tmp_path, arm="test_arm", scene="test_scene", task="test_task", policy="test_policy")
    assert not errors, errors
    assert len(rows) == 4
    (tmp_path / "policies" / "test_policy" / "policy.toml").write_text('[policy]\nname = "wrong_folder"\n')
    _, errors = check_static(tmp_path)
    assert any("policies/test_policy" in error for error in errors)


def test_new_dry_run_does_not_create_a_folder(tmp_path, capsys):
    (tmp_path / "policies").mkdir()
    assert main(["--dry-run", "new", "arm", "untouched"], repo=tmp_path) == 0
    assert not (tmp_path / "arms").exists()
    assert "-m cli.scaffold arm untouched" in capsys.readouterr().out


def test_default_check_covers_all_extensions_without_a_cartesian_product(monkeypatch):
    monkeypatch.setenv('S2R_ARM', 'so101')
    monkeypatch.setenv('S2R_SCENE', 'rpd_meeting_room')
    monkeypatch.setenv('S2R_TASK', 'pick_place_cube')
    rows = check_compositions(REPO)
    baseline = ('so101', 'rpd_meeting_room', 'pick_place_cube')
    assert {baseline, ('so101', 'blank_table', 'pick_place_cube'),
            ('so101', 'rpd_meeting_room', 'stack_two_cubes')} <= set(rows)
    expected = {baseline}
    for index, (folder, marker) in enumerate((('arms', 'arm.toml'), ('scenes', 'scene.toml'), ('tasks', 'task.toml'))):
        discovered = {path.name for path in (REPO / folder).iterdir()
                      if not path.name.startswith('_') and (path / marker).is_file()}
        assert {row[index] for row in rows} == discovered
        for name in discovered:
            composition = list(baseline)
            composition[index] = name
            expected.add(tuple(composition))
    assert len(rows) == len(set(rows)) and set(rows) == expected
    assert check_compositions(REPO, scene="blank_table", task="stack_two_cubes") == [("so101", "blank_table", "stack_two_cubes")]


def test_scaffolded_composition_builds_and_exports_sdk_config_without_source_edits(guide_extensions):
    pytest.importorskip("isaaclab_tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    assert not (guide_extensions / 'src').exists()
    with extension_roots(guide_extensions):
        from sim.envs import register_envs

        rows, errors = check_static(guide_extensions, arm='guide_arm', scene='guide_scene',
                                    task='guide_task', policy='guide_policy')
        assert not errors, errors
        assert len(rows) == 5
        task = register_envs(scene='guide_scene', task='guide_task')
        cfg = parse_env_cfg(task.gym_id('teleop'), device='cpu', num_envs=1)
        assert cfg.arm_spec.joint_names == ('shoulder', 'finger')
        assert cfg.scene_spec.name == 'guide_scene' and cfg.task_name == 'guide_task'
        assert cfg.actions.joint_positions.joint_names == ['Shoulder', 'Finger']
        assert cfg.scene.robot.init_state.pos == (0., 0., .75)
        assert set(cfg.scene.robot.init_state.joint_pos) == {'Shoulder', 'Finger'}
        assert cfg.scene.camera_room.height == 480
        for candidate in (cfg, cfg.copy(), cfg.replace()):
            candidate.validate()
            exported = candidate.to_dict()
            assert exported['arm_spec']['name'] == 'guide_arm'
            assert exported['scene_spec']['name'] == 'guide_scene'


@pytest.mark.parametrize('name', ['guide_policy', 'guide_fixed'])
def test_scaffolded_dynamic_and_fixed_policy_execute_real_factory_and_wire(guide_extensions, name, monkeypatch):
    from core.arms.calibration import resolve_calibration
    from core.arms.spec import load_arm_spec
    from core.policy_client.contract import ArmInfo, load_client_policy
    from core.policy_client.discovery import load_manifest
    from core.policy_client.transport import MsgSerializer

    monkeypatch.syspath_prepend(str(REPO / 'src/core'))
    from policy_server.contract import load_server_policy
    from policy_client.transport import MsgSerializer as ServerSerializer

    with extension_roots(guide_extensions):
        arm = load_arm_spec('guide_arm')[0]
        info = ArmInfo.from_arm(arm, resolve_calibration(arm, environ={}).mapping)
        manifest = load_manifest(guide_extensions / 'policies' / name)
        client = load_client_policy(name, manifest, {})
        client.configure(info)
        server = load_server_policy(manifest.path)
        server.load('', 'cpu', {'horizon': '3'})
        assert server.reset(seed=7) == {'seed': 7}
        assert server.info()['name'] == name
        state = np.array([-12., 37.], dtype=np.float32)
        images = {'room': np.arange(18, dtype=np.uint8).reshape(2, 3, 3)}
        request = ServerSerializer.from_bytes(MsgSerializer.to_bytes(client.build_request(state, images, 'stack')))
        response = MsgSerializer.from_bytes(ServerSerializer.to_bytes(server.infer(request)))
        actions = client.parse_response(response, list(info.feature_names))
        np.testing.assert_array_equal(actions, np.tile(state, (3, 1)))
        client.reset()
        _, errors = check_static(guide_extensions, arm='guide_arm', policy=name)
        assert not errors, errors
        if name == 'guide_fixed':
            path = manifest.path / 'policy.toml'
            path.write_text(path.read_text().replace('dim = 2', 'dim = 3'))
            _, errors = check_static(guide_extensions, arm='guide_arm', policy=name)
            assert any('3 action axes' in error and '2' in error for error in errors)
