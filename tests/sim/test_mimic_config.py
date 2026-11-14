"""Actual registered task configs satisfy the installed SDK Mimic boundary on CPU."""
import inspect
from types import SimpleNamespace

import pytest
import torch

from utils.arms.arms import load_arm_spec
from sim.datagen.mimic import MimicSpec, make_mimic_cfg


@pytest.mark.parametrize('scene_name', ['rpd_meeting_room', 'blank_table'])
def test_registered_scene_task_becomes_valid_sdk_mimic_cfg_without_launch(scene_name, monkeypatch):
    pytest.importorskip("isaaclab")
    from isaaclab.envs import MimicEnvCfg
    from isaaclab.managers import RecorderManager
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from sim.workspace.envs import register_envs

    monkeypatch.setenv('S2R_ARM', 'so101')
    task = register_envs(scene=scene_name, task='pick_place_cube')
    arm = load_arm_spec('so101')[0]
    cfg = parse_env_cfg(task.gym_id('teleop'), device='cpu', num_envs=1)
    before = cfg.to_dict()
    spec = MimicSpec.from_task(task)
    mixed = make_mimic_cfg(cfg, task, arm, spec, seed=101)
    assert isinstance(mixed, MimicEnvCfg)
    assert mixed.scene_spec.name == scene_name and mixed.arm_spec == arm
    assert mixed.scene.blue_cube.spawn.size == task.spec.objects['blue_cube'].size
    assert mixed.scene.white_box.spawn.usd_path == str(task.spec.objects['white_box'].usd)
    assert mixed.sim.device == 'cpu' and mixed.scene.num_envs == 1
    assert mixed.class_type == 'sim.datagen.mimic:S2RMimicEnv'
    assert mixed.recorders is None and mixed.terminations is None
    assert set(mixed.subtask_configs) == {arm.links.ee}
    assert len(mixed.subtask_configs[arm.links.ee]) == len(spec.subtasks)
    assert mixed.datagen_config.seed == 101 and not mixed.datagen_config.use_skillgen
    for candidate in (mixed, mixed.copy(), mixed.replace()):
        candidate.validate()
        exported = candidate.to_dict()
        for key in ('scene', 'actions', 'observations', 'events'):
            assert exported[key] == before[key]
    assert cfg.to_dict() == before
    # Exercise the SDK's real cfg assertion and final-subtask validation without an env constructor.
    pool = object()
    generator = DataGenerator(SimpleNamespace(cfg=mixed, device='cpu'), src_demo_datagen_info_pool=pool)
    assert generator.env_cfg is mixed and generator.src_demo_datagen_info_pool is pool
    # DataGenerator calls reset/set_success even with export_demo=False. Disabled SDK recorders support both.
    recorder = RecorderManager(None, SimpleNamespace(num_envs=1))
    assert recorder.reset(env_ids=torch.tensor([0])) == {}
    recorder.set_success_to_episodes(torch.tensor([0]), torch.tensor([[True]]))
    assert not torch.cuda._initialized


def test_mimic_adapter_matches_installed_constructor_and_generator_interface():
    pytest.importorskip("isaaclab")
    from isaaclab.envs import ManagerBasedRLMimicEnv
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from sim.datagen.mimic import S2RMimicEnv

    methods = ('get_robot_eef_pose', 'target_eef_pose_to_action', 'action_to_target_eef_pose',
               'actions_to_gripper_actions', 'get_object_poses', 'get_subtask_term_signals')
    for name in methods:
        sdk = inspect.signature(getattr(ManagerBasedRLMimicEnv, name)).parameters
        adapter = inspect.signature(getattr(S2RMimicEnv, name)).parameters
        assert list(adapter) == list(sdk), name
        assert [(value.kind, value.default) for value in adapter.values()] == [
            (value.kind, value.default) for value in sdk.values()], name
    assert S2RMimicEnv.__init__ is ManagerBasedRLMimicEnv.__init__
    assert not inspect.isabstract(S2RMimicEnv)
    assert inspect.iscoroutinefunction(DataGenerator.generate)
    inspect.signature(DataGenerator.__init__).bind(object(), object(), src_demo_datagen_info_pool=object())
    inspect.signature(DataGenerator.generate).bind(object(), 0, TerminationTermCfg(func=lambda env: None),
        env_reset_queue=object(), env_action_queue=object(), export_demo=False)
    assert not torch.cuda._initialized
