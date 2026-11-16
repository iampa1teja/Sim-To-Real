"""Bowl task composition retains three hollow assets and sufficient contact storage."""
import pytest
pytest.importorskip('isaaclab')
from utils.arms.arms import load_arm_spec
from core.tasks.task import load_task, resolve
from sim.workspace.envs import scene_builder


@pytest.mark.parametrize('variant',['teleop','eval'])
def test_three_bowls_are_declared_before_physics_import(variant):
    arm = load_arm_spec('so101')[0]
    cfg = scene_builder.build_scene_cfg('rpd_meeting_room', arm)
    task = load_task('stack_bowls')
    resolve(task.spec.entry_point(variant))(cfg,task)
    for name,spec in task.spec.objects.items():
        asset = getattr(cfg.scene,name)
        assert asset.spawn.usd_path == str(spec.usd)
        assert asset.spawn.visual_material.diffuse_color == tuple(spec.params['color'])
    assert cfg.scene.contact_grasp.filter_prim_paths_expr == [
        '{ENV_REGEX_NS}/'+name for name in task.spec.tracked_objects]
    assert cfg.sim.physics.solver_cfg.nconmax >= 32768
    assert cfg.sim.physics.solver_cfg.njmax >= 131072
    assert 1/(cfg.decimation*cfg.sim.dt) == pytest.approx(30.)
    from sim.workspace.envs.task_env import reset_task_tracking, task_success
    assert cfg.events.reset_tracking.func is reset_task_tracking
    assert cfg.events.reset_tracking.params == {'task': task.name}
    if variant=='eval':
        assert cfg.terminations.success.func is task_success
        assert cfg.terminations.success.params == {'task': task.name}


@pytest.mark.parametrize('flagged', [False, True])
def test_usd_grasp_material_is_selected_only_by_the_object_flag(flagged):
    from dataclasses import replace
    from types import SimpleNamespace
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from sim.workspace.envs.task_env import apply_variant, spawn_grasp_usd

    task = load_task('stack_bowls')
    name = task.spec.tracked_objects[0]
    original = task.spec.objects[name]
    obj = replace(original, params={**original.params, 'grasp_material': flagged})
    spec = replace(task.spec, objects={**task.spec.objects, name: obj})
    cfg = scene_builder.build_scene_cfg('blank_table', load_arm_spec('so101')[0])
    apply_variant(cfg, SimpleNamespace(spec=spec, name=task.name), 'teleop')
    spawn = getattr(cfg.scene, name).spawn.func
    spawn = resolve(spawn) if isinstance(spawn, str) else spawn
    assert spawn is (spawn_grasp_usd if flagged else spawn_from_usd)
    # The flag selects material binding; imported geometry and declared solver capacity stay intact.
    assert getattr(cfg.scene, name).spawn.usd_path == str(original.usd)
    assert cfg.sim.physics.solver_cfg.njmax >= 131072
