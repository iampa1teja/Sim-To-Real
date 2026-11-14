"""Bowl task composition retains three hollow assets and sufficient contact storage."""
import pytest
pytest.importorskip('isaaclab')
from utils.arms.arms import load_arm_spec
from core.tasks.loader import load_task, resolve
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
    assert cfg.events.reset_tracking.func.__module__.endswith('stack_bowls.mdp')
    if variant=='eval':
        assert cfg.terminations.success.func.__module__.endswith('stack_bowls.mdp')
