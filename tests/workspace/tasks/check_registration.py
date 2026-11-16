"""Live Phase-11 factory oracle: explicit registration equals direct task composition.

Run: s2r-entrypoint python tests/workspace/tasks/check_registration.py
"""
import os


def main():
    from core.tasks.task import load_task
    from sim import launch, close_app

    app = launch(headless=True, enable_cameras=True, device='cuda:0')
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        from sim.workspace.envs import register
        from sim.workspace.envs.scene_builder import build_scene_cfg
        from sim.workspace.envs.task_env import apply_variant
        from utils.arms.arms import load_arm_spec
        from check_stack_two_cubes import diff

        for arm_name in ('so101', 'sp7'):
            os.environ['S2R_ARM'] = arm_name
            arm = load_arm_spec(arm_name)[0]
            for scene in ('rpd_meeting_room', 'blank_table'):
                for task_name in ('pick_place_cube', 'stack_two_cubes', 'stack_bowls'):
                    task = register(scene, task_name)
                    assert task is load_task(task_name)
                    for variant in task.spec.all_variants():
                        expected = build_scene_cfg(scene, arm)
                        expected.task_name = task_name
                        apply_variant(expected, task, variant)
                        actual = parse_env_cfg(task.gym_id(variant), device='cuda:0', num_envs=1)
                        differences = list(diff(expected.to_dict(), actual.to_dict()))
                        assert not differences, (arm_name, scene, task_name, variant, differences)
                        print(f'MATCH {arm_name}/{scene}/{task_name}/{variant}', flush=True)
    finally:
        close_app(app)


if __name__ == '__main__':
    main()
