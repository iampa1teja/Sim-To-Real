"""Datagen must require the same confirmed physical placement as evaluation."""
from types import SimpleNamespace

import pytest
import torch

from utils.arms.arms import load_arm_spec
from core.tasks.loader import load_task


@pytest.mark.parametrize("with_termination", [False, True])
def test_placement_confirmation_counts_physics_steps_and_requires_grasp_history(with_termination):
    pytest.importorskip("isaaclab")
    from sim.workspace.envs.task_env import reset_task_tracking, task_signals, task_success

    task = load_task("pick_place_cube")
    count = task.stage_checks["success"].steps
    warmup = task.module.grasp.warmup_steps
    floor_z = task.spec.objects["white_box"].params["floor_z"]
    arm = load_arm_spec("so101")[0]
    quaternion = torch.tensor([[0., 0., 0., 1.]] * 2)
    cube = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., floor_z + .01]] * 2),
                                                root_quat_w=quaternion.clone()))
    box = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=quaternion.clone()))
    robot = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_0/Robot"),
        find_bodies=lambda name: ([0], [name]),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(2, 1, 3), body_link_quat_w=quaternion[:, None]),
    )
    contact = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(2, 1, 1, 3)),
                              reset=lambda env_ids=None: None)
    env = SimpleNamespace(
        cfg=SimpleNamespace(arm_spec=arm),
        scene={"robot": robot, "blue_cube": cube, "white_box": box, "contact_grasp": contact},
        num_envs=2, device="cpu", common_step_counter=0,
        episode_length_buf=torch.full((2,), warmup + 1),
        _task_grasp_offset=torch.zeros(3),
    )
    reset_task_tracking(env, None, task)
    # Only env 0 has actually held the cube earlier in this synthetic history.
    grasp, confirm = task.module.grasp, task.stage_checks["success"]
    grasp._state["ever"][0] = True

    def step(number):
        env.common_step_counter = number
        if with_termination:
            original = task_success(env, task).clone()
        first = task_signals(env, task)
        for _ in range(3):
            repeated = task_signals(env, task)
            torch.testing.assert_close(repeated["success"], first["success"])
        if with_termination:
            torch.testing.assert_close(first["success"], original)
        assert not first["success"][1]  # Being inside the box alone is not success.
        return first

    for number in range(1, count):
        assert not step(number)["success"].any()
        assert confirm._state["count"].tolist() == [number, 0]
    # A transient placement must restart the confirmation window.
    cube.data.root_pos_w[0, 0] = 1.
    assert not step(count)["success"].any()
    assert confirm._state["count"].tolist() == [0, 0]
    cube.data.root_pos_w[0, 0] = 0.
    for number in range(1, count + 1):
        signals = step(count + number)
        assert bool(signals["success"][0]) == (number == count)
        assert confirm._state["count"].tolist() == [number, 0]
    cube.data.root_pos_w[0, 0] = 1.
    assert not step(2 * count + 1)["success"].any()
    assert grasp is task.stage_checks["grasped"]
