"""Datagen must require the same confirmed physical placement as evaluation."""
from types import SimpleNamespace

import pytest
import torch

from utils.arms.arms import load_arm_spec
from core.tasks.loader import load_task, load_task_module


@pytest.mark.parametrize("with_termination", [False, True])
def test_placement_confirmation_counts_physics_steps_and_requires_grasp_history(with_termination):
    pytest.importorskip("isaaclab")
    task = load_task("pick_place_cube")
    mdp = load_task_module("mdp", task.name)
    config = load_task_module("env_cfg", task.name)
    params = dict(config._place_params(task), confirm_steps=task.spec.raw["success"]["confirm_steps"])
    count = params["confirm_steps"]
    arm = load_arm_spec("so101")[0]
    quaternion = torch.tensor([[0., 0., 0., 1.]] * 2)
    cube = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(size=task.spec.objects["blue_cube"].size)),
        data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., params["container_floor_z"] + .01]] * 2),
                             root_quat_w=quaternion.clone()),
    )
    box = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=quaternion.clone()))
    robot = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_0/Robot"),
        find_bodies=lambda name: ([0], [name]),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(2, 1, 3), body_link_quat_w=quaternion[:, None]),
    )
    contact = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(2, 1, 1, 3)))
    env = SimpleNamespace(
        cfg=SimpleNamespace(arm_spec=arm, terminations=SimpleNamespace(success=SimpleNamespace(params=params))
                            if with_termination else None),
        scene={"robot": robot, "blue_cube": cube, "white_box": box, "contact_grasp": contact},
        num_envs=2, device="cpu", common_step_counter=0,
        episode_length_buf=torch.full((2,), params["warmup_steps"] + 1),
        _pick_place_grasp_offset=torch.zeros(3),
    )
    mdp._ensure_pick_place_state(env, "blue_cube")
    # Only env 0 has actually held the cube earlier in this synthetic history.
    env._pick_place_ever_grasped[0] = True

    def step(number):
        env.common_step_counter = number
        if with_termination:
            original = mdp.object_placed_in_container_termination(env, **params).clone()
        first = mdp.stage_signals(env)
        for _ in range(3):
            repeated = mdp.stage_signals(env)
            torch.testing.assert_close(repeated["success"], first["success"])
        if with_termination:
            torch.testing.assert_close(first["success"], original)
        assert not first["success"][1]  # Being inside the box alone is not success.
        return first

    for number in range(1, count):
        assert not step(number)["success"].any()
        assert env._pick_place_placed_steps.tolist() == [number, 0]
    # A transient placement must restart the confirmation window.
    cube.data.root_pos_w[0, 0] = 1.
    assert not step(count)["success"].any()
    assert env._pick_place_placed_steps.tolist() == [0, 0]
    cube.data.root_pos_w[0, 0] = 0.
    for number in range(1, count + 1):
        signals = step(count + number)
        assert bool(signals["success"][0]) == (number == count)
        assert env._pick_place_placed_steps.tolist() == [number, 0]
    cube.data.root_pos_w[0, 0] = 1.
    assert not step(2 * count + 1)["success"].any()
