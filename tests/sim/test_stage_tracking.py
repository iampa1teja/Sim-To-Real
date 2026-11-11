"""Pre-reset stage capture handles each environment and preserves terminal returns."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip("isaaclab")
from core.arms.spec import load_arm_spec
from core.tasks.base import Task
from core.tasks.loader import load_task
from sim.eval.stage_tracking import install_stage_tracking


class SampleTask(Task):
    def signals(self, env):
        return {"arrived": env.arrived}

    def observe(self, state):
        return {"arrived": state.signals["arrived"], "done": state.signals["success"]}


def test_per_environment_terminal_state_is_observed_before_reset():
    spec = load_task("stack_two_cubes").spec
    task = SampleTask(replace(spec, stages=("arrived", "done"), objects={"cube_a": spec.objects["cube_a"]}))
    calls = []
    result = torch.tensor([True, False])
    cfg = SimpleNamespace(func=lambda env: calls.append(True) or result)
    robot = SimpleNamespace(find_bodies=lambda _: ([0], ["tool"]), find_joints=lambda *a, **k: ([0], ["Jaw"]),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(2, 1, 3), body_link_quat_w=torch.tensor([[[0., 0., 0., 1.]]] * 2),
                             joint_pos=torch.zeros(2, 1)))
    obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
                                               root_quat_w=torch.tensor([[0., 0., 0., 1.]] * 2)))
    env = SimpleNamespace(cfg=SimpleNamespace(arm_spec=load_arm_spec()[0]), scene={"robot": robot, "cube_a": obj},
        num_envs=2, common_step_counter=1, episode_length_buf=torch.tensor([10, 5]), arrived=torch.tensor([True, False]),
        termination_manager=SimpleNamespace(get_term_cfg=lambda _: cfg, set_term_cfg=lambda *_: None))
    install_stage_tracking(env, task)
    assert cfg.func(env) is result
    assert len(calls) == 1
    assert env._stage_traces[0].finish()["stages"] == {"arrived": True, "done": True}
    assert env._stage_traces[1].finish()["stages"] == {"arrived": False, "done": False}
    assert env._stage_traces[0].last_state.objects["cube_a"].position == (1., 0., 0.)
    env.common_step_counter += 1
    env.episode_length_buf = torch.tensor([1, 6])
    env.arrived = torch.tensor([False, True])
    result[:] = False
    cfg.func(env)
    assert not env._stage_traces[0].stages["arrived"]
    assert env._stage_traces[1].stages["arrived"]
