"""Exercise terminal snapshots and interrupted-report behavior across the split loop."""
from types import SimpleNamespace as NS
import json

import pytest
import torch

from core.policy_client.grasp import GraspConfig
from sim.eval import loop


class Trace:
    def __init__(self, success=False):
        self.stages = {'ready': True, 'done': success}

    def progress_label(self):
        return 'done' if self.stages['done'] else 'ready'

    def finish(self, success):
        return {'stages': self.stages.copy(), 'failure_mode': None if success else 'timeout'}


class Env:
    def __init__(self):
        self.unwrapped = self
        self.device = 'cpu'
        self.action_space = NS(shape=(1, 2))
        self.max_episode_length = 12
        self._task_start_index = [0]
        self._task_start_zone = ['test-zone']
        self._benchmark_trace = Trace()
        self.ticks = 0
        self.actions = []

    def observation(self):
        return {'policy': {'joint_pos_obs': torch.tensor([[float(self._task_start_index[0]), .5]])}, 'visual': {}}

    def reset(self):
        return self.observation(), {}

    def step(self, actions):
        self.actions.append(actions.clone())
        self.ticks += 1
        terminal = self.ticks % 12 == 0
        if terminal:
            self._benchmark_trace = Trace(success=self._task_start_index[0] == 0)
            # Match the SDK: terminal flags/trace refer to the old episode, observation/start metadata to the new one.
            self._task_start_index[0] += 1
        return self.observation(), None, torch.tensor([terminal]), torch.tensor([False]), {}


def context(tmp_path):
    env = Env()
    calls = []
    args = NS(video_dir=None, results_json=tmp_path/'report.json', num_episodes=2, save_videos=False,
              rerun=False, robot_start='recorded', action_repeat=1, checkpoint='fixture', seed=11,
              random_fraction=0., action_horizon=2, arm='test_arm', scene='test_scene', task='test_task',
              variant='eval', lang_instruction='test instruction', lang_instruction_by_color=None,
              policy='example', cube_starts=None, seed_per_episode=False)
    policy = NS(reset=lambda **kw: calls.append(kw), get_action=lambda joints, visual, log=False: joints,
                inference_calls=0, lang_instruction='')
    task = NS(stages=('ready', 'done'), episode_metadata=lambda raw: {'token': raw._task_start_index[0]})
    values = dict.fromkeys(loop.EvaluationContext.__dataclass_fields__)
    values.update(args_cli=args, simulation_app=NS(is_running=lambda: True), env=env,
                  task=task, env_id='S2R-test-eval', env_cfg=NS(scene=NS(robot=NS(spawn=NS(usd_path='test.usd')))),
                  resume=False, smoke_test=False, smoke_selection='first_five', render_warmup=0,
                  episode_length_s=1.2, step_dt=.1, replan_interval_s=.2,
                  keyboard_control=NS(reset_world=False), policy=policy, grasp_config=GraspConfig(),
                  grasp_kwargs={}, smoothing_defaults={}, smoothing_kwargs={}, policy_options={})
    return loop.EvaluationContext(**values), calls


def test_terminal_metadata_stays_with_pre_reset_episode(monkeypatch, tmp_path):
    ctx, calls = context(tmp_path)
    monkeypatch.setattr(loop, 'reset_stage_traces', lambda env: setattr(env, '_benchmark_trace', Trace()))
    loop.run_loop(ctx)
    report = json.loads(ctx.args_cli.results_json.read_text())
    assert report['complete'] and report['overall']['successes'] == 1
    assert [e['start_index'] for e in report['episodes']] == [0, 1]
    assert [e['token'] for e in report['episodes']] == [0, 1]
    assert [e['success'] for e in report['episodes']] == [True, False]
    assert [e['stages']['done'] for e in report['episodes']] == [True, False]
    assert all(e['steps'] == 12 for e in report['episodes'])
    assert calls == [{}, {}]
    assert ctx.env.actions[0][0, 0] == 0 and ctx.env.actions[12][0, 0] == 1


def test_stopped_app_writes_incomplete_report(monkeypatch, tmp_path):
    ctx, _ = context(tmp_path)
    ctx.simulation_app.is_running = lambda: False
    with pytest.raises(RuntimeError, match='0/2 episodes'):
        loop.run_loop(ctx)
    report = json.loads(ctx.args_cli.results_json.read_text())
    assert not report['complete'] and report['episodes'] == []
