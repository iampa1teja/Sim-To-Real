"""Exercise the real evaluator function with a simulated auto-reset environment."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sim_to_real_so101.utils.pick_place_eval',
                                            ROOT / 'source/sim_to_real_so101/utils/pick_place_eval.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


class FakePolicy:
    instances = []

    def __init__(self, **kwargs):
        self.instructions = []
        self.instances.append(self)

    def connect(self):
        pass

    def reset(self):
        self.instructions.append(getattr(self, '_lang_instruction', None))


class FakeEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.device = 'cpu'
        self.scene = {}
        self.cfg = SimpleNamespace()
        self.action_space = SimpleNamespace(shape=(1, 6))
        self.observation_space = 'fake'
        self.max_episode_length = 1
        self.explicit_resets = 0
        self.episode = 0
        self.closed = False

    def metadata(self):
        self._pick_place_start_index = [self.episode]
        self._pick_place_start_zone = ['NL' if self.episode % 2 == 0 else 'FR']
        self._pick_place_cube_color = ['blue' if self.episode % 2 == 0 else 'red']

    def reset(self):
        self.explicit_resets += 1
        self.metadata()
        return {}, {}

    def step(self, actions):
        self.episode += 1
        self.metadata()  # Isaac auto-reset overwrites metadata before returning.
        timeout = self.episode == 2
        return {}, 0, np.array([not timeout]), np.array([timeout]), {}

    def close(self):
        self.closed = True


class EvaluatorRolloutTests(unittest.TestCase):
    def evaluate(self, path, stop_after=None):
        env = FakeEnvironment()
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), events=SimpleNamespace(spawn_cube=SimpleNamespace(params={})))
        args = SimpleNamespace(task='Lerobot-So101-Teleop-Pick-Place-Eval', device='cpu', num_envs=1,
                               disable_fabric=False, seed=17, num_episodes=3, random_fraction=0.,
                               start_mode='cycle', cube_starts='/fake/metadata', rename_map=None,
                               lang_instruction_by_color={'blue': 'blue task', 'red': 'red task'},
                               lang_instruction='fallback', results_json=path, checkpoint='checkpoint-10',
                               rerun=False, policy_host='localhost', policy_port=5555, action_horizon=16)
        torch = SimpleNamespace(manual_seed=lambda _: None, cuda=SimpleNamespace(manual_seed_all=lambda _: None),
                                backends=SimpleNamespace(cudnn=SimpleNamespace()), inference_mode=nullcontext,
                                zeros=lambda shape, **kw: np.zeros(shape), tensor=lambda values, **kw: np.array(values))
        interface = SimpleNamespace(init_device=lambda **kw: None)
        simulation = SimpleNamespace(is_running=lambda: stop_after is None or env.episode < stop_after)
        context = dict(args_cli=args, parse_env_cfg=lambda *a, **kw: cfg, gym=SimpleNamespace(make=lambda *a, **kw: env),
                       KeyboardControl=lambda: SimpleNamespace(reset_world=False), torch=torch, np=np, random=random,
                       json=json, LeRobotSO101Interface=lambda **kw: interface, GR00TRemotePolicy=FakePolicy,
                       simulation_app=simulation, tqdm=lambda **kw: SimpleNamespace(update=lambda _: None, close=lambda: None))
        tree = ast.parse((ROOT / 'source/sim_to_real_so101/scripts/lerobot_eval.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_evaluate')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'lerobot_eval.py', 'exec'), context)
        with patch.dict(sys.modules, {'sim_to_real_so101.utils.pick_place_eval': HELPERS}), patch('builtins.print'):
            if stop_after is None:
                context['_evaluate']()
            else:
                with self.assertRaisesRegex(RuntimeError, 'Evaluation stopped'):
                    context['_evaluate']()
        return env, cfg

    def test_autoreset_preserves_episode_metadata_and_exact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, cfg = self.evaluate(path)
            report = json.loads(path.read_text())
        self.assertTrue(env.closed)
        self.assertEqual(env.explicit_resets, 1)
        self.assertEqual(env.episode, 3)
        self.assertEqual(cfg.events.spawn_cube.params['starts_dir'], '/fake/metadata')
        self.assertEqual([row['start_index'] for row in report['episodes']], [0, 1, 2])
        self.assertEqual([row['instruction'] for row in report['episodes']], ['blue task', 'red task', 'blue task'])
        self.assertEqual([row['success'] for row in report['episodes']], [True, False, True])
        self.assertTrue(report['complete'])
        self.assertEqual(report['overall']['success_rate'], 2/3)

    def test_closed_app_writes_partial_report_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, stop_after=1)
            report = json.loads(path.read_text())
        self.assertTrue(env.closed)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['episodes']), 1)


if __name__ == '__main__':
    unittest.main()
