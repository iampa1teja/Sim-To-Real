"""Exercise the real evaluator function with a simulated auto-reset environment."""
import ast
import argparse
from contextlib import nullcontext, redirect_stderr
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sim_to_real_so101.utils.pick_place_eval',
                                            ROOT / 'source/sim_to_real_so101/utils/pick_place_eval.py')
sys.path.insert(0, str(ROOT / 'source'))
from sim_to_real_so101.gr00t_client.grasp import GraspConfig, add_grasp_arguments
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
EVALUATOR = ROOT / 'source/sim_to_real_so101/scripts/lerobot_eval.py'


class RecordingStderr(io.StringIO):
    def __init__(self):
        super().__init__()
        self.last_flushed_text = ''

    def flush(self):
        self.last_flushed_text = self.getvalue()
        super().flush()


class FakePolicy:
    instances = []

    def __init__(self, **kwargs):
        self.instructions = []
        self.options = kwargs
        self.instances.append(self)

    def connect(self):
        pass

    def reset(self):
        self.instructions.append(getattr(self, '_lang_instruction', None))

    def get_action(self, *args, **kwargs):
        return np.zeros(6)


class FakeEnvironment:
    def __init__(self, outcomes=None, step_dt=1/30, failure_stage=None, close_error=None):
        self.unwrapped = self
        self.device = 'cpu'
        self.scene = {}
        self.cfg = SimpleNamespace()
        self.action_space = SimpleNamespace(shape=(1, 6))
        self.observation_space = 'fake'
        self.max_episode_length = 1
        self.step_dt = step_dt
        self.outcomes = outcomes or [(1, True, False), (1, False, True), (1, True, False)]
        self.episode_step = 0
        self.explicit_resets = 0
        self.episode = 0
        self.closed = False
        self.made_episode_length_s = None
        self.action_history = []
        self.failure_stage = failure_stage
        self.close_error = close_error
        self.on_close = None

    def reset_pose(self, episode=None):
        if episode is None:
            episode = self.episode
        return np.array([.2, -.3, .4, -.5, .6, .7]) + episode * .05

    def observation(self):
        # Physical drift distinguishes holding the reset pose from following
        # each subsequent observation during the settling period.
        position = self.reset_pose() + self.episode_step * .01
        return {'policy': {'joint_pos_obs': [SimpleNamespace(clone=lambda: position.copy())]}, 'visual': {}}

    def metadata(self):
        self._pick_place_start_index = [self.episode]
        self._pick_place_start_zone = ['NL' if self.episode % 2 == 0 else 'FR']
        self._pick_place_cube_color = ['blue' if self.episode % 2 == 0 else 'red']

    def reset(self):
        self.explicit_resets += 1
        if self.failure_stage == 'reset':
            raise RuntimeError('Injected environment reset failure')
        self.metadata()
        self.episode_step = 0
        return self.observation(), {}

    def step(self, actions):
        if self.failure_stage == 'step':
            raise RuntimeError('Injected environment step failure')
        self.action_history.append((self.episode, self.episode_step, actions.copy()))
        self.episode_step += 1
        duration, success, timeout = self.outcomes[self.episode]
        done = self.episode_step == duration
        if done:
            self.episode += 1
            self.episode_step = 0
            self.metadata()  # Isaac auto-reset overwrites metadata before returning.
        return self.observation(), 0, np.array([done and success]), np.array([done and timeout]), {}

    def close(self):
        self.closed = True
        if self.on_close is not None:
            self.on_close()
        if self.close_error is not None:
            raise self.close_error


class EvaluatorRolloutTests(unittest.TestCase):
    def evaluate(self, path, stop_after=None, args_override=None, task_episode_length_s=15., outcomes=None,
                 step_dt=1/30, expected_error=None, failure_stage=None, close_error=None, run_main=False):
        env = FakeEnvironment(outcomes, step_dt, failure_stage, close_error)
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), episode_length_s=task_episode_length_s,
                              events=SimpleNamespace(spawn_cube=SimpleNamespace(params={})))
        args = SimpleNamespace(task='Lerobot-So101-Teleop-Pick-Place-Eval', device='cpu', num_envs=1,
                               disable_fabric=False, seed=17, num_episodes=3, random_fraction=0.,
                               start_mode='cycle', robot_start='recorded', cube_starts='/fake/metadata', rename_map=None,
                               lang_instruction_by_color={'blue': 'blue task', 'red': 'red task'},
                               lang_instruction='fallback', results_json=path, checkpoint='checkpoint-10',
                               rerun=False, policy_host='localhost', policy_port=5555, action_horizon=16,
                               episode_length_s=None)
        for key, value in (vars(GraspConfig()) | {"seed_per_episode": False}).items():
            setattr(args, key, value)
        for key, value in (args_override or {}).items():
            setattr(args, key, value)
        torch = SimpleNamespace(manual_seed=lambda _: None, cuda=SimpleNamespace(manual_seed_all=lambda _: None),
                                backends=SimpleNamespace(cudnn=SimpleNamespace()), inference_mode=nullcontext,
                                zeros=lambda shape, **kw: np.zeros(shape), tensor=lambda values, **kw: np.array(values))
        interface = SimpleNamespace(init_device=lambda **kw: None)
        simulation = SimpleNamespace(is_running=lambda: stop_after is None or env.episode < stop_after,
                                     close=lambda: setattr(env, 'application_closed', True))
        def make_environment(*args, **kwargs):
            env.made_episode_length_s = kwargs['cfg'].episode_length_s
            env.max_episode_length = math.ceil(env.made_episode_length_s / env.step_dt)
            return env
        context = dict(GraspConfig=GraspConfig, args_cli=args, parse_env_cfg=lambda *a, **kw: cfg, gym=SimpleNamespace(make=make_environment),
                       KeyboardControl=lambda: SimpleNamespace(reset_world=False), torch=torch, np=np, random=random,
                       json=json, math=math, sys=sys, traceback=traceback,
                       LeRobotSO101Interface=lambda **kw: interface, GR00TRemotePolicy=FakePolicy,
                       simulation_app=simulation, tqdm=lambda **kw: SimpleNamespace(update=lambda _: None, close=lambda: None))
        tree = ast.parse(EVALUATOR.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_evaluate', 'main')]
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'lerobot_eval.py', 'exec'), context)
        invoke = context['main' if run_main else '_evaluate']
        stderr = RecordingStderr()

        def observe_close():
            env.stderr_at_close = stderr.getvalue()
            env.flushed_stderr_at_close = stderr.last_flushed_text

        real_print = print

        def capture_print(*values, **options):
            if options.get('file') is stderr:
                real_print(*values, **options)

        env.on_close = observe_close
        with patch.dict(sys.modules, {'sim_to_real_so101.utils.pick_place_eval': HELPERS}), \
                patch('builtins.print', side_effect=capture_print) as printer, redirect_stderr(stderr):
            if close_error is not None:
                with self.assertRaises(type(close_error)) as raised:
                    invoke()
                env.cleanup_exception = raised.exception
            elif failure_stage is not None:
                with self.assertRaisesRegex(RuntimeError, f'Injected environment {failure_stage} failure'):
                    invoke()
            elif expected_error is not None:
                with self.assertRaisesRegex(ValueError, expected_error):
                    invoke()
            elif stop_after is None:
                invoke()
            else:
                with self.assertRaisesRegex(RuntimeError, 'Evaluation stopped'):
                    invoke()
        env.printed = [call.args[0] for call in printer.call_args_list]
        env.stderr = stderr.getvalue()
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
        self.assertIs(cfg.events.spawn_cube.params['reset_robot'], True)
        self.assertEqual([row['start_index'] for row in report['episodes']], [0, 1, 2])
        self.assertEqual([row['instruction'] for row in report['episodes']], ['blue task', 'red task', 'blue task'])
        self.assertEqual([row['success'] for row in report['episodes']], [True, False, True])
        self.assertEqual([row['success_step'] for row in report['episodes']], [1, None, 1])
        self.assertTrue(report['complete'])
        self.assertEqual(report['overall']['success_rate'], 2/3)
        self.assertEqual(report['overall']['success_rate_within_15s'], 2/3)
        self.assertEqual(report['overall']['median_time_to_success_s'], 1/30)
        self.assertEqual(report['random_fraction'], 0)
        self.assertEqual(report['episode_length_s'], 15)
        self.assertEqual(report['action_horizon'], 16)
        self.assertEqual(report['step_dt'], 1/30)
        self.assertEqual(env.made_episode_length_s, 15)
        self.assertEqual(json.loads(next(line for line in env.printed if line.startswith('{'))), report)

    def test_override_is_applied_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, cfg = self.evaluate(path, args_override={'episode_length_s': 30., 'action_horizon': 8,
                                                        'random_fraction': .25})
            report = json.loads(path.read_text())
        self.assertEqual(env.made_episode_length_s, 30.)
        self.assertEqual(cfg.episode_length_s, 30.)
        self.assertEqual(report['episode_length_s'], 30.)
        self.assertEqual(report['action_horizon'], 8)
        self.assertEqual(report['random_fraction'], .25)
        self.assertEqual(cfg.events.spawn_cube.params['random_fraction'], .25)
        self.assertEqual(FakePolicy.instances[-1].options['action_horizon'], 8)

    def test_unset_timeout_preserves_the_selected_task_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, cfg = self.evaluate(path, task_episode_length_s=7.5)
            report = json.loads(path.read_text())
        self.assertEqual(env.made_episode_length_s, 7.5)
        self.assertEqual(cfg.episode_length_s, 7.5)
        self.assertEqual(report['episode_length_s'], 7.5)

    def test_recorded_starts_hold_each_reset_pose_for_ten_steps(self):
        for task in ('Lerobot-So101-Teleop-Pick-Place-Eval', 'Lerobot-So101-Teleop-Pick-Place-DR-Eval'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                env, _ = self.evaluate(path, args_override={'task': task}, outcomes=[(12, True, False)] * 3)
                report = json.loads(path.read_text())
            for episode, step, action in env.action_history:
                expected = env.reset_pose(episode) if step < 10 else np.zeros(6)
                np.testing.assert_allclose(action[0], expected)
            self.assertEqual([row['steps'] for row in report['episodes']], [12, 12, 12])
            self.assertEqual([row['success_step'] for row in report['episodes']], [12, 12, 12])
            self.assertEqual(env.explicit_resets, 1)

    def test_default_and_other_tasks_keep_calibrated_settling_pose(self):
        options = [
            {'robot_start': 'default'},
            {'task': 'Lerobot-So101-Teleop-MyRoom', 'cube_starts': None, 'lang_instruction_by_color': None},
        ]
        initial_action = np.array([-.2736, -.6109, -.0745, 1.5148, -1.6034, -.1465])
        for override in options:
            with self.subTest(options=override), tempfile.TemporaryDirectory() as directory:
                env, _ = self.evaluate(Path(directory) / 'report.json', args_override=override,
                                       outcomes=[(12, True, False)] * 3)
            for _, step, action in env.action_history:
                np.testing.assert_allclose(action[0], initial_action if step < 10 else np.zeros(6))

    def test_invalid_options_never_create_an_environment(self):
        for value in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(episode_length_s=value):
                env, _ = self.evaluate(None, args_override={'episode_length_s': value},
                                       expected_error='episode_length_s')
                self.assertIsNone(env.made_episode_length_s)
        for value in (0, -1, 17):
            with self.subTest(action_horizon=value):
                env, _ = self.evaluate(None, args_override={'action_horizon': value},
                                       expected_error='action_horizon')
                self.assertIsNone(env.made_episode_length_s)

    def test_action_horizon_boundaries_are_forwarded(self):
        for horizon in (1, 16):
            with self.subTest(action_horizon=horizon), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                self.evaluate(path, args_override={'action_horizon': horizon})
                self.assertEqual(json.loads(path.read_text())['action_horizon'], horizon)
                self.assertEqual(FakePolicy.instances[-1].options['action_horizon'], horizon)

    def test_30_second_rollout_distinguishes_boundary_late_success_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'episode_length_s': 30.},
                                   outcomes=[(450, True, False), (451, True, False), (900, False, True)])
            report = json.loads(path.read_text())
        self.assertEqual([row['steps'] for row in report['episodes']], [450, 451, 900])
        self.assertEqual([row['success_step'] for row in report['episodes']], [450, 451, None])
        self.assertEqual(report['overall']['success_rate'], 2/3)
        self.assertEqual(report['overall']['successes_within_15s'], 1)
        self.assertEqual(report['overall']['success_rate_within_15s'], 1/3)
        self.assertAlmostEqual(report['overall']['median_time_to_success_s'], (450 + 451) / 60)
        self.assertEqual(report['by_zone']['FR']['success_rate'], 1.)
        self.assertEqual(report['by_zone']['FR']['success_rate_within_15s'], 0.)
        self.assertEqual(report['by_cube_color']['red']['success_rate_within_15s'], 0.)
        self.assertEqual(env.explicit_resets, 1)

    def test_closed_app_writes_partial_report_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, stop_after=1)
            report = json.loads(path.read_text())
        self.assertTrue(env.closed)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['episodes']), 1)

    def test_app_stopped_before_first_step_reports_zero_episodes_and_fails_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, stop_after=0)
            report = json.loads(path.read_text())
        self.assertTrue(env.closed)
        self.assertEqual(env.action_history, [])
        self.assertEqual(report['episodes'], [])
        self.assertEqual(report['overall']['episodes'], 0)
        self.assertIsNone(report['overall']['success_rate'])
        self.assertFalse(report['complete'])
        self.assertIn('RuntimeError: Evaluation stopped after 0/3 episodes', env.stderr_at_close)
        self.assertEqual(env.flushed_stderr_at_close, env.stderr_at_close)

    def test_zero_episode_error_is_visible_when_environment_cleanup_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, stop_after=0, close_error=SystemExit(0))
            report = json.loads(path.read_text())
        self.assertEqual(env.cleanup_exception.code, 0)
        self.assertFalse(report['complete'])
        self.assertEqual(report['episodes'], [])
        self.assertIn('Traceback (most recent call last)', env.stderr_at_close)
        self.assertIn('RuntimeError: Evaluation stopped after 0/3 episodes', env.stderr_at_close)
        self.assertEqual(env.flushed_stderr_at_close, env.stderr_at_close)

    def test_reset_and_step_errors_are_visible_before_cleanup_exits_zero(self):
        for stage in ('reset', 'step'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                env, _ = self.evaluate(path, failure_stage=stage, close_error=SystemExit(0))
                report = json.loads(path.read_text())
            self.assertTrue(env.closed)
            self.assertEqual(env.cleanup_exception.code, 0)
            self.assertFalse(report['complete'])
            self.assertEqual(report['episodes'], [])
            self.assertIn('Traceback (most recent call last)', env.stderr_at_close)
            self.assertIn(f'RuntimeError: Injected environment {stage} failure', env.stderr_at_close)
            self.assertEqual(env.flushed_stderr_at_close, env.stderr_at_close)

    def test_main_preserves_rollout_failure_without_duplicate_tracebacks(self):
        with tempfile.TemporaryDirectory() as directory:
            env, _ = self.evaluate(Path(directory) / 'report.json', stop_after=0, run_main=True)
        self.assertTrue(env.closed)
        self.assertTrue(env.application_closed)
        self.assertEqual(env.stderr.count('Traceback (most recent call last)'), 1)
        self.assertEqual(env.stderr.count('RuntimeError: Evaluation stopped after 0/3 episodes'), 1)


class EvaluatorShutdownTests(unittest.TestCase):
    def invoke_main(self, error):
        tree = ast.parse(EVALUATOR.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        stderr = RecordingStderr()
        close_observations = []

        def evaluate():
            raise error

        def close():
            close_observations.append((stderr.getvalue(), stderr.last_flushed_text))
            raise SystemExit(0)

        context = dict(_evaluate=evaluate, simulation_app=SimpleNamespace(close=close), traceback=traceback, sys=sys)
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(EVALUATOR), 'exec'), context)
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            context['main']()
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(len(close_observations), 1)
        return close_observations[0]

    def test_setup_failure_is_printed_and_flushed_before_application_shutdown(self):
        printed, flushed = self.invoke_main(ValueError('Injected policy setup failure'))
        self.assertIn('Traceback (most recent call last)', printed)
        self.assertIn('ValueError: Injected policy setup failure', printed)
        self.assertEqual(flushed, printed)

class EvaluatorCliTests(unittest.TestCase):
    class Launcher:
        launches = 0

        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, args):
            type(self).launches += 1
            self.app = SimpleNamespace()

    def parse_cli(self, arguments):
        self.Launcher.launches = 0
        tree = ast.parse(EVALUATOR.read_text())
        # Run the real command-line setup through app launch, using an inert
        # launcher so invalid options must fail before any simulator starts.
        boundary = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Import)
                        and any(alias.name == 'gymnasium' for alias in node.names))
        body = [node for node in tree.body[:boundary] if not isinstance(node, (ast.Import, ast.ImportFrom))]
        context = dict(argparse=argparse, json=json, math=math, os=os, Path=Path, AppLauncher=self.Launcher, add_grasp_arguments=add_grasp_arguments)
        with patch.object(sys, 'argv', ['lerobot_eval', *arguments]), redirect_stderr(io.StringIO()):
            exec(compile(ast.Module(body=body, type_ignores=[]), str(EVALUATOR), 'exec'), context)
        return context['args_cli']

    def test_cli_defaults_and_valid_bounds(self):
        args = self.parse_cli([])
        self.assertIsNone(args.episode_length_s)
        self.assertEqual(args.action_horizon, 16)
        self.assertEqual(args.random_fraction, 0.)
        self.assertEqual(args.robot_start, 'recorded')
        for horizon in ('1', '16'):
            args = self.parse_cli(['--action_horizon', horizon, '--episode_length_s', '30'])
            self.assertEqual(args.action_horizon, int(horizon))
            self.assertEqual(args.episode_length_s, 30.)

    def test_invalid_cli_options_fail_before_simulator_launch(self):
        options = [('episode_length_s', value) for value in ('0', '-1', 'nan', 'inf', '-inf', 'bad')]
        options += [('action_horizon', value) for value in ('0', '17', '-1', '8.5', 'nan')]
        for name, value in options:
            with self.subTest(option=name, value=value), self.assertRaises(SystemExit) as error:
                self.parse_cli([f'--{name}={value}'])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(self.Launcher.launches, 0)


if __name__ == '__main__':
    unittest.main()
