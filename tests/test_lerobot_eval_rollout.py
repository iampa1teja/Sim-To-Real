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
from sim_to_real_so101.policy_client.grasp import GraspConfig, add_grasp_arguments
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


class FakeProgress:
    def __init__(self, **options):
        self.options = options
        self.postfixes = []
        self.closed = False

    def update(self, amount):
        pass

    def set_postfix_str(self, value):
        self.postfixes.append(value)

    def close(self):
        self.closed = True


class FakePolicy:
    instances = []

    def __init__(self, **kwargs):
        self.instructions = []
        self.reset_seeds = []
        self.options = kwargs
        self.instances.append(self)

    def connect(self):
        pass

    def reset(self, **kwargs):
        self.instructions.append(getattr(self, '_lang_instruction', None))
        self.reset_seeds.append(kwargs.get('seed'))

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
        pixels = np.full((1, 16, 24, 3), min(255, 40 + self.episode * 60 + self.episode_step), dtype=np.uint8)
        return {'policy': {'joint_pos_obs': [SimpleNamespace(clone=lambda: position.copy())]},
                'visual': {'rgb_realsense_rgb': pixels, 'rgb_wrist_cam': pixels.copy()}}

    def metadata(self):
        self._pick_place_start_index = [self.episode]
        self._pick_place_start_zone = ['NL' if self.episode % 2 == 0 else 'FR']
        self._pick_place_cube_color = ['blue' if self.episode % 2 == 0 else 'red']
        if getattr(self, 'benchmark', None) is not None:
            self.benchmark.sample()
            self._benchmark_start = self.benchmark.current

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
        robot = SimpleNamespace(spawn=SimpleNamespace(usd_path='/task/default.usd'))
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1, robot=robot), episode_length_s=task_episode_length_s,
                              events=SimpleNamespace(spawn_cube=SimpleNamespace(params={})))
        args = SimpleNamespace(task='Lerobot-So101-Teleop-Pick-Place-Eval', device='cpu', num_envs=1,
                               disable_fabric=False, seed=17, num_episodes=3, random_fraction=0.,
                               start_mode='cycle', robot_start='recorded', cube_starts='/fake/metadata', rename_map=None,
                               lang_instruction_by_color={'blue': 'blue task', 'red': 'red task'},
                               lang_instruction='fallback', results_json=path, checkpoint='checkpoint-10',
                               rerun=False, policy='groot_n17', policy_option=[], policy_api_token=None, policy_host='localhost', policy_port=None, action_horizon=16,
                               episode_length_s=None, robot_usd=None)
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
            env.made_robot_usd = kwargs['cfg'].scene.robot.spawn.usd_path
            env.made_overhead_camera = getattr(kwargs['cfg'].scene, 'camera_molmo_room', None)
            env.max_episode_length = math.ceil(env.made_episode_length_s / env.step_dt)
            params = kwargs['cfg'].events.spawn_cube.params
            if params.get('eval_set'):
                from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts
                env.benchmark = EvalSetStarts(params['eval_set'], params['splits'], params['repeats'], cfg.seed)
            return env
        env.progress_bars = []

        def progress_bar(**options):
            bar = FakeProgress(**options)
            env.progress_bars.append(bar)
            return bar

        def make_interface(**options):
            env.interface_options = options
            return interface

        context = dict(GraspConfig=GraspConfig, args_cli=args, parse_env_cfg=lambda *a, **kw: cfg, gym=SimpleNamespace(make=make_environment),
                       KeyboardControl=lambda: SimpleNamespace(reset_world=False), torch=torch, np=np, random=random,
                       json=json, math=math, sys=sys, traceback=traceback,
                       LeRobotSO101Interface=make_interface, SimRemotePolicy=FakePolicy,
                       simulation_app=simulation, tqdm=progress_bar)
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

        modules = {'sim_to_real_so101.utils.pick_place_eval': HELPERS}
        if getattr(args, 'eval_set', None):
            from sim_to_real_so101.utils.pick_place_benchmark import StageTrace, load_eval_set
            scene = load_eval_set(args.eval_set)['scene']
            modules['sim_to_real_so101.utils.pick_place_benchmark_runtime'] = SimpleNamespace(
                capture_scene=lambda _: scene,
                install_stage_tracking=lambda raw: setattr(raw, '_benchmark_trace', StageTrace()))
        env.on_close = observe_close
        with patch.dict(sys.modules, modules), \
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

    def test_smoke_test_runs_five_episodes_and_overrides_all_saved_outputs(self):
        recording = SimpleNamespace(flush=unittest.mock.Mock())
        rerun = SimpleNamespace(get_global_data_recording=lambda: recording, disconnect=unittest.mock.Mock())
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'rerun': rerun, 'sim_to_real_so101.utils.eval_video': None}):
            root = Path(directory)
            env, _ = self.evaluate(root / 'reports/report.json',
                                   args_override={'smoke_test': True, 'num_episodes': 99,
                                                  'episode_length_s': 300., 'save_videos': True,
                                                  'video_dir': root / 'videos', 'log_timing': True,
                                                  'timing_csv': str(root / 'timing.csv'), 'log_gripper': True,
                                                  'gripper_csv': str(root / 'gripper.csv')},
                                   outcomes=[(1, True, False)] * 6)
            self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(env.episode, 5)
        self.assertEqual(env.made_episode_length_s, 300.)
        self.assertTrue(env.closed)
        self.assertEqual(len(env.progress_bars), 5)
        self.assertNotIn('log_timing', FakePolicy.instances[-1].options)
        self.assertNotIn('timing_csv', FakePolicy.instances[-1].options)
        self.assertNotIn('grasp_config', FakePolicy.instances[-1].options)
        self.assertIn('[RESULT] complete: 5/5 episodes; success 5/5 (100.0%)', env.printed)
        self.assertFalse(any(line.startswith(('Results JSON:', '{')) for line in env.printed))
        recording.flush.assert_called_once_with(timeout_sec=5)
        rerun.disconnect.assert_called_once_with()

    def test_smoke_test_caps_fixed_benchmark_and_repeats_small_selections(self):
        from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts, eval_set_digest
        benchmark_path = ROOT / 'eval_sets/pick_place_v1_seed1984.json'
        for start_count in (150, 2):
            with self.subTest(starts=start_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                selection = json.loads(benchmark_path.read_text())
                selection['starts'] = selection['starts'][:start_count]
                selection['eval_set_id'] = eval_set_digest(selection)
                selected_path = root / 'selection.json'
                selected_path.write_text(json.dumps(selection))
                rerun = SimpleNamespace(get_global_data_recording=lambda: None, disconnect=lambda: None)
                with patch.dict(sys.modules, {'rerun': rerun, 'sim_to_real_so101.utils.eval_video': None}):
                    env, cfg = self.evaluate(root / 'report.json',
                                            args_override={'smoke_test': True, 'eval_set': selected_path,
                                                           'splits': None, 'repeats': 1, 'save_videos': True},
                                            outcomes=[(1, True, False)] * 6)
                self.assertEqual(env.episode, 5)
                self.assertEqual(list(root.iterdir()), [selected_path])
                repeats = cfg.events.spawn_cube.params['repeats']
                self.assertEqual(repeats, 1 if start_count >= 5 else 3)
                plan = EvalSetStarts(selected_path, repeats=repeats, seed=17)
                self.assertEqual(FakePolicy.instances[-1].reset_seeds,
                                 [s['policy_seed'] for s in plan.schedule[:5]])
                self.assertIn('[RESULT] complete: 5/5 episodes; success 5/5 (100.0%)', env.printed)

    def test_molmo_smoke_diagnostics_are_console_only_and_reset_each_episode(self):
        from sim_to_real_so101.utils.grasp_diagnostics import GraspTrace
        trace = GraspTrace()
        install = unittest.mock.Mock(return_value=trace)
        rerun = SimpleNamespace(get_global_data_recording=lambda: None, disconnect=lambda: None)
        module = SimpleNamespace(install_grasp_diagnostics=install)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'rerun': rerun, 'sim_to_real_so101.utils.grasp_diagnostics': module}):
            root = Path(directory)
            env, _ = self.evaluate(root / 'report.json',
                                   args_override={'smoke_test': True, 'policy': 'molmoact2',
                                                  'action_horizon': 30},
                                   outcomes=[(1, True, False)] * 5)
            self.assertEqual(list(root.iterdir()), [])
        install.assert_called_once()
        lines = [line for line in env.printed if line.startswith('[GRASP ')]
        self.assertEqual(len(lines), 5)
        self.assertTrue(all(json.loads(line.split('] ', 1)[1])['closest'] is None for line in lines))
        self.assertTrue(any('Replanning every 1.00s sim' in line for line in env.printed))

    def test_smoke_test_failure_closes_environment_without_saving_partial_report(self):
        rerun = SimpleNamespace(get_global_data_recording=lambda: None, disconnect=lambda: None)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'rerun': rerun}):
            root = Path(directory)
            env, _ = self.evaluate(root / 'report.json', args_override={'smoke_test': True, 'save_videos': True},
                                   failure_stage='step')
            self.assertEqual(list(root.iterdir()), [])
        self.assertTrue(env.closed)
        self.assertIn('[RESULT] incomplete: 0/5 episodes; success 0/0 (n/a)', env.printed)

    def test_molmo_overhead_camera_is_added_before_spawn_and_identified_in_report(self):
        camera = object()
        metadata = {'camera': 'molmo_room', 'position': [-.5, 0., .8]}
        calls = []

        def add_camera(cfg):
            calls.append(cfg)
            cfg.scene.camera_molmo_room = camera
            return metadata

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'sim_to_real_so101.utils.molmo_camera': SimpleNamespace(add_molmo_overhead_camera=add_camera)}):
            path = Path(directory) / 'report.json'
            env, cfg = self.evaluate(path, args_override={'policy': 'molmoact2', 'action_horizon': 30,
                                                        'molmo_overhead_camera': True,
                                                        'rename_map': '{"realsense_rgb":"room","wrist_cam":"wrist"}'})
            report = json.loads(path.read_text())
        self.assertEqual(calls, [cfg])
        self.assertIs(env.made_overhead_camera, camera)
        self.assertEqual(env.interface_options['rename_map'],
                         {'realsense_rgb': 'original_room', 'molmo_room': 'room', 'wrist_cam': 'wrist'})
        self.assertEqual(report['molmo_overhead_camera'], metadata)

    def test_default_rollout_never_imports_or_adds_molmo_camera(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'sim_to_real_so101.utils.molmo_camera': None}):
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path)
            report = json.loads(path.read_text())
        self.assertIsNone(env.made_overhead_camera)
        self.assertNotIn('molmo_overhead_camera', report)
        self.assertIsNone(env.interface_options['rename_map'])

    def test_disabling_molmo_camera_restores_original_room_mapping(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'sim_to_real_so101.utils.molmo_camera': None}):
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'policy': 'molmoact2', 'action_horizon': 30,
                                   'molmo_overhead_camera': False,
                                   'rename_map': '{"realsense_rgb":"original_room","molmo_room":"room","wrist_cam":"wrist"}'})
        self.assertIsNone(env.made_overhead_camera)
        self.assertEqual(env.interface_options['rename_map'], {'realsense_rgb': 'room', 'wrist_cam': 'wrist'})

    def test_molmo_overhead_rejects_other_policy_or_task_before_environment_creation(self):
        for changes in ({'policy': 'groot_n17'}, {'policy': 'molmoact2', 'task': 'Lerobot-So101-Teleop-MyRoom'}):
            with self.subTest(changes=changes):
                env, _ = self.evaluate(None, args_override={'molmo_overhead_camera': True, **changes},
                                       expected_error='requires molmoact2 and a pick-place Eval task')
                self.assertIsNone(env.made_episode_length_s)

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
        self.assertEqual(report['robot_usd'], '/task/default.usd')
        self.assertEqual(env.made_episode_length_s, 15)
        self.assertIn('[RESULT] complete: 3/3 episodes; success 2/3 (66.7%)', env.printed)
        self.assertFalse(any(line.startswith('{') for line in env.printed))
        self.assertIn(f'Results JSON: {path}', env.printed)

    def test_robot_usd_override_is_applied_before_spawn_and_saved_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            asset = ROOT / 'source/sim_to_real_so101/assets/usd/ARM_new.usd'
            env, _ = self.evaluate(path, args_override={'robot_usd': asset})
            report = json.loads(path.read_text())
        self.assertEqual(env.made_robot_usd, str(asset.resolve()))
        self.assertEqual(report['robot_usd'], str(asset.resolve()))

    def test_video_option_saves_playable_views_without_crossing_auto_reset(self):
        if importlib.util.find_spec('av') is None:
            self.skipTest('PyAV is required for real video encoding')
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'save_videos': True, 'num_episodes': 2},
                                   outcomes=[(2, True, False), (3, False, True)])
            report = json.loads(path.read_text())
            self.assertTrue(report['save_videos'])
            self.assertEqual(report['video_dir'], str(Path(directory) / 'report_videos'))
            for row, frame_count, lower, upper in zip(report['episodes'], (2, 3), (30, 90), (60, 120)):
                self.assertEqual(set(row['videos']), {'room', 'wrist'})
                for filename in row['videos'].values():
                    with av.open(filename) as recording:
                        self.assertEqual(recording.streams.video[0].average_rate, 30)
                        frames = [frame.to_ndarray(format='rgb24') for frame in recording.decode(video=0)]
                    self.assertEqual(len(frames), frame_count)
                    self.assertTrue(all(lower < frame.mean() < upper for frame in frames))
            self.assertEqual(len(list(Path(report['video_dir']).glob('*.mp4'))), 4)
        self.assertTrue(env.closed)

    def test_disabled_video_needs_no_recorder_import_or_output_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'sim_to_real_so101.utils.eval_video': None}):
            path = Path(directory) / 'report.json'
            self.evaluate(path)
            self.assertFalse((Path(directory) / 'report_videos').exists())
            self.assertNotIn('videos', json.loads(path.read_text())['episodes'][0])

    def test_video_files_are_finalized_when_the_rollout_fails(self):
        if importlib.util.find_spec('av') is None:
            self.skipTest('PyAV is required for real video encoding')
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'save_videos': True}, failure_stage='step')
            report = json.loads(path.read_text())
            self.assertFalse(report['complete'])
            files = list(Path(report['video_dir']).glob('*.mp4'))
            self.assertEqual(len(files), 2)
            for filename in files:
                with av.open(str(filename)) as recording:
                    self.assertEqual(len(list(recording.decode(video=0))), 1)
        self.assertTrue(env.closed)

    def test_rerun_flushes_and_disconnects_at_the_end_of_the_episode(self):
        calls = []
        recording = SimpleNamespace(flush=lambda **kw: calls.append(('flush', kw)))
        rerun = SimpleNamespace(get_global_data_recording=lambda: recording,
                                disconnect=lambda: calls.append(('disconnect', {})))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'rerun': rerun}):
            env, _ = self.evaluate(Path(directory) / 'report.json', args_override={'rerun': True})
        self.assertTrue(env.closed)
        self.assertEqual(calls, [('flush', {'timeout_sec': 5}), ('disconnect', {})])

    def test_missing_robot_usd_is_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'robot_usd': Path(directory) / 'missing.usd'},
                                   expected_error='Robot USD does not exist')
            self.assertFalse(path.exists())
        self.assertIsNone(env.made_episode_length_s)

    def test_progress_tracks_completed_successes_and_final_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            env, _ = self.evaluate(Path(directory) / 'report.json')
        bars = env.progress_bars
        self.assertEqual([bar.options['desc'] for bar in bars],
                         ['Rollout (ep 1/3)', 'Rollout (ep 2/3)', 'Rollout (ep 3/3)'])
        self.assertEqual([bar.postfixes[0] for bar in bars],
                         ['success=N/A (0/0)', 'success=100.0% (1/1)', 'success=50.0% (1/2)'])
        self.assertEqual([bar.postfixes[-1] for bar in bars],
                         ['success=100.0% (1/1)', 'success=50.0% (1/2)', 'success=66.7% (2/3)'])
        self.assertTrue(all(bar.closed for bar in bars))

    def test_progress_counts_simultaneous_success_and_timeout_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            env, _ = self.evaluate(Path(directory) / 'report.json', outcomes=[(1, True, True)] * 3)
        self.assertEqual(env.progress_bars[-1].postfixes[-1], 'success=0.0% (0/3)')

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

    def test_molmo_full_chunk_horizon_and_default_are_forwarded(self):
        for horizon in (None, 1, 30):
            with self.subTest(action_horizon=horizon), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                self.evaluate(path, args_override={'policy': 'molmoact2', 'action_horizon': horizon})
                expected = 30 if horizon is None else horizon
                self.assertEqual(json.loads(path.read_text())['action_horizon'], expected)
                self.assertEqual(FakePolicy.instances[-1].options['action_horizon'], expected)

    def test_molmo_horizon_above_chunk_length_never_creates_environment(self):
        env, _ = self.evaluate(None, args_override={'policy': 'molmoact2', 'action_horizon': 31},
                               expected_error='action_horizon must be in 1..30')
        self.assertIsNone(env.made_episode_length_s)

    def test_action_repeat_defaults_to_one_and_reports_nominal_replan_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            self.evaluate(path)
            report = json.loads(path.read_text())
        self.assertEqual(FakePolicy.instances[-1].options['action_repeat'], 1)
        self.assertEqual(report['action_repeat'], 1)
        self.assertAlmostEqual(report['replan_interval_s'], 16/30)

    def test_molmo_repeat_five_reports_five_second_replan_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'policy': 'molmoact2', 'action_horizon': 30,
                                                      'action_repeat': 5, 'prefetch_steps': 149})
            report = json.loads(path.read_text())
        self.assertEqual(FakePolicy.instances[-1].options['action_repeat'], 5)
        self.assertEqual(FakePolicy.instances[-1].options['prefetch_steps'], 149)
        self.assertEqual(report['action_repeat'], 5)
        self.assertEqual(report['replan_interval_s'], 5.)
        self.assertTrue(any('Replanning every 5.00s sim' in line for line in env.printed))

    def test_invalid_action_repeat_and_prefetch_fail_before_environment_creation(self):
        for repeat in (0, -1, True, False, 1.5):
            with self.subTest(action_repeat=repeat):
                env, _ = self.evaluate(None, args_override={'action_repeat': repeat},
                                       expected_error='action_repeat must be a positive integer')
                self.assertIsNone(env.made_episode_length_s)
        env, _ = self.evaluate(None, args_override={'policy': 'molmoact2', 'action_horizon': 30,
                                                   'action_repeat': 5, 'prefetch_steps': 150},
                               expected_error='prefetch_steps')
        self.assertIsNone(env.made_episode_length_s)

    def test_remote_decoder_preserves_every_action_of_full_thirty_step_chunk(self):
        tree = ast.parse((ROOT / 'source/sim_to_real_so101/utils/lerobot_interface.py').read_text())
        policy = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SimRemotePolicy')
        decoder = next(node for node in policy.body if isinstance(node, ast.FunctionDef) and node.name == '_decode_action_chunk')
        context = {'np': np}
        exec(compile(ast.Module(body=[decoder], type_ignores=[]), 'lerobot_interface.py', 'exec'), context)
        joint_order = HELPERS.SO101_JOINTS
        client = SimpleNamespace(_action_horizon=30, _action_repeat=1,
                                 _iface=SimpleNamespace(SO101_JOINT_ORDER=joint_order))
        chunk = np.arange(30 * 6, dtype=np.float32).reshape(30, 6)
        decoded = context['_decode_action_chunk'](client, chunk)
        self.assertEqual(len(decoded), 30)
        np.testing.assert_array_equal([[row[name] for name in joint_order] for row in decoded], chunk)

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

    def test_five_minute_attempt_records_success_and_timeout_on_both_clocks(self):
        with tempfile.TemporaryDirectory() as directory, patch('time.monotonic', side_effect=[100., 104.5, 200., 650.]):
            path = Path(directory) / 'report.json'
            env, _ = self.evaluate(path, args_override={'episode_length_s': 300., 'num_episodes': 2},
                                   outcomes=[(60, True, False), (9000, False, True)])
            report = json.loads(path.read_text())
        self.assertEqual(env.made_episode_length_s, 300.)
        success, timeout = report['episodes']
        self.assertEqual(success['time_to_success_s'], 2.)
        self.assertEqual(success['elapsed_sim_s'], 2.)
        self.assertEqual(success['wall_time_to_success_s'], 4.5)
        self.assertEqual(success['wall_time_s'], 4.5)
        self.assertEqual(timeout['elapsed_sim_s'], 300.)
        self.assertEqual(timeout['wall_time_s'], 450.)
        self.assertIsNone(timeout['time_to_success_s'])
        self.assertIsNone(timeout['wall_time_to_success_s'])
        episode_logs = [line for line in env.printed if line.startswith('[EPISODE')]
        self.assertIn('time_to_success=2.00s sim (4.50s wall)', episode_logs[0])
        self.assertIn('timeout_elapsed=300.00s sim (450.00s wall)', episode_logs[1])
        self.assertEqual(report['overall']['median_time_to_success_s'], 2.)
        self.assertIn('[RESULT] Median time to success: 2.00s sim', env.printed)

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
        self.assertIn('[RESULT] incomplete: 0/3 episodes; success 0/0 (n/a)', env.printed)
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
        self.assertEqual(args.action_repeat, 1)
        self.assertFalse(args.molmo_overhead_camera)
        self.assertEqual(args.random_fraction, 0.)
        self.assertEqual(args.robot_start, 'recorded')
        for horizon in ('1', '16'):
            args = self.parse_cli(['--action_horizon', horizon, '--episode_length_s', '30'])
            self.assertEqual(args.action_horizon, int(horizon))
            self.assertEqual(args.episode_length_s, 30.)

    def test_molmo_cli_defaults_to_thirty_and_accepts_full_chunk(self):
        args = self.parse_cli(['--policy', 'molmoact2'])
        self.assertEqual(args.action_horizon, 30)
        for horizon in ('1', '30'):
            args = self.parse_cli(['--policy', 'molmoact2', '--action_horizon', horizon])
            self.assertEqual(args.action_horizon, int(horizon))

    def test_molmo_cli_above_chunk_length_fails_before_simulator_launch(self):
        with self.assertRaises(SystemExit) as error:
            self.parse_cli(['--policy', 'molmoact2', '--action_horizon', '31'])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.Launcher.launches, 0)

    def test_molmo_overhead_cli_accepts_only_molmo_pick_place(self):
        for task in ('Lerobot-So101-Teleop-Pick-Place-Eval', 'Lerobot-So101-Teleop-Pick-Place-DR-Eval'):
            args = self.parse_cli(['--policy', 'molmoact2', '--task', task, '--molmo-overhead-camera'])
            self.assertTrue(args.molmo_overhead_camera)
        for arguments in (['--molmo-overhead-camera'],
                          ['--policy', 'molmoact2', '--molmo-overhead-camera'],
                          ['--task', 'Lerobot-So101-Teleop-Pick-Place-Eval', '--molmo-overhead-camera']):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                self.parse_cli(arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(self.Launcher.launches, 0)

    def test_repeat_cli_prefetch_uses_effective_physical_chunk_length(self):
        args = self.parse_cli(['--policy', 'molmoact2', '--action_repeat', '5', '--prefetch_steps', '149'])
        self.assertEqual(args.action_horizon, 30)
        self.assertEqual(args.action_repeat, 5)
        self.assertEqual(args.prefetch_steps, 149)
        with self.assertRaises(SystemExit) as error:
            self.parse_cli(['--policy', 'molmoact2', '--action_repeat', '5', '--prefetch_steps', '150'])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.Launcher.launches, 0)

    def test_invalid_repeat_cli_fails_before_simulator_launch(self):
        for repeat in ('0', '-1', 'true', 'false', '1.5'):
            with self.subTest(action_repeat=repeat), self.assertRaises(SystemExit) as error:
                self.parse_cli(['--action_repeat', repeat])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(self.Launcher.launches, 0)

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
