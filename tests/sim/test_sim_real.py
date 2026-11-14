"""Paired comparison guards and the trusted local server codec."""
import io
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sim.eval.sim_real.trials import limited_target


def _local_tcp():
    """Load policies/molmoact2/local_tcp.py by path; the policies folder is not an importable package."""
    import importlib.util
    import sys
    from core.policy_client.discovery import get_manifest

    module_name = '_s2r_policy_molmoact2_local_tcp'
    if module_name not in sys.modules:
        definition = importlib.util.spec_from_file_location(
            module_name, Path(get_manifest('molmoact2').path) / 'local_tcp.py')
        module = importlib.util.module_from_spec(definition)
        sys.modules[module_name] = module
        try:
            definition.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name]


def test_first_action_guard_precedes_step_limit():
    with pytest.raises(ValueError, match='First policy target'):
        limited_target({'a.pos': 50}, [0], ['a.pos'], [(-100, 100)],
                       first=True, start_limit=20, step_limit=10)


def test_each_side_limits_relative_to_its_own_observed_state():
    action = {'a.pos': 90, 'b.pos': -200}
    kwargs = dict(keys=list(action), ranges=[(-100, 100), (-100, 100)],
                  first=False, start_limit=20, step_limit=10)
    np.testing.assert_equal(limited_target(action, [0, 0], **kwargs), [10, -10])
    np.testing.assert_equal(limited_target(action, [60, 40], **kwargs), [70, 30])
    with pytest.raises(ValueError, match='finite'):
        limited_target(action, [float('nan'), 0], **kwargs)


def test_local_server_numpy_response_and_disallowed_classes():
    _ArrayUnpickler = _local_tcp()._ArrayUnpickler
    payload = {'chunk': np.ones((30, 6), dtype=np.float32)}
    result = _ArrayUnpickler(io.BytesIO(pickle.dumps(payload))).load()
    np.testing.assert_equal(result['chunk'], payload['chunk'])
    with pytest.raises(pickle.UnpicklingError):
        _ArrayUnpickler(io.BytesIO(pickle.dumps(SimpleNamespace()))).load()


def test_local_server_codec_does_not_double_transform(monkeypatch):
    from utils.arms.spec import load_arm_spec
    from utils.arms.calibration import resolve_calibration
    from core.policy_client.contract import ArmInfo
    LocalLeRobotPolicy = _local_tcp().LocalLeRobotPolicy
    import struct

    arm = load_arm_spec('so101')[0]
    selection = resolve_calibration(arm, unit='rpd_follower')
    policy = LocalLeRobotPolicy(ArmInfo.from_arm(arm, selection.mapping))
    state = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.float32)
    class Socket:
        def sendall(self, payload):
            request = pickle.loads(payload[4:])
            raw = request['raw_obs']
            degrees = np.asarray([raw[key] for key in policy.keys], dtype=np.float32)
            # A server returning the input pose after its own frame round-trip.
            response = pickle.dumps({'chunk': degrees[None, :]})
            self.response = io.BytesIO(struct.pack('>I', len(response)) + response)
        def recv(self, count):
            return self.response.read(count)
    policy.socket = Socket()
    images = {key: np.zeros((16, 16, 3), np.uint8) for key in ('room', 'wrist')}
    action = policy.get_raw_action(state, images)
    np.testing.assert_allclose(list(action.values()), state, atol=1e-5)


def test_video_preserves_wall_time_when_capture_drops_frames(tmp_path):
    cv2 = pytest.importorskip('cv2')
    from sim.eval.sim_real.trials import TrialVideo
    path = tmp_path / 'trial.mp4'
    images = {key: np.zeros((32, 32, 3), dtype=np.uint8) for key in ('room', 'wrist')}
    writer = TrialVideo(path, 10)
    writer.write(images, 'REAL', 0)
    writer.write(images, 'REAL', .5)
    writer.close(1.0)
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened()
    assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    assert reader.get(cv2.CAP_PROP_FPS) == 10
    reader.release()


def test_trial_video_keeps_room_panel_when_wrist_camera_is_disabled(tmp_path):
    cv2 = pytest.importorskip('cv2')
    from sim.eval.sim_real.trials import TrialVideo
    path = tmp_path / 'room-only.mp4'
    room = np.zeros((32, 32, 3), dtype=np.uint8)
    room[..., 1] = 200
    writer = TrialVideo(path, 10)
    writer.write({'room': room}, 'REAL', 0)
    writer.close(.1)
    reader = cv2.VideoCapture(str(path))
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape == (480, 1280, 3)
    assert frame[100:200, 100:200, 1].mean() > 180
    assert frame[300:400, 800:900].max() < 5
    assert frame[210:250, 740:1150].max() > 100  # Disabled-camera label is visible.


def test_leader_positions_before_start_and_is_not_read_when_control_hands_off():
    from sim.eval.sim_real.trials import position_with_leader
    commands = []
    reads = []
    def read():
        reads.append(True)
        return {'a.pos': 50, 'b.pos': -50}
    leader = SimpleNamespace(get_action=read)
    follower = SimpleNamespace(send_action=commands.append)
    kwargs = dict(leader=leader, follower=follower, measured=[0, 0], keys=['a.pos', 'b.pos'],
                  ranges=[(-100, 100)] * 2, step_limit=10, motion=True)
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=True), [10, -10])
    assert commands == [{'a.pos': 10, 'b.pos': -10}]
    # Policy takes over at Start; Stop and completion also disable operator control.
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=False), [0, 0])
    assert len(commands) == len(reads) == 1
    kwargs['motion'] = False
    np.testing.assert_equal(position_with_leader(**kwargs, enabled=True), [10, -10])
    assert len(commands) == 1  # Shadow still lets the sim follow, without physical writes.


def test_gui_can_stop_leader_positioning_before_a_trial():
    from sim.recording.web_gui import RecordingWebGui
    gui = RecordingWebGui([], port=0)
    try:
        gui.set_state(False, False, 'Positioning', positioning=True)
        assert gui._command('stop')
        assert gui.consume_requests()['stop']
        gui.set_state(False, False, 'Holding', positioning=False)
        assert not gui._command('stop')
    finally:
        gui.destroy()


def test_failed_policy_pair_closes_connections_and_can_retry():
    from concurrent.futures import ThreadPoolExecutor
    from sim.eval.sim_real.trials import connect_trial_policies
    events = []
    attempts = []
    class Policy:
        def __init__(self):
            self.index = len(attempts)
            attempts.append(self)
        def connect(self):
            if self.index == 1:
                raise ConnectionRefusedError('Server is warming up')
        def close(self):
            events.append(('close', self.index))
        def reset(self):
            events.append(('reset', self.index))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ConnectionRefusedError):
            connect_trial_policies(Policy, executor)
        assert events == [('close', 0), ('close', 1)]
        result = connect_trial_policies(Policy, executor)
        assert len(result) == 2
        assert events[-2:] == [('reset', 2), ('reset', 3)]


def test_no_object_is_opt_in_and_default_gui_requires_spawn():
    from sim.eval.sim_real.trials import parse_args
    from sim.recording.web_gui import RecordingWebGui
    default = parse_args([])
    assert default.no_object is False
    assert parse_args(['--no-object', '--shadow']).no_object is True
    gui = RecordingWebGui([], port=0)
    try:
        gui.set_state(False, default.no_object, 'Positioning')
        assert not gui._command('start')
        assert gui._command('spawn')
        assert gui.consume_requests()['spawn']
        gui.set_state(False, True, 'Spawned')
        assert gui._command('start')
        gui.consume_requests()
        # Opt-in reaches the same ready state without a Spawn command.
        gui.set_state(False, parse_args(['--no-object']).no_object, 'No object')
        assert gui._command('start')
    finally:
        gui.destroy()


@pytest.mark.parametrize('no_object,spawn,connects', [(False, False, False), (False, True, True), (True, False, True)])
def test_trial_loop_requires_spawn_unless_opted_out(no_object, spawn, connects, tmp_path, monkeypatch):
    """Exercise actual Start gating with fake hardware and no simulator launch."""
    import sys
    import torch
    import gymnasium
    from utils.arms.calibration import resolve_calibration
    from utils.arms.mapping import JointMapping
    from utils.arms.spec import load_arm_spec
    from sim.eval.sim_real.trials import parse_args, run
    from core.rollout import real_rollout
    from sim.teleop import agent, loop
    from sim.datagen import spawner
    from core.rollout import paired_gui
    import threading

    arm = load_arm_spec('so101')[0]
    selection = resolve_calibration(arm, unit='rpd_follower', environ={})
    raw = np.array([0, 0, 0, 0, 0, 60], dtype=np.float32)
    mapping = JointMapping.from_arm(arm, selection)
    observation = {'policy': {'joint_pos_obs': mapping.to_sim(raw)[None, :]}, 'visual': {}}
    noop = lambda *a, **k: None
    robot = SimpleNamespace(find_joints=lambda *a, **k: (list(range(6)),), find_bodies=lambda *a: ([0],),
                            write_joint_state_to_sim_index=noop, set_joint_position_target_index=noop)
    env = SimpleNamespace(scene={'robot': robot}, device='cpu', reset=lambda: (observation, {}),
                          step=lambda _: (observation, 0, False, False, {}), close=noop)
    env.unwrapped = env
    monkeypatch.setattr(gymnasium, 'make', lambda *a, **k: env)
    fake_envs = SimpleNamespace(TASK=SimpleNamespace(spec=SimpleNamespace(tracked_objects=())))
    import sim
    monkeypatch.setitem(sys.modules, 'sim.workspace.envs', fake_envs)
    monkeypatch.setitem(sys.modules, 'sim.workspace.envs.mdp._common', SimpleNamespace(as_torch=torch.as_tensor))
    monkeypatch.setattr(sim, 'envs', fake_envs, raising=False)
    monkeypatch.setattr(agent, '_env_id', lambda _: 'fake')
    monkeypatch.setattr(agent, '_make_env_cfg', lambda *a: None)
    monkeypatch.setattr(agent, '_sim_cameras', lambda _: [])
    monkeypatch.setattr(real_rollout, 'camera_specs', lambda _: {})
    monkeypatch.setattr(real_rollout, 'connect_follower', lambda _: SimpleNamespace(
        get_observation=lambda: dict(zip(arm.feature_names, raw)), disconnect=noop))
    attempted = []
    def policy_factory(_):
        attempted.append(True)
        raise ConnectionRefusedError('Test reached policy connection, before motion')
    monkeypatch.setattr(real_rollout, 'make_policy', policy_factory)
    spawned = []
    monkeypatch.setattr(spawner, 'ObjectSpawner', lambda *a, **k: SimpleNamespace(
        spawn=lambda: spawned.append(True), hold=noop, close=noop))
    body = dict(policy='molmoact2', prompt='Move the cube', host='localhost', port=5556, seconds=.1)
    requests = iter([{'connect': True}, {'spawn': True} if spawn else {}, {'run': body}, {'exit': True}])
    monkeypatch.setattr(paired_gui, 'PairedEvaluationGui', lambda *a, **k: SimpleNamespace(
        url='fake', publish=noop, update_images=noop, destroy=noop, clear_real_frames=noop,
        record_report=noop, cancel=threading.Event(), cancelled=False, motion_lock=threading.Lock(),
        consume_requests=lambda: next(requests)))
    monkeypatch.setattr(loop, 'gui_frames', lambda *a: {})
    args = parse_args(['--shadow', '--no-leader', '--output', str(tmp_path)] + (['--no-object'] if no_object else []))
    args.arm_spec, args.calibration_selection = arm, selection
    run(args, SimpleNamespace(is_running=lambda: True))
    assert bool(attempted) is connects
    assert bool(spawned) is spawn


@pytest.mark.parametrize('cancel_inference,operator_end', [(False,False),(True,False),(False,True)])
def test_both_trials_use_the_submitted_prompt_policy_endpoint_and_duration(tmp_path, monkeypatch, cancel_inference,operator_end):
    """Drive a complete real -> sim comparison with injected devices and policies."""
    import sys
    import threading
    import torch
    import gymnasium
    from utils.arms.spec import load_arm_spec
    from utils.arms.calibration import resolve_calibration
    from utils.arms.mapping import JointMapping
    from core.rollout import real_rollout, paired_gui
    from sim.eval import sim_real
    from sim.teleop import agent, loop
    from sim.datagen import spawner
    import sim

    arm=load_arm_spec('so101')[0]
    selection=resolve_calibration(arm,unit='rpd_follower',environ={})
    raw=np.array([0,0,0,0,0,60],np.float32)
    mapping=JointMapping.from_arm(arm,selection)
    observation={'policy':{'joint_pos_obs':mapping.to_sim(raw)[None,:]},'visual':{}}
    noop=lambda *a,**k:None
    robot=SimpleNamespace(find_joints=lambda *a,**k:(list(range(6)),),find_bodies=lambda *a:([0],),
                          write_joint_state_to_sim_index=noop,set_joint_position_target_index=noop)
    class Scene(dict):
        def update(self,dt):pass
    env=SimpleNamespace(scene=Scene(robot=robot),device='cpu',reset=lambda:(observation,{}),
        step=lambda _:(observation,0,False,False,{}),close=noop,
        sim=SimpleNamespace(forward=noop,render=noop),observation_manager=SimpleNamespace(compute=lambda:observation))
    env.unwrapped=env
    monkeypatch.setattr(gymnasium,'make',lambda *a,**k:env)
    fake=SimpleNamespace(TASK=SimpleNamespace(spec=SimpleNamespace(tracked_objects=())))
    monkeypatch.setitem(sys.modules,'sim.workspace.envs',fake);monkeypatch.setattr(sim,'envs',fake,raising=False)
    monkeypatch.setitem(sys.modules,'sim.workspace.envs.mdp._common',SimpleNamespace(as_torch=torch.as_tensor))
    monkeypatch.setattr(agent,'_env_id',lambda _:'fake')
    monkeypatch.setattr(agent,'_make_env_cfg',lambda *a:None)
    monkeypatch.setattr(agent,'_sim_cameras',lambda _:[])
    monkeypatch.setattr(real_rollout,'camera_specs',lambda _: {})
    writes=[]
    monkeypatch.setattr(real_rollout,'connect_follower',lambda _:SimpleNamespace(
        get_observation=lambda:dict(zip(arm.feature_names,raw)),send_action=writes.append,disconnect=noop))
    monkeypatch.setattr(spawner,'ObjectSpawner',lambda *a,**k:SimpleNamespace(spawn=noop,hold=noop,close=noop))
    monkeypatch.setattr(loop,'gui_frames',lambda *a:{})
    monkeypatch.setattr(sim_real,'TrialVideo',lambda *a:SimpleNamespace(write=noop,close=noop))
    selected=[];threads=[]
    labels={};ui_state={}
    entered,release=threading.Event(),threading.Event()
    sim_entered=threading.Event()
    def policy_factory(args):
        side_index=len(selected)
        selected.append(args)
        def call(*a):threads.append(threading.get_ident())
        def action(*a):
            call()
            if cancel_inference or (operator_end and side_index==0):
                entered.set()
                assert release.wait(3)
                if operator_end:
                    return {key:float(value)+1000 for key,value in zip(arm.feature_names,raw)}
            if side_index==1:sim_entered.set()
            return dict(zip(arm.feature_names,raw))
        return SimpleNamespace(connect=call,reset=call,close=call,get_raw_action=action)
    monkeypatch.setattr(real_rollout,'make_policy',policy_factory)
    duration=.5 if operator_end else .1
    body=dict(policy='groot_n17',prompt='Place the RED cube inside the white box',host='injected-host',port=5999,seconds=duration)
    requests=iter([{'connect':True},{'run':body}])
    reports=[]
    cancel=threading.Event()
    def consume():
        result=next(requests,None)
        if result is not None:return result
        if cancel_inference and entered.is_set() and not cancel.is_set():
            cancel.set();release.set();return {'stop':True}
        if operator_end:
            from core.rollout.paired_gui import operator_stage_label
            from core.tasks.loader import load_task
            import time
            side=ui_state.get('phase')
            if side=='real' and entered.is_set() and 'real' not in labels:
                labels['real']=dict(operator_stage_label(load_task('pick_place_cube').stages,'lifted'),
                                    side='real',at=time.monotonic(),confirmed_at='test')
                release.set();return {'end_trial':True}
            if side=='sim' and sim_entered.is_set() and 'sim' not in labels:
                labels['sim']=dict(operator_stage_label(load_task('pick_place_cube').stages,'success'),
                                   side='sim',at=time.monotonic(),confirmed_at='test')
                return {'end_trial':True}
        return {}
    def publish(phase,*a,**k):ui_state['phase']=phase
    class Gui:
        url='fake'
        motion_lock=threading.Lock()
        update_images=clear_real_frames=destroy=staticmethod(noop)
        trial_end_for=staticmethod(labels.get)
        record_report=staticmethod(reports.append)
        consume_requests=staticmethod(consume)
        @property
        def cancelled(self):return cancel.is_set()
    gui=Gui();gui.cancel=cancel;gui.publish=publish
    monkeypatch.setattr(paired_gui,'PairedEvaluationGui',lambda *a,**k:gui)
    args=sim_real.parse_args(([] if operator_end else ['--shadow'])+['--no-leader','--no-object','--output',str(tmp_path)])
    args.arm_spec,args.calibration_selection=arm,selection
    ticks=[0]
    def running():ticks[0]+=1;return not reports and ticks[0]<60
    sim_real.run(args,SimpleNamespace(is_running=running))
    assert len(reports)==1 and reports[0]['status']==('stopped' if cancel_inference else 'completed')
    assert [row['side'] for row in reports[0]['trials']]==(['real'] if cancel_inference else ['real','sim'])
    if cancel_inference:
        assert sum(row['steps'] for row in reports[0]['trials'])==0
    if operator_end:
        real,simulation=reports[0]['trials']
        assert real['status']=='operator_end' and real['stage']=='lifted' and real['outcome']=='incomplete'
        assert real['steps']==0  # The deliberately invalid late real reply was discarded.
        assert simulation['stage']=='success' and simulation['outcome_source']=='operator'
        assert simulation['wall_seconds']<duration and real['wall_seconds']<duration
        assert reports[0]['stage_comparison']==dict(real='lifted',sim='success',same_stage=False,source='operator')
    assert len(selected)==2
    for args in selected:
        assert (args.policy,args.lang_instruction,args.policy_host,args.policy_port,args.episode_seconds)==(
            'groot_n17',body['prompt'],'injected-host',5999,duration)
    assert len(set(threads))==1 and threading.get_ident() not in threads
    if operator_end:
        assert writes==[dict(zip(arm.feature_names,raw))]  # Only the explicit measured-pose hold, never the late policy action.
    else:
        assert not writes  # Shadow evaluation never issues physical commands.
