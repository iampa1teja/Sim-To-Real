"""Paired console requests are validated/frozen without device or simulator I/O."""
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from utils.arms.spec import load_arm_spec
from core.rollout.paired_gui import PairedEvaluationGui, trial_arguments


@pytest.fixture
def gui(tmp_path):
    args = SimpleNamespace(arm_spec=load_arm_spec('so101')[0], scene='rpd_meeting_room', task='pick_place_cube',
        follower_id='rpd_follower', follower_port='injected-device', camera_external='injected-room',
        camera_gripper='injected-wrist', rename_map=json.dumps({'external':'room','gripper':'wrist'}),
        backend='s2r', no_object=False, enable_motion=False, policy='molmoact2', policy_host='localhost',
        policy_port=5556, lang_instruction='Initial instruction', episode_seconds=30., fps=30,
        policy_option=[], policy_options={'molmoact2':{'input_joint_units':'lerobot_normalized'},
                                         'groot_n17':{'test_option':'groot-only'}}, gui_host='127.0.0.1', gui_port=0)
    manifest = SimpleNamespace(language=True, action_dim=6, cameras=('room','wrist'),
                               description='Injected test policy', default_port=5556)
    result = PairedEvaluationGui(args, ['Sim: realsense_rgb','Sim: wrist_cam','Real: external','Real: gripper'],
                                tmp_path, manifests={'molmoact2':manifest,'groot_n17':manifest})
    yield result
    result.destroy()


def body(**kwargs):
    return dict(policy='groot_n17',prompt='Place the red cube in the box',host='127.0.0.1',port=5555,
                seconds=12.,client_id='controller',**kwargs)


def test_connect_and_spawn_gate_the_real_then_sim_run(gui):
    assert not gui.snapshot()['connected']
    with pytest.raises(ValueError,match='Connect'):
        gui.request_run(body())
    gui.request_connect()
    assert gui.consume_requests()=={'connect':True}
    gui.publish('ready','Connected',connected=True)
    with pytest.raises(ValueError,match='Spawn'):
        gui.request_run(body())
    gui.request_spawn(); assert gui.consume_requests()=={'spawn':True}
    gui.publish('ready','Spawned',connected=True,can_start=True)
    gui.request_run(body())
    run=gui.consume_requests()['run']
    frozen=trial_arguments(gui.args,run)
    assert (frozen.policy,frozen.lang_instruction,frozen.policy_host,frozen.policy_port,frozen.episode_seconds)==(
        'groot_n17','Place the red cube in the box','127.0.0.1',5555,12.)
    assert frozen.policy_option==['test_option=groot-only']
    assert gui.args.lang_instruction=='Initial instruction'
    assert gui.snapshot()['owner']=='controller'
    with pytest.raises(ValueError):gui.request_spawn()
    gui.request_stop(); assert gui.cancelled


@pytest.mark.parametrize('changes',[{'seconds':float('nan')},{'port':True},{'prompt':'  '},{'policy':'missing'}])
def test_invalid_runs_are_rejected_before_start(gui,changes):
    gui.publish('ready','Ready',connected=True,can_start=True)
    with pytest.raises(ValueError):gui.request_run({**body(),**changes})
    assert not gui.consume_requests()


def test_four_views_share_only_preview_frames_and_require_both_camera_inputs(gui):
    config=gui.catalog()
    assert config['mode']=='eval-sim-real' and len(config['cameras'])==4
    assert config['default_views']==['Real: external','Sim: realsense_rgb']
    pixels=np.zeros((24,32,3),np.uint8)
    gui.update_images({name:pixels for name in config['cameras']})
    for name in config['cameras']:
        assert gui.frame(name).startswith(b'\xff\xd8')
    assert not gui.consume_requests()
    gui.clear_real_frames()
    assert gui.frame('Real: external') is None and gui.frame('Sim: realsense_rgb')
    gui._names.remove('Sim: wrist_cam')
    assert all(not row['compatible'] for row in gui.catalog()['policies'])


def test_http_serves_the_same_console_and_extra_spawn_command(gui):
    assert b'Task prompt' in urlopen(gui.url).read()
    assert b'camera-select' in urlopen(gui.url+'/app.js').read()
    def post(path,payload):
        return urlopen(Request(gui.url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}))
    gui.publish('ready','Connected',connected=True)
    assert post('/api/spawn',{}).status==202
    assert gui.consume_requests()=={'spawn':True}
    gui.publish('ready','Spawned',connected=True,can_start=True)
    assert post('/api/run',body()).status==202
    assert gui.consume_requests()['run']['prompt']=='Place the red cube in the box'
    with pytest.raises(HTTPError) as error:
        post('/api/spawn',{})
    assert error.value.code==409


def test_end_trial_labels_only_the_active_side_and_preserves_global_stop(gui):
    from core.rollout.paired_gui import operator_stage_label
    assert [row['value'] for row in gui.catalog()['stages']]==['not_reached',*gui.task_stages]
    gui.publish('ready','Ready',connected=True)
    with pytest.raises(ValueError):gui.request_end_trial({'side':'real','stage':'lifted'})
    gui.publish('real','Running',connected=True)
    with pytest.raises(ValueError):gui.request_end_trial({'side':'real','stage':'unknown'})
    with pytest.raises(ValueError):gui.request_end_trial({'side':'sim','stage':'lifted'})
    gui.request_end_trial({'side':'real','stage':'lifted'})
    label=gui.trial_end_for('real')
    assert gui.consume_requests()=={'end_trial':True}
    assert label['stage']=='lifted' and label['outcome']=='incomplete'
    assert label['outcome_source']=='operator'
    assert label['stages']==dict(reached=True,grasped=True,lifted=True,over_box=False,placed=False,success=False)
    assert not gui.cancelled  # Continue to simulation; global Stop is separate.
    with pytest.raises(ValueError):gui.request_end_trial({'side':'real','stage':'success'})
    gui.publish('sim','Sim running',connected=True)
    assert gui.trial_end_for('sim') is None
    gui.request_end_trial({'side':'sim','stage':'success'})
    assert gui.trial_end_for('sim')['outcome']=='success'
    gui.request_stop()
    assert gui.cancelled
    assert not any(operator_stage_label(gui.task_stages,'not_reached')['stages'].values())


def test_end_trial_http_endpoint_captures_an_operator_label(gui):
    gui.publish('real','Real trial',connected=True)
    req=Request(gui.url+'/api/end-trial',data=json.dumps({'side':'real','stage':'placed'}).encode(),
                headers={'Content-Type':'application/json'})
    assert urlopen(req).status==202
    assert gui.trial_end_for('real')['stage']=='placed'
    assert gui.snapshot()['ending_side']=='real'
