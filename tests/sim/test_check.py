"""Scene snapshots come from a checked live environment before its stage closes."""
import json
from types import SimpleNamespace

import pytest
import torch


def check_environment(monkeypatch, *, bad_camera=False):
    import gymnasium
    import isaaclab_tasks.utils
    import sim.envs as registry
    import sim.eval.scene as scene

    calls = []
    task = SimpleNamespace(name='guide_task', spec=SimpleNamespace(default_variant='teleop', objects={'cube': object()}),
                           gym_id=lambda variant: 'S2R-guide-task-' + variant)
    arm = SimpleNamespace(name='guide_arm', usd_joint_names=('Shoulder', 'Finger'),
                          links=SimpleNamespace(base='base', ee='tool'), gripper=None, cameras={})
    camera = SimpleNamespace(width=2, height=1)
    cfg = SimpleNamespace(arm_spec=arm, scene_spec=SimpleNamespace(cameras={'room': camera}))
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1, 2))),
        find_joints=lambda names, preserve_order: ([0, 1], list(names)),
        find_bodies=lambda name: ([0 if name == 'base' else 1], [name]))
    pixels = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    if bad_camera:
        pixels.zero_()
    obs = {'visual': {'rgb_room': pixels}}

    class Environment:
        def __init__(self):
            self.unwrapped = self
            self.scene = {'robot': robot, 'cube': SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((1, 3))))}
            self.closed = False

        def reset(self):
            calls.append('reset')
            return obs, {}

        def step(self, target):
            calls.append('step')
            return obs, None, None, None, {}

        def close(self):
            self.closed = True
            calls.append('close')

    env = Environment()
    geometry = {'arm': 'guide_arm', 'task': 'guide_task', 'base_position_world': [0., 0., .75],
                'base_quaternion_wxyz': [1., 0., 0., 0.], 'table_plane_world': [0., 0., .75],
                'frames': {'table': {'position_world': [0., 0., .75]}},
                'objects': {'cube': {'position': [.1, -.1, .01], 'size': [.02, .02, .02]}}}

    def capture(raw, selected):
        assert raw is env and not env.closed and selected is task
        assert calls == ['reset', 'step', 'step', 'step']
        calls.append('capture')
        return geometry

    monkeypatch.setattr(registry, 'TASK', task)
    monkeypatch.setattr(isaaclab_tasks.utils, 'parse_env_cfg', lambda *a, **kw: cfg)
    monkeypatch.setattr(gymnasium, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(scene, 'capture_scene', capture)
    return env, calls, geometry


def test_check_exports_named_scene_only_after_checks_before_close(tmp_path, monkeypatch):
    from sim.check import check

    env, calls, geometry = check_environment(monkeypatch)
    output = tmp_path / 'nested' / 'scene.json'
    args = SimpleNamespace(scene='guide_scene', variant='teleop', device='cpu', steps=3, scene_json=output)
    result = check(args)
    assert result['passed'] and result['objects'] == ['cube']
    assert json.loads(output.read_text()) == geometry
    assert calls == ['reset', 'step', 'step', 'step', 'capture', 'close'] and env.closed


def test_failed_composition_closes_without_publishing_scene(tmp_path, monkeypatch):
    from sim.check import check

    env, calls, _ = check_environment(monkeypatch, bad_camera=True)
    output = tmp_path / 'scene.json'
    args = SimpleNamespace(scene='guide_scene', variant='teleop', device='cpu', steps=3, scene_json=output)
    with pytest.raises(ValueError, match='constant pixels'):
        check(args)
    assert not output.exists() and env.closed and 'capture' not in calls
