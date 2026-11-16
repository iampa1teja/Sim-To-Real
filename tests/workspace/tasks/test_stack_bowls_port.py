"""The bowls port matches the frozen signals, rim expert, core rules and cfg plan."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from scipy.spatial.transform import Rotation

from core.tasks.loader import load_task
from core.tasks.base import StepState
from sim.actions.action import ExpertContext
from utils.arms.arms import load_arm_spec
from utils.geometry import Pose
from task_oracle import isolated_modules, legacy_task as make_legacy_task, same, plan

TASK = 'stack_bowls'
OBJECTS = ('yellow_bowl', 'red_bowl', 'blue_bowl')


@pytest.fixture
def legacy_task():
    return make_legacy_task(TASK, 'StackBowlsTask')


@pytest.fixture
def legacy(legacy_task):
    with isolated_modules(TASK, legacy_task) as modules:
        yield modules


def environment(n=8):
    scene = {}
    for name in OBJECTS:
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[0., 0., 0., 1.]] * n),
            root_vel_w=torch.zeros(n, 6)))
    scene['robot'] = SimpleNamespace(cfg=SimpleNamespace(prim_path='/World/envs/env_.*/Robot'),
        find_bodies=lambda name: ([1], [name]), data=SimpleNamespace(
            body_link_pos_w=torch.zeros(n, 2, 3), body_link_quat_w=torch.tensor([[[0., 0., 0., 1.]] * 2] * n)))
    scene['contact_grasp'] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n, 2, 3, 3)),
                                             reset=lambda *args, **kwargs: None)
    arm = SimpleNamespace(gripper=SimpleNamespace(kind='jaw', tip_mesh='must_not_be_read'),
                          ik=SimpleNamespace(position_offset_m=(.01, -.02, -.0981)), links=SimpleNamespace(ee='gripper'))
    return SimpleNamespace(scene=scene, cfg=SimpleNamespace(arm_spec=arm), num_envs=n, device='cpu',
                           common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long))


def rest(env, ids):
    for index, name in enumerate(OBJECTS):
        env.scene[name].data.root_pos_w[ids] = torch.tensor([.15 + .18 * index, -.1, .785])
        env.scene[name].data.root_quat_w[ids] = torch.tensor([0., 0., 0., 1.])
        env.scene[name].data.root_vel_w[ids] = 0.


@pytest.mark.parametrize('seed', range(6))
def test_signals_match_legacy_with_batch_reset_semantics(legacy, seed):
    rng = np.random.default_rng(seed)
    env = environment()
    ids = torch.arange(env.num_envs)
    rest(env, ids)
    old, new = legacy.mdp, legacy.task_env
    def reset(selected):
        old.reset_tracking(env, selected)
        new.reset_task_tracking(env, selected, TASK)
    def compare(label):
        expected, actual = old.stage_signals(env), new.task_signals(env, TASK)
        assert expected.keys() == actual.keys()
        for name in expected:
            same(actual[name], expected[name], (label, name))
        return expected
    compare('before startup reset')
    reset(None if seed % 2 else ids)
    seen = dict.fromkeys(load_task(TASK).stages, 0)
    for step in range(1, 601):
        phase = step % 100
        rest(env, ids)
        force = env.scene['contact_grasp'].data.force_matrix_w
        force[:] = 0.
        for row in range(env.num_envs):
            current = phase if row == 0 else (phase + row * 11) % 100
            if current >= 10:
                env.scene['red_bowl'].data.root_pos_w[row] = torch.tensor([.15, -.1, .815 if current < 20 else .807])
                if current < 20:
                    force[row, 0, 1, 2] = .3 if row == 0 else rng.choice([.149999, .15, .150001, .4])
            if 30 <= current < 40:
                env.scene['blue_bowl'].data.root_pos_w[row, 2] = .845
                force[row, 0, 2, 2] = .3 if row == 0 else rng.choice([0., .15, .2])
            if current >= 40:
                permutation = (0, 1, 2) if row == 0 else tuple(rng.permutation(3))
                for level, index in enumerate(permutation):
                    p = torch.tensor([.15, -.1, .785 + level * .022])
                    if row and level:
                        p[0] += rng.choice([0., .008999, .009, .009001])
                        p[2] += rng.choice([0., -.010001, -.01, .013, .013001])
                    data = env.scene[OBJECTS[index]].data
                    data.root_pos_w[row] = p
                    if row:
                        angle = rng.choice([0., 10., 12., 15., 30., 31., 180.])
                        data.root_quat_w[row] = torch.tensor(Rotation.from_euler('x', angle, degrees=True).as_quat())
                        data.root_vel_w[row, 0] = rng.choice([0., .024999, .025, .025001])
                        data.root_vel_w[row, 3] = rng.choice([0., .349999, .35, .350001])
                        force[row, 0, index, 2] = rng.choice([0., .019999, .02, .020001])
        robot = env.scene['robot'].data
        robot.body_link_pos_w[:, 1] = torch.tensor([.15, -.1, 1.2])
        if phase < 10:
            robot.body_link_pos_w[0, 1] = env.scene['red_bowl'].data.root_pos_w[0] - torch.tensor(env.cfg.arm_spec.ik.position_offset_m)
        env.common_step_counter = step
        env.episode_length_buf += 1
        same(new.task_success(env, TASK), old.success(env), (step, 'termination'))
        for repeat in range(2):
            signals = compare((step, repeat))
        for key in seen:
            seen[key] += int(signals[key].sum())
        selected = ids[torch.tensor(rng.random(env.num_envs) < .02) & (ids != 0)]
        if phase == 0:
            selected = ids
        if len(selected):
            rest(env, selected)
            env.episode_length_buf[selected] = 0
            reset(selected.tolist() if step % 2 else selected)
            # The original task invalidates the whole batch even for a partial reset.
            compare((step, 'after partial reset'))
    assert all(count > 0 for count in seen.values()), seen


def expert_case(seed):
    rng = np.random.default_rng(seed)
    base = rng.uniform([-.1, -.1, .7], [.1, .1, .8])
    positions = np.array([[-.2, -.15, .03], [0., -.22, .03], [.2, -.14, .03]]) + base
    objects = {name: Pose.from_lists(p + rng.uniform(-.005, .005, 3),
        Rotation.from_euler('xyz', rng.uniform(-.1, .1, 3)).as_quat()) for name, p in zip(OBJECTS, positions)}
    ee = Pose.from_lists(base + [0., -.1, .3], Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat())
    options = {'order': list(rng.permutation(OBJECTS))}
    for key, values in {
        'grasp_radius_m': [.04, .058], 'grasp_height_m': [.008, .015], 'lift_height_m': [.06, .1],
        'nesting_pitch_m': [.02, .025], 'move_seconds': [1., 3.], 'waypoint_timeout_seconds': [2., 8.],
        'tool_tilt_degrees': [10., 25.], 'rim_side': [-1., 1.], 'release_clearance_m': [.008, .015],
        'carry_height_above_support_m': [.09, .12], 'retreat_height_m': [.03, .06],
        'approach_height_m': [.02, .05], 'gripper_open_fraction': [.25, .4], 'gripper_closed_fraction': [0., .02],
        'gripper_open_gap_m': [.03, .04], 'gripper_closed_gap_m': [.001, .002],
        'orientation_xyzw': [(0., 0., 0., 1.)], 'upright_object_feedback': [False, True],
        'release_seconds': [.2], 'grasp_seconds': [.3], 'settle_seconds': [.1], 'position_tolerance_m': [.001],
    }.items():
        if rng.random() < .35:
            options[key] = values[int(rng.integers(len(values)))]
    normal = (float(rng.uniform(-.1, .1)), 0., 1.)
    return ExpertContext(StepState(objects, ee, 0., {'holding': True}), ee, normal, options,
                         Pose.from_lists(base), load_arm_spec('so101' if seed % 2 else 'sp7')[0])


@pytest.mark.parametrize('seed', range(50))
def test_expert_waypoints_and_feedback_match_legacy(legacy, legacy_task, seed):
    ctx = expert_case(seed)
    old, new = legacy.expert.create(legacy_task), load_task(TASK).expert()
    expected, actual = old.plan(ctx), new.plan(ctx)
    assert len(expected) == 18 and actual == expected
    # A moved support and source are sampled again at approach, then frozen through this placement.
    shifted = {name: replace(pose, position=tuple(np.asarray(pose.position) + [.005, -.003, .001]))
               for name, pose in ctx.state.objects.items()}
    current = replace(ctx, state=replace(ctx.state, objects=shifted))
    for index, point in enumerate(expected):
        a, b = old.adapt_waypoint(point, current), new.adapt_waypoint(point, current)
        assert a == b
        current.state.signals['holding'] = index % 3 != 0
        nominal = [0., 0., 0., 1.]
        a = old.control_orientation(a, current.state, current.control_pose, nominal, current, .6)
        b = new.control_orientation(b, current.state, current.control_pose, nominal, current, .6)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('changes', [
    {'options': {'order': ['red_bowl']}}, {'options': {'order': ['red_bowl'] * 3}}, {'arm_base': None},
])
def test_expert_refusals_match(legacy, legacy_task, changes):
    ctx = replace(expert_case(1), **changes)
    expected = plan(legacy.expert.create(legacy_task), ctx)
    assert isinstance(expected[0], type) and plan(load_task(TASK).expert(), ctx) == expected


def test_core_flags_sampling_and_geometry_match(legacy, legacy_task):
    task = load_task(TASK)
    rng = np.random.default_rng(8)
    for _ in range(1000):
        signals = {name: bool(rng.integers(2)) for name in (*task.stages, 'holding') if rng.random() < .8}
        state = replace(expert_case(1).state, signals=signals)
        flags = {name: bool(rng.integers(2)) for name in task.stages if rng.random() < .8}
        assert task.observe(state) == legacy_task.observe(state)
        for current in (state, None):
            assert task.failure_mode(flags, current) == legacy_task.failure_mode(flags, current)
            assert task.progress_label(flags, current) == legacy_task.progress_label(flags, current)
    for seed in range(100):
        assert task.sample_start(np.random.default_rng(seed)) == legacy_task.sample_start(np.random.default_rng(seed))
    assert task.stages == legacy_task.stages
    geometry = __import__('importlib').import_module('legacy_' + TASK + '.geometry')
    for _ in range(100):
        p = torch.tensor(rng.uniform(-.1, .1, (8, 3, 3)), dtype=torch.float32)
        q = torch.tensor(Rotation.random(24, random_state=rng).as_quat().reshape(8, 3, 4), dtype=torch.float32)
        v = torch.tensor(rng.normal(0., .02, (8, 3, 6)), dtype=torch.float32)
        for a, b in zip(task.module.nesting_metrics(p, q, v, task.module.SETTINGS),
                        geometry.nesting_metrics(p, q, v, legacy_task.spec.raw['success'])):
            same(a, b, 'nesting_metrics')
        same(task.module.any_nested_pair(p, task.module.SETTINGS, q),
             geometry.any_nested_pair(p, legacy_task.spec.raw['success'], q), 'any_nested_pair')


@pytest.mark.parametrize('variant', ['teleop', 'eval'])
@pytest.mark.parametrize('arm_name', ['so101', 'sp7'])
def test_env_cfg_plan_matches_legacy(legacy, variant, arm_name):
    import re
    task, m = load_task(TASK), legacy.task_env
    arm = load_arm_spec(arm_name)[0]
    O, S, T = m.ObjectPlan, m.SensorPlan, m.TermPlan
    bodies = arm.gripper.friction_bodies or arm.gripper.contact_bodies
    expression = '(' + '|'.join(map(re.escape, bodies)) + ')'
    items = (S('contact_grasp', '{ENV_REGEX_NS}/Robot/' + expression,
               tuple('{ENV_REGEX_NS}/' + name for name in OBJECTS)),) + tuple(
        O(name, '{ENV_REGEX_NS}/' + name, spec.usd, spec.size, None, tuple(spec.params['color']), .7, True,
          spec.position, (0., 0., 0.), 'arm_base',
          {'use_specular_workflow': 1, 'specular_color': [0., 0., 0.], 'clearcoat': 0.})
        for name, spec in task.spec.objects.items())
    prefix = 'sim.workspace.envs.task_env:'
    events = (T('check_eval_order', 'sim.workspace.envs.mdp.starts:assert_reset_event_order', mode='startup'),
              T('reset_from_starts', 'sim.workspace.envs.mdp.starts:reset_from_starts',
                {'task': TASK, 'mode': 'random', 'reset_robot': False}, mode='reset'),
              T('reset_tracking', prefix + 'reset_task_tracking', {'task': TASK}, mode='reset'))
    terms = () if variant == 'teleop' else (
        T('time_out', 'sim.workspace.envs.mdp:time_out', time_out=True),
        T('success', prefix + 'task_success', {'task': TASK}, time_out=False))
    expected = m.TaskCfgPlan(items, events, (), terms, 4, None if variant == 'teleop' else 2400,
                             variant != 'teleop', {'njmax': 131072, 'nconmax': 32768, 'iterations': 100})
    assert m.plan_task_cfg(task.spec, variant, arm, SimpleNamespace(appearance={})) == expected
