"""DR replay YAML uses declared parser options and keeps CLI flags authoritative."""
import pytest

from sim.datagen import dr_replay


def test_dr_replay_config_flags_and_source_validation(tmp_path, monkeypatch):
    def launcher_arguments(parser):
        parser.add_argument('--headless', action='store_true')
        return parser

    monkeypatch.setattr(dr_replay, 'add_launch_arguments', launcher_arguments)
    path = tmp_path / 'datagen.yaml'
    path.write_text('''arm: so101
scene: blank_table
task: pick_place_cube
variant: teleop
source: dr_replay
src_root: original
src_repo_id: local/original
dst_root: generated
dst_repo_id: local/generated
seed: 101
include_originals: true
headless: true
''')
    args = dr_replay.parse_args(['--config_path', str(path), '--seed', '17', '--dst_root', 'override'])
    assert (args.arm, args.scene, args.task, args.variant) == ('so101', 'blank_table', 'pick_place_cube', 'teleop')
    assert args.src_root == 'original' and args.dst_root == 'override'
    assert args.seed == 17 and args.include_originals is True and args.headless is True
    assert args.source == 'dr_replay'
    path.write_text(path.read_text().replace('source: dr_replay', 'source: mimic'))
    with pytest.raises(SystemExit):
        dr_replay.parse_args(['--config_path', str(path)])


def test_scripted_imported_arm_controls_are_explicit_and_validated(tmp_path, monkeypatch):
    from sim.datagen import scripted

    monkeypatch.setattr(scripted, 'add_generation_launch_arguments', lambda parser: parser)
    arguments = ['--destination', str(tmp_path / 'episodes'), '--ik_orientation_weight', '.2',
                 '--solver_njmax', '4096', '--solver_nconmax', '1024']
    args = scripted.parse_args(arguments)
    assert (args.ik_orientation_weight, args.solver_njmax, args.solver_nconmax) == (.2, 4096, 1024)
    for invalid in (['--solver_njmax', '0'], ['--ik_orientation_weight', 'nan'],
                    ['--ik_orientation_tolerance_rad', '-.1'], ['--ik_orientation_tolerance_rad', 'nan']):
        with pytest.raises(SystemExit):
            scripted.parse_args(arguments + invalid)


def test_start_variation_is_seeded_and_does_not_mutate_task_or_fingers(tmp_path, monkeypatch):
    import numpy as np
    from core.arms.spec import load_arm_spec
    from core.tasks.loader import load_task
    from sim.datagen import scripted

    monkeypatch.setattr(scripted, 'add_generation_launch_arguments', lambda parser: parser)
    task = load_task('pick_place_cube')
    original = dict(task.spec.start)
    options = {'xy_min': [-.20, -.18], 'xy_max': [-.08, -.06]}
    modified = scripted.task_with_start_options(task, options)
    assert task.spec.start == original
    assert modified.spec.start['xy_min'] == [-.20, -.18]
    for seed in range(5):
        xy = modified.sample_start(np.random.default_rng(seed))['blue_cube'].position[:2]
        assert np.all(np.asarray(xy) >= options['xy_min']) and np.all(np.asarray(xy) <= options['xy_max'])
    arm, _ = load_arm_spec('so101')
    positions = np.array([[j.initial for j in arm.action_joints]])
    varied = scripted.jitter_start_positions(arm, positions, .04, 123)
    assert np.array_equal(varied, scripted.jitter_start_positions(arm, positions, .04, 123))
    assert not np.array_equal(varied[:, :-1], positions[:, :-1])
    assert varied[0, -1] == positions[0, -1]
    for invalid in ('{"xy_min":[0]}', '{"yaw_deg":[0,"bad"]}'):
        with pytest.raises((SystemExit, TypeError)):
            scripted.parse_args(['--destination', str(tmp_path), '--start_options', invalid])
    with pytest.raises(ValueError, match='xy_min'):
        scripted.task_with_start_options(task, {'xy_min': [.3, .2]})


def test_carried_object_feedback_corrects_slip_and_freezes_after_release():
    import numpy as np
    from core.tasks.base import Pose, StepState
    from sim.datagen.scripted import carried_object_offset

    tcp = Pose((.3, .1, .2))
    state = StepState({'cube': Pose((.3, .1, .19))}, tcp, 0., {'holding': True})
    offset = carried_object_offset(state, tcp, 'cube')
    np.testing.assert_allclose(offset, [0., 0., .01])
    state.objects['cube'] = Pose((.294, .102, .19))
    corrected = carried_object_offset(state, tcp, 'cube', offset)
    goal = np.array([.4, .2, .05])
    np.testing.assert_allclose(goal + corrected, [.406, .198, .06])
    state.signals['holding'] = False
    state.objects['cube'] = Pose(tuple(goal))
    retreat = Pose((.406, .198, .15))
    np.testing.assert_allclose(carried_object_offset(state, retreat, 'cube', corrected), corrected)
