"""Smoke coverage and real stopping semantics, including Isaac term ordering."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from sim_to_real_so101.utils.eval_smoke import MilestoneDeadline, one_per_zone, install_milestone_deadline
from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts


def test_zone_coverage_preserves_original_starts_and_seeds():
    path = ROOT / 'eval_sets/pick_place_v1_seed1984.json'
    full = EvalSetStarts(path, seed=1984)
    selected = EvalSetStarts(path, seed=1984, selection='one_per_zone')
    assert len(selected.schedule) == 9
    assert {r['zone'] for r in selected.schedule} == {r['zone'] for r in full.schedule}
    for row in selected.schedule:
        assert row == next(r for r in full.schedule if r['id'] == row['id'])
    assert [r['id'] for r in selected.schedule] == [
        'id-000', 'id-001', 'id-002', 'id-003', 'id-004', 'id-006', 'id-011', 'id-013', 'ood-000']
    again = EvalSetStarts(path, seed=1984, selection='one_per_zone')
    assert selected.schedule == again.schedule
    other = EvalSetStarts(path, seed=9182, selection='one_per_zone')
    assert [r['id'] for r in selected.schedule] == [r['id'] for r in other.schedule]
    assert all(a['policy_seed'] != b['policy_seed'] for a, b in zip(selected.schedule, other.schedule))


def test_zone_selection_rejects_missing_names_and_empty_coverage():
    with pytest.raises(ValueError):
        one_per_zone([])
    with pytest.raises(ValueError):
        one_per_zone([{'id': 'a'}])


def test_no_reach_stops_at_sixty_seconds_and_cannot_be_revived():
    budget = MilestoneDeadline()
    assert not budget.observe(59.999, {})
    assert budget.observe(60, {})
    assert budget.summary()['stop_reason'] == 'reach_timeout'
    assert budget.observe(61, dict(reached=True, grasped=True), success=True)
    assert budget.deadline_s == 60


def test_new_milestones_at_boundary_are_accepted_with_only_thirty_more_seconds():
    budget = MilestoneDeadline()
    stages = {}
    for name, when in zip(('reached', 'grasped', 'lifted', 'over_box', 'placed'), (60, 90, 120, 150, 180)):
        stages[name] = True
        assert not budget.observe(when, stages)
        assert budget.deadline_s == when + 30
    assert not budget.observe(180 + 1 / 30, stages, success=True)
    assert budget.maximum_duration_s == 210


def test_repeated_reaching_or_grasping_never_extends_time():
    budget = MilestoneDeadline()
    assert not budget.observe(10, {'reached': True})
    assert not budget.observe(20, {'reached': True, 'grasped': True})
    assert budget.deadline_s == 50
    for when in (25, 30, 40, 49):
        assert not budget.observe(when, {'reached': True, 'grasped': True})
        assert budget.deadline_s == 50
    assert budget.observe(50, {'reached': True, 'grasped': True})
    assert budget.stop_reason == 'progress_timeout'


def test_late_new_progress_does_not_extend_an_expired_deadline():
    budget = MilestoneDeadline()
    assert budget.observe(60 + 1 / 30, {'reached': True})
    assert budget.stop_reason == 'reach_timeout'
    assert budget.deadline_s == 60


def test_multiple_milestones_on_one_frame_do_not_stack_extensions():
    budget = MilestoneDeadline()
    assert not budget.observe(10, dict(reached=True, grasped=True, lifted=True))
    assert budget.deadline_s == 40
    budget.reset()
    assert budget.deadline_s == 60 and budget.times == {} and budget.highest == -1


def test_knocking_cube_up_without_grasp_cannot_buy_an_extension():
    budget = MilestoneDeadline()
    assert not budget.observe(10, dict(reached=True))
    assert budget.deadline_s == 40
    assert not budget.observe(30, dict(reached=True, lifted=True))
    assert budget.deadline_s == 40 and budget.highest == 0
    assert budget.observe(40, dict(reached=True, lifted=True))


@pytest.mark.parametrize('bad', [True, 0, -1, float('inf'), float('nan')])
def test_invalid_budgets_are_rejected(bad):
    with pytest.raises(ValueError):
        MilestoneDeadline(bad, 30)


def fake_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(full_like=np.full_like))
    calls = []
    raw = SimpleNamespace(num_envs=1, common_step_counter=0, episode_length_buf=np.array([0]), step_dt=1.,
                          _benchmark_trace=SimpleNamespace(stages={}), next_stages={}, next_success=False)

    def success(env):
        calls.append(env.common_step_counter)
        env._benchmark_trace.stages.update(env.next_stages)
        return np.array([env.next_success], dtype=bool)

    terms = dict(success=SimpleNamespace(func=success, params={}),
                 time_out=SimpleNamespace(func=lambda env: np.array([env.episode_length_buf[0] >= 210]), params={}))
    raw.termination_manager = SimpleNamespace(get_term_cfg=terms.__getitem__, set_term_cfg=terms.__setitem__)
    return raw, terms, calls


@pytest.mark.parametrize('order', [('time_out', 'success'), ('success', 'time_out')])
def test_termination_order_uses_current_frame_once_and_preserves_success(monkeypatch, order):
    env, terms, calls = fake_environment(monkeypatch)
    budget = install_milestone_deadline(env, 60, 30)
    env.common_step_counter = 1
    env.episode_length_buf[0] = 60
    env.next_stages = {'reached': True}
    results = {name: terms[name].func(env) for name in order}
    assert not results['time_out'][0] and not results['success'][0]
    assert calls == [1] and budget.deadline_s == 90
    env.common_step_counter = 2
    env.episode_length_buf[0] = 90
    env.next_stages = {'grasped': True, 'lifted': True, 'over_box': True, 'placed': True}
    env.next_success = True
    results = {name: terms[name].func(env) for name in order}
    assert results['success'][0] and not results['time_out'][0]
    assert calls == [1, 2]


def test_timeout_is_a_truncation_and_next_episode_gets_a_fresh_budget(monkeypatch):
    env, terms, calls = fake_environment(monkeypatch)
    budget = install_milestone_deadline(env, 60, 30)
    env.common_step_counter = 1
    env.episode_length_buf[0] = 60
    assert terms['time_out'].func(env)[0]
    assert not terms['success'].func(env)[0]
    assert budget.stop_reason == 'reach_timeout' and calls == [1]
    env.common_step_counter = 2
    env.episode_length_buf[0] = 1
    env._benchmark_trace.stages = {}
    assert not terms['time_out'].func(env)[0]
    assert budget.stop_reason is None and budget.deadline_s == 60
