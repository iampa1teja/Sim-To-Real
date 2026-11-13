"""Mimic source coordinates remain the only gripper action axes."""


def test_fingers_track_through_resets_and_fifty_action_steps(imported_parallel_arm):
    check=next(row for row in imported_parallel_arm['report']['checks'] if row['name']=='mimic_tracking')
    assert check['passed'],check
    assert check['reset_count']>=2
    assert check['action_steps']>=100
    assert check['max_error_si']<=.001
    assert check['runtime'] in ('constraint','manual')
