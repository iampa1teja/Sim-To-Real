"""Mimic source coordinates remain the only gripper action axes."""
import pytest


def test_fingers_track_through_resets_and_fifty_action_steps(imported_urdf_arm):
    if imported_urdf_arm['name']!='seven_parallel':
        pytest.skip('The jaw fixture has no mimic follower')
    check=next(row for row in imported_urdf_arm['report']['checks'] if row['name']=='mimic_tracking')
    assert check['passed'],check
    assert check['reset_count']>=2
    assert check['action_steps']>=100
    assert check['max_error_si']<=.001
    assert check['runtime'] in ('constraint','manual')
