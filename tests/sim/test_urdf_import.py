"""End-to-end conversion and live checks; run explicitly on a free GPU."""


def test_full_import_preserves_names_fk_and_required_live_checks(imported_urdf_arm):
    result=imported_urdf_arm
    assert result['returncode']==0
    checks={row['name']:row for row in result['report']['checks']}
    required=['spec_action_dimension','conversion_names','fk_cross_check','ready_pose_gravity_hold',
              'mimic_tracking','reach_sweep','wrist_camera_frames']
    if result['name']=='seven_parallel':
        required+=['gripper_cube_lift']
    for name in required:
        assert checks[name]['passed'],checks[name]
    assert checks['fk_cross_check']['max_position_error_m']<.001
    assert len(checks['fk_cross_check']['poses'])==21
    assert checks['spec_action_dimension']['action_dimension']==(8 if result['name']=='seven_parallel' else 6)
