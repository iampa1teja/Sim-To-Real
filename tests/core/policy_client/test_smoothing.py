"""EMA and timing use declared roles, including reordered parallel fingers."""
import numpy as np

from core.policy_client.smoothing import JointEMA, TimingLog


def test_parallel_gripper_indices_bypass_ema_without_excluding_an_arm_axis():
    ema = JointEMA(.5, False, (0, 3))
    ema.apply(np.zeros(4))
    np.testing.assert_array_equal(ema.apply(np.array([10., 20., 30., 40.])), [10, 10, 15, 40])
    no_gripper = JointEMA(.5, False)
    no_gripper.apply(np.zeros(4))
    np.testing.assert_array_equal(no_gripper.apply(np.array([10., 20., 30., 40.])), [5, 10, 15, 20])


def test_timing_jerk_excludes_all_gripper_roles(tmp_path):
    log = TimingLog(tmp_path / 'timing.csv', ['left.pos', 'axis.pos', 'slide.pos', 'right.pos'],
                    gripper_indices=(0, 3))
    for step, value in enumerate((0, 10, -20)):
        sent = np.array([value, 1, 2, -value], dtype=float)
        log.record(step, 0, 0., False, sent, sent)
    assert log.jerk_sum == 0
    log.close()
