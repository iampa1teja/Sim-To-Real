"""CPU tests of grasp measurements and review classification."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_grasps import classify, episode_metrics


def fixture():
    n = 180
    actions = np.zeros((n, 6))
    actions[:, 5] = 100
    actions[60:67, 5] = np.linspace(100, 0, 7)
    actions[67:, 5] = 0
    joints = np.zeros((n, 6))
    joints[:, 1] = np.arange(n) * .1
    positions = np.zeros((n, 3))
    positions[90:, 2] = .02
    sidecar = dict(position=positions, orientation_wxyz=np.tile([1, 0, 0, 0], (n, 1)))
    return actions, joints, sidecar


class AuditTests(unittest.TestCase):
    def measure(self, data):
        return episode_metrics(*data, 0, 100, 1)

    def test_clean(self):
        row = self.measure(fixture())
        self.assertEqual(row['close_frame'], 64)
        self.assertEqual(row['close_duration_frames'], 6)
        self.assertEqual(row['lift_frame'], 90)
        self.assertTrue(row['lift_ok'])
        self.assertEqual(row['close_attempts'], 0)
        self.assertEqual(row['dither'], 0)
        self.assertEqual(row['hover_s'], 0)
        self.assertEqual(row['roll_alignment_deg'], 0)

    def test_hesitant(self):
        data = fixture()
        data[1][10:40, 1] = 1
        data[1][45:60, 2] = np.arange(15) % 2
        row = self.measure(data)
        self.assertAlmostEqual(row['hover_s'], 29/30)
        self.assertEqual(row['dither'], 13)
        refs = [dict(episode=i, **self.measure(fixture())) for i in (1, 2, 4, 5)]
        row['episode'] = 7
        classify(refs + [row])
        self.assertEqual(row['classification'], 'hesitant')
        self.assertTrue(all(r['classification'] == 'clean' for r in refs))

    def test_fumbled_and_forced(self):
        data = fixture()
        data[0][20:35, 5] = 0
        row = self.measure(data)
        self.assertEqual(row['close_frame'], 20)
        self.assertEqual(row['close_attempts'], 1)
        data[2]['position'][:] = 0
        failed = dict(episode=8, **self.measure(data))
        forced = dict(episode=72, **self.measure(fixture()))
        reference = dict(episode=1, **self.measure(fixture()))
        classify([reference, failed, forced])
        self.assertEqual(failed['classification'], 'bad')
        self.assertEqual(forced['classification'], 'bad')

    def test_short_close_initial_close_and_missing_full_close(self):
        data = fixture()
        data[0][:5, 5] = 0
        data[0][20:29, 5] = 0
        self.assertEqual(self.measure(data)['close_frame'], 64)
        self.assertEqual(self.measure(data)['close_attempts'], 1)
        data[0][64:, 5] = 40
        self.assertIsNone(self.measure(data)['close_duration_frames'])
        data[0][:, 5] = 0
        self.assertIsNone(self.measure(data)['close_frame'])
        self.assertFalse(self.measure(data)['lift_ok'])

    def test_square_symmetry_and_lift_before_close(self):
        data = fixture()
        data[1][:, 4] = 95
        self.assertAlmostEqual(self.measure(data)['roll_alignment_deg'], 5)
        data[2]['position'][10:20, 2] = .02
        data[2]['position'][90:, 2] = 0
        self.assertFalse(self.measure(data)['lift_ok'])


if __name__ == '__main__':
    unittest.main()
