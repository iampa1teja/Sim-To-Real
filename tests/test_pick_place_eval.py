"""CPU-only evaluation contract tests: python -m unittest discover -s tests -v."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'source/sim_to_real_so101/utils/pick_place_eval.py'
spec = importlib.util.spec_from_file_location('pick_place_eval', MODULE)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)


class RecordedStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        for i, xy in enumerate(((.1, -.1), (.4, .2), (.25, .05))):
            self.write(i, {'units': 'm', 'position': [[*xy, 999], [9, 9, 9]],
                           'orientation_wxyz': [[1, 0, 0, 0], [0, 1, 0, 0]]})

    def write(self, index, data):
        (self.directory / f'episode_{index:06d}.json').write_text(json.dumps(data))

    def test_frame_zero_only(self):
        starts = eval_utils.load_starts(self.directory)
        self.assertEqual(len(starts), 3)
        np.testing.assert_array_equal(starts[0][0], [.1, -.1, 999])
        np.testing.assert_array_equal(starts[0][1], [1, 0, 0, 0])

    def test_missing_empty_malformed_and_units(self):
        for directory in (None, self.directory / 'absent'):
            with self.assertRaisesRegex(ValueError, 'directory missing'):
                eval_utils.load_starts(directory)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, 'No episode_'):
                eval_utils.load_starts(empty)
        invalid = ({}, {'units': 'cm'}, {'units': 'm', 'position': []},
                   {'units': 'm', 'position': [[1, 2, 3]], 'orientation_wxyz': [[0, 0, 0, 0]]},
                   {'units': 'm', 'position': [[1, float('nan'), 3]], 'orientation_wxyz': [[1, 0, 0, 0]]})
        for payload in invalid:
            self.write(0, payload)
            with self.assertRaisesRegex(ValueError, 'Invalid recorded start.*episode_000000'):
                eval_utils.load_starts(self.directory)

    def test_cycle_coverage_and_reproducibility(self):
        a = eval_utils.RecordedStarts(self.directory, seed=42)
        b = eval_utils.RecordedStarts(self.directory, seed=42)
        sequence = [a.sample()[0] for _ in range(30)]
        self.assertEqual(sequence, [b.sample()[0] for _ in range(30)])
        for i in range(0, 30, 3):
            self.assertEqual(set(sequence[i:i+3]), {0, 1, 2})

    def test_random_recorded_uniform(self):
        sampler = eval_utils.RecordedStarts(self.directory, mode='random', seed=123)
        indices = [sampler.sample()[0] for _ in range(6000)]
        for count in np.bincount(indices):
            self.assertLess(abs(count - 2000), 180)

    def test_base_world_roundtrip_and_table_support(self):
        # Deliberately tilted base, translated scene, rotated cube, bogus recorded Z.
        base = np.array([.8, -.4, .7])
        q = np.array([.96, .1, -.05, .2])
        matrix = eval_utils.rotation_matrix(q)
        point = np.array([.23, -.09, .17])
        world = base + matrix @ point
        np.testing.assert_allclose(matrix.T @ (world-base), point, atol=1e-6)
        plane = np.array([.02, -.03, .79])
        world, rotation = eval_utils.resting_pose(point, [.9, .1, .2, .3], base, q, plane, [.02]*3)
        recovered = matrix.T @ (world-base)
        np.testing.assert_allclose(recovered[:2], point[:2], atol=1e-6)
        normal = np.r_[-plane[:2], 1.]
        support = .01 * np.abs(normal @ rotation).sum()
        self.assertAlmostEqual(normal @ world - support - .0002, plane[2], places=12)
        other, _ = eval_utils.resting_pose([*point[:2], -999], [.9, .1, .2, .3], base, q, plane, [.02]*3)
        np.testing.assert_allclose(world, other)

    def test_random_inside_bounds_and_outside_rotated_box(self):
        sampler = eval_utils.RecordedStarts(self.directory, random_fraction=1, seed=2)
        box = eval_utils.footprint([.25, .05, 0], eval_utils.rotation_matrix([.92, 0, 0, .38]), [.135, .085, .045], True)
        def accepts(p, q):
            return not eval_utils.footprints_overlap(eval_utils.footprint(p, eval_utils.rotation_matrix(q), [.02]*3), box)
        for _ in range(1000):
            index, p, q, zone = sampler.sample(accepts)
            self.assertEqual(index, -1)
            self.assertTrue(np.all(p[:2] >= sampler.low) and np.all(p[:2] <= sampler.high))
            self.assertTrue(accepts(p, q))
            self.assertTrue(0 <= 2*np.arctan2(q[3], q[0]) < np.pi/2)
            self.assertIn(zone, [a+b for a in 'NMF' for b in 'LCR'])

    def test_random_cube_is_upright_on_live_table(self):
        base = [.94, .1, .2, -.2]
        plane = [.02, -.04, .7]
        q = eval_utils.upright_orientation(base, plane, [np.cos(.3), 0, 0, np.sin(.3)])
        rotation = eval_utils.rotation_matrix(base) @ eval_utils.rotation_matrix(q)
        normal = np.array([-.02, .04, 1.])
        np.testing.assert_allclose(rotation[:, 2], normal / np.linalg.norm(normal), atol=1e-12)

    def test_overlap_known_cases(self):
        def square(x, y, angle=0):
            return eval_utils.footprint([x, y, 0], eval_utils.rotation_matrix([np.cos(angle/2), 0, 0, np.sin(angle/2)]), [2, 2, 1])
        self.assertTrue(eval_utils.footprints_overlap(square(0, 0), square(0, 0)))
        self.assertFalse(eval_utils.footprints_overlap(square(0, 0), square(3, 0)))
        self.assertTrue(eval_utils.footprints_overlap(square(0, 0), square(2.1, 0, np.pi/4)))
        self.assertFalse(eval_utils.footprints_overlap(square(0, 0), square(2, 0)))

    def test_zones_and_away_axis(self):
        sampler = eval_utils.RecordedStarts(self.directory)
        self.assertEqual(sampler.zone([.1, .2]), 'NL')
        self.assertEqual(sampler.zone([.4, -.1]), 'FR')
        self.assertEqual(sampler.zone([.25, .05]), 'MC')
        # Away is -Y here; left then points +X.
        sampler.low, sampler.high = np.array([-.1, -.6]), np.array([.2, -.3])
        sampler.away_axis, sampler.away_sign = 1, -1
        self.assertEqual(sampler.zone([.2, -.3]), 'NL')
        self.assertEqual(sampler.zone([-.1, -.6]), 'FR')
        sampler.high = sampler.low.copy()
        self.assertEqual(sampler.zone(sampler.low), 'MC')

    def test_invalid_sampling_options(self):
        for options in ({'mode': 'bad'}, {'random_fraction': -1}, {'random_fraction': float('nan')}):
            with self.assertRaises(ValueError):
                eval_utils.RecordedStarts(self.directory, **options)

    def test_color_instructions(self):
        mapping = {'blue': 'blue task', 'red': 'red task'}
        self.assertEqual(eval_utils.select_instruction('red', 'default', mapping), 'red task')
        self.assertEqual(eval_utils.select_instruction('blue', 'default'), 'default')
        for mapping in ({'blue': 'blue'}, [], {'red': ''}):
            with self.assertRaises(ValueError):
                eval_utils.select_instruction('red', 'default', mapping)

    def test_results_json_schema_and_rates(self):
        rows = [{'episode': i, 'start_index': index, 'start_kind': kind, 'zone': zone,
                 'cube_color': color, 'steps': 450, 'success': success, 'instruction': 'task'}
                for i, (index, kind, zone, color, success) in enumerate([
                    (0, 'recorded', 'NL', 'blue', True), (-1, 'random', 'FR', 'red', False),
                    (1, 'recorded', 'NL', 'red', False)])]
        report = json.loads(json.dumps(eval_utils.results_report(rows, 'task', 'checkpoint', 42), allow_nan=False))
        self.assertEqual(set(report), {'schema_version', 'task', 'checkpoint', 'seed', 'time',
                                      'overall', 'by_start', 'by_zone', 'by_cube_color', 'episodes'})
        self.assertEqual(report['overall'], {'episodes': 3, 'successes': 1, 'success_rate': 1/3})
        self.assertEqual(report['by_start']['recorded']['success_rate'], .5)
        self.assertEqual(report['by_zone']['FR']['success_rate'], 0)
        self.assertEqual(report['by_cube_color']['blue']['success_rate'], 1)
        self.assertIsNone(eval_utils.results_report([], 't', 'c', 0)['overall']['success_rate'])


if __name__ == '__main__':
    unittest.main()
