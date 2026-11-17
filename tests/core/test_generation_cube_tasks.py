"""WP10: the no-duplicates gate (contracts §13) on the three cube tasks, all three kinds, on ``fx_arm5l``.

Same pipeline as ``tests/core/test_no_duplicates.py``: 3 shards with random kill points and resume, merge, WP7
audit, independent fingerprint recomputation, fake DR ×2 audited against its source.
"""

import contextlib
import unittest

from v4_support import sandbox
from v4_support import harness as H

TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube")


class CubeTasksNoDuplicates(unittest.TestCase):
    """WP10 §13 gate for pick_place_cube, stack_two_cubes, pick_place_color_cube."""

    def test_gate(self):
        with contextlib.ExitStack() as stack:
            box = stack.enter_context(sandbox())
            H.prepare_sandbox(box, TASKS)
            out = H.run_no_duplicates_suite(self, TASKS, work=box.root / "gate", arm="fx_arm5l", seed=0,
                                            max_attempt_factor=10.0)
            self.assertEqual(sorted(out), sorted(TASKS))


if __name__ == "__main__":
    unittest.main()
