"""WP13: the no-duplicates gate (contracts §13) on ``stack_bowls``, all three kinds, on ``fx_arm5l``."""

import contextlib
import unittest

from v4_support import sandbox
from v4_support import harness as H


class StackBowlsNoDuplicates(unittest.TestCase):
    """WP13 §13 gate for stack_bowls."""

    def test_gate(self):
        with contextlib.ExitStack() as stack:
            box = stack.enter_context(sandbox())
            H.prepare_sandbox(box, ("stack_bowls",))
            out = H.run_no_duplicates_suite(self, ("stack_bowls",), work=box.root / "gate", arm="fx_arm5l", seed=0,
                                            max_attempt_factor=10.0)
            self.assertEqual(list(out), ["stack_bowls"])


if __name__ == "__main__":
    unittest.main()
