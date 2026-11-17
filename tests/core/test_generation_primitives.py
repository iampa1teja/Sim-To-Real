"""WP9: the no-duplicates gate (contracts §13) on the 13 primitive tasks, all three kinds, on ``fx_arm5l``.

Together with test_no_duplicates.py, test_generation_cube_tasks.py and test_generation_stack_bowls.py this covers all
17 tasks × 3 kinds. Also checks ``configs/generate/primitives.yaml``.
"""

import contextlib
import unittest

from v4_support import REPO_ROOT, sandbox
from v4_support import harness as H

PRIMITIVES = ("prim_joint_sweep", "prim_gripper_gap", "prim_reach_point", "prim_test_reach", "prim_calibration_reach",
              "prim_waypoint_path", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
              "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")


class PrimitivesNoDuplicates(unittest.TestCase):
    """WP9 §13 gate for the 13 primitives."""

    def test_gate(self):
        with contextlib.ExitStack() as stack:
            box = stack.enter_context(sandbox())
            H.prepare_sandbox(box, PRIMITIVES)
            out = H.run_no_duplicates_suite(self, PRIMITIVES, work=box.root / "gate", arm="fx_arm5l", seed=0,
                                            kills_per_shard=1, max_attempt_factor=10.0)
            self.assertEqual(sorted(out), sorted(PRIMITIVES))

    def test_config(self):
        from core.generate.config import load_generate_config
        cfg = load_generate_config(REPO_ROOT / "configs" / "generate" / "primitives.yaml")
        self.assertEqual(sorted(cfg.tasks), sorted(PRIMITIVES))
        self.assertEqual((cfg.arm, cfg.scene), ("so101", "rpd_meeting_room"))


if __name__ == "__main__":
    unittest.main()
