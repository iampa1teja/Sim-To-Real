"""WP14 smoke run: 5 episodes per task (3 successes, 1 failure, 1 recovery) for all 17 tasks, on ``fx_arm5l``.

Every task — the four target tasks and all 13 primitives — is generated with the wide randomisation of
contracts §16, audited, checked for duplicates and near-duplicates, then doubled twice by fake DR with camera
jitter and audited again. ``configs/generate/smoke_all.yaml`` is the container version of this run.
"""

import contextlib
import unittest
from collections import Counter

from v4_support import REPO_ROOT, sandbox
from v4_support import harness as H

ALL_TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube", "stack_bowls",
             "prim_joint_sweep", "prim_gripper_gap", "prim_reach_point", "prim_test_reach", "prim_calibration_reach",
             "prim_waypoint_path", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
             "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")
QUOTAS = {"success": 3, "failure": 1, "recovery": 1}
JITTER = {"position_m": 0.01, "rotation_deg": 2.0}


class AllTasksSmoke(unittest.TestCase):
    """Five randomised episodes per task for every one of the 17 tasks."""

    def test_five_episodes_per_task(self):
        from core.datasets.task_meta import list_sidecars, read_sidecar
        from core.generate.fingerprint import near_key, scenario_key
        from core.generate.loop import make_spec_fn
        from core.generate.protocols import Kind
        from core.tasks.loader import load_task
        with contextlib.ExitStack() as stack:
            box = stack.enter_context(sandbox())
            H.prepare_sandbox(box, ALL_TASKS)
            roots = H.generate_small(ALL_TASKS, quotas=QUOTAS, shards=1, work=box.root / "smoke", arm="fx_arm5l",
                                     max_attempt_factor=20.0)
            self.assertEqual(sorted(roots), sorted(ALL_TASKS))
            for task in ALL_TASKS:
                root = roots[task]
                with self.subTest(task=task):
                    H.assert_audit_passes(self, H.audit(root))
                    sidecars = [read_sidecar(root, i) for i in list_sidecars(root)]
                    self.assertEqual(len(sidecars), 5, f"{task}: expected 5 episodes")
                    self.assertEqual(Counter(sc["kind"] for sc in sidecars), Counter(QUOTAS))
                    counts = H.assert_no_duplicates(self, root)
                    self.assertEqual(counts, {"originals": 5, "copies": 0})
                    # the starts are pairwise more than one near-duplicate cell apart, and each differs in start pose
                    fn = make_spec_fn(load_task(task), "fx_arm5l", scene=H.SCENE)
                    specs = [fn(Kind(sc["kind"]), sc["attempt"]["index"], sc["attempt"]["seed"]) for sc in sidecars]
                    for sc, spec in zip(sidecars, specs):
                        self.assertEqual(sc["fingerprints"]["scenario_key"], scenario_key(spec))
                    self.assertEqual(len({near_key(s) for s in specs}), 5, f"{task}: near-duplicate starts")
                    self.assertEqual(len({tuple(round(v, 6) for v in s.initial_q) for s in specs}), 5)
                    self.assertEqual(len({s.expert_params.get("speed_scale") for s in specs}), 5)
                    # DR ×2 with camera jitter: 15 episodes, audited against the source
                    dr = H.fake_dr(root, box.root / "dr" / task, copies=2, camera_jitter=JITTER,
                                   cameras=("front", "wrist"))
                    H.assert_audit_passes(self, H.audit(dr, source_root=root))
                    self.assertEqual(H.assert_no_duplicates(self, dr, source_root=root), {"originals": 5, "copies": 10})

    def test_config(self):
        """configs/generate/smoke_all.yaml: all 17 tasks, 3/1/1 per task, DR ×2 with camera jitter."""
        from core.generate.config import load_generate_config
        cfg = load_generate_config(REPO_ROOT / "configs" / "generate" / "smoke_all.yaml")
        self.assertEqual(sorted(cfg.tasks), sorted(ALL_TASKS))
        self.assertEqual((cfg.arm, cfg.scene), ("so101", "rpd_meeting_room"))
        for task in ALL_TASKS:
            q = cfg.quotas[task]
            self.assertEqual((q.success, q.failure, q.recovery), (3, 1, 1), task)
        self.assertEqual(cfg.shard, (0, 1))
        self.assertGreaterEqual(cfg.max_attempt_factor, 20.0)
        self.assertEqual(cfg.dr.copies, 2)
        self.assertEqual(dict(cfg.dr.camera_jitter), JITTER)
        self.assertTrue(cfg.cameras)

    def test_dr_camera_jitter_config(self):
        """WP14 §16.7 ``dr.camera_jitter`` is optional (None by default) and validated."""
        from core.generate.config import DRConfig
        from core.generate.errors import ConfigError
        self.assertIsNone(DRConfig().camera_jitter)
        self.assertEqual(dict(DRConfig(camera_jitter=JITTER).camera_jitter), JITTER)
        for bad in ({"position_m": 0.01}, {"position_m": -0.01, "rotation_deg": 1.0},
                    {"position_m": 0.01, "rotation_deg": 1.0, "fov": 1.0}, {"position_m": "x", "rotation_deg": 1.0},
                    {"position_m": 0.2, "rotation_deg": 1.0}, {"position_m": 0.01, "rotation_deg": 30.0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                DRConfig(camera_jitter=bad)


if __name__ == "__main__":
    unittest.main()
