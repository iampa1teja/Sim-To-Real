"""WP1: StageTracker - ever-latched vs current flags, hold_until regression, furthest stage (contracts §4.3)."""

import unittest

import numpy as np

from core.generate.protocols import StageProgress
from core.tasks.base import StageTracker

STAGES = ("reached", "grasped", "lifted", "over_box", "placed", "success")
HOLD = {"reached": "grasped", "grasped": "placed", "lifted": "placed", "over_box": "placed"}


def _model(stages, hold_until, frames):
    """Independent transcription of the §4.3 per-frame algorithm; returns per-frame tuples and final state."""
    ever = dict.fromkeys(stages, False)
    current = dict.fromkeys(stages, False)
    out, regressions = [], []
    for t, checks in enumerate(frames):
        value = {s: bool(checks.get(s, False)) for s in stages}
        for s in stages:
            ever[s] = ever[s] or value[s]
            current[s] = current[s] or value[s]
        regressed = None
        for i, s in enumerate(stages):
            h = hold_until.get(s)
            if h is not None and current[s] and not value[s] and not current[h]:
                for later in stages[i:]:
                    current[later] = False
                regressions.append((t, s))
                regressed = s
                break

        def prefix(flags):
            index = -1
            for i, s in enumerate(stages):
                if not flags[s]:
                    break
                index = i
            return index
        out.append((prefix(current), prefix(ever), regressed))
    return out, dict(ever), regressions


def _run(stages, hold_until, frames):
    tracker = StageTracker(stages, hold_until)
    progress = [tracker.update(checks) for checks in frames]
    return tracker, progress


class StageTrackerTests(unittest.TestCase):
    """WP1 §4.3 StageTracker."""

    def test_initial_state(self):
        """§4.3 before any update: stage_index = furthest_index = -1, no regressions, all flags false."""
        tracker = StageTracker(STAGES, HOLD)
        self.assertEqual(tracker.stage_index, -1)
        self.assertEqual(tracker.furthest_index, -1)
        self.assertEqual(list(tracker.regressions), [])
        self.assertEqual(tracker.flags(), dict.fromkeys(STAGES, False))
        self.assertIsNone(tracker.furthest_stage)
        self.assertEqual(StageTracker(STAGES).flags(), dict.fromkeys(STAGES, False))   # hold_until defaults to {}

    def test_progress_record(self):
        """§3.2 / §4.3 update returns StageProgress(frame=t, stage_index, furthest_index, satisfied, regressed_stage)."""
        tracker = StageTracker(("a", "b"), {"a": "b"})
        p0 = tracker.update({"a": True, "helper": True})
        self.assertIsInstance(p0, StageProgress)
        self.assertEqual((p0.frame, p0.stage_index, p0.furthest_index, p0.regressed_stage), (0, 0, 0, None))
        self.assertEqual(dict(p0.satisfied), {"a": True, "helper": True})     # instantaneous values of all checks
        p1 = tracker.update({"a": False, "helper": False})
        self.assertEqual((p1.frame, p1.stage_index, p1.furthest_index, p1.regressed_stage), (1, -1, 0, "a"))
        self.assertEqual(dict(p1.satisfied), {"a": False, "helper": False})
        self.assertEqual(tracker.update({}).frame, 2)

    def test_ever_vs_current(self):
        """§4.3 / §14.10 ever flags latch forever; current flags drop on a hold_until regression."""
        frames = [{"reached": True}, {"reached": True, "grasped": True}, {"grasped": True},     # reached kept by grasp
                  {}, {"reached": True}, {}]
        tracker, progress = _run(STAGES, HOLD, frames)
        # t3: 'grasped' regresses (placed not reached); 'reached' survives because it is checked first and
        # current['grasped'] is still true at that point. t5: 'reached' regresses (grasped no longer current).
        self.assertEqual([p.stage_index for p in progress], [0, 1, 1, 0, 0, -1])
        self.assertEqual([p.furthest_index for p in progress], [0, 1, 1, 1, 1, 1])
        self.assertEqual([p.regressed_stage for p in progress], [None, None, None, "grasped", None, "reached"])
        self.assertEqual([tuple(r) for r in tracker.regressions], [(3, "grasped"), (5, "reached")])
        self.assertEqual(tracker.flags(), {"reached": True, "grasped": True, "lifted": False, "over_box": False,
                                           "placed": False, "success": False})
        self.assertEqual(tracker.furthest_stage, "grasped")
        self.assertEqual(tracker.stage_index, -1)
        self.assertEqual(tracker.furthest_index, 1)

    def test_no_hold_until_latches(self):
        """§4.3 a stage without hold_until never regresses: current stays latched like ever."""
        tracker, progress = _run(("a", "b", "c"), {}, [{"a": True}, {}, {"b": True}, {}, {}])
        self.assertEqual([p.stage_index for p in progress], [0, 0, 1, 1, 1])
        self.assertEqual(list(tracker.regressions), [])

    def test_hold_satisfied_by_later_stage(self):
        """§4.3 no regression while current[h] is true; once h is latched the stage may go false."""
        tracker, progress = _run(STAGES, HOLD, [{"reached": True}, {"reached": True, "grasped": True},
                                                {"grasped": True, "lifted": True}, {"grasped": True, "lifted": True},
                                                {"over_box": True, "grasped": True, "lifted": True},
                                                {"placed": True}, {}])
        self.assertEqual([p.stage_index for p in progress], [0, 1, 2, 2, 3, 4, 4])
        self.assertEqual(list(tracker.regressions), [])

    def test_regression_clears_later_stages(self):
        """§4.3 a regression of s clears current for s and every later stage (but not earlier ones)."""
        frames = [{"reached": True}, {"reached": True, "grasped": True}, {"grasped": True, "lifted": True},
                  {"lifted": False, "grasped": False}]       # grasp lost before placed: regress 'grasped'
        tracker, progress = _run(STAGES, HOLD, frames)
        self.assertEqual(progress[-1].regressed_stage, "grasped")
        self.assertEqual(progress[-1].stage_index, 0)         # 'reached' stays current (earlier than the regression)
        self.assertEqual([tuple(r) for r in tracker.regressions], [(3, "grasped")])
        self.assertEqual(tracker.furthest_stage, "lifted")

    def test_first_regressing_stage_wins(self):
        """§4.3 only the first stage (in order) satisfying the regression condition fires on a frame."""
        stages, hold = ("a", "b", "c"), {"a": "c", "b": "c"}
        tracker, progress = _run(stages, hold, [{"a": True, "b": True}, {}])
        self.assertEqual(progress[1].regressed_stage, "a")
        self.assertEqual(progress[1].stage_index, -1)
        self.assertEqual(list(tracker.regressions), [(1, "a")])

    def test_check_of_current_frame_counts_for_hold(self):
        """§4.3 step 1 runs before step 2: h true on this frame prevents the regression of s."""
        tracker, progress = _run(("a", "b"), {"a": "b"}, [{"a": True}, {"b": True}])
        self.assertEqual(progress[1].regressed_stage, None)
        self.assertEqual(progress[1].stage_index, 1)

    def test_missing_keys_are_false(self):
        """§4.3 missing keys in checks count as false."""
        tracker, progress = _run(("a", "b"), {"a": "b"}, [{"a": True}, {"unrelated": True}])
        self.assertEqual(progress[1].regressed_stage, "a")

    def test_matches_reference_model(self):
        """§4.3 random check sequences: per-frame stage_index, furthest_index, regressions and flags match the spec."""
        holds = [{}, HOLD, {"reached": "success", "lifted": "over_box"}]
        for k in range(60):
            rng = np.random.default_rng(k)
            hold = holds[k % len(holds)]
            n = int(rng.integers(1, 40))
            p_true = rng.uniform(0.2, 0.8)
            frames = [{s: bool(rng.random() < p_true) for s in STAGES if rng.random() < 0.9} for _ in range(n)]
            with self.subTest(seed=k):
                expected, ever, regressions = _model(STAGES, hold, frames)
                tracker, progress = _run(STAGES, hold, frames)
                self.assertEqual([(p.stage_index, p.furthest_index, p.regressed_stage) for p in progress], expected)
                self.assertEqual([p.frame for p in progress], list(range(n)))
                self.assertEqual(tracker.flags(), ever)
                self.assertEqual([tuple(r) for r in tracker.regressions], regressions)
                self.assertEqual(tracker.stage_index, expected[-1][0])
                self.assertEqual(tracker.furthest_index, expected[-1][1])
                furthest = [p.furthest_index for p in progress]
                self.assertEqual(furthest, sorted(furthest))            # monotone non-decreasing
                self.assertTrue(all(p.stage_index <= p.furthest_index for p in progress))
                self.assertEqual(tracker.furthest_stage, STAGES[expected[-1][1]] if expected[-1][1] >= 0 else None)

    def test_flags_is_a_copy(self):
        """§4.3 flags() returns a fresh dict of every stage; mutating it does not change the tracker."""
        tracker = StageTracker(("a", "b"))
        tracker.update({"a": True})
        flags = tracker.flags()
        self.assertIsInstance(flags, dict)
        flags["a"] = False
        self.assertTrue(tracker.flags()["a"])


if __name__ == "__main__":
    unittest.main()
