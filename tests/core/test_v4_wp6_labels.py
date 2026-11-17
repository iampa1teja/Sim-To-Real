"""WP6: ``core.generate.labels`` — per-frame labels, imitate, phases, segments, recovery frames and the
outcome-based acceptance table (contracts §7.2, §3.2, §9.1, §9.3, §11.1 step 3, §14 decisions 14–17,
docs/v4/prompts/WP6.md)."""

import itertools
import unittest

import numpy as np

from core.datasets.task_meta import LABEL_FEATURES
from core.generate.labels import (accept, build_imitate, build_phases, build_recovery, build_segments, frame_labels,
                                  label_columns)
from core.generate.protocols import FrameLabels, Kind, RecoveryInfo, RejectionReason, Segment, SegmentLabel
from core.tasks.base import Pose
from v4_support import assert_frozen
from v4_support.factories import random_pose

NOM, PERT, FAIL, REC = SegmentLabel.NOMINAL, SegmentLabel.PERTURBATION, SegmentLabel.FAILURE, SegmentLabel.RECOVERY
STAGES = ("reached", "grasped", "lifted", "over_box", "placed", "success")


def phases_for(n, rng=None):
    """Per-frame phase names in runs (``approach``, ``grasp``, ...)."""
    names = ["approach", "grasp", "lift", "transport", "place", "release", "retreat"]
    rng = rng or np.random.default_rng(0)
    cuts = sorted(int(c) for c in rng.integers(1, n, size=min(len(names) - 1, max(n - 1, 0)))) if n > 1 else []
    out, k, prev = [], 0, 0
    for c in cuts + [n]:
        out += [names[k % len(names)]] * (c - prev)
        prev, k = c, k + (1 if c > prev else 0)
    return out


def seg(label, start, end, phases):
    return Segment(label=label, start=start, end=end, phase=phases[start])


def assert_cover(test, segments, n, phases):
    """§7.2: non-empty, ordered, contiguous, covering [0, n); phase = phases[start]."""
    test.assertIsInstance(segments, tuple)
    test.assertGreaterEqual(len(segments), 1)
    test.assertEqual(segments[0].start, 0)
    test.assertEqual(segments[-1].end, n)
    for a, b in zip(segments, segments[1:]):
        test.assertEqual(a.end, b.start)
    for s in segments:
        test.assertIsInstance(s, Segment)
        test.assertLess(s.start, s.end)
        test.assertEqual(s.phase, phases[s.start])


class ImitateAndFrameLabelTests(unittest.TestCase):
    """WP6 §7.2 ``build_imitate``; §3.2 ``FrameLabels``; §9.3 ``LABEL_FEATURES`` columns."""

    def test_imitate_is_not_injected(self):
        """§7.2 imitate[t] = not injected[t]: false exactly on group-A window frames, true elsewhere — including
        recovery frames after the window and OU-noise frames (D14). Accepts lists and numpy arrays."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            injected = rng.random(int(rng.integers(1, 80))) < 0.3
            for given in (injected, list(injected), tuple(bool(v) for v in injected)):
                out = build_imitate(given)
                self.assertIsInstance(out, tuple)
                self.assertTrue(all(type(v) is bool for v in out))
                self.assertEqual(out, tuple(not v for v in injected))
        self.assertEqual(build_imitate([]), ())

    def test_frame_labels(self):
        """§11.1 step 3 ``FrameLabels(stage, not injected, ee_wxyz)``: the ee pose (XYZW in memory) is written in
        disk order [x, y, z, qw, qx, qy, qz] (§1.1, D17), float64 (7,)."""
        rng = np.random.default_rng(1)
        for _ in range(30):
            pose = random_pose(rng)
            stage = int(rng.integers(-1, 6))
            injected = bool(rng.integers(0, 2))
            labels = frame_labels(stage, injected, pose)
            with self.subTest(stage=stage, injected=injected):
                self.assertIsInstance(labels, FrameLabels)
                self.assertEqual(labels.stage, stage)
                self.assertIs(labels.imitate, not injected)
                x, y, z = pose.position
                qx, qy, qz, qw = pose.quat_xyzw
                self.assertEqual(labels.ee_pose_wxyz.dtype, np.float64)
                np.testing.assert_array_equal(labels.ee_pose_wxyz, [x, y, z, qw, qx, qy, qz])
                assert_frozen(labels)

    def test_label_columns_match_label_features(self):
        """§9.3 ``label_columns(labels)`` gives exactly the ``LABEL_FEATURES`` keys with their dtype and shape:
        ``label.stage`` int64 (1,), ``label.imitate`` int64 (1,) in {0, 1} (D16), ``label.ee_pose`` float32 (7,)."""
        rng = np.random.default_rng(2)
        for _ in range(30):
            labels = frame_labels(int(rng.integers(-1, 6)), bool(rng.integers(0, 2)), random_pose(rng))
            cols = label_columns(labels)
            self.assertEqual(list(cols), list(LABEL_FEATURES))
            for key, feature in LABEL_FEATURES.items():
                with self.subTest(key=key):
                    self.assertIsInstance(cols[key], np.ndarray)
                    self.assertEqual(str(cols[key].dtype), feature["dtype"])
                    self.assertEqual(cols[key].shape, tuple(feature["shape"]))
            self.assertEqual(int(cols["label.stage"][0]), labels.stage)
            self.assertEqual(int(cols["label.imitate"][0]), 1 if labels.imitate else 0)
            np.testing.assert_array_equal(cols["label.ee_pose"], labels.ee_pose_wxyz.astype(np.float32))

    def test_label_columns_do_not_alias(self):
        """The columns are fresh arrays: writing to them never touches the frozen FrameLabels."""
        labels = frame_labels(2, False, Pose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
        cols = label_columns(labels)
        for value in cols.values():
            if value.flags.writeable:
                value[...] = 0
        np.testing.assert_array_equal(labels.ee_pose_wxyz, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])


class PhaseTests(unittest.TestCase):
    """WP6 §9.1 ``phases``: run-length ``[{name, start, end}]`` of the per-frame expert phase."""

    def test_run_length(self):
        self.assertEqual(build_phases(["a", "a", "b", "a", "a", "a"]),
                         [{"name": "a", "start": 0, "end": 2}, {"name": "b", "start": 2, "end": 3},
                          {"name": "a", "start": 3, "end": 6}])
        self.assertEqual(build_phases([]), [])

    def test_property(self):
        """Contiguous runs of different names covering [0, N); expanding them gives back the input."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            n = int(rng.integers(1, 200))
            phases = phases_for(n, rng)
            runs = build_phases(phases)
            with self.subTest(n=n):
                self.assertEqual(runs[0]["start"], 0)
                self.assertEqual(runs[-1]["end"], n)
                for a, b in zip(runs, runs[1:]):
                    self.assertEqual(a["end"], b["start"])
                    self.assertNotEqual(a["name"], b["name"])
                self.assertEqual([r["name"] for r in runs for _ in range(r["end"] - r["start"])], phases)
                self.assertTrue(all(set(r) == {"name", "start", "end"} and type(r["start"]) is int for r in runs))

    def test_rejects_bad_names(self):
        """§6.2 phase is non-empty on every step: an empty or non-str name is a ``ValueError``."""
        for bad in (["a", ""], ["a", None], [3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_phases(bad)


class RecoveryFramesTests(unittest.TestCase):
    """WP6 §7.2: ``detected`` = first re-entry frame; ``back_on_plan`` = first t ≥ last re-entry frame with
    stage[t] ≥ index(last resumed_stage), N if none. ``build_recovery(reentries, stage, stages)``."""

    def test_no_reentry(self):
        self.assertIsNone(build_recovery([], [0, 1, 2], STAGES))

    def test_example(self):
        stage = [-1, 0, 0, 1, 2, 0, 0, 0, 1, 1, 2, 3, 4, 5]
        info = build_recovery([(5, "grasped"), (8, "lifted")], stage, STAGES)
        self.assertEqual(info, RecoveryInfo(detected_frame=5, resumed_stage="lifted", retries=2, back_on_plan_frame=10))
        self.assertEqual(build_recovery([(5, "grasped")], stage, STAGES).back_on_plan_frame, 8)
        self.assertEqual(build_recovery([(12, "success")], stage, STAGES).back_on_plan_frame, 13)
        self.assertEqual(build_recovery([(5, "success")], stage[:13], STAGES).back_on_plan_frame, 13, "N if none")

    def test_property(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            n = int(rng.integers(2, 120))
            stage = [int(v) for v in rng.integers(-1, len(STAGES), n)]
            frames = sorted(int(f) for f in rng.choice(n, size=int(rng.integers(1, 4)), replace=False))
            reentries = [(f, STAGES[int(rng.integers(0, len(STAGES)))]) for f in frames]
            info = build_recovery(reentries, stage, STAGES)
            k = STAGES.index(reentries[-1][1])
            with self.subTest(reentries=reentries):
                self.assertEqual(info.detected_frame, frames[0])
                self.assertEqual(info.resumed_stage, reentries[-1][1])
                self.assertEqual(info.retries, len(reentries))
                b = info.back_on_plan_frame
                self.assertGreaterEqual(b, frames[-1])
                self.assertLessEqual(b, n)
                self.assertTrue(all(stage[t] < k for t in range(frames[-1], b)))
                if b < n:
                    self.assertGreaterEqual(stage[b], k)


class SegmentTests(unittest.TestCase):
    """WP6 §7.2 segment table; zero-length segments dropped; contiguous cover of [0, N); phase = phases[start]."""

    def test_golden_rows(self):
        n = 20
        ph = phases_for(n)
        cases = [
            ("success", dict(onset=None, injected_end=None, detected=None, back_on_plan=None), [(NOM, 0, 20)]),
            ("failure A", dict(kind=Kind.FAILURE, onset=5, injected_end=9, detected=None, back_on_plan=None),
             [(NOM, 0, 5), (PERT, 5, 9), (FAIL, 9, 20)]),
            ("failure B (injected_end None -> onset + 1)",
             dict(kind=Kind.FAILURE, onset=7, injected_end=None, detected=None, back_on_plan=None),
             [(NOM, 0, 7), (PERT, 7, 8), (FAIL, 8, 20)]),
            ("recovery", dict(kind=Kind.RECOVERY, onset=4, injected_end=6, detected=8, back_on_plan=15),
             [(NOM, 0, 4), (PERT, 4, 8), (REC, 8, 15), (NOM, 15, 20)]),
            ("recovery never back on plan", dict(kind=Kind.RECOVERY, onset=4, injected_end=None, detected=8,
                                                 back_on_plan=None), [(NOM, 0, 4), (PERT, 4, 8), (REC, 8, 20)]),
            ("onset at 0 drops nominal", dict(kind=Kind.FAILURE, onset=0, injected_end=3, detected=None,
                                              back_on_plan=None), [(PERT, 0, 3), (FAIL, 3, 20)]),
            ("window open at the end", dict(kind=Kind.FAILURE, onset=12, injected_end=20, detected=None,
                                            back_on_plan=None), [(NOM, 0, 12), (PERT, 12, 20)]),
            ("window end beyond N is clamped", dict(kind=Kind.FAILURE, onset=12, injected_end=35, detected=None,
                                                    back_on_plan=None), [(NOM, 0, 12), (PERT, 12, 20)]),
            ("B onset on the last frame", dict(kind=Kind.FAILURE, onset=19, injected_end=None, detected=None,
                                               back_on_plan=None), [(NOM, 0, 19), (PERT, 19, 20)]),
            ("detected at onset", dict(kind=Kind.RECOVERY, onset=6, injected_end=None, detected=6, back_on_plan=9),
             [(NOM, 0, 6), (REC, 6, 9), (NOM, 9, 20)]),
            ("perturbation not applied (onset None)", dict(kind=Kind.FAILURE, onset=None, injected_end=None,
                                                           detected=None, back_on_plan=None), [(NOM, 0, 20)]),
        ]
        for name, kwargs, rows in cases:
            kind = kwargs.pop("kind", Kind.SUCCESS)
            with self.subTest(case=name):
                got = build_segments(n, kind, phases=ph, **kwargs)
                self.assertEqual(got, tuple(seg(label, s, e, ph) for label, s, e in rows))

    def test_kind_as_value(self):
        ph = ["p"] * 4
        self.assertEqual(build_segments(4, "failure", onset=1, injected_end=2, detected=None, back_on_plan=None,
                                        phases=ph),
                         build_segments(4, Kind.FAILURE, onset=1, injected_end=2, detected=None, back_on_plan=None,
                                        phases=ph))

    def test_random_episodes_cover(self):
        """Property over random episodes of every kind: ordered, contiguous, cover [0, N); labels in the kind's
        row order; the perturbation segment starts at the onset whenever it is non-empty."""
        order = {Kind.SUCCESS: [NOM], Kind.FAILURE: [NOM, PERT, FAIL], Kind.RECOVERY: [NOM, PERT, REC, NOM]}
        rng = np.random.default_rng(5)
        for i in range(400):
            n = int(rng.integers(1, 300))
            kind = [Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY][i % 3]
            ph = phases_for(n, rng)
            onset = int(rng.integers(0, n))
            injected_end = None if rng.random() < 0.3 else int(rng.integers(onset, n + 1))
            detected = int(rng.integers(onset, n + 1)) if kind is Kind.RECOVERY else None
            back = int(rng.integers(detected, n + 1)) if kind is Kind.RECOVERY and rng.random() < 0.8 else None
            segments = build_segments(n, kind, onset=onset, injected_end=injected_end, detected=detected,
                                      back_on_plan=back, phases=ph)
            with self.subTest(i=i, kind=kind.value, n=n):
                assert_cover(self, segments, n, ph)
                labels = [s.label for s in segments]
                it = iter(order[kind])
                self.assertTrue(all(any(lab == o for o in it) for lab in labels), f"{labels} not in row order")
                for s in segments:
                    if s.label is PERT:
                        self.assertEqual(s.start, onset)
                    assert_frozen(s)
                if kind is Kind.FAILURE:
                    p_end = onset + 1 if injected_end is None else injected_end
                    self.assertEqual([(s.start, s.end) for s in segments if s.label is FAIL],
                                     [(p_end, n)] if p_end < n else [])

    def test_group_a_window_equals_not_imitate(self):
        """§7.2 + D14 consistency for a group-A failure: the perturbation segment is exactly the frames with
        imitate false; a group-B failure has imitate true everywhere although it has a perturbation segment."""
        rng = np.random.default_rng(6)
        for _ in range(100):
            n = int(rng.integers(5, 200))
            onset = int(rng.integers(0, n - 1))
            end = int(rng.integers(onset + 1, n + 1))
            injected = [onset <= t < end for t in range(n)]
            imitate = build_imitate(injected)
            ph = phases_for(n, rng)
            segments = build_segments(n, Kind.FAILURE, onset=onset, injected_end=end, detected=None,
                                      back_on_plan=None, phases=ph)
            pert = [s for s in segments if s.label is PERT]
            self.assertEqual(len(pert), 1)
            self.assertEqual([t for t in range(n) if not imitate[t]], list(range(pert[0].start, pert[0].end)))
            world = build_segments(n, Kind.FAILURE, onset=onset, injected_end=None, detected=None,
                                   back_on_plan=None, phases=ph)
            self.assertEqual([(s.start, s.end) for s in world if s.label is PERT], [(onset, onset + 1)])
            self.assertEqual(build_imitate([False] * n), (True,) * n)

    def test_argument_errors(self):
        """``num_frames < 1`` or ``len(phases) != num_frames`` → ``ValueError``; a negative boundary →
        ``ValueError``; a non-int boundary → ``TypeError``."""
        ph = ["p"] * 5
        kw = dict(onset=1, injected_end=2, detected=None, back_on_plan=None)
        with self.assertRaises(ValueError):
            build_segments(0, Kind.SUCCESS, onset=None, injected_end=None, detected=None, back_on_plan=None, phases=[])
        with self.assertRaises(ValueError):
            build_segments(5, Kind.FAILURE, phases=ph[:4], **kw)
        with self.assertRaises(ValueError):
            build_segments(5, Kind.FAILURE, phases=ph, **dict(kw, onset=-1))
        with self.assertRaises(TypeError):
            build_segments(5, Kind.FAILURE, phases=ph, **dict(kw, onset=1.5))


def reference_accept(kind, success, applied, reentered, before_onset):
    """§7.2 acceptance table, rows evaluated in order (transcribed)."""
    if kind is Kind.SUCCESS:
        if not success:
            return "expert_failed"
        if reentered:
            return "unplanned_regression"
        return None
    if not applied:
        return "perturbation_not_applied"
    if before_onset:
        return "unplanned_regression"
    if kind is Kind.FAILURE:
        return "perturbation_ineffective" if success else None
    if not success:
        return "recovery_failed"
    return None if reentered else "no_regression"


class AcceptTests(unittest.TestCase):
    """WP6 §7.2 acceptance: the label follows the OBSERVED outcome; a mismatch with the intended kind rejects."""

    def test_exhaustive_table(self):
        """Every kind × success × applied × reentered × reentered_before_onset against the table."""
        for kind in Kind:
            for success, applied, reentered, before in itertools.product((False, True), repeat=4):
                got = accept(kind, success=success, applied=applied, reentered=reentered,
                             reentered_before_onset=before)
                want = reference_accept(kind, success, applied, reentered, before)
                with self.subTest(kind=kind.value, success=success, applied=applied, reentered=reentered,
                                  before=before):
                    if want is None:
                        self.assertIsNone(got)
                    else:
                        self.assertIs(got, RejectionReason(want))
                        self.assertEqual(got.value, want)

    def test_named_rows(self):
        """The rows the owner cares about, spelled out with their reason strings."""
        a = accept
        self.assertIsNone(a(Kind.SUCCESS, success=True, applied=False, reentered=False, reentered_before_onset=False))
        self.assertEqual(a(Kind.SUCCESS, success=False, applied=False, reentered=False,
                           reentered_before_onset=False).value, "expert_failed")
        self.assertEqual(a(Kind.SUCCESS, success=False, applied=False, reentered=True,
                           reentered_before_onset=True).value, "expert_failed", "rows are evaluated in order")
        self.assertEqual(a(Kind.SUCCESS, success=True, applied=False, reentered=True,
                           reentered_before_onset=False).value, "unplanned_regression", "D15")
        self.assertIsNone(a(Kind.FAILURE, success=False, applied=True, reentered=False, reentered_before_onset=False))
        self.assertEqual(a(Kind.FAILURE, success=True, applied=True, reentered=False,
                           reentered_before_onset=False).value, "perturbation_ineffective")
        self.assertEqual(a(Kind.FAILURE, success=False, applied=False, reentered=False,
                           reentered_before_onset=False).value, "perturbation_not_applied")
        self.assertEqual(a(Kind.RECOVERY, success=True, applied=True, reentered=True,
                           reentered_before_onset=True).value, "unplanned_regression")
        self.assertIsNone(a(Kind.RECOVERY, success=True, applied=True, reentered=True, reentered_before_onset=False))
        self.assertEqual(a(Kind.RECOVERY, success=False, applied=True, reentered=True,
                           reentered_before_onset=False).value, "recovery_failed")
        self.assertEqual(a(Kind.RECOVERY, success=True, applied=True, reentered=False,
                           reentered_before_onset=False).value, "no_regression")
        self.assertEqual(a(Kind.RECOVERY, success=False, applied=False, reentered=False,
                           reentered_before_onset=False).value, "perturbation_not_applied")

    def test_kind_as_value(self):
        self.assertEqual(accept("failure", success=True, applied=True, reentered=False, reentered_before_onset=False),
                         RejectionReason.PERTURBATION_INEFFECTIVE)


if __name__ == "__main__":
    unittest.main()
