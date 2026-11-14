"""Calibration changes select a separate synchronized dataset pair."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import unittest
from unittest.mock import call, patch

from utils.datasets.alignment import (
    MARKER,
    RecordingAlignmentError,
    recording_contract,
)
from sim.recording.setup import prepare_recording_roots
from utils.arms.arms import load_arm_spec
from utils.arms.calibration import resolve_calibration


class RecordingRootsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.roots = [str(self.directory / "sim"), str(self.directory / "real")]
        self.contract = {"version": 3, "calibration_sha256": "current"}
        self.previous = dict(self.contract, calibration_sha256="previous")

    def mark(self, root, contract):
        path = Path(root) / MARKER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract))

    def snapshot(self):
        return {
            str(path.relative_to(self.directory)): path.read_bytes() if path.is_file() else None
            for path in self.directory.rglob("*")
        }

    def candidate(self, contract=None):
        digest = hashlib.sha256(json.dumps(contract or self.contract, sort_keys=True).encode()).hexdigest()[:12]
        suffix = "-cal-" + digest
        return [str(Path(root).with_name(Path(root).name + suffix)) for root in self.roots], suffix

    def test_matching_pair_resumes_original_roots(self):
        for root in self.roots:
            self.mark(root, self.contract)
        before = self.snapshot()

        self.assertEqual(prepare_recording_roots(self.roots, self.contract), (self.roots, ""))
        self.assertEqual(self.snapshot(), before)

    def test_new_roots_are_selected_without_creating_them(self):
        self.assertEqual(prepare_recording_roots(self.roots, self.contract), (self.roots, ""))
        self.assertEqual(self.snapshot(), {})

    def test_empty_directories_are_accepted(self):
        for root in self.roots:
            Path(root).mkdir()
        before = self.snapshot()

        self.assertEqual(prepare_recording_roots(self.roots, self.contract), (self.roots, ""))
        self.assertEqual(self.snapshot(), before)

    def test_mismatch_in_either_root_moves_both(self):
        for mismatched in range(2):
            with self.subTest(mismatched=mismatched):
                for index, root in enumerate(self.roots):
                    self.mark(root, self.previous if index == mismatched else self.contract)
                before = self.snapshot()

                self.assertEqual(
                    prepare_recording_roots(self.roots, self.contract, on_mismatch="new"),
                    self.candidate(),
                )
                self.assertEqual(self.snapshot(), before)

    def test_repeated_launch_reuses_matching_calibration_pair(self):
        for root in self.roots:
            self.mark(root, self.previous)
        selected, suffix = prepare_recording_roots(self.roots, self.contract, on_mismatch="new")
        for root in selected:
            self.mark(root, self.contract)
        before = self.snapshot()

        self.assertEqual(
            prepare_recording_roots(self.roots, self.contract, on_mismatch="new"),
            (selected, suffix),
        )
        self.assertEqual(self.snapshot(), before)

    def test_missing_marker_preserves_original_payload(self):
        payload = Path(self.roots[0]) / "data" / "episode.parquet"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"existing recording")
        self.mark(self.roots[1], self.contract)
        before = self.snapshot()

        self.assertEqual(
            prepare_recording_roots(self.roots, self.contract, on_mismatch="new"),
            self.candidate(),
        )
        self.assertEqual(self.snapshot(), before)

    def test_mismatched_candidate_is_rejected_without_overwriting(self):
        for root in self.roots:
            self.mark(root, self.previous)
        candidate, _ = self.candidate()
        self.mark(candidate[0], dict(self.contract, calibration_sha256="collision"))
        before = self.snapshot()

        with self.assertRaises(RecordingAlignmentError):
            prepare_recording_roots(self.roots, self.contract, on_mismatch="new")
        self.assertEqual(self.snapshot(), before)

    def test_default_policy_rejects_different_calibration(self):
        for root in self.roots:
            self.mark(root, self.previous)
        before = self.snapshot()

        with self.assertRaises(RecordingAlignmentError):
            prepare_recording_roots(self.roots, self.contract)
        self.assertEqual(self.snapshot(), before)

    def test_recovers_base_and_candidate_before_validation(self):
        for root in self.roots:
            self.mark(root, self.previous)
        candidate, suffix = self.candidate()

        def recover(root):
            if root == candidate[0]:
                for selected in candidate:
                    self.mark(selected, self.contract)

        with patch("sim.recording.rerecording.recover_replacement", side_effect=recover) as recovery:
            self.assertEqual(
                prepare_recording_roots(self.roots, self.contract, on_mismatch="new"),
                (candidate, suffix),
            )
        self.assertEqual(recovery.call_args_list, [call(self.roots[0]), call(candidate[0])])


class RecordingCalibrationIdentityTests(unittest.TestCase):
    def contract(self, *, follower="follower-one", leader="leader-one"):
        arm, _ = load_arm_spec("so101")
        selection = resolve_calibration(arm, unit="rpd_robot", environ={})
        provenance = dict(selection.provenance,
                          follower_calibration={"sha256": follower},
                          leader_calibration={"sha256": leader})
        selection = replace(selection, provenance=provenance)
        return recording_contract(follower_id="follower", leader_id="leader", calibration_selection=selection)

    def test_contract_records_resolved_leader_without_environment_override(self):
        contract = self.contract()
        self.assertEqual(
            contract["selected_calibrations"],
            {"follower": "follower-one", "leader": "leader-one"},
        )

    def test_recalibrating_either_arm_changes_contract_and_selected_pair(self):
        original = self.contract()
        for changed in (self.contract(follower="follower-two"), self.contract(leader="leader-two")):
            with self.subTest(calibrations=changed["selected_calibrations"]):
                self.assertNotEqual(original["calibration_sha256"], changed["calibration_sha256"])
                with TemporaryDirectory() as directory:
                    roots = [str(Path(directory) / "sim"), str(Path(directory) / "real")]
                    for root in roots:
                        marker = Path(root) / MARKER
                        marker.parent.mkdir(parents=True)
                        marker.write_text(json.dumps(original))
                    selected, suffix = prepare_recording_roots(roots, changed, on_mismatch="new")
                    self.assertTrue(suffix)
                    self.assertTrue(all(new != old for new, old in zip(selected, roots)))


if __name__ == "__main__":
    unittest.main()
