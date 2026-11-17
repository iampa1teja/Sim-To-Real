"""WP6 → WP4: the label builders' output, together with ``Task.outcome`` and a drawn ``PerturbationSpec``, forms
the label part of a v2 sidecar that passes ``core.datasets.task_meta.validate_sidecar`` (contracts §7.2, §9.1,
§9.3, §4.2 outcome, §4.3 tracker, §8.4 commit order: only accepted attempts get a sidecar)."""

import json
import unittest

import numpy as np

from core.datasets.task_meta import validate_sidecar
from core.generate.labels import (accept, build_imitate, build_phases, build_recovery, build_segments, frame_labels,
                                  label_columns)
from core.generate.perturbations import draw_perturbation
from core.generate.protocols import Kind, PerturbationGroup
from core.tasks.base import Pose, StepState, load_task_spec
from core.tasks.generic import GenericTask
from v4_support import sandbox
from v4_support.factories import hex_digest, random_pose
from v4_support.tasks import STUB_USDA, contract_color_task_table, write_task

STAGES = ("reached", "grasped", "lifted", "over_box", "placed", "success")


def color_task():
    with sandbox() as box:
        write_task(box.tasks, "fx6_sidecar_color", contract_color_task_table(),
                   files={"usd/Small-Box-Physics.usda": STUB_USDA})
        return GenericTask(load_task_spec("fx6_sidecar_color"))


def base_sidecar(n, kind):
    """A valid v2 sidecar skeleton (§9.1) whose label keys the tests replace."""
    unit = [[1.0, 0.0, 0.0, 0.0]] * n
    return {
        "schema_version": 2, "episode_index": 0, "num_frames": n, "fps": 30, "task": "fx6_sidecar_color",
        "arm": "fx_arm5", "scene": "fx_scene", "reference_frame": "robot base link (base)", "units": "m",
        "angle_units": "rad", "joint_names": ["j1", "j2", "j3", "j4", "j5", "jaw"],
        "objects": {"cube_0": {"set": "cubes", "size_m": [0.02] * 3, "color": "red", "active": True, "role": "target",
                               "position": [[0.2, 0.0, 0.01]] * n, "orientation_wxyz": unit},
                    "small_box": {"set": None, "size_m": [0.05, 0.05, 0.035], "color": None, "active": True,
                                  "role": "container", "position": [[0.25, 0.0, 0.0]] * n, "orientation_wxyz": unit}},
        "ee_pose": None, "instruction": "Pick up the red cube and place it in the box",
        "roles": {"target": "cube_0", "container": "small_box"}, "kind": kind.value, "success": None,
        "failure_mode": None, "furthest_stage": None, "stages": list(STAGES), "stage": None, "imitate": [True] * n,
        "phases": [], "segments": [{"label": "nominal", "start": 0, "end": n, "phase": None}], "perturbation": None,
        "recovery": None, "noise": None,
        "attempt": {"index": 3, "seed": 2**64 - 1, "shard": [0, 1], "spec_fp": hex_digest("spec")},
        "fingerprints": {k: hex_digest(k) for k in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp",
                                                     "episode_fp")},
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "scripted_expert", "git_commit": None}, "legacy_v1": False,
    }


def label_block(*, kind, outcome, labels, phases, segments, perturbation, onset, applied, recovery):
    """The label keys of the sidecar as WP7 writes them from WP6 outputs (§9.1 table)."""
    return {
        "kind": kind.value, "success": outcome.success, "failure_mode": outcome.failure_mode,
        "furthest_stage": outcome.furthest_stage, "stages": list(STAGES),
        "stage": [int(lab.stage) for lab in labels], "imitate": [bool(lab.imitate) for lab in labels],
        "ee_pose": {"position": [[float(v) for v in lab.ee_pose_wxyz[:3]] for lab in labels],
                    "orientation_wxyz": [[float(v) for v in lab.ee_pose_wxyz[3:]] for lab in labels]},
        "phases": build_phases(phases), "segments": [s.to_json() for s in segments],
        "perturbation": None if perturbation is None else dict(perturbation.to_json(), onset_frame=onset,
                                                                applied=applied),
        "recovery": None if recovery is None else recovery.to_json(),
    }


def checks_frames(stage_targets, rng):
    """Per-frame check dicts that drive the tracker through ``stage_targets`` (index of the last true stage)."""
    frames = []
    for k in stage_targets:
        holding = 1 <= k <= 3
        frames.append({**{s: i <= k for i, s in enumerate(STAGES)}, "holding": holding,
                       "released": not holding, "distractor_in_box": False})
    return frames


class SidecarTests(unittest.TestCase):
    """WP6 + WP4: label builders produce validator-clean sidecars for accepted attempts of every kind."""

    @classmethod
    def setUpClass(cls):
        cls.task = color_task()

    def run_episode(self, kind, rng):
        """Simulate the loop's bookkeeping for one random attempt of ``kind`` (tracker, injected window, B onset,
        re-entries) and return everything accept/labels need."""
        n = int(rng.integers(30, 120))
        pert = draw_perturbation(self.task, kind, rng)
        tracker = self.task.make_tracker()
        onset = int(rng.integers(1, n // 2)) if pert is not None else None
        if kind is Kind.SUCCESS:
            targets = list(np.minimum(np.arange(n) * len(STAGES) // n, len(STAGES) - 1))
        elif kind is Kind.FAILURE:
            peak = int(rng.integers(0, 4))
            targets = [min(t * 6 // n, peak) for t in range(n)]
        else:
            targets = list(np.minimum(np.arange(n) * len(STAGES) // n, len(STAGES) - 1))
        reentries = []
        if kind is Kind.RECOVERY:
            d = int(rng.integers(onset, n - 5))
            targets = [min(v, 1) if onset <= t < d + 3 else v for t, v in enumerate(targets)]
            reentries = [(d, STAGES[max(0, min(targets[d] + 1, len(STAGES) - 1))])]
        progress = [tracker.update(c) for c in checks_frames(targets, rng)]
        stage = [p.stage_index for p in progress]
        final = checks_frames(targets[-1:], rng)[0]
        outcome = self.task.outcome(tracker, final, StepState(objects={}, ee=Pose((0.2, 0.0, 0.1)), gripper=0.0))
        injected_end = None
        injected = [False] * n
        if pert is not None and pert.group is PerturbationGroup.EXPERT:
            injected_end = int(rng.integers(onset + 1, n + 1))
            injected = [onset <= t < injected_end for t in range(n)]
        imitate = build_imitate(injected)
        labels = [frame_labels(stage[t], injected[t], random_pose(rng)) for t in range(n)]
        self.assertEqual(tuple(lab.imitate for lab in labels), imitate)
        recovery = build_recovery(reentries, stage, STAGES)
        phases = [["approach", "grasp", "lift", "transport", "release"][min(t * 5 // n, 4)] for t in range(n)]
        segments = build_segments(n, kind, onset=onset, injected_end=injected_end,
                                  detected=None if recovery is None else recovery.detected_frame,
                                  back_on_plan=None if recovery is None else recovery.back_on_plan_frame,
                                  phases=phases)
        return dict(n=n, pert=pert, outcome=outcome, labels=labels, phases=phases, segments=segments, onset=onset,
                    recovery=recovery, reentered=bool(reentries))

    def test_random_accepted_episodes_validate(self):
        """For every kind, random attempts that ``accept`` keeps give a sidecar that ``validate_sidecar`` accepts
        (lengths N, stage in [-1, len(stages)), unit quaternions, contiguous segments and phases, success ⇔ no
        failure_mode, success ⇒ no perturbation, recovery ⇒ recovery block)."""
        rng = np.random.default_rng(10)
        accepted = {k: 0 for k in Kind}
        for i in range(150):
            kind = list(Kind)[i % 3]
            ep = self.run_episode(kind, rng)
            reason = accept(kind, success=ep["outcome"].success, applied=ep["pert"] is not None,
                            reentered=ep["reentered"], reentered_before_onset=False)
            if reason is not None:
                continue
            accepted[kind] += 1
            payload = base_sidecar(ep["n"], kind)
            payload.update(label_block(kind=kind, outcome=ep["outcome"], labels=ep["labels"], phases=ep["phases"],
                                       segments=ep["segments"], perturbation=ep["pert"], onset=ep["onset"],
                                       applied=ep["pert"] is not None, recovery=ep["recovery"]))
            with self.subTest(i=i, kind=kind.value):
                validate_sidecar(json.loads(json.dumps(payload)), num_frames=ep["n"], fps=30)
                self.assertEqual(payload["success"], payload["failure_mode"] is None)
        self.assertTrue(all(v > 5 for v in accepted.values()), accepted)

    def test_label_columns_match_sidecar(self):
        """§10.1 ``labels_match``: stacking ``label_columns`` gives the sidecar ``stage`` / ``imitate`` / ``ee_pose``
        within float32 precision."""
        rng = np.random.default_rng(11)
        ep = self.run_episode(Kind.FAILURE, rng)
        block = label_block(kind=Kind.FAILURE, outcome=ep["outcome"], labels=ep["labels"], phases=ep["phases"],
                            segments=ep["segments"], perturbation=ep["pert"], onset=ep["onset"], applied=True,
                            recovery=None)
        cols = [label_columns(lab) for lab in ep["labels"]]
        self.assertEqual([int(c["label.stage"][0]) for c in cols], block["stage"])
        self.assertEqual([bool(c["label.imitate"][0]) for c in cols], block["imitate"])
        ee = np.stack([c["label.ee_pose"] for c in cols])
        np.testing.assert_allclose(ee[:, :3], block["ee_pose"]["position"], atol=1e-6)
        np.testing.assert_allclose(ee[:, 3:], block["ee_pose"]["orientation_wxyz"], atol=1e-6)


class OutcomeDrivenLabelTests(unittest.TestCase):
    """§7.2: the episode label is the OBSERVED outcome from ``Task.outcome`` (tracker + failure rules), never the
    intended kind; mismatches are rejected with the contract's reason."""

    @classmethod
    def setUpClass(cls):
        cls.task = color_task()

    def outcome_of(self, frames):
        tracker = self.task.make_tracker()
        for checks in frames:
            tracker.update(checks)
        return self.task.outcome(tracker, frames[-1], StepState(objects={}, ee=Pose((0.2, 0.0, 0.1)), gripper=0.0))

    @staticmethod
    def frame(k, **extra):
        holding = 1 <= k <= 3
        out = {**{s: i <= k for i, s in enumerate(STAGES)}, "holding": holding, "released": not holding,
               "distractor_in_box": False}
        out.update(extra)
        return out

    def test_failure_name_and_furthest_stage_from_the_task(self):
        """An intended failure (object dropped after lift) is accepted with ``failure_mode`` and ``furthest_stage``
        from the task's failure rules and tracker: dropped before the box after reaching ``lifted``."""
        frames = [self.frame(k) for k in (-1, 0, 0, 1, 2, 2)] + [self.frame(0)] * 5
        frames[-1] = dict(frames[-1], reached=False)
        outcome = self.outcome_of(frames)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.furthest_stage, "lifted")
        self.assertEqual(outcome.failure_mode, "dropped_before_box")
        self.assertIsNone(accept(Kind.FAILURE, success=outcome.success, applied=True, reentered=False,
                                 reentered_before_onset=False))

    def test_wrong_object_rule(self):
        """§4.6 helper flags count: a distractor in the box gives ``wrong_object_placed`` (the first rule)."""
        frames = [self.frame(k) for k in (0, 1, 2, 3)] + [self.frame(3, holding=False, distractor_in_box=True)]
        self.assertEqual(self.outcome_of(frames).failure_mode, "wrong_object_placed")

    def test_intent_mismatches_are_rejected(self):
        """Intended failure that succeeds → ``perturbation_ineffective``; intended success that fails →
        ``expert_failed``; intended recovery that does not recover → ``recovery_failed`` (rejected, never
        relabelled)."""
        success = self.outcome_of([self.frame(k) for k in range(6)])
        failed = self.outcome_of([self.frame(k) for k in (0, 1, 1, 1)])
        self.assertTrue(success.success)
        self.assertIsNone(success.failure_mode)
        self.assertEqual(success.furthest_stage, "success")
        self.assertFalse(failed.success)
        cases = [(Kind.FAILURE, success, False, "perturbation_ineffective"),
                 (Kind.SUCCESS, failed, False, "expert_failed"),
                 (Kind.RECOVERY, failed, True, "recovery_failed"),
                 (Kind.RECOVERY, success, False, "no_regression")]
        for kind, outcome, reentered, reason in cases:
            with self.subTest(kind=kind.value, reason=reason):
                got = accept(kind, success=outcome.success, applied=kind is not Kind.SUCCESS, reentered=reentered,
                             reentered_before_onset=False)
                self.assertEqual(got.value, reason)


if __name__ == "__main__":
    unittest.main()
