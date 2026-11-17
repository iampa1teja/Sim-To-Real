"""WP7: run_episode — one attempt from spec to labelled frames and an EpisodeResult (contracts §11.1, §7, §8.2).

Decisions pinned here (docs/v4/prompts/WP7.md):
* ``run_episode`` only adds frames: it never calls ``save_episode`` or ``discard_episode``; the result of an
  attempt that passes ``accept`` has ``status=accepted`` and ``episode_index=None`` (the loop assigns it).
* Per frame t: checks/tracker → ``expert.act`` → target = ``mapping.clamp_radians(q_target + noise[t])`` →
  ``add_frame(to_raw(target), to_raw(obs.q), env.render(), spec.instruction, FrameLabels(stage, not injected,
  ee))`` → stop if the expert has been done on ``settle_steps`` consecutive frames (counting t; at least 1) or
  t = max_steps − 1 → ``env.step(target)`` → world perturbation (``due``/``apply``/``observe``).
* ``settle_steps`` is ``[task.expert].settle_steps`` (default 10). Noise: ``ou_noise(spec.noise, arm_mask,
  spec.max_steps, rng_for(spec.seed, "noise"))`` with ``arm_mask = ~mapping.gripper_mask``; the world perturber
  rng is ``rng_for(spec.seed, "perturbation")`` (both overridable by the keyword arguments).
* Group A: onset = first injected frame, window end = last injected frame + 1, applied = any injected frame.
  Group B: from ``WorldPerturber`` (onset = the frame after the step it fired on).
* The outcome is ``task.outcome(tracker, checks of the last frame, state of the last frame)``.
* An exception from any GenEnv call (except ``SimulatedCrash``, which propagates) ends the episode with
  ``env_error``; a non-finite observation or target ends it with ``invalid_state`` (that frame is not added).
  Such results carry ``traj_fp = appearance_fp = episode_fp = None``.
* Fingerprints: ``scenario_key(spec)``, ``spec_fp(spec)``, ``traj_fp(action_raw, state_raw, objects0)`` with
  objects0 = frame-0 positions of the active objects rounded to 6 decimals (as the sidecar stores them, so
  the audit recomputes the same value), ``appearance_fp({active: colour}, instruction, None, None)``.
"""

import dataclasses
import unittest

import numpy as np

from core.generate.config import NoiseConfig
from core.generate.fake_env import SimulatedCrash
from core.generate.fingerprint import appearance_fp, episode_fp, scenario_key, spec_fp, traj_fp
from core.generate.labels import build_recovery, build_segments
from core.generate.loop import make_spec_fn, run_episode
from core.generate.perturbations import ou_noise
from core.generate.plan import attempt_seed
from core.generate.protocols import AttemptStatus, Kind, PerturbationGroup, RejectionReason, SegmentLabel
from core.generate.seeds import rng_for
from core.tasks.base import CheckContext
from test_v4_wp7_support import (ARM, FAST, ConstantExpert, FailingEnv, NaNEnv, RecordingEnv, SandboxCase)
from v4_support.fx_arms import fx_home_q
from v4_support.stubs import ListWriter


class LoggingExpert:
    """Wraps an Expert and logs every action it returns."""

    def __init__(self, inner):
        self.inner = inner
        self.actions = []

    phase = property(lambda self: self.inner.phase)
    done = property(lambda self: self.inner.done)

    def reset(self, spec, obs):
        self.inner.reset(spec, obs)

    def act(self, obs, progress):
        a = self.inner.act(obs, progress)
        self.actions.append(a)
        return a


class EpisodeBase(SandboxCase):
    TASKS = ("fx7_grip", "fx7_reach", "fx7_pick")

    def go(self, spec, task_name=FAST, *, env=None, expert=None, writer=None, **kw):
        env = env if env is not None else RecordingEnv(self.env(task_name))
        expert = expert if expert is not None else LoggingExpert(self.expert(task_name, spec))
        writer = writer if writer is not None else ListWriter()
        result, records = run_episode(spec, self.task(task_name), env, expert, writer, self.rig.mapping, **kw)
        return result, records, env, expert, writer

    def accepted_spec(self, kind, task_name=FAST, *, accept=None, start=0):
        """The first plan attempt of ``kind`` (from ``start``) that run_episode accepts."""
        for i in range(start, start + 60):
            try:
                spec = self.spec(task_name, kind, i)
            except Exception:  # noqa: BLE001 - SamplingError
                continue
            if accept is not None and not accept(spec):
                continue
            result, *_ = self.go(spec, task_name)
            if result.status is AttemptStatus.ACCEPTED:
                return spec
        raise AssertionError(f"no accepted {kind} attempt")


class SuccessEpisodeTests(EpisodeBase):
    """WP7 §11.1 a success attempt: frames, labels, the action invariant, the stop rule and fingerprints."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        self.spec_ = self.accepted_spec(Kind.SUCCESS, accept=lambda s: s.noise is None)
        self.result, self.records, self.envr, self.ex, self.w = self.go(self.spec_)

    def test_result_fields(self):
        r, spec = self.result, self.spec_
        self.assertIs(r.status, AttemptStatus.ACCEPTED)
        self.assertIsNone(r.reason)
        self.assertIsNone(r.episode_index)
        self.assertEqual((r.task, r.kind, r.attempt_index, r.seed), (spec.task, spec.kind, spec.attempt_index, spec.seed))
        self.assertTrue(r.success)
        self.assertIsNone(r.failure_mode)
        self.assertEqual(r.furthest_stage, self.task(FAST).stages[-1])
        self.assertEqual(r.num_frames, len(self.records))
        self.assertEqual(r.segments, build_segments(r.num_frames, Kind.SUCCESS, onset=None, injected_end=None,
                                                    detected=None, back_on_plan=None,
                                                    phases=[x.phase for x in self.records]))
        self.assertFalse(r.perturbation_applied)
        self.assertIsNone(r.perturbation_onset_frame)
        self.assertIsNone(r.recovery)

    def test_writer_holds_the_open_episode(self):
        """run_episode adds N frames and neither saves nor discards."""
        self.assertEqual(len(self.w.buffer), self.result.num_frames)
        self.assertEqual(self.w.episodes, [])
        self.assertNotIn("save_episode", self.w.calls)
        self.assertNotIn("discard_episode", self.w.calls)
        self.assertTrue(all(f["task"] == self.spec_.instruction for f in self.w.buffer))

    def test_records_match_frames(self):
        for t, (rec, frame) in enumerate(zip(self.records, self.w.buffer)):
            with self.subTest(t=t):
                self.assertEqual(rec.frame, t)
                np.testing.assert_array_equal(rec.action_raw, frame["action_raw"])
                np.testing.assert_array_equal(rec.state_raw, frame["state_raw"])
                self.assertEqual(rec.labels, frame["labels"])
                self.assertEqual(rec.labels.imitate, not rec.injected)
                self.assertEqual(rec.phase, self.ex.actions[t].phase)
        self.assertEqual(self.result.stage, tuple(r.labels.stage for r in self.records))
        self.assertEqual(self.result.imitate, tuple(r.labels.imitate for r in self.records))

    def test_action_equals_executed_target(self):
        """§11.1 invariant: to_radians(action_raw[t]) is the target sent to env.step within 1e-6 rad; the last
        frame is not stepped; targets are inside the runtime limits."""
        m = self.rig.mapping
        self.assertEqual(len(self.envr.targets), self.result.num_frames - 1)
        for t, target in enumerate(self.envr.targets):
            with self.subTest(t=t):
                np.testing.assert_allclose(m.to_radians(self.records[t].action_raw), target, atol=1e-6, rtol=0)
                self.assertTrue(np.all(target >= m.lower_rad - 1e-12) and np.all(target <= m.upper_rad + 1e-12))
                np.testing.assert_allclose(target, m.clamp_radians(self.ex.actions[t].q_target), atol=1e-12)

    def test_state_and_ee_label(self):
        m = self.rig.mapping
        q0 = np.array(self.spec_.initial_q)
        np.testing.assert_allclose(self.records[0].state_raw, m.to_raw(q0), atol=1e-9)
        T = self.rig.chain.fk(q0, "ee")
        ee = self.records[0].labels.ee_pose_wxyz
        np.testing.assert_allclose(ee[:3], T[:3, 3], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(ee[3:])), 1.0, places=9)

    def test_stop_rule(self):
        settle = self.task(FAST).spec.raw["expert"]["settle_steps"]
        done = [a.done for a in self.ex.actions]
        self.assertEqual(len(done), self.result.num_frames)
        self.assertTrue(all(done[-settle:]))
        run = 0
        for t, d in enumerate(done[:-1]):
            run = run + 1 if d else 0
            self.assertLess(run, settle, f"should have stopped at frame {t}")

    def test_fingerprints(self):
        spec, rec = self.spec_, self.records
        fp = self.result.fingerprints
        self.assertEqual(fp.scenario_key, scenario_key(spec))
        self.assertEqual(fp.spec_fp, spec_fp(spec))
        objects0 = {n: [round(float(c), 6) for c in p.position] for n, p in rec[0].objects.items()}
        t = traj_fp(np.stack([r.action_raw for r in rec]), np.stack([r.state_raw for r in rec]), objects0)
        a = appearance_fp({n: spec.objects[n].color for n in rec[0].objects}, spec.instruction, None, None)
        self.assertEqual((fp.traj_fp, fp.appearance_fp, fp.episode_fp), (t, a, episode_fp(t, a)))

    def test_deterministic(self):
        again, rec2, *_ = self.go(self.spec_)
        self.assertEqual(again, self.result)
        np.testing.assert_array_equal(np.stack([r.action_raw for r in rec2]), np.stack([r.action_raw for r in self.records]))


class PerturbedEpisodeTests(EpisodeBase):
    """WP7 §7.1/§7.2/§11.1 failure and recovery attempts: windows, onset, segments, recovery info."""

    def test_group_a_window(self):
        spec = self.accepted_spec(Kind.FAILURE)
        self.assertIs(spec.perturbation.group, PerturbationGroup.EXPERT)
        r, rec, env, ex, w = self.go(spec)
        injected = [t for t, a in enumerate(ex.actions) if a.injected]
        self.assertTrue(injected)
        self.assertEqual(r.perturbation_onset_frame, injected[0])
        self.assertTrue(r.perturbation_applied)
        self.assertEqual(r.imitate, tuple(not a.injected for a in ex.actions))
        self.assertEqual(r.segments, build_segments(r.num_frames, Kind.FAILURE, onset=injected[0],
                                                    injected_end=injected[-1] + 1, detected=None, back_on_plan=None,
                                                    phases=[a.phase for a in ex.actions]))
        self.assertFalse(r.success)
        self.assertIn(r.failure_mode, self.task(FAST).failure_names)

    def test_recovery_info(self):
        spec = self.accepted_spec(Kind.RECOVERY)
        r, rec, env, ex, w = self.go(spec)
        reentries = [(t, a.resumed_stage) for t, a in enumerate(ex.actions) if a.reentered]
        self.assertTrue(reentries)
        expected = build_recovery(reentries, list(r.stage), list(self.task(FAST).stages))
        self.assertEqual(r.recovery, expected)
        labels = [s.label for s in r.segments]
        self.assertIn(SegmentLabel.RECOVERY, labels)
        self.assertTrue(r.success)

    def test_world_perturbation_onset(self):
        """§7.1 group B: applied after env.step at frame t through set_object_pose; onset frame t + 1, whose
        record shows the new pose."""
        spec = self.first_spec("fx7_pick", Kind.RECOVERY,
                               accept=lambda s: s.perturbation.type in ("object_shift", "container_shift"))
        r, rec, env, ex, w = self.go(spec, "fx7_pick")
        self.assertEqual(len(env.teleports), 1)
        step_count, name, pose = env.teleports[0]
        self.assertTrue(r.perturbation_applied)
        self.assertEqual(r.perturbation_onset_frame, step_count)
        self.assertEqual(tuple(rec[step_count].objects[name].position), tuple(pose.position))
        self.assertTrue(all(lab for lab in r.imitate))

    def test_noise_on_arm_joints(self):
        """§7.1 group C: target = clamp(expert + ou_noise[t]); the gripper column has no noise; imitate stays true."""
        fn = make_spec_fn(self.task(FAST), ARM, scene="fx_scene", noise=NoiseConfig(fraction=1.0, sigma_rad=0.02,
                                                                                    max_abs_rad=0.03))
        spec = fn(Kind.SUCCESS, 0, attempt_seed(0, FAST, Kind.SUCCESS, 0))
        self.assertIsNotNone(spec.noise)
        r, rec, env, ex, w = self.go(spec)
        mask = ~np.asarray(self.rig.mapping.gripper_mask, bool)
        noise = ou_noise(spec.noise, mask, spec.max_steps, rng_for(spec.seed, "noise"))
        self.assertTrue(np.any(noise[1:, mask] != 0))
        np.testing.assert_array_equal(noise[:, ~mask], 0.0)
        for t, target in enumerate(env.targets):
            expected = self.rig.mapping.clamp_radians(np.asarray(ex.actions[t].q_target) + noise[t])
            np.testing.assert_allclose(target, expected, atol=1e-12, err_msg=f"frame {t}")
        self.assertTrue(all(r.imitate))


class RejectionTests(EpisodeBase):
    """WP7 §7.2 the acceptance table through run_episode, env errors, invalid states and crashes."""

    def test_expert_failed(self):
        spec = self.first_spec(FAST, Kind.SUCCESS, accept=lambda s: s.noise is None)
        r, rec, *_ = self.go(spec, expert=ConstantExpert(fx_home_q(ARM), frames=3))
        self.assertIs(r.status, AttemptStatus.REJECTED)
        self.assertIs(r.reason, RejectionReason.EXPERT_FAILED)
        self.assertIsNotNone(r.fingerprints.traj_fp)

    def test_perturbation_not_applied(self):
        spec = self.first_spec(FAST, Kind.FAILURE)
        r, *_ = self.go(spec, expert=ConstantExpert(fx_home_q(ARM), frames=3))
        self.assertIs(r.reason, RejectionReason.PERTURBATION_NOT_APPLIED)
        self.assertFalse(r.perturbation_applied)

    def test_env_error_mid_episode(self):
        spec = self.first_spec(FAST, Kind.SUCCESS, accept=lambda s: s.noise is None)
        env = FailingEnv(self.env(FAST), 3, RuntimeError("sim died"))
        r, rec, *_ = self.go(spec, env=env)
        self.assertIs(r.reason, RejectionReason.ENV_ERROR)
        self.assertEqual(r.num_frames, 4)
        self.assertEqual(len(rec), 4)
        self.assertIsNone(r.fingerprints.traj_fp)
        self.assertEqual(r.fingerprints.scenario_key, scenario_key(spec))

    def test_env_error_at_reset(self):
        spec = self.first_spec(FAST, Kind.SUCCESS)

        class BadReset(RecordingEnv):
            def reset(self, spec):
                raise RuntimeError("no scene")
        writer = ListWriter()
        r, rec, *_ = self.go(spec, env=BadReset(self.env(FAST)), writer=writer)
        self.assertIs(r.reason, RejectionReason.ENV_ERROR)
        self.assertEqual((r.num_frames, rec, writer.buffer), (0, [], []))

    def test_invalid_state(self):
        spec = self.first_spec(FAST, Kind.SUCCESS, accept=lambda s: s.noise is None)
        r, rec, *_ = self.go(spec, env=NaNEnv(self.env(FAST), 2))
        self.assertIs(r.reason, RejectionReason.INVALID_STATE)
        self.assertEqual(r.num_frames, 2)
        self.assertIsNone(r.fingerprints.episode_fp)

    def test_settle_counts_consecutive_done_frames(self):
        """§11.1 stop rule: only consecutive done frames count (settle_steps = 5 for fx7_grip)."""
        pattern = [True, True, True, False, True, True, True, True, True, True, True]

        class Toggle(ConstantExpert):
            done = property(lambda self: pattern[min(self._t, len(pattern)) - 1] if self._t else False)
        spec = self.first_spec(FAST, Kind.SUCCESS, accept=lambda s: s.noise is None)
        r, rec, env, ex, w = self.go(spec, expert=LoggingExpert(Toggle(fx_home_q(ARM))))
        self.assertEqual([a.done for a in ex.actions], pattern[:9])
        self.assertEqual(r.num_frames, 9)

    def test_targets_clamped_to_runtime_limits(self):
        """§11.1 target = mapping.clamp_radians(q_target + noise): an expert output outside the limits is clamped
        before it is executed and recorded."""
        spec = self.first_spec(FAST, Kind.SUCCESS, accept=lambda s: s.noise is None)
        wild = np.array(fx_home_q(ARM)) + np.array([5.0, -5.0, 5.0, -5.0, 5.0, 5.0])[:len(fx_home_q(ARM))]
        r, rec, env, ex, w = self.go(spec, expert=ConstantExpert(wild, frames=3))
        m = self.rig.mapping
        self.assertTrue(env.targets)
        for t, target in enumerate(env.targets):
            np.testing.assert_allclose(target, m.clamp_radians(wild), atol=1e-12)
            np.testing.assert_allclose(m.to_radians(rec[t].action_raw), target, atol=1e-6)

    def test_simulated_crash_propagates(self):
        spec = self.first_spec(FAST, Kind.SUCCESS)
        with self.assertRaises(SimulatedCrash):
            self.go(spec, env=self.env(FAST, crash_at_step=2))


if __name__ == "__main__":
    unittest.main()
