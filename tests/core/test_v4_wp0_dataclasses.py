"""WP0: dataclasses of ``core.generate.protocols`` - fields, defaults, immutability, invariants (contracts §1.3, §3.2)."""

import dataclasses
import types
import unittest

import numpy as np

from core.generate import protocols
from core.generate.protocols import (EpisodeResult, EpisodeSpec, ExpertAction, ExpertObs, Fingerprints, FrameLabels,
                                     Kind, NoiseSpec, ObjectState, PerturbationGroup, PerturbationSpec, RecoveryInfo,
                                     Segment, SegmentLabel, StageProgress, StepRecord)
from core.tasks.base import Pose, StepState
from v4_support import assert_frozen
from v4_support.factories import (DATACLASS_NAMES, make_episode_result, make_episode_spec, make_expert_action,
                                  make_expert_obs, make_fingerprints, make_frame_labels, make_noise, make_object_state,
                                  make_perturbation, make_pose, make_recovery_info, make_segment, make_stage_progress,
                                  make_step_record, random_instance)

MISSING = dataclasses.MISSING

# (field name, has default) in contract order.
FIELDS = {
    "ObjectState": [("pose", False), ("active", True), ("color", True), ("size", True)],  # size: WP14 §16.2
    "PerturbationSpec": [("type", False), ("group", False), ("onset_stage", False), ("delay_steps", True),
                         ("params", True)],
    "NoiseSpec": [("theta", False), ("sigma_rad", False), ("max_abs_rad", False)],
    "EpisodeSpec": [("task", False), ("arm", False), ("scene", False), ("kind", False), ("attempt_index", False),
                    ("seed", False), ("objects", False), ("roles", False), ("goal", False), ("instruction", False),
                    ("initial_q", False), ("held_object", True), ("perturbation", True), ("noise", True),
                    ("max_steps", True), ("fps", True), ("expert_params", True), ("physics", True)],  # WP14 §16.4-5
    "StageProgress": [("frame", False), ("stage_index", False), ("furthest_index", False), ("satisfied", False),
                      ("regressed_stage", False)],
    "ExpertObs": [("frame", False), ("q", False), ("ee", False), ("tcp", False), ("gripper_gap_m", False),
                  ("objects", False), ("signals", True)],
    "ExpertAction": [("q_target", False), ("phase", False), ("done", False), ("injected", False), ("reentered", False),
                     ("resumed_stage", False)],
    "FrameLabels": [("stage", False), ("imitate", False), ("ee_pose_wxyz", False)],
    "StepRecord": [("frame", False), ("state_raw", False), ("action_raw", False), ("labels", False), ("phase", False),
                   ("injected", False), ("reentered", False), ("objects", False)],
    "Segment": [("label", False), ("start", False), ("end", False), ("phase", True)],
    "RecoveryInfo": [("detected_frame", False), ("resumed_stage", False), ("retries", False),
                     ("back_on_plan_frame", False)],
    "Fingerprints": [("scenario_key", False), ("spec_fp", False), ("traj_fp", False), ("appearance_fp", False),
                     ("episode_fp", False)],
    "EpisodeResult": [("task", False), ("kind", False), ("attempt_index", False), ("seed", False), ("status", False),
                      ("reason", False), ("episode_index", False), ("num_frames", False), ("success", False),
                      ("failure_mode", False), ("furthest_stage", False), ("stage", False), ("imitate", False),
                      ("segments", False), ("perturbation_onset_frame", False), ("perturbation_applied", False),
                      ("recovery", False), ("fingerprints", False)],
}

MAKERS = {
    "ObjectState": make_object_state, "PerturbationSpec": make_perturbation, "NoiseSpec": make_noise,
    "EpisodeSpec": make_episode_spec, "StageProgress": make_stage_progress, "ExpertObs": make_expert_obs,
    "ExpertAction": make_expert_action, "FrameLabels": make_frame_labels, "StepRecord": make_step_record,
    "Segment": make_segment, "RecoveryInfo": make_recovery_info, "Fingerprints": make_fingerprints,
    "EpisodeResult": make_episode_result,
}


class FieldTests(unittest.TestCase):
    """WP0 §3.2 dataclass fields and defaults."""

    def test_every_contract_dataclass_is_a_frozen_dataclass(self):
        """§3 all thirteen types are @dataclass(frozen=True)."""
        self.assertEqual(set(FIELDS), set(DATACLASS_NAMES))
        for name in FIELDS:
            with self.subTest(name=name):
                cls = getattr(protocols, name)
                self.assertTrue(dataclasses.is_dataclass(cls))
                self.assertTrue(cls.__dataclass_params__.frozen)

    def test_field_names_order_and_required_fields(self):
        """§3.2 exact field names in contract order; defaulted fields exactly as listed."""
        for name, expected in FIELDS.items():
            with self.subTest(name=name):
                fields = dataclasses.fields(getattr(protocols, name))
                self.assertEqual([f.name for f in fields], [n for n, _ in expected])
                has_default = [f.default is not MISSING or f.default_factory is not MISSING for f in fields]
                self.assertEqual(has_default, [d for _, d in expected])

    def test_omitting_a_required_field_raises_type_error(self):
        """§3.2 fields without a default are required (e.g. PerturbationSpec.onset_stage)."""
        with self.assertRaises(TypeError):
            PerturbationSpec(type="grasp_offset", group=PerturbationGroup.EXPERT)
        with self.assertRaises(TypeError):
            StageProgress(frame=0, stage_index=-1, furthest_index=-1, satisfied={})
        with self.assertRaises(TypeError):
            Segment(label=SegmentLabel.NOMINAL, start=0)

    def test_object_state_defaults(self):
        """§3.2 ObjectState.active = True, color = None."""
        state = ObjectState(pose=make_pose())
        self.assertIs(state.active, True)
        self.assertIsNone(state.color)

    def test_perturbation_spec_defaults(self):
        """§3.2 PerturbationSpec.delay_steps = 0, params = {} (read-only mapping)."""
        spec = PerturbationSpec(type="stall", group=PerturbationGroup.EXPERT, onset_stage=None)
        self.assertEqual(spec.delay_steps, 0)
        self.assertEqual(spec.params, {})
        self.assertIsInstance(spec.params, types.MappingProxyType)

    def test_episode_spec_defaults(self):
        """§3.2 EpisodeSpec defaults: held_object/perturbation/noise None, max_steps 600, fps 30."""
        spec = EpisodeSpec(task="t", arm="a", scene="s", kind=Kind.SUCCESS, attempt_index=0, seed=0,
                           objects={"o": make_object_state()}, roles={"target": "o"}, goal={}, instruction="i",
                           initial_q=(0.0, 0.0))
        self.assertIsNone(spec.held_object)
        self.assertIsNone(spec.perturbation)
        self.assertIsNone(spec.noise)
        self.assertEqual(spec.max_steps, 600)
        self.assertEqual(spec.fps, 30)

    def test_expert_obs_signals_default_empty(self):
        """§3.2 ExpertObs.signals defaults to an empty read-only mapping."""
        obs = make_expert_obs()
        self.assertEqual(obs.signals, {})
        self.assertIsInstance(obs.signals, types.MappingProxyType)

    def test_segment_phase_default_none(self):
        """§3.2 Segment.phase defaults to None."""
        self.assertIsNone(Segment(label=SegmentLabel.NOMINAL, start=0, end=3).phase)


class ImmutabilityTests(unittest.TestCase):
    """WP0 §1.3/§3 immutability."""

    def test_factory_instances_are_deeply_frozen(self):
        """§3 frozen, mappings are MappingProxyType, sequences tuples, arrays read-only float64."""
        for name, maker in MAKERS.items():
            with self.subTest(name=name):
                instance = maker()
                self.assertIsInstance(instance, getattr(protocols, name))
                assert_frozen(instance)

    def test_random_instances_are_deeply_frozen(self):
        """§3 property: every valid instance is deeply immutable."""
        for k in range(3):
            rng = np.random.default_rng(k)
            for name in DATACLASS_NAMES:
                for _ in range(5):
                    with self.subTest(k=k, name=name):
                        assert_frozen(random_instance(name, rng))

    def test_sequence_fields_become_tuples(self):
        """§3 sequence fields are converted to tuples in __post_init__."""
        spec = make_episode_spec(initial_q=[0.1, 0.2, 0.3])
        self.assertIsInstance(spec.initial_q, tuple)
        self.assertEqual(spec.initial_q, (0.1, 0.2, 0.3))
        result = make_episode_result(num_frames=3, stage=[0, 0, 1], imitate=[True, False, True],
                                     segments=[make_segment("nominal", 0, 3)])
        self.assertEqual(result.stage, (0, 0, 1))
        self.assertEqual(result.imitate, (True, False, True))
        self.assertIsInstance(result.segments, tuple)
        for value in (result.stage, result.imitate):
            self.assertIsInstance(value, tuple)

    def test_mapping_fields_are_read_only_proxies(self):
        """§3 mapping fields are wrapped in types.MappingProxyType."""
        spec = make_episode_spec()
        for value in (spec.objects, spec.roles, spec.goal):
            self.assertIsInstance(value, types.MappingProxyType)
            with self.assertRaises(TypeError):
                value["new"] = 1
        self.assertIsInstance(make_stage_progress().satisfied, types.MappingProxyType)
        self.assertIsInstance(make_expert_obs().objects, types.MappingProxyType)
        self.assertIsInstance(make_step_record().objects, types.MappingProxyType)
        self.assertIsInstance(make_perturbation().params, types.MappingProxyType)

    def test_mapping_fields_do_not_alias_the_callers_dict(self):
        """§3 immutability: mutating the dict passed in does not change the instance."""
        objects = {"cube": make_object_state(color="red")}
        roles = {"target": "cube"}
        goal = {"point": [0.1, 0.2, 0.3]}
        spec = make_episode_spec(objects=objects, roles=roles, goal=goal)
        objects["other"] = make_object_state()
        roles["container"] = "other"
        goal["point"].append(9.0)
        goal["extra"] = 1
        self.assertEqual(set(spec.objects), {"cube"})
        self.assertEqual(dict(spec.roles), {"target": "cube"})
        self.assertEqual(set(spec.goal), {"point"})
        self.assertEqual(len(spec.goal["point"]), 3)

    def test_nested_json_values_are_frozen(self):
        """§3 goal / params values: nested lists become tuples and nested dicts read-only mappings."""
        spec = make_episode_spec(goal={"point": [0.1, 0.2, 0.3], "pts": [[1.0, 2.0], [3.0, 4.0]],
                                       "region": {"xy_min": [0.0, 0.1], "xy_max": [0.2, 0.3]}, "n": 2, "s": "x"})
        self.assertEqual(spec.goal["point"], (0.1, 0.2, 0.3))
        self.assertIsInstance(spec.goal["point"], tuple)
        self.assertEqual(spec.goal["pts"], ((1.0, 2.0), (3.0, 4.0)))
        self.assertIsInstance(spec.goal["region"], types.MappingProxyType)
        self.assertEqual(spec.goal["region"]["xy_min"], (0.0, 0.1))
        pert = make_perturbation(params={"xy_m": [0.01, 0.02], "steps": 3, "object": "@target"})
        self.assertEqual(pert.params["xy_m"], (0.01, 0.02))
        assert_frozen(spec)
        assert_frozen(pert)

    def test_arrays_are_float64_contiguous_read_only_copies(self):
        """§1.3 protocol arrays are float64, C-contiguous and read-only; the caller's array is not aliased."""
        source = np.arange(12, dtype=np.float32)[::2]          # non-contiguous float32 view, length 6
        cases = {
            "ExpertObs.q": lambda a: make_expert_obs(q=a).q,
            "ExpertAction.q_target": lambda a: make_expert_action(q_target=a).q_target,
            "StepRecord.state_raw": lambda a: make_step_record(state_raw=a, action_raw=np.zeros(6)).state_raw,
            "StepRecord.action_raw": lambda a: make_step_record(state_raw=np.zeros(6), action_raw=a).action_raw,
        }
        for label, get in cases.items():
            with self.subTest(field=label):
                original = np.arange(12, dtype=np.float32)[::2]
                value = get(original)
                self.assertEqual(value.dtype, np.float64)
                self.assertTrue(value.flags.c_contiguous)
                self.assertFalse(value.flags.writeable)
                np.testing.assert_array_equal(value, source.astype(np.float64))
                with self.assertRaises(ValueError):
                    value[0] = 99.0
                try:
                    original[0] = 42.0
                except ValueError:
                    pass                                         # caller's array was frozen instead of copied
                self.assertEqual(value[0], float(source[0]))
        labels = make_frame_labels(ee_pose_wxyz=[0.1, 0.2, 0.3, 1, 0, 0, 0])
        self.assertEqual(labels.ee_pose_wxyz.dtype, np.float64)
        self.assertEqual(labels.ee_pose_wxyz.shape, (7,))
        self.assertFalse(labels.ee_pose_wxyz.flags.writeable)

    def test_list_input_becomes_array(self):
        """§1.3 array fields accept sequences and store np.ndarray."""
        obs = make_expert_obs(q=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertIsInstance(obs.q, np.ndarray)
        self.assertEqual(obs.q.shape, (6,))

    def test_array_shapes_are_checked(self):
        """§1.3/§1.5 wrong array shapes raise ValueError: ee_pose_wxyz is (7,), joint vectors are 1-D."""
        with self.assertRaises(ValueError):
            make_frame_labels(ee_pose_wxyz=np.zeros(6))
        with self.assertRaises(ValueError):
            make_frame_labels(ee_pose_wxyz=np.zeros((1, 7)))
        with self.assertRaises(ValueError):
            make_expert_obs(q=np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            make_expert_action(q_target=np.zeros((6, 1)))

    def test_equality_with_array_fields(self):
        """§3 instances are comparable with ==, including those holding arrays."""
        self.assertEqual(make_expert_obs(q=np.ones(6)), make_expert_obs(q=np.ones(6)))
        self.assertNotEqual(make_expert_obs(q=np.ones(6)), make_expert_obs(q=np.zeros(6)))
        self.assertNotEqual(make_expert_obs(q=np.ones(6)), make_expert_obs(q=np.ones(5)))
        self.assertEqual(make_expert_action(), make_expert_action())
        self.assertNotEqual(make_expert_action(phase="a"), make_expert_action(phase="b"))
        self.assertEqual(make_frame_labels(), make_frame_labels())
        self.assertNotEqual(make_frame_labels(stage=1), make_frame_labels(stage=2))
        self.assertEqual(make_step_record(), make_step_record())
        self.assertNotEqual(make_step_record(), make_step_record(action_raw=np.ones(6)))
        self.assertIs(make_expert_obs() == make_expert_obs(), True)

    def test_equality_of_mapping_instances(self):
        """§3 equal field values compare equal; any field change compares unequal."""
        self.assertEqual(make_episode_spec(), make_episode_spec())
        self.assertNotEqual(make_episode_spec(), make_episode_spec(seed=1))
        self.assertNotEqual(make_episode_spec(), make_episode_spec(goal={"a": 1}))
        self.assertEqual(make_episode_result(), make_episode_result())

    def test_specs_are_not_hashable(self):
        """§3 instances with mapping fields are not hashable (spec_fp is the identity)."""
        for instance in (make_episode_spec(), make_perturbation(), make_stage_progress(), make_expert_obs()):
            with self.subTest(cls=type(instance).__name__):
                with self.assertRaises(TypeError):
                    hash(instance)


class EpisodeSpecInvariantTests(unittest.TestCase):
    """WP0 §3.2 EpisodeSpec invariants."""

    def test_valid_specs_for_every_kind(self):
        """§3.2 a success without perturbation and failure/recovery with one are valid."""
        for kind in Kind:
            with self.subTest(kind=kind):
                self.assertIs(make_episode_spec(kind).kind, kind)

    def test_success_with_perturbation_is_rejected(self):
        """§3.2 success ⇔ perturbation is None (success side)."""
        with self.assertRaises(ValueError):
            make_episode_spec(Kind.SUCCESS, perturbation=make_perturbation())

    def test_failure_or_recovery_without_perturbation_is_rejected(self):
        """§3.2 success ⇔ perturbation is None (failure/recovery side)."""
        for kind in (Kind.FAILURE, Kind.RECOVERY):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    make_episode_spec(kind, perturbation=None)

    def test_noise_only_on_success(self):
        """§3.2 noise only when kind is SUCCESS."""
        self.assertEqual(make_episode_spec(Kind.SUCCESS, noise=make_noise()).noise, make_noise())
        for kind in (Kind.FAILURE, Kind.RECOVERY):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    make_episode_spec(kind, noise=make_noise())

    def test_role_must_name_an_active_object(self):
        """§3.2 each role value is an active object."""
        with self.assertRaises(ValueError):
            make_episode_spec(roles={"target": "missing"})
        with self.assertRaises(ValueError):
            make_episode_spec(roles={"target": "cube_1"})          # cube_1 is inactive in the factory spec
        self.assertEqual(make_episode_spec(roles={}).roles, {})

    def test_kind_accepts_enum_value(self):
        """§3.1 the kind field holds a Kind member."""
        spec = make_episode_spec("recovery")
        self.assertIs(spec.kind, Kind.RECOVERY)

    def test_expert_mode(self):
        """§3.2 expert_mode is "blind" iff kind is FAILURE, else "closed_loop"."""
        self.assertEqual(make_episode_spec(Kind.SUCCESS).expert_mode, "closed_loop")
        self.assertEqual(make_episode_spec(Kind.RECOVERY).expert_mode, "closed_loop")
        self.assertEqual(make_episode_spec(Kind.FAILURE).expert_mode, "blind")
        self.assertIsInstance(EpisodeSpec.__dict__["expert_mode"], property)

    def test_inactive_objects_are_kept(self):
        """§3.2 objects holds every object, inactive ones with active=False."""
        spec = make_episode_spec()
        self.assertFalse(spec.objects["cube_1"].active)
        self.assertEqual(set(spec.objects), {"cube_0", "cube_1", "box"})


class StepStateTests(unittest.TestCase):
    """WP0 §3.2 ExpertObs.step_state."""

    def test_step_state_maps_fields(self):
        """§3.2 step_state = StepState(objects={active poses}, ee, gripper=gap, signals, tcp)."""
        obs = make_expert_obs(signals={"holding": True}, gripper_gap_m=0.021)
        state = obs.step_state()
        self.assertIsInstance(state, StepState)
        self.assertEqual(dict(state.objects), {"cube_0": obs.objects["cube_0"].pose})
        self.assertEqual(state.ee, obs.ee)
        self.assertEqual(state.gripper, 0.021)
        self.assertEqual(dict(state.signals), {"holding": True})
        if "tcp" in {f.name for f in dataclasses.fields(StepState)}:   # StepState.tcp arrives with WP1 (§4.2)
            self.assertEqual(state.tcp, obs.tcp)

    def test_step_state_excludes_inactive_objects(self):
        """§3.2 only active objects appear in StepState.objects."""
        obs = make_expert_obs(objects={"a": make_object_state(active=False), "b": make_object_state(active=False)})
        self.assertEqual(dict(obs.step_state().objects), {})

    def test_step_state_objects_are_poses(self):
        """§3.2 StepState.objects values are Pose (not ObjectState)."""
        for pose in make_expert_obs().step_state().objects.values():
            self.assertIsInstance(pose, Pose)


class ContainerTypeTests(unittest.TestCase):
    """WP0 §3.2 field types of nested values."""

    def test_nested_types(self):
        """§3.2 nested fields hold the contract types."""
        result = make_episode_result("recovery", recovery=make_recovery_info())
        self.assertIsInstance(result.fingerprints, Fingerprints)
        self.assertIsInstance(result.recovery, RecoveryInfo)
        self.assertTrue(all(isinstance(s, Segment) for s in result.segments))
        self.assertIsInstance(make_step_record().labels, FrameLabels)
        spec = make_episode_spec("failure")
        self.assertIsInstance(spec.perturbation, PerturbationSpec)
        self.assertTrue(all(isinstance(o, ObjectState) for o in spec.objects.values()))
        self.assertIsInstance(make_noise(), NoiseSpec)
        self.assertIsInstance(make_expert_action(), ExpertAction)
        self.assertIsInstance(make_stage_progress(), StageProgress)
        self.assertIsInstance(make_episode_result(), EpisodeResult)
        self.assertIsInstance(make_step_record(), StepRecord)
        self.assertIsInstance(make_expert_obs(), ExpertObs)


if __name__ == "__main__":
    unittest.main()
