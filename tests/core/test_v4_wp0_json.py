"""WP0: JSON round-trip of the protocol dataclasses (contracts §3.4)."""

import copy
import dataclasses
import inspect
import json
import math
import unittest

import numpy as np

from core.generate import protocols
from core.generate.errors import SchemaError
from core.generate.protocols import (AttemptStatus, EpisodeResult, EpisodeSpec, Kind, ObjectState, RejectionReason,
                                     SkipReason)
from core.utils.hashing import canonical_json
from v4_support.factories import (DATACLASS_NAMES, make_episode_result, make_episode_spec, make_expert_action,
                                  make_expert_obs, make_fingerprints, make_frame_labels, make_noise, make_object_state,
                                  make_perturbation, make_recovery_info, make_segment, make_stage_progress,
                                  make_step_record, random_instance)

MAKERS = {
    "ObjectState": make_object_state, "PerturbationSpec": make_perturbation, "NoiseSpec": make_noise,
    "EpisodeSpec": lambda: make_episode_spec("failure", goal={"p": [0.1, 0.2, 0.3], "n": 2}),
    "StageProgress": make_stage_progress, "ExpertObs": lambda: make_expert_obs(signals={"holding": True}),
    "ExpertAction": make_expert_action, "FrameLabels": make_frame_labels, "StepRecord": make_step_record,
    "Segment": lambda: make_segment("recovery", 3, 9, "grasp"), "RecoveryInfo": make_recovery_info,
    "Fingerprints": make_fingerprints,
    "EpisodeResult": lambda: make_episode_result("recovery", recovery=make_recovery_info()),
}
SCHEMA = {"EpisodeSpec": "episode_spec/1", "EpisodeResult": "episode_result/1"}


def round_trip(instance):
    return type(instance).from_json(json.loads(canonical_json(instance.to_json())))


def assert_plain_json(test, value, where="$"):
    """Only exact dict/list/str/int/float/bool/None, string keys, finite floats."""
    kind = type(value)
    if kind is dict:
        for key, item in value.items():
            test.assertIs(type(key), str, f"{where}: key {key!r}")
            assert_plain_json(test, item, f"{where}.{key}")
    elif kind is list:
        for index, item in enumerate(value):
            assert_plain_json(test, item, f"{where}[{index}]")
    elif kind is float:
        test.assertTrue(math.isfinite(value), where)
    else:
        test.assertIn(kind, (str, int, bool, type(None)), f"{where}: {kind.__name__}")


class ToJsonTests(unittest.TestCase):
    """WP0 §3.4 to_json encoding."""

    def test_keys_are_field_names_plus_schema_tag(self):
        """§3.4 to_json has one key per field, plus "schema" for EpisodeSpec and EpisodeResult only."""
        for name, maker in MAKERS.items():
            with self.subTest(name=name):
                payload = maker().to_json()
                expected = {f.name for f in dataclasses.fields(getattr(protocols, name))}
                if name in SCHEMA:
                    expected.add("schema")
                self.assertEqual(set(payload), expected)

    def test_schema_tags(self):
        """§3.4 schema tags episode_spec/1 and episode_result/1."""
        self.assertEqual(make_episode_spec().to_json()["schema"], "episode_spec/1")
        self.assertEqual(make_episode_result().to_json()["schema"], "episode_result/1")

    def test_output_is_plain_json(self):
        """§3.4 enums → value, tuples → lists, arrays → lists of float, mappings → dicts (exact builtin types)."""
        for name, maker in MAKERS.items():
            with self.subTest(name=name):
                payload = maker().to_json()
                assert_plain_json(self, payload)
                canonical_json(payload)

    def test_random_output_is_plain_json(self):
        """§3.4 property: to_json of any valid instance is plain, canonical-serialisable JSON."""
        for k in range(3):
            rng = np.random.default_rng(100 + k)
            for name in DATACLASS_NAMES:
                for _ in range(5):
                    with self.subTest(k=k, name=name):
                        payload = random_instance(name, rng).to_json()
                        assert_plain_json(self, payload)
                        canonical_json(payload)

    def test_episode_spec_encoding(self):
        """§3.4 EpisodeSpec: kind value, objects as dicts with Pose {position, quat_xyzw}, initial_q list."""
        spec = make_episode_spec("failure", initial_q=(0.0, 0.5), seed=2**64 - 1)
        payload = spec.to_json()
        self.assertEqual(payload["kind"], "failure")
        self.assertEqual(payload["initial_q"], [0.0, 0.5])
        self.assertEqual(payload["seed"], 2**64 - 1)
        self.assertIs(type(payload["seed"]), int)
        cube = payload["objects"]["cube_0"]
        self.assertEqual(set(cube), {"pose", "active", "color"})
        self.assertEqual(set(cube["pose"]), {"position", "quat_xyzw"})
        self.assertEqual(cube["pose"]["position"], list(spec.objects["cube_0"].pose.position))
        self.assertEqual(cube["pose"]["quat_xyzw"], list(spec.objects["cube_0"].pose.quat_xyzw))
        self.assertEqual(payload["roles"], {"target": "cube_0", "container": "box"})
        self.assertEqual(payload["perturbation"]["group"], "expert")
        self.assertIsNone(payload["noise"])
        self.assertIsNone(payload["held_object"])

    def test_nested_tuples_become_lists(self):
        """§3.4 tuples (also inside goal / params) become lists."""
        spec = make_episode_spec("failure", goal={"pts": [[1.0, 2.0], [3.0, 4.0]]},
                                 perturbation=make_perturbation(params={"r": (0.1, 0.2)}))
        payload = spec.to_json()
        self.assertEqual(payload["goal"], {"pts": [[1.0, 2.0], [3.0, 4.0]]})
        self.assertEqual(payload["perturbation"]["params"], {"r": [0.1, 0.2]})

    def test_array_fields_become_float_lists(self):
        """§3.4 arrays become (nested) lists of float."""
        obs_q = make_expert_obs(q=np.arange(6)).to_json()["q"]
        self.assertEqual(obs_q, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(all(type(v) is float for v in obs_q))
        labels = make_frame_labels().to_json()
        self.assertEqual(len(labels["ee_pose_wxyz"]), 7)
        self.assertTrue(all(type(v) is float for v in labels["ee_pose_wxyz"]))

    def test_episode_result_encoding(self):
        """§3.4 EpisodeResult: status/kind/reason values, segments/fingerprints as dicts."""
        payload = make_episode_result("failure", "rejected").to_json()
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["kind"], "failure")
        self.assertEqual(payload["reason"], "expert_failed")
        self.assertEqual(payload["segments"][0], {"label": "nominal", "start": 0, "end": 5, "phase": None})
        self.assertEqual(set(payload["fingerprints"]),
                         {"scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp"})
        self.assertIsInstance(payload["stage"], list)
        self.assertIsInstance(payload["imitate"], list)

    def test_step_record_objects_are_pose_dicts(self):
        """§3.4 StepRecord.objects → {name: {position, quat_xyzw}}; labels → FrameLabels dict."""
        payload = make_step_record().to_json()
        self.assertEqual(set(payload["objects"]["cube_0"]), {"position", "quat_xyzw"})
        self.assertEqual(set(payload["labels"]), {"stage", "imitate", "ee_pose_wxyz"})


class RoundTripTests(unittest.TestCase):
    """WP0 §3.4 from_json(json.loads(canonical_json(x.to_json()))) == x."""

    def test_factory_instances_round_trip(self):
        """§3.4 round-trip of one instance per class."""
        for name, maker in MAKERS.items():
            with self.subTest(name=name):
                instance = maker()
                back = round_trip(instance)
                self.assertIs(type(back), type(instance))
                self.assertEqual(back, instance)

    def test_random_instances_round_trip(self):
        """§3.4 property: round-trip holds for random valid instances of every class."""
        for k in range(5):
            rng = np.random.default_rng(k)
            for name in DATACLASS_NAMES:
                for _ in range(8):
                    instance = random_instance(name, rng)
                    with self.subTest(k=k, name=name):
                        self.assertEqual(round_trip(instance), instance)

    def test_round_trip_is_a_fixed_point(self):
        """§3.4 to_json(from_json(to_json(x))) == to_json(x) (canonical form is stable)."""
        for k in range(3):
            rng = np.random.default_rng(50 + k)
            for name in DATACLASS_NAMES:
                instance = random_instance(name, rng)
                with self.subTest(k=k, name=name):
                    first = canonical_json(instance.to_json())
                    self.assertEqual(canonical_json(round_trip(instance).to_json()), first)

    def test_round_trip_restores_types(self):
        """§3.4 from_json restores enums, tuples, mapping proxies, Pose and arrays."""
        spec = round_trip(make_episode_spec("recovery", goal={"p": [1.0, 2.0]}))
        self.assertIs(spec.kind, Kind.RECOVERY)
        self.assertIsInstance(spec.initial_q, tuple)
        self.assertIsInstance(spec.objects["cube_0"], ObjectState)
        self.assertEqual(spec.objects["cube_0"].pose, make_episode_spec().objects["cube_0"].pose)
        self.assertEqual(spec.goal["p"], (1.0, 2.0))
        result = round_trip(make_episode_result("failure", "skipped"))
        self.assertIs(result.status, AttemptStatus.SKIPPED)
        self.assertIs(result.reason, SkipReason.SCENARIO_SEEN)
        result = round_trip(make_episode_result("failure", "rejected", reason=RejectionReason.ENV_ERROR))
        self.assertIs(result.reason, RejectionReason.ENV_ERROR)
        obs = round_trip(make_expert_obs())
        self.assertIsInstance(obs.q, np.ndarray)
        self.assertEqual(obs.q.dtype, np.float64)
        self.assertFalse(obs.q.flags.writeable)

    def test_from_json_is_a_classmethod(self):
        """§3.4 from_json is a classmethod on every dataclass."""
        for name in DATACLASS_NAMES:
            with self.subTest(name=name):
                cls = getattr(protocols, name)
                self.assertIsInstance(inspect.getattr_static(cls, "from_json"), classmethod)
                self.assertTrue(callable(inspect.getattr_static(cls, "to_json")))

    def test_from_json_does_not_mutate_input(self):
        """§3.4 from_json reads its argument without changing it."""
        payload = json.loads(canonical_json(make_episode_spec("failure").to_json()))
        before = copy.deepcopy(payload)
        EpisodeSpec.from_json(payload)
        self.assertEqual(payload, before)


class FromJsonValidationTests(unittest.TestCase):
    """WP0 §3.4 from_json raises SchemaError."""

    def payload(self, name):
        return json.loads(canonical_json(MAKERS[name]().to_json()))

    def test_unknown_key(self):
        """§3.4 unknown keys raise SchemaError (top level, every class)."""
        for name in DATACLASS_NAMES:
            with self.subTest(name=name):
                payload = self.payload(name)
                payload["unexpected"] = 1
                with self.assertRaises(SchemaError):
                    getattr(protocols, name).from_json(payload)

    def test_unknown_nested_key(self):
        """§3.4 unknown keys inside nested objects (ObjectState, Pose) raise SchemaError."""
        payload = self.payload("EpisodeSpec")
        payload["objects"]["cube_0"]["extra"] = True
        with self.assertRaises(SchemaError):
            EpisodeSpec.from_json(payload)
        payload = self.payload("EpisodeSpec")
        payload["objects"]["cube_0"]["pose"]["quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(SchemaError):
            EpisodeSpec.from_json(payload)

    def test_missing_key(self):
        """§3.4 every missing key (including defaulted fields) raises SchemaError."""
        for name in DATACLASS_NAMES:
            for key in list(self.payload(name)):
                with self.subTest(name=name, key=key):
                    payload = self.payload(name)
                    del payload[key]
                    with self.assertRaises(SchemaError):
                        getattr(protocols, name).from_json(payload)

    def test_wrong_or_missing_schema_tag(self):
        """§3.4 a wrong schema tag raises SchemaError."""
        for cls, name, wrong in ((EpisodeSpec, "EpisodeSpec", "episode_spec/2"),
                                 (EpisodeResult, "EpisodeResult", "episode_spec/1")):
            with self.subTest(name=name):
                payload = self.payload(name)
                payload["schema"] = wrong
                with self.assertRaises(SchemaError):
                    cls.from_json(payload)

    def test_non_finite_floats(self):
        """§3.4 non-finite floats anywhere raise SchemaError."""
        def spec_q(p):
            p["initial_q"][0] = math.nan

        def spec_pose(p):
            p["objects"]["cube_0"]["pose"]["position"][2] = math.inf

        def spec_goal(p):
            p["goal"]["p"][1] = -math.inf

        def spec_params(p):
            p["perturbation"]["params"]["xy_m"] = math.nan

        for label, mutate in (("initial_q", spec_q), ("pose", spec_pose), ("goal", spec_goal), ("params", spec_params)):
            with self.subTest(field=label):
                payload = self.payload("EpisodeSpec")
                mutate(payload)
                with self.assertRaises(SchemaError):
                    EpisodeSpec.from_json(payload)
        for name, key in (("ExpertObs", "q"), ("FrameLabels", "ee_pose_wxyz"), ("StepRecord", "action_raw")):
            with self.subTest(name=name):
                payload = self.payload(name)
                payload[key][0] = math.nan
                with self.assertRaises(SchemaError):
                    getattr(protocols, name).from_json(payload)
        payload = self.payload("NoiseSpec")
        payload["sigma_rad"] = math.inf
        with self.assertRaises(SchemaError):
            protocols.NoiseSpec.from_json(payload)

    def test_invalid_enum_values(self):
        """§3.4 validation: unknown enum strings raise SchemaError."""
        payload = self.payload("EpisodeSpec")
        payload["kind"] = "mimic"
        with self.assertRaises(SchemaError):
            EpisodeSpec.from_json(payload)
        payload = self.payload("EpisodeResult")
        payload["status"] = "done"
        with self.assertRaises(SchemaError):
            EpisodeResult.from_json(payload)
        payload = self.payload("EpisodeResult")
        payload["reason"] = "bogus_reason"
        with self.assertRaises(SchemaError):
            EpisodeResult.from_json(payload)
        payload = self.payload("Segment")
        payload["label"] = "Nominal"
        with self.assertRaises(SchemaError):
            protocols.Segment.from_json(payload)

    def test_bad_pose_shape(self):
        """§3.4 Pose needs position [3] and quat_xyzw [4]."""
        payload = self.payload("EpisodeSpec")
        payload["objects"]["cube_0"]["pose"]["position"] = [0.1, 0.2]
        with self.assertRaises(SchemaError):
            EpisodeSpec.from_json(payload)
        payload = self.payload("ExpertObs")
        payload["ee"]["quat_xyzw"] = [0.0, 0.0, 1.0]
        with self.assertRaises(SchemaError):
            protocols.ExpertObs.from_json(payload)

    def test_seed_must_be_json_integer(self):
        """§1.3/§3.4 seeds are JSON integers; a string or float seed raises SchemaError."""
        for bad in ("12345", 12345.5):
            with self.subTest(seed=bad):
                payload = self.payload("EpisodeSpec")
                payload["seed"] = bad
                with self.assertRaises(SchemaError):
                    EpisodeSpec.from_json(payload)

    def test_not_a_dict(self):
        """§3.4 a non-object payload raises SchemaError."""
        for bad in ([], "x", None, 3):
            with self.subTest(payload=bad):
                with self.assertRaises(SchemaError):
                    EpisodeSpec.from_json(bad)

    def test_invariant_violation_raises_value_error(self):
        """§3.2/§3.4 a payload violating EpisodeSpec invariants is rejected with a ValueError."""
        payload = self.payload("EpisodeSpec")
        payload["perturbation"] = None                    # failure without perturbation
        with self.assertRaises(ValueError):
            EpisodeSpec.from_json(payload)


if __name__ == "__main__":
    unittest.main()
