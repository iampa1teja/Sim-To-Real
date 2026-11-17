"""WP0: the enums of ``core.generate.protocols`` (contracts §3.1)."""

import enum
import json
import unittest

from core.generate.protocols import (AttemptStatus, Kind, PerturbationGroup, RejectionReason, SegmentLabel,
                                     SkipReason)
from core.utils.hashing import canonical_json

EXPECTED = {
    Kind: {"SUCCESS": "success", "FAILURE": "failure", "RECOVERY": "recovery"},
    SegmentLabel: {"NOMINAL": "nominal", "PERTURBATION": "perturbation", "FAILURE": "failure", "RECOVERY": "recovery"},
    AttemptStatus: {"ACCEPTED": "accepted", "REJECTED": "rejected", "SKIPPED": "skipped"},
    SkipReason: {"SCENARIO_SEEN": "scenario_seen", "SAMPLING_FAILED": "sampling_failed"},
    RejectionReason: {name: name.lower() for name in (
        "EXPERT_FAILED", "UNPLANNED_REGRESSION", "PERTURBATION_NOT_APPLIED", "PERTURBATION_INEFFECTIVE",
        "NO_REGRESSION", "RECOVERY_FAILED", "DUPLICATE_TRAJECTORY", "INVALID_STATE", "ENV_ERROR")},
    PerturbationGroup: {"EXPERT": "expert", "WORLD": "world"},
}


class EnumTests(unittest.TestCase):
    """WP0 §3.1 enums."""

    def test_enums_are_str_enums(self):
        """§3.1 each enum is `class X(str, Enum)`."""
        for cls in EXPECTED:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, enum.Enum))
                self.assertTrue(issubclass(cls, str))

    def test_members_and_values_exactly(self):
        """§3.1 exact member names and on-disk values, in contract order, nothing extra."""
        for cls, members in EXPECTED.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual([(m.name, m.value) for m in cls], list(members.items()))

    def test_value_lookup_and_string_equality(self):
        """§3.1 the value is the on-disk string: X(value) is the member and member == value."""
        for cls, members in EXPECTED.items():
            for name, value in members.items():
                with self.subTest(cls=cls.__name__, name=name):
                    member = cls[name]
                    self.assertIs(cls(value), member)
                    self.assertEqual(member, value)
                    self.assertEqual(member.value, value)
                    self.assertIs(type(member.value), str)

    def test_unknown_value_raises_value_error(self):
        """§3.1 unknown on-disk strings are rejected."""
        for cls in EXPECTED:
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(ValueError):
                    cls("not_a_member")

    def test_json_serialises_to_value(self):
        """§3.4 enums become their value in JSON."""
        self.assertEqual(json.loads(canonical_json({"k": Kind.FAILURE})), {"k": "failure"})
        self.assertEqual(canonical_json([RejectionReason.ENV_ERROR]), '["env_error"]')

    def test_skip_and_rejection_values_are_disjoint(self):
        """§3.2 EpisodeResult.reason is SkipReason | RejectionReason, so the value sets must not overlap."""
        self.assertFalse({m.value for m in SkipReason} & {m.value for m in RejectionReason})


if __name__ == "__main__":
    unittest.main()
