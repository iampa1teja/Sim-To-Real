"""WP0: the exception hierarchy of ``core.generate.errors`` (contracts §1.5)."""

import inspect
import unittest

from core.generate import errors
from core.generate.errors import (AttemptBudgetExceeded, AuditFailed, ConfigError, DuplicateError, EnvError,
                                  GenerationError, KinematicsError, ResumeError, SamplingError, SchemaError,
                                  ScenarioSpaceExhausted)

EXPECTED_BASES = {
    "GenerationError": (Exception,),
    "ConfigError": (GenerationError, ValueError),
    "SchemaError": (GenerationError, ValueError),
    "KinematicsError": (GenerationError,),
    "SamplingError": (GenerationError,),
    "ScenarioSpaceExhausted": (GenerationError,),
    "AttemptBudgetExceeded": (GenerationError,),
    "ResumeError": (GenerationError,),
    "DuplicateError": (GenerationError,),
    "AuditFailed": (GenerationError,),
    "EnvError": (GenerationError,),
}


class ErrorHierarchyTests(unittest.TestCase):
    """WP0 §1.5 errors."""

    def test_every_error_class_has_exactly_the_contract_bases(self):
        """§1.5 each error class has exactly the listed bases, in order."""
        for name, bases in EXPECTED_BASES.items():
            with self.subTest(name=name):
                cls = getattr(errors, name)
                self.assertTrue(inspect.isclass(cls))
                self.assertEqual(cls.__bases__, bases)

    def test_module_defines_no_other_public_exception(self):
        """§1.5/§2 errors.py defines exactly the eleven public error classes (no extra public names)."""
        defined = {name for name, obj in vars(errors).items()
                   if not name.startswith("_") and inspect.isclass(obj) and obj.__module__ == errors.__name__}
        self.assertEqual(defined, set(EXPECTED_BASES))

    def test_all_errors_are_generation_errors(self):
        """§1.5 GenerationError is the base of all v4 errors."""
        for cls in (ConfigError, SchemaError, KinematicsError, SamplingError, ScenarioSpaceExhausted,
                    AttemptBudgetExceeded, ResumeError, DuplicateError, AuditFailed, EnvError):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, GenerationError))
                with self.assertRaises(GenerationError):
                    raise cls("boom")

    def test_config_and_schema_errors_are_value_errors(self):
        """§1.5/§14.1 ConfigError and SchemaError are caught by existing `except ValueError`."""
        for cls in (ConfigError, SchemaError):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(ValueError):
                    raise cls("bad")

    def test_other_errors_are_not_value_errors(self):
        """§1.5 only ConfigError and SchemaError subclass ValueError."""
        for cls in (GenerationError, KinematicsError, SamplingError, ScenarioSpaceExhausted, AttemptBudgetExceeded,
                    ResumeError, DuplicateError, AuditFailed, EnvError):
            with self.subTest(cls=cls.__name__):
                self.assertFalse(issubclass(cls, ValueError))
                self.assertFalse(issubclass(cls, TypeError))

    def test_errors_keep_message_and_are_distinct(self):
        """§1.5 the classes are distinct types that carry their message."""
        classes = [getattr(errors, name) for name in EXPECTED_BASES]
        self.assertEqual(len(set(classes)), len(classes))
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(str(cls("message 1")), "message 1")
        self.assertFalse(issubclass(SchemaError, ConfigError))
        self.assertFalse(issubclass(ConfigError, SchemaError))


if __name__ == "__main__":
    unittest.main()
