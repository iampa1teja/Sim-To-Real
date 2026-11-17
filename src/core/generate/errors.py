"""Shared generation errors (v4 contracts §1.5)."""


class GenerationError(Exception):
    """Base error for generation."""


class ConfigError(GenerationError, ValueError):
    """Invalid generation configuration."""


class SchemaError(GenerationError, ValueError):
    """Invalid serialized data."""


class KinematicsError(GenerationError):
    """A kinematic chain could not be constructed."""


class SamplingError(GenerationError):
    """The start sampler exhausted its tries."""


class ScenarioSpaceExhausted(GenerationError):
    """The scenario space cannot supply the requested quota."""


class AttemptBudgetExceeded(GenerationError):
    """Generation exceeded its attempt budget."""


class ResumeError(GenerationError):
    """Generation state cannot be reconciled."""


class DuplicateError(GenerationError):
    """A merge encountered a duplicate."""


class AuditFailed(GenerationError):
    """A passing, current audit is required."""


class EnvError(GenerationError):
    """An environment cannot honour an episode specification."""
