"""Custom exceptions used across SatBBA."""


class SatBBAError(Exception):
    """Base exception for SatBBA."""


class ConfigError(SatBBAError):
    """Raised when configuration parsing or validation fails."""


class MatcherError(SatBBAError):
    """Raised for matcher-specific failures."""


class WeightFileNotFoundError(MatcherError):
    """Raised when local weight file required by matcher is missing."""


class DataModelError(SatBBAError):
    """Raised when input data models are inconsistent."""
