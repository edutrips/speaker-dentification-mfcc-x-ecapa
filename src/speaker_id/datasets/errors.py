"""Exceptions for the dataset manifest contract."""


class ManifestError(Exception):
    """Base class for controlled manifest errors."""


class ManifestValidationError(ManifestError):
    """Raised when a manifest record or CSV structure is invalid."""
