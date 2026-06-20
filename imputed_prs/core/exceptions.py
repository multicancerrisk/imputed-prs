"""Custom exceptions for the imputed-prs library."""


class ImputedPRSError(Exception):
    """Base exception for all library errors."""

    pass


class DataLoadError(ImputedPRSError):
    """Failed to load input files (VCF, PRS, manifest)."""

    pass


class ValidationError(ImputedPRSError):
    """Invalid input data (wrong columns, bad values)."""

    pass


class IncompatibleBuildError(ImputedPRSError):
    """Genome build mismatch (GRCh37 vs GRCh38)."""

    pass


class IncompatiblePlatformError(ImputedPRSError):
    """Genotyping platform mismatch (e.g. model trained for 23andme_v5,
    upload declared as a different array)."""

    pass


class MissingVariantsError(ImputedPRSError):
    """Required variants not found in data."""

    pass


class ModelNotFittedError(ImputedPRSError):
    """Predict called before fit."""

    pass
