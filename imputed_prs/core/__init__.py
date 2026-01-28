"""Core module containing data types, exceptions, and harmonization functions."""

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    GenotypeData,
    GridSearchResult,
    ImputedVariantModel,
    PlatformInfo,
    PredictionResult,
    VariantInfo,
)
from imputed_prs.core.exceptions import (
    DataLoadError,
    ImputedPRSError,
    IncompatibleBuildError,
    MissingVariantsError,
    ModelNotFittedError,
    ValidationError,
)
from imputed_prs.core.harmonizer import (
    align_effect_alleles,
    filter_to_local_window,
    partition_variants,
    validate_genome_build,
    AlleleAlignmentResult,
    BuildValidationResult,
    PartitionResult,
    WindowFilterResult,
)

__all__ = [
    # Types
    "CalibrationParams",
    "EvaluationMetrics",
    "GenotypeData",
    "GridSearchResult",
    "ImputedVariantModel",
    "PlatformInfo",
    "PredictionResult",
    "VariantInfo",
    # Exceptions
    "DataLoadError",
    "ImputedPRSError",
    "IncompatibleBuildError",
    "MissingVariantsError",
    "ModelNotFittedError",
    "ValidationError",
    # Harmonization functions
    "align_effect_alleles",
    "filter_to_local_window",
    "partition_variants",
    "validate_genome_build",
    # Harmonization result types
    "AlleleAlignmentResult",
    "BuildValidationResult",
    "PartitionResult",
    "WindowFilterResult",
]
