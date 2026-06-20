"""Core module containing data types, exceptions, and harmonization functions."""

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    GenotypeData,
    GridSearchResult,
    ImputedVariantModel,
    OptunaSearchResult,
    PlatformInfo,
    PredictionResult,
    ProjectionRegionModel,
    ProjectionTrainingResult,
    TrainingFailure,
    TrainingResult,
    VariantInfo,
)
from imputed_prs.core.exceptions import (
    DataLoadError,
    ImputedPRSError,
    IncompatibleBuildError,
    IncompatiblePlatformError,
    MissingVariantsError,
    ModelNotFittedError,
    ValidationError,
)
from imputed_prs.core.harmonizer import (
    align_effect_alleles,
    check_predict_compatibility,
    filter_to_local_window,
    partition_variants,
    validate_genome_build,
    AlleleAlignmentResult,
    BuildValidationResult,
    PartitionResult,
    WindowFilterResult,
)
from imputed_prs.core.regions import (
    GenomicRegion,
    RegionDecompositionResult,
    merge_variant_windows,
)
from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
from imputed_prs.core.linear_projection_prs import LinearProjectionPRS

__all__ = [
    # Types
    "CalibrationParams",
    "EvaluationMetrics",
    "GenotypeData",
    "GridSearchResult",
    "ImputedVariantModel",
    "OptunaSearchResult",
    "PlatformInfo",
    "PredictionResult",
    "ProjectionRegionModel",
    "ProjectionTrainingResult",
    "TrainingFailure",
    "TrainingResult",
    "VariantInfo",
    # Exceptions
    "DataLoadError",
    "ImputedPRSError",
    "IncompatibleBuildError",
    "IncompatiblePlatformError",
    "MissingVariantsError",
    "ModelNotFittedError",
    "ValidationError",
    # Harmonization functions
    "align_effect_alleles",
    "check_predict_compatibility",
    "filter_to_local_window",
    "partition_variants",
    "validate_genome_build",
    # Harmonization result types
    "AlleleAlignmentResult",
    "BuildValidationResult",
    "PartitionResult",
    "WindowFilterResult",
    # Region decomposition
    "GenomicRegion",
    "RegionDecompositionResult",
    "merge_variant_windows",
    # Main API classes
    "LinearImputationPRS",
    "LinearProjectionPRS",
]
