"""imputed-prs: Polygenic Risk Score calculation with linear imputation.

This library enables calculation of Polygenic Risk Scores (PRS) on genotyping
platform data by performing linear imputation of missing variants.

Example:
    >>> from imputed_prs import (
    ...     LinearImputationPRS,
    ...     list_available_platforms,
    ...     search_pgs_catalog,
    ... )
    >>> model = LinearImputationPRS()
    >>> platforms = list_available_platforms()
    >>> scores = search_pgs_catalog("breast cancer")
"""

__version__ = "0.1.0"

# Main API classes
from imputed_prs.core import LinearImputationPRS, LinearProjectionPRS

# Convenience functions for platform discovery
from imputed_prs.io import (
    list_available_platforms,
    get_platform_info,
)

# Convenience functions for PGS Catalog
from imputed_prs.io import (
    search_pgs_catalog,
    fetch_pgs_catalog_score,
    clear_pgs_catalog_cache,
)

# Evaluation tools
from imputed_prs.evaluation import ImputationEvaluator

# Core types commonly needed by users
from imputed_prs.core import (
    PlatformInfo,
    PredictionResult,
)

# Base exception for catching all library errors
from imputed_prs.core import ImputedPRSError

__all__ = [
    # Version
    "__version__",
    # Main API
    "LinearImputationPRS",
    "LinearProjectionPRS",
    # Platform functions
    "list_available_platforms",
    "get_platform_info",
    # PGS Catalog functions
    "search_pgs_catalog",
    "fetch_pgs_catalog_score",
    "clear_pgs_catalog_cache",
    # Evaluation
    "ImputationEvaluator",
    # Types
    "PlatformInfo",
    "PredictionResult",
    # Exceptions
    "ImputedPRSError",
]
