"""
imputed-prs: Calculate Polygenic Risk Scores via linear imputation of missing variants.
"""

__version__ = "0.1.0"

from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

__all__ = [
    "LinearImputationPRS",
]
