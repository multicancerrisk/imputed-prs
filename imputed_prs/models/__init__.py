"""Models module for imputation models."""

from imputed_prs.core.types import GridSearchResult, SingleVariantModelResult
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.metrics import compute_cv_r2
from imputed_prs.models.tuning import global_hyperparameter_search

__all__ = [
    "fit_single_variant_model",
    "SingleVariantModelResult",
    "compute_cv_r2",
    "global_hyperparameter_search",
    "GridSearchResult",
]
