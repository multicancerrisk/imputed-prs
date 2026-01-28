"""Models module for imputation models."""

from imputed_prs.core.types import SingleVariantModelResult
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.metrics import compute_cv_r2

__all__ = ["fit_single_variant_model", "SingleVariantModelResult", "compute_cv_r2"]
