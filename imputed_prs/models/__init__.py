"""Models module for imputation models."""

from imputed_prs.core.types import SingleVariantModelResult
from imputed_prs.models.elastic_net import fit_single_variant_model

__all__ = ["fit_single_variant_model", "SingleVariantModelResult"]
