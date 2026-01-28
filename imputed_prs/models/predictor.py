"""PRS prediction from user genotypes.

This module provides functions for computing Polygenic Risk Scores from
user genotype data, combining observed variant contributions with
imputed variant predictions.
"""

from typing import Dict, List, Optional, Tuple

from imputed_prs.core.types import VariantInfo


def compute_observed_prs(
    user_dosages: Dict[str, Optional[float]],
    observed_variants: List[VariantInfo],
) -> Tuple[float, int]:
    """Compute PRS contribution from directly observed variants.

    Calculates sum(z_j * beta_j) for all variants j in the observed set
    where the user has a valid dosage value.

    Args:
        user_dosages: Dictionary mapping variant_id to dosage value
            (0.0, 1.0, 2.0) or None for missing variants.
        observed_variants: List of VariantInfo objects for variants
            present on the genotyping platform.

    Returns:
        Tuple of (prs_observed, n_used):
            - prs_observed: Sum of dosage × beta for observed variants
            - n_used: Count of variants with valid dosages
    """
    total = 0.0
    n_used = 0

    for variant in observed_variants:
        dosage = user_dosages.get(variant.variant_id)
        if dosage is not None:
            total += dosage * variant.beta
            n_used += 1

    return total, n_used
