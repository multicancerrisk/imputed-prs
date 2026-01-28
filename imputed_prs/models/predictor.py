"""PRS prediction from user genotypes.

This module provides functions for computing Polygenic Risk Scores from
user genotype data, combining observed variant contributions with
imputed variant predictions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from imputed_prs.core.types import ImputedVariantModel, VariantInfo
from imputed_prs.models.bounding import clip_and_adjust_variance


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


def compute_imputed_prs(
    user_dosages: Dict[str, Optional[float]],
    imputed_models: List[ImputedVariantModel],
) -> Tuple[float, float, int, int]:
    """Compute PRS contribution from imputed variants.

    For each imputed variant model:
    1. Gather predictor dosages from user data
    2. Compute raw prediction: x̂_j = z[L_j]' @ w_j + γ_j
    3. Apply dosage clipping with variance adjustment
    4. Multiply by beta

    Args:
        user_dosages: Dictionary mapping variant_id to dosage value
            (0.0, 1.0, 2.0) or None for missing variants.
        imputed_models: List of ImputedVariantModel objects for variants
            that need to be imputed.

    Returns:
        Tuple of (prs_imputed, total_variance, n_imputed, n_truncated):
            - prs_imputed: Sum of imputed_dosage × beta for imputed variants
            - total_variance: Sum of beta² × adjusted_residual_variance
            - n_imputed: Count of variants successfully imputed
            - n_truncated: Count of variants where dosage was clipped
    """
    total_prs = 0.0
    total_variance = 0.0
    n_imputed = 0
    n_truncated = 0

    for model in imputed_models:
        # Compute raw prediction
        if model.is_intercept_only:
            raw_prediction = model.intercept
        else:
            # Gather predictor dosages
            predictor_dosages = []
            all_predictors_available = True
            for pred_id in model.predictor_variant_ids:
                dosage = user_dosages.get(pred_id)
                if dosage is None:
                    all_predictors_available = False
                    break
                predictor_dosages.append(dosage)

            if not all_predictors_available:
                # Fall back to intercept-only (mean imputation)
                raw_prediction = model.intercept
            else:
                # Compute: x̂ = z' @ w + γ
                predictor_array = np.array(predictor_dosages)
                raw_prediction = np.dot(predictor_array, model.coefficients) + model.intercept

        # Apply clipping and variance adjustment
        clipped_dosage, adjusted_variance = clip_and_adjust_variance(
            raw_prediction, model.residual_variance
        )

        # Track truncation
        if clipped_dosage != raw_prediction:
            n_truncated += 1

        # Add contribution to PRS
        total_prs += clipped_dosage * model.beta

        # Add variance contribution: beta² × residual_variance
        total_variance += (model.beta ** 2) * adjusted_variance

        n_imputed += 1

    return total_prs, total_variance, n_imputed, n_truncated
