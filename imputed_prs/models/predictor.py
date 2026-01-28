"""PRS prediction from user genotypes.

This module provides functions for computing Polygenic Risk Scores from
user genotype data, combining observed variant contributions with
imputed variant predictions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    PredictionResult,
    VariantInfo,
)
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


class PRSPredictor:
    """Full PRS prediction combining observed and imputed components.

    This class orchestrates the complete PRS calculation pipeline:
    1. Compute contribution from directly observed variants
    2. Compute contribution from imputed (missing) variants
    3. Calculate uncertainty (standard error, confidence intervals)
    4. Optionally apply calibration scaling

    Attributes:
        observed_variants: List of VariantInfo for variants on the platform.
        imputed_models: List of ImputedVariantModel for missing variants.
        calibration_params: Optional calibration parameters for scaling.
    """

    def __init__(
        self,
        observed_variants: List[VariantInfo],
        imputed_models: List[ImputedVariantModel],
        calibration_params: Optional[CalibrationParams] = None,
    ):
        """Initialize the predictor.

        Args:
            observed_variants: Variants present on the genotyping platform.
            imputed_models: Trained imputation models for missing variants.
            calibration_params: Optional parameters for calibration scaling.
        """
        self.observed_variants = observed_variants
        self.imputed_models = imputed_models
        self.calibration_params = calibration_params

        # Pre-compute counts
        self._n_observed_variants = len(observed_variants)
        self._n_imputed_variants = len(imputed_models)
        self._n_intercept_only = sum(
            1 for m in imputed_models if m.is_intercept_only
        )

    def predict(
        self,
        user_genotypes: Dict[str, Optional[float]],
        apply_calibration: bool = True,
    ) -> PredictionResult:
        """Compute full PRS with uncertainty quantification.

        Args:
            user_genotypes: Dictionary mapping variant_id to dosage value
                (0.0, 1.0, 2.0) or None for missing variants.
            apply_calibration: Whether to apply calibration scaling
                (requires calibration_params to be set).

        Returns:
            PredictionResult with PRS value, confidence intervals,
            component breakdown, and optionally scaled values.
        """
        # Step 1: Compute observed component
        prs_observed, n_observed_used = compute_observed_prs(
            user_genotypes, self.observed_variants
        )

        # Step 2: Compute imputed component
        prs_imputed, total_variance, n_imputed, n_truncated = compute_imputed_prs(
            user_genotypes, self.imputed_models
        )

        # Step 3: Combine components
        prs_raw = prs_observed + prs_imputed

        # Step 4: Compute standard error and confidence intervals
        se = np.sqrt(total_variance) if total_variance > 0 else 0.0
        ci_lower = prs_raw - 1.96 * se
        ci_upper = prs_raw + 1.96 * se

        # Step 5: Count variants
        n_variants_used = n_observed_used + n_imputed
        n_user_variants_missing = (
            self._n_observed_variants + self._n_imputed_variants
        ) - n_variants_used

        # Step 6: Apply calibration if requested and available
        prs_scaled = None
        se_scaled = None
        ci_lower_scaled = None
        ci_upper_scaled = None

        if apply_calibration and self.calibration_params is not None:
            params = self.calibration_params
            prs_scaled = params.scaling_factor * prs_raw + params.calibration_intercept
            se_scaled = abs(params.scaling_factor) * se
            ci_lower_scaled = prs_scaled - 1.96 * se_scaled
            ci_upper_scaled = prs_scaled + 1.96 * se_scaled

        return PredictionResult(
            prs=prs_raw,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            prs_observed_component=prs_observed,
            prs_imputed_component=prs_imputed,
            n_variants_used=n_variants_used,
            n_variants_imputed=n_imputed,
            n_variants_intercept_only=self._n_intercept_only,
            n_user_variants_missing=n_user_variants_missing,
            n_truncated=n_truncated,
            prs_scaled=prs_scaled,
            se_scaled=se_scaled,
            ci_lower_scaled=ci_lower_scaled,
            ci_upper_scaled=ci_upper_scaled,
        )
