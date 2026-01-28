"""CV-predicted PRS computation and calibration parameter estimation."""

from typing import Dict

import numpy as np
from scipy import stats

from imputed_prs.core.types import CalibrationParams


def compute_cv_predicted_prs(
    X: np.ndarray,
    observed_variant_indices: np.ndarray,
    observed_betas: np.ndarray,
    cv_predictions: Dict[int, np.ndarray],
    missing_betas: np.ndarray,
) -> np.ndarray:
    """Compute CV-predicted PRS for each individual.

    Computes: S_i^CV = Σ(observed) x_ij * β_j + Σ(missing) x̂_ij^(-f) * β_j

    Where:
    - x_ij = true genotype dosage for individual i, variant j
    - x̂_ij^(-f) = cross-validated prediction (out-of-fold) for missing variant
    - β_j = effect weight from PRS definition

    Using out-of-fold predictions prevents overfitting when comparing to true PRS.

    Args:
        X: Genotype dosage matrix (n_samples, n_variants).
        observed_variant_indices: Indices into X for observed variants.
        observed_betas: Effect weights for observed variants (aligned with indices).
        cv_predictions: Dict mapping missing variant index to (n_samples,) CV predictions.
        missing_betas: Effect weights for missing variants (aligned with cv_predictions keys).

    Returns:
        Array of shape (n_samples,) with CV-predicted PRS values.
        Samples with any NaN predictions will have NaN PRS.
    """
    n_samples = X.shape[0]
    s_cv = np.zeros(n_samples)

    # Observed component: true genotypes × betas
    if len(observed_variant_indices) > 0:
        X_observed = X[:, observed_variant_indices]
        s_cv += X_observed @ observed_betas

    # Imputed component: CV predictions × betas
    for i, (var_idx, cv_pred) in enumerate(cv_predictions.items()):
        beta = missing_betas[i]
        s_cv += cv_pred * beta

    return s_cv


def estimate_cv_calibration(
    s_cv: np.ndarray,
    s_true: np.ndarray,
) -> CalibrationParams:
    """Estimate calibration parameters by regressing true PRS on CV-predicted PRS.

    Fits the model: S_true = α + β * S_cv

    The scaling factor β can be used to correct for attenuation in predictions.
    When imputation is imperfect, S_cv will have lower variance than S_true,
    and β > 1 will scale predictions back up.

    Args:
        s_cv: CV-predicted PRS values (n_samples,). NaN values are excluded.
        s_true: True PRS values (n_samples,). NaN values are excluded.

    Returns:
        CalibrationParams with regression results and summary statistics.

    Raises:
        ValueError: If fewer than 3 valid (non-NaN) samples remain after filtering.
    """
    # Filter out NaN values (use samples where both are valid)
    valid_mask = ~(np.isnan(s_cv) | np.isnan(s_true))
    s_cv_valid = s_cv[valid_mask]
    s_true_valid = s_true[valid_mask]

    n = len(s_cv_valid)
    if n < 3:
        raise ValueError(f"Need at least 3 valid samples, got {n}")

    # Linear regression: S_true = α + β * S_cv
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        s_cv_valid, s_true_valid
    )

    # Compute standard deviations
    sd_cv = np.std(s_cv_valid, ddof=1)
    sd_true = np.std(s_true_valid, ddof=1)

    # Scaled predictions: β * S_cv
    sd_scaled = abs(slope) * sd_cv if sd_cv > 0 else 0.0

    # Attenuation factor: how much variance is attenuated
    attenuation = sd_cv / sd_true if sd_true > 0 else 0.0

    return CalibrationParams(
        scaling_factor=slope,
        scaling_factor_se=std_err,
        calibration_intercept=intercept,
        calibration_r2=r_value**2,
        sd_cv_predicted=sd_cv,
        sd_true=sd_true,
        sd_scaled=sd_scaled,
        attenuation_factor=attenuation,
        n_calibration=n,
    )
