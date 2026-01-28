"""CV-predicted PRS computation and calibration parameter estimation."""

from typing import Dict

import numpy as np


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
