"""Metrics for evaluating imputation and PRS predictions."""

from typing import Tuple

import numpy as np


def standardize_columns(
    X: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize predictor columns to zero mean and unit variance.

    Standardizing before an ElasticNet fit makes the L1/L2 penalty scale-free
    across predictors (otherwise the effective penalty grows with each column's
    variance, i.e. with MAF for genotype dosages). The returned ``mean``/``scale``
    let callers back-transform the fitted coefficients to the raw-feature scale
    (see :func:`backtransform_linear_model`).

    Columns with (near-)zero variance are left unscaled (``scale = 1.0``) so the
    back-transform never divides by ~0. A constant column then standardizes to all
    zeros and is effectively ignored by the fit.

    Args:
        X: Predictor matrix. Shape: (n_samples, n_predictors).
        eps: Standard deviations below this are treated as zero variance and the
            column's scale is set to 1.0. Default: 1e-8.

    Returns:
        Tuple of (X_std, mean, scale):
            X_std: Standardized matrix, same shape as ``X``.
            mean: Per-column mean. Shape: (n_predictors,).
            scale: Per-column scale (std, or 1.0 for near-constant columns).
                Shape: (n_predictors,).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)  # ddof=0, matching sklearn's StandardScaler
    scale = np.where(std < eps, 1.0, std)
    return (X - mean) / scale, mean, scale


def backtransform_linear_model(
    coef_std: np.ndarray,
    intercept_std: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Map a linear model fitted on standardized columns back to raw-feature scale.

    A model fitted on ``X_std = (X - mean) / scale`` predicts::

        y = coef_std . ((x - mean) / scale) + intercept_std
          = (coef_std / scale) . x + (intercept_std - coef_std . (mean / scale))

    so the raw-scale model is ``coef_raw = coef_std / scale`` and
    ``intercept_raw = intercept_std - sum(coef_std * mean / scale)``. This is an
    algebraic identity: the raw model produces the same predictions as the
    standardized model (up to float rounding), so storage and inference on raw
    dosages are unchanged.

    Args:
        coef_std: Coefficients fitted on standardized columns. Shape: (n_predictors,).
        intercept_std: Intercept fitted on standardized columns.
        mean: Per-column mean used to standardize. Shape: (n_predictors,).
        scale: Per-column scale used to standardize. Shape: (n_predictors,).

    Returns:
        Tuple of (coef_raw, intercept_raw) on the raw-feature scale.
    """
    coef_raw = coef_std / scale
    intercept_raw = float(intercept_std - np.dot(coef_std, mean / scale))
    return coef_raw, intercept_raw


def compute_cv_r2(true_values: np.ndarray, cv_predictions: np.ndarray) -> float:
    """Compute R-squared from cross-validation predictions.

    R² = 1 - SS_res / SS_tot

    Where:
        SS_res = Σ(y_true - y_pred)²
        SS_tot = Σ(y_true - mean(y_true))²

    Args:
        true_values: True target values. Shape: (n_samples,).
        cv_predictions: Cross-validation predicted values. Shape: (n_samples,).

    Returns:
        R-squared value. Can be negative if predictions are worse than
        simply predicting the mean. Returns 0.0 if true_values has zero
        variance (constant values).

    Note:
        Unlike sklearn's r2_score, this function returns 0.0 (not NaN or
        negative) when the true values have zero variance, as this is a
        common edge case in genetic data.
    """
    if len(true_values) == 0:
        return 0.0

    ss_res = np.sum((true_values - cv_predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)

    if ss_tot < 1e-10:
        # Target has zero variance
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)
