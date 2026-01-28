"""Metrics for evaluating imputation and PRS predictions."""

import numpy as np


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
