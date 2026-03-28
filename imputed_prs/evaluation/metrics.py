"""PRS evaluation metrics computation."""

from typing import Dict, List

import numpy as np
from scipy import stats

from imputed_prs.core.types import EvaluationMetrics


def compute_prs_metrics(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
) -> EvaluationMetrics:
    """Compute comprehensive metrics comparing imputed vs true PRS.

    Args:
        s_imputed: Imputed PRS values (n_samples,). NaN values are excluded.
        s_true: True PRS values (n_samples,). NaN values are excluded.

    Returns:
        EvaluationMetrics with correlation, R², MAE, RMSE, Spearman rho,
        and calibration slope/intercept.

    Raises:
        ValueError: If fewer than 3 valid (non-NaN) samples remain after filtering.
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(s_imputed) | np.isnan(s_true))
    s_imp = s_imputed[valid_mask]
    s_tru = s_true[valid_mask]

    n = len(s_imp)
    if n < 3:
        raise ValueError(f"Need at least 3 valid samples, got {n}")

    # Pearson correlation
    correlation = np.corrcoef(s_imp, s_tru)[0, 1]

    # R² (squared correlation for simple case)
    r2 = correlation**2

    # Error metrics
    residuals = s_imp - s_tru
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    # Spearman rank correlation
    spearman_rho, _ = stats.spearmanr(s_imp, s_tru)

    # Calibration regression: S_true = a + b * S_imputed
    slope, intercept, _, _, _ = stats.linregress(s_imp, s_tru)

    return EvaluationMetrics(
        correlation=correlation,
        r2=r2,
        mae=mae,
        rmse=rmse,
        spearman_rho=spearman_rho,
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def compute_percentile_concordance(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
    percentiles: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute top/bottom percentile concordance between imputed and true PRS.

    For each percentile threshold, calculates what fraction of individuals
    in the top/bottom X% by imputed PRS are also in the top/bottom X% by true PRS.

    Args:
        s_imputed: Imputed PRS values (n_samples,). NaN values are excluded.
        s_true: True PRS values (n_samples,). NaN values are excluded.
        percentiles: List of percentile thresholds to evaluate (default [1, 5, 10]).

    Returns:
        Dict with keys like 'top_1_concordance', 'bottom_5_concordance', etc.,
        plus 'quintile_kappa' for Cohen's kappa on quintile assignments.

    Raises:
        ValueError: If fewer than 20 valid samples (minimum for meaningful quintiles).
    """
    # Filter NaN values
    valid_mask = ~(np.isnan(s_imputed) | np.isnan(s_true))
    s_imp = s_imputed[valid_mask]
    s_tru = s_true[valid_mask]

    n = len(s_imp)
    if n < 20:
        raise ValueError(f"Need at least 20 valid samples for concordance, got {n}")

    result: Dict[str, float] = {}

    for p in percentiles:
        # Top percentile
        top_threshold_imp = np.percentile(s_imp, 100 - p)
        top_threshold_tru = np.percentile(s_tru, 100 - p)
        top_imp = set(np.where(s_imp >= top_threshold_imp)[0])
        top_tru = set(np.where(s_tru >= top_threshold_tru)[0])
        top_concordance = len(top_imp & top_tru) / len(top_imp) if top_imp else 0.0
        result[f"top_{p}_concordance"] = top_concordance

        # Bottom percentile
        bot_threshold_imp = np.percentile(s_imp, p)
        bot_threshold_tru = np.percentile(s_tru, p)
        bot_imp = set(np.where(s_imp <= bot_threshold_imp)[0])
        bot_tru = set(np.where(s_tru <= bot_threshold_tru)[0])
        bot_concordance = len(bot_imp & bot_tru) / len(bot_imp) if bot_imp else 0.0
        result[f"bottom_{p}_concordance"] = bot_concordance

    # Quintile kappa (Cohen's kappa for 5-category agreement)
    quintile_imp = np.digitize(s_imp, np.percentile(s_imp, [20, 40, 60, 80]))
    quintile_tru = np.digitize(s_tru, np.percentile(s_tru, [20, 40, 60, 80]))
    result["quintile_kappa"] = _cohens_kappa(quintile_imp, quintile_tru)

    return result


def _cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa for two categorical arrays."""
    n = len(y1)
    categories = np.unique(np.concatenate([y1, y2]))

    # Observed agreement
    p_o = np.mean(y1 == y2)

    # Expected agreement by chance
    p_e = sum((np.sum(y1 == c) / n) * (np.sum(y2 == c) / n) for c in categories)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)
