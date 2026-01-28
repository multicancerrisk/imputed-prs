"""Per-variant imputation quality summary."""

from typing import Any, Dict

import numpy as np

from imputed_prs.core.types import ImputedVariantModel


def summarize_imputation_quality(
    models: Dict[str, ImputedVariantModel],
) -> Dict[str, Any]:
    """Summarize imputation quality across all trained variant models.

    Categorizes variants by imputation R² quality tiers, computes weighted
    mean R² (weighted by |beta|), and identifies intercept-only models.

    Args:
        models: Dictionary mapping variant_id to ImputedVariantModel.

    Returns:
        Dictionary with quality summary:
        - n_total: Total number of models
        - n_excellent: Count with R² > 0.8
        - n_good: Count with 0.6 < R² <= 0.8
        - n_moderate: Count with 0.4 < R² <= 0.6
        - n_poor: Count with R² <= 0.4
        - n_intercept_only: Count of intercept-only models
        - mean_r2: Unweighted mean R² across all models
        - weighted_mean_r2: Mean R² weighted by |beta|
        - median_r2: Median R² value
        - min_r2: Minimum R² value
        - max_r2: Maximum R² value

    Raises:
        ValueError: If models dict is empty.
    """
    if not models:
        raise ValueError("Models dict cannot be empty")

    r2_values = []
    betas = []
    n_intercept_only = 0

    for model in models.values():
        r2_values.append(model.imputation_r2)
        betas.append(abs(model.beta))
        if model.is_intercept_only:
            n_intercept_only += 1

    r2_arr = np.array(r2_values)
    beta_arr = np.array(betas)

    # Quality tier counts
    n_excellent = int(np.sum(r2_arr > 0.8))
    n_good = int(np.sum((r2_arr > 0.6) & (r2_arr <= 0.8)))
    n_moderate = int(np.sum((r2_arr > 0.4) & (r2_arr <= 0.6)))
    n_poor = int(np.sum(r2_arr <= 0.4))

    # Weighted mean R² (by |beta|)
    total_weight = np.sum(beta_arr)
    if total_weight > 0:
        weighted_mean_r2 = float(np.sum(r2_arr * beta_arr) / total_weight)
    else:
        weighted_mean_r2 = float(np.mean(r2_arr))

    return {
        "n_total": len(models),
        "n_excellent": n_excellent,
        "n_good": n_good,
        "n_moderate": n_moderate,
        "n_poor": n_poor,
        "n_intercept_only": n_intercept_only,
        "mean_r2": float(np.mean(r2_arr)),
        "weighted_mean_r2": weighted_mean_r2,
        "median_r2": float(np.median(r2_arr)),
        "min_r2": float(np.min(r2_arr)),
        "max_r2": float(np.max(r2_arr)),
    }
