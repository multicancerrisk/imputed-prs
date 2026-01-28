"""Hyperparameter tuning for elastic net imputation models."""

from typing import List, Optional

import numpy as np

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.types import GridSearchResult
from imputed_prs.models.elastic_net import fit_single_variant_model

# Default hyperparameter grids
DEFAULT_L1_RATIOS = [0.1, 0.5, 0.9]
DEFAULT_ALPHAS = [0.001, 0.01, 0.1]


def _evaluate_single_variant(
    Z: np.ndarray,
    target: np.ndarray,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
) -> Optional[float]:
    """Evaluate a single variant with given hyperparameters.

    Args:
        Z: Predictor dosages. Shape: (n_samples, n_predictors).
        target: Target variant dosages. Shape: (n_samples,).
        l1_ratio: ElasticNet L1 ratio parameter.
        alpha: ElasticNet regularization strength.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        CV MSE if model fit successfully and is not intercept-only, None otherwise.
    """
    try:
        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=Z,
            l1_ratio=l1_ratio,
            alpha=alpha,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        # Skip intercept-only models as they don't reflect hyperparameter impact
        if result.is_intercept_only:
            return None
        return result.cv_mse
    except Exception:
        return None


def global_hyperparameter_search(
    Z: np.ndarray,
    X_missing: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    l1_ratios: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> GridSearchResult:
    """Perform grid search over hyperparameters for elastic net imputation.

    Evaluates all combinations of l1_ratio and alpha on a set of variants
    to find optimal global hyperparameters.

    Args:
        Z: Predictor variant dosages. Shape: (n_samples, n_predictors).
        X_missing: Missing variant dosages to impute. Shape: (n_samples, n_missing_variants).
        sample_indices: Indices of variants in X_missing to use for tuning.
            If None, all variants are used.
        l1_ratios: L1 ratio values to search. Default: [0.1, 0.5, 0.9].
        alphas: Alpha values to search. Default: [0.001, 0.01, 0.1].
        cv_folds: Number of cross-validation folds. Default: 5.
        random_state: Random seed for reproducibility. Default: None.

    Returns:
        GridSearchResult with best hyperparameters and full grid results.

    Raises:
        ValidationError: If inputs are invalid (empty grids, shape mismatch,
            invalid indices, or all models fail).
    """
    # Convert to numpy arrays and ensure correct dtype
    Z = np.asarray(Z, dtype=np.float64)
    X_missing = np.asarray(X_missing, dtype=np.float64)

    # Handle 1D case for single variant
    if X_missing.ndim == 1:
        X_missing = X_missing.reshape(-1, 1)

    # Use defaults if not provided
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS.copy()
    if alphas is None:
        alphas = DEFAULT_ALPHAS.copy()

    # Validate grids are non-empty
    if len(l1_ratios) == 0:
        raise ValidationError("l1_ratios cannot be empty")
    if len(alphas) == 0:
        raise ValidationError("alphas cannot be empty")

    # Validate l1_ratio bounds
    for l1 in l1_ratios:
        if not (0.0 <= l1 <= 1.0):
            raise ValidationError(
                f"l1_ratio must be between 0 and 1, got {l1}"
            )

    # Validate alpha bounds
    for a in alphas:
        if a < 0:
            raise ValidationError(f"alpha must be non-negative, got {a}")

    # Validate shapes match
    n_samples_Z = Z.shape[0]
    n_samples_X = X_missing.shape[0]
    if n_samples_Z != n_samples_X:
        raise ValidationError(
            f"Shape mismatch: Z has {n_samples_Z} samples but "
            f"X_missing has {n_samples_X} samples"
        )

    n_missing_variants = X_missing.shape[1]

    # Handle sample_indices
    if sample_indices is None:
        sample_indices = list(range(n_missing_variants))
    else:
        # Validate indices
        for idx in sample_indices:
            if not isinstance(idx, (int, np.integer)):
                raise ValidationError(
                    f"sample_indices must contain integers, got {type(idx)}"
                )
            if idx < 0 or idx >= n_missing_variants:
                raise ValidationError(
                    f"sample_indices contains invalid index {idx}. "
                    f"Valid range: 0 to {n_missing_variants - 1}"
                )

    n_variants_sampled = len(sample_indices)

    # Edge case: no predictors
    n_predictors = Z.shape[1] if Z.ndim == 2 and Z.size > 0 else 0
    if n_predictors == 0:
        # Return defaults with inf MSE
        grid_results = []
        for l1_ratio in l1_ratios:
            for alpha in alphas:
                grid_results.append({
                    "l1_ratio": l1_ratio,
                    "alpha": alpha,
                    "mean_cv_mse": float("inf"),
                    "std_cv_mse": 0.0,
                    "n_variants_evaluated": 0,
                })
        return GridSearchResult(
            best_l1_ratio=l1_ratios[0],
            best_alpha=alphas[0],
            best_mean_cv_mse=float("inf"),
            grid_results=grid_results,
            n_variants_sampled=n_variants_sampled,
            n_variants_failed=n_variants_sampled,
        )

    # Grid search
    grid_results = []
    best_mean_mse = float("inf")
    best_l1_ratio = l1_ratios[0]
    best_alpha = alphas[0]
    total_failed = 0

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            mse_values = []
            n_failed_this_combo = 0

            for var_idx in sample_indices:
                target = X_missing[:, var_idx]
                mse = _evaluate_single_variant(
                    Z=Z,
                    target=target,
                    l1_ratio=l1_ratio,
                    alpha=alpha,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
                if mse is not None:
                    mse_values.append(mse)
                else:
                    n_failed_this_combo += 1

            # Compute statistics for this combination
            if len(mse_values) > 0:
                mean_mse = float(np.mean(mse_values))
                std_mse = float(np.std(mse_values))
            else:
                mean_mse = float("inf")
                std_mse = 0.0

            grid_results.append({
                "l1_ratio": l1_ratio,
                "alpha": alpha,
                "mean_cv_mse": mean_mse,
                "std_cv_mse": std_mse,
                "n_variants_evaluated": len(mse_values),
            })

            # Update best if this is better
            if mean_mse < best_mean_mse:
                best_mean_mse = mean_mse
                best_l1_ratio = l1_ratio
                best_alpha = alpha

    # Count total failures (based on best combo to be consistent)
    # Actually, we need to track failures properly - count variants that failed
    # for ALL hyperparameter combinations
    # For simplicity, we'll count based on the best combination
    for result in grid_results:
        if (result["l1_ratio"] == best_l1_ratio and
                result["alpha"] == best_alpha):
            total_failed = n_variants_sampled - result["n_variants_evaluated"]
            break

    # Check if all models failed
    if best_mean_mse == float("inf"):
        raise ValidationError(
            "All models failed during hyperparameter search. "
            "Check that input data has sufficient variance and valid samples."
        )

    return GridSearchResult(
        best_l1_ratio=best_l1_ratio,
        best_alpha=best_alpha,
        best_mean_cv_mse=best_mean_mse,
        grid_results=grid_results,
        n_variants_sampled=n_variants_sampled,
        n_variants_failed=total_failed,
    )
