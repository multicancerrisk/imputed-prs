"""Hyperparameter tuning for elastic net imputation models."""

import time
from typing import List, Optional, Tuple

import numpy as np

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.types import GridSearchResult, OptunaSearchResult
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


def optuna_hyperparameter_search(
    Z: np.ndarray,
    X_missing: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    n_trials: int = 50,
    l1_ratio_range: Tuple[float, float] = (0.0, 1.0),
    alpha_range: Tuple[float, float] = (1e-4, 1.0),
    cv_folds: int = 5,
    seed: Optional[int] = None,
    timeout: Optional[float] = None,
    show_progress: bool = False,
) -> OptunaSearchResult:
    """Bayesian hyperparameter optimization using Optuna TPE sampler.

    Uses Tree-structured Parzen Estimator (TPE) for efficient hyperparameter
    search over continuous parameter ranges.

    Args:
        Z: Predictor variant dosages. Shape: (n_samples, n_predictors).
        X_missing: Missing variant dosages. Shape: (n_samples, n_missing_variants).
        sample_indices: Indices of variants to use for tuning. Default: all.
        n_trials: Number of Optuna trials. Default: 50.
        l1_ratio_range: (min, max) for L1 ratio search. Default: (0.0, 1.0).
        alpha_range: (min, max) for alpha search (log-uniform). Default: (1e-4, 1.0).
        cv_folds: Number of CV folds. Default: 5.
        seed: Random seed for reproducibility. Default: None.
        timeout: Maximum optimization time in seconds. Default: None.
        show_progress: Show Optuna progress bar. Default: False.

    Returns:
        OptunaSearchResult with best parameters and trial history.

    Raises:
        ImportError: If optuna is not installed.
        ValidationError: If inputs are invalid.
    """
    # Lazy import of optuna
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "Optuna is required for Bayesian hyperparameter optimization. "
            "Install it with: pip install optuna"
        ) from e

    # Convert to numpy arrays and ensure correct dtype
    Z = np.asarray(Z, dtype=np.float64)
    X_missing = np.asarray(X_missing, dtype=np.float64)

    # Handle 1D case for single variant
    if X_missing.ndim == 1:
        X_missing = X_missing.reshape(-1, 1)

    # Validate l1_ratio_range
    if len(l1_ratio_range) != 2:
        raise ValidationError("l1_ratio_range must be a tuple of (min, max)")
    l1_min, l1_max = l1_ratio_range
    if not (0.0 <= l1_min <= l1_max <= 1.0):
        raise ValidationError(
            f"l1_ratio_range must satisfy 0 <= min <= max <= 1, "
            f"got ({l1_min}, {l1_max})"
        )

    # Validate alpha_range
    if len(alpha_range) != 2:
        raise ValidationError("alpha_range must be a tuple of (min, max)")
    alpha_min, alpha_max = alpha_range
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValidationError(
            f"alpha_range values must be positive, got ({alpha_min}, {alpha_max})"
        )
    if alpha_min > alpha_max:
        raise ValidationError(
            f"alpha_range min must be <= max, got ({alpha_min}, {alpha_max})"
        )

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
        return OptunaSearchResult(
            best_l1_ratio=l1_min,
            best_alpha=alpha_min,
            best_mean_cv_mse=float("inf"),
            n_trials=0,
            n_variants_sampled=n_variants_sampled,
            n_variants_failed=n_variants_sampled,
            trial_history=[],
            optimization_time_seconds=0.0,
        )

    # Trial history tracking
    trial_history: List[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters
        l1_ratio = trial.suggest_float("l1_ratio", l1_min, l1_max)
        alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)

        # Evaluate all sampled variants
        mse_values = []
        for var_idx in sample_indices:
            target = X_missing[:, var_idx]
            mse = _evaluate_single_variant(
                Z=Z,
                target=target,
                l1_ratio=l1_ratio,
                alpha=alpha,
                cv_folds=cv_folds,
                random_state=seed,
            )
            if mse is not None:
                mse_values.append(mse)

        # Compute mean MSE
        if len(mse_values) > 0:
            mean_mse = float(np.mean(mse_values))
        else:
            mean_mse = float("inf")

        # Store trial details
        trial_history.append({
            "trial_number": trial.number,
            "l1_ratio": l1_ratio,
            "alpha": alpha,
            "mean_cv_mse": mean_mse,
            "n_variants_evaluated": len(mse_values),
        })

        return mean_mse

    # Create sampler with seed for reproducibility
    sampler = TPESampler(seed=seed)

    # Configure optuna logging
    if not show_progress:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Run optimization
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
    )
    optimization_time = time.time() - start_time

    # Extract best parameters
    if study.best_trial is not None:
        best_l1_ratio = study.best_trial.params["l1_ratio"]
        best_alpha = study.best_trial.params["alpha"]
        best_mean_cv_mse = study.best_value

        # Count failures at best parameters
        best_trial_entry = None
        for entry in trial_history:
            if entry["trial_number"] == study.best_trial.number:
                best_trial_entry = entry
                break

        if best_trial_entry is not None:
            n_variants_failed = n_variants_sampled - best_trial_entry["n_variants_evaluated"]
        else:
            n_variants_failed = 0
    else:
        # No trials completed (shouldn't happen normally)
        best_l1_ratio = l1_min
        best_alpha = alpha_min
        best_mean_cv_mse = float("inf")
        n_variants_failed = n_variants_sampled

    return OptunaSearchResult(
        best_l1_ratio=best_l1_ratio,
        best_alpha=best_alpha,
        best_mean_cv_mse=best_mean_cv_mse,
        n_trials=len(study.trials),
        n_variants_sampled=n_variants_sampled,
        n_variants_failed=n_variants_failed,
        trial_history=trial_history,
        optimization_time_seconds=optimization_time,
    )
