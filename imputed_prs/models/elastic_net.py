"""Elastic net imputation model for single variant imputation."""

from typing import Optional

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from imputed_prs.core.types import SingleVariantModelResult
from imputed_prs.models.metrics import compute_cv_r2


def _fit_intercept_only_model(
    target_dosages: np.ndarray,
    n_predictors: int,
    l1_ratio: float,
    alpha: float,
) -> SingleVariantModelResult:
    """Fit an intercept-only model (fallback when no predictors available).

    Args:
        target_dosages: Target variant dosages. Shape: (n_samples,).
        n_predictors: Number of predictors (for metadata, typically 0).
        l1_ratio: L1 ratio parameter (stored in result).
        alpha: Alpha parameter (stored in result).

    Returns:
        SingleVariantModelResult with intercept set to mean of valid targets.
    """
    n_samples = len(target_dosages)

    # Find valid (non-NaN) samples
    valid_mask = ~np.isnan(target_dosages)
    valid_targets = target_dosages[valid_mask]

    if len(valid_targets) == 0:
        # All targets are NaN - use 0 as intercept
        intercept = 0.0
    else:
        intercept = float(np.mean(valid_targets))

    # CV predictions are constant intercept for valid samples, NaN for invalid
    cv_predictions = np.full(n_samples, np.nan)
    cv_predictions[valid_mask] = intercept

    # MSE is variance of target (prediction is mean)
    if len(valid_targets) > 0:
        cv_mse = float(np.var(valid_targets))
    else:
        cv_mse = 0.0

    return SingleVariantModelResult(
        coefficients=np.array([]),
        intercept=intercept,
        cv_predictions=cv_predictions,
        cv_mse=cv_mse,
        cv_r2=0.0,
        is_intercept_only=True,
        n_predictors=n_predictors,
        n_samples=n_samples,
        l1_ratio=l1_ratio,
        alpha=alpha,
    )


def fit_single_variant_model(
    target_dosages: np.ndarray,
    predictor_dosages: np.ndarray,
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> SingleVariantModelResult:
    """Fit ElasticNet imputation model for a single target variant.

    Uses cross-validation to collect out-of-fold predictions for all samples,
    which are needed for downstream calibration. The final model is fit on
    all valid samples.

    Args:
        target_dosages: Target variant dosages to predict. Shape: (n_samples,).
            Values should be 0-2 (allele counts). NaN values are allowed and
            will be excluded from fitting.
        predictor_dosages: Predictor variant dosages. Shape: (n_samples, n_predictors).
            Values should be 0-2 (allele counts). Samples with any NaN values
            in predictors will be excluded from fitting.
        l1_ratio: ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).
            Default: 0.5.
        alpha: Regularization strength. Larger values mean more regularization.
            Default: 0.01.
        cv_folds: Number of cross-validation folds. Default: 5.
        random_state: Random seed for reproducibility. Default: None.

    Returns:
        SingleVariantModelResult containing model coefficients, intercept,
        out-of-fold predictions, and performance metrics.

    Raises:
        ValueError: If target_dosages and predictor_dosages have incompatible
            shapes (different number of samples).
    """
    # Convert to float64 for numerical stability
    target_dosages = np.asarray(target_dosages, dtype=np.float64)
    predictor_dosages = np.asarray(predictor_dosages, dtype=np.float64)

    # Handle 1D predictor case (single predictor)
    if predictor_dosages.ndim == 1:
        predictor_dosages = predictor_dosages.reshape(-1, 1)

    n_samples = len(target_dosages)
    n_predictors = predictor_dosages.shape[1] if predictor_dosages.size > 0 else 0

    # Input validation
    if predictor_dosages.size > 0 and predictor_dosages.shape[0] != n_samples:
        raise ValueError(
            f"Shape mismatch: target has {n_samples} samples but "
            f"predictors have {predictor_dosages.shape[0]} samples"
        )

    # Edge case: no predictors
    if n_predictors == 0:
        return _fit_intercept_only_model(target_dosages, n_predictors, l1_ratio, alpha)

    # Create valid mask: non-NaN target AND no NaN in any predictor
    valid_target_mask = ~np.isnan(target_dosages)
    valid_predictor_mask = ~np.any(np.isnan(predictor_dosages), axis=1)
    valid_mask = valid_target_mask & valid_predictor_mask

    n_valid = np.sum(valid_mask)

    # Edge case: too few valid samples for CV
    if n_valid < cv_folds:
        return _fit_intercept_only_model(target_dosages, n_predictors, l1_ratio, alpha)

    # Extract valid data
    y_valid = target_dosages[valid_mask]
    X_valid = predictor_dosages[valid_mask]

    # Edge case: zero variance in target
    if np.std(y_valid) < 1e-10:
        return _fit_intercept_only_model(target_dosages, n_predictors, l1_ratio, alpha)

    # Initialize CV predictions array (NaN for invalid samples)
    cv_predictions = np.full(n_samples, np.nan)

    # Cross-validation loop
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_mses = []

    # Map from valid indices back to original indices
    valid_indices = np.where(valid_mask)[0]

    for train_idx, val_idx in kfold.split(X_valid):
        X_train, X_val = X_valid[train_idx], X_valid[val_idx]
        y_train, y_val = y_valid[train_idx], y_valid[val_idx]

        # Fit ElasticNet on training fold
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # Predict on validation fold
        y_pred = model.predict(X_val)

        # Store out-of-fold predictions at original indices
        original_val_indices = valid_indices[val_idx]
        cv_predictions[original_val_indices] = y_pred

        # Track fold MSE
        fold_mse = np.mean((y_val - y_pred) ** 2)
        fold_mses.append(fold_mse)

    # Fit final model on all valid data
    final_model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=10000,
        random_state=random_state,
    )
    final_model.fit(X_valid, y_valid)

    # Compute metrics
    cv_mse = float(np.mean(fold_mses))

    # Compute CV R² from valid predictions
    valid_cv_preds = cv_predictions[valid_mask]
    cv_r2 = compute_cv_r2(y_valid, valid_cv_preds)

    # Check if all coefficients were shrunk to zero
    is_intercept_only = np.allclose(final_model.coef_, 0, atol=1e-10)

    return SingleVariantModelResult(
        coefficients=final_model.coef_.copy(),
        intercept=float(final_model.intercept_),
        cv_predictions=cv_predictions,
        cv_mse=cv_mse,
        cv_r2=cv_r2,
        is_intercept_only=is_intercept_only,
        n_predictors=n_predictors,
        n_samples=n_samples,
        l1_ratio=l1_ratio,
        alpha=alpha,
    )
