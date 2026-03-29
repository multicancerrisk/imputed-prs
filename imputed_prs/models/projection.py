"""Elastic net projection model for single genomic region fitting."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from imputed_prs.models.metrics import compute_cv_r2


@dataclass
class SingleRegionModelResult:
    """Result from fitting a single region projection model.

    Mirrors SingleVariantModelResult but for a region whose target is
    S_R = X_R @ beta_R (a continuous PRS contribution, not a dosage).

    Attributes:
        coefficients: Regression coefficients for predictor variants.
            Shape: (n_predictors,). Empty array for intercept-only.
        intercept: Model intercept (mean of target for intercept-only).
        cv_predictions: Out-of-fold CV predictions. Shape: (n_samples,).
            Contains NaN for samples excluded due to missing values.
        cv_mse: Mean squared error from cross-validation.
        cv_r2: R-squared from cross-validation.
        is_intercept_only: True if no predictors or all coefficients zero.
        n_predictors: Number of predictor variants.
        n_samples: Total number of samples.
        l1_ratio: ElasticNet L1 ratio used.
        alpha: ElasticNet alpha used.
    """

    coefficients: np.ndarray
    intercept: float
    cv_predictions: np.ndarray
    cv_mse: float
    cv_r2: float
    is_intercept_only: bool
    n_predictors: int
    n_samples: int
    l1_ratio: float
    alpha: float


def _fit_intercept_only_region(
    target_prs_contribution: np.ndarray,
    n_predictors: int,
    l1_ratio: float,
    alpha: float,
) -> SingleRegionModelResult:
    """Fit an intercept-only model (fallback when no predictors available).

    Args:
        target_prs_contribution: Regional PRS contribution values. Shape: (n_samples,).
        n_predictors: Number of predictors (for metadata, typically 0).
        l1_ratio: L1 ratio parameter (stored in result).
        alpha: Alpha parameter (stored in result).

    Returns:
        SingleRegionModelResult with intercept set to mean of valid targets.
    """
    n_samples = len(target_prs_contribution)

    # Find valid (non-NaN) samples
    valid_mask = ~np.isnan(target_prs_contribution)
    valid_targets = target_prs_contribution[valid_mask]

    if len(valid_targets) == 0:
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

    return SingleRegionModelResult(
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


def fit_single_region_model(
    target_prs_contribution: np.ndarray,
    predictor_dosages: np.ndarray,
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> SingleRegionModelResult:
    """Fit ElasticNet projection model for a single genomic region.

    The target is S_R = X_R @ beta_R (regional PRS contribution from missing
    variants). The predictors are platform variant dosages Z_R in the region.

    This mirrors fit_single_variant_model() from elastic_net.py but:
    - Target is a weighted sum (continuous, possibly negative), not a dosage.
    - No dosage bounds [0, 2] apply to the target.
    - The intercept represents the mean regional PRS contribution.

    Args:
        target_prs_contribution: S_R values for each sample. Shape: (n_samples,).
        predictor_dosages: Platform variant dosages. Shape: (n_samples, n_predictors).
        l1_ratio: ElasticNet L1/L2 mixing (0=Ridge, 1=Lasso). Default: 0.5.
        alpha: Regularization strength. Default: 0.01.
        cv_folds: Number of CV folds. Default: 5.
        random_state: Random seed. Default: None.

    Returns:
        SingleRegionModelResult with fitted model parameters and CV metrics.

    Raises:
        ValueError: If target and predictor arrays have incompatible shapes.
    """
    # Convert to float64 for numerical stability
    target_prs_contribution = np.asarray(target_prs_contribution, dtype=np.float64)
    predictor_dosages = np.asarray(predictor_dosages, dtype=np.float64)

    # Handle 1D predictor case (single predictor)
    if predictor_dosages.ndim == 1:
        predictor_dosages = predictor_dosages.reshape(-1, 1)

    n_samples = len(target_prs_contribution)
    n_predictors = predictor_dosages.shape[1] if predictor_dosages.size > 0 else 0

    # Input validation
    if predictor_dosages.size > 0 and predictor_dosages.shape[0] != n_samples:
        raise ValueError(
            f"Shape mismatch: target has {n_samples} samples but "
            f"predictors have {predictor_dosages.shape[0]} samples"
        )

    # Edge case: no predictors
    if n_predictors == 0:
        return _fit_intercept_only_region(
            target_prs_contribution, n_predictors, l1_ratio, alpha
        )

    # Create valid mask: non-NaN target AND no NaN in any predictor
    valid_target_mask = ~np.isnan(target_prs_contribution)
    valid_predictor_mask = ~np.any(np.isnan(predictor_dosages), axis=1)
    valid_mask = valid_target_mask & valid_predictor_mask

    n_valid = np.sum(valid_mask)

    # Edge case: too few valid samples for CV
    if n_valid < cv_folds:
        return _fit_intercept_only_region(
            target_prs_contribution, n_predictors, l1_ratio, alpha
        )

    # Extract valid data
    y_valid = target_prs_contribution[valid_mask]
    X_valid = predictor_dosages[valid_mask]

    # Edge case: zero variance in target
    if np.std(y_valid) < 1e-10:
        return _fit_intercept_only_region(
            target_prs_contribution, n_predictors, l1_ratio, alpha
        )

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

    return SingleRegionModelResult(
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
