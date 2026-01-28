"""Training loop for imputation models across all missing variants."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.harmonizer import filter_to_local_window
from imputed_prs.core.types import (
    ImputedVariantModel,
    SingleVariantModelResult,
    TrainingResult,
)
from imputed_prs.models.elastic_net import fit_single_variant_model


def compute_residual_variance(allele_frequency: float, r2: float) -> float:
    """Compute residual variance for imputed dosage.

    The residual variance is: 2 * q * (1 - q) * (1 - r2)

    Where q is the allele frequency and r2 is the imputation R².

    Args:
        allele_frequency: Allele frequency (0-1).
        r2: Imputation R² (clipped to [0, 1] for variance calculation).

    Returns:
        Residual variance.
    """
    q = allele_frequency
    # Clip r2 to [0, 1] for variance calculation
    r2_clipped = max(0.0, min(1.0, r2))
    return 2.0 * q * (1.0 - q) * (1.0 - r2_clipped)


def _convert_to_imputed_model(
    variant_row: pd.Series,
    result: SingleVariantModelResult,
    predictor_ids: List[str],
    target_dosages: np.ndarray,
) -> ImputedVariantModel:
    """Convert SingleVariantModelResult to ImputedVariantModel.

    Args:
        variant_row: Row from prs_variants DataFrame with variant info.
        result: Result from fit_single_variant_model.
        predictor_ids: List of predictor variant IDs.
        target_dosages: Target dosage values for computing allele frequency.

    Returns:
        ImputedVariantModel with all fields populated.
    """
    # Compute allele frequency from target dosages
    valid_mask = ~np.isnan(target_dosages)
    if np.any(valid_mask):
        allele_frequency = float(np.mean(target_dosages[valid_mask]) / 2.0)
    else:
        allele_frequency = 0.0

    # Clip allele frequency to [0, 1]
    allele_frequency = max(0.0, min(1.0, allele_frequency))

    # Compute residual variance using clipped R²
    r2_clipped = max(0.0, result.cv_r2)
    residual_variance = compute_residual_variance(allele_frequency, r2_clipped)

    # Get other_allele if present
    other_allele = variant_row.get("other_allele")
    if pd.isna(other_allele):
        other_allele = None

    return ImputedVariantModel(
        variant_id=variant_row["variant_id"],
        chromosome=str(variant_row["chromosome"]),
        position=int(variant_row["position"]),
        effect_allele=variant_row["effect_allele"],
        other_allele=other_allele,
        beta=float(variant_row["beta"]),
        allele_frequency=allele_frequency,
        imputation_r2=result.cv_r2,  # Keep original (can be negative)
        residual_variance=residual_variance,
        intercept=result.intercept,
        predictor_variant_ids=predictor_ids,
        coefficients=result.coefficients,
        is_intercept_only=result.is_intercept_only,
    )


def _compute_training_summary(models: Dict[str, ImputedVariantModel]) -> Dict[str, Any]:
    """Compute summary statistics from trained models.

    Args:
        models: Dictionary mapping variant_id to ImputedVariantModel.

    Returns:
        Dictionary with summary statistics.
    """
    if not models:
        return {
            "mean_r2": 0.0,
            "median_r2": 0.0,
            "std_r2": 0.0,
            "min_r2": 0.0,
            "max_r2": 0.0,
            "n_high_quality": 0,
            "n_medium_quality": 0,
            "n_low_quality": 0,
            "mean_n_predictors": 0.0,
        }

    r2_values = [m.imputation_r2 for m in models.values()]
    n_predictors = [len(m.predictor_variant_ids) for m in models.values()]

    r2_array = np.array(r2_values)

    return {
        "mean_r2": float(np.mean(r2_array)),
        "median_r2": float(np.median(r2_array)),
        "std_r2": float(np.std(r2_array)),
        "min_r2": float(np.min(r2_array)),
        "max_r2": float(np.max(r2_array)),
        "n_high_quality": int(np.sum(r2_array > 0.8)),
        "n_medium_quality": int(np.sum((r2_array > 0.4) & (r2_array <= 0.8))),
        "n_low_quality": int(np.sum(r2_array <= 0.4)),
        "mean_n_predictors": float(np.mean(n_predictors)),
    }


def _fit_one_variant(
    variant_idx: int,
    variant_row: pd.Series,
    Z: np.ndarray,
    X: np.ndarray,
    platform_variant_info: pd.DataFrame,
    window_size: int,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
    max_predictors: Optional[int],
) -> Tuple[int, Optional[ImputedVariantModel], Optional[np.ndarray], bool]:
    """Fit imputation model for a single variant.

    Args:
        variant_idx: Index of the variant in X.
        variant_row: Row from prs_variants DataFrame.
        Z: Platform genotype dosages (n_samples, n_platform_variants).
        X: Missing variant dosages (n_samples, n_missing_variants).
        platform_variant_info: Platform variant info DataFrame.
        window_size: Window size in base pairs.
        l1_ratio: ElasticNet L1 ratio.
        alpha: ElasticNet regularization strength.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        max_predictors: Maximum number of predictors to use.

    Returns:
        Tuple of (variant_idx, model, cv_predictions, is_intercept_only).
        Model and cv_predictions are None if fitting failed.
    """
    try:
        # Extract target dosages
        target_dosages = X[:, variant_idx]

        # Filter to local window
        window_result = filter_to_local_window(
            target_chrom=str(variant_row["chromosome"]),
            target_pos=int(variant_row["position"]),
            variant_info=platform_variant_info,
            window_size=window_size,
            exclude_target=True,
            max_variants=max_predictors,
        )

        # Extract predictor dosages
        if window_result.n_variants > 0:
            predictor_dosages = Z[:, window_result.variant_indices]
            predictor_ids = window_result.variant_ids
        else:
            # No predictors in window - will result in intercept-only model
            predictor_dosages = np.empty((Z.shape[0], 0))
            predictor_ids = []

        # Fit model
        result = fit_single_variant_model(
            target_dosages=target_dosages,
            predictor_dosages=predictor_dosages,
            l1_ratio=l1_ratio,
            alpha=alpha,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        # Convert to ImputedVariantModel
        model = _convert_to_imputed_model(
            variant_row=variant_row,
            result=result,
            predictor_ids=predictor_ids,
            target_dosages=target_dosages,
        )

        return (variant_idx, model, result.cv_predictions, result.is_intercept_only)

    except Exception:
        # Return None to indicate failure
        return (variant_idx, None, None, False)


class ImputationModelTrainer:
    """Orchestrates training of imputation models for all missing variants.

    This class handles the training loop for fitting ElasticNet imputation
    models to predict missing variant dosages from observed platform variants.
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        verbose: int = 1,
    ):
        """Initialize the trainer.

        Args:
            window_size: Window size in base pairs for local predictor selection.
                Default: 1,000,000 (1 Mb).
            l1_ratio: ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).
                Default: 0.5.
            alpha: ElasticNet regularization strength. Default: 0.01.
            cv_folds: Number of cross-validation folds. Default: 5.
            n_jobs: Number of parallel jobs. Default: 1 (sequential).
            random_state: Random seed for reproducibility. Default: None.
            max_predictors: Maximum predictors per variant. Default: None (no limit).
            progress_callback: Callback function(variant_id, current, total) for
                progress reporting. Default: None.
            verbose: Verbosity level (0=silent, 1=progress, 2=debug). Default: 1.
        """
        self.window_size = window_size
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.progress_callback = progress_callback
        self.verbose = verbose

    def _validate_inputs(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        prs_variants: pd.DataFrame,
        platform_variant_info: pd.DataFrame,
    ) -> None:
        """Validate input shapes and required columns.

        Args:
            Z: Platform genotype dosages.
            X: Missing variant dosages.
            prs_variants: Missing variants DataFrame.
            platform_variant_info: Platform variant info DataFrame.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Check required columns in prs_variants
        required_prs_cols = ["variant_id", "chromosome", "position", "effect_allele", "beta"]
        missing_prs_cols = [c for c in required_prs_cols if c not in prs_variants.columns]
        if missing_prs_cols:
            raise ValidationError(
                f"Missing required columns in prs_variants: {missing_prs_cols}"
            )

        # Check required columns in platform_variant_info
        required_platform_cols = ["variant_id", "chromosome", "position"]
        missing_platform_cols = [
            c for c in required_platform_cols if c not in platform_variant_info.columns
        ]
        if missing_platform_cols:
            raise ValidationError(
                f"Missing required columns in platform_variant_info: {missing_platform_cols}"
            )

        # Check shape compatibility
        if Z.shape[0] != X.shape[0]:
            raise ValidationError(
                f"Sample count mismatch: Z has {Z.shape[0]} samples, "
                f"X has {X.shape[0]} samples"
            )

        if X.shape[1] != len(prs_variants):
            raise ValidationError(
                f"Variant count mismatch: X has {X.shape[1]} variants, "
                f"prs_variants has {len(prs_variants)} rows"
            )

        if Z.shape[1] != len(platform_variant_info):
            raise ValidationError(
                f"Platform variant count mismatch: Z has {Z.shape[1]} variants, "
                f"platform_variant_info has {len(platform_variant_info)} rows"
            )

    def fit_all_variants(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        prs_variants: pd.DataFrame,
        platform_variant_info: pd.DataFrame,
    ) -> TrainingResult:
        """Fit imputation models for all missing variants.

        Args:
            Z: Platform genotype dosages. Shape: (n_samples, n_platform_variants).
                Values are 0-2 representing allele counts, with NaN for missing.
            X: Missing variant dosages. Shape: (n_samples, n_missing_variants).
                Values are 0-2 representing allele counts, with NaN for missing.
            prs_variants: DataFrame with missing variant information. Required columns:
                variant_id, chromosome, position, effect_allele, beta.
                Optional: other_allele.
            platform_variant_info: DataFrame with platform variant information.
                Required columns: variant_id, chromosome, position.

        Returns:
            TrainingResult containing trained models, CV predictions, and summary.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate inputs
        self._validate_inputs(Z, X, prs_variants, platform_variant_info)

        # Handle empty input
        if len(prs_variants) == 0:
            return TrainingResult(
                models={},
                cv_predictions={},
                n_variants_trained=0,
                n_variants_failed=0,
                n_intercept_only=0,
                training_summary=_compute_training_summary({}),
            )

        # Reset DataFrame index to ensure sequential indexing
        prs_variants = prs_variants.reset_index(drop=True)

        n_variants = len(prs_variants)

        # Prepare arguments for parallel processing
        fit_args = [
            (
                idx,
                prs_variants.iloc[idx],
                Z,
                X,
                platform_variant_info,
                self.window_size,
                self.l1_ratio,
                self.alpha,
                self.cv_folds,
                self.random_state,
                self.max_predictors,
            )
            for idx in range(n_variants)
        ]

        # Run fitting (parallel or sequential)
        if self.n_jobs == 1:
            # Sequential execution
            results = []
            for i, args in enumerate(fit_args):
                result = _fit_one_variant(*args)
                results.append(result)
                if self.progress_callback is not None:
                    variant_id = prs_variants.iloc[args[0]]["variant_id"]
                    self.progress_callback(variant_id, i + 1, n_variants)
        else:
            # Parallel execution
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_one_variant)(*args) for args in fit_args
            )

        # Collect results
        models: Dict[str, ImputedVariantModel] = {}
        cv_predictions: Dict[str, np.ndarray] = {}
        n_variants_failed = 0
        n_intercept_only = 0

        for variant_idx, model, cv_preds, is_intercept_only in results:
            if model is None:
                n_variants_failed += 1
                continue

            variant_id = model.variant_id
            models[variant_id] = model
            cv_predictions[variant_id] = cv_preds

            if is_intercept_only:
                n_intercept_only += 1

        n_variants_trained = len(models)

        # Compute summary statistics
        training_summary = _compute_training_summary(models)

        return TrainingResult(
            models=models,
            cv_predictions=cv_predictions,
            n_variants_trained=n_variants_trained,
            n_variants_failed=n_variants_failed,
            n_intercept_only=n_intercept_only,
            training_summary=training_summary,
        )
