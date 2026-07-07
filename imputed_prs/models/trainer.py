"""Training loop for imputation models across all missing variants."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.types import (
    ImputedVariantModel,
    SingleVariantModelResult,
    TrainingFailure,
    TrainingResult,
)
from imputed_prs.core.window_index import ChromosomeIndex
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.tuning import tune_single_variant_model


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
    predictor_rows: pd.DataFrame,
    predictor_dosages: np.ndarray,
) -> ImputedVariantModel:
    """Convert SingleVariantModelResult to ImputedVariantModel.

    Args:
        variant_row: Row from prs_variants DataFrame with variant info.
        result: Result from fit_single_variant_model.
        predictor_ids: List of predictor variant IDs (index-aligned with
            predictor_rows / the columns of predictor_dosages).
        target_dosages: Target dosage values for computing allele frequency.
        predictor_rows: Platform reference rows backing each predictor (the Z
            columns), with columns chromosome/position/ref_allele/alt_allele.
            Z counts the ALT allele, so counted=alt_allele, other=ref_allele.
        predictor_dosages: Predictor dosage columns (n_samples, n_predictors),
            used to compute the counted-allele frequency of each predictor.

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

    # Derive index-aligned predictor allele metadata from the reference rows
    # backing each Z column. Z counts the ALT allele, so counted=alt, other=ref.
    # The counted-allele frequency is the mean predictor dosage / 2.
    if len(predictor_ids) > 0:
        predictor_chromosomes = [str(c) for c in predictor_rows["chromosome"].tolist()]
        predictor_positions = [int(p) for p in predictor_rows["position"].tolist()]
        predictor_counted_alleles = [str(a) for a in predictor_rows["alt_allele"].tolist()]
        predictor_other_alleles = [str(a) for a in predictor_rows["ref_allele"].tolist()]
        predictor_allele_frequencies = np.nanmean(predictor_dosages, axis=0) / 2.0
    else:
        predictor_chromosomes = []
        predictor_positions = []
        predictor_counted_alleles = []
        predictor_other_alleles = []
        predictor_allele_frequencies = np.array([])

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
        predictor_chromosomes=predictor_chromosomes,
        predictor_positions=predictor_positions,
        predictor_counted_alleles=predictor_counted_alleles,
        predictor_other_alleles=predictor_other_alleles,
        predictor_allele_frequencies=predictor_allele_frequencies,
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
    chrom_index: ChromosomeIndex,
    window_size: int,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
    max_predictors: Optional[int],
    tuning_scope: str = "none",
    l1_ratios: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
) -> Tuple[
    int, Optional[ImputedVariantModel], Optional[np.ndarray], bool, Optional[TrainingFailure]
]:
    """Fit imputation model for a single variant.

    Args:
        variant_idx: Index of the variant in X.
        variant_row: Row from prs_variants DataFrame.
        Z: Platform genotype dosages (n_samples, n_platform_variants).
        X: Missing variant dosages (n_samples, n_missing_variants).
        platform_variant_info: Platform variant info DataFrame.
        chrom_index: Prebuilt ``ChromosomeIndex`` over ``platform_variant_info``
            (O(log n) window queries; replaces the per-call ``filter_to_local_window``
            scan). Must be built from the same frame passed as
            ``platform_variant_info`` so positional indices align.
        window_size: Window size in base pairs.
        l1_ratio: ElasticNet L1 ratio (used when tuning_scope != "per_variant").
        alpha: ElasticNet regularization strength (used when not "per_variant").
        cv_folds: Number of CV folds.
        random_state: Random seed.
        max_predictors: Maximum number of predictors to use.
        tuning_scope: "per_variant" runs a per-variant grid search on this
            variant's local window (selecting its own l1_ratio/alpha); any other
            value fits a single model with the supplied l1_ratio/alpha.
        l1_ratios: Grid for per-variant search (defaults applied when None).
        alphas: Grid for per-variant search (defaults applied when None).

    Returns:
        Tuple of (variant_idx, model, cv_predictions, is_intercept_only, failure).
        Model and cv_predictions are None if fitting failed; ``failure`` is a
        TrainingFailure on failure and None otherwise.
    """
    # Diagnostics captured for failure reporting (P5.1). Initialized to None and
    # bound progressively so a failure reports whatever context was available
    # when it raised.
    n_valid_samples: Optional[int] = None
    target_variance: Optional[float] = None
    n_predictors: Optional[int] = None
    try:
        # Extract target dosages
        target_dosages = X[:, variant_idx]

        valid = ~np.isnan(target_dosages)
        n_valid_samples = int(valid.sum())
        target_variance = float(np.var(target_dosages[valid])) if valid.any() else 0.0

        # Filter to local window (Phase 9: O(log n) prebuilt index; identical result
        # to filter_to_local_window — same WindowFilterResult, positional indices).
        window_result = chrom_index.window(
            target_chrom=str(variant_row["chromosome"]),
            target_pos=int(variant_row["position"]),
            window_size=window_size,
            exclude_target=True,
            max_variants=max_predictors,
        )

        # Extract predictor dosages and the reference rows backing each Z column
        if window_result.n_variants > 0:
            predictor_dosages = Z[:, window_result.variant_indices]
            predictor_ids = window_result.variant_ids
            predictor_rows = platform_variant_info.iloc[window_result.variant_indices]
        else:
            # No predictors in window - will result in intercept-only model
            predictor_dosages = np.empty((Z.shape[0], 0))
            predictor_ids = []
            predictor_rows = platform_variant_info.iloc[[]]
        n_predictors = int(predictor_dosages.shape[1])

        # Fit model. For per-variant tuning, grid-search this variant's own local
        # window (same matrix used below) and keep the best fit; otherwise fit once
        # with the supplied (globally tuned or default) hyperparameters.
        if tuning_scope == "per_variant" and predictor_dosages.shape[1] > 0:
            result = tune_single_variant_model(
                target_dosages,
                predictor_dosages,
                l1_ratios=l1_ratios,
                alphas=alphas,
                cv_folds=cv_folds,
                random_state=random_state,
            )
        else:
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
            predictor_rows=predictor_rows,
            predictor_dosages=predictor_dosages,
        )

        return (variant_idx, model, result.cv_predictions, result.is_intercept_only, None)

    except Exception as exc:
        # Capture a structured failure reason instead of silently dropping it (P5.1).
        failure = TrainingFailure(
            unit_id=str(variant_row["variant_id"]),
            error_type=type(exc).__name__,
            error_message=str(exc),
            n_valid_samples=n_valid_samples,
            target_variance=target_variance,
            n_predictors=n_predictors,
        )
        return (variant_idx, None, None, False, failure)


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
        tuning_scope: str = "none",
        tuning_l1_ratios: Optional[List[float]] = None,
        tuning_alphas: Optional[List[float]] = None,
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
        self.tuning_scope = tuning_scope
        self.tuning_l1_ratios = tuning_l1_ratios
        self.tuning_alphas = tuning_alphas
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
        required_platform_cols = [
            "variant_id", "chromosome", "position", "ref_allele", "alt_allele"
        ]
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
                Required columns: variant_id, chromosome, position, ref_allele,
                alt_allele.

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

        # Phase 9: build the window index once over the (call-invariant) platform
        # frame; every _fit_one_variant does O(log n) lookups instead of an O(n)
        # per-call chromosome re-normalization + scan. Shared read-only across the
        # prefer="threads" pool.
        chrom_index = ChromosomeIndex(platform_variant_info)

        # Prepare arguments for parallel processing
        fit_args = [
            (
                idx,
                prs_variants.iloc[idx],
                Z,
                X,
                platform_variant_info,
                chrom_index,
                self.window_size,
                self.l1_ratio,
                self.alpha,
                self.cv_folds,
                self.random_state,
                self.max_predictors,
                self.tuning_scope,
                self.tuning_l1_ratios,
                self.tuning_alphas,
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
        failures: Dict[str, TrainingFailure] = {}
        n_variants_failed = 0
        n_intercept_only = 0

        for variant_idx, model, cv_preds, is_intercept_only, failure in results:
            if model is None:
                n_variants_failed += 1
                if failure is not None:
                    failures[failure.unit_id] = failure
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
            failures=failures,
        )
