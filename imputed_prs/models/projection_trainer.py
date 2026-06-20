"""Training loop for projection models across all genomic regions."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.harmonizer import _normalize_chromosome
from imputed_prs.core.regions import GenomicRegion, merge_variant_windows
from imputed_prs.core.types import (
    ProjectionRegionModel,
    ProjectionTrainingResult,
)
from imputed_prs.models.projection import fit_single_region_model


def _compute_projection_training_summary(
    region_models: Dict[str, ProjectionRegionModel],
) -> Dict[str, Any]:
    """Compute summary statistics across all trained region models.

    Args:
        region_models: Dictionary mapping region_id to ProjectionRegionModel.

    Returns:
        Dictionary with summary statistics.
    """
    if not region_models:
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
            "mean_n_prs_variants_per_region": 0.0,
        }

    r2_values = [m.cv_r2 for m in region_models.values()]
    n_predictors = [len(m.predictor_variant_ids) for m in region_models.values()]
    n_prs_variants = [len(m.prs_variant_ids) for m in region_models.values()]

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
        "mean_n_prs_variants_per_region": float(np.mean(n_prs_variants)),
    }


def _find_platform_variants_in_region(
    region: GenomicRegion,
    platform_variant_info: pd.DataFrame,
    max_predictors: Optional[int],
) -> Tuple[List[str], np.ndarray]:
    """Find platform variants within a genomic region.

    Args:
        region: The genomic region to search within.
        platform_variant_info: DataFrame with columns: variant_id, chromosome,
            position, ref_allele, alt_allele.
        max_predictors: Maximum number of predictors. If set, select the closest
            to region center. None means no limit.

    Returns:
        Tuple of (predictor_variant_ids, predictor_indices_into_dataframe).
    """
    # Normalize chromosomes for matching
    chrom_normalized = platform_variant_info["chromosome"].apply(
        lambda c: _normalize_chromosome(str(c))
    )
    mask = (
        (chrom_normalized == region.chromosome)
        & (platform_variant_info["position"] >= region.start)
        & (platform_variant_info["position"] <= region.end)
    )

    matched_indices = np.where(mask.values)[0]

    if len(matched_indices) == 0:
        return [], np.array([], dtype=int)

    # Optionally limit to max_predictors closest to region center
    if max_predictors is not None and len(matched_indices) > max_predictors:
        center = (region.start + region.end) // 2
        positions = platform_variant_info.iloc[matched_indices]["position"].values
        distances = np.abs(positions - center)
        closest_order = np.argsort(distances)[:max_predictors]
        matched_indices = matched_indices[closest_order]

    predictor_ids = platform_variant_info.iloc[matched_indices]["variant_id"].tolist()
    return predictor_ids, matched_indices


def _fit_one_region(
    region: GenomicRegion,
    Z: np.ndarray,
    X: np.ndarray,
    prs_variants: pd.DataFrame,
    platform_variant_info: pd.DataFrame,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
    max_predictors: Optional[int],
) -> Tuple[str, Optional[ProjectionRegionModel], Optional[np.ndarray], bool]:
    """Fit projection model for a single region (parallelizable).

    Args:
        region: The genomic region to fit.
        Z: Platform genotype dosages (n_samples, n_platform_variants).
        X: Missing PRS variant dosages (n_samples, n_missing_variants).
        prs_variants: DataFrame for missing PRS variants.
        platform_variant_info: DataFrame with platform variant info.
        l1_ratio: ElasticNet L1 ratio.
        alpha: ElasticNet regularization strength.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        max_predictors: Maximum number of predictors per region.

    Returns:
        Tuple of (region_id, model_or_None, cv_predictions_or_None, is_intercept_only).
    """
    region_id = f"chr{region.chromosome}:{region.start}-{region.end}"

    try:
        # Get betas for PRS variants in this region
        indices = region.prs_variant_indices
        prs_rows = prs_variants.iloc[indices]
        betas = prs_rows["beta"].values.astype(np.float64)

        # Per-PRS-variant locus + alleles, index-aligned with betas. Lets the
        # evaluator orient the true PRS (effect==REF, strand-flip, multiallelic)
        # instead of assuming effect==ALT at the first reference row.
        prs_positions = [int(p) for p in prs_rows["position"].tolist()]
        prs_effect_alleles = [str(a) for a in prs_rows["effect_allele"].tolist()]
        if "other_allele" in prs_rows.columns:
            prs_other_alleles = [
                None if pd.isna(a) else str(a)
                for a in prs_rows["other_allele"].tolist()
            ]
        else:
            prs_other_alleles = [None] * len(indices)

        # Compute target: S_R = X[:, indices] @ betas
        X_region = X[:, indices]
        target = X_region @ betas

        # Find platform variants in region
        predictor_ids, platform_indices = _find_platform_variants_in_region(
            region, platform_variant_info, max_predictors
        )

        # Extract predictor dosages
        if len(platform_indices) > 0:
            predictor_dosages = Z[:, platform_indices]
        else:
            predictor_dosages = np.empty((Z.shape[0], 0))

        # Fit model
        result = fit_single_region_model(
            target_prs_contribution=target,
            predictor_dosages=predictor_dosages,
            l1_ratio=l1_ratio,
            alpha=alpha,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        # Compute predictor allele frequencies and the reference-row allele
        # metadata backing each Z column. Z counts the ALT allele, so the
        # counted allele is alt_allele and the other allele is ref_allele;
        # the counted-allele frequency is the mean predictor dosage / 2.
        if len(platform_indices) > 0:
            predictor_afs = np.array([
                float(np.nanmean(Z[:, idx]) / 2.0)
                for idx in platform_indices
            ])
            predictor_rows = platform_variant_info.iloc[platform_indices]
            predictor_chromosomes = [str(c) for c in predictor_rows["chromosome"].tolist()]
            predictor_positions = [int(p) for p in predictor_rows["position"].tolist()]
            predictor_counted_alleles = [str(a) for a in predictor_rows["alt_allele"].tolist()]
            predictor_other_alleles = [str(a) for a in predictor_rows["ref_allele"].tolist()]
        else:
            predictor_afs = np.array([])
            predictor_chromosomes = []
            predictor_positions = []
            predictor_counted_alleles = []
            predictor_other_alleles = []

        # Wrap into ProjectionRegionModel
        model = ProjectionRegionModel(
            region_id=region_id,
            chromosome=region.chromosome,
            start=region.start,
            end=region.end,
            prs_variant_ids=region.prs_variant_ids,
            betas=betas,
            predictor_variant_ids=predictor_ids,
            coefficients=result.coefficients,
            intercept=result.intercept,
            cv_mse=result.cv_mse,
            cv_r2=result.cv_r2,
            is_intercept_only=result.is_intercept_only,
            mean_prs_contribution=float(np.nanmean(target)),
            predictor_allele_frequencies=predictor_afs,
            predictor_chromosomes=predictor_chromosomes,
            predictor_positions=predictor_positions,
            predictor_counted_alleles=predictor_counted_alleles,
            predictor_other_alleles=predictor_other_alleles,
            prs_positions=prs_positions,
            prs_effect_alleles=prs_effect_alleles,
            prs_other_alleles=prs_other_alleles,
        )

        return (region_id, model, result.cv_predictions, result.is_intercept_only)

    except Exception:
        return (region_id, None, None, False)


class ProjectionRegionTrainer:
    """Orchestrates training of projection models for all genomic regions.

    Mirrors ImputationModelTrainer but operates at region granularity instead
    of per-variant.
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
        verbose: int = 1,
    ):
        """Initialize the trainer.

        Args:
            window_size: Size of genomic window (bp) for defining regions
                and selecting predictor variants. Default: 1,000,000 (1 Mb).
            l1_ratio: ElasticNet L1/L2 mixing parameter. Default: 0.5.
            alpha: ElasticNet regularization strength. Default: 0.01.
            cv_folds: Number of CV folds. Default: 5.
            n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs). Default: 1.
            random_state: Random seed. Default: None.
            max_predictors: Max predictor variants per region. Default: None (no limit).
            verbose: Verbosity level (0=silent, 1=progress, 2=debug). Default: 1.
        """
        self.window_size = window_size
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.verbose = verbose

    def _validate_inputs(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        prs_variants: pd.DataFrame,
        platform_variant_info: pd.DataFrame,
    ) -> None:
        """Validate input shapes and required columns.

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

    def fit_all_regions(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        prs_variants: pd.DataFrame,
        platform_variant_info: pd.DataFrame,
    ) -> ProjectionTrainingResult:
        """Fit projection models for all regions.

        Args:
            Z: Platform genotype dosages (n_samples, n_platform_variants).
            X: Missing PRS variant dosages (n_samples, n_missing_variants).
            prs_variants: DataFrame for missing PRS variants with columns:
                variant_id, chromosome, position, effect_allele, beta.
            platform_variant_info: DataFrame with columns:
                variant_id, chromosome, position, ref_allele, alt_allele.

        Returns:
            ProjectionTrainingResult with trained region models and CV predictions.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate inputs
        self._validate_inputs(Z, X, prs_variants, platform_variant_info)

        # Handle empty input
        if len(prs_variants) == 0:
            return ProjectionTrainingResult(
                region_models={},
                cv_predictions={},
                n_regions_trained=0,
                n_regions_failed=0,
                n_intercept_only=0,
                training_summary=_compute_projection_training_summary({}),
            )

        # Reset DataFrame index to ensure sequential indexing
        prs_variants = prs_variants.reset_index(drop=True)

        # Decompose into regions
        decomposition = merge_variant_windows(prs_variants, self.window_size)

        # Prepare arguments for each region
        fit_args = [
            (
                region,
                Z,
                X,
                prs_variants,
                platform_variant_info,
                self.l1_ratio,
                self.alpha,
                self.cv_folds,
                self.random_state,
                self.max_predictors,
            )
            for region in decomposition.regions
        ]

        # Run fitting (parallel or sequential)
        if self.n_jobs == 1:
            results = []
            for args in fit_args:
                result = _fit_one_region(*args)
                results.append(result)
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_one_region)(*args) for args in fit_args
            )

        # Collect results
        region_models: Dict[str, ProjectionRegionModel] = {}
        cv_predictions: Dict[str, np.ndarray] = {}
        n_regions_failed = 0
        n_intercept_only = 0

        for region_id, model, cv_preds, is_intercept_only in results:
            if model is None:
                n_regions_failed += 1
                continue

            region_models[region_id] = model
            cv_predictions[region_id] = cv_preds

            if is_intercept_only:
                n_intercept_only += 1

        n_regions_trained = len(region_models)

        # Compute summary statistics
        training_summary = _compute_projection_training_summary(region_models)

        return ProjectionTrainingResult(
            region_models=region_models,
            cv_predictions=cv_predictions,
            n_regions_trained=n_regions_trained,
            n_regions_failed=n_regions_failed,
            n_intercept_only=n_intercept_only,
            training_summary=training_summary,
        )
