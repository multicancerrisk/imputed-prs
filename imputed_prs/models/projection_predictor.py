"""Prediction pipeline for projection-based PRS computation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from imputed_prs.core.types import (
    CalibrationParams,
    PredictionResult,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.io.user_genotypes import (
    RawUserGenotypeCollection,
    resolve_counted_dosage,
)
from imputed_prs.models.predictor import (
    compute_observed_prs,
    compute_observed_prs_oriented,
)


def _region_effective_variance(
    model: ProjectionRegionModel, n_substituted: int
) -> float:
    """Missingness-aware region variance (P3.3).

    Interpolates from the full-model ``cv_mse`` toward the intercept-only error
    variance (``target_variance``) in proportion to how many of the region's
    predictors were mean-substituted::

        f         = n_substituted / n_predictors
        effective = cv_mse * (1 - f) + target_variance * f

    With no predictors (intercept-only region) or none substituted, returns
    ``cv_mse`` unchanged — so a fully-observed score is scored exactly as before.
    Note: this grows with missingness only when ``target_variance >= cv_mse``
    (non-negative CV R²); a region whose model predicts worse out-of-fold than
    its own mean (cv_mse > target_variance) correctly interpolates *downward*
    toward the better intercept-only fallback.
    """
    n_pred = len(model.predictor_variant_ids)
    if n_pred == 0 or n_substituted == 0:
        return model.cv_mse
    f = n_substituted / n_pred
    return model.cv_mse * (1.0 - f) + model.target_variance * f


def compute_projected_prs(
    user_dosages: Dict[str, Optional[float]],
    region_models: List[ProjectionRegionModel],
) -> Tuple[float, float, int, int]:
    """Compute PRS contribution from projection regions.

    For each region model:
    1. Gather predictor dosages from user data.
    2. For any missing predictor, substitute 2 * AF (population mean dosage).
    3. Compute prediction: S_hat_R = z_R^T @ a_R + intercept_R.
    4. No dosage clipping (target is PRS contribution, not a dosage).

    Args:
        user_dosages: Dict mapping variant_id to dosage (0-2) or None.
        region_models: List of ProjectionRegionModel objects.

    Returns:
        Tuple of (prs_projected, total_variance, n_regions_used,
                  n_predictors_substituted).
    """
    if not region_models:
        return (0.0, 0.0, 0, 0)

    total_prs = 0.0
    total_variance = 0.0
    n_regions_used = 0
    n_predictors_substituted = 0

    for model in region_models:
        n_sub_region = 0
        if model.is_intercept_only:
            prediction = model.intercept
        else:
            # Gather predictor dosages, substituting 2*AF for missing
            predictor_dosages = []
            for i, pred_id in enumerate(model.predictor_variant_ids):
                dosage = user_dosages.get(pred_id)
                if dosage is None:
                    dosage = 2.0 * model.predictor_allele_frequencies[i]
                    n_predictors_substituted += 1
                    n_sub_region += 1
                predictor_dosages.append(dosage)

            predictor_array = np.array(predictor_dosages)
            prediction = np.dot(predictor_array, model.coefficients) + model.intercept

        total_prs += prediction
        total_variance += _region_effective_variance(model, n_sub_region)
        n_regions_used += 1

    return (total_prs, total_variance, n_regions_used, n_predictors_substituted)


def compute_projected_prs_oriented(
    raw_genotypes: RawUserGenotypeCollection,
    region_models: List[ProjectionRegionModel],
    *,
    allow_ambiguous: bool,
    allow_strand_flip: bool = True,
) -> Tuple[float, float, int, int]:
    """Allele-aware PRS contribution from projection regions, from raw genotypes.

    The browser/upload counterpart to :func:`compute_projected_prs`. Each region
    predictor dosage is counted allele-aware — copies of the stored ALT allele
    (``predictor_counted_alleles[i]``, the allele the reference ``Z`` column was
    built from) — instead of read from an allele-blind homozygosity dosage dict,
    so a ``"GG"`` call no longer contributes 2 when ``G`` is the other (REF)
    allele. A predictor that is missing or unresolvable is mean-substituted with
    ``2 * AF`` (population mean dosage), exactly as :func:`compute_projected_prs`
    already does for missing ids. No dosage clipping (the target is a PRS
    contribution, not a dosage).

    This is the canonical projected-scoring path for real uploads; the legacy
    :func:`compute_projected_prs` (dosage-dict, allele-blind) is retained only for
    the evaluator / back-compat path until P1.6.

    Args:
        raw_genotypes: User genotypes as a multi-key resolvable collection.
        region_models: Region models carrying P1.3 predictor allele metadata
            (counted/other alleles + allele frequencies), index-aligned with
            ``coefficients``/``predictor_variant_ids``.
        allow_ambiguous: Whether palindromic predictor loci may be counted. The
            orchestrator passes ``True`` so palindromic predictors are counted
            (not mean-substituted), matching the oriented reference computation.
        allow_strand_flip: Whether to retry on the complementary strand.

    Returns:
        Tuple of (prs_projected, total_variance, n_regions_used,
        n_predictors_substituted), matching :func:`compute_projected_prs`.
    """
    if not region_models:
        return (0.0, 0.0, 0, 0)

    total_prs = 0.0
    total_variance = 0.0
    n_regions_used = 0
    n_predictors_substituted = 0

    for model in region_models:
        n_sub_region = 0
        if model.is_intercept_only:
            prediction = model.intercept
        else:
            # Count each predictor's ALT-allele dosage, substituting 2*AF for any
            # missing or unresolvable predictor.
            predictor_dosages = []
            for i, pred_id in enumerate(model.predictor_variant_ids):
                dosage = resolve_counted_dosage(
                    raw_genotypes,
                    variant_id=pred_id,
                    chromosome=model.predictor_chromosomes[i],
                    position=model.predictor_positions[i],
                    counted_allele=model.predictor_counted_alleles[i],
                    other_allele=model.predictor_other_alleles[i],
                    allow_ambiguous=allow_ambiguous,
                    allow_strand_flip=allow_strand_flip,
                )
                if dosage is None:
                    dosage = 2.0 * model.predictor_allele_frequencies[i]
                    n_predictors_substituted += 1
                    n_sub_region += 1
                predictor_dosages.append(dosage)

            predictor_array = np.array(predictor_dosages)
            prediction = np.dot(predictor_array, model.coefficients) + model.intercept

        total_prs += prediction
        total_variance += _region_effective_variance(model, n_sub_region)
        n_regions_used += 1

    return (total_prs, total_variance, n_regions_used, n_predictors_substituted)


class ProjectionPredictor:
    """Full PRS prediction combining observed and projected components.

    Mirrors PRSPredictor but uses region-based projection models
    instead of per-variant imputation models.

    Prediction: PRS = S_observed + sum_R(S_hat_R)
    """

    def __init__(
        self,
        observed_variants: List[VariantInfo],
        region_models: List[ProjectionRegionModel],
        calibration_params: Optional[CalibrationParams] = None,
        *,
        allow_ambiguous: bool = True,
        allow_strand_flip: bool = True,
    ):
        """Initialize the projection predictor.

        Args:
            observed_variants: List of observed PRS variants (on the platform).
            region_models: List of trained ProjectionRegionModel objects.
            calibration_params: Optional calibration parameters for scaling.
            allow_ambiguous: Allele policy for the oriented observed scorer —
                whether palindromic (A/T, C/G) loci may be scored. Default True
                mirrors training (see ``compute_observed_prs_oriented``).
            allow_strand_flip: Whether the oriented scorer retries on the
                complementary strand. Default True mirrors ``match_oriented_dosage``.
        """
        self.observed_variants = observed_variants
        self.region_models = region_models
        self.calibration_params = calibration_params
        self.allow_ambiguous = allow_ambiguous
        self.allow_strand_flip = allow_strand_flip

        # Pre-compute counts
        self._n_observed_variants = len(observed_variants)
        self._n_projected_variants = sum(
            len(m.prs_variant_ids) for m in region_models
        )
        self._n_intercept_only = sum(
            1 for m in region_models if m.is_intercept_only
        )

    def predict(
        self,
        user_genotypes: Dict[str, Optional[float]],
        apply_calibration: bool = True,
        *,
        raw_genotypes: Optional[RawUserGenotypeCollection] = None,
    ) -> PredictionResult:
        """Compute full PRS with uncertainty quantification.

        Args:
            user_genotypes: Dictionary mapping variant_id to dosage value
                (0.0, 1.0, 2.0) or None for missing variants. Used by the projected
                component (and the legacy observed scorer when ``raw_genotypes`` is
                not supplied).
            apply_calibration: Whether to apply calibration scaling.
            raw_genotypes: When provided, the observed component is scored
                allele-aware from these raw genotype strings (the path for real
                uploads). When omitted, the legacy allele-blind dosage-dict scorer
                is used (evaluators / back-compat until P1.6).

        Returns:
            PredictionResult with PRS value, confidence intervals,
            component breakdown, and optionally scaled values.
        """
        # Step 1: Compute observed component. With raw genotype strings, use the
        # allele-aware oriented scorer; otherwise the legacy allele-blind scorer.
        n_observed_scored_direct: Optional[int] = None
        n_observed_scored_via_fallback: Optional[int] = None
        weighted_beta_via_fallback: Optional[float] = None
        unresolved_observed_ids: Optional[Tuple[str, ...]] = None
        observed_fallback_variance = 0.0
        if raw_genotypes is not None:
            observed_score = compute_observed_prs_oriented(
                raw_genotypes,
                self.observed_variants,
                allow_ambiguous=self.allow_ambiguous,
                allow_strand_flip=self.allow_strand_flip,
            )
            prs_observed = observed_score.prs
            # Observed fallbacks are not trained for the projection product until
            # P2.4, so these are zero today; the shared scorer surfaces them
            # uniformly so projection and imputation report the same diagnostics.
            n_observed_used = (
                observed_score.n_scored_direct + observed_score.n_scored_fallback
            )
            n_observed_scored_direct = observed_score.n_scored_direct
            n_observed_scored_via_fallback = observed_score.n_scored_fallback
            weighted_beta_via_fallback = observed_score.weighted_beta_fallback
            unresolved_observed_ids = observed_score.unresolved_ids
            observed_fallback_variance = observed_score.fallback_variance
        else:
            prs_observed, n_observed_used = compute_observed_prs(
                user_genotypes, self.observed_variants
            )

        # Step 2: Compute projected component. With raw genotype strings, orient
        # predictor inputs allele-aware; otherwise the legacy allele-blind scorer.
        if raw_genotypes is not None:
            (
                prs_projected,
                total_variance,
                n_regions_used,
                _,
            ) = compute_projected_prs_oriented(
                raw_genotypes,
                self.region_models,
                allow_ambiguous=self.allow_ambiguous,
                allow_strand_flip=self.allow_strand_flip,
            )
        else:
            prs_projected, total_variance, n_regions_used, _ = compute_projected_prs(
                user_genotypes, self.region_models
            )

        # Observed fallbacks (P2.4) would contribute variance here; 0.0 today.
        total_variance += observed_fallback_variance

        # Step 3: Combine components
        prs_raw = prs_observed + prs_projected

        # Step 4: Compute standard error and confidence intervals
        se = np.sqrt(total_variance) if total_variance > 0 else 0.0
        ci_lower = prs_raw - 1.96 * se
        ci_upper = prs_raw + 1.96 * se

        # Step 5: Count variants
        n_variants_used = n_observed_used + self._n_projected_variants
        n_user_variants_missing = (
            self._n_observed_variants + self._n_projected_variants
        ) - n_variants_used

        # Step 6: Apply calibration if requested and available
        prs_scaled = None
        se_scaled = None
        ci_lower_scaled = None
        ci_upper_scaled = None

        if apply_calibration and self.calibration_params is not None:
            params = self.calibration_params
            prs_scaled = params.scaling_factor * prs_raw + params.calibration_intercept
            se_scaled = abs(params.scaling_factor) * se
            ci_lower_scaled = prs_scaled - 1.96 * se_scaled
            ci_upper_scaled = prs_scaled + 1.96 * se_scaled

        return PredictionResult(
            prs=prs_raw,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            prs_observed_component=prs_observed,
            prs_imputed_component=prs_projected,
            n_variants_used=n_variants_used,
            n_variants_imputed=self._n_projected_variants,
            n_variants_intercept_only=self._n_intercept_only,
            n_user_variants_missing=n_user_variants_missing,
            n_truncated=0,
            prs_scaled=prs_scaled,
            se_scaled=se_scaled,
            ci_lower_scaled=ci_lower_scaled,
            ci_upper_scaled=ci_upper_scaled,
            n_observed_scored_direct=n_observed_scored_direct,
            n_observed_scored_via_fallback=n_observed_scored_via_fallback,
            weighted_beta_via_fallback=weighted_beta_via_fallback,
            unresolved_observed_ids=unresolved_observed_ids,
        )
