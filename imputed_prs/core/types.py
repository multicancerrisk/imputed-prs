"""Core data types for the imputed-prs library."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PlatformInfo:
    """Metadata about a genotyping platform.

    Attributes:
        name: Internal platform identifier (e.g., "23andme_v5").
        display_name: Human-readable name (e.g., "23andMe V5").
        description: Brief description of the platform.
        genome_build: Reference genome build ("GRCh37" or "GRCh38").
        n_variants: Number of variants on the platform.
        chip_technology: Underlying chip technology (e.g., "Illumina GSA").
        company: Company that offers the platform.
        version: Platform version identifier.
        date_introduced: When the platform was introduced (optional).
        source_url: URL for more information about the platform (optional).
    """

    name: str
    display_name: str
    description: str
    genome_build: str
    n_variants: int
    chip_technology: str
    company: str
    version: str
    date_introduced: Optional[str] = None
    source_url: Optional[str] = None


@dataclass
class VariantInfo:
    """Represents a single variant in a PRS definition.

    Attributes:
        variant_id: rsID or unique identifier for the variant.
        chromosome: Chromosome (1-22, X, Y, MT).
        position: Genomic position.
        effect_allele: Allele associated with the effect.
        other_allele: Reference/alternate allele (optional).
        beta: Effect size (log odds ratio or beta coefficient).
        fallback: Optional per-variant imputation-style model (P1.8). Predicts
            this variant's *effect-allele* dosage from local-window platform
            predictors (excluding its own locus) so the variant can be recovered
            when the user's upload cannot resolve/call it directly, instead of
            being silently dropped. ``None`` when no fallback was trained (e.g.
            locus absent from the reference, or no platform predictors in window).
    """

    variant_id: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: Optional[str]
    beta: float
    # Forward reference: ImputedVariantModel is defined below and this module
    # does not use `from __future__ import annotations`, so the annotation must
    # stay a string to avoid evaluating the name at class-definition time.
    fallback: Optional["ImputedVariantModel"] = None


@dataclass(frozen=True)
class VariantIdentity:
    """Stable, multi-key identity for a single scored variant.

    Used to resolve a user's raw genotype against the several identifiers a DTC
    file may use, and to carry the *role-specific* counted/other alleles for
    oriented scoring. The same physical locus can appear as different
    ``VariantIdentity`` instances in different roles (e.g. an observed-PRS term
    counts the effect allele, while a predictor counts the ALT allele), so the
    counted allele lives on the identity, not on a shared per-locus dict.

    Attributes:
        feature_id: Canonical, collision-free key ("chr:pos:ref:alt").
        variant_id: Primary identifier (rsID or source-provided id).
        accepted_ids: All identifiers that should match this variant in a user
            file (rsID, PRS id, platform id, "chr:pos"); every one is tried.
        chromosome: Chromosome (1-22, X, Y, MT).
        position: Genomic position.
        counted_allele: Allele whose copies are counted for this role.
        other_allele: The complementary allele of the biallelic pair.
    """

    feature_id: str
    variant_id: str
    accepted_ids: Tuple[str, ...]
    chromosome: str
    position: int
    counted_allele: str
    other_allele: str


@dataclass
class ImputedVariantModel:
    """Stores the trained imputation model for a missing variant.

    Attributes:
        variant_id: rsID or unique identifier for the variant.
        chromosome: Chromosome (1-22, X, Y, MT).
        position: Genomic position.
        effect_allele: Allele associated with the effect.
        other_allele: Reference/alternate allele (optional).
        beta: Effect size (log odds ratio or beta coefficient).
        allele_frequency: Population allele frequency.
        imputation_r2: Cross-validated R² of imputation.
        residual_variance: Residual variance after imputation.
        intercept: Model intercept (2*AF for intercept-only).
        predictor_variant_ids: IDs of predictor variants.
        coefficients: Regression coefficients.
        is_intercept_only: True if no predictors (fallback to mean).
        predictor_chromosomes: Chromosome of each predictor. Index-aligned with
            predictor_variant_ids/coefficients.
        predictor_positions: Genomic position of each predictor. Index-aligned.
        predictor_counted_alleles: Allele each predictor coefficient counts
            (= ALT of the reference row backing the Z column). Index-aligned.
        predictor_other_alleles: The non-counted allele of each predictor
            (= REF of the reference row). Index-aligned. Together with
            chromosome/position this identifies the exact reference row, which
            disambiguates multiallelic loci.
        predictor_allele_frequencies: Frequency of the counted (ALT) allele for
            each predictor. Shape: (n_predictors,). Used for mean-substitution of
            missing predictors at inference time (mean dosage = 2*AF).
    """

    variant_id: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: Optional[str]
    beta: float
    allele_frequency: float
    imputation_r2: float
    residual_variance: float
    intercept: float
    predictor_variant_ids: List[str] = field(default_factory=list)
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    is_intercept_only: bool = False
    predictor_chromosomes: List[str] = field(default_factory=list)
    predictor_positions: List[int] = field(default_factory=list)
    predictor_counted_alleles: List[str] = field(default_factory=list)
    predictor_other_alleles: List[str] = field(default_factory=list)
    predictor_allele_frequencies: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    def to_dict(self) -> dict:
        """Convert to dictionary, handling numpy arrays.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        result = asdict(self)
        result["coefficients"] = self.coefficients.tolist()
        result["predictor_allele_frequencies"] = (
            self.predictor_allele_frequencies.tolist()
        )
        return result


@dataclass
class PredictionResult:
    """Output from PRS prediction.

    Attributes:
        prs: Raw PRS value.
        se: Standard error.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        prs_observed_component: Contribution from observed variants.
        prs_imputed_component: Contribution from imputed variants.
        n_variants_used: Total variants contributing.
        n_variants_imputed: Count of imputed variants.
        n_variants_intercept_only: Count using intercept-only models.
        n_user_variants_missing: User variants not available.
        n_truncated: Imputed dosages that were clipped.
        prs_scaled: Scaled PRS value (optional, for calibrated output).
        se_scaled: Scaled standard error (optional).
        ci_lower_scaled: Scaled CI lower bound (optional).
        ci_upper_scaled: Scaled CI upper bound (optional).
        n_observed_scored_direct: Count of observed variants scored from a direct,
            allele-oriented effect-allele dosage. ``None`` on the legacy
            allele-blind (dosage-dict) prediction path.
        n_observed_scored_via_fallback: Count of observed variants that could not be
            scored directly and were recovered via their per-variant fallback model
            (P1.8). ``None`` on the legacy allele-blind path.
        weighted_beta_via_fallback: Sum of ``|beta|`` over observed variants scored
            via fallback — a QC magnitude of how much PRS weight was recovered
            through the lower-confidence fallback path. ``None`` on the legacy path.
        unresolved_observed_ids: variant_ids of observed variants that could be scored
            neither directly nor via fallback (never silently dropped — surfaced
            here). ``None`` on the legacy allele-blind path.
        se_diagonal_lower_bound: The per-prediction diagonal SE,
            ``sqrt(Σ beta² · effective_residual_variance)`` for *this* user (inflated by
            P3.3 mean-substitution as predictors go missing). Since P4.1 the reported
            ``se`` is ``max(empirical_residual_sd, se_diagonal_lower_bound)``: the
            empirical SD is the LD-aware panel-wide baseline, and this diagonal value is
            a genuine lower bound that becomes the binding floor under heavy user
            missingness. Distinct from the fit-time, full-data
            ``CalibrationParams.diagonal_model_se_lower_bound``. ``None`` until set by
            ``predict`` (always populated on the predict path).
    """

    prs: float
    se: float
    ci_lower: float
    ci_upper: float
    prs_observed_component: float
    prs_imputed_component: float
    n_variants_used: int
    n_variants_imputed: int
    n_variants_intercept_only: int
    n_user_variants_missing: int
    n_truncated: int
    prs_scaled: Optional[float] = None
    se_scaled: Optional[float] = None
    ci_lower_scaled: Optional[float] = None
    ci_upper_scaled: Optional[float] = None
    n_observed_scored_direct: Optional[int] = None
    n_observed_scored_via_fallback: Optional[int] = None
    weighted_beta_via_fallback: Optional[float] = None
    unresolved_observed_ids: Optional[Tuple[str, ...]] = None
    se_diagonal_lower_bound: Optional[float] = None


@dataclass
class CalibrationParams:
    """Internal CV calibration parameters.

    Attributes:
        scaling_factor: Slope from regressing true on CV-predicted.
        scaling_factor_se: Standard error of scaling factor.
        calibration_intercept: Intercept from calibration regression.
        calibration_r2: R² of calibration fit.
        sd_cv_predicted: SD of CV-predicted PRS.
        sd_true: SD of true PRS.
        sd_scaled: SD of scaled predictions.
        attenuation_factor: Ratio of sd_cv/sd_true.
        n_calibration: Sample size used for calibration.
        raw_empirical_residual_sd: Empirical, score-level approximation error on the
            *raw* scale, ``std(s_true - s_cv, ddof=1)`` over out-of-fold CV scores.
            Because ``prs_raw`` is built exactly like ``s_cv`` (exact observed dosages
            plus CV-predicted imputed/projected terms), this is the honest SD for the
            raw interval — it captures the full ``betaᵀ Σ beta`` residual covariance
            (including LD off-diagonals) that the diagonal SE omits. ``None`` on
            artifacts predating P4.1 (forces the diagonal-SE fallback in ``predict``).
        calibrated_empirical_residual_sd: Empirical residual SD after the calibration
            transform, ``std(s_true - (intercept + slope * s_cv), ddof=1)`` — the SD
            for the ``prs_scaled`` interval. ``None`` on pre-P4.1 artifacts.
        diagonal_model_se_lower_bound: The *full-data* (no-missingness) diagonal SE
            scalar measured at fit time — imputation ``sqrt(Σ beta² · residual_var)``,
            projection ``sqrt(Σ cv_mse)``. A reference/QC lower bound on the model SE.
            NOTE: this is distinct from ``PredictionResult.se_diagonal_lower_bound``,
            which is the *per-prediction*, user-specific diagonal SE (inflated by P3.3
            missingness). ``predict`` recomputes the per-user value and must NOT read
            this stored scalar. ``None`` on pre-P4.1 artifacts.
    """

    scaling_factor: float
    scaling_factor_se: float
    calibration_intercept: float
    calibration_r2: float
    sd_cv_predicted: float
    sd_true: float
    sd_scaled: float
    attenuation_factor: float
    n_calibration: int
    raw_empirical_residual_sd: Optional[float] = None
    calibrated_empirical_residual_sd: Optional[float] = None
    diagonal_model_se_lower_bound: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Metrics from evaluation.

    Attributes:
        correlation: Pearson correlation.
        r2: R-squared.
        mae: Mean absolute error.
        rmse: Root mean squared error.
        spearman_rho: Spearman rank correlation.
        calibration_slope: Slope from calibration regression.
        calibration_intercept: Intercept from calibration regression.
    """

    correlation: float
    r2: float
    mae: float
    rmse: float
    spearman_rho: float
    calibration_slope: float
    calibration_intercept: float


@dataclass
class GenotypeData:
    """Container for loaded reference genotype data.

    Attributes:
        dosage_matrix: Genotype dosage matrix (n_samples x n_variants).
            Values are 0-2 representing allele counts, with NaN for missing.
        variant_info: DataFrame with variant metadata. Contains columns:
            variant_id, chromosome, position, ref_allele, alt_allele.
        sample_ids: List of sample identifiers.
        genome_build: Reference genome build (e.g., "GRCh37", "GRCh38").
        source_file: Path to the source file.
    """

    dosage_matrix: np.ndarray  # (n_samples x n_variants), values 0-2
    variant_info: pd.DataFrame  # variant_id, chromosome, position, ref_allele, alt_allele
    sample_ids: List[str]
    genome_build: Optional[str] = None
    source_file: Optional[str] = None

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self.dosage_matrix.shape[0]

    @property
    def n_variants(self) -> int:
        """Return the number of variants."""
        return self.dosage_matrix.shape[1]


@dataclass
class SingleVariantModelResult:
    """Result from fitting a single variant imputation model.

    Attributes:
        coefficients: Regression coefficients for predictor variants.
            Shape: (n_predictors,). Empty array for intercept-only models.
        intercept: Model intercept (mean of target for intercept-only models).
        cv_predictions: Out-of-fold cross-validation predictions.
            Shape: (n_samples,). Contains NaN for samples excluded due to
            missing values.
        cv_mse: Mean squared error from cross-validation.
        cv_r2: R-squared from cross-validation (0.0 for intercept-only).
        is_intercept_only: True if no predictors were used (fallback to mean).
        n_predictors: Number of predictor variants used.
        n_samples: Total number of samples (including those with NaN).
        l1_ratio: ElasticNet L1 ratio parameter used.
        alpha: ElasticNet regularization strength used.
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

    def to_dict(self) -> dict:
        """Convert to dictionary, handling numpy arrays.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "cv_predictions": self.cv_predictions.tolist(),
            "cv_mse": self.cv_mse,
            "cv_r2": self.cv_r2,
            "is_intercept_only": self.is_intercept_only,
            "n_predictors": self.n_predictors,
            "n_samples": self.n_samples,
            "l1_ratio": self.l1_ratio,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class TrainingFailure:
    """Structured reason a per-variant or per-region training fit failed.

    Captured when an ElasticNet fit raises a genuine exception inside the
    trainer. Degenerate-but-handled cases (zero-variance target, too few
    samples, no predictors) are downgraded to intercept-only models elsewhere
    and are *not* failures. Surfaced through the orchestrators'
    ``variant_dispositions`` / ``summary`` so a failed variant reports *why* it
    failed, not merely that it did.

    Attributes:
        unit_id: variant_id (imputation) or region_id (projection) that failed.
        error_type: Exception class name (``type(exc).__name__``).
        error_message: Exception message (``str(exc)``).
        n_valid_samples: Non-missing target samples at the failed fit, if known.
        target_variance: Variance of the non-missing target, if known.
        n_predictors: Number of windowed predictors at the failed fit, if known.
        member_ids: PRS variant IDs covered by a failed projection region, so a
            region failure can be attributed to each affected PRS variant. Empty
            for imputation, where the unit *is* the variant.
    """

    unit_id: str
    error_type: str
    error_message: str
    n_valid_samples: Optional[int] = None
    target_variance: Optional[float] = None
    n_predictors: Optional[int] = None
    member_ids: Tuple[str, ...] = ()


@dataclass
class GridSearchResult:
    """Result from global hyperparameter search.

    Attributes:
        best_l1_ratio: Optimal L1 ratio from grid search.
        best_alpha: Optimal regularization strength from grid search.
        best_mean_cv_mse: Mean CV MSE at optimal parameters.
        grid_results: Full grid search results. List of dicts with keys:
            l1_ratio, alpha, mean_cv_mse, std_cv_mse, n_variants_evaluated.
        n_variants_sampled: Number of variants used in search.
        n_variants_failed: Number of variants where fitting failed.
        failure_reasons: Map of exception class name -> count of sampled
            variants whose fit raised that exception during the search.
    """

    best_l1_ratio: float
    best_alpha: float
    best_mean_cv_mse: float
    grid_results: List[Dict[str, Any]]
    n_variants_sampled: int
    n_variants_failed: int
    failure_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result from training imputation models for all missing variants.

    Attributes:
        models: Dictionary mapping variant_id to trained ImputedVariantModel.
        cv_predictions: Dictionary mapping variant_id to out-of-fold CV predictions
            (shape (n_samples,) each), or ``None`` when the streaming backend
            accumulated calibration in-stream (no per-variant predictions retained).
        n_variants_trained: Number of variants successfully trained.
        n_variants_failed: Number of variants where training failed.
        n_intercept_only: Number of variants using intercept-only models.
        training_summary: Summary statistics including:
            - mean_r2: Mean imputation R² across variants.
            - median_r2: Median imputation R² across variants.
            - std_r2: Standard deviation of R² values.
            - min_r2: Minimum R² value.
            - max_r2: Maximum R² value.
            - n_high_quality: Count with R² > 0.8.
            - n_medium_quality: Count with 0.4 < R² <= 0.8.
            - n_low_quality: Count with R² <= 0.4.
            - mean_n_predictors: Average number of predictors per model.
        failures: Map of variant_id -> TrainingFailure for variants whose fit
            raised a genuine exception (structured reason; see TrainingFailure).
    """

    models: Dict[str, "ImputedVariantModel"]
    cv_predictions: Optional[Dict[str, np.ndarray]]
    n_variants_trained: int
    n_variants_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]
    failures: Dict[str, "TrainingFailure"] = field(default_factory=dict)


@dataclass
class OptunaSearchResult:
    """Result from Optuna hyperparameter search.

    Attributes:
        best_l1_ratio: Optimal L1 ratio from Optuna.
        best_alpha: Optimal regularization strength from Optuna.
        best_mean_cv_mse: Mean CV MSE at optimal parameters.
        n_trials: Number of trials completed.
        n_variants_sampled: Number of variants used in search.
        n_variants_failed: Number of variants where fitting failed at best params.
        trial_history: List of dicts with trial details (trial_number, l1_ratio, alpha, mean_cv_mse).
        optimization_time_seconds: Total optimization time.
        failure_reasons: Map of exception class name -> count of trial fits that
            raised that exception.
    """

    best_l1_ratio: float
    best_alpha: float
    best_mean_cv_mse: float
    n_trials: int
    n_variants_sampled: int
    n_variants_failed: int
    trial_history: List[Dict[str, Any]]
    optimization_time_seconds: float
    failure_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProjectionRegionModel:
    """Trained projection model for a single genomic region.

    For region R, this stores the learned weights a_R such that
    z_R^T a_R + intercept approximates S_R = sum(x_j * beta_j) for
    missing PRS variants j in R.

    Attributes:
        region_id: Unique identifier, format "chr{chrom}:{start}-{end}".
        chromosome: Chromosome.
        start: Region start position.
        end: Region end position.
        prs_variant_ids: Missing PRS variant IDs in this region.
        betas: Effect sizes for PRS variants in this region. Shape: (n_prs_variants,).
        predictor_variant_ids: Platform variant IDs used as predictors.
        coefficients: Learned regression coefficients. Shape: (n_predictors,).
            Empty array for intercept-only models.
        intercept: Model intercept (mean of S_R for intercept-only models).
        cv_mse: Cross-validated MSE for this region's PRS contribution.
        cv_r2: Cross-validated R-squared for this region's PRS contribution.
        is_intercept_only: True if no predictors available or all coefficients
            shrunk to zero.
        mean_prs_contribution: Mean of S_R across training samples.
        predictor_allele_frequencies: Frequency of the counted (ALT) allele for
            each predictor. Shape: (n_predictors,). Used for mean-substitution of
            missing predictors at inference time (mean dosage = 2*AF).
        predictor_chromosomes: Chromosome of each predictor. Index-aligned with
            predictor_variant_ids/coefficients.
        predictor_positions: Genomic position of each predictor. Index-aligned.
        predictor_counted_alleles: Allele each predictor coefficient counts
            (= ALT of the reference row backing the Z column). Index-aligned.
        predictor_other_alleles: The non-counted allele of each predictor
            (= REF of the reference row). Index-aligned. Together with
            chromosome/position this identifies the exact reference row, which
            disambiguates multiallelic loci.
        prs_positions: Genomic position of each PRS variant. Index-aligned with
            prs_variant_ids/betas. The chromosome is the region's `chromosome`
            (regions are single-chromosome by construction).
        prs_effect_alleles: Effect allele of each PRS variant (the allele `beta`
            is oriented to). Index-aligned with prs_variant_ids/betas.
        prs_other_alleles: Non-effect allele of each PRS variant (may be None).
            Index-aligned. Together with position these let a standalone scorer
            orient the true PRS via match_oriented_dosage instead of assuming
            effect==ALT at the first reference row.
        target_variance: Variance of the region target S_R across reference
            samples (the error variance of predicting with the regional mean, i.e.
            the intercept-only model). Used as the intercept_only_variance term in
            the missingness-aware uncertainty inflation at inference time (P3.3):
            as predictors are mean-substituted, the region's effective variance
            interpolates from cv_mse toward target_variance.
    """

    region_id: str
    chromosome: str
    start: int
    end: int
    prs_variant_ids: List[str]
    betas: np.ndarray
    predictor_variant_ids: List[str]
    coefficients: np.ndarray
    intercept: float
    cv_mse: float
    cv_r2: float
    is_intercept_only: bool
    mean_prs_contribution: float
    predictor_allele_frequencies: np.ndarray
    predictor_chromosomes: List[str] = field(default_factory=list)
    predictor_positions: List[int] = field(default_factory=list)
    predictor_counted_alleles: List[str] = field(default_factory=list)
    predictor_other_alleles: List[str] = field(default_factory=list)
    prs_positions: List[int] = field(default_factory=list)
    prs_effect_alleles: List[str] = field(default_factory=list)
    prs_other_alleles: List[Optional[str]] = field(default_factory=list)
    target_variance: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary, handling numpy arrays.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        result = asdict(self)
        result["betas"] = self.betas.tolist()
        result["coefficients"] = self.coefficients.tolist()
        result["predictor_allele_frequencies"] = self.predictor_allele_frequencies.tolist()
        return result


@dataclass
class ProjectionTrainingResult:
    """Result from training projection models for all regions.

    Attributes:
        region_models: Dict mapping region_id to ProjectionRegionModel.
        cv_predictions: Dict mapping region_id to out-of-fold CV predictions
            of S_R for each sample (shape (n_samples,) each), or ``None`` when the
            streaming backend accumulated calibration in-stream.
        n_regions_trained: Number of regions successfully trained.
        n_regions_failed: Number of regions where training failed.
        n_intercept_only: Number of regions using intercept-only models.
        training_summary: Summary statistics dict with keys:
            mean_r2, median_r2, std_r2, min_r2, max_r2,
            n_high_quality (r2 > 0.8), n_medium_quality (0.4-0.8),
            n_low_quality (r2 <= 0.4), mean_n_predictors,
            mean_n_prs_variants_per_region.
        failures: Map of region_id -> TrainingFailure for regions whose fit
            raised a genuine exception. Each carries member_ids (the PRS variant
            IDs in the failed region) so the failure can be attributed per variant.
    """

    region_models: Dict[str, "ProjectionRegionModel"]
    cv_predictions: Optional[Dict[str, np.ndarray]]
    n_regions_trained: int
    n_regions_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]
    failures: Dict[str, "TrainingFailure"] = field(default_factory=dict)
