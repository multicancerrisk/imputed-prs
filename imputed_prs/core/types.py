"""Core data types for the imputed-prs library."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
    """

    variant_id: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: Optional[str]
    beta: float


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

    def to_dict(self) -> dict:
        """Convert to dictionary, handling numpy arrays.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        result = asdict(self)
        result["coefficients"] = self.coefficients.tolist()
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
    """

    best_l1_ratio: float
    best_alpha: float
    best_mean_cv_mse: float
    grid_results: List[Dict[str, Any]]
    n_variants_sampled: int
    n_variants_failed: int


@dataclass
class TrainingResult:
    """Result from training imputation models for all missing variants.

    Attributes:
        models: Dictionary mapping variant_id to trained ImputedVariantModel.
        cv_predictions: Dictionary mapping variant_id to out-of-fold CV predictions.
            Shape of each array: (n_samples,).
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
    """

    models: Dict[str, "ImputedVariantModel"]
    cv_predictions: Dict[str, np.ndarray]
    n_variants_trained: int
    n_variants_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]
