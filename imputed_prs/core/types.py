"""Core data types for the imputed-prs library."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import numpy as np


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
