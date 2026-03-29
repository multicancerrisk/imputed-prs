"""Evaluation module for calibration and metrics."""

from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
)
from imputed_prs.evaluation.evaluator import (
    CrossValidationResult,
    ImputationEvaluator,
    SensitivityResult,
)
from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator
from imputed_prs.evaluation.metrics import (
    compute_prs_metrics,
    compute_percentile_concordance,
)
from imputed_prs.evaluation.quality import (
    summarize_imputation_quality,
)
from imputed_prs.evaluation.plotting import (
    plot_calibration,
    plot_imputation_quality,
    plot_variance_contribution,
    plot_truncation_diagnostics,
)

__all__ = [
    "compute_cv_predicted_prs",
    "estimate_cv_calibration",
    "compute_prs_metrics",
    "compute_percentile_concordance",
    "summarize_imputation_quality",
    "CrossValidationResult",
    "ImputationEvaluator",
    "ProjectionEvaluator",
    "SensitivityResult",
    "plot_calibration",
    "plot_imputation_quality",
    "plot_variance_contribution",
    "plot_truncation_diagnostics",
]
