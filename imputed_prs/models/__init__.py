"""Models module for imputation models."""

from imputed_prs.core.types import (
    GridSearchResult,
    OptunaSearchResult,
    SingleVariantModelResult,
    TrainingResult,
)
from imputed_prs.models.bounding import (
    clip_and_adjust_variance,
    compute_truncation_adjustment_factor,
    truncated_normal_mean,
    truncated_normal_variance,
)
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.metrics import compute_cv_r2
from imputed_prs.models.predictor import PRSPredictor, compute_imputed_prs, compute_observed_prs
from imputed_prs.models.trainer import ImputationModelTrainer
from imputed_prs.models.tuning import global_hyperparameter_search, optuna_hyperparameter_search

__all__ = [
    "clip_and_adjust_variance",
    "compute_cv_r2",
    "compute_imputed_prs",
    "compute_observed_prs",
    "compute_truncation_adjustment_factor",
    "fit_single_variant_model",
    "global_hyperparameter_search",
    "GridSearchResult",
    "ImputationModelTrainer",
    "optuna_hyperparameter_search",
    "OptunaSearchResult",
    "PRSPredictor",
    "SingleVariantModelResult",
    "TrainingResult",
    "truncated_normal_mean",
    "truncated_normal_variance",
]
