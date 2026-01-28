"""Main LinearImputationPRS class for training and prediction."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    PredictionResult,
    TrainingResult,
    VariantInfo,
)


class LinearImputationPRS:
    """High-level API for training and using imputation-based PRS models.

    This class provides a unified interface for:
    - Loading PRS definitions and platform information
    - Training imputation models on reference genotype data
    - Computing PRS predictions with uncertainty estimates
    - Exporting trained models to portable formats

    Example:
        >>> model = LinearImputationPRS(window_size=1_000_000, cv_folds=5)
        >>> model.fit(
        ...     reference_genotypes="1000g_eur.vcf.gz",
        ...     prs_definition="PGS000004",
        ...     platform_name="23andme_v5",
        ... )
        >>> result = model.predict("user_genotypes.txt")
        >>> print(f"PRS: {result.prs:.3f} (95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")

    Attributes:
        window_size: Size of genomic window (bp) for selecting predictor variants.
        tuning_scope: Hyperparameter tuning strategy ("global", "per_variant", or "none").
        l1_ratio: ElasticNet L1/L2 mixing parameter (0=Ridge, 1=Lasso).
        alpha: ElasticNet regularization strength.
        cv_folds: Number of cross-validation folds.
        n_jobs: Number of parallel jobs for training.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        tuning_scope: Literal["global", "per_variant", "none"] = "global",
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        verbose: int = 1,
    ):
        """Initialize LinearImputationPRS model.

        Args:
            window_size: Size of genomic window (bp) for selecting predictor variants.
                Larger windows include more potential predictors but increase computation.
                Default: 1,000,000 (1 Mb).
            tuning_scope: Hyperparameter tuning strategy:
                - "global": Tune once on subset of variants (recommended)
                - "per_variant": Tune separately for each variant (slow)
                - "none": Use provided l1_ratio and alpha directly
                Default: "global".
            l1_ratio: ElasticNet L1/L2 mixing parameter. 0=pure Ridge, 1=pure Lasso.
                Only used when tuning_scope="none". Default: 0.5.
            alpha: ElasticNet regularization strength. Larger values = more regularization.
                Only used when tuning_scope="none". Default: 0.01.
            cv_folds: Number of cross-validation folds for training and calibration.
                Default: 5.
            n_jobs: Number of parallel jobs for training (-1 for all CPUs).
                Default: 1 (sequential).
            random_state: Random seed for reproducibility. Default: None.
            max_predictors: Maximum number of predictor variants per model.
                If None, uses all variants in window. Default: None.
            verbose: Verbosity level. 0=silent, 1=progress bar, 2=debug output.
                Default: 1.
        """
        # Configuration parameters
        self.window_size = window_size
        self.tuning_scope = tuning_scope
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.verbose = verbose

        # Fitted state (populated by fit())
        self._is_fitted: bool = False
        self._observed_variants: Optional[List[VariantInfo]] = None
        self._imputed_models: Optional[List[ImputedVariantModel]] = None
        self._calibration_params: Optional[CalibrationParams] = None
        self._evaluation_metrics: Optional[EvaluationMetrics] = None
        self._training_result: Optional[TrainingResult] = None
        self._platform_variant_index: Optional[Dict[str, int]] = None

        # Metadata (populated by fit())
        self._prs_id: Optional[str] = None
        self._platform_name: Optional[str] = None
        self._genome_build: Optional[str] = None
        self._model_name: Optional[str] = None

    def fit(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        genome_build: Optional[str] = None,
        prs_id: Optional[str] = None,
        model_name: Optional[str] = None,
        evaluation_genotypes: Optional[Union[str, Path]] = None,
    ) -> "LinearImputationPRS":
        """Train imputation models on reference genotype data.

        Exactly one platform source must be provided (platform_name, platform_manifest,
        or platform_variants).

        Args:
            reference_genotypes: Path to reference genotype file (VCF or PLINK format).
            prs_definition: PRS definition as PGS Catalog ID (e.g., "PGS000004"),
                file path, or DataFrame with variant weights.
            platform_name: Name of pre-built platform (e.g., "23andme_v5").
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            genome_build: Genome build ("GRCh37" or "GRCh38"). Auto-detected if None.
            prs_id: PRS identifier for metadata.
            model_name: Human-readable model name for metadata.
            evaluation_genotypes: Optional holdout genotypes for external evaluation.

        Returns:
            self (for method chaining).

        Raises:
            ValidationError: If inputs are invalid or incompatible.
            DataLoadError: If files cannot be loaded.
        """
        # TODO: Implement in Phase 7.2
        raise NotImplementedError("fit() will be implemented in Phase 7.2")

    def predict(
        self,
        user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
        apply_calibration: bool = True,
    ) -> PredictionResult:
        """Compute PRS for user genotypes.

        Args:
            user_genotypes: User genotype data as:
                - File path (DTC format auto-detected)
                - DataFrame with variant_id and genotype columns
                - Dict mapping variant_id to dosage values
            apply_calibration: Whether to apply calibration scaling.
                Default: True.

        Returns:
            PredictionResult with PRS value, uncertainty estimates, and diagnostics.

        Raises:
            ModelNotFittedError: If fit() has not been called.
            DataLoadError: If user genotype file cannot be loaded.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict()."
            )
        # TODO: Implement in Phase 7.3
        raise NotImplementedError("predict() will be implemented in Phase 7.3")

    def export(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        formats: Optional[List[str]] = None,
        include_variance_scaling: bool = True,
    ) -> Dict[str, Path]:
        """Export trained model to portable formats.

        Args:
            output_dir: Directory for output files.
            model_name: Base name for output files. Uses self._model_name if None.
            formats: List of formats to export. Options: "json", "arrow", "parquet",
                "hdf5", "csv". Default: ["json", "hdf5"].
            include_variance_scaling: Whether to include variance/SE components.
                Default: True.

        Returns:
            Dict mapping format name to output file path.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before export()."
            )
        # TODO: Implement in Phase 7.4
        raise NotImplementedError("export() will be implemented in Phase 7.4")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearImputationPRS":
        """Load a trained model from file.

        Args:
            path: Path to saved model file (HDF5 or JSON format).

        Returns:
            Loaded LinearImputationPRS instance ready for prediction.

        Raises:
            DataLoadError: If file cannot be loaded.
        """
        # TODO: Implement in Phase 7.4
        raise NotImplementedError("load() will be implemented in Phase 7.4")

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def variant_table(self) -> pd.DataFrame:
        """Per-variant summary table with status and quality metrics.

        Returns:
            DataFrame with columns: variant_id, chromosome, position, effect_allele,
            other_allele, beta, status, imputation_r2, allele_frequency, n_predictors.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        # Build variant table from observed and imputed variants
        rows = []

        for var in self._observed_variants or []:
            rows.append({
                "variant_id": var.variant_id,
                "chromosome": var.chromosome,
                "position": var.position,
                "effect_allele": var.effect_allele,
                "other_allele": var.other_allele,
                "beta": var.beta,
                "status": "observed",
                "imputation_r2": None,
                "allele_frequency": None,
                "n_predictors": 0,
            })

        for model in self._imputed_models or []:
            rows.append({
                "variant_id": model.variant_id,
                "chromosome": model.chromosome,
                "position": model.position,
                "effect_allele": model.effect_allele,
                "other_allele": model.other_allele,
                "beta": model.beta,
                "status": "intercept_only" if model.is_intercept_only else "imputed",
                "imputation_r2": model.imputation_r2,
                "allele_frequency": model.allele_frequency,
                "n_predictors": len(model.predictor_variant_ids),
            })

        return pd.DataFrame(rows)

    @property
    def summary(self) -> Dict[str, Any]:
        """Model summary with counts and quality statistics.

        Returns:
            Dict with keys: n_total_variants, n_observed, n_imputed, n_intercept_only,
            mean_imputation_r2, prs_id, platform_name, genome_build, etc.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )

        n_observed = len(self._observed_variants or [])
        n_imputed = len(self._imputed_models or [])
        n_intercept_only = sum(
            1 for m in (self._imputed_models or []) if m.is_intercept_only
        )

        # Compute mean R² for imputed variants
        r2_values = [m.imputation_r2 for m in (self._imputed_models or [])
                     if not m.is_intercept_only]
        mean_r2 = float(np.mean(r2_values)) if r2_values else None

        return {
            "n_total_variants": n_observed + n_imputed,
            "n_observed": n_observed,
            "n_imputed": n_imputed,
            "n_intercept_only": n_intercept_only,
            "mean_imputation_r2": mean_r2,
            "prs_id": self._prs_id,
            "platform_name": self._platform_name,
            "genome_build": self._genome_build,
            "model_name": self._model_name,
            "window_size": self.window_size,
            "cv_folds": self.cv_folds,
        }

    @property
    def evaluation_metrics(self) -> Optional[EvaluationMetrics]:
        """Evaluation metrics from training (if available).

        Returns:
            EvaluationMetrics or None if no evaluation was performed.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._evaluation_metrics

    @property
    def calibration_params(self) -> Optional[CalibrationParams]:
        """Calibration parameters from CV training.

        Returns:
            CalibrationParams or None if calibration was not performed.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._calibration_params

    @property
    def observed_variants(self) -> List[VariantInfo]:
        """List of observed (directly measured) variants.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._observed_variants or []

    @property
    def imputed_models(self) -> List[ImputedVariantModel]:
        """List of imputed variant models.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._imputed_models or []

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LinearImputationPRS(window_size={self.window_size}, "
            f"cv_folds={self.cv_folds}, status={status})"
        )
