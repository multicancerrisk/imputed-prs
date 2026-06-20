"""ImputationEvaluator class for evaluating fitted LinearImputationPRS models."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import warnings

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.types import (
    EvaluationMetrics,
    GenotypeData,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.evaluation.metrics import compute_prs_metrics, compute_percentile_concordance
from imputed_prs.evaluation.quality import summarize_imputation_quality
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.evaluation._scoring import (
    NeededVariant,
    is_hard_called,
    iter_sample_collections,
    observed_component_numeric,
    oriented_predictor_matrix,
)
from imputed_prs.models.predictor import PRSPredictor


@dataclass
class CrossValidationResult:
    """Result from cross-validation evaluation.

    Attributes:
        fold_metrics: List of EvaluationMetrics for each fold.
        mean_correlation: Mean Pearson correlation across folds.
        std_correlation: Standard deviation of correlation across folds.
        mean_r2: Mean R-squared across folds.
        std_r2: Standard deviation of R-squared across folds.
        mean_mae: Mean MAE across folds.
        mean_rmse: Mean RMSE across folds.
        mean_spearman: Mean Spearman correlation across folds.
        percentile_concordance: Aggregated percentile concordance metrics.
        n_folds: Number of folds used.
        n_samples_per_fold: Number of test samples in each fold.
    """

    fold_metrics: List[EvaluationMetrics]
    mean_correlation: float
    std_correlation: float
    mean_r2: float
    std_r2: float
    mean_mae: float
    mean_rmse: float
    mean_spearman: float
    percentile_concordance: Dict[str, float]
    n_folds: int
    n_samples_per_fold: List[int]


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis.

    Attributes:
        parameter_results: List of results for each parameter combination.
            Each dict contains 'params', 'metrics', and 'quality_summary'.
        best_params: Parameter combination with highest mean R-squared.
        best_metrics: EvaluationMetrics for best parameters.
        quality_summaries: List of quality summaries for each parameter combination.
    """

    parameter_results: List[Dict[str, Any]]
    best_params: Dict[str, float]
    best_metrics: EvaluationMetrics
    quality_summaries: List[Dict[str, Any]]


class ImputationEvaluator:
    """Evaluator for fitted LinearImputationPRS models.

    Provides methods to evaluate imputation-based PRS models on external
    held-out data, perform cross-validation, and analyze sensitivity to
    hyperparameters.

    Example:
        >>> model = LinearImputationPRS().fit(...)
        >>> evaluator = ImputationEvaluator(model)
        >>> metrics = evaluator.evaluate("held_out_genotypes.vcf.gz")
        >>> print(f"Correlation: {metrics.correlation:.3f}")

    Attributes:
        model: The fitted LinearImputationPRS model to evaluate.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
    """

    def __init__(self, model: "LinearImputationPRS", verbose: int = 1):
        """Initialize the evaluator.

        Args:
            model: A fitted LinearImputationPRS model.
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        # Import here to avoid circular import
        from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

        if not model.is_fitted:
            raise ModelNotFittedError("ImputationEvaluator requires a fitted model.")

        self.model = model
        self.verbose = verbose

    def evaluate(
        self,
        evaluation_genotypes: Union[str, Path, GenotypeData],
        percentile_thresholds: List[int] = None,
    ) -> EvaluationMetrics:
        """Evaluate the model on held-out genotype data.

        Computes imputed PRS and true PRS for all samples in the evaluation
        dataset, then calculates metrics comparing them.

        Args:
            evaluation_genotypes: Path to genotype file (VCF/PLINK) or
                pre-loaded GenotypeData object.
            percentile_thresholds: Percentile thresholds for concordance
                calculation. Default: [1, 5, 10].

        Returns:
            EvaluationMetrics comparing imputed vs true PRS.

        Raises:
            ValidationError: If evaluation data is invalid.
        """
        if percentile_thresholds is None:
            percentile_thresholds = [1, 5, 10]

        # Step 1: Load evaluation genotypes
        if isinstance(evaluation_genotypes, GenotypeData):
            genotype_data = evaluation_genotypes
        else:
            needed_variants = self._get_all_needed_variant_ids()
            genotype_data = load_genotypes(
                path=evaluation_genotypes,
                variant_ids=needed_variants,
            )

        if self.verbose >= 2:
            print(
                f"Loaded {genotype_data.n_samples} samples, "
                f"{genotype_data.n_variants} variants"
            )

        # Step 2: Compute true PRS
        true_prs = self._compute_true_prs(genotype_data)

        # Step 3: Compute imputed PRS
        imputed_prs = self._compute_imputed_prs_batch(genotype_data)

        if self.verbose >= 2:
            print(f"Computed PRS for {len(true_prs)} samples")

        # Step 4: Compute metrics
        metrics = compute_prs_metrics(imputed_prs, true_prs)

        return metrics

    def cross_validate(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> CrossValidationResult:
        """Perform k-fold cross-validation.

        Splits the reference genotype data into k folds, trains a model on
        k-1 folds and evaluates on the held-out fold. Repeats for all folds.

        Args:
            reference_genotypes: Path to reference genotype file.
            prs_definition: PRS definition (PGS ID, path, or DataFrame).
            platform_name: Name of pre-built platform.
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            n_folds: Number of cross-validation folds. Must be >= 2.
            random_state: Random seed for reproducibility.

        Returns:
            CrossValidationResult with fold metrics and aggregated statistics.

        Raises:
            ValidationError: If inputs are invalid (e.g., no platform source,
                n_folds < 2).
        """
        # Import here to avoid circular import
        from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

        # Validate inputs
        platform_sources = [
            platform_name is not None,
            platform_manifest is not None,
            platform_variants is not None,
        ]
        if sum(platform_sources) != 1:
            raise ValidationError(
                "Exactly one platform source must be provided: "
                "platform_name, platform_manifest, or platform_variants"
            )

        if n_folds < 2:
            raise ValidationError(f"n_folds must be >= 2, got {n_folds}")

        # Load full reference genotypes
        genotype_data = load_genotypes(path=reference_genotypes)

        if self.verbose >= 1:
            print(f"Loaded {genotype_data.n_samples} samples for cross-validation")

        # Create fold indices
        n_samples = genotype_data.n_samples
        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        fold_size = n_samples // n_folds
        fold_indices = []
        for i in range(n_folds):
            start = i * fold_size
            if i == n_folds - 1:
                # Last fold gets remaining samples
                end = n_samples
            else:
                end = start + fold_size
            fold_indices.append(indices[start:end])

        # Run cross-validation
        fold_metrics: List[EvaluationMetrics] = []
        n_samples_per_fold: List[int] = []
        all_imputed_prs = []
        all_true_prs = []

        for fold_idx in range(n_folds):
            if self.verbose >= 1:
                print(f"Processing fold {fold_idx + 1}/{n_folds}...")

            # Split samples
            test_indices = fold_indices[fold_idx]
            train_indices = np.concatenate(
                [fold_indices[i] for i in range(n_folds) if i != fold_idx]
            )

            # Create train/test genotype data
            train_data = self._subset_genotype_data(genotype_data, train_indices)
            test_data = self._subset_genotype_data(genotype_data, test_indices)

            # Train model on training data
            fold_model = LinearImputationPRS(
                window_size=self.model.window_size,
                tuning_scope=self.model.tuning_scope,
                l1_ratio=self.model.l1_ratio,
                alpha=self.model.alpha,
                cv_folds=self.model.cv_folds,
                n_jobs=self.model.n_jobs,
                random_state=random_state,
                max_predictors=self.model.max_predictors,
                verbose=0,  # Suppress output during CV
            )

            # Create temporary file for training data
            # For CV, we need to fit on train data and evaluate on test data
            # This requires writing the train data to a temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                self._write_genotype_data_to_vcf(train_data, tmp_path)

                fold_model.fit(
                    reference_genotypes=tmp_path,
                    prs_definition=prs_definition,
                    platform_name=platform_name,
                    platform_manifest=platform_manifest,
                    platform_variants=platform_variants,
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            # Create evaluator for fold model and evaluate on test data
            fold_evaluator = ImputationEvaluator(fold_model, verbose=0)
            fold_true_prs = fold_evaluator._compute_true_prs(test_data)
            fold_imputed_prs = fold_evaluator._compute_imputed_prs_batch(test_data)

            # Compute fold metrics
            metrics = compute_prs_metrics(fold_imputed_prs, fold_true_prs)
            fold_metrics.append(metrics)
            n_samples_per_fold.append(len(test_indices))

            # Collect for percentile concordance
            all_imputed_prs.extend(fold_imputed_prs.tolist())
            all_true_prs.extend(fold_true_prs.tolist())

        # Aggregate metrics
        correlations = [m.correlation for m in fold_metrics]
        r2_values = [m.r2 for m in fold_metrics]
        mae_values = [m.mae for m in fold_metrics]
        rmse_values = [m.rmse for m in fold_metrics]
        spearman_values = [m.spearman_rho for m in fold_metrics]

        # Compute percentile concordance on all samples
        percentile_concordance = compute_percentile_concordance(
            np.array(all_imputed_prs),
            np.array(all_true_prs),
        )

        return CrossValidationResult(
            fold_metrics=fold_metrics,
            mean_correlation=float(np.mean(correlations)),
            std_correlation=float(np.std(correlations)),
            mean_r2=float(np.mean(r2_values)),
            std_r2=float(np.std(r2_values)),
            mean_mae=float(np.mean(mae_values)),
            mean_rmse=float(np.mean(rmse_values)),
            mean_spearman=float(np.mean(spearman_values)),
            percentile_concordance=percentile_concordance,
            n_folds=n_folds,
            n_samples_per_fold=n_samples_per_fold,
        )

    def sensitivity_analysis(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        parameter_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> SensitivityResult:
        """Analyze model sensitivity to hyperparameters.

        Trains models across a grid of hyperparameter values and compares
        their performance.

        Args:
            reference_genotypes: Path to reference genotype file.
            prs_definition: PRS definition (PGS ID, path, or DataFrame).
            platform_name: Name of pre-built platform.
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            parameter_grid: Dictionary mapping parameter names to lists of
                values to try. Default grid:
                {'window_size': [500_000, 1_000_000, 2_000_000],
                 'l1_ratio': [0.1, 0.5, 0.9],
                 'alpha': [0.001, 0.01, 0.1]}
            cv_folds: Number of CV folds for evaluation. Default: 5.
            random_state: Random seed for reproducibility.

        Returns:
            SensitivityResult with results for each parameter combination.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Import here to avoid circular import
        from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

        # Validate inputs
        platform_sources = [
            platform_name is not None,
            platform_manifest is not None,
            platform_variants is not None,
        ]
        if sum(platform_sources) != 1:
            raise ValidationError(
                "Exactly one platform source must be provided: "
                "platform_name, platform_manifest, or platform_variants"
            )

        # Default parameter grid
        if parameter_grid is None:
            parameter_grid = {
                "window_size": [500_000, 1_000_000, 2_000_000],
                "l1_ratio": [0.1, 0.5, 0.9],
                "alpha": [0.001, 0.01, 0.1],
            }

        # Generate parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))

        if self.verbose >= 1:
            print(f"Testing {len(param_combinations)} parameter combinations...")

        # Load reference genotypes once
        genotype_data = load_genotypes(path=reference_genotypes)

        # Create temporary file for genotype data
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            self._write_genotype_data_to_vcf(genotype_data, tmp_path)

            # Test each parameter combination
            parameter_results: List[Dict[str, Any]] = []
            quality_summaries: List[Dict[str, Any]] = []
            best_r2 = -np.inf
            best_params: Dict[str, float] = {}
            best_metrics: Optional[EvaluationMetrics] = None

            for idx, combo in enumerate(param_combinations):
                params = dict(zip(param_names, combo))

                if self.verbose >= 1:
                    print(f"  [{idx + 1}/{len(param_combinations)}] Testing {params}")

                try:
                    # Create model with these parameters
                    test_model = LinearImputationPRS(
                        window_size=params.get("window_size", self.model.window_size),
                        tuning_scope="none",  # Use specified params directly
                        l1_ratio=params.get("l1_ratio", self.model.l1_ratio),
                        alpha=params.get("alpha", self.model.alpha),
                        cv_folds=cv_folds,
                        n_jobs=self.model.n_jobs,
                        random_state=random_state,
                        verbose=0,
                    )

                    test_model.fit(
                        reference_genotypes=tmp_path,
                        prs_definition=prs_definition,
                        platform_name=platform_name,
                        platform_manifest=platform_manifest,
                        platform_variants=platform_variants,
                    )

                    # Get quality summary
                    if test_model._imputed_models:
                        models_dict = {
                            m.variant_id: m for m in test_model._imputed_models
                        }
                        quality_summary = summarize_imputation_quality(models_dict)
                    else:
                        quality_summary = {"mean_r2": None, "n_total": 0}

                    # Evaluate using internal CV metrics
                    evaluator = ImputationEvaluator(test_model, verbose=0)
                    metrics = evaluator.evaluate(genotype_data)

                    parameter_results.append({
                        "params": params,
                        "metrics": metrics,
                        "quality_summary": quality_summary,
                    })
                    quality_summaries.append(quality_summary)

                    # Track best
                    if metrics.r2 > best_r2:
                        best_r2 = metrics.r2
                        best_params = params
                        best_metrics = metrics

                except Exception as e:
                    if self.verbose >= 1:
                        print(f"    Warning: Failed for {params}: {e}")
                    parameter_results.append({
                        "params": params,
                        "metrics": None,
                        "error": str(e),
                    })

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if best_metrics is None:
            raise ValidationError("All parameter combinations failed")

        return SensitivityResult(
            parameter_results=parameter_results,
            best_params=best_params,
            best_metrics=best_metrics,
            quality_summaries=quality_summaries,
        )

    def _get_all_needed_variant_ids(self) -> Set[str]:
        """Get union of PRS variant IDs and predictor variant IDs.

        Returns:
            Set of all variant IDs needed for evaluation.
        """
        needed = set()

        # Add observed variant IDs
        for var in self.model.observed_variants:
            needed.add(var.variant_id)

        # Add imputed variant IDs and their predictors
        for model in self.model.imputed_models:
            needed.add(model.variant_id)
            needed.update(model.predictor_variant_ids)

        return needed

    def _compute_true_prs(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute true PRS from full genotype data.

        Calculates sum(effect_dosage * beta) for all placed PRS variants, using
        effect-allele-oriented dosages so the gold standard is allele-correct
        (and consistent with the imputation target the models were trained on).

        Args:
            genotype_data: GenotypeData containing all samples.

        Returns:
            Array of true PRS values (n_samples,).
        """
        from imputed_prs.core.harmonizer import (
            build_reference_allele_index,
            match_oriented_dosage,
        )

        reference_index = build_reference_allele_index(genotype_data.variant_info)
        n_samples = genotype_data.n_samples
        true_prs = np.zeros(n_samples)

        # All placed variants (observed + imputed); dropped variants are absent
        # from both lists and therefore correctly excluded.
        placed = [
            (v.chromosome, v.position, v.effect_allele, v.other_allele, v.beta)
            for v in self.model.observed_variants
        ] + [
            (m.chromosome, m.position, m.effect_allele, m.other_allele, m.beta)
            for m in self.model.imputed_models
        ]

        for chromosome, position, effect_allele, other_allele, beta in placed:
            match = match_oriented_dosage(
                chromosome, position, effect_allele, other_allele,
                genotype_data.variant_info, genotype_data.dosage_matrix,
                reference_index,
            )
            if match is None:
                continue
            dosages = match[1]
            valid_mask = ~np.isnan(dosages)
            true_prs[valid_mask] += dosages[valid_mask] * beta

        return true_prs

    def _compute_imputed_prs_batch(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute predicted (observed + imputed) PRS for all samples.

        Routes through the same allele-oriented semantics as the browser/upload
        path so train/eval and browser cannot diverge (P1.6):

        - **hard-called** integer dosages → render genotype strings per sample and
          replay ``PRSPredictor.predict`` (the literal upload path);
        - **continuous** DS/GP dosages → a role-aware numeric scorer that orients
          each predictor via ``match_oriented_dosage`` from the stored P1.3 allele
          metadata.

        The two paths agree on integer biallelic data (golden test in
        ``tests/test_round_trip.py``).

        Args:
            genotype_data: GenotypeData containing all samples.

        Returns:
            Array of predicted PRS values (n_samples,).
        """
        if is_hard_called(genotype_data.dosage_matrix):
            return self._predicted_prs_via_strings(genotype_data)
        return self._predicted_prs_numeric(genotype_data)

    def _needed_for_render(self) -> List[NeededVariant]:
        """Variants the string scorer must resolve: observed terms (effect/other)
        plus every predictor (counted/other from P1.3 metadata). The allele pair
        only selects the reference row; ``count_allele`` re-orients per role."""
        needed: List[NeededVariant] = []
        for var in self.model.observed_variants:
            needed.append(
                (
                    var.variant_id,
                    var.chromosome,
                    var.position,
                    var.effect_allele,
                    var.other_allele,
                )
            )
        for model in self.model.imputed_models:
            if model.is_intercept_only:
                continue
            for i, pred_id in enumerate(model.predictor_variant_ids):
                needed.append(
                    (
                        pred_id,
                        model.predictor_chromosomes[i],
                        model.predictor_positions[i],
                        model.predictor_counted_alleles[i],
                        model.predictor_other_alleles[i],
                    )
                )
        return needed

    def _predicted_prs_via_strings(self, genotype_data: GenotypeData) -> np.ndarray:
        """Hard-call path: render genotype strings per sample and replay the
        browser scorer (observed + imputed in one oriented ``predict`` call)."""
        needed = self._needed_for_render()
        predictor = PRSPredictor(
            observed_variants=self.model.observed_variants,
            imputed_models=self.model.imputed_models,
            calibration_params=None,  # evaluation scores the raw, uncalibrated model
        )
        predicted = np.zeros(genotype_data.n_samples)
        for sample_idx, collection in enumerate(
            iter_sample_collections(genotype_data, needed)
        ):
            result = predictor.predict(
                {}, apply_calibration=False, raw_genotypes=collection
            )
            predicted[sample_idx] = result.prs
        return predicted

    def _predicted_prs_numeric(self, genotype_data: GenotypeData) -> np.ndarray:
        """Continuous path: orient observed and predictor dosages numerically via
        ``match_oriented_dosage`` and run the imputation regression (clipped to
        ``[0, 2]`` to match ``clip_and_adjust_variance`` in the per-user scorer)."""
        from imputed_prs.core.harmonizer import build_reference_allele_index

        reference_index = build_reference_allele_index(genotype_data.variant_info)
        n_samples = genotype_data.n_samples

        predicted = observed_component_numeric(
            genotype_data, reference_index, self.model.observed_variants
        )

        for model in self.model.imputed_models:
            if model.is_intercept_only or not model.predictor_variant_ids:
                raw = np.full(n_samples, float(model.intercept))
            else:
                z = oriented_predictor_matrix(
                    genotype_data,
                    reference_index,
                    model.predictor_chromosomes,
                    model.predictor_positions,
                    model.predictor_counted_alleles,
                    model.predictor_other_alleles,
                    model.predictor_allele_frequencies,
                )
                raw = z @ np.asarray(model.coefficients, dtype=np.float64) + float(
                    model.intercept
                )
            predicted += np.clip(raw, 0.0, 2.0) * model.beta

        return predicted

    def _subset_genotype_data(
        self, genotype_data: GenotypeData, sample_indices: np.ndarray
    ) -> GenotypeData:
        """Create a subset of genotype data with selected samples.

        Args:
            genotype_data: Original GenotypeData.
            sample_indices: Indices of samples to include.

        Returns:
            New GenotypeData with only selected samples.
        """
        return GenotypeData(
            dosage_matrix=genotype_data.dosage_matrix[sample_indices, :],
            variant_info=genotype_data.variant_info.copy(),
            sample_ids=[genotype_data.sample_ids[i] for i in sample_indices],
            genome_build=genotype_data.genome_build,
            source_file=genotype_data.source_file,
        )

    def _write_genotype_data_to_vcf(
        self, genotype_data: GenotypeData, path: Path
    ) -> None:
        """Write GenotypeData to a VCF file.

        Args:
            genotype_data: GenotypeData to write.
            path: Output path.
        """
        lines = []

        # Header
        lines.append("##fileformat=VCFv4.2")
        lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

        # Column header
        header_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        header_cols.extend(genotype_data.sample_ids)
        lines.append("\t".join(header_cols))

        # Variants
        for var_idx, row in genotype_data.variant_info.iterrows():
            chrom = str(row["chromosome"])
            pos = str(int(row["position"]))
            var_id = row["variant_id"]
            ref = row.get("ref_allele", "N")
            alt = row.get("alt_allele", "N")
            if pd.isna(ref):
                ref = "N"
            if pd.isna(alt):
                alt = "N"

            record = [chrom, pos, var_id, ref, alt, ".", ".", ".", "GT"]

            # Add genotypes for each sample
            for sample_idx in range(genotype_data.n_samples):
                dosage = genotype_data.dosage_matrix[sample_idx, var_idx]
                if np.isnan(dosage):
                    gt = "./."
                elif dosage < 0.5:
                    gt = "0/0"
                elif dosage < 1.5:
                    gt = "0/1"
                else:
                    gt = "1/1"
                record.append(gt)

            lines.append("\t".join(record))

        path.write_text("\n".join(lines) + "\n")
