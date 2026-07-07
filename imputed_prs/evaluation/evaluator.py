"""ImputationEvaluator class for evaluating fitted LinearImputationPRS models."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
from imputed_prs.io.platform_loader import resolve_platform_variant_set
from imputed_prs.core.harmonizer import ReferenceAlleleResolver
from imputed_prs.evaluation._scoring import (
    NeededVariant,
    iter_sample_collections,
    observed_component_numeric,
    oriented_predictor_matrix,
    should_use_batch,
)
from imputed_prs.models.predictor import PRSPredictor
from imputed_prs.models.vectorized_predictor import (
    accumulate_true_prs,
    build_chip_axis,
    build_coef_csr,
    oriented_chip_matrix,
    panel_impute_prs,
)


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


def _run_sensitivity_combo(payload):
    """Fit + evaluate ONE hyperparameter combo (picklable worker for ``parallel_map``).

    Returns the ``parameter_results`` dict — ``{"params", "metrics", "quality_summary"}``
    on success, ``{"params", "metrics": None, "error"}`` on failure — so the parent can
    collect in canonical combo order and pick the best deterministically. The inner fit
    runs single-process (``n_workers=1``); the outer combo pool owns the cores.
    """
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

    (params, base, prs_definition, platform_name, platform_manifest,
     platform_variants, cv_folds, random_state, inner_n_jobs, genotype_data,
     platform_variant_set, platform_info, cache_dir) = payload
    try:
        test_model = LinearImputationPRS(
            window_size=params.get("window_size", base["window_size"]),
            tuning_scope="none",  # use the specified params directly
            l1_ratio=params.get("l1_ratio", base["l1_ratio"]),
            alpha=params.get("alpha", base["alpha"]),
            cv_folds=cv_folds,
            n_jobs=inner_n_jobs,
            n_workers=1,
            random_state=random_state,
            backend=base.get("backend", "auto"),
            verbose=0,
        )
        test_model.fit(
            reference_genotypes=genotype_data,
            prs_definition=prs_definition,
            platform_name=platform_name,
            platform_manifest=platform_manifest,
            platform_variants=platform_variants,
            _platform_variant_set=platform_variant_set,
            _platform_info=platform_info,
            _cache_dir=cache_dir,
        )
        if test_model._imputed_models:
            models_dict = {m.variant_id: m for m in test_model._imputed_models}
            quality_summary = summarize_imputation_quality(models_dict)
        else:
            quality_summary = {"mean_r2": None, "n_total": 0}
        evaluator = ImputationEvaluator(test_model, verbose=0)
        metrics = evaluator.evaluate(genotype_data)
        return {"params": params, "metrics": metrics, "quality_summary": quality_summary}
    except Exception as e:  # noqa: BLE001 — mirror the serial path: record, don't crash
        return {"params": params, "metrics": None, "error": str(e)}


def _run_sensitivity_group(payload):
    """Fit + evaluate a GROUP of combos that share an accumulation key (Phase 9).

    The combos run serially in this one process sharing a block memo: the first fit
    collects the streaming Gram blocks into it, the rest re-solve them for their own
    ``(l1_ratio, alpha)`` — so the streaming accumulation runs ONCE per group, not once
    per combo. The evaluator scores raw models, so the re-solved (uncalibrated) grid
    models give identical metrics to a full fit. Returns the list of per-combo result
    dicts (same shape as :func:`_run_sensitivity_combo`), one per combo in the group.
    """
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

    (group, base, prs_definition, platform_name, platform_manifest, platform_variants,
     cv_folds, random_state, inner_n_jobs, genotype_data, platform_variant_set,
     platform_info, cache_dir) = payload
    memo: Dict[str, Any] = {}  # shared across the group: {ref_info, collected}
    out: List[Dict[str, Any]] = []
    for params in group:
        try:
            test_model = LinearImputationPRS(
                window_size=params.get("window_size", base["window_size"]),
                tuning_scope="none",
                l1_ratio=params.get("l1_ratio", base["l1_ratio"]),
                alpha=params.get("alpha", base["alpha"]),
                cv_folds=cv_folds,
                n_jobs=inner_n_jobs,
                n_workers=1,
                random_state=random_state,
                backend=base.get("backend", "auto"),
                verbose=0,
            )
            test_model.fit(
                reference_genotypes=genotype_data,
                prs_definition=prs_definition,
                platform_name=platform_name,
                platform_manifest=platform_manifest,
                platform_variants=platform_variants,
                _platform_variant_set=platform_variant_set,
                _platform_info=platform_info,
                _block_memo=memo,
                _cache_dir=cache_dir,
            )
            if test_model._imputed_models:
                models_dict = {m.variant_id: m for m in test_model._imputed_models}
                quality_summary = summarize_imputation_quality(models_dict)
            else:
                quality_summary = {"mean_r2": None, "n_total": 0}
            evaluator = ImputationEvaluator(test_model, verbose=0)
            metrics = evaluator.evaluate(genotype_data)
            out.append(
                {"params": params, "metrics": metrics, "quality_summary": quality_summary}
            )
        except Exception as e:  # noqa: BLE001 — mirror the serial path: record, don't crash
            out.append({"params": params, "metrics": None, "error": str(e)})
    return out


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

        imputed_prs, true_prs = self.compute_score_arrays(evaluation_genotypes)
        return compute_prs_metrics(imputed_prs, true_prs)

    def compute_score_arrays(
        self,
        evaluation_genotypes: Union[str, Path, GenotypeData],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(s_estimated, s_true)`` PRS arrays for all samples.

        ``s_estimated`` is the library-scored (observed + imputed) PRS routed
        through the same allele-oriented semantics as the browser/upload path
        (P1.6); ``s_true`` is the gold-standard PRS summed over all placed variants
        from the full reference dosages. Both are effect-allele-oriented, raw
        (uncalibrated), and sample-aligned — the array contract that
        :func:`compute_prs_metrics`/:func:`compute_percentile_concordance` and the
        masking-validation harness consume. ``evaluate`` is a thin wrapper over
        this method.

        Args:
            evaluation_genotypes: Path to genotype file (VCF/PLINK) or a pre-loaded
                GenotypeData object.

        Returns:
            Tuple ``(s_estimated, s_true)`` of shape ``(n_samples,)`` each.
        """
        if isinstance(evaluation_genotypes, GenotypeData):
            genotype_data = evaluation_genotypes
        else:
            genotype_data = load_genotypes(
                path=evaluation_genotypes,
                variant_ids=self._get_all_needed_variant_ids(),
            )

        if self.verbose >= 2:
            print(
                f"Loaded {genotype_data.n_samples} samples, "
                f"{genotype_data.n_variants} variants"
            )

        s_true = self._compute_true_prs(genotype_data)
        s_estimated = self._compute_imputed_prs_batch(genotype_data)
        return s_estimated, s_true

    def cross_validate(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        backend: Optional[str] = None,
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

        # Phase 9: resolve the platform set ONCE here and thread it through the
        # reference-CV pass and every dense fold refit, so the platform file is read
        # once per cross_validate — not once per fold.
        platform_variant_set, platform_info, _ = resolve_platform_variant_set(
            platform_name, platform_manifest, platform_variants
        )

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

        # Phase 6 fast-path: assemble every training fold's models from ONE streaming
        # pass by additive subtraction (S_full − S_fold(k)) instead of k independent
        # refits. Returns None for the dense oracle, so small/test inputs stay on the
        # refit-per-fold path below (keeping the golden gate exact). The fold partition
        # is identical either way, so metrics/reproducibility are preserved.
        cv_driver = LinearImputationPRS(
            window_size=self.model.window_size,
            tuning_scope=self.model.tuning_scope,
            l1_ratio=self.model.l1_ratio,
            alpha=self.model.alpha,
            cv_folds=self.model.cv_folds,
            n_jobs=self.model.n_jobs,
            n_workers=self.model.n_workers,
            random_state=random_state,
            max_predictors=self.model.max_predictors,
            backend=backend if backend is not None else self.model.backend,
            verbose=0,
        )
        cv_models = cv_driver._reference_cv_fold_models(
            genotype_data,
            prs_definition,
            platform_name=platform_name,
            platform_manifest=platform_manifest,
            platform_variants=platform_variants,
            fold_indices=fold_indices,
            _platform_variant_set=platform_variant_set,
        )

        # Run cross-validation
        fold_metrics: List[EvaluationMetrics] = []
        n_samples_per_fold: List[int] = []
        all_imputed_prs = []
        all_true_prs = []

        genome_build = getattr(self.model, "_genome_build", None)
        for fold_idx in range(n_folds):
            if self.verbose >= 1:
                print(f"Processing fold {fold_idx + 1}/{n_folds}...")

            test_indices = fold_indices[fold_idx]
            test_data = self._subset_genotype_data(genotype_data, test_indices)

            if cv_models is not None:
                # Fast-path: a hermetic model from the additive per-fold coefficients
                # (no refit). Observed terms are fold-independent; only the imputed
                # models change per fold.
                fold_model = LinearImputationPRS._from_components(
                    cv_models.observed_variants,
                    cv_models.fold_imputed_models[fold_idx],
                    None,
                    None,
                    {"genome_build": genome_build},
                )
            else:
                # Refit oracle: train a fresh model on the training fold. Phase 4: fit
                # the in-RAM fold directly (no temp-VCF round-trip). backend="streaming"
                # streams the fold via InMemoryGenotypeSource; "auto"/"dense" use the
                # dense matrix in place. The fold backend defaults to the parent model's.
                train_indices = np.concatenate(
                    [fold_indices[i] for i in range(n_folds) if i != fold_idx]
                )
                train_data = self._subset_genotype_data(genotype_data, train_indices)
                fold_model = LinearImputationPRS(
                    window_size=self.model.window_size,
                    tuning_scope=self.model.tuning_scope,
                    l1_ratio=self.model.l1_ratio,
                    alpha=self.model.alpha,
                    cv_folds=self.model.cv_folds,
                    n_jobs=self.model.n_jobs,
                    n_workers=self.model.n_workers,
                    random_state=random_state,
                    max_predictors=self.model.max_predictors,
                    backend=backend if backend is not None else self.model.backend,
                    verbose=0,  # Suppress output during CV
                )
                fold_model.fit(
                    reference_genotypes=train_data,
                    prs_definition=prs_definition,
                    platform_name=platform_name,
                    platform_manifest=platform_manifest,
                    platform_variants=platform_variants,
                    _platform_variant_set=platform_variant_set,
                    _platform_info=platform_info,
                )

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
        cache_dir: Optional[Union[str, Path]] = None,
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
            cache_dir: Optional directory for the opt-in persisted sufficient-statistics
                cache (Phase 9). When set and the fit streams, the collected Gram blocks
                are keyed on (reference, chip+target set, window params) — NOT on
                (alpha, l1) — so the first run writes them through and a later
                sensitivity / re-tune on the same panel skips the accumulation pass
                entirely, re-solving cached blocks per combo. Default None → no disk I/O.
                Serves raw-model reuse (coefficients + CV-R²); grid models are scored
                uncalibrated, so no calibration re-stream is needed.

        Returns:
            SensitivityResult with results for each parameter combination.

        Raises:
            ValidationError: If inputs are invalid.
        """
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

        # Load reference genotypes once, then fit each combo in-memory — Phase 4:
        # no temp-VCF round-trip (fit accepts a GenotypeData directly). Phase 7:
        # combos are independent, so fan them out across a process pool when n_workers>1
        # (each combo's inner fit stays single-process to avoid oversubscription).
        genotype_data = load_genotypes(path=reference_genotypes)

        # Phase 9: resolve the platform set ONCE and thread it into every combo's fit
        # (via the payload) so the platform file is read once, not once per combo. The
        # chip set is independent of the grid's window_size/l1_ratio/alpha.
        platform_variant_set, platform_info, _ = resolve_platform_variant_set(
            platform_name, platform_manifest, platform_variants
        )

        from imputed_prs.compute.parallel import parallel_map, resolve_n_workers

        resolved = resolve_n_workers(self.model.n_workers)
        inner_n_jobs = self.model.n_jobs if resolved <= 1 else 1
        base_config = {
            "window_size": self.model.window_size,
            "l1_ratio": self.model.l1_ratio,
            "alpha": self.model.alpha,
            "backend": self.model.backend,
        }
        combo_dicts = [dict(zip(param_names, combo)) for combo in param_combinations]
        common = (
            base_config, prs_definition, platform_name, platform_manifest,
            platform_variants, cv_folds, random_state, inner_n_jobs, genotype_data,
            platform_variant_set, platform_info, cache_dir,
        )

        # Phase 9: when the fit will stream, combos sharing a window_size share the
        # streaming accumulation — group them so it runs ONCE per window_size and the
        # collected Gram blocks are re-solved per (l1_ratio, alpha). Dense/small inputs
        # keep the flat per-combo fan-out (each dense fit is cheap and independently
        # parallel; grouping would only cost parallelism there). The heuristic mirrors
        # fit's auto size-gate (n_variants over-estimates |needed| → conservative).
        from imputed_prs.core.linear_imputation_prs import _AUTO_STREAMING_BYTES_THRESHOLD

        est_bytes = genotype_data.n_samples * max(genotype_data.n_variants, 1) * 4
        will_stream = self.model.backend == "streaming" or (
            self.model.backend == "auto" and est_bytes > _AUTO_STREAMING_BYTES_THRESHOLD
        )

        if will_stream:
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for cd in combo_dicts:
                groups.setdefault(
                    cd.get("window_size", base_config["window_size"]), []
                ).append(cd)
            group_payloads = [(group, *common) for group in groups.values()]
            if self.verbose >= 1:
                print(
                    f"Testing {len(param_combinations)} combinations in "
                    f"{len(group_payloads)} accumulation group(s) (streaming reuse)..."
                )
            nested = parallel_map(
                _run_sensitivity_group, group_payloads, n_workers=self.model.n_workers
            )
            results = [r for group_res in nested for r in group_res]
        else:
            payloads = [(cd, *common) for cd in combo_dicts]
            if self.verbose >= 1:
                how = "serial" if resolved <= 1 else f"{resolved} workers"
                print(f"Testing {len(param_combinations)} parameter combinations ({how})...")
            results = parallel_map(
                _run_sensitivity_combo, payloads, n_workers=self.model.n_workers
            )

        # Restore canonical combo order (grouping reorders results) so the best-scan's
        # first-max-wins tie-break stays deterministic regardless of grouping/fan-out.
        _pos = {combo: i for i, combo in enumerate(param_combinations)}
        results.sort(key=lambda r: _pos[tuple(r["params"][k] for k in param_names)])

        # Collect in canonical combo order → deterministic best (first max wins ties,
        # matching the pre-Phase-7 serial loop's strict '>' scan over combos in order).
        parameter_results: List[Dict[str, Any]] = []
        quality_summaries: List[Dict[str, Any]] = []
        best_r2 = -np.inf
        best_params: Dict[str, float] = {}
        best_metrics: Optional[EvaluationMetrics] = None
        for res in results:
            parameter_results.append(res)
            if res.get("metrics") is not None:
                quality_summaries.append(res["quality_summary"])
                if res["metrics"].r2 > best_r2:
                    best_r2 = res["metrics"].r2
                    best_params = res["params"]
                    best_metrics = res["metrics"]
            elif self.verbose >= 1:
                print(f"    Warning: Failed for {res['params']}: {res.get('error')}")

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
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)

        # All placed variants (observed + imputed); dropped variants are absent
        # from both lists and therefore correctly excluded.
        placed = [
            (v.chromosome, v.position, v.effect_allele, v.other_allele, v.beta)
            for v in self.model.observed_variants
        ] + [
            (m.chromosome, m.position, m.effect_allele, m.other_allele, m.beta)
            for m in self.model.imputed_models
        ]

        # Large panels: vectorized gather (float32 product, float64 accumulate),
        # matching the per-variant oracle up to block-sum re-association (~1e-14).
        if should_use_batch(len(placed)):
            return accumulate_true_prs(genotype_data.dosage_matrix, resolver, placed)

        # Small inputs (golden fixtures): byte-exact per-variant oracle loop.
        n_samples = genotype_data.n_samples
        true_prs = np.zeros(n_samples)
        for chromosome, position, effect_allele, other_allele, beta in placed:
            match = resolver.resolve(
                chromosome, position, effect_allele, other_allele,
                genotype_data.dosage_matrix,
            )
            if match is None:
                continue
            dosages = match[1]
            valid_mask = ~np.isnan(dosages)
            true_prs[valid_mask] += dosages[valid_mask] * beta

        return true_prs

    def _compute_imputed_prs_batch(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute predicted (observed + imputed) PRS for all samples.

        Both dosage modes score through the role-aware **numeric** path
        (``_predicted_prs_numeric``): it orients observed and predictor dosages
        from the stored P1.3 allele metadata and runs the imputation regression,
        size-selecting the vectorized CSR batch scorer at/above
        ``_BATCH_MIN_TARGETS`` targets and the byte-exact per-model oracle below.

        This is what allows hard-called reference CV / masking validation to score
        at matvec speed instead of the per-sample string replay (P5). On hard-called
        integer dosages the numeric path is byte-identical to the string replay
        (golden ``TestNumericVsStringGolden`` at ``atol=1e-12``; CSR batch matches
        the oracle at ``atol~1e-9``), so metrics are unchanged within statistical
        parity. ``_predicted_prs_via_strings`` is retained as the browser-faithful
        oracle for the golden tests; it is no longer on the metric path.

        Args:
            genotype_data: GenotypeData containing all samples.

        Returns:
            Array of predicted PRS values (n_samples,).
        """
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

    def _predicted_prs_numeric(
        self, genotype_data: GenotypeData, *, _force_batch: Optional[bool] = None
    ) -> np.ndarray:
        """Continuous path: orient observed and predictor dosages numerically and
        run the imputation regression (clipped to ``[0, 2]`` to match
        ``clip_and_adjust_variance`` in the per-user scorer).

        Size-selected: at/above ``_BATCH_MIN_TARGETS`` imputed variants the
        vectorized CSR batch scorer runs (validated at ``atol~1e-9``); below it the
        byte-exact per-model oracle loop runs (golden ``atol=1e-12``).
        ``_force_batch`` overrides the size gate for tests.
        """
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)
        n_samples = genotype_data.n_samples
        models = self.model.imputed_models

        predicted = observed_component_numeric(
            genotype_data, resolver, self.model.observed_variants
        )

        use_batch = (
            should_use_batch(len(models)) if _force_batch is None else _force_batch
        )
        if use_batch:
            axis = build_chip_axis(models, resolver)
            z_chip = oriented_chip_matrix(genotype_data.dosage_matrix, axis)
            W, intercepts, betas = build_coef_csr(models, axis.chip_index)
            predicted += panel_impute_prs(z_chip, W, intercepts, betas)
            return predicted

        for model in models:
            if model.is_intercept_only or not model.predictor_variant_ids:
                raw = np.full(n_samples, float(model.intercept))
            else:
                z = oriented_predictor_matrix(
                    genotype_data,
                    resolver,
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
