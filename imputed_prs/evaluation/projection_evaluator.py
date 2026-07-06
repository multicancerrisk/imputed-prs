"""ProjectionEvaluator class for evaluating fitted LinearProjectionPRS models."""

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import (
    EvaluationMetrics,
    GenotypeData,
)
from imputed_prs.core.harmonizer import ReferenceAlleleResolver
from imputed_prs.evaluation._scoring import (
    NeededVariant,
    iter_sample_collections,
    observed_component_numeric,
    oriented_predictor_matrix,
    should_use_batch,
)
from imputed_prs.evaluation.metrics import compute_prs_metrics
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.models.projection_predictor import ProjectionPredictor
from imputed_prs.models.vectorized_predictor import (
    accumulate_true_prs,
    build_chip_axis,
    build_projection_weff,
    oriented_chip_matrix,
    panel_project_prs,
)


class ProjectionEvaluator:
    """Evaluator for fitted LinearProjectionPRS models.

    Mirrors ImputationEvaluator but uses ProjectionPredictor for
    computing projected PRS values.

    Example:
        >>> model = LinearProjectionPRS().fit(...)
        >>> evaluator = ProjectionEvaluator(model)
        >>> metrics = evaluator.evaluate("held_out_genotypes.vcf.gz")
        >>> print(f"Correlation: {metrics.correlation:.3f}")
    """

    def __init__(self, model: "LinearProjectionPRS", verbose: int = 1):
        """Initialize the evaluator.

        Args:
            model: A fitted LinearProjectionPRS model.
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        # Import here to avoid circular import
        from imputed_prs.core.linear_projection_prs import LinearProjectionPRS

        if not model.is_fitted:
            raise ModelNotFittedError("ProjectionEvaluator requires a fitted model.")

        self.model = model
        self.verbose = verbose

    def evaluate(
        self,
        evaluation_genotypes: Union[str, Path, GenotypeData],
    ) -> EvaluationMetrics:
        """Evaluate the model on held-out genotype data.

        Computes projected PRS and true PRS for all samples in the evaluation
        dataset, then calculates metrics comparing them.

        Args:
            evaluation_genotypes: Path to genotype file (VCF/PLINK) or
                pre-loaded GenotypeData object.

        Returns:
            EvaluationMetrics comparing projected vs true PRS.
        """
        projected_prs, true_prs = self.compute_score_arrays(evaluation_genotypes)
        return compute_prs_metrics(projected_prs, true_prs)

    def compute_score_arrays(
        self,
        evaluation_genotypes: Union[str, Path, GenotypeData],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(s_estimated, s_true)`` PRS arrays for all samples.

        ``s_estimated`` is the library-scored (observed + projected) PRS routed
        through the same allele-oriented semantics as the browser/upload path
        (P1.6); ``s_true`` is the gold-standard PRS summed over all placed variants
        from the full reference dosages. Both are effect-allele-oriented, raw
        (uncalibrated), and sample-aligned — the array contract that
        :func:`compute_prs_metrics`/:func:`compute_percentile_concordance` and the
        masking-validation harness consume. ``evaluate`` is a thin wrapper over
        this method. The signature mirrors
        :meth:`ImputationEvaluator.compute_score_arrays`, so the harness has a
        single duck-typed call site after dispatch.

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
        s_estimated = self._compute_projected_prs_batch(genotype_data)
        return s_estimated, s_true

    def _get_all_needed_variant_ids(self) -> Set[str]:
        """Get union of PRS variant IDs and predictor variant IDs.

        Returns:
            Set of all variant IDs needed for evaluation.
        """
        needed = set()

        # Add observed variant IDs
        for var in self.model.observed_variants:
            needed.add(var.variant_id)

        # Add PRS variant IDs and predictor IDs from region models
        for region_model in self.model.region_models:
            needed.update(region_model.prs_variant_ids)
            needed.update(region_model.predictor_variant_ids)

        return needed

    def _compute_true_prs(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute true PRS from full genotype data.

        Calculates sum(dosage * beta) for all PRS variants (observed + missing),
        using true genotypes.

        Args:
            genotype_data: GenotypeData containing all samples.

        Returns:
            Array of true PRS values (n_samples,).
        """
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)

        # All placed variants: observed terms + every region's PRS variants, each
        # effect-allele-oriented per the stored locus + alleles so effect==REF,
        # strand-flipped, and multiallelic loci score correctly (parity with the
        # imputation evaluator).
        placed = [
            (v.chromosome, v.position, v.effect_allele, v.other_allele, v.beta)
            for v in self.model.observed_variants
        ]
        for region_model in self.model.region_models:
            for i, beta in enumerate(region_model.betas):
                placed.append(
                    (
                        region_model.chromosome,
                        int(region_model.prs_positions[i]),
                        region_model.prs_effect_alleles[i],
                        region_model.prs_other_alleles[i],
                        float(beta),
                    )
                )

        # Large panels: vectorized gather (float32 product, float64 accumulate).
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

    def _compute_projected_prs_batch(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute predicted (observed + projected) PRS for all samples.

        Both dosage modes score through the role-aware **numeric** path
        (``_predicted_prs_numeric``): it orients observed and predictor dosages
        from the stored P1.3 allele metadata and runs the region regression,
        size-selecting the vectorized ``Z @ w_eff + const`` mat-vec at/above
        ``_BATCH_MIN_TARGETS`` regions and the byte-exact per-region oracle below.

        This is what allows hard-called reference CV / masking validation to score
        at matvec speed instead of the per-sample string replay (P5). On hard-called
        integer dosages the numeric path is byte-identical to the string replay
        (golden ``TestNumericVsStringGolden`` at ``atol=1e-12``; batch matches the
        oracle at ``atol~1e-9``), so metrics are unchanged within statistical
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
        plus every region predictor (counted/other from P1.3 metadata). The PRS
        target variants are predicted from the predictors, not read as inputs."""
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
        for model in self.model.region_models:
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
        browser scorer (observed + projected in one oriented ``predict`` call)."""
        needed = self._needed_for_render()
        predictor = ProjectionPredictor(
            observed_variants=self.model.observed_variants,
            region_models=self.model.region_models,
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
        run the region regression (no clipping — the projection target is a PRS
        contribution, not a dosage).

        Size-selected: at/above ``_BATCH_MIN_TARGETS`` regions the projection
        collapses to a single ``Z @ w_eff + const`` mat-vec (validated at
        ``atol~1e-9``); below it the byte-exact per-region oracle loop runs (golden
        ``atol=1e-12``). ``_force_batch`` overrides the size gate for tests.
        """
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)
        models = self.model.region_models

        predicted = observed_component_numeric(
            genotype_data, resolver, self.model.observed_variants
        )

        use_batch = (
            should_use_batch(len(models)) if _force_batch is None else _force_batch
        )
        if use_batch:
            axis = build_chip_axis(models, resolver)
            z_chip = oriented_chip_matrix(genotype_data.dosage_matrix, axis)
            w_eff, const = build_projection_weff(models, axis.chip_index)
            predicted += panel_project_prs(z_chip, w_eff, const)
            return predicted

        for model in models:
            if model.is_intercept_only or not model.predictor_variant_ids:
                predicted += float(model.intercept)
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
                predicted += z @ np.asarray(model.coefficients, dtype=np.float64) + float(
                    model.intercept
                )

        return predicted

    # ------------------------------------------------------------------
    # Reference cross-validation (Phase 6).
    # ------------------------------------------------------------------
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
    ):
        """Perform k-fold reference cross-validation for the projection model.

        Splits the reference panel into ``k`` folds, trains region models on ``k-1``
        folds and scores the held-out fold; repeats for all folds. Phase 6: when the
        resolved backend streams, all ``k`` training folds are assembled from **one**
        streaming pass by additive subtraction (``S_full − S_fold(k)``) rather than ``k``
        independent refits; the dense/small path refits each fold as the size-selected
        oracle (keeping the golden gate exact). Mirrors
        :meth:`ImputationEvaluator.cross_validate` and returns the same
        ``CrossValidationResult``.
        """
        from imputed_prs.core.exceptions import ValidationError
        from imputed_prs.core.linear_projection_prs import LinearProjectionPRS
        from imputed_prs.evaluation.evaluator import CrossValidationResult
        from imputed_prs.evaluation.metrics import compute_percentile_concordance

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

        genotype_data = load_genotypes(path=reference_genotypes)

        if self.verbose >= 1:
            print(f"Loaded {genotype_data.n_samples} samples for cross-validation")

        # Fold partition — identical construction to the imputation evaluator.
        n_samples = genotype_data.n_samples
        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        fold_size = n_samples // n_folds
        fold_indices = []
        for i in range(n_folds):
            start = i * fold_size
            end = n_samples if i == n_folds - 1 else start + fold_size
            fold_indices.append(indices[start:end])

        # Phase 6 fast-path: one streaming pass → per-fold region models by subtraction.
        cv_driver = LinearProjectionPRS(
            window_size=self.model.window_size,
            tuning_scope=self.model.tuning_scope,
            l1_ratio=self.model.l1_ratio,
            alpha=self.model.alpha,
            cv_folds=self.model.cv_folds,
            n_jobs=self.model.n_jobs,
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
        )

        fold_metrics: List[EvaluationMetrics] = []
        n_samples_per_fold: List[int] = []
        all_estimated_prs = []
        all_true_prs = []
        genome_build = getattr(self.model, "_genome_build", None)

        for fold_idx in range(n_folds):
            if self.verbose >= 1:
                print(f"Processing fold {fold_idx + 1}/{n_folds}...")

            test_indices = fold_indices[fold_idx]
            test_data = self._subset_genotype_data(genotype_data, test_indices)

            if cv_models is not None:
                fold_model = LinearProjectionPRS._from_components(
                    cv_models.observed_variants,
                    cv_models.fold_region_models[fold_idx],
                    None,
                    None,
                    {"genome_build": genome_build},
                )
            else:
                train_indices = np.concatenate(
                    [fold_indices[i] for i in range(n_folds) if i != fold_idx]
                )
                train_data = self._subset_genotype_data(genotype_data, train_indices)
                fold_model = LinearProjectionPRS(
                    window_size=self.model.window_size,
                    tuning_scope=self.model.tuning_scope,
                    l1_ratio=self.model.l1_ratio,
                    alpha=self.model.alpha,
                    cv_folds=self.model.cv_folds,
                    n_jobs=self.model.n_jobs,
                    random_state=random_state,
                    max_predictors=self.model.max_predictors,
                    backend=backend if backend is not None else self.model.backend,
                    verbose=0,
                )
                fold_model.fit(
                    reference_genotypes=train_data,
                    prs_definition=prs_definition,
                    platform_name=platform_name,
                    platform_manifest=platform_manifest,
                    platform_variants=platform_variants,
                )

            fold_evaluator = ProjectionEvaluator(fold_model, verbose=0)
            fold_true_prs = fold_evaluator._compute_true_prs(test_data)
            fold_estimated_prs = fold_evaluator._compute_projected_prs_batch(test_data)

            metrics = compute_prs_metrics(fold_estimated_prs, fold_true_prs)
            fold_metrics.append(metrics)
            n_samples_per_fold.append(len(test_indices))
            all_estimated_prs.extend(fold_estimated_prs.tolist())
            all_true_prs.extend(fold_true_prs.tolist())

        correlations = [m.correlation for m in fold_metrics]
        r2_values = [m.r2 for m in fold_metrics]
        mae_values = [m.mae for m in fold_metrics]
        rmse_values = [m.rmse for m in fold_metrics]
        spearman_values = [m.spearman_rho for m in fold_metrics]
        percentile_concordance = compute_percentile_concordance(
            np.array(all_estimated_prs),
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

    def _subset_genotype_data(
        self, genotype_data: GenotypeData, sample_indices: np.ndarray
    ) -> GenotypeData:
        """Row-subset the panel for a CV fold (Phase 4 in-RAM fold, no temp-VCF)."""
        return GenotypeData(
            dosage_matrix=genotype_data.dosage_matrix[sample_indices, :],
            variant_info=genotype_data.variant_info.copy(),
            sample_ids=[genotype_data.sample_ids[i] for i in sample_indices],
            genome_build=genotype_data.genome_build,
            source_file=genotype_data.source_file,
        )
