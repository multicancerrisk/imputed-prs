"""ProjectionEvaluator class for evaluating fitted LinearProjectionPRS models."""

from pathlib import Path
from typing import Dict, Optional, Set, Union

import numpy as np

from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import (
    EvaluationMetrics,
    GenotypeData,
)
from imputed_prs.evaluation.metrics import compute_prs_metrics
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.models.projection_predictor import ProjectionPredictor


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

        # Step 3: Compute projected PRS
        projected_prs = self._compute_projected_prs_batch(genotype_data)

        if self.verbose >= 2:
            print(f"Computed PRS for {len(true_prs)} samples")

        # Step 4: Compute metrics
        metrics = compute_prs_metrics(projected_prs, true_prs)

        return metrics

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
        from imputed_prs.core.harmonizer import (
            build_reference_allele_index,
            match_oriented_dosage,
        )

        reference_index = build_reference_allele_index(genotype_data.variant_info)
        n_samples = genotype_data.n_samples
        true_prs = np.zeros(n_samples)

        # Observed variant contributions use effect-allele-oriented dosages.
        for var in self.model.observed_variants:
            match = match_oriented_dosage(
                var.chromosome, var.position, var.effect_allele, var.other_allele,
                genotype_data.variant_info, genotype_data.dosage_matrix,
                reference_index,
            )
            if match is None:
                continue
            dosages = match[1]
            valid_mask = ~np.isnan(dosages)
            true_prs[valid_mask] += dosages[valid_mask] * var.beta

        # Region (missing) variant contributions. Region models do not store
        # per-variant alleles, so these use the first reference row at each locus
        # (correct for the common effect==ALT biallelic case). The analysis
        # pipeline avoids this path by scoring the true PRS via the imputation
        # evaluator, which has full allele information.
        var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            var_to_idx.setdefault(row["variant_id"], idx)
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            var_to_idx.setdefault(f"{chrom}:{int(row['position'])}", idx)

        for region_model in self.model.region_models:
            for i, var_id in enumerate(region_model.prs_variant_ids):
                beta = float(region_model.betas[i])
                idx = var_to_idx.get(var_id)
                if idx is not None:
                    dosages = genotype_data.dosage_matrix[:, idx]
                    valid_mask = ~np.isnan(dosages)
                    true_prs[valid_mask] += dosages[valid_mask] * beta

        return true_prs

    def _compute_projected_prs_batch(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute projected PRS for all samples using ProjectionPredictor.

        For each sample, extracts platform dosages and computes the
        projected PRS using the model's predictor.

        Args:
            genotype_data: GenotypeData containing all samples.

        Returns:
            Array of projected PRS values (n_samples,).
        """
        from imputed_prs.core.harmonizer import (
            build_reference_allele_index,
            match_oriented_dosage,
        )

        n_samples = genotype_data.n_samples

        # Observed component: effect-allele-oriented dosages (consistent with the
        # oriented true PRS used in evaluation and with the oriented projection
        # targets the region models were trained on).
        reference_index = build_reference_allele_index(genotype_data.variant_info)
        observed_prs = np.zeros(n_samples)
        for var in self.model.observed_variants:
            match = match_oriented_dosage(
                var.chromosome, var.position, var.effect_allele, var.other_allele,
                genotype_data.variant_info, genotype_data.dosage_matrix,
                reference_index,
            )
            if match is None:
                continue
            dosages = match[1]
            valid_mask = ~np.isnan(dosages)
            observed_prs[valid_mask] += dosages[valid_mask] * var.beta

        # Projected component: predict each region's S_R from raw platform
        # dosages (predictors kept raw, matching how the models were trained).
        var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            var_to_idx.setdefault(row["variant_id"], idx)
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            var_to_idx.setdefault(f"{chrom}:{int(row['position'])}", idx)

        predictor_ids: Set[str] = set()
        for region_model in self.model.region_models:
            predictor_ids.update(region_model.predictor_variant_ids)

        # Observed variants are scored above (oriented); the predictor only
        # contributes the projected component here.
        predictor = ProjectionPredictor(
            observed_variants=[],
            region_models=self.model.region_models,
            calibration_params=None,
        )

        projected_component = np.zeros(n_samples)
        for sample_idx in range(n_samples):
            user_dosages: Dict[str, Optional[float]] = {}
            for var_id in predictor_ids:
                geno_idx = var_to_idx.get(var_id)
                if geno_idx is not None:
                    dosage = genotype_data.dosage_matrix[sample_idx, geno_idx]
                    user_dosages[var_id] = None if np.isnan(dosage) else float(dosage)
                else:
                    user_dosages[var_id] = None

            result = predictor.predict(user_dosages, apply_calibration=False)
            projected_component[sample_idx] = result.prs

        return observed_prs + projected_component
