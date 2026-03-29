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
        # Build variant-to-index mapping (by variant_id AND chr:pos)
        var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            var_to_idx[row["variant_id"]] = idx
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            var_to_idx[f"{chrom}:{int(row['position'])}"] = idx

        n_samples = genotype_data.n_samples
        true_prs = np.zeros(n_samples)

        def _find_idx(variant_id, chromosome, position):
            """Find genotype index by variant_id or chr:pos."""
            if variant_id in var_to_idx:
                return var_to_idx[variant_id]
            chrpos = f"{chromosome}:{position}"
            if chrpos in var_to_idx:
                return var_to_idx[chrpos]
            return None

        # Add observed variant contributions
        for var in self.model.observed_variants:
            idx = _find_idx(var.variant_id, var.chromosome, var.position)
            if idx is not None:
                dosages = genotype_data.dosage_matrix[:, idx]
                valid_mask = ~np.isnan(dosages)
                true_prs[valid_mask] += dosages[valid_mask] * var.beta

        # Add missing variant contributions (using true genotypes, not projected)
        # Region models contain multiple PRS variants each
        for region_model in self.model.region_models:
            for i, var_id in enumerate(region_model.prs_variant_ids):
                beta = float(region_model.betas[i])
                # Look up variant in genotype data
                # We need chromosome and position - extract from region
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
        # Build variant-to-index mapping (by variant_id AND chr:pos)
        var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            var_to_idx[row["variant_id"]] = idx
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            var_to_idx[f"{chrom}:{int(row['position'])}"] = idx

        n_samples = genotype_data.n_samples
        projected_prs = np.zeros(n_samples)

        # Build mapping from variant_id to genotype index for all needed variants
        variant_id_to_geno_idx: Dict[str, int] = {}

        for var in self.model.observed_variants:
            for key in [var.variant_id, f"{var.chromosome}:{var.position}"]:
                if key in var_to_idx:
                    variant_id_to_geno_idx[var.variant_id] = var_to_idx[key]
                    break

        for region_model in self.model.region_models:
            for pred_id in region_model.predictor_variant_ids:
                if pred_id in var_to_idx:
                    variant_id_to_geno_idx[pred_id] = var_to_idx[pred_id]

        # Collect all variant IDs the predictor will request
        platform_variant_ids = set()
        for var in self.model.observed_variants:
            platform_variant_ids.add(var.variant_id)
        for region_model in self.model.region_models:
            platform_variant_ids.update(region_model.predictor_variant_ids)

        # Create predictor (no calibration for evaluation)
        predictor = ProjectionPredictor(
            observed_variants=self.model.observed_variants,
            region_models=self.model.region_models,
            calibration_params=None,
        )

        # Compute for each sample
        for sample_idx in range(n_samples):
            user_dosages: Dict[str, Optional[float]] = {}
            for var_id in platform_variant_ids:
                geno_idx = variant_id_to_geno_idx.get(var_id)
                if geno_idx is not None:
                    dosage = genotype_data.dosage_matrix[sample_idx, geno_idx]
                    if np.isnan(dosage):
                        user_dosages[var_id] = None
                    else:
                        user_dosages[var_id] = float(dosage)
                else:
                    user_dosages[var_id] = None

            result = predictor.predict(user_dosages, apply_calibration=False)
            projected_prs[sample_idx] = result.prs

        return projected_prs
