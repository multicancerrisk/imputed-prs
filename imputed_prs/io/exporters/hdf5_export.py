"""HDF5 export for trained imputation models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)


def _write_metadata(
    f: h5py.File,
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    calibration_params: Optional[CalibrationParams],
    evaluation_metrics: Optional[EvaluationMetrics],
    platform_name: Optional[str],
    prs_id: Optional[str],
    genome_build: Optional[str],
    model_name: Optional[str],
    include_variance_scaling: bool,
    training_summary: Optional[Dict[str, Any]],
    reference_panel_id: Optional[str],
    training_ancestry: Optional[str],
    ambiguous_policy: Optional[str],
) -> None:
    """Write metadata to HDF5 file."""
    meta = f.create_group("metadata")

    # Scalar attributes
    meta.attrs["format_version"] = "2.0"
    meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    meta.attrs["n_observed_variants"] = len(observed_variants)
    meta.attrs["n_imputed_variants"] = len(imputed_models)
    meta.attrs["n_intercept_only"] = sum(1 for m in imputed_models if m.is_intercept_only)
    meta.attrs["include_variance_scaling"] = include_variance_scaling

    # Optional string attributes (store empty string if None)
    meta.attrs["model_name"] = model_name or ""
    meta.attrs["prs_id"] = prs_id or ""
    meta.attrs["platform_name"] = platform_name or ""
    meta.attrs["genome_build"] = genome_build or ""

    # Provenance scalars (v2). `centering_scaling` is the calibration object,
    # already stored as `calibration_params_json` below.
    meta.attrs["reference_panel_id"] = reference_panel_id or ""
    meta.attrs["training_ancestry"] = training_ancestry or ""
    meta.attrs["ambiguous_policy"] = ambiguous_policy or ""

    # Complex objects stored as JSON datasets
    str_dtype = h5py.string_dtype(encoding='utf-8')

    if calibration_params is not None:
        meta.create_dataset(
            "calibration_params_json",
            data=json.dumps(asdict(calibration_params)),
            dtype=str_dtype,
        )

    if evaluation_metrics is not None:
        meta.create_dataset(
            "evaluation_metrics_json",
            data=json.dumps(asdict(evaluation_metrics)),
            dtype=str_dtype,
        )

    if training_summary is not None:
        meta.create_dataset(
            "training_summary_json",
            data=json.dumps(training_summary),
            dtype=str_dtype,
        )


def _write_observed_variants(
    f: h5py.File,
    observed_variants: List[VariantInfo],
) -> None:
    """Write observed variants to HDF5 file."""
    grp = f.create_group("observed_variants")
    str_dtype = h5py.string_dtype(encoding='utf-8')

    n = len(observed_variants)
    if n == 0:
        # Create empty datasets with correct dtypes
        grp.create_dataset("variant_id", shape=(0,), dtype=str_dtype)
        grp.create_dataset("chromosome", shape=(0,), dtype=str_dtype)
        grp.create_dataset("position", shape=(0,), dtype=np.int64)
        grp.create_dataset("effect_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("other_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("beta", shape=(0,), dtype=np.float64)
        grp.create_dataset("platform_index", shape=(0,), dtype=np.int32)
        return

    grp.create_dataset("variant_id", data=[v.variant_id for v in observed_variants], dtype=str_dtype)
    grp.create_dataset("chromosome", data=[v.chromosome for v in observed_variants], dtype=str_dtype)
    grp.create_dataset("position", data=np.array([v.position for v in observed_variants], dtype=np.int64))
    grp.create_dataset("effect_allele", data=[v.effect_allele for v in observed_variants], dtype=str_dtype)
    grp.create_dataset("other_allele", data=[v.other_allele or "" for v in observed_variants], dtype=str_dtype)
    grp.create_dataset("beta", data=np.array([v.beta for v in observed_variants], dtype=np.float64))
    grp.create_dataset("platform_index", data=np.arange(n, dtype=np.int32))


def _write_imputed_variants(
    f: h5py.File,
    imputed_models: List[ImputedVariantModel],
    include_variance_scaling: bool,
    group_name: str = "imputed_variants",
) -> None:
    """Write a variant-model group to HDF5.

    Used for both the ``imputed_variants`` group and the ``observed_fallbacks``
    group (P1.8) — identical layout, different group name.
    """
    grp = f.create_group(group_name)
    str_dtype = h5py.string_dtype(encoding='utf-8')

    n = len(imputed_models)
    if n == 0:
        # Create empty datasets with correct dtypes
        grp.create_dataset("variant_id", shape=(0,), dtype=str_dtype)
        grp.create_dataset("chromosome", shape=(0,), dtype=str_dtype)
        grp.create_dataset("position", shape=(0,), dtype=np.int64)
        grp.create_dataset("effect_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("other_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("beta", shape=(0,), dtype=np.float64)
        grp.create_dataset("allele_frequency", shape=(0,), dtype=np.float64)
        grp.create_dataset("imputation_r2", shape=(0,), dtype=np.float64)
        grp.create_dataset("intercept", shape=(0,), dtype=np.float64)
        grp.create_dataset("is_intercept_only", shape=(0,), dtype=bool)
        grp.create_dataset("n_predictors", shape=(0,), dtype=np.int32)
        if include_variance_scaling:
            grp.create_dataset("residual_variance", shape=(0,), dtype=np.float64)
        return

    grp.create_dataset("variant_id", data=[m.variant_id for m in imputed_models], dtype=str_dtype)
    grp.create_dataset("chromosome", data=[m.chromosome for m in imputed_models], dtype=str_dtype)
    grp.create_dataset("position", data=np.array([m.position for m in imputed_models], dtype=np.int64))
    grp.create_dataset("effect_allele", data=[m.effect_allele for m in imputed_models], dtype=str_dtype)
    grp.create_dataset("other_allele", data=[m.other_allele or "" for m in imputed_models], dtype=str_dtype)
    grp.create_dataset("beta", data=np.array([m.beta for m in imputed_models], dtype=np.float64))
    grp.create_dataset("allele_frequency", data=np.array([m.allele_frequency for m in imputed_models], dtype=np.float64))
    grp.create_dataset("imputation_r2", data=np.array([m.imputation_r2 for m in imputed_models], dtype=np.float64))
    grp.create_dataset("intercept", data=np.array([m.intercept for m in imputed_models], dtype=np.float64))
    grp.create_dataset("is_intercept_only", data=np.array([m.is_intercept_only for m in imputed_models], dtype=bool))
    grp.create_dataset("n_predictors", data=np.array([len(m.predictor_variant_ids) for m in imputed_models], dtype=np.int32))

    if include_variance_scaling:
        grp.create_dataset("residual_variance", data=np.array([m.residual_variance for m in imputed_models], dtype=np.float64))


def _write_coefficients(
    f: h5py.File,
    imputed_models: List[ImputedVariantModel],
) -> None:
    """Write sparse coefficients to HDF5 file.

    One row per (target, predictor) pair. Each row also carries the predictor's
    allele metadata (counted/other allele, locus, allele frequency) so a reloaded
    model can orient raw genotypes (schema v2). Missing metadata is written as
    ``""`` / ``NaN`` and reconstructed as absent on load.
    """
    grp = f.create_group("coefficients")
    str_dtype = h5py.string_dtype(encoding='utf-8')

    # Build sparse representation
    target_variant_ids = []
    predictor_variant_ids = []
    coefficients = []
    predictor_chromosomes = []
    predictor_positions = []
    predictor_counted_alleles = []
    predictor_other_alleles = []
    predictor_allele_frequencies = []

    for model in imputed_models:
        for i, (pred_id, coef) in enumerate(
            zip(model.predictor_variant_ids, model.coefficients.tolist())
        ):
            target_variant_ids.append(model.variant_id)
            predictor_variant_ids.append(pred_id)
            coefficients.append(coef)
            predictor_chromosomes.append(
                model.predictor_chromosomes[i]
                if i < len(model.predictor_chromosomes)
                else ""
            )
            predictor_positions.append(
                int(model.predictor_positions[i])
                if i < len(model.predictor_positions)
                else 0
            )
            predictor_counted_alleles.append(
                model.predictor_counted_alleles[i]
                if i < len(model.predictor_counted_alleles)
                else ""
            )
            predictor_other_alleles.append(
                model.predictor_other_alleles[i]
                if i < len(model.predictor_other_alleles)
                else ""
            )
            predictor_allele_frequencies.append(
                float(model.predictor_allele_frequencies[i])
                if i < len(model.predictor_allele_frequencies)
                else np.nan
            )

    n = len(target_variant_ids)
    if n == 0:
        grp.create_dataset("target_variant_id", shape=(0,), dtype=str_dtype)
        grp.create_dataset("predictor_variant_id", shape=(0,), dtype=str_dtype)
        grp.create_dataset("coefficient", shape=(0,), dtype=np.float64)
        grp.create_dataset("predictor_chromosome", shape=(0,), dtype=str_dtype)
        grp.create_dataset("predictor_position", shape=(0,), dtype=np.int64)
        grp.create_dataset("predictor_counted_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("predictor_other_allele", shape=(0,), dtype=str_dtype)
        grp.create_dataset("predictor_allele_frequency", shape=(0,), dtype=np.float64)
        return

    grp.create_dataset("target_variant_id", data=target_variant_ids, dtype=str_dtype)
    grp.create_dataset("predictor_variant_id", data=predictor_variant_ids, dtype=str_dtype)
    grp.create_dataset("coefficient", data=np.array(coefficients, dtype=np.float64))
    grp.create_dataset("predictor_chromosome", data=predictor_chromosomes, dtype=str_dtype)
    grp.create_dataset("predictor_position", data=np.array(predictor_positions, dtype=np.int64))
    grp.create_dataset("predictor_counted_allele", data=predictor_counted_alleles, dtype=str_dtype)
    grp.create_dataset("predictor_other_allele", data=predictor_other_alleles, dtype=str_dtype)
    grp.create_dataset(
        "predictor_allele_frequency",
        data=np.array(predictor_allele_frequencies, dtype=np.float64),
    )


def export_to_hdf5(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    calibration_params: Optional[CalibrationParams] = None,
    evaluation_metrics: Optional[EvaluationMetrics] = None,
    platform_name: Optional[str] = None,
    prs_id: Optional[str] = None,
    genome_build: Optional[str] = None,
    model_name: Optional[str] = None,
    include_variance_scaling: bool = True,
    training_summary: Optional[Dict[str, Any]] = None,
    reference_panel_id: Optional[str] = None,
    training_ancestry: Optional[str] = None,
    ambiguous_policy: Optional[str] = None,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> Path:
    """Export trained imputation model to HDF5 format (schema v2.0).

    Creates a single HDF5 file with hierarchical structure for efficient storage
    and fast Python reloading. The sparse ``coefficients`` group carries per-predictor
    allele metadata so a reloaded model can orient raw genotypes.

    Args:
        output_path: Path for output HDF5 file.
        observed_variants: List of VariantInfo for directly observed variants.
        imputed_models: List of ImputedVariantModel for imputed variants.
        calibration_params: Optional calibration scaling parameters.
        evaluation_metrics: Optional evaluation metrics from training.
        platform_name: Name of the genotyping platform.
        prs_id: PRS identifier (e.g., "PGS000004").
        genome_build: Genome build (e.g., "GRCh37").
        model_name: Human-readable model name.
        include_variance_scaling: Whether to include variance/SE components.
        training_summary: Optional summary statistics from training.
        reference_panel_id: Provenance — reference panel used for training.
        training_ancestry: Provenance — ancestry of the training cohort.
        ambiguous_policy: Provenance — palindromic-SNP handling policy.
        compression: Compression filter ("gzip", "lzf", or None).
        compression_opts: Compression level for gzip (0-9).

    Returns:
        Path to the created HDF5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        _write_metadata(
            f, observed_variants, imputed_models, calibration_params,
            evaluation_metrics, platform_name, prs_id, genome_build,
            model_name, include_variance_scaling, training_summary,
            reference_panel_id, training_ancestry, ambiguous_policy,
        )
        _write_observed_variants(f, observed_variants)
        _write_imputed_variants(f, imputed_models, include_variance_scaling)
        # Per-observed-variant fallback models (P1.8): written as a parallel group
        # with their predictors folded into the shared sparse coefficients table.
        fallback_models = [
            v.fallback for v in observed_variants if v.fallback is not None
        ]
        _write_imputed_variants(
            f, fallback_models, include_variance_scaling,
            group_name="observed_fallbacks",
        )
        _write_coefficients(f, imputed_models + fallback_models)

    return output_path
