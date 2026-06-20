"""Arrow and Parquet export for trained imputation models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)


def _build_metadata_table(
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
) -> pa.Table:
    """Build metadata as a single-row table."""
    data = {
        "format_version": ["2.0"],
        "created_at": [datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")],
        "model_name": [model_name],
        "prs_id": [prs_id],
        "platform_name": [platform_name],
        "genome_build": [genome_build],
        "reference_panel_id": [reference_panel_id],
        "training_ancestry": [training_ancestry],
        "ambiguous_policy": [ambiguous_policy],
        "n_observed_variants": [len(observed_variants)],
        "n_imputed_variants": [len(imputed_models)],
        "n_intercept_only": [sum(1 for m in imputed_models if m.is_intercept_only)],
        "include_variance_scaling": [include_variance_scaling],
        "calibration_params_json": [
            json.dumps(asdict(calibration_params)) if calibration_params else None
        ],
        "evaluation_metrics_json": [
            json.dumps(asdict(evaluation_metrics)) if evaluation_metrics else None
        ],
        "training_summary_json": [
            json.dumps(training_summary) if training_summary else None
        ],
    }
    return pa.Table.from_pydict(data)


def _build_observed_variants_table(
    observed_variants: List[VariantInfo],
) -> pa.Table:
    """Build observed variants table."""
    if not observed_variants:
        # Return empty table with correct schema
        schema = pa.schema([
            ("variant_id", pa.string()),
            ("chromosome", pa.string()),
            ("position", pa.int64()),
            ("effect_allele", pa.string()),
            ("other_allele", pa.string()),
            ("beta", pa.float64()),
            ("platform_index", pa.int32()),
        ])
        return pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)

    data = {
        "variant_id": [v.variant_id for v in observed_variants],
        "chromosome": [v.chromosome for v in observed_variants],
        "position": [v.position for v in observed_variants],
        "effect_allele": [v.effect_allele for v in observed_variants],
        "other_allele": [v.other_allele for v in observed_variants],
        "beta": [v.beta for v in observed_variants],
        "platform_index": list(range(len(observed_variants))),
    }
    return pa.Table.from_pydict(data)


def _build_imputed_variants_table(
    imputed_models: List[ImputedVariantModel],
    include_variance_scaling: bool,
) -> pa.Table:
    """Build imputed variants table (excluding coefficients)."""
    if not imputed_models:
        # Return empty table with correct schema
        fields = [
            ("variant_id", pa.string()),
            ("chromosome", pa.string()),
            ("position", pa.int64()),
            ("effect_allele", pa.string()),
            ("other_allele", pa.string()),
            ("beta", pa.float64()),
            ("allele_frequency", pa.float64()),
            ("imputation_r2", pa.float64()),
            ("intercept", pa.float64()),
            ("is_intercept_only", pa.bool_()),
            ("n_predictors", pa.int32()),
        ]
        if include_variance_scaling:
            fields.append(("residual_variance", pa.float64()))
        schema = pa.schema(fields)
        return pa.Table.from_pydict({f[0]: [] for f in fields}, schema=schema)

    data = {
        "variant_id": [m.variant_id for m in imputed_models],
        "chromosome": [m.chromosome for m in imputed_models],
        "position": [m.position for m in imputed_models],
        "effect_allele": [m.effect_allele for m in imputed_models],
        "other_allele": [m.other_allele for m in imputed_models],
        "beta": [m.beta for m in imputed_models],
        "allele_frequency": [m.allele_frequency for m in imputed_models],
        "imputation_r2": [m.imputation_r2 for m in imputed_models],
        "intercept": [m.intercept for m in imputed_models],
        "is_intercept_only": [m.is_intercept_only for m in imputed_models],
        "n_predictors": [len(m.predictor_variant_ids) for m in imputed_models],
    }
    if include_variance_scaling:
        data["residual_variance"] = [m.residual_variance for m in imputed_models]

    return pa.Table.from_pydict(data)


def _build_coefficients_table(
    imputed_models: List[ImputedVariantModel],
) -> pa.Table:
    """Build sparse coefficients table (one row per predictor-target pair).

    Each row carries the predictor's allele metadata (counted/other allele, locus,
    allele frequency) so a reloaded model can orient raw genotypes (schema v2).
    Missing metadata is written as ""/NaN and reconstructed as absent on load.
    """
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
                else float("nan")
            )

    if not target_variant_ids:
        schema = pa.schema([
            ("target_variant_id", pa.string()),
            ("predictor_variant_id", pa.string()),
            ("coefficient", pa.float64()),
            ("predictor_chromosome", pa.string()),
            ("predictor_position", pa.int64()),
            ("predictor_counted_allele", pa.string()),
            ("predictor_other_allele", pa.string()),
            ("predictor_allele_frequency", pa.float64()),
        ])
        return pa.Table.from_pydict(
            {f.name: [] for f in schema},
            schema=schema,
        )

    return pa.Table.from_pydict({
        "target_variant_id": target_variant_ids,
        "predictor_variant_id": predictor_variant_ids,
        "coefficient": coefficients,
        "predictor_chromosome": predictor_chromosomes,
        "predictor_position": predictor_positions,
        "predictor_counted_allele": predictor_counted_alleles,
        "predictor_other_allele": predictor_other_alleles,
        "predictor_allele_frequency": predictor_allele_frequencies,
    })


def export_to_arrow(
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
) -> Path:
    """Export trained imputation model to Arrow IPC format (schema v2.0).

    Creates an Arrow IPC file containing multiple record batches for efficient
    in-memory access and inter-process communication. The sparse ``coefficients``
    table carries per-predictor allele metadata for allele-aware reloading.

    Args:
        output_path: Path for output Arrow file.
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

    Returns:
        Path to the created Arrow file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build tables
    metadata_table = _build_metadata_table(
        observed_variants,
        imputed_models,
        calibration_params,
        evaluation_metrics,
        platform_name,
        prs_id,
        genome_build,
        model_name,
        include_variance_scaling,
        training_summary,
        reference_panel_id,
        training_ancestry,
        ambiguous_policy,
    )
    observed_table = _build_observed_variants_table(observed_variants)
    # Per-observed-variant fallback models (P1.8): a parallel table reusing the
    # imputed-variants layout, with predictors folded into the shared coefficients.
    fallback_models = [v.fallback for v in observed_variants if v.fallback is not None]
    imputed_table = _build_imputed_variants_table(
        imputed_models, include_variance_scaling
    )
    fallback_table = _build_imputed_variants_table(
        fallback_models, include_variance_scaling
    )
    coefficients_table = _build_coefficients_table(imputed_models + fallback_models)

    # Store table names in custom metadata
    schema = pa.schema([
        ("table_name", pa.string()),
        ("data", pa.binary()),
    ])

    # Serialize each table to bytes
    tables_data = []
    for name, table in [
        ("metadata", metadata_table),
        ("observed_variants", observed_table),
        ("imputed_variants", imputed_table),
        ("observed_fallbacks", fallback_table),
        ("coefficients", coefficients_table),
    ]:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as stream_writer:
            stream_writer.write_table(table)
        tables_data.append({"table_name": name, "data": sink.getvalue().to_pybytes()})

    # Write container table
    container = pa.Table.from_pylist(tables_data, schema=schema)
    with pa.ipc.new_file(str(output_path), container.schema) as writer:
        writer.write_table(container)

    return output_path


def export_to_parquet(
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
    compression: str = "snappy",
) -> Dict[str, Path]:
    """Export trained imputation model to Parquet format (schema v2.0).

    Creates multiple Parquet files (one per table) for efficient columnar storage
    with compression. The sparse ``coefficients`` table carries per-predictor allele
    metadata for allele-aware reloading.

    Args:
        output_path: Base path for output (directory will be created).
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
        compression: Compression codec ("snappy", "gzip", "zstd", "none").

    Returns:
        Dictionary mapping table names to file paths.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build tables
    metadata_table = _build_metadata_table(
        observed_variants,
        imputed_models,
        calibration_params,
        evaluation_metrics,
        platform_name,
        prs_id,
        genome_build,
        model_name,
        include_variance_scaling,
        training_summary,
        reference_panel_id,
        training_ancestry,
        ambiguous_policy,
    )
    observed_table = _build_observed_variants_table(observed_variants)
    # Per-observed-variant fallback models (P1.8): a parallel table reusing the
    # imputed-variants layout, with predictors folded into the shared coefficients.
    fallback_models = [v.fallback for v in observed_variants if v.fallback is not None]
    imputed_table = _build_imputed_variants_table(
        imputed_models, include_variance_scaling
    )
    fallback_table = _build_imputed_variants_table(
        fallback_models, include_variance_scaling
    )
    coefficients_table = _build_coefficients_table(imputed_models + fallback_models)

    # Write each table as a separate Parquet file
    paths = {}
    compression_arg = None if compression == "none" else compression

    for name, table in [
        ("metadata", metadata_table),
        ("observed_variants", observed_table),
        ("imputed_variants", imputed_table),
        ("observed_fallbacks", fallback_table),
        ("coefficients", coefficients_table),
    ]:
        file_path = output_path / f"{name}.parquet"
        pq.write_table(table, file_path, compression=compression_arg)
        paths[name] = file_path

    return paths
