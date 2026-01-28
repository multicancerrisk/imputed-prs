"""JSON export for trained imputation models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)


def export_to_json(
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
) -> Path:
    """Export trained imputation model to JSON format.

    Creates a portable JSON file containing all model components
    needed for PRS prediction in any environment (including JavaScript).

    Args:
        output_path: Path for output JSON file.
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

    Returns:
        Path to the created JSON file.
    """
    output_path = Path(output_path)

    # Build metadata section
    metadata = {
        "format_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "prs_id": prs_id,
        "platform_name": platform_name,
        "genome_build": genome_build,
        "n_observed_variants": len(observed_variants),
        "n_imputed_variants": len(imputed_models),
        "n_intercept_only": sum(1 for m in imputed_models if m.is_intercept_only),
        "include_variance_scaling": include_variance_scaling,
    }

    # Serialize observed variants
    observed_variants_data = [
        {
            "variant_id": v.variant_id,
            "chromosome": v.chromosome,
            "position": v.position,
            "effect_allele": v.effect_allele,
            "other_allele": v.other_allele,
            "beta": v.beta,
        }
        for v in observed_variants
    ]

    # Serialize imputed variants with sparse coefficient representation
    imputed_variants_data = []
    for model in imputed_models:
        model_dict = {
            "variant_id": model.variant_id,
            "chromosome": model.chromosome,
            "position": model.position,
            "effect_allele": model.effect_allele,
            "other_allele": model.other_allele,
            "beta": model.beta,
            "allele_frequency": model.allele_frequency,
            "imputation_r2": model.imputation_r2,
            "intercept": model.intercept,
            "is_intercept_only": model.is_intercept_only,
            "predictor_variant_ids": model.predictor_variant_ids,
            "coefficients": model.coefficients.tolist(),
        }
        if include_variance_scaling:
            model_dict["residual_variance"] = model.residual_variance
        imputed_variants_data.append(model_dict)

    # Build platform variant index (maps variant_id to position for fast lookup)
    platform_variant_index = {
        v.variant_id: i for i, v in enumerate(observed_variants)
    }

    # Build output structure
    output = {
        "metadata": metadata,
        "observed_variants": observed_variants_data,
        "imputed_variants": imputed_variants_data,
        "platform_variant_index": platform_variant_index,
    }

    # Add calibration params if available
    if calibration_params is not None:
        output["calibration_params"] = asdict(calibration_params)

    # Add evaluation metrics if available
    if evaluation_metrics is not None:
        output["evaluation_metrics"] = asdict(evaluation_metrics)

    # Add training summary if available
    if training_summary is not None:
        output["training_summary"] = training_summary

    # Write JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path
