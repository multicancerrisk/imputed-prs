"""JSON export for trained imputation models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from imputed_prs.core.harmonizer import _is_ambiguous_snp
from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)

# Default deploy-time policy for palindromic (A/T, C/G) variants. Recorded in the
# artifact's provenance; the browser scorer enforces it (training-side
# `exclude_ambiguous` stays False so the model still carries these variants).
DEFAULT_AMBIGUOUS_POLICY = "exclude_unless_platform_strand_known"


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
    reference_panel_id: Optional[str] = None,
    training_ancestry: Optional[str] = None,
    ambiguous_policy: str = DEFAULT_AMBIGUOUS_POLICY,
    require_other_allele: bool = True,
    require_provenance: bool = True,
) -> Path:
    """Export trained imputation model to portable JSON (schema v2.0).

    Creates a browser-deployable JSON artifact carrying everything a client-side
    scorer needs to orient raw user genotypes and validate compatibility:
    per-predictor allele metadata (so each coefficient knows which allele it
    counts), observed-term accepted ids + ambiguity flags, and a top-level
    provenance block (genome build, platform, reference panel, ancestry, ambiguity
    policy, centering/scaling).

    Args:
        output_path: Path for output JSON file.
        observed_variants: List of VariantInfo for directly observed variants.
        imputed_models: List of ImputedVariantModel for imputed variants.
        calibration_params: Optional calibration scaling parameters.
        evaluation_metrics: Optional evaluation metrics from training.
        platform_name: Name of the genotyping platform (also recorded as the
            provenance ``platform_id``).
        prs_id: PRS identifier (e.g., "PGS000004").
        genome_build: Genome build (e.g., "GRCh37").
        model_name: Human-readable model name.
        include_variance_scaling: Whether to include variance/SE components.
        training_summary: Optional summary statistics from training.
        reference_panel_id: Provenance — reference panel used for training (e.g.,
            "1000G_phase3_EUR"). Consumed by the build/platform compatibility check.
        training_ancestry: Provenance — ancestry of the training cohort (e.g.,
            "EUR").
        ambiguous_policy: Provenance — how palindromic (A/T, C/G) variants must be
            handled by the deployed scorer. Recorded in the artifact.
        require_other_allele: Deploy gate. When True (default) the export raises if
            any scored variant (observed term or predictor) lacks ``other_allele``,
            which the browser needs for strand-safe orientation. Pass False for a
            non-deployable research export.
        require_provenance: Deploy gate. When True (default) the export raises if
            ``genome_build``, ``reference_panel_id``, or ``training_ancestry`` is
            missing, since the scorer needs them to validate that an upload is
            compatible with the trained model. Pass False for a non-deployable
            research export.

    Returns:
        Path to the created JSON file.

    Raises:
        ValueError: If ``require_other_allele`` is True and any observed variant or
            predictor lacks ``other_allele``; or if ``require_provenance`` is True
            and a required provenance field is missing.
    """
    output_path = Path(output_path)

    # Deploy gate: a browser cannot orient a scored variant without the other
    # allele of the biallelic pair. "Scored" = observed terms (the effect allele is
    # counted from the user's genotype) plus predictors (the ALT allele is counted
    # to feed the imputation model). The imputed *target* allele is predicted, not
    # counted, so it is not gated here.
    if require_other_allele:
        missing: List[str] = []
        for v in observed_variants:
            if not v.other_allele:
                missing.append(f"observed:{v.variant_id}")
        for model in imputed_models:
            for i, pred_id in enumerate(model.predictor_variant_ids):
                other = (
                    model.predictor_other_alleles[i]
                    if i < len(model.predictor_other_alleles)
                    else None
                )
                if not other:
                    missing.append(f"predictor:{model.variant_id}<-{pred_id}")
        if missing:
            raise ValueError(
                "Deployable JSON export requires 'other_allele' for every scored "
                f"variant; {len(missing)} are missing it (e.g. {missing[:5]}). "
                "Pass require_other_allele=False for a non-deployable research export."
            )

    # Deploy gate: a deployable artifact must carry the provenance the scorer
    # uses to validate that an upload is compatible with the trained model (the
    # build/platform hard-check, P1.7). These are recorded but allowed to be null
    # by the schema, so they are enforced here rather than in the schema.
    if require_provenance:
        missing_provenance = [
            name
            for name, value in (
                ("genome_build", genome_build),
                ("reference_panel_id", reference_panel_id),
                ("training_ancestry", training_ancestry),
            )
            if not value
        ]
        if missing_provenance:
            raise ValueError(
                "Deployable JSON export requires provenance fields "
                f"{missing_provenance} to be set (non-null); they identify the "
                "build, reference panel, and training ancestry the scorer needs to "
                "validate compatibility. Pass require_provenance=False for a "
                "non-deployable research export."
            )

    # Build metadata section. All v1.0 keys are retained for continuity; the
    # browser reads the richer `provenance` block below.
    metadata = {
        "format_version": "2.0",
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

    # Provenance — identity + compatibility metadata consumed by the build/platform
    # hard-check (P1.7) and surfaced to the browser. `platform_id` is the platform
    # name (there is no distinct platform-id concept). `centering_scaling` reuses
    # the existing calibration object.
    provenance = {
        "genome_build": genome_build,
        "platform_id": platform_name,
        "reference_panel_id": reference_panel_id,
        "training_ancestry": training_ancestry,
        "ambiguous_policy": ambiguous_policy,
        "centering_scaling": (
            asdict(calibration_params) if calibration_params is not None else None
        ),
    }

    # Serialize observed variants with multi-key ids + a palindrome flag. Both are
    # derived from fields that already round-trip, so they need not be stored.
    observed_variants_data = [
        {
            "variant_id": v.variant_id,
            "chromosome": v.chromosome,
            "position": v.position,
            "effect_allele": v.effect_allele,
            "other_allele": v.other_allele,
            "beta": v.beta,
            "accepted_ids": [v.variant_id, f"{v.chromosome}:{v.position}"],
            "ambiguous": (
                v.other_allele is not None
                and _is_ambiguous_snp(v.effect_allele, v.other_allele)
            ),
        }
        for v in observed_variants
    ]

    # Serialize imputed variants. Each predictor is a self-describing object so a
    # browser can orient it without trusting cross-array index alignment.
    imputed_variants_data = []
    for model in imputed_models:
        predictors = [
            {
                "variant_id": pred_id,
                "chromosome": model.predictor_chromosomes[i],
                "position": model.predictor_positions[i],
                "counted_allele": model.predictor_counted_alleles[i],
                "other_allele": model.predictor_other_alleles[i],
                "allele_frequency": float(model.predictor_allele_frequencies[i]),
                "coefficient": float(model.coefficients[i]),
            }
            for i, pred_id in enumerate(model.predictor_variant_ids)
        ]
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
            "predictors": predictors,
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
        "provenance": provenance,
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
