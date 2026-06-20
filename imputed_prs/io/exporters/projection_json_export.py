"""JSON export for trained projection models (schema v2.0).

Mirrors :mod:`imputed_prs.io.exporters.json_export` (the imputation exporter) for
the projection product, which previously had no portable artifact. The schema is
the same browser-deployable v2.0 family: a top-level ``metadata`` + ``provenance``
block, ``observed_variants`` carrying multi-key ids + ambiguity flags, and
``region_models`` whose predictors and (projected) PRS variants are each
self-describing objects so a client-side scorer can orient raw genotypes without
trusting cross-array index alignment.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from imputed_prs.core.harmonizer import _is_ambiguous_snp
from imputed_prs.core.types import (
    CalibrationParams,
    ProjectionRegionModel,
    VariantInfo,
)

# Reuse the imputation exporter's policy constant and per-variant model serializer
# (used for the optional observed-variant fallback block, P2.4) so the two
# artifacts share a single source of truth.
from imputed_prs.io.exporters.json_export import (
    DEFAULT_AMBIGUOUS_POLICY,
    _serialize_imputed_model,
)


def _serialize_region_model(
    region: ProjectionRegionModel, *, include_variance_scaling: bool
) -> Dict[str, Any]:
    """Serialize a ProjectionRegionModel to a self-describing dict.

    Each predictor and each projected PRS variant is an object carrying its own
    allele metadata so a browser can orient it without trusting cross-array index
    alignment. ``region_id`` and the explicit ``chromosome``/``start``/``end`` are
    both emitted so a loader need not parse the id (the ``chr`` prefix plus ``X``/
    ``MT`` make string-parsing fragile). ``prs_variants[].chromosome`` is
    denormalized from the region's chromosome (regions are single-chromosome) for
    the scorer's convenience; it is not a distinct stored field on the dataclass.

    ``include_variance_scaling`` is accepted for signature parity with the
    imputation serializer but unused: projection regions have no per-region
    residual-variance field.

    Numpy ``float64``/int scalars from ``betas``/``coefficients``/AF arrays and the
    region's float fields are cast to Python ``float``/``int`` so ``json.dump``
    succeeds (it cannot serialize numpy scalars).
    """
    # The dataclass permits ragged arrays (every list field defaults to []), so
    # validate index-alignment loudly here rather than emit nulls a P2.2 loader
    # cannot use. This also documents the alignment contract at the export boundary.
    n_pred = len(region.predictor_variant_ids)
    if not (
        len(region.coefficients) == n_pred
        and len(region.predictor_chromosomes) == n_pred
        and len(region.predictor_positions) == n_pred
        and len(region.predictor_counted_alleles) == n_pred
        and len(region.predictor_other_alleles) == n_pred
        and len(region.predictor_allele_frequencies) == n_pred
    ):
        raise ValueError(
            f"region {region.region_id}: predictor_* arrays are not index-aligned "
            f"with predictor_variant_ids (n={n_pred})"
        )
    n_prs = len(region.prs_variant_ids)
    if not (
        len(region.betas) == n_prs
        and len(region.prs_positions) == n_prs
        and len(region.prs_effect_alleles) == n_prs
        and len(region.prs_other_alleles) == n_prs
    ):
        raise ValueError(
            f"region {region.region_id}: prs_* arrays are not index-aligned with "
            f"prs_variant_ids (n={n_prs})"
        )

    predictors = [
        {
            "variant_id": pred_id,
            "chromosome": region.predictor_chromosomes[i],
            "position": int(region.predictor_positions[i]),
            "counted_allele": region.predictor_counted_alleles[i],
            "other_allele": region.predictor_other_alleles[i],
            "allele_frequency": float(region.predictor_allele_frequencies[i]),
            "coefficient": float(region.coefficients[i]),
        }
        for i, pred_id in enumerate(region.predictor_variant_ids)
    ]
    prs_variants = [
        {
            "variant_id": prs_id,
            "chromosome": region.chromosome,
            "position": int(region.prs_positions[i]),
            "effect_allele": region.prs_effect_alleles[i],
            # May legitimately be None (PRS source lacked other_allele); emit
            # verbatim so it round-trips as JSON null.
            "other_allele": region.prs_other_alleles[i],
            "beta": float(region.betas[i]),
        }
        for i, prs_id in enumerate(region.prs_variant_ids)
    ]
    return {
        "region_id": region.region_id,
        "chromosome": region.chromosome,
        "start": int(region.start),
        "end": int(region.end),
        "intercept": float(region.intercept),
        "cv_mse": float(region.cv_mse),
        "cv_r2": float(region.cv_r2),
        "is_intercept_only": bool(region.is_intercept_only),
        "mean_prs_contribution": float(region.mean_prs_contribution),
        "target_variance": float(region.target_variance),
        "predictors": predictors,
        "prs_variants": prs_variants,
    }


def export_projection_to_json(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    region_models: List[ProjectionRegionModel],
    calibration_params: Optional[CalibrationParams] = None,
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
    """Export a trained projection model to portable JSON (schema v2.0).

    Creates a browser-deployable JSON artifact carrying everything a client-side
    scorer needs to orient raw user genotypes and validate compatibility:
    per-predictor allele metadata for every region, observed-term accepted ids +
    ambiguity flags, and a top-level provenance block (genome build, platform,
    reference panel, ancestry, ambiguity policy, centering/scaling).

    The projection analog of the imputation exporter. It diverges in two ways:
    the per-variant blocks are grouped into ``region_models`` (instead of a flat
    ``imputed_variants`` list), and there is no ``evaluation_metrics`` parameter
    (the projection model stores none).

    Args:
        output_path: Path for output JSON file.
        observed_variants: List of VariantInfo for directly observed variants.
        region_models: List of ProjectionRegionModel for the projected regions.
        calibration_params: Optional calibration scaling parameters.
        platform_name: Name of the genotyping platform (also recorded as the
            provenance ``platform_id``).
        prs_id: PRS identifier (e.g., "PGS000004").
        genome_build: Genome build (e.g., "GRCh37").
        model_name: Human-readable model name.
        include_variance_scaling: Accepted for parity with the imputation exporter;
            projection regions have no residual-variance field, so it is unused.
        training_summary: Optional summary statistics from training.
        reference_panel_id: Provenance — reference panel used for training (e.g.,
            "1000G_phase3_EUR"). Consumed by the build/platform compatibility check.
        training_ancestry: Provenance — ancestry of the training cohort (e.g., "EUR").
        ambiguous_policy: Provenance — how palindromic (A/T, C/G) variants must be
            handled by the deployed scorer. Recorded in the artifact.
        require_other_allele: Deploy gate. When True (default) the export raises if
            any variant the scorer counts from the upload (an observed term or a
            region predictor) lacks ``other_allele``. Region PRS variants are
            projected from predictors, not counted, so they are not gated. Pass
            False for a non-deployable research export.
        require_provenance: Deploy gate. When True (default) the export raises if
            ``genome_build``, ``reference_panel_id``, or ``training_ancestry`` is
            missing. Pass False for a non-deployable research export.

    Returns:
        Path to the created JSON file.

    Raises:
        ValueError: If ``require_other_allele`` is True and an observed variant or
            region predictor lacks ``other_allele``; if ``require_provenance`` is
            True and a required provenance field is missing; or if a region's
            predictor/PRS arrays are not index-aligned.
    """
    output_path = Path(output_path)

    # Deploy gate: a browser cannot orient a scored variant without the other allele
    # of the biallelic pair. "Scored" = observed terms (the effect allele is counted
    # from the user's genotype) plus region predictors (the ALT allele is counted to
    # feed the region model). Region PRS variants are *projected* from the predictors,
    # not counted from the upload (the projection analog of the imputed target), so
    # they are not gated here — mirroring the imputation exporter, which skips the
    # imputed target allele for the same reason.
    if require_other_allele:
        missing: List[str] = []
        for v in observed_variants:
            if not v.other_allele:
                missing.append(f"observed:{v.variant_id}")
            # A fallback's predictors are also counted from the upload (P2.4).
            if v.fallback is not None:
                for i, pred_id in enumerate(v.fallback.predictor_variant_ids):
                    other = (
                        v.fallback.predictor_other_alleles[i]
                        if i < len(v.fallback.predictor_other_alleles)
                        else None
                    )
                    if not other:
                        missing.append(f"fallback:{v.variant_id}<-{pred_id}")
        for region in region_models:
            for i, pred_id in enumerate(region.predictor_variant_ids):
                other = (
                    region.predictor_other_alleles[i]
                    if i < len(region.predictor_other_alleles)
                    else None
                )
                if not other:
                    missing.append(f"predictor:{region.region_id}<-{pred_id}")
        if missing:
            raise ValueError(
                "Deployable JSON export requires 'other_allele' for every scored "
                f"variant; {len(missing)} are missing it (e.g. {missing[:5]}). "
                "Pass require_other_allele=False for a non-deployable research export."
            )

    # Deploy gate: a deployable artifact must carry the provenance the scorer uses
    # to validate that an upload is compatible with the trained model (P1.7).
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

    # Metadata section (v1.0 keys retained for continuity; the browser reads the
    # richer `provenance` block below).
    metadata = {
        "format_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "prs_id": prs_id,
        "platform_name": platform_name,
        "genome_build": genome_build,
        "n_observed_variants": len(observed_variants),
        "n_region_models": len(region_models),
        "n_intercept_only_regions": sum(
            1 for r in region_models if r.is_intercept_only
        ),
        "include_variance_scaling": include_variance_scaling,
    }

    # Provenance — identity + compatibility metadata consumed by the build/platform
    # hard-check (P1.7). `platform_id` is the platform name; `centering_scaling`
    # reuses the calibration object.
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

    # Observed variants — identical shape to the imputation exporter (multi-key ids
    # + palindrome flag + optional per-variant fallback model).
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
            "fallback": (
                _serialize_imputed_model(
                    v.fallback, include_variance_scaling=include_variance_scaling
                )
                if v.fallback is not None
                else None
            ),
        }
        for v in observed_variants
    ]

    region_models_data = [
        _serialize_region_model(
            region, include_variance_scaling=include_variance_scaling
        )
        for region in region_models
    ]

    # Platform variant index — maps each observed variant_id to its position in the
    # observed list. Rebuilt here from `observed_variants` (NOT the model's
    # `_platform_variant_index`, which is keyed by the full platform-manifest row
    # set) to match the imputation exporter's semantics.
    platform_variant_index = {
        v.variant_id: i for i, v in enumerate(observed_variants)
    }

    output: Dict[str, Any] = {
        "metadata": metadata,
        "provenance": provenance,
        "observed_variants": observed_variants_data,
        "region_models": region_models_data,
        "platform_variant_index": platform_variant_index,
    }

    if calibration_params is not None:
        output["calibration_params"] = asdict(calibration_params)

    if training_summary is not None:
        output["training_summary"] = training_summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path
