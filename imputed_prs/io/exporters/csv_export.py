"""CSV export for per-variant summary table."""

from pathlib import Path
from typing import List, Union

import pandas as pd

from imputed_prs.core.types import (
    ImputedVariantModel,
    VariantInfo,
)

# Columns of the companion long-format coefficients table (one row per
# predictor->target pair), carrying per-predictor allele metadata (schema v2).
COEFFICIENTS_COLUMNS = [
    "target_variant_id",
    "predictor_variant_id",
    "predictor_chromosome",
    "predictor_position",
    "predictor_counted_allele",
    "predictor_other_allele",
    "predictor_allele_frequency",
    "coefficient",
]


def coefficients_path_for(variants_path: Union[str, Path]) -> Path:
    """Return the companion coefficients-CSV path for a per-variant table path.

    ``<name>_variants.csv`` -> ``<name>_coefficients.csv``; any other stem gets a
    ``<stem>_coefficients.csv`` sibling. Used by both the exporter and the loader so
    the two files stay paired.
    """
    variants_path = Path(variants_path)
    stem = variants_path.stem
    base = stem[: -len("_variants")] if stem.endswith("_variants") else stem
    return variants_path.with_name(f"{base}_coefficients.csv")


def _write_coefficients_csv(
    coefficients_path: Path,
    imputed_models: List[ImputedVariantModel],
) -> None:
    """Write the companion long-format predictor coefficients table.

    Always written (header-only when there are no predictors). Missing per-predictor
    allele metadata is left blank and reconstructed as absent on load.
    """
    rows = []
    for model in imputed_models:
        for i, (pred_id, coef) in enumerate(
            zip(model.predictor_variant_ids, model.coefficients.tolist())
        ):
            rows.append(
                {
                    "target_variant_id": model.variant_id,
                    "predictor_variant_id": pred_id,
                    "predictor_chromosome": (
                        model.predictor_chromosomes[i]
                        if i < len(model.predictor_chromosomes)
                        else None
                    ),
                    "predictor_position": (
                        int(model.predictor_positions[i])
                        if i < len(model.predictor_positions)
                        else None
                    ),
                    "predictor_counted_allele": (
                        model.predictor_counted_alleles[i]
                        if i < len(model.predictor_counted_alleles)
                        else None
                    ),
                    "predictor_other_allele": (
                        model.predictor_other_alleles[i]
                        if i < len(model.predictor_other_alleles)
                        else None
                    ),
                    "predictor_allele_frequency": (
                        float(model.predictor_allele_frequencies[i])
                        if i < len(model.predictor_allele_frequencies)
                        else None
                    ),
                    "coefficient": coef,
                }
            )

    if rows:
        df = pd.DataFrame(rows)[COEFFICIENTS_COLUMNS]
    else:
        df = pd.DataFrame(columns=COEFFICIENTS_COLUMNS)

    coefficients_path = Path(coefficients_path)
    coefficients_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(coefficients_path, index=False)


def export_variant_table(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    include_variance_scaling: bool = True,
    include_predictor_details: bool = False,
) -> Path:
    """Export per-variant summary table to CSV format (schema v2).

    Creates a flat CSV file with one row per variant, combining observed and
    imputed variants with status indicators, plus a companion
    ``<name>_coefficients.csv`` (see :func:`coefficients_path_for`) holding the
    per-predictor allele metadata the flat table cannot represent. The pair
    round-trips through :func:`imputed_prs.io.loaders.load_model_csv`.

    Args:
        output_path: Path for output CSV file.
        observed_variants: List of VariantInfo for directly observed variants.
        imputed_models: List of ImputedVariantModel for imputed variants.
        include_variance_scaling: Whether to include residual_variance column.
        include_predictor_details: Whether to include predictor_variant_ids column.

    Returns:
        Path to the created per-variant CSV file (the companion coefficients CSV is
        written alongside it).

    Example:
        >>> export_variant_table("variants.csv", observed, imputed)
        >>> df = pd.read_csv("variants.csv")
        >>> print(df['status'].value_counts())
    """
    rows = []

    # Add observed variants
    for var in observed_variants:
        row = {
            "variant_id": var.variant_id,
            "chromosome": var.chromosome,
            "position": var.position,
            "effect_allele": var.effect_allele,
            "other_allele": var.other_allele,
            "beta": var.beta,
            "status": "observed",
            "imputation_r2": None,
            "allele_frequency": None,
            "intercept": None,
            "n_predictors": 0,
        }
        if include_variance_scaling:
            row["residual_variance"] = None
        if include_predictor_details:
            row["predictor_variant_ids"] = None
        rows.append(row)

    # Add imputed variants
    for model in imputed_models:
        status = "intercept_only" if model.is_intercept_only else "imputed"
        row = {
            "variant_id": model.variant_id,
            "chromosome": model.chromosome,
            "position": model.position,
            "effect_allele": model.effect_allele,
            "other_allele": model.other_allele,
            "beta": model.beta,
            "status": status,
            "imputation_r2": model.imputation_r2,
            "allele_frequency": model.allele_frequency,
            "intercept": model.intercept,
            "n_predictors": len(model.predictor_variant_ids),
        }
        if include_variance_scaling:
            row["residual_variance"] = model.residual_variance
        if include_predictor_details:
            # Semicolon-separated list of predictor variant IDs
            row["predictor_variant_ids"] = (
                ";".join(model.predictor_variant_ids)
                if model.predictor_variant_ids
                else None
            )
        rows.append(row)

    # Add per-observed-variant fallback models (P1.8) as dedicated rows so the flat
    # table round-trips them; their predictors go in the companion coefficients CSV.
    # is_intercept_only is encoded in `status`, mirroring the imputed convention.
    fallback_models = [v.fallback for v in observed_variants if v.fallback is not None]
    for model in fallback_models:
        status = (
            "observed_fallback_intercept_only"
            if model.is_intercept_only
            else "observed_fallback"
        )
        row = {
            "variant_id": model.variant_id,
            "chromosome": model.chromosome,
            "position": model.position,
            "effect_allele": model.effect_allele,
            "other_allele": model.other_allele,
            "beta": model.beta,
            "status": status,
            "imputation_r2": model.imputation_r2,
            "allele_frequency": model.allele_frequency,
            "intercept": model.intercept,
            "n_predictors": len(model.predictor_variant_ids),
        }
        if include_variance_scaling:
            row["residual_variance"] = model.residual_variance
        if include_predictor_details:
            row["predictor_variant_ids"] = (
                ";".join(model.predictor_variant_ids)
                if model.predictor_variant_ids
                else None
            )
        rows.append(row)

    # Ensure consistent column order
    base_cols = [
        "variant_id",
        "chromosome",
        "position",
        "effect_allele",
        "other_allele",
        "beta",
        "status",
        "imputation_r2",
        "allele_frequency",
        "intercept",
        "n_predictors",
    ]
    if include_variance_scaling:
        base_cols.append("residual_variance")
    if include_predictor_details:
        base_cols.append("predictor_variant_ids")

    # Create DataFrame with explicit column order
    if rows:
        df = pd.DataFrame(rows)[base_cols]
    else:
        # Handle empty case - create empty DataFrame with correct columns
        df = pd.DataFrame(columns=base_cols)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Companion long-format coefficients table with per-predictor allele metadata
    # (schema v2) so a reloaded CSV model can orient raw genotypes. The flat
    # per-variant table above has no slot for per-predictor rows.
    _write_coefficients_csv(
        coefficients_path_for(output_path), imputed_models + fallback_models
    )

    return output_path
