"""CSV export for per-variant summary table."""

from pathlib import Path
from typing import List, Union

import pandas as pd

from imputed_prs.core.types import (
    ImputedVariantModel,
    VariantInfo,
)


def export_variant_table(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    include_variance_scaling: bool = True,
    include_predictor_details: bool = False,
) -> Path:
    """Export per-variant summary table to CSV format.

    Creates a flat CSV file with one row per variant, combining
    observed and imputed variants with status indicators.

    Args:
        output_path: Path for output CSV file.
        observed_variants: List of VariantInfo for directly observed variants.
        imputed_models: List of ImputedVariantModel for imputed variants.
        include_variance_scaling: Whether to include residual_variance column.
        include_predictor_details: Whether to include predictor_variant_ids column.

    Returns:
        Path to the created CSV file.

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

    return output_path
