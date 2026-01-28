"""Model loaders for restoring trained imputation models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)


def load_model_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load trained imputation model from JSON format.

    Returns the complete JSON dictionary for JavaScript interop
    verification and lightweight model inspection.

    Args:
        path: Path to JSON model file.

    Returns:
        Dictionary containing all model components:
        - metadata: Model metadata dict
        - observed_variants: List of observed variant dicts
        - imputed_variants: List of imputed variant dicts with coefficients
        - platform_variant_index: Dict mapping variant_id to index
        - calibration_params: Optional calibration parameters dict
        - evaluation_metrics: Optional evaluation metrics dict
        - training_summary: Optional training summary dict

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required keys are missing.
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    # Validate required keys
    required_keys = ["metadata", "observed_variants", "imputed_variants"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in JSON model: {missing_keys}")

    return data


def _load_observed_variants_hdf5(f: h5py.File) -> List[VariantInfo]:
    """Load observed variants from HDF5 file."""
    grp = f["observed_variants"]
    n = len(grp["variant_id"])

    if n == 0:
        return []

    variant_ids = [v.decode("utf-8") for v in grp["variant_id"][:]]
    chromosomes = [v.decode("utf-8") for v in grp["chromosome"][:]]
    positions = grp["position"][:]
    effect_alleles = [v.decode("utf-8") for v in grp["effect_allele"][:]]
    other_alleles = [v.decode("utf-8") for v in grp["other_allele"][:]]
    betas = grp["beta"][:]

    return [
        VariantInfo(
            variant_id=variant_ids[i],
            chromosome=chromosomes[i],
            position=int(positions[i]),
            effect_allele=effect_alleles[i],
            other_allele=other_alleles[i] if other_alleles[i] else None,
            beta=float(betas[i]),
        )
        for i in range(n)
    ]


def _load_imputed_models_hdf5(f: h5py.File) -> List[ImputedVariantModel]:
    """Load imputed variant models from HDF5 file."""
    grp = f["imputed_variants"]
    coef_grp = f["coefficients"]
    n = len(grp["variant_id"])

    if n == 0:
        return []

    # Load base variant data
    variant_ids = [v.decode("utf-8") for v in grp["variant_id"][:]]
    chromosomes = [v.decode("utf-8") for v in grp["chromosome"][:]]
    positions = grp["position"][:]
    effect_alleles = [v.decode("utf-8") for v in grp["effect_allele"][:]]
    other_alleles = [v.decode("utf-8") for v in grp["other_allele"][:]]
    betas = grp["beta"][:]
    allele_frequencies = grp["allele_frequency"][:]
    imputation_r2s = grp["imputation_r2"][:]
    intercepts = grp["intercept"][:]
    is_intercept_only = grp["is_intercept_only"][:]

    # Load residual variance if present
    has_residual_variance = "residual_variance" in grp
    if has_residual_variance:
        residual_variances = grp["residual_variance"][:]
    else:
        residual_variances = np.zeros(n)

    # Load sparse coefficients
    n_coef = len(coef_grp["target_variant_id"])
    if n_coef > 0:
        target_ids = [v.decode("utf-8") for v in coef_grp["target_variant_id"][:]]
        predictor_ids = [v.decode("utf-8") for v in coef_grp["predictor_variant_id"][:]]
        coefficients = coef_grp["coefficient"][:]
    else:
        target_ids = []
        predictor_ids = []
        coefficients = np.array([])

    # Build lookup: variant_id -> list of (predictor_id, coefficient)
    coef_lookup: Dict[str, List[Tuple[str, float]]] = {}
    for i in range(n_coef):
        target = target_ids[i]
        if target not in coef_lookup:
            coef_lookup[target] = []
        coef_lookup[target].append((predictor_ids[i], coefficients[i]))

    # Reconstruct ImputedVariantModel objects
    models = []
    for i in range(n):
        var_id = variant_ids[i]
        coef_pairs = coef_lookup.get(var_id, [])
        pred_ids = [p[0] for p in coef_pairs]
        coefs = np.array([p[1] for p in coef_pairs], dtype=np.float64)

        models.append(
            ImputedVariantModel(
                variant_id=var_id,
                chromosome=chromosomes[i],
                position=int(positions[i]),
                effect_allele=effect_alleles[i],
                other_allele=other_alleles[i] if other_alleles[i] else None,
                beta=float(betas[i]),
                allele_frequency=float(allele_frequencies[i]),
                imputation_r2=float(imputation_r2s[i]),
                residual_variance=float(residual_variances[i]),
                intercept=float(intercepts[i]),
                predictor_variant_ids=pred_ids,
                coefficients=coefs,
                is_intercept_only=bool(is_intercept_only[i]),
            )
        )

    return models


def _load_calibration_params_hdf5(f: h5py.File) -> Optional[CalibrationParams]:
    """Load calibration parameters from HDF5 metadata if present."""
    meta = f["metadata"]
    if "calibration_params_json" not in meta:
        return None

    json_str = meta["calibration_params_json"][()].decode("utf-8")
    data = json.loads(json_str)
    return CalibrationParams(**data)


def _load_evaluation_metrics_hdf5(f: h5py.File) -> Optional[EvaluationMetrics]:
    """Load evaluation metrics from HDF5 metadata if present."""
    meta = f["metadata"]
    if "evaluation_metrics_json" not in meta:
        return None

    json_str = meta["evaluation_metrics_json"][()].decode("utf-8")
    data = json.loads(json_str)
    return EvaluationMetrics(**data)


def _load_metadata_hdf5(f: h5py.File) -> Dict[str, Any]:
    """Load metadata attributes from HDF5 file."""
    meta = f["metadata"]
    result = {}

    for key in meta.attrs:
        result[key] = meta.attrs[key]

    # Load training summary if present
    if "training_summary_json" in meta:
        json_str = meta["training_summary_json"][()].decode("utf-8")
        result["training_summary"] = json.loads(json_str)

    return result


def load_model_hdf5(
    path: Union[str, Path],
) -> Tuple[
    List[VariantInfo],
    List[ImputedVariantModel],
    Optional[CalibrationParams],
    Optional[EvaluationMetrics],
    Dict[str, Any],
]:
    """Load trained imputation model from HDF5 format.

    Reconstructs all model components needed for PRSPredictor
    from the HDF5 hierarchical structure.

    Args:
        path: Path to HDF5 model file.

    Returns:
        Tuple of:
        - observed_variants: List of VariantInfo for directly observed variants
        - imputed_models: List of ImputedVariantModel with reconstructed coefficients
        - calibration_params: Optional CalibrationParams if present
        - evaluation_metrics: Optional EvaluationMetrics if present
        - metadata: Dict of metadata attributes (format_version, prs_id, etc.)

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required groups are missing from the HDF5 file.

    Example:
        >>> observed, imputed, calib, metrics, meta = load_model_hdf5("model.h5")
        >>> predictor = PRSPredictor(observed, imputed, calib)
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        # Validate required groups
        required_groups = ["metadata", "observed_variants", "imputed_variants", "coefficients"]
        missing_groups = [g for g in required_groups if g not in f]
        if missing_groups:
            raise KeyError(f"Missing required groups in HDF5 file: {missing_groups}")

        observed_variants = _load_observed_variants_hdf5(f)
        imputed_models = _load_imputed_models_hdf5(f)
        calibration_params = _load_calibration_params_hdf5(f)
        evaluation_metrics = _load_evaluation_metrics_hdf5(f)
        metadata = _load_metadata_hdf5(f)

    return observed_variants, imputed_models, calibration_params, evaluation_metrics, metadata
