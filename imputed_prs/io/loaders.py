"""Model loaders for restoring trained imputation models.

All formats reconstruct the same `(observed_variants, imputed_models,
calibration_params, evaluation_metrics, metadata)` tuple. The format-specific readers
each marshal their storage (HDF5 datasets, Arrow tables, CSV frames) into plain
column dicts and hand them to the shared `_rebuild_observed_variants` /
`_rebuild_imputed_models` helpers, so allele-aware reconstruction lives in one place.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.csv_export import (
    COEFFICIENTS_COLUMNS,
    coefficients_path_for,
)


# ---------------------------------------------------------------------------
# Shared reconstruction helpers (format-agnostic)
# ---------------------------------------------------------------------------


def _is_nan(value: Any) -> bool:
    """True for a float NaN (e.g. pandas-read empty cell)."""
    return isinstance(value, float) and np.isnan(value)


def _to_str(value: Any) -> str:
    """Decode/stringify a required string cell."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_str_or_none(value: Any) -> Optional[str]:
    """Decode/stringify an optional string cell; None/""/NaN -> None."""
    if value is None or _is_nan(value):
        return None
    s = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return s if s != "" else None


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce a cell to int; None/NaN -> default."""
    if value is None or _is_nan(value):
        return default
    return int(float(value))


def _to_float(value: Any, default: float = float("nan")) -> float:
    """Coerce a cell to float; None/""/NaN -> default."""
    if value is None or (isinstance(value, str) and value == ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    """Coerce a cell to bool, tolerating CSV "True"/"False" strings."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return bool(value)


def _rebuild_observed_variants(obs_cols: Dict[str, List[Any]]) -> List[VariantInfo]:
    """Reconstruct observed VariantInfo list from a column dict."""
    n = len(obs_cols.get("variant_id", []))
    return [
        VariantInfo(
            variant_id=_to_str(obs_cols["variant_id"][i]),
            chromosome=_to_str(obs_cols["chromosome"][i]),
            position=_to_int(obs_cols["position"][i]),
            effect_allele=_to_str(obs_cols["effect_allele"][i]),
            other_allele=_to_str_or_none(obs_cols["other_allele"][i]),
            beta=_to_float(obs_cols["beta"][i]),
        )
        for i in range(n)
    ]


def _rebuild_imputed_models(
    imp_cols: Dict[str, List[Any]],
    coef_cols: Dict[str, List[Any]],
) -> List[ImputedVariantModel]:
    """Reconstruct ImputedVariantModel list from per-variant + sparse-coefficient
    column dicts.

    The sparse coefficient rows carry per-predictor allele metadata when present
    (schema v2). When the metadata columns are absent (v1 artifacts) the predictor
    metadata is reconstructed as empty, mirroring the JSON v1 fallback.
    """
    n = len(imp_cols.get("variant_id", []))
    residual_variances = imp_cols.get("residual_variance")
    if residual_variances is None:
        residual_variances = [0.0] * n

    has_meta = "predictor_counted_allele" in coef_cols
    targets = coef_cols.get("target_variant_id", [])
    pred_ids_all = coef_cols.get("predictor_variant_id", [])
    coefs_all = coef_cols.get("coefficient", [])

    # variant_id -> list of (predictor_id, coef[, chrom, pos, counted, other, af])
    coef_lookup: Dict[str, List[tuple]] = {}
    for i in range(len(targets)):
        entry = coef_lookup.setdefault(_to_str(targets[i]), [])
        if has_meta:
            entry.append(
                (
                    _to_str(pred_ids_all[i]),
                    _to_float(coefs_all[i], 0.0),
                    coef_cols["predictor_chromosome"][i],
                    coef_cols["predictor_position"][i],
                    coef_cols["predictor_counted_allele"][i],
                    coef_cols["predictor_other_allele"][i],
                    coef_cols["predictor_allele_frequency"][i],
                )
            )
        else:
            entry.append((_to_str(pred_ids_all[i]), _to_float(coefs_all[i], 0.0)))

    models = []
    for j in range(n):
        vid = _to_str(imp_cols["variant_id"][j])
        pairs = coef_lookup.get(vid, [])
        pred_ids = [p[0] for p in pairs]
        coefs = np.array([p[1] for p in pairs], dtype=np.float64)
        if has_meta:
            pred_chroms = [_to_str_or_none(p[2]) or "" for p in pairs]
            pred_pos = [_to_int(p[3]) for p in pairs]
            pred_counted = [_to_str_or_none(p[4]) or "" for p in pairs]
            pred_other = [_to_str_or_none(p[5]) or "" for p in pairs]
            pred_af = np.array([_to_float(p[6]) for p in pairs], dtype=np.float64)
        else:
            pred_chroms, pred_pos, pred_counted, pred_other = [], [], [], []
            pred_af = np.array([], dtype=np.float64)

        models.append(
            ImputedVariantModel(
                variant_id=vid,
                chromosome=_to_str(imp_cols["chromosome"][j]),
                position=_to_int(imp_cols["position"][j]),
                effect_allele=_to_str(imp_cols["effect_allele"][j]),
                other_allele=_to_str_or_none(imp_cols["other_allele"][j]),
                beta=_to_float(imp_cols["beta"][j]),
                allele_frequency=_to_float(imp_cols["allele_frequency"][j]),
                imputation_r2=_to_float(imp_cols["imputation_r2"][j]),
                residual_variance=_to_float(residual_variances[j], 0.0),
                intercept=_to_float(imp_cols["intercept"][j]),
                predictor_variant_ids=pred_ids,
                coefficients=coefs,
                is_intercept_only=_to_bool(imp_cols["is_intercept_only"][j]),
                predictor_chromosomes=pred_chroms,
                predictor_positions=pred_pos,
                predictor_counted_alleles=pred_counted,
                predictor_other_alleles=pred_other,
                predictor_allele_frequencies=pred_af,
            )
        )
    return models


def _attach_observed_fallbacks(
    observed: List[VariantInfo],
    fallback_cols: Dict[str, List[Any]],
    coef_cols: Dict[str, List[Any]],
) -> None:
    """Reconstruct observed fallback models (P1.8) and attach them in place.

    ``fallback_cols`` is laid out exactly like the ``imputed_variants`` columns and
    ``coef_cols`` is the shared sparse coefficients table (imputed + fallback
    predictors keyed by ``target_variant_id``), so the fallbacks reconstruct through
    the same :func:`_rebuild_imputed_models` path and attach to their observed
    VariantInfo by ``variant_id``. A no-op when no fallbacks were serialized (a
    pre-P1.8 artifact, or a model whose observed variants all resolve directly).
    """
    if not fallback_cols.get("variant_id"):
        return
    fallback_models = _rebuild_imputed_models(fallback_cols, coef_cols)
    by_id = {m.variant_id: m for m in fallback_models}
    for v in observed:
        model = by_id.get(v.variant_id)
        if model is not None:
            v.fallback = model


def load_model_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load trained imputation model from JSON format.

    Returns the complete JSON dictionary for JavaScript interop
    verification and lightweight model inspection.

    Args:
        path: Path to JSON model file.

    Returns:
        Dictionary containing all model components:
        - metadata: Model metadata dict
        - provenance: Optional provenance dict (v2.0): genome_build, platform_id,
          reference_panel_id, training_ancestry, ambiguous_policy, centering_scaling
        - observed_variants: List of observed variant dicts (v2.0 adds accepted_ids
          and an ambiguous flag)
        - imputed_variants: List of imputed variant dicts; v2.0 carries a
          self-describing `predictors` list, v1.0 carried parallel
          predictor_variant_ids/coefficients arrays
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
    obs_cols = {
        "variant_id": [v.decode("utf-8") for v in grp["variant_id"][:]],
        "chromosome": [v.decode("utf-8") for v in grp["chromosome"][:]],
        "position": grp["position"][:].tolist(),
        "effect_allele": [v.decode("utf-8") for v in grp["effect_allele"][:]],
        "other_allele": [v.decode("utf-8") for v in grp["other_allele"][:]],
        "beta": grp["beta"][:].tolist(),
    }
    return _rebuild_observed_variants(obs_cols)


def _hdf5_model_cols(grp: h5py.Group) -> Dict[str, List[Any]]:
    """Read a variant-model HDF5 group (``imputed_variants`` or
    ``observed_fallbacks``) into a plain column dict."""
    cols: Dict[str, List[Any]] = {
        "variant_id": [v.decode("utf-8") for v in grp["variant_id"][:]],
        "chromosome": [v.decode("utf-8") for v in grp["chromosome"][:]],
        "position": grp["position"][:].tolist(),
        "effect_allele": [v.decode("utf-8") for v in grp["effect_allele"][:]],
        "other_allele": [v.decode("utf-8") for v in grp["other_allele"][:]],
        "beta": grp["beta"][:].tolist(),
        "allele_frequency": grp["allele_frequency"][:].tolist(),
        "imputation_r2": grp["imputation_r2"][:].tolist(),
        "intercept": grp["intercept"][:].tolist(),
        "is_intercept_only": grp["is_intercept_only"][:].tolist(),
    }
    if "residual_variance" in grp:
        cols["residual_variance"] = grp["residual_variance"][:].tolist()
    return cols


def _hdf5_coef_cols(coef_grp: h5py.Group) -> Dict[str, List[Any]]:
    """Read the sparse ``coefficients`` HDF5 group into a plain column dict,
    including the per-predictor allele metadata when present (schema v2)."""
    coef_cols: Dict[str, List[Any]] = {
        "target_variant_id": [
            v.decode("utf-8") for v in coef_grp["target_variant_id"][:]
        ],
        "predictor_variant_id": [
            v.decode("utf-8") for v in coef_grp["predictor_variant_id"][:]
        ],
        "coefficient": coef_grp["coefficient"][:].tolist(),
    }
    if "predictor_counted_allele" in coef_grp:
        coef_cols["predictor_chromosome"] = [
            v.decode("utf-8") for v in coef_grp["predictor_chromosome"][:]
        ]
        coef_cols["predictor_position"] = coef_grp["predictor_position"][:].tolist()
        coef_cols["predictor_counted_allele"] = [
            v.decode("utf-8") for v in coef_grp["predictor_counted_allele"][:]
        ]
        coef_cols["predictor_other_allele"] = [
            v.decode("utf-8") for v in coef_grp["predictor_other_allele"][:]
        ]
        coef_cols["predictor_allele_frequency"] = coef_grp[
            "predictor_allele_frequency"
        ][:].tolist()
    return coef_cols


def _load_imputed_models_hdf5(f: h5py.File) -> List[ImputedVariantModel]:
    """Load imputed variant models from HDF5 file.

    Reads the sparse ``coefficients`` group, including the per-predictor allele
    metadata datasets when present (schema v2); v1 files without them reconstruct
    with empty predictor metadata.
    """
    return _rebuild_imputed_models(
        _hdf5_model_cols(f["imputed_variants"]),
        _hdf5_coef_cols(f["coefficients"]),
    )


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
        # Attach observed fallbacks (P1.8) when present (optional group).
        if "observed_fallbacks" in f:
            _attach_observed_fallbacks(
                observed_variants,
                _hdf5_model_cols(f["observed_fallbacks"]),
                _hdf5_coef_cols(f["coefficients"]),
            )
        calibration_params = _load_calibration_params_hdf5(f)
        evaluation_metrics = _load_evaluation_metrics_hdf5(f)
        metadata = _load_metadata_hdf5(f)

    return observed_variants, imputed_models, calibration_params, evaluation_metrics, metadata


# ---------------------------------------------------------------------------
# Arrow / Parquet loaders
# ---------------------------------------------------------------------------

ModelComponents = Tuple[
    List[VariantInfo],
    List[ImputedVariantModel],
    Optional[CalibrationParams],
    Optional[EvaluationMetrics],
    Dict[str, Any],
]


def _columns_dict(table: pa.Table) -> Dict[str, List[Any]]:
    """Materialize a pyarrow Table as a plain {column: list} dict."""
    return {name: table.column(name).to_pylist() for name in table.column_names}


def _assemble_from_tables(tables: Dict[str, pa.Table]) -> ModelComponents:
    """Reconstruct model components from the four Arrow/Parquet tables.

    Shared by the Arrow IPC and Parquet readers (identical table layout).
    """
    observed = _rebuild_observed_variants(_columns_dict(tables["observed_variants"]))
    coef_cols = _columns_dict(tables["coefficients"])
    imputed = _rebuild_imputed_models(
        _columns_dict(tables["imputed_variants"]),
        coef_cols,
    )
    # Attach observed fallbacks (P1.8) when the optional table is present.
    if "observed_fallbacks" in tables:
        _attach_observed_fallbacks(
            observed,
            _columns_dict(tables["observed_fallbacks"]),
            coef_cols,
        )

    meta_cols = _columns_dict(tables["metadata"])
    metadata: Dict[str, Any] = {
        name: (values[0] if values else None) for name, values in meta_cols.items()
    }

    calibration_params = None
    cp_json = metadata.pop("calibration_params_json", None)
    if cp_json:
        calibration_params = CalibrationParams(**json.loads(cp_json))

    evaluation_metrics = None
    em_json = metadata.pop("evaluation_metrics_json", None)
    if em_json:
        evaluation_metrics = EvaluationMetrics(**json.loads(em_json))

    ts_json = metadata.pop("training_summary_json", None)
    if ts_json:
        metadata["training_summary"] = json.loads(ts_json)

    return observed, imputed, calibration_params, evaluation_metrics, metadata


def load_model_arrow(path: Union[str, Path]) -> ModelComponents:
    """Load a trained imputation model from Arrow IPC format (schema v2).

    The Arrow file is a container table whose rows are the serialized
    ``metadata``/``observed_variants``/``imputed_variants``/``coefficients`` tables.

    Args:
        path: Path to the ``.arrow`` model file.

    Returns:
        Same 5-tuple as :func:`load_model_hdf5`.
    """
    path = Path(path)
    reader = pa.ipc.open_file(str(path))
    container = reader.read_all()

    names = container.column("table_name").to_pylist()
    blobs = container.column("data").to_pylist()
    tables: Dict[str, pa.Table] = {}
    for name, blob in zip(names, blobs):
        stream = pa.ipc.open_stream(pa.BufferReader(blob))
        tables[name] = stream.read_all()

    required = {"metadata", "observed_variants", "imputed_variants", "coefficients"}
    missing = required - set(tables)
    if missing:
        raise KeyError(f"Missing required tables in Arrow file: {sorted(missing)}")

    return _assemble_from_tables(tables)


def load_model_parquet(path: Union[str, Path]) -> ModelComponents:
    """Load a trained imputation model from a Parquet directory (schema v2).

    Args:
        path: Path to the export directory containing ``metadata.parquet``,
            ``observed_variants.parquet``, ``imputed_variants.parquet`` and
            ``coefficients.parquet``.

    Returns:
        Same 5-tuple as :func:`load_model_hdf5`.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(
            f"Parquet model path must be the export directory, got: {path}"
        )

    tables: Dict[str, pa.Table] = {}
    for name in ("metadata", "observed_variants", "imputed_variants", "coefficients"):
        file_path = path / f"{name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing Parquet table: {file_path}")
        tables[name] = pq.read_table(file_path)

    # Optional observed-fallback table (P1.8); absent in pre-P1.8 exports.
    fallback_path = path / "observed_fallbacks.parquet"
    if fallback_path.exists():
        tables["observed_fallbacks"] = pq.read_table(fallback_path)

    return _assemble_from_tables(tables)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_model_csv(path: Union[str, Path]) -> ModelComponents:
    """Load a trained imputation model from the CSV pair (schema v2).

    Reads the per-variant table at ``path`` (split into observed vs imputed rows by
    the ``status`` column) and its companion ``<name>_coefficients.csv`` for the
    per-predictor allele metadata. The flat CSV format carries no
    calibration/provenance, so those are returned as ``None`` / an empty metadata
    dict; a CSV-loaded model scores the (oriented) raw PRS only.

    Args:
        path: Path to the per-variant ``*_variants.csv`` file.

    Returns:
        Same 5-tuple as :func:`load_model_hdf5` (calibration/metrics are ``None``).
    """
    path = Path(path)
    variants = pd.read_csv(path, dtype=str)

    obs_df = variants[variants["status"] == "observed"]
    imp_df = variants[variants["status"].isin(["imputed", "intercept_only"])]
    fb_df = variants[
        variants["status"].isin(
            ["observed_fallback", "observed_fallback_intercept_only"]
        )
    ]

    observed = _rebuild_observed_variants(
        {col: obs_df[col].tolist() for col in obs_df.columns}
    )

    imp_cols: Dict[str, List[Any]] = {
        col: imp_df[col].tolist() for col in imp_df.columns
    }
    # is_intercept_only is encoded by `status` in the flat table, not a column.
    imp_cols["is_intercept_only"] = [
        s == "intercept_only" for s in imp_df["status"].tolist()
    ]

    coef_path = coefficients_path_for(path)
    if coef_path.exists():
        coef_df = pd.read_csv(coef_path, dtype=str)
        coef_cols: Dict[str, List[Any]] = {
            col: coef_df[col].tolist() for col in coef_df.columns
        }
    else:
        coef_cols = {col: [] for col in COEFFICIENTS_COLUMNS}

    imputed = _rebuild_imputed_models(imp_cols, coef_cols)

    # Attach observed fallbacks (P1.8) from their dedicated status rows; their
    # predictors share the companion coefficients table (keyed by target id).
    if not fb_df.empty:
        fb_cols: Dict[str, List[Any]] = {
            col: fb_df[col].tolist() for col in fb_df.columns
        }
        fb_cols["is_intercept_only"] = [
            s == "observed_fallback_intercept_only"
            for s in fb_df["status"].tolist()
        ]
        _attach_observed_fallbacks(observed, fb_cols, coef_cols)

    return observed, imputed, None, None, {}
