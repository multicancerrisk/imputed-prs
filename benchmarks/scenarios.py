"""Scenario registry — the units of work the isolated child executes.

Two families:

* ``_selftest_*`` — hermetic ops (no ``imputed_prs``, no data) that let the harness be
  validated fast and deterministically: allocate a known amount, raise, grow until the
  watchdog fires, self-SIGKILL, and a numpy-attribution probe.
* ``load_genotypes`` / ``fit`` / ``predict`` — drive the real library via its public API.

Also exposes :func:`predict_bytes`, the analytical byte model used by the scaling
projection as a cross-check (and as the attribution fallback if tracemalloc's numpy
domain is ever unavailable).
"""
from __future__ import annotations

import os
import signal
import time
from typing import Any, Callable, Dict, List, Optional

from benchmarks.meters import PhaseRegistry, phase

_MIB = 1024 * 1024

# Objects deliberately kept alive until the process exits, so tracemalloc's *live*
# snapshot (taken by the child after the scenario returns) attributes their resident
# bytes to the source line that allocated them. Peak RSS is a high-water mark and does
# not need this; per-site attribution does. Mirrors how the library retains
# ``dosage_matrix`` / the fitted model.
_RETAINED: List[Any] = []


# --------------------------------------------------------------------------------------
# Hermetic self-test ops
# --------------------------------------------------------------------------------------
def _op_selftest_alloc(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    """Allocate ``alloc_mib`` MiB (pages touched) so peak RSS is measurable."""
    import numpy as np

    mib = int(spec.params.get("alloc_mib", 0))
    dwell = float(spec.params.get("dwell_s", 0.0))
    held: List[Any] = []
    with phase("alloc", registry, trace=spec.tracemalloc):
        if mib > 0:
            arr = np.ones(mib * _MIB // 8, dtype=np.float64)
            arr[::512] = 2.0  # touch enough pages to force resident memory
            held.append(arr)
    if dwell > 0:
        time.sleep(dwell)
    return {"alloc_mib": mib, "held_bytes": sum(a.nbytes for a in held)}


def _op_selftest_raise(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    raise ValueError(str(spec.params.get("message", "selftest failure")))


def _op_selftest_grow(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    """Grow memory in chunks until the watchdog aborts the process."""
    import numpy as np

    chunk_mib = int(spec.params.get("chunk_mib", 32))
    held: List[Any] = []
    for _ in range(1_000_000):
        a = np.ones(chunk_mib * _MIB // 8, dtype=np.float64)
        a[::512] = 1.0
        held.append(a)
        time.sleep(0.005)
    return {}  # unreachable in practice


def _op_selftest_sigkill(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    os.kill(os.getpid(), signal.SIGKILL)
    return {}  # unreachable


def _op_selftest_numpy_attrib(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    """Mimic the library's ``np.column_stack(cols).astype(np.float32)`` allocation so the
    attribution test has a known caller line to assert against."""
    import numpy as np

    n = int(spec.params.get("rows", 200_000))
    k = int(spec.params.get("cols", 8))
    cols = [np.random.default_rng(i).standard_normal(n) for i in range(k)]
    with phase("stack", registry, trace=spec.tracemalloc):
        mat = np.column_stack(cols).astype(np.float32)  # <-- attribution target line
    _RETAINED.append(mat)  # keep live for the post-return attribution snapshot
    return {"mat_bytes": int(mat.nbytes), "shape": list(mat.shape)}


# --------------------------------------------------------------------------------------
# Real library ops
# --------------------------------------------------------------------------------------
def _op_load_genotypes(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    from imputed_prs.io import load_genotypes

    p = spec.params
    variant_ids = set(p["variant_ids"]) if p.get("variant_ids") else None
    with phase("load", registry, trace=spec.tracemalloc):
        gd = load_genotypes(path=p["path"], variant_ids=variant_ids, samples=p.get("samples"))
    _RETAINED.append(gd)  # keep dosage_matrix live for attribution
    return {
        "n_samples": int(gd.n_samples),
        "n_variants": int(gd.n_variants),
        "dosage_matrix_bytes": int(gd.dosage_matrix.nbytes),
    }


def _make_model(method: str, config: Dict[str, Any]):
    from imputed_prs import LinearImputationPRS, LinearProjectionPRS

    common = dict(
        window_size=config.get("window_size", 1_000_000),
        l1_ratio=config.get("l1_ratio", 0.5),
        alpha=config.get("alpha", 0.01),
        cv_folds=config.get("cv_folds", 5),
        n_jobs=config.get("n_jobs", 1),
        random_state=config.get("random_state", 42),
        max_predictors=config.get("max_predictors"),
        tuning_scope=config.get("tuning_scope", "none"),
        verbose=config.get("verbose", 1),
    )
    if method == "projection":
        common["max_tuning_regions"] = config.get("max_tuning_regions", 50)
        return LinearProjectionPRS(**common)
    common["max_tuning_variants"] = config.get("max_tuning_variants", 50)
    return LinearImputationPRS(**common)


def _fit_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "reference_genotypes": params["reference_genotypes"],
        "prs_definition": params["prs_definition"],
        "genome_build": params.get("genome_build"),
    }
    if params.get("platform_variants_file"):
        with open(params["platform_variants_file"]) as fh:
            kwargs["platform_variants"] = [ln.strip() for ln in fh if ln.strip()]
    elif params.get("platform_variants"):
        kwargs["platform_variants"] = list(params["platform_variants"])
    elif params.get("platform_manifest"):
        kwargs["platform_manifest"] = params["platform_manifest"]
    elif params.get("platform_name"):
        kwargs["platform_name"] = params["platform_name"]
    for key in ("prs_id", "model_name", "reference_panel_id", "training_ancestry"):
        if params.get(key) is not None:
            kwargs[key] = params[key]
    return kwargs


def _op_fit(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    from benchmarks.oracle import extract_oracle

    method = spec.params.get("method", "imputation")
    model = _make_model(method, spec.config)
    _RETAINED.append(model)  # keep the fitted model (and its cv_predictions) live for attribution
    with phase("fit", registry, trace=spec.tracemalloc):
        model.fit(**_fit_kwargs(spec.params))
    probe = None
    if spec.params.get("predict_probe"):
        probe = _predict_probe(model, spec.params["predict_probe"])
    payload = extract_oracle(model, method, probe=probe)
    if spec.params.get("export_dir"):
        with phase("export", registry, trace=spec.tracemalloc):
            try:
                paths = model.export(spec.params["export_dir"])
                payload["export_ok"] = True
                payload["export_paths"] = {k: str(v) for k, v in dict(paths).items()}
            except Exception as exc:  # never lose the oracle over an export hiccup
                payload["export_ok"] = False
                payload["export_error"] = str(exc)
    return payload


def _predict_probe(model, probe_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Score one deterministic user via the public ``predict`` for a parity probe."""
    import pandas as pd

    df = pd.DataFrame(probe_spec["genotypes"])  # {variant_id: [...], genotype: [...]}
    result = model.predict(df, apply_calibration=bool(probe_spec.get("apply_calibration", True)))
    return {
        "prs": float(getattr(result, "prs", float("nan"))),
        "prs_scaled": _maybe_float(getattr(result, "prs_scaled", None)),
        "se": _maybe_float(getattr(result, "se", None)),
        "n_variants_used": _maybe_int(getattr(result, "n_variants_used", None)),
    }


def _op_predict(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

    model = LinearImputationPRS.load(spec.params["model_path"])
    with phase("predict", registry, trace=spec.tracemalloc):
        result = model.predict(spec.params["user_genotypes"])
    return {"prs": _maybe_float(getattr(result, "prs", None))}


def _maybe_float(v):
    return None if v is None else float(v)


def _maybe_int(v):
    return None if v is None else int(v)


# --------------------------------------------------------------------------------------
# Dispatch + analytical model
# --------------------------------------------------------------------------------------
SCENARIOS: Dict[str, Callable[[Any, PhaseRegistry], Dict[str, Any]]] = {
    "_selftest_alloc": _op_selftest_alloc,
    "_selftest_raise": _op_selftest_raise,
    "_selftest_grow": _op_selftest_grow,
    "_selftest_sigkill": _op_selftest_sigkill,
    "_selftest_numpy_attrib": _op_selftest_numpy_attrib,
    "load_genotypes": _op_load_genotypes,
    "fit": _op_fit,
    "predict": _op_predict,
}


def run_scenario(spec, registry: PhaseRegistry) -> Dict[str, Any]:
    fn = SCENARIOS.get(spec.operation)
    if fn is None:
        raise KeyError(f"Unknown operation: {spec.operation!r}")
    return fn(spec, registry)


def predict_bytes(
    operation: str,
    n_samples: int,
    n_variants: int,
    *,
    n_missing: Optional[int] = None,
    itemsize: int = 4,
    concurrent_copies: int = 4,
) -> Dict[str, int]:
    """Analytical peak-byte model — the cross-check for the empirical scaling fit.

    * dense dosage matrix (float32): ``n_samples * n_variants * itemsize``
    * fit holds several concurrent dense copies (dosage_matrix, Z, X, X_full)
    * imputation ``cv_predictions`` is one float64 array per missing variant:
      ``n_missing * n_samples * 8``
    """
    dense_one = int(n_samples) * int(n_variants) * int(itemsize)
    if n_missing is None:
        n_missing = n_variants
    if operation == "load_genotypes":
        # the loader stacks a python list then transposes -> ~2x transiently
        return {"dense_dosage_matrix": dense_one, "transient_vstack": dense_one,
                "total_est": 2 * dense_one}
    if operation == "fit":
        cv = int(n_missing) * int(n_samples) * 8
        dense = concurrent_copies * dense_one
        return {"dense_copies": dense, "cv_predictions": cv, "total_est": dense + cv}
    return {"total_est": dense_one}
