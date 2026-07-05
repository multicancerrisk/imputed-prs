"""Statistical-parity **oracle**: a compact, deterministic snapshot of a fitted model's
counts, per-unit R² distribution, calibration parameters, and a prediction probe.

This is the contract every later phase must preserve: the new scalable training path is
required to reproduce these numbers within tolerance on the same data (the golden
allele-orientation / export round-trip tests remain the exact ``atol=1e-12`` gate;
this oracle is the R²/calibration parity gate). :func:`compare_oracle` is the reusable
checker.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

ORACLE_SCHEMA_VERSION = 1


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    s = sorted(vals)
    mid = n // 2
    median = s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": math.sqrt(var),
        "min": s[0],
        "max": s[-1],
    }


def extract_oracle(model, method: str, probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the oracle dict from a fitted ``LinearImputationPRS``/``LinearProjectionPRS``.

    Uses only public accessors: ``.summary``, ``.calibration_params``, and
    ``.imputed_models``/``.region_models``.
    """
    summary = dict(model.summary)  # counts + provenance
    cal = model.calibration_params
    if method == "projection":
        units = model.region_models
        r2_values = [u.cv_r2 for u in units if not getattr(u, "is_intercept_only", False)]
    else:
        units = model.imputed_models
        r2_values = [u.imputation_r2 for u in units if not getattr(u, "is_intercept_only", False)]

    return {
        "oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "method": method,
        "summary": summary,
        "calibration": (dataclasses.asdict(cal) if cal is not None else None),
        "r2_summary": _stats(r2_values),
        "n_units": len(units),
        "predict_probe": probe,
    }


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def compare_oracle(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, Tuple[Any, Any, bool]]:
    """Field-by-field comparison of two oracle dicts.

    Numeric leaves compare with ``math.isclose(rel_tol=rtol, abs_tol=atol)``; everything
    else compares by equality. Returns ``{dotted_field: (base, cand, ok)}`` — a later
    phase asserts every ``ok`` is True (or filters to the ``False`` ones to see drift).
    """
    fb: Dict[str, Any] = {}
    fc: Dict[str, Any] = {}
    _flatten("", base, fb)
    _flatten("", cand, fc)
    diff: Dict[str, Tuple[Any, Any, bool]] = {}
    for key in sorted(set(fb) | set(fc)):
        a = fb.get(key)
        b = fc.get(key)
        if isinstance(a, bool) or isinstance(b, bool):
            ok = a == b
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            af, bf = float(a), float(b)
            if math.isnan(af) and math.isnan(bf):
                ok = True
            else:
                ok = math.isclose(af, bf, rel_tol=rtol, abs_tol=atol)
        else:
            ok = a == b
        diff[key] = (a, b, ok)
    return diff


def oracle_matches(base: Dict[str, Any], cand: Dict[str, Any], **kw) -> bool:
    return all(ok for _, _, ok in compare_oracle(base, cand, **kw).values())
