"""Fit peak-RSS and wall-time as functions of (n_samples, n_variants) from measured
points and extrapolate to the production scale (500,000 samples × ~2,000,000 variants).

Models (all via ``numpy.linalg.lstsq`` — no new dependencies):

1. **Combined mechanistic multilinear** ``y ≈ b0 + b_v·v + b_sv·(s·v)`` — the headline.
   The dense float32 copies (4 B) and the imputation ``cv_predictions`` array (8 B, one
   per missing variant ≈ ``v``) are both ∝ ``s·v``, so they collapse into one coefficient
   ``b_sv`` (effective bytes per sample per variant). ``b_v`` absorbs the O(v) Python /
   pandas overhead; ``b0`` is the interpreter + libraries baseline.
2. **Power law** ``y = A·(s·v)^k`` (fit in log space) — an independent confirmation;
   ``k ≈ 1`` for memory.
3. **Per-axis linear** slices for interpretability.

An **analytical byte model** is computed independently as an honesty anchor: one dense
float32 copy at 500K×2M is exactly 4 TB. Every projection carries the caveat that the
sample axis is measured only to a few thousand samples, so 500K is a large extrapolation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.harness import DEFAULT_RESULTS_DIR, MeasurementResult, dumps, load_results

TARGET_SAMPLES = 500_000
TARGET_VARIANTS = 2_000_000
# Decimal (SI) byte units for display, so the analytical anchor reads as the plan states
# it (one dense float32 copy = 4 * 5e5 * 2e6 = 4.0 TB). Raw byte values in the JSON are
# exact and unit-free.
_TB = 1e12
_GB = 1e9
_MB = 1e6


# --------------------------------------------------------------------------------------
# Pure fitting primitives (unit-tested directly)
# --------------------------------------------------------------------------------------
def _lstsq(design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares fit; returns (coefficients, R²)."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def fit_combined(s: Sequence[float], v: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Fit ``y ≈ b0 + b_v·v + b_sv·(s·v)``."""
    s = np.asarray(s, float)
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    design = np.column_stack([np.ones_like(s), v, s * v])
    coef, r2 = _lstsq(design, y)
    n_params = design.shape[1]
    # Collinear inputs (e.g. all points the same size) make the fit meaningless even with
    # many rows, so require enough *distinct* (s, v) combinations, not just enough rows.
    distinct = len({(round(a, 6), round(b, 6)) for a, b in zip(s, v)})
    confident = len(y) >= n_params + 2 and distinct >= n_params and r2 == r2
    return {
        "model": "b0 + b_v*v + b_sv*(s*v)",
        "coef": {"b0": float(coef[0]), "b_v": float(coef[1]), "b_sv": float(coef[2])},
        "r2": r2,
        "n_points": int(len(y)),
        "n_distinct_sizes": distinct,
        "confidence": "ok" if confident else "low",
    }


def predict_combined(fit: Dict[str, Any], s: float, v: float) -> float:
    c = fit["coef"]
    return c["b0"] + c["b_v"] * v + c["b_sv"] * (s * v)


def fit_powerlaw(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Fit ``y = A·x^k`` in log space (positive points only)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return {"model": "A*x^k", "A": float("nan"), "k": float("nan"), "r2_log": float("nan"),
                "n_points": int(mask.sum()), "confidence": "low"}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    coef, r2 = _lstsq(np.column_stack([np.ones_like(lx), lx]), ly)
    return {
        "model": "A*x^k",
        "A": float(math.exp(coef[0])),
        "k": float(coef[1]),
        "r2_log": r2,
        "n_points": int(mask.sum()),
        "confidence": "ok" if mask.sum() >= 4 else "low",
    }


def fit_linear_1d(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Fit ``y = slope·x + intercept``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n_points": int(len(x)), "confidence": "low"}
    coef, r2 = _lstsq(np.column_stack([x, np.ones_like(x)]), y)
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2,
            "n_points": int(len(x)), "confidence": "ok" if len(x) >= 3 else "low"}


# --------------------------------------------------------------------------------------
# Analytical byte model (honesty anchor)
# --------------------------------------------------------------------------------------
def analytical_memory_bytes(
    s: float,
    v: float,
    *,
    itemsize: int = 4,
    concurrent_copies: int = 4,
    include_cv_predictions: bool = True,
) -> Dict[str, float]:
    dense_one = float(s) * float(v) * itemsize
    cv = float(v) * float(s) * 8 if include_cv_predictions else 0.0
    return {
        "dense_one_copy": dense_one,
        "dense_concurrent_copies": concurrent_copies * dense_one,
        "cv_predictions": cv,
        "total": concurrent_copies * dense_one + cv,
    }


# --------------------------------------------------------------------------------------
# Point extraction from measurement results
# --------------------------------------------------------------------------------------
@dataclass
class Points:
    s: List[float]
    v: List[float]
    y: List[float]
    labels: List[str]

    def __len__(self) -> int:
        return len(self.y)


def build_points(results: List[MeasurementResult], y_field: str, *, operation: Optional[str] = None) -> Points:
    """Extract (n_samples, n_variants, y) from completed runs.

    ``y_field`` is ``"peak_rss_bytes"`` or ``"wall_seconds"``. ``n_variants`` prefers the
    actually-loaded count in ``result["n_variants"]`` and falls back to ``spec.n_variants``.
    """
    s: List[float] = []
    v: List[float] = []
    y: List[float] = []
    labels: List[str] = []
    for r in results:
        if r.outcome != "completed":
            continue
        if operation is not None and r.spec.operation != operation:
            continue
        yi = getattr(r, y_field, None)
        n_s = r.spec.n_samples
        n_v = (r.result or {}).get("n_variants") if r.result else None
        if n_v is None:
            n_v = r.spec.n_variants
        if yi is None or n_s is None or n_v is None:
            continue
        # Peak RSS is only trustworthy when tracemalloc was off
        if y_field == "peak_rss_bytes" and not r.peak_rss_is_authoritative:
            continue
        s.append(float(n_s))
        v.append(float(n_v))
        y.append(float(yi))
        labels.append(r.spec.label)
    return Points(s, v, y, labels)


# --------------------------------------------------------------------------------------
# Projection report
# --------------------------------------------------------------------------------------
def project_response(
    points: Points,
    *,
    target_samples: int,
    target_variants: int,
    is_memory: bool,
) -> Dict[str, Any]:
    if len(points) == 0:
        return {"error": "no usable measured points"}
    combined = fit_combined(points.s, points.v, points.y)
    prod = [si * vi for si, vi in zip(points.s, points.v)]
    power = fit_powerlaw(prod, points.y)

    # per-axis slices: fix the most common value on the other axis
    s_arr = np.asarray(points.s)
    v_arr = np.asarray(points.v)
    per_axis: Dict[str, Any] = {}
    if len(set(points.s)) > 1:
        v_star = _mode(v_arr)
        m = np.isclose(v_arr, v_star)
        if m.sum() >= 2:
            per_axis["vs_samples_at_fixed_v"] = {
                "fixed_v": float(v_star),
                **fit_linear_1d(s_arr[m], np.asarray(points.y)[m]),
            }
    if len(set(points.v)) > 1:
        s_star = _mode(s_arr)
        m = np.isclose(s_arr, s_star)
        if m.sum() >= 2:
            per_axis["vs_variants_at_fixed_s"] = {
                "fixed_s": float(s_star),
                **fit_linear_1d(v_arr[m], np.asarray(points.y)[m]),
            }

    extrap = float(predict_combined(combined, target_samples, target_variants))
    extrap_power = (
        power["A"] * (target_samples * target_variants) ** power["k"]
        if power["confidence"] != "low"
        else float("nan")
    )

    out: Dict[str, Any] = {
        "combined": combined,
        "power_law": power,
        "per_axis": per_axis,
        "extrapolation": {"combined": extrap, "power_law": extrap_power},
        "measured_range": {
            "n_samples": [min(points.s), max(points.s)],
            "n_variants": [min(points.v), max(points.v)],
        },
    }
    if is_memory:
        analytical = analytical_memory_bytes(target_samples, target_variants)
        out["analytical"] = analytical
        b_sv = combined["coef"]["b_sv"]
        out["effective_concurrent_float32_copies"] = b_sv / 4.0 if b_sv == b_sv else float("nan")
    return out


def make_report(
    results: List[MeasurementResult],
    *,
    target_samples: int = TARGET_SAMPLES,
    target_variants: int = TARGET_VARIANTS,
) -> Dict[str, Any]:
    # The headline memory/time models use the real fit workload; load-only cells (dense
    # matrix only) are modeled separately so the two footprints are not conflated.
    fit_results = [r for r in results if r.spec.operation == "fit"]
    load_results_ = [r for r in results if r.spec.operation == "load_genotypes"]
    report: Dict[str, Any] = {
        "targets": {"n_samples": target_samples, "n_variants": target_variants},
        "n_results": len(results),
        "memory": project_response(
            build_points(fit_results, "peak_rss_bytes"),
            target_samples=target_samples, target_variants=target_variants, is_memory=True,
        ),
        "time": project_response(
            build_points(fit_results, "wall_seconds"),
            target_samples=target_samples, target_variants=target_variants, is_memory=False,
        ),
    }
    if load_results_:
        report["dense_isolation"] = project_response(
            build_points(load_results_, "peak_rss_bytes"),
            target_samples=target_samples, target_variants=target_variants, is_memory=True,
        )
    report["caveats"] = [
        "Sample axis measured only up to a few thousand samples; 500K is a large "
        "extrapolation (~150x on the sample axis).",
        "Peak RSS points with tracemalloc enabled are excluded (inflated).",
        "Wall-time super-linearity from O(n_variants) pandas hotspots is only "
        "partially captured by a linear-in-(s*v) model.",
    ]
    return report


def _mode(arr: np.ndarray) -> float:
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[int(np.argmax(counts))])


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------
def _human_bytes(n: float) -> str:
    if n != n:
        return "nan"
    for unit, size in (("TB", _TB), ("GB", _GB), ("MB", _MB)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f} B"


def _human_time(sec: float) -> str:
    if sec != sec:
        return "nan"
    if sec >= 86400:
        return f"{sec / 86400:.1f} days"
    if sec >= 3600:
        return f"{sec / 3600:.1f} h"
    if sec >= 60:
        return f"{sec / 60:.1f} min"
    return f"{sec:.1f} s"


def format_report(report: Dict[str, Any]) -> str:
    t = report["targets"]
    lines = [
        "=" * 72,
        f"Scaling projection to {t['n_samples']:,} samples x {t['n_variants']:,} variants",
        f"(from {report['n_results']} measured runs)",
        "=" * 72,
    ]
    mem = report["memory"]
    if "error" in mem:
        lines.append(f"MEMORY: {mem['error']}")
    else:
        c = mem["combined"]
        lines += [
            "",
            "PEAK MEMORY",
            f"  model: y = {c['coef']['b0']:.3g} + {c['coef']['b_v']:.3g}*v + "
            f"{c['coef']['b_sv']:.4g}*(s*v)   R^2={c['r2']:.5f}  [{c['confidence']}, n={c['n_points']}]",
            f"  power law: k={mem['power_law']['k']:.3f} (R^2_log={mem['power_law']['r2_log']:.4f})",
            f"  -> projected peak RSS: {_human_bytes(mem['extrapolation']['combined'])}",
            f"  analytical: 1 dense float32 copy = {_human_bytes(mem['analytical']['dense_one_copy'])}; "
            f"~{mem['analytical']['dense_concurrent_copies'] / _TB:.1f} TB across copies; "
            f"cv_predictions = {_human_bytes(mem['analytical']['cv_predictions'])}; "
            f"total ~ {_human_bytes(mem['analytical']['total'])}",
            f"  effective concurrent float32 copies (b_sv/4): "
            f"{mem.get('effective_concurrent_float32_copies', float('nan')):.2f}",
        ]
    tm = report["time"]
    if "error" not in tm:
        c = tm["combined"]
        lines += [
            "",
            "WALL TIME",
            f"  model R^2={c['r2']:.5f} [{c['confidence']}, n={c['n_points']}]  "
            f"power-law k={tm['power_law']['k']:.3f}",
            f"  -> projected wall time: {_human_time(tm['extrapolation']['combined'])}",
        ]
    dense = report.get("dense_isolation")
    if dense and "error" not in dense:
        c = dense["combined"]
        lines += [
            "",
            "DENSE MATRIX (load-only isolation)",
            f"  model R^2={c['r2']:.5f} [{c['confidence']}, n={c['n_points']}, "
            f"distinct sizes={c.get('n_distinct_sizes')}]",
            f"  -> projected dense load peak: {_human_bytes(dense['extrapolation']['combined'])}",
        ]
    lines += ["", "CAVEATS"]
    lines += [f"  - {c}" for c in report["caveats"]]
    lines.append("=" * 72)
    return "\n".join(lines)


def maybe_plot(report: Dict[str, Any], results: List[MeasurementResult], out_dir: Path) -> Optional[List[Path]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fit_results = [r for r in results if r.spec.operation in ("fit", "load_genotypes")]
    saved: List[Path] = []
    for y_field, fname, human in (
        ("peak_rss_bytes", "projection_rss.png", "peak RSS (bytes)"),
        ("wall_seconds", "projection_time.png", "wall time (s)"),
    ):
        pts = build_points(fit_results, y_field)
        if len(pts) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        prod = np.asarray(pts.s) * np.asarray(pts.v)
        ax.scatter(prod, pts.y, label="measured")
        ax.set_xlabel("n_samples * n_variants")
        ax.set_ylabel(human)
        ax.set_title(f"{human} vs problem size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        path = out_dir / fname
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extrapolate imputed-prs time/memory to 500K x 2M.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--target-samples", type=int, default=TARGET_SAMPLES)
    ap.add_argument("--target-variants", type=int, default=TARGET_VARIANTS)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    results = load_results(args.results_dir)
    report = make_report(
        results, target_samples=args.target_samples, target_variants=args.target_variants
    )
    print(format_report(report))
    out_json = args.out_json or (args.results_dir / "projection.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(dumps(report))
    print(f"\nwrote {out_json}")
    if args.plot:
        saved = maybe_plot(report, results, args.results_dir)
        if saved:
            print("plots:", ", ".join(str(p) for p in saved))
        else:
            print("(matplotlib unavailable; skipped plots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
