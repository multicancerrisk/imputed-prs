"""Hermetic tests for ``benchmarks/scaling_projection.py``.

No I/O, no measured data — synthesize points from a *known* model and assert the fitter
recovers it and that extrapolation matches the closed form. This is the provable core of
Phase 0's extrapolation claim.
"""
from __future__ import annotations

import math

from benchmarks.harness import MeasurementResult, WorkSpec
from benchmarks.scaling_projection import (
    analytical_memory_bytes,
    build_points,
    fit_combined,
    fit_powerlaw,
    make_report,
    predict_combined,
)


def _grid(s_vals, v_vals, fn):
    s, v, y = [], [], []
    for sv in s_vals:
        for vv in v_vals:
            s.append(sv)
            v.append(vv)
            y.append(fn(sv, vv))
    return s, v, y


def test_fit_combined_recovers_coefficients_well_conditioned():
    # small magnitudes => well-conditioned => tight coefficient recovery
    b0, b_v, b_sv = 2.0, 3.0, 5.0
    s, v, y = _grid([1, 2, 3, 4], [1, 2, 3, 4, 5], lambda sv, vv: b0 + b_v * vv + b_sv * (sv * vv))
    fit = fit_combined(s, v, y)
    assert fit["r2"] > 1 - 1e-12
    assert fit["confidence"] == "ok"
    assert math.isclose(fit["coef"]["b0"], b0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(fit["coef"]["b_v"], b_v, rel_tol=1e-9)
    assert math.isclose(fit["coef"]["b_sv"], b_sv, rel_tol=1e-9)


def test_extrapolation_matches_closed_form_realistic_scale():
    # realistic byte-scale coefficients; extrapolation dominated by the b_sv term
    b0, b_v, b_sv = 1e9, 500.0, 4.0
    s, v, y = _grid([500, 1000, 2000, 3202], [2e4, 6e4, 1.2e5, 3e5],
                    lambda sv, vv: b0 + b_v * vv + b_sv * (sv * vv))
    fit = fit_combined(s, v, y)
    assert fit["r2"] > 1 - 1e-9
    pred = predict_combined(fit, 500_000, 2_000_000)
    exact = b0 + b_v * 2_000_000 + b_sv * (500_000 * 2_000_000)
    assert math.isclose(pred, exact, rel_tol=1e-6)


def test_powerlaw_recovers_exponent():
    x = [1, 2, 4, 8, 16, 32, 64]
    lin = fit_powerlaw(x, [4.0 * xi for xi in x])
    assert math.isclose(lin["k"], 1.0, rel_tol=1e-6)
    assert math.isclose(lin["A"], 4.0, rel_tol=1e-6)
    quad = fit_powerlaw(x, [3.0 * xi ** 2 for xi in x])
    assert math.isclose(quad["k"], 2.0, rel_tol=1e-6)


def test_degenerate_input_flagged_low_confidence():
    fit = fit_combined([1, 2], [1, 2], [10.0, 20.0])  # 2 points, 3 params
    assert fit["confidence"] == "low"


def test_analytical_anchor_reproduces_4TB_dense_copy():
    a = analytical_memory_bytes(500_000, 2_000_000)
    # one dense float32 copy: 4 * 5e5 * 2e6 bytes
    assert a["dense_one_copy"] == 4e12
    # imputation cv_predictions (float64, one per missing variant ~= n_variants)
    assert a["cv_predictions"] == 8e12


def _mk(s, v, peak, wall):
    return MeasurementResult(
        spec=WorkSpec(operation="fit", label=f"{s}x{v}", n_samples=s, n_variants=v),
        outcome="completed",
        run_id=f"r{s}_{v}",
        peak_rss_bytes=peak,
        wall_seconds=wall,
        peak_rss_is_authoritative=True,
        result={"n_variants": v},
    )


def test_make_report_end_to_end_on_synthetic_results():
    # peak = 1e9 + 4*(s*v); wall = 1e-6*(s*v)
    results = [
        _mk(s, v, 1e9 + 4 * s * v, 1e-6 * s * v)
        for s in (500, 1000, 2000, 3202)
        for v in (20_000, 60_000, 120_000)
    ]
    report = make_report(results)
    mem = report["memory"]
    assert mem["combined"]["r2"] > 1 - 1e-9
    # b_sv should recover ~4 bytes/sample/variant
    assert math.isclose(mem["combined"]["coef"]["b_sv"], 4.0, rel_tol=1e-3)
    assert mem["extrapolation"]["combined"] > 1e12  # multi-TB at 500K x 2M
    assert mem["analytical"]["dense_one_copy"] == 4e12
    assert report["time"]["combined"]["r2"] > 1 - 1e-9


def test_build_points_excludes_non_authoritative_peak():
    good = _mk(500, 20_000, 1e8, 1.0)
    bad = _mk(1000, 20_000, 1e8, 1.0)
    bad.peak_rss_is_authoritative = False
    pts = build_points([good, bad], "peak_rss_bytes")
    assert len(pts) == 1
    # wall_seconds is unaffected by the RSS-authoritative filter
    assert len(build_points([good, bad], "wall_seconds")) == 2
