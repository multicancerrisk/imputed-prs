"""Phase-11 full-scale integration validation (chromosome-22 scoped).

Composes the Phase-2/4/6 building blocks into ONE consolidated end-to-end run on the real
GRCh38 1000G high-coverage panel (3,202 samples) under ``benchmarks/data/`` — the phase that
finally exercises fit + reference CV together, rather than each lever in isolation:

* ``fit``     — both methods, chr22 PGS000027, ``backend="streaming"`` via ``harness.measure``
  (isolated subprocess, authoritative peak RSS). **No export here**: serializing a ~4 GB
  dense-score artifact inflates peak RSS ~10× and is an export-path concern, not the
  streaming-fit footprint — so the fit RAM headline is measured clean.
* ``refcv``   — additive single-pass reference CV (Phase 6) vs the dense refit-per-fold oracle,
  **both methods, 3 folds**; reuses ``verify_predict_scale.part_refcv`` / ``part_refcv_projection``
  verbatim (they self-assert additive==refit metric parity and guard against a temp-VCF regression).
* ``project`` — 500K extrapolation via the streaming-imputation 1D peak-vs-n fit + the analytical
  byte model (NOT ``make_report``, whose ``(s,v)`` multilinear model is degenerate on a
  single-chromosome run), pooling the Phase-2 streaming RAM-sweep points; plus the full 500K×2M
  dense-vs-streaming anchor.

Opt-in legs (NOT in ``--part all`` — heavy / need extra deps):

* ``golden``  — real-scale export→reload→predict at atol=1e-12 (imputation, json-only, exported to
  a temp dir and deleted). The invariant is also covered continuously by ``tests/test_golden.py``
  and ``tests/test_scale_integration.py``.
* ``mps``     — ``device="cpu"`` vs ``device="mps"`` imputation fit parity; run under ``.venv-gpu``.

chr22 only (user directive); the full 22-autosome / 2.1M run is out of scope. Each streaming fit is
~9–16 min. Curated → ``benchmarks/results/integration/``.

Run:  .venv/bin/python -m benchmarks.verify_integration --part all --folds 3
      .venv/bin/python -m benchmarks.verify_integration --part golden
      .venv-gpu/bin/python -m benchmarks.verify_integration --part mps
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from benchmarks.harness import collect_metadata, load_results, measure
from benchmarks.oracle import compare_oracle
from benchmarks.run_baseline import (
    DEFAULT_CHIP_FILE,
    GENOME_BUILD,
    PGS_SCALE,
    _BENCH_DATA,
    _fit_spec,
)
from benchmarks.scaling_projection import analytical_memory_bytes
from benchmarks.scenarios import predict_bytes
from benchmarks.verify_gpu_scale import _resolve_gpu_device
from benchmarks.verify_predict_scale import CHR22_REF, part_refcv, part_refcv_projection
from benchmarks.verify_streaming_scale import RESULTS as STREAM_RESULTS
from benchmarks.verify_streaming_scale import (
    PARITY_ATOL,
    PARITY_RTOL,
    _config,
    _lstsq_linear,
    _restricted_prs_csv,
    _scale_record,
)

log = logging.getLogger("verify_integration")

RESULTS = _BENCH_DATA.parent / "results" / "integration"
N_SAMPLES = 3202
TARGET_SAMPLES = 500_000
TARGET_VARIANTS = 2_000_000
PROBE_N = 40
GOLDEN_ATOL = 1e-12


# --------------------------------------------------------------------------------------
def _model_cls(method: str):
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.core.linear_projection_prs import LinearProjectionPRS

    return LinearImputationPRS if method == "imputation" else LinearProjectionPRS


def _build_probe(sub_df, n: int = PROBE_N) -> Dict:
    """Deterministic biallelic-SNP het probe: ``{rsid: model variant_id, genotype: effect+other}``.

    The golden invariant is *child-model prs == reloaded-artifact prs* (atol=1e-12) on the **same**
    input, so the probe only has to be deterministic and re-buildable; restricting to biallelic SNPs
    keeps the two-character genotype string unambiguous.
    """
    ea = sub_df["effect_allele"].astype(str)
    oa = sub_df["other_allele"].astype(str)
    snp = sub_df[(ea.str.len() == 1) & (oa.str.len() == 1)].head(n)
    vids = snp["variant_id"].astype(str).tolist()
    genos = [f"{e}{o}" for e, o in zip(snp["effect_allele"], snp["other_allele"])]
    return {"genotypes": {"rsid": vids, "genotype": genos}, "apply_calibration": True}


def _golden_rescore(method: str, rec: Dict) -> Dict:
    """Reload the exported artifact, re-score the probe, and compare to the child model's prs."""
    import pandas as pd

    paths = rec.get("export_paths") or {}
    jpath = paths.get("json")
    probe = rec.get("probe")
    child = (rec.get("child_probe") or {}).get("prs")
    if not jpath or not Path(jpath).exists() or not probe:
        return {"status": "skipped", "reason": "no exported json / probe"}

    model = _model_cls(method).load(jpath)
    df = pd.DataFrame(probe["genotypes"])
    result = model.predict(df, apply_calibration=bool(probe.get("apply_calibration", True)))
    reload_prs = float(getattr(result, "prs", float("nan")))

    if child is None:
        diff, status = None, "no_child_probe"
    else:
        cf = float(child)
        diff = 0.0 if (math.isnan(cf) and math.isnan(reload_prs)) else abs(reload_prs - cf)
        status = "ok" if diff <= GOLDEN_ATOL else "mismatch"

    units = model.imputed_models if method == "imputation" else model.region_models
    return {
        "status": status, "atol": GOLDEN_ATOL,
        "child_prs": child, "reload_prs": reload_prs, "abs_diff": diff,
        "reload_n_units": len(units), "oracle_n_units": rec.get("n_units"),
        "probe_n_variants": len(probe["genotypes"]["rsid"]),
    }


def _oracle_view(rec: Dict) -> Dict:
    """The statistical-parity subset of a fit record (what ``compare_oracle`` should judge)."""
    return {k: rec.get(k) for k in ("summary", "calibration", "r2_summary", "n_units")}


def _run_fit(method: str, meta, timeout_s: float, *, device: Optional[str] = None,
             export_dir: Optional[Path] = None, export_formats=None) -> Dict:
    """One streaming chr22 fit in an isolated child (peak RSS authoritative).

    ``export_dir``/probe are set only for the golden leg — the fit RAM headline is measured
    WITHOUT export.
    """
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    cfg = _config("streaming") if device is None else _config("streaming", device=device)
    label = f"integ_{method}_chr22_streaming" + (f"_{device}" if device else "")
    probe = _build_probe(sub) if export_dir else None
    spec = _fit_spec(
        method, CHR22_REF, prs_csv, DEFAULT_CHIP_FILE, GENOME_BUILD,
        label=label, n_samples=N_SAMPLES, n_variants=len(sub),
        config=cfg, soft_ceiling=None, tracemalloc=False,
        probe=probe, export_dir=(str(export_dir) if export_dir else None),
    )
    if export_dir and export_formats:
        spec.params["export_formats"] = list(export_formats)
    res = measure(spec, RESULTS, timeout_s=timeout_s, metadata=meta)
    log.info(
        "%-36s outcome=%s wall=%.1fs peak=%.2fGB%s",
        label, res.outcome, res.wall_seconds or -1.0, (res.peak_rss_bytes or 0) / 1e9,
        "" if res.ok else f"  ERROR: {res.error_message}",
    )
    rec = _scale_record(res, CHR22_REF, len(sub))
    o = res.result or {}
    rec["device"] = device or "cpu"
    if export_dir:
        rec["export_paths"] = o.get("export_paths")
        rec["export_ok"] = o.get("export_ok")
        rec["child_probe"] = o.get("predict_probe")
        rec["probe"] = probe
    return rec


# --------------------------------------------------------------------------------------
def part_fit(meta) -> Dict:
    """Both methods, chr22 streaming fit — clean peak RSS (no export; see ``_run_fit``)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: Dict = {"pgs_id": PGS_SCALE, "genome_build": GENOME_BUILD,
                 "n_samples": N_SAMPLES, "methods": {}}
    for method in ("imputation", "projection"):
        rec = _run_fit(method, meta, timeout_s=5400)
        (RESULTS / f"{method}_chr22.json").write_text(json.dumps(rec, indent=2, default=str))
        out["methods"][method] = rec
    (RESULTS / "fit_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def part_golden(meta) -> Dict:
    """Real-scale export→reload→predict at atol=1e-12 (imputation; json-only, temp, cleaned up).

    Opt-in (NOT in ``--part all``): exporting a ~4 GB dense-score artifact needs ~20 GB RAM. The
    invariant is also covered by ``tests/test_golden.py`` + ``tests/test_scale_integration.py``.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="integ_golden_"))
    out: Dict = {"methods": {}}
    try:
        for method in ("imputation",):  # the heavier dense-score case; projection covered by tests
            rec = _run_fit(method, meta, timeout_s=5400,
                           export_dir=tmp / method, export_formats=["json"])
            g = _golden_rescore(method, rec) if rec.get("ok") else {"status": "fit_failed"}
            out["methods"][method] = g
            log.info("golden[%-10s] %s  child=%s reload=%s Δ=%s", method, g.get("status"),
                     g.get("child_prs"), g.get("reload_prs"), g.get("abs_diff"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # free the multi-GB temp artifact
    (RESULTS / "golden.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def part_refcv_all(meta, folds: int = 3) -> Dict:
    """Additive single-pass reference CV vs dense refit oracle, both methods (reused verbatim)."""
    log.info("refcv: imputation additive-vs-refit (%d folds) ...", folds)
    imp = part_refcv(meta, n_folds=folds)
    log.info("refcv: projection additive-vs-refit (%d folds) ...", folds)
    proj = part_refcv_projection(meta, n_folds=folds)
    rec = {"n_folds": folds, "imputation": imp, "projection": proj}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reference_cv_chr22.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def _bounded_chr22(start: int, end: int):
    """Prefilter CHR22_REF + window the PGS000027 PRS to a small chr22 region (fast real-data refcv)."""
    from benchmarks.run_baseline import _prs_to_csv
    from benchmarks.verify_streaming_scale import STREAM_WORK

    sub, _ = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    pos = sub["position"].astype(int)
    win = sub[(pos >= start) & (pos <= end)].copy()
    prs_csv = _prs_to_csv(win, STREAM_WORK / "prs" / f"PGS000027_22_{start}_{end}.csv")
    out_vcf = STREAM_WORK / "pos" / f"chr22_{start}_{end}.vcf.gz"
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    if not out_vcf.exists():
        # CHR22_REF is GRCh38 with 'chr'-prefixed contigs (chr22); the library normalizes
        # chr22<->22 downstream, but the raw bcftools region query needs the file's contig.
        subprocess.run(["bcftools", "view", "-r", f"chr22:{start}-{end}", "-Oz",
                        "-o", str(out_vcf), str(CHR22_REF)], check=True)
        subprocess.run(["bcftools", "index", "-t", str(out_vcf)], check=True)
    return out_vcf, prs_csv, len(win)


def part_refcv_bounded(meta, start: int = 16_000_000, end: int = 18_000_000, folds: int = 3) -> Dict:
    """Real-data additive-vs-refit reference-CV parity on a bounded chr22 region (fast, foreground).

    The full 3,202-sample × whole-chr22 refcv is ~35 min of per-variant solves (the solve is
    n_samples-independent, so subsampling samples does not shorten it) and would not run to
    completion in this environment (long-lived jobs are terminated). The additive==refit parity is
    the same math at any variant count — and is proven hermetically by
    ``tests/test_scale_integration.py`` + ``tests/test_cv_stats.py`` — so this confirms it on REAL
    1000G chr22 dosages within the time budget and records the k-fold-saving extrapolation.
    """
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.core.linear_projection_prs import LinearProjectionPRS
    from imputed_prs.evaluation import ImputationEvaluator
    from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator

    ref, prs_csv, n_win = _bounded_chr22(start, end)
    with open(DEFAULT_CHIP_FILE) as fh:
        platform = [ln.strip() for ln in fh if ln.strip()]
    common = dict(reference_genotypes=str(ref), prs_definition=str(prs_csv),
                  platform_variants=platform, n_folds=folds, random_state=42)
    log.info("refcv-bounded: 22:%d-%d, %d PRS variants, %d samples, %d folds",
             start, end, n_win, N_SAMPLES, folds)

    out: Dict = {"region": f"22:{start}-{end}", "n_prs_variants": n_win, "n_folds": folds,
                 "n_samples": N_SAMPLES, "methods": {}}
    for name, cls, ev_cls in [("imputation", LinearImputationPRS, ImputationEvaluator),
                              ("projection", LinearProjectionPRS, ProjectionEvaluator)]:
        parent = cls(window_size=1_000_000, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
                     cv_folds=5, random_state=42, backend="streaming", verbose=0)
        t0 = time.perf_counter()
        parent.fit(reference_genotypes=str(ref), prs_definition=str(prs_csv),
                   platform_variants=platform, genome_build=GENOME_BUILD)
        parent_wall = time.perf_counter() - t0
        ev = ev_cls(parent, verbose=0)

        # Guard: additive CV must not write a temp VCF (Phase-4 regression check).
        real_tnf = tempfile.NamedTemporaryFile

        def _no_temp(*a, **k):
            raise AssertionError("cross_validate wrote a temporary file (temp-VCF regression)")

        tempfile.NamedTemporaryFile = _no_temp
        try:
            t1 = time.perf_counter()
            cv_a = ev.cross_validate(backend="streaming", **common)
            a_wall = time.perf_counter() - t1
        finally:
            tempfile.NamedTemporaryFile = real_tnf

        t2 = time.perf_counter()
        cv_b = ev.cross_validate(backend="dense", **common)
        b_wall = time.perf_counter() - t2

        dr2 = abs(cv_a.mean_r2 - cv_b.mean_r2)
        out["methods"][name] = {
            "parent_fit_seconds": parent_wall,
            "additive": {"backend": "streaming", "seconds": a_wall, "mean_r2": cv_a.mean_r2,
                         "mean_correlation": cv_a.mean_correlation},
            "refit_oracle": {"backend": "dense", "seconds": b_wall, "mean_r2": cv_b.mean_r2,
                             "mean_correlation": cv_b.mean_correlation},
            "abs_mean_r2_diff": dr2,
            "parity_ok": dr2 < 1e-3,
            "no_temp_vcf": True,
            "measured_speedup_x": (b_wall / a_wall) if a_wall > 0 else None,
            "kfold_extrapolation": {
                str(k): {"refit_predicted_seconds": k * parent_wall,
                         "additive_predicted_seconds": parent_wall, "predicted_speedup_x": float(k)}
                for k in (3, 5, 10)
            },
        }
        log.info("refcv-bounded[%-10s] additive %.1fs (r2=%.4f) vs refit %.1fs (r2=%.4f)  |Δr2|=%.2e %s",
                 name, a_wall, cv_a.mean_r2, b_wall, cv_b.mean_r2, dr2, "OK" if dr2 < 1e-3 else "MISMATCH")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reference_cv_bounded_chr22.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def part_mps(meta) -> Dict:
    """device=cpu vs device=mps imputation chr22 fit: parity + wall/peak-RSS (self-skips w/o torch)."""
    dev = _resolve_gpu_device()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if dev is None:
        log.info("mps: no GPU device (torch missing or no mps/cuda) — self-skip. "
                 "Run under .venv-gpu/bin/python -m benchmarks.verify_integration --part mps")
        rec = {"status": "skipped_no_device"}
        (RESULTS / "mps_parity.json").write_text(json.dumps(rec, indent=2, default=str))
        return rec
    log.info("mps: device=cpu vs device=%s (imputation chr22) ...", dev)
    cpu = _run_fit("imputation", meta, timeout_s=5400, device="cpu")
    gpu = _run_fit("imputation", meta, timeout_s=5400, device=dev)
    rec: Dict = {"device": dev, "tolerance": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
                 "cpu": cpu, "gpu": gpu}
    if cpu.get("ok") and gpu.get("ok"):
        diff = compare_oracle(_oracle_view(cpu), _oracle_view(gpu),
                              rtol=PARITY_RTOL, atol=PARITY_ATOL)
        mism = {k: {"cpu": a, "gpu": b} for k, (a, b, ok) in diff.items() if not ok}
        cw, gw = cpu.get("wall_seconds"), gpu.get("wall_seconds")
        rec.update({
            "status": "ok" if not mism else "mismatch",
            "n_fields_compared": len(diff), "n_mismatch": len(mism), "mismatches": mism,
            "cpu_wall_seconds": cw, "gpu_wall_seconds": gw,
            "cpu_over_gpu_speedup": (cw / gw) if (cw and gw) else None,
            "cpu_peak_rss_gb": cpu.get("peak_rss_gb"), "gpu_peak_rss_gb": gpu.get("peak_rss_gb"),
        })
    else:
        rec["status"] = "run_failed"
    (RESULTS / "mps_parity.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


# --------------------------------------------------------------------------------------
def _streaming_imputation_points(pool) -> list:
    """Completed, authoritative streaming-*imputation* fits, restricted to the dominant workload.

    The scalable per-variant profile: constant variant count, varying n. (Dense oracle fits,
    projection mega-region fits, and the tiny PRS-313 point are filtered out — pooling them into
    one model conflates footprints; see :func:`part_project`.)
    """
    pts = []
    for r in pool:
        c = getattr(r.spec, "config", {}) or {}
        p = getattr(r.spec, "params", {}) or {}
        if (c.get("backend") == "streaming" and p.get("method", "imputation") == "imputation"
                and r.outcome == "completed" and r.peak_rss_is_authoritative and r.peak_rss_bytes):
            v = (r.result or {}).get("n_variants") or r.spec.n_variants
            pts.append({"label": r.spec.label, "n_samples": r.spec.n_samples, "n_variants": v,
                        "peak_rss_bytes": r.peak_rss_bytes, "peak_rss_gb": r.peak_rss_bytes / 1e9,
                        "wall_seconds": r.wall_seconds})
    if pts:  # keep the dominant-v group (chr22 PGS000027); drops the v=313 PRS-313 parity point
        vmode = Counter(p["n_variants"] for p in pts).most_common(1)[0][0]
        pts = [p for p in pts if p["n_variants"] == vmode]
    pts.sort(key=lambda p: p["n_samples"])
    return pts


def part_project() -> Dict:
    """500K extrapolation — the chr22-scoped streaming methodology, NOT the blocker-grid model.

    A single-chromosome run has ~one variant-axis point, so ``scaling_projection.make_report``'s
    multilinear ``(s, v)`` model is degenerate here and conflates dense/streaming footprints.
    The principled estimate (mirroring ``results/streaming/ram_extrapolation_imputation.json``) is
    a 1D peak-vs-n fit over the streaming-imputation points + the analytical band-buffer byte model,
    plus the full-problem 500K×2M dense anchor (the "dense would be TB" honesty line).
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    pool = list(load_results(RESULTS))
    n_integ = len(pool)
    n_stream = 0
    try:
        stream = list(load_results(STREAM_RESULTS))
        pool += stream
        n_stream = len(stream)
    except Exception as exc:  # noqa: BLE001
        log.warning("project: streaming raw results unavailable (%s); integration-only", exc)

    pts = _streaming_imputation_points(pool)
    xs = [p["n_samples"] for p in pts]
    n_var = pts[0]["n_variants"] if pts else 27_854
    empirical = None
    if len(set(xs)) >= 2:
        a, b, r2 = _lstsq_linear(xs, [float(p["peak_rss_bytes"]) for p in pts])
        empirical = {"model": "peak_rss = a + b*n_samples", "a_bytes": a, "b_bytes_per_sample": b,
                     "r2": r2, "predicted_500k_gb": (a + b * TARGET_SAMPLES) / 1e9,
                     "caveat": "measured n<=3202: the band buffer is tiny vs fixed overhead, so the "
                               "slope is noisy; the analytical model is the principled estimate."}

    band = 512  # imputation slides a ~few-hundred-column chip band; see scenarios.predict_bytes
    stream_chr22 = predict_bytes("streaming_fit", TARGET_SAMPLES, n_var, n_missing=band)
    dense_chr22 = predict_bytes("fit", TARGET_SAMPLES, n_var)
    stream_full = predict_bytes("streaming_fit", TARGET_SAMPLES, TARGET_VARIANTS, n_missing=band)
    dense_full = analytical_memory_bytes(TARGET_SAMPLES, TARGET_VARIANTS)

    report: Dict = {
        "method": "imputation",
        "workload": f"chr22 PGS000027 (~{n_var} variants)",
        "points": pts,
        "n_points": len(pts),
        "n_distinct_n": len(set(xs)),
        "empirical_fit_500k": empirical,
        "analytical_500k_chr22": {
            "band_proxy": band,
            "streaming_total_gb": stream_chr22["total_est"] / 1e9,
            "dense_total_gb": dense_chr22["total_est"] / 1e9,
            "dense_over_streaming_ratio": dense_chr22["total_est"] / max(stream_chr22["total_est"], 1),
        },
        "full_scale_anchor_500k_x_2m": {
            "streaming_total_gb": stream_full["total_est"] / 1e9,
            "dense_one_copy_tb": dense_full["dense_one_copy"] / 1e12,
            "dense_total_tb": dense_full["total"] / 1e12,
            "dense_breakdown": dense_full,
            "note": "the plan's headline: at 500K×2M one dense float32 copy is ~4 TB (peak ~24 TB "
                    "across working copies + cv_predictions); the streaming band-buffer path stays "
                    "in GB (sample dimension contracted).",
        },
        "pool": {"integration_results": n_integ, "streaming_results": n_stream},
        "methodology_note": "make_report (Phase-0 blocker-grid multilinear (s,v) model) is NOT used "
                            "here: a chr22-only run has one variant-axis point, so that model is "
                            "degenerate and mixes dense+streaming. This is the streaming 1D-in-n "
                            "peak fit + analytical byte model, per results/streaming/.",
    }
    if n_stream == 0:
        report["low_confidence"] = True
        log.warning("project: no Phase-2 streaming RAM-sweep points under %s — few distinct sample "
                    "sizes (low_confidence). Run `verify_streaming_scale --part ram_impute` first.",
                    STREAM_RESULTS)
    (RESULTS / "projection.json").write_text(json.dumps(report, indent=2, default=str))
    _print_project(report)
    return report


# --------------------------------------------------------------------------------------
def _print_fit(out: Dict) -> None:
    print("\n=== Phase-11 integration: chr22 PGS000027 end-to-end (both methods, streaming) ===")
    print(f"  panel: real GRCh38 1000G high-coverage, {out.get('n_samples')} samples "
          "(fit RAM measured WITHOUT export)")
    for method, r in out.get("methods", {}).items():
        if not r.get("ok"):
            print(f"  {method}: RUN FAILED (outcome={r.get('outcome')}) {r.get('error')}")
            continue
        r2 = (r.get("r2_summary") or {})
        cal = (r.get("calibration") or {})
        print(f"  {method}: peak={r.get('peak_rss_gb', 0):.2f} GB  wall={ (r.get('wall_seconds') or 0):.0f}s"
              f"  n_units={r.get('n_units')}  mean_r2={r2.get('mean')}"
              f"  calib_scale={cal.get('scaling_factor')}")


def _print_project(rep: Dict) -> None:
    print("\n=== Phase-11 500K extrapolation (chr22 streaming imputation) ===")
    print(f"  workload: {rep.get('workload')}  ({rep.get('n_points')} points, "
          f"{rep.get('n_distinct_n')} distinct n)")
    emp = rep.get("empirical_fit_500k")
    if emp:
        print(f"  empirical peak = {emp['a_bytes']/1e9:.2f} GB + {emp['b_bytes_per_sample']/1e3:.0f} "
              f"KB/sample  (R²={emp['r2']:.3f})  → 500K ≈ {emp['predicted_500k_gb']:.0f} GB")
    an = rep.get("analytical_500k_chr22", {})
    print(f"  analytical (chr22 @500K): streaming ≈ {an.get('streaming_total_gb', 0):.1f} GB vs "
          f"dense ≈ {an.get('dense_total_gb', 0):.0f} GB ({an.get('dense_over_streaming_ratio', 0):.0f}×)")
    fs = rep.get("full_scale_anchor_500k_x_2m", {})
    print(f"  full 500K×2M anchor: streaming ≈ {fs.get('streaming_total_gb', 0):.0f} GB vs "
          f"dense ≈ {fs.get('dense_one_copy_tb', 0):.0f} TB/copy (~{fs.get('dense_total_tb', 0):.0f} TB peak)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--part",
                    choices=["fit", "refcv", "refcv-bounded", "project", "golden", "mps", "all"],
                    default="all")
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = collect_metadata()
    RESULTS.mkdir(parents=True, exist_ok=True)

    # `all` = the core, robust deliverables. golden (heavy export) and mps (needs a torch venv)
    # are opt-in only.
    if args.part in ("fit", "all"):
        _print_fit(part_fit(meta))
    if args.part in ("refcv", "all"):
        part_refcv_all(meta, args.folds)
    if args.part == "refcv-bounded":
        part_refcv_bounded(meta, folds=args.folds)
    if args.part in ("project", "all"):
        part_project()
    if args.part == "golden":
        part_golden(meta)
    if args.part == "mps":
        part_mps(meta)


if __name__ == "__main__":
    main()
