"""Phase-2 bounded scale validation for the streaming backend (both methods).

Reuses the Phase-0 rig (``prefilter`` + ``harness.measure`` + ``oracle``) on the real
GRCh38 1000G high-coverage panel (3,202 samples) under ``benchmarks/data/``:

* ``parity``     — PRS-313 (PGS000004) fit **dense vs streaming**, both methods; assert the
  statistical-parity oracle (counts, R² summary, calibration, provenance) matches within a
  documented tolerance. Closes the "parity holds on PRS-313" Done-when on *real* data.
* ``projection`` — projection streaming fit on **chr20-22** PGS000027 (132,107 reference
  positions, 3,202 samples); records wall-clock + peak RSS. Genome-fraction scale, all
  samples, multiple chromosomes merged.
* ``impute``     — imputation streaming fit on **chr22** PGS000027 (34,388 positions, 3,202
  samples); records per-unit throughput (variants/s) + peak RSS.
* ``ram``        — streaming peak RSS at n ∈ {500, 1000, 2000, 3202} on chr22 (via the
  method-agnostic band buffer), plus the analytical streaming-vs-dense byte model, both
  extrapolated to 500K samples.

The full 22-chromosome / 2.1M-variant run is Phase 6's job (full-scale integration); here we
prove the mechanism end-to-end on real multi-chromosome data with peak-RAM measured and
extrapolated. Curated results -> ``benchmarks/results/streaming/*.json``.

Run:  .venv/bin/python -m benchmarks.verify_streaming_scale --part all
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarks import prefilter
from benchmarks.harness import RunMetadata, WorkSpec, collect_metadata, measure
from benchmarks.oracle import compare_oracle
from benchmarks.run_baseline import (
    DEFAULT_CHIP_FILE,
    DEFAULT_KG_DIR,
    GENOME_BUILD,
    PGS_BASELINE,
    PGS_SCALE,
    _BENCH_DATA,
    _fetch_prs,
    _load_chip_set,
    _prepare_reference,
    _prs_chroms,
    _prs_to_csv,
)
from benchmarks.scenarios import predict_bytes

log = logging.getLogger("verify_streaming_scale")

RESULTS = _BENCH_DATA.parent / "results" / "streaming"
WORK = _BENCH_DATA / "work"                # existing Phase-0 cache (chr22 / chr20-22 pos + cells)
STREAM_WORK = _BENCH_DATA / "work_streaming"   # fresh workdir (no collision with the PGS000027 cache)
WINDOW = 1_000_000

# Statistical-parity band: streaming's Gram coordinate descent reproduces the dense oracle
# to well within this on a no-missing panel (the L1-kink OOF sensitivity washes out in the
# aggregate R²/calibration summary; see tests/test_gram_solve.py::test_scale_highreg_kink_parity).
PARITY_RTOL, PARITY_ATOL = 5e-3, 1e-3

_PRS_CACHE: Dict[str, object] = {}


def _prs(pgs_id: str):
    if pgs_id not in _PRS_CACHE:
        df, _ = _fetch_prs(pgs_id, GENOME_BUILD)
        _PRS_CACHE[pgs_id] = df
    return _PRS_CACHE[pgs_id]


def _config(backend: str, **over) -> Dict:
    cfg = dict(
        window_size=WINDOW, cv_folds=5, tuning_scope="none", l1_ratio=0.5, alpha=0.01,
        n_jobs=1, random_state=42, verbose=1, backend=backend,
    )
    cfg.update(over)
    return cfg


def _spec(label: str, method: str, ref_vcf, prs_csv, config: Dict,
          n_samples: int, n_variants: int) -> WorkSpec:
    params = dict(
        method=method,
        reference_genotypes=str(ref_vcf),
        prs_definition=str(prs_csv),
        platform_variants_file=str(DEFAULT_CHIP_FILE),
        genome_build=GENOME_BUILD,
        reference_panel_id="1000G_highcov_GRCh38",
        training_ancestry="ALL",
    )
    return WorkSpec(
        operation="fit", label=label, params=params, config=config,
        n_samples=n_samples, n_variants=n_variants, seed=42, tracemalloc=False,
    )


def _run(spec: WorkSpec, meta: RunMetadata, timeout_s: float):
    res = measure(spec, RESULTS, timeout_s=timeout_s, metadata=meta)
    log.info(
        "%-28s outcome=%s wall=%.1fs peak_rss=%.2fGB%s",
        spec.label, res.outcome, res.wall_seconds or -1.0,
        (res.peak_rss_bytes or 0) / 1e9,
        "" if res.ok else f"  ERROR: {res.error_message}",
    )
    return res


def _restricted_prs_csv(pgs_id: str, chroms: set, tag: str):
    df = _prs(pgs_id)
    sub = df[df["chromosome"].isin({str(c) for c in chroms})].copy()
    csv = _prs_to_csv(sub, STREAM_WORK / "prs" / f"{pgs_id}_{GENOME_BUILD}_{tag}.csv")
    return sub, csv


def _scale_record(res, ref, n_prs: int) -> Dict:
    o = res.result or {}
    return {
        "label": res.spec.label,
        "outcome": res.outcome,
        "ok": res.ok,
        "reference": str(ref),
        "n_prs_variants_on_chroms": n_prs,
        "wall_seconds": res.wall_seconds,
        "peak_rss_bytes": res.peak_rss_bytes,
        "peak_rss_gb": (res.peak_rss_bytes or 0) / 1e9,
        "peak_rss_authoritative": res.peak_rss_is_authoritative,
        "n_units": o.get("n_units"),
        "summary": o.get("summary"),
        "calibration": o.get("calibration"),
        "r2_summary": o.get("r2_summary"),
        "phases": [dataclasses.asdict(p) for p in res.phases],
        "error": res.error_message,
    }


# --------------------------------------------------------------------------------------
# Part: parity (PRS-313, dense vs streaming, both methods)
# --------------------------------------------------------------------------------------
def _parity(dres, sres) -> Dict:
    if not dres.ok or not sres.ok:
        return {
            "status": "run_failed",
            "dense_outcome": dres.outcome, "dense_error": dres.error_message,
            "stream_outcome": sres.outcome, "stream_error": sres.error_message,
        }
    diff = compare_oracle(dres.result, sres.result, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    mism = {k: {"dense": a, "stream": b} for k, (a, b, ok) in diff.items() if not ok}
    return {
        "status": "ok" if not mism else "mismatch",
        "tolerance": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
        "n_fields_compared": len(diff),
        "n_mismatch": len(mism),
        "mismatches": mism,
        "dense_oracle": dres.result,
        "stream_oracle": sres.result,
    }


def part_parity(meta: RunMetadata) -> Dict:
    prs_df = _prs(PGS_BASELINE)
    chip = _load_chip_set(DEFAULT_CHIP_FILE)
    chroms = _prs_chroms(prs_df)
    STREAM_WORK.mkdir(parents=True, exist_ok=True)
    log.info("parity: prefiltering %s reference (%d variants, chroms=%s) ...",
             PGS_BASELINE, len(prs_df), ",".join(chroms))
    ref_vcf, n_used, n_pos = _prepare_reference(
        prs_df, chip, chroms, None, DEFAULT_KG_DIR, STREAM_WORK,
        prefilter.GRCH38_HIGHCOV_PATTERN, WINDOW,
    )
    prs_csv = _prs_to_csv(prs_df, STREAM_WORK / "prs" / f"{PGS_BASELINE}_{GENOME_BUILD}.csv")
    log.info("parity: reference ready (%d samples, %d positions)", n_used, n_pos)

    out: Dict = {"pgs_id": PGS_BASELINE, "genome_build": GENOME_BUILD,
                 "n_samples": n_used, "n_positions": n_pos, "methods": {}}
    for method in ("imputation", "projection"):
        dres = _run(_spec(f"parity_{method}_dense", method, ref_vcf, prs_csv,
                          _config("dense"), n_used, len(prs_df)), meta, 1800)
        sres = _run(_spec(f"parity_{method}_streaming", method, ref_vcf, prs_csv,
                          _config("streaming"), n_used, len(prs_df)), meta, 1800)
        out["methods"][method] = _parity(dres, sres)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "parity_prs313.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------------------
# Parts: projection / imputation scale runs
# --------------------------------------------------------------------------------------
def part_projection(meta: RunMetadata) -> Dict:
    ref = WORK / "pos" / "22+21+20.vcf.gz"  # 132,107 positions, 3,202 samples (cached, complete)
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"20", "21", "22"}, "20-22")
    res = _run(_spec("proj_pgs027_chr20-22_streaming", "projection", ref, prs_csv,
                     _config("streaming"), 3202, len(sub)), meta, 3600)
    rec = _scale_record(res, ref, len(sub))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "projection_chr20-22.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def part_impute(meta: RunMetadata) -> Dict:
    ref = WORK / "pos" / "22.vcf.gz"  # 34,388 positions, 3,202 samples (cached, complete)
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    res = _run(_spec("impute_pgs027_chr22_streaming", "imputation", ref, prs_csv,
                     _config("streaming"), 3202, len(sub)), meta, 5400)
    rec = _scale_record(res, ref, len(sub))
    if res.ok and res.result:
        summ = res.result.get("summary") or {}
        n_trained = summ.get("n_imputed") or summ.get("n_variants_trained") or res.result.get("n_units")
        if n_trained and res.wall_seconds:
            rec["variants_trained"] = n_trained
            rec["variants_per_sec"] = n_trained / res.wall_seconds
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "imputation_chr22.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


# --------------------------------------------------------------------------------------
# Part: RAM scaling + 500K extrapolation
# --------------------------------------------------------------------------------------
def _lstsq_linear(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (intercept, slope, r2) for y = a + b*x by ordinary least squares."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def part_ram(meta: RunMetadata, method: str = "imputation") -> Dict:
    # Streaming peak RSS = fixed interpreter/BLAS/cyvcf2 overhead + the chip band buffer
    # (imputation: ±2W sliding window, O(n·band)) + O(n) calibration accumulators. Imputation
    # is the clean scalable profile; projection on a *dense* score merges into chromosome-
    # spanning mega-regions whose per-fold Gram Ghold=(K,cap,cap) dominates RAM independently
    # of n (see the projection finding), so its RAM-vs-n slope is not the band-buffer scaling.
    refs = {
        500: WORK / "cells" / "22_s500.vcf.gz",
        1000: WORK / "cells" / "22_s1000.vcf.gz",
        2000: WORK / "cells" / "22_s2000.vcf.gz",
        3202: WORK / "pos" / "22.vcf.gz",
    }
    if method == "imputation":
        refs.pop(3202, None)  # reuse the 4c chr22 imputation point (n=3202) to save a ~9-min run
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    points: List[Dict] = []
    for n, ref in refs.items():
        if not Path(ref).exists():
            log.warning("ram: missing cached reference %s (skip n=%d)", ref, n)
            continue
        res = _run(_spec(f"ram_{method[:4]}_chr22_s{n}", method, ref, prs_csv,
                         _config("streaming"), n, len(sub)), meta, 5400)
        if res.ok and res.peak_rss_bytes:
            points.append({"n_samples": n, "peak_rss_bytes": res.peak_rss_bytes,
                           "peak_rss_gb": res.peak_rss_bytes / 1e9,
                           "wall_seconds": res.wall_seconds,
                           "authoritative": res.peak_rss_is_authoritative})
    if method == "imputation":  # fold in the already-measured n=3202 chr22 imputation point (4c)
        try:
            d = json.loads((RESULTS / "imputation_chr22.json").read_text())
            if d.get("ok") and d.get("peak_rss_bytes"):
                points.append({"n_samples": 3202, "peak_rss_bytes": d["peak_rss_bytes"],
                               "peak_rss_gb": d["peak_rss_bytes"] / 1e9,
                               "wall_seconds": d.get("wall_seconds"), "authoritative": True})
        except Exception as exc:  # noqa: BLE001
            log.warning("ram: could not reuse 4c n=3202 point: %s", exc)
        points.sort(key=lambda p: p["n_samples"])

    rec: Dict = {"method": method, "reference": "chr22 PGS000027 (34,388 positions)",
                 "n_prs_variants": len(sub), "points": points}

    # Empirical linear fit peak_rss = a + b*n (dominated by fixed overhead at these small n;
    # the slope is the streaming per-sample cost).
    if len(points) >= 2:
        xs = [p["n_samples"] for p in points]
        ys = [float(p["peak_rss_bytes"]) for p in points]
        a, b, r2 = _lstsq_linear(xs, ys)
        rec["empirical_fit"] = {
            "model": "peak_rss = a + b*n_samples", "a_bytes": a, "b_bytes_per_sample": b,
            "r2": r2, "predicted_500k_bytes": a + b * 500_000,
            "predicted_500k_gb": (a + b * 500_000) / 1e9,
            "caveat": "measured n<=3202: band buffer is tiny vs fixed overhead, so the slope "
                      "is noisy; the analytical model below is the principled 500K estimate.",
        }

    # Analytical byte model (the principled cross-check), streaming vs dense, at 500K x 34,388.
    n_var = 34_388
    # Band proxy: imputation slides a ±2W window (~hundreds of co-windowed chip cols); a dense
    # projection region buffers thousands. Drives the streaming band_matrix term at 500K.
    band = 512 if method == "imputation" else 8192
    stream_500k = predict_bytes("streaming_fit", 500_000, n_var, n_missing=band)
    dense_500k = predict_bytes("fit", 500_000, n_var)
    rec["analytical_500k"] = {
        "n_samples": 500_000, "n_variants": n_var, "band_proxy": band,
        "streaming_total_bytes": stream_500k["total_est"],
        "streaming_total_gb": stream_500k["total_est"] / 1e9,
        "dense_total_bytes": dense_500k["total_est"],
        "dense_total_tb": dense_500k["total_est"] / 1e12,
        "dense_over_streaming_ratio": dense_500k["total_est"] / max(stream_500k["total_est"], 1),
    }
    rec["band_proxy"] = band
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"ram_extrapolation_{method}.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


# --------------------------------------------------------------------------------------
def _print_parity(out: Dict) -> None:
    print("\n=== PRS-313 dense-vs-streaming parity (real GRCh38 1000G) ===")
    print(f"  samples={out.get('n_samples')} positions={out.get('n_positions')}")
    for method, r in out.get("methods", {}).items():
        status = r.get("status")
        if status == "run_failed":
            print(f"  {method}: RUN FAILED (dense={r.get('dense_outcome')} "
                  f"stream={r.get('stream_outcome')})")
            continue
        print(f"  {method}: {status.upper()}  "
              f"({r.get('n_fields_compared')} fields, {r.get('n_mismatch')} outside "
              f"rtol={PARITY_RTOL}/atol={PARITY_ATOL})")
        for k, v in list(r.get("mismatches", {}).items())[:12]:
            print(f"      Δ {k}: dense={v['dense']} stream={v['stream']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["parity", "projection", "impute", "ram", "ram_impute", "all"],
                    default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = collect_metadata()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.part in ("parity", "all"):
        _print_parity(part_parity(meta))
    if args.part in ("projection", "all"):
        part_projection(meta)
    if args.part in ("impute", "all"):
        part_impute(meta)
    if args.part in ("ram", "all"):
        part_ram(meta, "projection")
    if args.part in ("ram_impute", "all"):
        part_ram(meta, "imputation")


if __name__ == "__main__":
    main()
