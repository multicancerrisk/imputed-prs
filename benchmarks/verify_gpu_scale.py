"""Phase-3 GPU-acceleration validation: ``device="cpu"`` vs ``device="mps"`` (or CUDA).

Two parts, both on the real GRCh38 1000G high-coverage panel (3,202 samples) under
``benchmarks/data/`` and reusing the Phase-0/2 rig (``harness.measure`` + ``oracle``):

* ``parity`` — chr22 imputation **streaming** fit, ``device="cpu"`` vs ``device="mps"``, via
  the public ``fit()`` → oracle. Asserts statistical parity (rtol=5e-3/atol=1e-3) and records
  wall-clock + peak RSS + the CPU→GPU speedup. This is the end-to-end headline. **Honest
  framing:** at n=3,202 the O(n) accumulation is ~1% of the fit (Phase-2 profiling), so any
  end-to-end GPU win here comes from batching the per-unit *solve* (the n-independent floor),
  not the accumulation — the accumulation win is an O(n) effect shown by ``accum`` below.

* ``accum`` — isolated microbench of the two O(n) accumulation seams (Seam B: the incremental
  band-Gram ``add``; Seam A: the chunk cross-product ``C=ZᵀY``), timed **CPU numpy vs device**
  across n ∈ {3,202 .. 500,000}. The per-sample cost is fit and extrapolated to the 500K
  production scale, which is where the accumulation actually dominates and the GPU win lands.

Run under the torch venv:  ``.venv-gpu/bin/python -m benchmarks.verify_gpu_scale --part all``
Under the torch-free ``.venv`` the GPU parts self-skip (no device) with a clear message.
Curated results -> ``benchmarks/results/gpu/*.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks import prefilter
from benchmarks.harness import RunMetadata, collect_metadata, measure
from benchmarks.oracle import compare_oracle
from benchmarks.run_baseline import (
    DEFAULT_CHIP_FILE,
    DEFAULT_KG_DIR,
    GENOME_BUILD,
    PGS_BASELINE,
    _BENCH_DATA,
    _load_chip_set,
    _prepare_reference,
    _prs_chroms,
    _prs_to_csv,
)
from benchmarks.verify_streaming_scale import (
    PARITY_ATOL,
    PARITY_RTOL,
    STREAM_WORK,
    _lstsq_linear,
    _prs,
    _scale_record,
    _spec,
)

log = logging.getLogger("verify_gpu_scale")

RESULTS = _BENCH_DATA.parent / "results" / "gpu"
WINDOW = 1_000_000

# Isolated-seam microbench sweep (synthetic 0/1/2 dosages; the accumulation cost is
# n-linear with a fixed band, so a handful of points pins the per-sample slope cleanly).
ACCUM_NS: Tuple[int, ...] = (3_202, 25_000, 100_000, 500_000)
ACCUM_M = 256    # chip columns per band-fill (one stream block)
ACCUM_T = 64     # stacked targets for the Seam-A cross-product C=ZᵀY
ACCUM_K = 5      # CV folds (per-fold Gram maintenance is part of Seam B)
TARGET_N = 500_000


# --------------------------------------------------------------------------------------
# device probing (only run GPU work when a real device is present in THIS interpreter —
# measure() spawns the fit child with the same sys.executable, so torch propagates)
# --------------------------------------------------------------------------------------
def _resolve_gpu_device() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        return None
    return None


def _sync(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _config(device: str, **over) -> Dict:
    cfg = dict(
        window_size=WINDOW, cv_folds=5, tuning_scope="none", l1_ratio=0.5, alpha=0.01,
        n_jobs=1, random_state=42, verbose=1, backend="streaming", device=device,
    )
    cfg.update(over)
    return cfg


def _run(spec, meta: RunMetadata, timeout_s: float):
    res = measure(spec, RESULTS, timeout_s=timeout_s, metadata=meta)
    log.info(
        "%-30s outcome=%s wall=%.1fs peak_rss=%.2fGB%s",
        spec.label, res.outcome, res.wall_seconds or -1.0,
        (res.peak_rss_bytes or 0) / 1e9,
        "" if res.ok else f"  ERROR: {res.error_message}",
    )
    return res


# --------------------------------------------------------------------------------------
# Part: end-to-end real-data parity + throughput (PRS-313, cpu vs gpu, both methods)
# --------------------------------------------------------------------------------------
# Why PRS-313 and not a dense score (PGS000027) for the headline: an end-to-end fit runs one
# per-unit solve per trained unit, and each tiny solve is *launch-bound* on MPS. A dense chr22
# score is ~24,600 units — that many launch-bound solves time out (>15 min) even at n=500 (the
# de-risk measured peak 5.2 GB — bounded, no OOM — but wall > 900 s). PRS-313 is 227 imputation /
# 161 projection units, which completes on MPS and still validates real-data parity on the full
# 1000G panel. The *scale* win lives in the accumulation kernels (see part_accum), not the solve.
_SMALL_N_FINDING = (
    "A dense chr22 PGS000027 fit (~24,600 units) is launch-bound on MPS and does not complete in "
    "budget (de-risk n=500: peak 5.2 GB bounded, wall > 900 s timeout); the per-unit solve is "
    "n-independent, so this is a unit-count/launch-overhead effect, not memory. device='auto' "
    "routes fits below GPU_AUTO_MIN_SAMPLES (25,000) to CPU for exactly this reason."
)


def _prepare_prs313_reference():
    """Prepare (cached) the real GRCh38 1000G PRS-313 reference across its chromosomes."""
    prs_df = _prs(PGS_BASELINE)
    chip = _load_chip_set(DEFAULT_CHIP_FILE)
    chroms = _prs_chroms(prs_df)
    STREAM_WORK.mkdir(parents=True, exist_ok=True)
    ref_vcf, n_used, n_pos = _prepare_reference(
        prs_df, chip, chroms, None, DEFAULT_KG_DIR, STREAM_WORK,
        prefilter.GRCH38_HIGHCOV_PATTERN, WINDOW,
    )
    prs_csv = _prs_to_csv(prs_df, STREAM_WORK / "prs" / f"{PGS_BASELINE}_{GENOME_BUILD}.csv")
    return ref_vcf, prs_csv, n_used, n_pos, len(prs_df)


def _parity_record(cpu, gpu, ref, n_prs) -> Dict:
    rec: Dict = {"cpu": _scale_record(cpu, ref, n_prs), "gpu": _scale_record(gpu, ref, n_prs)}
    if not (cpu.ok and gpu.ok):
        rec["status"] = "run_failed"
        rec["cpu_outcome"], rec["cpu_error"] = cpu.outcome, cpu.error_message
        rec["gpu_outcome"], rec["gpu_error"] = gpu.outcome, gpu.error_message
        return rec
    diff = compare_oracle(cpu.result, gpu.result, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    mism = {k: {"cpu": a, "gpu": b} for k, (a, b, ok) in diff.items() if not ok}
    cw, gw = cpu.wall_seconds, gpu.wall_seconds
    rec.update({
        "status": "ok" if not mism else "mismatch",
        "n_fields_compared": len(diff), "n_mismatch": len(mism), "mismatches": mism,
        "cpu_wall_seconds": cw, "gpu_wall_seconds": gw,
        "cpu_over_gpu_speedup": (cw / gw) if (cw and gw) else None,
        "cpu_peak_rss_gb": (cpu.peak_rss_bytes or 0) / 1e9,
        "gpu_peak_rss_gb": (gpu.peak_rss_bytes or 0) / 1e9,
    })
    return rec


def part_parity(meta: RunMetadata, gpu_device: str,
                methods=("imputation", "projection")) -> Dict:
    ref_vcf, prs_csv, n_used, n_pos, n_prs = _prepare_prs313_reference()
    log.info("parity: PRS-313 reference ready (%d samples, %d positions)", n_used, n_pos)
    out: Dict = {
        "pgs_id": PGS_BASELINE, "gpu_device": gpu_device, "n_samples": n_used,
        "n_positions": n_pos, "n_prs_variants": n_prs,
        "tolerance": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
        "small_n_finding": _SMALL_N_FINDING, "methods": {},
    }
    for method in methods:
        cpu = _run(_spec(f"gpu_parity_{method}_cpu", method, ref_vcf, prs_csv,
                         _config("cpu"), n_used, n_prs), meta, 1800)
        gpu = _run(_spec(f"gpu_parity_{method}_{gpu_device}", method, ref_vcf, prs_csv,
                         _config(gpu_device), n_used, n_prs), meta, 1800)
        out["methods"][method] = _parity_record(cpu, gpu, ref_vcf, n_prs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "parity_prs313.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------------------
# Part: isolated accumulation microbench (Seam A + Seam B), CPU vs device, n-sweep
# --------------------------------------------------------------------------------------
def _time_cpu(fn: Callable[[], None], reps: int = 3) -> float:
    fn()  # warm (allocations, BLAS threadpool)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _time_gpu(fn: Callable[[], None], torch, device: str, reps: int = 3) -> float:
    fn(); _sync(torch, device)  # warm (kernel compile, buffer alloc)
    best = float("inf")
    for _ in range(reps):
        _sync(torch, device)
        t = time.perf_counter()
        fn(); _sync(torch, device)
        best = min(best, time.perf_counter() - t)
    return best


def _synth_columns(n: int, m: int, k: int, seed: int):
    """A stream of (col_perm, platform_idx, position, af) synthetic 0/1/2 dosages."""
    from imputed_prs.compute.sufficient_stats import GlobalFolds

    rng = np.random.RandomState(seed)
    folds = GlobalFolds(n, k, 42)
    cols = []
    for c in range(m):
        freq = rng.uniform(0.1, 0.9)
        raw = rng.binomial(2, freq, size=n).astype(np.float64)
        cols.append((folds.permute(raw), c, 1000 * (c + 1), float(freq)))
    return folds, cols


def _seam_b_point(n: int, gpu_device: str) -> Dict:
    """Band-fill ACCUM_M chip columns: numpy per-column ``add`` vs device ``add_batch``."""
    import torch

    from imputed_prs.compute.gpu_backend import GpuBackend
    from imputed_prs.compute.sufficient_stats import _ChipGramBuffer

    folds, cols = _synth_columns(n, ACCUM_M, ACCUM_K, seed=1)
    be = GpuBackend(gpu_device)

    def fill_cpu() -> None:
        buf = _ChipGramBuffer(folds.n, folds)
        for col, p, pos, af in cols:
            buf.add(col, p, pos, af)

    def fill_gpu() -> None:  # flush the whole block as one batched GEMM (the 3D win)
        buf = be.make_buffer(folds.n, folds)
        buf.add_batch([c for c, *_ in cols], [p for _, p, *_ in cols],
                      [ps for *_, ps, _ in cols], [af for *_, af in cols])

    tc = _time_cpu(fill_cpu)
    tg = _time_gpu(fill_gpu, torch, gpu_device)
    return {"n_samples": n, "cpu_seconds": tc, "gpu_seconds": tg,
            "cpu_over_gpu_speedup": tc / tg if tg else None}


def _seam_a_point(n: int, gpu_device: str) -> Dict:
    """Chunk cross-product C=ZᵀY (m×T): numpy float64 vs device float32 GEMM."""
    import torch

    rng = np.random.RandomState(2)
    Z = rng.binomial(2, 0.3, size=(n, ACCUM_M)).astype(np.float64)
    Y = rng.standard_normal((n, ACCUM_T)).astype(np.float64)
    dev = gpu_device
    Zt = torch.as_tensor(Z, device=dev, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)

    tc = _time_cpu(lambda: (Z.T @ Y, None)[1])
    tg = _time_gpu(lambda: (Zt.t() @ Yt, None)[1], torch, dev)
    return {"n_samples": n, "cpu_seconds": tc, "gpu_seconds": tg,
            "cpu_over_gpu_speedup": tc / tg if tg else None}


def _extrapolate(points: List[Dict], key_cpu: str = "cpu_seconds",
                 key_gpu: str = "gpu_seconds") -> Dict:
    """Linear (in n) fit of CPU and device seam cost → predict at 500K."""
    xs = [float(p["n_samples"]) for p in points]
    a_c, b_c, r2_c = _lstsq_linear(xs, [float(p[key_cpu]) for p in points])
    a_g, b_g, r2_g = _lstsq_linear(xs, [float(p[key_gpu]) for p in points])
    cpu_500k = a_c + b_c * TARGET_N
    gpu_500k = a_g + b_g * TARGET_N
    return {
        "model": "seconds = a + b*n_samples",
        "cpu_fit": {"a": a_c, "b_per_sample": b_c, "r2": r2_c},
        "gpu_fit": {"a": a_g, "b_per_sample": b_g, "r2": r2_g},
        "predicted_500k": {"cpu_seconds": cpu_500k, "gpu_seconds": gpu_500k,
                           "cpu_over_gpu_speedup": cpu_500k / gpu_500k if gpu_500k > 0 else None},
    }


def part_accum(gpu_device: str) -> Dict:
    log.info("accum: Seam-A/B microbench on %s over n=%s (M=%d cols, T=%d, K=%d)",
             gpu_device, list(ACCUM_NS), ACCUM_M, ACCUM_T, ACCUM_K)
    seam_b, seam_a = [], []
    for n in ACCUM_NS:
        b = _seam_b_point(n, gpu_device)
        a = _seam_a_point(n, gpu_device)
        seam_b.append(b)
        seam_a.append(a)
        log.info("  n=%-7d  Seam-B(add) cpu=%.4fs gpu=%.4fs (%.2fx)   "
                 "Seam-A(ZᵀY) cpu=%.4fs gpu=%.4fs (%.2fx)",
                 n, b["cpu_seconds"], b["gpu_seconds"], b["cpu_over_gpu_speedup"] or 0.0,
                 a["cpu_seconds"], a["gpu_seconds"], a["cpu_over_gpu_speedup"] or 0.0)
    rec = {
        "gpu_device": gpu_device, "m_cols": ACCUM_M, "t_targets": ACCUM_T, "k_folds": ACCUM_K,
        "seam_b_band_gram_add": {"points": seam_b, "extrapolation_500k": _extrapolate(seam_b)},
        "seam_a_cross_product": {"points": seam_a, "extrapolation_500k": _extrapolate(seam_a)},
        "note": "Isolated accumulation kernels (synthetic 0/1/2 dosages). Seam B = incremental "
                "band-Gram maintenance (add_batch GEMM vs numpy per-column add); Seam A = the "
                "chunk cross-product C=ZᵀY. Both O(n) with a fixed band, so the per-sample slope "
                "extrapolates cleanly; 500K is where the accumulation dominates the fit.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "accumulation_scaling.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


# --------------------------------------------------------------------------------------
def _print_parity(rec: Dict) -> None:
    print("\n=== PRS-313 real-data parity: device=cpu vs device=%s (GRCh38 1000G, n=%s) ==="
          % (rec.get("gpu_device"), rec.get("n_samples")))
    for method, p in rec.get("methods", {}).items():
        if p.get("status") == "run_failed":
            print(f"  {method}: RUN FAILED (cpu={p.get('cpu_outcome')} gpu={p.get('gpu_outcome')})")
            print(f"      cpu_error={p.get('cpu_error')}")
            print(f"      gpu_error={p.get('gpu_error')}")
            continue
        print(f"  {method}: {p.get('status', '?').upper()}  "
              f"({p.get('n_fields_compared')} fields, {p.get('n_mismatch')} outside "
              f"rtol={PARITY_RTOL}/atol={PARITY_ATOL})")
        for k, v in list(p.get("mismatches", {}).items())[:12]:
            print(f"      Δ {k}: cpu={v['cpu']} gpu={v['gpu']}")
        cw, gw, sp = p.get("cpu_wall_seconds"), p.get("gpu_wall_seconds"), p.get("cpu_over_gpu_speedup")
        if cw and gw:
            print(f"      wall: cpu={cw:.1f}s  gpu={gw:.1f}s  ({sp:.2f}x)   "
                  f"peak RSS: cpu={p.get('cpu_peak_rss_gb', 0):.2f}GB gpu={p.get('gpu_peak_rss_gb', 0):.2f}GB")
    print(f"  note: {rec.get('small_n_finding', '')}")


def _print_accum(rec: Dict) -> None:
    print(f"\n=== accumulation microbench (device={rec.get('gpu_device')}) ===")
    for seam, title in (("seam_b_band_gram_add", "Seam B  band-Gram add"),
                        ("seam_a_cross_product", "Seam A  C=ZᵀY")):
        ex = rec[seam]["extrapolation_500k"]["predicted_500k"]
        print(f"  {title}: measured speedups "
              + ", ".join(f"n={p['n_samples']}:{p['cpu_over_gpu_speedup']:.2f}x"
                          for p in rec[seam]["points"]))
        print(f"      -> 500K extrapolation: cpu={ex['cpu_seconds']:.3f}s gpu={ex['gpu_seconds']:.3f}s "
              f"({ex['cpu_over_gpu_speedup']:.2f}x)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["parity", "accum", "all"], default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gpu_device = _resolve_gpu_device()
    if gpu_device is None:
        print("No GPU device available in this interpreter (torch missing or no mps/cuda).")
        print("Run under the torch venv:  .venv-gpu/bin/python -m benchmarks.verify_gpu_scale")
        return
    log.info("GPU device resolved to %r", gpu_device)
    meta = collect_metadata()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.part in ("parity", "all"):
        _print_parity(part_parity(meta, gpu_device))
    if args.part in ("accum", "all"):
        _print_accum(part_accum(gpu_device))


if __name__ == "__main__":
    main()
