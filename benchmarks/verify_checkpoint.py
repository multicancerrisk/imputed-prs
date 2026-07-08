"""Phase 10 — micro-benchmark for the resumable streaming fit (io/checkpoint.py).

A streaming fit contracts the sample dimension into one per-chromosome partial per
chromosome, produced by ``_run_one_chromosome`` and merged by an order-independent reduce.
With ``checkpoint_dir`` set, each partial is persisted the instant it completes; a killed
run resumes by loading the finished chromosomes (a cheap disk read) and recomputing only
the rest — bit-identical to an uninterrupted run.

This script measures the two per-chromosome costs on a synthetic streaming panel:

    t_fit    fit one chromosome  (stream + band Gram + local solves)   — paid on a cold run
    t_load   load one chromosome's persisted partial from disk         — paid on a resume

and composes them into the wall-clock of a run killed after completing a fraction ``f`` of
the chromosomes: a cold restart re-pays ``C·t_fit``, while a resume pays
``f·C·t_load + (1−f)·C·t_fit`` — a saving of ``f·C·(t_fit − t_load)``. Because ``t_fit``
grows ~linearly in sample count while ``t_load`` is disk-bound, the resume saving grows with
n; at 500K samples the saving approaches the full fraction of already-done compute. Bit-parity
of a resumed vs uninterrupted artifact is asserted inline and covered exhaustively by
tests/test_checkpoint.py.

Run:  .venv/bin/python -m benchmarks.verify_checkpoint
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from imputed_prs import LinearImputationPRS
from imputed_prs.core.types import GenotypeData
from imputed_prs.io.checkpoint import get_checkpoint_info

RESULTS = Path(__file__).resolve().parent / "results" / "checkpoint"

N_CHROM = 8
N_VAR_PER_CHROM = 30
N_SAMPLES = 800
SCALE_N = 500_000
SEED = 20260708


def _build_panel(n_chrom=N_CHROM, n_var=N_VAR_PER_CHROM, n_samples=N_SAMPLES, seed=SEED):
    """An integer-dosage panel: (n_var-1) chip predictors + 1 missing target / chrom."""
    rng = np.random.default_rng(seed)
    vrows, cols, prows, platform = [], [], [], []
    for c in range(1, n_chrom + 1):
        chrom, base = str(c), c * 1_000_000
        for i in range(n_var):
            vid = f"{chrom}:{base + i * 500}"
            vrows.append({
                "variant_id": vid, "chromosome": chrom, "position": base + i * 500,
                "ref_allele": "A", "alt_allele": "G",
            })
            cols.append(rng.integers(0, 3, size=n_samples).astype(np.float32))
            prows.append({
                "variant_id": vid, "chromosome": chrom, "position": base + i * 500,
                "effect_allele": "G", "other_allele": "A", "beta": 0.05 * (i + 1),
            })
            if i < n_var - 1:
                platform.append(vid)
    gd = GenotypeData(
        dosage_matrix=np.stack(cols, axis=1),
        variant_info=pd.DataFrame(vrows),
        sample_ids=[f"S{i}" for i in range(n_samples)],
    )
    return gd, pd.DataFrame(prows), platform


def _fit(gd, prs, plat, checkpoint_dir):
    m = LinearImputationPRS(
        backend="streaming", tuning_scope="none", cv_folds=3, random_state=17, verbose=0,
    )
    m.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
          genome_build="GRCh37", checkpoint_dir=str(checkpoint_dir))
    return m


def _coefs(model):
    return {mm.variant_id: np.asarray(mm.coefficients) for mm in model._imputed_models}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    gd, prs, plat = _build_panel()

    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"

        t0 = perf_counter()
        cold = _fit(gd, prs, plat, ckpt)  # cold: every chromosome fitted + persisted
        t_cold = perf_counter() - t0
        info = get_checkpoint_info(ckpt)
        n_shards = info["entries"][0]["n_shards"]

        t0 = perf_counter()
        warm = _fit(gd, prs, plat, ckpt)  # warm: every chromosome resumed from disk
        t_warm = perf_counter() - t0

        # Simulate a kill after half the chromosomes: drop half the shards, then resume.
        entry = Path(info["path"]) / info["entries"][0]["digest"]
        shards = sorted(entry.glob("chr*.ckpt"))
        killed = len(shards) // 2
        for s in shards[:killed]:
            s.unlink()
        t0 = perf_counter()
        resumed = _fit(gd, prs, plat, ckpt)  # recompute `killed`, resume the rest
        t_partial = perf_counter() - t0

    # Correctness: a resumed fit is bit-identical to the uninterrupted (cold) one.
    cold_c, warm_c, res_c = _coefs(cold), _coefs(warm), _coefs(resumed)
    assert cold_c.keys() == warm_c.keys() == res_c.keys()
    for vid in cold_c:
        assert np.array_equal(cold_c[vid], warm_c[vid]), vid
        assert np.array_equal(cold_c[vid], res_c[vid]), vid

    # Decompose per-chromosome costs (t_warm ≈ C·t_load + fixed overhead; the fixed
    # metadata/plan overhead is paid on every fit, so difference it out).
    per_fit = max(t_cold - t_warm, 1e-9) / n_shards       # marginal fit cost / chromosome
    per_load = t_warm / n_shards                           # load + reduce cost / chromosome (+overhead share)
    saved_half = killed * per_fit                          # compute skipped by resuming half

    def resume_saving(fraction, total_fit_time):
        # A run killed after fraction f of the chromosomes saves ~f of the compute on resume.
        return fraction * total_fit_time

    report = {
        "panel": {"n_chrom": N_CHROM, "n_var_per_chrom": N_VAR_PER_CHROM, "n_samples": N_SAMPLES},
        "n_shards": n_shards,
        "t_cold_s": round(t_cold, 4),
        "t_warm_full_resume_s": round(t_warm, 4),
        "t_partial_resume_s": round(t_partial, 4),
        "per_chrom_fit_s": round(per_fit, 5),
        "per_chrom_load_s": round(per_load, 5),
        "killed_chroms": killed,
        "compute_skipped_half_resume_s": round(saved_half, 4),
        "extrapolation": {
            "note": (
                "t_fit scales ~linearly in n_samples; t_load is disk-bound. A run killed "
                "after fraction f of the chromosomes resumes in ~f·C·t_load + (1-f)·C·t_fit, "
                "saving ~f of the compute — approaching the full done-fraction at scale."
            ),
            "scale_n_samples": SCALE_N,
            "est_per_chrom_fit_at_scale_s": round(per_fit * SCALE_N / N_SAMPLES, 2),
            "est_saving_resume_at_90pct_done_s": round(
                resume_saving(0.9, N_CHROM * per_fit * SCALE_N / N_SAMPLES), 1
            ),
        },
    }
    out = RESULTS / "verify_checkpoint.json"
    out.write_text(json.dumps(report, indent=2))

    print("Phase 10 checkpoint micro-benchmark")
    print(f"  panel: {N_CHROM} chrom x {N_VAR_PER_CHROM} var x {N_SAMPLES} samples")
    print(f"  shards written (cold): {n_shards}")
    print(f"  t_cold (full fit)          : {t_cold*1e3:8.1f} ms")
    print(f"  t_warm (full resume)       : {t_warm*1e3:8.1f} ms  ({t_cold/max(t_warm,1e-9):.1f}x faster)")
    print(f"  t_partial (resume half)    : {t_partial*1e3:8.1f} ms")
    print(f"  per-chrom fit / load       : {per_fit*1e3:.2f} / {per_load*1e3:.2f} ms")
    print("  resumed artifact == cold artifact: bit-identical ✓")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
