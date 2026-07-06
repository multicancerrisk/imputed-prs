"""Phase-4 bounded scale validation: vectorized panel scoring + reference CV.

Phase 4 vectorizes the evaluator's panel scoring (a ``scipy.sparse`` CSR mat-mul +
resolver-based orientation replacing the per-model / per-predictor Python loops) and
removes the temp-VCF round-trip from reference CV / sensitivity (``fit`` now ingests
in-memory genotypes). Correctness/parity is locked by the unit suite
(``tests/test_vectorized_predictor.py``, ``test_round_trip.py`` forced-batch,
``test_streaming_backend.py`` three-way in-memory-fit parity). This script measures
the *scale* behavior on the real GRCh38 1000G high-coverage panel (3,202 samples).

Parts (curated results -> ``benchmarks/results/predict/*.json``):

* ``evaluator`` — chr22 PGS000027 (34,388 positions): fit once, then time the
  vectorized batch estimated-PRS scorer vs the per-unit oracle and the auto-batched
  true-PRS; report speedup, batch-vs-oracle parity, and the O(n) 500K extrapolation.
* ``refcv``     — reworked ``cross_validate`` with ``backend="streaming"`` and a
  ``tempfile`` guard asserting **no temp-VCF is written**; per-fold wall. Defaults to
  chr22 + 3 folds (the locked bounded scope); the full-2M / 10-fold run is Phase 6.
* ``masking``   — ``run_masking_validation`` (scoring-only) on the chr22 model, timed.

Full 2M x 500K is Phase 6. Wall-clock budgets: each fit ~9 min at chr22, so ``refcv``
(3 folds + parent) ~35 min; a config projected past ~60 min should shrink + extrapolate.

Run:  .venv/bin/python -m benchmarks.verify_predict_scale --part evaluator
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.harness import RunMetadata, WorkSpec, collect_metadata, measure
from benchmarks.run_baseline import (
    DEFAULT_CHIP_FILE,
    GENOME_BUILD,
    PGS_SCALE,
    _BENCH_DATA,
)
from benchmarks.verify_streaming_scale import (
    WORK,
    _config,
    _lstsq_linear,
    _restricted_prs_csv,
)

log = logging.getLogger("verify_predict_scale")

RESULTS = _BENCH_DATA.parent / "results" / "predict"
TARGET_SAMPLES = 500_000

# chr22 PGS000027 cached reference (34,388 positions, 3,202 samples) — same panel the
# Phase-2 streaming validation used, so the fit cost is a known ~9 min.
CHR22_REF = WORK / "pos" / "22.vcf.gz"


def _phase_map(res) -> Dict[str, float]:
    return {p.name: p.wall_seconds for p in res.phases}


def _score_spec(label: str, method: str, ref, prs_csv, n_samples: int, n_variants: int) -> WorkSpec:
    params = dict(
        method=method,
        reference_genotypes=str(ref),
        prs_definition=str(prs_csv),
        platform_variants_file=str(DEFAULT_CHIP_FILE),
        genome_build=GENOME_BUILD,
        reference_panel_id="1000G_highcov_GRCh38",
        training_ancestry="ALL",
    )
    return WorkSpec(
        operation="score_arrays", label=label, params=params,
        config=_config("streaming"), n_samples=n_samples, n_variants=n_variants,
        seed=42, tracemalloc=False,
    )


# --------------------------------------------------------------------------------------
def part_evaluator(meta: RunMetadata, timeout_s: float = 1800.0) -> Dict:
    """Vectorized batch scorer vs per-unit oracle on chr22 (real 3,202-sample panel)."""
    if not CHR22_REF.exists():
        raise FileNotFoundError(
            f"missing cached chr22 reference {CHR22_REF}; run verify_streaming_scale "
            "--part impute first to build it."
        )
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    spec = _score_spec("eval_pgs027_chr22", "imputation", CHR22_REF, prs_csv, 3202, len(sub))
    res = measure(spec, RESULTS, timeout_s=timeout_s, metadata=meta)
    o = res.result or {}
    phases = _phase_map(res)

    est_oracle = phases.get("est_oracle")
    est_batch = phases.get("est_batch")
    speedup = (est_oracle / est_batch) if (est_oracle and est_batch) else None
    # Batch estimated-PRS scoring is O(n_samples * nnz(W)); extrapolate linearly in n.
    proj_500k = (
        est_batch * TARGET_SAMPLES / 3202 if est_batch else None
    )

    rec = {
        "part": "evaluator",
        "reference": str(CHR22_REF),
        "outcome": res.outcome,
        "ok": res.ok,
        "n_samples": o.get("n_samples"),
        "n_targets": o.get("n_targets"),
        "n_placed": o.get("n_placed"),
        "batch_vs_oracle_max_abs_diff": o.get("batch_vs_oracle_max_abs_diff"),
        "phase_wall_seconds": phases,
        "estimated_prs_speedup_oracle_over_batch": speedup,
        "peak_rss_gb": (res.peak_rss_bytes or 0) / 1e9,
        "peak_rss_authoritative": res.peak_rss_is_authoritative,
        "extrapolation_500k_samples": {
            "model": "batch scoring wall ~ O(n_samples) (Z @ Wᵀ)",
            "batch_scoring_wall_at_3202": est_batch,
            "predicted_batch_scoring_wall_500k_seconds": proj_500k,
        },
        "error": res.error_message,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "evaluator_speedup.json").write_text(json.dumps(rec, indent=2, default=str))
    log.info(
        "evaluator: outcome=%s est_oracle=%.2fs est_batch=%.2fs speedup=%.1fx "
        "parity_maxdiff=%s true_prs=%.2fs",
        res.outcome, est_oracle or -1, est_batch or -1, speedup or -1,
        o.get("batch_vs_oracle_max_abs_diff"), phases.get("true_prs") or -1,
    )
    return rec


# --------------------------------------------------------------------------------------
def part_refcv(
    meta: RunMetadata, n_folds: int = 3, ref: Optional[Path] = None, timeout_s: float = 3600.0
) -> Dict:
    """Reworked cross_validate: streaming folds, NO temp-VCF (asserted), per-fold wall.

    Runs in-process (cross_validate is a Python call). Patches
    ``tempfile.NamedTemporaryFile`` to raise so any regressed temp-VCF path fails loudly.
    Defaults to chr22 + 3 folds (the locked bounded scope). Requires a fitted parent
    model for the evaluator, so the total is (1 parent + n_folds) chr22 fits.
    """
    import tempfile

    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.evaluation import ImputationEvaluator

    ref = Path(ref) if ref is not None else CHR22_REF
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    with open(DEFAULT_CHIP_FILE) as fh:
        platform = [ln.strip() for ln in fh if ln.strip()]

    log.info("refcv: fitting parent model (chr22, streaming) ...")
    t0 = time.perf_counter()
    parent = LinearImputationPRS(
        window_size=1_000_000, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=42, backend="streaming", verbose=0,
    )
    parent.fit(reference_genotypes=str(ref), prs_definition=str(prs_csv),
               platform_variants=platform, genome_build=GENOME_BUILD)
    parent_wall = time.perf_counter() - t0
    evaluator = ImputationEvaluator(parent, verbose=0)

    # Guard: cross_validate must not write a temp VCF anymore.
    real_tnf = tempfile.NamedTemporaryFile

    def _no_temp(*a, **k):
        raise AssertionError("cross_validate wrote a temporary file (temp-VCF regression)")

    tempfile.NamedTemporaryFile = _no_temp
    try:
        t1 = time.perf_counter()
        cv = evaluator.cross_validate(
            reference_genotypes=str(ref), prs_definition=str(prs_csv),
            platform_variants=platform, n_folds=n_folds, random_state=42,
            backend="streaming",
        )
        cv_wall = time.perf_counter() - t1
    finally:
        tempfile.NamedTemporaryFile = real_tnf

    rec = {
        "part": "refcv",
        "reference": str(ref),
        "backend": "streaming",
        "n_folds": n_folds,
        "no_temp_vcf": True,
        "parent_fit_seconds": parent_wall,
        "cross_validate_seconds": cv_wall,
        "mean_seconds_per_fold": cv_wall / n_folds,
        "mean_correlation": cv.mean_correlation,
        "mean_r2": cv.mean_r2,
        "std_r2": cv.std_r2,
        "n_prs_variants_on_chr22": len(sub),
        "note": "full 2M / 10-fold reference CV over all of 1000G is Phase 6.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "refcv_bounded.json").write_text(json.dumps(rec, indent=2, default=str))
    log.info(
        "refcv: no_temp_vcf=True folds=%d cv_wall=%.1fs (%.1fs/fold) mean_r2=%.3f",
        n_folds, cv_wall, cv_wall / n_folds, cv.mean_r2,
    )
    return rec


# --------------------------------------------------------------------------------------
def part_masking(meta: RunMetadata, ref: Optional[Path] = None) -> Dict:
    """run_masking_validation (scoring-only) on the chr22 model, timed in-process."""
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.evaluation.validation import run_masking_validation

    ref = Path(ref) if ref is not None else CHR22_REF
    sub, prs_csv = _restricted_prs_csv(PGS_SCALE, {"22"}, "22")
    with open(DEFAULT_CHIP_FILE) as fh:
        platform = [ln.strip() for ln in fh if ln.strip()]

    log.info("masking: fitting chr22 model (streaming) ...")
    model = LinearImputationPRS(
        window_size=1_000_000, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=42, backend="streaming", verbose=0,
    )
    model.fit(reference_genotypes=str(ref), prs_definition=str(prs_csv),
              platform_variants=platform, genome_build=GENOME_BUILD)

    t0 = time.perf_counter()
    report = run_masking_validation(
        model, str(ref), platform_variants=platform, verbose=0,
    )
    wall = time.perf_counter() - t0
    rd = report.to_dict() if hasattr(report, "to_dict") else {}
    rec = {
        "part": "masking",
        "reference": str(ref),
        "masking_wall_seconds": wall,
        "n_prs_variants_on_chr22": len(sub),
        "report": rd,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "masking_bounded.json").write_text(json.dumps(rec, indent=2, default=str))
    log.info("masking: wall=%.1fs", wall)
    return rec


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["evaluator", "refcv", "masking", "all"], default="evaluator")
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = collect_metadata()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.part in ("evaluator", "all"):
        part_evaluator(meta)
    if args.part in ("refcv", "all"):
        part_refcv(meta, n_folds=args.folds)
    if args.part in ("masking", "all"):
        part_masking(meta)


if __name__ == "__main__":
    main()
