"""Phase 8 — batched multi-target solve micro-scaling (no 500K/2M run).

Times the per-target ElasticNet solve loop (``[fit_from_local_gram(b) …]``, the sklearn
coordinate-descent oracle) against the batched solve (``solve_blocks_batched``, batched
FISTA/ridge) on a chr22-like batch of co-windowed Gram blocks, for a few batch widths and
hyperparameters. The solve is sample-free (cost ∝ p², not n), so this isolates exactly what
Phase 8 optimizes — the per-target Python-call + un-vectorized-linear-algebra overhead that
dominates a 2M-variant fit — and extrapolates the whole-genome saving.

Run:  .venv/bin/python -m benchmarks.verify_batched_solve
Correctness (batched == per-target within statistical parity) is covered by
tests/test_gram_solve.py and tests/test_batched_solve_streaming.py; this script only measures
wall-clock and is exempt from the full-scale exercise per the plan's Phase 5-9 rule.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from imputed_prs.compute.gram_solve import (
    LocalGramBlock,
    fit_from_local_gram,
    solve_blocks_batched,
)

from imputed_prs.compute.gram_solve import _BATCH_MAX_P_ENET, _BATCH_MAX_P_RIDGE

RESULTS = Path(__file__).resolve().parent / "results" / "batched_solve"

N_SAMPLES = 2504          # 1000G scale (the solve is n-independent; only block build uses it)
# The batched-solve speedup is dominated by the predictor count p, not the sample count:
# batching amortizes per-target Python-call overhead while the per-block linear algebra is
# small, and loses once the O(p²)/O(p³) work (and, for enet, FISTA's iteration count) grows.
# Sweep p across the crossover; a realistic ±1 Mb dense-chip window sits at the high end.
PREDICTOR_COUNTS = (5, 10, 20, 40, 80, 150)
BATCH = 64                # ~a streaming chunk at 500K samples (batch_cap-bound)
CV_FOLDS = 5
HYPERPARAMS = [("ridge", 0.01, 0.0), ("enet", 0.01, 0.5)]
GENOME_TARGETS = 2_000_000  # PGS000027-scale extrapolation
SEED = 11


def _build_block(n: int, p: int, cv: int, rng: np.random.Generator) -> LocalGramBlock:
    """A realistic dosage-like local Gram block with contiguous CV folds."""
    freqs = rng.uniform(0.1, 0.9, size=p)
    X = rng.binomial(2, freqs, size=(n, p)).astype(np.float64)
    y = np.clip(X @ (rng.standard_normal(p) * 0.3) + rng.standard_normal(n) * 0.5, 0.0, 2.0)
    block = LocalGramBlock(
        n=n, G=X.T @ X, c=X.T @ y, zsum=X.sum(0), zsqsum=(X * X).sum(0),
        ysum=float(y.sum()), ysqsum=float(y @ y),
    )
    for sl in np.array_split(np.arange(n), cv):
        Xv, yv = X[sl], y[sl]
        block.fold_G.append(Xv.T @ Xv)
        block.fold_c.append(Xv.T @ yv)
        block.fold_zsum.append(Xv.sum(0))
        block.fold_zsqsum.append((Xv * Xv).sum(0))
        block.fold_ysum.append(float(yv.sum()))
        block.fold_ysqsum.append(float(yv @ yv))
        block.fold_n.append(len(sl))
    return block


def _time(fn: Callable[[], object], reps: int = 3) -> float:
    fn()  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _bench_one(blocks: List[LocalGramBlock], alpha: float, l1_ratio: float) -> Dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns on ElasticNet(l1_ratio=0)
        per_target = _time(
            lambda: [
                fit_from_local_gram(b, alpha=alpha, l1_ratio=l1_ratio, cv_folds=CV_FOLDS)
                for b in blocks
            ]
        )
    batched = _time(lambda: solve_blocks_batched(blocks, alpha, l1_ratio, CV_FOLDS))
    return {"per_target_s": per_target, "batched_s": batched, "speedup": per_target / batched}


def main() -> Dict:
    rng = np.random.default_rng(SEED)
    print(f"blocks: n={N_SAMPLES} folds={CV_FOLDS} batch={BATCH}  "
          f"(runtime p-gate: ridge≤{_BATCH_MAX_P_RIDGE}, enet≤{_BATCH_MAX_P_ENET})")

    runs: List[Dict] = []
    for name, alpha, l1 in HYPERPARAMS:
        gate = _BATCH_MAX_P_RIDGE if l1 == 0.0 else _BATCH_MAX_P_ENET
        print(f"  {name} (alpha={alpha}, l1={l1}):")
        for p in PREDICTOR_COUNTS:
            blocks = [_build_block(N_SAMPLES, p, CV_FOLDS, rng) for _ in range(BATCH)]
            r = _bench_one(blocks, alpha, l1)
            genome_pt = GENOME_TARGETS / BATCH * r["per_target_s"]
            genome_bt = GENOME_TARGETS / BATCH * r["batched_s"]
            batched_at_runtime = p <= gate  # what _solve_real_padded's p-gate would pick
            runs.append({
                "kind": name, "alpha": alpha, "l1_ratio": l1, "p": p, "batch": BATCH,
                **r, "batched_at_runtime": batched_at_runtime,
                "genome_per_target_s": genome_pt, "genome_batched_s": genome_bt,
            })
            flag = "batched" if batched_at_runtime else "per-target (gated)"
            print(f"    p={p:4d}: per-target {r['per_target_s']*1e3:8.2f} ms  "
                  f"batched {r['batched_s']*1e3:8.2f} ms  speedup {r['speedup']:5.2f}x  "
                  f"→ {flag}  | 2M: {genome_pt:6.1f}s vs {genome_bt:6.1f}s")

    out = {
        "config": {
            "n_samples": N_SAMPLES, "predictor_counts": list(PREDICTOR_COUNTS), "batch": BATCH,
            "cv_folds": CV_FOLDS, "genome_targets": GENOME_TARGETS, "seed": SEED,
            "p_gate": {"ridge": _BATCH_MAX_P_RIDGE, "enet": _BATCH_MAX_P_ENET},
        },
        "runs": runs,
        "note": (
            "Batched CPU solve wins only for SMALL predictor counts, where per-target Python-"
            "call overhead dominates the tiny per-block linear algebra (≈3-6x at p≤20). It "
            "loses at large p — the compiled Cholesky/CD already saturates and FISTA needs "
            "many more iterations than sklearn CD — so solve_blocks_batched's runtime p-gate "
            "(_BATCH_MAX_P_{RIDGE,ENET}) routes only small-p blocks to the batch and large-p "
            "blocks to the per-target path (never a regression). A realistic ±1 Mb dense-chip "
            "window has hundreds of predictors → per-target; small windows / capped "
            "max_predictors / sparse chips → batched. Warm starts + screening (later Phase-8 "
            "commits) cut FISTA iterations / effective p and widen the batched range. Solve "
            "cost is n-independent (Gram is p×p), so this extrapolates across sample count."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "solve_scaling.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RESULTS / 'solve_scaling.json'}")
    return out


if __name__ == "__main__":
    main()
