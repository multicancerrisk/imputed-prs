# Phase-4 panel-scale prediction — bounded scale validation

Real GRCh38 1000G high-coverage panel (3,202 samples), produced by
`benchmarks/verify_predict_scale.py`. Phase 4 vectorizes the evaluator's panel scoring
(a `scipy.sparse` CSR mat-mul + `ReferenceAlleleResolver` orientation replacing the
per-model / per-predictor Python loops) and removes the temp-VCF round-trip from
reference CV / sensitivity (`fit` now ingests in-memory genotypes).

**Correctness/parity is locked by the unit suite**, not these runs:
`tests/test_vectorized_predictor.py` (orientation byte-identical to the oracle; batch ==
oracle at `atol=1e-9`), `tests/test_bounding.py` (array == scalar at `atol=1e-12`),
`tests/test_round_trip.py::TestForcedBatchParity` (evaluator batch == oracle, both
methods), and `tests/test_streaming_backend.py::TestInMemoryFitParity` (dense-from-
`GenotypeData` exact, streaming-from-`InMemoryGenotypeSource` within the parity band).
These runs measure the **scale** behavior.

## 1. Vectorized panel scoring — speedup + parity at scale
`evaluator_speedup.json` — chr22 PGS000027, **24,597 imputed targets, 3,202 samples**.
The estimated-PRS numeric path is timed both ways via the `_force_batch` hook; the
auto-batched true-PRS is timed once.

| stage | wall | note |
|---|---|---|
| estimated PRS — per-model oracle loop | 67.6 s | the pre-Phase-4 path |
| estimated PRS — **vectorized CSR batch** | **6.9 s** | **9.8× faster** |
| true PRS over 27,731 placed variants (`accumulate_true_prs`) | 0.28 s | was a per-variant `match_oriented_dosage` loop |

- **batch vs oracle max abs diff = 7.7e-15** — effectively exact (on hard-called integer
  dosages the float32 orientation is exact; only the CSR SpMM reassociation shows, far
  inside the `atol=1e-9` bar). Confirms the batch scorer matches the oracle at 24.6K targets.
- peak RSS **5.2 GB** (authoritative), fit 560 s (matches the Phase-2 chr22 baseline).
- **500K-sample extrapolation** (batch scoring is O(n_samples) in the `Z @ Wᵀ` mat-mul):
  ~1,074 s (~18 min) to score a 500K panel at this variant count — vs ~9,900 s for the
  per-model loop.

The estimated component on *hard-called* reference data still routes through the string
replay by design (`is_hard_called` → the browser-parity path); the batch numeric scorer is
the 2M-variant *continuous* / measurement path and is what the ≥256-target size-select
selects. The true-PRS vectorization applies on both modes.

## 2. Reference CV — no temp-VCF round-trip (bounded: chr22, 3 folds)
`ImputationEvaluator.cross_validate` was reworked to fit each fold from an in-memory
`InMemoryGenotypeSource` (streaming, no serialization). **The "no temp-VCF is written"
guarantee is locked by construction and the unit suite** —
`tests/test_streaming_backend.py::TestInMemoryFitParity` plus the deletion of
`_write_genotype_data_to_vcf` — not by this timing run.

The real-data timing run (`--part refcv`, chr22 + 3 folds) was **not carried to
completion**. On this *hard-called* 1000G reference the per-fold **held-out scoring**
routes through the per-sample string-replay path (`_predicted_prs_via_strings`, the
browser-parity path — see the hard-called note in §1), which is O(held-out samples ×
variants) pure-Python and is *not* what Phase 4 vectorized. It dominated wall-clock
(killed at >2 h; the four streaming fits themselves were ~9 min each and completed
early) while telling us nothing about the no-temp-VCF seam actually under test. A
bounded real-data CV number needs either a much smaller PRS subset or a **vectorized
hard-called scorer** (a recommended follow-up); the full 2M / 10-fold run is Phase 6
regardless.

## Reproduce
```
.venv/bin/python -m benchmarks.verify_predict_scale --part evaluator   # ~10 min (fit + score)
.venv/bin/python -m benchmarks.verify_predict_scale --part refcv --folds 3   # bounded CV, no temp VCF
.venv/bin/python -m benchmarks.verify_predict_scale --part masking     # run_masking_validation, timed
```
Requires the cached chr22 reference `benchmarks/data/work/pos/22.vcf.gz` (built by
`verify_streaming_scale --part impute`).
