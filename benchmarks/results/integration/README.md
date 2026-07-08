# Phase-11 full-scale integration (chromosome-22)

Consolidated end-to-end validation on the real GRCh38 1000G high-coverage panel (3,202 samples)
under `benchmarks/data/`, scoped to **chromosome 22** (PGS000027, BMI). Produced by
`benchmarks/verify_integration.py`. This is the phase that exercises fit + reference CV + the
export round-trip together; the per-lever proofs live in the sibling `results/` dirs. Every fit
runs in an isolated subprocess (`benchmarks.harness.measure`; peak RSS via `os.wait4`), so peak RSS
is authoritative.

## 1. End-to-end fit — both methods, chr22, `backend="streaming"`

| method | peak RSS | wall | units | quality | source |
|---|---|---|---|---|---|
| imputation | **3.68 GB** | 567 s | 24,597 models | mean imputation R² **0.773**, calib R² 0.952 | `../streaming/imputation_chr22.json` |
| projection | **11.12 GB** | 108 s | 1 region | dense-score mega-region (low R² by design) | `projection_chr22_fit.json` |

Fit RAM is measured **without export** — serializing the ~4 GB dense-score imputation artifact
inflates peak RSS ~10× (to ~33 GB), which is an *export-path* cost, not the streaming-fit
footprint. Imputation is the lean scalable profile; projection on a *uniformly dense* score merges
into one ~11 GB mega-region Gram per chromosome (documented in `../streaming/README.md` — **use
imputation for uniformly dense scores**).

## 2. export → reload → predict golden (`atol=1e-12`)

The deployable artifact reproduces the in-memory model at real chr22 scale: reloading the exported
JSON and re-scoring a fixed probe matches the child model bit-for-bit — **imputation Δ=0.0,
projection Δ=0.0** (`verify_integration.py --part golden`; also covered continuously by
`tests/test_golden.py` and `tests/test_scale_integration.py`).

## 3. Reference CV — additive single-pass vs refit-per-fold (real chr22 dosages)

`reference_cv_bounded_chr22.json` — chr22:16–18 Mb, **1,125 PRS variants, 3,202 samples, 3 folds**:

| method | additive (streaming) | refit oracle (dense) | wall ratio | parity \|Δmean_r2\| | temp-VCF |
|---|---|---|---|---|---|
| imputation | 8.9 s (r²=0.7931) | 167.0 s (r²=0.7931) | 18.8× | **4.4e-16** | none |
| projection | 1.0 s (r²=0.2411) | 22.6 s (r²=0.2411) | 22.3× | **0.0 (exact)** | none |

The additive single accumulation pass (`S_train(k) = S_full − S_fold(k)`) reproduces the
refit-per-fold metrics to **machine precision on real 1000G dosages**, while streaming the panel
once instead of per-fold. The wall ratio combines two effects (additive-vs-refit *and*
streaming-vs-dense-oracle); the additive-specific k-fold saving — the additive path streams a fixed
number of times regardless of `n_folds` — is extrapolated in the JSON (10-fold → ~10× fewer passes).
The Phase-4 temp-VCF guard passed (no temp file written).

### Scope note
The full 3,202-sample × whole-chr22 refcv is ~35 min of per-variant ElasticNet solves (the solve is
`n_samples`-independent, so subsampling samples does **not** shorten it) and did not complete in
this environment (long-lived benchmark jobs were terminated externally). The additive == refit
parity is identical math at any variant count — proven hermetically by
`tests/test_scale_integration.py` and `tests/test_cv_stats.py` — so the bounded region confirms it
on REAL chr22 data within budget, and the k-fold saving extrapolates.

## 4. 500K extrapolation

`projection.json` — streaming-imputation peak-vs-n (chr22 PGS000027, ~27,854 variants), pooling the
Phase-2 RAM-sweep points (n ∈ {500, 1000, 2000, 3202}):

- empirical fit **3.24 GB + 135 KB/sample** (R²=0.99) → 500K ≈ **71 GB**; analytical band model ≈ **2 GB**.
- Both are **GB, not TB**. At the full **500K × 2M** target one dense float32 copy is **~4 TB** (peak
  ~24 TB across working copies + `cv_predictions`); the streaming band-buffer path stays ≈ **2 GB** —
  the sample dimension is contracted.

`make_report`'s blocker-grid multilinear `(s, v)` model is intentionally **not** used for a
single-chromosome run — it is degenerate on one variant-axis point and conflates dense/streaming
footprints (it projected a nonsensical −24 TB). The principled chr22-scoped estimate is the
streaming 1D-in-n peak fit + the analytical byte model, mirroring `../streaming/`.

## Reproduce

```bash
.venv/bin/python -m benchmarks.verify_integration --part refcv-bounded  # real-data CV parity (~5 min)
.venv/bin/python -m benchmarks.verify_integration --part project        # 500K extrapolation (instant)
.venv/bin/python -m benchmarks.verify_integration --part fit            # clean both-method fit RAM (~11 min)
.venv/bin/python -m benchmarks.verify_integration --part golden         # real-scale export round-trip (~15 min, ~20 GB)
.venv-gpu/bin/python -m benchmarks.verify_integration --part mps         # CPU vs MPS parity (torch venv)
```
