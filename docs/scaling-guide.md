# Scaling guide

`imputed-prs` trains linear imputation / projection models on a reference panel so a PRS can be
computed from a fixed DTC chip. Training the toy path materializes the whole reference dosage
matrix in RAM — at 2M variants × 500K samples one dense float32 copy is **~4 TB**. This guide
covers the streaming/GPU/parallel/checkpoint machinery (Phases 1–10 of the scaling refactor)
that trains the same models by **contracting the sample dimension into sufficient statistics**,
never materializing the full matrix.

The **browser export artifact is unchanged** — it is `#variants`-scale (sample-free) already.
Everything here is a train-time concern; `predict`/`export` are untouched.

> TL;DR — the defaults already scale. `LinearImputationPRS().fit(...)` auto-selects the streaming
> backend once the estimated dense matrix exceeds 8 GiB, runs on CPU (or GPU if you ask), and
> produces a byte-for-byte identical browser artifact.

---

## The `backend` knob — dense vs streaming vs auto

Set on the constructor: `LinearImputationPRS(backend="auto")` (default), `LinearProjectionPRS`
mirrors it.

| `backend` | What it does | RAM | When |
|---|---|---|---|
| `"dense"` | The original in-RAM oracle: materializes the dosage matrix. | ~`n_samples × n_variants × 4 B` (×several working copies) | small panels; the correctness oracle |
| `"streaming"` | Trains from band-limited sufficient statistics (`G = ZᵀZ`, `C = ZᵀX`) accumulated in one chunked pass; the matrix is never resident. | a chip-band buffer + O(n) accumulators (GB, not TB) | large panels |
| `"auto"` (default) | Streams when the estimated dense matrix `n_samples × n_needed_variants × 4 B` exceeds **8 GiB**; otherwise dense. PGEN input and a bare `GenotypeSource` always stream. | picks the above | almost always leave this |

Statistical parity, not bit-for-bit: streaming uses a Gram-based coordinate descent (equivalent to
sklearn's `precompute=True`), so it reproduces the dense path's R²/calibration within tolerance,
not sklearn coefficients bit-for-bit. The golden allele-orientation / export round-trip tests stay
exact (`atol=1e-12`).

**Streaming constraints** (validated at construction): `tuning_scope` must be `"none"` or
`"global"` (per-variant tuning has no sample-free analogue), and `exclude_ambiguous=False`.

---

## Input formats

`fit(reference_genotypes=...)` accepts a path **or** an in-memory object:

| Format | dense | streaming | Notes |
|---|---|---|---|
| VCF / BCF (`.vcf`, `.vcf.gz`) | ✓ | ✓ | streaming uses tabix region pushdown (no whole-file scan) |
| PLINK1 (`.bed`) | ✓ | — | dense only; `"auto"` falls back to dense |
| **PLINK2 PGEN (`.pgen`)** | — | ✓ | streaming-native, no dense reader; **always streams**. Install the `scale` extra (`pip install imputed-prs[scale]`, pulls `pgenlib`). This is the recommended production reader; convert your reference once with `plink2 --make-pgen`. |
| `GenotypeData` (in-RAM) | ✓ | ✓ | e.g. a CV fold |
| `GenotypeSource` (streaming) | — | ✓ | always streams; `backend="dense"` rejects it |

---

## The `device` knob — CPU, Apple MPS, NVIDIA CUDA

Set on the constructor: `device="auto"` (default) | `"cpu"` | `"mps"` | `"cuda"`. Requires the
`gpu` extra (`pip install imputed-prs[gpu]`, pulls `torch`); **the CPU path is fully functional
without torch** — `"auto"` resolves to CPU when torch is absent.

- `"auto"` probes CUDA → Apple MPS → CPU, and additionally routes fits below
  `GPU_AUTO_MIN_SAMPLES` (25,000 samples) to CPU (tiny per-unit GPU solves are launch-bound).
- **Only the streaming path is device-aware**; the dense path is always CPU.
- Where the GPU wins is the O(n) accumulation kernels (the band-Gram `add_batch` GEMM and the
  chunk cross-product `C = ZᵀY`) at 500K samples — not the launch-bound per-unit solve. See
  `benchmarks/results/gpu/README.md`. On Python 3.14 install torch in a separate venv if wheels lag.

---

## The `n_workers` knob — multi-core process fan-out

Set on the constructor: `n_workers=1` (default) | `-1` (performance cores) | `k`. Shards the
streaming accumulation + solves **by chromosome** across a process pool (chromosome shards are
zero-halo, so the reduce is order-independent → **bit-identical** artifact regardless of worker
count). CPU-only, orthogonal to `n_jobs` (the intra-fit sklearn thread pool). BLAS is pinned to
one thread inside workers to avoid oversubscription.

---

## `checkpoint_dir` — resumable long runs

`fit(..., checkpoint_dir="ckpt/")` and `cross_validate(..., checkpoint_dir="ckpt/")` (streaming
backend only) persist each per-chromosome partial as it completes; a killed run re-invoked with
the same `checkpoint_dir` (and fold partition) resumes to a **bit-identical** result. `None`
(default) → no disk I/O. Ignored on the dense backend.

---

## Streaming reference CV

`ImputationEvaluator(model).cross_validate(..., backend="streaming")` (and the new
`ProjectionEvaluator.cross_validate`) use **additive** leave-one-fold-out statistics: because the
sufficient statistics are sums over samples, each training fold's normal equations are
`S_full − S_fold(k)` — one accumulation pass replaces `k` independent refits (~k× less work for
k-fold CV). The refit-per-fold path is retained as the size-selected oracle. Exact on a
no-missing panel; on panels with missingness it is a mean-imputation-validated approximation.

---

## Method choice at scale

- **Imputation** (`LinearImputationPRS`) is the lean, high-R² scalable profile for **uniformly
  dense** scores (a PRS variant every few kb). RAM is a small chip band + O(n) accumulators.
- **Projection** (`LinearProjectionPRS`) is for scores whose PRS variants **cluster into a modest
  number of separated regions**. On a uniformly dense score the ±1 Mb windows merge into ~one
  mega-region per chromosome whose per-fold Gram dominates RAM (~12 GB on chr22, independent of
  n) and whose fit is low-R² — this is a property of the projection method on dense scores, not a
  defect. **Use imputation for uniformly dense scores.**

---

## Validation evidence

Curated benchmarks (real GRCh38 1000G high-coverage panel, 3,202 samples; peak RSS is
authoritative via isolated-subprocess measurement):

| Area | Dir | Headline |
|---|---|---|
| Streaming parity + scale | `benchmarks/results/streaming/` | PRS-313 dense-vs-streaming parity: 0 mismatches both methods. chr22 PGS000027 imputation: **3.68 GB** peak / mean R² **0.773**. |
| Full-scale integration (chr22) | `benchmarks/results/integration/` | Both methods end-to-end + 3-fold additive reference CV + export→reload→predict golden (Phase 11). |
| GPU | `benchmarks/results/gpu/` | CPU-vs-MPS streaming parity + O(n) accumulation speedup extrapolation. |
| Parallel | `benchmarks/results/parallel/` | `n_workers` process fan-out ~linear speedup, bit-identical. |
| Batched solve | `benchmarks/results/batched_solve/` | Shared-Gram multi-target solve speedup. |

**500K extrapolation (chr22 streaming imputation):** peak RSS grows as `3.24 GB + 135 KB/sample`
(R²=0.99) → an empirical ~71 GB vs an analytical ~2 GB at 500K (both **GB, not the dense TB
wall**). At the full **500K × 2M** target, one dense float32 copy is ~4 TB (peak ~24 TB across
working copies + `cv_predictions`); the streaming band-buffer path stays in the low GB — the
sample dimension is contracted.

---

## Reproducing

```bash
# Hermetic mini-1000G scale regression (seconds, no panel, no torch) — also the CI gate:
.venv/bin/pytest tests/test_scale_integration.py -q

# Bounded real-data scale validation (needs the GRCh38 1000G panel under benchmarks/data/):
.venv/bin/python -m benchmarks.verify_streaming_scale --part all      # Phase 2: parity + fit + RAM
.venv/bin/python -m benchmarks.verify_integration     --part all      # Phase 11: chr22 end-to-end
.venv-gpu/bin/python -m benchmarks.verify_integration --part mps      # CPU vs MPS (needs torch)
```
