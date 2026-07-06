# Phase-3 GPU acceleration — curated results

Validation of `device="cpu"` vs `device="mps"` (Apple MPS; the code path is device-agnostic and
also targets CUDA, benchmark deferred to NVIDIA hardware) for the streaming sufficient-statistics
fit. Produced by `benchmarks/verify_gpu_scale.py` on the real GRCh38 1000G high-coverage panel
(3,202 samples), run under the torch venv:

```
.venv-gpu/bin/python -m benchmarks.verify_gpu_scale --part all
```

Fidelity bar = **statistical parity** (`rtol=5e-3`, `atol=1e-3`), not bit-parity; float32 on MPS is
exact for 0/1/2 dosages and within the sanctioned band after mean-imputation. The dense/small path
is always CPU/numpy, so the golden gate (`tests/test_golden.py`, exact `1e-12`) is untouched by
`device`.

## `parity_prs313.json` — end-to-end real-data parity (PRS-313, both methods)

`fit()` on `device="cpu"` vs `device="mps"`, compared field-by-field through the R²/calibration
oracle. n = 3,202, 109,089 reference positions.

| method     | parity                         | cpu wall | mps wall | speedup | cpu RSS | mps RSS |
|------------|--------------------------------|---------:|---------:|--------:|--------:|--------:|
| imputation | **OK** (40 fields, 0 outside)  | 121.5 s  | 101.7 s  | 1.20×   | 3.43 GB | 4.22 GB |
| projection | **OK** (61 fields, 0 outside)  | 121.0 s  |  98.2 s  | 1.23×   | 3.57 GB | 3.86 GB |

Every oracle field (counts, R² summary, calibration, provenance) matches within the parity band —
the GPU FISTA/Cholesky solve reproduces the CPU coordinate-descent optimum. MPS is modestly faster
here (batched solve + faster accumulation) even though at n=3,202 the O(n) accumulation is only
~1 % of the fit.

## `accumulation_scaling.json` — the O(n) accumulation win (where GPU actually pays off)

Isolated microbench of the two accumulation seams, CPU numpy vs device, across a sample sweep
(synthetic 0/1/2 dosages, 256 chip columns, K=5, T=64). These are `O(n)` with a fixed band, so the
per-sample slope extrapolates cleanly to the 500K production scale.

| seam                          | n=3,202 | n=25,000 | n=100,000 | n=500,000 | 500K extrapolation |
|-------------------------------|--------:|---------:|----------:|----------:|--------------------|
| **B** band-Gram `add` (ZᵀZ)   | 2.23×   | 5.41×    | 4.15×     | 4.02×     | 4.0× (cpu 3.60 s → gpu 0.89 s) |
| **A** cross-product `C=ZᵀY`   | 0.20×   | 2.55×    | 4.79×     | 18.85×    | 18.1× (cpu 0.055 s → gpu 0.003 s) |

Seam B (band-Gram maintenance, the `_GpuChipGramBuffer.add_batch` GEMM vs numpy per-column) wins at
every scale. Seam A (the chunk cross-product) is memory-bound and *loses* at n=3,202 (transfer
overhead) but crosses over by n=25,000 and reaches ~19× at 500K — this is the seam that dominates
the fit at production scale.

## The crossover / size-guard finding

The per-unit local solve is **n-independent** (it runs off the Gram) and, on MPS, **launch-bound**:
each tiny per-unit FISTA/Cholesky solve is a burst of small kernels whose launch overhead the
M-series' Accelerate BLAS avoids on CPU. Consequences, measured:

- A **dense** score (PGS000027 chr22, ~24,600 trained units) issues ~24,600 launch-bound solves and
  does **not complete in budget** on MPS — de-risk at n=500 timed out (> 900 s) with **bounded**
  peak RSS (5.2 GB; an earlier bug hit 63 GB — fixed, see below). Because the solve is n-independent,
  this is a *unit-count* effect, not a memory or large-n effect.
- A **sparse** score (PRS-313, ~227/161 units) completes fine and MPS is ~1.2× faster at n=3,202.

So `device="auto"` engages the GPU only when `n ≥ GPU_AUTO_MIN_SAMPLES` (**25,000**, aligned with the
Seam-A crossover); below that it stays on CPU, which avoids the dense-score timeout at the cost of a
marginal sparse-score speedup. Explicit `device="mps"`/`"cuda"` is always honored. At the 500K target
the O(n) accumulation dominates and the GPU wins net; that win is shown by the accumulation
extrapolation above, not by the (accumulation-light) n=3,202 end-to-end.

## OOM fix (device memory is bounded regardless of chunk width)

The chunk kernel originally gathered *every* co-windowed unit's `(K, p, p)` fold-Gram onto the device
before solving; on a wide chunk (small n ⇒ large batch width) this ballooned MPS to 63 GB and OOM'd.
The kernel now gathers **and** solves in device-memory-bounded sub-batches (`_SOLVE_GRAM_BUDGET`,
sorted by predictor count to minimize padding), so only one sub-batch's Grams are ever resident —
peak is bounded (5.2 GB at the de-risk point) at any n or chunk width. Result-invariant: padding is
masked out and results are stored by original unit index.

## Phase-3E band-limited per-fold Gram (projection memory)

Finding #1: on a dense projection score the merged regions span whole chromosome arms, so the
band buffer's per-fold Gram `Ghold = (K, cap, cap)` (with `cap` ~ 7.7k predictors) ballooned to
~12 GB — independent of `n`. The fix is a projection-only **lazy per-fold Gram**: the buffer keeps
only the band `Z` + column moments and never allocates `Gfull`/`Ghold`; `gather` recomputes the
full + per-fold Gram on-demand from `Z` over each region's ≤`max_predictors` fit predictors
(imputation keeps the incremental path — many small units). Measured on the real GRCh38 1000G
chr20-22 projection fit (`verify_streaming_scale.py --part finding1`, n=3,202) against the 11.51 GB
Phase-2 baseline (`../streaming/projection_chr20-22.json`):

| `max_predictors` | peak RSS | vs. baseline | wall | calibration R² |
|------------------|---------:|-------------:|-----:|---------------:|
| `None` (apples-to-apples) | **8.90 GB** | 1.29× lower | 359.8 s | 0.6167005036 (identical to 16 digits) |
| `500` (recommended)       | **2.38 GB** | 4.84× lower | 290.8 s | 0.6250 (≥ baseline)                    |

The `None` run reproduces the baseline calibration **bit-for-bit** (the high-coverage panel is
complete 0/1/2 dosages, so integer Gram sums are order-invariant in float64) — a rigorous parity
proof. Its 1.29× is modest because a 7,673-predictor `(K, P, P)` solve is inherently large (the
"use imputation for dense scores" anti-pattern; 3/4 regions intercept-only). The real win: the
per-fold Gram now tracks the **fit** size, so `max_predictors` is an effective RAM control (pre-fix
the buffer allocated `(K, cap, cap)` regardless of it) — 4.84× lower at `max_predictors=500`, with
*higher* R² (the distant predictors were noise). Full result: `../streaming/finding1_projection_gram.json`.

## Raw artifacts

`raw/` (git-ignored) holds the per-cell `child_result.json` measurement artifacts from the isolated
subprocess runs (`benchmarks.harness.measure`). Only the curated `*.json` + this README are committed.

CUDA: the torch kernels are device-agnostic (`.to(device)`) and exercised via MPS + the torch-CPU
proxy; hardware throughput on NVIDIA is deferred (no CUDA device on this machine).
