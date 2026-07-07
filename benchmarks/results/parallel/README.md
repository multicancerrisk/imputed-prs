# Phase 7 — process fan-out scaling

`streaming_fit_scaling.json` — micro-scaling of the chromosome-sharded streaming
imputation fit (`n_workers ∈ {1,2,4,8}`) on a small synthetic 8-chromosome panel
(2000 samples × 6320 variants, 720 imputed models). Produced by
`python -m benchmarks.verify_parallel_scale`. Per the plan's Phase 5–9 rule this is a
micro-benchmark + extrapolation, not a 500K/2M run; **bit-identity across worker counts**
is proven separately in `tests/test_parallel_streaming.py`.

| n_workers | wall (s) | speedup | efficiency |
|-----------|---------:|--------:|-----------:|
| 1 | 18.99 | 1.00× | 1.00 |
| 2 | 10.64 | 1.78× | 0.89 |
| 4 |  6.70 | 2.83× | 0.71 |
| 8 |  3.88 | 4.89× | 0.61 |

Near-linear at low worker counts; Amdahl serial fraction **f ≈ 0.116** (plan build, source
setup, and the canonical reduce are serial). Extrapolated to the machine's 6 performance
cores over 22 autosomes: Amdahl ceiling ≈ 3.79×, chromosome load-imbalance factor ≈ 1.09
(⌈22/6⌉ tail rounding) → **projected ≈ 3.5× speedup**. The ceiling is `min(P-cores,
n_chromosomes)` × imbalance; sub-chromosome window sharding (out of scope — it breaks
bit-identity) would raise it. Process fan-out is a CPU accelerator, orthogonal to the
Phase-3 GPU path (GPU stays single-process).
