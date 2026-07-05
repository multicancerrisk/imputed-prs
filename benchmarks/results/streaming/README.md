# Phase-2 streaming backend — bounded scale validation

Real GRCh38 1000G high-coverage panel (3,202 samples), produced by
`benchmarks/verify_streaming_scale.py`. Every fit runs in an isolated subprocess
(`benchmarks.harness.measure`; peak RSS via `os.wait4`, `PYTHONHASHSEED` pinned), so peak
RSS is authoritative.

## 1. Parity — streaming reproduces the dense oracle (PRS-313 / PGS000004)
`parity_prs313.json` — dense vs streaming, both methods, 3,202 samples, 109,089 reference
positions. Oracle (counts, R² summary, calibration, provenance) compared field-by-field at
rtol=5e-3 / atol=1e-3:

| method | fields | mismatches | verdict |
|---|---|---|---|
| imputation | 40 | 0 | **OK** |
| projection | 61 | 0 | **OK** |

Streaming peak RSS was *below* dense here (imputation 3.37 vs 5.50 GB; projection 3.53 vs
5.87 GB) — the full dosage matrix is never materialized. This closes the "parity holds on
PRS-313" Done-when on **real** data (previously only synthetic panels).

## 2. Imputation at real scale (chr22 PGS000027, 27,854 variants)
`imputation_chr22.json` — `backend="streaming"`, 3,202 samples:
- peak RSS **3.68 GB** (authoritative), wall 567 s
- 24,597 models trained, mean imputation R² **0.773** (median 0.832)
- calibration R² 0.952, scaling ≈ 1.02 — high quality
- throughput ≈ **43 variants/s** (per-unit Gram coordinate descent; batched solves are the
  Phase-3 lever)

## 3. Projection at genome-fraction scale (chr20-22 PGS000027, 109,252 variants)
`projection_chr20-22.json` — `backend="streaming"`, 3,202 samples:
- peak RSS **11.5 GB**, wall 360 s — completes within laptop RAM
- the dense score merges into only **4 regions** (mean 24,159 PRS variants / 7,673 chip
  predictors per region) → see the finding below.

## Finding: projection on *uniformly dense* scores degenerates into mega-regions
On PGS000027 the ±1 Mb windows merge into ~one region per chromosome. The streaming chip
band buffer then holds a per-fold Gram `Ghold = (K, cap, cap)` whose `cap` is the region's
chip-column count (thousands), so RAM is dominated by this Gram — **~12 GB on chr22,
independent of n_samples** (`ram_extrapolation_projection.json`: flat across
n ∈ {500, 1000, 2000, 3202}) — and the mega-region fit is itself low-R² (~0.01). This is a
property of the *projection method* on dense scores (present in both backends; the streaming
path adds the O(K·cap²) memory), not a correctness defect — parity (§1) holds.
**Use `LinearImputationPRS` for uniformly dense scores** (lean + high-R², §2); projection is
for scores whose PRS variants cluster into a modest number of separated regions (§1, PRS-313).
`max_predictors` caps the *fit* predictors but not the buffered `cap`; a band-limited
per-fold Gram is a deferred Phase-3 optimization.

## 4. Peak-RAM vs n_samples → 500K extrapolation
`ram_extrapolation_imputation.json` (the scalable per-variant path) and
`ram_extrapolation_projection.json` (dense-score mega-region characterization).

Imputation streaming peak RSS grows gently and linearly over n ∈ {500, 1000, 2000, 3202}:
**3.33 → 3.36 → 3.51 → 3.68 GB** (fit `a = 3.24 GB` fixed + `b = 135 KB/sample`, R² = 0.99).
Two extrapolations to 500K bracket the answer:
- analytical band + accumulator model: **~2 GB** (band cols × n + O(n) scalars);
- empirical linear fit: **~71 GB** (the measured per-sample slope — dominated by n-dependent
  block-read / cyvcf2 / per-batch buffers — extrapolated 156× beyond the data).

Both are **GB, not the dense ~0.41 TB wall** (dense/streaming ≈ 200× by the analytical model).
The sample dimension is contracted: no n × n_variants dosage matrix is resident
(`collect_reference_variant_info` keeps only variant metadata; the fit holds an O(n × cap) band
buffer). Tightening the 2–71 GB bracket — profiling the n-dependent transients and measuring at
larger n — is a Phase-6 item.

## Deferred to Phase 6 (full-scale integration)
The full 22-chromosome / 2.1M-variant fit on all 3,202 samples + 10-fold reference CV is the
Phase-6 job. (The cached all-autosome prefilter under `benchmarks/data/work/` is a partial,
aborted-blocker extract — 109,737 variants, fewer than chr20-22 alone — and was not used.)
