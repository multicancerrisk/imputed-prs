# `benchmarks/` — Phase 0 measurement rig

Dev-only tooling (not shipped in the `imputed_prs` wheel; never imported by the library).
It measures wall-clock and **peak RSS**, attributes memory to source lines, captures a
statistical-parity **oracle**, demonstrates the scaling blocker on real data, and
extrapolates time/memory to 500K samples × 2M variants.

All verification is on **GRCh38** (the All of Us production build): the NYGC 1000G
high-coverage 30x panel (3,202 samples) + PGS000027 (BMI, ~2.1M variants). Nothing under
`imputed_prs/` is modified.

## Layout
| file | role |
|---|---|
| `harness.py` | `measure()` / `sweep()` — run a unit of work in an isolated subprocess; peak RSS via `os.wait4`, graceful OOM/kill/timeout capture; result schema + JSON IO |
| `_child.py` | the isolated worker (`python -m benchmarks._child <run_dir>`) |
| `meters.py` | in-child: sub-phase timing, tracemalloc per-site attribution, soft-memory watchdog |
| `scenarios.py` | op dispatch (`load_genotypes`/`fit`/`predict` + hermetic self-tests) + analytical byte model |
| `prefilter.py` | self-contained bcftools prefilter (`view -R` index seek) + sample subset |
| `grid.py` | blocker growth grid (variants × samples, + load-only isolation) |
| `oracle.py` | oracle snapshot extraction + `compare_oracle` parity checker |
| `run_baseline.py` | CLI driver (`confirm` / `baseline` / `blocker` / `profile` / `all`, `--smoke`) |
| `scaling_projection.py` | fit + extrapolate peak-RSS / wall-time to 500K×2M |
| `smoke.py` | hermetic end-to-end shakedown on synthetic data |
| `data_prep/` | `download_1kg_grch38.sh`, `liftover_chip.py` |

## One-time setup
```bash
.venv/bin/pip install -e ".[bench]"                       # pyliftover (+ .[plotting] for PNGs)
benchmarks/data_prep/download_1kg_grch38.sh               # ~20-30 GB GRCh38 panel -> benchmarks/data/1kg_grch38/
.venv/bin/python -m benchmarks.data_prep.liftover_chip    # GRCh37 chip -> benchmarks/data/23andme_v5_GRCh38_variants.txt
```
`bcftools` must be on PATH for the real-data prefilter.

## Run
```bash
.venv/bin/python -m benchmarks.run_baseline --smoke        # fast, no panel needed (proves plumbing)
.venv/bin/python -m benchmarks.run_baseline --mode confirm # PGS000027/PGS000004 variant counts (GRCh38)
.venv/bin/python -m benchmarks.run_baseline --mode baseline # oracle -> results/baseline/{method}.json
.venv/bin/python -m benchmarks.run_baseline --mode blocker  # grow until OOM; attributes the failure
.venv/bin/python -m benchmarks.run_baseline --mode profile  # cProfile the pandas hotspots
.venv/bin/python -m benchmarks.scaling_projection --plot    # extrapolate to 500K x 2M
```

## Notes
- **Reproducibility:** the harness pins `PYTHONHASHSEED=0` in the child. The library's fit
  has a set/dict iteration-order dependence beyond `random_state` (calibration/R² drift
  ~1e-3 across hash seeds); pinning it keeps the parity oracle exact.
- **Peak RSS is authoritative only with tracemalloc off** (tracemalloc inflates RSS and
  misses cyvcf2/BLAS C buffers). Attribution runs (`--tracemalloc`) flag this.
- **Results:** raw per-run JSON in `results/raw/<run_id>/` (git-ignored); curated oracle in
  `results/baseline/` (committed for cross-phase diffing).
