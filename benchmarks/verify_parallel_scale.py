"""Phase 7 — process fan-out micro-scaling (no 500K/2M run).

Times a streaming imputation fit over a small synthetic multi-chromosome panel at
n_workers in {1,2,4,8}, reports speedup / efficiency / an Amdahl serial-fraction fit, and
extrapolates to the 22-autosome / performance-core operating point. Chromosome shards are
the parallel unit, so the panel carries N_CHROM independent chromosomes; per-chromosome
compute is sized to dominate pool + pickle overhead.

Run:  .venv/bin/python -m benchmarks.verify_parallel_scale
Correctness (bit-identity across worker counts) is covered by tests/test_parallel_streaming.py;
this script only measures wall-clock and is exempt from the full-scale exercise per the
plan's Phase 5-9 rule.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from imputed_prs import LinearImputationPRS
from imputed_prs.compute.parallel import _performance_cores
from imputed_prs.core.types import GenotypeData

RESULTS = Path(__file__).resolve().parent / "results" / "parallel"

N_SAMPLES = 2000
N_CHROM = 8
CHIP_PER_CHROM = 700
TARGETS_PER_CHROM = 90
SPACING = 6_000  # bp between variants → ~1 Mb window spans ~160 predictors
WORKER_COUNTS = (1, 2, 4, 8)
SEED = 7


def _build_panel():
    """A synthetic N_CHROM-chromosome 0/1/2 panel: CHIP_PER_CHROM chip + TARGETS missing."""
    rng = np.random.default_rng(SEED)
    vrows, cols, prows, platform = [], [], [], []
    per = CHIP_PER_CHROM + TARGETS_PER_CHROM
    for c in range(1, N_CHROM + 1):
        chrom = str(c)
        # Interleave targets among chip variants so every target has in-window predictors.
        target_slots = set(
            np.linspace(0, per - 1, TARGETS_PER_CHROM, dtype=int).tolist()
        )
        for i in range(per):
            pos = 1_000_000 + i * SPACING
            vid = f"{chrom}:{pos}"
            vrows.append({
                "variant_id": vid, "chromosome": chrom, "position": pos,
                "ref_allele": "A", "alt_allele": "G",
            })
            freq = rng.uniform(0.05, 0.95)
            cols.append(rng.binomial(2, freq, size=N_SAMPLES).astype(np.float32))
            prows.append({
                "variant_id": vid, "chromosome": chrom, "position": pos,
                "effect_allele": "G", "other_allele": "A", "beta": float(rng.normal(0, 0.1)),
            })
            if i not in target_slots:
                platform.append(vid)
    gd = GenotypeData(
        dosage_matrix=np.stack(cols, axis=1),
        variant_info=pd.DataFrame(vrows),
        sample_ids=[f"S{i}" for i in range(N_SAMPLES)],
    )
    return gd, pd.DataFrame(prows), platform


def _time(fn: Callable[[], object], reps: int = 2) -> float:
    fn()  # warm: spawn the loky pool, allocate, prime BLAS
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _fit_once(panel, n_workers: int):
    gd, prs, plat = panel
    m = LinearImputationPRS(
        backend="streaming", tuning_scope="none", cv_folds=5,
        random_state=17, n_workers=n_workers, verbose=0,
    )
    m.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
          genome_build="GRCh37")
    return m


def _amdahl_serial_fraction(speedups: Dict[int, float]) -> float:
    """Least-squares f for S(p) = 1/(f + (1-f)/p) over the measured p>1 points."""
    ps = [p for p in speedups if p > 1 and speedups[p] > 0]
    fs = [(p / speedups[p] - 1.0) / (p - 1.0) for p in ps]
    return float(np.clip(np.mean(fs), 0.0, 1.0)) if fs else float("nan")


def main() -> Dict:
    panel = _build_panel()
    n_variants = len(panel[0].variant_info)
    base = _fit_once(panel, 1)
    n_models = len(base._imputed_models)
    print(f"panel: {N_SAMPLES} samples x {n_variants} variants x {N_CHROM} chrom; "
          f"{n_models} imputed models")

    walls: Dict[int, float] = {}
    for k in WORKER_COUNTS:
        walls[k] = _time(lambda: _fit_once(panel, k))
        print(f"  n_workers={k}: {walls[k]:.3f}s")

    t1 = walls[1]
    speedup = {k: t1 / walls[k] for k in WORKER_COUNTS}
    efficiency = {k: speedup[k] / k for k in WORKER_COUNTS}
    f = _amdahl_serial_fraction(speedup)

    # Extrapolate to the real operating point: 22 autosomes over the machine's P-cores.
    p_cores = _performance_cores()
    workers_real = min(p_cores, 22)
    amdahl_ceiling = 1.0 / (f + (1.0 - f) / workers_real) if not np.isnan(f) else None
    # Chromosome load imbalance (chr1 ~5x chr22) caps real speedup below the Amdahl ideal;
    # 22 chromosomes / p workers with a ~2x max/mean size ratio → ceil(22/p) rounds tail up.
    import math
    imbalance = math.ceil(22 / workers_real) / (22 / workers_real)
    projected_real = (amdahl_ceiling / imbalance) if amdahl_ceiling else None

    out = {
        "config": {
            "n_samples": N_SAMPLES, "n_chrom": N_CHROM, "n_variants": n_variants,
            "n_imputed_models": n_models, "worker_counts": list(WORKER_COUNTS),
            "reps": 2, "seed": SEED,
        },
        "wall_seconds": walls,
        "speedup": speedup,
        "efficiency": efficiency,
        "amdahl_serial_fraction": f,
        "extrapolation": {
            "performance_cores": p_cores,
            "workers_real": workers_real,
            "amdahl_ceiling": amdahl_ceiling,
            "chrom_imbalance_factor": imbalance,
            "projected_speedup_22_autosomes": projected_real,
            "note": (
                "Process fan-out is a CPU accelerator orthogonal to the GPU path. "
                "Real speedup is bounded by min(P-cores, n_chromosomes) and chromosome "
                "size imbalance; sub-chromosome sharding (out of scope, breaks bit-identity) "
                "would lift the ceiling."
            ),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "streaming_fit_scaling.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Amdahl serial fraction f={f:.3f}; projected {workers_real}-worker speedup "
          f"~{projected_real:.2f}x" if projected_real else "no projection")
    print(f"wrote {RESULTS / 'streaming_fit_scaling.json'}")
    return out


if __name__ == "__main__":
    main()
