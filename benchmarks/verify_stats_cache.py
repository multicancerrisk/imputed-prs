"""Phase 9 commit 6 — micro-benchmark for the opt-in stats cache (io/stats_cache.py).

A streaming sensitivity / re-tune over an (alpha, l1, window) grid pays, per window_size,
one O(n·band²) accumulation pass (decode + band Gram + block assembly) to reach the raw
Gram blocks, then re-solves them cheaply per combo. The blocks are alpha/l1-independent,
so the opt-in disk cache lets a SECOND invocation on the same panel skip every
accumulation pass and load the blocks instead — re-solving from disk.

This script measures the three per-unit costs on a synthetic streaming panel:

    t_collect   one accumulation pass  (paid W times on a cold run, 0 on a warm run)
    t_load      load one window's blocks from disk  (paid W times on a warm run)
    t_solve     solve one window's blocks at one (alpha, l1)  (paid by every combo, both)

and composes them into the cold vs warm wall-clock for a default 3×3×3 grid (3 window
sizes → 3 accumulation groups, 27 solves), then EXTRAPOLATES: t_collect scales ~linearly
in sample count while t_load (disk-bound) and t_solve (sample-free, O(band³)) do not, so
the warm-run speedup grows with n. Correctness (a cache-loaded block re-solves to the same
coefficients as the live block) is asserted inline; bit-parity is covered by
tests/test_stats_cache.py + tests/test_evaluator.py::TestSensitivityStatsCache.

Run:  .venv/bin/python -m benchmarks.verify_stats_cache
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from imputed_prs.compute.sufficient_stats import (
    ObservedVar,
    StreamPlan,
    StreamingImputationFitter,
    TargetVar,
)
from imputed_prs.io.stats_cache import (
    load_collected,
    make_stats_key,
    store_collected,
)

RESULTS = Path(__file__).resolve().parent / "results" / "stats_cache"

# Grid shape the extrapolation composes over (the sensitivity_analysis default grid).
N_WINDOWS = 3          # window_size values → one accumulation group each
N_COMBOS = 27          # 3 window × 3 l1 × 3 alpha
BENCH_N = 4000         # bench sample count (kept small; the point is the scaling law)
SCALE_N = 500_000      # extrapolation target
N_VARIANTS_PER_CHROM = 40
N_CHROMS = 4
ALPHA, L1_RATIO = 0.01, 0.5
SEED = 7


@dataclass
class _Block:
    variant_info: pd.DataFrame
    dosages: np.ndarray

    @property
    def n_variants(self):
        return self.dosages.shape[1]


class _MemSource:
    """Minimal in-memory streaming source: position-sorted blocks per chromosome."""

    def __init__(self, info, dosage, sample_ids, block_size=16):
        self._info = info.reset_index(drop=True)
        self._dos = dosage
        self._sample_ids = list(sample_ids)
        self._bs = block_size

    @property
    def sample_ids(self):
        return self._sample_ids

    def iter_variant_blocks(self, region=None, block_size=None):
        bs = block_size or self._bs
        info = self._info
        if region is not None:
            rows = np.nonzero((info["chromosome"].astype(str) == str(region)).to_numpy())[0]
        else:
            rows = np.arange(len(info))
        rows = rows[np.argsort(info["position"].to_numpy()[rows], kind="stable")]
        for s in range(0, len(rows), bs):
            sel = rows[s : s + bs]
            yield _Block(info.iloc[sel].reset_index(drop=True), self._dos[:, sel])


def _build_panel(n_samples, seed):
    rng = np.random.RandomState(seed)
    records = []
    for c in range(1, N_CHROMS + 1):
        for i in range(N_VARIANTS_PER_CHROM):
            records.append(
                dict(variant_id=f"v{c}_{i}", chromosome=str(c),
                     position=100_000 * (i + 1), ref_allele="A", alt_allele="G")
            )
    info = pd.DataFrame(records)
    freqs = rng.uniform(0.15, 0.85, size=len(info))
    dosage = rng.binomial(2, freqs, size=(n_samples, len(info))).astype(np.float32)
    sample_ids = [f"s{i}" for i in range(n_samples)]
    return info, dosage, sample_ids


def _build_plan(info, sample_ids, window_size):
    """Chip = ~80% of variants; PRS = every other variant (some observed, some missing)."""
    n = len(info)
    chip_rows = [i for i in range(n) if i % 5 != 2]
    platform_info = info.iloc[chip_rows].reset_index(drop=True)
    chip_ids = {info.iloc[r]["variant_id"]: pi for pi, r in enumerate(chip_rows)}
    targets, observed = {}, {}
    rng = np.random.RandomState(SEED)
    for r in range(0, n, 2):
        vid = info.iloc[r]["variant_id"]
        beta = float(rng.uniform(-0.6, 0.6))
        if vid in chip_ids:
            observed[vid] = ObservedVar(beta=beta, effect_flip=False)
        else:
            targets[vid] = TargetVar(
                prs_variant_id=vid, chromosome=str(info.iloc[r]["chromosome"]),
                position=int(info.iloc[r]["position"]), effect_allele="G",
                other_allele="A", beta=beta, effect_flip=False,
            )
    return StreamPlan(
        sample_ids=sample_ids, platform_variant_info=platform_info, chip_ids=chip_ids,
        targets=targets, observed=observed, window_size=window_size, max_predictors=None,
        alpha=ALPHA, l1_ratio=L1_RATIO, cv_folds=5, random_state=SEED,
    )


def _best(fn, reps=5):
    fn()  # warm caches
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _key_for(plan, sample_ids):
    tokens = [f"c:{c}" for c in plan.chip_ids] + [f"t:{t}:0" for t in plan.targets]
    return make_stats_key(
        sample_ids=sample_ids, chip_ids=tokens, window_size=plan.window_size,
        max_predictors=None, cv_folds=5, random_state=SEED, n_variants=len(sample_ids),
    )


def main() -> Dict:
    print(f"stats-cache micro-bench: n={BENCH_N} variants={N_CHROMS * N_VARIANTS_PER_CHROM} "
          f"grid={N_COMBOS} combos / {N_WINDOWS} windows")
    info, dosage, sample_ids = _build_panel(BENCH_N, SEED)
    plan = _build_plan(info, sample_ids, window_size=1_000_000)
    source = _MemSource(info, dosage, sample_ids)
    fitter = StreamingImputationFitter(plan)
    tmp = Path(tempfile.mkdtemp())

    t_collect = _best(lambda: fitter.run_collect(source), reps=3)
    collected = fitter.run_collect(source)
    key = _key_for(plan, sample_ids)
    store_collected(key, collected, cache_dir=tmp)

    t_load = _best(lambda: load_collected(key, cache_dir=tmp))
    t_solve = _best(lambda: fitter.solve_collected(collected, ALPHA, L1_RATIO))

    # Correctness: a cache-loaded block re-solves to the SAME coefficients as the live one.
    loaded = load_collected(key, cache_dir=tmp)
    live_m = fitter.solve_collected(collected, ALPHA, L1_RATIO).models
    warm_m = fitter.solve_collected(loaded, ALPHA, L1_RATIO).models
    max_diff = max(
        (float(np.abs(warm_m[v].coefficients - live_m[v].coefficients).max())
         for v in live_m if live_m[v].coefficients.size),
        default=0.0,
    )
    assert max_diff == 0.0, f"cache-loaded solve diverged by {max_diff}"

    # Compose cold vs warm wall-clock for the default grid.
    cold = N_WINDOWS * t_collect + N_COMBOS * t_solve
    warm = N_WINDOWS * t_load + N_COMBOS * t_solve
    speedup = cold / warm

    # Extrapolate: t_collect ∝ n (decode + band Gram); t_load / t_solve ~ n-independent.
    scale = SCALE_N / BENCH_N
    t_collect_scaled = t_collect * scale
    cold_s = N_WINDOWS * t_collect_scaled + N_COMBOS * t_solve
    warm_s = N_WINDOWS * t_load + N_COMBOS * t_solve
    speedup_scaled = cold_s / warm_s

    print(f"  t_collect  {t_collect*1e3:8.2f} ms   (per window; paid W×  cold, 0× warm)")
    print(f"  t_load     {t_load*1e3:8.2f} ms   (per window; warm hit)")
    print(f"  t_solve    {t_solve*1e3:8.2f} ms   (per combo;  both)")
    print(f"  correctness: cache-loaded coefficients bit-identical (max diff {max_diff:.1e})")
    print(f"  grid cold {cold*1e3:7.1f} ms  vs warm {warm*1e3:7.1f} ms  →  {speedup:5.2f}x "
          f"(2nd invocation)")
    print(f"  extrapolated to n={SCALE_N:,} (t_collect ×{scale:.0f}):  "
          f"cold {cold_s:7.1f} s  vs warm {warm_s*1e3:6.1f} ms  →  {speedup_scaled:6.1f}x")

    out = {
        "config": {
            "bench_n": BENCH_N, "scale_n": SCALE_N, "n_variants": len(info),
            "n_windows": N_WINDOWS, "n_combos": N_COMBOS, "alpha": ALPHA,
            "l1_ratio": L1_RATIO, "seed": SEED,
        },
        "t_collect_s": t_collect, "t_load_s": t_load, "t_solve_s": t_solve,
        "grid_cold_s": cold, "grid_warm_s": warm, "speedup": speedup,
        "extrapolation": {
            "scale_factor": scale, "t_collect_scaled_s": t_collect_scaled,
            "grid_cold_s": cold_s, "grid_warm_s": warm_s, "speedup": speedup_scaled,
        },
        "max_abs_coef_diff": max_diff,
        "verdict": (
            f"On a warm 2nd invocation the disk cache elides all {N_WINDOWS} accumulation "
            f"passes, loading blocks in {t_load*1e3:.1f} ms each instead of streaming "
            f"({t_collect*1e3:.1f} ms each). At bench n={BENCH_N} the grid is {speedup:.2f}× "
            f"faster; because accumulation is O(n) while load/solve are ~n-independent, the "
            f"win grows to ~{speedup_scaled:.0f}× extrapolated to n={SCALE_N:,}. Cached "
            f"blocks re-solve bit-identically (max coef diff {max_diff:.0e}). Serves "
            f"coefficient + CV-R² reuse; per-sample calibration is not cached (it needs the "
            f"resident dosage band) and re-streams when required."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats_cache.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RESULTS / 'stats_cache.json'}")
    return out


if __name__ == "__main__":
    main()
