"""Micro-benchmark: ChromosomeIndex window-query throughput at panel scale.

Builds a ~633K-variant index (23andMe-chip scale) and runs millions of window
queries, then contrasts against the legacy O(n_variants)-per-call
filter_to_local_window on a small sample (extrapolated). Demonstrates the
O(log n) query cost the Phase-2 Gram band-limiting relies on.

Standalone (time.perf_counter), not a pytest gate. Run:
    .venv/bin/python -m benchmarks.bench_window_index [n_queries]
"""

import sys
import time

import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import filter_to_local_window
from imputed_prs.core.window_index import ChromosomeIndex

N_INDEX = 633_000       # 23andMe v5 chip scale
CHROM_LEN = 250_000_000


def build_panel(n, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "variant_id": [f"rs{i}" for i in range(n)],
        "chromosome": rng.randint(1, 23, size=n).astype(str),
        "position": rng.randint(1, CHROM_LEN, size=n),
    })


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    panel = build_panel(N_INDEX)

    t = time.perf_counter()
    idx = ChromosomeIndex(panel)
    t_build = time.perf_counter() - t

    rng = np.random.RandomState(1)
    q_chrom = rng.randint(1, 23, size=n_queries).astype(str)
    q_pos = rng.randint(1, CHROM_LEN, size=n_queries)

    t = time.perf_counter()
    hits = 0
    for i in range(n_queries):
        hits += idx.window(q_chrom[i], int(q_pos[i])).n_variants
    t_idx = time.perf_counter() - t

    # Legacy oracle on a small sample -> extrapolate to n_queries.
    n_legacy = 200
    t = time.perf_counter()
    for i in range(n_legacy):
        filter_to_local_window(q_chrom[i], int(q_pos[i]), panel)
    t_legacy_sample = time.perf_counter() - t
    legacy_full = t_legacy_sample / n_legacy * n_queries

    print(f"panel: {N_INDEX:,} variants | index build: {t_build:.2f}s")
    print(f"ChromosomeIndex: {n_queries:,} queries in {t_idx:.2f}s "
          f"({n_queries / t_idx / 1e6:.2f}M q/s, mean {hits / n_queries:.0f} hits)")
    print(f"legacy filter_to_local_window: {t_legacy_sample / n_legacy * 1e3:.1f} ms/query "
          f"-> {legacy_full / 60:.0f} min for {n_queries:,} queries")
    print(f"speedup: {legacy_full / t_idx:.0f}x")


if __name__ == "__main__":
    main()
