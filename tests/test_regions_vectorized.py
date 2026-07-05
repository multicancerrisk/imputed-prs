"""Differential test: vectorized merge_variant_windows == legacy (Phase 1, D).

Only the per-variant window construction was vectorized; the sweep-merge is
unchanged. This compares the full RegionDecompositionResult against a verbatim
reimplementation of the pre-vectorization loop on a messy frame (chr-prefixed,
duplicate positions, overlapping windows, shuffled rows).
"""

import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import _normalize_chromosome
from imputed_prs.core.regions import (
    GenomicRegion,
    RegionDecompositionResult,
    _chromosome_sort_key,
    merge_variant_windows,
)


def _legacy_merge(prs_variants, window_size=1_000_000):
    """Verbatim pre-vectorization implementation, kept as an oracle."""
    if prs_variants.empty:
        return RegionDecompositionResult([], 0, 0, [], 0)
    windows = []
    for idx, row in prs_variants.iterrows():
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = int(row["position"])
        windows.append((chrom, max(0, pos - window_size), pos + window_size,
                        row["variant_id"], idx))
    chrom_groups: dict = {}
    for chrom, start, end, variant_id, idx in windows:
        chrom_groups.setdefault(chrom, []).append((start, end, variant_id, idx))
    regions = []
    for chrom in sorted(chrom_groups.keys(), key=_chromosome_sort_key):
        intervals = chrom_groups[chrom]
        intervals.sort(key=lambda x: (x[0], x[1]))
        cs, ce = intervals[0][0], intervals[0][1]
        cids, cidx = [intervals[0][2]], [intervals[0][3]]
        for i in range(1, len(intervals)):
            s, e, vid, ix = intervals[i]
            if s <= ce:
                ce = max(ce, e)
                cids.append(vid)
                cidx.append(ix)
            else:
                regions.append(GenomicRegion(chrom, cs, ce, cids, cidx))
                cs, ce, cids, cidx = s, e, [vid], [ix]
        regions.append(GenomicRegion(chrom, cs, ce, cids, cidx))
    vpr = [len(r.prs_variant_ids) for r in regions]
    spans = [r.end - r.start for r in regions]
    return RegionDecompositionResult(
        regions, len(regions), sum(vpr), vpr, max(spans) if spans else 0)


def _messy_frame():
    rng = np.random.RandomState(7)
    rows = []
    vid = 0
    for chrom in ["chr2", "1", "2", "X", "1", "chr2"]:
        for p in rng.randint(1_000_000, 8_000_000, size=25):
            rows.append({"variant_id": f"v{vid}", "chromosome": chrom,
                         "position": int(p)})
            vid += 1
    # duplicate positions (multiallelic) + shuffle
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df.sample(frac=1.0, random_state=2).reset_index(drop=True)


def test_vectorized_matches_legacy_on_messy_frame():
    df = _messy_frame()
    for w in (1_000_000, 250_000, 2_000_000):
        assert merge_variant_windows(df, w) == _legacy_merge(df, w)


def test_empty_and_single():
    empty = pd.DataFrame(columns=["variant_id", "chromosome", "position"])
    assert merge_variant_windows(empty) == _legacy_merge(empty)
    one = pd.DataFrame({"variant_id": ["a"], "chromosome": ["chr7"], "position": [5_000_000]})
    assert merge_variant_windows(one) == _legacy_merge(one)


def test_indices_are_the_reset_positional_labels():
    df = pd.DataFrame({
        "variant_id": ["a", "b", "c"],
        "chromosome": ["1", "1", "2"],
        "position": [1_000_000, 1_500_000, 9_000_000],
    })
    result = merge_variant_windows(df)
    all_indices = sorted(i for r in result.regions for i in r.prs_variant_indices)
    assert all_indices == [0, 1, 2]
