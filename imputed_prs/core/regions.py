"""Region decomposition for linear projection PRS.

Merges overlapping per-variant genomic windows into non-overlapping regions,
each of which becomes the unit of prediction in the projection approach.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

from imputed_prs.core.harmonizer import _normalize_chromosome


@dataclass
class GenomicRegion:
    """A contiguous genomic interval containing one or more missing PRS variants.

    Created by merging overlapping per-variant windows.

    Attributes:
        chromosome: Chromosome identifier (normalized, e.g., "1", "X").
        start: Start position (inclusive). Derived from min(pos - window_size)
            across all PRS variants whose windows contributed to this region,
            clamped to >= 0.
        end: End position (inclusive). Derived from max(pos + window_size).
        prs_variant_ids: List of missing PRS variant IDs in this region.
        prs_variant_indices: Indices into the missing_prs_df for variants
            in this region. Used to index into the X matrix.
    """

    chromosome: str
    start: int
    end: int
    prs_variant_ids: List[str]
    prs_variant_indices: List[int]


@dataclass
class RegionDecompositionResult:
    """Result of decomposing PRS variants into non-overlapping regions.

    Attributes:
        regions: List of GenomicRegion objects, sorted by (chromosome, start).
        n_regions: Total number of merged regions.
        n_variants_in_regions: Total PRS variants covered.
        variants_per_region: List of counts (how many PRS variants per region).
        max_region_span_bp: Largest region span in base pairs.
    """

    regions: List[GenomicRegion]
    n_regions: int
    n_variants_in_regions: int
    variants_per_region: List[int]
    max_region_span_bp: int


def merge_variant_windows(
    prs_variants: pd.DataFrame,
    window_size: int = 1_000_000,
) -> RegionDecompositionResult:
    """Merge overlapping per-variant windows into non-overlapping genomic regions.

    Algorithm:
    1. For each missing PRS variant, compute window = [pos - W, pos + W].
       Clamp start to >= 0.
    2. Group by chromosome (normalized).
    3. Within each chromosome, sort by start position.
    4. Sweep-line merge: if current interval overlaps or is adjacent to previous,
       extend previous; otherwise start a new interval.
    5. Track which variant IDs/indices belong to each merged region.

    Args:
        prs_variants: DataFrame with columns: variant_id, chromosome, position.
            These are the missing PRS variants (not observed ones).
        window_size: Window size in base pairs on each side. Default: 1,000,000.

    Returns:
        RegionDecompositionResult with merged regions sorted by (chromosome, start).
    """
    if prs_variants.empty:
        return RegionDecompositionResult(
            regions=[],
            n_regions=0,
            n_variants_in_regions=0,
            variants_per_region=[],
            max_region_span_bp=0,
        )

    # Compute per-variant windows with normalized chromosomes
    windows = []
    for idx, row in prs_variants.iterrows():
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = int(row["position"])
        start = max(0, pos - window_size)
        end = pos + window_size
        windows.append((chrom, start, end, row["variant_id"], idx))

    # Group by chromosome
    chrom_groups: dict = {}
    for chrom, start, end, variant_id, idx in windows:
        if chrom not in chrom_groups:
            chrom_groups[chrom] = []
        chrom_groups[chrom].append((start, end, variant_id, idx))

    # Sort chromosomes for deterministic output
    sorted_chroms = sorted(chrom_groups.keys(), key=_chromosome_sort_key)

    regions = []
    for chrom in sorted_chroms:
        intervals = chrom_groups[chrom]
        # Sort by start position, then by end for ties
        intervals.sort(key=lambda x: (x[0], x[1]))

        # Sweep-line merge
        current_start, current_end = intervals[0][0], intervals[0][1]
        current_ids = [intervals[0][2]]
        current_indices = [intervals[0][3]]

        for i in range(1, len(intervals)):
            start, end, variant_id, idx = intervals[i]
            if start <= current_end:
                # Overlapping or adjacent: extend
                current_end = max(current_end, end)
                current_ids.append(variant_id)
                current_indices.append(idx)
            else:
                # Non-overlapping: emit current region, start new
                regions.append(GenomicRegion(
                    chromosome=chrom,
                    start=current_start,
                    end=current_end,
                    prs_variant_ids=current_ids,
                    prs_variant_indices=current_indices,
                ))
                current_start = start
                current_end = end
                current_ids = [variant_id]
                current_indices = [idx]

        # Emit final region for this chromosome
        regions.append(GenomicRegion(
            chromosome=chrom,
            start=current_start,
            end=current_end,
            prs_variant_ids=current_ids,
            prs_variant_indices=current_indices,
        ))

    variants_per_region = [len(r.prs_variant_ids) for r in regions]
    spans = [r.end - r.start for r in regions]

    return RegionDecompositionResult(
        regions=regions,
        n_regions=len(regions),
        n_variants_in_regions=sum(variants_per_region),
        variants_per_region=variants_per_region,
        max_region_span_bp=max(spans) if spans else 0,
    )


def _chromosome_sort_key(chrom: str):
    """Sort key for chromosomes: numeric first (1-22), then X, Y, M."""
    try:
        return (0, int(chrom))
    except ValueError:
        order = {"X": 23, "Y": 24, "M": 25}
        return (0, order.get(chrom, 26))
