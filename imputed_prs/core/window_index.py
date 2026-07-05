"""O(log n) local-window queries over a fixed variant panel.

``filter_to_local_window`` (harmonizer.py) re-normalizes every variant's
chromosome with a pandas ``.apply`` and scans the full position array on *every*
call -- and it is called once per missing/target variant, so it is O(n_variants)
per query. ``ChromosomeIndex`` precomputes, once, per-chromosome sorted position
arrays, turning each window query into an ``np.searchsorted`` slice
(O(log n + k)). It returns a ``WindowFilterResult`` identical to
``filter_to_local_window`` (guarded by a differential property test), which is
retained as the oracle.

This is a validated, self-contained component in Phase 1. It is consumed by the
Phase-2 sufficient-statistics pass (which uses it to band-limit the ZᵀZ Gram to
co-windowed chip pairs); the legacy per-variant trainer is *not* rewired here,
since Phase 2 rewrites that path.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import (
    WindowFilterResult,
    _normalize_chromosome,
    normalize_chromosome_array,
)


class ChromosomeIndex:
    """Per-chromosome sorted-position index for fast window queries.

    Parameters
    ----------
    variant_info : pd.DataFrame
        Panel metadata with columns ``variant_id``, ``chromosome``,
        ``position``. Built once; queried many times. ``window`` returns
        positional indices into this exact frame (usable directly as dosage
        matrix columns), matching ``filter_to_local_window``.
    """

    def __init__(self, variant_info: pd.DataFrame):
        chroms = normalize_chromosome_array(variant_info["chromosome"])
        positions = variant_info["position"].to_numpy(dtype=np.int64)
        self._variant_ids = variant_info["variant_id"].to_numpy()
        # Per normalized chromosome: (positions sorted ascending, original
        # positional row indices aligned to them). A stable sort keeps equal
        # positions in ascending original-row order, which lets ``window``
        # reproduce ``np.where`` ordering exactly.
        self._by_chrom: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for chrom in pd.unique(chroms):
            rows = np.nonzero(chroms == chrom)[0]  # ascending original indices
            pos_c = positions[rows]
            order = np.argsort(pos_c, kind="stable")
            self._by_chrom[chrom] = (pos_c[order], rows[order])

    def window(
        self,
        target_chrom: str,
        target_pos: int,
        window_size: int = 1_000_000,
        exclude_target: bool = True,
        max_variants: Optional[int] = None,
    ) -> WindowFilterResult:
        """Return variants within ``±window_size`` bp of the target.

        Equivalent to ``filter_to_local_window(target_chrom, target_pos,
        variant_info, window_size, exclude_target, max_variants)`` for the frame
        this index was built from.
        """
        chrom_norm = _normalize_chromosome(str(target_chrom))
        bucket = self._by_chrom.get(chrom_norm)
        if bucket is None:
            return self._empty()

        sorted_pos, row_ids = bucket
        # Inclusive window (|pos - target| <= W): left bound at target-W, right
        # bound just past target+W.
        lo = np.searchsorted(sorted_pos, target_pos - window_size, side="left")
        hi = np.searchsorted(sorted_pos, target_pos + window_size, side="right")
        sel_rows = row_ids[lo:hi]
        sel_pos = sorted_pos[lo:hi]

        if exclude_target:
            keep = sel_pos != target_pos
            sel_rows = sel_rows[keep]
            sel_pos = sel_pos[keep]

        if sel_rows.size == 0:
            return self._empty()

        # Reproduce np.where(mask) order: ascending original ROW index. The
        # legacy path derives ``filtered_distances`` from this ordering before
        # any max_variants selection, so matching it makes the subsequent
        # argsort identical.
        order = np.argsort(sel_rows, kind="stable")
        indices = sel_rows[order]
        distances = np.abs(sel_pos[order] - target_pos)

        if max_variants is not None and len(indices) > max_variants:
            sort_order = np.argsort(distances)[:max_variants]
            indices = indices[sort_order]
            distances = distances[sort_order]

        return WindowFilterResult(
            # Vectorized gather + tolist: same objects/order as the oracle's
            # ``variant_info.iloc[indices]["variant_id"].tolist()``, far faster
            # than a per-element Python comprehension.
            variant_ids=self._variant_ids[indices].tolist(),
            variant_indices=indices,
            distances=distances,
            n_variants=len(indices),
        )

    @staticmethod
    def _empty() -> WindowFilterResult:
        return WindowFilterResult(
            variant_ids=[],
            variant_indices=np.array([], dtype=int),
            distances=np.array([], dtype=int),
            n_variants=0,
        )
