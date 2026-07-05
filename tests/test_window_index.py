"""Differential + edge-case tests for ChromosomeIndex (Phase 1, Workstream B).

ChromosomeIndex.window must return a WindowFilterResult *identical* to the
oracle filter_to_local_window across many loci and parameter combinations,
including the ordering-sensitive cases (duplicate/multiallelic positions,
unsorted panel rows, inclusive boundary at distance == window_size,
exclude_target removing several rows, and max_variants tie-breaking).
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.harmonizer import WindowFilterResult, filter_to_local_window
from imputed_prs.core.window_index import ChromosomeIndex


def _assert_same(actual: WindowFilterResult, expected: WindowFilterResult):
    assert actual.n_variants == expected.n_variants
    assert list(actual.variant_ids) == list(expected.variant_ids)
    np.testing.assert_array_equal(actual.variant_indices, expected.variant_indices)
    np.testing.assert_array_equal(actual.distances, expected.distances)


def _make_panel():
    """Multi-chromosome panel with duplicate positions and shuffled rows."""
    rng = np.random.RandomState(0)
    rows = []
    vid = 0
    for chrom in ["1", "2", "X", "chr3"]:  # chr-prefixed exercises normalization
        pos = rng.randint(1_000_000, 3_000_000, size=120).tolist()
        pos += pos[:15]  # duplicate positions == multiallelic split rows
        for p in pos:
            rows.append(
                {"variant_id": f"v{vid}", "chromosome": chrom, "position": int(p)}
            )
            vid += 1
    df = pd.DataFrame(rows)
    # Shuffle so positional (row) order differs from sorted-position order --
    # the case where reproducing np.where ordering actually matters.
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


PANEL = _make_panel()
INDEX = ChromosomeIndex(PANEL)
# Normalized chromosome -> sorted unique positions present, for target choice.
_NORM = PANEL["chromosome"].str.upper().str.replace("CHR", "", regex=False)


def _targets():
    W = 1_000_000
    targets = []
    for chrom in ["1", "2", "X", "3"]:
        present = sorted(PANEL.loc[_NORM == chrom, "position"].unique().tolist())
        picks = present[:3] + present[-3:] + present[len(present) // 2 : len(present) // 2 + 2]
        for p in picks:
            # exact, and boundary at +/- exactly W and one bp either side
            targets += [
                (chrom, p), (chrom, p + W), (chrom, p - W),
                (chrom, p + W + 1), (chrom, p - W - 1), (chrom, p + 500_000),
            ]
    # chr-prefixed target + an absent chromosome
    targets += [("chr3", PANEL.loc[_NORM == "3", "position"].iloc[0]), ("22", 1_500_000)]
    return targets


TARGETS = _targets()


@pytest.mark.parametrize("window_size", [1_000_000, 50_000])
@pytest.mark.parametrize("exclude_target", [True, False])
@pytest.mark.parametrize("max_variants", [None, 1, 3, 7, 10_000])
def test_matches_oracle_over_grid(window_size, exclude_target, max_variants):
    for chrom, pos in TARGETS:
        expected = filter_to_local_window(
            chrom, pos, PANEL, window_size=window_size,
            exclude_target=exclude_target, max_variants=max_variants,
        )
        actual = INDEX.window(
            chrom, pos, window_size=window_size,
            exclude_target=exclude_target, max_variants=max_variants,
        )
        _assert_same(actual, expected)


class TestEdgeCases:
    def test_inclusive_boundary_at_exactly_window_size(self):
        df = pd.DataFrame({
            "variant_id": ["a", "b", "c"],
            "chromosome": ["1", "1", "1"],
            "position": [1_000_000, 2_000_000, 3_000_001],
        })
        idx = ChromosomeIndex(df)
        # target 2M, window 1M: a (exactly -1M) and b included; c (+1_000_001) out.
        res = idx.window("1", 2_000_000, window_size=1_000_000, exclude_target=True)
        _assert_same(res, filter_to_local_window(
            "1", 2_000_000, df, window_size=1_000_000, exclude_target=True))
        assert sorted(res.variant_ids) == ["a"]  # b is the target, excluded

    def test_exclude_target_drops_all_multiallelic_rows(self):
        df = pd.DataFrame({
            "variant_id": ["a1", "a2", "b"],
            "chromosome": ["1", "1", "1"],
            "position": [500_000, 500_000, 600_000],  # a1,a2 share the target locus
        })
        idx = ChromosomeIndex(df)
        res = idx.window("1", 500_000, window_size=1_000_000, exclude_target=True)
        _assert_same(res, filter_to_local_window(
            "1", 500_000, df, window_size=1_000_000, exclude_target=True))
        assert list(res.variant_ids) == ["b"]
        # Without exclusion both rows at the locus come back.
        res2 = idx.window("1", 500_000, window_size=1_000_000, exclude_target=False)
        assert res2.n_variants == 3

    def test_absent_chromosome_is_empty(self):
        res = INDEX.window("22", 1_500_000)
        assert res.n_variants == 0
        assert res.variant_ids == []
        assert res.variant_indices.tolist() == []

    def test_indices_are_positional_into_source_frame(self):
        # variant_indices must index the exact source frame (used as Z columns).
        res = INDEX.window("1", int(PANEL.loc[_NORM == "1", "position"].iloc[0]),
                           max_variants=5)
        for i, vid in zip(res.variant_indices, res.variant_ids):
            assert PANEL.iloc[i]["variant_id"] == vid

    def test_empty_panel(self):
        df = pd.DataFrame({"variant_id": [], "chromosome": [], "position": []})
        idx = ChromosomeIndex(df)
        assert idx.window("1", 1000).n_variants == 0
