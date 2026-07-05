"""Differential tests for the vectorized reference index and resolver (Phase 1, C).

Both new code paths must be *exactly* equivalent to the untouched oracles they
optimize:
- vectorized ``build_reference_allele_index`` vs the original iterrows/setdefault
  construction;
- ``ReferenceAlleleResolver.resolve`` vs ``match_oriented_dosage``.

These reach the golden gate through ``_compute_true_prs``, and the golden
fixture is narrow (biallelic, one palindrome), so these differential batteries
-- not the round-trip -- are the real guard.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.harmonizer import (
    ReferenceAlleleResolver,
    _normalize_chromosome,
    build_reference_allele_index,
    hoist_columns,
    match_oriented_dosage,
)


def _legacy_build_reference_allele_index(variant_info):
    """The pre-vectorization implementation, kept here as an oracle."""
    index = {}
    for pos_idx, (_, row) in enumerate(variant_info.iterrows()):
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = int(row["position"])
        index.setdefault(f"{chrom}:{pos}", []).append(pos_idx)
    return index


def _make_reference(records):
    """records: list of (chrom, pos, ref, alt, dosages)."""
    variant_info = pd.DataFrame([
        {"variant_id": f"{c}:{p}:{ref}:{alt}", "chromosome": c, "position": p,
         "ref_allele": ref, "alt_allele": alt}
        for c, p, ref, alt, _ in records
    ])
    dosage_matrix = np.array([d for *_, d in records], dtype=float).T
    return variant_info, dosage_matrix


# A deliberately messy reference: multiallelic split rows, duplicate positions
# across chromosomes, out-of-row-order positions, a chr-prefixed and a float
# chromosome, and an ALT-less (None) record.
_RECORDS = [
    ("1", 100, "A", "G", [0.0, 1.0, 2.0]),
    ("1", 100, "A", "T", [2.0, 1.0, 0.0]),       # multiallelic at 1:100
    ("chr1", 200, "C", "T", [0.0, 2.0, 1.0]),    # chr-prefixed -> normalizes to "1"
    ("2", 50, "G", "C", [1.0, 1.0, 1.0]),        # palindromic
    ("22.0", 300, "A", "G", [2.0, 0.0, 2.0]),    # float chrom -> "22"
    ("2", 100, "A", None, [0.0, 0.0, 1.0]),      # ALT-less record (alt None)
    ("X", 400, "T", "A", [1.0, 2.0, 0.0]),
]


class TestBuildReferenceAlleleIndexVectorized:
    def test_matches_legacy_on_messy_frame(self):
        vi, _ = _make_reference(_RECORDS)
        assert build_reference_allele_index(vi) == _legacy_build_reference_allele_index(vi)

    def test_index_lists_are_python_ints(self):
        # Downstream ``if not candidates`` / iteration require plain lists, not
        # numpy arrays (which raise on truthiness).
        vi, _ = _make_reference(_RECORDS)
        idx = build_reference_allele_index(vi)
        for rows in idx.values():
            assert isinstance(rows, list)
            assert all(isinstance(i, int) for i in rows)

    def test_multiallelic_locus_groups_both_rows_in_order(self):
        vi, _ = _make_reference(_RECORDS)
        idx = build_reference_allele_index(vi)
        assert idx["1:100"] == [0, 1]

    def test_empty_frame(self):
        vi = pd.DataFrame(
            columns=["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]
        )
        assert build_reference_allele_index(vi) == {}


class TestResolverEquivalence:
    """resolver.resolve must equal match_oriented_dosage for every query."""

    # (chrom, pos, effect, other) queries spanning every branch.
    QUERIES = [
        ("1", 100, "G", "A"),          # effect == ALT, no flip
        ("1", 100, "A", "G"),          # effect == REF, flip
        ("1", 100, "T", "A"),          # multiallelic: picks the A/T row
        ("1", 100, "T", None),         # other None, multiallelic
        ("1", 200, "A", "G"),          # strand complement (C/T locus)
        ("2", 50, "G", "C"),           # palindromic direct
        ("22.0", 300, "G", "A"),       # float chromosome on the query side
        ("2", 100, "A", None),         # ALT-less record, effect == REF
        ("1", 100, "C", "A"),          # no allele match -> None
        ("9", 100, "A", "G"),          # absent locus -> None
        ("1", 100, "G", float("nan")), # other float-NaN -> treated as ""
        ("1", 100, "G", pd.NA),        # other pd.NA -> literal "<NA>" (NOT "")
    ]

    @pytest.mark.parametrize("chrom,pos,effect,other", QUERIES)
    def test_resolve_matches_oracle(self, chrom, pos, effect, other):
        vi, dm = _make_reference(_RECORDS)
        legacy_index = build_reference_allele_index(vi)
        resolver = ReferenceAlleleResolver(vi)

        expected = match_oriented_dosage(chrom, pos, effect, other, vi, dm, legacy_index)
        actual = resolver.resolve(chrom, pos, effect, other, dm)

        if expected is None:
            assert actual is None
            return
        assert actual is not None
        exp_idx, exp_dose, exp_flip = expected
        act_idx, act_dose, act_flip = actual
        assert act_idx == exp_idx
        assert act_flip == exp_flip
        np.testing.assert_array_equal(act_dose, exp_dose)

    def test_pd_na_other_is_not_treated_as_empty(self):
        # Guards the exact predicate: a blanket pd.isna would set other="" and
        # spuriously match; both oracle and resolver must return None here.
        vi, dm = _make_reference(_RECORDS)
        resolver = ReferenceAlleleResolver(vi)
        legacy_index = build_reference_allele_index(vi)
        assert match_oriented_dosage("1", 100, "G", pd.NA, vi, dm, legacy_index) is None
        assert resolver.resolve("1", 100, "G", pd.NA, dm) is None

    def test_returned_dosage_is_a_copy(self):
        # No-flip path must not alias the reference matrix column.
        vi, dm = _make_reference(_RECORDS)
        resolver = ReferenceAlleleResolver(vi)
        out = resolver.resolve("1", 100, "G", "A", dm)
        assert out is not None
        _, dose, _ = out
        dose[0] = 999.0
        assert dm[0, 0] != 999.0


class TestHoistColumns:
    """hoist_columns is the iterrows replacement in the harmonization loops."""

    def test_values_and_absent_optional_column(self):
        df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "22", "X"],
            "position": pd.array([100, 200, 300], dtype="Int64"),
            "effect_allele": ["A", "C", "G"],
            "beta": [0.1, -0.2, 0.3],
        })
        vids, chroms, pos, effs, oths, betas = hoist_columns(
            df, "variant_id", "chromosome", "position", "effect_allele",
            "other_allele", "beta",
        )
        assert vids == ["rs1", "rs2", "rs3"]
        assert chroms == ["1", "22", "X"]
        assert [int(p) for p in pos] == [100, 200, 300]
        assert effs == ["A", "C", "G"]
        assert oths == [None, None, None]  # absent column mirrors row.get
        assert [round(float(b), 6) for b in betas] == [0.1, -0.2, 0.3]

    def test_optional_column_missing_values_match_iterrows_isna(self):
        # None / NaN / pd.NA must all be pd.isna-equivalent to the iterrows path,
        # since the loops branch on ``pd.isna(other)``.
        df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3", "rs4"],
            "other_allele": ["G", None, np.nan, pd.NA],
        })
        (oths,) = hoist_columns(df, "other_allele")
        for i, (_, row) in enumerate(df.iterrows()):
            assert pd.isna(oths[i]) == pd.isna(row.get("other_allele"))
        assert oths[0] == "G"
