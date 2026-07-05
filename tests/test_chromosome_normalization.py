"""Tests for chromosome-normalization hardening (Phase 1, Workstream A).

Covers two folded Phase-0 correctness fixes and the shared vectorized helper
that the windowing/reference-indexing hotspots reuse:

1. ``_normalize_chromosome`` repairs float-stringified numeric chromosomes
   ("22.0" -> "22") without clobbering versioned scaffold accessions.
2. ``normalize_chromosome_array`` is element-wise identical to the scalar
   function (guards the drop-in equivalence relied on by ChromosomeIndex,
   build_reference_allele_index, and merge_variant_windows).

These are deliberate behavior changes (they change which variants match a
reference), so they are validated on their own rather than under the
statistical-parity umbrella.
"""

import numpy as np
import pandas as pd

from imputed_prs.core.harmonizer import (
    _normalize_chromosome,
    build_reference_allele_index,
    match_oriented_dosage,
    normalize_chromosome_array,
)


class TestNormalizeChromosomeFloatFix:
    """The numeric-guarded '.0' strip."""

    def test_float_stringified_autosome_repaired(self):
        assert _normalize_chromosome("22.0") == "22"
        assert _normalize_chromosome("1.0") == "1"

    def test_float_repaired_matches_chr_prefixed(self):
        # This equality is exactly what lets a "22.0" PRS variant match a
        # "chr22" reference locus after the fix.
        assert _normalize_chromosome("22.0") == _normalize_chromosome("chr22")

    def test_chr_prefixed_float_repaired(self):
        assert _normalize_chromosome("chr22.0") == "22"

    def test_scaffold_accession_not_clobbered(self):
        # Numeric guard: a versioned accession must be left untouched.
        assert _normalize_chromosome("GL000220.0") == "GL000220.0"

    def test_plain_names_unchanged(self):
        assert _normalize_chromosome("chr1") == "1"
        assert _normalize_chromosome("CHR22") == "22"
        assert _normalize_chromosome("X") == "X"
        assert _normalize_chromosome("MT") == "M"
        assert _normalize_chromosome("22") == "22"

    def test_deduped_loader_copy_is_the_fixed_one(self):
        # io.genotype_loader re-exports the canonical harmonizer function, so its
        # copy carries the fix too (no silent drift between the two paths).
        from imputed_prs.io.genotype_loader import (
            _normalize_chromosome as loader_norm,
        )

        assert loader_norm is _normalize_chromosome
        assert loader_norm("22.0") == "22"


class TestNormalizeChromosomeArray:
    """Vectorized helper must equal the scalar function element-wise."""

    def test_matches_scalar_over_mixed_inputs(self):
        inputs = [
            "chr1", "CHR22", "1", "22.0", "X", "chrX", "MT", "chrMT", "M",
            "GL000220.0", "Y", "chrY", 5, 7.0, 22.0, None, np.nan,
        ]
        expected = [_normalize_chromosome(str(x)) for x in inputs]
        result = normalize_chromosome_array(inputs).tolist()
        assert result == expected

    def test_matches_scalar_apply_on_series(self):
        series = pd.Series(["chr22", "22.0", "chr22", "X", "22.0"])
        expected = series.apply(lambda x: _normalize_chromosome(str(x))).tolist()
        assert normalize_chromosome_array(series).tolist() == expected

    def test_preserves_alignment_and_length(self):
        series = pd.Series(["chr1", "chr2", "chr1"])
        out = normalize_chromosome_array(series)
        assert len(out) == 3
        assert out.tolist() == ["1", "2", "1"]


class TestCoverageDeltaThroughMatching:
    """The fix flips a previously-dropped '22.0' variant to matched."""

    @staticmethod
    def _ref(records):
        variant_info = pd.DataFrame([
            {"variant_id": f"{c}:{p}", "chromosome": c, "position": p,
             "ref_allele": ref, "alt_allele": alt}
            for c, p, ref, alt, _ in records
        ])
        dosage_matrix = np.array([d for *_, d in records], dtype=float).T
        return variant_info, dosage_matrix, build_reference_allele_index(variant_info)

    def test_float_chrom_now_matches_reference(self):
        # Reference stored as normalized "22"; PRS locus arrives as "22.0".
        # Before the fix, "22.0" normalized to "22.0" and missed -> dropped.
        vi, dm, idx = self._ref([("22", 16050075, "A", "G", [0.0, 1.0, 2.0])])
        out = match_oriented_dosage("22.0", 16050075, "G", "A", vi, dm, idx)
        assert out is not None
        _, dosage, flipped = out
        assert flipped is False
        np.testing.assert_array_equal(dosage, [0.0, 1.0, 2.0])
