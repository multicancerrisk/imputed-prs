"""Tests for variant harmonization functions."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import IncompatibleBuildError
from imputed_prs.core.harmonizer import (
    AlleleAlignmentResult,
    BuildValidationResult,
    PartitionResult,
    WindowFilterResult,
    _complement,
    _is_ambiguous_snp,
    _normalize_build,
    _normalize_chromosome,
    align_effect_alleles,
    filter_to_local_window,
    partition_variants,
    validate_genome_build,
)
from imputed_prs.core.types import GenotypeData


class TestComplement:
    """Tests for the _complement helper function."""

    def test_a_to_t(self):
        """A complements to T."""
        assert _complement("A") == "T"
        assert _complement("a") == "T"

    def test_t_to_a(self):
        """T complements to A."""
        assert _complement("T") == "A"
        assert _complement("t") == "A"

    def test_c_to_g(self):
        """C complements to G."""
        assert _complement("C") == "G"
        assert _complement("c") == "G"

    def test_g_to_c(self):
        """G complements to C."""
        assert _complement("G") == "C"
        assert _complement("g") == "C"

    def test_unknown_allele(self):
        """Unknown alleles return uppercase of input."""
        assert _complement("N") == "N"
        assert _complement("X") == "X"


class TestIsAmbiguousSNP:
    """Tests for the _is_ambiguous_snp helper function."""

    def test_at_pair_is_ambiguous(self):
        """A/T pairs are ambiguous."""
        assert _is_ambiguous_snp("A", "T") is True
        assert _is_ambiguous_snp("T", "A") is True
        assert _is_ambiguous_snp("a", "t") is True

    def test_cg_pair_is_ambiguous(self):
        """C/G pairs are ambiguous."""
        assert _is_ambiguous_snp("C", "G") is True
        assert _is_ambiguous_snp("G", "C") is True
        assert _is_ambiguous_snp("c", "g") is True

    def test_ac_pair_not_ambiguous(self):
        """A/C pairs are not ambiguous."""
        assert _is_ambiguous_snp("A", "C") is False
        assert _is_ambiguous_snp("C", "A") is False

    def test_ag_pair_not_ambiguous(self):
        """A/G pairs are not ambiguous."""
        assert _is_ambiguous_snp("A", "G") is False
        assert _is_ambiguous_snp("G", "A") is False

    def test_ct_pair_not_ambiguous(self):
        """C/T pairs are not ambiguous."""
        assert _is_ambiguous_snp("C", "T") is False
        assert _is_ambiguous_snp("T", "C") is False

    def test_gt_pair_not_ambiguous(self):
        """G/T pairs are not ambiguous."""
        assert _is_ambiguous_snp("G", "T") is False
        assert _is_ambiguous_snp("T", "G") is False


class TestNormalizeBuild:
    """Tests for the _normalize_build helper function."""

    def test_grch37_aliases(self):
        """Test GRCh37 alias normalization."""
        assert _normalize_build("GRCh37") == "GRCh37"
        assert _normalize_build("grch37") == "GRCh37"
        assert _normalize_build("hg19") == "GRCh37"
        assert _normalize_build("b37") == "GRCh37"

    def test_grch38_aliases(self):
        """Test GRCh38 alias normalization."""
        assert _normalize_build("GRCh38") == "GRCh38"
        assert _normalize_build("grch38") == "GRCh38"
        assert _normalize_build("hg38") == "GRCh38"
        assert _normalize_build("b38") == "GRCh38"

    def test_none_input(self):
        """None input returns None."""
        assert _normalize_build(None) is None

    def test_unknown_build(self):
        """Unknown build returns as-is."""
        assert _normalize_build("unknown") == "unknown"

    def test_whitespace_handling(self):
        """Whitespace is stripped."""
        assert _normalize_build("  GRCh37  ") == "GRCh37"
        assert _normalize_build(" hg19 ") == "GRCh37"


class TestNormalizeChromosome:
    """Tests for the _normalize_chromosome helper function."""

    def test_strips_chr_prefix(self):
        """chr prefix is stripped."""
        assert _normalize_chromosome("chr1") == "1"
        assert _normalize_chromosome("Chr22") == "22"
        assert _normalize_chromosome("CHR10") == "10"

    def test_uppercase_sex_chromosomes(self):
        """Sex chromosomes are uppercased."""
        assert _normalize_chromosome("chrX") == "X"
        assert _normalize_chromosome("chrY") == "Y"
        assert _normalize_chromosome("x") == "X"

    def test_mitochondrial_normalized(self):
        """MT is normalized to M."""
        assert _normalize_chromosome("chrMT") == "M"
        assert _normalize_chromosome("MT") == "M"


class TestValidateGenomeBuild:
    """Tests for validate_genome_build function."""

    def test_same_build_is_compatible(self):
        """Same build is compatible."""
        result = validate_genome_build("GRCh37", "GRCh37")
        assert result.is_compatible is True
        assert result.prs_build == "GRCh37"
        assert result.genotype_build == "GRCh37"
        assert result.warning is None

    def test_alias_matching(self):
        """hg19 matches GRCh37."""
        result = validate_genome_build("hg19", "GRCh37")
        assert result.is_compatible is True
        assert result.prs_build == "GRCh37"
        assert result.genotype_build == "GRCh37"

    def test_mismatch_strict_raises(self):
        """Mismatch raises error in strict mode."""
        with pytest.raises(IncompatibleBuildError, match="Genome build mismatch"):
            validate_genome_build("GRCh37", "GRCh38", strict=True)

    def test_mismatch_non_strict_returns_false(self):
        """Mismatch returns is_compatible=False in non-strict mode."""
        result = validate_genome_build("GRCh37", "GRCh38", strict=False)
        assert result.is_compatible is False
        assert result.prs_build == "GRCh37"
        assert result.genotype_build == "GRCh38"

    def test_none_prs_build_produces_warning(self):
        """None PRS build produces warning."""
        result = validate_genome_build(None, "GRCh37")
        assert result.is_compatible is True
        assert result.prs_build is None
        assert result.genotype_build == "GRCh37"
        assert "PRS build is unknown" in result.warning

    def test_none_genotype_build_produces_warning(self):
        """None genotype build produces warning."""
        result = validate_genome_build("GRCh37", None)
        assert result.is_compatible is True
        assert result.prs_build == "GRCh37"
        assert result.genotype_build is None
        assert "Genotype build is unknown" in result.warning

    def test_both_none_produces_warning(self):
        """Both None builds produce warning."""
        result = validate_genome_build(None, None)
        assert result.is_compatible is True
        assert result.prs_build is None
        assert result.genotype_build is None
        assert "Both PRS and genotype builds are unknown" in result.warning


class TestPartitionVariants:
    """Tests for partition_variants function."""

    @pytest.fixture
    def sample_prs_df(self):
        """Create a sample PRS DataFrame."""
        return pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "chromosome": ["1", "1", "2", "2", "X"],
            "position": [100, 200, 300, 400, 500],
            "effect_allele": ["A", "C", "G", "T", "A"],
            "beta": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

    def test_exact_rsid_match(self, sample_prs_df):
        """Test exact rsID matching."""
        platform_variants = {"rs1", "rs3", "rs5"}
        result = partition_variants(sample_prs_df, platform_variants)

        assert "rs1" in result.observed
        assert "rs3" in result.observed
        assert "rs5" in result.observed
        assert "rs2" in result.missing
        assert "rs4" in result.missing
        assert result.observed_by_rsid == 3
        assert result.observed_by_chrpos == 0

    def test_chrpos_fallback(self, sample_prs_df):
        """Test chr:pos fallback matching."""
        platform_variants = {"1:100", "2:300"}  # No rsIDs
        result = partition_variants(sample_prs_df, platform_variants)

        assert "rs1" in result.observed
        assert "rs3" in result.observed
        assert len(result.missing) == 3
        assert result.observed_by_rsid == 0
        assert result.observed_by_chrpos == 2

    def test_case_insensitive_rsid(self, sample_prs_df):
        """Test case-insensitive rsID matching."""
        platform_variants = {"RS1", "Rs3"}
        result = partition_variants(sample_prs_df, platform_variants)

        assert "rs1" in result.observed
        assert "rs3" in result.observed
        assert result.observed_by_rsid == 2

    def test_mixed_id_formats(self, sample_prs_df):
        """Test mixed rsID and chr:pos in platform variants."""
        platform_variants = {"rs1", "2:400", "X:500"}
        result = partition_variants(sample_prs_df, platform_variants)

        assert "rs1" in result.observed
        assert "rs4" in result.observed
        assert "rs5" in result.observed
        assert len(result.observed) == 3
        assert result.observed_by_rsid == 1
        assert result.observed_by_chrpos == 2

    def test_empty_platform(self, sample_prs_df):
        """Test with empty platform variants."""
        result = partition_variants(sample_prs_df, set())

        assert len(result.observed) == 0
        assert len(result.missing) == 5

    def test_prs_to_platform_mapping(self, sample_prs_df):
        """Test that prs_to_platform_id mapping is correct."""
        platform_variants = {"rs1", "2:300"}
        result = partition_variants(sample_prs_df, platform_variants)

        assert result.prs_to_platform_id["rs1"] == "rs1"
        assert result.prs_to_platform_id["rs3"] == "2:300"

    def test_total_equals_prs_count(self, sample_prs_df):
        """Test that observed + missing equals total PRS variants."""
        platform_variants = {"rs1", "rs3", "2:400"}
        result = partition_variants(sample_prs_df, platform_variants)

        assert len(result.observed) + len(result.missing) == len(sample_prs_df)


class TestFilterToLocalWindow:
    """Tests for filter_to_local_window function."""

    @pytest.fixture
    def sample_variant_info(self):
        """Create sample variant info DataFrame."""
        return pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"],
            "chromosome": ["1", "1", "1", "1", "2", "1"],
            "position": [100, 500, 1000, 2000, 100, 1500],
        })

    def test_basic_window_filtering(self, sample_variant_info):
        """Test basic window filtering."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=1000,
            variant_info=sample_variant_info,
            window_size=1000,
            exclude_target=False,
        )

        # Should include rs1 (distance 900), rs2 (500), rs3 (0), rs6 (500)
        # Not rs4 (distance 1000 is on the boundary, should be included)
        assert result.n_variants >= 3
        assert "rs3" in result.variant_ids  # At target position

    def test_exclude_target_position(self, sample_variant_info):
        """Test excluding target position."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=1000,
            variant_info=sample_variant_info,
            window_size=1000,
            exclude_target=True,
        )

        # Should not include rs3 (at target position)
        assert "rs3" not in result.variant_ids

    def test_wrong_chromosome_excluded(self, sample_variant_info):
        """Test that variants on wrong chromosome are excluded."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=100,
            variant_info=sample_variant_info,
            window_size=10000,
        )

        # rs5 is on chromosome 2, should be excluded
        assert "rs5" not in result.variant_ids

    def test_max_variants_limit(self, sample_variant_info):
        """Test max_variants limit."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=1000,
            variant_info=sample_variant_info,
            window_size=2000,
            exclude_target=True,
            max_variants=2,
        )

        assert result.n_variants <= 2

    def test_no_variants_in_window(self, sample_variant_info):
        """Test when no variants are in window."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=100000,  # Far from all variants
            variant_info=sample_variant_info,
            window_size=100,
        )

        assert result.n_variants == 0
        assert len(result.variant_ids) == 0

    def test_distances_correct(self, sample_variant_info):
        """Test that distances are correctly computed."""
        result = filter_to_local_window(
            target_chrom="1",
            target_pos=1000,
            variant_info=sample_variant_info,
            window_size=2000,
            exclude_target=False,
        )

        # Find rs3 which is at position 1000
        for i, vid in enumerate(result.variant_ids):
            if vid == "rs3":
                assert result.distances[i] == 0
                break

    def test_chr_prefix_normalization(self, sample_variant_info):
        """Test that chr prefix is normalized."""
        result = filter_to_local_window(
            target_chrom="chr1",  # With prefix
            target_pos=1000,
            variant_info=sample_variant_info,
            window_size=1000,
        )

        # Should still find variants on chromosome 1
        assert result.n_variants > 0


class TestAlignEffectAlleles:
    """Tests for align_effect_alleles function."""

    def _make_genotype_data(self, variant_info, dosage_matrix, sample_ids=None):
        """Helper to create GenotypeData."""
        if sample_ids is None:
            sample_ids = [f"S{i}" for i in range(dosage_matrix.shape[0])]
        return GenotypeData(
            dosage_matrix=dosage_matrix,
            variant_info=variant_info,
            sample_ids=sample_ids,
        )

    def test_no_flip_needed(self):
        """Test when effect_allele matches alt_allele (no flip needed)."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["G"],
            "other_allele": ["A"],
            "beta": [0.5],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "ref_allele": ["A"],
            "alt_allele": ["G"],
        })

        # Dosage counts alt allele (G), which is the effect allele
        dosage_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_matched == 1
        assert result.n_flipped == 0
        # Dosage should be unchanged
        np.testing.assert_array_equal(result.aligned_dosage_matrix[:, 0], [0.0, 1.0, 2.0])

    def test_flip_needed(self):
        """Test when effect_allele matches ref_allele (flip needed)."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],  # Matches ref
            "other_allele": ["G"],
            "beta": [0.5],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "ref_allele": ["A"],
            "alt_allele": ["G"],
        })

        # Dosage counts alt allele (G), but effect allele is A (ref)
        dosage_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_matched == 1
        assert result.n_flipped == 1
        assert result.flip_mask[0] == True
        # Dosage should be flipped: 2 - original
        np.testing.assert_array_equal(result.aligned_dosage_matrix[:, 0], [2.0, 1.0, 0.0])

    def test_complement_matching(self):
        """Test complement matching for strand issues."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["C"],  # Complement of G (alt)
            "other_allele": ["T"],   # Complement of A (ref)
            "beta": [0.5],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "ref_allele": ["A"],
            "alt_allele": ["G"],
        })

        dosage_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_matched == 1
        # C complements to G which is alt, so no flip needed
        assert result.n_flipped == 0

    def test_ambiguous_snp_detection(self):
        """Test detection of ambiguous A/T or C/G SNPs."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            "other_allele": ["T"],
            "beta": [0.5],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "ref_allele": ["A"],
            "alt_allele": ["T"],
        })

        dosage_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_ambiguous == 1

    def test_unmatched_alleles(self):
        """Test handling of unmatched alleles."""
        # Use alleles that won't match even with complement
        # N and X are not standard nucleotides, so no complement matching
        prs_df = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["N"],  # Non-standard allele
            "other_allele": ["X"],
            "beta": [0.5],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "ref_allele": ["A"],
            "alt_allele": ["G"],
        })

        dosage_matrix = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        # Should not match - N and X don't match A/G or their complements
        assert result.n_matched == 0
        assert result.n_unmatched_alleles == 1

    def test_multiple_variants(self):
        """Test alignment with multiple variants."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "2"],
            "position": [100, 200, 300],
            "effect_allele": ["G", "A", "C"],
            "other_allele": ["A", "G", "T"],
            "beta": [0.5, 0.3, 0.2],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "2"],
            "position": [100, 200, 300],
            "ref_allele": ["A", "A", "T"],
            "alt_allele": ["G", "G", "C"],
        })

        dosage_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        observed = frozenset(["rs1", "rs2", "rs3"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_matched == 3
        # rs1: effect=G=alt, no flip
        # rs2: effect=A=ref, flip
        # rs3: effect=C=alt, no flip
        assert result.n_flipped == 1

    def test_variant_not_in_observed(self):
        """Test that variants not in observed set are skipped."""
        prs_df = pd.DataFrame({
            "variant_id": ["rs1", "rs2"],
            "chromosome": ["1", "1"],
            "position": [100, 200],
            "effect_allele": ["G", "C"],
            "other_allele": ["A", "T"],
            "beta": [0.5, 0.3],
        })

        variant_info = pd.DataFrame({
            "variant_id": ["rs1", "rs2"],
            "chromosome": ["1", "1"],
            "position": [100, 200],
            "ref_allele": ["A", "T"],
            "alt_allele": ["G", "C"],
        })

        dosage_matrix = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        genotype_data = self._make_genotype_data(variant_info, dosage_matrix)

        # Only rs1 is observed
        observed = frozenset(["rs1"])
        result = align_effect_alleles(prs_df, genotype_data, observed)

        assert result.n_matched == 1
        assert result.variant_mask[0] == True
        assert result.variant_mask[1] == False


class TestIntegration:
    """Integration tests with real data structures."""

    @pytest.mark.skip(reason="Requires network access to PGS Catalog")
    def test_prs313_23andme_v5_partition(self):
        """Test real data: PRS-313 vs 23andMe V5 platform.

        This test requires network access and downloads data from PGS Catalog.
        Run with: pytest -v -k "test_prs313" --run-integration
        """
        from imputed_prs.io.pgs_catalog import download_pgs_catalog_score
        from imputed_prs.io.platform_loader import load_platform_from_name

        prs_df, _ = download_pgs_catalog_score("PGS000004", "GRCh37")
        platform_variants, _ = load_platform_from_name("23andme_v5")

        result = partition_variants(prs_df, platform_variants)

        # PRS-313 has 313 variants, expect some observed on 23andMe
        assert result.observed
        assert result.missing
        assert len(result.observed) + len(result.missing) == len(prs_df)

        # Log results for analysis
        print(f"\nPRS variants: {len(prs_df)}")
        print(f"Observed: {len(result.observed)}")
        print(f"Missing: {len(result.missing)}")
        print(f"Matched by rsID: {result.observed_by_rsid}")
        print(f"Matched by chr:pos: {result.observed_by_chrpos}")
