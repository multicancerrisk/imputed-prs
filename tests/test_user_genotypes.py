"""Tests for user genotype loading from DTC genetic testing files."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.types import VariantIdentity
from imputed_prs.io.user_genotypes import (
    count_allele,
    detect_genome_build,
    genotype_to_dosage,
    load_raw_user_genotypes,
    load_user_genotype_strings,
    load_user_genotypes,
    render_genotype_string,
)


class TestGenotypeToDosage:
    """Tests for genotype to dosage conversion."""

    def test_homozygous(self):
        """Test homozygous genotypes return 2.0."""
        assert genotype_to_dosage("AA") == 2.0
        assert genotype_to_dosage("GG") == 2.0
        assert genotype_to_dosage("TT") == 2.0
        assert genotype_to_dosage("CC") == 2.0

    def test_heterozygous(self):
        """Test heterozygous genotypes return 1.0."""
        assert genotype_to_dosage("AG") == 1.0
        assert genotype_to_dosage("GA") == 1.0
        assert genotype_to_dosage("CT") == 1.0
        assert genotype_to_dosage("TC") == 1.0
        assert genotype_to_dosage("AT") == 1.0
        assert genotype_to_dosage("CG") == 1.0

    def test_missing_double_dash(self):
        """Test double dash missing genotypes return None."""
        assert genotype_to_dosage("--") is None

    def test_missing_empty_string(self):
        """Test empty string returns None."""
        assert genotype_to_dosage("") is None

    def test_missing_none(self):
        """Test None input returns None."""
        assert genotype_to_dosage(None) is None

    def test_missing_nan(self):
        """Test NaN input returns None."""
        assert genotype_to_dosage(np.nan) is None

    def test_missing_na_variants(self):
        """Test various NA indicators return None."""
        assert genotype_to_dosage("NA") is None
        assert genotype_to_dosage("N/A") is None
        assert genotype_to_dosage("NULL") is None
        assert genotype_to_dosage("..") is None
        assert genotype_to_dosage("NN") is None
        assert genotype_to_dosage("00") is None

    def test_indels_single_char(self):
        """Test single character indel indicators return None."""
        assert genotype_to_dosage("I") is None
        assert genotype_to_dosage("D") is None

    def test_indels_double_char(self):
        """Test double character indel genotypes return None."""
        assert genotype_to_dosage("II") is None
        assert genotype_to_dosage("DD") is None
        assert genotype_to_dosage("ID") is None
        assert genotype_to_dosage("DI") is None

    def test_partial_missing(self):
        """Test partial missing genotypes return None."""
        assert genotype_to_dosage("-A") is None
        assert genotype_to_dosage("A-") is None
        assert genotype_to_dosage("N0") is None

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert genotype_to_dosage("aa") == 2.0
        assert genotype_to_dosage("gg") == 2.0
        assert genotype_to_dosage("ag") == 1.0
        assert genotype_to_dosage("Ag") == 1.0

    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        assert genotype_to_dosage(" AA ") == 2.0
        assert genotype_to_dosage("\tAG\n") == 1.0

    def test_unrecognized_format(self):
        """Test unrecognized formats return None."""
        assert genotype_to_dosage("AAA") is None
        assert genotype_to_dosage("X") is None
        assert genotype_to_dosage("123") is None


class TestLoadUserGenotypesDataFrame:
    """Tests for loading genotypes from DataFrame input."""

    def test_load_basic_dataframe(self):
        """Test loading from a basic DataFrame."""
        df = pd.DataFrame({
            "genotype": ["AA", "AG", "GG"],
        }, index=["rs1", "rs2", "rs3"])

        dosages = load_user_genotypes(df)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] == 1.0
        assert dosages["rs3"] == 2.0

    def test_load_with_rsid_column(self):
        """Test loading from DataFrame with rsid column."""
        df = pd.DataFrame({
            "rsid": ["rs1", "rs2", "rs3"],
            "genotype": ["AA", "AG", "GG"],
            "chrom": ["1", "1", "1"],
            "pos": [100, 200, 300],
        })

        dosages = load_user_genotypes(df)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] == 1.0
        assert dosages["rs3"] == 2.0

    def test_load_with_variant_id_column(self):
        """Test loading from DataFrame with variant_id column."""
        df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3"],
            "genotype": ["AA", "AG", "GG"],
        })

        dosages = load_user_genotypes(df)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] == 1.0
        assert dosages["rs3"] == 2.0

    def test_expected_variants_filtering(self):
        """Test filtering to expected variants."""
        df = pd.DataFrame({
            "genotype": ["AA", "AG", "GG", "TT"],
        }, index=["rs1", "rs2", "rs3", "rs4"])

        dosages = load_user_genotypes(df, expected_variants={"rs1", "rs2", "rs5"})

        assert "rs1" in dosages
        assert "rs2" in dosages
        assert "rs5" in dosages  # Missing variant included
        assert "rs3" not in dosages  # Not in expected_variants
        assert "rs4" not in dosages
        assert dosages["rs5"] is None  # Missing variant is None

    def test_missing_genotype_column_raises(self):
        """Test that missing genotype column raises ValidationError."""
        df = pd.DataFrame({
            "rsid": ["rs1", "rs2"],
            "alleles": ["AA", "AG"],  # Wrong column name
        })

        with pytest.raises(ValidationError, match="genotype"):
            load_user_genotypes(df)

    def test_handles_missing_genotypes(self):
        """Test handling of missing genotypes in DataFrame."""
        df = pd.DataFrame({
            "genotype": ["AA", "--", None, "GG"],
        }, index=["rs1", "rs2", "rs3", "rs4"])

        dosages = load_user_genotypes(df)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] is None
        assert dosages["rs3"] is None
        assert dosages["rs4"] == 2.0


class TestLoadUserGenotypesFile:
    """Tests for loading genotypes from file input."""

    def test_file_not_found(self):
        """Test that nonexistent file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="File not found"):
            load_user_genotypes("/nonexistent/file.txt")

    def test_load_with_mock_snps(self, tmp_path):
        """Test loading from file using mocked snps package."""
        # Create a mock SNPs object
        mock_snps_instance = MagicMock()
        mock_snps_instance.snps = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "pos": [100, 200, 300],
            "genotype": ["AA", "AG", "GG"],
        }, index=["rs1", "rs2", "rs3"])
        mock_snps_instance.source = "23andMe"
        mock_snps_instance.build = 37

        # Create a dummy file
        test_file = tmp_path / "test_genotypes.txt"
        test_file.write_text("dummy content")

        with patch("snps.SNPs") as mock_snps_class:
            mock_snps_class.return_value = mock_snps_instance

            dosages = load_user_genotypes(test_file)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] == 1.0
        assert dosages["rs3"] == 2.0

    def test_empty_file_raises(self, tmp_path):
        """Test that empty file raises DataLoadError."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        mock_snps_instance = MagicMock()
        mock_snps_instance.snps = pd.DataFrame()  # Empty DataFrame

        with patch("snps.SNPs") as mock_snps_class:
            mock_snps_class.return_value = mock_snps_instance

            with pytest.raises(DataLoadError, match="No genotype data"):
                load_user_genotypes(test_file)


class TestLoadUserGenotypesSNPs:
    """Tests for loading genotypes from SNPs object input."""

    def test_load_from_snps_object(self):
        """Test loading from pre-loaded SNPs object."""
        from snps import SNPs

        # Create a mock that is an instance of SNPs
        mock_snps = MagicMock(spec=SNPs)
        mock_snps.snps = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "pos": [100, 200, 300],
            "genotype": ["AA", "AG", "GG"],
        }, index=["rs1", "rs2", "rs3"])

        dosages = load_user_genotypes(mock_snps)

        assert dosages["rs1"] == 2.0
        assert dosages["rs2"] == 1.0
        assert dosages["rs3"] == 2.0

    def test_load_from_snps_with_expected_variants(self):
        """Test loading from SNPs object with expected_variants filter."""
        from snps import SNPs

        mock_snps = MagicMock(spec=SNPs)
        mock_snps.snps = pd.DataFrame({
            "chrom": ["1", "1", "2", "2"],
            "pos": [100, 200, 300, 400],
            "genotype": ["AA", "AG", "GG", "TT"],
        }, index=["rs1", "rs2", "rs3", "rs4"])

        dosages = load_user_genotypes(
            mock_snps, expected_variants={"rs1", "rs3", "rs99"}
        )

        assert "rs1" in dosages
        assert "rs3" in dosages
        assert "rs99" in dosages
        assert dosages["rs99"] is None  # Missing
        assert "rs2" not in dosages
        assert "rs4" not in dosages


class TestDetectGenomeBuild:
    """Tests for genome build detection."""

    def test_detect_build_from_file(self, tmp_path):
        """Test genome build detection from file."""
        test_file = tmp_path / "test_genotypes.txt"
        test_file.write_text("dummy content")

        mock_snps = MagicMock()
        mock_snps.build = 37

        with patch("snps.SNPs") as mock_snps_class:
            mock_snps_class.return_value = mock_snps

            build = detect_genome_build(test_file)

        assert build == 37

    def test_detect_build_from_snps_object(self):
        """Test genome build detection from SNPs object."""
        from snps import SNPs

        mock_snps = MagicMock(spec=SNPs)
        mock_snps.build = 38

        build = detect_genome_build(mock_snps)

        assert build == 38

    def test_detect_build_file_not_found(self):
        """Test that missing file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="File not found"):
            detect_genome_build("/nonexistent/file.txt")


class TestUnsupportedInputType:
    """Tests for unsupported input types."""

    def test_list_raises_validation_error(self):
        """Test that list input raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported input type"):
            load_user_genotypes(["rs1", "rs2"])

    def test_dict_raises_validation_error(self):
        """Test that dict input raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported input type"):
            load_user_genotypes({"rs1": "AA"})

    def test_int_raises_validation_error(self):
        """Test that int input raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported input type"):
            load_user_genotypes(123)


class TestLoadFromPGP:
    """Integration tests using real PGP data.

    These tests download real genotype files from the Personal Genome Project
    and verify the loading functions work correctly with actual DTC data.

    Note: These tests require network access and may be skipped if the PGP
    server is unavailable or returns non-genotype data.
    """

    @pytest.fixture
    def pgp_23andme_file(self, tmp_path):
        """Download 23andMe file from PGP for testing."""
        import urllib.request

        url = "https://my.pgp-hms.org/user_file/download/4181"
        filepath = tmp_path / "23andme_hu09B28E.txt"
        try:
            urllib.request.urlretrieve(url, filepath)
            # Check if we got actual genotype data (not HTML)
            with open(filepath, "r") as f:
                first_line = f.readline()
                if first_line.startswith("<!DOCTYPE") or first_line.startswith("<"):
                    pytest.skip("PGP server returned HTML instead of genotype data")
        except Exception as e:
            pytest.skip(f"Failed to download PGP file: {e}")
        return filepath

    @pytest.fixture
    def pgp_ancestry_file(self, tmp_path):
        """Download AncestryDNA file from PGP for testing."""
        import urllib.request

        url = "https://my.pgp-hms.org/user_file/download/4127"
        filepath = tmp_path / "ancestry_hu09FF8C.txt"
        try:
            urllib.request.urlretrieve(url, filepath)
            # Check if we got actual genotype data (not HTML)
            with open(filepath, "r") as f:
                first_line = f.readline()
                if first_line.startswith("<!DOCTYPE") or first_line.startswith("<"):
                    pytest.skip("PGP server returned HTML instead of genotype data")
        except Exception as e:
            pytest.skip(f"Failed to download PGP file: {e}")
        return filepath

    @pytest.mark.integration
    def test_load_23andme_file(self, pgp_23andme_file):
        """Test loading real 23andMe file from PGP."""
        dosages = load_user_genotypes(pgp_23andme_file)

        # Should load many variants
        assert len(dosages) > 500000

        # Check some common SNPs exist and have valid values
        # rs1426654 - SLC24A5 skin pigmentation
        if "rs1426654" in dosages:
            assert dosages["rs1426654"] in (0.0, 1.0, 2.0, None)

        # Count valid dosages
        valid_count = sum(1 for v in dosages.values() if v is not None)
        assert valid_count > 400000  # Most variants should be valid

    @pytest.mark.integration
    def test_load_ancestry_file(self, pgp_ancestry_file):
        """Test loading real AncestryDNA file from PGP."""
        dosages = load_user_genotypes(pgp_ancestry_file)

        # Should load many variants
        assert len(dosages) > 500000

        # Count valid dosages
        valid_count = sum(1 for v in dosages.values() if v is not None)
        assert valid_count > 400000

    @pytest.mark.integration
    def test_detect_build_23andme(self, pgp_23andme_file):
        """Test genome build detection on 23andMe file."""
        build = detect_genome_build(pgp_23andme_file)
        # 23andMe typically uses GRCh37 (build 37)
        assert build in (37, 38, None)

    @pytest.mark.integration
    def test_expected_variants_with_real_file(self, pgp_23andme_file):
        """Test expected_variants filtering with real file."""
        # Some known common SNPs
        expected = {"rs1426654", "rs12913832", "rs1800407", "rs1805007", "rs12345_fake"}

        dosages = load_user_genotypes(pgp_23andme_file, expected_variants=expected)

        # Should only have the expected variants
        assert set(dosages.keys()) == expected

        # The fake one should be None
        assert dosages["rs12345_fake"] is None

        # At least some real ones should have values
        real_variants = [v for k, v in dosages.items() if k != "rs12345_fake"]
        assert any(v is not None for v in real_variants)


class TestCountAllele:
    """Golden table for the oriented allele-counting primitive (P1.1)."""

    def test_effect_alt_counts_named_allele(self):
        # counted=A (e.g. ALT), other=G: AA/AG/GG -> 2/1/0 copies of A.
        assert count_allele("AA", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 2.0
        assert count_allele("AG", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 1.0
        assert count_allele("GG", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 0.0

    def test_heterozygote_order_invariant(self):
        assert count_allele("GA", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 1.0

    def test_effect_ref_symmetry(self):
        # counted=G (e.g. REF) on the same genotypes: AA/AG/GG -> 0/1/2 copies of G.
        assert count_allele("AA", "G", "A", allow_ambiguous=False, allow_strand_flip=False) == 0.0
        assert count_allele("AG", "G", "A", allow_ambiguous=False, allow_strand_flip=False) == 1.0
        assert count_allele("GG", "G", "A", allow_ambiguous=False, allow_strand_flip=False) == 2.0

    def test_partial_overlap_is_unresolved(self):
        # "AC" against {A, G}: C is foreign -> unresolved, NOT a single copy of A.
        assert count_allele("AC", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("CG", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_full_mismatch_is_unresolved(self):
        assert count_allele("CT", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_palindromic_blocked_unless_allowed(self):
        assert count_allele("AA", "A", "T", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("CC", "C", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_palindromic_allowed(self):
        assert count_allele("AA", "A", "T", allow_ambiguous=True, allow_strand_flip=False) == 2.0
        assert count_allele("AT", "A", "T", allow_ambiguous=True, allow_strand_flip=False) == 1.0
        assert count_allele("TT", "A", "T", allow_ambiguous=True, allow_strand_flip=False) == 0.0

    def test_strand_flip_disabled_does_not_resolve(self):
        # "TT" complements to "AA" (in {A, G}), but only with the flip enabled.
        assert count_allele("TT", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_strand_flip_enabled_resolves(self):
        assert count_allele("TT", "A", "G", allow_ambiguous=False, allow_strand_flip=True) == 2.0
        assert count_allele("TC", "A", "G", allow_ambiguous=False, allow_strand_flip=True) == 1.0
        assert count_allele("CC", "A", "G", allow_ambiguous=False, allow_strand_flip=True) == 0.0

    def test_strand_flip_still_unresolved(self):
        # "AC" -> complement "TG"; neither orientation is a subset of {A, G}.
        assert count_allele("AC", "A", "G", allow_ambiguous=False, allow_strand_flip=True) is None

    def test_palindrome_with_flip_does_not_double_resolve(self):
        # Direct match already resolves; enabling the flip must not change the count.
        assert count_allele("AT", "A", "T", allow_ambiguous=True, allow_strand_flip=True) == 1.0
        assert count_allele("AA", "A", "T", allow_ambiguous=True, allow_strand_flip=True) == 2.0

    def test_separators(self):
        assert count_allele("A/G", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 1.0
        assert count_allele("A|G", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 1.0
        assert count_allele("A/A", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 2.0

    def test_lowercase_and_whitespace(self):
        assert count_allele("ag", "a", "g", allow_ambiguous=False, allow_strand_flip=False) == 1.0
        assert count_allele(" AG ", "A", "G", allow_ambiguous=False, allow_strand_flip=False) == 1.0

    def test_missing_indel_haploid_unresolved(self):
        bad_tokens = [
            "--", "", "NA", "N/A", "NULL", "..", "NN", "00",
            "I", "D", "II", "DD", "ID", "-A", "A-", "N0",
            "A", "G", "AAA", "X", "123", None, np.nan,
        ]
        for bad in bad_tokens:
            assert (
                count_allele(bad, "A", "G", allow_ambiguous=False, allow_strand_flip=False)
                is None
            ), bad

    def test_degenerate_pair_unresolved(self):
        # counted == other is not a valid biallelic locus.
        assert count_allele("AA", "A", "A", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("AG", "A", "A", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_non_acgt_alleles_unresolved(self):
        assert count_allele("AG", "A", "I", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("AG", "-", "G", allow_ambiguous=False, allow_strand_flip=False) is None
        # Empty string is a substring of "ACGT"; must still be rejected.
        assert count_allele("AG", "", "G", allow_ambiguous=False, allow_strand_flip=False) is None
        # Multi-base allele argument must be rejected.
        assert count_allele("AG", "AG", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_numeric_cell_unresolved(self):
        assert count_allele(12.0, "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None


class TestRenderGenotypeString:
    """render_genotype_string is the inverse of count_allele for integer dosages."""

    def test_renders_homozygous_and_het(self):
        # dosage counts ALT: 0 -> ref/ref, 1 -> ref/alt, 2 -> alt/alt.
        assert render_genotype_string("A", "G", 0) == "AA"
        assert render_genotype_string("A", "G", 1) == "AG"
        assert render_genotype_string("A", "G", 2) == "GG"

    def test_round_trips_with_count_allele(self):
        # Rendering then counting the ALT allele recovers the dosage; counting the
        # REF allele recovers 2 - dosage. This is the train/eval == browser bridge.
        for ref, alt in [("A", "G"), ("G", "A"), ("C", "T"), ("A", "T")]:
            for d in (0, 1, 2):
                geno = render_genotype_string(ref, alt, d)
                assert geno is not None
                assert count_allele(
                    geno, alt, ref, allow_ambiguous=True, allow_strand_flip=True
                ) == float(d)
                assert count_allele(
                    geno, ref, alt, allow_ambiguous=True, allow_strand_flip=True
                ) == float(2 - d)

    def test_accepts_float_and_numpy_dosage(self):
        assert render_genotype_string("A", "G", 1.0) == "AG"
        assert render_genotype_string("A", "G", np.float32(2.0)) == "GG"
        assert render_genotype_string("a", "g", 1) == "AG"  # alleles upper-cased

    def test_missing_or_noninteger_dosage_is_none(self):
        assert render_genotype_string("A", "G", np.nan) is None
        assert render_genotype_string("A", "G", None) is None
        assert render_genotype_string("A", "G", 0.7) is None  # continuous -> no string
        assert render_genotype_string("A", "G", 1.3) is None
        assert render_genotype_string("A", "G", -1) is None
        assert render_genotype_string("A", "G", 3) is None

    def test_invalid_alleles_is_none(self):
        assert render_genotype_string("A", "A", 1) is None  # degenerate pair
        assert render_genotype_string("A", "I", 1) is None  # indel/non-base
        assert render_genotype_string("", "G", 0) is None
        assert render_genotype_string("AG", "C", 1) is None  # multi-base allele


class TestLoadUserGenotypeStrings:
    """Tests for the raw-string loader (P1.1)."""

    def test_dataframe_index(self):
        df = pd.DataFrame({"genotype": ["AA", "AG", "GG"]}, index=["rs1", "rs2", "rs3"])
        strings = load_user_genotype_strings(df)
        assert strings["rs1"] == "AA"
        assert strings["rs2"] == "AG"
        assert strings["rs3"] == "GG"

    def test_returns_strings_not_dosages(self):
        df = pd.DataFrame({"genotype": ["AG"]}, index=["rs1"])
        strings = load_user_genotype_strings(df)
        assert strings["rs1"] == "AG"
        assert strings["rs1"] != 1.0

    def test_dataframe_rsid_column(self):
        df = pd.DataFrame({"rsid": ["rs1", "rs2"], "genotype": ["AA", "AG"]})
        strings = load_user_genotype_strings(df)
        assert strings["rs1"] == "AA"
        assert strings["rs2"] == "AG"

    def test_dataframe_variant_id_column(self):
        df = pd.DataFrame({"variant_id": ["rs1", "rs2"], "genotype": ["AA", "AG"]})
        strings = load_user_genotype_strings(df)
        assert strings["rs1"] == "AA"
        assert strings["rs2"] == "AG"

    def test_snps_object(self):
        from snps import SNPs

        mock_snps = MagicMock(spec=SNPs)
        mock_snps.snps = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "pos": [100, 200, 300],
            "genotype": ["AA", "AG", "GG"],
        }, index=["rs1", "rs2", "rs3"])

        strings = load_user_genotype_strings(mock_snps)
        assert strings["rs1"] == "AA"
        assert strings["rs2"] == "AG"
        assert strings["rs3"] == "GG"

    def test_expected_variants_filter_omits_absent(self):
        df = pd.DataFrame({"genotype": ["AA", "AG"]}, index=["rs1", "rs2"])
        strings = load_user_genotype_strings(df, expected_variants={"rs1", "rs99"})
        assert strings == {"rs1": "AA"}

    def test_missing_genotype_column_raises(self):
        df = pd.DataFrame({"rsid": ["rs1"], "alleles": ["AA"]})
        with pytest.raises(ValidationError, match="genotype"):
            load_user_genotype_strings(df)

    def test_duplicate_rsid_does_not_raise(self):
        # Regression: a duplicated index must not crash (no df.loc on a dup index).
        df = pd.DataFrame({"genotype": ["AA", "AG"]}, index=["rs1", "rs1"])
        strings = load_user_genotype_strings(df)
        assert strings["rs1"] == "AG"  # last write wins in the projected dict


class TestLoadRawUserGenotypes:
    """Tests for the raw multi-key collection build (P1.1b)."""

    def test_indexes_by_rsid_lowercased_and_chrpos_normalized(self):
        df = pd.DataFrame({
            "rsid": ["RS1"],
            "genotype": ["AG"],
            "chrom": ["chr1"],
            "pos": [100],
        })
        collection = load_raw_user_genotypes(df)
        assert "rs1" in collection._by_rsid
        assert "1:100" in collection._by_chrpos

    def test_duplicate_rsid_kept_as_multiple_records(self):
        df = pd.DataFrame({"genotype": ["AA", "AG"]}, index=["rs1", "rs1"])
        collection = load_raw_user_genotypes(df)
        assert len(collection._by_rsid["rs1"]) == 2

    def test_no_chrpos_index_when_absent(self):
        df = pd.DataFrame({"genotype": ["AG"]}, index=["rs1"])
        collection = load_raw_user_genotypes(df)
        assert collection._by_chrpos == {}


def _identity(accepted_ids, chromosome, position, counted="A", other="G"):
    """Build a VariantIdentity for resolver tests (alleles are arbitrary here:
    resolution is allele-agnostic; counting happens later in count_allele)."""
    return VariantIdentity(
        feature_id=f"{chromosome}:{position}:{other}:{counted}",
        variant_id=accepted_ids[0],
        accepted_ids=tuple(accepted_ids),
        chromosome=chromosome,
        position=position,
        counted_allele=counted,
        other_allele=other,
    )


class TestVariantIdentityResolver:
    """Tests for multi-key, conflict-aware resolution (P1.1b)."""

    def _collection(self, rows):
        # rows: list of (rsid, chrom, pos, genotype)
        df = pd.DataFrame({
            "rsid": [r[0] for r in rows],
            "chrom": [r[1] for r in rows],
            "pos": [r[2] for r in rows],
            "genotype": [r[3] for r in rows],
        })
        return load_raw_user_genotypes(df)

    def test_rsid_match_resolves(self):
        coll = self._collection([("rs1", "1", 100, "AG")])
        res = coll.resolve(_identity(["rs1"], "1", 100))
        assert res.status == "resolved"
        assert res.genotype == "AG"
        assert res.matched_id == "rs1"

    def test_chrpos_only_match_resolves(self):
        # The user row uses an rsID the model does not know; only chr:pos links them.
        coll = self._collection([("rs_user", "1", 100, "AG")])
        res = coll.resolve(_identity(["rs_model", "1:100"], "1", 100))
        assert res.status == "resolved"
        assert res.genotype == "AG"

    def test_both_resolve(self):
        coll = self._collection([
            ("rs1", "1", 100, "AG"),
            ("rs_user", "2", 200, "CC"),
        ])
        assert coll.resolve(_identity(["rs1"], "1", 100)).status == "resolved"
        assert coll.resolve(_identity(["rs_model", "2:200"], "2", 200)).status == "resolved"

    def test_not_found(self):
        coll = self._collection([("rs1", "1", 100, "AG")])
        res = coll.resolve(_identity(["rs999", "9:999"], "9", 999))
        assert res.status == "not_found"
        assert res.genotype is None

    def test_conflict_by_genotype(self):
        coll = self._collection([
            ("rs1", "1", 100, "AG"),
            ("rs1", "1", 100, "GG"),
        ])
        res = coll.resolve(_identity(["rs1"], "1", 100))
        assert res.status == "duplicate_conflict"
        assert res.genotype is None

    def test_conflict_by_locus(self):
        coll = self._collection([
            ("rs1", "1", 100, "AG"),
            ("rs1", "1", 200, "AG"),
        ])
        res = coll.resolve(_identity(["rs1"], "1", 100))
        assert res.status == "duplicate_conflict"

    def test_same_record_via_rsid_and_chrpos_not_conflict(self):
        # rsID and chr:pos both point at the SAME single record -> dedup, resolved.
        coll = self._collection([("rs1", "1", 100, "AG")])
        res = coll.resolve(_identity(["rs1", "1:100"], "1", 100))
        assert res.status == "resolved"
        assert res.genotype == "AG"

    def test_none_chrpos_plus_present_chrpos_not_conflict(self):
        # record A matched by rsID has no chr/pos; record B matched by chr:pos has
        # both; same genotype -> resolved, not a false-positive conflict.
        df = pd.DataFrame({
            "rsid": ["rs1", "rs_user"],
            "chrom": [None, "1"],
            "pos": [None, 100],
            "genotype": ["AG", "AG"],
        })
        coll = load_raw_user_genotypes(df)
        res = coll.resolve(_identity(["rs1", "1:100"], "1", 100))
        assert res.status == "resolved"
        assert res.genotype == "AG"

    def test_case_insensitive_rsid(self):
        coll = self._collection([("rs1", "1", 100, "AG")])
        assert coll.resolve(_identity(["RS1"], "1", 100)).status == "resolved"

    def test_chromosome_normalization(self):
        coll = self._collection([("rs_user", "chr1", 100, "AG")])
        assert coll.resolve(_identity(["1:100"], "1", 100)).status == "resolved"

        coll_mt = self._collection([("rs_user", "MT", 50, "AG")])
        assert coll_mt.resolve(_identity(["M:50"], "M", 50)).status == "resolved"
