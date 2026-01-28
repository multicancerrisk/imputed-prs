"""Tests for user genotype loading from DTC genetic testing files."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.io.user_genotypes import (
    detect_genome_build,
    genotype_to_dosage,
    load_user_genotypes,
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
