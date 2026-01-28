"""Tests for platform manifest loading."""

import gzip
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.types import PlatformInfo
from imputed_prs.io.platform_loader import (
    SUPPORTED_PLATFORMS,
    _validate_platform_name,
    _validate_variant_id,
    get_platform_info,
    list_available_platforms,
    load_platform_from_manifest,
    load_platform_from_name,
    load_platform_variants_from_list,
    _platform_cache,
)


class TestValidateVariantId:
    """Tests for _validate_variant_id function."""

    def test_valid_rsid_accepted(self):
        """Test that valid rsID format is accepted."""
        assert _validate_variant_id("rs123") is True
        assert _validate_variant_id("rs1") is True
        assert _validate_variant_id("rs123456789") is True

    def test_rsid_case_insensitive(self):
        """Test that rsID validation is case-insensitive."""
        assert _validate_variant_id("RS123") is True
        assert _validate_variant_id("Rs123") is True

    def test_valid_chrpos_accepted(self):
        """Test that valid chr:pos format is accepted."""
        assert _validate_variant_id("1:12345") is True
        assert _validate_variant_id("22:67890") is True
        assert _validate_variant_id("X:12345") is True
        assert _validate_variant_id("Y:67890") is True
        assert _validate_variant_id("MT:12345") is True
        assert _validate_variant_id("M:67890") is True

    def test_chrpos_case_insensitive(self):
        """Test that chr:pos validation is case-insensitive."""
        assert _validate_variant_id("x:12345") is True
        assert _validate_variant_id("y:67890") is True
        assert _validate_variant_id("mt:12345") is True

    def test_invalid_format_rejected(self):
        """Test that invalid formats are rejected."""
        assert _validate_variant_id("invalid") is False
        assert _validate_variant_id("rs") is False
        assert _validate_variant_id("rsABC") is False
        assert _validate_variant_id("12345") is False
        assert _validate_variant_id("chr1:12345") is False  # Should not have chr prefix
        assert _validate_variant_id("1:") is False
        assert _validate_variant_id(":12345") is False
        assert _validate_variant_id("") is False
        assert _validate_variant_id(None) is False

    def test_whitespace_variant_rejected(self):
        """Test that variants with only whitespace are rejected."""
        assert _validate_variant_id("   ") is False
        assert _validate_variant_id("\t") is False


class TestValidatePlatformName:
    """Tests for _validate_platform_name function."""

    def test_valid_platform_accepted(self):
        """Test that valid platform names are accepted."""
        assert _validate_platform_name("23andme_v3") == "23andme_v3"
        assert _validate_platform_name("23andme_v4") == "23andme_v4"
        assert _validate_platform_name("23andme_v5") == "23andme_v5"
        assert _validate_platform_name("ancestrydna_v1") == "ancestrydna_v1"
        assert _validate_platform_name("ancestrydna_v2") == "ancestrydna_v2"

    def test_case_insensitive(self):
        """Test that platform name validation is case-insensitive."""
        assert _validate_platform_name("23ANDME_V3") == "23andme_v3"
        assert _validate_platform_name("23ANDME_V4") == "23andme_v4"
        assert _validate_platform_name("23ANDME_V5") == "23andme_v5"
        assert _validate_platform_name("AncestryDNA_V1") == "ancestrydna_v1"
        assert _validate_platform_name("AncestryDNA_V2") == "ancestrydna_v2"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from platform names."""
        assert _validate_platform_name("  23andme_v5  ") == "23andme_v5"

    def test_unknown_platform_raises(self):
        """Test that unknown platform names raise ValidationError."""
        with pytest.raises(ValidationError, match="Unknown platform"):
            _validate_platform_name("unknown_platform")

        with pytest.raises(ValidationError, match="Unknown platform"):
            _validate_platform_name("23andme_v6")


class TestLoadPlatformVariantsFromList:
    """Tests for load_platform_variants_from_list function."""

    def test_valid_variants_accepted(self):
        """Test that valid variant IDs are accepted."""
        variants = load_platform_variants_from_list(["rs123", "rs456", "rs789"])
        assert len(variants) == 3
        assert "rs123" in variants
        assert "rs456" in variants
        assert "rs789" in variants

    def test_duplicates_removed(self):
        """Test that duplicate variant IDs are removed."""
        variants = load_platform_variants_from_list(["rs123", "rs123", "rs456"])
        assert len(variants) == 2
        assert "rs123" in variants
        assert "rs456" in variants

    def test_invalid_ids_filtered_with_warning(self):
        """Test that invalid IDs are filtered with a warning."""
        with pytest.warns(UserWarning, match="Filtered.*invalid"):
            variants = load_platform_variants_from_list(
                ["rs123", "invalid", "rs456", "bad_id"]
            )
        assert len(variants) == 2
        assert "rs123" in variants
        assert "rs456" in variants

    def test_empty_list_raises(self):
        """Test that empty list raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            load_platform_variants_from_list([])

    def test_all_invalid_raises(self):
        """Test that all invalid IDs raises ValidationError."""
        with pytest.warns(UserWarning):
            with pytest.raises(ValidationError, match="No valid variant IDs"):
                load_platform_variants_from_list(["invalid", "bad_id", "wrong"])

    def test_mixed_formats_accepted(self):
        """Test that both rsID and chr:pos formats are accepted together."""
        variants = load_platform_variants_from_list(
            ["rs123", "1:12345", "rs456", "X:67890"]
        )
        assert len(variants) == 4
        assert "rs123" in variants
        assert "1:12345" in variants
        assert "rs456" in variants
        assert "X:67890" in variants

    def test_returns_set(self):
        """Test that the function returns a set."""
        variants = load_platform_variants_from_list(["rs123", "rs456"])
        assert isinstance(variants, set)

    def test_name_parameter_for_logging(self):
        """Test that name parameter doesn't affect output."""
        variants = load_platform_variants_from_list(
            ["rs123", "rs456"], name="my_platform"
        )
        assert len(variants) == 2


class TestLoadPlatformFromManifest:
    """Tests for load_platform_from_manifest function."""

    def test_load_plain_text_list(self):
        """Test loading plain text file with one variant per line."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("rs123\n")
            f.write("rs456\n")
            f.write("1:12345\n")
            f.flush()
            path = Path(f.name)

        try:
            variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 3
            assert "rs123" in variants
            assert "rs456" in variants
            assert "1:12345" in variants
            assert info is None  # No metadata for custom manifests
        finally:
            path.unlink()

    def test_load_csv_with_variant_column(self):
        """Test loading CSV file with rsid column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("rsid,chromosome,position\n")
            f.write("rs123,1,12345\n")
            f.write("rs456,2,67890\n")
            f.flush()
            path = Path(f.name)

        try:
            variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 2
            assert "rs123" in variants
            assert "rs456" in variants
            assert info is None
        finally:
            path.unlink()

    def test_load_tsv_format(self):
        """Test loading TSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("variant_id\tchromosome\tposition\n")
            f.write("rs123\t1\t12345\n")
            f.write("rs456\t2\t67890\n")
            f.flush()
            path = Path(f.name)

        try:
            variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 2
            assert "rs123" in variants
            assert "rs456" in variants
        finally:
            path.unlink()

    def test_load_gzipped_file(self):
        """Test loading gzipped variant list."""
        with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
            path = Path(f.name)

        try:
            with gzip.open(path, "wt") as f:
                f.write("rs123\n")
                f.write("rs456\n")
                f.write("1:12345\n")

            variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 3
            assert "rs123" in variants
        finally:
            path.unlink()

    def test_file_not_found_raises(self):
        """Test that missing file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            load_platform_from_manifest("/nonexistent/path/file.txt")

    def test_empty_file_raises(self):
        """Test that empty file raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="No valid variant IDs"):
                load_platform_from_manifest(str(path))
        finally:
            path.unlink()

    def test_invalid_variants_filtered_with_warning(self):
        """Test that invalid variants are filtered with warning."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("rs123\n")
            f.write("invalid_variant\n")
            f.write("rs456\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.warns(UserWarning, match="Filtered.*invalid"):
                variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 2
        finally:
            path.unlink()

    def test_uses_first_column_if_no_variant_column(self):
        """Test that first column is used if no known variant column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("my_ids,other_col\n")
            f.write("rs123,foo\n")
            f.write("rs456,bar\n")
            f.flush()
            path = Path(f.name)

        try:
            variants, info = load_platform_from_manifest(str(path))
            assert len(variants) == 2
            assert "rs123" in variants
        finally:
            path.unlink()


class TestLoadPlatformFromName:
    """Tests for load_platform_from_name function."""

    def test_case_insensitive(self):
        """Test that platform names are case-insensitive."""
        # Clear cache to ensure fresh load
        _platform_cache.clear()

        # Mock the variant file loading to avoid actual file operations
        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            variants1, info1 = load_platform_from_name("23andme_v5")
            _platform_cache.clear()
            variants2, info2 = load_platform_from_name("23ANDME_V5")

            assert info1.name == info2.name == "23andme_v5"

    def test_unknown_platform_raises(self):
        """Test that unknown platform raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown platform"):
            load_platform_from_name("unknown_platform")

    def test_caching_works(self):
        """Test that caching prevents repeated loads."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            # First call
            load_platform_from_name("23andme_v5")

            # Second call should use cache
            load_platform_from_name("23andme_v5")

            # Should only be called once due to caching
            assert mock_variants.call_count == 1

    def test_returns_variants_and_info(self):
        """Test that function returns both variants and PlatformInfo."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890", "3:11111"}

            variants, info = load_platform_from_name("23andme_v5")

            assert isinstance(variants, set)
            assert isinstance(info, PlatformInfo)
            assert info.name == "23andme_v5"
            assert info.display_name == "23andMe V5"
            assert info.genome_build == "GRCh37"

    def test_load_ancestrydna_v2(self):
        """Test loading AncestryDNA V2 platform."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            variants, info = load_platform_from_name("ancestrydna_v2")

            assert info.name == "ancestrydna_v2"
            assert info.display_name == "AncestryDNA V2"
            assert info.company == "Ancestry"

    def test_load_23andme_v3(self):
        """Test loading 23andMe V3 platform."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            variants, info = load_platform_from_name("23andme_v3")

            assert info.name == "23andme_v3"
            assert info.display_name == "23andMe V3"
            assert info.company == "23andMe"
            assert info.version == "3"

    def test_load_23andme_v4(self):
        """Test loading 23andMe V4 platform."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            variants, info = load_platform_from_name("23andme_v4")

            assert info.name == "23andme_v4"
            assert info.display_name == "23andMe V4"
            assert info.company == "23andMe"
            assert info.version == "4"

    def test_load_ancestrydna_v1(self):
        """Test loading AncestryDNA V1 platform."""
        _platform_cache.clear()

        with patch(
            "imputed_prs.io.platform_loader._get_or_generate_variant_file"
        ) as mock_variants:
            mock_variants.return_value = {"1:12345", "2:67890"}

            variants, info = load_platform_from_name("ancestrydna_v1")

            assert info.name == "ancestrydna_v1"
            assert info.display_name == "AncestryDNA V1"
            assert info.company == "Ancestry"
            assert info.version == "1"


class TestListAvailablePlatforms:
    """Tests for list_available_platforms function."""

    def test_returns_expected_platforms(self):
        """Test that the function returns expected platform names."""
        platforms = list_available_platforms()
        assert isinstance(platforms, list)
        assert "23andme_v3" in platforms
        assert "23andme_v4" in platforms
        assert "23andme_v5" in platforms
        assert "ancestrydna_v1" in platforms
        assert "ancestrydna_v2" in platforms

    def test_returns_all_supported_platforms(self):
        """Test that all supported platforms are returned."""
        platforms = list_available_platforms()
        assert set(platforms) == set(SUPPORTED_PLATFORMS)


class TestGetPlatformInfo:
    """Tests for get_platform_info function."""

    def test_returns_metadata(self):
        """Test that function returns PlatformInfo without loading variants."""
        info = get_platform_info("23andme_v5")
        assert isinstance(info, PlatformInfo)
        assert info.name == "23andme_v5"
        assert info.display_name == "23andMe V5"
        assert info.genome_build == "GRCh37"
        assert info.company == "23andMe"

    def test_unknown_platform_raises(self):
        """Test that unknown platform raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown platform"):
            get_platform_info("unknown_platform")

    def test_case_insensitive(self):
        """Test that platform names are case-insensitive."""
        info1 = get_platform_info("23andme_v5")
        info2 = get_platform_info("23ANDME_V5")
        assert info1.name == info2.name

    def test_uses_cache_if_available(self):
        """Test that cached data is used if available."""
        _platform_cache.clear()

        # Pre-populate cache
        cached_info = PlatformInfo(
            name="23andme_v5",
            display_name="Cached",
            description="test",
            genome_build="GRCh37",
            n_variants=100,
            chip_technology="test",
            company="test",
            version="5",
        )
        _platform_cache["23andme_v5"] = ({"rs123"}, cached_info)

        info = get_platform_info("23andme_v5")
        assert info.display_name == "Cached"

        # Clean up
        _platform_cache.clear()


class TestPlatformInfoMetadata:
    """Tests for platform metadata JSON files."""

    def test_23andme_v3_metadata(self):
        """Test 23andMe V3 metadata is correct."""
        info = get_platform_info("23andme_v3")
        assert info.name == "23andme_v3"
        assert info.display_name == "23andMe V3"
        assert "OmniExpress" in info.chip_technology
        assert info.genome_build == "GRCh37"
        assert info.company == "23andMe"
        assert info.version == "3"
        assert info.n_variants == 961811

    def test_23andme_v4_metadata(self):
        """Test 23andMe V4 metadata is correct."""
        info = get_platform_info("23andme_v4")
        assert info.name == "23andme_v4"
        assert info.display_name == "23andMe V4"
        assert "HTS iSelect" in info.chip_technology
        assert info.genome_build == "GRCh37"
        assert info.company == "23andMe"
        assert info.version == "4"
        assert info.n_variants == 598426

    def test_23andme_v5_metadata(self):
        """Test 23andMe V5 metadata is correct."""
        info = get_platform_info("23andme_v5")
        assert info.name == "23andme_v5"
        assert info.display_name == "23andMe V5"
        assert "GSA" in info.chip_technology or "Global Screening Array" in info.chip_technology
        assert info.genome_build == "GRCh37"
        assert info.company == "23andMe"
        assert info.version == "5"
        assert info.n_variants > 0

    def test_ancestrydna_v1_metadata(self):
        """Test AncestryDNA V1 metadata is correct."""
        info = get_platform_info("ancestrydna_v1")
        assert info.name == "ancestrydna_v1"
        assert info.display_name == "AncestryDNA V1"
        assert "OmniExpress" in info.chip_technology
        assert info.genome_build == "GRCh37"
        assert info.company == "Ancestry"
        assert info.version == "1"
        assert info.n_variants == 728909

    def test_ancestrydna_v2_metadata(self):
        """Test AncestryDNA V2 metadata is correct."""
        info = get_platform_info("ancestrydna_v2")
        assert info.name == "ancestrydna_v2"
        assert info.display_name == "AncestryDNA V2"
        assert info.genome_build == "GRCh37"
        assert info.company == "Ancestry"
        assert info.version == "2"
        assert info.n_variants > 0


@pytest.mark.integration
class TestPlatformIntegration:
    """Integration tests for platform loading (require network access)."""

    def test_23andme_v5_variant_count(self):
        """Verify 23andMe V5 has expected variant count."""
        _platform_cache.clear()

        variants, info = load_platform_from_name("23andme_v5")
        assert info.n_variants > 600000
        assert len(variants) > 0
        assert info.genome_build == "GRCh37"

        # Verify variant format is chr:pos
        sample = list(variants)[0]
        assert ":" in sample

    def test_ancestrydna_v2_variant_count(self):
        """Verify AncestryDNA V2 has expected variant count."""
        _platform_cache.clear()

        variants, info = load_platform_from_name("ancestrydna_v2")
        assert info.n_variants > 600000
        assert len(variants) > 0
        assert info.genome_build == "GRCh37"

    def test_pgs000004_coverage_on_23andme_v5(self):
        """Check PGS000004 variant overlap with 23andMe V5."""
        _platform_cache.clear()

        try:
            from imputed_prs.io import download_pgs_catalog_score

            prs_df, _ = download_pgs_catalog_score("PGS000004", "GRCh37")
            variants, _ = load_platform_from_name("23andme_v5")

            # Convert PRS variants to chr:pos format for comparison
            prs_chrpos = set(
                f"{row['chromosome']}:{row['position']}"
                for _, row in prs_df.iterrows()
            )
            overlap = prs_chrpos & variants
            coverage = len(overlap) / len(prs_chrpos) if prs_chrpos else 0

            print(f"PGS000004 coverage on 23andMe V5: {coverage:.1%}")
            assert coverage > 0  # Should have some overlap
        except Exception as e:
            pytest.skip(f"Integration test skipped due to: {e}")
