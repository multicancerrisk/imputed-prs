"""Tests for PGS Catalog integration."""

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.io.pgs_catalog import (
    PGSCatalogMetadata,
    PGSSearchResult,
    _get_cache_dir,
    _get_harmonized_file_url,
    _get_metadata_cache_path,
    _get_scoring_file_cache_path,
    _load_cached_metadata,
    _parse_api_metadata,
    _process_harmonized_columns,
    _save_metadata_to_cache,
    _validate_genome_build,
    _validate_pgs_id,
    clear_pgs_catalog_cache,
    download_pgs_catalog_score,
    fetch_pgs_catalog_metadata,
    get_pgs_catalog_cache_info,
    search_pgs_catalog,
)


class TestValidatePgsId:
    """Tests for _validate_pgs_id function."""

    def test_valid_pgs_id(self):
        """Test that valid PGS ID is returned unchanged."""
        assert _validate_pgs_id("PGS000004") == "PGS000004"

    def test_lowercase_normalized(self):
        """Test that lowercase PGS ID is normalized to uppercase."""
        assert _validate_pgs_id("pgs000001") == "PGS000001"

    def test_mixed_case_normalized(self):
        """Test that mixed case is normalized."""
        assert _validate_pgs_id("Pgs000004") == "PGS000004"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        assert _validate_pgs_id("  PGS000004  ") == "PGS000004"

    def test_invalid_prefix_raises(self):
        """Test that invalid prefix raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid PGS ID format"):
            _validate_pgs_id("RS000004")

    def test_invalid_length_raises(self):
        """Test that wrong number of digits raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid PGS ID format"):
            _validate_pgs_id("PGS0004")

    def test_too_many_digits_raises(self):
        """Test that too many digits raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid PGS ID format"):
            _validate_pgs_id("PGS0000004")

    def test_non_numeric_raises(self):
        """Test that non-numeric suffix raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid PGS ID format"):
            _validate_pgs_id("PGS00000A")


class TestValidateGenomeBuild:
    """Tests for _validate_genome_build function."""

    def test_grch37_valid(self):
        """Test that GRCh37 is accepted."""
        assert _validate_genome_build("GRCh37") == "GRCh37"

    def test_grch38_valid(self):
        """Test that GRCh38 is accepted."""
        assert _validate_genome_build("GRCh38") == "GRCh38"

    def test_hg19_normalized(self):
        """Test that hg19 is normalized to GRCh37."""
        assert _validate_genome_build("hg19") == "GRCh37"

    def test_hg38_normalized(self):
        """Test that hg38 is normalized to GRCh38."""
        assert _validate_genome_build("hg38") == "GRCh38"

    def test_case_insensitive(self):
        """Test that build names are case-insensitive."""
        assert _validate_genome_build("grch37") == "GRCh37"
        assert _validate_genome_build("GRCH38") == "GRCh38"
        assert _validate_genome_build("HG19") == "GRCh37"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        assert _validate_genome_build("  GRCh37  ") == "GRCh37"

    def test_invalid_build_raises(self):
        """Test that unsupported build raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported genome build"):
            _validate_genome_build("GRCh36")

    def test_unknown_build_raises(self):
        """Test that unknown build raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported genome build"):
            _validate_genome_build("hg18")


class TestProcessHarmonizedColumns:
    """Tests for _process_harmonized_columns function."""

    def test_prefers_hm_columns(self):
        """Test that hm_* columns are preferred over original columns."""
        df = pd.DataFrame({
            "rsID": ["rs123", "rs456"],
            "hm_rsID": ["rs123_hm", "rs456_hm"],
            "chr_name": ["1", "2"],
            "hm_chr": ["chr1", "chr2"],
            "chr_position": [100, 200],
            "hm_pos": [101, 201],
            "effect_allele": ["A", "G"],
            "effect_weight": [0.1, 0.2],
            "hm_code": [0, 1],
        })
        result = _process_harmonized_columns(df)

        # hm_rsID should replace rsID
        assert "rsID" in result.columns
        assert result["rsID"].tolist() == ["rs123_hm", "rs456_hm"]

        # hm_chr should replace chr_name
        assert "chr_name" in result.columns
        assert result["chr_name"].tolist() == ["chr1", "chr2"]

        # hm_pos should replace chr_position
        assert "chr_position" in result.columns
        assert result["chr_position"].tolist() == [101, 201]

    def test_filters_failed_mappings(self):
        """Test that variants with hm_code < 0 are removed."""
        df = pd.DataFrame({
            "rsID": ["rs123", "rs456", "rs789"],
            "effect_allele": ["A", "G", "T"],
            "effect_weight": [0.1, 0.2, 0.3],
            "hm_code": [0, -1, 1],
        })
        result = _process_harmonized_columns(df, filter_failed_mappings=True)

        assert len(result) == 2
        assert result["rsID"].tolist() == ["rs123", "rs789"]

    def test_keeps_all_when_disabled(self):
        """Test that filter_failed_mappings=False keeps all variants."""
        df = pd.DataFrame({
            "rsID": ["rs123", "rs456", "rs789"],
            "effect_allele": ["A", "G", "T"],
            "effect_weight": [0.1, 0.2, 0.3],
            "hm_code": [0, -1, 1],
        })
        result = _process_harmonized_columns(df, filter_failed_mappings=False)

        assert len(result) == 3

    def test_no_hm_code_column(self):
        """Test handling when hm_code column is missing."""
        df = pd.DataFrame({
            "rsID": ["rs123", "rs456"],
            "effect_allele": ["A", "G"],
            "effect_weight": [0.1, 0.2],
        })
        result = _process_harmonized_columns(df, filter_failed_mappings=True)

        assert len(result) == 2

    def test_only_harmonized_columns_present(self):
        """Test handling when only harmonized columns are present."""
        df = pd.DataFrame({
            "hm_rsID": ["rs123", "rs456"],
            "hm_chr": ["1", "2"],
            "hm_pos": [100, 200],
            "effect_allele": ["A", "G"],
            "effect_weight": [0.1, 0.2],
            "hm_code": [0, 1],
        })
        result = _process_harmonized_columns(df)

        assert "rsID" in result.columns
        assert "chr_name" in result.columns
        assert "chr_position" in result.columns

    def test_generates_variant_id_when_no_rsid(self):
        """Test that variant_id is generated from position when rsID is null."""
        df = pd.DataFrame({
            "chr_name": ["1", "2", "3"],
            "chr_position": [100, 200, 300],
            "effect_allele": ["A", "G", "T"],
            "effect_weight": [0.1, 0.2, 0.3],
        })
        result = _process_harmonized_columns(df)

        assert "variant_id" in result.columns
        assert result["variant_id"].tolist() == ["1:100:A", "2:200:G", "3:300:T"]

    def test_generates_variant_id_when_rsid_all_null(self):
        """Test that variant_id is generated when rsID column exists but all null."""
        df = pd.DataFrame({
            "rsID": [None, None, None],
            "chr_name": ["1", "2", "3"],
            "chr_position": [100, 200, 300],
            "effect_allele": ["A", "G", "T"],
            "effect_weight": [0.1, 0.2, 0.3],
        })
        result = _process_harmonized_columns(df)

        assert "variant_id" in result.columns
        assert result["variant_id"].tolist() == ["1:100:A", "2:200:G", "3:300:T"]

    def test_uses_rsid_when_available(self):
        """Test that rsID is used when available and not all null."""
        df = pd.DataFrame({
            "rsID": ["rs123", "rs456", "rs789"],
            "chr_name": ["1", "2", "3"],
            "chr_position": [100, 200, 300],
            "effect_allele": ["A", "G", "T"],
            "effect_weight": [0.1, 0.2, 0.3],
        })
        result = _process_harmonized_columns(df)

        # Should NOT generate variant_id since rsID is available
        assert "variant_id" not in result.columns
        assert "rsID" in result.columns


class TestCacheHelpers:
    """Tests for cache helper functions."""

    def test_get_cache_dir_creates_directory(self):
        """Test that _get_cache_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            result = _get_cache_dir(cache_dir)

            assert result == cache_dir
            assert cache_dir.exists()

    def test_get_metadata_cache_path(self):
        """Test metadata cache path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            path = _get_metadata_cache_path("PGS000004", cache_dir)

            assert path == cache_dir / "PGS000004_metadata.json"

    def test_get_scoring_file_cache_path(self):
        """Test scoring file cache path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            path = _get_scoring_file_cache_path("PGS000004", "GRCh37", cache_dir)

            assert path == cache_dir / "PGS000004_hmPOS_GRCh37.txt.gz"

    def test_get_harmonized_file_url(self):
        """Test harmonized file URL generation."""
        url = _get_harmonized_file_url("PGS000004", "GRCh37")

        expected = (
            "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"
            "PGS000004/ScoringFiles/Harmonized/PGS000004_hmPOS_GRCh37.txt.gz"
        )
        assert url == expected


class TestParseApiMetadata:
    """Tests for _parse_api_metadata function."""

    def test_parse_full_metadata(self):
        """Test parsing complete API response."""
        data = {
            "id": "PGS000004",
            "name": "PRS313",
            "trait_reported": "Breast cancer",
            "trait_efo": [{"id": "EFO_0000305"}],
            "variants_number": 313,
            "original_genome_build": "GRCh37",
            "ftp_scoring_file": "https://example.com/file.txt.gz",
            "ftp_harmonized_scoring_files": {
                "GRCh37": {"positions": "https://example.com/grch37.txt.gz"},
                "GRCh38": {"positions": "https://example.com/grch38.txt.gz"},
            },
            "publication": {
                "doi": "10.1234/example",
                "PMID": "12345678",
            },
            "date_release": "2020-01-15",
        }

        result = _parse_api_metadata(data)

        assert result.pgs_id == "PGS000004"
        assert result.name == "PRS313"
        assert result.trait_reported == "Breast cancer"
        assert result.trait_efo == ["EFO_0000305"]
        assert result.variants_number == 313
        assert result.genome_build == "GRCh37"
        assert result.ftp_harmonized_files["GRCh37"] == "https://example.com/grch37.txt.gz"
        assert result.publication_doi == "10.1234/example"
        assert result.publication_pmid == "12345678"

    def test_parse_minimal_metadata(self):
        """Test parsing minimal API response."""
        data = {
            "id": "PGS000001",
        }

        result = _parse_api_metadata(data)

        assert result.pgs_id == "PGS000001"
        assert result.name == ""
        assert result.variants_number == 0
        assert result.ftp_harmonized_files == {}


class TestMetadataCache:
    """Tests for metadata caching functions."""

    def test_save_and_load_metadata(self):
        """Test saving and loading metadata from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            metadata = PGSCatalogMetadata(
                pgs_id="PGS000004",
                name="PRS313",
                trait_reported="Breast cancer",
                trait_efo=["EFO_0000305"],
                variants_number=313,
                genome_build="GRCh37",
                ftp_scoring_file="https://example.com/file.txt.gz",
                ftp_harmonized_files={"GRCh37": "https://example.com/grch37.txt.gz"},
                publication_doi="10.1234/example",
            )

            _save_metadata_to_cache(metadata, cache_dir)
            loaded = _load_cached_metadata("PGS000004", cache_dir)

            assert loaded is not None
            assert loaded.pgs_id == "PGS000004"
            assert loaded.name == "PRS313"
            assert loaded.variants_number == 313

    def test_load_missing_cache_returns_none(self):
        """Test that loading non-existent cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            result = _load_cached_metadata("PGS999999", cache_dir)

            assert result is None

    def test_load_invalid_cache_returns_none(self):
        """Test that loading invalid JSON cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_path = cache_dir / "PGS000004_metadata.json"
            cache_path.write_text("invalid json")

            result = _load_cached_metadata("PGS000004", cache_dir)

            assert result is None


class TestFetchMetadata:
    """Tests for fetch_pgs_catalog_metadata function."""

    @patch("imputed_prs.io.pgs_catalog._request_with_retry")
    def test_fetch_success(self, mock_request):
        """Test successful metadata fetch with mocked API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "PGS000004",
            "name": "PRS313",
            "trait_reported": "Breast cancer",
            "trait_efo": [],
            "variants_number": 313,
        }
        mock_request.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            result = fetch_pgs_catalog_metadata("PGS000004", use_cache=False, cache_dir=cache_dir)

            assert result.pgs_id == "PGS000004"
            assert result.name == "PRS313"
            assert result.variants_number == 313

    def test_invalid_pgs_id_raises(self):
        """Test that invalid PGS ID raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid PGS ID format"):
            fetch_pgs_catalog_metadata("INVALID")

    @patch("imputed_prs.io.pgs_catalog._request_with_retry")
    def test_uses_cache(self, mock_request):
        """Test that cached metadata is used when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create cached metadata
            metadata = PGSCatalogMetadata(
                pgs_id="PGS000004",
                name="Cached PRS",
                trait_reported="Cached trait",
                trait_efo=[],
                variants_number=100,
                genome_build="GRCh37",
                ftp_scoring_file=None,
            )
            _save_metadata_to_cache(metadata, cache_dir)

            result = fetch_pgs_catalog_metadata("PGS000004", use_cache=True, cache_dir=cache_dir)

            # Should use cache, not call API
            mock_request.assert_not_called()
            assert result.name == "Cached PRS"


class TestDownloadScore:
    """Tests for download_pgs_catalog_score function."""

    @patch("imputed_prs.io.pgs_catalog._request_with_retry")
    def test_download_and_normalize(self, mock_request):
        """Test full download and normalization pipeline with mocks."""
        # Mock metadata response
        metadata_response = MagicMock()
        metadata_response.json.return_value = {
            "id": "PGS000004",
            "name": "PRS313",
            "trait_reported": "Breast cancer",
            "trait_efo": [],
            "variants_number": 3,
        }

        # Create test scoring file content
        scoring_content = (
            "# PGS000004\n"
            "rsID\thm_rsID\tchr_name\thm_chr\tchr_position\thm_pos\teffect_allele\teffect_weight\thm_code\n"
            "rs123\trs123\t1\t1\t100\t100\tA\t0.1\t0\n"
            "rs456\trs456\t2\t2\t200\t200\tG\t0.2\t0\n"
            "rs789\trs789\t3\t3\t300\t300\tT\t0.3\t-1\n"
        )

        # Mock file download response
        download_response = MagicMock()
        download_response.content = gzip.compress(scoring_content.encode())

        mock_request.side_effect = [metadata_response, download_response]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            df, metadata = download_pgs_catalog_score(
                "PGS000004",
                genome_build="GRCh37",
                cache_dir=cache_dir,
                use_cache=False,
            )

            # Check metadata
            assert metadata.pgs_id == "PGS000004"

            # Check DataFrame - should have 2 rows (one filtered by hm_code=-1)
            assert len(df) == 2
            assert "variant_id" in df.columns
            assert "chromosome" in df.columns
            assert "position" in df.columns
            assert "effect_allele" in df.columns
            assert "beta" in df.columns

    @patch("imputed_prs.io.pgs_catalog._request_with_retry")
    def test_cache_used(self, mock_request):
        """Test that cached scoring file is used when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create cached metadata
            metadata = PGSCatalogMetadata(
                pgs_id="PGS000004",
                name="PRS313",
                trait_reported="Breast cancer",
                trait_efo=[],
                variants_number=2,
                genome_build="GRCh37",
                ftp_scoring_file=None,
            )
            _save_metadata_to_cache(metadata, cache_dir)

            # Create cached scoring file
            scoring_content = (
                "rsID\teffect_allele\teffect_weight\n"
                "rs123\tA\t0.1\n"
                "rs456\tG\t0.2\n"
            )
            cache_path = cache_dir / "PGS000004_hmPOS_GRCh37.txt.gz"
            with gzip.open(cache_path, "wt") as f:
                f.write(scoring_content)

            df, _ = download_pgs_catalog_score(
                "PGS000004",
                genome_build="GRCh37",
                cache_dir=cache_dir,
                use_cache=True,
            )

            # API should not be called for cached data
            mock_request.assert_not_called()
            assert len(df) == 2

    def test_invalid_pgs_id_raises(self):
        """Test that invalid PGS ID raises ValidationError."""
        with pytest.raises(ValidationError):
            download_pgs_catalog_score("INVALID")

    def test_invalid_build_raises(self):
        """Test that invalid genome build raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported genome build"):
            download_pgs_catalog_score("PGS000004", genome_build="GRCh36")


class TestSearchCatalog:
    """Tests for search_pgs_catalog function."""

    @patch("imputed_prs.io.pgs_catalog._request_with_retry")
    def test_search_returns_results(self, mock_request):
        """Test search with mocked API."""
        # Mock search response
        search_response = MagicMock()
        search_response.json.return_value = {
            "results": [
                {"associated_pgs_ids": ["PGS000004", "PGS000005"]},
            ]
        }

        # Mock metadata responses - use a function to return appropriate response
        def mock_api_call(url, *args, **kwargs):
            response = MagicMock()
            if "PGS000004" in url:
                response.json.return_value = {
                    "id": "PGS000004",
                    "name": "PRS313",
                    "trait_reported": "Breast cancer",
                    "trait_efo": [],
                    "variants_number": 313,
                }
            elif "PGS000005" in url:
                response.json.return_value = {
                    "id": "PGS000005",
                    "name": "Other PRS",
                    "trait_reported": "Breast cancer risk",
                    "trait_efo": [],
                    "variants_number": 100,
                }
            else:
                # Search response
                response.json.return_value = {
                    "results": [
                        {"associated_pgs_ids": ["PGS000004", "PGS000005"]},
                    ]
                }
            return response

        mock_request.side_effect = mock_api_call

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use temp cache to avoid polluting real cache
            with patch("imputed_prs.io.pgs_catalog.DEFAULT_CACHE_DIR", Path(tmpdir)):
                results = search_pgs_catalog("breast cancer", limit=10)

        assert len(results) == 2
        pgs_ids = {r.pgs_id for r in results}
        assert pgs_ids == {"PGS000004", "PGS000005"}

    def test_empty_query_raises(self):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            search_pgs_catalog("")

    def test_whitespace_query_raises(self):
        """Test that whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            search_pgs_catalog("   ")


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_clear_cache(self):
        """Test clearing cache removes all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create some cache files
            (cache_dir / "PGS000004_metadata.json").write_text("{}")
            (cache_dir / "PGS000004_hmPOS_GRCh37.txt.gz").write_bytes(b"test")
            (cache_dir / "PGS000005_metadata.json").write_text("{}")

            count = clear_pgs_catalog_cache(cache_dir)

            assert count == 3
            assert len(list(cache_dir.iterdir())) == 0

    def test_clear_empty_cache(self):
        """Test clearing non-existent cache returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nonexistent"
            count = clear_pgs_catalog_cache(cache_dir)

            assert count == 0

    def test_get_cache_info(self):
        """Test getting cache info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create some cache files
            (cache_dir / "PGS000004_metadata.json").write_text('{"test": 1}')
            (cache_dir / "PGS000004_hmPOS_GRCh37.txt.gz").write_bytes(b"test content")
            (cache_dir / "PGS000005_metadata.json").write_text('{"test": 2}')

            info = get_pgs_catalog_cache_info(cache_dir)

            assert info["path"] == str(cache_dir)
            assert info["n_files"] == 3
            assert info["size_bytes"] > 0
            assert set(info["cached_scores"]) == {"PGS000004", "PGS000005"}

    def test_get_cache_info_empty(self):
        """Test getting cache info for non-existent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nonexistent"
            info = get_pgs_catalog_cache_info(cache_dir)

            assert info["n_files"] == 0
            assert info["size_bytes"] == 0
            assert info["cached_scores"] == []


# Integration tests - require network access
@pytest.mark.integration
@pytest.mark.network
class TestPGSCatalogIntegration:
    """Integration tests that require network access."""

    def test_fetch_pgs000004_metadata(self):
        """Fetch real PGS000004 metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = fetch_pgs_catalog_metadata("PGS000004", cache_dir=cache_dir)

            assert metadata.pgs_id == "PGS000004"
            assert metadata.variants_number == 313
            assert "breast" in metadata.trait_reported.lower()

    def test_download_pgs000004_grch37(self):
        """Download real PGS000004 for GRCh37."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            df, metadata = download_pgs_catalog_score(
                "PGS000004",
                genome_build="GRCh37",
                cache_dir=cache_dir,
            )

            assert "variant_id" in df.columns
            assert "chromosome" in df.columns
            assert "position" in df.columns
            assert "effect_allele" in df.columns
            assert "beta" in df.columns
            assert len(df) > 300  # ~313 after filtering
            assert df["beta"].dtype == float

    def test_search_breast_cancer(self):
        """Search for breast cancer scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("imputed_prs.io.pgs_catalog.DEFAULT_CACHE_DIR", Path(tmpdir)):
                results = search_pgs_catalog("breast cancer", limit=5)

        assert len(results) > 0
        # Check that results have valid structure
        for result in results:
            assert result.pgs_id.startswith("PGS")
            assert result.variants_number > 0
