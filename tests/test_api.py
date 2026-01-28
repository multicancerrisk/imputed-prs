"""Tests for top-level API imports."""

import pytest


class TestTopLevelImports:
    """Test that all documented imports work from top level."""

    def test_import_main_class(self):
        """Test LinearImputationPRS import."""
        from imputed_prs import LinearImputationPRS
        assert callable(LinearImputationPRS)

    def test_import_platform_functions(self):
        """Test platform convenience functions."""
        from imputed_prs import list_available_platforms, get_platform_info
        assert callable(list_available_platforms)
        assert callable(get_platform_info)

    def test_import_pgs_catalog_functions(self):
        """Test PGS Catalog convenience functions."""
        from imputed_prs import (
            search_pgs_catalog,
            fetch_pgs_catalog_score,
            clear_pgs_catalog_cache,
        )
        assert callable(search_pgs_catalog)
        assert callable(fetch_pgs_catalog_score)
        assert callable(clear_pgs_catalog_cache)

    def test_import_evaluator(self):
        """Test ImputationEvaluator import."""
        from imputed_prs import ImputationEvaluator
        assert ImputationEvaluator is not None

    def test_import_types(self):
        """Test type imports."""
        from imputed_prs import PlatformInfo, PredictionResult
        assert PlatformInfo is not None
        assert PredictionResult is not None

    def test_import_exception(self):
        """Test base exception import."""
        from imputed_prs import ImputedPRSError
        assert issubclass(ImputedPRSError, Exception)

    def test_version_available(self):
        """Test __version__ is available."""
        from imputed_prs import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_all_exports_defined(self):
        """Test __all__ contains expected exports."""
        import imputed_prs
        expected = {
            "__version__",
            "LinearImputationPRS",
            "list_available_platforms",
            "get_platform_info",
            "search_pgs_catalog",
            "fetch_pgs_catalog_score",
            "clear_pgs_catalog_cache",
            "ImputationEvaluator",
            "PlatformInfo",
            "PredictionResult",
            "ImputedPRSError",
        }
        assert set(imputed_prs.__all__) == expected


class TestConvenienceFunctionsBehavior:
    """Test that convenience functions work correctly from top level."""

    def test_list_available_platforms_returns_list(self):
        """Test list_available_platforms returns expected platforms."""
        from imputed_prs import list_available_platforms
        platforms = list_available_platforms()
        assert isinstance(platforms, list)
        assert "23andme_v5" in platforms

    def test_get_platform_info_returns_platforminfo(self):
        """Test get_platform_info returns PlatformInfo."""
        from imputed_prs import get_platform_info, PlatformInfo
        info = get_platform_info("23andme_v5")
        assert isinstance(info, PlatformInfo)
        assert info.name == "23andme_v5"

    def test_fetch_is_alias_for_download(self):
        """Test fetch_pgs_catalog_score is alias for download_pgs_catalog_score."""
        from imputed_prs import fetch_pgs_catalog_score
        from imputed_prs.io import download_pgs_catalog_score
        assert fetch_pgs_catalog_score is download_pgs_catalog_score
