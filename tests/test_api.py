"""Tests for top-level API imports and the predict/export compatibility guards."""

import contextlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS, PredictionResult
from imputed_prs.core import IncompatibleBuildError, IncompatiblePlatformError
from imputed_prs.io.exporters.json_export import export_to_json


@contextlib.contextmanager
def warnings_recorder():
    """Record all warnings without turning them into errors."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield records


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
            "LinearProjectionPRS",
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


# =============================================================================
# Fixtures for the predict/export compatibility guards (P1.7)
# =============================================================================

# 20-sample biallelic reference; effect_allele == ALT for every PRS variant.
# Platform is rs1-rs3 (a partial overlap), so the fitted model has observed
# (rs1-rs3) and imputed (rs4, rs5) terms. Mirrors tests/test_round_trip.py.
_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\tS5\tS6\tS7\tS8\tS9\tS10\tS11\tS12\tS13\tS14\tS15\tS16\tS17\tS18\tS19\tS20
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1
1\t102000\trs5\tG\tC\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1
"""

_PRS_DF = pd.DataFrame(
    {
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "C"],
        "other_allele": ["A", "C", "G", "T", "G"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    }
)

_PLATFORM = ["rs1", "rs2", "rs3"]


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "ref.vcf"
    path.write_text(_VCF)
    return path


@pytest.fixture
def fitted_model(vcf_file):
    """An imputation model fit as a deployable artifact: build + provenance set."""
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file,
        prs_definition=_PRS_DF,
        platform_variants=_PLATFORM,
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def model_no_provenance(vcf_file):
    """A research model fit without build/reference/ancestry provenance."""
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file, prs_definition=_PRS_DF, platform_variants=_PLATFORM
    )
    return model


@pytest.fixture
def user_df():
    """A user upload covering the platform variants (rs1-rs3), as genotype strings."""
    return pd.DataFrame(
        {
            "rsid": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "1"],
            "position": [100000, 100500, 101000],
            "genotype": ["AG", "CC", "AA"],
        }
    )


class TestPredictBuildPlatformGuard:
    """predict() refuses/hard-blocks on incompatible build or platform (P1.7).

    The fitted model declares build GRCh37 and (via platform_variants) the
    platform name "custom".
    """

    def test_build_match_returns_result(self, fitted_model, user_df):
        result = fitted_model.predict(user_df, apply_calibration=False, genome_build="GRCh37")
        assert isinstance(result, PredictionResult)
        assert np.isfinite(result.prs)

    def test_build_alias_match_ok(self, fitted_model, user_df):
        # "hg19" normalizes to GRCh37 and must be accepted.
        result = fitted_model.predict(user_df, apply_calibration=False, genome_build="hg19")
        assert isinstance(result, PredictionResult)

    def test_build_mismatch_raises(self, fitted_model, user_df):
        with pytest.raises(IncompatibleBuildError, match="build mismatch"):
            fitted_model.predict(user_df, apply_calibration=False, genome_build="GRCh38")

    def test_unknown_user_build_warns_and_scores(self, fitted_model, user_df):
        # A DataFrame upload carries no build and cannot be auto-detected; the
        # model declares GRCh37, so the guard blocks loudly but still scores.
        with pytest.warns(UserWarning, match="determine the genome build"):
            result = fitted_model.predict(user_df, apply_calibration=False)
        assert isinstance(result, PredictionResult)

    def test_strict_false_downgrades_build_mismatch(self, fitted_model, user_df):
        with pytest.warns(UserWarning, match="mismatch"):
            result = fitted_model.predict(
                user_df, apply_calibration=False, genome_build="GRCh38", strict=False
            )
        assert isinstance(result, PredictionResult)

    def test_platform_match_ok(self, fitted_model, user_df):
        result = fitted_model.predict(
            user_df, apply_calibration=False, genome_build="GRCh37", platform_id="custom"
        )
        assert isinstance(result, PredictionResult)

    def test_platform_mismatch_raises(self, fitted_model, user_df):
        with pytest.raises(IncompatiblePlatformError, match="platform mismatch"):
            fitted_model.predict(
                user_df,
                apply_calibration=False,
                genome_build="GRCh37",
                platform_id="illumina_gsa",
            )

    def test_strict_false_downgrades_platform_mismatch(self, fitted_model, user_df):
        with pytest.warns(UserWarning, match="platform mismatch"):
            result = fitted_model.predict(
                user_df,
                apply_calibration=False,
                genome_build="GRCh37",
                platform_id="illumina_gsa",
                strict=False,
            )
        assert isinstance(result, PredictionResult)

    def test_guard_is_side_effect_free(self, fitted_model, user_df):
        # The guard's warn/raise decision must never perturb the score: a matched
        # build and an unknown (warned) build score the same upload identically.
        r_match = fitted_model.predict(user_df, apply_calibration=False, genome_build="GRCh37")
        with pytest.warns(UserWarning):
            r_unknown = fitted_model.predict(user_df, apply_calibration=False)
        np.testing.assert_allclose(r_unknown.prs, r_match.prs, rtol=0, atol=1e-12)

    def test_model_without_build_predicts_silently(self, model_no_provenance, user_df):
        # A model with no declared build cannot check compatibility, so the guard
        # must emit no build/platform warning.
        with warnings_recorder() as records:
            result = model_no_provenance.predict(user_df, apply_calibration=False)
        assert isinstance(result, PredictionResult)
        guard_msgs = [
            str(r.message)
            for r in records
            if "genome build" in str(r.message).lower()
            or "platform mismatch" in str(r.message).lower()
        ]
        assert guard_msgs == [], guard_msgs

    def test_projection_predict_guard_raises(self, vcf_file, user_df):
        # The same guard applies to the projection product.
        pytest.importorskip("cyvcf2")
        model = LinearProjectionPRS(window_size=500_000, cv_folds=3, verbose=0, random_state=42)
        model.fit(
            reference_genotypes=vcf_file,
            prs_definition=_PRS_DF,
            platform_variants=_PLATFORM,
            genome_build="GRCh37",
        )
        with pytest.raises(IncompatibleBuildError, match="build mismatch"):
            model.predict(user_df, apply_calibration=False, genome_build="GRCh38")


class TestExportProvenanceGate:
    """A deployable export must carry non-null provenance (P1.7)."""

    def test_model_export_without_provenance_raises(self, model_no_provenance, tmp_path):
        with pytest.raises(ValueError, match="provenance"):
            model_no_provenance.export(tmp_path, model_name="m", formats=["json"])

    def test_model_export_with_provenance_succeeds(self, fitted_model, tmp_path):
        paths = fitted_model.export(tmp_path, model_name="m", formats=["json"])
        assert paths["json"].exists()
        data = json.loads(paths["json"].read_text())
        prov = data["provenance"]
        assert prov["genome_build"] == "GRCh37"
        assert prov["reference_panel_id"] == "1000G_phase3_EUR"
        assert prov["training_ancestry"] == "EUR"

    def test_export_to_json_missing_provenance_raises(self, fitted_model, tmp_path):
        with pytest.raises(ValueError, match="provenance"):
            export_to_json(
                output_path=tmp_path / "research.json",
                observed_variants=fitted_model._observed_variants,
                imputed_models=fitted_model._imputed_models,
            )

    def test_export_to_json_research_escape_hatch(self, fitted_model, tmp_path):
        path = tmp_path / "research.json"
        export_to_json(
            output_path=path,
            observed_variants=fitted_model._observed_variants,
            imputed_models=fitted_model._imputed_models,
            require_provenance=False,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["provenance"]["genome_build"] is None
