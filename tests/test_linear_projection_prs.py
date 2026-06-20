"""Tests for the LinearProjectionPRS class."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearProjectionPRS
from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.types import (
    PredictionResult,
    ProjectionRegionModel,
)


# =============================================================================
# Fixtures (same synthetic data as test_linear_imputation_prs.py)
# =============================================================================


@pytest.fixture
def synthetic_vcf_file(tmp_path):
    """Create a synthetic VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\tSAMPLE3\tSAMPLE4\tSAMPLE5\tSAMPLE6\tSAMPLE7\tSAMPLE8\tSAMPLE9\tSAMPLE10
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0
1\t102000\trs5\tA\tT\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1
"""
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    return vcf_path


@pytest.fixture
def synthetic_prs_df():
    """Create a synthetic PRS DataFrame for testing."""
    return pd.DataFrame({
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "T"],
        "other_allele": ["A", "C", "G", "T", "A"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    })


@pytest.fixture
def platform_variants_partial():
    """Platform variants with partial overlap: rs4, rs5 need projection."""
    return ["rs1", "rs2", "rs3"]


@pytest.fixture
def platform_variants_all():
    """Platform variants with full overlap: no projection needed."""
    return ["rs1", "rs2", "rs3", "rs4", "rs5"]


@pytest.fixture
def fitted_model(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial):
    """Create a fitted LinearProjectionPRS model."""
    cyvcf2 = pytest.importorskip("cyvcf2")

    model = LinearProjectionPRS(
        window_size=500_000,
        cv_folds=3,
        verbose=0,
        random_state=42,
    )
    model.fit(
        reference_genotypes=synthetic_vcf_file,
        prs_definition=synthetic_prs_df,
        platform_variants=platform_variants_partial,
    )
    return model


class TestLinearProjectionPRSObservedAlleleGate:
    """P1.2 fit-time allele-aware observed inclusion (projection product)."""

    def _model(self):
        return LinearProjectionPRS(
            window_size=500_000,
            cv_folds=3,
            verbose=0,
            random_state=42,
        )

    def test_allele_incompatible_observed_is_reclassified(self, synthetic_vcf_file):
        """An observed locus whose alleles mismatch the reference is not labeled
        observed; it is dropped-with-reason, never mis-scored as 2*beta."""
        pytest.importorskip("cyvcf2")
        # Reference rs1 is A/G at 1:100000; declare an incompatible A/C there.
        prs_df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "1"],
            "position": [100000, 100500, 101000],
            "effect_allele": ["A", "T", "A"],
            "other_allele": ["C", "C", "G"],
            "beta": [0.5, -0.05, 0.2],
        })
        model = self._model()
        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=prs_df,
            platform_variants=["rs1", "rs2", "rs3"],
        )

        observed_ids = {v.variant_id for v in model.observed_variants}
        assert "rs1" not in observed_ids
        assert "rs2" in observed_ids
        assert "rs3" in observed_ids

        disp = model.variant_dispositions.set_index("variant_id")
        assert disp.loc["rs1", "status"] != "observed"
        assert disp.loc["rs1", "reason"] == "allele_mismatch"

    def test_not_in_reference_observed_is_kept(self, synthetic_vcf_file):
        """A platform variant absent from the (chr1-only) reference stays observed."""
        pytest.importorskip("cyvcf2")
        prs_df = pd.DataFrame({
            "variant_id": ["rs1", "rs2", "rs99"],
            "chromosome": ["1", "1", "2"],
            "position": [100000, 100500, 5000],
            "effect_allele": ["G", "T", "A"],
            "other_allele": ["A", "C", "G"],
            "beta": [0.1, -0.05, 0.3],
        })
        model = self._model()
        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=prs_df,
            platform_variants=["rs1", "rs2", "rs99"],
        )

        observed_ids = {v.variant_id for v in model.observed_variants}
        assert "rs99" in observed_ids
        assert "rs1" in observed_ids


@pytest.fixture
def user_dosages_dict():
    """Sample user dosages dictionary."""
    return {
        "rs1": 1.0,
        "rs2": 0.0,
        "rs3": 2.0,
        "rs4": 1.0,
        "rs5": 0.0,
    }


# =============================================================================
# Constructor tests
# =============================================================================


class TestLinearProjectionPRSConstructor:
    """Test constructor and initialization."""

    def test_default_parameters(self):
        """Default parameters match expected values."""
        model = LinearProjectionPRS()

        assert model.window_size == 1_000_000
        assert model.tuning_scope == "global"
        assert model.l1_ratio == 0.5
        assert model.alpha == 0.01
        assert model.cv_folds == 5
        assert model.n_jobs == 1
        assert model.random_state is None
        assert model.max_predictors is None
        assert model.max_tuning_regions == 50
        assert model.verbose == 1

    def test_invalid_tuning_scope_raises(self):
        """An unsupported tuning_scope is rejected (projection has no per_variant)."""
        from imputed_prs.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            LinearProjectionPRS(tuning_scope="per_variant")

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""
        model = LinearProjectionPRS(
            window_size=500_000,
            l1_ratio=0.8,
            alpha=0.05,
            cv_folds=10,
            n_jobs=4,
            random_state=42,
            max_predictors=100,
            verbose=2,
        )

        assert model.window_size == 500_000
        assert model.l1_ratio == 0.8
        assert model.alpha == 0.05
        assert model.cv_folds == 10
        assert model.n_jobs == 4
        assert model.random_state == 42
        assert model.max_predictors == 100
        assert model.verbose == 2

    def test_initial_unfitted_state(self):
        """is_fitted returns False before fit()."""
        model = LinearProjectionPRS()
        assert model.is_fitted is False


# =============================================================================
# Unfitted error tests
# =============================================================================


class TestLinearProjectionPRSUnfittedErrors:
    """Test ModelNotFittedError is raised for unfitted model."""

    def test_predict_raises_error(self):
        """predict() before fit() raises ModelNotFittedError."""
        model = LinearProjectionPRS()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict({})

    def test_summary_raises_error(self):
        """summary before fit() raises ModelNotFittedError."""
        model = LinearProjectionPRS()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.summary

    def test_region_models_raises_error(self):
        """region_models before fit() raises ModelNotFittedError."""
        model = LinearProjectionPRS()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.region_models

    def test_variant_table_raises_error(self):
        """variant_table before fit() raises ModelNotFittedError."""
        model = LinearProjectionPRS()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.variant_table


# =============================================================================
# Fit validation tests
# =============================================================================


class TestLinearProjectionPRSFitValidation:
    """Test fit() method input validation."""

    def test_no_platform_source(self, synthetic_vcf_file, synthetic_prs_df):
        """No platform source raises ValidationError."""
        model = LinearProjectionPRS(verbose=0)
        with pytest.raises(ValidationError, match="Exactly one platform source"):
            model.fit(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
            )

    def test_multiple_platform_sources(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Multiple platform sources raises ValidationError."""
        model = LinearProjectionPRS(verbose=0)
        with pytest.raises(ValidationError, match="Exactly one platform source"):
            model.fit(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                platform_name="23andme_v5",
                platform_variants=platform_variants_partial,
            )


# =============================================================================
# Predict tests
# =============================================================================


class TestLinearProjectionPRSPredict:
    """Test predict() method."""

    def test_predict_returns_prediction_result(self, fitted_model, user_dosages_dict):
        """predict() returns PredictionResult type."""
        result = fitted_model.predict(user_dosages_dict)
        assert isinstance(result, PredictionResult)

    def test_predict_dict_input(self, fitted_model, user_dosages_dict):
        """predict() accepts Dict[str, float] input."""
        result = fitted_model.predict(user_dosages_dict)
        assert isinstance(result.prs, float)
        assert isinstance(result.se, float)

    def test_predict_with_calibration(self, fitted_model, user_dosages_dict):
        """Calibration is applied when available and requested."""
        result = fitted_model.predict(user_dosages_dict, apply_calibration=True)
        # If calibration was computed, scaled values should be present
        if fitted_model.calibration_params is not None:
            assert result.prs_scaled is not None

    def test_predict_without_calibration(self, fitted_model, user_dosages_dict):
        """No calibration when apply_calibration=False."""
        result = fitted_model.predict(user_dosages_dict, apply_calibration=False)
        assert result.prs_scaled is None


# =============================================================================
# Property tests
# =============================================================================


class TestLinearProjectionPRSProperties:
    """Test properties of fitted model."""

    def test_summary_keys(self, fitted_model):
        """summary dict has all expected keys."""
        summary = fitted_model.summary
        expected_keys = {
            "n_observed_variants",
            "n_observed_with_fallback",
            "n_missing_variants",
            "n_definition_variants",
            "n_dropped",
            "dropped_by_reason",
            "coverage",
            "n_regions",
            "n_intercept_only_regions",
            "training_summary",
            "calibration",
            "prs_id",
            "platform_name",
            "genome_build",
            "model_name",
            "window_size",
            "cv_folds",
        }
        assert set(summary.keys()) == expected_keys

    def test_variant_table_columns(self, fitted_model):
        """variant_table DataFrame has expected columns."""
        vt = fitted_model.variant_table
        expected_columns = {
            "region_id",
            "chromosome",
            "start",
            "end",
            "n_prs_variants",
            "n_predictors",
            "cv_r2",
            "cv_mse",
            "is_intercept_only",
            "prs_variant_ids",
        }
        assert set(vt.columns) == expected_columns

    def test_region_models_type(self, fitted_model):
        """region_models returns List[ProjectionRegionModel]."""
        models = fitted_model.region_models
        assert isinstance(models, list)
        for m in models:
            assert isinstance(m, ProjectionRegionModel)

    def test_is_fitted_after_fit(self, fitted_model):
        """is_fitted returns True after fit()."""
        assert fitted_model.is_fitted is True


class TestObservedFallbackTraining:
    """P2.4 — per-observed-variant fallback models trained at fit time, with
    parity to the imputation product's P1.8."""

    def _fit(self, vcf, prs_df, platform, **kwargs):
        pytest.importorskip("cyvcf2")
        model = LinearProjectionPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42, **kwargs,
        )
        model.fit(reference_genotypes=vcf, prs_definition=prs_df,
                  platform_variants=platform)
        return model

    def test_observed_variants_get_fallbacks(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Every in-reference observed variant carries a trained fallback model."""
        model = self._fit(synthetic_vcf_file, synthetic_prs_df,
                          platform_variants_partial)
        assert model.observed_variants
        assert all(v.fallback is not None for v in model.observed_variants)
        assert model.summary["n_observed_with_fallback"] == len(
            model.observed_variants
        )
        disp = model.variant_dispositions.set_index("variant_id")
        for vid in ["rs1", "rs2", "rs3"]:
            assert bool(disp.loc[vid, "has_fallback"]) is True
            assert disp.loc[vid, "fallback_reason"] is None

    def test_fallback_training_is_deterministic(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Two fits with the same random_state produce identical fallback models."""
        m1 = self._fit(synthetic_vcf_file, synthetic_prs_df,
                       platform_variants_partial)
        m2 = self._fit(synthetic_vcf_file, synthetic_prs_df,
                       platform_variants_partial)
        fb1 = {v.variant_id: v.fallback for v in m1.observed_variants}
        fb2 = {v.variant_id: v.fallback for v in m2.observed_variants}
        assert fb1.keys() == fb2.keys()
        for vid in fb1:
            np.testing.assert_array_equal(
                fb1[vid].coefficients, fb2[vid].coefficients
            )
            assert fb1[vid].intercept == fb2[vid].intercept
            assert fb1[vid].predictor_variant_ids == fb2[vid].predictor_variant_ids

    def test_no_call_observed_recovered_end_to_end(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """A no-call observed variant is recovered via its fallback, not dropped."""
        model = self._fit(synthetic_vcf_file, synthetic_prs_df,
                          platform_variants_partial)
        upload = pd.DataFrame(
            {"rsid": ["rs1", "rs2", "rs3"], "genotype": ["AG", "--", "AA"]}
        )
        r = model.predict(upload, apply_calibration=False)
        assert r.n_observed_scored_direct == 2  # rs1, rs3 counted directly
        assert r.n_observed_scored_via_fallback == 1  # rs2 recovered
        assert "rs2" not in (r.unresolved_observed_ids or ())
        assert r.weighted_beta_via_fallback == 0.05  # |beta(rs2)| = 0.05

    def test_fully_resolvable_upload_uses_no_fallback(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """When every observed variant resolves, scoring is direct and exact."""
        model = self._fit(synthetic_vcf_file, synthetic_prs_df,
                          platform_variants_partial)
        upload = pd.DataFrame(
            {"rsid": ["rs1", "rs2", "rs3"], "genotype": ["AG", "CC", "AA"]}
        )
        r = model.predict(upload, apply_calibration=False)
        assert r.n_observed_scored_direct == 3
        assert r.n_observed_scored_via_fallback == 0
        # rs1 "AG" counts 1 G (effect=G, beta 0.1); rs2 "CC" counts 0 T
        # (effect=T, beta -0.05); rs3 "AA" counts 2 A (effect=A, beta 0.2).
        np.testing.assert_allclose(
            r.prs_observed_component, 1 * 0.1 + 0 * -0.05 + 2 * 0.2,
            rtol=0, atol=1e-12,
        )

    def test_observed_absent_from_reference_has_no_fallback(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """An observed variant whose locus is absent from the reference gets no
        fallback (no training target) and is recorded, not silently dropped."""
        prs = pd.concat([synthetic_prs_df, pd.DataFrame([{
            "variant_id": "rs999", "chromosome": "1", "position": 5000,
            "effect_allele": "A", "other_allele": "G", "beta": 0.3,
        }])], ignore_index=True)
        platform = platform_variants_partial + ["rs999"]
        model = self._fit(synthetic_vcf_file, prs, platform)
        by_id = {v.variant_id: v for v in model.observed_variants}
        assert "rs999" in by_id
        assert by_id["rs999"].fallback is None
        disp = model.variant_dispositions.set_index("variant_id")
        assert disp.loc["rs999", "status"] == "observed"
        assert bool(disp.loc["rs999", "has_fallback"]) is False
        assert disp.loc["rs999", "fallback_reason"] == "no_reference_target"
