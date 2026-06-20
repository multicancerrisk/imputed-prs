"""Tests for the LinearImputationPRS class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import DataLoadError, ModelNotFittedError, ValidationError
from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    PredictionResult,
    VariantInfo,
)


class TestLinearImputationPRSConstructor:
    """Test constructor and initialization."""

    def test_default_parameters(self):
        """Test constructor with default parameters."""
        model = LinearImputationPRS()

        assert model.window_size == 1_000_000
        assert model.tuning_scope == "global"
        assert model.l1_ratio == 0.5
        assert model.alpha == 0.01
        assert model.cv_folds == 5
        assert model.n_jobs == 1
        assert model.random_state is None
        assert model.max_predictors is None
        assert model.max_tuning_variants == 50
        assert model.verbose == 1

    def test_invalid_tuning_scope_raises(self):
        """An unsupported tuning_scope is rejected, not silently accepted."""
        from imputed_prs.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            LinearImputationPRS(tuning_scope="bogus")

    def test_custom_parameters(self):
        """Test constructor with custom parameters."""
        model = LinearImputationPRS(
            window_size=500_000,
            tuning_scope="per_variant",
            l1_ratio=0.8,
            alpha=0.05,
            cv_folds=10,
            n_jobs=4,
            random_state=42,
            max_predictors=100,
            verbose=2,
        )

        assert model.window_size == 500_000
        assert model.tuning_scope == "per_variant"
        assert model.l1_ratio == 0.8
        assert model.alpha == 0.05
        assert model.cv_folds == 10
        assert model.n_jobs == 4
        assert model.random_state == 42
        assert model.max_predictors == 100
        assert model.verbose == 2

    def test_tuning_scope_none(self):
        """Test constructor with tuning_scope='none'."""
        model = LinearImputationPRS(tuning_scope="none")
        assert model.tuning_scope == "none"

    def test_initial_unfitted_state(self):
        """Test that model starts in unfitted state."""
        model = LinearImputationPRS()

        assert model._is_fitted is False
        assert model._observed_variants is None
        assert model._imputed_models is None
        assert model._calibration_params is None
        assert model._evaluation_metrics is None
        assert model._training_result is None
        assert model._platform_variant_index is None
        assert model._prs_id is None
        assert model._platform_name is None
        assert model._genome_build is None
        assert model._model_name is None


class TestLinearImputationPRSIsFitted:
    """Test is_fitted property."""

    def test_is_fitted_returns_false_before_fit(self):
        """Test is_fitted returns False before fit() is called."""
        model = LinearImputationPRS()
        assert model.is_fitted is False


class TestLinearImputationPRSModelNotFittedErrors:
    """Test ModelNotFittedError is raised for unfitted model."""

    def test_predict_raises_model_not_fitted_error(self):
        """Test predict() raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict({})

    def test_export_raises_model_not_fitted_error(self):
        """Test export() raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.export("/tmp/output")

    def test_variant_table_raises_model_not_fitted_error(self):
        """Test variant_table raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.variant_table

    def test_summary_raises_model_not_fitted_error(self):
        """Test summary raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.summary

    def test_evaluation_metrics_raises_model_not_fitted_error(self):
        """Test evaluation_metrics raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.evaluation_metrics

    def test_calibration_params_raises_model_not_fitted_error(self):
        """Test calibration_params raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.calibration_params

    def test_observed_variants_raises_model_not_fitted_error(self):
        """Test observed_variants raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.observed_variants

    def test_imputed_models_raises_model_not_fitted_error(self):
        """Test imputed_models raises ModelNotFittedError before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.imputed_models


class TestLinearImputationPRSRepr:
    """Test __repr__ method."""

    def test_repr_unfitted_model(self):
        """Test __repr__ works for unfitted model."""
        model = LinearImputationPRS(window_size=500_000, cv_folds=10)
        repr_str = repr(model)

        assert "LinearImputationPRS" in repr_str
        assert "window_size=500000" in repr_str
        assert "cv_folds=10" in repr_str
        assert "status=not fitted" in repr_str

    def test_repr_default_model(self):
        """Test __repr__ with default parameters."""
        model = LinearImputationPRS()
        repr_str = repr(model)

        assert "window_size=1000000" in repr_str
        assert "cv_folds=5" in repr_str


class TestLinearImputationPRSImports:
    """Test that LinearImputationPRS is properly exported."""

    def test_import_from_package_root(self):
        """Test import from imputed_prs package root."""
        from imputed_prs import LinearImputationPRS as PRS
        assert PRS is not None
        model = PRS()
        assert model.is_fitted is False

    def test_import_from_core(self):
        """Test import from imputed_prs.core."""
        from imputed_prs.core import LinearImputationPRS as PRS
        assert PRS is not None
        model = PRS()
        assert model.is_fitted is False

    def test_import_from_module(self):
        """Test import from full module path."""
        from imputed_prs.core.linear_imputation_prs import LinearImputationPRS as PRS
        assert PRS is not None
        model = PRS()
        assert model.is_fitted is False


# =============================================================================
# Fixtures for fit() method tests
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
    """Create a list of platform variants (partial overlap with PRS)."""
    # Only rs1, rs2, rs3 are on platform; rs4, rs5 need imputation
    return ["rs1", "rs2", "rs3"]


@pytest.fixture
def platform_variants_all():
    """Create a list of platform variants (full overlap with PRS)."""
    return ["rs1", "rs2", "rs3", "rs4", "rs5"]


class TestLinearImputationPRSObservedAlleleGate:
    """P1.2 fit-time allele-aware observed inclusion."""

    def _model(self):
        return LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

    def test_allele_incompatible_observed_is_reclassified(self, synthetic_vcf_file):
        """An observed locus whose alleles mismatch the reference is not labeled
        observed; it is dropped-with-reason, never mis-scored as 2*beta."""
        pytest.importorskip("cyvcf2")
        # Reference rs1 is A/G at 1:100000; declare an incompatible A/C there
        # ({A,C} is neither {A,G} nor its complement {T,C}).
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
        assert "rs1" not in observed_ids  # not mis-labeled observed
        # Allele-compatible siblings remain observed.
        assert "rs2" in observed_ids
        assert "rs3" in observed_ids

        disp = model.variant_dispositions.set_index("variant_id")
        assert disp.loc["rs1", "status"] != "observed"
        assert disp.loc["rs1", "reason"] == "allele_mismatch"

    def test_not_in_reference_observed_is_kept(self, synthetic_vcf_file):
        """A platform variant absent from the (chr1-only) reference stays observed:
        it remains directly scoreable from the user's genotype."""
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
        assert "rs99" in observed_ids  # not dropped despite absence from reference
        assert "rs1" in observed_ids  # in-reference, allele-compatible: stays observed


class TestLinearImputationPRSOrientedPredict:
    """P1.2 end-to-end: predict() scores observed terms by the effect allele."""

    def test_predict_dataframe_is_allele_oriented(self, synthetic_vcf_file):
        """A genotype-string upload counts the effect allele, not homozygosity."""
        pytest.importorskip("cyvcf2")
        # rs1 effect=A is the REF at 1:100000 (ref A/G); rs2 effect=T is the ALT.
        prs_df = pd.DataFrame({
            "variant_id": ["rs1", "rs2"],
            "chromosome": ["1", "1"],
            "position": [100000, 100500],
            "effect_allele": ["A", "T"],
            "other_allele": ["G", "C"],
            "beta": [1.0, 2.0],
        })
        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, tuning_scope="none",
            verbose=0, random_state=42,
        )
        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=prs_df,
            platform_variants=["rs1", "rs2"],
        )

        # rs1 "GG" -> 0 copies of effect A (NOT 2); rs2 "TT" -> 2 copies of effect T.
        user_df = pd.DataFrame({
            "rsid": ["rs1", "rs2"],
            "chrom": ["1", "1"],
            "pos": [100000, 100500],
            "genotype": ["GG", "TT"],
        })
        result = model.predict(user_df, apply_calibration=False)

        # Oriented observed = 0*1.0 + 2*2.0 = 4.0 (allele-blind would give 6.0).
        np.testing.assert_allclose(
            result.prs_observed_component, 4.0, rtol=0, atol=1e-12
        )
        assert result.n_observed_scored_direct == 2
        assert result.unresolved_observed_ids == ()


# =============================================================================
# Tests for fit() method input validation
# =============================================================================


class TestLinearImputationPRSFitInputValidation:
    """Test fit() method input validation."""

    def test_fit_raises_validation_error_if_no_platform_source(self, synthetic_vcf_file, synthetic_prs_df):
        """Test fit() raises ValidationError if no platform source is provided."""
        model = LinearImputationPRS(verbose=0)

        with pytest.raises(ValidationError, match="Exactly one platform source"):
            model.fit(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                # No platform source provided
            )

    def test_fit_raises_validation_error_if_multiple_platform_sources(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() raises ValidationError if multiple platform sources are provided."""
        model = LinearImputationPRS(verbose=0)

        with pytest.raises(ValidationError, match="Exactly one platform source"):
            model.fit(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                platform_name="23andme_v5",
                platform_variants=platform_variants_partial,
            )


# =============================================================================
# Tests for fit() method with platform_variants
# =============================================================================


class TestLinearImputationPRSFitWithPlatformVariants:
    """Test fit() with platform_variants list."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_platform_variants_sets_is_fitted(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() with platform_variants sets _is_fitted to True."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        result = model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model._is_fitted is True
        assert model.is_fitted is True

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_threads_predictor_allele_metadata(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """fit() threads ALT-counted predictor allele metadata to each model.

        P1.3: every imputed model must expose, per predictor, the reference row
        backing its Z column (counted=ALT, other=REF) plus chr/pos/AF, index-
        aligned with the coefficients.
        """
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )
        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        # Reference rows from the synthetic VCF: (chrom, pos, ALT, REF).
        expected = {
            "rs1": ("1", 100000, "G", "A"),
            "rs2": ("1", 100500, "T", "C"),
            "rs3": ("1", 101000, "A", "G"),
        }

        n_with_predictors = 0
        for m in model._imputed_models:
            n_pred = len(m.predictor_variant_ids)
            # All predictor metadata is index-aligned with predictor_variant_ids.
            assert len(m.predictor_chromosomes) == n_pred
            assert len(m.predictor_positions) == n_pred
            assert len(m.predictor_counted_alleles) == n_pred
            assert len(m.predictor_other_alleles) == n_pred
            assert len(m.predictor_allele_frequencies) == n_pred

            if n_pred > 0:
                n_with_predictors += 1
            for i, pred_id in enumerate(m.predictor_variant_ids):
                chrom, pos, counted, other = expected[pred_id]
                assert m.predictor_chromosomes[i] == chrom
                assert m.predictor_positions[i] == pos
                # Z counts ALT: counted == VCF ALT, other == VCF REF.
                assert m.predictor_counted_alleles[i] == counted
                assert m.predictor_other_alleles[i] == other
                assert 0.0 <= m.predictor_allele_frequencies[i] <= 1.0

        # rs4 and rs5 are imputed and have rs1-rs3 as in-window predictors.
        assert n_with_predictors > 0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_platform_variants_returns_self(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() returns self for method chaining."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        result = model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert result is model

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_platform_variants_populates_observed_variants(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() populates _observed_variants correctly."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model._observed_variants is not None
        assert len(model._observed_variants) == 3  # rs1, rs2, rs3

        # Check that observed variants are VariantInfo objects
        for var in model._observed_variants:
            assert isinstance(var, VariantInfo)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_platform_variants_populates_imputed_models(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() populates _imputed_models correctly."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model._imputed_models is not None
        assert len(model._imputed_models) == 2  # rs4, rs5

        # Check that imputed models are ImputedVariantModel objects
        for m in model._imputed_models:
            assert isinstance(m, ImputedVariantModel)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_platform_variants_populates_metadata(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() populates metadata correctly."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            prs_id="TEST_PRS",
            model_name="Test Model",
        )

        assert model._prs_id == "TEST_PRS"
        assert model._platform_name == "custom"
        assert model._model_name == "Test Model"


# =============================================================================
# Tests for fit() with DataFrame PRS definition
# =============================================================================


class TestLinearImputationPRSFitWithDataFrame:
    """Test fit() with DataFrame PRS definition."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_dataframe_prs_definition(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() with DataFrame PRS definition."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model.is_fitted is True


# =============================================================================
# Tests for fit() with tuning_scope options
# =============================================================================


class TestLinearImputationPRSFitTuningScope:
    """Test fit() with different tuning_scope options."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_tuning_scope_none(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() with tuning_scope='none' uses provided hyperparameters."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            l1_ratio=0.8,
            alpha=0.05,
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model.is_fitted is True

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_tuning_scope_global(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() with tuning_scope='global' runs hyperparameter search."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="global",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model.is_fitted is True

    def test_fit_with_tuning_scope_per_variant(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() with tuning_scope='per_variant' tunes each variant and fits."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="per_variant",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert model.is_fitted is True
        # rs4, rs5 are off-platform and recovered via per-variant-tuned imputation.
        assert len(model.imputed_models) == 2


# =============================================================================
# Tests for properties after fit()
# =============================================================================


class TestLinearImputationPRSPropertiesAfterFit:
    """Test properties work correctly after fit()."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_variant_table_after_fit(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test variant_table property works after fit()."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        vt = model.variant_table
        assert isinstance(vt, pd.DataFrame)
        assert len(vt) == 5  # All variants
        assert "variant_id" in vt.columns
        assert "status" in vt.columns

        # Check that observed and imputed variants are correctly labeled
        observed_count = (vt["status"] == "observed").sum()
        imputed_count = vt["status"].isin(["imputed", "intercept_only"]).sum()
        assert observed_count == 3
        assert imputed_count == 2

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_summary_after_fit(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test summary property works after fit()."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        summary = model.summary
        assert isinstance(summary, dict)
        assert summary["n_total_variants"] == 5
        assert summary["n_observed"] == 3
        assert summary["n_imputed"] == 2
        assert summary["window_size"] == 500_000
        assert summary["cv_folds"] == 3

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_observed_variants_after_fit(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test observed_variants property works after fit()."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        obs_vars = model.observed_variants
        assert isinstance(obs_vars, list)
        assert len(obs_vars) == 3

        variant_ids = {v.variant_id for v in obs_vars}
        assert variant_ids == {"rs1", "rs2", "rs3"}

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_imputed_models_after_fit(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test imputed_models property works after fit()."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        imp_models = model.imputed_models
        assert isinstance(imp_models, list)
        assert len(imp_models) == 2

        variant_ids = {m.variant_id for m in imp_models}
        assert variant_ids == {"rs4", "rs5"}


# =============================================================================
# Tests for fit() with all variants observed (no imputation needed)
# =============================================================================


class TestLinearImputationPRSFitAllObserved:
    """Test fit() when all PRS variants are on platform."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_all_variants_observed(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_all
    ):
        """Test fit() when all PRS variants are observed (no imputation needed)."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_all,
        )

        assert model.is_fitted is True
        assert len(model.observed_variants) == 5
        assert len(model.imputed_models) == 0

        summary = model.summary
        assert summary["n_observed"] == 5
        assert summary["n_imputed"] == 0


# =============================================================================
# Tests for fit() method chaining
# =============================================================================


class TestLinearImputationPRSFitMethodChaining:
    """Test fit() method chaining."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_returns_self_for_chaining(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test fit() returns self for method chaining."""
        pytest.importorskip("cyvcf2")

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        # Method chaining should work
        result = model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        assert result is model
        assert result.is_fitted is True


# =============================================================================
# Tests for fit() with PRS file path
# =============================================================================


class TestLinearImputationPRSFitWithFilePath:
    """Test fit() with PRS definition as file path."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_fit_with_prs_file_path(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial, tmp_path
    ):
        """Test fit() with PRS definition loaded from file path."""
        pytest.importorskip("cyvcf2")

        # Write PRS to file
        prs_path = tmp_path / "test_prs.csv"
        synthetic_prs_df.to_csv(prs_path, index=False)

        model = LinearImputationPRS(
            window_size=500_000,
            cv_folds=3,
            tuning_scope="none",
            verbose=0,
            random_state=42,
        )

        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=str(prs_path),
            platform_variants=platform_variants_partial,
        )

        assert model.is_fitted is True
        assert len(model.observed_variants) == 3
        assert len(model.imputed_models) == 2


# =============================================================================
# Fixtures for predict() method tests
# =============================================================================


@pytest.fixture
def fitted_model(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial):
    """Create a fitted LinearImputationPRS model for prediction tests."""
    cyvcf2 = pytest.importorskip("cyvcf2")

    model = LinearImputationPRS(
        window_size=500_000,
        cv_folds=3,
        tuning_scope="none",
        verbose=0,
        random_state=42,
    )

    model.fit(
        reference_genotypes=synthetic_vcf_file,
        prs_definition=synthetic_prs_df,
        platform_variants=platform_variants_partial,
    )

    return model


@pytest.fixture
def deployable_model(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial):
    """A fitted model carrying provenance, so export() passes the deploy gate (P1.7).

    Kept separate from ``fitted_model`` so build-less predict tests stay quiet:
    a model with a known build warns when predicted on a build-less dict/frame.
    """
    pytest.importorskip("cyvcf2")

    model = LinearImputationPRS(
        window_size=500_000,
        cv_folds=3,
        tuning_scope="none",
        verbose=0,
        random_state=42,
    )

    model.fit(
        reference_genotypes=synthetic_vcf_file,
        prs_definition=synthetic_prs_df,
        platform_variants=platform_variants_partial,
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )

    return model


@pytest.fixture
def fitted_model_all_observed(synthetic_vcf_file, synthetic_prs_df, platform_variants_all):
    """Create a fitted model where all variants are observed (no imputation)."""
    cyvcf2 = pytest.importorskip("cyvcf2")

    model = LinearImputationPRS(
        window_size=500_000,
        cv_folds=3,
        tuning_scope="none",
        verbose=0,
        random_state=42,
    )

    model.fit(
        reference_genotypes=synthetic_vcf_file,
        prs_definition=synthetic_prs_df,
        platform_variants=platform_variants_all,
    )

    return model


@pytest.fixture
def user_dosages_dict():
    """Create a sample user dosages dictionary."""
    return {
        "rs1": 1.0,  # heterozygous
        "rs2": 0.0,  # homozygous ref
        "rs3": 2.0,  # homozygous alt
        "rs4": 1.0,  # heterozygous
        "rs5": 0.0,  # homozygous ref
    }


@pytest.fixture
def user_genotypes_df():
    """Create a sample user genotypes DataFrame."""
    return pd.DataFrame({
        "rsid": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "genotype": ["AG", "CC", "AA", "TC", "AA"],
    })


# =============================================================================
# Tests for predict() method
# =============================================================================


class TestLinearImputationPRSPredict:
    """Test predict() method."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_raises_model_not_fitted_error(self):
        """Test predict() raises ModelNotFittedError if called before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict({"rs1": 1.0})

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_with_dict_input(self, fitted_model, user_dosages_dict):
        """Test predict() with Dict[str, float] input works correctly."""
        result = fitted_model.predict(user_dosages_dict)

        assert isinstance(result, PredictionResult)
        assert result.prs is not None
        assert isinstance(result.prs, float)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_with_dataframe_input(self, fitted_model, user_genotypes_df):
        """Test predict() with DataFrame input loads genotypes correctly."""
        result = fitted_model.predict(user_genotypes_df)

        assert isinstance(result, PredictionResult)
        assert result.prs is not None
        assert isinstance(result.prs, float)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_returns_prediction_result_with_valid_values(
        self, fitted_model, user_dosages_dict
    ):
        """Test predict() returns PredictionResult with valid values."""
        result = fitted_model.predict(user_dosages_dict)

        # Check that all required fields are populated
        assert hasattr(result, "prs")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "prs_observed_component")
        assert hasattr(result, "prs_imputed_component")
        assert hasattr(result, "n_variants_used")
        assert hasattr(result, "n_variants_imputed")
        assert hasattr(result, "n_variants_intercept_only")
        assert hasattr(result, "n_user_variants_missing")
        assert hasattr(result, "n_truncated")

        # Verify numeric types
        assert isinstance(result.prs, float)
        assert isinstance(result.se, float)
        assert isinstance(result.ci_lower, float)
        assert isinstance(result.ci_upper, float)
        assert isinstance(result.n_variants_used, int)
        assert isinstance(result.n_variants_imputed, int)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_confidence_intervals_are_valid(self, fitted_model, user_dosages_dict):
        """Test predict() confidence intervals are valid (ci_lower < prs < ci_upper)."""
        result = fitted_model.predict(user_dosages_dict)

        # CI should bracket the point estimate (or be equal if SE is 0)
        assert result.ci_lower <= result.prs
        assert result.prs <= result.ci_upper

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_component_counts_are_accurate(self, fitted_model, user_dosages_dict):
        """Test predict() component counts are accurate."""
        result = fitted_model.predict(user_dosages_dict)

        # Total variants used should be sum of observed (that user has) + imputed
        # n_variants_imputed should match the number of imputed models
        assert result.n_variants_imputed == len(fitted_model.imputed_models)

        # n_variants_used should be reasonable
        assert result.n_variants_used >= 0
        assert result.n_variants_used <= (
            len(fitted_model.observed_variants) + len(fitted_model.imputed_models)
        )

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_with_apply_calibration_false(self, fitted_model, user_dosages_dict):
        """Test predict() with apply_calibration=False returns raw values."""
        result = fitted_model.predict(user_dosages_dict, apply_calibration=False)

        assert isinstance(result, PredictionResult)
        assert result.prs is not None

        # When calibration is not applied, scaled values should be None
        # (unless calibration params are None anyway)
        if fitted_model.calibration_params is None:
            assert result.prs_scaled is None
        # If calibration was requested but not applied, scaled values should be None
        assert result.prs_scaled is None

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_with_apply_calibration_true(self, fitted_model, user_dosages_dict):
        """Test predict() with apply_calibration=True applies scaling when available."""
        result = fitted_model.predict(user_dosages_dict, apply_calibration=True)

        assert isinstance(result, PredictionResult)

        # If calibration params exist, scaled values should be populated
        if fitted_model.calibration_params is not None:
            assert result.prs_scaled is not None
            assert result.se_scaled is not None
            assert result.ci_lower_scaled is not None
            assert result.ci_upper_scaled is not None
        else:
            # No calibration params means scaled values are None
            assert result.prs_scaled is None

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_handles_missing_user_variants_gracefully(self, fitted_model):
        """Test predict() handles missing user variants gracefully."""
        # Only provide some variants
        partial_dosages = {
            "rs1": 1.0,
            "rs2": 0.0,
            # Missing: rs3, rs4, rs5
        }

        result = fitted_model.predict(partial_dosages)

        assert isinstance(result, PredictionResult)
        assert result.prs is not None
        # Should report missing variants
        assert result.n_user_variants_missing >= 0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_handles_empty_dosages(self, fitted_model):
        """Test predict() handles empty dosages dictionary."""
        result = fitted_model.predict({})

        assert isinstance(result, PredictionResult)
        # With no user data, imputation falls back to intercept-only
        assert result.n_user_variants_missing > 0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_with_all_observed_model(
        self, fitted_model_all_observed, user_dosages_dict
    ):
        """Test predict() with model where all variants are observed."""
        result = fitted_model_all_observed.predict(user_dosages_dict)

        assert isinstance(result, PredictionResult)
        assert result.n_variants_imputed == 0
        assert result.prs_imputed_component == 0.0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_prs_components_sum_to_total(self, fitted_model, user_dosages_dict):
        """Test predict() observed + imputed components sum to total PRS."""
        result = fitted_model.predict(user_dosages_dict)

        # The raw PRS should be sum of observed and imputed components
        expected_prs = result.prs_observed_component + result.prs_imputed_component
        assert abs(result.prs - expected_prs) < 1e-10

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_standard_error_non_negative(self, fitted_model, user_dosages_dict):
        """Test predict() standard error is non-negative."""
        result = fitted_model.predict(user_dosages_dict)

        assert result.se >= 0.0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_predict_consistent_results(self, fitted_model, user_dosages_dict):
        """Test predict() returns consistent results for same input."""
        result1 = fitted_model.predict(user_dosages_dict)
        result2 = fitted_model.predict(user_dosages_dict)

        assert result1.prs == result2.prs
        assert result1.se == result2.se
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper


# =============================================================================
# Tests for _get_expected_variants() helper method
# =============================================================================


class TestLinearImputationPRSGetExpectedVariants:
    """Test _get_expected_variants() helper method."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_expected_variants_includes_observed(self, fitted_model):
        """Test _get_expected_variants includes observed variant IDs."""
        expected = fitted_model._get_expected_variants()

        for var in fitted_model.observed_variants:
            assert var.variant_id in expected

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_expected_variants_includes_predictor_variants(self, fitted_model):
        """Test _get_expected_variants includes predictor variant IDs from models."""
        expected = fitted_model._get_expected_variants()

        for model in fitted_model.imputed_models:
            for pred_id in model.predictor_variant_ids:
                assert pred_id in expected

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_expected_variants_returns_set(self, fitted_model):
        """Test _get_expected_variants returns a set."""
        expected = fitted_model._get_expected_variants()

        assert isinstance(expected, set)


# =============================================================================
# Tests for export() method
# =============================================================================


class TestLinearImputationPRSExport:
    """Test export() method."""

    @pytest.fixture
    def fitted_model(self, deployable_model):
        # export() enforces the provenance deploy gate (P1.7), so these tests run
        # against a model that declares build + reference panel + ancestry.
        return deployable_model

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_raises_model_not_fitted_error(self, tmp_path):
        """Test export() raises ModelNotFittedError if called before fit()."""
        model = LinearImputationPRS()

        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.export(tmp_path)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_default_formats(self, fitted_model, tmp_path):
        """Test export() with default formats (json, hdf5)."""
        paths = fitted_model.export(tmp_path)

        assert "json" in paths
        assert "hdf5" in paths
        assert paths["json"].exists()
        assert paths["hdf5"].exists()

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_json_format(self, fitted_model, tmp_path):
        """Test export() to JSON format."""
        paths = fitted_model.export(tmp_path, formats=["json"])

        assert "json" in paths
        assert paths["json"].suffix == ".json"
        assert paths["json"].exists()

        # Verify JSON is valid
        import json
        with open(paths["json"]) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "observed_variants" in data
        assert "imputed_variants" in data

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_hdf5_format(self, fitted_model, tmp_path):
        """Test export() to HDF5 format."""
        paths = fitted_model.export(tmp_path, formats=["hdf5"])

        assert "hdf5" in paths
        assert paths["hdf5"].suffix == ".h5"
        assert paths["hdf5"].exists()

        # Verify HDF5 is valid
        import h5py
        with h5py.File(paths["hdf5"], "r") as f:
            assert "metadata" in f
            assert "observed_variants" in f
            assert "imputed_variants" in f

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_arrow_format(self, fitted_model, tmp_path):
        """Test export() to Arrow format."""
        paths = fitted_model.export(tmp_path, formats=["arrow"])

        assert "arrow" in paths
        assert paths["arrow"].suffix == ".arrow"
        assert paths["arrow"].exists()

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_parquet_format(self, fitted_model, tmp_path):
        """Test export() to Parquet format."""
        paths = fitted_model.export(tmp_path, formats=["parquet"])

        assert "parquet" in paths
        # Parquet returns a directory
        assert paths["parquet"].is_dir()
        # Should have parquet files inside
        parquet_files = list(paths["parquet"].glob("*.parquet"))
        assert len(parquet_files) > 0

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_csv_format(self, fitted_model, tmp_path):
        """Test export() to CSV format."""
        paths = fitted_model.export(tmp_path, formats=["csv"])

        assert "csv" in paths
        assert paths["csv"].suffix == ".csv"
        assert paths["csv"].exists()

        # Verify CSV contents
        df = pd.read_csv(paths["csv"])
        assert "variant_id" in df.columns
        assert "status" in df.columns
        # One row per observed + imputed variant, plus one per observed-variant
        # fallback model (P1.8), which is serialized as its own status row.
        n_fallback = sum(
            1 for v in fitted_model.observed_variants if v.fallback is not None
        )
        assert len(df) == (
            len(fitted_model.observed_variants)
            + len(fitted_model.imputed_models)
            + n_fallback
        )

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_multiple_formats(self, fitted_model, tmp_path):
        """Test export() with multiple formats."""
        paths = fitted_model.export(tmp_path, formats=["json", "hdf5", "csv"])

        assert len(paths) == 3
        assert all(p.exists() for p in paths.values())

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_with_custom_model_name(self, fitted_model, tmp_path):
        """Test export() with custom model name."""
        paths = fitted_model.export(tmp_path, model_name="my_custom_model", formats=["json"])

        assert "my_custom_model" in str(paths["json"])

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_invalid_format_raises_error(self, fitted_model, tmp_path):
        """Test export() raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="Unsupported export formats"):
            fitted_model.export(tmp_path, formats=["invalid_format"])

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_creates_output_directory(self, fitted_model, tmp_path):
        """Test export() creates output directory if it doesn't exist."""
        new_dir = tmp_path / "new_subdir" / "export"
        assert not new_dir.exists()

        paths = fitted_model.export(new_dir, formats=["json"])

        assert new_dir.exists()
        assert paths["json"].exists()

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_export_without_variance_scaling(self, fitted_model, tmp_path):
        """Test export() with include_variance_scaling=False."""
        paths = fitted_model.export(
            tmp_path, formats=["json"], include_variance_scaling=False
        )

        import json
        with open(paths["json"]) as f:
            data = json.load(f)

        # Check metadata indicates variance scaling is excluded
        assert data["metadata"]["include_variance_scaling"] is False


# =============================================================================
# Tests for load() class method
# =============================================================================


class TestLinearImputationPRSLoad:
    """Test load() class method."""

    def test_load_raises_error_for_nonexistent_file(self, tmp_path):
        """Test load() raises DataLoadError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.h5"

        with pytest.raises(DataLoadError, match="Model file not found"):
            LinearImputationPRS.load(nonexistent)

    def test_load_raises_error_for_unsupported_format(self, tmp_path):
        """Test load() raises DataLoadError for unsupported file format."""
        unsupported = tmp_path / "model.xyz"
        unsupported.touch()

        with pytest.raises(DataLoadError, match="Unsupported model file format"):
            LinearImputationPRS.load(unsupported)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_load_hdf5_roundtrip(self, fitted_model, tmp_path, user_dosages_dict):
        """Test save to HDF5 and load roundtrip."""
        # Export to HDF5
        paths = fitted_model.export(tmp_path, formats=["hdf5"])

        # Load back
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # Verify loaded model is fitted
        assert loaded_model.is_fitted

        # Verify variant counts match
        assert len(loaded_model.observed_variants) == len(fitted_model.observed_variants)
        assert len(loaded_model.imputed_models) == len(fitted_model.imputed_models)

        # Verify predictions match
        original_result = fitted_model.predict(user_dosages_dict)
        loaded_result = loaded_model.predict(user_dosages_dict)

        assert np.isclose(original_result.prs, loaded_result.prs)
        assert np.isclose(original_result.se, loaded_result.se)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_load_json_roundtrip(self, deployable_model, tmp_path, user_dosages_dict):
        """Test save to JSON and load roundtrip."""
        # JSON export enforces the provenance deploy gate (P1.7), so use a model
        # that declares build + provenance.
        fitted_model = deployable_model
        # Export to JSON
        paths = fitted_model.export(tmp_path, formats=["json"])

        # Load back
        loaded_model = LinearImputationPRS.load(paths["json"])

        # Verify loaded model is fitted
        assert loaded_model.is_fitted

        # Verify variant counts match
        assert len(loaded_model.observed_variants) == len(fitted_model.observed_variants)
        assert len(loaded_model.imputed_models) == len(fitted_model.imputed_models)

        # Verify predictions match (pass the model's build so the guard is silent)
        original_result = fitted_model.predict(user_dosages_dict, genome_build="GRCh37")
        loaded_result = loaded_model.predict(user_dosages_dict, genome_build="GRCh37")

        assert np.isclose(original_result.prs, loaded_result.prs)
        assert np.isclose(original_result.se, loaded_result.se)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_load_preserves_metadata(self, fitted_model, tmp_path):
        """Test load() preserves model metadata."""
        # Set some metadata
        fitted_model._prs_id = "TEST_PRS_ID"
        fitted_model._platform_name = "test_platform"
        fitted_model._genome_build = "GRCh37"
        fitted_model._model_name = "test_model"

        # Export and load
        paths = fitted_model.export(tmp_path, formats=["hdf5"])
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # Verify metadata
        assert loaded_model._prs_id == "TEST_PRS_ID"
        assert loaded_model._platform_name == "test_platform"
        assert loaded_model._genome_build == "GRCh37"
        assert loaded_model._model_name == "test_model"

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_load_preserves_calibration_params(self, fitted_model, tmp_path):
        """Test load() preserves calibration parameters if present."""
        # Export and load
        paths = fitted_model.export(tmp_path, formats=["hdf5"])
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # If original had calibration params, loaded should too
        if fitted_model.calibration_params is not None:
            assert loaded_model.calibration_params is not None
            assert np.isclose(
                loaded_model.calibration_params.scaling_factor,
                fitted_model.calibration_params.scaling_factor,
            )

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_loaded_model_can_predict(self, fitted_model, tmp_path, user_dosages_dict):
        """Test loaded model can make predictions."""
        # Export and load
        paths = fitted_model.export(tmp_path, formats=["hdf5"])
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # Should be able to predict
        result = loaded_model.predict(user_dosages_dict)

        assert isinstance(result, PredictionResult)
        assert not np.isnan(result.prs)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_loaded_model_summary_works(self, fitted_model, tmp_path):
        """Test loaded model summary property works."""
        # Export and load
        paths = fitted_model.export(tmp_path, formats=["hdf5"])
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # Summary should work
        summary = loaded_model.summary
        assert "n_observed" in summary
        assert "n_imputed" in summary
        assert summary["n_observed"] == len(fitted_model.observed_variants)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_loaded_model_variant_table_works(self, fitted_model, tmp_path):
        """Test loaded model variant_table property works."""
        # Export and load
        paths = fitted_model.export(tmp_path, formats=["hdf5"])
        loaded_model = LinearImputationPRS.load(paths["hdf5"])

        # Variant table should work
        vt = loaded_model.variant_table
        assert isinstance(vt, pd.DataFrame)
        assert len(vt) == len(fitted_model.observed_variants) + len(fitted_model.imputed_models)


class TestVariantDispositionsAndCoverage:
    """Tests for honest coverage reporting and per-variant dispositions."""

    def _fit(self, vcf, prs_df, platform, **kwargs):
        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, tuning_scope="none",
            verbose=0, random_state=42, **kwargs,
        )
        model.fit(reference_genotypes=vcf, prs_definition=prs_df,
                  platform_variants=platform)
        return model

    def test_summary_coverage_keys_all_found(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """All variants found -> full coverage and the new honest-coverage keys."""
        pytest.importorskip("cyvcf2")
        s = self._fit(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial).summary
        assert s["n_definition_variants"] == 5
        assert s["n_total_variants"] == 5
        assert s["n_dropped"] == 0
        assert s["dropped_by_reason"] == {}
        assert s["coverage"] == 1.0

    def test_variant_table_has_reason_column(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        pytest.importorskip("cyvcf2")
        vt = self._fit(synthetic_vcf_file, synthetic_prs_df,
                       platform_variants_partial).variant_table
        assert "reason" in vt.columns
        assert len(vt) == 5

    def test_dropped_variant_not_in_reference(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """A PRS variant absent from the reference is recorded, not silently lost."""
        pytest.importorskip("cyvcf2")
        prs = pd.concat([synthetic_prs_df, pd.DataFrame([{
            "variant_id": "rs_missing", "chromosome": "1", "position": 200000,
            "effect_allele": "A", "other_allele": "G", "beta": 0.3,
        }])], ignore_index=True)

        model = self._fit(synthetic_vcf_file, prs, platform_variants_partial)
        s = model.summary
        assert s["n_definition_variants"] == 6  # full definition, not post-drop
        assert s["n_dropped"] == 1
        assert s["coverage"] < 1.0
        assert s["dropped_by_reason"].get("not_in_reference") == 1

        vt = model.variant_table
        assert len(vt) == 6
        row = vt[vt["variant_id"] == "rs_missing"].iloc[0]
        assert row["status"] == "dropped"
        assert row["reason"] == "not_in_reference"

    def test_reference_contig_missing(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """A variant on a contig absent from the reference is flagged distinctly."""
        pytest.importorskip("cyvcf2")
        prs = pd.concat([synthetic_prs_df, pd.DataFrame([{
            "variant_id": "rsX", "chromosome": "X", "position": 5000,
            "effect_allele": "A", "other_allele": "G", "beta": 0.3,
        }])], ignore_index=True)

        model = self._fit(synthetic_vcf_file, prs, platform_variants_partial)
        vt = model.variant_table
        row = vt[vt["variant_id"] == "rsX"].iloc[0]
        assert row["status"] == "dropped"
        assert row["reason"] == "reference_contig_missing"
        assert model.summary["dropped_by_reason"].get("reference_contig_missing") == 1

    def test_exclude_ambiguous_drops_palindrome(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """rs5 is a palindromic A/T SNP with MAF~0.45; QC drops it only when enabled."""
        pytest.importorskip("cyvcf2")
        kept = self._fit(synthetic_vcf_file, synthetic_prs_df,
                         platform_variants_partial, exclude_ambiguous=False)
        assert "rs5" not in set(
            kept.variant_table.loc[kept.variant_table.status == "dropped", "variant_id"]
        )

        dropped = self._fit(synthetic_vcf_file, synthetic_prs_df,
                            platform_variants_partial,
                            exclude_ambiguous=True, ambiguous_maf_threshold=0.4)
        vt = dropped.variant_table
        row = vt[vt["variant_id"] == "rs5"].iloc[0]
        assert row["status"] == "dropped"
        assert row["reason"] == "ambiguous_excluded"
        assert dropped.summary["dropped_by_reason"].get("ambiguous_excluded") == 1

    def test_disposition_completeness(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Every input PRS variant appears exactly once in the variant table."""
        pytest.importorskip("cyvcf2")
        model = self._fit(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial)
        vt = model.variant_table
        assert len(vt) == len(synthetic_prs_df)
        assert sorted(vt["variant_id"]) == sorted(synthetic_prs_df["variant_id"])


class TestObservedFallbackTraining:
    """P1.8 — per-observed-variant fallback models trained at fit time."""

    def _fit(self, vcf, prs_df, platform, **kwargs):
        pytest.importorskip("cyvcf2")
        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, tuning_scope="none",
            verbose=0, random_state=42, **kwargs,
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
        fallback (no training target) and is recorded, not silently."""
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


class TestLinearImputationPRSTrainingFailureSurfacing:
    """P5.1: a missing-variant training failure surfaces *why* it failed in both
    variant_dispositions and summary (not merely an opaque count)."""

    def test_training_failure_in_dispositions_and_summary(
        self, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial, monkeypatch
    ):
        pytest.importorskip("cyvcf2")
        from imputed_prs.models.elastic_net import (
            fit_single_variant_model as real_fit,
        )

        # Fail only the first per-variant fit (the first missing variant, rs4);
        # rs5 and the observed fallbacks still train, so the pipeline completes.
        calls = {"n": 0}

        def fail_first(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("synthetic fit failure")
            return real_fit(*args, **kwargs)

        monkeypatch.setattr(
            "imputed_prs.models.trainer.fit_single_variant_model", fail_first
        )

        model = LinearImputationPRS(
            window_size=500_000, cv_folds=3, tuning_scope="none",
            verbose=0, random_state=42,
        )
        model.fit(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
        )

        disp = model.variant_dispositions.set_index("variant_id")
        # rs4 failed training -> dropped, reason unchanged, but now explained.
        assert disp.loc["rs4", "status"] == "dropped"
        assert disp.loc["rs4", "reason"] == "training_failed"
        assert disp.loc["rs4", "failure_error_type"] == "ValueError"
        assert "synthetic fit failure" in disp.loc["rs4", "failure_error_message"]
        # A non-failed variant carries no failure detail.
        assert pd.isna(disp.loc["rs1", "failure_error_type"])

        summary = model.summary
        assert summary["n_training_failed"] == 1
        assert summary["training_failures_by_type"] == {"ValueError": 1}
