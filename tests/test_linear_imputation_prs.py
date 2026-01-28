"""Tests for the LinearImputationPRS class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.types import CalibrationParams, VariantInfo, ImputedVariantModel


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
        assert model.verbose == 1

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


class TestLinearImputationPRSStubMethods:
    """Test stub methods raise NotImplementedError."""

    def test_load_raises_not_implemented_error(self):
        """Test load() raises NotImplementedError (stub)."""
        with pytest.raises(NotImplementedError, match="Phase 7.4"):
            LinearImputationPRS.load("/path/to/model.hdf5")


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
