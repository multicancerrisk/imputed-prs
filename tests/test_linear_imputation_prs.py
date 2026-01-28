"""Tests for the LinearImputationPRS class shell."""

import pytest

from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import ModelNotFittedError


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

    def test_fit_raises_not_implemented_error(self):
        """Test fit() raises NotImplementedError (stub)."""
        model = LinearImputationPRS()

        with pytest.raises(NotImplementedError, match="Phase 7.2"):
            model.fit(
                reference_genotypes="test.vcf",
                prs_definition="PGS000004",
                platform_name="test_platform",
            )

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
