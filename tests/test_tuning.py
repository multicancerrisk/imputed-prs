"""Tests for hyperparameter tuning module."""

import numpy as np
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.types import GridSearchResult
from imputed_prs.models.tuning import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    global_hyperparameter_search,
)


class TestGlobalHyperparameterSearch:
    """Tests for basic global_hyperparameter_search functionality."""

    def test_basic_usage(self):
        """Test basic grid search with simple data."""
        rng = np.random.default_rng(42)
        n, m, p = 500, 100, 20

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        assert isinstance(result, GridSearchResult)
        assert result.best_l1_ratio in DEFAULT_L1_RATIOS
        assert result.best_alpha in DEFAULT_ALPHAS
        assert result.best_mean_cv_mse >= 0
        assert result.n_variants_sampled == p
        assert result.n_variants_failed >= 0

    def test_custom_hyperparameter_grids(self):
        """Test grid search with custom l1_ratios and alphas."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        custom_l1_ratios = [0.2, 0.8]
        custom_alphas = [0.005, 0.05]

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            l1_ratios=custom_l1_ratios,
            alphas=custom_alphas,
            cv_folds=3,
            random_state=42,
        )

        assert result.best_l1_ratio in custom_l1_ratios
        assert result.best_alpha in custom_alphas

    def test_sample_indices(self):
        """Test using sample_indices to select specific variants."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 20

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        # Only use first 5 variants
        sample_indices = list(range(5))

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            sample_indices=sample_indices,
            cv_folds=3,
            random_state=42,
        )

        assert result.n_variants_sampled == 5

    def test_default_grids_used_when_none(self):
        """Test that default grids are used when not specified."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            l1_ratios=None,
            alphas=None,
            cv_folds=3,
            random_state=42,
        )

        # Should have evaluated all default combinations
        expected_combos = len(DEFAULT_L1_RATIOS) * len(DEFAULT_ALPHAS)
        assert len(result.grid_results) == expected_combos


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_l1_ratios_raises_error(self):
        """Test that empty l1_ratios raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="l1_ratios cannot be empty"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                l1_ratios=[],
            )

    def test_empty_alphas_raises_error(self):
        """Test that empty alphas raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="alphas cannot be empty"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                alphas=[],
            )

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (80, 10)).astype(float)  # Different n_samples

        with pytest.raises(ValidationError, match="Shape mismatch"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
            )

    def test_invalid_l1_ratio_raises_error(self):
        """Test that l1_ratio outside [0, 1] raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="l1_ratio must be between"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                l1_ratios=[0.5, 1.5],  # 1.5 is invalid
            )

    def test_negative_alpha_raises_error(self):
        """Test that negative alpha raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="alpha must be non-negative"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                alphas=[0.01, -0.1],  # -0.1 is invalid
            )

    def test_invalid_sample_indices_out_of_range(self):
        """Test that out-of-range sample_indices raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="invalid index"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                sample_indices=[0, 5, 15],  # 15 is out of range (only 10 variants)
            )

    def test_invalid_sample_indices_negative(self):
        """Test that negative sample_indices raises ValidationError."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (100, 50)).astype(float)
        X_missing = rng.binomial(2, 0.25, (100, 10)).astype(float)

        with pytest.raises(ValidationError, match="invalid index"):
            global_hyperparameter_search(
                Z=Z,
                X_missing=X_missing,
                sample_indices=[0, -1],
            )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_predictors(self):
        """Test handling of empty predictor matrix."""
        rng = np.random.default_rng(42)
        n, p = 100, 10

        Z = np.empty((n, 0))  # No predictors
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        # Should return defaults with inf MSE
        assert result.best_l1_ratio == DEFAULT_L1_RATIOS[0]
        assert result.best_alpha == DEFAULT_ALPHAS[0]
        assert result.best_mean_cv_mse == float("inf")
        assert result.n_variants_failed == p

    def test_single_hyperparameter_value(self):
        """Test grid search with single values for hyperparameters."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            l1_ratios=[0.5],
            alphas=[0.01],
            cv_folds=3,
            random_state=42,
        )

        assert result.best_l1_ratio == 0.5
        assert result.best_alpha == 0.01
        assert len(result.grid_results) == 1

    def test_single_variant(self):
        """Test grid search with single target variant."""
        rng = np.random.default_rng(42)
        n, m = 200, 50

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, 1)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        assert result.n_variants_sampled == 1

    def test_1d_x_missing_input(self):
        """Test that 1D X_missing input is handled correctly."""
        rng = np.random.default_rng(42)
        n, m = 200, 50

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, n).astype(float)  # 1D array

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        assert result.n_variants_sampled == 1


class TestGridResults:
    """Tests for grid_results structure and content."""

    def test_grid_results_structure(self):
        """Test that grid_results has correct structure."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        l1_ratios = [0.1, 0.5]
        alphas = [0.01, 0.1]

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            l1_ratios=l1_ratios,
            alphas=alphas,
            cv_folds=3,
            random_state=42,
        )

        # Should have all combinations
        assert len(result.grid_results) == len(l1_ratios) * len(alphas)

        # Check each result has required keys
        for grid_result in result.grid_results:
            assert "l1_ratio" in grid_result
            assert "alpha" in grid_result
            assert "mean_cv_mse" in grid_result
            assert "std_cv_mse" in grid_result
            assert "n_variants_evaluated" in grid_result

    def test_best_is_minimum_mse(self):
        """Test that best hyperparameters correspond to minimum mean MSE."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        # Find minimum from grid_results
        min_mse = min(r["mean_cv_mse"] for r in result.grid_results)

        assert result.best_mean_cv_mse == min_mse

        # Find the best params from grid_results
        best_from_grid = None
        for r in result.grid_results:
            if r["mean_cv_mse"] == min_mse:
                best_from_grid = r
                break

        assert result.best_l1_ratio == best_from_grid["l1_ratio"]
        assert result.best_alpha == best_from_grid["alpha"]

    def test_n_variants_evaluated_reasonable(self):
        """Test that n_variants_evaluated is within expected range."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=42,
        )

        for grid_result in result.grid_results:
            assert 0 <= grid_result["n_variants_evaluated"] <= p


class TestReproducibility:
    """Tests for reproducibility with random_state."""

    def test_same_random_state_same_results(self):
        """Test that same random_state produces identical results."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result1 = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=123,
        )

        result2 = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=123,
        )

        assert result1.best_l1_ratio == result2.best_l1_ratio
        assert result1.best_alpha == result2.best_alpha
        assert result1.best_mean_cv_mse == result2.best_mean_cv_mse
        assert result1.n_variants_failed == result2.n_variants_failed

        # Grid results should be identical
        for r1, r2 in zip(result1.grid_results, result2.grid_results):
            assert r1["l1_ratio"] == r2["l1_ratio"]
            assert r1["alpha"] == r2["alpha"]
            assert r1["mean_cv_mse"] == r2["mean_cv_mse"]
            assert r1["std_cv_mse"] == r2["std_cv_mse"]

    def test_different_random_state_may_differ(self):
        """Test that different random_state may produce different CV metrics."""
        rng = np.random.default_rng(42)
        n, m, p = 200, 50, 10

        Z = rng.binomial(2, 0.3, (n, m)).astype(float)
        X_missing = rng.binomial(2, 0.25, (n, p)).astype(float)

        result1 = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=123,
        )

        result2 = global_hyperparameter_search(
            Z=Z,
            X_missing=X_missing,
            cv_folds=3,
            random_state=456,
        )

        # Results may or may not be identical depending on the data
        # Just verify both complete successfully
        assert isinstance(result1, GridSearchResult)
        assert isinstance(result2, GridSearchResult)


class TestImportPaths:
    """Tests for import paths."""

    def test_import_from_models(self):
        """Test that global_hyperparameter_search can be imported from models."""
        from imputed_prs.models import global_hyperparameter_search, GridSearchResult

        assert callable(global_hyperparameter_search)
        assert GridSearchResult is not None

    def test_import_grid_search_result_from_core(self):
        """Test that GridSearchResult can be imported from core."""
        from imputed_prs.core import GridSearchResult

        assert GridSearchResult is not None
