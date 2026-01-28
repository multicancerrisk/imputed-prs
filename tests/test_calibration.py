"""Tests for the calibration module."""

import numpy as np
import pytest

from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
)


class TestComputeCvPredictedPrs:
    """Tests for compute_cv_predicted_prs function."""

    def test_basic_computation(self):
        """Test basic PRS computation with observed and imputed components."""
        n, p = 100, 10
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = rng.normal(0, 0.05, p)

        # Split: 6 observed, 4 missing
        observed_indices = np.array([0, 1, 2, 3, 4, 5])
        missing_indices = [6, 7, 8, 9]

        observed_betas = betas[observed_indices]
        missing_betas = betas[missing_indices]

        # Create CV predictions (add some noise to true values)
        cv_predictions = {
            idx: X[:, idx] + rng.normal(0, 0.1, n) for idx in missing_indices
        }

        s_cv = compute_cv_predicted_prs(
            X, observed_indices, observed_betas, cv_predictions, missing_betas
        )
        s_true = X @ betas

        assert len(s_cv) == n
        assert np.corrcoef(s_cv, s_true)[0, 1] > 0.8  # Should be correlated

    def test_observed_only(self):
        """Test with no missing variants."""
        n, p = 50, 5
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = rng.normal(0, 0.05, p)

        s_cv = compute_cv_predicted_prs(X, np.arange(p), betas, {}, np.array([]))
        s_true = X @ betas

        np.testing.assert_allclose(s_cv, s_true)

    def test_imputed_only(self):
        """Test with no observed variants."""
        n, p = 50, 5
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = rng.normal(0, 0.05, p)

        cv_predictions = {i: X[:, i] for i in range(p)}

        s_cv = compute_cv_predicted_prs(
            X, np.array([]), np.array([]), cv_predictions, betas
        )
        s_true = X @ betas

        np.testing.assert_allclose(s_cv, s_true)

    def test_nan_propagation(self):
        """Test that NaN in CV predictions propagates to PRS."""
        n, p = 50, 5
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = rng.normal(0, 0.05, p)

        cv_pred = X[:, 4].astype(float).copy()
        cv_pred[0] = np.nan  # First sample excluded from CV

        cv_predictions = {4: cv_pred}

        s_cv = compute_cv_predicted_prs(
            X, np.arange(4), betas[:4], cv_predictions, betas[4:5]
        )

        assert np.isnan(s_cv[0])  # First sample should be NaN
        assert not np.isnan(s_cv[1])  # Other samples should be valid


class TestEdgeCases:
    """Edge case tests for compute_cv_predicted_prs."""

    def test_empty_x_matrix(self):
        """Test with empty X matrix."""
        X = np.empty((0, 5))
        betas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        s_cv = compute_cv_predicted_prs(X, np.arange(5), betas, {}, np.array([]))

        assert len(s_cv) == 0

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1.0, 2.0, 0.0]])
        betas = np.array([0.1, 0.2, 0.3])

        s_cv = compute_cv_predicted_prs(X, np.arange(3), betas, {}, np.array([]))

        expected = 1.0 * 0.1 + 2.0 * 0.2 + 0.0 * 0.3
        np.testing.assert_allclose(s_cv, [expected])

    def test_single_variant_observed(self):
        """Test with single observed variant."""
        n = 10
        X = np.arange(n).reshape(n, 1).astype(float)
        betas = np.array([0.5])

        s_cv = compute_cv_predicted_prs(X, np.array([0]), betas, {}, np.array([]))

        expected = X[:, 0] * 0.5
        np.testing.assert_allclose(s_cv, expected)

    def test_single_variant_imputed(self):
        """Test with single imputed variant."""
        n = 10
        X = np.arange(n).reshape(n, 1).astype(float)
        betas = np.array([0.5])
        cv_predictions = {0: X[:, 0] + 0.1}

        s_cv = compute_cv_predicted_prs(
            X, np.array([]), np.array([]), cv_predictions, betas
        )

        expected = (X[:, 0] + 0.1) * 0.5
        np.testing.assert_allclose(s_cv, expected)

    def test_zero_betas(self):
        """Test with zero effect weights."""
        n, p = 20, 3
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = np.zeros(p)

        s_cv = compute_cv_predicted_prs(X, np.arange(p), betas, {}, np.array([]))

        np.testing.assert_allclose(s_cv, np.zeros(n))

    def test_multiple_nan_samples(self):
        """Test NaN propagation with multiple samples having NaN predictions."""
        n, p = 20, 3
        rng = np.random.default_rng(42)
        X = rng.binomial(2, 0.3, (n, p)).astype(float)
        betas = np.array([0.1, 0.2, 0.3])

        cv_pred = X[:, 2].astype(float).copy()
        cv_pred[[0, 5, 10]] = np.nan

        cv_predictions = {2: cv_pred}

        s_cv = compute_cv_predicted_prs(
            X, np.arange(2), betas[:2], cv_predictions, betas[2:3]
        )

        assert np.isnan(s_cv[0])
        assert np.isnan(s_cv[5])
        assert np.isnan(s_cv[10])
        assert not np.isnan(s_cv[1])
        assert not np.isnan(s_cv[6])


class TestEstimateCvCalibration:
    """Tests for estimate_cv_calibration function."""

    def test_basic_calibration(self):
        """Test calibration with synthetic attenuated data."""
        rng = np.random.default_rng(42)
        n = 500
        s_true = rng.normal(0, 1, n)
        # Attenuated predictions (multiplied by 0.9, plus noise)
        s_cv = 0.9 * s_true + rng.normal(0, 0.2, n)

        params = estimate_cv_calibration(s_cv, s_true)

        # Slope should be approximately 1/0.9 ≈ 1.11
        assert 1.0 < params.scaling_factor < 1.3
        assert params.calibration_r2 > 0.8
        assert 0 < params.attenuation_factor < 1
        assert params.n_calibration == n

    def test_perfect_prediction(self):
        """Test with perfect prediction (s_cv = s_true)."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_cv = s_true.copy()

        params = estimate_cv_calibration(s_cv, s_true)

        np.testing.assert_allclose(params.scaling_factor, 1.0, atol=1e-10)
        np.testing.assert_allclose(params.calibration_r2, 1.0, atol=1e-10)
        np.testing.assert_allclose(params.attenuation_factor, 1.0, atol=1e-10)

    def test_nan_filtering(self):
        """Test that NaN values are properly filtered."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_cv = s_true + rng.normal(0, 0.1, 100)

        # Introduce NaN values
        s_cv[0] = np.nan
        s_true[5] = np.nan
        s_cv[10] = np.nan
        s_true[10] = np.nan

        params = estimate_cv_calibration(s_cv, s_true)

        # Should have filtered out 3 samples (indices 0, 5, 10)
        assert params.n_calibration == 97

    def test_insufficient_samples(self):
        """Test error when too few valid samples."""
        s_true = np.array([1.0, 2.0, np.nan])
        s_cv = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="at least 3"):
            estimate_cv_calibration(s_cv, s_true)

    def test_scaling_factor_se(self):
        """Test that standard error is computed correctly."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 500)
        s_cv = s_true + rng.normal(0, 0.3, 500)

        params = estimate_cv_calibration(s_cv, s_true)

        # SE should be small and positive
        assert params.scaling_factor_se > 0
        assert params.scaling_factor_se < 0.1  # Reasonable for n=500

    def test_zero_variance_true(self):
        """Test handling of constant true PRS."""
        s_true = np.full(50, 1.5)
        s_cv = np.random.default_rng(42).normal(1.5, 0.1, 50)

        params = estimate_cv_calibration(s_cv, s_true)

        assert params.attenuation_factor == 0.0
        assert params.sd_true == 0.0
