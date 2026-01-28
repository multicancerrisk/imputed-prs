"""Tests for the calibration module."""

import numpy as np

from imputed_prs.evaluation.calibration import compute_cv_predicted_prs


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
