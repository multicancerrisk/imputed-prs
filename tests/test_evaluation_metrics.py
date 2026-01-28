"""Tests for PRS evaluation metrics."""

import numpy as np
import pytest

from imputed_prs.evaluation.metrics import (
    compute_prs_metrics,
    compute_percentile_concordance,
)


class TestComputePrsMetrics:
    """Tests for compute_prs_metrics function."""

    def test_basic_metrics(self):
        """Test metrics with correlated synthetic data."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 1000)
        s_imputed = s_true + rng.normal(0, 0.2, 1000)

        metrics = compute_prs_metrics(s_imputed, s_true)

        assert metrics.correlation > 0.9
        assert metrics.r2 > 0.8
        assert metrics.mae < 0.5
        assert metrics.rmse < 0.5
        assert metrics.spearman_rho > 0.9
        assert 0.9 < metrics.calibration_slope < 1.1

    def test_perfect_prediction(self):
        """Test with perfect prediction."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_imputed = s_true.copy()

        metrics = compute_prs_metrics(s_imputed, s_true)

        np.testing.assert_allclose(metrics.correlation, 1.0)
        np.testing.assert_allclose(metrics.r2, 1.0)
        np.testing.assert_allclose(metrics.mae, 0.0)
        np.testing.assert_allclose(metrics.rmse, 0.0)
        np.testing.assert_allclose(metrics.calibration_slope, 1.0)
        np.testing.assert_allclose(metrics.calibration_intercept, 0.0, atol=1e-10)

    def test_nan_filtering(self):
        """Test that NaN values are properly filtered."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_imputed = s_true + rng.normal(0, 0.1, 100)

        s_imputed[0] = np.nan
        s_true[5] = np.nan

        metrics = compute_prs_metrics(s_imputed, s_true)
        # Should complete without error, using 98 samples
        assert metrics.correlation > 0.9

    def test_insufficient_samples(self):
        """Test error when too few valid samples."""
        s_true = np.array([1.0, 2.0, np.nan])
        s_imputed = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="at least 3"):
            compute_prs_metrics(s_imputed, s_true)


class TestComputePercentileConcordance:
    """Tests for compute_percentile_concordance function."""

    def test_basic_concordance(self):
        """Test concordance with correlated data."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 1000)
        s_imputed = s_true + rng.normal(0, 0.2, 1000)

        concordance = compute_percentile_concordance(s_imputed, s_true)

        assert concordance["top_10_concordance"] > 0.5
        assert concordance["bottom_10_concordance"] > 0.5
        assert "quintile_kappa" in concordance
        assert concordance["quintile_kappa"] > 0.5

    def test_perfect_concordance(self):
        """Test with perfect prediction."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_imputed = s_true.copy()

        concordance = compute_percentile_concordance(s_imputed, s_true)

        assert concordance["top_10_concordance"] == 1.0
        assert concordance["bottom_10_concordance"] == 1.0
        assert concordance["quintile_kappa"] == 1.0

    def test_custom_percentiles(self):
        """Test with custom percentile list."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 500)
        s_imputed = s_true + rng.normal(0, 0.3, 500)

        concordance = compute_percentile_concordance(
            s_imputed, s_true, percentiles=[5, 20]
        )

        assert "top_5_concordance" in concordance
        assert "top_20_concordance" in concordance
        assert "top_10_concordance" not in concordance  # Not requested

    def test_insufficient_samples(self):
        """Test error when too few samples for concordance."""
        s_true = np.arange(10, dtype=float)
        s_imputed = s_true + 0.1

        with pytest.raises(ValueError, match="at least 20"):
            compute_percentile_concordance(s_imputed, s_true)

    def test_nan_filtering(self):
        """Test NaN filtering in concordance."""
        rng = np.random.default_rng(42)
        s_true = rng.normal(0, 1, 100)
        s_imputed = s_true + rng.normal(0, 0.1, 100)

        s_imputed[:5] = np.nan

        concordance = compute_percentile_concordance(s_imputed, s_true)
        # Should complete with 95 samples
        assert "quintile_kappa" in concordance
