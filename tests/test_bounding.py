"""Tests for dosage bounding with variance adjustment."""

import numpy as np
import pytest
from scipy.stats import truncnorm

from imputed_prs.models.bounding import (
    clip_and_adjust_variance,
    clip_and_adjust_variance_array,
    compute_truncation_adjustment_factor,
    truncated_normal_mean,
    truncated_normal_variance,
    truncated_normal_variance_array,
)


class TestTruncatedNormalVariance:
    """Tests for truncated_normal_variance function."""

    def test_well_within_bounds_variance_nearly_unchanged(self):
        """When mu is well within bounds with small sigma, variance should be nearly unchanged."""
        mu, sigma = 1.0, 0.2
        var = truncated_normal_variance(mu, sigma)
        # With mu=1.0 (center of [0,2]) and small sigma, truncation has minimal effect
        # Using sigma=0.2 so bounds are 5 sigma away
        assert np.isclose(var, sigma**2, rtol=0.01)

    def test_near_lower_bound_variance_reduced(self):
        """When mu is near lower bound, variance should be reduced."""
        mu, sigma = 0.1, 0.5
        var = truncated_normal_variance(mu, sigma)
        assert var < sigma**2

    def test_near_upper_bound_variance_reduced(self):
        """When mu is near upper bound, variance should be reduced."""
        mu, sigma = 1.9, 0.5
        var = truncated_normal_variance(mu, sigma)
        assert var < sigma**2

    def test_outside_lower_bound_variance_significantly_reduced(self):
        """When mu is outside lower bound, variance should be significantly reduced."""
        mu, sigma = -0.5, 0.5
        var = truncated_normal_variance(mu, sigma)
        assert var < sigma**2 * 0.5  # More than 50% reduction

    def test_outside_upper_bound_variance_significantly_reduced(self):
        """When mu is outside upper bound, variance should be significantly reduced."""
        mu, sigma = 2.5, 0.5
        var = truncated_normal_variance(mu, sigma)
        assert var < sigma**2 * 0.5  # More than 50% reduction

    def test_zero_sigma_returns_zero(self):
        """Zero sigma should return zero variance."""
        var = truncated_normal_variance(1.0, 0.0)
        assert var == 0.0

    def test_tiny_sigma_returns_zero(self):
        """Very small sigma should return zero variance."""
        var = truncated_normal_variance(1.0, 1e-15)
        assert var == 0.0

    def test_variance_always_non_negative(self):
        """Variance should always be non-negative."""
        test_cases = [
            (0.0, 0.5),
            (2.0, 0.5),
            (-1.0, 0.5),
            (3.0, 0.5),
            (1.0, 0.01),
            (1.0, 2.0),
        ]
        for mu, sigma in test_cases:
            var = truncated_normal_variance(mu, sigma)
            assert var >= 0.0, f"Negative variance for mu={mu}, sigma={sigma}"

    def test_variance_at_most_original(self):
        """Truncated variance should never exceed original variance."""
        test_cases = [
            (0.5, 0.3),
            (1.0, 0.5),
            (1.5, 0.4),
            (0.0, 0.2),
            (2.0, 0.2),
        ]
        for mu, sigma in test_cases:
            var = truncated_normal_variance(mu, sigma)
            assert var <= sigma**2 + 1e-10, f"Variance exceeds original for mu={mu}, sigma={sigma}"

    def test_scipy_validation_center(self):
        """Validate against scipy.stats.truncnorm for centered distribution."""
        mu, sigma = 1.0, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        our_var = truncated_normal_variance(mu, sigma)
        assert np.isclose(our_var, scipy_var, rtol=1e-6)

    def test_scipy_validation_near_lower(self):
        """Validate against scipy.stats.truncnorm for distribution near lower bound."""
        mu, sigma = 0.3, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        our_var = truncated_normal_variance(mu, sigma)
        assert np.isclose(our_var, scipy_var, rtol=1e-6)

    def test_scipy_validation_near_upper(self):
        """Validate against scipy.stats.truncnorm for distribution near upper bound."""
        mu, sigma = 1.7, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        our_var = truncated_normal_variance(mu, sigma)
        assert np.isclose(our_var, scipy_var, rtol=1e-6)

    def test_scipy_validation_outside_lower(self):
        """Validate against scipy.stats.truncnorm for mu outside lower bound."""
        mu, sigma = -0.2, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        our_var = truncated_normal_variance(mu, sigma)
        assert np.isclose(our_var, scipy_var, rtol=1e-6)

    def test_scipy_validation_outside_upper(self):
        """Validate against scipy.stats.truncnorm for mu outside upper bound."""
        mu, sigma = 2.2, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        our_var = truncated_normal_variance(mu, sigma)
        assert np.isclose(our_var, scipy_var, rtol=1e-6)

    def test_custom_bounds(self):
        """Test with custom bounds."""
        mu, sigma = 0.5, 0.2
        var = truncated_normal_variance(mu, sigma, lower=-1.0, upper=1.0)
        # With bounds [-1, 1] and mu=0.5, there's some truncation at upper
        a, b = (-1.0 - mu) / sigma, (1.0 - mu) / sigma
        scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
        assert np.isclose(var, scipy_var, rtol=1e-6)


class TestTruncatedNormalMean:
    """Tests for truncated_normal_mean function."""

    def test_well_within_bounds_mean_nearly_unchanged(self):
        """When mu is well within bounds, mean should be nearly unchanged."""
        mu, sigma = 1.0, 0.3
        mean = truncated_normal_mean(mu, sigma)
        assert np.isclose(mean, mu, rtol=0.01)

    def test_near_lower_bound_mean_shifted_up(self):
        """When mu is near lower bound, mean should be shifted up."""
        mu, sigma = 0.1, 0.5
        mean = truncated_normal_mean(mu, sigma)
        assert mean > mu

    def test_near_upper_bound_mean_shifted_down(self):
        """When mu is near upper bound, mean should be shifted down."""
        mu, sigma = 1.9, 0.5
        mean = truncated_normal_mean(mu, sigma)
        assert mean < mu

    def test_outside_lower_bound_mean_at_lower(self):
        """When mu is far outside lower bound, mean should be near lower."""
        mu, sigma = -2.0, 0.5
        mean = truncated_normal_mean(mu, sigma)
        assert mean > 0.0  # Mean is within bounds
        assert mean < 0.5  # But close to lower bound

    def test_outside_upper_bound_mean_at_upper(self):
        """When mu is far outside upper bound, mean should be near upper."""
        mu, sigma = 4.0, 0.5
        mean = truncated_normal_mean(mu, sigma)
        assert mean < 2.0  # Mean is within bounds
        assert mean > 1.5  # But close to upper bound

    def test_zero_sigma_returns_clipped_mu(self):
        """Zero sigma should return mu clipped to bounds."""
        assert truncated_normal_mean(1.0, 0.0) == 1.0
        assert truncated_normal_mean(-0.5, 0.0) == 0.0
        assert truncated_normal_mean(2.5, 0.0) == 2.0

    def test_mean_always_within_bounds(self):
        """Mean should always be within bounds."""
        test_cases = [
            (0.0, 0.5),
            (2.0, 0.5),
            (-1.0, 0.5),
            (3.0, 0.5),
            (1.0, 0.01),
            (1.0, 2.0),
        ]
        for mu, sigma in test_cases:
            mean = truncated_normal_mean(mu, sigma)
            assert 0.0 <= mean <= 2.0, f"Mean out of bounds for mu={mu}, sigma={sigma}"

    def test_scipy_validation(self):
        """Validate against scipy.stats.truncnorm."""
        mu, sigma = 0.3, 0.5
        a, b = (0 - mu) / sigma, (2 - mu) / sigma
        scipy_mean = truncnorm.mean(a, b, loc=mu, scale=sigma)
        our_mean = truncated_normal_mean(mu, sigma)
        assert np.isclose(our_mean, scipy_mean, rtol=1e-6)


class TestClipAndAdjustVariance:
    """Tests for clip_and_adjust_variance function."""

    def test_well_within_bounds_variance_unchanged(self):
        """Well within bounds with small variance - should be nearly unchanged."""
        # Use small variance (sigma=0.2) so bounds are 5 sigma away
        pred, var = clip_and_adjust_variance(1.0, 0.04)
        assert pred == 1.0
        assert np.isclose(var, 0.04, rtol=0.01)

    def test_near_boundary_variance_adjusted(self):
        """Near boundary - variance should be reduced."""
        pred, var = clip_and_adjust_variance(-0.2, 0.25)
        assert pred == 0.0
        assert var < 0.25  # Truncation reduces variance

    def test_negative_prediction_clipped(self):
        """Negative prediction should be clipped to 0."""
        pred, var = clip_and_adjust_variance(-0.5, 0.1)
        assert pred == 0.0

    def test_above_upper_bound_clipped(self):
        """Prediction above 2 should be clipped to 2."""
        pred, var = clip_and_adjust_variance(2.3, 0.1)
        assert pred == 2.0

    def test_zero_variance_returns_zero(self):
        """Zero variance should return zero adjusted variance."""
        pred, var = clip_and_adjust_variance(1.0, 0.0)
        assert pred == 1.0
        assert var == 0.0

    def test_negative_variance_raises_error(self):
        """Negative variance should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clip_and_adjust_variance(1.0, -0.1)

    def test_tiny_variance_returns_zero(self):
        """Very small variance should return zero."""
        pred, var = clip_and_adjust_variance(1.0, 1e-25)
        assert pred == 1.0
        assert var == 0.0

    def test_custom_bounds(self):
        """Test with custom bounds."""
        pred, var = clip_and_adjust_variance(1.5, 0.25, lower=0.0, upper=1.0)
        assert pred == 1.0  # Clipped to upper
        assert var < 0.25  # Variance reduced due to truncation

    def test_variance_never_exceeds_original(self):
        """Adjusted variance should never exceed original."""
        test_cases = [
            (0.5, 0.1),
            (1.0, 0.25),
            (1.5, 0.3),
            (-0.1, 0.2),
            (2.1, 0.2),
        ]
        for raw_pred, orig_var in test_cases:
            _, adj_var = clip_and_adjust_variance(raw_pred, orig_var)
            assert adj_var <= orig_var + 1e-10


class TestComputeTruncationAdjustmentFactor:
    """Tests for compute_truncation_adjustment_factor function."""

    def test_center_factor_near_one(self):
        """Factor should be near 1 when mu is at center with small sigma."""
        factor = compute_truncation_adjustment_factor(1.0, 0.3)
        assert 0.95 < factor <= 1.0

    def test_near_boundary_factor_less_than_one(self):
        """Factor should be less than 1 near boundaries."""
        factor_lower = compute_truncation_adjustment_factor(0.1, 0.5)
        factor_upper = compute_truncation_adjustment_factor(1.9, 0.5)
        assert factor_lower < 1.0
        assert factor_upper < 1.0

    def test_outside_bounds_factor_small(self):
        """Factor should be small when mu is outside bounds."""
        factor = compute_truncation_adjustment_factor(-0.5, 0.5)
        assert factor < 0.5

    def test_factor_in_zero_one_range(self):
        """Factor should always be in [0, 1]."""
        test_cases = [
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, 0.5),
            (-1.0, 0.5),
            (3.0, 0.5),
            (1.0, 0.01),
            (1.0, 2.0),
        ]
        for mu, sigma in test_cases:
            factor = compute_truncation_adjustment_factor(mu, sigma)
            assert 0.0 <= factor <= 1.0, f"Factor out of range for mu={mu}, sigma={sigma}"

    def test_zero_sigma_returns_zero(self):
        """Zero sigma should return factor of 0."""
        factor = compute_truncation_adjustment_factor(1.0, 0.0)
        assert factor == 0.0

    def test_symmetry(self):
        """Factor should be symmetric around center of bounds."""
        factor_lower = compute_truncation_adjustment_factor(0.3, 0.5)
        factor_upper = compute_truncation_adjustment_factor(1.7, 0.5)
        assert np.isclose(factor_lower, factor_upper, rtol=1e-6)


class TestIntegrationWithPrediction:
    """Integration tests simulating realistic imputation scenarios."""

    def test_typical_imputation_scenario(self):
        """Test typical imputation scenario with various predictions."""
        # Simulate a batch of predictions with residual variance from model
        predictions = [0.5, 1.0, 1.5, -0.1, 2.1, 0.0, 2.0]
        residual_var = 0.15

        for raw_pred in predictions:
            clipped, adj_var = clip_and_adjust_variance(raw_pred, residual_var)

            # Clipped should be in valid dosage range
            assert 0.0 <= clipped <= 2.0

            # Adjusted variance should be non-negative and at most original
            assert 0.0 <= adj_var <= residual_var

            # At center of bounds, variance reduction is less than at edges
            # With residual_var=0.15 (sigma~0.387), truncation still has effect
            if raw_pred == 1.0:
                # At center, expect ~93% of original variance preserved
                assert adj_var > 0.9 * residual_var

    def test_high_uncertainty_near_boundary(self):
        """Test behavior with high uncertainty near boundary."""
        # High variance prediction near lower bound
        raw_pred = 0.2
        residual_var = 0.5  # High uncertainty

        clipped, adj_var = clip_and_adjust_variance(raw_pred, residual_var)

        assert clipped == 0.2
        # With high variance near boundary, truncation has significant effect
        assert adj_var < residual_var

    def test_low_uncertainty_well_within_bounds(self):
        """Test behavior with low uncertainty well within bounds."""
        # Low variance prediction at center
        raw_pred = 1.0
        residual_var = 0.01  # Low uncertainty

        clipped, adj_var = clip_and_adjust_variance(raw_pred, residual_var)

        assert clipped == 1.0
        # With low variance at center, truncation has minimal effect
        assert np.isclose(adj_var, residual_var, rtol=0.001)

    def test_batch_processing(self):
        """Test processing a batch of predictions."""
        np.random.seed(42)
        n_samples = 100

        # Simulate raw predictions with some outside bounds
        raw_predictions = np.random.normal(1.0, 0.6, n_samples)
        residual_var = 0.2

        clipped_preds = []
        adjusted_vars = []

        for raw_pred in raw_predictions:
            clipped, adj_var = clip_and_adjust_variance(raw_pred, residual_var)
            clipped_preds.append(clipped)
            adjusted_vars.append(adj_var)

        clipped_preds = np.array(clipped_preds)
        adjusted_vars = np.array(adjusted_vars)

        # All clipped predictions should be in valid range
        assert np.all(clipped_preds >= 0.0)
        assert np.all(clipped_preds <= 2.0)

        # All adjusted variances should be valid
        assert np.all(adjusted_vars >= 0.0)
        assert np.all(adjusted_vars <= residual_var)

        # Average adjusted variance should be less than original
        # (since some predictions are near boundaries)
        assert np.mean(adjusted_vars) < residual_var


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_extreme_mu_below_bounds(self):
        """Test with mu far below bounds."""
        var = truncated_normal_variance(-10.0, 0.5)
        mean = truncated_normal_mean(-10.0, 0.5)

        # Should return valid (though small) values
        assert var >= 0.0
        assert 0.0 <= mean <= 2.0

    def test_extreme_mu_above_bounds(self):
        """Test with mu far above bounds."""
        var = truncated_normal_variance(10.0, 0.5)
        mean = truncated_normal_mean(10.0, 0.5)

        # Should return valid values
        assert var >= 0.0
        assert 0.0 <= mean <= 2.0

    def test_very_large_sigma(self):
        """Test with very large sigma."""
        var = truncated_normal_variance(1.0, 100.0)
        mean = truncated_normal_mean(1.0, 100.0)

        # Should handle gracefully
        assert var >= 0.0
        assert 0.0 <= mean <= 2.0

    def test_very_small_sigma(self):
        """Test with very small (but non-zero) sigma."""
        var = truncated_normal_variance(1.0, 1e-8)
        mean = truncated_normal_mean(1.0, 1e-8)

        # Should return small variance and mean ≈ mu
        assert var >= 0.0
        assert np.isclose(mean, 1.0, atol=1e-6)

    def test_mu_exactly_at_bounds(self):
        """Test with mu exactly at bounds."""
        # At lower bound
        var_lower = truncated_normal_variance(0.0, 0.5)
        mean_lower = truncated_normal_mean(0.0, 0.5)
        assert var_lower >= 0.0
        assert 0.0 <= mean_lower <= 2.0

        # At upper bound
        var_upper = truncated_normal_variance(2.0, 0.5)
        mean_upper = truncated_normal_mean(2.0, 0.5)
        assert var_upper >= 0.0
        assert 0.0 <= mean_upper <= 2.0


class TestVectorizedBounding:
    """Array twins must be elementwise-identical to the scalar oracle (atol=1e-12).

    The grids deliberately straddle the guard boundaries — ``sigma`` at/below
    ``_MIN_SIGMA`` (=1e-10) and ``mu`` far outside ``[lower, upper]`` that drive
    ``Z = Phi_hi - Phi_lo`` below ``_MIN_Z`` — so the ``np.where`` safe-denominator
    pattern is exercised where both scalar early-returns fire.
    """

    _SIGMAS = [0.0, 1e-15, 1e-10, 1e-3, 0.5, 5.0]
    _MUS = [-1.0, 0.0, 1.0, 2.0, 3.0, 100.0]

    def test_variance_array_equals_scalar(self):
        mu_grid, sig_grid = np.meshgrid(self._MUS, self._SIGMAS, indexing="ij")
        got = truncated_normal_variance_array(mu_grid, sig_grid)
        expected = np.array(
            [[truncated_normal_variance(m, s) for s in self._SIGMAS] for m in self._MUS]
        )
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    def test_variance_array_custom_bounds_equals_scalar(self):
        mus = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        sigs = np.array([0.2, 0.5, 0.5, 0.2, 0.5])
        got = truncated_normal_variance_array(mus, sigs, lower=-1.0, upper=1.0)
        expected = np.array(
            [
                truncated_normal_variance(m, s, lower=-1.0, upper=1.0)
                for m, s in zip(mus, sigs)
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    def test_variance_array_finite_in_guarded_regions(self):
        # Extreme mu (Z->0) and sub-floor sigma must not leak NaN/inf from the
        # unused np.where branch.
        mu_grid, sig_grid = np.meshgrid(
            [-1e6, -5.0, 0.0, 1.0, 2.0, 7.0, 1e6],
            [0.0, 1e-20, 1e-12, 1e-10, 0.3, 50.0],
            indexing="ij",
        )
        out = truncated_normal_variance_array(mu_grid, sig_grid)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_clip_adjust_array_equals_scalar(self):
        raws = [-0.5, 0.0, 0.2, 1.0, 1.9, 2.0, 2.5]
        rvars = [0.0, 1e-25, 1e-20, 1e-3, 0.25, 0.5]
        raw_grid, rv_grid = np.meshgrid(raws, rvars, indexing="ij")
        clip_got, adj_got = clip_and_adjust_variance_array(raw_grid, rv_grid)
        clip_exp = np.empty_like(raw_grid)
        adj_exp = np.empty_like(raw_grid)
        for i, r in enumerate(raws):
            for j, v in enumerate(rvars):
                c, a = clip_and_adjust_variance(r, v)
                clip_exp[i, j] = c
                adj_exp[i, j] = a
        np.testing.assert_allclose(clip_got, clip_exp, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(adj_got, adj_exp, rtol=0.0, atol=1e-12)

    def test_clip_adjust_array_broadcasts_scalar_variance(self):
        raws = np.array([-0.5, 1.0, 2.5])
        clip_got, adj_got = clip_and_adjust_variance_array(raws, 0.25)
        for i, r in enumerate(raws):
            c, a = clip_and_adjust_variance(r, 0.25)
            assert np.isclose(clip_got[i], c, rtol=0.0, atol=1e-12)
            assert np.isclose(adj_got[i], a, rtol=0.0, atol=1e-12)

    def test_clip_adjust_array_negative_variance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            clip_and_adjust_variance_array(
                np.array([1.0, 1.0]), np.array([0.1, -0.1])
            )
