"""Tests for utility helper functions."""

import numpy as np
import pytest

from imputed_prs.utils.helpers import (
    clip_dosage,
    hardy_weinberg_variance,
    compute_residual_variance,
    compute_standard_error,
)


class TestClipDosage:
    """Tests for clip_dosage function."""

    def test_value_at_lower_boundary(self):
        """Test value exactly at lower boundary (0)."""
        assert clip_dosage(0.0) == 0.0

    def test_value_at_upper_boundary(self):
        """Test value exactly at upper boundary (2)."""
        assert clip_dosage(2.0) == 2.0

    def test_value_below_lower_boundary(self):
        """Test value below lower boundary gets clipped."""
        assert clip_dosage(-0.5) == 0.0

    def test_value_above_upper_boundary(self):
        """Test value above upper boundary gets clipped."""
        assert clip_dosage(2.5) == 2.0

    def test_value_in_range(self):
        """Test value in valid range is unchanged."""
        assert clip_dosage(1.0) == 1.0
        assert clip_dosage(0.5) == 0.5
        assert clip_dosage(1.75) == 1.75

    def test_custom_bounds(self):
        """Test with custom lower and upper bounds."""
        assert clip_dosage(0.5, lower=0.0, upper=1.0) == 0.5
        assert clip_dosage(1.5, lower=0.0, upper=1.0) == 1.0
        assert clip_dosage(-1.0, lower=-0.5, upper=0.5) == -0.5


class TestHardyWeinbergVariance:
    """Tests for hardy_weinberg_variance function."""

    def test_af_half(self):
        """Test AF=0.5 gives variance=0.5."""
        result = hardy_weinberg_variance(0.5)
        assert result == pytest.approx(0.5)

    def test_af_low(self):
        """Test AF=0.1 gives variance=0.18."""
        result = hardy_weinberg_variance(0.1)
        assert result == pytest.approx(0.18)

    def test_af_zero(self):
        """Test AF=0 gives variance=0."""
        result = hardy_weinberg_variance(0.0)
        assert result == 0.0

    def test_af_one(self):
        """Test AF=1 gives variance=0."""
        result = hardy_weinberg_variance(1.0)
        assert result == 0.0

    def test_af_quarter(self):
        """Test AF=0.25 gives variance=0.375."""
        result = hardy_weinberg_variance(0.25)
        # 2 * 0.25 * 0.75 = 0.375
        assert result == pytest.approx(0.375)

    def test_symmetry(self):
        """Test that AF and 1-AF give the same variance."""
        assert hardy_weinberg_variance(0.3) == pytest.approx(hardy_weinberg_variance(0.7))


class TestComputeResidualVariance:
    """Tests for compute_residual_variance function."""

    def test_r2_zero(self):
        """Test r2=0 gives full Hardy-Weinberg variance."""
        af = 0.3
        result = compute_residual_variance(af, r2=0.0)
        expected = hardy_weinberg_variance(af)
        assert result == pytest.approx(expected)

    def test_r2_one(self):
        """Test r2=1 gives zero residual variance."""
        result = compute_residual_variance(0.3, r2=1.0)
        assert result == 0.0

    def test_partial_r2(self):
        """Test partial R2 gives proportionally reduced variance."""
        af = 0.4
        r2 = 0.7
        result = compute_residual_variance(af, r2)
        hw_var = hardy_weinberg_variance(af)
        expected = hw_var * (1.0 - r2)
        assert result == pytest.approx(expected)

    def test_typical_values(self):
        """Test with typical imputation values."""
        # AF=0.2, R2=0.85
        af = 0.2
        r2 = 0.85
        result = compute_residual_variance(af, r2)
        # HW variance = 2 * 0.2 * 0.8 = 0.32
        # Residual = 0.32 * 0.15 = 0.048
        assert result == pytest.approx(0.048)


class TestComputeStandardError:
    """Tests for compute_standard_error function."""

    def test_single_variant(self):
        """Test SE calculation with a single variant."""
        betas = np.array([0.5])
        residual_variances = np.array([0.1])
        result = compute_standard_error(betas, residual_variances)
        # SE = sqrt(0.5^2 * 0.1) = sqrt(0.025) = 0.158...
        expected = np.sqrt(0.5**2 * 0.1)
        assert result == pytest.approx(expected)

    def test_multiple_variants(self):
        """Test SE calculation with multiple variants."""
        betas = np.array([0.5, 0.3, 0.2])
        residual_variances = np.array([0.1, 0.2, 0.15])
        result = compute_standard_error(betas, residual_variances)
        # SE = sqrt(0.5^2 * 0.1 + 0.3^2 * 0.2 + 0.2^2 * 0.15)
        # = sqrt(0.025 + 0.018 + 0.006) = sqrt(0.049)
        expected = np.sqrt(0.5**2 * 0.1 + 0.3**2 * 0.2 + 0.2**2 * 0.15)
        assert result == pytest.approx(expected)

    def test_empty_arrays(self):
        """Test SE calculation with empty arrays returns 0."""
        result = compute_standard_error(np.array([]), np.array([]))
        assert result == 0.0

    def test_zero_residual_variance(self):
        """Test SE is 0 when all residual variances are 0."""
        betas = np.array([0.5, 0.3])
        residual_variances = np.array([0.0, 0.0])
        result = compute_standard_error(betas, residual_variances)
        assert result == 0.0

    def test_negative_betas(self):
        """Test SE calculation works with negative betas (squared)."""
        betas = np.array([-0.5, 0.3])
        residual_variances = np.array([0.1, 0.2])
        result = compute_standard_error(betas, residual_variances)
        # SE = sqrt((-0.5)^2 * 0.1 + 0.3^2 * 0.2) = sqrt(0.025 + 0.018)
        expected = np.sqrt(0.5**2 * 0.1 + 0.3**2 * 0.2)
        assert result == pytest.approx(expected)

    def test_list_input(self):
        """Test that regular Python lists are also accepted."""
        betas = [0.5, 0.3]
        residual_variances = [0.1, 0.2]
        result = compute_standard_error(betas, residual_variances)
        expected = np.sqrt(0.5**2 * 0.1 + 0.3**2 * 0.2)
        assert result == pytest.approx(expected)

    def test_returns_float(self):
        """Test that result is a Python float, not numpy scalar."""
        betas = np.array([0.5])
        residual_variances = np.array([0.1])
        result = compute_standard_error(betas, residual_variances)
        assert isinstance(result, float)
