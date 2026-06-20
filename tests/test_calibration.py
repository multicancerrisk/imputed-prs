"""Tests for the calibration module."""

import numpy as np
import pytest

from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
    mean_impute_columns,
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


class TestMeanImputeColumns:
    """Tests for the mean_impute_columns helper (per-column NaN imputation, P3.4)."""

    def test_fills_with_column_mean(self):
        """NaN cells are filled with each column's non-missing mean."""
        nan = np.nan
        X = np.array([[0.0, 2.0], [2.0, nan], [nan, 4.0], [4.0, 6.0]])
        # col0 observed = {0, 2, 4} -> mean 2.0; col1 observed = {2, 4, 6} -> mean 4.0
        expected = np.array([[0.0, 2.0], [2.0, 4.0], [2.0, 4.0], [4.0, 6.0]])
        result = mean_impute_columns(X)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        # Pin the individual filled cells for clarity.
        assert result[2, 0] == 2.0
        assert result[1, 1] == 4.0

    def test_differs_from_nan_to_num(self):
        """Mean imputation differs from nan_to_num's zero-fill at the missing cells."""
        nan = np.nan
        X = np.array([[0.0, 2.0], [2.0, nan], [nan, 4.0], [4.0, 6.0]])
        result = mean_impute_columns(X)
        zero_filled = np.nan_to_num(X)
        assert not np.allclose(result, zero_filled)
        # The knocked-out cell becomes the column mean, not 0.0 (= homozygous non-effect).
        assert result[2, 0] == 2.0
        assert zero_filled[2, 0] == 0.0

    def test_all_nan_column_filled_with_zero(self):
        """A column that is entirely NaN falls back to 0.0 (degenerate, matmul-safe)."""
        nan = np.nan
        X = np.array([[1.0, nan], [3.0, nan]])
        expected = np.array([[1.0, 0.0], [3.0, 0.0]])
        np.testing.assert_allclose(mean_impute_columns(X), expected, rtol=0, atol=1e-12)

    def test_no_nan_returned_unchanged_and_equals_nan_to_num(self):
        """On NaN-free input the helper is a no-op and matches nan_to_num.

        This invariant underwrites the no-regression claim: every fit-time fixture is
        fully observed, so mean_impute_columns and np.nan_to_num return identical arrays.
        """
        rng = np.random.default_rng(0)
        X = rng.binomial(2, 0.3, size=(20, 5)).astype(np.float32)
        result = mean_impute_columns(X)
        assert np.array_equal(result, X)
        assert np.array_equal(result, np.nan_to_num(X))

    def test_dtype_preserved(self):
        """The output preserves the input dtype."""
        nan = np.nan
        X32 = np.array([[0.0, nan], [2.0, 4.0]], dtype=np.float32)
        assert mean_impute_columns(X32).dtype == np.float32
        X64 = np.array([[0.0, nan], [2.0, 4.0]], dtype=np.float64)
        assert mean_impute_columns(X64).dtype == np.float64

    def test_input_not_mutated(self):
        """The helper does not mutate its argument."""
        nan = np.nan
        X = np.array([[0.0, 2.0], [2.0, nan], [nan, 4.0]])
        X_before = X.copy()
        mean_impute_columns(X)
        assert np.array_equal(np.isnan(X), np.isnan(X_before))
        observed = ~np.isnan(X)
        np.testing.assert_array_equal(X[observed], X_before[observed])


def _missingness_fixture():
    """Complete dosage matrix + a non-uniform missingness pattern + a plausible s_cv.

    Missingness is scattered (per-row dosage loss varies) so the buggy zero-fill moves
    the calibration slope/r2, not just the intercept (OLS is invariant to a constant
    shift of s_true). Seed is frozen so the asserted margins are reproducible.
    """
    rng = np.random.default_rng(0)
    n, p = 400, 12
    af = rng.uniform(0.1, 0.5, p)
    X_complete = rng.binomial(2, af, size=(n, p)).astype(np.float64)
    betas = rng.normal(0, 0.1, p)
    mask = rng.random((n, p)) < 0.15
    X_missing = X_complete.copy()
    X_missing[mask] = np.nan
    s_cv = 0.85 * (X_complete @ betas) + rng.normal(0, 0.05, n)
    return X_complete, X_missing, mask, betas, s_cv


class TestCalibrationMeanImputation:
    """Acceptance tests for the NaN->mean calibration fix (P3.4).

    Calibration previously built its score matrix with np.nan_to_num, filling a missing
    reference dosage with 0 (homozygous non-effect) and biasing s_true / the observed
    part of s_cv toward zero. These tests show the per-column mean fix (a) preserves the
    column means the bug destroys, (b) reconstructs the complete-case reference better
    than zero-fill, and (c) produces materially different calibration parameters.
    """

    def test_fix_preserves_column_means_unlike_nan_to_num(self):
        """Mean-fill preserves each column's mean exactly; zero-fill shrinks it toward 0."""
        _, X_missing, _, _, _ = _missingness_fixture()
        observed_means = np.nanmean(X_missing, axis=0)
        X_fix = mean_impute_columns(X_missing)
        X_zero = np.nan_to_num(X_missing)
        # The fix preserves the observed column means exactly.
        np.testing.assert_allclose(
            X_fix.mean(axis=0), observed_means, rtol=0, atol=1e-10
        )
        # Zero-fill shrinks every (positive) column mean toward 0 by ~(1 - missing_frac).
        assert np.all(observed_means > 0)
        assert np.all(X_zero.mean(axis=0) / observed_means < 0.95)

    def test_fix_reconstructs_complete_case_better_than_nan_to_num(self):
        """Mean-fill is closer to the complete-case reference than zero-fill (matrix & score).

        Per column the mean is the best constant fill and 2*AF > 0, so mean-fill has
        strictly smaller reconstruction error than zero-fill against the complete panel.
        """
        X_complete, X_missing, _, betas, _ = _missingness_fixture()
        X_fix = mean_impute_columns(X_missing)
        X_zero = np.nan_to_num(X_missing)
        # Matrix-level reconstruction error.
        assert np.sum((X_fix - X_complete) ** 2) < np.sum((X_zero - X_complete) ** 2)
        # Score-level (s_true) reconstruction error.
        s_true_complete = X_complete @ betas
        err_fix = np.sum((X_fix @ betas - s_true_complete) ** 2)
        err_zero = np.sum((X_zero @ betas - s_true_complete) ** 2)
        assert err_fix < err_zero

    def test_fix_calibration_differs_from_nan_to_num(self):
        """The calibration parameters change materially relative to the buggy path."""
        _, X_missing, _, betas, s_cv = _missingness_fixture()
        calib_fixed = estimate_cv_calibration(s_cv, mean_impute_columns(X_missing) @ betas)
        calib_buggy = estimate_cv_calibration(s_cv, np.nan_to_num(X_missing) @ betas)
        # Robust margins (empirical gaps at this seed are ~0.095 / ~0.07 / ~0.01).
        assert abs(calib_fixed.calibration_r2 - calib_buggy.calibration_r2) > 0.03
        assert (
            abs(calib_fixed.attenuation_factor - calib_buggy.attenuation_factor) > 0.02
        )
        assert not np.isclose(
            calib_fixed.scaling_factor, calib_buggy.scaling_factor, atol=1e-3
        )

    def test_mean_filled_reference_is_idempotent_and_matches_calibration(self):
        """Calibration via mean-fill matches the mean-filled complete-case reference exactly.

        The complete-case reference under mean imputation is the population-mean-filled
        panel. Re-masking it and re-imputing reproduces it bit-for-bit, so its calibration
        is reproduced exactly, while zero-fill on the same matrix still diverges.
        """
        _, X_missing, mask, betas, s_cv = _missingness_fixture()
        # Mean-filled complete-case reference.
        X_ref = mean_impute_columns(X_missing)
        X_ref_missing = X_ref.copy()
        X_ref_missing[mask] = np.nan
        # Imputing an already-mean-filled panel is idempotent.
        np.testing.assert_allclose(
            mean_impute_columns(X_ref_missing), X_ref, rtol=0, atol=1e-12
        )
        calib_ref = estimate_cv_calibration(s_cv, X_ref @ betas)
        calib_fix = estimate_cv_calibration(
            s_cv, mean_impute_columns(X_ref_missing) @ betas
        )
        for field in (
            "scaling_factor",
            "calibration_intercept",
            "calibration_r2",
            "attenuation_factor",
            "sd_true",
        ):
            np.testing.assert_allclose(
                getattr(calib_fix, field), getattr(calib_ref, field), rtol=0, atol=1e-9
            )
        # Zero-fill on the same masked matrix still differs from the reference.
        calib_buggy = estimate_cv_calibration(
            s_cv, np.nan_to_num(X_ref_missing) @ betas
        )
        assert not np.isclose(
            calib_fix.scaling_factor, calib_buggy.scaling_factor, atol=1e-3
        )
