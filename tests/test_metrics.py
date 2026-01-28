"""Tests for the metrics module."""

import numpy as np

from imputed_prs.models.metrics import compute_cv_r2


class TestComputeCvR2:
    """Tests for compute_cv_r2 function."""

    def test_perfect_prediction(self):
        """R² = 1.0 for perfect predictions."""
        true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv_predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert r2 == 1.0

    def test_mean_prediction(self):
        """R² = 0.0 when predicting the mean."""
        true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv_predictions = np.full(5, np.mean(true_values))

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert abs(r2) < 1e-10

    def test_negative_r2(self):
        """R² < 0 when predictions worse than mean."""
        true_values = np.array([1.0, 2.0, 3.0])
        # Predictions in opposite direction
        cv_predictions = np.array([3.0, 2.0, 1.0])

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert r2 < 0

    def test_zero_variance_target(self):
        """R² = 0.0 for constant target (zero variance)."""
        true_values = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        cv_predictions = np.array([2.9, 3.1, 3.0, 3.0, 3.0])

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert r2 == 0.0

    def test_empty_arrays(self):
        """R² = 0.0 for empty arrays."""
        r2 = compute_cv_r2(np.array([]), np.array([]))
        assert r2 == 0.0

    def test_single_sample(self):
        """Handle single sample edge case."""
        true_values = np.array([5.0])
        cv_predictions = np.array([4.0])

        # Single sample has zero variance, so should return 0.0
        r2 = compute_cv_r2(true_values, cv_predictions)
        assert r2 == 0.0

    def test_good_but_imperfect_prediction(self):
        """R² between 0 and 1 for reasonable predictions."""
        true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Predictions with small error
        cv_predictions = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert 0 < r2 < 1

    def test_very_bad_predictions(self):
        """R² can be very negative for terrible predictions."""
        true_values = np.array([1.0, 2.0, 3.0])
        # Predictions that are way off
        cv_predictions = np.array([10.0, 20.0, 30.0])

        r2 = compute_cv_r2(true_values, cv_predictions)
        assert r2 < -1  # Much worse than predicting the mean
