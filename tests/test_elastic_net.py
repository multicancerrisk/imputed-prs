"""Tests for the elastic net imputation model."""

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet

from imputed_prs.models.elastic_net import (
    _fit_intercept_only_model,
    fit_single_variant_model,
)
from imputed_prs.models.metrics import standardize_columns
from imputed_prs.core.types import SingleVariantModelResult


class TestFitSingleVariantModel:
    """Tests for basic fitting functionality."""

    def test_basic_fitting(self):
        """Test fitting with a known linear relationship."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Create predictor with known relationship
        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        noise = rng.normal(0, 0.2, n_samples)
        target = 0.8 * predictor + 0.4 + noise

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        assert isinstance(result, SingleVariantModelResult)
        assert result.n_samples == n_samples
        assert result.n_predictors == 1
        assert not result.is_intercept_only
        # Coefficient should be close to 0.8
        assert abs(result.coefficients[0] - 0.8) < 0.2
        # R² should be reasonably high
        assert result.cv_r2 > 0.3
        assert result.cv_mse > 0

    def test_multiple_predictors(self):
        """Test fitting with multiple predictors."""
        rng = np.random.default_rng(123)
        n_samples = 500
        n_predictors = 5

        # Create predictors
        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)
        # Only first two predictors have effect
        coefficients_true = np.array([0.5, 0.3, 0.0, 0.0, 0.0])
        noise = rng.normal(0, 0.2, n_samples)
        target = predictors @ coefficients_true + 0.5 + noise

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=123,
        )

        assert result.n_predictors == n_predictors
        assert len(result.coefficients) == n_predictors
        assert not result.is_intercept_only
        # First two coefficients should be larger than others
        assert result.coefficients[0] > result.coefficients[3]
        assert result.coefficients[1] > result.coefficients[4]

    def test_cv_predictions_shape(self):
        """Test that CV predictions have correct shape."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=42,
        )

        assert result.cv_predictions.shape == (n_samples,)
        # All predictions should be non-NaN for complete data
        assert not np.any(np.isnan(result.cv_predictions))

    def test_1d_predictor_input(self):
        """Test that 1D predictor array is handled correctly."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Pass predictor as 1D array instead of 2D
        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor,  # 1D array
            random_state=42,
        )

        assert result.n_predictors == 1
        assert len(result.coefficients) == 1


class TestInterceptOnlyModel:
    """Tests for intercept-only model fallback."""

    def test_empty_predictors(self):
        """Test fitting with no predictors returns intercept-only model."""
        rng = np.random.default_rng(42)
        n_samples = 100
        target = rng.binomial(2, 0.4, n_samples).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=np.empty((n_samples, 0)),
            random_state=42,
        )

        assert result.is_intercept_only
        assert result.n_predictors == 0
        assert len(result.coefficients) == 0
        assert result.cv_r2 == 0.0
        # Intercept should be close to mean
        assert abs(result.intercept - np.mean(target)) < 0.01
        # CV predictions should all be the intercept
        assert np.allclose(result.cv_predictions, result.intercept)

    def test_zero_variance_target(self):
        """Test that zero-variance target returns intercept-only model."""
        n_samples = 100
        target = np.ones(n_samples) * 0.6  # Constant target
        predictors = np.random.default_rng(42).binomial(2, 0.3, (n_samples, 3)).astype(
            float
        )

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            random_state=42,
        )

        assert result.is_intercept_only
        assert result.cv_r2 == 0.0
        assert abs(result.intercept - 0.6) < 0.01

    def test_too_few_samples_for_cv(self):
        """Test that too few samples returns intercept-only model."""
        rng = np.random.default_rng(42)
        n_samples = 3  # Less than default cv_folds=5
        target = rng.binomial(2, 0.4, n_samples).astype(float)
        predictors = rng.binomial(2, 0.3, (n_samples, 2)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            cv_folds=5,
            random_state=42,
        )

        assert result.is_intercept_only

    def test_all_coefficients_shrunk_to_zero(self):
        """Test that model detects when all coefficients are shrunk to zero."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Create predictors with no relationship to target
        predictors = rng.binomial(2, 0.3, (n_samples, 3)).astype(float)
        target = rng.binomial(2, 0.4, n_samples).astype(float)

        # Use high alpha to force coefficients to zero
        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            alpha=100.0,  # Very high regularization
            random_state=42,
        )

        # Should be marked as intercept-only
        assert result.is_intercept_only


class TestNaNHandling:
    """Tests for handling NaN values."""

    def test_nan_in_target(self):
        """Test that NaN in target excludes those samples."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        # Set some targets to NaN
        target[10:20] = np.nan

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=42,
        )

        # CV predictions for NaN targets should be NaN
        assert np.all(np.isnan(result.cv_predictions[10:20]))
        # Other predictions should be valid
        valid_mask = ~np.isnan(target)
        assert not np.any(np.isnan(result.cv_predictions[valid_mask]))
        assert result.n_samples == n_samples

    def test_nan_in_predictors(self):
        """Test that NaN in predictors excludes those samples."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_predictors = 3

        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)
        target = predictors[:, 0] * 0.5 + rng.normal(0, 0.1, n_samples)

        # Set some predictor values to NaN
        predictors[5:10, 1] = np.nan

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            random_state=42,
        )

        # CV predictions for samples with NaN predictors should be NaN
        assert np.all(np.isnan(result.cv_predictions[5:10]))

    def test_nan_in_both_target_and_predictors(self):
        """Test handling NaN in both target and predictors."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        # Set different samples to NaN
        target[10:15] = np.nan
        predictor[20:25] = np.nan

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=42,
        )

        # Both sets of NaN samples should have NaN predictions
        assert np.all(np.isnan(result.cv_predictions[10:15]))
        assert np.all(np.isnan(result.cv_predictions[20:25]))

    def test_all_targets_nan(self):
        """Test that all NaN targets returns intercept-only model."""
        n_samples = 100
        target = np.full(n_samples, np.nan)
        predictor = np.random.default_rng(42).binomial(2, 0.3, n_samples).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=42,
        )

        assert result.is_intercept_only
        assert result.intercept == 0.0
        assert np.all(np.isnan(result.cv_predictions))


class TestInputValidation:
    """Tests for input validation."""

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        target = np.array([1.0, 2.0, 3.0])
        predictors = np.array([[1.0], [2.0]])  # Different number of samples

        with pytest.raises(ValueError, match="Shape mismatch"):
            fit_single_variant_model(
                target_dosages=target,
                predictor_dosages=predictors,
            )

    def test_converts_to_float64(self):
        """Test that inputs are converted to float64."""
        target = np.array([0, 1, 2, 1, 0], dtype=np.int32)
        predictors = np.array([[0], [1], [2], [1], [0]], dtype=np.int32)

        # Should not raise - arrays should be converted
        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            random_state=42,
        )

        assert result.cv_predictions.dtype == np.float64


class TestFitInterceptOnlyModel:
    """Tests for _fit_intercept_only_model helper function."""

    def test_basic_intercept_only(self):
        """Test basic intercept-only model."""
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

        result = _fit_intercept_only_model(
            target_dosages=target,
            n_predictors=0,
            l1_ratio=0.5,
            alpha=0.01,
        )

        assert result.is_intercept_only
        assert result.intercept == np.mean(target)
        assert np.allclose(result.cv_predictions, result.intercept)
        assert result.cv_r2 == 0.0
        assert len(result.coefficients) == 0

    def test_intercept_only_with_nan(self):
        """Test intercept-only model with NaN values."""
        target = np.array([0.0, 1.0, np.nan, 1.0, 0.0])

        result = _fit_intercept_only_model(
            target_dosages=target,
            n_predictors=0,
            l1_ratio=0.5,
            alpha=0.01,
        )

        valid_mean = np.mean([0.0, 1.0, 1.0, 0.0])
        assert result.intercept == valid_mean
        # NaN target should have NaN prediction
        assert np.isnan(result.cv_predictions[2])
        # Valid targets should have intercept prediction
        assert result.cv_predictions[0] == valid_mean


class TestReproducibility:
    """Tests for reproducibility with random_state."""

    def test_same_random_state_same_results(self):
        """Test that same random_state produces identical results."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result1 = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=123,
        )

        result2 = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=123,
        )

        assert np.allclose(result1.coefficients, result2.coefficients)
        assert result1.intercept == result2.intercept
        assert np.allclose(result1.cv_predictions, result2.cv_predictions)
        assert result1.cv_r2 == result2.cv_r2
        assert result1.cv_mse == result2.cv_mse

    def test_different_random_state_different_results(self):
        """Test that different random_state produces different CV splits."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result1 = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=123,
        )

        result2 = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=456,
        )

        # Final coefficients should be very similar (same data)
        # but CV predictions may differ due to different fold assignments
        assert np.allclose(result1.coefficients, result2.coefficients, atol=1e-5)
        # CV metrics should be similar but not necessarily identical
        assert abs(result1.cv_r2 - result2.cv_r2) < 0.1


class TestParameterStorage:
    """Tests for parameter storage in results."""

    def test_parameters_stored_correctly(self):
        """Test that input parameters are stored in result."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            l1_ratio=0.7,
            alpha=0.05,
            random_state=42,
        )

        assert result.l1_ratio == 0.7
        assert result.alpha == 0.05
        assert result.n_samples == n_samples
        assert result.n_predictors == 1


class TestToDict:
    """Tests for SingleVariantModelResult.to_dict method."""

    def test_to_dict_serialization(self):
        """Test that to_dict correctly serializes numpy arrays."""
        rng = np.random.default_rng(42)
        n_samples = 50

        predictor = rng.binomial(2, 0.3, n_samples).astype(float)
        target = predictor * 0.5 + rng.normal(0, 0.1, n_samples)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictor.reshape(-1, 1),
            random_state=42,
        )

        d = result.to_dict()

        assert isinstance(d["coefficients"], list)
        assert isinstance(d["cv_predictions"], list)
        assert isinstance(d["intercept"], float)
        assert isinstance(d["cv_mse"], float)
        assert isinstance(d["cv_r2"], float)
        assert isinstance(d["is_intercept_only"], bool)


class TestStandardization:
    """Predictors are standardized before fitting; coefficients are back-transformed
    to the raw-dosage scale so storage and inference are unchanged."""

    def test_fold_scaler_fit_on_training_only(self):
        """standardize_columns derives stats from the rows it is given; applying those
        stats to differently-distributed rows does not re-center them (no leakage)."""
        rng = np.random.default_rng(0)
        X_train = rng.binomial(2, 0.3, (100, 4)).astype(float)
        X_val = rng.binomial(2, 0.5, (40, 4)).astype(float)  # different distribution

        X_train_std, mean, scale = standardize_columns(X_train)

        # Scaler fit on the training rows only.
        np.testing.assert_allclose(mean, X_train.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(scale, X_train.std(axis=0), atol=1e-12)
        np.testing.assert_allclose(X_train_std.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_train_std.std(axis=0), 1.0, atol=1e-10)

        # Applying train stats to a differently-distributed val set does NOT
        # zero-center it (the val mean leaks nothing into the train scaler).
        X_val_std = (X_val - mean) / scale
        assert np.any(np.abs(X_val_std.mean(axis=0)) > 0.05)

        # Integration: a full fit still produces finite OOF predictions.
        target = X_train[:, 0] * 0.7 + rng.normal(0, 0.1, 100)
        result = fit_single_variant_model(target, X_train, random_state=42)
        assert not np.any(np.isnan(result.cv_predictions))

    def test_zero_variance_predictor_column_handled(self):
        """A constant (zero-variance) predictor column yields no NaN coefficients."""
        rng = np.random.default_rng(1)
        n_samples = 200
        signal = rng.binomial(2, 0.3, n_samples).astype(float)
        constant = np.full(n_samples, 1.0)
        predictors = np.column_stack([signal, constant])
        target = 0.8 * signal + 0.3 + rng.normal(0, 0.1, n_samples)

        result = fit_single_variant_model(target, predictors, random_state=42)

        assert np.isfinite(result.coefficients).all()
        assert np.isfinite(result.intercept)
        assert not np.any(np.isnan(result.cv_predictions))
        # Constant column carries no signal -> coefficient is exactly ~0.
        assert abs(result.coefficients[1]) < 1e-6

    def test_backtransform_predictions_match_standardized(self):
        """The stored raw model predicts identically to the standardized final model."""
        rng = np.random.default_rng(2)
        n_samples = 300
        predictors = rng.binomial(2, 0.3, (n_samples, 5)).astype(float)
        target = (
            predictors @ np.array([0.5, 0.3, 0.0, 0.2, 0.0])
            + 0.4
            + rng.normal(0, 0.2, n_samples)
        )

        result = fit_single_variant_model(
            target, predictors, alpha=0.01, l1_ratio=0.5, random_state=42
        )

        # Reconstruct the standardized final model the function fit internally.
        X_std, mean, scale = standardize_columns(predictors)
        m = ElasticNet(
            alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=10000, random_state=42
        )
        m.fit(X_std, target)

        raw_pred = predictors @ result.coefficients + result.intercept
        std_pred = m.predict((predictors - mean) / scale)
        np.testing.assert_allclose(raw_pred, std_pred, rtol=1e-9, atol=1e-9)

    def test_cv_predictions_invariant_under_positive_rescaling(self):
        """CV predictions are invariant when a predictor column is positively rescaled.

        This FAILS on the pre-fix raw-dosage fit (the L1/L2 penalty depended on each
        column's scale) and PASSES once predictors are standardized before fitting.
        """
        rng = np.random.default_rng(3)
        n_samples = 300
        predictors = rng.binomial(2, 0.3, (n_samples, 5)).astype(float)
        target = (
            predictors @ np.array([0.5, -0.3, 0.2, 0.0, 0.1])
            + 0.4
            + rng.normal(0, 0.2, n_samples)
        )

        result_a = fit_single_variant_model(target, predictors, random_state=7)

        rescaled = predictors.copy()
        rescaled[:, 0] *= 4.0  # power of two -> clean float cancellation
        result_b = fit_single_variant_model(target, rescaled, random_state=7)

        np.testing.assert_allclose(
            result_a.cv_predictions,
            result_b.cv_predictions,
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True,
        )
        assert result_a.cv_r2 == pytest.approx(result_b.cv_r2, rel=1e-6)
        assert result_a.cv_mse == pytest.approx(result_b.cv_mse, rel=1e-6)
        # The raw-scale coefficient for the rescaled column scales by 1/4.
        assert result_b.coefficients[0] == pytest.approx(
            result_a.coefficients[0] / 4.0, rel=1e-6
        )

    def test_is_intercept_only_consistent_after_backtransform(self):
        """High regularization zeros every coefficient; the flag and the stored raw
        coefficients agree (0 / scale stays 0)."""
        rng = np.random.default_rng(4)
        predictors = rng.binomial(2, 0.3, (200, 5)).astype(float)
        target = rng.normal(0, 0.01, 200)  # negligible signal

        result = fit_single_variant_model(
            target, predictors, alpha=100.0, random_state=42
        )

        assert result.is_intercept_only
        assert np.allclose(result.coefficients, 0, atol=1e-12)
