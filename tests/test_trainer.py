"""Tests for the imputation model trainer."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.types import ImputedVariantModel, TrainingResult
from imputed_prs.models.trainer import (
    ImputationModelTrainer,
    compute_residual_variance,
    _compute_training_summary,
    _convert_to_imputed_model,
)
from imputed_prs.core.types import SingleVariantModelResult


def create_test_data(
    n_samples: int = 100,
    n_platform_variants: int = 50,
    n_missing_variants: int = 10,
    random_state: int = 42,
):
    """Create synthetic test data for trainer tests.

    Returns:
        Tuple of (Z, X, prs_variants, platform_variant_info).
    """
    rng = np.random.default_rng(random_state)

    # Create platform variants (predictors)
    Z = rng.binomial(2, 0.3, (n_samples, n_platform_variants)).astype(float)

    # Create platform variant info
    platform_variant_info = pd.DataFrame({
        "variant_id": [f"rs{1000 + i}" for i in range(n_platform_variants)],
        "chromosome": ["1"] * n_platform_variants,
        "position": [10000 + i * 1000 for i in range(n_platform_variants)],
    })

    # Create missing variants (targets) with some relationship to predictors
    X = np.zeros((n_samples, n_missing_variants))
    prs_variants_data = []

    for i in range(n_missing_variants):
        # Each missing variant is related to nearby platform variants
        nearby_start = max(0, i * 5 - 2)
        nearby_end = min(n_platform_variants, i * 5 + 3)
        if nearby_end > nearby_start:
            coeffs = rng.uniform(0.2, 0.5, nearby_end - nearby_start)
            X[:, i] = np.sum(Z[:, nearby_start:nearby_end] * coeffs, axis=1)
            X[:, i] += rng.normal(0, 0.5, n_samples)
            # Clip to valid dosage range
            X[:, i] = np.clip(X[:, i], 0, 2)
        else:
            X[:, i] = rng.binomial(2, 0.3, n_samples).astype(float)

        prs_variants_data.append({
            "variant_id": f"rs{2000 + i}",
            "chromosome": "1",
            "position": 15000 + i * 5000,
            "effect_allele": "A",
            "other_allele": "G",
            "beta": rng.uniform(0.1, 0.5),
        })

    prs_variants = pd.DataFrame(prs_variants_data)

    return Z, X, prs_variants, platform_variant_info


class TestImputationModelTrainer:
    """Tests for basic training functionality."""

    def test_basic_training(self):
        """Test basic training with synthetic data."""
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=200,
            n_platform_variants=50,
            n_missing_variants=5,
        )

        trainer = ImputationModelTrainer(
            window_size=100_000,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            n_jobs=1,
            random_state=42,
        )

        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        assert isinstance(result, TrainingResult)
        assert len(result.models) > 0
        assert result.n_variants_trained == len(result.models)
        assert result.n_variants_failed == 0
        assert result.n_variants_trained + result.n_variants_failed == len(prs_variants)

    def test_model_attributes(self):
        """Test that trained models have correct attributes."""
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=200,
            n_platform_variants=50,
            n_missing_variants=3,
        )

        trainer = ImputationModelTrainer(
            window_size=100_000,
            random_state=42,
        )

        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        for var_id, model in result.models.items():
            assert isinstance(model, ImputedVariantModel)
            # imputation_r2 can be negative
            assert model.imputation_r2 >= -1.0
            # allele_frequency should be in [0, 1]
            assert 0 <= model.allele_frequency <= 1
            # residual_variance should be non-negative
            assert model.residual_variance >= 0
            # beta should match input
            prs_row = prs_variants[prs_variants["variant_id"] == var_id].iloc[0]
            assert model.beta == prs_row["beta"]

    def test_cv_predictions_shape(self):
        """Test that CV predictions have correct shape."""
        n_samples = 150
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=n_samples,
            n_platform_variants=40,
            n_missing_variants=4,
        )

        trainer = ImputationModelTrainer(random_state=42)
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        for var_id, cv_preds in result.cv_predictions.items():
            assert cv_preds.shape == (n_samples,)

    def test_training_summary_statistics(self):
        """Test that training summary contains expected keys."""
        Z, X, prs_variants, platform_variant_info = create_test_data()

        trainer = ImputationModelTrainer(random_state=42)
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        expected_keys = [
            "mean_r2", "median_r2", "std_r2", "min_r2", "max_r2",
            "n_high_quality", "n_medium_quality", "n_low_quality",
            "mean_n_predictors",
        ]

        for key in expected_keys:
            assert key in result.training_summary

        # Check quality counts sum to total
        total_quality = (
            result.training_summary["n_high_quality"] +
            result.training_summary["n_medium_quality"] +
            result.training_summary["n_low_quality"]
        )
        assert total_quality == result.n_variants_trained


class TestParallelProcessing:
    """Tests for parallel processing."""

    def test_sequential_vs_parallel_same_results(self):
        """Test that n_jobs=1 and n_jobs>1 produce same results."""
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=100,
            n_platform_variants=30,
            n_missing_variants=5,
        )

        # Sequential
        trainer_seq = ImputationModelTrainer(
            window_size=50_000,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            n_jobs=1,
            random_state=42,
        )
        result_seq = trainer_seq.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Parallel
        trainer_par = ImputationModelTrainer(
            window_size=50_000,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            n_jobs=2,
            random_state=42,
        )
        result_par = trainer_par.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Same number of models
        assert result_seq.n_variants_trained == result_par.n_variants_trained
        assert set(result_seq.models.keys()) == set(result_par.models.keys())

        # Models should be identical
        for var_id in result_seq.models:
            model_seq = result_seq.models[var_id]
            model_par = result_par.models[var_id]

            assert model_seq.imputation_r2 == model_par.imputation_r2
            assert model_seq.intercept == model_par.intercept
            assert np.allclose(model_seq.coefficients, model_par.coefficients)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_predictors_in_window(self):
        """Test handling when no predictors are in the window."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Platform variants far from missing variant
        Z = rng.binomial(2, 0.3, (n_samples, 10)).astype(float)
        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            "position": [1_000_000 + i * 1000 for i in range(10)],  # Far away
        })

        # Missing variant at position 100
        X = rng.binomial(2, 0.3, (n_samples, 1)).astype(float)
        prs_variants = pd.DataFrame({
            "variant_id": ["rs9999"],
            "chromosome": ["1"],
            "position": [100],  # Far from platform variants
            "effect_allele": ["A"],
            "other_allele": ["G"],
            "beta": [0.5],
        })

        trainer = ImputationModelTrainer(
            window_size=1000,  # Small window
            random_state=42,
        )
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Should result in intercept-only model
        assert result.n_intercept_only == 1
        assert result.models["rs9999"].is_intercept_only

    def test_empty_prs_variants(self):
        """Test handling empty prs_variants DataFrame."""
        Z = np.zeros((100, 10))
        X = np.zeros((100, 0))
        prs_variants = pd.DataFrame(columns=[
            "variant_id", "chromosome", "position", "effect_allele", "beta"
        ])
        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            "position": list(range(10)),
        })

        trainer = ImputationModelTrainer()
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        assert result.n_variants_trained == 0
        assert result.n_variants_failed == 0
        assert len(result.models) == 0

    def test_all_nan_target(self):
        """Test handling when all target values are NaN."""
        rng = np.random.default_rng(42)
        n_samples = 100

        Z = rng.binomial(2, 0.3, (n_samples, 20)).astype(float)
        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(20)],
            "chromosome": ["1"] * 20,
            "position": [1000 + i * 100 for i in range(20)],
        })

        # Target with all NaN
        X = np.full((n_samples, 1), np.nan)
        prs_variants = pd.DataFrame({
            "variant_id": ["rs9999"],
            "chromosome": ["1"],
            "position": [1500],
            "effect_allele": ["A"],
            "other_allele": ["G"],
            "beta": [0.5],
        })

        trainer = ImputationModelTrainer(random_state=42)
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Should result in intercept-only model
        assert result.n_intercept_only == 1
        model = result.models["rs9999"]
        assert model.is_intercept_only
        assert model.intercept == 0.0
        assert model.allele_frequency == 0.0


class TestInputValidation:
    """Tests for input validation."""

    def test_missing_prs_columns(self):
        """Test that missing required columns raises ValidationError."""
        Z = np.zeros((100, 10))
        X = np.zeros((100, 1))

        # Missing 'beta' column
        prs_variants = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            # Missing 'beta'
        })

        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            "position": list(range(10)),
        })

        trainer = ImputationModelTrainer()
        with pytest.raises(ValidationError, match="Missing required columns in prs_variants"):
            trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

    def test_missing_platform_columns(self):
        """Test that missing platform columns raises ValidationError."""
        Z = np.zeros((100, 10))
        X = np.zeros((100, 1))

        prs_variants = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            "beta": [0.5],
        })

        # Missing 'position' column
        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            # Missing 'position'
        })

        trainer = ImputationModelTrainer()
        with pytest.raises(ValidationError, match="Missing required columns in platform_variant_info"):
            trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

    def test_sample_count_mismatch(self):
        """Test that sample count mismatch raises ValidationError."""
        Z = np.zeros((100, 10))
        X = np.zeros((50, 1))  # Different sample count

        prs_variants = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            "beta": [0.5],
        })

        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            "position": list(range(10)),
        })

        trainer = ImputationModelTrainer()
        with pytest.raises(ValidationError, match="Sample count mismatch"):
            trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

    def test_variant_count_mismatch(self):
        """Test that variant count mismatch raises ValidationError."""
        Z = np.zeros((100, 10))
        X = np.zeros((100, 3))  # 3 columns

        # But only 1 variant in DataFrame
        prs_variants = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            "beta": [0.5],
        })

        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(10)],
            "chromosome": ["1"] * 10,
            "position": list(range(10)),
        })

        trainer = ImputationModelTrainer()
        with pytest.raises(ValidationError, match="Variant count mismatch"):
            trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

    def test_platform_variant_count_mismatch(self):
        """Test that platform variant count mismatch raises ValidationError."""
        Z = np.zeros((100, 10))  # 10 columns
        X = np.zeros((100, 1))

        prs_variants = pd.DataFrame({
            "variant_id": ["rs1"],
            "chromosome": ["1"],
            "position": [100],
            "effect_allele": ["A"],
            "beta": [0.5],
        })

        # But only 5 variants in DataFrame
        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(5)],
            "chromosome": ["1"] * 5,
            "position": list(range(5)),
        })

        trainer = ImputationModelTrainer()
        with pytest.raises(ValidationError, match="Platform variant count mismatch"):
            trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)


class TestComputations:
    """Tests for computation helper functions."""

    def test_compute_residual_variance(self):
        """Test residual variance computation."""
        # At AF=0.5 and r2=0, variance is maximal: 2*0.5*0.5*1 = 0.5
        var = compute_residual_variance(0.5, 0.0)
        assert np.isclose(var, 0.5)

        # At r2=1, variance is 0
        var = compute_residual_variance(0.5, 1.0)
        assert np.isclose(var, 0.0)

        # Negative r2 should be clipped to 0 for variance
        var = compute_residual_variance(0.5, -0.5)
        assert np.isclose(var, 0.5)

        # At AF=0 or AF=1, variance is 0
        var = compute_residual_variance(0.0, 0.5)
        assert np.isclose(var, 0.0)
        var = compute_residual_variance(1.0, 0.5)
        assert np.isclose(var, 0.0)

    def test_allele_frequency_computation(self):
        """Test that allele frequency is computed correctly."""
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Create data with known allele frequency
        true_af = 0.3
        Z = rng.binomial(2, 0.3, (n_samples, 20)).astype(float)
        X = rng.binomial(2, true_af, (n_samples, 1)).astype(float)

        platform_variant_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(20)],
            "chromosome": ["1"] * 20,
            "position": [1000 + i * 100 for i in range(20)],
        })

        prs_variants = pd.DataFrame({
            "variant_id": ["rs9999"],
            "chromosome": ["1"],
            "position": [1500],
            "effect_allele": ["A"],
            "beta": [0.5],
        })

        trainer = ImputationModelTrainer(random_state=42)
        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Allele frequency should be close to true_af
        model = result.models["rs9999"]
        assert abs(model.allele_frequency - true_af) < 0.05

    def test_compute_training_summary_empty(self):
        """Test training summary with empty models."""
        summary = _compute_training_summary({})

        assert summary["mean_r2"] == 0.0
        assert summary["n_high_quality"] == 0
        assert summary["mean_n_predictors"] == 0.0


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called(self):
        """Test that progress callback is called correctly."""
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=100,
            n_platform_variants=20,
            n_missing_variants=5,
        )

        callback_calls = []

        def progress_callback(variant_id, current, total):
            callback_calls.append((variant_id, current, total))

        trainer = ImputationModelTrainer(
            n_jobs=1,  # Sequential to ensure callback is called
            random_state=42,
            progress_callback=progress_callback,
        )

        trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        # Should be called once per variant
        assert len(callback_calls) == len(prs_variants)

        # Check that current increases from 1 to n_variants
        currents = [call[1] for call in callback_calls]
        assert currents == list(range(1, len(prs_variants) + 1))

        # All calls should have same total
        totals = [call[2] for call in callback_calls]
        assert all(t == len(prs_variants) for t in totals)


class TestConvertToImputedModel:
    """Tests for _convert_to_imputed_model helper."""

    def test_convert_basic(self):
        """Test basic conversion from SingleVariantModelResult."""
        variant_row = pd.Series({
            "variant_id": "rs123",
            "chromosome": "1",
            "position": 1000,
            "effect_allele": "A",
            "other_allele": "G",
            "beta": 0.5,
        })

        result = SingleVariantModelResult(
            coefficients=np.array([0.1, 0.2]),
            intercept=0.3,
            cv_predictions=np.array([0.1, 0.2, 0.3]),
            cv_mse=0.01,
            cv_r2=0.85,
            is_intercept_only=False,
            n_predictors=2,
            n_samples=3,
            l1_ratio=0.5,
            alpha=0.01,
        )

        target_dosages = np.array([0.0, 1.0, 2.0])  # Mean = 1.0, AF = 0.5
        predictor_ids = ["rs1", "rs2"]

        model = _convert_to_imputed_model(variant_row, result, predictor_ids, target_dosages)

        assert model.variant_id == "rs123"
        assert model.chromosome == "1"
        assert model.position == 1000
        assert model.effect_allele == "A"
        assert model.other_allele == "G"
        assert model.beta == 0.5
        assert np.isclose(model.allele_frequency, 0.5)
        assert model.imputation_r2 == 0.85
        assert model.intercept == 0.3
        assert model.predictor_variant_ids == ["rs1", "rs2"]
        assert np.allclose(model.coefficients, [0.1, 0.2])
        assert model.is_intercept_only is False

    def test_convert_with_none_other_allele(self):
        """Test conversion when other_allele is None."""
        variant_row = pd.Series({
            "variant_id": "rs123",
            "chromosome": "1",
            "position": 1000,
            "effect_allele": "A",
            "other_allele": None,
            "beta": 0.5,
        })

        result = SingleVariantModelResult(
            coefficients=np.array([]),
            intercept=0.6,
            cv_predictions=np.array([0.6, 0.6, 0.6]),
            cv_mse=0.1,
            cv_r2=0.0,
            is_intercept_only=True,
            n_predictors=0,
            n_samples=3,
            l1_ratio=0.5,
            alpha=0.01,
        )

        target_dosages = np.array([0.0, 1.0, 2.0])
        predictor_ids = []

        model = _convert_to_imputed_model(variant_row, result, predictor_ids, target_dosages)

        assert model.other_allele is None
        assert model.is_intercept_only is True


class TestMaxPredictors:
    """Tests for max_predictors parameter."""

    def test_max_predictors_limits_coefficients(self):
        """Test that max_predictors limits number of predictor variants."""
        Z, X, prs_variants, platform_variant_info = create_test_data(
            n_samples=200,
            n_platform_variants=100,
            n_missing_variants=3,
        )

        # Use a large window but limit predictors
        trainer = ImputationModelTrainer(
            window_size=1_000_000,  # Large window
            max_predictors=5,
            random_state=42,
        )

        result = trainer.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        for model in result.models.values():
            # Should have at most max_predictors
            assert len(model.predictor_variant_ids) <= 5


class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_random_state_same_results(self):
        """Test that same random state produces identical results."""
        Z, X, prs_variants, platform_variant_info = create_test_data()

        trainer1 = ImputationModelTrainer(random_state=42)
        result1 = trainer1.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        trainer2 = ImputationModelTrainer(random_state=42)
        result2 = trainer2.fit_all_variants(Z, X, prs_variants, platform_variant_info)

        assert result1.n_variants_trained == result2.n_variants_trained
        assert set(result1.models.keys()) == set(result2.models.keys())

        for var_id in result1.models:
            m1 = result1.models[var_id]
            m2 = result2.models[var_id]
            assert m1.imputation_r2 == m2.imputation_r2
            assert np.allclose(m1.coefficients, m2.coefficients)
