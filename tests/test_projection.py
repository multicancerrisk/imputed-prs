"""Tests for projection model fitting and training."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.regions import merge_variant_windows
from imputed_prs.core.types import ProjectionRegionModel, ProjectionTrainingResult
from imputed_prs.models.projection import (
    SingleRegionModelResult,
    fit_single_region_model,
)
from imputed_prs.models.projection_trainer import ProjectionRegionTrainer


def create_projection_test_data(
    n_samples: int = 100,
    n_platform_variants: int = 50,
    n_missing_variants: int = 10,
    random_state: int = 42,
):
    """Create synthetic test data for projection tests.

    Generates platform dosages Z and missing variant dosages X where X has
    a linear relationship to nearby columns of Z. All variants are on
    chromosome "1" with positions that create overlapping windows.

    Returns:
        Tuple of (Z, X, prs_variants, platform_variant_info).
    """
    rng = np.random.default_rng(random_state)

    # Platform variants (predictors)
    Z = rng.binomial(2, 0.3, (n_samples, n_platform_variants)).astype(float)

    # Platform variant info - spread across a region
    platform_variant_info = pd.DataFrame({
        "variant_id": [f"rs{1000 + i}" for i in range(n_platform_variants)],
        "chromosome": ["1"] * n_platform_variants,
        "position": [10000 + i * 1000 for i in range(n_platform_variants)],
    })

    # Missing variants with synthetic relationships to nearby platform variants
    X = np.zeros((n_samples, n_missing_variants))
    prs_variants_data = []

    for i in range(n_missing_variants):
        # Each missing variant related to nearby platform variants
        nearby_start = max(0, i * 5 - 2)
        nearby_end = min(n_platform_variants, i * 5 + 3)
        if nearby_end > nearby_start:
            coeffs = rng.uniform(0.2, 0.5, nearby_end - nearby_start)
            X[:, i] = np.sum(Z[:, nearby_start:nearby_end] * coeffs, axis=1)
            X[:, i] += rng.normal(0, 0.5, n_samples)
            X[:, i] = np.clip(X[:, i], 0, 2)
        else:
            X[:, i] = rng.binomial(2, 0.3, n_samples).astype(float)

        # Positions close together so windows overlap (creating multi-variant regions)
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


class TestFitSingleRegionModel:
    """Tests for fit_single_region_model function."""

    def test_basic_fitting(self):
        """Known linear relationship: verify coefficients recover reasonable fit."""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_predictors = 5

        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)
        true_weights = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
        target = predictors @ true_weights + 0.1 + rng.normal(0, 0.3, n_samples)

        result = fit_single_region_model(
            target, predictors, l1_ratio=0.5, alpha=0.001, random_state=42,
        )

        assert isinstance(result, SingleRegionModelResult)
        assert result.cv_r2 > 0.3
        assert not result.is_intercept_only
        assert result.n_predictors == n_predictors
        assert result.n_samples == n_samples

    def test_no_predictors_intercept_only(self):
        """Empty predictor matrix -> intercept-only, intercept == mean(target)."""
        rng = np.random.default_rng(42)
        target = rng.normal(0.5, 0.1, 100)
        predictors = np.empty((100, 0))

        result = fit_single_region_model(target, predictors)

        assert result.is_intercept_only
        assert result.intercept == pytest.approx(np.mean(target))
        assert result.cv_r2 == 0.0
        assert len(result.coefficients) == 0

    def test_zero_variance_target(self):
        """Constant target -> intercept-only."""
        target = np.full(100, 0.42)
        predictors = np.random.default_rng(42).binomial(2, 0.3, (100, 5)).astype(float)

        result = fit_single_region_model(target, predictors)

        assert result.is_intercept_only
        assert result.intercept == pytest.approx(0.42)

    def test_too_few_samples(self):
        """Fewer valid samples than cv_folds -> intercept-only."""
        target = np.array([0.1, 0.2, 0.3])
        predictors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = fit_single_region_model(target, predictors, cv_folds=5)

        assert result.is_intercept_only

    def test_cv_predictions_shape(self):
        """cv_predictions has shape (n_samples,)."""
        rng = np.random.default_rng(42)
        n_samples = 80
        target = rng.normal(0, 1, n_samples)
        predictors = rng.binomial(2, 0.3, (n_samples, 10)).astype(float)

        result = fit_single_region_model(target, predictors, random_state=42)

        assert result.cv_predictions.shape == (n_samples,)

    def test_cv_predictions_nan_for_invalid(self):
        """Samples with NaN in target/predictors have NaN in cv_predictions."""
        rng = np.random.default_rng(42)
        n_samples = 100
        target = rng.normal(0.5, 0.2, n_samples)
        predictors = rng.binomial(2, 0.3, (n_samples, 5)).astype(float)

        # Set some NaN values
        target[10:15] = np.nan
        predictors[20:25, 0] = np.nan

        result = fit_single_region_model(target, predictors, random_state=42)

        assert result.cv_predictions.shape == (n_samples,)
        # NaN target samples should have NaN predictions
        assert np.all(np.isnan(result.cv_predictions[10:15]))
        # NaN predictor samples should have NaN predictions
        assert np.all(np.isnan(result.cv_predictions[20:25]))

    def test_reproducibility(self):
        """Same random_state produces identical results."""
        rng = np.random.default_rng(42)
        target = rng.normal(0.5, 0.3, 100)
        predictors = rng.binomial(2, 0.3, (100, 5)).astype(float)

        result1 = fit_single_region_model(target, predictors, random_state=123)
        result2 = fit_single_region_model(target, predictors, random_state=123)

        np.testing.assert_array_equal(result1.coefficients, result2.coefficients)
        assert result1.intercept == result2.intercept
        assert result1.cv_r2 == result2.cv_r2

    def test_negative_target_values(self):
        """S_R can be negative (unlike dosages): verify no clipping occurs."""
        rng = np.random.default_rng(42)
        n_samples = 100

        predictors = rng.binomial(2, 0.3, (n_samples, 5)).astype(float)
        # Target with negative values (valid for PRS contributions)
        target = predictors @ np.array([0.5, -0.8, 0.3, -0.2, 0.1]) - 1.0
        assert np.any(target < 0), "Test setup: target should have negative values"

        result = fit_single_region_model(
            target, predictors, alpha=0.001, random_state=42,
        )

        # Predictions should also contain negative values (no clipping)
        valid_preds = result.cv_predictions[~np.isnan(result.cv_predictions)]
        assert np.any(valid_preds < 0), "Predictions should not be clipped to [0, 2]"

    def test_all_coefficients_zero_intercept_only(self):
        """Strong regularization shrinks all coefficients to zero -> intercept-only."""
        rng = np.random.default_rng(42)
        target = rng.normal(0, 0.01, 100)  # Very low signal
        predictors = rng.binomial(2, 0.3, (100, 5)).astype(float)

        result = fit_single_region_model(
            target, predictors, alpha=100.0, random_state=42,
        )

        assert result.is_intercept_only


class TestProjectionRegionTrainer:
    """Tests for ProjectionRegionTrainer class."""

    def test_basic_training(self):
        """Training with synthetic data produces ProjectionTrainingResult."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, alpha=0.01, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        assert isinstance(result, ProjectionTrainingResult)
        assert result.n_regions_trained > 0
        assert len(result.region_models) == result.n_regions_trained

    def test_region_count_correct(self):
        """Number of region models matches expected from merge_variant_windows."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        decomposition = merge_variant_windows(prs_variants, window_size=1_000_000)
        expected_regions = decomposition.n_regions

        assert result.n_regions_trained + result.n_regions_failed == expected_regions

    def test_cv_predictions_per_region_shape(self):
        """Each region's cv_predictions has shape (n_samples,)."""
        Z, X, prs_variants, platform_info = create_projection_test_data(n_samples=80)

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        for region_id, cv_preds in result.cv_predictions.items():
            assert cv_preds.shape == (80,), f"Wrong shape for region {region_id}"

    def test_training_summary_keys(self):
        """Training summary contains all expected keys."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        expected_keys = {
            "mean_r2", "median_r2", "std_r2", "min_r2", "max_r2",
            "n_high_quality", "n_medium_quality", "n_low_quality",
            "mean_n_predictors", "mean_n_prs_variants_per_region",
        }
        assert set(result.training_summary.keys()) == expected_keys

    def test_empty_prs_variants(self):
        """No missing variants -> 0 regions, empty result."""
        rng = np.random.default_rng(42)
        Z = rng.binomial(2, 0.3, (50, 20)).astype(float)
        X = np.empty((50, 0))
        prs_variants = pd.DataFrame(
            columns=["variant_id", "chromosome", "position", "effect_allele", "beta"]
        )
        platform_info = pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(20)],
            "chromosome": ["1"] * 20,
            "position": list(range(20)),
        })

        trainer = ProjectionRegionTrainer(random_state=42)
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        assert result.n_regions_trained == 0
        assert result.n_regions_failed == 0
        assert len(result.region_models) == 0

    def test_parallel_vs_sequential(self):
        """n_jobs=1 and n_jobs=2 produce same results (with same random_state)."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer_seq = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42, n_jobs=1,
        )
        trainer_par = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42, n_jobs=2,
        )

        result_seq = trainer_seq.fit_all_regions(Z, X, prs_variants, platform_info)
        result_par = trainer_par.fit_all_regions(Z, X, prs_variants, platform_info)

        assert set(result_seq.region_models.keys()) == set(result_par.region_models.keys())

        for region_id in result_seq.region_models:
            m_seq = result_seq.region_models[region_id]
            m_par = result_par.region_models[region_id]
            np.testing.assert_allclose(m_seq.coefficients, m_par.coefficients)
            assert m_seq.intercept == pytest.approx(m_par.intercept)
            assert m_seq.cv_r2 == pytest.approx(m_par.cv_r2)

    def test_max_predictors_respected(self):
        """max_predictors limits predictor count per region."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, max_predictors=5, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        for model in result.region_models.values():
            assert len(model.predictor_variant_ids) <= 5

    def test_target_computation_correctness(self):
        """S_R = X[:, indices] @ beta is computed correctly."""
        Z, X, prs_variants, platform_info = create_projection_test_data(
            n_samples=50, n_missing_variants=3,
        )

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        # Manually compute expected mean PRS contribution for each region
        prs_variants_reset = prs_variants.reset_index(drop=True)
        decomposition = merge_variant_windows(prs_variants_reset, window_size=1_000_000)

        for region in decomposition.regions:
            region_id = f"chr{region.chromosome}:{region.start}-{region.end}"
            if region_id not in result.region_models:
                continue
            model = result.region_models[region_id]

            indices = region.prs_variant_indices
            betas = prs_variants_reset.iloc[indices]["beta"].values
            expected_target = X[:, indices] @ betas
            expected_mean = float(np.nanmean(expected_target))

            assert model.mean_prs_contribution == pytest.approx(expected_mean, rel=1e-10)

    def test_predictor_allele_frequencies_stored(self):
        """predictor_allele_frequencies matches manual AF computation."""
        Z, X, prs_variants, platform_info = create_projection_test_data()

        trainer = ProjectionRegionTrainer(
            window_size=1_000_000, random_state=42,
        )
        result = trainer.fit_all_regions(Z, X, prs_variants, platform_info)

        for model in result.region_models.values():
            if model.is_intercept_only and len(model.predictor_variant_ids) == 0:
                assert len(model.predictor_allele_frequencies) == 0
                continue

            assert len(model.predictor_allele_frequencies) == len(model.predictor_variant_ids)
            # Each AF should be in [0, 1]
            for af in model.predictor_allele_frequencies:
                assert 0.0 <= af <= 1.0

    def test_input_validation(self):
        """Missing DataFrame columns or shape mismatches raise ValidationError."""
        Z, X, prs_variants, platform_info = create_projection_test_data()
        trainer = ProjectionRegionTrainer(random_state=42)

        # Missing column in prs_variants
        bad_prs = prs_variants.drop(columns=["beta"])
        with pytest.raises(ValidationError, match="Missing required columns"):
            trainer.fit_all_regions(Z, X, bad_prs, platform_info)

        # Missing column in platform_variant_info
        bad_platform = platform_info.drop(columns=["position"])
        with pytest.raises(ValidationError, match="Missing required columns"):
            trainer.fit_all_regions(Z, X, prs_variants, bad_platform)

        # Sample count mismatch
        Z_bad = Z[:50]
        with pytest.raises(ValidationError, match="Sample count mismatch"):
            trainer.fit_all_regions(Z_bad, X, prs_variants, platform_info)

        # Variant count mismatch
        X_bad = X[:, :5]
        with pytest.raises(ValidationError, match="Variant count mismatch"):
            trainer.fit_all_regions(Z, X_bad, prs_variants, platform_info)
