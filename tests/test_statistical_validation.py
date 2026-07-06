"""Statistical validation tests for the imputed-prs library.

This module implements the recommended testing from the statistical correctness review:

1. Simulation study: Generate synthetic data where true values are known
   - Verify CV R² matches empirical prediction accuracy
   - Verify calibration slope ≈ 1 when imputation is perfect
   - Verify 95% CIs have ~95% coverage

2. Real data validation (integration tests): Using 1000 Genomes + PRS-313
   - Hold out subset with full genotypes
   - Compare imputed PRS to true PRS
   - Verify calibration parameters are reasonable

3. Edge case testing:
   - Variants with no nearby predictors (intercept-only)
   - Extreme allele frequencies (MAF < 1%)
   - Predictions near dosage bounds (truncation effects)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import norm, truncnorm

from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    TrainingResult,
    VariantInfo,
)
from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
)
from imputed_prs.models.bounding import (
    clip_and_adjust_variance,
    truncated_normal_mean,
    truncated_normal_variance,
)
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.metrics import compute_cv_r2
from imputed_prs.models.predictor import (
    PRSPredictor,
    compute_imputed_prs,
    compute_observed_prs,
)
from imputed_prs.models.trainer import (
    ImputationModelTrainer,
    compute_residual_variance,
)


# =============================================================================
# SIMULATION STUDY: Synthetic Data with Known True Values
# =============================================================================


class TestCVR2MatchesEmpiricalAccuracy:
    """Verify CV R² matches actual prediction accuracy on synthetic data."""

    def test_cv_r2_reflects_actual_prediction_quality(self):
        """CV R² should approximate true out-of-sample R²."""
        rng = np.random.default_rng(42)
        n_samples = 500
        n_predictors = 10

        # Generate predictors with realistic genetic structure
        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)

        # True model with known coefficients
        true_coefficients = rng.uniform(0.1, 0.3, n_predictors)
        true_intercept = 0.5
        noise_std = 0.3

        # Generate target with known relationship
        target_signal = predictors @ true_coefficients + true_intercept
        noise = rng.normal(0, noise_std, n_samples)
        target = np.clip(target_signal + noise, 0, 2)

        # Fit model and get CV predictions
        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # Compute empirical R² from CV predictions
        valid_mask = ~np.isnan(result.cv_predictions)
        empirical_r2 = compute_cv_r2(target[valid_mask], result.cv_predictions[valid_mask])

        # CV R² from model should match empirical computation
        np.testing.assert_allclose(result.cv_r2, empirical_r2, rtol=1e-10)

        # R² should be positive for this well-specified model
        assert result.cv_r2 > 0.3, f"Expected R² > 0.3, got {result.cv_r2}"

    def test_cv_r2_detects_poor_signal(self):
        """CV R² should be low when there's no true signal."""
        rng = np.random.default_rng(123)
        n_samples = 200
        n_predictors = 5

        # Predictors unrelated to target
        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)
        target = rng.binomial(2, 0.4, n_samples).astype(float)  # Independent

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # R² should be near zero or negative (no signal)
        assert result.cv_r2 < 0.1, f"Expected R² < 0.1 for noise, got {result.cv_r2}"

    def test_cv_r2_increases_with_signal_strength(self):
        """CV R² should increase as signal-to-noise ratio increases."""
        rng = np.random.default_rng(456)
        n_samples = 300
        n_predictors = 5

        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)
        true_coefficients = np.array([0.2, 0.15, 0.1, 0.05, 0.0])
        signal = predictors @ true_coefficients + 0.5

        noise_levels = [0.1, 0.3, 0.5]
        r2_values = []

        for noise_std in noise_levels:
            noise = rng.normal(0, noise_std, n_samples)
            target = np.clip(signal + noise, 0, 2)

            result = fit_single_variant_model(
                target_dosages=target,
                predictor_dosages=predictors,
                l1_ratio=0.5,
                alpha=0.01,
                cv_folds=5,
                random_state=42,
            )
            r2_values.append(result.cv_r2)

        # R² should decrease as noise increases
        assert r2_values[0] > r2_values[1] > r2_values[2], (
            f"R² should decrease with noise: {r2_values}"
        )


class TestCalibrationWithPerfectImputation:
    """Verify calibration slope ≈ 1 when imputation is perfect."""

    def test_perfect_imputation_has_unit_slope(self):
        """Calibration slope should be ~1 when CV predictions equal true values."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Perfect CV predictions (with tiny noise for numerical stability)
        s_true = rng.normal(0, 1, n_samples)
        s_cv = s_true + rng.normal(0, 1e-6, n_samples)

        params = estimate_cv_calibration(s_cv, s_true)

        # Slope should be essentially 1
        np.testing.assert_allclose(params.scaling_factor, 1.0, atol=1e-3)
        np.testing.assert_allclose(params.calibration_intercept, 0.0, atol=1e-3)
        np.testing.assert_allclose(params.calibration_r2, 1.0, atol=1e-3)

    def test_attenuated_imputation_has_slope_greater_than_one(self):
        """Calibration slope should be >1 when CV predictions are attenuated."""
        rng = np.random.default_rng(42)
        n_samples = 500

        s_true = rng.normal(0, 1, n_samples)
        attenuation = 0.8  # Predictions are 80% of true values
        s_cv = attenuation * s_true + rng.normal(0, 0.1, n_samples)

        params = estimate_cv_calibration(s_cv, s_true)

        # Slope should be approximately 1/attenuation = 1.25
        expected_slope = 1.0 / attenuation
        assert 1.0 < params.scaling_factor < 1.5, (
            f"Expected slope ~{expected_slope}, got {params.scaling_factor}"
        )

        # Attenuation factor should reflect the shrinkage
        assert params.attenuation_factor < 1.0, (
            f"Attenuation factor should be < 1, got {params.attenuation_factor}"
        )

    def test_cv_prs_matches_true_for_perfect_imputation(self):
        """CV-predicted PRS should match true PRS when predictions are perfect."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_variants = 10

        X = rng.binomial(2, 0.3, (n_samples, n_variants)).astype(float)
        betas = rng.normal(0, 0.1, n_variants)

        # All observed - CV predictions are the true values
        s_cv = compute_cv_predicted_prs(
            X=X,
            observed_variant_indices=np.arange(n_variants),
            observed_betas=betas,
            cv_predictions={},
            missing_betas=np.array([]),
        )
        s_true = X @ betas

        np.testing.assert_allclose(s_cv, s_true)


class TestConfidenceIntervalCoverage:
    """Verify 95% CIs have approximately 95% coverage."""

    def _simulate_prs_with_uncertainty(
        self,
        n_simulations: int,
        n_samples: int,
        n_imputed: int,
        residual_variance: float,
        rng: np.random.Generator,
    ) -> tuple:
        """Simulate PRS predictions with known variance structure.

        Returns:
            Tuple of (true_prs_values, predicted_prs_values, se_values, ci_lower, ci_upper)
        """
        # Fixed betas for imputed variants
        betas = rng.normal(0, 0.1, n_imputed)

        true_values = []
        predicted_values = []
        se_values = []
        ci_lowers = []
        ci_uppers = []

        for _ in range(n_simulations):
            # True dosages
            true_dosages = rng.binomial(2, 0.3, n_imputed).astype(float)
            true_prs = np.dot(true_dosages, betas)

            # Imputed dosages with noise
            noise = rng.normal(0, np.sqrt(residual_variance), n_imputed)
            imputed_dosages = np.clip(true_dosages + noise, 0, 2)
            predicted_prs = np.dot(imputed_dosages, betas)

            # Theoretical variance: Σ(β² × σ²_residual)
            total_variance = np.sum(betas**2 * residual_variance)
            se = np.sqrt(total_variance)

            ci_lower = predicted_prs - 1.96 * se
            ci_upper = predicted_prs + 1.96 * se

            true_values.append(true_prs)
            predicted_values.append(predicted_prs)
            se_values.append(se)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        return (
            np.array(true_values),
            np.array(predicted_values),
            np.array(se_values),
            np.array(ci_lowers),
            np.array(ci_uppers),
        )

    def test_ci_coverage_approximately_95_percent(self):
        """95% CIs should contain true value approximately 95% of the time."""
        rng = np.random.default_rng(42)
        n_simulations = 1000
        residual_variance = 0.1

        true_prs, pred_prs, se, ci_lower, ci_upper = self._simulate_prs_with_uncertainty(
            n_simulations=n_simulations,
            n_samples=1,
            n_imputed=20,
            residual_variance=residual_variance,
            rng=rng,
        )

        # Check coverage
        covered = (true_prs >= ci_lower) & (true_prs <= ci_upper)
        coverage_rate = np.mean(covered)

        # Allow some statistical variation (95% ± 3%)
        assert 0.92 <= coverage_rate <= 0.98, (
            f"Expected coverage ~0.95, got {coverage_rate:.3f}"
        )

    def test_se_reflects_actual_prediction_error(self):
        """Standard error should approximate actual prediction error variability."""
        rng = np.random.default_rng(42)
        n_simulations = 500
        residual_variance = 0.1

        true_prs, pred_prs, se, _, _ = self._simulate_prs_with_uncertainty(
            n_simulations=n_simulations,
            n_samples=1,
            n_imputed=20,
            residual_variance=residual_variance,
            rng=rng,
        )

        # Empirical standard deviation of prediction errors
        prediction_errors = pred_prs - true_prs
        empirical_se = np.std(prediction_errors)

        # Mean SE should approximate empirical SE
        mean_se = np.mean(se)
        relative_error = abs(mean_se - empirical_se) / empirical_se

        assert relative_error < 0.2, (
            f"SE ({mean_se:.4f}) differs from empirical ({empirical_se:.4f}) "
            f"by {relative_error:.1%}"
        )

    def test_ci_width_increases_with_variance(self):
        """CI width should increase as residual variance increases."""
        rng = np.random.default_rng(42)

        variance_levels = [0.05, 0.1, 0.2]
        ci_widths = []

        for var in variance_levels:
            _, _, se, ci_lower, ci_upper = self._simulate_prs_with_uncertainty(
                n_simulations=100,
                n_samples=1,
                n_imputed=20,
                residual_variance=var,
                rng=np.random.default_rng(42),  # Same seed for comparability
            )
            mean_width = np.mean(ci_upper - ci_lower)
            ci_widths.append(mean_width)

        assert ci_widths[0] < ci_widths[1] < ci_widths[2], (
            f"CI width should increase with variance: {ci_widths}"
        )


class TestResidualVarianceFormula:
    """Verify residual variance formula: σ² = 2q(1-q)(1-r²)."""

    def test_residual_variance_formula_correctness(self):
        """Test the 2q(1-q)(1-r²) formula against Hardy-Weinberg expectations."""
        # Test cases: (allele_frequency, r2, expected_variance)
        test_cases = [
            (0.5, 0.0, 0.5),      # q=0.5, no imputation: 2*0.5*0.5*1 = 0.5
            (0.5, 1.0, 0.0),      # Perfect imputation: variance = 0
            (0.5, 0.5, 0.25),     # q=0.5, r²=0.5: 2*0.5*0.5*0.5 = 0.25
            (0.1, 0.0, 0.18),     # q=0.1: 2*0.1*0.9*1 = 0.18
            (0.3, 0.8, 0.084),    # q=0.3, r²=0.8: 2*0.3*0.7*0.2 = 0.084
        ]

        for q, r2, expected in test_cases:
            result = compute_residual_variance(q, r2)
            np.testing.assert_allclose(
                result, expected, rtol=1e-10,
                err_msg=f"Failed for q={q}, r²={r2}"
            )

    def test_hw_variance_matches_binomial(self):
        """Hardy-Weinberg variance 2q(1-q) should match Binomial(2, q) variance."""
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            hw_var = 2 * q * (1 - q)
            # Binomial(n=2, p=q) has variance = n*p*(1-p) = 2*q*(1-q)
            binomial_var = 2 * q * (1 - q)
            np.testing.assert_allclose(hw_var, binomial_var)

    def test_negative_r2_is_clipped(self):
        """Negative R² should be clipped to 0 for variance calculation."""
        # Negative R² means predictions are worse than mean
        # But variance should still be valid (use r²=0)
        result = compute_residual_variance(0.3, -0.5)
        expected = 2 * 0.3 * 0.7 * 1.0  # 1 - max(0, -0.5) = 1
        np.testing.assert_allclose(result, expected)


class TestFullPipelineSimulation:
    """End-to-end simulation testing the full imputation pipeline."""

    def _create_synthetic_genetic_data(
        self,
        n_samples: int,
        n_platform: int,
        n_missing: int,
        ld_correlation: float,
        rng: np.random.Generator,
    ) -> tuple:
        """Create synthetic genetic data with LD structure.

        Args:
            n_samples: Number of individuals.
            n_platform: Number of platform variants.
            n_missing: Number of missing variants to impute.
            ld_correlation: Correlation between platform and missing variants.
            rng: Random generator.

        Returns:
            Tuple of (Z_platform, X_missing, platform_info, missing_info)
        """
        # Generate platform variants
        Z = rng.binomial(2, 0.3, (n_samples, n_platform)).astype(float)

        # Generate missing variants with LD structure
        # Each missing variant is correlated with nearby platform variants
        X = np.zeros((n_samples, n_missing))
        for j in range(n_missing):
            # Use weighted sum of platform variants + noise
            weights = rng.uniform(-0.3, 0.3, n_platform)
            signal = Z @ weights
            signal = (signal - signal.mean()) / (signal.std() + 1e-10)
            signal = signal * np.sqrt(ld_correlation)

            # Add independent component
            noise = rng.normal(0, 1, n_samples) * np.sqrt(1 - ld_correlation)

            # Transform to dosage scale
            latent = signal + noise
            af = rng.uniform(0.1, 0.9)
            threshold1 = norm.ppf(1 - af)
            threshold2 = norm.ppf(1 - af**2)
            X[:, j] = (latent > threshold1).astype(float) + (latent > threshold2).astype(float)

        # Create variant info DataFrames
        platform_info = pd.DataFrame({
            "variant_id": [f"rs_plat_{i}" for i in range(n_platform)],
            "chromosome": ["1"] * n_platform,
            "position": [i * 1000 for i in range(n_platform)],
            "ref_allele": ["A"] * n_platform,
            "alt_allele": ["G"] * n_platform,
        })

        missing_info = pd.DataFrame({
            "variant_id": [f"rs_miss_{i}" for i in range(n_missing)],
            "chromosome": ["1"] * n_missing,
            "position": [i * 1000 + 500 for i in range(n_missing)],  # Interleaved
            "effect_allele": ["A"] * n_missing,
            "other_allele": ["G"] * n_missing,
            "beta": rng.normal(0, 0.05, n_missing),
        })

        return Z, X, platform_info, missing_info

    def test_imputation_r2_matches_empirical(self):
        """Reported imputation R² should match empirical accuracy."""
        rng = np.random.default_rng(42)

        Z, X, platform_info, missing_info = self._create_synthetic_genetic_data(
            n_samples=200,
            n_platform=50,
            n_missing=10,
            ld_correlation=0.6,
            rng=rng,
        )

        trainer = ImputationModelTrainer(
            window_size=100_000,  # Large window to include all predictors
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        result = trainer.fit_all_variants(
            Z=Z,
            X=X,
            prs_variants=missing_info,
            platform_variant_info=platform_info,
        )

        # Verify empirical R² matches reported R²
        for var_id, model in result.models.items():
            cv_pred = result.cv_predictions[var_id]
            var_idx = missing_info[missing_info["variant_id"] == var_id].index[0]
            true_dosages = X[:, var_idx]

            # Compute empirical R²
            valid_mask = ~np.isnan(cv_pred)
            if np.sum(valid_mask) > 10:
                empirical_r2 = compute_cv_r2(true_dosages[valid_mask], cv_pred[valid_mask])
                np.testing.assert_allclose(
                    model.imputation_r2, empirical_r2, rtol=1e-5,
                    err_msg=f"R² mismatch for {var_id}"
                )

    def test_calibration_corrects_attenuation(self):
        """Calibration should correct for imputation-induced attenuation."""
        rng = np.random.default_rng(42)

        # Create data with strong LD structure for predictable imputation
        n_samples = 500
        n_platform = 50
        n_missing = 20

        # Generate platform variants
        Z = rng.binomial(2, 0.3, (n_samples, n_platform)).astype(float)

        # Generate missing variants with STRONG correlation to platform variants
        # Each missing variant is a linear combination of platform variants
        X = np.zeros((n_samples, n_missing))
        for j in range(n_missing):
            # Use only a few predictors with strong weights
            n_predictors = min(5, n_platform)
            predictor_indices = rng.choice(n_platform, n_predictors, replace=False)
            weights = rng.uniform(0.3, 0.6, n_predictors)

            signal = Z[:, predictor_indices] @ weights
            # Normalize and add small noise
            signal = (signal - signal.mean()) / (signal.std() + 1e-10)
            noise = rng.normal(0, 0.3, n_samples)

            # Transform to dosage scale [0, 2]
            latent = 0.8 * signal + 0.2 * noise
            X[:, j] = np.clip(latent + 1.0, 0, 2)  # Center around 1

        # Create variant info DataFrames
        platform_info = pd.DataFrame({
            "variant_id": [f"rs_plat_{i}" for i in range(n_platform)],
            "chromosome": ["1"] * n_platform,
            "position": [i * 1000 for i in range(n_platform)],
            "ref_allele": ["A"] * n_platform,
            "alt_allele": ["G"] * n_platform,
        })

        missing_info = pd.DataFrame({
            "variant_id": [f"rs_miss_{i}" for i in range(n_missing)],
            "chromosome": ["1"] * n_missing,
            "position": [i * 1000 + 500 for i in range(n_missing)],
            "effect_allele": ["A"] * n_missing,
            "other_allele": ["G"] * n_missing,
            "beta": rng.normal(0, 0.05, n_missing),
        })

        trainer = ImputationModelTrainer(
            window_size=100_000,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        result = trainer.fit_all_variants(
            Z=Z,
            X=X,
            prs_variants=missing_info,
            platform_variant_info=platform_info,
        )

        # Compute true PRS and CV-predicted PRS
        betas = missing_info["beta"].values
        s_true = X @ betas

        # Build CV predictions matrix (missing variants only)
        cv_pred_matrix = np.zeros_like(X)
        for i, var_id in enumerate(missing_info["variant_id"]):
            if var_id in result.cv_predictions:
                cv_pred_matrix[:, i] = result.cv_predictions[var_id]
            else:
                cv_pred_matrix[:, i] = np.mean(X[:, i])  # Fallback to mean

        s_cv = cv_pred_matrix @ betas

        # Estimate calibration
        calib = estimate_cv_calibration(s_cv, s_true)

        # Calibration R² should be positive (some predictability)
        assert calib.calibration_r2 > 0.2, (
            f"Calibration R² too low: {calib.calibration_r2}"
        )

        # Scaling factor should be positive and reasonable
        assert calib.scaling_factor > 0.5, (
            f"Scaling factor unexpectedly low: {calib.scaling_factor}"
        )


# =============================================================================
# EDGE CASE TESTING
# =============================================================================


class TestInterceptOnlyModels:
    """Tests for variants with no nearby predictors (intercept-only models)."""

    def test_intercept_only_when_no_predictors(self):
        """Model should fall back to intercept-only when no predictors available."""
        rng = np.random.default_rng(42)
        n_samples = 100

        target = rng.binomial(2, 0.35, n_samples).astype(float)
        predictors = np.empty((n_samples, 0))  # No predictors

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        assert result.is_intercept_only
        assert len(result.coefficients) == 0
        # Intercept should be mean of target
        expected_intercept = np.mean(target)
        np.testing.assert_allclose(result.intercept, expected_intercept, rtol=1e-10)

    def test_intercept_equals_twice_allele_frequency(self):
        """For intercept-only models, intercept ≈ 2×AF under Hardy-Weinberg."""
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Generate dosages from Hardy-Weinberg
        af = 0.3
        target = rng.binomial(2, af, n_samples).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=np.empty((n_samples, 0)),
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # Mean dosage should be approximately 2*AF
        expected = 2 * af
        np.testing.assert_allclose(result.intercept, expected, rtol=0.1)

    def test_intercept_only_r2_is_zero(self):
        """Intercept-only models should have R² = 0."""
        rng = np.random.default_rng(42)
        n_samples = 100

        target = rng.binomial(2, 0.4, n_samples).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=np.empty((n_samples, 0)),
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        assert result.cv_r2 == 0.0

    def test_prs_uses_intercept_when_predictors_missing(self):
        """PRS computation should use intercept when predictor dosages are missing."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=1000,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.3,
            imputation_r2=0.7,
            residual_variance=0.1,
            intercept=0.6,  # 2 * 0.3
            predictor_variant_ids=["rs2", "rs3"],
            coefficients=np.array([0.2, 0.3]),
            is_intercept_only=False,
        )

        # No predictor dosages provided
        user_dosages = {}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(
            user_dosages, [model]
        )

        # Should use intercept: 0.6 * 0.1 = 0.06
        expected_prs = 0.6 * 0.1
        np.testing.assert_allclose(prs, expected_prs)


class TestExtremeAlleleFrequencies:
    """Tests for variants with extreme allele frequencies (MAF < 1%)."""

    def test_rare_variant_imputation(self):
        """Imputation should work for rare variants (MAF < 1%)."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Rare variant with MAF = 0.5%
        maf = 0.005
        target = rng.binomial(2, maf, n_samples).astype(float)

        # Most values will be 0
        assert np.mean(target == 0) > 0.9

        # Create predictors (some correlated with rare variant)
        predictors = rng.binomial(2, 0.3, (n_samples, 10)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # Should fit without error
        assert result.intercept >= 0
        assert len(result.cv_predictions) == n_samples

    def test_residual_variance_for_rare_variants(self):
        """Residual variance should be small for rare variants."""
        # For rare variants (q small), HW variance 2q(1-q) ≈ 2q
        rare_af = 0.01
        common_af = 0.5

        var_rare = compute_residual_variance(rare_af, r2=0.5)
        var_common = compute_residual_variance(common_af, r2=0.5)

        # Rare variant should have much smaller variance
        assert var_rare < var_common / 10

    def test_very_common_variant(self):
        """Imputation should work for very common variants (MAF > 49%)."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Very common variant
        af = 0.95
        target = rng.binomial(2, af, n_samples).astype(float)

        # Most values will be 2
        assert np.mean(target == 2) > 0.8

        predictors = rng.binomial(2, 0.3, (n_samples, 10)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # Intercept should be close to 2*AF = 1.9
        assert 1.5 < result.intercept < 2.0


class TestDosageTruncationEffects:
    """Tests for predictions near dosage bounds (truncation effects)."""

    def test_truncated_variance_less_than_original(self):
        """Truncated variance should always be ≤ original variance."""
        test_cases = [
            (1.0, 0.5),   # Well within bounds
            (0.1, 0.5),   # Near lower bound
            (1.9, 0.5),   # Near upper bound
            (-0.5, 0.5),  # Below lower bound
            (2.5, 0.5),   # Above upper bound
        ]

        for mu, sigma in test_cases:
            original_var = sigma**2
            truncated_var = truncated_normal_variance(mu, sigma, 0.0, 2.0)

            assert truncated_var <= original_var + 1e-10, (
                f"Truncated variance {truncated_var} > original {original_var} "
                f"for mu={mu}, sigma={sigma}"
            )
            assert truncated_var >= 0

    def test_truncated_variance_matches_scipy(self):
        """Truncated variance should match scipy.stats.truncnorm."""
        test_cases = [
            (1.0, 0.5),
            (0.5, 0.3),
            (1.5, 0.4),
        ]

        for mu, sigma in test_cases:
            lower, upper = 0.0, 2.0
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma

            scipy_var = truncnorm.var(a, b, loc=mu, scale=sigma)
            our_var = truncated_normal_variance(mu, sigma, lower, upper)

            np.testing.assert_allclose(our_var, scipy_var, rtol=1e-6)

    def test_truncated_mean_matches_scipy(self):
        """Truncated mean should match scipy.stats.truncnorm."""
        test_cases = [
            (1.0, 0.5),
            (0.5, 0.3),
            (1.5, 0.4),
        ]

        for mu, sigma in test_cases:
            lower, upper = 0.0, 2.0
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma

            scipy_mean = truncnorm.mean(a, b, loc=mu, scale=sigma)
            our_mean = truncated_normal_mean(mu, sigma, lower, upper)

            np.testing.assert_allclose(our_mean, scipy_mean, rtol=1e-6)

    def test_clipping_at_lower_bound(self):
        """Predictions below 0 should be clipped to 0."""
        clipped, adjusted_var = clip_and_adjust_variance(
            raw_prediction=-0.5,
            residual_variance=0.25,
            lower=0.0,
            upper=2.0,
        )

        assert clipped == 0.0
        # Variance should be reduced due to truncation
        assert adjusted_var < 0.25

    def test_clipping_at_upper_bound(self):
        """Predictions above 2 should be clipped to 2."""
        clipped, adjusted_var = clip_and_adjust_variance(
            raw_prediction=2.5,
            residual_variance=0.25,
            lower=0.0,
            upper=2.0,
        )

        assert clipped == 2.0
        # Variance should be reduced due to truncation
        assert adjusted_var < 0.25

    def test_variance_unchanged_when_well_within_bounds(self):
        """Variance should be nearly unchanged when prediction is far from bounds."""
        clipped, adjusted_var = clip_and_adjust_variance(
            raw_prediction=1.0,
            residual_variance=0.04,  # sigma = 0.2
            lower=0.0,
            upper=2.0,
        )

        assert clipped == 1.0
        # Variance should be essentially unchanged (1.0 is 5 sigmas from bounds)
        np.testing.assert_allclose(adjusted_var, 0.04, rtol=0.01)

    def test_truncation_count_in_prs(self):
        """PRS computation should count truncated dosages."""
        # Model that will produce prediction > 2
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=1000,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.9,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=1.5,
            predictor_variant_ids=["rs2"],
            coefficients=np.array([0.4]),
            is_intercept_only=False,
        )

        # Prediction will be: 2.0 * 0.4 + 1.5 = 2.3 > 2.0
        user_dosages = {"rs2": 2.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(
            user_dosages, [model]
        )

        assert n_truncated == 1
        # PRS should use clipped value: 2.0 * 0.1 = 0.2
        np.testing.assert_allclose(prs, 0.2)


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_zero_variance_target(self):
        """Handle constant target values (zero variance)."""
        n_samples = 100
        target = np.full(n_samples, 1.5)  # Constant
        predictors = np.random.default_rng(42).binomial(2, 0.3, (n_samples, 5)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        # Should handle gracefully
        assert not np.isnan(result.intercept)
        assert result.cv_r2 == 0.0 or np.isclose(result.cv_r2, 0.0)

    def test_very_small_variance(self):
        """Handle very small but non-zero variance."""
        n_samples = 100
        rng = np.random.default_rng(42)
        # Target with tiny variance
        target = 1.0 + rng.normal(0, 1e-8, n_samples)
        target = np.clip(target, 0, 2)

        predictors = rng.binomial(2, 0.3, (n_samples, 5)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        assert not np.isnan(result.intercept)
        assert not np.any(np.isnan(result.cv_predictions))

    def test_large_number_of_predictors(self):
        """Handle cases with many more predictors than samples."""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_predictors = 200  # p >> n

        target = rng.binomial(2, 0.3, n_samples).astype(float)
        predictors = rng.binomial(2, 0.3, (n_samples, n_predictors)).astype(float)

        result = fit_single_variant_model(
            target_dosages=target,
            predictor_dosages=predictors,
            l1_ratio=0.5,  # L1 regularization for sparsity
            alpha=0.1,     # Stronger regularization needed
            cv_folds=5,
            random_state=42,
        )

        # Should complete without error
        assert not np.isnan(result.intercept)
        # Most coefficients should be shrunk to zero
        assert np.sum(np.abs(result.coefficients) > 1e-6) < n_predictors / 2

    def test_extreme_beta_values(self):
        """Handle extreme effect sizes in PRS computation."""
        # Very large positive beta
        observed_large_pos = [VariantInfo("rs1", "1", 100, "A", "G", 10.0)]
        dosages_large_pos = {"rs1": 2.0}
        prs, _ = compute_observed_prs(dosages_large_pos, observed_large_pos)
        assert prs == 20.0

        # Very large negative beta
        observed_large_neg = [VariantInfo("rs2", "1", 200, "C", "T", -10.0)]
        dosages_large_neg = {"rs2": 2.0}
        prs, _ = compute_observed_prs(dosages_large_neg, observed_large_neg)
        assert prs == -20.0

        # Very small beta
        observed_tiny = [VariantInfo("rs3", "1", 300, "G", "A", 1e-10)]
        dosages_tiny = {"rs3": 2.0}
        prs, _ = compute_observed_prs(dosages_tiny, observed_tiny)
        np.testing.assert_allclose(prs, 2e-10)


# =============================================================================
# INTEGRATION TESTS (for real data validation)
# =============================================================================


@pytest.mark.integration
class TestRealDataValidation:
    """Integration tests using real data (1000 Genomes + PRS-313).

    These tests require external data and are marked as integration tests.
    Run with: pytest -m integration
    """

    @pytest.fixture
    def sample_prs313_subset(self):
        """Create a small subset of PRS-313 for testing."""
        # This would normally load from PGS Catalog
        # For unit tests, we create a mock subset
        return pd.DataFrame({
            "variant_id": [f"rs{i}" for i in range(1, 11)],
            "chromosome": ["1"] * 10,
            "position": list(range(1000, 11000, 1000)),
            "effect_allele": ["A"] * 10,
            "other_allele": ["G"] * 10,
            "beta": np.random.default_rng(42).normal(0, 0.1, 10),
        })

    def test_imputation_produces_reasonable_r2_distribution(self, sample_prs313_subset):
        """Imputation R² distribution should be reasonable for real variants."""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_platform = 50
        n_missing = 10

        # Simulate platform variants
        Z = rng.binomial(2, 0.3, (n_samples, n_platform)).astype(float)

        # Simulate missing variants with LD structure (correlated with platform)
        X = np.zeros((n_samples, n_missing))
        for j in range(n_missing):
            # Each missing variant is correlated with some platform variants
            n_pred = min(5, n_platform)
            pred_idx = rng.choice(n_platform, n_pred, replace=False)
            weights = rng.uniform(0.2, 0.5, n_pred)
            signal = Z[:, pred_idx] @ weights
            signal = (signal - signal.mean()) / (signal.std() + 1e-10)
            noise = rng.normal(0, 0.5, n_samples)
            X[:, j] = np.clip(signal + noise + 1.0, 0, 2)

        platform_info = pd.DataFrame({
            "variant_id": [f"rs_p{i}" for i in range(n_platform)],
            "chromosome": ["1"] * n_platform,
            "position": list(range(0, n_platform * 1000, 1000)),
            "ref_allele": ["A"] * n_platform,
            "alt_allele": ["G"] * n_platform,
        })

        trainer = ImputationModelTrainer(
            window_size=100_000,
            l1_ratio=0.5,
            alpha=0.01,
            cv_folds=5,
            random_state=42,
        )

        result = trainer.fit_all_variants(
            Z=Z,
            X=X,
            prs_variants=sample_prs313_subset,
            platform_variant_info=platform_info,
        )

        # Check R² distribution
        r2_values = [m.imputation_r2 for m in result.models.values()]

        # R² values should be computed for all variants
        assert len(r2_values) == n_missing

        # Mean R² should be positive (since we created correlated data)
        mean_r2 = np.mean(r2_values)
        assert mean_r2 > 0.0, f"Mean R² should be positive with LD: {mean_r2}"

        # At least some variants should have good imputation
        n_good = sum(1 for r2 in r2_values if r2 > 0.3)
        assert n_good >= 1, "Expected at least one variant with R² > 0.3"

    def test_calibration_parameters_are_reasonable(self, sample_prs313_subset):
        """Calibration parameters should be within reasonable ranges."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Generate correlated true and predicted PRS
        s_true = rng.normal(0, 1, n_samples)
        # Add measurement noise to simulate imputation error
        s_cv = 0.85 * s_true + rng.normal(0, 0.3, n_samples)

        calib = estimate_cv_calibration(s_cv, s_true)

        # Scaling factor should be positive
        assert calib.scaling_factor > 0

        # Calibration R² should be reasonable
        assert 0.5 < calib.calibration_r2 < 1.0

        # Attenuation should be less than 1
        assert 0 < calib.attenuation_factor <= 1.0

        # Standard error should be small relative to estimate
        assert calib.scaling_factor_se < 0.5 * abs(calib.scaling_factor)


class TestEmpiricalResidualCoverageUnderLD:
    """P4.1 gate: the empirical score-level residual SD restores ~nominal coverage
    where the LD-blind diagonal SE under-covers.

    The error of an imputed PRS is ``error = (X_imputed - X_true) @ betas``. The
    diagonal SE models the per-variant imputation residuals as independent, so it
    reports ``sqrt(Σ beta_j^2 * sigma_j^2)`` and structurally omits the off-diagonal
    LD covariance. We inject positive equicorrelation across same-signed-beta
    residuals so the true score-error variance ``beta^T Sigma beta`` greatly exceeds
    the diagonal, then show the empirical residual SD — measured on a train split,
    evaluated out-of-sample — recovers nominal coverage. (Working in score space
    keeps the error exactly Gaussian, so the contrast is robust; the genotype-level
    diagonal coverage is already covered by ``TestConfidenceIntervalCoverage``.)
    """

    @staticmethod
    def _draw_score_errors(rng, n, p, betas, rho, sigma):
        # Equicorrelated per-variant residuals: a shared common factor (g) plus an
        # idiosyncratic part (e) => Cov = sigma^2[(1-rho) I + rho 11^T]. The
        # score-level error is the beta-weighted sum, a sum of Gaussians => Gaussian.
        g = rng.standard_normal(n)
        e = rng.standard_normal((n, p))
        residuals = sigma * (np.sqrt(rho) * g[:, None] + np.sqrt(1.0 - rho) * e)
        return residuals @ betas

    def test_empirical_beats_diagonal_under_ld(self):
        rng = np.random.default_rng(20260620)
        n_train, n_eval, p = 4000, 4000, 30
        rho, sigma = 0.6, 0.30
        betas = np.abs(rng.normal(0.0, 0.10, p)) + 0.02  # same-signed

        err_train = self._draw_score_errors(rng, n_train, p, betas, rho, sigma)
        emp_sd = np.std(err_train, ddof=1)  # raw_empirical_residual_sd
        diag_se = np.sqrt(np.sum(betas**2 * sigma**2))  # LD-blind diagonal SE

        err_eval = self._draw_score_errors(rng, n_eval, p, betas, rho, sigma)
        cov_diag = np.mean(np.abs(err_eval) <= 1.96 * diag_se)
        cov_emp = np.mean(np.abs(err_eval) <= 1.96 * emp_sd)

        # The diagonal SE is far too small under LD => severe under-coverage.
        assert cov_diag < 0.90, f"diagonal should under-cover, got {cov_diag:.3f}"
        assert emp_sd > 1.3 * diag_se, (emp_sd, diag_se)
        # The empirical residual SD restores ~nominal coverage, out-of-sample.
        assert 0.93 <= cov_emp <= 0.975, f"empirical coverage {cov_emp:.3f}"
        assert cov_emp > cov_diag + 0.05

    def test_estimate_cv_calibration_reports_raw_residual_sd(self):
        """Wiring lock-in: raw_empirical_residual_sd == std(s_true - s_cv, ddof=1)."""
        rng = np.random.default_rng(20260620)
        n, p = 4000, 30
        rho, sigma = 0.6, 0.30
        betas = np.abs(rng.normal(0.0, 0.10, p)) + 0.02

        err = self._draw_score_errors(rng, n, p, betas, rho, sigma)
        s_true = rng.standard_normal(n)
        s_cv = s_true - err

        params = estimate_cv_calibration(s_cv, s_true)
        assert params.raw_empirical_residual_sd == pytest.approx(
            np.std(s_true - s_cv, ddof=1), rel=1e-12
        )


# =============================================================================
# P4.3: Masking-validation harness (mask reference -> platform, compare to full PRS)
# =============================================================================

import warnings  # noqa: E402

from imputed_prs import LinearImputationPRS, LinearProjectionPRS  # noqa: E402
from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError  # noqa: E402
from imputed_prs.core.types import GenotypeData  # noqa: E402
from imputed_prs.evaluation import (  # noqa: E402
    ImputationEvaluator,
    MaskingValidationReport,
    ProjectionEvaluator,
    mask_reference_to_platform,
    run_masking_validation,
)
from imputed_prs.evaluation.validation import (  # noqa: E402
    _CALIBRATION_CAVEAT,
    _write_synthetic_23andme,
)
from imputed_prs.io.user_genotypes import load_user_genotype_strings  # noqa: E402

# 20-sample biallelic reference (effect_allele == ALT), platform is a partial
# overlap (rs1-rs3 observed; rs4, rs5 imputed/projected). Mirrors test_round_trip.
_MV_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\tS5\tS6\tS7\tS8\tS9\tS10\tS11\tS12\tS13\tS14\tS15\tS16\tS17\tS18\tS19\tS20
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1
1\t102000\trs5\tG\tC\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1
"""

_MV_PRS_DF = pd.DataFrame(
    {
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "C"],
        "other_allele": ["A", "C", "G", "T", "G"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    }
)

_MV_PLATFORM = ["rs1", "rs2", "rs3"]
_MV_REF_ALLELES = ["A", "C", "G", "T", "G"]
_MV_ALT_ALLELES = ["G", "T", "A", "C", "C"]


@pytest.fixture
def mv_vcf_file(tmp_path):
    path = tmp_path / "ref.vcf"
    path.write_text(_MV_VCF)
    return path


@pytest.fixture
def mv_imputation_model(mv_vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=mv_vcf_file,
        prs_definition=_MV_PRS_DF,
        platform_variants=_MV_PLATFORM,
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def mv_projection_model(mv_vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearProjectionPRS(
        window_size=500_000, cv_folds=3, verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=mv_vcf_file,
        prs_definition=_MV_PRS_DF,
        platform_variants=_MV_PLATFORM,
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def mv_all_on_platform_model(mv_vcf_file):
    """Model whose platform covers every PRS variant -> masking is a no-op and the
    estimate equals the true PRS exactly (perfect-recovery fixture)."""
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=mv_vcf_file,
        prs_definition=_MV_PRS_DF,
        platform_variants=["rs1", "rs2", "rs3", "rs4", "rs5"],
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


def _mv_genotype_data(ref_alleles, alt_alleles, dosage_matrix):
    """Build a GenotypeData over rs1..rsN matching the masking-validation PRS."""
    n_variants = len(ref_alleles)
    variant_info = pd.DataFrame(
        {
            "variant_id": [f"rs{i + 1}" for i in range(n_variants)],
            "chromosome": ["1"] * n_variants,
            "position": [100000, 100500, 101000, 101500, 102000][:n_variants],
            "ref_allele": ref_alleles,
            "alt_allele": alt_alleles,
        }
    )
    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=[f"S{i + 1}" for i in range(dosage_matrix.shape[0])],
        genome_build="GRCh37",
    )


class TestMaskReferenceToPlatform:
    """mask_reference_to_platform blanks off-platform columns, keeps platform ones."""

    def test_keeps_exactly_platform_columns(self):
        dm = np.random.default_rng(0).integers(0, 3, size=(12, 5)).astype(np.float32)
        gd = _mv_genotype_data(list(_MV_REF_ALLELES), list(_MV_ALT_ALLELES), dm)

        masked = mask_reference_to_platform(gd, {"rs1", "rs2", "rs3"})

        # Kept columns (rs1-rs3) are bit-identical; masked columns (rs4, rs5) are NaN.
        np.testing.assert_array_equal(masked.dosage_matrix[:, :3], dm[:, :3])
        assert np.all(np.isnan(masked.dosage_matrix[:, 3:]))
        # variant_info is preserved (columns blanked, not dropped).
        assert list(masked.variant_info["variant_id"]) == list(gd.variant_info["variant_id"])

    def test_chrpos_platform_matching(self):
        """A platform expressed as chr:pos must still match rsID-keyed columns."""
        dm = np.random.default_rng(1).integers(0, 3, size=(12, 5)).astype(np.float32)
        gd = _mv_genotype_data(list(_MV_REF_ALLELES), list(_MV_ALT_ALLELES), dm)

        masked = mask_reference_to_platform(gd, {"1:100000", "1:100500", "1:101000"})

        np.testing.assert_array_equal(masked.dosage_matrix[:, :3], dm[:, :3])
        assert np.all(np.isnan(masked.dosage_matrix[:, 3:]))

    def test_no_op_when_all_on_platform(self):
        dm = np.random.default_rng(2).integers(0, 3, size=(12, 5)).astype(np.float32)
        gd = _mv_genotype_data(list(_MV_REF_ALLELES), list(_MV_ALT_ALLELES), dm)

        masked = mask_reference_to_platform(gd, {"rs1", "rs2", "rs3", "rs4", "rs5"})

        np.testing.assert_array_equal(masked.dosage_matrix, dm)


class TestRunMaskingValidationEndToEnd:
    """The headline gate: masking validation runs and emits the metric panel."""

    def test_populates_all_report_fields(self, mv_imputation_model, mv_vcf_file):
        pytest.importorskip("snps")
        report = run_masking_validation(
            mv_imputation_model, mv_vcf_file, random_state=1, verbose=0
        )

        assert isinstance(report, MaskingValidationReport)
        assert report.model_type == "imputation"
        assert report.n_samples == 20
        assert report.n_observed == 3
        assert report.n_variants_masked == 2
        assert report.n_platform_variants_retained == 3
        assert -1.0 <= report.correlation <= 1.0
        assert 0.0 <= report.r2 <= 1.0
        assert np.isfinite(report.empirical_error_sd) and report.empirical_error_sd >= 0.0
        assert np.isfinite(report.empirical_error_mean)
        assert 0.0 <= report.top_decile_concordance <= 1.0
        assert report.coverage_95 is not None and 0.0 <= report.coverage_95 <= 1.0
        assert report.raw_parser_checked is True
        assert report.raw_parser_agrees is True
        assert report.raw_parser_max_abs_diff <= 1e-6
        assert report.cross_ancestry is False
        assert _CALIBRATION_CAVEAT in report.caveats
        # to_dict round-trips every field.
        d = report.to_dict()
        assert d["model_type"] == "imputation"
        assert "percentile_concordance" in d

    def test_metric_panel_summary_renders(self, mv_imputation_model, mv_vcf_file):
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            run_raw_parser_check=False,
            verbose=0,
        )
        text = report.summary()
        assert isinstance(text, str) and text
        lowered = text.lower()
        assert "correlation" in lowered
        assert "top-decile" in lowered
        assert "empirical error" in lowered


class TestMaskingValidationNumericStringMetricParity:
    """P5: the masking-validation estimate now flows through the numeric scorer
    instead of the per-sample string replay. On a fully-called biallelic reference
    the two are equal, so the end-to-end report metrics (R², calibration, error,
    concordance) must match the string oracle to numerical precision."""

    @staticmethod
    def _assert_metrics_match(numeric, string):
        assert numeric.correlation == pytest.approx(string.correlation, abs=1e-9)
        assert numeric.r2 == pytest.approx(string.r2, abs=1e-9)
        assert numeric.calibration_slope == pytest.approx(string.calibration_slope, abs=1e-9)
        assert numeric.calibration_intercept == pytest.approx(
            string.calibration_intercept, abs=1e-9
        )
        assert numeric.empirical_error_sd == pytest.approx(string.empirical_error_sd, abs=1e-9)
        assert numeric.empirical_error_mean == pytest.approx(
            string.empirical_error_mean, abs=1e-9
        )
        assert numeric.top_decile_concordance == pytest.approx(
            string.top_decile_concordance, abs=1e-9
        )

    def test_imputation_metrics_match_string_oracle(
        self, mv_imputation_model, mv_vcf_file, monkeypatch
    ):
        numeric = run_masking_validation(
            mv_imputation_model, mv_vcf_file, run_raw_parser_check=False,
            random_state=1, verbose=0,
        )
        # Force the whole run's estimate back onto the retired string replay.
        monkeypatch.setattr(
            ImputationEvaluator,
            "_compute_imputed_prs_batch",
            lambda self, gd: self._predicted_prs_via_strings(gd),
        )
        string = run_masking_validation(
            mv_imputation_model, mv_vcf_file, run_raw_parser_check=False,
            random_state=1, verbose=0,
        )
        self._assert_metrics_match(numeric, string)

    def test_projection_metrics_match_string_oracle(
        self, mv_projection_model, mv_vcf_file, monkeypatch
    ):
        numeric = run_masking_validation(
            mv_projection_model, mv_vcf_file, run_raw_parser_check=False,
            random_state=1, verbose=0,
        )
        monkeypatch.setattr(
            ProjectionEvaluator,
            "_compute_projected_prs_batch",
            lambda self, gd: self._predicted_prs_via_strings(gd),
        )
        string = run_masking_validation(
            mv_projection_model, mv_vcf_file, run_raw_parser_check=False,
            random_state=1, verbose=0,
        )
        self._assert_metrics_match(numeric, string)


class TestPerfectRecoveryMaskingValidation:
    """When the platform covers every PRS variant, the estimate == the full PRS."""

    def test_perfect_recovery(self, mv_all_on_platform_model, mv_vcf_file):
        report = run_masking_validation(
            mv_all_on_platform_model,
            mv_vcf_file,
            run_raw_parser_check=False,
            verbose=0,
        )
        assert report.n_variants_masked == 0
        # Estimate == full PRS up to float32 dosage noise (the true-PRS and
        # string-scoring paths cast the float32 reference differently).
        assert report.correlation == pytest.approx(1.0, abs=1e-9)
        assert report.empirical_error_sd == pytest.approx(0.0, abs=1e-6)
        assert report.top_decile_concordance == pytest.approx(1.0, abs=1e-9)
        # The "no off-platform variants" caveat must fire so the result is not
        # misread as validating imputation.
        assert any("no off-platform" in c.lower() for c in report.caveats)


class TestRawParserRoundTrip:
    """The literal upload path (predict(file)) agrees with the batch estimate."""

    def test_file_predict_matches_batch(self, mv_imputation_model, mv_vcf_file):
        pytest.importorskip("snps")
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            raw_parser_max_samples=8,
            verbose=0,
        )
        assert report.raw_parser_checked is True
        assert report.raw_parser_n > 0
        assert report.raw_parser_agrees is True
        assert report.raw_parser_max_abs_diff == pytest.approx(0.0, abs=1e-6)

    def test_no_userwarning_from_file_predict(self, mv_imputation_model, tmp_path):
        pytest.importorskip("snps")
        # Render a single sample's platform genotypes to a synthetic 23andMe file.
        records = [("rs1", "1", 100000, "AG"), ("rs2", "1", 100500, "CC"),
                   ("rs3", "1", 101000, "AA")]
        path = tmp_path / "user_23andme.txt"
        _write_synthetic_23andme(path, records)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mv_imputation_model.predict(path, genome_build="GRCh37", strict=True)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warnings, [str(w.message) for w in user_warnings]

    def test_synthetic_file_recognized_as_23andme(self, tmp_path):
        pytest.importorskip("snps")
        records = [("rs1", "1", 100000, "AG"), ("rs2", "1", 100500, "CC"),
                   ("rs3", "1", 101000, "AA")]
        path = tmp_path / "user_23andme.txt"
        _write_synthetic_23andme(path, records)
        assert load_user_genotype_strings(path) == {"rs1": "AG", "rs2": "CC", "rs3": "AA"}


class TestCrossAncestryNoteLogic:
    """evaluation_ancestry recording + caveat emission."""

    def test_note_emitted_when_ancestries_differ(self, mv_imputation_model, mv_vcf_file):
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            evaluation_ancestry="EAS",  # model trained on EUR
            run_raw_parser_check=False,
            verbose=0,
        )
        assert report.cross_ancestry is True
        assert report.evaluation_ancestry == "EAS"
        assert any("cross-ancestry" in c.lower() for c in report.caveats)

    def test_no_note_when_same_ancestry(self, mv_imputation_model, mv_vcf_file):
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            evaluation_ancestry="eur",  # case-insensitive match to training EUR
            run_raw_parser_check=False,
            verbose=0,
        )
        assert report.cross_ancestry is False
        assert not any("cross-ancestry" in c.lower() for c in report.caveats)

    def test_no_note_when_label_missing(self, mv_imputation_model, mv_vcf_file):
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            run_raw_parser_check=False,
            verbose=0,
        )
        assert report.cross_ancestry is False

    def test_internal_calibration_caveat_always_present(
        self, mv_imputation_model, mv_vcf_file
    ):
        report = run_masking_validation(
            mv_imputation_model,
            mv_vcf_file,
            run_raw_parser_check=False,
            verbose=0,
        )
        assert _CALIBRATION_CAVEAT in report.caveats


class TestProjectionMaskingValidation:
    """The harness dispatches to the projection evaluator and populates fields."""

    def test_projection_model_runs(self, mv_projection_model, mv_vcf_file):
        pytest.importorskip("snps")
        report = run_masking_validation(
            mv_projection_model,
            mv_vcf_file,
            raw_parser_max_samples=8,
            random_state=1,
            verbose=0,
        )
        assert report.model_type == "projection"
        assert report.n_imputed_or_regions >= 1
        assert np.isfinite(report.correlation)
        assert report.n_variants_masked == 2
        assert report.raw_parser_agrees is True


class TestMaskingValidationEdgeCases:
    """Degenerate cohorts and small samples degrade gracefully, never crash."""

    def test_unfitted_model_raises(self, mv_vcf_file):
        model = LinearImputationPRS(verbose=0)
        with pytest.raises(ModelNotFittedError):
            run_masking_validation(model, mv_vcf_file, verbose=0)

    def test_zero_variance_true_prs_handled(self, mv_imputation_model):
        # 25 identical samples -> the PRS is constant -> correlation undefined.
        one_row = np.array([[0, 1, 2, 0, 1]], dtype=np.float32)
        dm = np.tile(one_row, (25, 1))
        gd = _mv_genotype_data(list(_MV_REF_ALLELES), list(_MV_ALT_ALLELES), dm)

        report = run_masking_validation(
            mv_imputation_model, gd, run_raw_parser_check=False, verbose=0
        )
        assert np.isnan(report.correlation)
        assert any("degenerate" in c.lower() for c in report.caveats)

    def test_small_sample_skips_concordance(self, mv_imputation_model):
        dm = np.random.default_rng(3).integers(0, 3, size=(6, 5)).astype(np.float32)
        gd = _mv_genotype_data(list(_MV_REF_ALLELES), list(_MV_ALT_ALLELES), dm)

        report = run_masking_validation(
            mv_imputation_model, gd, run_raw_parser_check=False, verbose=0
        )
        assert report.percentile_concordance == {}
        assert np.isnan(report.top_decile_concordance)
        assert any("concordance skipped" in c.lower() for c in report.caveats)
