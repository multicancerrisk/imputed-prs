"""Tests for the predictor module."""

import numpy as np

from imputed_prs.core.types import ImputedVariantModel, VariantInfo
from imputed_prs.models.predictor import compute_imputed_prs, compute_observed_prs


class TestComputeObservedPrs:
    """Tests for compute_observed_prs function."""

    def test_basic_calculation(self):
        """Basic calculation with known values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 + 1.0 * 0.2 = 0.2 + 0.2 = 0.4
        assert abs(prs - 0.4) < 1e-10
        assert n_used == 2

    def test_missing_variants_none_dosages(self):
        """Handle missing variants (None dosages)."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 2.0, "rs2": None}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 = 0.2 (rs2 is skipped)
        assert abs(prs - 0.2) < 1e-10
        assert n_used == 1

    def test_empty_observed_variants(self):
        """Empty observed variants list returns zero."""
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, [])

        assert prs == 0.0
        assert n_used == 0

    def test_mixed_valid_missing_dosages(self):
        """Mixed valid and missing dosages."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
            VariantInfo("rs3", "1", 300, "G", "A", 0.3),
            VariantInfo("rs4", "1", 400, "T", "C", 0.4),
        ]
        dosages = {
            "rs1": 1.0,
            "rs2": None,
            "rs3": 2.0,
            # rs4 not in dosages dict
        }

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 1.0 * 0.1 + 2.0 * 0.3 = 0.1 + 0.6 = 0.7
        assert abs(prs - 0.7) < 1e-10
        assert n_used == 2

    def test_negative_betas(self):
        """Handle negative beta values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", -0.05),
        ]
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 + 1.0 * (-0.05) = 0.2 - 0.05 = 0.15
        assert abs(prs - 0.15) < 1e-10
        assert n_used == 2

    def test_zero_dosage(self):
        """Handle zero dosage values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.5),
            VariantInfo("rs2", "1", 200, "C", "T", 0.3),
        ]
        dosages = {"rs1": 0.0, "rs2": 2.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 0.0 * 0.5 + 2.0 * 0.3 = 0.0 + 0.6 = 0.6
        assert abs(prs - 0.6) < 1e-10
        assert n_used == 2

    def test_all_missing_dosages(self):
        """All variants have missing dosages."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": None, "rs2": None}

        prs, n_used = compute_observed_prs(dosages, observed)

        assert prs == 0.0
        assert n_used == 0

    def test_empty_dosages_dict(self):
        """Empty dosages dictionary."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
        ]
        dosages = {}

        prs, n_used = compute_observed_prs(dosages, observed)

        assert prs == 0.0
        assert n_used == 0

    def test_fractional_dosages(self):
        """Handle fractional dosage values (imputed-like)."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 1.5, "rs2": 0.7}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 1.5 * 0.1 + 0.7 * 0.2 = 0.15 + 0.14 = 0.29
        assert abs(prs - 0.29) < 1e-10
        assert n_used == 2


class TestComputeImputedPrs:
    """Tests for compute_imputed_prs function."""

    def test_basic_calculation(self):
        """Basic calculation with known coefficients and intercept."""
        model = ImputedVariantModel(
            variant_id="rs3",
            chromosome="1",
            position=300,
            effect_allele="A",
            other_allele="G",
            beta=0.05,
            allele_frequency=0.3,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.5,
            predictor_variant_ids=["rs1", "rs2"],
            coefficients=np.array([0.3, 0.2]),
            is_intercept_only=False,
        )

        user_dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # raw = 2*0.3 + 1*0.2 + 0.5 = 1.3 (within bounds, no clipping)
        # PRS = 1.3 * 0.05 = 0.065
        expected_raw = 2 * 0.3 + 1 * 0.2 + 0.5
        expected_prs = expected_raw * 0.05
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1
        assert n_truncated == 0
        # Variance should be approximately beta^2 * residual_variance
        # (slightly adjusted due to truncation, but raw is well within bounds)
        assert variance > 0

    def test_intercept_only_model(self):
        """Intercept-only model uses intercept as prediction."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.4,
            imputation_r2=0.0,
            residual_variance=0.2,
            intercept=0.8,  # 2 * AF = 2 * 0.4 = 0.8
            predictor_variant_ids=[],
            coefficients=np.array([]),
            is_intercept_only=True,
        )

        user_dosages = {"rs2": 1.0, "rs3": 2.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # PRS = 0.8 * 0.1 = 0.08
        expected_prs = 0.8 * 0.1
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1
        assert n_truncated == 0

    def test_missing_predictor_fallback_to_intercept(self):
        """Missing predictor dosages fall back to intercept-only."""
        model = ImputedVariantModel(
            variant_id="rs3",
            chromosome="1",
            position=300,
            effect_allele="A",
            other_allele="G",
            beta=0.05,
            allele_frequency=0.3,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.6,
            predictor_variant_ids=["rs1", "rs2"],
            coefficients=np.array([0.3, 0.2]),
            is_intercept_only=False,
        )

        # rs2 is missing
        user_dosages = {"rs1": 2.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # Falls back to intercept: PRS = 0.6 * 0.05 = 0.03
        expected_prs = 0.6 * 0.05
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1

    def test_dosage_clipping_upper_bound(self):
        """Dosage clipped at upper boundary triggers truncation count."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.9,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=1.0,
            predictor_variant_ids=["rs2"],
            coefficients=np.array([0.8]),
            is_intercept_only=False,
        )

        # raw = 2.0 * 0.8 + 1.0 = 2.6 > 2.0, should be clipped
        user_dosages = {"rs2": 2.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # Clipped to 2.0, PRS = 2.0 * 0.1 = 0.2
        expected_prs = 2.0 * 0.1
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1
        assert n_truncated == 1

    def test_dosage_clipping_lower_bound(self):
        """Dosage clipped at lower boundary triggers truncation count."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.1,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.5,
            predictor_variant_ids=["rs2"],
            coefficients=np.array([-0.4]),
            is_intercept_only=False,
        )

        # raw = 2.0 * (-0.4) + 0.5 = -0.3 < 0.0, should be clipped
        user_dosages = {"rs2": 2.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # Clipped to 0.0, PRS = 0.0 * 0.1 = 0.0
        assert abs(prs - 0.0) < 1e-10
        assert n_imputed == 1
        assert n_truncated == 1

    def test_empty_imputed_models(self):
        """Empty imputed models list returns zeros."""
        user_dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [])

        assert prs == 0.0
        assert variance == 0.0
        assert n_imputed == 0
        assert n_truncated == 0

    def test_multiple_models_mixed(self):
        """Multiple models with mixed intercept-only and full models."""
        model1 = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=0.1,
            allele_frequency=0.4,
            imputation_r2=0.0,
            residual_variance=0.2,
            intercept=0.8,
            predictor_variant_ids=[],
            coefficients=np.array([]),
            is_intercept_only=True,
        )
        model2 = ImputedVariantModel(
            variant_id="rs2",
            chromosome="1",
            position=200,
            effect_allele="C",
            other_allele="T",
            beta=0.2,
            allele_frequency=0.3,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.3,
            predictor_variant_ids=["rs3"],
            coefficients=np.array([0.5]),
            is_intercept_only=False,
        )

        user_dosages = {"rs3": 1.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(
            user_dosages, [model1, model2]
        )

        # model1: intercept-only, PRS = 0.8 * 0.1 = 0.08
        # model2: raw = 1.0 * 0.5 + 0.3 = 0.8, PRS = 0.8 * 0.2 = 0.16
        # Total PRS = 0.08 + 0.16 = 0.24
        expected_prs = 0.8 * 0.1 + 0.8 * 0.2
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 2
        assert n_truncated == 0
        assert variance > 0

    def test_variance_contribution(self):
        """Variance contribution is beta^2 * adjusted_residual_variance."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=0.5,
            allele_frequency=0.5,
            imputation_r2=0.8,
            residual_variance=0.04,  # sigma = 0.2
            intercept=1.0,  # Well within bounds
            predictor_variant_ids=[],
            coefficients=np.array([]),
            is_intercept_only=True,
        )

        user_dosages = {}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # Prediction is 1.0 (well within [0,2]), so variance adjustment is minimal
        # Variance ≈ beta^2 * residual_variance = 0.5^2 * 0.04 = 0.01
        assert variance > 0
        # Should be close to the unadjusted value since 1.0 is well within bounds
        expected_approx = 0.5**2 * 0.04
        assert abs(variance - expected_approx) < 0.005  # Allow small truncation adjustment

    def test_negative_betas(self):
        """Handle negative beta values correctly."""
        model = ImputedVariantModel(
            variant_id="rs1",
            chromosome="1",
            position=100,
            effect_allele="A",
            other_allele="G",
            beta=-0.1,
            allele_frequency=0.4,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.8,
            predictor_variant_ids=["rs2"],
            coefficients=np.array([0.2]),
            is_intercept_only=False,
        )

        user_dosages = {"rs2": 1.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # raw = 1.0 * 0.2 + 0.8 = 1.0
        # PRS = 1.0 * (-0.1) = -0.1
        expected_prs = 1.0 * (-0.1)
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1
        # Variance is still positive (beta^2 is always positive)
        assert variance > 0

    def test_predictor_none_dosage(self):
        """Predictor with None dosage triggers fallback to intercept."""
        model = ImputedVariantModel(
            variant_id="rs3",
            chromosome="1",
            position=300,
            effect_allele="A",
            other_allele="G",
            beta=0.05,
            allele_frequency=0.3,
            imputation_r2=0.8,
            residual_variance=0.1,
            intercept=0.6,
            predictor_variant_ids=["rs1", "rs2"],
            coefficients=np.array([0.3, 0.2]),
            is_intercept_only=False,
        )

        # rs1 has None dosage
        user_dosages = {"rs1": None, "rs2": 1.0}

        prs, variance, n_imputed, n_truncated = compute_imputed_prs(user_dosages, [model])

        # Falls back to intercept: PRS = 0.6 * 0.05 = 0.03
        expected_prs = 0.6 * 0.05
        assert abs(prs - expected_prs) < 1e-10
        assert n_imputed == 1
