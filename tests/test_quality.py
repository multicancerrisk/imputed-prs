"""Tests for per-variant imputation quality summary."""

import numpy as np
import pytest

from imputed_prs.core.types import ImputedVariantModel
from imputed_prs.evaluation.quality import summarize_imputation_quality


def _make_model(
    imputation_r2: float,
    beta: float,
    is_intercept_only: bool = False,
) -> ImputedVariantModel:
    """Helper to create test ImputedVariantModel instances."""
    return ImputedVariantModel(
        variant_id="test",
        chromosome="1",
        position=100000,
        effect_allele="A",
        other_allele="G",
        beta=beta,
        allele_frequency=0.3,
        imputation_r2=imputation_r2,
        residual_variance=0.1,
        intercept=0.6,
        predictor_variant_ids=[],
        coefficients=np.array([]),
        is_intercept_only=is_intercept_only,
    )


class TestSummarizeImputationQuality:
    def test_basic_summary(self):
        """Test with mixed quality models."""
        models = {
            "rs1": _make_model(imputation_r2=0.95, beta=0.1),   # Excellent
            "rs2": _make_model(imputation_r2=0.70, beta=0.05),  # Good
            "rs3": _make_model(imputation_r2=0.50, beta=0.08),  # Moderate
            "rs4": _make_model(imputation_r2=0.10, beta=0.02),  # Poor
        }

        summary = summarize_imputation_quality(models)

        assert summary["n_total"] == 4
        assert summary["n_excellent"] == 1
        assert summary["n_good"] == 1
        assert summary["n_moderate"] == 1
        assert summary["n_poor"] == 1
        assert 0 < summary["mean_r2"] < 1

    def test_intercept_only_count(self):
        """Test counting of intercept-only models."""
        models = {
            "rs1": _make_model(imputation_r2=0.0, beta=0.1, is_intercept_only=True),
            "rs2": _make_model(imputation_r2=0.8, beta=0.05, is_intercept_only=False),
            "rs3": _make_model(imputation_r2=0.0, beta=0.08, is_intercept_only=True),
        }

        summary = summarize_imputation_quality(models)

        assert summary["n_intercept_only"] == 2

    def test_weighted_mean_r2(self):
        """Test that weighted mean uses |beta| as weights."""
        models = {
            "rs1": _make_model(imputation_r2=0.9, beta=0.1),   # High r2, high weight
            "rs2": _make_model(imputation_r2=0.1, beta=0.01),  # Low r2, low weight
        }

        summary = summarize_imputation_quality(models)

        # Weighted mean should be closer to 0.9 due to higher beta weight
        assert summary["weighted_mean_r2"] > summary["mean_r2"]

    def test_empty_models_error(self):
        """Test error when no models provided."""
        with pytest.raises(ValueError, match="cannot be empty"):
            summarize_imputation_quality({})

    def test_all_excellent(self):
        """Test when all models are excellent quality."""
        models = {
            f"rs{i}": _make_model(imputation_r2=0.9 + i * 0.01, beta=0.05)
            for i in range(5)
        }

        summary = summarize_imputation_quality(models)

        assert summary["n_excellent"] == 5
        assert summary["n_good"] == 0
        assert summary["n_moderate"] == 0
        assert summary["n_poor"] == 0

    def test_single_model(self):
        """Test with single model."""
        models = {"rs1": _make_model(imputation_r2=0.75, beta=0.1)}

        summary = summarize_imputation_quality(models)

        assert summary["n_total"] == 1
        assert summary["mean_r2"] == 0.75
        assert summary["median_r2"] == 0.75
        assert summary["min_r2"] == 0.75
        assert summary["max_r2"] == 0.75

    def test_negative_beta_uses_absolute_value(self):
        """Test that negative betas are converted to absolute value for weighting."""
        models = {
            "rs1": _make_model(imputation_r2=0.9, beta=-0.1),  # Negative beta
            "rs2": _make_model(imputation_r2=0.1, beta=0.01),
        }

        summary = summarize_imputation_quality(models)

        # Should still weight by |beta|, so weighted mean should be closer to 0.9
        assert summary["weighted_mean_r2"] > summary["mean_r2"]

    def test_zero_betas_fallback(self):
        """Test that zero betas fall back to unweighted mean."""
        models = {
            "rs1": _make_model(imputation_r2=0.9, beta=0.0),
            "rs2": _make_model(imputation_r2=0.1, beta=0.0),
        }

        summary = summarize_imputation_quality(models)

        # With all zero weights, should fall back to unweighted mean
        assert summary["weighted_mean_r2"] == summary["mean_r2"]
        assert summary["weighted_mean_r2"] == pytest.approx(0.5)

    def test_boundary_values(self):
        """Test models exactly on tier boundaries."""
        models = {
            "rs1": _make_model(imputation_r2=0.8, beta=0.1),  # On boundary - good
            "rs2": _make_model(imputation_r2=0.6, beta=0.1),  # On boundary - moderate
            "rs3": _make_model(imputation_r2=0.4, beta=0.1),  # On boundary - poor
        }

        summary = summarize_imputation_quality(models)

        # 0.8 is <= 0.8, so not excellent (> 0.8)
        assert summary["n_excellent"] == 0
        # 0.8 is in (0.6, 0.8], so good
        assert summary["n_good"] == 1
        # 0.6 is in (0.4, 0.6], so moderate
        assert summary["n_moderate"] == 1
        # 0.4 is <= 0.4, so poor
        assert summary["n_poor"] == 1

    def test_negative_r2_values(self):
        """Test that negative R² values (possible from CV) are handled."""
        models = {
            "rs1": _make_model(imputation_r2=-0.1, beta=0.1),  # Negative R²
            "rs2": _make_model(imputation_r2=0.5, beta=0.1),
        }

        summary = summarize_imputation_quality(models)

        assert summary["n_poor"] == 1  # -0.1 is <= 0.4
        assert summary["min_r2"] == pytest.approx(-0.1)
        assert summary["mean_r2"] == pytest.approx(0.2)
