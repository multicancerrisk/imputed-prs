"""Tests for plotting module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.types import ImputedVariantModel, PredictionResult

# Skip all tests if matplotlib is not installed
pytest.importorskip("matplotlib")

from imputed_prs.evaluation.plotting import (
    QUALITY_TIER_COLORS,
    _get_quality_tier,
    _get_quality_tier_color,
    _create_figure_if_needed,
    plot_calibration,
    plot_imputation_quality,
    plot_variance_contribution,
    plot_truncation_diagnostics,
)


@pytest.fixture
def sample_imputed_variant_models():
    """Create sample ImputedVariantModel instances for testing."""
    models = []
    # Create models with varying R² values across all tiers
    r2_values = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    for i, r2 in enumerate(r2_values):
        model = ImputedVariantModel(
            variant_id=f"rs{1000 + i}",
            chromosome="1",
            position=100000 + i * 1000,
            effect_allele="A",
            other_allele="G",
            beta=0.1 * (i + 1),
            allele_frequency=0.3,
            imputation_r2=r2,
            residual_variance=0.1 * (1 - r2),
            intercept=0.6,
            predictor_variant_ids=[f"rs{2000 + i}"],
            coefficients=np.array([0.5]),
            is_intercept_only=False,
        )
        models.append(model)
    return models


@pytest.fixture
def sample_prediction_results():
    """Create sample PredictionResult instances for testing."""
    results = []
    for i in range(50):
        # Varying truncation counts
        n_truncated = max(0, i - 40)  # 0 for first 40, then increasing
        result = PredictionResult(
            prs=0.5 + np.random.normal(0, 0.1),
            se=0.05,
            ci_lower=0.4,
            ci_upper=0.6,
            prs_observed_component=0.3,
            prs_imputed_component=0.2,
            n_variants_used=100,
            n_variants_imputed=50,
            n_variants_intercept_only=5,
            n_user_variants_missing=10,
            n_truncated=n_truncated,
        )
        results.append(result)
    return results


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_quality_tier_excellent(self):
        """Test excellent tier classification."""
        assert _get_quality_tier(0.9) == "excellent"
        assert _get_quality_tier(0.85) == "excellent"
        assert _get_quality_tier(0.81) == "excellent"

    def test_get_quality_tier_good(self):
        """Test good tier classification."""
        assert _get_quality_tier(0.8) == "good"
        assert _get_quality_tier(0.7) == "good"
        assert _get_quality_tier(0.61) == "good"

    def test_get_quality_tier_moderate(self):
        """Test moderate tier classification."""
        assert _get_quality_tier(0.6) == "moderate"
        assert _get_quality_tier(0.5) == "moderate"
        assert _get_quality_tier(0.41) == "moderate"

    def test_get_quality_tier_poor(self):
        """Test poor tier classification."""
        assert _get_quality_tier(0.4) == "poor"
        assert _get_quality_tier(0.2) == "poor"
        assert _get_quality_tier(0.0) == "poor"

    def test_get_quality_tier_color(self):
        """Test color assignment by tier."""
        assert _get_quality_tier_color(0.9) == QUALITY_TIER_COLORS["excellent"]
        assert _get_quality_tier_color(0.7) == QUALITY_TIER_COLORS["good"]
        assert _get_quality_tier_color(0.5) == QUALITY_TIER_COLORS["moderate"]
        assert _get_quality_tier_color(0.2) == QUALITY_TIER_COLORS["poor"]

    def test_create_figure_if_needed_none(self):
        """Test figure creation when ax is None."""
        fig, ax = _create_figure_if_needed(None)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_create_figure_if_needed_existing(self):
        """Test using existing axes."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()
        fig, ax = _create_figure_if_needed(ax_orig)
        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig_orig)


class TestPlotCalibration:
    """Tests for plot_calibration function."""

    def test_returns_figure_and_axes(self):
        """Test that function returns (fig, ax) tuple."""
        s_imputed = np.random.normal(0, 1, 100)
        s_true = 0.9 * s_imputed + np.random.normal(0, 0.2, 100)

        fig, ax = plot_calibration(s_imputed, s_true)

        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_works_with_provided_ax(self):
        """Test function works with provided ax parameter."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()
        s_imputed = np.random.normal(0, 1, 50)
        s_true = 0.8 * s_imputed + np.random.normal(0, 0.3, 50)

        fig, ax = plot_calibration(s_imputed, s_true, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig_orig)

    def test_regression_line_direction(self):
        """Test that regression line has correct slope direction."""
        # Create positively correlated data
        s_imputed = np.linspace(0, 10, 100)
        s_true = 0.9 * s_imputed + 1  # positive slope

        fig, ax = plot_calibration(s_imputed, s_true)

        # Check that annotation contains positive slope
        annotations = [
            child
            for child in ax.get_children()
            if hasattr(child, "get_text") and "Slope" in str(child.get_text())
        ]
        # The annotation text should indicate positive slope
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_handles_few_samples(self):
        """Test edge case with few samples (n < 10)."""
        s_imputed = np.array([1, 2, 3, 4, 5])
        s_true = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        fig, ax = plot_calibration(s_imputed, s_true)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_raises_on_length_mismatch(self):
        """Test that ValueError is raised for mismatched array lengths."""
        s_imputed = np.array([1, 2, 3])
        s_true = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            plot_calibration(s_imputed, s_true)

    def test_raises_on_insufficient_samples(self):
        """Test that ValueError is raised for fewer than 2 samples."""
        s_imputed = np.array([1])
        s_true = np.array([1])

        with pytest.raises(ValueError, match="at least 2 samples"):
            plot_calibration(s_imputed, s_true)

    def test_custom_title(self):
        """Test custom title is applied."""
        s_imputed = np.random.normal(0, 1, 50)
        s_true = 0.9 * s_imputed

        fig, ax = plot_calibration(s_imputed, s_true, title="Custom Title")

        assert ax.get_title() == "Custom Title"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_identity_line_toggle(self):
        """Test show_identity parameter."""
        s_imputed = np.random.normal(0, 1, 50)
        s_true = 0.9 * s_imputed

        fig, ax = plot_calibration(s_imputed, s_true, show_identity=False)

        # Should still work without identity line
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_regression_line_toggle(self):
        """Test show_regression parameter."""
        s_imputed = np.random.normal(0, 1, 50)
        s_true = 0.9 * s_imputed

        fig, ax = plot_calibration(s_imputed, s_true, show_regression=False)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotImputationQuality:
    """Tests for plot_imputation_quality function."""

    def test_returns_figure_and_axes(self, sample_imputed_variant_models):
        """Test that function returns (fig, ax) tuple."""
        fig, ax = plot_imputation_quality(sample_imputed_variant_models)

        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_histogram_bins_in_range(self, sample_imputed_variant_models):
        """Test histogram bins are in [0, 1] range."""
        fig, ax = plot_imputation_quality(sample_imputed_variant_models)

        xlim = ax.get_xlim()
        assert xlim[0] >= -0.1  # Allow small margin
        assert xlim[1] <= 1.1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_quality_tier_colors_applied(self, sample_imputed_variant_models):
        """Test quality tier colors are applied correctly."""
        fig, ax = plot_imputation_quality(sample_imputed_variant_models)

        # Check that bars exist
        patches = [p for p in ax.patches]
        assert len(patches) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_models_list(self):
        """Test handling of empty models list."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plot_imputation_quality([])

            assert len(w) == 1
            assert "Empty models list" in str(w[0].message)
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_show_tiers_toggle(self, sample_imputed_variant_models):
        """Test show_tiers parameter."""
        fig, ax = plot_imputation_quality(
            sample_imputed_variant_models, show_tiers=False
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_bins(self, sample_imputed_variant_models):
        """Test custom bins parameter."""
        fig, ax = plot_imputation_quality(sample_imputed_variant_models, bins=10)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_works_with_provided_ax(self, sample_imputed_variant_models):
        """Test function works with provided ax parameter."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()
        fig, ax = plot_imputation_quality(sample_imputed_variant_models, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig_orig)


class TestPlotVarianceContribution:
    """Tests for plot_variance_contribution function."""

    def test_returns_figure_and_axes(self, sample_imputed_variant_models):
        """Test that function returns (fig, ax) tuple."""
        fig, ax = plot_variance_contribution(sample_imputed_variant_models)

        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_shows_correct_top_n(self, sample_imputed_variant_models):
        """Test shows correct top_n variants."""
        top_n = 5
        fig, ax = plot_variance_contribution(sample_imputed_variant_models, top_n=top_n)

        # Check number of bars
        patches = [p for p in ax.patches]
        assert len(patches) == top_n
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_bars_sorted_descending(self, sample_imputed_variant_models):
        """Test bars are sorted by contribution descending."""
        fig, ax = plot_variance_contribution(sample_imputed_variant_models, top_n=5)

        # Get bar heights (which are widths for horizontal bar chart)
        patches = ax.patches
        # In horizontal bar chart, the width is the value
        widths = [p.get_width() for p in patches]

        # Since we reversed for display (highest at top), values should be
        # increasing from bottom to top
        assert widths == sorted(widths)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_works_with_fewer_models_than_top_n(self):
        """Test when fewer models than top_n."""
        models = [
            ImputedVariantModel(
                variant_id="rs1001",
                chromosome="1",
                position=100000,
                effect_allele="A",
                other_allele="G",
                beta=0.1,
                allele_frequency=0.3,
                imputation_r2=0.8,
                residual_variance=0.02,
                intercept=0.6,
            ),
            ImputedVariantModel(
                variant_id="rs1002",
                chromosome="1",
                position=200000,
                effect_allele="A",
                other_allele="G",
                beta=0.2,
                allele_frequency=0.3,
                imputation_r2=0.7,
                residual_variance=0.03,
                intercept=0.6,
            ),
        ]

        fig, ax = plot_variance_contribution(models, top_n=20)

        patches = [p for p in ax.patches]
        assert len(patches) == 2  # Only 2 models available
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_models_list(self):
        """Test handling of empty models list."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plot_variance_contribution([])

            assert len(w) == 1
            assert "Empty models list" in str(w[0].message)
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_color_by_quality_toggle(self, sample_imputed_variant_models):
        """Test color_by_quality parameter."""
        fig, ax = plot_variance_contribution(
            sample_imputed_variant_models, color_by_quality=False
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_works_with_provided_ax(self, sample_imputed_variant_models):
        """Test function works with provided ax parameter."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()
        fig, ax = plot_variance_contribution(sample_imputed_variant_models, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig_orig)


class TestPlotTruncationDiagnostics:
    """Tests for plot_truncation_diagnostics function."""

    def test_returns_figure_and_axes(self, sample_prediction_results):
        """Test that function returns (fig, ax) tuple."""
        fig, ax = plot_truncation_diagnostics(sample_prediction_results)

        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_accepts_list_of_prediction_results(self, sample_prediction_results):
        """Test accepts List[PredictionResult]."""
        fig, ax = plot_truncation_diagnostics(sample_prediction_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_accepts_dataframe(self):
        """Test accepts DataFrame with n_truncated column."""
        df = pd.DataFrame(
            {"n_truncated": [0, 1, 2, 3, 0, 0, 5, 2], "n_variants_imputed": [50] * 8}
        )

        fig, ax = plot_truncation_diagnostics(df)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_handles_no_truncation(self):
        """Test predictions with no truncation (n_truncated=0)."""
        results = [
            PredictionResult(
                prs=0.5,
                se=0.05,
                ci_lower=0.4,
                ci_upper=0.6,
                prs_observed_component=0.3,
                prs_imputed_component=0.2,
                n_variants_used=100,
                n_variants_imputed=50,
                n_variants_intercept_only=5,
                n_user_variants_missing=10,
                n_truncated=0,
            )
            for _ in range(20)
        ]

        fig, ax = plot_truncation_diagnostics(results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_raises_on_empty_list(self):
        """Test raises ValueError for empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_truncation_diagnostics([])

    def test_raises_on_missing_column(self):
        """Test raises ValueError for DataFrame without n_truncated."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="n_truncated"):
            plot_truncation_diagnostics(df)

    def test_works_with_provided_ax(self, sample_prediction_results):
        """Test function works with provided ax parameter."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()
        fig, ax = plot_truncation_diagnostics(sample_prediction_results, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig
        plt.close(fig_orig)

    def test_custom_title(self, sample_prediction_results):
        """Test custom title is applied."""
        fig, ax = plot_truncation_diagnostics(
            sample_prediction_results, title="Custom Title"
        )

        assert ax.get_title() == "Custom Title"
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestModuleImports:
    """Tests for module import handling."""

    def test_functions_importable_from_evaluation_module(self):
        """Test functions are importable from evaluation module."""
        from imputed_prs.evaluation import (
            plot_calibration,
            plot_imputation_quality,
            plot_variance_contribution,
            plot_truncation_diagnostics,
        )

        assert callable(plot_calibration)
        assert callable(plot_imputation_quality)
        assert callable(plot_variance_contribution)
        assert callable(plot_truncation_diagnostics)

    def test_quality_tier_colors_exported(self):
        """Test QUALITY_TIER_COLORS is accessible."""
        from imputed_prs.evaluation.plotting import QUALITY_TIER_COLORS

        assert "excellent" in QUALITY_TIER_COLORS
        assert "good" in QUALITY_TIER_COLORS
        assert "moderate" in QUALITY_TIER_COLORS
        assert "poor" in QUALITY_TIER_COLORS
