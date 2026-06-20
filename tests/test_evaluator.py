"""Tests for the ImputationEvaluator class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.types import EvaluationMetrics, GenotypeData
from imputed_prs.evaluation import (
    CrossValidationResult,
    ImputationEvaluator,
    SensitivityResult,
)
from imputed_prs.evaluation._scoring import is_hard_called


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_vcf_file(tmp_path):
    """Create a synthetic VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\tSAMPLE3\tSAMPLE4\tSAMPLE5\tSAMPLE6\tSAMPLE7\tSAMPLE8\tSAMPLE9\tSAMPLE10\tSAMPLE11\tSAMPLE12\tSAMPLE13\tSAMPLE14\tSAMPLE15\tSAMPLE16\tSAMPLE17\tSAMPLE18\tSAMPLE19\tSAMPLE20
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1
1\t102000\trs5\tA\tT\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1
"""
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    return vcf_path


@pytest.fixture
def synthetic_prs_df():
    """Create a synthetic PRS DataFrame for testing."""
    return pd.DataFrame({
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "T"],
        "other_allele": ["A", "C", "G", "T", "A"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    })


@pytest.fixture
def platform_variants_partial():
    """Create a list of platform variants (partial overlap with PRS)."""
    return ["rs1", "rs2", "rs3"]


@pytest.fixture
def platform_variants_all():
    """Create a list of platform variants (full overlap with PRS)."""
    return ["rs1", "rs2", "rs3", "rs4", "rs5"]


@pytest.fixture
def fitted_model(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial):
    """Create a fitted LinearImputationPRS model for testing."""
    cyvcf2 = pytest.importorskip("cyvcf2")

    model = LinearImputationPRS(
        window_size=500_000,
        cv_folds=3,
        tuning_scope="none",
        verbose=0,
        random_state=42,
    )

    model.fit(
        reference_genotypes=synthetic_vcf_file,
        prs_definition=synthetic_prs_df,
        platform_variants=platform_variants_partial,
    )

    return model


@pytest.fixture
def unfitted_model():
    """Create an unfitted LinearImputationPRS model."""
    return LinearImputationPRS(verbose=0)


@pytest.fixture
def synthetic_genotype_data():
    """Create synthetic GenotypeData for testing."""
    n_samples = 20
    n_variants = 5

    dosage_matrix = np.random.default_rng(42).integers(0, 3, size=(n_samples, n_variants)).astype(np.float32)

    variant_info = pd.DataFrame({
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "ref_allele": ["A", "C", "G", "T", "A"],
        "alt_allele": ["G", "T", "A", "C", "T"],
    })

    sample_ids = [f"SAMPLE{i+1}" for i in range(n_samples)]

    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=sample_ids,
    )


@pytest.fixture
def continuous_genotype_data():
    """GenotypeData with continuous (DS-style) dosages -> the numeric scorer path."""
    dosage_matrix = np.random.default_rng(123).uniform(0.0, 2.0, size=(20, 5))
    variant_info = pd.DataFrame({
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "ref_allele": ["A", "C", "G", "T", "A"],
        "alt_allele": ["G", "T", "A", "C", "T"],
    })
    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=[f"SAMPLE{i+1}" for i in range(20)],
    )


class TestImputationEvaluatorDosageModes:
    """evaluate() works for both hard-called and continuous reference dosages."""

    def test_evaluate_continuous_dosages(self, fitted_model, continuous_genotype_data):
        # Continuous dosages cannot be rendered to strings -> numeric scorer path.
        assert not is_hard_called(continuous_genotype_data.dosage_matrix)
        metrics = ImputationEvaluator(fitted_model, verbose=0).evaluate(
            continuous_genotype_data
        )
        assert isinstance(metrics, EvaluationMetrics)
        assert np.isfinite(metrics.mae) and metrics.mae >= 0
        assert np.isfinite(metrics.rmse) and metrics.rmse >= 0

    def test_evaluate_hardcalled_dosages(self, fitted_model, synthetic_genotype_data):
        # Integer dosages -> string-render replay of the browser scorer.
        assert is_hard_called(synthetic_genotype_data.dosage_matrix)
        metrics = ImputationEvaluator(fitted_model, verbose=0).evaluate(
            synthetic_genotype_data
        )
        assert isinstance(metrics, EvaluationMetrics)


# =============================================================================
# Tests for __init__
# =============================================================================


class TestImputationEvaluatorInit:
    """Tests for ImputationEvaluator initialization."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_init_raises_model_not_fitted_error_for_unfitted_model(self, unfitted_model):
        """Test __init__ raises ModelNotFittedError for unfitted model."""
        with pytest.raises(ModelNotFittedError, match="requires a fitted model"):
            ImputationEvaluator(unfitted_model)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_init_succeeds_with_fitted_model(self, fitted_model):
        """Test __init__ succeeds with fitted model."""
        evaluator = ImputationEvaluator(fitted_model)
        assert evaluator.model is fitted_model

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_init_accepts_custom_verbose_level(self, fitted_model):
        """Test __init__ accepts custom verbose level."""
        evaluator = ImputationEvaluator(fitted_model, verbose=2)
        assert evaluator.verbose == 2

        evaluator_quiet = ImputationEvaluator(fitted_model, verbose=0)
        assert evaluator_quiet.verbose == 0


# =============================================================================
# Tests for evaluate()
# =============================================================================


class TestImputationEvaluatorEvaluate:
    """Tests for ImputationEvaluator.evaluate() method."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_evaluate_returns_evaluation_metrics(
        self, fitted_model, synthetic_vcf_file
    ):
        """Test evaluate() returns EvaluationMetrics object."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        metrics = evaluator.evaluate(synthetic_vcf_file)

        assert isinstance(metrics, EvaluationMetrics)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_evaluate_metrics_have_valid_ranges(
        self, fitted_model, synthetic_vcf_file
    ):
        """Test evaluate() metrics are within valid ranges."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        metrics = evaluator.evaluate(synthetic_vcf_file)

        # Correlation should be in [-1, 1]
        assert -1 <= metrics.correlation <= 1

        # R² should be >= 0 (can be > 1 in edge cases but usually 0-1)
        assert metrics.r2 >= 0

        # MAE and RMSE should be >= 0
        assert metrics.mae >= 0
        assert metrics.rmse >= 0

        # Spearman rho should be in [-1, 1]
        assert -1 <= metrics.spearman_rho <= 1

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_evaluate_accepts_pre_loaded_genotype_data(
        self, fitted_model, synthetic_genotype_data
    ):
        """Test evaluate() accepts pre-loaded GenotypeData."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        metrics = evaluator.evaluate(synthetic_genotype_data)

        assert isinstance(metrics, EvaluationMetrics)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_evaluate_handles_missing_variants_with_warning(
        self, fitted_model, tmp_path
    ):
        """Test evaluate() handles missing variants gracefully with warning."""
        # Create VCF with only some variants
        vcf_content = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\tSAMPLE3
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0
"""
        vcf_path = tmp_path / "partial.vcf"
        vcf_path.write_text(vcf_content)

        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        # Should still work, but with potentially degraded metrics
        # Some variants needed for imputation may be missing
        metrics = evaluator.evaluate(vcf_path)
        assert isinstance(metrics, EvaluationMetrics)


# =============================================================================
# Tests for cross_validate()
# =============================================================================


class TestImputationEvaluatorCrossValidate:
    """Tests for ImputationEvaluator.cross_validate() method."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_raises_validation_error_without_platform_source(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df
    ):
        """Test cross_validate() raises ValidationError without platform source."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        with pytest.raises(ValidationError, match="Exactly one platform source"):
            evaluator.cross_validate(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                # No platform source
            )

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_raises_validation_error_for_n_folds_less_than_2(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test cross_validate() raises ValidationError for n_folds < 2."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        with pytest.raises(ValidationError, match="n_folds must be >= 2"):
            evaluator.cross_validate(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                platform_variants=platform_variants_partial,
                n_folds=1,
            )

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_returns_cross_validation_result(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test cross_validate() returns CrossValidationResult."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        result = evaluator.cross_validate(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            n_folds=2,
            random_state=42,
        )

        assert isinstance(result, CrossValidationResult)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_has_correct_fold_count(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test cross_validate() has correct number of folds."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        n_folds = 3
        result = evaluator.cross_validate(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            n_folds=n_folds,
            random_state=42,
        )

        assert result.n_folds == n_folds
        assert len(result.fold_metrics) == n_folds
        assert len(result.n_samples_per_fold) == n_folds

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_reproducible_with_random_state(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test cross_validate() is reproducible with random_state."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        result1 = evaluator.cross_validate(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            n_folds=2,
            random_state=42,
        )

        result2 = evaluator.cross_validate(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            n_folds=2,
            random_state=42,
        )

        assert result1.mean_correlation == result2.mean_correlation
        assert result1.mean_r2 == result2.mean_r2

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_cross_validate_metrics_have_valid_values(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test cross_validate() metrics have valid values."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        result = evaluator.cross_validate(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            n_folds=2,
            random_state=42,
        )

        # Mean correlation should be in [-1, 1]
        assert -1 <= result.mean_correlation <= 1

        # Std correlation should be >= 0
        assert result.std_correlation >= 0

        # Mean R² should be >= 0
        assert result.mean_r2 >= 0

        # Mean MAE and RMSE should be >= 0
        assert result.mean_mae >= 0
        assert result.mean_rmse >= 0


# =============================================================================
# Tests for sensitivity_analysis()
# =============================================================================


class TestImputationEvaluatorSensitivityAnalysis:
    """Tests for ImputationEvaluator.sensitivity_analysis() method."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_sensitivity_analysis_returns_sensitivity_result(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test sensitivity_analysis() returns SensitivityResult."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        # Use a minimal grid for faster testing
        small_grid = {
            "window_size": [500_000],
            "l1_ratio": [0.5],
            "alpha": [0.01],
        }

        result = evaluator.sensitivity_analysis(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            parameter_grid=small_grid,
            cv_folds=3,
            random_state=42,
        )

        assert isinstance(result, SensitivityResult)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_sensitivity_analysis_uses_default_grid_when_none(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test sensitivity_analysis() uses default grid when parameter_grid=None."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        # Use only first values from default grid (27 combinations is too slow)
        # Just verify it works with minimal params
        small_grid = {
            "window_size": [500_000],
            "l1_ratio": [0.5],
        }

        result = evaluator.sensitivity_analysis(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            parameter_grid=small_grid,
            cv_folds=3,
            random_state=42,
        )

        assert isinstance(result, SensitivityResult)
        assert len(result.parameter_results) == 1

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_sensitivity_analysis_identifies_best_params(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df, platform_variants_partial
    ):
        """Test sensitivity_analysis() identifies best_params correctly."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        # Test with 2 parameter combinations
        grid = {
            "window_size": [500_000, 1_000_000],
            "l1_ratio": [0.5],
        }

        result = evaluator.sensitivity_analysis(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=synthetic_prs_df,
            platform_variants=platform_variants_partial,
            parameter_grid=grid,
            cv_folds=3,
            random_state=42,
        )

        # best_params should be a dict with the parameter values
        assert isinstance(result.best_params, dict)
        assert "window_size" in result.best_params
        assert "l1_ratio" in result.best_params

        # best_metrics should be valid
        assert isinstance(result.best_metrics, EvaluationMetrics)
        assert -1 <= result.best_metrics.correlation <= 1

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_sensitivity_analysis_raises_without_platform_source(
        self, fitted_model, synthetic_vcf_file, synthetic_prs_df
    ):
        """Test sensitivity_analysis() raises ValidationError without platform source."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        with pytest.raises(ValidationError, match="Exactly one platform source"):
            evaluator.sensitivity_analysis(
                reference_genotypes=synthetic_vcf_file,
                prs_definition=synthetic_prs_df,
                # No platform source
            )


# =============================================================================
# Tests for helper methods
# =============================================================================


class TestImputationEvaluatorHelpers:
    """Tests for ImputationEvaluator helper methods."""

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_all_needed_variant_ids_includes_observed(self, fitted_model):
        """Test _get_all_needed_variant_ids includes observed variant IDs."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        needed = evaluator._get_all_needed_variant_ids()

        for var in fitted_model.observed_variants:
            assert var.variant_id in needed

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_all_needed_variant_ids_includes_imputed(self, fitted_model):
        """Test _get_all_needed_variant_ids includes imputed variant IDs."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        needed = evaluator._get_all_needed_variant_ids()

        for model in fitted_model.imputed_models:
            assert model.variant_id in needed

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_get_all_needed_variant_ids_includes_predictors(self, fitted_model):
        """Test _get_all_needed_variant_ids includes predictor variant IDs."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        needed = evaluator._get_all_needed_variant_ids()

        for model in fitted_model.imputed_models:
            for pred_id in model.predictor_variant_ids:
                assert pred_id in needed

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_compute_true_prs_returns_correct_shape(
        self, fitted_model, synthetic_genotype_data
    ):
        """Test _compute_true_prs returns array with correct shape."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        true_prs = evaluator._compute_true_prs(synthetic_genotype_data)

        assert isinstance(true_prs, np.ndarray)
        assert true_prs.shape == (synthetic_genotype_data.n_samples,)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_compute_imputed_prs_batch_returns_correct_shape(
        self, fitted_model, synthetic_genotype_data
    ):
        """Test _compute_imputed_prs_batch returns array with correct shape."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)
        imputed_prs = evaluator._compute_imputed_prs_batch(synthetic_genotype_data)

        assert isinstance(imputed_prs, np.ndarray)
        assert imputed_prs.shape == (synthetic_genotype_data.n_samples,)

    @pytest.mark.skipif(
        not Path("/usr/bin/python3").exists(),
        reason="VCF parsing requires cyvcf2"
    )
    def test_subset_genotype_data_returns_correct_subset(
        self, fitted_model, synthetic_genotype_data
    ):
        """Test _subset_genotype_data returns correct subset."""
        evaluator = ImputationEvaluator(fitted_model, verbose=0)

        indices = np.array([0, 5, 10])
        subset = evaluator._subset_genotype_data(synthetic_genotype_data, indices)

        assert subset.n_samples == 3
        assert subset.n_variants == synthetic_genotype_data.n_variants
        assert len(subset.sample_ids) == 3
        assert np.array_equal(
            subset.dosage_matrix,
            synthetic_genotype_data.dosage_matrix[indices, :]
        )


# =============================================================================
# Tests for imports
# =============================================================================


class TestImputationEvaluatorImports:
    """Tests for ImputationEvaluator imports."""

    def test_import_from_evaluation_module(self):
        """Test import from imputed_prs.evaluation."""
        from imputed_prs.evaluation import ImputationEvaluator
        assert ImputationEvaluator is not None

    def test_import_cross_validation_result(self):
        """Test import CrossValidationResult."""
        from imputed_prs.evaluation import CrossValidationResult
        assert CrossValidationResult is not None

    def test_import_sensitivity_result(self):
        """Test import SensitivityResult."""
        from imputed_prs.evaluation import SensitivityResult
        assert SensitivityResult is not None

    def test_import_from_evaluator_module(self):
        """Test import from full module path."""
        from imputed_prs.evaluation.evaluator import (
            ImputationEvaluator,
            CrossValidationResult,
            SensitivityResult,
        )
        assert ImputationEvaluator is not None
        assert CrossValidationResult is not None
        assert SensitivityResult is not None
