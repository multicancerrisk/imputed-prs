"""Tests for the ProjectionEvaluator class."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearProjectionPRS
from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import EvaluationMetrics, GenotypeData
from imputed_prs.evaluation._scoring import is_hard_called
from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_vcf_file(tmp_path):
    """Create a synthetic VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\tSAMPLE3\tSAMPLE4\tSAMPLE5\tSAMPLE6\tSAMPLE7\tSAMPLE8\tSAMPLE9\tSAMPLE10
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0
1\t102000\trs5\tA\tT\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1
"""
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    return vcf_path


@pytest.fixture
def synthetic_prs_df():
    """Create a synthetic PRS DataFrame."""
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
    """Platform variants with partial overlap."""
    return ["rs1", "rs2", "rs3"]


@pytest.fixture
def fitted_model(synthetic_vcf_file, synthetic_prs_df, platform_variants_partial):
    """Create a fitted LinearProjectionPRS model."""
    cyvcf2 = pytest.importorskip("cyvcf2")

    model = LinearProjectionPRS(
        window_size=500_000,
        cv_folds=3,
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
def synthetic_genotype_data():
    """Create synthetic GenotypeData for evaluation."""
    rng = np.random.default_rng(42)
    n_samples = 20
    n_variants = 5

    dosage_matrix = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_variants))
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


class TestProjectionEvaluatorDosageModes:
    """evaluate() works for both hard-called and continuous reference dosages."""

    def test_evaluate_continuous_dosages(self, fitted_model, continuous_genotype_data):
        # Continuous dosages cannot be rendered to strings -> numeric scorer path.
        assert not is_hard_called(continuous_genotype_data.dosage_matrix)
        metrics = ProjectionEvaluator(fitted_model, verbose=0).evaluate(
            continuous_genotype_data
        )
        assert isinstance(metrics, EvaluationMetrics)
        assert np.isfinite(metrics.mae) and metrics.mae >= 0
        assert np.isfinite(metrics.rmse) and metrics.rmse >= 0

    def test_evaluate_hardcalled_dosages(self, fitted_model, synthetic_genotype_data):
        # Integer dosages -> string-render replay of the browser scorer.
        assert is_hard_called(synthetic_genotype_data.dosage_matrix)
        metrics = ProjectionEvaluator(fitted_model, verbose=0).evaluate(
            synthetic_genotype_data
        )
        assert isinstance(metrics, EvaluationMetrics)


# =============================================================================
# Tests
# =============================================================================


class TestProjectionEvaluator:
    """Tests for the ProjectionEvaluator class."""

    def test_evaluate_returns_evaluation_metrics(self, fitted_model, synthetic_genotype_data):
        """evaluate() returns EvaluationMetrics type."""
        evaluator = ProjectionEvaluator(fitted_model, verbose=0)
        metrics = evaluator.evaluate(synthetic_genotype_data)
        assert isinstance(metrics, EvaluationMetrics)

    def test_evaluate_correlation_positive(self, fitted_model, synthetic_vcf_file):
        """With well-correlated synthetic data, correlation is reasonable."""
        evaluator = ProjectionEvaluator(fitted_model, verbose=0)
        metrics = evaluator.evaluate(synthetic_vcf_file)
        # With synthetic data, correlation should be positive
        # (model was trained on same VCF, so in-sample evaluation)
        assert metrics.correlation > 0.0

    def test_true_prs_computation(self, fitted_model, synthetic_genotype_data):
        """_compute_true_prs matches manual dot-product calculation."""
        evaluator = ProjectionEvaluator(fitted_model, verbose=0)
        true_prs = evaluator._compute_true_prs(synthetic_genotype_data)

        assert true_prs.shape == (synthetic_genotype_data.n_samples,)

        # Manual computation for sample 0
        betas = {"rs1": 0.1, "rs2": -0.05, "rs3": 0.2, "rs4": 0.15, "rs5": -0.1}
        variant_ids = list(synthetic_genotype_data.variant_info["variant_id"])
        expected = 0.0
        for j, var_id in enumerate(variant_ids):
            dosage = synthetic_genotype_data.dosage_matrix[0, j]
            if not np.isnan(dosage) and var_id in betas:
                expected += dosage * betas[var_id]

        assert true_prs[0] == pytest.approx(expected, abs=1e-6)

    def test_projected_prs_batch_shape(self, fitted_model, synthetic_genotype_data):
        """_compute_projected_prs_batch returns array of shape (n_samples,)."""
        evaluator = ProjectionEvaluator(fitted_model, verbose=0)
        projected = evaluator._compute_projected_prs_batch(synthetic_genotype_data)
        assert projected.shape == (synthetic_genotype_data.n_samples,)

    def test_unfitted_model_raises_error(self):
        """Unfitted model raises ModelNotFittedError."""
        model = LinearProjectionPRS()
        with pytest.raises(ModelNotFittedError, match="requires a fitted model"):
            ProjectionEvaluator(model)

    def test_needed_variant_ids(self, fitted_model):
        """_get_all_needed_variant_ids includes observed + PRS + predictor IDs."""
        evaluator = ProjectionEvaluator(fitted_model, verbose=0)
        needed = evaluator._get_all_needed_variant_ids()

        # Should include observed variant IDs
        for var in fitted_model.observed_variants:
            assert var.variant_id in needed

        # Should include predictor IDs from region models
        for region_model in fitted_model.region_models:
            for pred_id in region_model.predictor_variant_ids:
                assert pred_id in needed
            # Should include PRS variant IDs from regions
            for var_id in region_model.prs_variant_ids:
                assert var_id in needed
