"""Tests for the ProjectionEvaluator class."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearProjectionPRS
from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import (
    EvaluationMetrics,
    GenotypeData,
    ProjectionRegionModel,
)
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
def effect_ref_prs_df():
    """PRS where the missing variant rs5 has effect_allele == reference REF.

    rs5 is A>T in the reference; setting its effect allele to "A" (the REF) forces
    the true-PRS scorer to flip to (2 - alt_dosage). rs5 is absent from the platform
    (see platform_variants_partial) so it is scored through the region path.
    """
    return pd.DataFrame({
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "A"],
        "other_allele": ["A", "C", "G", "T", "T"],
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

    def test_true_prs_effect_ref_oriented_and_parity(
        self, synthetic_vcf_file, effect_ref_prs_df,
        platform_variants_partial, synthetic_genotype_data,
    ):
        """A missing variant with effect==REF is flipped to (2 - alt_dosage)*beta,
        matching a hand calculation and the imputation evaluator (P2.3 Done-when)."""
        pytest.importorskip("cyvcf2")
        from imputed_prs import LinearImputationPRS
        from imputed_prs.evaluation.evaluator import ImputationEvaluator

        fit_kwargs = dict(
            reference_genotypes=synthetic_vcf_file,
            prs_definition=effect_ref_prs_df,
            platform_variants=platform_variants_partial,
        )
        proj = LinearProjectionPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42
        )
        proj.fit(**fit_kwargs)
        imp = LinearImputationPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42
        )
        imp.fit(**fit_kwargs)

        proj_true = ProjectionEvaluator(proj, verbose=0)._compute_true_prs(
            synthetic_genotype_data
        )
        imp_true = ImputationEvaluator(imp, verbose=0)._compute_true_prs(
            synthetic_genotype_data
        )

        # rs5 (effect==REF) must be missing and scored through the region path.
        assert any("rs5" in rm.prs_variant_ids for rm in proj.region_models)
        # No variant was dropped, so a hand calc over all 5 is valid below.
        placed = {v.variant_id for v in proj.observed_variants}
        for rm in proj.region_models:
            placed.update(rm.prs_variant_ids)
        assert placed == set(effect_ref_prs_df["variant_id"])

        # Parity with the allele-correct imputation evaluator.
        np.testing.assert_allclose(proj_true, imp_true, rtol=0, atol=1e-12)

        # Hand calculation, orienting each variant (effect==REF -> 2 - alt_dosage).
        vinfo = synthetic_genotype_data.variant_info
        dm = synthetic_genotype_data.dosage_matrix
        pos_to_col = {int(p): j for j, p in enumerate(vinfo["position"])}
        expected = np.zeros(synthetic_genotype_data.n_samples)
        for _, r in effect_ref_prs_df.iterrows():
            j = pos_to_col[int(r["position"])]
            alt_dosage = dm[:, j]
            oriented = (
                2.0 - alt_dosage
                if r["effect_allele"] == vinfo.iloc[j]["ref_allele"]
                else alt_dosage
            )
            expected += oriented * r["beta"]
        np.testing.assert_allclose(proj_true, expected, rtol=0, atol=1e-12)

        # rs5's flip is observable (raw != flipped for some sample), so the old
        # un-oriented region path would have produced a different score.
        assert not np.allclose(dm[:, pos_to_col[102000]], 2.0 - dm[:, pos_to_col[102000]])

    def test_true_prs_multiallelic_selects_correct_alt(self):
        """At a multiallelic locus the missing variant's effect allele selects the
        matching ALT row, not the first reference row at that position."""
        # Two reference rows at locus 1:200000 -> A>G (col 0) and A>T (col 1).
        dosage_matrix = np.array([
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
        ])
        variant_info = pd.DataFrame({
            "variant_id": ["1:200000:A:G", "1:200000:A:T"],
            "chromosome": ["1", "1"],
            "position": [200000, 200000],
            "ref_allele": ["A", "A"],
            "alt_allele": ["G", "T"],
        })
        gd = GenotypeData(
            dosage_matrix=dosage_matrix,
            variant_info=variant_info,
            sample_ids=["S0", "S1", "S2"],
        )
        # Missing PRS variant targets the SECOND ALT (T) with beta=0.5.
        region_model = ProjectionRegionModel(
            region_id="chr1:200000-200000",
            chromosome="1",
            start=200000,
            end=200000,
            prs_variant_ids=["rs_multi"],
            betas=np.array([0.5]),
            predictor_variant_ids=[],
            coefficients=np.array([]),
            intercept=0.0,
            cv_mse=0.0,
            cv_r2=0.0,
            is_intercept_only=True,
            mean_prs_contribution=0.0,
            predictor_allele_frequencies=np.array([]),
            prs_positions=[200000],
            prs_effect_alleles=["T"],
            prs_other_alleles=["A"],
        )
        model = LinearProjectionPRS()
        model._is_fitted = True
        model._observed_variants = []
        model._region_models = [region_model]

        true_prs = ProjectionEvaluator(model, verbose=0)._compute_true_prs(gd)

        # Effect allele T -> A>T row (col 1) dosage [2, 0, 1] * 0.5.
        np.testing.assert_allclose(
            true_prs, np.array([1.0, 0.0, 0.5]), rtol=0, atol=1e-12
        )
        # The first row (A>G, col 0) would have given [0, 1, 0.5] -> different.
        assert not np.allclose(true_prs, dosage_matrix[:, 0] * 0.5)

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
