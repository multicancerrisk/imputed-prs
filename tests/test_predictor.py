"""Tests for the predictor module."""

import numpy as np
import pandas as pd

from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.user_genotypes import load_raw_user_genotypes
from imputed_prs.models.bounding import clip_and_adjust_variance
from imputed_prs.models.predictor import (
    ObservedScore,
    PRSPredictor,
    _effective_residual_variance,
    _predict_model_dosage,
    compute_imputed_prs,
    compute_imputed_prs_oriented,
    compute_observed_prs,
    compute_observed_prs_oriented,
)
from imputed_prs.utils.helpers import (
    compute_residual_variance,
    hardy_weinberg_variance,
)


def _collection(rows):
    """Build a RawUserGenotypeCollection from (rsid, chrom, pos, genotype) rows."""
    df = pd.DataFrame(
        {
            "rsid": [r[0] for r in rows],
            "chrom": [r[1] for r in rows],
            "pos": [r[2] for r in rows],
            "genotype": [r[3] for r in rows],
        }
    )
    return load_raw_user_genotypes(df)


def _render_genotype(ref: str, alt: str, alt_dosage: int) -> str:
    """Render an integer ALT dosage to a genotype string (inverse of counting ALT).

    ``alt_dosage`` copies of ALT plus ``2 - alt_dosage`` copies of REF, e.g.
    ``("G", "A", 1) -> "AG"``. Used to render reference ALT dosages to the raw
    genotype strings the oriented scorer consumes.
    """
    return alt * alt_dosage + ref * (2 - alt_dosage)


def _imputed_model(
    variant_id="rs_target",
    chromosome="1",
    position=5000,
    effect_allele="A",
    other_allele="G",
    beta=0.05,
    allele_frequency=0.3,
    imputation_r2=0.8,
    residual_variance=0.1,
    intercept=0.6,
    predictor_variant_ids=None,
    coefficients=None,
    is_intercept_only=False,
    predictor_chromosomes=None,
    predictor_positions=None,
    predictor_counted_alleles=None,
    predictor_other_alleles=None,
    predictor_allele_frequencies=None,
):
    """Build an ImputedVariantModel carrying P1.3 predictor allele metadata.

    Metadata defaults are length-aligned to ``predictor_variant_ids`` so the
    oriented scorer can resolve every predictor; override per-test to control
    orientation (counted = ALT, other = REF).
    """
    if predictor_variant_ids is None:
        predictor_variant_ids = ["rs_p0", "rs_p1"]
    n = len(predictor_variant_ids)
    if coefficients is None:
        coefficients = np.full(n, 0.2)
    if predictor_chromosomes is None:
        predictor_chromosomes = [chromosome] * n
    if predictor_positions is None:
        predictor_positions = [1000 + 100 * i for i in range(n)]
    if predictor_counted_alleles is None:
        predictor_counted_alleles = ["A"] * n
    if predictor_other_alleles is None:
        predictor_other_alleles = ["G"] * n
    if predictor_allele_frequencies is None:
        predictor_allele_frequencies = np.full(n, allele_frequency)
    return ImputedVariantModel(
        variant_id=variant_id,
        chromosome=chromosome,
        position=position,
        effect_allele=effect_allele,
        other_allele=other_allele,
        beta=beta,
        allele_frequency=allele_frequency,
        imputation_r2=imputation_r2,
        residual_variance=residual_variance,
        intercept=intercept,
        predictor_variant_ids=predictor_variant_ids,
        coefficients=coefficients,
        is_intercept_only=is_intercept_only,
        predictor_chromosomes=predictor_chromosomes,
        predictor_positions=predictor_positions,
        predictor_counted_alleles=predictor_counted_alleles,
        predictor_other_alleles=predictor_other_alleles,
        predictor_allele_frequencies=predictor_allele_frequencies,
    )


class TestComputeObservedPrsOriented:
    """Allele-aware observed scoring (P1.2): counts the effect allele."""

    def test_effect_alt_hand_calc(self):
        """effect=ALT: 'AA' counts 2 copies of A -> 2*beta."""
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1)]
        coll = _collection([("rs1", "1", 100, "AA")])
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        assert isinstance(result, ObservedScore)
        np.testing.assert_allclose(result.prs, 0.2, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 1
        assert result.unresolved_ids == ()

    def test_effect_ref_hand_calc(self):
        """effect=REF: 'AA' counts 0 copies of effect=G -> 0.0, NOT 2*beta."""
        observed = [VariantInfo("rs2", "1", 200, "G", "A", 0.5)]
        coll = _collection([("rs2", "1", 200, "AA")])
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        np.testing.assert_allclose(result.prs, 0.0, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 1
        # Contrast: the allele-blind path would score genotype_to_dosage('AA')=2.
        blind, _ = compute_observed_prs({"rs2": 2.0}, observed)
        np.testing.assert_allclose(blind, 1.0, rtol=0, atol=1e-12)

    def test_heterozygote_order_invariant(self):
        """'AG' and 'GA' both count 1 copy of the effect allele."""
        observed = [VariantInfo("rs3", "1", 300, "A", "G", 0.7)]
        for geno in ("AG", "GA"):
            result = compute_observed_prs_oriented(
                _collection([("rs3", "1", 300, geno)]),
                observed,
                allow_ambiguous=True,
            )
            np.testing.assert_allclose(result.prs, 0.7, rtol=0, atol=1e-12)
            assert result.n_scored_direct == 1

    def test_resolves_by_chrpos_when_rsid_differs(self):
        """A variant whose rsID is absent resolves via chr:pos."""
        observed = [VariantInfo("rs_prs_only", "1", 400, "A", "G", 0.3)]
        # User file has a different id at the same locus.
        coll = _collection([("rs_platform", "1", 400, "AA")])
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        np.testing.assert_allclose(result.prs, 0.6, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 1
        assert result.unresolved_ids == ()

    def test_duplicate_conflict_is_unresolved(self):
        """Conflicting duplicate user entries -> unresolved, never scored."""
        observed = [VariantInfo("rs5", "1", 500, "A", "G", 0.9)]
        coll = _collection(
            [("rs5", "1", 500, "AA"), ("rs5", "1", 500, "GG")]
        )
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        np.testing.assert_allclose(result.prs, 0.0, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 0
        assert result.unresolved_ids == ("rs5",)

    def test_palindromic_policy_knob(self):
        """A/T palindrome: unresolved when allow_ambiguous=False, scored when True."""
        observed = [VariantInfo("rs6", "1", 600, "A", "T", 0.4)]
        coll = _collection([("rs6", "1", 600, "AA")])

        blocked = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=False
        )
        assert blocked.n_scored_direct == 0
        assert blocked.unresolved_ids == ("rs6",)

        allowed = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        np.testing.assert_allclose(allowed.prs, 0.8, rtol=0, atol=1e-12)
        assert allowed.n_scored_direct == 1

    def test_strand_flip_knob(self):
        """Complementary-strand genotype scored only with allow_strand_flip=True."""
        observed = [VariantInfo("rs7", "1", 700, "A", "G", 0.5)]
        coll = _collection([("rs7", "1", 700, "TT")])  # complement of AA

        no_flip = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True, allow_strand_flip=False
        )
        assert no_flip.n_scored_direct == 0
        assert no_flip.unresolved_ids == ("rs7",)

        with_flip = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True, allow_strand_flip=True
        )
        np.testing.assert_allclose(with_flip.prs, 1.0, rtol=0, atol=1e-12)
        assert with_flip.n_scored_direct == 1

    def test_missing_other_allele_is_unresolved(self):
        """other_allele=None cannot be browser-safely oriented -> unresolved."""
        observed = [VariantInfo("rs8", "1", 800, "A", None, 0.6)]
        coll = _collection([("rs8", "1", 800, "AA")])
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        assert result.n_scored_direct == 0
        assert result.unresolved_ids == ("rs8",)

    def test_not_found_is_unresolved(self):
        """A variant absent from the user file is unresolved, not scored."""
        observed = [VariantInfo("rs9", "1", 900, "A", "G", 0.6)]
        coll = _collection([("rsX", "2", 111, "AA")])
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=True
        )
        np.testing.assert_allclose(result.prs, 0.0, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 0
        assert result.unresolved_ids == ("rs9",)

    def test_empty_observed_list(self):
        """No observed variants -> zeroed ObservedScore."""
        result = compute_observed_prs_oriented(
            _collection([("rs1", "1", 100, "AA")]), [], allow_ambiguous=True
        )
        assert result == ObservedScore(0.0, 0, ())

    def test_mixed_set_exact_sum_and_unresolved_order(self):
        """Mixed observed set: exact PRS, count, and unresolved ordering."""
        observed = [
            VariantInfo("rs_alt", "1", 100, "A", "G", 0.1),   # AA -> 2*0.1
            VariantInfo("rs_ref", "1", 200, "G", "A", 0.5),   # AA -> 0*0.5
            VariantInfo("rs_absent", "1", 300, "A", "G", 1.0),  # not in file
            VariantInfo("rs_pal", "1", 400, "C", "G", 2.0),   # palindrome blocked
        ]
        coll = _collection(
            [
                ("rs_alt", "1", 100, "AA"),
                ("rs_ref", "1", 200, "AA"),
                ("rs_pal", "1", 400, "CC"),
            ]
        )
        result = compute_observed_prs_oriented(
            coll, observed, allow_ambiguous=False
        )
        np.testing.assert_allclose(result.prs, 0.2, rtol=0, atol=1e-12)
        assert result.n_scored_direct == 2  # rs_alt and rs_ref
        # Unresolved preserves observed_variants order.
        assert result.unresolved_ids == ("rs_absent", "rs_pal")


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


class TestComputeImputedPrsOriented:
    """Allele-aware imputed scoring (P1.4): counts predictor ALT alleles."""

    def test_full_predictor_set_matches_reference_dot_product(self):
        """Genotype-string scoring equals the oriented reference ALT-dosage dot product."""
        # Three non-palindromic predictors; reference ALT dosages [2, 1, 0].
        model = _imputed_model(
            beta=0.05,
            intercept=0.1,
            residual_variance=0.05,
            predictor_variant_ids=["rs_p0", "rs_p1", "rs_p2"],
            coefficients=np.array([0.3, -0.2, 0.15]),
            predictor_chromosomes=["1", "1", "1"],
            predictor_positions=[1000, 1100, 1200],
            predictor_counted_alleles=["A", "C", "T"],  # ALT
            predictor_other_alleles=["G", "T", "C"],  # REF
            predictor_allele_frequencies=np.array([0.3, 0.4, 0.25]),
        )
        coll = _collection(
            [
                ("rs_p0", "1", 1000, _render_genotype("G", "A", 2)),  # "AA" -> 2
                ("rs_p1", "1", 1100, _render_genotype("T", "C", 1)),  # "CT" -> 1
                ("rs_p2", "1", 1200, _render_genotype("C", "T", 0)),  # "CC" -> 0
            ]
        )
        prs, variance, n_imputed, n_truncated = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # raw = 2*0.3 + 1*(-0.2) + 0*0.15 + 0.1 = 0.5 (within [0, 2], no clipping).
        expected_prs = 0.5 * 0.05
        np.testing.assert_allclose(prs, expected_prs, rtol=0, atol=1e-12)
        assert n_imputed == 1
        assert n_truncated == 0
        assert variance > 0

    def test_alt_ref_orientation_bugfix(self):
        """A 'GG' call with ALT=A counts 0 copies, not homozygous 2 (the bug)."""
        model = _imputed_model(
            beta=1.0,
            intercept=0.0,
            residual_variance=0.0,
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            predictor_chromosomes=["1"],
            predictor_positions=[1000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
            predictor_allele_frequencies=np.array([0.1]),
        )
        coll = _collection([("rs_p0", "1", 1000, "GG")])
        prs, _, n_imputed, _ = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # Oriented: 0 copies of ALT A -> raw 0 -> prs 0.
        np.testing.assert_allclose(prs, 0.0, rtol=0, atol=1e-12)
        assert n_imputed == 1
        # Contrast: the allele-blind dosage-dict path counts "GG" homozygous -> 2.
        blind_prs, _, _, _ = compute_imputed_prs({"rs_p0": 2.0}, [model])
        np.testing.assert_allclose(blind_prs, 2.0, rtol=0, atol=1e-12)

    def test_missing_predictor_mean_substitution(self):
        """A missing predictor uses 2*AF; the model does NOT collapse to intercept."""
        model = _imputed_model(
            beta=0.1,
            intercept=0.2,
            residual_variance=0.05,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.3]),
            predictor_chromosomes=["1", "1"],
            predictor_positions=[1000, 1100],
            predictor_counted_alleles=["A", "C"],
            predictor_other_alleles=["G", "T"],
            predictor_allele_frequencies=np.array([0.3, 0.4]),
        )
        # rs_p0 present (dosage 2 -> "AA"); rs_p1 omitted -> 2*0.4 = 0.8.
        coll = _collection([("rs_p0", "1", 1000, "AA")])
        prs, _, n_imputed, n_truncated = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # raw = 2*0.5 + 0.8*(-0.3) + 0.2 = 0.96  (NOT the intercept fallback 0.2*0.1).
        np.testing.assert_allclose(prs, 0.96 * 0.1, rtol=0, atol=1e-12)
        assert n_imputed == 1
        assert n_truncated == 0

    def test_unresolved_predictor_substitutes(self):
        """A duplicate-conflict locus is unresolved and mean-substituted, not guessed."""
        model = _imputed_model(
            beta=0.1,
            intercept=0.2,
            residual_variance=0.05,
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([0.5]),
            predictor_chromosomes=["1"],
            predictor_positions=[1000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
            predictor_allele_frequencies=np.array([0.3]),
        )
        # Two conflicting genotype calls at the same locus -> duplicate_conflict.
        coll = _collection(
            [("rs_p0", "1", 1000, "AA"), ("rs_p0", "1", 1000, "GG")]
        )
        prs, _, _, _ = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # Unresolved -> 2*0.3 = 0.6: raw = 0.6*0.5 + 0.2 = 0.5.
        np.testing.assert_allclose(prs, 0.5 * 0.1, rtol=0, atol=1e-12)

    def test_palindromic_predictor_counted_when_allow_ambiguous(self):
        """A palindromic (A/T) predictor is counted with allow_ambiguous, else substituted."""
        model = _imputed_model(
            beta=0.1,
            intercept=0.0,
            residual_variance=0.0,
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            predictor_chromosomes=["1"],
            predictor_positions=[1000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["T"],  # A/T palindrome
            predictor_allele_frequencies=np.array([0.2]),
        )
        coll = _collection([("rs_p0", "1", 1000, "AA")])
        # allow_ambiguous=True: counted -> 2 copies -> prs = 2*1.0*0.1 = 0.2.
        prs, _, _, _ = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        np.testing.assert_allclose(prs, 0.2, rtol=0, atol=1e-12)
        # allow_ambiguous=False: palindrome unresolved -> 2*0.2 = 0.4 -> prs = 0.04.
        prs_blocked, _, _, _ = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=False
        )
        np.testing.assert_allclose(prs_blocked, 0.04, rtol=0, atol=1e-12)

    def test_clipping_applies_to_raw_prediction(self):
        """The imputed dosage is clipped to [0, 2] before * beta, with truncation count."""
        model = _imputed_model(
            beta=0.1,
            intercept=1.0,
            residual_variance=0.1,
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([2.0]),
            predictor_chromosomes=["1"],
            predictor_positions=[1000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
            predictor_allele_frequencies=np.array([0.3]),
        )
        coll = _collection([("rs_p0", "1", 1000, "AA")])  # dosage 2
        prs, _, _, n_truncated = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # raw = 2*2.0 + 1.0 = 5.0 -> clipped to 2.0 -> prs = 2.0 * 0.1 = 0.2.
        np.testing.assert_allclose(prs, 0.2, rtol=0, atol=1e-12)
        assert n_truncated == 1

    def test_heterozygote_order_invariant(self):
        """'AG' and 'GA' both count one copy of the ALT allele."""
        model = _imputed_model(
            beta=0.1,
            intercept=0.0,
            residual_variance=0.0,
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            predictor_chromosomes=["1"],
            predictor_positions=[1000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
            predictor_allele_frequencies=np.array([0.3]),
        )
        for geno in ("AG", "GA"):
            coll = _collection([("rs_p0", "1", 1000, geno)])
            prs, _, _, _ = compute_imputed_prs_oriented(
                coll, [model], allow_ambiguous=True
            )
            np.testing.assert_allclose(prs, 0.1, rtol=0, atol=1e-12)

    def test_intercept_only_unchanged(self):
        """An intercept-only model scores its intercept and reads no metadata."""
        model = _imputed_model(
            beta=0.1,
            intercept=0.8,
            is_intercept_only=True,
            predictor_variant_ids=[],
            coefficients=np.array([]),
        )
        coll = _collection([("rs_x", "1", 999, "AA")])  # irrelevant
        prs, _, n_imputed, n_truncated = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        np.testing.assert_allclose(prs, 0.08, rtol=0, atol=1e-12)
        assert n_imputed == 1
        assert n_truncated == 0


class TestPRSPredictor:
    """Tests for PRSPredictor class."""

    def _create_observed_variants(self):
        """Create sample observed variants for testing."""
        return [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]

    def _create_imputed_model(
        self,
        variant_id="rs3",
        beta=0.05,
        intercept=0.6,
        predictor_ids=None,
        coefficients=None,
        is_intercept_only=False,
        residual_variance=0.1,
    ):
        """Create a sample imputed model for testing."""
        if predictor_ids is None:
            predictor_ids = ["rs1", "rs2"]
        if coefficients is None:
            coefficients = np.array([0.3, 0.2])
        return ImputedVariantModel(
            variant_id=variant_id,
            chromosome="1",
            position=300,
            effect_allele="A",
            other_allele="G",
            beta=beta,
            allele_frequency=0.3,
            imputation_r2=0.8,
            residual_variance=residual_variance,
            intercept=intercept,
            predictor_variant_ids=predictor_ids,
            coefficients=coefficients,
            is_intercept_only=is_intercept_only,
        )

    def _create_calibration_params(self):
        """Create sample calibration parameters for testing."""
        return CalibrationParams(
            scaling_factor=1.1,
            scaling_factor_se=0.05,
            calibration_intercept=0.01,
            calibration_r2=0.95,
            sd_cv_predicted=0.5,
            sd_true=0.55,
            sd_scaled=0.55,
            attenuation_factor=0.91,
            n_calibration=500,
        )

    def test_basic_prediction_mixed_observed_and_imputed(self):
        """Basic prediction with mixed observed and imputed variants."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor(observed, imputed)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # Observed: 2*0.1 + 1*0.2 = 0.4
        # Imputed raw: 2*0.3 + 1*0.2 + 0.6 = 1.4
        # Imputed PRS: 1.4 * 0.05 = 0.07
        # Total: 0.4 + 0.07 = 0.47
        expected_observed = 0.4
        expected_imputed = 1.4 * 0.05
        expected_prs = expected_observed + expected_imputed

        assert abs(result.prs_observed_component - expected_observed) < 1e-10
        assert abs(result.prs_imputed_component - expected_imputed) < 1e-10
        assert abs(result.prs - expected_prs) < 1e-10
        assert result.n_variants_used == 3
        assert result.n_variants_imputed == 1

    def test_legacy_dict_path_leaves_oriented_diagnostics_none(self):
        """The allele-blind dosage-dict path does not populate oriented fields."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor(observed, imputed)

        result = predictor.predict({"rs1": 2.0, "rs2": 1.0}, apply_calibration=False)

        assert result.n_observed_scored_direct is None
        assert result.unresolved_observed_ids is None
        # Score itself is unchanged from the pre-P1.2 baseline.
        assert abs(result.prs - 0.47) < 1e-10

    def test_oriented_observed_via_raw_genotypes(self):
        """raw_genotypes routes observed AND predictor scoring through the oriented path."""
        observed = self._create_observed_variants()  # rs1=(A,G,0.1), rs2=(C,T,0.2)
        # Distinct predictor loci (rs10, rs11) so the observed and predictor roles do
        # not collide; metadata + rendered collection reproduce predictor dosages 2, 1.
        imputed = [
            _imputed_model(
                variant_id="rs3",
                beta=0.05,
                intercept=0.6,
                residual_variance=0.1,
                predictor_variant_ids=["rs10", "rs11"],
                coefficients=np.array([0.3, 0.2]),
                predictor_chromosomes=["1", "1"],
                predictor_positions=[1000, 1100],
                predictor_counted_alleles=["A", "C"],
                predictor_other_alleles=["G", "T"],
                predictor_allele_frequencies=np.array([0.3, 0.3]),
            )
        ]
        predictor = PRSPredictor(observed, imputed)

        # With raw_genotypes the dosage dict is ignored; everything is scored from
        # strings: observed rs1 "GG" -> 0 copies of effect A, rs2 "CC" -> 2 copies of
        # effect C; predictors rs10 "AA" -> 2 copies of ALT A, rs11 "CT" -> 1 copy of C.
        coll = _collection(
            [
                ("rs1", "1", 100, "GG"),
                ("rs2", "1", 200, "CC"),
                ("rs10", "1", 1000, "AA"),
                ("rs11", "1", 1100, "CT"),
            ]
        )
        result = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)

        # Oriented observed = 0*0.1 + 2*0.2 = 0.4 (allele-blind would give 0.6).
        np.testing.assert_allclose(
            result.prs_observed_component, 0.4, rtol=0, atol=1e-12
        )
        # Oriented imputed raw = 2*0.3 + 1*0.2 + 0.6 = 1.4 -> 1.4 * 0.05 = 0.07.
        np.testing.assert_allclose(
            result.prs_imputed_component, 0.07, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(result.prs, 0.47, rtol=0, atol=1e-12)
        assert result.n_observed_scored_direct == 2
        assert result.unresolved_observed_ids == ()
        assert result.n_variants_used == 3

    def test_predict_imputed_via_raw_genotypes_matches_oriented(self):
        """predict() routes the imputed component through the oriented scorer."""
        # A 'GG' predictor with ALT=A: oriented counts 0; the blind dict counts 2.
        imputed = [
            _imputed_model(
                variant_id="rs_t",
                beta=0.1,
                intercept=0.0,
                residual_variance=0.0,
                predictor_variant_ids=["rs_p0"],
                coefficients=np.array([1.0]),
                predictor_chromosomes=["1"],
                predictor_positions=[1000],
                predictor_counted_alleles=["A"],
                predictor_other_alleles=["G"],
                predictor_allele_frequencies=np.array([0.1]),
            )
        ]
        predictor = PRSPredictor([], imputed)
        coll = _collection([("rs_p0", "1", 1000, "GG")])
        result = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)
        np.testing.assert_allclose(
            result.prs_imputed_component, 0.0, rtol=0, atol=1e-12
        )
        # The legacy dosage-dict path (no raw_genotypes) counts "GG" homozygous -> 2.
        blind = predictor.predict({"rs_p0": 2.0}, apply_calibration=False)
        np.testing.assert_allclose(
            blind.prs_imputed_component, 0.2, rtol=0, atol=1e-12
        )

    def test_all_observed_variants_no_imputation(self):
        """All observed variants, no imputation needed."""
        observed = self._create_observed_variants()
        predictor = PRSPredictor(observed, [])

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # Observed: 2*0.1 + 1*0.2 = 0.4
        assert abs(result.prs_observed_component - 0.4) < 1e-10
        assert abs(result.prs_imputed_component - 0.0) < 1e-10
        assert abs(result.prs - 0.4) < 1e-10
        assert result.n_variants_used == 2
        assert result.n_variants_imputed == 0
        assert result.se == 0.0  # No variance from imputation

    def test_all_imputed_variants_no_observed(self):
        """All imputed variants, no observed."""
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor([], imputed)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # Imputed raw: 2*0.3 + 1*0.2 + 0.6 = 1.4
        # Imputed PRS: 1.4 * 0.05 = 0.07
        assert abs(result.prs_observed_component - 0.0) < 1e-10
        assert abs(result.prs_imputed_component - 0.07) < 1e-10
        assert abs(result.prs - 0.07) < 1e-10
        assert result.n_variants_used == 1
        assert result.n_variants_imputed == 1
        assert result.se > 0  # Has variance from imputation

    def test_empty_user_genotypes_all_missing(self):
        """Empty user genotypes, all variants missing."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor(observed, imputed)

        user_dosages = {}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # Observed: 0 (no dosages)
        # Imputed: falls back to intercept = 0.6 * 0.05 = 0.03
        assert result.prs_observed_component == 0.0
        assert abs(result.prs_imputed_component - 0.03) < 1e-10
        assert result.n_variants_used == 1  # Only imputed variant (via intercept fallback)
        assert result.n_user_variants_missing == 2  # 2 observed variants missing

    def test_prediction_without_calibration(self):
        """Prediction without calibration (apply_calibration=False)."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        calib = self._create_calibration_params()
        predictor = PRSPredictor(observed, imputed, calib)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        assert result.prs_scaled is None
        assert result.se_scaled is None
        assert result.ci_lower_scaled is None
        assert result.ci_upper_scaled is None

    def test_prediction_with_calibration_scaling(self):
        """Prediction with calibration scaling."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        calib = self._create_calibration_params()
        predictor = PRSPredictor(observed, imputed, calib)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=True)

        # Verify scaled values are computed
        assert result.prs_scaled is not None
        assert result.se_scaled is not None
        assert result.ci_lower_scaled is not None
        assert result.ci_upper_scaled is not None

        # Verify scaling math
        expected_scaled_prs = calib.scaling_factor * result.prs + calib.calibration_intercept
        expected_scaled_se = abs(calib.scaling_factor) * result.se
        assert abs(result.prs_scaled - expected_scaled_prs) < 1e-10
        assert abs(result.se_scaled - expected_scaled_se) < 1e-10

    def test_confidence_interval_correctness(self):
        """95% CI math is correct (prs ± 1.96*se)."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor(observed, imputed)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        expected_ci_lower = result.prs - 1.96 * result.se
        expected_ci_upper = result.prs + 1.96 * result.se

        assert abs(result.ci_lower - expected_ci_lower) < 1e-10
        assert abs(result.ci_upper - expected_ci_upper) < 1e-10
        assert result.ci_lower < result.prs < result.ci_upper

    def test_scaled_confidence_interval_correctness(self):
        """Scaled 95% CI math is correct."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        calib = self._create_calibration_params()
        predictor = PRSPredictor(observed, imputed, calib)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=True)

        expected_ci_lower = result.prs_scaled - 1.96 * result.se_scaled
        expected_ci_upper = result.prs_scaled + 1.96 * result.se_scaled

        assert abs(result.ci_lower_scaled - expected_ci_lower) < 1e-10
        assert abs(result.ci_upper_scaled - expected_ci_upper) < 1e-10

    def test_component_accounting_n_variants_used(self):
        """n_variants_used matches sum of observed and imputed components."""
        observed = self._create_observed_variants()
        imputed = [
            self._create_imputed_model("rs3"),
            self._create_imputed_model("rs4"),
        ]
        predictor = PRSPredictor(observed, imputed)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # 2 observed + 2 imputed = 4 total
        assert result.n_variants_used == 4
        assert result.n_variants_imputed == 2

    def test_intercept_only_count_tracking(self):
        """Intercept-only count is tracked correctly."""
        observed = self._create_observed_variants()
        imputed = [
            self._create_imputed_model(
                "rs3",
                predictor_ids=[],
                coefficients=np.array([]),
                is_intercept_only=True,
            ),
            self._create_imputed_model("rs4", is_intercept_only=False),
        ]
        predictor = PRSPredictor(observed, imputed)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # 1 intercept-only model
        assert result.n_variants_intercept_only == 1

    def test_truncation_count_passthrough(self):
        """Truncation count is passed through from imputed component."""
        observed = self._create_observed_variants()
        # Create model that will trigger clipping (raw > 2.0)
        imputed = [
            ImputedVariantModel(
                variant_id="rs3",
                chromosome="1",
                position=300,
                effect_allele="A",
                other_allele="G",
                beta=0.1,
                allele_frequency=0.9,
                imputation_r2=0.8,
                residual_variance=0.1,
                intercept=1.0,
                predictor_variant_ids=["rs1"],
                coefficients=np.array([0.8]),
                is_intercept_only=False,
            )
        ]
        predictor = PRSPredictor(observed, imputed)

        # raw = 2.0 * 0.8 + 1.0 = 2.6 > 2.0, should be clipped
        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        assert result.n_truncated == 1

    def test_missing_user_variants_count(self):
        """Missing user variants count is computed correctly."""
        observed = self._create_observed_variants()  # 2 variants
        imputed = [self._create_imputed_model()]  # 1 variant
        predictor = PRSPredictor(observed, imputed)

        # Only provide dosage for rs1, rs2 is missing
        user_dosages = {"rs1": 2.0}
        result = predictor.predict(user_dosages, apply_calibration=False)

        # Total variants: 2 observed + 1 imputed = 3
        # Used: 1 observed + 1 imputed (via intercept fallback) = 2
        # Missing: 3 - 2 = 1
        assert result.n_user_variants_missing == 1

    def test_calibration_without_params_returns_none(self):
        """apply_calibration=True without calibration_params returns None scaled values."""
        observed = self._create_observed_variants()
        imputed = [self._create_imputed_model()]
        predictor = PRSPredictor(observed, imputed, calibration_params=None)

        user_dosages = {"rs1": 2.0, "rs2": 1.0}
        result = predictor.predict(user_dosages, apply_calibration=True)

        assert result.prs_scaled is None
        assert result.se_scaled is None
        assert result.ci_lower_scaled is None
        assert result.ci_upper_scaled is None

def _intercept_only_fallback(variant_id, beta, intercept, residual_variance=0.1):
    """An intercept-only ImputedVariantModel usable as an observed fallback (P1.8).

    Predicts ``clip(intercept)`` copies of the effect allele regardless of the
    upload, so tests can pin the recovered contribution to ``clip(intercept)*beta``.
    """
    return _imputed_model(
        variant_id=variant_id,
        effect_allele="A",
        other_allele="G",
        beta=beta,
        intercept=intercept,
        residual_variance=residual_variance,
        is_intercept_only=True,
        predictor_variant_ids=[],
    )


class TestObservedFallback:
    """P1.8 — observed variants recovered via per-variant fallback models."""

    def test_direct_scoring_ignores_biased_fallback(self):
        """A resolvable variant is scored direct; its (biased) fallback is unused."""
        # The biased fallback predicts dosage 0 -> would contribute 0.0 if used;
        # the direct "AA" count contributes 2*0.1 = 0.2. They differ, so the
        # asserted 0.2 proves the fallback was not consulted.
        fb = _intercept_only_fallback("rs1", beta=0.1, intercept=0.0)
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1, fallback=fb)]
        coll = _collection([("rs1", "1", 100, "AA")])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        np.testing.assert_allclose(r.prs, 0.2, rtol=0, atol=1e-12)
        assert r.n_scored_direct == 1
        assert r.n_scored_fallback == 0
        assert r.fallback_variance == 0.0
        assert r.unresolved_ids == ()

    def test_homozygous_other_allele_is_direct_not_fallback(self):
        """A 0.0 direct dosage (homozygous other allele) is a score, not a failure."""
        fb = _intercept_only_fallback("rs1", beta=0.1, intercept=2.0)  # would be 0.2
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1, fallback=fb)]
        coll = _collection([("rs1", "1", 100, "GG")])  # zero copies of A
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.prs == 0.0
        assert r.n_scored_direct == 1
        assert r.n_scored_fallback == 0

    def test_no_call_recovered_via_fallback(self):
        """A no-call observed genotype is recovered via fallback, not dropped."""
        fb = _intercept_only_fallback("rs_big", beta=1.5, intercept=0.6)
        observed = [VariantInfo("rs_big", "1", 100, "A", "G", 1.5, fallback=fb)]
        coll = _collection([("rs_big", "1", 100, "--")])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.n_scored_direct == 0
        assert r.n_scored_fallback == 1
        assert r.unresolved_ids == ()
        # intercept-only fallback predicts clip(0.6) = 0.6 -> 0.6 * 1.5
        np.testing.assert_allclose(r.prs, 0.6 * 1.5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.weighted_beta_fallback, 1.5, rtol=0, atol=1e-12)
        assert r.fallback_variance > 0.0

    def test_no_fallback_stays_unresolved(self):
        """Without a fallback model, a no-call variant is reported unresolved."""
        observed = [VariantInfo("rs_big", "1", 100, "A", "G", 1.5)]
        coll = _collection([("rs_big", "1", 100, "--")])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.n_scored_fallback == 0
        assert r.unresolved_ids == ("rs_big",)
        assert r.prs == 0.0
        assert r.fallback_variance == 0.0

    def test_partial_overlap_routes_to_fallback(self):
        """A genotype partially overlapping the allele pair routes to fallback."""
        fb = _intercept_only_fallback("rs1", beta=0.7, intercept=0.6)
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.7, fallback=fb)]
        coll = _collection([("rs1", "1", 100, "AC")])  # C not in {A, G}
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.n_scored_direct == 0
        assert r.n_scored_fallback == 1
        assert r.unresolved_ids == ()

    def test_duplicate_conflict_routes_to_fallback(self):
        """A duplicate-conflict (same id, conflicting genotype) routes to fallback."""
        fb = _intercept_only_fallback("rs1", beta=0.7, intercept=0.6)
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.7, fallback=fb)]
        coll = _collection([("rs1", "1", 100, "AA"), ("rs1", "1", 100, "GG")])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.n_scored_direct == 0
        assert r.n_scored_fallback == 1
        assert r.unresolved_ids == ()

    def test_weighted_beta_via_fallback_sums_absolute_betas(self):
        """weighted_beta_fallback = sum of |beta| over fallback-scored variants."""
        observed = [
            VariantInfo("rs_a", "1", 100, "A", "G", 0.1),  # scored direct
            VariantInfo("rs_b", "1", 200, "A", "G", -2.0,
                        fallback=_intercept_only_fallback("rs_b", -2.0, 0.6)),
            VariantInfo("rs_c", "1", 300, "A", "G", 0.5,
                        fallback=_intercept_only_fallback("rs_c", 0.5, 0.6)),
        ]
        coll = _collection([
            ("rs_a", "1", 100, "AA"),
            ("rs_b", "1", 200, "--"),
            ("rs_c", "1", 300, "--"),
        ])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert r.n_scored_direct == 1
        assert r.n_scored_fallback == 2
        # |-2.0| + |0.5| = 2.5 (a signed sum would be -1.5).
        np.testing.assert_allclose(r.weighted_beta_fallback, 2.5, rtol=0, atol=1e-12)

    def test_fallback_with_predictors_mean_substitutes_missing(self):
        """A fallback's missing predictor is mean-substituted (2*AF), matching the
        imputed scorer; the present predictor uses its real counted dosage."""
        fb = _imputed_model(
            variant_id="rs_t", effect_allele="A", other_allele="G", beta=1.0,
            intercept=0.1, residual_variance=0.05, is_intercept_only=False,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.3, 0.4]),
            predictor_chromosomes=["1", "1"],
            predictor_positions=[1000, 2000],
            predictor_counted_alleles=["A", "A"],
            predictor_other_alleles=["G", "G"],
            predictor_allele_frequencies=np.array([0.4, 0.4]),
        )
        observed = [VariantInfo("rs_t", "1", 5000, "A", "G", 1.0, fallback=fb)]
        # rs_t no-call -> fallback; rs_p0 "AA" -> 2 copies of A; rs_p1 omitted -> 2*0.4.
        coll = _collection([
            ("rs_t", "1", 5000, "--"),
            ("rs_p0", "1", 1000, "AA"),
        ])
        r = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        # raw = 2.0*0.3 + (2*0.4)*0.4 + 0.1 = 0.6 + 0.32 + 0.1 = 1.02; clip -> 1.02.
        np.testing.assert_allclose(r.prs, 1.02, rtol=0, atol=1e-12)
        assert r.n_scored_fallback == 1

    def test_predict_model_dosage_matches_imputed_scorer(self):
        """_predict_model_dosage is the per-model body of compute_imputed_prs_oriented."""
        model = _imputed_model(beta=0.05)
        coll = _collection([
            ("rs_p0", "1", 1000, "AA"),
            ("rs_p1", "1", 1100, "AG"),
        ])
        dosage, var, trunc = _predict_model_dosage(
            model, coll, allow_ambiguous=True, allow_strand_flip=True
        )
        prs, total_var, _, n_trunc = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        np.testing.assert_allclose(prs, dosage * model.beta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            total_var, (model.beta ** 2) * var, rtol=0, atol=1e-12
        )
        assert n_trunc == (1 if trunc else 0)


class TestPRSPredictorFallback:
    """P1.8 fallback wired through the full PRSPredictor.predict path."""

    def test_fallback_variance_reaches_se_and_ci(self):
        fb = _intercept_only_fallback("rs1", beta=1.0, intercept=0.6)
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 1.0, fallback=fb)]
        predictor = PRSPredictor(observed, imputed_models=[])
        coll = _collection([("rs1", "1", 100, "--")])
        r = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)
        assert r.n_observed_scored_direct == 0
        assert r.n_observed_scored_via_fallback == 1
        assert r.unresolved_observed_ids == ()
        assert r.se > 0.0
        assert r.ci_lower < r.prs < r.ci_upper
        assert r.n_variants_used == 1

    def test_no_fallback_control_has_zero_se(self):
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 1.0)]
        predictor = PRSPredictor(observed, imputed_models=[])
        coll = _collection([("rs1", "1", 100, "--")])
        r = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)
        assert r.se == 0.0
        assert r.unresolved_observed_ids == ("rs1",)
        assert r.n_observed_scored_via_fallback == 0

    def test_direct_unchanged_with_fallback_present(self):
        fb = _intercept_only_fallback("rs1", beta=0.1, intercept=0.0)  # biased
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1, fallback=fb)]
        predictor = PRSPredictor(observed, imputed_models=[])
        coll = _collection([("rs1", "1", 100, "AA")])
        r = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)
        np.testing.assert_allclose(r.prs_observed_component, 0.2, rtol=0, atol=1e-12)
        assert r.n_observed_scored_via_fallback == 0
        assert r.se == 0.0  # exact integer count

    def test_calibration_applies_on_top_of_fallback(self):
        calib = CalibrationParams(
            scaling_factor=2.0, scaling_factor_se=0.1, calibration_intercept=0.5,
            calibration_r2=0.9, sd_cv_predicted=1.0, sd_true=1.0, sd_scaled=1.0,
            attenuation_factor=1.0, n_calibration=100,
        )
        fb = _intercept_only_fallback("rs1", beta=1.0, intercept=0.6)
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 1.0, fallback=fb)]
        predictor = PRSPredictor(observed, imputed_models=[], calibration_params=calib)
        coll = _collection([("rs1", "1", 100, "--")])
        r = predictor.predict({}, apply_calibration=True, raw_genotypes=coll)
        np.testing.assert_allclose(r.prs_scaled, 2.0 * r.prs + 0.5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.se_scaled, 2.0 * r.se, rtol=0, atol=1e-12)

    def test_legacy_dict_path_leaves_fallback_fields_none(self):
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1)]
        predictor = PRSPredictor(observed, imputed_models=[])
        r = predictor.predict({"rs1": 2.0}, apply_calibration=False)
        assert r.n_observed_scored_via_fallback is None
        assert r.weighted_beta_via_fallback is None


class TestMissingnessAwareVariance:
    """P3.3: a model's residual variance is inflated from the full-model value
    toward the intercept-only Hardy-Weinberg variance 2q(1-q) in proportion to the
    fraction of predictors that were mean-substituted. This affects only the
    reported variance, never the point estimate.
    """

    def test_effective_residual_variance_formula(self):
        """The helper interpolates residual_variance -> 2q(1-q) by f = n_sub/n_pred."""
        af = 0.3
        residual = compute_residual_variance(af, 0.5)  # 0.42 * 0.5 = 0.21
        hw = hardy_weinberg_variance(af)  # 0.42
        model = _imputed_model(
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1"],
        )
        for n_sub in (0, 1, 2):
            f = n_sub / 2
            np.testing.assert_allclose(
                _effective_residual_variance(model, n_sub),
                residual * (1 - f) + hw * f,
                rtol=0,
                atol=1e-12,
            )

    def test_effective_residual_variance_endpoints(self):
        """f=0 -> residual_variance (unchanged); f=1 -> Hardy-Weinberg variance."""
        af = 0.3
        residual = compute_residual_variance(af, 0.5)
        model = _imputed_model(
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1"],
        )
        np.testing.assert_allclose(
            _effective_residual_variance(model, 0), residual, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            _effective_residual_variance(model, 2),
            hardy_weinberg_variance(af),
            rtol=0,
            atol=1e-12,
        )

    def test_effective_residual_variance_monotonic(self):
        """Effective variance is non-decreasing in the substituted fraction.

        Guaranteed because the trainer sets residual = 2q(1-q)*(1-r2) with
        r2 in [0,1], so 2q(1-q) >= residual_variance always.
        """
        af = 0.3
        residual = compute_residual_variance(af, 0.8)
        model = _imputed_model(
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1", "rs_p2", "rs_p3"],
            coefficients=np.full(4, 0.2),
            predictor_allele_frequencies=np.full(4, 0.4),
        )
        variances = [_effective_residual_variance(model, k) for k in range(5)]
        assert all(
            variances[i] <= variances[i + 1] + 1e-12
            for i in range(len(variances) - 1)
        )

    def test_intercept_only_model_variance_unchanged(self):
        """An intercept-only model (no predictors) keeps its full residual variance."""
        model = _imputed_model(
            residual_variance=0.07,
            is_intercept_only=True,
            predictor_variant_ids=[],
            coefficients=np.array([]),
            predictor_allele_frequencies=np.array([]),
        )
        np.testing.assert_allclose(
            _effective_residual_variance(model, 0), 0.07, rtol=0, atol=1e-12
        )

    def test_oriented_scorer_passes_effective_variance(self):
        """compute_imputed_prs_oriented feeds the inflated variance into clipping;
        the point estimate is unchanged by the inflation."""
        af = 0.3
        residual = compute_residual_variance(af, 0.5)  # 0.21
        beta = 0.1
        model = _imputed_model(
            beta=beta,
            intercept=0.2,
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.3]),
            predictor_chromosomes=["1", "1"],
            predictor_positions=[1000, 1100],
            predictor_counted_alleles=["A", "A"],
            predictor_other_alleles=["G", "G"],
            predictor_allele_frequencies=np.array([0.4, 0.4]),
        )
        # rs_p0 present ("AA" -> 2); rs_p1 omitted -> 2*0.4 = 0.8. One of two missing.
        coll = _collection([("rs_p0", "1", 1000, "AA")])
        prs, total_var, n_imputed, _ = compute_imputed_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        raw = 2 * 0.5 + 0.8 * (-0.3) + 0.2  # 0.96 (interior of [0, 2])
        effective = residual * 0.5 + hardy_weinberg_variance(af) * 0.5  # 0.315
        np.testing.assert_allclose(prs, raw * beta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            total_var,
            beta**2 * clip_and_adjust_variance(raw, effective)[1],
            rtol=0,
            atol=1e-12,
        )
        # The inflation genuinely changed the variance vs the un-inflated residual.
        uninflated = beta**2 * clip_and_adjust_variance(raw, residual)[1]
        assert total_var > uninflated
        assert n_imputed == 1

    def test_legacy_collapse_uses_hw_variance(self):
        """Legacy all-or-nothing collapse to intercept now reports the intercept-only
        Hardy-Weinberg variance, not the full-model residual variance (P3.3 f=1)."""
        af = 0.3
        residual = compute_residual_variance(af, 0.5)  # 0.21 < HW = 0.42
        beta = 0.1
        intercept = 0.2
        model = _imputed_model(
            beta=beta,
            intercept=intercept,
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.3]),
        )
        # rs_p1 missing -> collapse to intercept (allele-blind dosage-dict path).
        prs, variance, _, _ = compute_imputed_prs({"rs_p0": 2.0}, [model])
        np.testing.assert_allclose(prs, intercept * beta, rtol=0, atol=1e-12)
        hw = hardy_weinberg_variance(af)
        np.testing.assert_allclose(
            variance,
            beta**2 * clip_and_adjust_variance(intercept, hw)[1],
            rtol=0,
            atol=1e-12,
        )
        # Strictly larger than the old (buggy) full-residual-variance report.
        old = beta**2 * clip_and_adjust_variance(intercept, residual)[1]
        assert variance > old

    def test_legacy_all_present_variance_unchanged(self):
        """Legacy path with every predictor present keeps the full residual variance."""
        af = 0.3
        residual = compute_residual_variance(af, 0.5)
        beta = 0.1
        model = _imputed_model(
            beta=beta,
            intercept=0.2,
            allele_frequency=af,
            residual_variance=residual,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.3]),
        )
        prs, variance, _, _ = compute_imputed_prs({"rs_p0": 2.0, "rs_p1": 1.0}, [model])
        raw = 2 * 0.5 + 1 * (-0.3) + 0.2  # 0.9
        np.testing.assert_allclose(prs, raw * beta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            variance,
            beta**2 * clip_and_adjust_variance(raw, residual)[1],
            rtol=0,
            atol=1e-12,
        )
