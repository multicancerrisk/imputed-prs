"""Tests for projection prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.types import (
    CalibrationParams,
    PredictionResult,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.io.user_genotypes import load_raw_user_genotypes
from imputed_prs.models.projection_predictor import (
    ProjectionPredictor,
    compute_projected_prs,
    compute_projected_prs_oriented,
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
    ``("G", "A", 1) -> "AG"``.
    """
    return alt * alt_dosage + ref * (2 - alt_dosage)


def _make_region_model(
    region_id="chr1:1000000-3000000",
    chromosome="1",
    start=1_000_000,
    end=3_000_000,
    prs_variant_ids=None,
    betas=None,
    predictor_variant_ids=None,
    coefficients=None,
    intercept=0.1,
    cv_mse=0.01,
    cv_r2=0.8,
    is_intercept_only=False,
    mean_prs_contribution=0.5,
    predictor_allele_frequencies=None,
    predictor_chromosomes=None,
    predictor_positions=None,
    predictor_counted_alleles=None,
    predictor_other_alleles=None,
):
    """Helper to create a ProjectionRegionModel with sensible defaults.

    P1.3 predictor allele metadata defaults are length-aligned to
    ``predictor_variant_ids`` (counted = ALT, other = REF) so the oriented scorer
    can resolve every predictor; override per-test to control orientation.
    """
    if prs_variant_ids is None:
        prs_variant_ids = ["rs2000"]
    if betas is None:
        betas = np.array([0.3])
    if predictor_variant_ids is None:
        predictor_variant_ids = ["rs1000", "rs1001", "rs1002"]
    if coefficients is None:
        coefficients = np.array([0.2, -0.1, 0.15])
    if predictor_allele_frequencies is None:
        predictor_allele_frequencies = np.array([0.3, 0.4, 0.25])
    n = len(predictor_variant_ids)
    if predictor_chromosomes is None:
        predictor_chromosomes = [chromosome] * n
    if predictor_positions is None:
        predictor_positions = [start + 1000 * (i + 1) for i in range(n)]
    if predictor_counted_alleles is None:
        predictor_counted_alleles = ["A"] * n
    if predictor_other_alleles is None:
        predictor_other_alleles = ["G"] * n
    return ProjectionRegionModel(
        region_id=region_id,
        chromosome=chromosome,
        start=start,
        end=end,
        prs_variant_ids=prs_variant_ids,
        betas=betas,
        predictor_variant_ids=predictor_variant_ids,
        coefficients=coefficients,
        intercept=intercept,
        cv_mse=cv_mse,
        cv_r2=cv_r2,
        is_intercept_only=is_intercept_only,
        mean_prs_contribution=mean_prs_contribution,
        predictor_allele_frequencies=predictor_allele_frequencies,
        predictor_chromosomes=predictor_chromosomes,
        predictor_positions=predictor_positions,
        predictor_counted_alleles=predictor_counted_alleles,
        predictor_other_alleles=predictor_other_alleles,
    )


def _make_observed_variants():
    """Helper to create sample observed variants."""
    return [
        VariantInfo(
            variant_id="rs100",
            chromosome="1",
            position=500_000,
            effect_allele="A",
            other_allele="G",
            beta=0.5,
        ),
        VariantInfo(
            variant_id="rs101",
            chromosome="1",
            position=600_000,
            effect_allele="C",
            other_allele="T",
            beta=-0.3,
        ),
    ]


def _make_calibration_params():
    """Helper to create sample calibration params."""
    return CalibrationParams(
        scaling_factor=1.05,
        scaling_factor_se=0.02,
        calibration_intercept=0.01,
        calibration_r2=0.98,
        sd_cv_predicted=0.5,
        sd_true=0.52,
        sd_scaled=0.525,
        attenuation_factor=0.96,
        n_calibration=200,
    )


class TestComputeProjectedPrs:
    """Tests for the compute_projected_prs function."""

    def test_basic_calculation(self):
        """Known coefficients and intercept: verify z^T a + intercept."""
        model = _make_region_model(
            coefficients=np.array([0.5, -0.2]),
            intercept=0.1,
            predictor_variant_ids=["rs1", "rs2"],
            predictor_allele_frequencies=np.array([0.3, 0.4]),
            cv_mse=0.05,
        )
        user_dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, variance, n_regions, n_sub = compute_projected_prs(
            user_dosages, [model],
        )

        # z^T a + intercept = 2.0*0.5 + 1.0*(-0.2) + 0.1 = 0.9
        assert prs == pytest.approx(0.9)
        assert variance == pytest.approx(0.05)
        assert n_regions == 1
        assert n_sub == 0

    def test_intercept_only_region(self):
        """Region with is_intercept_only=True uses intercept value."""
        model = _make_region_model(
            is_intercept_only=True,
            intercept=0.42,
            coefficients=np.array([]),
            predictor_variant_ids=[],
            predictor_allele_frequencies=np.array([]),
            cv_mse=0.02,
        )

        prs, variance, n_regions, n_sub = compute_projected_prs({}, [model])

        assert prs == pytest.approx(0.42)
        assert variance == pytest.approx(0.02)
        assert n_regions == 1
        assert n_sub == 0

    def test_missing_predictor_substitution(self):
        """Missing user variant -> substituted with 2*AF."""
        model = _make_region_model(
            coefficients=np.array([0.5, -0.2]),
            intercept=0.1,
            predictor_variant_ids=["rs1", "rs2"],
            predictor_allele_frequencies=np.array([0.3, 0.4]),
        )
        # rs2 is missing -> substituted with 2*0.4 = 0.8
        user_dosages = {"rs1": 2.0}

        prs, _, _, n_sub = compute_projected_prs(user_dosages, [model])

        expected = 2.0 * 0.5 + 0.8 * (-0.2) + 0.1
        assert prs == pytest.approx(expected)
        assert n_sub == 1

    def test_all_predictors_missing(self):
        """All predictors missing -> all substituted with 2*AF."""
        model = _make_region_model(
            coefficients=np.array([0.5, -0.2]),
            intercept=0.1,
            predictor_variant_ids=["rs1", "rs2"],
            predictor_allele_frequencies=np.array([0.3, 0.4]),
        )

        prs, _, _, n_sub = compute_projected_prs({}, [model])

        # 2*0.3*0.5 + 2*0.4*(-0.2) + 0.1 = 0.3 - 0.16 + 0.1 = 0.24
        expected = 2 * 0.3 * 0.5 + 2 * 0.4 * (-0.2) + 0.1
        assert prs == pytest.approx(expected)
        assert n_sub == 2

    def test_no_dosage_clipping(self):
        """Projected value can be negative: verify no clipping to [0, 2]."""
        model = _make_region_model(
            coefficients=np.array([-1.0]),
            intercept=-0.5,
            predictor_variant_ids=["rs1"],
            predictor_allele_frequencies=np.array([0.3]),
        )
        user_dosages = {"rs1": 2.0}

        prs, _, _, _ = compute_projected_prs(user_dosages, [model])

        # -1.0 * 2.0 + (-0.5) = -2.5
        assert prs == pytest.approx(-2.5)
        assert prs < 0  # Explicitly verify negative

    def test_empty_region_models(self):
        """Empty list -> (0.0, 0.0, 0, 0)."""
        prs, variance, n_regions, n_sub = compute_projected_prs({"rs1": 1.0}, [])

        assert prs == 0.0
        assert variance == 0.0
        assert n_regions == 0
        assert n_sub == 0

    def test_multiple_regions(self):
        """Two regions: verify sum is correct."""
        model1 = _make_region_model(
            region_id="chr1:0-2000000",
            coefficients=np.array([0.5]),
            intercept=0.1,
            predictor_variant_ids=["rs1"],
            predictor_allele_frequencies=np.array([0.3]),
            cv_mse=0.02,
        )
        model2 = _make_region_model(
            region_id="chr1:5000000-7000000",
            coefficients=np.array([0.3]),
            intercept=-0.05,
            predictor_variant_ids=["rs10"],
            predictor_allele_frequencies=np.array([0.5]),
            cv_mse=0.03,
        )
        user_dosages = {"rs1": 1.0, "rs10": 2.0}

        prs, variance, n_regions, _ = compute_projected_prs(
            user_dosages, [model1, model2],
        )

        expected1 = 1.0 * 0.5 + 0.1  # 0.6
        expected2 = 2.0 * 0.3 + (-0.05)  # 0.55
        assert prs == pytest.approx(expected1 + expected2)
        assert variance == pytest.approx(0.02 + 0.03)
        assert n_regions == 2

    def test_variance_from_cv_mse(self):
        """total_variance == sum of region.cv_mse."""
        models = [
            _make_region_model(region_id=f"chr1:{i}M-{i+2}M", cv_mse=0.01 * (i + 1))
            for i in range(3)
        ]

        _, variance, _, _ = compute_projected_prs(
            {"rs1000": 1.0, "rs1001": 1.0, "rs1002": 1.0}, models,
        )

        assert variance == pytest.approx(0.01 + 0.02 + 0.03)


class TestComputeProjectedPrsOriented:
    """Allele-aware projected scoring (P1.4): counts predictor ALT alleles, no clip."""

    def test_full_predictor_set_matches_reference_dot_product(self):
        """Genotype-string scoring equals the oriented reference ALT-dosage dot product."""
        model = _make_region_model(
            predictor_variant_ids=["rs_p0", "rs_p1", "rs_p2"],
            coefficients=np.array([0.2, -0.1, 0.15]),
            intercept=0.05,
            cv_mse=0.02,
            predictor_allele_frequencies=np.array([0.3, 0.4, 0.25]),
            predictor_chromosomes=["1", "1", "1"],
            predictor_positions=[1_000_001, 1_000_002, 1_000_003],
            predictor_counted_alleles=["A", "C", "T"],  # ALT
            predictor_other_alleles=["G", "T", "C"],  # REF
        )
        # Reference ALT dosages [2, 1, 0].
        coll = _collection(
            [
                ("rs_p0", "1", 1_000_001, _render_genotype("G", "A", 2)),  # "AA"
                ("rs_p1", "1", 1_000_002, _render_genotype("T", "C", 1)),  # "CT"
                ("rs_p2", "1", 1_000_003, _render_genotype("C", "T", 0)),  # "CC"
            ]
        )
        prs, variance, n_regions, n_sub = compute_projected_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # 2*0.2 + 1*(-0.1) + 0*0.15 + 0.05 = 0.35 (no clipping on the projection path).
        np.testing.assert_allclose(prs, 0.35, rtol=0, atol=1e-12)
        np.testing.assert_allclose(variance, 0.02, rtol=0, atol=1e-12)
        assert n_regions == 1
        assert n_sub == 0

    def test_alt_ref_orientation_bugfix(self):
        """A 'GG' call with ALT=A counts 0 copies, not homozygous 2 (the bug)."""
        model = _make_region_model(
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            intercept=0.0,
            cv_mse=0.01,
            predictor_allele_frequencies=np.array([0.1]),
            predictor_chromosomes=["1"],
            predictor_positions=[1_000_001],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
        )
        coll = _collection([("rs_p0", "1", 1_000_001, "GG")])
        prs, _, _, n_sub = compute_projected_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        np.testing.assert_allclose(prs, 0.0, rtol=0, atol=1e-12)
        assert n_sub == 0  # resolved to 0 copies, not substituted
        # Contrast: the allele-blind dosage-dict path counts "GG" homozygous -> 2.
        blind, _, _, _ = compute_projected_prs({"rs_p0": 2.0}, [model])
        np.testing.assert_allclose(blind, 2.0, rtol=0, atol=1e-12)

    def test_missing_predictor_mean_substitution(self):
        """A missing predictor is substituted with 2*AF and counted in n_substituted."""
        model = _make_region_model(
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.2]),
            intercept=0.1,
            cv_mse=0.01,
            predictor_allele_frequencies=np.array([0.3, 0.4]),
            predictor_chromosomes=["1", "1"],
            predictor_positions=[1_000_001, 1_000_002],
            predictor_counted_alleles=["A", "C"],
            predictor_other_alleles=["G", "T"],
        )
        # rs_p0 present (dosage 2 -> "AA"); rs_p1 omitted -> 2*0.4 = 0.8.
        coll = _collection([("rs_p0", "1", 1_000_001, "AA")])
        prs, _, _, n_sub = compute_projected_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # 2*0.5 + 0.8*(-0.2) + 0.1 = 0.94.
        np.testing.assert_allclose(prs, 0.94, rtol=0, atol=1e-12)
        assert n_sub == 1

    def test_no_clipping_projection(self):
        """A prediction above 2 is NOT clipped (target is a PRS contribution)."""
        model = _make_region_model(
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([3.0]),
            intercept=1.0,
            cv_mse=0.01,
            predictor_allele_frequencies=np.array([0.3]),
            predictor_chromosomes=["1"],
            predictor_positions=[1_000_001],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
        )
        coll = _collection([("rs_p0", "1", 1_000_001, "AA")])  # dosage 2
        prs, _, _, _ = compute_projected_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        # raw = 2*3.0 + 1.0 = 7.0, left unclipped.
        np.testing.assert_allclose(prs, 7.0, rtol=0, atol=1e-12)

    def test_heterozygote_order_invariant(self):
        """'AG' and 'GA' both count one copy of the ALT allele."""
        model = _make_region_model(
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            intercept=0.0,
            cv_mse=0.01,
            predictor_allele_frequencies=np.array([0.3]),
            predictor_chromosomes=["1"],
            predictor_positions=[1_000_001],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
        )
        for geno in ("AG", "GA"):
            coll = _collection([("rs_p0", "1", 1_000_001, geno)])
            prs, _, _, _ = compute_projected_prs_oriented(
                coll, [model], allow_ambiguous=True
            )
            np.testing.assert_allclose(prs, 1.0, rtol=0, atol=1e-12)

    def test_intercept_only_unchanged(self):
        """An intercept-only region scores its intercept and reads no metadata."""
        model = _make_region_model(
            predictor_variant_ids=[],
            coefficients=np.array([]),
            intercept=0.42,
            cv_mse=0.01,
            is_intercept_only=True,
            predictor_allele_frequencies=np.array([]),
        )
        coll = _collection([("rs_x", "1", 999, "AA")])  # irrelevant
        prs, variance, n_regions, n_sub = compute_projected_prs_oriented(
            coll, [model], allow_ambiguous=True
        )
        np.testing.assert_allclose(prs, 0.42, rtol=0, atol=1e-12)
        np.testing.assert_allclose(variance, 0.01, rtol=0, atol=1e-12)
        assert n_regions == 1
        assert n_sub == 0


class TestProjectionPredictor:
    """Tests for the ProjectionPredictor class."""

    def test_basic_prediction_mixed(self):
        """Both observed and projected components contribute."""
        observed = _make_observed_variants()
        model = _make_region_model(
            coefficients=np.array([0.4]),
            intercept=0.05,
            predictor_variant_ids=["rs500"],
            predictor_allele_frequencies=np.array([0.3]),
            cv_mse=0.01,
        )
        predictor = ProjectionPredictor(observed, [model])

        user = {"rs100": 1.0, "rs101": 2.0, "rs500": 1.0}
        result = predictor.predict(user, apply_calibration=False)

        # Observed: 1.0*0.5 + 2.0*(-0.3) = -0.1
        # Projected: 1.0*0.4 + 0.05 = 0.45
        assert result.prs_observed_component == pytest.approx(-0.1)
        assert result.prs_imputed_component == pytest.approx(0.45)
        assert result.prs == pytest.approx(-0.1 + 0.45)

    def test_legacy_dict_path_leaves_oriented_diagnostics_none(self):
        """The allele-blind dosage-dict path does not populate oriented fields."""
        observed = _make_observed_variants()
        predictor = ProjectionPredictor(observed, [])

        result = predictor.predict({"rs100": 2.0, "rs101": 0.0}, apply_calibration=False)

        assert result.n_observed_scored_direct is None
        assert result.unresolved_observed_ids is None

    def test_oriented_observed_via_raw_genotypes(self):
        """raw_genotypes routes observed AND predictor scoring through the oriented path."""
        observed = _make_observed_variants()  # rs100=(A,G,0.5), rs101=(C,T,-0.3)
        model = _make_region_model(
            coefficients=np.array([0.4]),
            intercept=0.05,
            predictor_variant_ids=["rs500"],
            predictor_allele_frequencies=np.array([0.3]),
            predictor_chromosomes=["1"],
            predictor_positions=[2_000_000],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
            cv_mse=0.01,
        )
        predictor = ProjectionPredictor(observed, [model])

        # With raw_genotypes the dosage dict is ignored; everything is scored from
        # strings: observed rs100 "GG" -> 0 copies of effect A, rs101 "CC" -> 2 copies
        # of effect C; predictor rs500 "AG" -> 1 copy of ALT A.
        coll = _collection(
            [
                ("rs100", "1", 500_000, "GG"),
                ("rs101", "1", 600_000, "CC"),
                ("rs500", "1", 2_000_000, "AG"),
            ]
        )
        result = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)

        # Oriented observed = 0*0.5 + 2*(-0.3) = -0.6 (allele-blind would give 0.4).
        np.testing.assert_allclose(
            result.prs_observed_component, -0.6, rtol=0, atol=1e-12
        )
        # Oriented projected = 1*0.4 + 0.05 = 0.45.
        np.testing.assert_allclose(
            result.prs_imputed_component, 0.45, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(result.prs, -0.15, rtol=0, atol=1e-12)
        assert result.n_observed_scored_direct == 2
        assert result.unresolved_observed_ids == ()

    def test_predict_projected_via_raw_genotypes_matches_oriented(self):
        """predict() routes the projected component through the oriented scorer."""
        # A 'GG' predictor with ALT=A: oriented counts 0; the blind dict counts 2.
        model = _make_region_model(
            predictor_variant_ids=["rs_p0"],
            coefficients=np.array([1.0]),
            intercept=0.0,
            cv_mse=0.0,
            predictor_allele_frequencies=np.array([0.1]),
            predictor_chromosomes=["1"],
            predictor_positions=[1_000_001],
            predictor_counted_alleles=["A"],
            predictor_other_alleles=["G"],
        )
        predictor = ProjectionPredictor([], [model])
        coll = _collection([("rs_p0", "1", 1_000_001, "GG")])
        result = predictor.predict({}, apply_calibration=False, raw_genotypes=coll)
        # Projected component is exposed as prs_imputed_component.
        np.testing.assert_allclose(
            result.prs_imputed_component, 0.0, rtol=0, atol=1e-12
        )
        # The legacy dosage-dict path (no raw_genotypes) counts "GG" homozygous -> 2.
        blind = predictor.predict({"rs_p0": 2.0}, apply_calibration=False)
        np.testing.assert_allclose(
            blind.prs_imputed_component, 2.0, rtol=0, atol=1e-12
        )

    def test_all_observed_no_projection(self):
        """No region models: PRS == observed component."""
        observed = _make_observed_variants()
        predictor = ProjectionPredictor(observed, [])

        user = {"rs100": 2.0, "rs101": 0.0}
        result = predictor.predict(user, apply_calibration=False)

        assert result.prs == pytest.approx(2.0 * 0.5 + 0.0 * (-0.3))
        assert result.prs_imputed_component == 0.0
        assert result.se == 0.0

    def test_all_projected_no_observed(self):
        """No observed variants: PRS == projected component."""
        model = _make_region_model(
            coefficients=np.array([0.3]),
            intercept=0.1,
            predictor_variant_ids=["rs500"],
            predictor_allele_frequencies=np.array([0.3]),
        )
        predictor = ProjectionPredictor([], [model])

        user = {"rs500": 1.0}
        result = predictor.predict(user, apply_calibration=False)

        expected = 1.0 * 0.3 + 0.1
        assert result.prs == pytest.approx(expected)
        assert result.prs_observed_component == 0.0

    def test_with_calibration(self):
        """Calibration scaling applied correctly."""
        observed = _make_observed_variants()
        model = _make_region_model(cv_mse=0.04)
        cal = _make_calibration_params()
        predictor = ProjectionPredictor(observed, [model], calibration_params=cal)

        user = {"rs100": 1.0, "rs101": 1.0, "rs1000": 1.0, "rs1001": 1.0, "rs1002": 1.0}
        result = predictor.predict(user, apply_calibration=True)

        # Verify scaling
        assert result.prs_scaled is not None
        assert result.prs_scaled == pytest.approx(
            cal.scaling_factor * result.prs + cal.calibration_intercept
        )
        se = np.sqrt(0.04)
        assert result.se_scaled == pytest.approx(abs(cal.scaling_factor) * se)

    def test_without_calibration(self):
        """No scaling when apply_calibration=False or params=None."""
        observed = _make_observed_variants()
        model = _make_region_model()

        # With params but apply=False
        cal = _make_calibration_params()
        predictor = ProjectionPredictor(observed, [model], calibration_params=cal)
        user = {"rs100": 1.0, "rs101": 1.0, "rs1000": 1.0, "rs1001": 1.0, "rs1002": 1.0}
        result = predictor.predict(user, apply_calibration=False)
        assert result.prs_scaled is None

        # Without params
        predictor2 = ProjectionPredictor(observed, [model])
        result2 = predictor2.predict(user, apply_calibration=True)
        assert result2.prs_scaled is None

    def test_confidence_interval(self):
        """CI = prs +/- 1.96 * SE."""
        model = _make_region_model(cv_mse=0.04)
        predictor = ProjectionPredictor([], [model])

        user = {"rs1000": 1.0, "rs1001": 1.0, "rs1002": 1.0}
        result = predictor.predict(user, apply_calibration=False)

        se = np.sqrt(0.04)
        assert result.se == pytest.approx(se)
        assert result.ci_lower == pytest.approx(result.prs - 1.96 * se)
        assert result.ci_upper == pytest.approx(result.prs + 1.96 * se)

    def test_returns_prediction_result_type(self):
        """Return type is PredictionResult (same as imputation)."""
        predictor = ProjectionPredictor([], [])
        result = predictor.predict({}, apply_calibration=False)
        assert isinstance(result, PredictionResult)
