"""Tests for projection prediction pipeline."""

import numpy as np
import pytest

from imputed_prs.core.types import (
    CalibrationParams,
    PredictionResult,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.models.projection_predictor import (
    ProjectionPredictor,
    compute_projected_prs,
)


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
):
    """Helper to create a ProjectionRegionModel with sensible defaults."""
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
