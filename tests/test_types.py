"""Tests for core data types."""

import numpy as np
from dataclasses import asdict

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    PlatformInfo,
    PredictionResult,
    ProjectionRegionModel,
    VariantInfo,
)


class TestVariantInfo:
    """Tests for VariantInfo dataclass."""

    def test_instantiation(self):
        """Test creating a VariantInfo with valid data."""
        variant = VariantInfo(
            variant_id="rs123",
            chromosome="1",
            position=12345,
            effect_allele="A",
            other_allele="G",
            beta=0.5,
        )
        assert variant.variant_id == "rs123"
        assert variant.chromosome == "1"
        assert variant.position == 12345
        assert variant.effect_allele == "A"
        assert variant.other_allele == "G"
        assert variant.beta == 0.5

    def test_optional_other_allele(self):
        """Test that other_allele can be None."""
        variant = VariantInfo(
            variant_id="rs456",
            chromosome="2",
            position=67890,
            effect_allele="C",
            other_allele=None,
            beta=-0.3,
        )
        assert variant.other_allele is None

    def test_asdict_serialization(self):
        """Test that asdict serializes correctly."""
        variant = VariantInfo(
            variant_id="rs789",
            chromosome="X",
            position=11111,
            effect_allele="T",
            other_allele="A",
            beta=0.1,
        )
        d = asdict(variant)
        assert d == {
            "variant_id": "rs789",
            "chromosome": "X",
            "position": 11111,
            "effect_allele": "T",
            "other_allele": "A",
            "beta": 0.1,
            "fallback": None,
        }


class TestImputedVariantModel:
    """Tests for ImputedVariantModel dataclass."""

    def test_instantiation(self):
        """Test creating an ImputedVariantModel with valid data."""
        model = ImputedVariantModel(
            variant_id="rs123",
            chromosome="1",
            position=12345,
            effect_allele="A",
            other_allele="G",
            beta=0.5,
            allele_frequency=0.3,
            imputation_r2=0.85,
            residual_variance=0.063,
            intercept=0.6,
            predictor_variant_ids=["rs100", "rs101"],
            coefficients=np.array([0.2, 0.3]),
            is_intercept_only=False,
        )
        assert model.variant_id == "rs123"
        assert model.allele_frequency == 0.3
        assert model.imputation_r2 == 0.85
        assert len(model.predictor_variant_ids) == 2
        assert len(model.coefficients) == 2

    def test_default_values(self):
        """Test default values work correctly."""
        model = ImputedVariantModel(
            variant_id="rs123",
            chromosome="1",
            position=12345,
            effect_allele="A",
            other_allele=None,
            beta=0.5,
            allele_frequency=0.3,
            imputation_r2=0.0,
            residual_variance=0.42,
            intercept=0.6,
        )
        assert model.predictor_variant_ids == []
        assert len(model.coefficients) == 0
        assert model.is_intercept_only is False
        # Predictor allele metadata defaults to empty, index-aligned arrays.
        assert model.predictor_chromosomes == []
        assert model.predictor_positions == []
        assert model.predictor_counted_alleles == []
        assert model.predictor_other_alleles == []
        assert len(model.predictor_allele_frequencies) == 0

    def test_to_dict_serialization(self):
        """Test that to_dict converts numpy arrays to lists."""
        model = ImputedVariantModel(
            variant_id="rs123",
            chromosome="1",
            position=12345,
            effect_allele="A",
            other_allele="G",
            beta=0.5,
            allele_frequency=0.3,
            imputation_r2=0.85,
            residual_variance=0.063,
            intercept=0.6,
            predictor_variant_ids=["rs100"],
            coefficients=np.array([0.2, 0.3, 0.4]),
            is_intercept_only=False,
            predictor_chromosomes=["1"],
            predictor_positions=[11000],
            predictor_counted_alleles=["T"],
            predictor_other_alleles=["C"],
            predictor_allele_frequencies=np.array([0.25]),
        )
        d = model.to_dict()
        assert d["coefficients"] == [0.2, 0.3, 0.4]
        assert isinstance(d["coefficients"], list)
        # Predictor allele metadata round-trips; the AF array becomes a list.
        assert d["predictor_chromosomes"] == ["1"]
        assert d["predictor_positions"] == [11000]
        assert d["predictor_counted_alleles"] == ["T"]
        assert d["predictor_other_alleles"] == ["C"]
        assert d["predictor_allele_frequencies"] == [0.25]
        assert isinstance(d["predictor_allele_frequencies"], list)

    def test_intercept_only_model(self):
        """Test intercept-only model configuration."""
        model = ImputedVariantModel(
            variant_id="rs999",
            chromosome="5",
            position=50000,
            effect_allele="C",
            other_allele="T",
            beta=0.2,
            allele_frequency=0.4,
            imputation_r2=0.0,
            residual_variance=0.48,
            intercept=0.8,
            predictor_variant_ids=[],
            coefficients=np.array([]),
            is_intercept_only=True,
        )
        assert model.is_intercept_only is True
        assert len(model.predictor_variant_ids) == 0


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_instantiation(self):
        """Test creating a PredictionResult with valid data."""
        result = PredictionResult(
            prs=1.5,
            se=0.2,
            ci_lower=1.1,
            ci_upper=1.9,
            prs_observed_component=1.0,
            prs_imputed_component=0.5,
            n_variants_used=100,
            n_variants_imputed=20,
            n_variants_intercept_only=5,
            n_user_variants_missing=10,
            n_truncated=2,
        )
        assert result.prs == 1.5
        assert result.se == 0.2
        assert result.n_variants_used == 100
        assert result.n_variants_imputed == 20

    def test_default_scaled_values(self):
        """Test that scaled values default to None."""
        result = PredictionResult(
            prs=1.5,
            se=0.2,
            ci_lower=1.1,
            ci_upper=1.9,
            prs_observed_component=1.0,
            prs_imputed_component=0.5,
            n_variants_used=100,
            n_variants_imputed=20,
            n_variants_intercept_only=5,
            n_user_variants_missing=10,
            n_truncated=2,
        )
        assert result.prs_scaled is None
        assert result.se_scaled is None
        assert result.ci_lower_scaled is None
        assert result.ci_upper_scaled is None

    def test_with_scaled_values(self):
        """Test setting scaled values."""
        result = PredictionResult(
            prs=1.5,
            se=0.2,
            ci_lower=1.1,
            ci_upper=1.9,
            prs_observed_component=1.0,
            prs_imputed_component=0.5,
            n_variants_used=100,
            n_variants_imputed=20,
            n_variants_intercept_only=5,
            n_user_variants_missing=10,
            n_truncated=2,
            prs_scaled=1.8,
            se_scaled=0.25,
            ci_lower_scaled=1.3,
            ci_upper_scaled=2.3,
        )
        assert result.prs_scaled == 1.8
        assert result.se_scaled == 0.25

    def test_asdict_serialization(self):
        """Test asdict serialization."""
        result = PredictionResult(
            prs=1.5,
            se=0.2,
            ci_lower=1.1,
            ci_upper=1.9,
            prs_observed_component=1.0,
            prs_imputed_component=0.5,
            n_variants_used=100,
            n_variants_imputed=20,
            n_variants_intercept_only=5,
            n_user_variants_missing=10,
            n_truncated=2,
        )
        d = asdict(result)
        assert d["prs"] == 1.5
        assert d["n_variants_used"] == 100


class TestCalibrationParams:
    """Tests for CalibrationParams dataclass."""

    def test_instantiation(self):
        """Test creating CalibrationParams with valid data."""
        params = CalibrationParams(
            scaling_factor=1.1,
            scaling_factor_se=0.05,
            calibration_intercept=0.01,
            calibration_r2=0.95,
            sd_cv_predicted=0.9,
            sd_true=1.0,
            sd_scaled=0.99,
            attenuation_factor=0.9,
            n_calibration=1000,
        )
        assert params.scaling_factor == 1.1
        assert params.calibration_r2 == 0.95
        assert params.n_calibration == 1000

    def test_asdict_serialization(self):
        """Test asdict serialization."""
        params = CalibrationParams(
            scaling_factor=1.1,
            scaling_factor_se=0.05,
            calibration_intercept=0.01,
            calibration_r2=0.95,
            sd_cv_predicted=0.9,
            sd_true=1.0,
            sd_scaled=0.99,
            attenuation_factor=0.9,
            n_calibration=1000,
        )
        d = asdict(params)
        assert d["scaling_factor"] == 1.1
        assert d["n_calibration"] == 1000


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_instantiation(self):
        """Test creating EvaluationMetrics with valid data."""
        metrics = EvaluationMetrics(
            correlation=0.95,
            r2=0.90,
            mae=0.1,
            rmse=0.15,
            spearman_rho=0.94,
            calibration_slope=1.02,
            calibration_intercept=0.01,
        )
        assert metrics.correlation == 0.95
        assert metrics.r2 == 0.90
        assert metrics.mae == 0.1
        assert metrics.rmse == 0.15
        assert metrics.spearman_rho == 0.94

    def test_asdict_serialization(self):
        """Test asdict serialization."""
        metrics = EvaluationMetrics(
            correlation=0.95,
            r2=0.90,
            mae=0.1,
            rmse=0.15,
            spearman_rho=0.94,
            calibration_slope=1.02,
            calibration_intercept=0.01,
        )
        d = asdict(metrics)
        assert d["correlation"] == 0.95
        assert d["calibration_slope"] == 1.02


class TestPlatformInfo:
    """Tests for PlatformInfo dataclass."""

    def test_instantiation(self):
        """Test creating a PlatformInfo with valid data."""
        info = PlatformInfo(
            name="test_platform",
            display_name="Test Platform",
            description="A test genotyping platform",
            genome_build="GRCh37",
            n_variants=500000,
            chip_technology="Illumina GSA",
            company="Test Company",
            version="1",
        )
        assert info.name == "test_platform"
        assert info.display_name == "Test Platform"
        assert info.genome_build == "GRCh37"
        assert info.n_variants == 500000
        assert info.chip_technology == "Illumina GSA"
        assert info.company == "Test Company"
        assert info.version == "1"

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        info = PlatformInfo(
            name="test_platform",
            display_name="Test Platform",
            description="A test genotyping platform",
            genome_build="GRCh37",
            n_variants=500000,
            chip_technology="Illumina GSA",
            company="Test Company",
            version="1",
        )
        assert info.date_introduced is None
        assert info.source_url is None

    def test_with_optional_fields(self):
        """Test setting optional fields."""
        info = PlatformInfo(
            name="test_platform",
            display_name="Test Platform",
            description="A test genotyping platform",
            genome_build="GRCh37",
            n_variants=500000,
            chip_technology="Illumina GSA",
            company="Test Company",
            version="1",
            date_introduced="2020-01",
            source_url="https://example.com",
        )
        assert info.date_introduced == "2020-01"
        assert info.source_url == "https://example.com"

    def test_asdict_serialization(self):
        """Test asdict serialization."""
        info = PlatformInfo(
            name="test_platform",
            display_name="Test Platform",
            description="A test genotyping platform",
            genome_build="GRCh37",
            n_variants=500000,
            chip_technology="Illumina GSA",
            company="Test Company",
            version="1",
            date_introduced="2020-01",
            source_url="https://example.com",
        )
        d = asdict(info)
        assert d["name"] == "test_platform"
        assert d["n_variants"] == 500000
        assert d["date_introduced"] == "2020-01"


class TestProjectionRegionModel:
    """Tests for the ProjectionRegionModel dataclass."""

    def _model(self, **overrides):
        kwargs = dict(
            region_id="chr1:1-2",
            chromosome="1",
            start=1,
            end=2,
            prs_variant_ids=["rs2000"],
            betas=np.array([0.3]),
            predictor_variant_ids=["rs1000"],
            coefficients=np.array([0.2]),
            intercept=0.1,
            cv_mse=0.01,
            cv_r2=0.8,
            is_intercept_only=False,
            mean_prs_contribution=0.5,
            predictor_allele_frequencies=np.array([0.3]),
        )
        kwargs.update(overrides)
        return ProjectionRegionModel(**kwargs)

    def test_target_variance_defaults_to_zero(self):
        """target_variance (P3.3) is optional and defaults to 0.0 for back-compat."""
        assert self._model().target_variance == 0.0

    def test_target_variance_round_trips_through_to_dict(self):
        """to_dict carries target_variance (a plain float handled by asdict)."""
        d = self._model(target_variance=0.42).to_dict()
        assert d["target_variance"] == 0.42
