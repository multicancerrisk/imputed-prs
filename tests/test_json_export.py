"""Tests for JSON export functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.json_export import export_to_json


@pytest.fixture
def sample_observed_variants():
    """Create sample observed variants for testing."""
    return [
        VariantInfo("rs1", "1", 100, "A", "G", 0.1),
        VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        VariantInfo("rs3", "2", 300, "G", "A", -0.15),
    ]


@pytest.fixture
def sample_imputed_models():
    """Create sample imputed variant models for testing."""
    return [
        ImputedVariantModel(
            variant_id="rs4",
            chromosome="1",
            position=150,
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
        ),
        ImputedVariantModel(
            variant_id="rs5",
            chromosome="2",
            position=400,
            effect_allele="T",
            other_allele="C",
            beta=0.02,
            allele_frequency=0.5,
            imputation_r2=0.0,
            residual_variance=0.5,
            intercept=1.0,
            predictor_variant_ids=[],
            coefficients=np.array([]),
            is_intercept_only=True,
        ),
    ]


@pytest.fixture
def sample_calibration_params():
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


@pytest.fixture
def sample_evaluation_metrics():
    """Create sample evaluation metrics for testing."""
    return EvaluationMetrics(
        correlation=0.95,
        r2=0.90,
        mae=0.1,
        rmse=0.15,
        spearman_rho=0.94,
        calibration_slope=1.05,
        calibration_intercept=0.02,
    )


class TestBasicExport:
    """Tests for basic export functionality."""

    def test_basic_export_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic export with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            result_path = export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path == output_path
            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "observed_variants" in data
            assert "imputed_variants" in data
            assert "platform_variant_index" in data

    def test_export_with_calibration_params(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test export includes calibration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "calibration_params" in data
            assert data["calibration_params"]["scaling_factor"] == 1.1
            assert data["calibration_params"]["n_calibration"] == 500

    def test_export_with_evaluation_metrics(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test export includes evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "evaluation_metrics" in data
            assert data["evaluation_metrics"]["correlation"] == 0.95
            assert data["evaluation_metrics"]["r2"] == 0.90


class TestVarianceScaling:
    """Tests for variance scaling option."""

    def test_export_without_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export without variance scaling excludes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check that residual_variance is not included
            for imputed in data["imputed_variants"]:
                assert "residual_variance" not in imputed

            # Check metadata reflects this
            assert data["metadata"]["include_variance_scaling"] is False

    def test_export_with_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export with variance scaling includes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check that residual_variance is included
            for imputed in data["imputed_variants"]:
                assert "residual_variance" in imputed

            assert data["metadata"]["include_variance_scaling"] is True


class TestJSONValidity:
    """Tests for JSON validity and parsing."""

    def test_json_can_be_parsed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that output is valid JSON that can be parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            # This should not raise an exception
            with open(output_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_coefficients_are_lists_not_numpy(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that numpy arrays are converted to lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Find the model with coefficients
            model_with_coeffs = next(
                m for m in data["imputed_variants"] if not m["is_intercept_only"]
            )
            assert isinstance(model_with_coeffs["coefficients"], list)
            assert model_with_coeffs["coefficients"] == [0.3, 0.2]


class TestRoundTrip:
    """Tests for round-trip serialization."""

    def test_all_fields_present(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
        sample_evaluation_metrics,
    ):
        """Test that all fields are present after export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
                evaluation_metrics=sample_evaluation_metrics,
                platform_name="23andme_v5",
                prs_id="PGS000004",
                genome_build="GRCh37",
                model_name="Test PRS Model",
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check metadata
            assert data["metadata"]["prs_id"] == "PGS000004"
            assert data["metadata"]["platform_name"] == "23andme_v5"
            assert data["metadata"]["genome_build"] == "GRCh37"
            assert data["metadata"]["model_name"] == "Test PRS Model"

            # Check counts
            assert data["metadata"]["n_observed_variants"] == 3
            assert data["metadata"]["n_imputed_variants"] == 2

            # Check observed variants
            assert len(data["observed_variants"]) == 3
            assert data["observed_variants"][0]["variant_id"] == "rs1"

            # Check imputed variants
            assert len(data["imputed_variants"]) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_imputed_models_list(self, sample_observed_variants):
        """Test export with empty imputed models list (all observed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["metadata"]["n_imputed_variants"] == 0
            assert data["metadata"]["n_intercept_only"] == 0
            assert len(data["imputed_variants"]) == 0

    def test_empty_observed_variants_list(self, sample_imputed_models):
        """Test export with empty observed variants list (all imputed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["metadata"]["n_observed_variants"] == 0
            assert len(data["observed_variants"]) == 0
            assert data["platform_variant_index"] == {}

    def test_output_path_with_nonexistent_parent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that non-existent parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_model.json"
            result_path = export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path.exists()
            assert result_path.parent.exists()


class TestMetadata:
    """Tests for metadata content."""

    def test_intercept_only_count_in_metadata(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that intercept-only model count is correct in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            # One of the sample_imputed_models is intercept-only
            assert data["metadata"]["n_intercept_only"] == 1

    def test_format_version_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that format version is present in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["metadata"]["format_version"] == "1.0"

    def test_created_at_timestamp_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that created_at timestamp is present and valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "created_at" in data["metadata"]
            assert data["metadata"]["created_at"].endswith("Z")


class TestPlatformVariantIndex:
    """Tests for platform variant index."""

    def test_platform_variant_index_correct(self, sample_observed_variants):
        """Test that platform variant index maps correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["platform_variant_index"]["rs1"] == 0
            assert data["platform_variant_index"]["rs2"] == 1
            assert data["platform_variant_index"]["rs3"] == 2


class TestTrainingSummary:
    """Tests for training summary inclusion."""

    def test_export_with_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export includes training summary when provided."""
        training_summary = {
            "mean_r2": 0.75,
            "median_r2": 0.80,
            "n_high_quality": 100,
            "n_low_quality": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                training_summary=training_summary,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "training_summary" in data
            assert data["training_summary"]["mean_r2"] == 0.75
            assert data["training_summary"]["n_high_quality"] == 100

    def test_export_without_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export without training summary does not include the key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.json"
            export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "training_summary" not in data


class TestStringPath:
    """Tests for string path input."""

    def test_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_model.json"
            result_path = export_to_json(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert isinstance(result_path, Path)
            assert result_path.exists()
