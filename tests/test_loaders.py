"""Tests for model loader functionality."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.arrow_export import (
    export_to_arrow,
    export_to_parquet,
)
from imputed_prs.io.exporters.csv_export import export_variant_table
from imputed_prs.io.exporters.hdf5_export import export_to_hdf5
from imputed_prs.io.exporters.json_export import export_to_json
from imputed_prs.io.loaders import load_model_hdf5, load_model_json
from imputed_prs.io.user_genotypes import load_raw_user_genotypes
from imputed_prs.models.predictor import compute_imputed_prs_oriented


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
            predictor_chromosomes=["1", "1"],
            predictor_positions=[100, 200],
            predictor_counted_alleles=["G", "T"],
            predictor_other_alleles=["A", "C"],
            predictor_allele_frequencies=np.array([0.4, 0.3]),
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


class TestJSONLoader:
    """Tests for JSON loader functionality."""

    def test_basic_json_loading_with_required_keys(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic JSON loading with all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            data = load_model_json(json_path)

            assert "metadata" in data
            assert "observed_variants" in data
            assert "imputed_variants" in data
            assert len(data["observed_variants"]) == 3
            assert len(data["imputed_variants"]) == 2

    def test_json_loading_raises_value_error_for_missing_keys(self):
        """Test JSON loading raises ValueError for missing required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "invalid_model.json"
            # Write JSON with missing required keys
            with open(json_path, "w") as f:
                json.dump({"metadata": {}, "observed_variants": []}, f)

            with pytest.raises(ValueError) as exc_info:
                load_model_json(json_path)

            assert "imputed_variants" in str(exc_info.value)

    def test_json_loading_raises_file_not_found_for_nonexistent_file(self):
        """Test JSON loading raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_model_json("/nonexistent/path/model.json")

    def test_json_loading_with_calibration_params(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test JSON loading includes calibration params when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            data = load_model_json(json_path)

            assert "calibration_params" in data
            assert data["calibration_params"]["scaling_factor"] == 1.1
            assert data["calibration_params"]["n_calibration"] == 500

    def test_json_loading_with_evaluation_metrics(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test JSON loading includes evaluation metrics when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            data = load_model_json(json_path)

            assert "evaluation_metrics" in data
            assert data["evaluation_metrics"]["correlation"] == 0.95
            assert data["evaluation_metrics"]["r2"] == 0.90

    def test_json_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly for JSON loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = f"{tmpdir}/model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            data = load_model_json(json_path)

            assert "metadata" in data
            assert len(data["observed_variants"]) == 3


class TestHDF5Loader:
    """Tests for HDF5 loader functionality."""

    def test_basic_hdf5_loading_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic HDF5 loading with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            obs, imp, calib, metrics, meta = load_model_hdf5(hdf5_path)

            assert len(obs) == 3
            assert len(imp) == 2
            assert calib is None
            assert metrics is None
            assert "format_version" in meta

    def test_hdf5_loading_reconstructs_variant_info_correctly(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading reconstructs VariantInfo correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            obs, _, _, _, _ = load_model_hdf5(hdf5_path)

            # Find rs1 and verify all fields
            rs1 = next(v for v in obs if v.variant_id == "rs1")
            assert rs1.chromosome == "1"
            assert rs1.position == 100
            assert rs1.effect_allele == "A"
            assert rs1.other_allele == "G"
            assert rs1.beta == 0.1

    def test_hdf5_loading_reconstructs_imputed_model_with_coefficients(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading reconstructs ImputedVariantModel with coefficients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            _, imp, _, _, _ = load_model_hdf5(hdf5_path)

            # Find rs4 and verify coefficients
            rs4 = next(m for m in imp if m.variant_id == "rs4")
            assert rs4.chromosome == "1"
            assert rs4.position == 150
            assert rs4.effect_allele == "A"
            assert rs4.other_allele == "G"
            assert rs4.beta == 0.05
            assert rs4.allele_frequency == 0.3
            assert rs4.imputation_r2 == 0.8
            assert rs4.residual_variance == 0.1
            assert rs4.intercept == 0.6
            assert rs4.is_intercept_only is False
            assert len(rs4.predictor_variant_ids) == 2
            assert set(rs4.predictor_variant_ids) == {"rs1", "rs2"}
            assert len(rs4.coefficients) == 2

    def test_hdf5_loading_reconstructs_calibration_params_when_present(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test HDF5 loading reconstructs CalibrationParams when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            _, _, calib, _, _ = load_model_hdf5(hdf5_path)

            assert calib is not None
            assert calib.scaling_factor == 1.1
            assert calib.scaling_factor_se == 0.05
            assert calib.calibration_intercept == 0.01
            assert calib.calibration_r2 == 0.95
            assert calib.sd_cv_predicted == 0.5
            assert calib.sd_true == 0.55
            assert calib.sd_scaled == 0.55
            assert calib.attenuation_factor == 0.91
            assert calib.n_calibration == 500

    def test_hdf5_loading_reconstructs_evaluation_metrics_when_present(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test HDF5 loading reconstructs EvaluationMetrics when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            _, _, _, metrics, _ = load_model_hdf5(hdf5_path)

            assert metrics is not None
            assert metrics.correlation == 0.95
            assert metrics.r2 == 0.90
            assert metrics.mae == 0.1
            assert metrics.rmse == 0.15
            assert metrics.spearman_rho == 0.94
            assert metrics.calibration_slope == 1.05
            assert metrics.calibration_intercept == 0.02

    def test_hdf5_loading_handles_missing_calibration_params(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading handles missing calibration params (returns None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            _, _, calib, _, _ = load_model_hdf5(hdf5_path)

            assert calib is None

    def test_hdf5_loading_handles_empty_variant_lists(self):
        """Test HDF5 loading handles empty variant lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=[],
                imputed_models=[],
            )

            obs, imp, _, _, meta = load_model_hdf5(hdf5_path)

            assert len(obs) == 0
            assert len(imp) == 0
            assert meta["n_observed_variants"] == 0
            assert meta["n_imputed_variants"] == 0

    def test_hdf5_loading_preserves_numpy_array_dtypes(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading preserves numpy array dtypes for coefficients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            _, imp, _, _, _ = load_model_hdf5(hdf5_path)

            # Find rs4 which has coefficients
            rs4 = next(m for m in imp if m.variant_id == "rs4")
            assert isinstance(rs4.coefficients, np.ndarray)
            assert rs4.coefficients.dtype == np.float64

    def test_hdf5_loading_handles_none_other_allele(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading handles None other_allele (stored as empty string)."""
        # Create variant with None other_allele
        observed_with_none = [
            VariantInfo("rs99", "1", 500, "A", None, 0.1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=observed_with_none,
                imputed_models=sample_imputed_models,
            )

            obs, _, _, _, _ = load_model_hdf5(hdf5_path)

            assert obs[0].other_allele is None

    def test_hdf5_loading_reconstructs_training_summary_in_metadata(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 loading reconstructs training_summary in metadata."""
        training_summary = {
            "mean_r2": 0.75,
            "median_r2": 0.80,
            "n_high_quality": 100,
            "n_low_quality": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                training_summary=training_summary,
            )

            _, _, _, _, meta = load_model_hdf5(hdf5_path)

            assert "training_summary" in meta
            assert meta["training_summary"]["mean_r2"] == 0.75
            assert meta["training_summary"]["n_high_quality"] == 100

    def test_hdf5_loading_raises_key_error_for_missing_groups(self):
        """Test HDF5 loading raises KeyError for missing required groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "invalid_model.h5"
            # Create HDF5 file with missing groups
            with h5py.File(hdf5_path, "w") as f:
                f.create_group("metadata")
                # Missing observed_variants, imputed_variants, coefficients

            with pytest.raises(KeyError) as exc_info:
                load_model_hdf5(hdf5_path)

            assert "Missing required groups" in str(exc_info.value)

    def test_hdf5_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly for HDF5 loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = f"{tmpdir}/model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            obs, imp, _, _, _ = load_model_hdf5(hdf5_path)

            assert len(obs) == 3
            assert len(imp) == 2


class TestRoundTrip:
    """Tests for round-trip serialization."""

    def test_hdf5_round_trip_data_integrity(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
        sample_evaluation_metrics,
    ):
        """Test round-trip: export to HDF5 and load back, verify data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
                evaluation_metrics=sample_evaluation_metrics,
                platform_name="23andme_v5",
                prs_id="PGS000004",
                genome_build="GRCh37",
            )

            obs, imp, calib, metrics, meta = load_model_hdf5(hdf5_path)

            # Verify observed variants
            assert len(obs) == len(sample_observed_variants)
            for orig, loaded in zip(sample_observed_variants, obs):
                assert orig.variant_id == loaded.variant_id
                assert orig.chromosome == loaded.chromosome
                assert orig.position == loaded.position
                assert orig.effect_allele == loaded.effect_allele
                assert orig.other_allele == loaded.other_allele
                assert orig.beta == loaded.beta

            # Verify imputed models
            assert len(imp) == len(sample_imputed_models)
            for orig in sample_imputed_models:
                loaded = next(m for m in imp if m.variant_id == orig.variant_id)
                assert orig.chromosome == loaded.chromosome
                assert orig.position == loaded.position
                assert orig.effect_allele == loaded.effect_allele
                assert orig.beta == loaded.beta
                assert orig.allele_frequency == loaded.allele_frequency
                assert orig.imputation_r2 == loaded.imputation_r2
                assert orig.is_intercept_only == loaded.is_intercept_only
                assert len(orig.predictor_variant_ids) == len(loaded.predictor_variant_ids)
                assert np.allclose(orig.coefficients, loaded.coefficients)

            # Verify calibration params
            assert calib is not None
            assert calib.scaling_factor == sample_calibration_params.scaling_factor
            assert calib.n_calibration == sample_calibration_params.n_calibration

            # Verify evaluation metrics
            assert metrics is not None
            assert metrics.correlation == sample_evaluation_metrics.correlation
            assert metrics.r2 == sample_evaluation_metrics.r2

            # Verify metadata
            assert meta["platform_name"] == "23andme_v5"
            assert meta["prs_id"] == "PGS000004"
            assert meta["genome_build"] == "GRCh37"

    def test_json_round_trip_structure_verification(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test round-trip: export to JSON and load back, verify structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
                prs_id="PGS000004",
            )

            data = load_model_json(json_path)

            # Verify structure
            assert "metadata" in data
            assert "observed_variants" in data
            assert "imputed_variants" in data
            assert "platform_variant_index" in data
            assert "calibration_params" in data

            # Verify metadata
            assert data["metadata"]["prs_id"] == "PGS000004"
            assert data["metadata"]["n_observed_variants"] == 3
            assert data["metadata"]["n_imputed_variants"] == 2

            # Verify observed variants
            assert len(data["observed_variants"]) == 3
            rs1 = next(v for v in data["observed_variants"] if v["variant_id"] == "rs1")
            assert rs1["chromosome"] == "1"
            assert rs1["position"] == 100
            assert rs1["beta"] == 0.1

            # Verify imputed variants
            assert len(data["imputed_variants"]) == 2
            rs4 = next(v for v in data["imputed_variants"] if v["variant_id"] == "rs4")
            assert rs4["allele_frequency"] == 0.3
            assert rs4["imputation_r2"] == 0.8
            assert [p["variant_id"] for p in rs4["predictors"]] == ["rs1", "rs2"]
            assert [p["coefficient"] for p in rs4["predictors"]] == [0.3, 0.2]

            # Verify calibration params
            assert data["calibration_params"]["scaling_factor"] == 1.1


class TestInterceptOnlyModels:
    """Tests for intercept-only model handling."""

    def test_hdf5_round_trip_intercept_only_model(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that intercept-only models are correctly round-tripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            _, imp, _, _, _ = load_model_hdf5(hdf5_path)

            # Find rs5 which is intercept-only
            rs5 = next(m for m in imp if m.variant_id == "rs5")
            assert rs5.is_intercept_only is True
            assert len(rs5.predictor_variant_ids) == 0
            assert len(rs5.coefficients) == 0
            assert rs5.intercept == 1.0


class TestEmptyObservedVariants:
    """Tests for empty observed variants list."""

    def test_hdf5_loading_empty_observed_full_imputed(self, sample_imputed_models):
        """Test HDF5 loading with empty observed variants but full imputed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            obs, imp, _, _, meta = load_model_hdf5(hdf5_path)

            assert len(obs) == 0
            assert len(imp) == 2
            assert meta["n_observed_variants"] == 0
            assert meta["n_imputed_variants"] == 2


class TestEmptyImputedModels:
    """Tests for empty imputed models list."""

    def test_hdf5_loading_full_observed_empty_imputed(self, sample_observed_variants):
        """Test HDF5 loading with full observed variants but empty imputed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            obs, imp, _, _, meta = load_model_hdf5(hdf5_path)

            assert len(obs) == 3
            assert len(imp) == 0
            assert meta["n_observed_variants"] == 3
            assert meta["n_imputed_variants"] == 0


class TestTypePreservation:
    """Tests for type preservation in loaded data."""

    def test_hdf5_loading_preserves_python_types(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that Python types are correctly preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            obs, imp, _, _, _ = load_model_hdf5(hdf5_path)

            # Check VariantInfo types
            assert isinstance(obs[0].variant_id, str)
            assert isinstance(obs[0].chromosome, str)
            assert isinstance(obs[0].position, int)
            assert isinstance(obs[0].effect_allele, str)
            assert isinstance(obs[0].beta, float)

            # Check ImputedVariantModel types
            rs4 = next(m for m in imp if m.variant_id == "rs4")
            assert isinstance(rs4.variant_id, str)
            assert isinstance(rs4.position, int)
            assert isinstance(rs4.beta, float)
            assert isinstance(rs4.allele_frequency, float)
            assert isinstance(rs4.imputation_r2, float)
            assert isinstance(rs4.residual_variance, float)
            assert isinstance(rs4.intercept, float)
            assert isinstance(rs4.is_intercept_only, bool)
            assert isinstance(rs4.predictor_variant_ids, list)
            assert isinstance(rs4.coefficients, np.ndarray)

    def test_hdf5_loading_bool_not_numpy_bool(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that is_intercept_only is Python bool, not numpy bool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            _, imp, _, _, _ = load_model_hdf5(hdf5_path)

            for model in imp:
                assert type(model.is_intercept_only) is bool


class TestMetadataLoading:
    """Tests for metadata loading."""

    def test_hdf5_metadata_includes_all_attributes(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that all metadata attributes are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "model.h5"
            export_to_hdf5(
                output_path=hdf5_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                platform_name="test_platform",
                prs_id="PGS000001",
                genome_build="GRCh38",
                model_name="Test Model",
            )

            _, _, _, _, meta = load_model_hdf5(hdf5_path)

            assert meta["format_version"] == "2.0"
            assert meta["platform_name"] == "test_platform"
            assert meta["prs_id"] == "PGS000001"
            assert meta["genome_build"] == "GRCh38"
            assert meta["model_name"] == "Test Model"
            assert meta["n_observed_variants"] == 3
            assert meta["n_imputed_variants"] == 2
            assert meta["n_intercept_only"] == 1
            assert "created_at" in meta


class TestJSONv2RoundTrip:
    """v2.0 JSON: provenance, predictor-metadata restore, and v1.0 back-compat."""

    def test_json_loader_returns_provenance(
        self, sample_observed_variants, sample_imputed_models
    ):
        """The raw-dict loader surfaces the v2 provenance block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                platform_name="23andme_v5",
                genome_build="GRCh37",
                reference_panel_id="1000G_phase3_EUR",
                training_ancestry="EUR",
            )

            data = load_model_json(json_path)

            assert "provenance" in data
            assert data["provenance"]["platform_id"] == "23andme_v5"
            assert data["provenance"]["reference_panel_id"] == "1000G_phase3_EUR"
            assert data["provenance"]["training_ancestry"] == "EUR"
            assert (
                data["provenance"]["ambiguous_policy"]
                == "exclude_unless_platform_strand_known"
            )

    def test_json_round_trip_via_predict(self):
        """Export -> load -> oriented score is identical to the in-memory model.

        This is the teeth: ``compute_imputed_prs_oriented`` indexes the predictor
        allele metadata, so a loader that dropped it would mis-score (or
        ``IndexError``). The score-equivalence assertion is what proves the
        restore.
        """
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1)]
        imputed = [
            ImputedVariantModel(
                variant_id="rs4", chromosome="1", position=150,
                effect_allele="A", other_allele="G", beta=0.05,
                allele_frequency=0.3, imputation_r2=0.8, residual_variance=0.1,
                intercept=0.6, predictor_variant_ids=["rs1", "rs2"],
                coefficients=np.array([0.3, 0.2]), is_intercept_only=False,
                predictor_chromosomes=["1", "1"], predictor_positions=[100, 200],
                predictor_counted_alleles=["G", "T"],
                predictor_other_alleles=["A", "C"],
                predictor_allele_frequencies=np.array([0.4, 0.3]),
            ),
            ImputedVariantModel(
                variant_id="rs5", chromosome="2", position=400,
                effect_allele="T", other_allele="C", beta=0.02,
                allele_frequency=0.5, imputation_r2=0.0, residual_variance=0.5,
                intercept=1.0, predictor_variant_ids=[],
                coefficients=np.array([]), is_intercept_only=True,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "model.json"
            export_to_json(
                output_path=json_path,
                observed_variants=observed,
                imputed_models=imputed,
            )
            loaded = LinearImputationPRS.load(json_path)

        loaded_rs4 = next(m for m in loaded._imputed_models if m.variant_id == "rs4")
        orig_rs4 = imputed[0]

        # Exact reconstruction of ids / alleles / structure.
        assert loaded_rs4.predictor_variant_ids == orig_rs4.predictor_variant_ids
        assert loaded_rs4.predictor_chromosomes == orig_rs4.predictor_chromosomes
        assert loaded_rs4.predictor_positions == orig_rs4.predictor_positions
        assert (
            loaded_rs4.predictor_counted_alleles
            == orig_rs4.predictor_counted_alleles
        )
        assert (
            loaded_rs4.predictor_other_alleles == orig_rs4.predictor_other_alleles
        )
        assert loaded_rs4.is_intercept_only == orig_rs4.is_intercept_only

        # Allclose for floats.
        np.testing.assert_allclose(
            loaded_rs4.coefficients, orig_rs4.coefficients, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            loaded_rs4.predictor_allele_frequencies,
            orig_rs4.predictor_allele_frequencies,
            rtol=0, atol=1e-12,
        )

        # Score-equivalence on the oriented (raw-genotype) path.
        raw = load_raw_user_genotypes(
            pd.DataFrame(
                {
                    "rsid": ["rs1", "rs2"],
                    "chrom": ["1", "1"],
                    "pos": [100, 200],
                    "genotype": ["AG", "TT"],
                }
            )
        )
        orig_score = compute_imputed_prs_oriented(raw, imputed, allow_ambiguous=True)
        loaded_score = compute_imputed_prs_oriented(
            raw, loaded._imputed_models, allow_ambiguous=True
        )
        np.testing.assert_allclose(
            orig_score[0], loaded_score[0], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            orig_score[1], loaded_score[1], rtol=0, atol=1e-12
        )

    def test_v1_file_still_loads(self):
        """A legacy v1.0 JSON (parallel arrays, no provenance) still loads."""
        v1 = {
            "metadata": {
                "format_version": "1.0",
                "prs_id": "PGS000004",
                "platform_name": "23andme_v5",
                "genome_build": "GRCh37",
                "model_name": "legacy",
            },
            "observed_variants": [
                {
                    "variant_id": "rs1", "chromosome": "1", "position": 100,
                    "effect_allele": "A", "other_allele": "G", "beta": 0.1,
                }
            ],
            "imputed_variants": [
                {
                    "variant_id": "rs4", "chromosome": "1", "position": 150,
                    "effect_allele": "A", "other_allele": "G", "beta": 0.05,
                    "allele_frequency": 0.3, "imputation_r2": 0.8,
                    "intercept": 0.6, "is_intercept_only": False,
                    "predictor_variant_ids": ["rs1"], "coefficients": [0.3],
                }
            ],
            "platform_variant_index": {"rs1": 0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "v1_model.json"
            with open(json_path, "w") as f:
                json.dump(v1, f)

            loaded = LinearImputationPRS.load(json_path)

        rs4 = loaded._imputed_models[0]
        assert rs4.predictor_variant_ids == ["rs1"]
        np.testing.assert_allclose(rs4.coefficients, [0.3], rtol=0, atol=1e-12)
        # v1.0 carried no predictor allele metadata -> stays empty (documented).
        assert rs4.predictor_counted_alleles == []
        assert rs4.predictor_allele_frequencies.size == 0
        # Identity still restored from `metadata`; provenance defaults to None.
        assert loaded._genome_build == "GRCh37"
        assert loaded._platform_name == "23andme_v5"
        assert loaded._reference_panel_id is None
        assert loaded._training_ancestry is None


class TestV2FormatRoundTrip:
    """Export -> ``load()`` -> oriented-score round trip for the non-JSON v2 formats.

    Mirrors ``TestJSONv2RoundTrip.test_json_round_trip_via_predict`` for HDF5, Arrow,
    Parquet and CSV: the teeth is score-equivalence on ``compute_imputed_prs_oriented``,
    which indexes the predictor allele metadata, so a loader that dropped it would
    mis-score or ``IndexError``.
    """

    @staticmethod
    def _model():
        observed = [VariantInfo("rs1", "1", 100, "A", "G", 0.1)]
        imputed = [
            ImputedVariantModel(
                variant_id="rs4", chromosome="1", position=150,
                effect_allele="A", other_allele="G", beta=0.05,
                allele_frequency=0.3, imputation_r2=0.8, residual_variance=0.1,
                intercept=0.6, predictor_variant_ids=["rs1", "rs2"],
                coefficients=np.array([0.3, 0.2]), is_intercept_only=False,
                predictor_chromosomes=["1", "1"], predictor_positions=[100, 200],
                predictor_counted_alleles=["G", "T"],
                predictor_other_alleles=["A", "C"],
                predictor_allele_frequencies=np.array([0.4, 0.3]),
            ),
            ImputedVariantModel(
                variant_id="rs5", chromosome="2", position=400,
                effect_allele="T", other_allele="C", beta=0.02,
                allele_frequency=0.5, imputation_r2=0.0, residual_variance=0.5,
                intercept=1.0, predictor_variant_ids=[],
                coefficients=np.array([]), is_intercept_only=True,
            ),
        ]
        return observed, imputed

    @staticmethod
    def _export(fmt, tmpdir, observed, imputed):
        prov = dict(
            platform_name="23andme_v5", prs_id="PGS000004", genome_build="GRCh37",
            reference_panel_id="1000G_phase3_EUR", training_ancestry="EUR",
            ambiguous_policy="exclude_unless_platform_strand_known",
        )
        d = Path(tmpdir)
        if fmt == "hdf5":
            path = d / "model.h5"
            export_to_hdf5(path, observed_variants=observed,
                           imputed_models=imputed, **prov)
        elif fmt == "arrow":
            path = d / "model.arrow"
            export_to_arrow(path, observed_variants=observed,
                            imputed_models=imputed, **prov)
        elif fmt == "parquet":
            path = d / "model_parquet"  # parquet export is a directory
            export_to_parquet(path, observed_variants=observed,
                              imputed_models=imputed, **prov)
        elif fmt == "csv":
            path = d / "model_variants.csv"  # CSV carries no provenance by design
            export_variant_table(path, observed_variants=observed,
                                 imputed_models=imputed)
        else:  # pragma: no cover
            raise ValueError(fmt)
        return path

    @pytest.mark.parametrize("fmt", ["hdf5", "arrow", "parquet", "csv"])
    def test_round_trip_via_predict(self, fmt):
        observed, imputed = self._model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._export(fmt, tmpdir, observed, imputed)
            loaded = LinearImputationPRS.load(path)

        loaded_rs4 = next(
            m for m in loaded._imputed_models if m.variant_id == "rs4"
        )
        orig_rs4 = imputed[0]

        # Exact reconstruction of ids / alleles / structure.
        assert loaded_rs4.predictor_variant_ids == orig_rs4.predictor_variant_ids
        assert loaded_rs4.predictor_chromosomes == orig_rs4.predictor_chromosomes
        assert loaded_rs4.predictor_positions == orig_rs4.predictor_positions
        assert (
            loaded_rs4.predictor_counted_alleles
            == orig_rs4.predictor_counted_alleles
        )
        assert (
            loaded_rs4.predictor_other_alleles == orig_rs4.predictor_other_alleles
        )
        assert loaded_rs4.is_intercept_only == orig_rs4.is_intercept_only

        # Allclose for floats.
        np.testing.assert_allclose(
            loaded_rs4.coefficients, orig_rs4.coefficients, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            loaded_rs4.predictor_allele_frequencies,
            orig_rs4.predictor_allele_frequencies, rtol=0, atol=1e-12,
        )

        # Intercept-only model reconstructs with empty predictor metadata.
        loaded_rs5 = next(
            m for m in loaded._imputed_models if m.variant_id == "rs5"
        )
        assert loaded_rs5.is_intercept_only
        assert loaded_rs5.predictor_counted_alleles == []

        # Score-equivalence on the oriented (raw-genotype) path.
        raw = load_raw_user_genotypes(
            pd.DataFrame(
                {
                    "rsid": ["rs1", "rs2"],
                    "chrom": ["1", "1"],
                    "pos": [100, 200],
                    "genotype": ["AG", "TT"],
                }
            )
        )
        orig_score = compute_imputed_prs_oriented(
            raw, imputed, allow_ambiguous=True
        )
        loaded_score = compute_imputed_prs_oriented(
            raw, loaded._imputed_models, allow_ambiguous=True
        )
        np.testing.assert_allclose(
            orig_score[0], loaded_score[0], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            orig_score[1], loaded_score[1], rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("fmt", ["hdf5", "arrow", "parquet"])
    def test_provenance_restored(self, fmt):
        """HDF5/Arrow/Parquet restore provenance through ``load()``."""
        observed, imputed = self._model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._export(fmt, tmpdir, observed, imputed)
            loaded = LinearImputationPRS.load(path)

        assert loaded._genome_build == "GRCh37"
        assert loaded._platform_name == "23andme_v5"
        assert loaded._reference_panel_id == "1000G_phase3_EUR"
        assert loaded._training_ancestry == "EUR"
        assert loaded._ambiguous_policy == "exclude_unless_platform_strand_known"

    def test_csv_carries_no_provenance(self):
        """CSV is a flat variant table: provenance/calibration are not stored."""
        observed, imputed = self._model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._export("csv", tmpdir, observed, imputed)
            loaded = LinearImputationPRS.load(path)

        assert loaded._reference_panel_id is None
        assert loaded._training_ancestry is None
        assert loaded._calibration_params is None

    def test_v1_hdf5_without_predictor_metadata_loads(self):
        """A legacy v1 HDF5 (no predictor allele metadata datasets) still loads."""
        observed, imputed = self._model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v1.h5"
            export_to_hdf5(
                path, observed_variants=observed, imputed_models=imputed
            )
            # Simulate a legacy v1 artifact: drop the v2 predictor metadata datasets.
            with h5py.File(path, "r+") as f:
                for name in (
                    "predictor_chromosome",
                    "predictor_position",
                    "predictor_counted_allele",
                    "predictor_other_allele",
                    "predictor_allele_frequency",
                ):
                    del f["coefficients"][name]
                f["metadata"].attrs["format_version"] = "1.0"
            loaded = LinearImputationPRS.load(path)

        rs4 = next(m for m in loaded._imputed_models if m.variant_id == "rs4")
        # ids/coefficients still restored; predictor allele metadata stays empty.
        assert rs4.predictor_variant_ids == ["rs1", "rs2"]
        np.testing.assert_allclose(
            rs4.coefficients, [0.3, 0.2], rtol=0, atol=1e-12
        )
        assert rs4.predictor_counted_alleles == []
        assert rs4.predictor_allele_frequencies.size == 0
