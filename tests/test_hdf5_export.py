"""Tests for HDF5 export functionality."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.hdf5_export import export_to_hdf5


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


class TestBasicHDF5Export:
    """Tests for basic HDF5 export functionality."""

    def test_basic_hdf5_export_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic HDF5 export with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            result_path = export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path == output_path
            assert output_path.exists()

    def test_hdf5_file_can_be_read_back(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that HDF5 file can be read back with h5py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                assert "metadata" in f
                assert "observed_variants" in f
                assert "imputed_variants" in f
                assert "coefficients" in f

    def test_hdf5_export_with_calibration_params(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test HDF5 export includes calibration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            with h5py.File(output_path, "r") as f:
                calibration_json = f["metadata/calibration_params_json"][()].decode("utf-8")
                calibration_data = json.loads(calibration_json)
                assert calibration_data["scaling_factor"] == 1.1
                assert calibration_data["n_calibration"] == 500

    def test_hdf5_export_with_evaluation_metrics(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test HDF5 export includes evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            with h5py.File(output_path, "r") as f:
                metrics_json = f["metadata/evaluation_metrics_json"][()].decode("utf-8")
                metrics_data = json.loads(metrics_json)
                assert metrics_data["correlation"] == 0.95
                assert metrics_data["r2"] == 0.90


class TestVarianceScaling:
    """Tests for variance scaling option."""

    def test_hdf5_export_without_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export without variance scaling excludes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            with h5py.File(output_path, "r") as f:
                assert "residual_variance" not in f["imputed_variants"]

    def test_hdf5_export_with_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export with variance scaling includes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            with h5py.File(output_path, "r") as f:
                assert "residual_variance" in f["imputed_variants"]
                residual_variance = f["imputed_variants/residual_variance"][:]
                assert len(residual_variance) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_hdf5_export_empty_imputed_models(self, sample_observed_variants):
        """Test HDF5 export with empty imputed models list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            with h5py.File(output_path, "r") as f:
                assert f["metadata"].attrs["n_imputed_variants"] == 0
                assert len(f["imputed_variants/variant_id"]) == 0
                assert len(f["coefficients/coefficient"]) == 0

    def test_hdf5_export_empty_observed_variants(self, sample_imputed_models):
        """Test HDF5 export with empty observed variants list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                assert f["metadata"].attrs["n_observed_variants"] == 0
                assert len(f["observed_variants/variant_id"]) == 0

    def test_hdf5_output_path_with_nonexistent_parent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that non-existent parent directories are created for HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_model.h5"
            result_path = export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path.exists()
            assert result_path.parent.exists()


class TestGroupStructure:
    """Tests for HDF5 group structure."""

    def test_verify_group_structure(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that HDF5 file has correct group structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                # Top-level groups
                assert isinstance(f["metadata"], h5py.Group)
                assert isinstance(f["observed_variants"], h5py.Group)
                assert isinstance(f["imputed_variants"], h5py.Group)
                assert isinstance(f["coefficients"], h5py.Group)

    def test_observed_variants_datasets(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test observed variants group has correct datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                expected_datasets = {
                    "variant_id",
                    "chromosome",
                    "position",
                    "effect_allele",
                    "other_allele",
                    "beta",
                    "platform_index",
                }
                actual_datasets = set(f["observed_variants"].keys())
                assert actual_datasets == expected_datasets

    def test_imputed_variants_datasets_with_variance(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test imputed variants group has correct datasets with variance scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            with h5py.File(output_path, "r") as f:
                expected_datasets = {
                    "variant_id",
                    "chromosome",
                    "position",
                    "effect_allele",
                    "other_allele",
                    "beta",
                    "allele_frequency",
                    "imputation_r2",
                    "intercept",
                    "is_intercept_only",
                    "n_predictors",
                    "residual_variance",
                }
                actual_datasets = set(f["imputed_variants"].keys())
                assert actual_datasets == expected_datasets

    def test_imputed_variants_datasets_without_variance(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test imputed variants group has correct datasets without variance scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            with h5py.File(output_path, "r") as f:
                expected_datasets = {
                    "variant_id",
                    "chromosome",
                    "position",
                    "effect_allele",
                    "other_allele",
                    "beta",
                    "allele_frequency",
                    "imputation_r2",
                    "intercept",
                    "is_intercept_only",
                    "n_predictors",
                }
                actual_datasets = set(f["imputed_variants"].keys())
                assert actual_datasets == expected_datasets

    def test_coefficients_datasets(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test coefficients group has correct datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                # v2: coefficients carry per-predictor allele metadata.
                expected_datasets = {
                    "target_variant_id",
                    "predictor_variant_id",
                    "coefficient",
                    "predictor_chromosome",
                    "predictor_position",
                    "predictor_counted_allele",
                    "predictor_other_allele",
                    "predictor_allele_frequency",
                }
                actual_datasets = set(f["coefficients"].keys())
                assert actual_datasets == expected_datasets

    def test_coefficients_predictor_metadata_values(
        self, sample_observed_variants, sample_imputed_models
    ):
        """v2 coefficients store per-predictor allele metadata, index-aligned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                grp = f["coefficients"]
                targets = [v.decode("utf-8") for v in grp["target_variant_id"][:]]
                preds = [v.decode("utf-8") for v in grp["predictor_variant_id"][:]]
                counted = [
                    v.decode("utf-8") for v in grp["predictor_counted_allele"][:]
                ]
                other = [
                    v.decode("utf-8") for v in grp["predictor_other_allele"][:]
                ]
                chroms = [
                    v.decode("utf-8") for v in grp["predictor_chromosome"][:]
                ]
                positions = grp["predictor_position"][:].tolist()
                afs = grp["predictor_allele_frequency"][:]

            # Only rs4 has predictors: rs1 (G/A) and rs2 (T/C); rs5 is intercept-only.
            assert targets == ["rs4", "rs4"]
            assert preds == ["rs1", "rs2"]
            assert counted == ["G", "T"]
            assert other == ["A", "C"]
            assert chroms == ["1", "1"]
            assert positions == [100, 200]
            np.testing.assert_allclose(afs, [0.4, 0.3], rtol=0, atol=1e-12)


class TestRoundTrip:
    """Tests for round-trip serialization."""

    def test_hdf5_round_trip_data_integrity(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
        sample_evaluation_metrics,
    ):
        """Test that data is preserved after HDF5 export and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
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

            with h5py.File(output_path, "r") as f:
                # Verify metadata attributes
                assert f["metadata"].attrs["prs_id"] == "PGS000004"
                assert f["metadata"].attrs["platform_name"] == "23andme_v5"
                assert f["metadata"].attrs["genome_build"] == "GRCh37"
                assert f["metadata"].attrs["model_name"] == "Test PRS Model"
                assert f["metadata"].attrs["n_observed_variants"] == 3
                assert f["metadata"].attrs["n_imputed_variants"] == 2

                # Verify observed variants
                variant_ids = [v.decode("utf-8") for v in f["observed_variants/variant_id"][:]]
                assert "rs1" in variant_ids
                assert len(variant_ids) == 3

                positions = f["observed_variants/position"][:]
                assert positions[variant_ids.index("rs1")] == 100

                betas = f["observed_variants/beta"][:]
                assert np.isclose(betas[variant_ids.index("rs1")], 0.1)

                # Verify imputed variants
                imputed_ids = [v.decode("utf-8") for v in f["imputed_variants/variant_id"][:]]
                assert "rs4" in imputed_ids
                assert "rs5" in imputed_ids

                imputation_r2 = f["imputed_variants/imputation_r2"][:]
                assert np.isclose(imputation_r2[imputed_ids.index("rs4")], 0.8)

                is_intercept_only = f["imputed_variants/is_intercept_only"][:]
                assert is_intercept_only[imputed_ids.index("rs5")] == True
                assert is_intercept_only[imputed_ids.index("rs4")] == False


class TestCoefficientsSparseRepresentation:
    """Tests for coefficients sparse representation."""

    def test_coefficients_sparse_representation(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test coefficients stored in sparse representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                target_ids = [v.decode("utf-8") for v in f["coefficients/target_variant_id"][:]]
                predictor_ids = [v.decode("utf-8") for v in f["coefficients/predictor_variant_id"][:]]
                coefficients = f["coefficients/coefficient"][:]

                # rs4 has 2 predictors (rs1, rs2), rs5 has none
                assert len(target_ids) == 2

                # Check rs4's coefficients
                rs4_indices = [i for i, t in enumerate(target_ids) if t == "rs4"]
                assert len(rs4_indices) == 2

                rs4_predictors = {predictor_ids[i] for i in rs4_indices}
                assert rs4_predictors == {"rs1", "rs2"}

                # Check coefficient values
                for i in rs4_indices:
                    if predictor_ids[i] == "rs1":
                        assert np.isclose(coefficients[i], 0.3)
                    elif predictor_ids[i] == "rs2":
                        assert np.isclose(coefficients[i], 0.2)


class TestCompressionOptions:
    """Tests for HDF5 compression options."""

    def test_hdf5_compression_gzip(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export with gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="gzip",
                compression_opts=4,
            )

            assert output_path.exists()
            with h5py.File(output_path, "r") as f:
                assert "metadata" in f

    def test_hdf5_compression_lzf(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export with lzf compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="lzf",
            )

            assert output_path.exists()
            with h5py.File(output_path, "r") as f:
                assert "metadata" in f

    def test_hdf5_compression_none(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export with no compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression=None,
            )

            assert output_path.exists()
            with h5py.File(output_path, "r") as f:
                assert "metadata" in f


class TestMetadataAttributes:
    """Tests for metadata attributes."""

    def test_metadata_attributes_correctly_stored(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that metadata attributes are correctly stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                platform_name="23andme_v5",
                prs_id="PGS000004",
                genome_build="GRCh37",
                model_name="Test Model",
            )

            with h5py.File(output_path, "r") as f:
                meta = f["metadata"]
                assert meta.attrs["format_version"] == "2.0"
                assert meta.attrs["n_observed_variants"] == 3
                assert meta.attrs["n_imputed_variants"] == 2
                assert meta.attrs["n_intercept_only"] == 1
                assert meta.attrs["include_variance_scaling"] == True
                assert meta.attrs["platform_name"] == "23andme_v5"
                assert meta.attrs["prs_id"] == "PGS000004"
                assert meta.attrs["genome_build"] == "GRCh37"
                assert meta.attrs["model_name"] == "Test Model"
                # v2 provenance attrs exist (empty when not supplied).
                assert meta.attrs["reference_panel_id"] == ""
                assert meta.attrs["training_ancestry"] == ""
                assert meta.attrs["ambiguous_policy"] == ""

    def test_format_version_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that format version is present in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                assert f["metadata"].attrs["format_version"] == "2.0"

    def test_created_at_timestamp_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that created_at timestamp is present and valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                created_at = f["metadata"].attrs["created_at"]
                assert created_at is not None
                assert created_at.endswith("Z")

    def test_intercept_only_count_in_metadata(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that intercept-only model count is correct in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                # One of the sample_imputed_models is intercept-only
                assert f["metadata"].attrs["n_intercept_only"] == 1

    def test_optional_metadata_empty_when_not_provided(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that optional metadata stores empty strings when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                meta = f["metadata"]
                assert meta.attrs["platform_name"] == ""
                assert meta.attrs["prs_id"] == ""
                assert meta.attrs["genome_build"] == ""
                assert meta.attrs["model_name"] == ""


class TestStringEncoding:
    """Tests for string datasets using UTF-8 encoding."""

    def test_string_datasets_use_utf8(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string datasets use UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                # Check variant_id encoding
                variant_id_ds = f["observed_variants/variant_id"]
                assert h5py.check_string_dtype(variant_id_ds.dtype) is not None

                # Check chromosome encoding
                chromosome_ds = f["observed_variants/chromosome"]
                assert h5py.check_string_dtype(chromosome_ds.dtype) is not None


class TestNumpyDtypes:
    """Tests for numpy array dtype preservation."""

    def test_numpy_arrays_preserve_dtype(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that numpy arrays preserve their dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                # Check position dtype (int64)
                position = f["observed_variants/position"]
                assert position.dtype == np.int64

                # Check beta dtype (float64)
                beta = f["observed_variants/beta"]
                assert beta.dtype == np.float64

                # Check platform_index dtype (int32)
                platform_index = f["observed_variants/platform_index"]
                assert platform_index.dtype == np.int32

                # Check coefficient dtype (float64)
                coefficient = f["coefficients/coefficient"]
                assert coefficient.dtype == np.float64

                # Check is_intercept_only dtype (bool)
                is_intercept_only = f["imputed_variants/is_intercept_only"]
                assert is_intercept_only.dtype == bool


class TestTrainingSummary:
    """Tests for training summary inclusion."""

    def test_hdf5_export_with_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export includes training summary when provided."""
        training_summary = {
            "mean_r2": 0.75,
            "median_r2": 0.80,
            "n_high_quality": 100,
            "n_low_quality": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                training_summary=training_summary,
            )

            with h5py.File(output_path, "r") as f:
                summary_json = f["metadata/training_summary_json"][()].decode("utf-8")
                summary_data = json.loads(summary_json)
                assert summary_data["mean_r2"] == 0.75
                assert summary_data["n_high_quality"] == 100

    def test_hdf5_export_without_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test HDF5 export works without training summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.h5"
            export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            with h5py.File(output_path, "r") as f:
                assert "training_summary_json" not in f["metadata"]


class TestStringPath:
    """Tests for string path input."""

    def test_hdf5_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly for HDF5 export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_model.h5"
            result_path = export_to_hdf5(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert isinstance(result_path, Path)
            assert result_path.exists()
