"""Tests for Arrow and Parquet export functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.arrow_export import export_to_arrow, export_to_parquet


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


class TestBasicArrowExport:
    """Tests for basic Arrow export functionality."""

    def test_basic_arrow_export_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic Arrow export with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            result_path = export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path == output_path
            assert output_path.exists()

    def test_arrow_file_can_be_read_back(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that Arrow file can be read back with pyarrow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            # Read the container table
            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            assert container.num_rows == 4  # 4 tables
            assert "table_name" in container.column_names
            assert "data" in container.column_names

            # Extract table names
            table_names = container.column("table_name").to_pylist()
            assert "metadata" in table_names
            assert "observed_variants" in table_names
            assert "imputed_variants" in table_names
            assert "coefficients" in table_names

    def test_arrow_export_with_calibration_params(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test Arrow export includes calibration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            # Read and extract metadata table
            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()
            metadata_idx = table_names.index("metadata")
            metadata_bytes = container.column("data")[metadata_idx].as_py()

            with pa.ipc.open_stream(metadata_bytes) as stream:
                metadata_table = stream.read_all()

            # Check calibration params are stored as JSON
            calibration_json = metadata_table.column("calibration_params_json")[
                0
            ].as_py()
            assert calibration_json is not None
            calibration_data = json.loads(calibration_json)
            assert calibration_data["scaling_factor"] == 1.1
            assert calibration_data["n_calibration"] == 500

    def test_arrow_export_with_evaluation_metrics(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test Arrow export includes evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            # Read and extract metadata table
            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()
            metadata_idx = table_names.index("metadata")
            metadata_bytes = container.column("data")[metadata_idx].as_py()

            with pa.ipc.open_stream(metadata_bytes) as stream:
                metadata_table = stream.read_all()

            # Check evaluation metrics are stored as JSON
            metrics_json = metadata_table.column("evaluation_metrics_json")[0].as_py()
            assert metrics_json is not None
            metrics_data = json.loads(metrics_json)
            assert metrics_data["correlation"] == 0.95
            assert metrics_data["r2"] == 0.90


class TestBasicParquetExport:
    """Tests for basic Parquet export functionality."""

    def test_basic_parquet_export_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic Parquet export with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert "metadata" in paths
            assert "observed_variants" in paths
            assert "imputed_variants" in paths
            assert "coefficients" in paths

            for table_path in paths.values():
                assert table_path.exists()

    def test_parquet_files_can_be_read_back(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that Parquet files can be read back with pyarrow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            # Read observed variants
            observed_df = pq.read_table(paths["observed_variants"]).to_pandas()
            assert len(observed_df) == 3
            assert "variant_id" in observed_df.columns
            assert "beta" in observed_df.columns

            # Read imputed variants
            imputed_df = pq.read_table(paths["imputed_variants"]).to_pandas()
            assert len(imputed_df) == 2
            assert "variant_id" in imputed_df.columns
            assert "imputation_r2" in imputed_df.columns

            # Read coefficients
            coefficients_df = pq.read_table(paths["coefficients"]).to_pandas()
            assert len(coefficients_df) == 2  # rs4 has 2 predictors
            assert "target_variant_id" in coefficients_df.columns
            assert "predictor_variant_id" in coefficients_df.columns
            assert "coefficient" in coefficients_df.columns

    def test_parquet_export_with_calibration_params(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
    ):
        """Test Parquet export includes calibration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                calibration_params=sample_calibration_params,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            calibration_json = metadata_df["calibration_params_json"].iloc[0]
            assert calibration_json is not None
            calibration_data = json.loads(calibration_json)
            assert calibration_data["scaling_factor"] == 1.1

    def test_parquet_export_with_evaluation_metrics(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_evaluation_metrics,
    ):
        """Test Parquet export includes evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                evaluation_metrics=sample_evaluation_metrics,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            metrics_json = metadata_df["evaluation_metrics_json"].iloc[0]
            assert metrics_json is not None
            metrics_data = json.loads(metrics_json)
            assert metrics_data["correlation"] == 0.95


class TestVarianceScaling:
    """Tests for variance scaling option."""

    def test_arrow_export_without_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Arrow export without variance scaling excludes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            # Read and extract imputed_variants table
            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()
            imputed_idx = table_names.index("imputed_variants")
            imputed_bytes = container.column("data")[imputed_idx].as_py()

            with pa.ipc.open_stream(imputed_bytes) as stream:
                imputed_table = stream.read_all()

            assert "residual_variance" not in imputed_table.column_names

    def test_arrow_export_with_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Arrow export with variance scaling includes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            # Read and extract imputed_variants table
            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()
            imputed_idx = table_names.index("imputed_variants")
            imputed_bytes = container.column("data")[imputed_idx].as_py()

            with pa.ipc.open_stream(imputed_bytes) as stream:
                imputed_table = stream.read_all()

            assert "residual_variance" in imputed_table.column_names

    def test_parquet_export_without_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export without variance scaling excludes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            imputed_df = pq.read_table(paths["imputed_variants"]).to_pandas()
            assert "residual_variance" not in imputed_df.columns

    def test_parquet_export_with_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export with variance scaling includes residual_variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            imputed_df = pq.read_table(paths["imputed_variants"]).to_pandas()
            assert "residual_variance" in imputed_df.columns


class TestEdgeCases:
    """Tests for edge cases."""

    def test_arrow_export_empty_imputed_models(self, sample_observed_variants):
        """Test Arrow export with empty imputed models list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()

            # Check metadata
            metadata_idx = table_names.index("metadata")
            metadata_bytes = container.column("data")[metadata_idx].as_py()
            with pa.ipc.open_stream(metadata_bytes) as stream:
                metadata_table = stream.read_all()
            assert metadata_table.column("n_imputed_variants")[0].as_py() == 0

            # Check imputed_variants is empty
            imputed_idx = table_names.index("imputed_variants")
            imputed_bytes = container.column("data")[imputed_idx].as_py()
            with pa.ipc.open_stream(imputed_bytes) as stream:
                imputed_table = stream.read_all()
            assert imputed_table.num_rows == 0

            # Check coefficients is empty
            coef_idx = table_names.index("coefficients")
            coef_bytes = container.column("data")[coef_idx].as_py()
            with pa.ipc.open_stream(coef_bytes) as stream:
                coef_table = stream.read_all()
            assert coef_table.num_rows == 0

    def test_arrow_export_empty_observed_variants(self, sample_imputed_models):
        """Test Arrow export with empty observed variants list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()

            # Check metadata
            metadata_idx = table_names.index("metadata")
            metadata_bytes = container.column("data")[metadata_idx].as_py()
            with pa.ipc.open_stream(metadata_bytes) as stream:
                metadata_table = stream.read_all()
            assert metadata_table.column("n_observed_variants")[0].as_py() == 0

            # Check observed_variants is empty
            observed_idx = table_names.index("observed_variants")
            observed_bytes = container.column("data")[observed_idx].as_py()
            with pa.ipc.open_stream(observed_bytes) as stream:
                observed_table = stream.read_all()
            assert observed_table.num_rows == 0

    def test_parquet_export_empty_imputed_models(self, sample_observed_variants):
        """Test Parquet export with empty imputed models list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            assert metadata_df["n_imputed_variants"].iloc[0] == 0

            imputed_df = pq.read_table(paths["imputed_variants"]).to_pandas()
            assert len(imputed_df) == 0

            coef_df = pq.read_table(paths["coefficients"]).to_pandas()
            assert len(coef_df) == 0

    def test_parquet_export_empty_observed_variants(self, sample_imputed_models):
        """Test Parquet export with empty observed variants list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            assert metadata_df["n_observed_variants"].iloc[0] == 0

            observed_df = pq.read_table(paths["observed_variants"]).to_pandas()
            assert len(observed_df) == 0

    def test_arrow_output_path_with_nonexistent_parent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that non-existent parent directories are created for Arrow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_model.arrow"
            result_path = export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path.exists()
            assert result_path.parent.exists()

    def test_parquet_output_path_with_nonexistent_parent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that non-existent parent directories are created for Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert output_path.exists()
            for table_path in paths.values():
                assert table_path.exists()


class TestTableSchemas:
    """Tests for table schema correctness."""

    def test_observed_variants_schema(self, sample_observed_variants):
        """Test observed variants table has correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            observed_table = pq.read_table(paths["observed_variants"])
            expected_columns = {
                "variant_id",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta",
                "platform_index",
            }
            assert set(observed_table.column_names) == expected_columns

    def test_imputed_variants_schema_with_variance(self, sample_imputed_models):
        """Test imputed variants table has correct schema with variance scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            imputed_table = pq.read_table(paths["imputed_variants"])
            expected_columns = {
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
            assert set(imputed_table.column_names) == expected_columns

    def test_imputed_variants_schema_without_variance(self, sample_imputed_models):
        """Test imputed variants table has correct schema without variance scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            imputed_table = pq.read_table(paths["imputed_variants"])
            expected_columns = {
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
            assert set(imputed_table.column_names) == expected_columns

    def test_coefficients_schema(self, sample_imputed_models):
        """Test coefficients table has correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            coef_table = pq.read_table(paths["coefficients"])
            # v2: coefficients carry per-predictor allele metadata.
            expected_columns = {
                "target_variant_id",
                "predictor_variant_id",
                "coefficient",
                "predictor_chromosome",
                "predictor_position",
                "predictor_counted_allele",
                "predictor_other_allele",
                "predictor_allele_frequency",
            }
            assert set(coef_table.column_names) == expected_columns

    def test_metadata_schema(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test metadata table has correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            metadata_table = pq.read_table(paths["metadata"])
            expected_columns = {
                "format_version",
                "created_at",
                "model_name",
                "prs_id",
                "platform_name",
                "genome_build",
                "reference_panel_id",
                "training_ancestry",
                "ambiguous_policy",
                "n_observed_variants",
                "n_imputed_variants",
                "n_intercept_only",
                "include_variance_scaling",
                "calibration_params_json",
                "evaluation_metrics_json",
                "training_summary_json",
            }
            assert set(metadata_table.column_names) == expected_columns

    def test_coefficients_predictor_metadata_values(self, sample_imputed_models):
        """v2 coefficients store per-predictor allele metadata, index-aligned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            coef = pq.read_table(paths["coefficients"]).to_pandas()
            rs4 = coef[coef["target_variant_id"] == "rs4"].reset_index(drop=True)
            assert list(rs4["predictor_variant_id"]) == ["rs1", "rs2"]
            assert list(rs4["predictor_counted_allele"]) == ["G", "T"]
            assert list(rs4["predictor_other_allele"]) == ["A", "C"]
            assert list(rs4["predictor_chromosome"].astype(str)) == ["1", "1"]
            assert list(rs4["predictor_position"]) == [100, 200]
            np.testing.assert_allclose(
                rs4["predictor_allele_frequency"], [0.4, 0.3], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                rs4["coefficient"], [0.3, 0.2], rtol=0, atol=1e-12
            )
            # rs5 is intercept-only -> contributes no coefficient rows.
            assert "rs5" not in set(coef["target_variant_id"])


class TestCoefficientsSparseRepresentation:
    """Tests for coefficients sparse representation."""

    def test_coefficients_sparse_representation(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test coefficients table has correct sparse representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            coef_df = pq.read_table(paths["coefficients"]).to_pandas()

            # rs4 has 2 predictors (rs1, rs2), rs5 has none
            assert len(coef_df) == 2

            # Check rs4's coefficients
            rs4_coefs = coef_df[coef_df["target_variant_id"] == "rs4"]
            assert len(rs4_coefs) == 2
            assert set(rs4_coefs["predictor_variant_id"]) == {"rs1", "rs2"}

            # Check coefficient values
            rs1_coef = rs4_coefs[rs4_coefs["predictor_variant_id"] == "rs1"][
                "coefficient"
            ].iloc[0]
            rs2_coef = rs4_coefs[rs4_coefs["predictor_variant_id"] == "rs2"][
                "coefficient"
            ].iloc[0]
            assert np.isclose(rs1_coef, 0.3)
            assert np.isclose(rs2_coef, 0.2)


class TestRoundTrip:
    """Tests for round-trip serialization."""

    def test_parquet_round_trip_data_integrity(
        self,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
        sample_evaluation_metrics,
    ):
        """Test that data is preserved after Parquet export and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
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

            # Read and verify metadata
            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            assert metadata_df["prs_id"].iloc[0] == "PGS000004"
            assert metadata_df["platform_name"].iloc[0] == "23andme_v5"
            assert metadata_df["genome_build"].iloc[0] == "GRCh37"
            assert metadata_df["model_name"].iloc[0] == "Test PRS Model"
            assert metadata_df["n_observed_variants"].iloc[0] == 3
            assert metadata_df["n_imputed_variants"].iloc[0] == 2

            # Read and verify observed variants
            observed_df = pq.read_table(paths["observed_variants"]).to_pandas()
            assert len(observed_df) == 3
            rs1_row = observed_df[observed_df["variant_id"] == "rs1"].iloc[0]
            assert rs1_row["chromosome"] == "1"
            assert rs1_row["position"] == 100
            assert rs1_row["effect_allele"] == "A"
            assert np.isclose(rs1_row["beta"], 0.1)

            # Read and verify imputed variants
            imputed_df = pq.read_table(paths["imputed_variants"]).to_pandas()
            assert len(imputed_df) == 2
            rs4_row = imputed_df[imputed_df["variant_id"] == "rs4"].iloc[0]
            assert rs4_row["chromosome"] == "1"
            assert np.isclose(rs4_row["imputation_r2"], 0.8)
            assert rs4_row["is_intercept_only"] == False

            rs5_row = imputed_df[imputed_df["variant_id"] == "rs5"].iloc[0]
            assert rs5_row["is_intercept_only"] == True


class TestParquetCompression:
    """Tests for Parquet compression options."""

    def test_parquet_compression_snappy(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export with snappy compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="snappy",
            )

            for table_path in paths.values():
                assert table_path.exists()
                # Verify file can be read
                pq.read_table(table_path)

    def test_parquet_compression_gzip(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export with gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="gzip",
            )

            for table_path in paths.values():
                assert table_path.exists()
                pq.read_table(table_path)

    def test_parquet_compression_zstd(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export with zstd compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="zstd",
            )

            for table_path in paths.values():
                assert table_path.exists()
                pq.read_table(table_path)

    def test_parquet_compression_none(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export with no compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                compression="none",
            )

            for table_path in paths.values():
                assert table_path.exists()
                pq.read_table(table_path)


class TestStringPath:
    """Tests for string path input."""

    def test_arrow_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly for Arrow export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_model.arrow"
            result_path = export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert isinstance(result_path, Path)
            assert result_path.exists()

    def test_parquet_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly for Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            for table_path in paths.values():
                assert isinstance(table_path, Path)
                assert table_path.exists()


class TestTrainingSummary:
    """Tests for training summary inclusion."""

    def test_parquet_export_with_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Parquet export includes training summary when provided."""
        training_summary = {
            "mean_r2": 0.75,
            "median_r2": 0.80,
            "n_high_quality": 100,
            "n_low_quality": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                training_summary=training_summary,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            summary_json = metadata_df["training_summary_json"].iloc[0]
            assert summary_json is not None
            summary_data = json.loads(summary_json)
            assert summary_data["mean_r2"] == 0.75
            assert summary_data["n_high_quality"] == 100

    def test_arrow_export_with_training_summary(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test Arrow export includes training summary when provided."""
        training_summary = {
            "mean_r2": 0.75,
            "median_r2": 0.80,
            "n_high_quality": 100,
            "n_low_quality": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.arrow"
            export_to_arrow(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                training_summary=training_summary,
            )

            with pa.ipc.open_file(str(output_path)) as reader:
                container = reader.read_all()

            table_names = container.column("table_name").to_pylist()
            metadata_idx = table_names.index("metadata")
            metadata_bytes = container.column("data")[metadata_idx].as_py()

            with pa.ipc.open_stream(metadata_bytes) as stream:
                metadata_table = stream.read_all()

            summary_json = metadata_table.column("training_summary_json")[0].as_py()
            assert summary_json is not None
            summary_data = json.loads(summary_json)
            assert summary_data["mean_r2"] == 0.75


class TestMetadata:
    """Tests for metadata content."""

    def test_intercept_only_count_in_metadata(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that intercept-only model count is correct in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            # One of the sample_imputed_models is intercept-only
            assert metadata_df["n_intercept_only"].iloc[0] == 1

    def test_format_version_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that format version is present in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            assert metadata_df["format_version"].iloc[0] == "2.0"

    def test_created_at_timestamp_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that created_at timestamp is present and valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model"
            paths = export_to_parquet(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            metadata_df = pq.read_table(paths["metadata"]).to_pandas()
            created_at = metadata_df["created_at"].iloc[0]
            assert created_at is not None
            assert created_at.endswith("Z")
