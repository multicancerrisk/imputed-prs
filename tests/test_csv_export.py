"""Tests for CSV export functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.types import (
    ImputedVariantModel,
    VariantInfo,
)
from imputed_prs.io.exporters.csv_export import (
    COEFFICIENTS_COLUMNS,
    coefficients_path_for,
    export_variant_table,
)


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


class TestBasicExport:
    """Tests for basic export functionality."""

    def test_basic_export_with_observed_and_imputed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test basic export with observed and imputed variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            result_path = export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path == output_path
            assert output_path.exists()

            df = pd.read_csv(output_path)
            assert len(df) == 5  # 3 observed + 2 imputed

    def test_correct_columns_present(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that all expected columns are present in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)
            expected_cols = [
                "variant_id",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta",
                "status",
                "imputation_r2",
                "allele_frequency",
                "intercept",
                "n_predictors",
                "residual_variance",
            ]
            assert list(df.columns) == expected_cols

    def test_status_values_correct(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that status values are correct for each variant type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)

            # Check status counts
            status_counts = df["status"].value_counts().to_dict()
            assert status_counts["observed"] == 3
            assert status_counts["imputed"] == 1
            assert status_counts["intercept_only"] == 1

            # Check specific variants
            assert df[df["variant_id"] == "rs1"]["status"].values[0] == "observed"
            assert df[df["variant_id"] == "rs4"]["status"].values[0] == "imputed"
            assert df[df["variant_id"] == "rs5"]["status"].values[0] == "intercept_only"


class TestObservedVariantFields:
    """Tests for observed variant field values."""

    def test_observed_variants_have_none_for_imputation_fields(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that observed variants have None/NaN for imputation fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)
            observed = df[df["status"] == "observed"]

            assert observed["imputation_r2"].isna().all()
            assert observed["allele_frequency"].isna().all()
            assert observed["intercept"].isna().all()
            assert observed["residual_variance"].isna().all()
            assert (observed["n_predictors"] == 0).all()


class TestImputedVariantFields:
    """Tests for imputed variant field values."""

    def test_imputed_variants_have_imputation_fields_populated(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that imputed variants have imputation fields populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)
            imputed = df[df["status"].isin(["imputed", "intercept_only"])]

            assert not imputed["imputation_r2"].isna().any()
            assert not imputed["allele_frequency"].isna().any()
            assert not imputed["intercept"].isna().any()

            # Check specific values
            rs4 = df[df["variant_id"] == "rs4"].iloc[0]
            assert rs4["imputation_r2"] == 0.8
            assert rs4["allele_frequency"] == 0.3
            assert rs4["intercept"] == 0.6
            assert rs4["n_predictors"] == 2


class TestVarianceScaling:
    """Tests for variance scaling option."""

    def test_export_without_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export without variance scaling excludes residual_variance column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=False,
            )

            df = pd.read_csv(output_path)
            assert "residual_variance" not in df.columns

    def test_export_with_variance_scaling(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test export with variance scaling includes residual_variance column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
            )

            df = pd.read_csv(output_path)
            assert "residual_variance" in df.columns

            # Check that imputed variants have values
            rs4 = df[df["variant_id"] == "rs4"].iloc[0]
            assert rs4["residual_variance"] == 0.1


class TestPredictorDetails:
    """Tests for predictor details option."""

    def test_export_with_predictor_details(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that include_predictor_details=True adds predictor_variant_ids column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_predictor_details=True,
            )

            df = pd.read_csv(output_path)
            assert "predictor_variant_ids" in df.columns

    def test_predictor_ids_semicolon_separated(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that predictor IDs are semicolon-separated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_predictor_details=True,
            )

            df = pd.read_csv(output_path)
            rs4 = df[df["variant_id"] == "rs4"].iloc[0]
            assert rs4["predictor_variant_ids"] == "rs1;rs2"

    def test_observed_variants_have_none_predictor_ids(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that observed variants have None for predictor_variant_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_predictor_details=True,
            )

            df = pd.read_csv(output_path)
            observed = df[df["status"] == "observed"]
            assert observed["predictor_variant_ids"].isna().all()

    def test_intercept_only_has_none_predictor_ids(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that intercept-only variants have None for predictor_variant_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_predictor_details=True,
            )

            df = pd.read_csv(output_path)
            intercept_only = df[df["status"] == "intercept_only"]
            assert intercept_only["predictor_variant_ids"].isna().all()

    def test_export_without_predictor_details_excludes_column(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that include_predictor_details=False excludes the column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_predictor_details=False,
            )

            df = pd.read_csv(output_path)
            assert "predictor_variant_ids" not in df.columns


class TestNPredictors:
    """Tests for n_predictors count."""

    def test_n_predictors_count_correct(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that n_predictors count is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)

            # Observed variants have 0 predictors
            observed = df[df["status"] == "observed"]
            assert (observed["n_predictors"] == 0).all()

            # rs4 has 2 predictors
            rs4 = df[df["variant_id"] == "rs4"].iloc[0]
            assert rs4["n_predictors"] == 2

            # rs5 (intercept-only) has 0 predictors
            rs5 = df[df["variant_id"] == "rs5"].iloc[0]
            assert rs5["n_predictors"] == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_observed_variants_list(self, sample_imputed_models):
        """Test export with empty observed variants list (all imputed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=[],
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert "observed" not in df["status"].values

    def test_empty_imputed_models_list(self, sample_observed_variants):
        """Test export with empty imputed models list (all observed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            df = pd.read_csv(output_path)
            assert len(df) == 3
            assert (df["status"] == "observed").all()

    def test_both_lists_empty(self):
        """Test export with both lists empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=[],
                imputed_models=[],
            )

            df = pd.read_csv(output_path)
            assert len(df) == 0
            # Check columns are still present
            assert "variant_id" in df.columns
            assert "status" in df.columns

    def test_output_path_with_nonexistent_parent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that non-existent parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "variants.csv"
            result_path = export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert result_path.exists()
            assert result_path.parent.exists()


class TestStringPath:
    """Tests for string path input."""

    def test_string_path_input(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that string paths work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/variants.csv"
            result_path = export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            assert isinstance(result_path, Path)
            assert result_path.exists()


class TestCSVReadBack:
    """Tests for reading back the CSV."""

    def test_csv_can_be_read_and_parsed(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that CSV can be read back and parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            # Read back and verify
            df = pd.read_csv(output_path)

            # Check that numeric columns are numeric
            assert pd.api.types.is_integer_dtype(df["position"])
            assert pd.api.types.is_float_dtype(df["beta"])

            # Check that numeric values can be used
            total_beta = df["beta"].sum()
            assert isinstance(total_beta, (float, np.floating))

    def test_csv_roundtrip_preserves_values(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that CSV roundtrip preserves all values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)

            # Check observed variant values
            rs1 = df[df["variant_id"] == "rs1"].iloc[0]
            assert str(rs1["chromosome"]) == "1"
            assert rs1["position"] == 100
            assert rs1["effect_allele"] == "A"
            assert rs1["other_allele"] == "G"
            assert rs1["beta"] == 0.1

            # Check imputed variant values
            rs4 = df[df["variant_id"] == "rs4"].iloc[0]
            assert str(rs4["chromosome"]) == "1"
            assert rs4["position"] == 150
            assert rs4["effect_allele"] == "A"
            assert rs4["beta"] == 0.05
            assert rs4["imputation_r2"] == 0.8


class TestColumnOrder:
    """Tests for column ordering."""

    def test_column_order_is_consistent(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test that column order is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            df = pd.read_csv(output_path)
            expected_order = [
                "variant_id",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta",
                "status",
                "imputation_r2",
                "allele_frequency",
                "intercept",
                "n_predictors",
                "residual_variance",
            ]
            assert list(df.columns) == expected_order

    def test_column_order_with_all_options(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Test column order with all options enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
                include_variance_scaling=True,
                include_predictor_details=True,
            )

            df = pd.read_csv(output_path)
            expected_order = [
                "variant_id",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta",
                "status",
                "imputation_r2",
                "allele_frequency",
                "intercept",
                "n_predictors",
                "residual_variance",
                "predictor_variant_ids",
            ]
            assert list(df.columns) == expected_order


class TestCompanionCoefficients:
    """Tests for the companion long-format coefficients CSV (schema v2)."""

    def test_companion_file_written_with_predictor_metadata(
        self, sample_observed_variants, sample_imputed_models
    ):
        """The companion *_coefficients.csv carries per-predictor allele metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=sample_imputed_models,
            )

            coef_path = coefficients_path_for(output_path)
            assert coef_path.name == "model_coefficients.csv"
            assert coef_path.exists()

            coef = pd.read_csv(coef_path)
            assert list(coef.columns) == COEFFICIENTS_COLUMNS

            # rs4 has predictors rs1 (G/A) and rs2 (T/C); rs5 is intercept-only.
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
            assert "rs5" not in set(coef["target_variant_id"])

    def test_companion_file_header_only_without_predictors(
        self, sample_observed_variants
    ):
        """With no imputed predictors the companion is a header-only table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_variants.csv"
            export_variant_table(
                output_path=output_path,
                observed_variants=sample_observed_variants,
                imputed_models=[],
            )

            coef = pd.read_csv(coefficients_path_for(output_path))
            assert list(coef.columns) == COEFFICIENTS_COLUMNS
            assert len(coef) == 0
