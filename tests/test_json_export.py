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

            # Find the model with coefficients (v2: per-predictor objects).
            model_with_coeffs = next(
                m for m in data["imputed_variants"] if not m["is_intercept_only"]
            )
            assert isinstance(model_with_coeffs["predictors"], list)
            coeffs = [p["coefficient"] for p in model_with_coeffs["predictors"]]
            assert all(isinstance(c, float) for c in coeffs)
            assert coeffs == [0.3, 0.2]


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

            assert data["metadata"]["format_version"] == "2.0"

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


class TestV2Schema:
    """Tests for the v2.0 browser-deployable schema additions."""

    def _export(self, tmpdir, observed, imputed, **kwargs):
        output_path = Path(tmpdir) / "model.json"
        export_to_json(
            output_path=output_path,
            observed_variants=observed,
            imputed_models=imputed,
            **kwargs,
        )
        with open(output_path) as f:
            return json.load(f)

    def test_provenance_block(
        self, sample_observed_variants, sample_imputed_models,
        sample_calibration_params,
    ):
        """Provenance carries identity + policy + centering/scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(
                tmpdir, sample_observed_variants, sample_imputed_models,
                calibration_params=sample_calibration_params,
                platform_name="23andme_v5", genome_build="GRCh37",
                reference_panel_id="1000G_phase3_EUR", training_ancestry="EUR",
            )
            prov = data["provenance"]
            assert prov["genome_build"] == "GRCh37"
            assert prov["platform_id"] == "23andme_v5"
            assert prov["reference_panel_id"] == "1000G_phase3_EUR"
            assert prov["training_ancestry"] == "EUR"
            assert prov["ambiguous_policy"] == "exclude_unless_platform_strand_known"
            np.testing.assert_allclose(
                prov["centering_scaling"]["scaling_factor"], 1.1, rtol=0, atol=1e-12
            )

    def test_provenance_centering_scaling_null_without_calibration(
        self, sample_observed_variants, sample_imputed_models
    ):
        """centering_scaling is null when no calibration was fitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(tmpdir, sample_observed_variants, sample_imputed_models)
            assert data["provenance"]["centering_scaling"] is None

    def test_observed_accepted_ids_and_ambiguous_flag(self, sample_imputed_models):
        """Observed terms carry accepted_ids and a palindrome flag."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),   # not palindromic
            VariantInfo("rs_pal", "3", 500, "A", "T", 0.3),  # A/T palindrome
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(tmpdir, observed, sample_imputed_models)
            by_id = {v["variant_id"]: v for v in data["observed_variants"]}
            assert by_id["rs1"]["accepted_ids"] == ["rs1", "1:100"]
            assert by_id["rs1"]["ambiguous"] is False
            assert by_id["rs_pal"]["accepted_ids"] == ["rs_pal", "3:500"]
            assert by_id["rs_pal"]["ambiguous"] is True

    def test_predictor_objects_are_self_describing(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Each predictor is an object carrying its own locus + alleles + AF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(tmpdir, sample_observed_variants, sample_imputed_models)
            rs4 = next(m for m in data["imputed_variants"] if m["variant_id"] == "rs4")
            preds = rs4["predictors"]
            assert [p["variant_id"] for p in preds] == ["rs1", "rs2"]
            assert preds[0]["counted_allele"] == "G"
            assert preds[0]["other_allele"] == "A"
            assert preds[1]["counted_allele"] == "T"
            assert preds[1]["other_allele"] == "C"
            np.testing.assert_allclose(
                [p["allele_frequency"] for p in preds], [0.4, 0.3], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                [p["coefficient"] for p in preds], [0.3, 0.2], rtol=0, atol=1e-12
            )

    def test_intercept_only_has_empty_predictors(
        self, sample_observed_variants, sample_imputed_models
    ):
        """Intercept-only imputed variants emit an empty predictors list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(tmpdir, sample_observed_variants, sample_imputed_models)
            rs5 = next(m for m in data["imputed_variants"] if m["variant_id"] == "rs5")
            assert rs5["is_intercept_only"] is True
            assert rs5["predictors"] == []

    def test_deploy_gate_raises_on_missing_observed_other_allele(
        self, sample_imputed_models
    ):
        """Default export refuses a scored observed term without other_allele."""
        observed = [VariantInfo("rsX", "1", 100, "A", None, 0.1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="other_allele"):
                self._export(tmpdir, observed, sample_imputed_models)

    def test_deploy_gate_raises_on_predictor_without_metadata(self):
        """A predictor lacking other_allele metadata cannot be deployed."""
        imputed = [
            ImputedVariantModel(
                variant_id="rs4", chromosome="1", position=150,
                effect_allele="A", other_allele="G", beta=0.05,
                allele_frequency=0.3, imputation_r2=0.8, residual_variance=0.1,
                intercept=0.6, predictor_variant_ids=["rs1"],
                coefficients=np.array([0.3]), is_intercept_only=False,
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="other_allele"):
                self._export(tmpdir, [], imputed)

    def test_escape_hatch_allows_missing_other_allele(self, sample_imputed_models):
        """require_other_allele=False writes a non-deployable research export."""
        observed = [VariantInfo("rsX", "1", 100, "A", None, 0.1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(
                tmpdir, observed, sample_imputed_models, require_other_allele=False
            )
            assert data["observed_variants"][0]["other_allele"] is None
            assert data["observed_variants"][0]["ambiguous"] is False


# Committed machine-readable schema for the v2.0 deployable artifact (P1.5c).
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "imputation_model_v2.schema.json"
)


@pytest.fixture
def jsonschema_mod():
    """The jsonschema module; schema tests skip if the dev dep is absent."""
    return pytest.importorskip("jsonschema")


@pytest.fixture
def imputation_v2_validator(jsonschema_mod):
    """A validator built from the committed v2 schema.

    Building it also asserts the committed schema is itself a valid Draft
    2020-12 schema (``check_schema``), so a malformed schema fails loudly.
    """
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    validator_cls = jsonschema_mod.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


class TestSchemaValidation:
    """Validate exported v2 artifacts against the committed JSON Schema (P1.5c)."""

    def _export(self, tmpdir, observed, imputed, **kwargs):
        output_path = Path(tmpdir) / "model.json"
        export_to_json(
            output_path=output_path,
            observed_variants=observed,
            imputed_models=imputed,
            **kwargs,
        )
        with open(output_path) as f:
            return json.load(f)

    def test_committed_schema_is_valid(self, imputation_v2_validator):
        """The committed schema parses and passes Draft 2020-12 check_schema."""
        # Reaching here means the fixture built the validator without raising.
        assert imputation_v2_validator is not None

    def test_full_export_validates(
        self,
        imputation_v2_validator,
        sample_observed_variants,
        sample_imputed_models,
        sample_calibration_params,
        sample_evaluation_metrics,
    ):
        """A fully-populated deployable export conforms to the schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(
                tmpdir,
                sample_observed_variants,
                sample_imputed_models,
                calibration_params=sample_calibration_params,
                evaluation_metrics=sample_evaluation_metrics,
                training_summary={"mean_r2": 0.75, "n_high_quality": 100},
                platform_name="23andme_v5",
                prs_id="PGS000004",
                genome_build="GRCh37",
                model_name="Test PRS Model",
                reference_panel_id="1000G_phase3_EUR",
                training_ancestry="EUR",
                include_variance_scaling=True,
            )
        imputation_v2_validator.validate(data)  # raises on invalid

    def test_minimal_export_validates(
        self,
        imputation_v2_validator,
        sample_observed_variants,
        sample_imputed_models,
    ):
        """A minimal export still conforms.

        No calibration/evaluation/training_summary, ``include_variance_scaling``
        off — so the optional top-level blocks are absent, ``centering_scaling``
        is null, and imputed variants carry no ``residual_variance``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(
                tmpdir,
                sample_observed_variants,
                sample_imputed_models,
                include_variance_scaling=False,
            )
        assert "calibration_params" not in data
        assert "evaluation_metrics" not in data
        assert "training_summary" not in data
        assert data["provenance"]["centering_scaling"] is None
        assert all("residual_variance" not in m for m in data["imputed_variants"])
        imputation_v2_validator.validate(data)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda d: d["metadata"].__setitem__("unexpected", 1),
                id="extra-field",
            ),
            pytest.param(
                lambda d: d["imputed_variants"][0]["predictors"][0].pop(
                    "counted_allele"
                ),
                id="missing-predictor-counted-allele",
            ),
            pytest.param(
                lambda d: d["observed_variants"][0].__setitem__("position", "100"),
                id="position-wrong-type",
            ),
            pytest.param(
                lambda d: d["metadata"].__setitem__("format_version", "1.0"),
                id="wrong-format-version",
            ),
        ],
    )
    def test_invalid_artifacts_are_rejected(
        self,
        jsonschema_mod,
        imputation_v2_validator,
        sample_observed_variants,
        sample_imputed_models,
        mutate,
    ):
        """The schema actually constrains: malformed artifacts fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._export(
                tmpdir, sample_observed_variants, sample_imputed_models
            )
        mutate(data)
        with pytest.raises(jsonschema_mod.ValidationError):
            imputation_v2_validator.validate(data)
