"""Tests for projection JSON export (P2.1).

Mirrors ``tests/test_json_export.py`` (the imputation exporter's v2.0 tests) for
the projection product. Asserts the serialized JSON structure, allele orientation,
provenance, deploy gates, and numpy-clean serialization. The full export -> load
-> predict round trip is deferred to P2.2 (the projection loader does not exist
yet), so these tests stop at the serialized artifact.

Convention: exact for ids / alleles / counts; ``np.testing.assert_allclose(...,
rtol=0, atol=1e-12)`` for floats.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearProjectionPRS
from imputed_prs.core.exceptions import ModelNotFittedError
from imputed_prs.core.types import (
    CalibrationParams,
    ImputedVariantModel,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.io.exporters.projection_json_export import (
    _serialize_region_model,
    export_projection_to_json,
)


# =============================================================================
# Helpers / fixtures
# =============================================================================


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
    prs_positions=None,
    prs_effect_alleles=None,
    prs_other_alleles=None,
    target_variance=0.5,
):
    """Build a ProjectionRegionModel with sensible, index-aligned defaults.

    Predictor and PRS allele metadata default to lengths matching their id lists so
    the oriented scorer can resolve every term; override per-test.
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
    m = len(prs_variant_ids)
    if prs_positions is None:
        prs_positions = [start + 1_000_000 * (i + 1) for i in range(m)]
    if prs_effect_alleles is None:
        prs_effect_alleles = ["A"] * m
    if prs_other_alleles is None:
        prs_other_alleles = ["G"] * m
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
        prs_positions=prs_positions,
        prs_effect_alleles=prs_effect_alleles,
        prs_other_alleles=prs_other_alleles,
        target_variance=target_variance,
    )


def _make_observed_variants():
    return [
        VariantInfo("rs100", "1", 500_000, "A", "G", 0.5),
        VariantInfo("rs101", "1", 600_000, "C", "T", -0.3),
    ]


def _make_calibration_params():
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


def _export(observed, regions, **kwargs):
    """Export to a temp file and return the parsed JSON.

    Defaults to ``require_provenance=False`` so structure tests need not supply
    provenance; tests that exercise the provenance gate opt back in.
    """
    kwargs.setdefault("require_provenance", False)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "model.json"
        result = export_projection_to_json(
            output_path=out,
            observed_variants=observed,
            region_models=regions,
            **kwargs,
        )
        assert result == out
        with open(out) as f:
            return json.load(f)


# 20-sample biallelic reference (from tests/test_round_trip.py): partial platform
# overlap so the fitted model has observed (rs1-rs3) and projected (rs4, rs5) terms.
_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\tS5\tS6\tS7\tS8\tS9\tS10\tS11\tS12\tS13\tS14\tS15\tS16\tS17\tS18\tS19\tS20
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1
1\t102000\trs5\tG\tC\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1
"""

_PRS_DF = pd.DataFrame(
    {
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "C"],
        "other_allele": ["A", "C", "G", "T", "G"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    }
)

_PLATFORM = ["rs1", "rs2", "rs3"]


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "ref.vcf"
    path.write_text(_VCF)
    return path


# =============================================================================
# Structure
# =============================================================================


class TestProjectionExportStructure:
    def test_top_level_keys(self):
        data = _export(_make_observed_variants(), [_make_region_model()])
        assert set(data) >= {
            "metadata",
            "provenance",
            "observed_variants",
            "region_models",
            "platform_variant_index",
        }
        assert data["metadata"]["format_version"] == "2.0"

    def test_region_predictors_are_self_describing(self):
        data = _export(_make_observed_variants(), [_make_region_model()])
        region = data["region_models"][0]
        assert set(region) >= {
            "region_id",
            "chromosome",
            "start",
            "end",
            "intercept",
            "cv_mse",
            "cv_r2",
            "is_intercept_only",
            "mean_prs_contribution",
            "predictors",
            "prs_variants",
        }
        for pred in region["predictors"]:
            assert set(pred) == {
                "variant_id",
                "chromosome",
                "position",
                "counted_allele",
                "other_allele",
                "allele_frequency",
                "coefficient",
            }
        for pv in region["prs_variants"]:
            assert set(pv) == {
                "variant_id",
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta",
            }

    def test_prs_variant_chromosome_denormalized_from_region(self):
        region = _make_region_model(chromosome="7", region_id="chr7:1-2")
        data = _export(_make_observed_variants(), [region])
        for pv in data["region_models"][0]["prs_variants"]:
            assert pv["chromosome"] == "7"

    def test_region_values_match_model(self):
        region = _make_region_model(
            prs_variant_ids=["rsZ"],
            betas=np.array([0.42]),
            prs_positions=[2_500_000],
            prs_effect_alleles=["T"],
            prs_other_alleles=["A"],
            predictor_variant_ids=["pX"],
            coefficients=np.array([-0.17]),
            predictor_allele_frequencies=np.array([0.33]),
            predictor_positions=[1_234_567],
            predictor_chromosomes=["1"],
            predictor_counted_alleles=["C"],
            predictor_other_alleles=["G"],
            target_variance=0.37,
        )
        data = _export(_make_observed_variants(), [region])
        r = data["region_models"][0]
        pred, pv = r["predictors"][0], r["prs_variants"][0]
        # Ids / alleles / positions: exact.
        assert pred["variant_id"] == "pX"
        assert pred["counted_allele"] == "C" and pred["other_allele"] == "G"
        assert pred["position"] == 1_234_567
        assert pv["variant_id"] == "rsZ"
        assert pv["effect_allele"] == "T" and pv["other_allele"] == "A"
        assert pv["position"] == 2_500_000
        # Floats: allclose.
        np.testing.assert_allclose(pred["coefficient"], -0.17, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            pred["allele_frequency"], 0.33, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(pv["beta"], 0.42, rtol=0, atol=1e-12)
        # target_variance (P3.3 intercept-only variance) is exported per region.
        np.testing.assert_allclose(r["target_variance"], 0.37, rtol=0, atol=1e-12)

    def test_metadata_counts(self):
        regions = [
            _make_region_model(region_id="r1"),
            _make_region_model(region_id="r2", is_intercept_only=True),
        ]
        data = _export(_make_observed_variants(), regions)
        assert data["metadata"]["n_observed_variants"] == 2
        assert data["metadata"]["n_region_models"] == 2
        assert data["metadata"]["n_intercept_only_regions"] == 1

    def test_platform_variant_index_maps_observed(self):
        data = _export(_make_observed_variants(), [_make_region_model()])
        assert data["platform_variant_index"] == {"rs100": 0, "rs101": 1}

    def test_observed_variant_accepted_ids_and_ambiguous(self):
        data = _export(_make_observed_variants(), [_make_region_model()])
        obs = data["observed_variants"][0]
        assert obs["accepted_ids"] == ["rs100", "1:500000"]
        assert obs["ambiguous"] is False  # A/G is not palindromic
        assert obs["fallback"] is None  # projection fallback (P2.4) not present


# =============================================================================
# Provenance
# =============================================================================


class TestProjectionProvenance:
    def test_provenance_block(self):
        data = _export(
            _make_observed_variants(),
            [_make_region_model()],
            calibration_params=_make_calibration_params(),
            platform_name="23andme_v5",
            genome_build="GRCh37",
            reference_panel_id="1000G_phase3_EUR",
            training_ancestry="EUR",
        )
        prov = data["provenance"]
        assert prov["genome_build"] == "GRCh37"
        assert prov["platform_id"] == "23andme_v5"
        assert prov["reference_panel_id"] == "1000G_phase3_EUR"
        assert prov["training_ancestry"] == "EUR"
        assert prov["ambiguous_policy"] == "exclude_unless_platform_strand_known"
        np.testing.assert_allclose(
            prov["centering_scaling"]["scaling_factor"], 1.05, rtol=0, atol=1e-12
        )

    def test_centering_scaling_none_without_calibration(self):
        data = _export(_make_observed_variants(), [_make_region_model()])
        assert data["provenance"]["centering_scaling"] is None
        assert "calibration_params" not in data

    def test_calibration_block_present_when_given(self):
        data = _export(
            _make_observed_variants(),
            [_make_region_model()],
            calibration_params=_make_calibration_params(),
        )
        assert data["calibration_params"]["n_calibration"] == 200


# =============================================================================
# Numpy-clean serialization (the catcher)
# =============================================================================


class TestProjectionNumpyClean:
    def test_numpy_arrays_and_int64_positions_serialize(self):
        """json.dump must not choke on numpy float64/int64 from the arrays."""
        region = _make_region_model(
            prs_variant_ids=["rsA", "rsB"],
            betas=np.array([0.3, -0.2]),
            prs_positions=[np.int64(2_000_000), np.int64(3_000_000)],
            prs_effect_alleles=["A", "C"],
            prs_other_alleles=["G", "T"],
            predictor_variant_ids=["p0", "p1"],
            coefficients=np.array([0.1, 0.2]),
            predictor_allele_frequencies=np.array([0.3, 0.4]),
            predictor_positions=[np.int64(1_000_001), np.int64(1_000_002)],
            predictor_chromosomes=["1", "1"],
            predictor_counted_alleles=["A", "C"],
            predictor_other_alleles=["G", "T"],
        )
        # Would raise TypeError ("Object of type float64/int64 is not JSON
        # serializable") without the float()/int() casts in the serializer.
        data = _export(_make_observed_variants(), [region])
        r = data["region_models"][0]
        for pred in r["predictors"]:
            assert type(pred["coefficient"]) is float
            assert type(pred["allele_frequency"]) is float
            assert type(pred["position"]) is int
        for pv in r["prs_variants"]:
            assert type(pv["beta"]) is float
            assert type(pv["position"]) is int
        # Top-level region floats are plain floats too.
        assert type(r["intercept"]) is float
        assert type(r["cv_mse"]) is float
        assert type(r["cv_r2"]) is float
        assert type(r["mean_prs_contribution"]) is float
        np.testing.assert_allclose(
            [pv["beta"] for pv in r["prs_variants"]], [0.3, -0.2], rtol=0, atol=1e-12
        )


# =============================================================================
# Deploy gates
# =============================================================================


class TestProjectionDeployGates:
    def test_gate_raises_on_missing_observed_other_allele(self):
        observed = [VariantInfo("rsX", "1", 100, "A", None, 0.1)]
        with pytest.raises(ValueError, match="other_allele"):
            _export(observed, [_make_region_model()], require_other_allele=True)

    def test_gate_raises_on_missing_predictor_other_allele(self):
        region = _make_region_model(predictor_other_alleles=[None, "G", "G"])
        with pytest.raises(ValueError, match="other_allele"):
            _export(_make_observed_variants(), [region], require_other_allele=True)

    def test_gate_raises_on_missing_provenance(self):
        with pytest.raises(ValueError, match="provenance"):
            _export(
                _make_observed_variants(),
                [_make_region_model()],
                require_provenance=True,
            )

    def test_escape_hatch_allows_missing_other_allele(self):
        observed = [VariantInfo("rsX", "1", 100, "A", None, 0.1)]
        data = _export(observed, [_make_region_model()], require_other_allele=False)
        assert data["observed_variants"][0]["other_allele"] is None
        assert data["observed_variants"][0]["ambiguous"] is False

    def test_escape_hatch_allows_missing_provenance(self):
        # _export defaults require_provenance=False; just confirm it writes.
        data = _export(_make_observed_variants(), [_make_region_model()])
        assert data["provenance"]["genome_build"] is None

    def test_prs_variant_other_allele_not_gated(self):
        """Region PRS variants are projected, not counted, so None other_allele is
        allowed even under the default deploy gate."""
        region = _make_region_model(prs_other_alleles=[None])
        data = _export(
            _make_observed_variants(),
            [region],
            require_other_allele=True,  # gate ON; should still pass
        )
        assert data["region_models"][0]["prs_variants"][0]["other_allele"] is None

    def test_prs_other_allele_none_round_trips_as_null(self):
        region = _make_region_model(
            prs_variant_ids=["a", "b"],
            betas=np.array([0.1, 0.2]),
            prs_positions=[1, 2],
            prs_effect_alleles=["A", "C"],
            prs_other_alleles=["G", None],
        )
        data = _export(_make_observed_variants(), [region])
        pvs = data["region_models"][0]["prs_variants"]
        assert pvs[0]["other_allele"] == "G"
        assert pvs[1]["other_allele"] is None


# =============================================================================
# Edge cases
# =============================================================================


class TestProjectionEdgeCases:
    def test_empty_region_models(self):
        data = _export(_make_observed_variants(), [])
        assert data["region_models"] == []
        assert data["metadata"]["n_region_models"] == 0
        assert data["metadata"]["n_intercept_only_regions"] == 0

    def test_intercept_only_region_keeps_nonzero_predictors(self):
        """is_intercept_only with predictors present (zero-shrunk coefficients) must
        still emit those predictors verbatim."""
        region = _make_region_model(is_intercept_only=True)  # keeps 3 predictors
        data = _serialize_region_model(region, include_variance_scaling=True)
        assert data["is_intercept_only"] is True
        assert len(data["predictors"]) == 3

    def test_index_misalignment_raises(self):
        region = _make_region_model()
        region.prs_positions = []  # break alignment vs 1 prs_variant_id
        with pytest.raises(ValueError, match="index-aligned"):
            _serialize_region_model(region, include_variance_scaling=True)

    def test_predictor_misalignment_raises(self):
        region = _make_region_model()
        region.predictor_counted_alleles = ["A"]  # 1 vs 3 predictors
        with pytest.raises(ValueError, match="index-aligned"):
            _serialize_region_model(region, include_variance_scaling=True)


# =============================================================================
# LinearProjectionPRS.export() integration
# =============================================================================


class TestProjectionExportMethod:
    def test_export_with_provenance_writes_json(self, vcf_file, tmp_path):
        pytest.importorskip("cyvcf2")
        model = LinearProjectionPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42
        )
        model.fit(
            reference_genotypes=vcf_file,
            prs_definition=_PRS_DF,
            platform_variants=_PLATFORM,
            genome_build="GRCh37",
            reference_panel_id="1000G_phase3_EUR",
            training_ancestry="EUR",
        )
        paths = model.export(tmp_path)
        assert "json" in paths and paths["json"].exists()

        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["metadata"]["format_version"] == "2.0"
        assert data["provenance"]["genome_build"] == "GRCh37"
        assert data["provenance"]["reference_panel_id"] == "1000G_phase3_EUR"
        assert data["provenance"]["training_ancestry"] == "EUR"
        # Observed terms exist; every region term is self-describing.
        assert data["metadata"]["n_observed_variants"] >= 1
        assert data["region_models"], "expected at least one projected region"
        for region in data["region_models"]:
            for pred in region["predictors"]:
                assert pred["other_allele"] and pred["counted_allele"]
            for pv in region["prs_variants"]:
                assert pv["effect_allele"]

    def test_export_without_provenance_raises(self, vcf_file, tmp_path):
        pytest.importorskip("cyvcf2")
        model = LinearProjectionPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42
        )
        model.fit(
            reference_genotypes=vcf_file,
            prs_definition=_PRS_DF,
            platform_variants=_PLATFORM,
        )
        with pytest.raises(ValueError, match="provenance"):
            model.export(tmp_path)

    def test_export_unsupported_format_raises(self, vcf_file, tmp_path):
        pytest.importorskip("cyvcf2")
        model = LinearProjectionPRS(
            window_size=500_000, cv_folds=3, verbose=0, random_state=42
        )
        model.fit(
            reference_genotypes=vcf_file,
            prs_definition=_PRS_DF,
            platform_variants=_PLATFORM,
            genome_build="GRCh37",
            reference_panel_id="1000G_phase3_EUR",
            training_ancestry="EUR",
        )
        with pytest.raises(ValueError, match="Unsupported export formats"):
            model.export(tmp_path, formats=["hdf5"])

    def test_export_not_fitted_raises(self, tmp_path):
        model = LinearProjectionPRS(verbose=0)
        with pytest.raises(ModelNotFittedError):
            model.export(tmp_path)


# =============================================================================
# JSON Schema validation
# =============================================================================

# Committed machine-readable schema for the v2.0 deployable projection artifact.
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "projection_model_v2.schema.json"
)


@pytest.fixture
def jsonschema_mod():
    """The jsonschema module; schema tests skip if the dev dep is absent."""
    return pytest.importorskip("jsonschema")


@pytest.fixture
def projection_v2_validator(jsonschema_mod):
    """A validator built from the committed v2 projection schema.

    Building it also asserts the committed schema is itself a valid Draft
    2020-12 schema (``check_schema``), so a malformed schema fails loudly.
    """
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    validator_cls = jsonschema_mod.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


class TestSchemaValidation:
    """Validate exported v2 projection artifacts against the committed schema."""

    def test_committed_schema_is_valid(self, projection_v2_validator):
        """The committed schema parses and passes Draft 2020-12 check_schema."""
        # Reaching here means the fixture built the validator without raising.
        assert projection_v2_validator is not None

    def test_full_export_validates(self, projection_v2_validator):
        """A fully-populated deployable export conforms to the schema."""
        data = _export(
            _make_observed_variants(),
            [
                _make_region_model(region_id="r1"),
                _make_region_model(region_id="r2", is_intercept_only=True),
            ],
            calibration_params=_make_calibration_params(),
            training_summary={"mean_r2": 0.75, "n_high_quality": 100},
            platform_name="23andme_v5",
            prs_id="PGS000004",
            genome_build="GRCh37",
            model_name="Test PRS Model",
            reference_panel_id="1000G_phase3_EUR",
            training_ancestry="EUR",
            include_variance_scaling=True,
            require_provenance=True,
        )
        projection_v2_validator.validate(data)  # raises on invalid

    def test_minimal_export_validates(self, projection_v2_validator):
        """A minimal export still conforms.

        No calibration/training_summary, so the optional top-level blocks are
        absent and ``centering_scaling`` is null.
        """
        data = _export(_make_observed_variants(), [_make_region_model()])
        assert "calibration_params" not in data
        assert "training_summary" not in data
        assert data["provenance"]["centering_scaling"] is None
        projection_v2_validator.validate(data)

    def test_empty_region_models_validates(self, projection_v2_validator):
        """An export with no projected regions still conforms."""
        data = _export(_make_observed_variants(), [])
        assert data["region_models"] == []
        projection_v2_validator.validate(data)

    def test_observed_fallback_validates(self, projection_v2_validator):
        """An observed variant carrying a populated fallback model conforms (P2.4)."""
        fallback = ImputedVariantModel(
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
        )
        observed = [
            VariantInfo("rs100", "1", 500_000, "A", "G", 0.5, fallback=fallback),
            VariantInfo("rs101", "1", 600_000, "C", "T", -0.3),  # fallback=None
        ]
        data = _export(observed, [_make_region_model()])
        assert data["observed_variants"][0]["fallback"] is not None
        assert data["observed_variants"][1]["fallback"] is None
        projection_v2_validator.validate(data)

    def test_prs_variant_null_other_allele_validates(self, projection_v2_validator):
        """A region PRS variant with a null other_allele conforms (it is allowed
        to be null since PRS variants are projected, not counted)."""
        region = _make_region_model(prs_other_alleles=[None])
        data = _export(_make_observed_variants(), [region])
        assert data["region_models"][0]["prs_variants"][0]["other_allele"] is None
        projection_v2_validator.validate(data)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda d: d["metadata"].__setitem__("unexpected", 1),
                id="extra-field",
            ),
            pytest.param(
                lambda d: d["region_models"][0]["predictors"][0].pop(
                    "counted_allele"
                ),
                id="missing-predictor-counted-allele",
            ),
            pytest.param(
                lambda d: d["region_models"][0].pop("target_variance"),
                id="missing-region-target-variance",
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
        self, jsonschema_mod, projection_v2_validator, mutate
    ):
        """The schema actually constrains: malformed artifacts fail validation."""
        data = _export(_make_observed_variants(), [_make_region_model()])
        mutate(data)
        with pytest.raises(jsonschema_mod.ValidationError):
            projection_v2_validator.validate(data)
