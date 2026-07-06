"""Round-trip and numeric-vs-string golden tests for the unified scorers (P1.6).

Two guarantees are locked here:

1. **Round trip** — an imputation model exported and re-loaded scores a user
   upload identically to the in-memory model, through the public ``predict``
   (the browser/upload path), for every export format. Exact for ids/counts,
   allclose (atol=1e-12) for floats.
2. **Golden numeric == string** — on hard-called integer dosages the evaluator's
   numeric predicted-PRS path equals its string-render path (which replays the
   browser scorer), for imputation and projection, both in the common
   effect==ALT orientation and a flipped (effect==REF) panel. This is what keeps
   train/eval and the browser from diverging on continuous data, where only the
   numeric path can run.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.core.exceptions import DataLoadError
from imputed_prs.core.types import (
    GenotypeData,
    ImputedVariantModel,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.evaluation import ImputationEvaluator
from imputed_prs.evaluation._scoring import is_hard_called
from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator


# =============================================================================
# Fixtures
# =============================================================================

# 20-sample biallelic reference. effect_allele == ALT for every PRS variant
# (the common case); platform is a partial overlap so the model has both
# observed (rs1-rs3) and imputed/projected (rs4, rs5) terms with real predictors.
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

# Standard panel orientation: ref/alt match the PRS (other/effect).
_REF_ALLELES = ["A", "C", "G", "T", "G"]
_ALT_ALLELES = ["G", "T", "A", "C", "C"]


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "ref.vcf"
    path.write_text(_VCF)
    return path


@pytest.fixture
def fitted_imputation_model(vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file,
        prs_definition=_PRS_DF,
        platform_variants=_PLATFORM,
        # Provenance so the JSON round trip exercises a deployable artifact (P1.7).
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def fitted_projection_model(vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearProjectionPRS(
        window_size=500_000, cv_folds=3, verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file,
        prs_definition=_PRS_DF,
        platform_variants=_PLATFORM,
        # Provenance so the JSON round trip exercises a deployable artifact (P1.7);
        # symmetric with fitted_imputation_model.
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def user_genotype_df():
    """A user upload covering the platform variants (rs1-rs3), as genotype strings."""
    return pd.DataFrame(
        {
            "rsid": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "1"],
            "position": [100000, 100500, 101000],
            "genotype": ["AG", "CC", "AA"],
        }
    )


def _genotype_data(ref_alleles, alt_alleles, dosage_matrix):
    """Build a GenotypeData over rs1..rsN with the given alleles and dosages."""
    n_variants = len(ref_alleles)
    variant_info = pd.DataFrame(
        {
            "variant_id": [f"rs{i + 1}" for i in range(n_variants)],
            "chromosome": ["1"] * n_variants,
            "position": [100000, 100500, 101000, 101500, 102000][:n_variants],
            "ref_allele": ref_alleles,
            "alt_allele": alt_alleles,
        }
    )
    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=[f"S{i + 1}" for i in range(dosage_matrix.shape[0])],
    )


@pytest.fixture
def integer_genotype_data():
    """Hard-called integer dosages, standard (effect==ALT) panel orientation."""
    dm = np.random.default_rng(7).integers(0, 3, size=(15, 5)).astype(np.float32)
    return _genotype_data(list(_REF_ALLELES), list(_ALT_ALLELES), dm)


@pytest.fixture
def flipped_integer_genotype_data():
    """Hard-called integers with ref/alt swapped vs training, so the model's
    counted allele is the panel REF: exercises the 2-dosage flip in
    match_oriented_dosage and the matching count via the rendered string."""
    dm = np.random.default_rng(8).integers(0, 3, size=(15, 5)).astype(np.float32)
    return _genotype_data(list(_ALT_ALLELES), list(_REF_ALLELES), dm)


@pytest.fixture
def continuous_genotype_data():
    """Continuous DS-style dosages (no genotype string can be rendered)."""
    dm = np.random.default_rng(9).uniform(0.0, 2.0, size=(12, 5))
    return _genotype_data(list(_REF_ALLELES), list(_ALT_ALLELES), dm)


# =============================================================================
# Round trip: export -> load -> predict
# =============================================================================


class TestImputationRoundTrip:
    @pytest.mark.parametrize(
        "fmt,dep",
        [
            ("json", None),
            ("hdf5", "h5py"),
            ("arrow", "pyarrow"),
            ("parquet", "pyarrow"),
            ("csv", None),
        ],
    )
    def test_export_load_predict_equivalence(
        self, fitted_imputation_model, user_genotype_df, tmp_path, fmt, dep
    ):
        if dep is not None:
            pytest.importorskip(dep)
        model = fitted_imputation_model
        paths = model.export(tmp_path, model_name="rt", formats=[fmt])
        loaded = LinearImputationPRS.load(paths[fmt])

        # The model declares GRCh37; pass it so the build guard stays silent.
        r0 = model.predict(user_genotype_df, apply_calibration=False, genome_build="GRCh37")
        r1 = loaded.predict(user_genotype_df, apply_calibration=False, genome_build="GRCh37")

        # Floats: allclose. The round trip must not perturb the score.
        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component, r1.prs_imputed_component],
            [r0.prs, r0.prs_observed_component, r0.prs_imputed_component],
            rtol=0,
            atol=1e-12,
            err_msg=f"format={fmt}",
        )
        # Ids / counts: exact.
        assert r1.n_variants_used == r0.n_variants_used, fmt
        assert r1.unresolved_observed_ids == r0.unresolved_observed_ids, fmt

    def test_fitted_model_has_real_predictors(self, fitted_imputation_model):
        # Guards the golden tests: at least one imputed model must actually use
        # predictors, else numeric==string would be trivially true.
        assert any(
            not m.is_intercept_only and len(m.predictor_variant_ids) > 0
            for m in fitted_imputation_model.imputed_models
        )

    def test_fitted_model_trains_observed_fallbacks(self, fitted_imputation_model):
        # Guards the fallback round trip below: every in-reference observed
        # variant (rs1-rs3) must carry a per-variant fallback model (P1.8).
        observed = fitted_imputation_model.observed_variants
        assert observed, "expected observed variants"
        assert all(v.fallback is not None for v in observed)
        assert fitted_imputation_model.summary["n_observed_with_fallback"] == len(
            observed
        )

    @pytest.mark.parametrize(
        "fmt,dep",
        [
            ("json", None),
            ("hdf5", "h5py"),
            ("arrow", "pyarrow"),
            ("parquet", "pyarrow"),
            ("csv", None),
        ],
    )
    def test_round_trip_recovers_observed_fallback(
        self, fitted_imputation_model, tmp_path, fmt, dep
    ):
        """An upload that no-calls an observed variant recovers it via fallback,
        and that recovery survives export->load for *every* format (P1.8)."""
        if dep is not None:
            pytest.importorskip(dep)
        model = fitted_imputation_model
        # rs2 is a no-call; rs1/rs3 are called so rs2's fallback can use them.
        upload = pd.DataFrame(
            {
                "rsid": ["rs1", "rs2", "rs3"],
                "chromosome": ["1", "1", "1"],
                "position": [100000, 100500, 101000],
                "genotype": ["AG", "--", "AA"],
            }
        )
        paths = model.export(tmp_path, model_name="fb", formats=[fmt])
        loaded = LinearImputationPRS.load(paths[fmt])

        r0 = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        r1 = loaded.predict(upload, apply_calibration=False, genome_build="GRCh37")

        # rs2 recovered via fallback (not dropped), both in-memory and loaded.
        assert r0.n_observed_scored_via_fallback == 1, fmt
        assert r1.n_observed_scored_via_fallback == 1, fmt
        assert r0.unresolved_observed_ids == (), fmt
        assert r1.unresolved_observed_ids == (), fmt
        # The fallback recovery survives the round trip to float tolerance, so the
        # serialized fallback block (JSON nest / flat observed_fallbacks) is exact.
        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component],
            [r0.prs, r0.prs_observed_component],
            rtol=0,
            atol=1e-12,
            err_msg=f"format={fmt}",
        )
        # SE round trip (P4.1): calibration-carrying formats reproduce the
        # empirical-floored SE exactly. CSV carries no calibration, so its SE
        # falls back to the per-user diagonal — which must still equal the
        # in-memory diagonal lower bound, proving the fallback variance (the only
        # SE input CSV preserves) round-tripped.
        if fmt == "csv":
            np.testing.assert_allclose(
                r1.se, r0.se_diagonal_lower_bound, rtol=0, atol=1e-12
            )
        else:
            np.testing.assert_allclose(r1.se, r0.se, rtol=0, atol=1e-12, err_msg=fmt)


class TestProjectionRoundTrip:
    """Export -> load() -> predict equivalence for the projection product (P2.2).

    Projection exports to JSON only, so this is the single-format analog of
    TestImputationRoundTrip: a reloaded model must score a user upload identically
    to the in-memory model through the public ``predict`` (the browser/upload path).
    """

    def test_export_load_predict_equivalence(
        self, fitted_projection_model, user_genotype_df, tmp_path
    ):
        model = fitted_projection_model
        paths = model.export(tmp_path, model_name="rt", formats=["json"])
        loaded = LinearProjectionPRS.load(paths["json"])

        # The model declares GRCh37; pass it so the build guard stays silent.
        r0 = model.predict(
            user_genotype_df, apply_calibration=False, genome_build="GRCh37"
        )
        r1 = loaded.predict(
            user_genotype_df, apply_calibration=False, genome_build="GRCh37"
        )

        # Floats: allclose. The round trip must not perturb the score.
        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component, r1.prs_imputed_component],
            [r0.prs, r0.prs_observed_component, r0.prs_imputed_component],
            rtol=0,
            atol=1e-12,
        )
        # Ids / counts: exact.
        assert r1.n_variants_used == r0.n_variants_used
        assert r1.unresolved_observed_ids == r0.unresolved_observed_ids

    def test_fitted_model_trains_observed_fallbacks(self, fitted_projection_model):
        # Guards the fallback round trip below: every in-reference observed
        # variant (rs1-rs3) must carry a per-variant fallback model (P2.4).
        observed = fitted_projection_model.observed_variants
        assert observed, "expected observed variants"
        assert all(v.fallback is not None for v in observed)
        assert fitted_projection_model.summary["n_observed_with_fallback"] == len(
            observed
        )

    def test_round_trip_recovers_observed_fallback(
        self, fitted_projection_model, tmp_path
    ):
        """An upload that no-calls an observed variant recovers it via fallback,
        and that recovery survives the JSON export->load (P2.4)."""
        model = fitted_projection_model
        # rs2 is a no-call; rs1/rs3 are called so rs2's fallback can use them.
        upload = pd.DataFrame(
            {
                "rsid": ["rs1", "rs2", "rs3"],
                "chromosome": ["1", "1", "1"],
                "position": [100000, 100500, 101000],
                "genotype": ["AG", "--", "AA"],
            }
        )
        loaded = LinearProjectionPRS.load(
            model.export(tmp_path, model_name="fb", formats=["json"])["json"]
        )

        r0 = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        r1 = loaded.predict(upload, apply_calibration=False, genome_build="GRCh37")

        # rs2 recovered via fallback (not dropped), both in-memory and loaded.
        assert r0.n_observed_scored_via_fallback == 1
        assert r1.n_observed_scored_via_fallback == 1
        assert r0.unresolved_observed_ids == ()
        assert r1.unresolved_observed_ids == ()
        # The fallback recovery survives the round trip to float tolerance, so the
        # serialized projection fallback block is exact.
        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component, r1.se],
            [r0.prs, r0.prs_observed_component, r0.se],
            rtol=0,
            atol=1e-12,
        )

    def test_calibration_survives_round_trip(
        self, fitted_projection_model, user_genotype_df, tmp_path
    ):
        """Calibration params restore, so the calibrated score round-trips too."""
        model = fitted_projection_model
        loaded = LinearProjectionPRS.load(
            model.export(tmp_path, formats=["json"])["json"]
        )
        assert (loaded.calibration_params is None) == (
            model.calibration_params is None
        )
        r0 = model.predict(
            user_genotype_df, apply_calibration=True, genome_build="GRCh37"
        )
        r1 = loaded.predict(
            user_genotype_df, apply_calibration=True, genome_build="GRCh37"
        )
        np.testing.assert_allclose(r1.prs, r0.prs, rtol=0, atol=1e-12)
        if r0.prs_scaled is not None:
            np.testing.assert_allclose(
                r1.prs_scaled, r0.prs_scaled, rtol=0, atol=1e-12
            )

    def test_provenance_restored(self, fitted_projection_model, tmp_path):
        """The build/platform guard's inputs survive the round trip (P1.7)."""
        model = fitted_projection_model
        loaded = LinearProjectionPRS.load(
            model.export(tmp_path, formats=["json"])["json"]
        )
        assert loaded._genome_build == "GRCh37"
        assert loaded._reference_panel_id == "1000G_phase3_EUR"
        assert loaded._training_ancestry == "EUR"

    def test_summary_works_after_load(self, fitted_projection_model, tmp_path):
        """summary tolerates the un-serialized training_result (stays None)."""
        model = fitted_projection_model
        loaded = LinearProjectionPRS.load(
            model.export(tmp_path, formats=["json"])["json"]
        )
        summary = loaded.summary
        assert summary["n_regions"] == len(model.region_models)

    def test_load_rejects_non_json(self, tmp_path):
        """Projection exports JSON only; a non-JSON path is a clear error."""
        bad = tmp_path / "model.hdf5"
        bad.write_text("not json")
        with pytest.raises(DataLoadError):
            LinearProjectionPRS.load(bad)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(DataLoadError):
            LinearProjectionPRS.load(tmp_path / "does_not_exist.json")


# =============================================================================
# Golden: numeric path == rendered-string path on integer dosages
# =============================================================================


class TestNumericVsStringGolden:
    def test_imputation_numeric_equals_string(
        self, fitted_imputation_model, integer_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(integer_genotype_data),
            ev._predicted_prs_via_strings(integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_imputation_numeric_equals_string_flipped(
        self, fitted_imputation_model, flipped_integer_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(flipped_integer_genotype_data),
            ev._predicted_prs_via_strings(flipped_integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_projection_numeric_equals_string(
        self, fitted_projection_model, integer_genotype_data
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(integer_genotype_data),
            ev._predicted_prs_via_strings(integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_projection_numeric_equals_string_flipped(
        self, fitted_projection_model, flipped_integer_genotype_data
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(flipped_integer_genotype_data),
            ev._predicted_prs_via_strings(flipped_integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )


# =============================================================================
# Dispatch: hard-call -> string path, continuous -> numeric path
# =============================================================================


class TestDosageModeDispatch:
    def test_hardcall_dispatches_to_numeric_path(
        self, fitted_imputation_model, integer_genotype_data, monkeypatch
    ):
        """P5: hard-called panels now score through the numeric path, not the
        per-sample string replay. Proven by making the string path raise: the
        dispatch must not touch it, and must return the numeric result."""
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        assert is_hard_called(integer_genotype_data.dosage_matrix)

        def _boom(*_a, **_k):
            raise AssertionError("string replay must not run on the metric path (P5)")

        monkeypatch.setattr(ev, "_predicted_prs_via_strings", _boom)
        np.testing.assert_allclose(
            ev._compute_imputed_prs_batch(integer_genotype_data),
            ev._predicted_prs_numeric(integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_projection_hardcall_dispatches_to_numeric_path(
        self, fitted_projection_model, integer_genotype_data, monkeypatch
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        assert is_hard_called(integer_genotype_data.dosage_matrix)

        def _boom(*_a, **_k):
            raise AssertionError("string replay must not run on the metric path (P5)")

        monkeypatch.setattr(ev, "_predicted_prs_via_strings", _boom)
        np.testing.assert_allclose(
            ev._compute_projected_prs_batch(integer_genotype_data),
            ev._predicted_prs_numeric(integer_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_continuous_dispatches_to_numeric_path(
        self, fitted_imputation_model, continuous_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        assert not is_hard_called(continuous_genotype_data.dosage_matrix)
        np.testing.assert_allclose(
            ev._compute_imputed_prs_batch(continuous_genotype_data),
            ev._predicted_prs_numeric(continuous_genotype_data),
            rtol=0,
            atol=1e-12,
        )

    def test_projection_continuous_dispatches_to_numeric_path(
        self, fitted_projection_model, continuous_genotype_data
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        assert not is_hard_called(continuous_genotype_data.dosage_matrix)
        np.testing.assert_allclose(
            ev._compute_projected_prs_batch(continuous_genotype_data),
            ev._predicted_prs_numeric(continuous_genotype_data),
            rtol=0,
            atol=1e-12,
        )


class TestEmpiricalResidualCalibration:
    """P4.1 end-to-end: a fitted model carries the empirical residual SDs and
    ``predict`` reports the empirical-floored SE (rule B), for both products."""

    def _upload(self):
        return pd.DataFrame(
            {
                "rsid": ["rs1", "rs2", "rs3"],
                "chromosome": ["1", "1", "1"],
                "position": [100000, 100500, 101000],
                "genotype": ["AG", "CT", "AA"],
            }
        )

    def _check(self, model):
        cp = model.calibration_params
        assert cp is not None
        # All three P4.1 fields populated and finite on a real fit.
        for field in (
            "raw_empirical_residual_sd",
            "calibrated_empirical_residual_sd",
            "diagonal_model_se_lower_bound",
        ):
            val = getattr(cp, field)
            assert val is not None and np.isfinite(val) and val >= 0.0, field

        r = model.predict(
            self._upload(), apply_calibration=True, genome_build="GRCh37"
        )
        assert r.se_diagonal_lower_bound is not None
        # Rule (B) identity: reported SE == max(empirical baseline, per-user diagonal).
        np.testing.assert_allclose(
            r.se,
            max(cp.raw_empirical_residual_sd, r.se_diagonal_lower_bound),
            rtol=0,
            atol=1e-12,
        )
        assert r.se >= cp.raw_empirical_residual_sd - 1e-12
        assert r.se >= r.se_diagonal_lower_bound - 1e-12
        # Calibrated SE == max(calibrated empirical, |slope| * per-user diagonal).
        np.testing.assert_allclose(
            r.se_scaled,
            max(
                cp.calibrated_empirical_residual_sd,
                abs(cp.scaling_factor) * r.se_diagonal_lower_bound,
            ),
            rtol=0,
            atol=1e-12,
        )
        # CI uses the reported SE.
        np.testing.assert_allclose(r.ci_lower, r.prs - 1.96 * r.se, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.ci_upper, r.prs + 1.96 * r.se, rtol=0, atol=1e-12)

    def test_imputation_reports_empirical_se(self, fitted_imputation_model):
        self._check(fitted_imputation_model)

    def test_projection_reports_empirical_se(self, fitted_projection_model):
        self._check(fitted_projection_model)


# =============================================================================
# Phase 4: vectorized batch scorer parity (forced onto the golden fixtures)
# =============================================================================
class TestForcedBatchParity:
    """The size-selected batch path must match the per-unit oracle at atol=1e-9.

    The golden fixtures have only 2 imputed/projected targets, so they naturally
    take the oracle; ``_force_batch=True`` drives the CSR/collapse path over the
    same tiny model to compare the two. Bit identity is impossible (SpMM reorder),
    hence atol=1e-9 rather than the golden 1e-12.
    """

    def test_imputation_batch_equals_oracle_continuous(
        self, fitted_imputation_model, continuous_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        oracle = ev._predicted_prs_numeric(continuous_genotype_data, _force_batch=False)
        batch = ev._predicted_prs_numeric(continuous_genotype_data, _force_batch=True)
        np.testing.assert_allclose(batch, oracle, rtol=0.0, atol=1e-9)

    def test_imputation_batch_equals_oracle_integer(
        self, fitted_imputation_model, integer_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        oracle = ev._predicted_prs_numeric(integer_genotype_data, _force_batch=False)
        batch = ev._predicted_prs_numeric(integer_genotype_data, _force_batch=True)
        np.testing.assert_allclose(batch, oracle, rtol=0.0, atol=1e-9)

    def test_imputation_batch_equals_oracle_flipped(
        self, fitted_imputation_model, flipped_integer_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        oracle = ev._predicted_prs_numeric(
            flipped_integer_genotype_data, _force_batch=False
        )
        batch = ev._predicted_prs_numeric(
            flipped_integer_genotype_data, _force_batch=True
        )
        np.testing.assert_allclose(batch, oracle, rtol=0.0, atol=1e-9)

    def test_projection_batch_equals_oracle_continuous(
        self, fitted_projection_model, continuous_genotype_data
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        oracle = ev._predicted_prs_numeric(continuous_genotype_data, _force_batch=False)
        batch = ev._predicted_prs_numeric(continuous_genotype_data, _force_batch=True)
        np.testing.assert_allclose(batch, oracle, rtol=0.0, atol=1e-9)

    def test_projection_batch_equals_oracle_flipped(
        self, fitted_projection_model, flipped_integer_genotype_data
    ):
        ev = ProjectionEvaluator(fitted_projection_model, verbose=0)
        oracle = ev._predicted_prs_numeric(
            flipped_integer_genotype_data, _force_batch=False
        )
        batch = ev._predicted_prs_numeric(
            flipped_integer_genotype_data, _force_batch=True
        )
        np.testing.assert_allclose(batch, oracle, rtol=0.0, atol=1e-9)


# =============================================================================
# P5: hard-called panels score through the numeric path (string replay retired
# from the metric path). The module fixtures above are fully-called biallelic
# panels; these lock the parity the routing relies on on the two panels the
# fixtures do not reach — no-call (NaN) samples, and multiallelic loci — using
# hermetic fitted models that drive the *real* evaluator methods. Observed
# variants carry NO fallback here so the P1.8 deviation cannot confound the
# biallelic parity checks (that deviation is exercised in TestP18FallbackGuard).
# =============================================================================


def _imputed_model(
    vid, chrom, pos, beta, intercept, predictors, coeffs, *, is_intercept_only=False
):
    """predictors: list of (pid, chrom, pos, counted, other, af)."""
    return ImputedVariantModel(
        variant_id=vid,
        chromosome=chrom,
        position=pos,
        effect_allele="A",
        other_allele="G",
        beta=beta,
        allele_frequency=0.1,
        imputation_r2=0.5,
        residual_variance=0.1,
        intercept=intercept,
        predictor_variant_ids=[p[0] for p in predictors],
        coefficients=np.asarray(coeffs, dtype=np.float64),
        is_intercept_only=is_intercept_only,
        predictor_chromosomes=[p[1] for p in predictors],
        predictor_positions=[p[2] for p in predictors],
        predictor_counted_alleles=[p[3] for p in predictors],
        predictor_other_alleles=[p[4] for p in predictors],
        predictor_allele_frequencies=np.asarray(
            [p[5] for p in predictors], dtype=np.float64
        ),
    )


def _region_model(rid, chrom, betas, predictors, coeffs, intercept):
    return ProjectionRegionModel(
        region_id=rid,
        chromosome=chrom,
        start=0,
        end=10_000,
        prs_variant_ids=[f"{rid}:prs{i}" for i in range(len(betas))],
        betas=np.asarray(betas, dtype=np.float64),
        predictor_variant_ids=[p[0] for p in predictors],
        coefficients=np.asarray(coeffs, dtype=np.float64),
        intercept=intercept,
        cv_mse=0.1,
        cv_r2=0.5,
        is_intercept_only=False,
        mean_prs_contribution=0.0,
        predictor_allele_frequencies=np.asarray(
            [p[5] for p in predictors], dtype=np.float64
        ),
        predictor_chromosomes=[p[1] for p in predictors],
        predictor_positions=[p[2] for p in predictors],
        predictor_counted_alleles=[p[3] for p in predictors],
        predictor_other_alleles=[p[4] for p in predictors],
    )


def _panel(records):
    """records: (chrom, pos, ref, alt, dosages) -> hard-called GenotypeData with
    variant_id 'chrom:pos:ref:alt'. A NaN dosage models a no-call."""
    vi = pd.DataFrame(
        [
            {
                "variant_id": f"{c}:{p}:{r}:{a}",
                "chromosome": c,
                "position": p,
                "ref_allele": r,
                "alt_allele": a,
            }
            for c, p, r, a, _ in records
        ]
    )
    dm = np.array([d for *_, d in records], dtype=np.float32).T
    return GenotypeData(
        dosage_matrix=dm,
        variant_info=vi,
        sample_ids=[f"s{i}" for i in range(dm.shape[0])],
        genome_build="GRCh37",
        source_file=None,
    )


def _imputation_model(observed, imputed):
    return LinearImputationPRS._from_components(
        observed, imputed, None, None, {"genome_build": "GRCh37"}
    )


def _projection_model(observed, regions):
    m = LinearProjectionPRS()
    m._is_fitted = True
    m._observed_variants = observed
    m._region_models = regions
    return m


# A biallelic hard-called panel with no-calls (NaN) in a direct predictor, a
# flipped (counted==REF) predictor, and an observed term. Observed variants carry
# no fallback, so both paths drop no-call observed samples identically.
_NOCALL_RECORDS = [
    ("1", 100, "A", "G", [0.0, 1.0, 2.0, np.nan, 1.0, 2.0]),  # p_direct, no-call s3
    ("1", 200, "C", "T", [0.0, 2.0, 1.0, 2.0, np.nan, 0.0]),  # p_flip (2-dosage), no-call s4
    ("1", 300, "A", "G", [1.0, 1.0, np.nan, 0.0, 2.0, 1.0]),  # observed obs1, no-call s2
    ("2", 50, "G", "C", [1.0, 0.0, 2.0, 1.0, 0.0, 2.0]),      # observed obs2
]
_NOCALL_OBSERVED = [
    VariantInfo("1:300:A:G", "1", 300, "G", "A", 0.3, fallback=None),
    VariantInfo("2:50:G:C", "2", 50, "C", "G", -0.2, fallback=None),
]
_P_DIRECT = ("1:100:A:G", "1", 100, "G", "A", 0.3)
_P_FLIP = ("1:200:C:T", "1", 200, "C", "T", 0.4)


class TestHardCalledNoCallParity:
    """On a hard-called panel with no-call (NaN) samples, the numeric metric path
    must equal the retired string replay element-wise: both mean-substitute a
    predictor no-call by ``2*AF`` and drop a fallback-free observed no-call. The
    dispatch, the per-unit oracle, and the vectorized batch must all agree with
    the string oracle at ``atol=1e-9``."""

    def test_imputation_no_call_matches_string(self):
        panel = _panel(_NOCALL_RECORDS)
        model = _imputation_model(
            _NOCALL_OBSERVED,
            [
                _imputed_model(
                    "1:400:A:G", "1", 400, 1.5, 0.1, [_P_DIRECT, _P_FLIP], [0.5, 0.25]
                ),
                _imputed_model("1:500:A:G", "1", 500, 0.7, -0.2, [], [], is_intercept_only=True),
            ],
        )
        ev = ImputationEvaluator(model, verbose=0)
        string = ev._predicted_prs_via_strings(panel)
        np.testing.assert_allclose(ev._compute_imputed_prs_batch(panel), string, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(panel, _force_batch=False), string, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(panel, _force_batch=True), string, rtol=0, atol=1e-9
        )

    def test_projection_no_call_matches_string(self):
        panel = _panel(_NOCALL_RECORDS)
        model = _projection_model(
            _NOCALL_OBSERVED,
            [
                _region_model(
                    "r1", "1", [0.8, -0.4], [_P_DIRECT, _P_FLIP], [0.5, 0.25], 0.1
                ),
            ],
        )
        ev = ProjectionEvaluator(model, verbose=0)
        string = ev._predicted_prs_via_strings(panel)
        np.testing.assert_allclose(ev._compute_projected_prs_batch(panel), string, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(panel, _force_batch=False), string, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(panel, _force_batch=True), string, rtol=0, atol=1e-9
        )


class TestP18FallbackGuard:
    """The numeric observed scorer drops (does not fallback-recover) an
    unresolvable/no-call observed variant that carries a trained fallback (P1.8).
    That deviation must be loud, never silent."""

    def test_warns_when_fallback_variant_unresolved(self):
        fb = _imputed_model(
            "fb", "1", 100, 1.0, 0.0, [_P_DIRECT], [0.5]
        )
        # Observed locus 9:999 is absent from the panel -> resolver returns None.
        observed = [VariantInfo("9:999:G:A", "9", 999, "G", "A", 0.3, fallback=fb)]
        panel = _panel([("1", 100, "A", "G", [0.0, 1.0, 2.0])])
        ev = ImputationEvaluator(_imputation_model(observed, []), verbose=0)
        with pytest.warns(UserWarning, match="P1.8"):
            ev._predicted_prs_numeric(panel)

    def test_warns_when_fallback_variant_has_no_call(self):
        fb = _imputed_model("fb", "1", 100, 1.0, 0.0, [_P_DIRECT], [0.5])
        observed = [VariantInfo("1:300:A:G", "1", 300, "G", "A", 0.3, fallback=fb)]
        panel = _panel([("1", 300, "A", "G", [1.0, np.nan, 2.0])])  # no-call at s1
        ev = ImputationEvaluator(_imputation_model(observed, []), verbose=0)
        with pytest.warns(UserWarning, match="P1.8"):
            ev._predicted_prs_numeric(panel)

    def test_no_warning_when_fully_resolved(self, fitted_imputation_model, integer_genotype_data):
        # fitted fixture's observed variants carry fallbacks, but are fully
        # resolved/called here -> the guard must stay silent.
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev._predicted_prs_numeric(integer_genotype_data)
        assert not [w for w in caught if "P1.8" in str(w.message)]


class TestMultiallelicCoPredictorDeviation:
    """P5 deviation: when two ALT alleles of one multiallelic locus are both used
    as co-predictors for the same target, the retired string replay conflates the
    locus (chr:pos duplicate-conflict -> mean-fill) while the numeric path
    resolves each allele into its own column, exactly as the model was trained.
    The numeric result is the training-faithful one; routing hard-called panels
    through it is an intended improvement, not a regression. (Biallelic loci and
    multiallelic loci with a single used allele agree exactly -- see the no-call
    and golden parity tests.)"""

    _RECORDS = [
        ("1", 100, "A", "G", [0.0, 1.0, 2.0, 0.0]),
        ("1", 100, "A", "T", [2.0, 1.0, 0.0, 1.0]),  # multiallelic sibling at 1:100
        ("1", 200, "C", "T", [0.0, 2.0, 1.0, 2.0]),
    ]
    _P_G = ("1:100:A:G", "1", 100, "G", "A", 0.3)
    _P_T = ("1:100:A:T", "1", 100, "T", "A", 0.2)

    def _ev(self):
        model = _imputation_model(
            [],
            [
                _imputed_model(
                    "1:400:A:G", "1", 400, 1.5, 0.1,
                    [self._P_G, self._P_T, _P_FLIP], [0.5, -0.3, 0.25],
                )
            ],
        )
        return ImputationEvaluator(model, verbose=0), _panel(self._RECORDS)

    def test_numeric_paths_are_self_consistent(self):
        ev, panel = self._ev()
        oracle = ev._predicted_prs_numeric(panel, _force_batch=False)
        # The dispatch routes to numeric, and the vectorized batch agrees with it.
        np.testing.assert_allclose(ev._compute_imputed_prs_batch(panel), oracle, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            ev._predicted_prs_numeric(panel, _force_batch=True), oracle, rtol=0, atol=1e-9
        )

    def test_documents_divergence_from_retired_string_path(self):
        """Executable documentation of the intended behavior change: the numeric
        path and the retired string replay differ at multiallelic co-predictors."""
        ev, panel = self._ev()
        numeric = ev._predicted_prs_numeric(panel)
        string = ev._predicted_prs_via_strings(panel)
        assert np.max(np.abs(numeric - string)) > 0.1
