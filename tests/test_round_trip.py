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

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.core.types import GenotypeData
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
        reference_genotypes=vcf_file, prs_definition=_PRS_DF, platform_variants=_PLATFORM
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
            [r1.prs, r1.prs_observed_component, r1.se],
            [r0.prs, r0.prs_observed_component, r0.se],
            rtol=0,
            atol=1e-12,
            err_msg=f"format={fmt}",
        )


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
    def test_hardcall_dispatches_to_string_path(
        self, fitted_imputation_model, integer_genotype_data
    ):
        ev = ImputationEvaluator(fitted_imputation_model, verbose=0)
        assert is_hard_called(integer_genotype_data.dosage_matrix)
        np.testing.assert_allclose(
            ev._compute_imputed_prs_batch(integer_genotype_data),
            ev._predicted_prs_via_strings(integer_genotype_data),
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
