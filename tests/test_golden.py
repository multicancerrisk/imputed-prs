"""Consolidated golden-test battery — the Python hard gate (P5.2).

This file is the single authoritative statement of the allele-orientation
invariants the deep review surfaced. Each invariant is asserted at the contract
level it ships at, so the allele break — training orients to the effect allele but
the browser/upload path once counted raw homozygosity — cannot regress silently.

The invariants are also tested, more exhaustively, in their per-problem files
(``test_user_genotypes.py``, ``test_predictor.py``, ``test_projection_predictor.py``,
``test_round_trip.py``, ``test_projection_evaluator.py``); a ``# exhaustive cases``
comment cites those. The value here is one auditable gate plus end-to-end hand
calculations and the two browser-facing cases.

Almost everything is **hermetic** (built by directly constructing models — the same
state ``load()`` reconstructs and round-trip tests already ``predict()`` on), so it
runs without cyvcf2. Only ``TestRoundTripGolden`` needs a real ``fit()`` and rides
the cyvcf2-gated conftest fixtures (``fitted_imputation_model`` /
``fitted_projection_model``); it skips cleanly on a minimal install while every
orientation invariant below still gates CI.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.core.types import (
    GenotypeData,
    ImputedVariantModel,
    ProjectionRegionModel,
    VariantIdentity,
    VariantInfo,
)
from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator
from imputed_prs.io.user_genotypes import (
    count_allele,
    load_raw_user_genotypes,
    resolve_counted_dosage,
)
from imputed_prs.models.predictor import compute_observed_prs_oriented

# =============================================================================
# Inline helpers (repo convention: these tiny builders are duplicated per test
# file rather than imported, since the suite has no cross-test-module imports).
# =============================================================================


def _collection(rows):
    """Build a RawUserGenotypeCollection from (rsid, chrom, pos, genotype) rows."""
    df = pd.DataFrame(
        {
            "rsid": [r[0] for r in rows],
            "chrom": [r[1] for r in rows],
            "pos": [r[2] for r in rows],
            "genotype": [r[3] for r in rows],
        }
    )
    return load_raw_user_genotypes(df)


def _identity(accepted_ids, chromosome, position, counted="A", other="G"):
    """Build a VariantIdentity (resolution is allele-agnostic; counting is later)."""
    return VariantIdentity(
        feature_id=f"{chromosome}:{position}:{other}:{counted}",
        variant_id=accepted_ids[0],
        accepted_ids=tuple(accepted_ids),
        chromosome=chromosome,
        position=position,
        counted_allele=counted,
        other_allele=other,
    )


def _imputed_model(
    variant_id="rs_target",
    chromosome="1",
    position=5000,
    effect_allele="A",
    other_allele="G",
    beta=0.05,
    allele_frequency=0.3,
    imputation_r2=0.8,
    residual_variance=0.1,
    intercept=0.6,
    predictor_variant_ids=None,
    coefficients=None,
    is_intercept_only=False,
    predictor_chromosomes=None,
    predictor_positions=None,
    predictor_counted_alleles=None,
    predictor_other_alleles=None,
    predictor_allele_frequencies=None,
):
    """Build an ImputedVariantModel carrying P1.3 predictor allele metadata.

    Metadata defaults are length-aligned to ``predictor_variant_ids`` (counted =
    ALT, other = REF) so the oriented scorer can resolve every predictor.
    """
    if predictor_variant_ids is None:
        predictor_variant_ids = ["rs_p0", "rs_p1"]
    n = len(predictor_variant_ids)
    if coefficients is None:
        coefficients = np.full(n, 0.2)
    if predictor_chromosomes is None:
        predictor_chromosomes = [chromosome] * n
    if predictor_positions is None:
        predictor_positions = [1000 + 100 * i for i in range(n)]
    if predictor_counted_alleles is None:
        predictor_counted_alleles = ["A"] * n
    if predictor_other_alleles is None:
        predictor_other_alleles = ["G"] * n
    if predictor_allele_frequencies is None:
        predictor_allele_frequencies = np.full(n, allele_frequency)
    return ImputedVariantModel(
        variant_id=variant_id,
        chromosome=chromosome,
        position=position,
        effect_allele=effect_allele,
        other_allele=other_allele,
        beta=beta,
        allele_frequency=allele_frequency,
        imputation_r2=imputation_r2,
        residual_variance=residual_variance,
        intercept=intercept,
        predictor_variant_ids=predictor_variant_ids,
        coefficients=coefficients,
        is_intercept_only=is_intercept_only,
        predictor_chromosomes=predictor_chromosomes,
        predictor_positions=predictor_positions,
        predictor_counted_alleles=predictor_counted_alleles,
        predictor_other_alleles=predictor_other_alleles,
        predictor_allele_frequencies=predictor_allele_frequencies,
    )


def _imputation_model(observed, imputed):
    """A predict()-ready LinearImputationPRS built by setting the attributes that
    predict() reads (the same state ``load()`` reconstructs). No fit / cyvcf2."""
    m = LinearImputationPRS()
    m._observed_variants = observed
    m._imputed_models = imputed
    m._calibration_params = None
    m._genome_build = "GRCh37"
    m._platform_name = None
    m._is_fitted = True
    return m


def _upload(rsids, positions, genotypes):
    """A user upload as genotype strings (the browser/file path -> allele-aware)."""
    return pd.DataFrame(
        {
            "rsid": list(rsids),
            "chromosome": ["1"] * len(rsids),
            "position": list(positions),
            "genotype": list(genotypes),
        }
    )


# Canonical observed PRS terms (betas/alleles from the conftest _PRS_DF rs1-rs3).
# rs1/rs2 have effect == ALT; rs3 has effect == REF-of-panel — the orientation the
# original bug mis-scored.
def _canonical_observed():
    return [
        VariantInfo(variant_id="rs1", chromosome="1", position=100000,
                    effect_allele="G", other_allele="A", beta=0.1),
        VariantInfo(variant_id="rs2", chromosome="1", position=100500,
                    effect_allele="T", other_allele="C", beta=-0.05),
        VariantInfo(variant_id="rs3", chromosome="1", position=101000,
                    effect_allele="A", other_allele="G", beta=0.2),
    ]


# =============================================================================
# Items 1-5: the count_allele primitive
# =============================================================================


class TestAlleleCounting:
    """The oriented allele-counting golden table.

    # exhaustive cases: test_user_genotypes.py TestCountAllele (:437-476)
    """

    def test_homozygote_dosage_effect_alt_and_ref(self):
        """Item 1: counting the named allele is orientation-correct both ways."""
        k = dict(allow_ambiguous=False, allow_strand_flip=False)
        # effect == ALT (count A): AA/AG/GG -> 2/1/0
        assert count_allele("AA", "A", "G", **k) == 2.0
        assert count_allele("AG", "A", "G", **k) == 1.0
        assert count_allele("GG", "A", "G", **k) == 0.0
        # effect == REF (count G): symmetric -> AA/AG/GG -> 0/1/2
        assert count_allele("AA", "G", "A", **k) == 0.0
        assert count_allele("AG", "G", "A", **k) == 1.0
        assert count_allele("GG", "G", "A", **k) == 2.0

    def test_heterozygote_order_invariant(self):
        """Item 2: AG and GA count identically."""
        k = dict(allow_ambiguous=False, allow_strand_flip=False)
        assert count_allele("AG", "A", "G", **k) == count_allele("GA", "A", "G", **k) == 1.0

    def test_partial_overlap_is_unresolved(self):
        """Item 3: a genotype not fully inside the pair is None, never dosage 1."""
        assert count_allele("AC", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None

    def test_palindromic_policy(self):
        """Item 4: A/T (palindromic) is None unless ambiguity is explicitly allowed."""
        assert count_allele("AA", "A", "T", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("AA", "A", "T", allow_ambiguous=True, allow_strand_flip=False) == 2.0

    def test_strand_complement_policy(self):
        """Item 5: a reverse-strand genotype resolves only with strand flip on."""
        assert count_allele("TT", "A", "G", allow_ambiguous=False, allow_strand_flip=False) is None
        assert count_allele("TT", "A", "G", allow_ambiguous=False, allow_strand_flip=True) == 2.0

    def test_indel_allele_counting(self):
        """Item 6: arbitrary-length indel alleles count as whole tokens (structured
        allele/dosage browser scorer), both orientations, matching the numeric path."""
        k = dict(allow_ambiguous=True, allow_strand_flip=True)
        # counted = insertion ALT "AT": A/A, A/AT, AT/AT -> 0, 1, 2.
        assert count_allele("A/A", "AT", "A", **k) == 0.0
        assert count_allele("A/AT", "AT", "A", **k) == 1.0
        assert count_allele("AT/AT", "AT", "A", **k) == 2.0
        # deletion (counted = shorter allele) and a foreign allele -> unresolved.
        assert count_allele("AT/A", "A", "AT", **k) == 1.0
        assert count_allele("A/C", "AT", "A", **k) is None


# =============================================================================
# Items 1,3,11 in the observed-scoring contract
# =============================================================================


class TestObservedScoring:
    """Orientation / unresolved handling through the canonical observed scorer.

    # exhaustive cases: test_predictor.py TestComputeObservedPrsOriented (:118-263)
    """

    def test_effect_ref_homozygote_scores_zero_not_two(self):
        """Item 1 (the bug): effect == REF, genotype homozygous-other -> 0 copies."""
        observed = [VariantInfo(variant_id="rs_ref", chromosome="1", position=200,
                                effect_allele="G", other_allele="A", beta=0.5)]
        coll = _collection([("rs_ref", "1", 200, "AA")])
        score = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        np.testing.assert_allclose(score.prs, 0.0, rtol=0, atol=1e-12)
        assert score.n_scored_direct == 1
        assert score.unresolved_ids == ()

    def test_partial_overlap_falls_through_to_unresolved(self):
        """Item 3: a partial-overlap call with no fallback is surfaced, not scored."""
        observed = [VariantInfo(variant_id="rs1", chromosome="1", position=100,
                                effect_allele="A", other_allele="G", beta=0.9)]
        coll = _collection([("rs1", "1", 100, "AC")])
        score = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        np.testing.assert_allclose(score.prs, 0.0, rtol=0, atol=1e-12)
        assert score.n_scored_direct == 0
        assert score.unresolved_ids == ("rs1",)

    def test_duplicate_conflict_is_unresolved_not_first_match(self):
        """Item 11 (scorer level): conflicting duplicate -> unresolved, never a guess."""
        observed = [VariantInfo(variant_id="rs5", chromosome="1", position=500,
                                effect_allele="A", other_allele="G", beta=0.9)]
        coll = _collection([("rs5", "1", 500, "AA"), ("rs5", "1", 500, "GG")])
        score = compute_observed_prs_oriented(coll, observed, allow_ambiguous=True)
        assert score.n_scored_direct == 0
        assert score.unresolved_ids == ("rs5",)
        np.testing.assert_allclose(score.prs, 0.0, rtol=0, atol=1e-12)


# =============================================================================
# Items 6-8: raw genotype -> public predict() == hand calculation
# =============================================================================


class TestEndToEndImputation:
    """Hand-calculated scores through the public ``predict`` (the upload path).

    A DataFrame upload (not a dict) routes both components through the
    allele-aware oriented scorers (``predictor.py:541,570``).
    """

    def test_raw_genotype_to_predict_equals_handcalc(self):
        """Item 6: observed PRS from a raw upload matches a hand calculation."""
        model = _imputation_model(_canonical_observed(), imputed=[])
        upload = _upload(["rs1", "rs2", "rs3"], [100000, 100500, 101000],
                         ["AG", "CC", "AA"])
        r = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        # rs1 G in "AG" = 1 -> 0.1 ; rs2 T in "CC" = 0 -> 0 ; rs3 A in "AA" = 2 -> 0.4
        np.testing.assert_allclose(r.prs, 0.5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.prs_observed_component, 0.5, rtol=0, atol=1e-12)
        assert r.prs_imputed_component == 0.0
        assert r.n_observed_scored_direct == 3
        assert r.unresolved_observed_ids == ()
        assert r.n_variants_used == 3

    def test_observed_only_score_equals_handcalc(self):
        """Item 7: a distinct genotype vector, observed-only, matches by hand."""
        model = _imputation_model(_canonical_observed(), imputed=[])
        upload = _upload(["rs1", "rs2", "rs3"], [100000, 100500, 101000],
                         ["GG", "CT", "AG"])
        r = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        # rs1 "GG"=2 -> 0.2 ; rs2 "CT"=1 -> -0.05 ; rs3 "AG"=1 -> 0.2  => 0.35
        np.testing.assert_allclose(r.prs, 0.35, rtol=0, atol=1e-12)
        assert r.prs_imputed_component == 0.0
        assert r.n_observed_scored_direct == 3

    def test_one_missing_predictor_uses_remaining_plus_mean_not_intercept(self):
        """Item 8: one missing predictor -> remaining predictor + mean substitution
        (P3.3), never an all-or-nothing collapse to the intercept."""
        imp = _imputed_model(
            variant_id="rs_target", effect_allele="A", other_allele="G", beta=0.1,
            intercept=0.2, residual_variance=0.05,
            predictor_variant_ids=["rs_p0", "rs_p1"],
            coefficients=np.array([0.5, -0.3]),
            predictor_chromosomes=["1", "1"],
            predictor_positions=[1000, 1100],
            predictor_counted_alleles=["A", "C"],
            predictor_other_alleles=["G", "T"],
            predictor_allele_frequencies=np.array([0.3, 0.4]),
        )
        model = _imputation_model(observed=[], imputed=[imp])
        # rs_p0 = "AA" (2 copies of ALT A); rs_p1 absent -> mean 2*0.4 = 0.8.
        upload = _upload(["rs_p0"], [1000], ["AA"])
        r = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        # predicted target dosage = clip(2*0.5 + 0.8*(-0.3) + 0.2, 0, 2) = 0.96
        # contribution = 0.96 * beta(0.1) = 0.096
        np.testing.assert_allclose(r.prs, 0.096, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.prs_imputed_component, 0.096, rtol=0, atol=1e-12)
        assert r.n_variants_imputed == 1
        # It did NOT collapse to intercept-only (which would be 0.2 * 0.1 = 0.02).
        assert abs(r.prs - 0.02) > 1e-6


# =============================================================================
# Item 9: projection effect=REF / multiallelic true-PRS (P2.3)
# =============================================================================


def _intercept_only_region(prs_id, position, effect, other, beta):
    """A predictor-free region carrying one missing PRS variant (true-PRS only)."""
    return ProjectionRegionModel(
        region_id=f"chr1:{position}-{position}",
        chromosome="1",
        start=position,
        end=position,
        prs_variant_ids=[prs_id],
        betas=np.array([beta]),
        predictor_variant_ids=[],
        coefficients=np.array([]),
        intercept=0.0,
        cv_mse=0.0,
        cv_r2=0.0,
        is_intercept_only=True,
        mean_prs_contribution=0.0,
        predictor_allele_frequencies=np.array([]),
        prs_positions=[position],
        prs_effect_alleles=[effect],
        prs_other_alleles=[other],
    )


def _projection_true_prs(genotype_data, region_models):
    model = LinearProjectionPRS()
    model._is_fitted = True
    model._observed_variants = []
    model._region_models = region_models
    return ProjectionEvaluator(model, verbose=0)._compute_true_prs(genotype_data)


class TestProjectionTruePrs:
    """The region-path true PRS is effect-allele-oriented per the stored alleles.

    # exhaustive cases: test_projection_evaluator.py (:201-314)
    """

    def test_effect_ref_is_flipped(self):
        """Item 9a: effect == REF scores (2 - alt_dosage) * beta, not raw dosage."""
        gd = GenotypeData(
            dosage_matrix=np.array([[0.0], [1.0], [2.0]]),
            variant_info=pd.DataFrame({
                "variant_id": ["1:300000:A:G"], "chromosome": ["1"],
                "position": [300000], "ref_allele": ["A"], "alt_allele": ["G"],
            }),
            sample_ids=["S0", "S1", "S2"],
        )
        region = _intercept_only_region("rs_ref", 300000, effect="A", other="G", beta=0.5)
        true_prs = _projection_true_prs(gd, [region])
        # effect A == REF -> oriented (2 - alt) = [2, 1, 0]; * 0.5 = [1.0, 0.5, 0.0].
        np.testing.assert_allclose(true_prs, np.array([1.0, 0.5, 0.0]), rtol=0, atol=1e-12)
        # The raw (un-oriented) reading would have been [0.0, 0.5, 1.0].
        assert not np.allclose(true_prs, np.array([0.0, 0.5, 1.0]))

    def test_multiallelic_selects_correct_alt(self):
        """Item 9b: at a multiallelic locus the effect allele picks the matching
        ALT row, not the first reference row at that position."""
        gd = GenotypeData(
            dosage_matrix=np.array([[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]]),
            variant_info=pd.DataFrame({
                "variant_id": ["1:200000:A:G", "1:200000:A:T"],
                "chromosome": ["1", "1"], "position": [200000, 200000],
                "ref_allele": ["A", "A"], "alt_allele": ["G", "T"],
            }),
            sample_ids=["S0", "S1", "S2"],
        )
        # Missing PRS variant targets the SECOND ALT (T) with beta=0.5.
        region = _intercept_only_region("rs_multi", 200000, effect="T", other="A", beta=0.5)
        true_prs = _projection_true_prs(gd, [region])
        # Effect T -> A>T row (col 1) dosage [2, 0, 1] * 0.5 = [1.0, 0.0, 0.5].
        np.testing.assert_allclose(true_prs, np.array([1.0, 0.0, 0.5]), rtol=0, atol=1e-12)


# =============================================================================
# Item 10: export -> load -> predict round trip (imputation & projection)
# =============================================================================


def _round_trip_upload():
    return _upload(["rs1", "rs2", "rs3"], [100000, 100500, 101000], ["AG", "CC", "AA"])


class TestRoundTripGolden:
    """A reloaded model scores identically and preserves schema/alleles exactly.

    Rides the cyvcf2-gated conftest fixtures, so this class skips on a minimal
    install while every hermetic invariant above still gates CI.
    """

    @pytest.mark.parametrize("fmt,dep", [("json", None), ("hdf5", "h5py")])
    def test_imputation_round_trip(self, fitted_imputation_model, tmp_path, fmt, dep):
        if dep is not None:
            pytest.importorskip(dep)
        model = fitted_imputation_model
        loaded = LinearImputationPRS.load(
            model.export(tmp_path, model_name="g", formats=[fmt])[fmt]
        )
        upload = _round_trip_upload()
        r0 = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        r1 = loaded.predict(upload, apply_calibration=False, genome_build="GRCh37")

        # Floats: allclose. The round trip must not perturb the score.
        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component, r1.prs_imputed_component],
            [r0.prs, r0.prs_observed_component, r0.prs_imputed_component],
            rtol=0, atol=1e-12, err_msg=fmt,
        )
        # Counts: exact.
        assert r1.n_variants_used == r0.n_variants_used, fmt
        assert r1.n_observed_scored_direct == r0.n_observed_scored_direct, fmt
        assert r1.unresolved_observed_ids == r0.unresolved_observed_ids, fmt
        # Schema / alleles: exact on the loaded model state (the clause the
        # prediction-level round trip in test_round_trip.py does not check).
        assert _alleles(loaded.observed_variants) == _alleles(model.observed_variants), fmt

    def test_projection_round_trip(self, fitted_projection_model, tmp_path):
        model = fitted_projection_model
        loaded = LinearProjectionPRS.load(
            model.export(tmp_path, model_name="g", formats=["json"])["json"]
        )
        upload = _round_trip_upload()
        r0 = model.predict(upload, apply_calibration=False, genome_build="GRCh37")
        r1 = loaded.predict(upload, apply_calibration=False, genome_build="GRCh37")

        np.testing.assert_allclose(
            [r1.prs, r1.prs_observed_component, r1.prs_imputed_component],
            [r0.prs, r0.prs_observed_component, r0.prs_imputed_component],
            rtol=0, atol=1e-12,
        )
        assert r1.n_variants_used == r0.n_variants_used
        assert r1.unresolved_observed_ids == r0.unresolved_observed_ids
        # Observed alleles and per-region PRS-variant effect alleles round-trip exactly.
        assert _alleles(loaded.observed_variants) == _alleles(model.observed_variants)
        assert _region_prs_effect_alleles(loaded.region_models) == \
            _region_prs_effect_alleles(model.region_models)


def _alleles(variants):
    return [(v.variant_id, v.effect_allele, v.other_allele) for v in variants]


def _region_prs_effect_alleles(region_models):
    return [
        (rm.region_id, tuple(rm.prs_variant_ids), tuple(rm.prs_effect_alleles))
        for rm in region_models
    ]


# =============================================================================
# Items 11-12: browser-facing multi-key resolution
# =============================================================================


class TestBrowserResolution:
    """Real-upload resolution: conflicts are surfaced, multiallelic loci are
    disambiguated by chr:pos:allele-pair — never by rsID alone.

    # exhaustive cases: test_user_genotypes.py TestVariantIdentityResolver (:667-755)
    """

    def test_browser_case1_duplicate_conflict_reported(self):
        """Item 11: a duplicate rsID with a conflicting genotype or locus resolves
        to a duplicate-conflict (None), never an arbitrary first match."""
        # Conflicting genotype at the same locus.
        coll_g = _collection([("rs1", "1", 100, "AG"), ("rs1", "1", 100, "GG")])
        res_g = coll_g.resolve(_identity(["rs1"], "1", 100))
        assert res_g.status == "duplicate_conflict"
        assert res_g.genotype is None
        # Conflicting locus for the same rsID.
        coll_l = _collection([("rs1", "1", 100, "AG"), ("rs1", "1", 200, "AG")])
        assert coll_l.resolve(_identity(["rs1"], "1", 100)).status == "duplicate_conflict"
        # And the user-visible scorer surfaces it as unresolved, not a guessed score.
        observed = [VariantInfo(variant_id="rs1", chromosome="1", position=100,
                                effect_allele="A", other_allele="G", beta=0.9)]
        score = compute_observed_prs_oriented(coll_g, observed, allow_ambiguous=True)
        assert score.n_scored_direct == 0
        assert score.unresolved_ids == ("rs1",)

    def test_browser_case2_multiallelic_allele_pair_discriminates(self):
        """Item 12: at one multiallelic locus, the allele pair — not the shared
        rsID — decides which genotypes are countable."""
        coll = _collection([("rs_multi", "1", 100, "AG")])
        # Term declaring the A/G pair counts the effect allele A once.
        d_ag = resolve_counted_dosage(
            coll, variant_id="rs_multi", chromosome="1", position=100,
            counted_allele="A", other_allele="G",
            allow_ambiguous=True, allow_strand_flip=False,
        )
        assert d_ag == 1.0
        # A term declaring a different multiallelic alt (A/C) cannot count "AG" —
        # the foreign G makes it unresolved, not mis-scored as 1.
        d_ac = resolve_counted_dosage(
            coll, variant_id="rs_multi", chromosome="1", position=100,
            counted_allele="A", other_allele="C",
            allow_ambiguous=True, allow_strand_flip=False,
        )
        assert d_ac is None

    def test_browser_case2_same_rsid_two_loci_needs_chrpos(self):
        """Item 12: the same rsID at two physical loci is ambiguous by rsID alone,
        so a by-rsID lookup is refused (None); chr:pos breaks the tie."""
        coll = _collection([("rs_multi", "1", 100, "AG"), ("rs_multi", "1", 200, "AC")])
        # rsID matches both loci -> ambiguous -> unresolved (never an arbitrary pick).
        ambiguous = resolve_counted_dosage(
            coll, variant_id="rs_multi", chromosome="1", position=100,
            counted_allele="A", other_allele="G",
            allow_ambiguous=True, allow_strand_flip=False,
        )
        assert ambiguous is None
        # A distinct user id at 1:200 resolves cleanly by chr:pos to the A/C record.
        coll2 = _collection([("rs_multi", "1", 100, "AG"), ("rs_other", "1", 200, "AC")])
        by_chrpos = resolve_counted_dosage(
            coll2, variant_id="rs_model_200", chromosome="1", position=200,
            counted_allele="A", other_allele="C",
            allow_ambiguous=True, allow_strand_flip=False,
        )
        assert by_chrpos == 1.0
