"""Parity tests for the Phase-4 vectorized panel predictor.

The batch orientation must be *byte-identical* to the per-predictor oracle
(``match_oriented_dosage`` with NaN → ``2*AF`` fill); the CSR mat-mul score is
validated at ``atol=1e-9`` (bit identity is impossible — ``scipy`` canonicalizes
CSR indices and the SpMM reorders float additions). Oracles are reimplemented
inline here (against the stable ``match_oriented_dosage``) so these tests do not
couple to the evolving ``evaluation/_scoring.py`` signatures.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.harmonizer import (
    build_reference_allele_index,
    match_oriented_dosage,
)
from imputed_prs.core.types import (
    GenotypeData,
    ImputedVariantModel,
    ProjectionRegionModel,
    VariantInfo,
)
from imputed_prs.models.vectorized_predictor import (
    VectorizedPredictor,
    accumulate_true_prs,
    build_chip_axis,
    build_coef_csr,
    build_projection_weff,
    oriented_chip_matrix,
    panel_impute_prs,
    panel_project_prs,
)


# --------------------------------------------------------------------------- #
# Fixtures / inline oracles
# --------------------------------------------------------------------------- #
def _make_reference(records):
    """records: list of (chrom, pos, ref, alt, dosages). Reference dosages float32."""
    variant_info = pd.DataFrame(
        [
            {
                "variant_id": f"{c}:{p}:{ref}:{alt}",
                "chromosome": c,
                "position": p,
                "ref_allele": ref,
                "alt_allele": alt,
            }
            for c, p, ref, alt, _ in records
        ]
    )
    dosage_matrix = np.array([d for *_, d in records], dtype=np.float32).T
    return variant_info, dosage_matrix


def _genotype_data(records):
    vi, dm = _make_reference(records)
    return GenotypeData(
        dosage_matrix=dm,
        variant_info=vi,
        sample_ids=[f"s{i}" for i in range(dm.shape[0])],
        genome_build="GRCh37",
        source_file=None,
    )


def _imodel(
    variant_id,
    chrom,
    pos,
    beta,
    intercept,
    predictors,
    coeffs,
    *,
    effect="A",
    other="G",
    is_intercept_only=False,
):
    """predictors: list of (pid, chrom, pos, counted, other, af)."""
    return ImputedVariantModel(
        variant_id=variant_id,
        chromosome=chrom,
        position=pos,
        effect_allele=effect,
        other_allele=other,
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


def _region(region_id, chrom, beta_list, predictors, coeffs, intercept, *, is_intercept_only=False):
    return ProjectionRegionModel(
        region_id=region_id,
        chromosome=chrom,
        start=0,
        end=10_000,
        prs_variant_ids=[f"{region_id}:prs{i}" for i in range(len(beta_list))],
        betas=np.asarray(beta_list, dtype=np.float64),
        predictor_variant_ids=[p[0] for p in predictors],
        coefficients=np.asarray(coeffs, dtype=np.float64),
        intercept=intercept,
        cv_mse=0.1,
        cv_r2=0.5,
        is_intercept_only=is_intercept_only,
        mean_prs_contribution=0.0,
        predictor_allele_frequencies=np.asarray(
            [p[5] for p in predictors], dtype=np.float64
        ),
        predictor_chromosomes=[p[1] for p in predictors],
        predictor_positions=[p[2] for p in predictors],
        predictor_counted_alleles=[p[3] for p in predictors],
        predictor_other_alleles=[p[4] for p in predictors],
    )


def _chip_meta(models):
    """Ordered unique (chrom, pos, counted, other, af) — mirrors build_chip_axis dedup."""
    seen = set()
    order = []
    for m in models:
        if m.is_intercept_only or not m.predictor_variant_ids:
            continue
        for i, pid in enumerate(m.predictor_variant_ids):
            if pid in seen:
                continue
            seen.add(pid)
            order.append(
                (
                    m.predictor_chromosomes[i],
                    m.predictor_positions[i],
                    m.predictor_counted_alleles[i],
                    m.predictor_other_alleles[i],
                    float(m.predictor_allele_frequencies[i]),
                )
            )
    return order


def _oracle_oriented(vi, dm, meta):
    """Column-stacked oriented_predictor_matrix oracle (NaN/unresolved → 2*AF)."""
    ref_index = build_reference_allele_index(vi)
    n = dm.shape[0]
    M = np.empty((n, len(meta)), dtype=np.float64)
    for i, (chrom, pos, counted, other, af) in enumerate(meta):
        mean = 2.0 * float(af)
        match = match_oriented_dosage(chrom, pos, counted, other, vi, dm, ref_index)
        if match is None:
            M[:, i] = mean
            continue
        col = np.asarray(match[1], dtype=np.float64).copy()
        col[np.isnan(col)] = mean
        M[:, i] = col
    return M


def _oracle_impute_prs(gd, models):
    """Per-model oracle: sum_j clip(z_j·w_j + b_j, 0, 2) * beta_j (imputed component)."""
    ref_index = build_reference_allele_index(gd.variant_info)
    n = gd.n_samples
    predicted = np.zeros(n)
    for m in models:
        if m.is_intercept_only or not m.predictor_variant_ids:
            raw = np.full(n, float(m.intercept))
        else:
            meta = [
                (
                    m.predictor_chromosomes[i],
                    m.predictor_positions[i],
                    m.predictor_counted_alleles[i],
                    m.predictor_other_alleles[i],
                    float(m.predictor_allele_frequencies[i]),
                )
                for i in range(len(m.predictor_variant_ids))
            ]
            z = _oracle_oriented(gd.variant_info, gd.dosage_matrix, meta)
            raw = z @ np.asarray(m.coefficients, dtype=np.float64) + float(m.intercept)
        predicted += np.clip(raw, 0.0, 2.0) * m.beta
    return predicted


# A messy reference: direct, flip (effect==REF), multiallelic, NaN sample.
_RECORDS = [
    ("1", 100, "A", "G", [0.0, 1.0, 2.0, np.nan]),  # p1 direct (counted=G=ALT)
    ("1", 100, "A", "T", [2.0, 1.0, 0.0, 1.0]),  # p2 multiallelic at 1:100, counted=T
    ("1", 200, "C", "T", [0.0, 2.0, 1.0, 2.0]),  # p3 flip (counted=C=REF -> 2-dosage)
    ("1", 300, "A", "G", [1.0, 1.0, 1.0, 0.0]),  # observed obs1
    ("2", 50, "G", "C", [1.0, 0.0, 2.0, 1.0]),  # observed obs2
]


def _models():
    p1 = ("rs_p1", "1", 100, "G", "A", 0.2)
    p2 = ("rs_p2", "1", 100, "T", "A", 0.3)
    p3 = ("rs_p3", "1", 200, "C", "T", 0.4)
    p4 = ("rs_p4", "9", 999, "X", "Y", 0.25)  # unresolved locus -> mean fill
    t1 = _imodel("t1", "1", 400, beta=1.5, intercept=0.1, predictors=[p1, p3], coeffs=[0.5, -0.3])
    t2 = _imodel(
        "t2", "1", 500, beta=0.7, intercept=-0.2, predictors=[p2, p1, p4], coeffs=[0.2, 0.4, 0.1]
    )
    t3 = _imodel("t3", "2", 60, beta=2.0, intercept=0.05, predictors=[], coeffs=[], is_intercept_only=True)
    return [t1, t2, t3]


def _observed():
    return [
        VariantInfo("obs1", "1", 300, "G", "A", 0.9),
        VariantInfo("obs2", "2", 50, "C", "G", -0.5),
    ]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestOrientedChipMatrix:
    def test_byte_identical_to_oracle(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        models = _models()
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(models, resolver)
        got = oriented_chip_matrix(gd.dosage_matrix, axis)

        expected = _oracle_oriented(gd.variant_info, gd.dosage_matrix, _chip_meta(models))
        np.testing.assert_array_equal(got, expected)

    def test_chip_index_first_occurrence_order(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(_models(), resolver)
        assert axis.chip_index == {"rs_p1": 0, "rs_p3": 1, "rs_p2": 2, "rs_p4": 3}
        assert not axis.resolved[axis.chip_index["rs_p4"]]  # unresolved locus

    def test_empty_axis(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        resolver = ReferenceAlleleResolver(gd.variant_info)
        # all intercept-only -> no chip columns
        m = _imodel("t", "1", 1, 1.0, 0.0, [], [], is_intercept_only=True)
        axis = build_chip_axis([m], resolver)
        Z = oriented_chip_matrix(gd.dosage_matrix, axis)
        assert Z.shape == (gd.n_samples, 0)


class TestBuildCoefCsr:
    def test_structure(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        models = _models()
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(models, resolver)
        W, intercepts, betas = build_coef_csr(models, axis.chip_index)

        assert W.shape == (3, 4)
        # nnz == sum of non-intercept coefficient lengths (2 + 3 + 0)
        assert W.nnz == 5
        np.testing.assert_allclose(intercepts, [0.1, -0.2, 0.05])
        np.testing.assert_allclose(betas, [1.5, 0.7, 2.0])
        # intercept-only row t3 is all zeros
        assert W[2].nnz == 0
        # t1 row: coef 0.5 at col(rs_p1)=0, -0.3 at col(rs_p3)=1
        row0 = W[0].toarray().ravel()
        assert row0[0] == 0.5 and row0[1] == -0.3

    def test_shared_predictor_column(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        models = _models()
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(models, resolver)
        W, _, _ = build_coef_csr(models, axis.chip_index)
        # rs_p1 (col 0) appears in t1 (0.5) and t2 (0.4)
        col0 = W[:, 0].toarray().ravel()
        assert col0[0] == 0.5 and col0[1] == 0.4


class TestPanelImputePrs:
    def test_equals_oracle(self):
        gd = _genotype_data(_RECORDS)
        models = _models()
        observed = _observed()
        pred = VectorizedPredictor(observed, imputed_models=models)
        got = pred.predict_panel(gd)

        # oracle: observed (f64) + imputed
        ref_index = build_reference_allele_index(gd.variant_info)
        obs = np.zeros(gd.n_samples)
        for var in observed:
            match = match_oriented_dosage(
                var.chromosome, var.position, var.effect_allele, var.other_allele,
                gd.variant_info, gd.dosage_matrix, ref_index,
            )
            if match is None:
                continue
            d = np.asarray(match[1], dtype=np.float64)
            valid = ~np.isnan(d)
            obs[valid] += d[valid] * var.beta
        expected = obs + _oracle_impute_prs(gd, models)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)

    def test_block_size_invariance(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        models = _models()
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(models, resolver)
        Z = oriented_chip_matrix(gd.dosage_matrix, axis)
        W, intercepts, betas = build_coef_csr(models, axis.chip_index)
        base = panel_impute_prs(Z, W, intercepts, betas, block_size=8192)
        for bs in (1, 2, 7, 100):
            got = panel_impute_prs(Z, W, intercepts, betas, block_size=bs)
            np.testing.assert_allclose(got, base, rtol=0.0, atol=1e-12)


class TestProjectionCollapse:
    def _proj_models(self):
        p1 = ("rs_p1", "1", 100, "G", "A", 0.2)
        p2 = ("rs_p2", "1", 100, "T", "A", 0.3)
        p3 = ("rs_p3", "1", 200, "C", "T", 0.4)
        r1 = _region("chr1:0-1000", "1", [0.5, 0.3], [p1, p3], [0.6, -0.2], 0.15)
        r2 = _region("chr1:1000-2000", "1", [0.1], [p2, p1], [0.25, 0.35], -0.05)
        r3 = _region("chr2:0-1000", "2", [0.2], [], [], 0.4, is_intercept_only=True)
        return [r1, r2, r3]

    def test_weff_collapse_equals_oracle(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        models = self._proj_models()
        resolver = ReferenceAlleleResolver(gd.variant_info)
        axis = build_chip_axis(models, resolver)
        Z = oriented_chip_matrix(gd.dosage_matrix, axis)
        w_eff, const = build_projection_weff(models, axis.chip_index)
        got = panel_project_prs(Z, w_eff, const)

        # oracle: per-region z@coef + intercept (no clip, no beta)
        expected = np.zeros(gd.n_samples)
        for m in models:
            expected += float(m.intercept)
            if m.is_intercept_only or not m.predictor_variant_ids:
                continue
            meta = [
                (
                    m.predictor_chromosomes[i],
                    m.predictor_positions[i],
                    m.predictor_counted_alleles[i],
                    m.predictor_other_alleles[i],
                    float(m.predictor_allele_frequencies[i]),
                )
                for i in range(len(m.predictor_variant_ids))
            ]
            z = _oracle_oriented(gd.variant_info, gd.dosage_matrix, meta)
            expected += z @ np.asarray(m.coefficients, dtype=np.float64)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)


class TestAccumulateTruePrs:
    def test_equals_per_variant_oracle(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        # Mix: direct (obs at 1:300, 2:50), flip (1:200 counted=C=REF), multiallelic
        # + NaN sample (1:100 sample index 3 is NaN), and an unresolved locus.
        placed = [
            ("1", 300, "G", "A", 0.9),
            ("2", 50, "C", "G", -0.5),
            ("1", 200, "C", "T", 1.2),  # flip
            ("1", 100, "G", "A", -0.7),  # has a NaN sample
            ("9", 999, "X", "Y", 5.0),  # unresolved -> skipped
        ]

        resolver = ReferenceAlleleResolver(gd.variant_info)
        got = accumulate_true_prs(gd.dosage_matrix, resolver, placed)

        ref_index = build_reference_allele_index(gd.variant_info)
        expected = np.zeros(gd.n_samples)
        for chrom, pos, effect, other, beta in placed:
            match = match_oriented_dosage(
                chrom, pos, effect, other, gd.variant_info, gd.dosage_matrix, ref_index
            )
            if match is None:
                continue
            dosages = match[1]
            valid = ~np.isnan(dosages)
            expected[valid] += dosages[valid] * beta
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)

    def test_block_size_invariance(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        placed = [
            ("1", 300, "G", "A", 0.9),
            ("2", 50, "C", "G", -0.5),
            ("1", 200, "C", "T", 1.2),
            ("1", 100, "G", "A", -0.7),
        ]
        resolver = ReferenceAlleleResolver(gd.variant_info)
        base = accumulate_true_prs(gd.dosage_matrix, resolver, placed, block_size=8192)
        for bs in (1, 2, 3):
            got = accumulate_true_prs(gd.dosage_matrix, resolver, placed, block_size=bs)
            np.testing.assert_allclose(got, base, rtol=0.0, atol=1e-12)


class TestConsistencyGuard:
    def test_inconsistent_shared_predictor_raises(self):
        from imputed_prs.core.harmonizer import ReferenceAlleleResolver

        gd = _genotype_data(_RECORDS)
        resolver = ReferenceAlleleResolver(gd.variant_info)
        # same pid "rs_x" but different counted allele across two models
        m1 = _imodel("t1", "1", 400, 1.0, 0.0, [("rs_x", "1", 100, "G", "A", 0.2)], [0.5])
        m2 = _imodel("t2", "1", 500, 1.0, 0.0, [("rs_x", "1", 100, "T", "A", 0.2)], [0.5])
        with pytest.raises(ValueError, match="Inconsistent predictor metadata"):
            build_chip_axis([m1, m2], resolver)
