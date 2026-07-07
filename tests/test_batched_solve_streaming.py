"""Phase 8 — forced batched-solve path: end-to-end parity + cross-worker bit-identity.

The batched FISTA/ridge solve (``gram_solve.solve_blocks_batched``) engages in the streaming
fit only for large panels (``n_samples >= _BATCH_MIN_SAMPLES``) or when ``IMPUTED_PRS_SOLVE``
forces it. These tests force it on small panels (so they run in seconds) and check:

  * imputation + projection models / calibration match the per-target sklearn coordinate-
    descent oracle within statistical-parity tolerance (batched FISTA is a different
    optimizer, ~1e-3 coef / ~5e-3 CV metrics — the sanctioned band);
  * the batched path is deterministic under the Phase-7 chromosome fan-out — ``n_workers=1``
    vs ``2`` produce a **bit-identical** artifact (the ``IMPUTED_PRS_SOLVE`` env is inherited
    by the loky workers; BLAS is pinned so GEMM reduction order is the only-controlled var).
"""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.core.types import GenotypeData
from tests.test_parallel_streaming import _assert_imputation_identical
from tests.test_streaming_backend import (
    SEED,
    WINDOW,
    _synthetic_prs,
    _write_synthetic_vcf,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _fit_imputation_vcf(path, prs_df, platform, solve_mode, monkeypatch):
    monkeypatch.setenv("IMPUTED_PRS_SOLVE", solve_mode)
    m = LinearImputationPRS(
        window_size=WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=SEED, backend="streaming", device="cpu", verbose=0,
    )
    m.fit(reference_genotypes=path, prs_definition=prs_df,
          platform_variants=platform, genome_build="GRCh38")
    return m


def _fit_projection_vcf(path, prs_df, platform, solve_mode, monkeypatch):
    monkeypatch.setenv("IMPUTED_PRS_SOLVE", solve_mode)
    m = LinearProjectionPRS(
        window_size=WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=SEED, backend="streaming", device="cpu", verbose=0,
    )
    m.fit(reference_genotypes=path, prs_definition=prs_df,
          platform_variants=platform, genome_build="GRCh38")
    return m


def test_batched_imputation_matches_per_target(tmp_path, monkeypatch):
    """Forced batched FISTA == per-target sklearn CD (streaming imputation), stat. parity."""
    pytest.importorskip("cyvcf2")
    path = tmp_path / "panel.vcf"
    _write_synthetic_vcf(path)
    prs_df, platform = _synthetic_prs()

    ref = _fit_imputation_vcf(path, prs_df, platform, "per_target", monkeypatch)
    bat = _fit_imputation_vcf(path, prs_df, platform, "batched", monkeypatch)

    rmods = {m.variant_id: m for m in ref.imputed_models}
    bmods = {m.variant_id: m for m in bat.imputed_models}
    assert set(rmods) == set(bmods) and len(rmods) > 10  # meaningful #targets trained
    for vid, rm in rmods.items():
        bm = bmods[vid]
        assert list(bm.predictor_variant_ids) == list(rm.predictor_variant_ids)
        np.testing.assert_allclose(
            np.asarray(bm.coefficients), np.asarray(rm.coefficients), atol=2e-3, rtol=2e-3
        )
        assert abs(bm.intercept - rm.intercept) < 5e-3
        assert abs(bm.imputation_r2 - rm.imputation_r2) < 5e-3

    # Calibration is an aggregate over all targets' OOF predictions.
    rc, bc = ref._calibration_params, bat._calibration_params
    assert abs(rc.scaling_factor - bc.scaling_factor) < 1e-2
    assert abs(rc.calibration_intercept - bc.calibration_intercept) < 1e-2
    assert abs(rc.calibration_r2 - bc.calibration_r2) < 5e-3


def test_batched_projection_matches_per_target(tmp_path, monkeypatch):
    """Forced batched solve == per-target oracle for streaming projection (shared seam)."""
    pytest.importorskip("cyvcf2")
    path = tmp_path / "panel.vcf"
    _write_synthetic_vcf(path)
    prs_df, platform = _synthetic_prs()

    ref = _fit_projection_vcf(path, prs_df, platform, "per_target", monkeypatch)
    bat = _fit_projection_vcf(path, prs_df, platform, "batched", monkeypatch)

    rmods = {m.region_id: m for m in ref._region_models}
    bmods = {m.region_id: m for m in bat._region_models}
    assert set(rmods) == set(bmods) and len(rmods) >= 1
    for rid, rm in rmods.items():
        bm = bmods[rid]
        np.testing.assert_allclose(
            np.asarray(bm.coefficients), np.asarray(rm.coefficients), atol=2e-3, rtol=2e-3
        )
        # Region intercept is on the S_R (region-PRS) scale, which is large — use a
        # relative tolerance (the absolute ~5e-3 gap is a ~1e-3 relative FISTA/CD wobble).
        np.testing.assert_allclose(bm.intercept, rm.intercept, atol=2e-3, rtol=5e-3)


def _multichrom_panel(n_chrom=3, per_chrom=24, n_samples=80, seed=99):
    """Multi-chromosome integer-dosage panel with several targets per chromosome, so the
    forced batched solve handles genuine multi-target chunks inside each chromosome shard."""
    rng = np.random.default_rng(seed)
    vrows, cols, prows, platform = [], [], [], []
    for c in range(1, n_chrom + 1):
        chrom, base = str(c), c * 1_000_000
        for i in range(per_chrom):
            pos = base + i * 5_000
            vid = f"{chrom}:{pos}"
            vrows.append({
                "variant_id": vid, "chromosome": chrom, "position": pos,
                "ref_allele": "A", "alt_allele": "G",
            })
            cols.append(rng.binomial(2, rng.uniform(0.2, 0.8), size=n_samples).astype(np.float32))
            prows.append({
                "variant_id": vid, "chromosome": chrom, "position": pos,
                "effect_allele": "G", "other_allele": "A", "beta": float(rng.uniform(-0.4, 0.4)),
            })
            if i % 3 != 0:  # ~2/3 on-platform (chip), ~1/3 missing targets
                platform.append(vid)
    gd = GenotypeData(
        dosage_matrix=np.stack(cols, axis=1),
        variant_info=pd.DataFrame(vrows),
        sample_ids=[f"S{i}" for i in range(n_samples)],
    )
    return gd, pd.DataFrame(prows), platform


@pytest.mark.parametrize("n_workers", [2])
def test_batched_bit_identical_across_workers(monkeypatch, n_workers):
    """Forced batched fit is bit-identical across worker counts (Phase-7 composition)."""
    monkeypatch.setenv("IMPUTED_PRS_SOLVE", "batched")
    gd, prs, plat = _multichrom_panel()

    def _fit(nw):
        m = LinearImputationPRS(
            window_size=100_000, backend="streaming", tuning_scope="none", cv_folds=3,
            alpha=0.01, l1_ratio=0.5, random_state=17, n_workers=nw, device="cpu", verbose=0,
        )
        m.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
              genome_build="GRCh37")
        return m

    with threadpool_limits(limits=1):
        base = _fit(1)
        got = _fit(n_workers)
    assert len(base._imputed_models) > 3  # multiple targets per chromosome trained
    _assert_imputation_identical(base, got)


# --- Reference CV (leave-one-fold-out, additive S_full − S_fold(k)) -------------------
# The batched solve routes through the same _run_cv_chunk for imputation and projection.


def _cv_fold_indices(n_samples, n_folds, seed):
    rng = np.random.default_rng(seed)
    return [np.asarray(idx) for idx in np.array_split(rng.permutation(n_samples), n_folds)]


def test_batched_reference_cv_matches_per_target(monkeypatch):
    """Forced-batched reference CV == per-target ``fit_reference_folds`` (statistical parity).

    Every outer fold's per-target model (``solve_reference_folds_batched`` under the hood)
    must match the sklearn-CD oracle within the FISTA band, so the CV metric it feeds is
    unchanged. Small panel + forced batched path so it runs in seconds.
    """
    gd, prs, plat = _multichrom_panel(n_samples=90)
    fi = _cv_fold_indices(90, 3, seed=5)

    def _run(mode):
        monkeypatch.setenv("IMPUTED_PRS_SOLVE", mode)
        d = LinearImputationPRS(
            backend="streaming", tuning_scope="none", cv_folds=3, alpha=0.01, l1_ratio=0.5,
            random_state=17, n_workers=1, device="cpu", verbose=0,
        )
        return d._reference_cv_fold_models(gd, prs, platform_variants=plat,
                                           fold_indices=fi, genome_build="GRCh37")

    with threadpool_limits(limits=1):
        ref = _run("per_target")
        bat = _run("batched")
    assert ref is not None and bat is not None
    checked = 0
    for k in ref.fold_imputed_models:
        rm = {m.variant_id: m for m in ref.fold_imputed_models[k]}
        bm = {m.variant_id: m for m in bat.fold_imputed_models[k]}
        assert set(rm) == set(bm) and rm, (k, set(rm), set(bm))
        for vid, r in rm.items():
            b = bm[vid]
            assert list(b.predictor_variant_ids) == list(r.predictor_variant_ids), (k, vid)
            np.testing.assert_allclose(
                np.asarray(b.coefficients), np.asarray(r.coefficients), atol=2e-3, rtol=2e-3
            )
            assert abs(b.intercept - r.intercept) < 5e-3
            checked += 1
    assert checked > 5  # multiple targets across folds actually compared


@pytest.mark.parametrize("n_workers", [2])
def test_batched_reference_cv_bit_identical_across_workers(monkeypatch, n_workers):
    """Forced-batched reference CV is bit-identical across worker counts (Phase-7 compose)."""
    monkeypatch.setenv("IMPUTED_PRS_SOLVE", "batched")
    gd, prs, plat = _multichrom_panel(n_samples=90)
    fi = _cv_fold_indices(90, 3, seed=5)

    def _run(nw):
        d = LinearImputationPRS(
            backend="streaming", tuning_scope="none", cv_folds=3, alpha=0.01, l1_ratio=0.5,
            random_state=17, n_workers=nw, device="cpu", verbose=0,
        )
        return d._reference_cv_fold_models(gd, prs, platform_variants=plat,
                                           fold_indices=fi, genome_build="GRCh37")

    with threadpool_limits(limits=1):
        base, got = _run(1), _run(n_workers)
    assert base is not None and got is not None
    for k in base.fold_imputed_models:
        ba = {m.variant_id: m for m in base.fold_imputed_models[k]}
        gb = {m.variant_id: m for m in got.fold_imputed_models[k]}
        assert set(ba) == set(gb) and ba, (k, set(ba), set(gb))
        for vid in ba:
            assert np.array_equal(
                np.asarray(ba[vid].coefficients), np.asarray(gb[vid].coefficients)
            ), (k, vid)
            assert ba[vid].intercept == gb[vid].intercept, (k, vid)
