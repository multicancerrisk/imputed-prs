"""Phase 6: leave-one-fold-out reference CV via additive sufficient statistics.

Commit 1 scope — the reusable per-fold solver ``compute.gram_solve.fit_reference_folds``
and ``compute.sufficient_stats.GlobalFolds.from_partition``. Asserts the additive
``S_full − S_fold(k)`` subtraction reproduces a *direct* fit on the fold-``k``-excluded
rows (the refit oracle) from sufficient statistics alone, and that the fold partition
is validated. Runs in seconds; no streaming/panel needed.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.compute.gram_solve import (
    LocalGramBlock,
    fit_from_local_gram,
    fit_reference_folds,
)
from imputed_prs.compute.sufficient_stats import GlobalFolds

pytestmark = pytest.mark.filterwarnings("ignore")

# Synthetic panel geometry — big enough for real ±W windows (many predictors/target).
N_SAMPLES = 60
N_VARIANTS = 120
SPACING = 10_000
WINDOW = 200_000
SEED = 42


# ---------------------------------------------------------------------------
# Helpers — synthetic dosage data + reference-CV partition + block builders.
# ---------------------------------------------------------------------------
def make_dosage_data(n, p, seed):
    """Synthetic 0-2 dosage-like data with a real linear signal (no NaN)."""
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(0.1, 0.9, size=p)
    X = rng.binomial(2, freqs, size=(n, p)).astype(np.float64)
    true_w = rng.randn(p) * 0.3
    y = X @ true_w + rng.randn(n) * 0.5
    y = np.clip(y - y.min(), 0.0, 2.0)
    return X, y


def reference_cv_folds(n, n_folds, random_state):
    """Reproduce ImputationEvaluator.cross_validate's outer partition byte-for-byte
    (evaluator.py:250-265): shuffle arange(n), contiguous chunks, last absorbs rest.
    """
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = n if i == n_folds - 1 else start + fold_size
        folds.append(idx[start:end])
    return folds


def write_synthetic_vcf(path, n_samples=N_SAMPLES, n_variants=N_VARIANTS, seed=7):
    """chr1 VCF with random biallelic genotypes (no missing calls)."""
    rng = np.random.RandomState(seed)
    samples = [f"S{i}" for i in range(n_samples)]
    alleles = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1,length=249250621>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    gt_str = {0: "0/0", 1: "0/1", 2: "1/1"}
    for v in range(n_variants):
        pos = 100_000 + v * SPACING
        ref, alt = alleles[v % len(alleles)]
        p = 0.2 + 0.6 * rng.rand()
        dos = rng.binomial(2, p, size=n_samples)
        gts = "\t".join(gt_str[int(d)] for d in dos)
        lines.append(f"1\t{pos}\trs{v}\t{ref}\t{alt}\t.\t.\t.\tGT\t{gts}")
    path.write_text("\n".join(lines) + "\n")
    return path


def synthetic_prs(seed=0):
    """A PRS over the synthetic panel; ~1/3 on-platform (observed), rest missing."""
    rng = np.random.RandomState(seed)
    alleles = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]
    rows, platform = [], []
    for v in range(N_VARIANTS):
        pos = 100_000 + v * SPACING
        ref, alt = alleles[v % len(alleles)]
        flip = v % 3 == 0  # exercise both effect orientations
        rows.append(dict(
            variant_id=f"rs{v}", chromosome="1", position=pos,
            effect_allele=(ref if flip else alt),
            other_allele=(alt if flip else ref),
            beta=float(rng.uniform(-0.5, 0.5)),
        ))
        if v % 3 == 1:
            platform.append(f"rs{v}")
    return pd.DataFrame(rows), platform


def plain_block(X, y):
    """A LocalGramBlock with full stats only (no per-fold slabs) — a direct fit."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return LocalGramBlock(
        n=X.shape[0],
        G=X.T @ X,
        c=X.T @ y,
        zsum=X.sum(axis=0),
        zsqsum=(X * X).sum(axis=0),
        ysum=float(y.sum()),
        ysqsum=float(y @ y),
    )


def reference_block(X, y, fold_indices):
    """Full LocalGramBlock whose per-fold slabs are the reference-CV outer folds."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    block = plain_block(X, y)
    for val_idx in fold_indices:
        Xv, yv = X[val_idx], y[val_idx]
        block.fold_G.append(Xv.T @ Xv)
        block.fold_c.append(Xv.T @ yv)
        block.fold_zsum.append(Xv.sum(axis=0))
        block.fold_zsqsum.append((Xv * Xv).sum(axis=0))
        block.fold_ysum.append(float(yv.sum()))
        block.fold_ysqsum.append(float(yv @ yv))
        block.fold_n.append(len(val_idx))
    return block


# ---------------------------------------------------------------------------
# GlobalFolds.from_partition
# ---------------------------------------------------------------------------
def test_from_partition_matches_explicit_partition():
    n, n_folds = 213, 4  # non-divisible ⇒ last fold absorbs the remainder
    fold_indices = reference_cv_folds(n, n_folds, random_state=42)
    folds = GlobalFolds.from_partition(fold_indices)

    assert folds.n == n
    assert folds.n_folds == n_folds
    # perm concatenates the folds in order; fold_slice(k) recovers each block.
    assert np.array_equal(folds.perm, np.concatenate(fold_indices))
    for k, val_idx in enumerate(fold_indices):
        sl = folds.fold_slice(k)
        assert np.array_equal(folds.perm[sl], val_idx)
    # permute round-trips a natural-order array into fold-block order.
    natural = np.arange(n) * 10.0
    assert np.array_equal(folds.permute(natural), natural[folds.perm])


def test_from_partition_rejects_non_partition():
    # Duplicated index (overlap) — silently corrupts every train Gram if allowed (R5).
    with pytest.raises(ValueError, match="complete, disjoint partition"):
        GlobalFolds.from_partition([np.array([0, 1, 2]), np.array([2, 3, 4])])
    # Missing index (gap): covers {0,1,3,4}, drops 2.
    with pytest.raises(ValueError, match="complete, disjoint partition"):
        GlobalFolds.from_partition([np.array([0, 1]), np.array([3, 4])])


# ---------------------------------------------------------------------------
# Numerical identity of the additive subtraction (verification (a)).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,p,n_folds", [(400, 12, 5), (250, 6, 3), (511, 20, 10)])
def test_subtraction_equals_direct_train_gram(n, p, n_folds):
    X, y = make_dosage_data(n, p, seed=n + p)
    fold_indices = reference_cv_folds(n, n_folds, random_state=7)
    block = reference_block(X, y, fold_indices)

    # Σ_k fold_G[k] == G_full  (the invariant the subtraction rests on).
    fold_sum = np.sum(block.fold_G, axis=0)
    np.testing.assert_allclose(fold_sum, block.G, atol=1e-9, rtol=0)

    # G_full − fold_G[k] == direct ZᵀZ over the fold-k-excluded (training) rows.
    for k in range(n_folds):
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(n_folds) if i != k]
        )
        Xtr = X[train_idx]
        np.testing.assert_allclose(
            block.G - block.fold_G[k], Xtr.T @ Xtr, atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            block.c - block.fold_c[k], Xtr.T @ y[train_idx], atol=1e-9, rtol=0
        )
        assert block.n - block.fold_n[k] == len(train_idx)  # R1: n_train exact


# ---------------------------------------------------------------------------
# Per-fold model parity: additive subtraction vs direct refit (verification (b)).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,l1_ratio", [(0.01, 0.5), (0.1, 0.9), (0.001, 0.1)])
@pytest.mark.parametrize("n,p,n_folds", [(400, 12, 5), (300, 8, 3)])
def test_fit_reference_folds_matches_direct_refit(n, p, n_folds, alpha, l1_ratio):
    X, y = make_dosage_data(n, p, seed=n + p)
    fold_indices = reference_cv_folds(n, n_folds, random_state=7)
    block = reference_block(X, y, fold_indices)

    cv_results = fit_reference_folds(
        block, alpha=alpha, l1_ratio=l1_ratio, cv_folds=5
    )
    assert len(cv_results) == n_folds

    for k in range(n_folds):
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(n_folds) if i != k]
        )
        # The refit oracle: a direct fit on exactly the fold-k-excluded rows.
        direct = fit_from_local_gram(
            plain_block(X[train_idx], y[train_idx]),
            alpha=alpha,
            l1_ratio=l1_ratio,
            cv_folds=5,
        )
        # Same solver, stats differ only by float64 subtraction — tight parity.
        # This also guards R1: a wrong n would rescale the penalty and blow past 1e-9.
        np.testing.assert_allclose(
            cv_results[k].coefficients, direct.coefficients, atol=1e-9, rtol=0
        )
        assert abs(cv_results[k].intercept - direct.intercept) < 1e-9
        assert cv_results[k].is_intercept_only == direct.is_intercept_only


def test_fit_reference_folds_intercept_only_uses_training_fold_mean():
    """A no-predictor target ⇒ each outer fold's model is intercept-only at the
    *training-fold* mean (matching a per-fold refit), not the full-panel mean."""
    n, n_folds = 200, 5
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 2.0, size=n)
    fold_indices = reference_cv_folds(n, n_folds, random_state=3)

    block = LocalGramBlock(
        n=n,
        G=np.empty((0, 0)),
        c=np.empty(0),
        zsum=np.empty(0),
        zsqsum=np.empty(0),
        ysum=float(y.sum()),
        ysqsum=float(y @ y),
    )
    for val_idx in fold_indices:  # only target moments — no predictor slabs
        yv = y[val_idx]
        block.fold_ysum.append(float(yv.sum()))
        block.fold_ysqsum.append(float(yv @ yv))
        block.fold_n.append(len(val_idx))

    results = fit_reference_folds(block, alpha=0.01, l1_ratio=0.5, cv_folds=5)
    assert len(results) == n_folds
    for k, val_idx in enumerate(fold_indices):
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(n_folds) if i != k]
        )
        assert results[k].is_intercept_only
        assert results[k].coefficients.size == 0
        assert abs(results[k].intercept - float(y[train_idx].mean())) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end streaming: one-pass additive CV vs a direct per-fold refit (b).
# ---------------------------------------------------------------------------
def test_streaming_reference_cv_matches_direct_refit(tmp_path):
    """streaming_reference_cv_impute's per-fold models (one pass, S_full − S_fold(k))
    reproduce a direct streaming refit on each training fold — within streaming tol."""
    pytest.importorskip("cyvcf2")
    from imputed_prs.compute.cv_stats import streaming_reference_cv_impute
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.io.genotype_loader import load_genotypes
    from imputed_prs.io.genotype_source import InMemoryGenotypeSource
    from imputed_prs.io.platform_loader import load_platform_variants_from_list

    path = tmp_path / "panel.vcf"
    write_synthetic_vcf(path)
    prs_df, platform = synthetic_prs()
    # Resolve the platform the same way LinearImputationPRS.fit does internally, so the
    # additive path and the direct refit build predictors in the same order.
    platform_set = load_platform_variants_from_list(platform)
    gd = load_genotypes(path=path)
    n = gd.n_samples
    n_folds = 3
    fold_indices = reference_cv_folds(n, n_folds, random_state=42)
    hp = dict(window_size=WINDOW, alpha=0.01, l1_ratio=0.5, cv_folds=5)

    # One streaming pass → per-outer-fold models by additive subtraction.
    cv = streaming_reference_cv_impute(
        InMemoryGenotypeSource(gd),
        prs_df,
        platform_set,
        fold_indices=fold_indices,
        random_state=SEED,
        device="cpu",
        **hp,
    )
    assert cv.fold_imputed_models is not None
    assert set(cv.fold_imputed_models) == set(range(n_folds))

    for k in range(n_folds):
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(n_folds) if i != k]
        )
        # The refit oracle: a full streaming fit on exactly the fold-k-excluded rows.
        direct = LinearImputationPRS(
            backend="streaming", device="cpu", tuning_scope="none",
            random_state=SEED, verbose=0, **hp,
        )
        direct.fit(
            reference_genotypes=InMemoryGenotypeSource(gd, sample_indices=train_idx),
            prs_definition=prs_df,
            platform_variants=platform,
            genome_build="GRCh38",
        )
        dmods = {m.variant_id: m for m in direct.imputed_models}
        amods = {m.variant_id: m for m in cv.fold_imputed_models[k]}
        assert set(amods) == set(dmods)
        assert len(amods) > 5  # a meaningful number of targets were trained
        for vid, dm in dmods.items():
            am = amods[vid]
            # Compare per predictor id (permutation-equivariant), not by position.
            d_coef = dict(zip(dm.predictor_variant_ids, np.asarray(dm.coefficients)))
            a_coef = dict(zip(am.predictor_variant_ids, np.asarray(am.coefficients)))
            assert set(a_coef) == set(d_coef)
            # Same solver, same training samples (subtraction vs direct) ⇒ ~1e-9.
            # Also the R1 guard: a wrong n_train would rescale the penalty >> 1e-9.
            for pid, dc in d_coef.items():
                assert abs(a_coef[pid] - dc) < 1e-9
            assert abs(am.intercept - dm.intercept) < 1e-9
            assert am.is_intercept_only == dm.is_intercept_only
