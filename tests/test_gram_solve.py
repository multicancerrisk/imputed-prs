"""Stage 0: sample-free Gram solver parity vs the legacy per-variant fit.

Asserts ``compute.gram_solve.fit_from_local_gram`` reproduces
``models.elastic_net.fit_single_variant_model`` (coefficients, intercept, cv_mse,
cv_r2, and per-sample out-of-fold predictions) from sufficient statistics alone,
across shapes/hyperparameters, plus the three intercept-only fallbacks and the
sklearn-private-API guardrail. Runs in seconds.
"""

import warnings

import numpy as np
import pytest
from sklearn.model_selection import KFold

from imputed_prs.compute.gram_solve import (
    _BATCH_MAX_P_ENET,
    LocalGramBlock,
    _batched_fista,
    _select_solver,
    fit_from_local_gram,
    ridge_gram_fit,
    solve_blocks_batched,
    standardize_from_moments,
)
from imputed_prs.models.elastic_net import fit_single_variant_model


def build_block(X: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> LocalGramBlock:
    """Assemble a LocalGramBlock from raw (X, y), folded exactly like the legacy fit.

    The legacy fit runs ``KFold(shuffle=True, random_state=seed).split(X_valid)``;
    with no NaNs X_valid == X, so the same call reproduces its held-out partition.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape

    block = LocalGramBlock(
        n=n,
        G=X.T @ X,
        c=X.T @ y,
        zsum=X.sum(axis=0),
        zsqsum=(X * X).sum(axis=0),
        ysum=float(y.sum()),
        ysqsum=float(y @ y),
    )
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for _train_idx, val_idx in kfold.split(X):
        Xv, yv = X[val_idx], y[val_idx]
        block.fold_G.append(Xv.T @ Xv)
        block.fold_c.append(Xv.T @ yv)
        block.fold_zsum.append(Xv.sum(axis=0))
        block.fold_zsqsum.append((Xv * Xv).sum(axis=0))
        block.fold_ysum.append(float(yv.sum()))
        block.fold_ysqsum.append(float(yv @ yv))
        block.fold_n.append(len(val_idx))
    return block


def oof_from_fold_models(result, X, cv_folds, seed):
    """Reconstruct per-sample OOF predictions from the returned raw fold models."""
    n = X.shape[0]
    oof = np.full(n, np.nan)
    if result.is_intercept_only:
        oof[:] = result.intercept
        return oof
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for k, (_train_idx, val_idx) in enumerate(kfold.split(X)):
        fm = result.fold_models[k]
        oof[val_idx] = X[val_idx] @ fm.coef + fm.intercept
    return oof


def make_dosage_data(n, p, seed):
    """Synthetic 0-2 dosage-like data with a real linear signal (no NaN)."""
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(0.1, 0.9, size=p)
    X = rng.binomial(2, freqs, size=(n, p)).astype(np.float64)
    true_w = rng.randn(p) * 0.3
    y = X @ true_w + rng.randn(n) * 0.5
    # Keep y in a plausible dosage range so std/variance edge guards behave alike.
    y = np.clip(y - y.min(), 0.0, 2.0)
    return X, y


@pytest.mark.parametrize("n,p", [(200, 8), (500, 20), (137, 3), (300, 1)])
@pytest.mark.parametrize("alpha,l1_ratio", [(0.01, 0.5), (0.1, 0.9), (0.001, 0.1)])
def test_gram_matches_legacy(n, p, alpha, l1_ratio):
    seed = 42
    cv_folds = 5
    X, y = make_dosage_data(n, p, seed=n + p)

    legacy = fit_single_variant_model(
        y, X, l1_ratio=l1_ratio, alpha=alpha, cv_folds=cv_folds, random_state=seed
    )
    block = build_block(X, y, cv_folds, seed)
    gram = fit_from_local_gram(block, alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv_folds)

    assert gram.is_intercept_only == legacy.is_intercept_only
    np.testing.assert_allclose(
        gram.coefficients, legacy.coefficients, atol=1e-8, rtol=1e-6
    )
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(gram.cv_mse, legacy.cv_mse, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(gram.cv_r2, legacy.cv_r2, atol=1e-7, rtol=1e-6)

    # Per-sample out-of-fold predictions (consumed by streaming calibration).
    oof = oof_from_fold_models(gram, X, cv_folds, seed)
    np.testing.assert_allclose(oof, legacy.cv_predictions, atol=1e-7, rtol=1e-6)


def test_scale_highreg_kink_parity():
    """At 1000G scale, strong-L1 fits match on the exported model but wobble on OOF.

    With heavy L1 (alpha=0.1, l1=0.9) a marginal predictor can sit exactly on the
    L1 zero-kink; the tol=1e-4 solver then includes/excludes it differently between
    the legacy per-fold fit and the Gram (full - held-out) fold fit — two equally
    valid optima. The **final** model (what is exported) still matches to ~1e-12;
    the per-fold OOF predictions can differ by ~1e-3, which is uncorrelated across
    variants and washes out in the aggregate calibration score (statistical parity).
    """
    n, p, alpha, l1 = 2504, 50, 0.1, 0.9
    seed, cv_folds = 42, 5
    X, y = make_dosage_data(n, p, seed=n + p)

    legacy = fit_single_variant_model(
        y, X, l1_ratio=l1, alpha=alpha, cv_folds=cv_folds, random_state=seed
    )
    block = build_block(X, y, cv_folds, seed)
    gram = fit_from_local_gram(block, alpha=alpha, l1_ratio=l1, cv_folds=cv_folds)

    # Exported model reproduces legacy tightly.
    np.testing.assert_allclose(gram.coefficients, legacy.coefficients, atol=1e-9)
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-9)
    # CV metrics + OOF are within a small statistical band (kink sensitivity).
    assert abs(gram.cv_r2 - legacy.cv_r2) < 5e-3
    oof = oof_from_fold_models(gram, X, cv_folds, seed)
    assert np.max(np.abs(oof - legacy.cv_predictions)) < 5e-3


@pytest.mark.parametrize("n,p", [(200, 8), (500, 20), (137, 3), (300, 1)])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.001])
def test_ridge_closed_form_exact(n, p, alpha):
    """The ridge fast-path solves ``(G_std + alpha*n*I) w = q_std`` exactly.

    Cholesky (``ridge_gram_fit``) must agree with a direct ``np.linalg.solve`` to
    machine precision — this is a deterministic closed form, not an iteration.
    """
    X, y = make_dosage_data(n, p, seed=n + p)
    block = build_block(X, y, 5, 42)
    G_std, q_std, _yc2, _mean, _scale, _ybar = standardize_from_moments(
        block.G, block.c, block.zsum, block.zsqsum, block.ysum, block.ysqsum, n
    )
    w = ridge_gram_fit(G_std, q_std, n, alpha)
    w_ref = np.linalg.solve(G_std + alpha * n * np.eye(p), q_std)
    np.testing.assert_allclose(w, w_ref, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("n,p", [(200, 8), (500, 20), (137, 3), (300, 1)])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.001])
def test_ridge_matches_legacy(n, p, alpha):
    """``l1_ratio=0`` fast-path agrees with the legacy sklearn ridge (ElasticNet l1=0).

    The fast-path is the *exact* ridge optimum (``test_ridge_closed_form_exact``); the
    legacy path reaches it only iteratively, and how tightly is **sklearn-version
    dependent** — its coordinate descent for the degenerate ``l1_ratio=0`` case converges
    to ~1e-14 under sklearn 1.8 but only ~1e-5 under sklearn 1.9. So the fast-path is if
    anything *more* accurate than legacy, and the two agree within the sanctioned
    statistical-parity band (observed worst ~1e-5, checked with generous headroom below
    the 5e-3 band) rather than bit-for-bit.
    """
    seed, cv_folds = 42, 5
    X, y = make_dosage_data(n, p, seed=n + p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns on ElasticNet(l1_ratio=0)
        legacy = fit_single_variant_model(
            y, X, l1_ratio=0.0, alpha=alpha, cv_folds=cv_folds, random_state=seed
        )
    gram = fit_from_local_gram(
        build_block(X, y, cv_folds, seed), alpha=alpha, l1_ratio=0.0, cv_folds=cv_folds
    )

    assert gram.is_intercept_only == legacy.is_intercept_only
    np.testing.assert_allclose(gram.coefficients, legacy.coefficients, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(gram.cv_mse, legacy.cv_mse, atol=1e-3, rtol=1e-3)
    assert abs(gram.cv_r2 - legacy.cv_r2) < 5e-3
    oof = oof_from_fold_models(gram, X, cv_folds, seed)
    assert np.max(np.abs(oof - legacy.cv_predictions)) < 5e-3


def test_fallback_no_predictors():
    X, y = make_dosage_data(100, 1, seed=1)
    X0 = np.empty((100, 0))
    legacy = fit_single_variant_model(y, X0, cv_folds=5, random_state=0)
    block = LocalGramBlock(
        n=100,
        G=np.empty((0, 0)),
        c=np.empty(0),
        zsum=np.empty(0),
        zsqsum=np.empty(0),
        ysum=float(y.sum()),
        ysqsum=float(y @ y),
    )
    gram = fit_from_local_gram(block, alpha=0.01, l1_ratio=0.5, cv_folds=5)
    assert gram.is_intercept_only and legacy.is_intercept_only
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-12)
    np.testing.assert_allclose(gram.cv_mse, legacy.cv_mse, atol=1e-12)
    assert gram.cv_r2 == 0.0


def test_fallback_too_few_valid():
    X, y = make_dosage_data(4, 3, seed=2)  # n_valid=4 < cv_folds=5
    legacy = fit_single_variant_model(y, X, cv_folds=5, random_state=0)
    block = build_block(X, y, cv_folds=4, seed=0)  # folds irrelevant; fallback fires
    block.fold_G.clear()  # simulate: caller need not populate folds when n < cv_folds
    block.fold_n.clear()
    gram = fit_from_local_gram(block, alpha=0.01, l1_ratio=0.5, cv_folds=5)
    assert gram.is_intercept_only and legacy.is_intercept_only
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-12)


def test_fallback_zero_variance_target():
    rng = np.random.RandomState(3)
    X = rng.binomial(2, 0.5, size=(200, 5)).astype(np.float64)
    y = np.full(200, 1.0)  # constant target
    legacy = fit_single_variant_model(y, X, cv_folds=5, random_state=0)
    block = build_block(X, y, cv_folds=5, seed=0)
    gram = fit_from_local_gram(block, alpha=0.01, l1_ratio=0.5, cv_folds=5)
    assert gram.is_intercept_only and legacy.is_intercept_only
    np.testing.assert_allclose(gram.intercept, legacy.intercept, atol=1e-12)


def test_private_solver_guardrail():
    """The sklearn private Gram CD must be present and agree with ElasticNet here."""
    assert _select_solver() is True, (
        "sklearn private enet_coordinate_descent_gram failed the self-test; "
        "the pinned sklearn may have broken the private API."
    )


# ---------------------------------------------------------------------------
# Phase 8 — batched multi-target solve (solve_blocks_batched). Ridge is exact
# (closed form), so it is the cleanest oracle for the pad/mask/cv-reconstruct
# scaffolding, independent of the FISTA optimizer (commit 2).
# ---------------------------------------------------------------------------
def _assert_result_close(got, exp, atol=1e-8, rtol=1e-6):
    assert got.is_intercept_only == exp.is_intercept_only
    assert got.n_predictors == exp.n_predictors
    np.testing.assert_allclose(got.coefficients, exp.coefficients, atol=atol, rtol=rtol)
    np.testing.assert_allclose(got.intercept, exp.intercept, atol=atol, rtol=rtol)
    np.testing.assert_allclose(got.cv_mse, exp.cv_mse, atol=atol, rtol=rtol)
    np.testing.assert_allclose(got.cv_r2, exp.cv_r2, atol=atol, rtol=rtol)
    assert len(got.fold_models) == len(exp.fold_models)
    for fg, fe in zip(got.fold_models, exp.fold_models):
        np.testing.assert_allclose(fg.coef, fe.coef, atol=atol, rtol=rtol)
        np.testing.assert_allclose(fg.intercept, fe.intercept, atol=atol, rtol=rtol)


def test_solve_blocks_batched_ridge_matches_per_target():
    """Batched ridge solve == per-target fit_from_local_gram (exact closed form).

    Heterogeneous predictor counts (incl. an underdetermined p>n block) exercise the
    pad-to-uniform-P + mask path; the sort-by-p sub-batching must not perturb results.
    """
    cv_folds, alpha = 5, 0.05
    specs = [(200, 8), (500, 20), (137, 3), (300, 1), (250, 12), (10, 15)]  # last: p > n
    blocks = [
        build_block(*make_dosage_data(n, p, seed=n + p), cv_folds, seed=42)
        for n, p in specs
    ]
    got = solve_blocks_batched(blocks, alpha=alpha, l1_ratio=0.0, cv_folds=cv_folds)
    for block, g in zip(blocks, got):
        exp = fit_from_local_gram(block, alpha=alpha, l1_ratio=0.0, cv_folds=cv_folds)
        _assert_result_close(g, exp)


def test_solve_blocks_batched_partitions_fallbacks_and_none():
    """The 3 intercept-only fallbacks + ``None`` pass-through match the per-target path.

    Fallbacks are partitioned on the host and reuse ``_intercept_only_result``, so they are
    bit-identical (atol=1e-12); ``None`` inputs stay ``None`` and never enter the solve.
    """
    cv_folds, alpha = 5, 0.0
    X, y = make_dosage_data(200, 6, seed=7)
    real = build_block(X, y, cv_folds, seed=42)

    no_pred = LocalGramBlock(
        n=100, G=np.empty((0, 0)), c=np.empty(0),
        zsum=np.empty(0), zsqsum=np.empty(0),
        ysum=float(y[:100].sum()), ysqsum=float(y[:100] @ y[:100]),
    )
    Xc, _ = make_dosage_data(200, 4, seed=9)
    zerovar = build_block(Xc, np.full(200, 1.0), cv_folds, seed=42)  # constant target

    blocks = [real, None, no_pred, zerovar]
    got = solve_blocks_batched(blocks, alpha=alpha, l1_ratio=0.0, cv_folds=cv_folds)
    assert got[1] is None
    for b, g in zip(blocks, got):
        if b is None:
            continue
        exp = fit_from_local_gram(b, alpha=alpha, l1_ratio=0.0, cv_folds=cv_folds)
        atol = 1e-12 if exp.is_intercept_only else 1e-8
        _assert_result_close(g, exp, atol=atol)


def _blocks_batched_fixture(cv=5):
    """Well-determined heterogeneous predictor counts — the GPU parity fixture's set.

    The batched pad/mask handling of an underdetermined p>n block is covered exactly by the
    ridge test above and (for FISTA) by ``test_solve_blocks_batched_underdetermined`` below.
    """
    specs = [(500, 8), (500, 20), (500, 3), (500, 1), (500, 12), (500, 40)]
    return [build_block(*make_dosage_data(n, p, seed=100 + p), cv, seed=7) for n, p in specs]


@pytest.mark.parametrize(
    "alpha,l1_ratio",
    [
        (0.01, 0.0), (0.1, 0.0),  # ridge fast-path (exact)
        (0.01, 0.5), (0.1, 0.9), (0.01, 0.1),  # elastic net (batched FISTA)
    ],
)
def test_solve_blocks_batched_matches_per_target(alpha, l1_ratio):
    """Batched solve == per-target ``fit_from_local_gram`` within statistical parity.

    Torch-free CPU analogue of ``test_compute_backend.py::test_solve_blocks_matches_cpu``,
    at the same tolerances: ridge is exact; the batched-FISTA L1 path matches sklearn
    coordinate descent to ~1e-3 on coefficients / ~5e-3 on CV metrics — the sanctioned band
    the GPU FISTA already satisfies against the same oracle.
    """
    cv = 5
    blocks = _blocks_batched_fixture(cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns on ElasticNet(l1_ratio=0)
        exp = [
            fit_from_local_gram(b, alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv)
            for b in blocks
        ]
    got = solve_blocks_batched(blocks, alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv)
    for a, g in zip(exp, got):
        assert a.is_intercept_only == g.is_intercept_only
        np.testing.assert_allclose(g.coefficients, a.coefficients, atol=1e-3, rtol=1e-3)
        assert abs(a.intercept - g.intercept) < 1e-3
        assert abs(a.cv_r2 - g.cv_r2) < 5e-3
        assert abs(a.cv_mse - g.cv_mse) < 5e-3


def test_batched_fista_warm_start_same_optimum_fewer_iters():
    """Warm-starting FISTA near the optimum yields the same solution in fewer iterations.

    This is the mechanism ``_solve_padded_core`` uses to seed each fold model from the
    full-data solution (a ~1/K perturbation). FISTA is convex, so warm==cold; only the
    iteration count drops.
    """
    rng = np.random.RandomState(0)
    B, P = 8, 25
    M = rng.randn(B, P, P)
    G_std = np.einsum("bij,bkj->bik", M, M) / P  # SPD standardized Gram blocks
    q_std = rng.randn(B, P)
    n = np.full(B, 500.0)
    mask = np.ones((B, P))
    alpha, l1 = 0.01, 0.5

    w_cold, i_cold = _batched_fista(G_std, q_std, n, alpha, l1, mask)
    w_seed = w_cold + rng.randn(B, P) * 1e-3  # a near-optimal seed (like fold-from-final)
    w_warm, i_warm = _batched_fista(G_std, q_std, n, alpha, l1, mask, w_init=w_seed)

    np.testing.assert_allclose(w_warm, w_cold, atol=1e-5)  # same optimum (convex)
    assert i_warm < i_cold  # warm start converges in fewer iterations


def test_solve_blocks_batched_p_gate_falls_back_large_p():
    """The predictor-count crossover gate routes large-p blocks to the exact per-target path.

    Batching amortizes per-target overhead only for small p; above ``_BATCH_MAX_P_ENET`` the
    per-target solve wins, so those blocks must return the *exact* ``fit_from_local_gram``
    result (not a FISTA approximation), while small-p blocks in the same call still batch.
    """
    cv, alpha, l1 = 5, 0.01, 0.5  # l1_ratio>0 → _BATCH_MAX_P_ENET gate
    small = build_block(*make_dosage_data(500, 10, seed=1), cv, seed=7)  # batched
    big_p = _BATCH_MAX_P_ENET + 40
    large = build_block(*make_dosage_data(500, big_p, seed=2), cv, seed=7)  # per-target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = [fit_from_local_gram(b, alpha=alpha, l1_ratio=l1, cv_folds=cv) for b in (small, large)]
    got = solve_blocks_batched([small, large], alpha, l1, cv)
    # Small p: batched FISTA within statistical parity.
    np.testing.assert_allclose(got[0].coefficients, exp[0].coefficients, atol=1e-3, rtol=1e-3)
    # Large p: routed to the per-target path → exact (bit-for-bit, not a parity band).
    np.testing.assert_allclose(got[1].coefficients, exp[1].coefficients, atol=1e-12)
    assert abs(got[1].intercept - exp[1].intercept) < 1e-12
    np.testing.assert_allclose(got[1].cv_r2, exp[1].cv_r2, atol=1e-12)
