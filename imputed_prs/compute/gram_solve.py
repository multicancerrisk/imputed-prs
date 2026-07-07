"""Sample-free ElasticNet fit from local Gram blocks (Phase 2 solver core).

The legacy per-variant / per-region fit (``models/elastic_net.py::
fit_single_variant_model`` and ``models/projection.py::fit_single_region_model``)
regresses a target ``y`` on a local predictor block ``Z`` with sklearn ElasticNet,
5-fold CV, per-fold train-only standardization, and a back-transform to raw scale.
Coordinate descent (which is what sklearn runs with ``precompute=True``) needs only
the **Gram matrix** ``ZᵀZ`` and the cross-product ``Zᵀy`` — not the raw samples. So
this module reproduces that fit from **sufficient statistics** alone:

    G   = Σ z_a z_b   (raw predictor Gram, p×p)
    c   = Σ z_a y     (raw cross-product, p)
    Σz, Σz², Σy, Σy², n   (column / target moments)

plus the same statistics accumulated over each CV fold's held-out (validation)
samples, which let us reconstruct per-fold train-only standardization and the
held-out MSE without ever touching a sample matrix.

**Parity.** Standardization, penalty scaling, and back-transform are reproduced
exactly (verified to ~1e-14). The ElasticNet optimum is obtained by calling
sklearn's own compiled Gram coordinate descent on the moment-derived standardized
Gram — a faithful hand-rolled coordinate descent lands ~1e-5 away, so we do **not**
hand-roll the primary path. See :func:`enet_gram_fit`.

**Exactness caveat.** Reproducing the legacy fit bit-for-bit requires the local
Gram to be taken over the *same* samples the legacy fit would use. Legacy listwise-
deletes samples with a NaN in the target or any predictor; the streaming caller
instead mean-imputes (it cannot drop per-variant-varying rows from a shared Gram).
These coincide **iff there are no missing dosages** (true for the dense 1000G
reference). Under missingness this is mean-imputation, a documented, statistically-
valid deviation validated on quality (R²/calibration), not bit-parity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from imputed_prs.models.metrics import compute_cv_r2

# ---------------------------------------------------------------------------
# sklearn private Gram coordinate descent (primary solver) + a robust fallback.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard exercised only on incompatible sklearn
    from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram
    from sklearn.utils import check_random_state

    _HAVE_PRIVATE_GRAM_CD = True
    # The compiled routine takes a RandomState only for ``selection='random'``;
    # we always run cyclic (``random=False``) so it is never drawn from. Build it
    # once at import and reuse — constructing one per solve (millions of fits) cost
    # ~5% of the streaming pass (posix.urandom + validation) for a never-used arg.
    _GRAM_CD_RNG = check_random_state(0)
except Exception:  # noqa: BLE001 - any import failure falls back
    _HAVE_PRIVATE_GRAM_CD = False
    _GRAM_CD_RNG = None


_STD_EPS = 1e-8  # matches models/metrics.py::standardize_columns
_INTERCEPT_ONLY_ATOL = 1e-10  # matches elastic_net.py is_intercept_only test


@dataclass
class FoldModel:
    """A per-fold model on the **raw** predictor scale (for out-of-fold scoring).

    The out-of-fold prediction for a held-out sample with raw predictor row ``z``
    is ``z @ coef + intercept`` — the streaming driver applies this to buffered
    validation-fold columns to produce the per-sample OOF used by calibration.
    """

    coef: np.ndarray  # (p,) raw-dosage scale
    intercept: float


@dataclass
class GramFitResult:
    """Sample-free analogue of ``SingleVariantModelResult`` (minus per-sample OOF).

    ``fold_models`` are returned so the streaming driver can compute per-sample
    out-of-fold predictions from buffered columns; ``cv_mse``/``cv_r2`` are already
    reconstructed here from per-fold Gram blocks and match the legacy scalars.
    """

    coefficients: np.ndarray  # (p,) raw scale; empty for intercept-only
    intercept: float
    cv_mse: float
    cv_r2: float
    is_intercept_only: bool
    n_predictors: int
    n_samples: int  # number of samples contributing to the block (n_valid)
    fold_models: List[FoldModel] = field(default_factory=list)


@dataclass
class LocalGramBlock:
    """Raw sufficient statistics for one target's local regression.

    All arrays are on the **raw dosage scale** (no centering/standardization —
    that is done analytically here). Per-fold entries hold the statistics over each
    CV fold's *held-out* (validation) samples; the train-fold statistics are then
    ``full − held-out`` (deterministic leave-one-fold-out).
    """

    # Full valid-subset statistics
    n: int
    G: np.ndarray  # (p, p)   Σ z_a z_b
    c: np.ndarray  # (p,)     Σ z_a y
    zsum: np.ndarray  # (p,)  Σ z_a
    zsqsum: np.ndarray  # (p,) Σ z_a²
    ysum: float  # Σ y
    ysqsum: float  # Σ y²
    # Per-fold held-out (validation) statistics; each list has length n_folds
    fold_G: List[np.ndarray] = field(default_factory=list)
    fold_c: List[np.ndarray] = field(default_factory=list)
    fold_zsum: List[np.ndarray] = field(default_factory=list)
    fold_zsqsum: List[np.ndarray] = field(default_factory=list)
    fold_ysum: List[float] = field(default_factory=list)
    fold_ysqsum: List[float] = field(default_factory=list)
    fold_n: List[int] = field(default_factory=list)

    @property
    def n_predictors(self) -> int:
        return int(self.G.shape[0]) if self.G.size > 0 else 0


# ---------------------------------------------------------------------------
# Moment → standardized-Gram algebra (reproduces models/metrics.py exactly).
# ---------------------------------------------------------------------------
def standardize_from_moments(
    G: np.ndarray,
    c: np.ndarray,
    zsum: np.ndarray,
    zsqsum: np.ndarray,
    ysum: float,
    ysqsum: float,
    n: int,
    eps: float = _STD_EPS,
):
    """Analytically center/standardize a raw Gram block.

    Reproduces ``standardize_columns`` (ddof=0 std, ``scale=1.0`` for near-constant
    columns) followed by sklearn's internal centering for ``fit_intercept=True``
    (X standardized already has zero mean, so only y is centered):

        mean_a  = Σz_a / n
        scale_a = sqrt(Σz²_a/n − mean_a²)   (or 1.0 if < eps)
        ybar    = Σy / n
        G_std[a,b] = (G[a,b] − n·mean_a·mean_b) / (scale_a·scale_b)
        q_std[a]   = (c[a] − n·mean_a·ybar) / scale_a      # = X_std,aᵀ(y − ȳ)
        yc2        = Σy² − n·ybar²                          # = ‖y − ȳ‖²

    Returns ``(G_std, q_std, yc2, mean, scale, ybar)``.
    """
    mean = zsum / n
    var = zsqsum / n - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))  # guard tiny negatives from cancellation
    scale = np.where(std < eps, 1.0, std)
    ybar = ysum / n

    inv_scale = 1.0 / scale
    # G_std[a,b] = (G[a,b] − n·mean_a·mean_b)·inv_scale_a·inv_scale_b. Scale rows/cols
    # by broadcasting (no p×p temporary), then subtract a single rank-1 outer of the
    # standardized means ms = mean·inv_scale — one np.outer instead of two.
    ms = mean * inv_scale
    G_std = G * inv_scale[:, None]
    G_std *= inv_scale[None, :]
    G_std -= n * np.outer(ms, ms)
    q_std = (c - n * mean * ybar) * inv_scale
    yc2 = float(ysqsum - n * ybar * ybar)
    return G_std, q_std, yc2, mean, scale, ybar


def _enet_gram_private(
    G_std: np.ndarray,
    q_std: np.ndarray,
    yc2: float,
    n: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Primary path: sklearn's compiled Gram coordinate descent.

    Minimises ½wᵀQw − qᵀw + l1_reg‖w‖₁ + ½l2_reg‖w‖² with
    ``l1_reg = α·l1_ratio·n`` and ``l2_reg = α·(1−l1_ratio)·n`` (sklearn's ElasticNet
    penalty scaling). The routine reads ``y`` only as ‖y‖² for the duality-gap
    tolerance, so a length-1 synthetic ``[sqrt(yc2)]`` reproduces the exact stopping
    behaviour of a fit on the real centered target.
    """
    p = G_std.shape[0]
    l1_reg = alpha * l1_ratio * n
    l2_reg = alpha * (1.0 - l1_ratio) * n
    w = np.zeros(p, dtype=np.float64)
    Q = np.ascontiguousarray(G_std, dtype=np.float64)
    q = np.ascontiguousarray(q_std, dtype=np.float64)
    y_synth = np.array([np.sqrt(max(yc2, 0.0))], dtype=np.float64)
    # selection='cyclic' (random=False) ⇒ the RNG is never drawn from; reuse the
    # module singleton instead of constructing one per call.
    result = enet_coordinate_descent_gram(
        w, l1_reg, l2_reg, Q, q, y_synth, max_iter, tol, _GRAM_CD_RNG, False, False
    )
    # Returns (w, gap, tol, n_iter); w is also modified in place.
    return np.asarray(result[0], dtype=np.float64)


def _enet_gram_handrolled(
    G_std: np.ndarray,
    q_std: np.ndarray,
    yc2: float,
    n: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Break-glass fallback: plain Gram coordinate descent (no private sklearn API).

    Only used if sklearn's private routine is missing or fails the self-test. Lands
    within ~1e-5 of the exact optimum — inside statistical-parity tolerance, and does
    not require a positive-definite Gram (unlike a Cholesky synthetic-design trick).
    """
    p = G_std.shape[0]
    l1_reg = alpha * l1_ratio * n
    l2_reg = alpha * (1.0 - l1_ratio) * n
    w = np.zeros(p, dtype=np.float64)
    diag = np.diag(G_std).astype(np.float64)
    Hw = G_std @ w  # maintained residual gradient term Q·w
    tol_scaled = tol * max(yc2, 1e-12)
    for _ in range(max_iter):
        w_max = 0.0
        max_step = 0.0
        for j in range(p):
            if diag[j] == 0.0:
                continue
            w_j = w[j]
            # rho = q_j − Σ_{k≠j} G[j,k] w_k = q_j − (Hw_j − G[j,j] w_j)
            rho = q_std[j] - (Hw[j] - diag[j] * w_j)
            if rho > l1_reg:
                new_w = (rho - l1_reg) / (diag[j] + l2_reg)
            elif rho < -l1_reg:
                new_w = (rho + l1_reg) / (diag[j] + l2_reg)
            else:
                new_w = 0.0
            d = new_w - w_j
            if d != 0.0:
                Hw += G_std[:, j] * d  # rank-1 update of Q·w
                w[j] = new_w
            max_step = max(max_step, abs(d))
            w_max = max(w_max, abs(new_w))
        if w_max == 0.0 or max_step / (w_max + 1e-12) < tol_scaled / (yc2 + 1e-12):
            break
    return w


# One-time self-test result: True ⇒ use the private sklearn routine.
_USE_PRIVATE: Optional[bool] = None


def _select_solver() -> bool:
    """Decide once whether the private sklearn Gram CD is available and correct.

    Guards against sklearn changing/removing the private ``enet_coordinate_descent_gram``
    signature: if the routine is gone or disagrees with the public ``ElasticNet`` on a
    fixture, we fall back to the hand-rolled path and warn (loudly) instead of shipping
    silently-wrong coefficients.
    """
    global _USE_PRIVATE
    if _USE_PRIVATE is not None:
        return _USE_PRIVATE
    if not _HAVE_PRIVATE_GRAM_CD:
        _USE_PRIVATE = False
        warnings.warn(
            "sklearn private Gram coordinate descent unavailable; using the "
            "hand-rolled fallback (statistical parity only, ~1e-5).",
            RuntimeWarning,
            stacklevel=2,
        )
        return _USE_PRIVATE
    try:
        from sklearn.linear_model import ElasticNet

        rng = np.random.RandomState(0)
        X = rng.randn(64, 6)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        y = rng.randn(64)
        yc = y - y.mean()
        G = X.T @ X
        q = X.T @ yc
        n = X.shape[0]
        w_priv = _enet_gram_private(G, q, float(yc @ yc), n, 0.05, 0.5, 10000, 1e-4)
        ref = ElasticNet(alpha=0.05, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
        ref.fit(X, y)
        _USE_PRIVATE = bool(np.allclose(w_priv, ref.coef_, atol=1e-8))
        if not _USE_PRIVATE:
            warnings.warn(
                "sklearn private Gram coordinate descent disagrees with ElasticNet; "
                "falling back to the hand-rolled solver (statistical parity only).",
                RuntimeWarning,
                stacklevel=2,
            )
    except Exception as exc:  # noqa: BLE001
        _USE_PRIVATE = False
        warnings.warn(
            f"Gram solver self-test failed ({exc!r}); using hand-rolled fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _USE_PRIVATE


def ridge_gram_fit(
    G_std: np.ndarray, q_std: np.ndarray, n: int, alpha: float
) -> np.ndarray:
    """Closed-form ridge (``l1_ratio=0``) coefficients from a standardized Gram block.

    With no L1 term the ElasticNet objective ``½wᵀG_std w − q_stdᵀw + ½·l2·‖w‖²`` is a
    strictly-convex quadratic (``l2 = α·(1−0)·n = α·n``), minimized in closed form by

        w = (G_std + α·n·I)⁻¹ q_std

    solved via Cholesky (``G_std + α·n·I`` is symmetric positive-definite for ``α>0``).
    This is exact — it equals the coordinate-descent optimum but skips the iteration —
    and is the batched-Cholesky ridge fast-path (Phase 3, item c). It replaces routing
    ``l1_ratio=0`` through the (correct but iterative) ElasticNet coordinate descent.
    """
    p = G_std.shape[0]
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    l2 = alpha * n
    a = np.array(G_std, dtype=np.float64, copy=True)
    a[np.diag_indices(p)] += l2
    try:
        return cho_solve(cho_factor(a, check_finite=False), q_std, check_finite=False)
    except LinAlgError:
        # α≈0 (unregularized) or a numerically singular Gram: least-squares fallback.
        return np.linalg.lstsq(a, q_std, rcond=None)[0]


def enet_gram_fit(
    G_std: np.ndarray,
    q_std: np.ndarray,
    yc2: float,
    n: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 10000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Fit standardized-scale ElasticNet coefficients ``w_std`` from a Gram block.

    ``l1_ratio == 0`` (pure ridge) takes the exact closed-form :func:`ridge_gram_fit`
    fast-path; any L1 component routes through sklearn's compiled Gram coordinate
    descent (or the hand-rolled fallback).
    """
    if l1_ratio == 0.0:
        return ridge_gram_fit(G_std, q_std, n, alpha)
    if _select_solver():
        return _enet_gram_private(G_std, q_std, yc2, n, alpha, l1_ratio, max_iter, tol)
    return _enet_gram_handrolled(G_std, q_std, yc2, n, alpha, l1_ratio, max_iter, tol)


# ---------------------------------------------------------------------------
# Full per-unit fit (reproduces fit_single_variant_model / fit_single_region_model).
# ---------------------------------------------------------------------------
def _intercept_only_result(
    ysum: float, ysqsum: float, n: int, n_predictors: int
) -> GramFitResult:
    """Sample-free analogue of ``_fit_intercept_only_model``.

    intercept = mean target; cv_mse = population variance of the target; cv_r2 = 0.
    The out-of-fold "prediction" legacy uses is the **global** target mean on every
    valid sample (no per-fold split), so ``fold_models`` is empty and the driver
    scores OOF as the constant ``intercept``.
    """
    intercept = ysum / n if n > 0 else 0.0
    var = ysqsum / n - intercept * intercept if n > 0 else 0.0
    return GramFitResult(
        coefficients=np.array([]),
        intercept=float(intercept),
        cv_mse=float(max(var, 0.0)),
        cv_r2=0.0,
        is_intercept_only=True,
        n_predictors=n_predictors,
        n_samples=n,
        fold_models=[],
    )


def _raw_model_from_std(
    w_std: np.ndarray, mean: np.ndarray, scale: np.ndarray, ybar: float
):
    """Back-transform standardized coef/intercept to raw scale (matches metrics.py)."""
    coef_raw = w_std / scale
    intercept_raw = float(ybar - np.dot(w_std, mean / scale))
    return coef_raw, intercept_raw


def _fold_mse_from_block(
    G_val: np.ndarray,
    c_val: np.ndarray,
    zsum_val: np.ndarray,
    ysum_val: float,
    ysqsum_val: float,
    n_val: int,
    coef: np.ndarray,
    intercept: float,
) -> float:
    """Held-out MSE for a fold, from its val-subset Gram block and raw fold model.

    Σ_val (y − z·a − b)² expands to a quadratic in the fold's raw sufficient stats:
        Σy² + aᵀG_val a + n·b² − 2aᵀc_val − 2b·Σy + 2b·aᵀΣz
    """
    a = coef
    b = intercept
    sse = (
        ysqsum_val
        + float(a @ (G_val @ a))
        + n_val * b * b
        - 2.0 * float(a @ c_val)
        - 2.0 * b * ysum_val
        + 2.0 * b * float(a @ zsum_val)
    )
    return sse / n_val if n_val > 0 else 0.0


def fit_from_local_gram(
    block: LocalGramBlock,
    alpha: float,
    l1_ratio: float,
    cv_folds: int = 5,
    max_iter: int = 10000,
    tol: float = 1e-4,
) -> GramFitResult:
    """Fit one target's ElasticNet model from its local Gram block, sample-free.

    Reproduces ``fit_single_variant_model``: the three intercept-only fallbacks (in
    order), the final model on the full valid subset, and the K per-fold models with
    train-only standardization used to reconstruct ``cv_mse``/``cv_r2``.
    """
    p = block.n_predictors
    n = block.n

    # --- Fallbacks (same order as elastic_net.py) --------------------------------
    if p == 0:
        return _intercept_only_result(block.ysum, block.ysqsum, n, p)
    if n < cv_folds:
        return _intercept_only_result(block.ysum, block.ysqsum, n, p)
    ybar_full = block.ysum / n
    y_var = block.ysqsum / n - ybar_full * ybar_full
    if np.sqrt(max(y_var, 0.0)) < 1e-10:  # std(y_valid) < 1e-10 (ddof=0)
        return _intercept_only_result(block.ysum, block.ysqsum, n, p)

    # --- Cross-validation folds (held-out Gram blocks) ---------------------------
    fold_mses: List[float] = []
    fold_models: List[FoldModel] = []
    ss_res = 0.0
    n_folds = len(block.fold_n)
    for k in range(n_folds):
        # Train-fold statistics = full − held-out-k (leave-one-fold-out).
        G_tr = block.G - block.fold_G[k]
        c_tr = block.c - block.fold_c[k]
        zsum_tr = block.zsum - block.fold_zsum[k]
        zsqsum_tr = block.zsqsum - block.fold_zsqsum[k]
        ysum_tr = block.ysum - block.fold_ysum[k]
        ysqsum_tr = block.ysqsum - block.fold_ysqsum[k]
        n_tr = n - block.fold_n[k]

        G_std, q_std, yc2, mean_k, scale_k, ybar_k = standardize_from_moments(
            G_tr, c_tr, zsum_tr, zsqsum_tr, ysum_tr, ysqsum_tr, n_tr
        )
        w_std_k = enet_gram_fit(
            G_std, q_std, yc2, n_tr, alpha, l1_ratio, max_iter, tol
        )
        coef_k, intercept_k = _raw_model_from_std(w_std_k, mean_k, scale_k, ybar_k)
        fold_models.append(FoldModel(coef=coef_k, intercept=intercept_k))

        n_val = block.fold_n[k]
        fmse = _fold_mse_from_block(
            block.fold_G[k],
            block.fold_c[k],
            block.fold_zsum[k],
            block.fold_ysum[k],
            block.fold_ysqsum[k],
            n_val,
            coef_k,
            intercept_k,
        )
        fold_mses.append(fmse)
        ss_res += fmse * n_val  # pooled SS_res over all held-out samples

    # --- Final model on the full valid subset ------------------------------------
    G_std, q_std, yc2, mean_all, scale_all, ybar_all = standardize_from_moments(
        block.G, block.c, block.zsum, block.zsqsum, block.ysum, block.ysqsum, n
    )
    w_std = enet_gram_fit(G_std, q_std, yc2, n, alpha, l1_ratio, max_iter, tol)
    coef_raw, intercept_raw = _raw_model_from_std(w_std, mean_all, scale_all, ybar_all)

    cv_mse = float(np.mean(fold_mses)) if fold_mses else 0.0
    ss_tot = yc2  # Σ(y − ȳ)² over the full valid subset
    cv_r2 = 1.0 - ss_res / ss_tot if ss_tot >= 1e-10 else 0.0
    is_intercept_only = bool(np.allclose(w_std, 0.0, atol=_INTERCEPT_ONLY_ATOL))

    return GramFitResult(
        coefficients=coef_raw.copy(),
        intercept=intercept_raw,
        cv_mse=cv_mse,
        cv_r2=float(cv_r2),
        is_intercept_only=is_intercept_only,
        n_predictors=p,
        n_samples=n,
        fold_models=fold_models,
    )


# ---------------------------------------------------------------------------
# Leave-one-fold-out reference CV: per-outer-fold models by additive subtraction.
# ---------------------------------------------------------------------------
def fit_reference_folds(
    full_block: LocalGramBlock,
    alpha: float,
    l1_ratio: float,
    cv_folds: int = 5,
    max_iter: int = 10000,
    tol: float = 1e-4,
) -> List[GramFitResult]:
    """Fit one target's per-outer-fold models for leave-one-fold-out reference CV.

    ``full_block``'s per-fold entries are the **reference-CV outer folds** — a
    disjoint sample partition. For each outer fold ``k`` this returns the model
    trained on the *training fold* (all samples except fold ``k``), assembled by the
    additive subtraction ``S_train(k) = S_full − S_fold(k)`` and solved via the same
    :func:`fit_from_local_gram` main-fit path a direct refit on those samples would
    use. It reproduces ``fold_model.fit(train_data_k)`` (the refit oracle) without
    re-streaming the panel ``k`` times — the whole point of Phase 6.

    Two correctness invariants (verified by tests):

    * **R1** — each subtracted training block carries ``n = n_full − fold_n[k]`` and
      *every* moment subtracted consistently, so the ElasticNet penalty
      (``l1_reg = α·l1_ratio·n``) scales with the training-fold sample count. Passing
      the full ``n`` would over-penalize every fold by ``k/(k−1)``.
    * **R2** — ``cv_folds`` is the model's **inner** CV count (the ``n < cv_folds``
      intercept-only guard), *not* the number of outer folds; pass the same value the
      refit oracle would. Each training block's own fold lists are left empty, so the
      nested CV loop in :func:`fit_from_local_gram` is a no-op (its ``cv_mse``/``cv_r2``
      are 0 and unused — reference CV scores raw).
    """
    n_folds = len(full_block.fold_n)
    has_predictors = full_block.n_predictors > 0
    results: List[GramFitResult] = []
    for k in range(n_folds):
        n_tr = int(full_block.n - full_block.fold_n[k])
        ysum_tr = float(full_block.ysum - full_block.fold_ysum[k])
        ysqsum_tr = float(full_block.ysqsum - full_block.fold_ysqsum[k])
        if has_predictors:
            train_block = LocalGramBlock(
                n=n_tr,
                G=full_block.G - full_block.fold_G[k],
                c=full_block.c - full_block.fold_c[k],
                zsum=full_block.zsum - full_block.fold_zsum[k],
                zsqsum=full_block.zsqsum - full_block.fold_zsqsum[k],
                ysum=ysum_tr,
                ysqsum=ysqsum_tr,
            )
        else:
            train_block = LocalGramBlock(
                n=n_tr,
                G=np.empty((0, 0)),
                c=np.empty(0),
                zsum=np.empty(0),
                zsqsum=np.empty(0),
                ysum=ysum_tr,
                ysqsum=ysqsum_tr,
            )
        results.append(
            fit_from_local_gram(
                train_block,
                alpha=alpha,
                l1_ratio=l1_ratio,
                cv_folds=cv_folds,
                max_iter=max_iter,
                tol=tol,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Batched multi-target solve (Phase 8) — numpy analogue of the GPU
# GpuBackend._solve_blocks / _solve_padded_core / _standardize_solve_backtransform /
# _batched_ridge / _batched_fista (compute/gpu_backend.py:492-707). Many co-windowed
# targets share the chip-predictor band, so their local Gram blocks are solved together:
# pad heterogeneous predictor sets to a uniform P with a per-target mask (padded rows get
# an identity so they solve to 0), solve final + K leave-one-fold-out fold models as
# batched linear algebra, and reconstruct held-out MSE / cv_r2 batched. This amortizes the
# per-target Python-call overhead that dominates 2M-variant fits on CPU (the streaming
# _run_fit_chunk loop). CPU stays torch-free, so this is a *parallel* numpy implementation
# of the torch code rather than a shared array-module seam; the two are kept in lockstep
# and coupled only by the parity test (batched == per-target fit_from_local_gram within
# the sanctioned statistical band). Requires alpha > 0: the batched solve is all-or-nothing
# on a singular block (one degenerate block raises for the whole batch), and alpha > 0 makes
# every G_std + α·n·I / + l2 SPD, so there is no per-item lstsq fallback here.
# ---------------------------------------------------------------------------

# FISTA controls — identical to GpuBackend for parity with its proven behavior
# (compute/gpu_backend.py:323-327). Reused by _batched_fista (Phase 8 commit 2).
_FISTA_MAX_ITER = 2000
_FISTA_TOL = 1e-7
_POWER_ITERS = 12
# Power-iteration's Rayleigh quotient under-estimates λ_max, so 1/Lip would be a hair too
# large a step; nudge Lip up for a safe contraction (cheap robustness, numpy float64).
_LIPSCHITZ_MARGIN = 1.01
# Cap the padded (B, K, P, P) fold-Gram working set per sub-batch (matches the GPU budget).
_SOLVE_GRAM_BUDGET = 512 * 1024 * 1024

# Empirical CPU crossover (measured by benchmarks/verify_batched_solve.py): batching amortizes
# the per-target Python-call overhead only while the per-block linear algebra is small. Above
# these predictor counts the per-target scipy path wins — the compiled Cholesky / coordinate
# descent already saturates, and FISTA in particular needs many more iterations than sklearn's
# CD at large p — so large-p blocks fall back to the per-target solve (never a regression). The
# ridge closed form tolerates a larger p than FISTA. Warm starts + screening (later Phase-8
# commits) attack FISTA's iteration count and effective p, widening the batched range.
_BATCH_MAX_P_RIDGE = 64
_BATCH_MAX_P_ENET = 32


def solve_blocks_batched(
    blocks: List[Optional[LocalGramBlock]],
    alpha: float,
    l1_ratio: float,
    cv_folds: int = 5,
) -> List[Optional[GramFitResult]]:
    """Batched analogue of ``[fit_from_local_gram(b, alpha, l1_ratio, cv_folds) for b in blocks]``.

    Numpy float64 port of ``GpuBackend._solve_blocks`` (compute/gpu_backend.py:492). The
    three intercept-only fallbacks (``p==0``, ``n<cv_folds``, ``std(y)<1e-10``) are
    partitioned on the host in the exact order :func:`fit_from_local_gram` applies them, so
    they reuse :func:`_intercept_only_result` bit-for-bit; only "real" blocks enter the
    padded solve. ``None`` inputs pass through as ``None`` (mirrors the streaming caller,
    which drops failed assembles). Ridge (``l1_ratio == 0``) is exact; the L1 case routes
    through batched FISTA (statistical parity with sklearn CD, same as the GPU).
    """
    results: List[Optional[GramFitResult]] = [None] * len(blocks)
    real: List[LocalGramBlock] = []
    real_pos: List[int] = []
    for i, b in enumerate(blocks):
        if b is None:
            continue
        p, n = b.n_predictors, b.n
        if p == 0 or n < cv_folds:
            results[i] = _intercept_only_result(b.ysum, b.ysqsum, n, p)
            continue
        ybar = b.ysum / n
        if np.sqrt(max(b.ysqsum / n - ybar * ybar, 0.0)) < 1e-10:
            results[i] = _intercept_only_result(b.ysum, b.ysqsum, n, p)
            continue
        real.append(b)
        real_pos.append(i)
    for pos, res in zip(real_pos, _solve_real_padded(real, alpha, l1_ratio, cv_folds)):
        results[pos] = res
    return results


def _solve_real_padded(
    blocks: List[LocalGramBlock], alpha: float, l1_ratio: float, cv_folds: int
) -> List[Optional[GramFitResult]]:
    """Sort blocks by predictor count, then sub-batch under :data:`_SOLVE_GRAM_BUDGET`.

    Sorting by ``p`` keeps each sub-batch's padding close to its own max predictor count —
    a free, exact pad-waste cut the GPU path does not do (it only sub-batches by memory,
    gpu_backend.py:534-547). Results are returned in the caller's original order.
    """
    if not blocks:
        return []
    max_p = _BATCH_MAX_P_RIDGE if l1_ratio == 0.0 else _BATCH_MAX_P_ENET
    order = sorted(range(len(blocks)), key=lambda i: blocks[i].n_predictors)
    K = len(blocks[0].fold_n)
    out: List[Optional[GramFitResult]] = [None] * len(blocks)
    n = len(order)
    s = 0
    while s < n:
        # Blocks are sorted ascending by p, so once one exceeds max_p, so do all that follow:
        # solve the whole large-p tail per-target (batching overhead would exceed the win).
        if blocks[order[s]].n_predictors > max_p:
            for i in range(s, n):
                bi = order[i]
                out[bi] = fit_from_local_gram(
                    blocks[bi], alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv_folds
                )
            break
        e = s + 1  # always at least one block per sub-batch (guarantees progress)
        while e < n:
            p_next = blocks[order[e]].n_predictors  # sorted asc → the new max in [s, e]
            if p_next > max_p:
                break  # large-p blocks handled by the per-target tail above
            cnt = e - s + 1
            if cnt * K * p_next * p_next * 8 > _SOLVE_GRAM_BUDGET:
                break
            e += 1
        sub = [blocks[order[i]] for i in range(s, e)]
        for i, res in zip(range(s, e), _solve_padded_core(sub, alpha, l1_ratio)):
            out[order[i]] = res
        s = e
    return out


def _solve_padded_core(
    blks: List[LocalGramBlock], alpha: float, l1_ratio: float
) -> List[GramFitResult]:
    """Pad a list of real Gram blocks to a uniform ``P`` and solve final + K fold models.

    Numpy port of ``GpuBackend._solve_padded_core`` (gpu_backend.py:549-624). Fold sizes /
    moments are carried **per block** (shape ``(B, K)``) rather than the GPU's shared
    ``(K,)`` — identical under the streaming global-fold partition (every target sees the
    same fold sizes), but also correct for arbitrary blocks (the block-level parity test).
    Held-out MSE / cv_r2 are reconstructed with the same quadratic as
    :func:`_fold_mse_from_block`.
    """
    B = len(blks)
    P = max(b.n_predictors for b in blks)
    K = len(blks[0].fold_n)
    zeros = lambda *shp: np.zeros(shp, dtype=np.float64)  # noqa: E731

    Gp = zeros(B, P, P); cp = zeros(B, P); zsp = zeros(B, P); zsqp = zeros(B, P)
    mask = zeros(B, P)
    fGp = zeros(B, K, P, P); fcp = zeros(B, K, P); fzsp = zeros(B, K, P); fzsqp = zeros(B, K, P)
    fysum_m = zeros(B, K); fysqsum_m = zeros(B, K); fn_m = zeros(B, K)
    ysum_v = zeros(B); ysqsum_v = zeros(B); n_v = zeros(B)
    for i, b in enumerate(blks):
        p = b.n_predictors
        mask[i, :p] = 1.0
        Gp[i, :p, :p] = b.G; cp[i, :p] = b.c
        zsp[i, :p] = b.zsum; zsqp[i, :p] = b.zsqsum
        fGp[i, :, :p, :p] = np.stack(b.fold_G)
        fcp[i, :, :p] = np.stack(b.fold_c)
        fzsp[i, :, :p] = np.stack(b.fold_zsum)
        fzsqp[i, :, :p] = np.stack(b.fold_zsqsum)
        fysum_m[i] = np.asarray(b.fold_ysum, dtype=np.float64)
        fysqsum_m[i] = np.asarray(b.fold_ysqsum, dtype=np.float64)
        fn_m[i] = np.asarray(b.fold_n, dtype=np.float64)
        ysum_v[i] = b.ysum; ysqsum_v[i] = b.ysqsum; n_v[i] = float(b.n)

    # Final model on the full valid subset.
    w_final, coef_raw, intercept_raw, is_io = _batched_standardize_solve(
        Gp, cp, zsp, zsqp, ysum_v, ysqsum_v, n_v, mask, alpha, l1_ratio
    )
    # K fold models (train-fold = full − held-out-k). Penalty scales with n_tr (invariant R1).
    # Warm-start each fold's FISTA from the full-data solution: a fold model is a ~1/K
    # perturbation of the final model in the (near-identical) standardized coordinates, so
    # this converges in far fewer iterations without changing the optimum (convex).
    fcoef = zeros(B, K, P); fint = zeros(B, K)
    for k in range(K):
        n_tr = n_v - fn_m[:, k]
        _wk, ck, ik, _ = _batched_standardize_solve(
            Gp - fGp[:, k], cp - fcp[:, k], zsp - fzsp[:, k], zsqp - fzsqp[:, k],
            ysum_v - fysum_m[:, k], ysqsum_v - fysqsum_m[:, k], n_tr, mask, alpha, l1_ratio,
            w_init=w_final,
        )
        fcoef[:, k] = ck; fint[:, k] = ik

    # Held-out MSE + cv metrics, batched. Per fold, over that fold's held-out stats:
    #   sse = Σy² + aᵀG_val a + n·b² − 2aᵀc_val − 2b·Σy + 2b·aᵀΣz    (a=fold coef, b=fold int).
    a = fcoef
    Ga = np.einsum("bkij,bkj->bki", fGp, a)
    aGa = (a * Ga).sum(-1); ac = (a * fcp).sum(-1); az = (a * fzsp).sum(-1)
    bt = fint
    sse = fysqsum_m + aGa + fn_m * bt * bt - 2.0 * ac - 2.0 * bt * fysum_m + 2.0 * bt * az
    with np.errstate(divide="ignore", invalid="ignore"):
        per_fold_mse = np.where(fn_m > 0, sse / fn_m, 0.0)
    cv_mse = per_fold_mse.mean(axis=1)
    ss_res = sse.sum(axis=1)  # pooled SS_res over all held-out samples
    ss_tot = ysqsum_v - ysum_v * ysum_v / n_v  # = Σ(y − ȳ)² over the full valid subset
    cv_r2 = np.where(ss_tot >= 1e-10, 1.0 - ss_res / ss_tot, 0.0)

    out: List[GramFitResult] = []
    for i, b in enumerate(blks):
        p = b.n_predictors
        fold_models = [
            FoldModel(coef=fcoef[i, k, :p].copy(), intercept=float(fint[i, k]))
            for k in range(K)
        ]
        out.append(
            GramFitResult(
                coefficients=coef_raw[i, :p].copy(),
                intercept=float(intercept_raw[i]),
                cv_mse=float(cv_mse[i]),
                cv_r2=float(cv_r2[i]),
                is_intercept_only=bool(is_io[i]),
                n_predictors=p,
                n_samples=int(b.n),
                fold_models=fold_models,
            )
        )
    return out


def _batched_standardize_solve(
    G: np.ndarray,
    c: np.ndarray,
    zsum: np.ndarray,
    zsqsum: np.ndarray,
    ysum: np.ndarray,
    ysqsum: np.ndarray,
    n: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    l1_ratio: float,
    w_init: Optional[np.ndarray] = None,
):
    """Batched :func:`standardize_from_moments` + :func:`enet_gram_fit` + back-transform.

    Numpy port of ``GpuBackend._standardize_solve_backtransform`` (gpu_backend.py:627). All
    inputs carry a leading batch axis ``B``; ``mask`` (B, P) marks real predictor slots.
    Padded coordinates are forced to zero: their Gram cross-terms are masked out and an
    identity is placed on the padded diagonal, so they solve to ``w_std=0`` and decouple
    from the real block (making padding exactly parity-preserving). ``w_init`` warm-starts
    the FISTA iterate (standardized scale). Returns the standardized coefficients too, so a
    caller can reuse them as a warm start: ``(w_std (B,P), coef_raw (B,P),
    intercept_raw (B,), is_intercept_only (B,))``.
    """
    B, P, _ = G.shape
    ninv = 1.0 / n
    mean = zsum * ninv[:, None]
    var = zsqsum * ninv[:, None] - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    scale = np.where(std < _STD_EPS, 1.0, std)
    ybar = ysum * ninv
    inv_scale = 1.0 / scale
    ms = mean * inv_scale
    G_std = G * inv_scale[:, :, None] * inv_scale[:, None, :]
    G_std = G_std - n[:, None, None] * ms[:, :, None] * ms[:, None, :]
    outer = mask[:, :, None] * mask[:, None, :]
    G_std = G_std * outer
    diag = np.arange(P)
    G_std[:, diag, diag] += 1.0 - mask  # padded rows → identity (real rows: mask=1 → +0)
    q_std = (c - n[:, None] * mean * ybar[:, None]) * inv_scale
    q_std = q_std * mask

    if l1_ratio == 0.0:
        w_std = _batched_ridge(G_std, q_std, n, alpha)
    else:
        w_std, _n_iters = _batched_fista(G_std, q_std, n, alpha, l1_ratio, mask, w_init=w_init)
    w_std = w_std * mask

    coef_raw = (w_std * inv_scale) * mask
    intercept_raw = ybar - (w_std * (mean * inv_scale)).sum(axis=1)
    is_io = np.abs(w_std).max(axis=1) <= _INTERCEPT_ONLY_ATOL
    return w_std, coef_raw, intercept_raw, is_io


def _batched_ridge(
    G_std: np.ndarray, q_std: np.ndarray, n: np.ndarray, alpha: float
) -> np.ndarray:
    """Batched closed-form ridge ``w = (G_std + α·n·I)⁻¹ q_std`` per batch element.

    Numpy analogue of :func:`ridge_gram_fit` / ``GpuBackend._batched_ridge``. Uses a batched
    LU solve — numpy has no batched ``cholesky_solve``, and ``G_std + α·n·I`` is SPD for
    ``α>0`` so LU is exact. The RHS is passed as ``(B, P, 1)`` (numpy≥2.0 rejects a ``(B, P)``
    stack-of-vectors RHS).
    """
    B, P, _ = G_std.shape
    A = G_std.copy()
    diag = np.arange(P)
    A[:, diag, diag] += alpha * n[:, None]
    return np.linalg.solve(A, q_std[:, :, None])[:, :, 0]


def _batched_fista(
    G_std: np.ndarray,
    q_std: np.ndarray,
    n: np.ndarray,
    alpha: float,
    l1_ratio: float,
    mask: np.ndarray,
    w_init: Optional[np.ndarray] = None,
):
    """Batched masked FISTA (accelerated proximal gradient) for the ElasticNet L1 case.

    Numpy float64 port of ``GpuBackend._batched_fista`` (gpu_backend.py:666). Minimizes, per
    batch element, the standardized ElasticNet objective (same convention as the sklearn Gram
    CD that :func:`enet_gram_fit` calls)::

        ½·wᵀG_std·w − q_stdᵀw + l1·‖w‖₁ + ½·l2·‖w‖²,
        l1 = α·l1_ratio·n,   l2 = α·(1−l1_ratio)·n

    The smooth part has gradient ``H·w − q_std`` with ``H = G_std + l2·I``; the L1 prox is a
    soft-threshold at ``l1·step``. FISTA (not coordinate descent) is used because it is pure
    matmul/elementwise and so vectorizes across the target batch — coordinate descent's
    sequential coordinate sweep does not. This is a *different optimizer* than the per-target
    sklearn CD, so it matches within statistical parity (~1e-3 coef / ~5e-3 CV metrics), which
    is exactly what the GPU FISTA already satisfies against the same oracle.

    ``mask`` (B, P) zeroes padded coordinates every step so they stay 0 and decouple from the
    real block. Convergence is checked every iteration (numpy has no device→host sync to
    amortize, unlike the GPU's ``check_every``), so it early-stops as tightly as possible.

    ``w_init`` (B, P), if given, warm-starts the iterate (masked): the K leave-one-fold-out
    fold models are ~1/K perturbations of the full-data solution, so seeding each from the
    final model's coefficients cuts iterations sharply. FISTA is convex, so the converged
    optimum is unchanged; only the iteration count differs. Returns ``(w, n_iters)``.
    """
    B, P = q_std.shape
    l1_reg = alpha * l1_ratio * n            # (B,)
    l2_reg = alpha * (1.0 - l1_ratio) * n    # (B,)
    H = G_std.copy()
    diag = np.arange(P)
    H[:, diag, diag] += l2_reg[:, None]
    lip = _power_lipschitz(H, mask) * _LIPSCHITZ_MARGIN  # (B,)
    step = 1.0 / np.maximum(lip, 1e-12)                   # (B,)
    thr = (l1_reg * step)[:, None]                        # (B, 1) soft-threshold level
    w = np.zeros((B, P), dtype=np.float64) if w_init is None else (w_init * mask)
    w_prev = w.copy()
    tk = 1.0
    n_iters = 0
    for _ in range(_FISTA_MAX_ITER):
        n_iters += 1
        tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        mom = (tk - 1.0) / tk_new
        v = (w + mom * (w - w_prev)) * mask
        grad = np.einsum("bij,bj->bi", H, v) - q_std
        zt = v - step[:, None] * grad
        w_new = np.sign(zt) * np.maximum(np.abs(zt) - thr, 0.0)
        w_new = w_new * mask
        converged = np.max(np.abs(w_new - w)) < _FISTA_TOL
        w_prev, w, tk = w, w_new, tk_new
        if converged:
            break
    return w, n_iters


def _power_lipschitz(H: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-batch largest-eigenvalue (Lipschitz) estimate of ``H`` via power iteration.

    Numpy port of ``GpuBackend._power_lipschitz`` (gpu_backend.py:698). ``mask`` restricts the
    iterate to real coordinates so the padded rows (identity) never dominate the estimate.
    Returns a (B,) Rayleigh-quotient value — an *under*-estimate of ``λ_max`` in general, which
    is why callers scale it up by :data:`_LIPSCHITZ_MARGIN` before taking ``step = 1/Lip``.
    """
    v = mask.copy()
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    for _ in range(_POWER_ITERS):
        hv = np.einsum("bij,bj->bi", H, v) * mask
        v = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-12)
    hv = np.einsum("bij,bj->bi", H, v) * mask
    return (v * hv).sum(axis=1)
