"""Streaming sufficient-statistics driver for imputation training (Phase 2).

Instead of materializing the whole reference dosage matrix (≈4 TB at 2M×500K),
this streams the panel one chromosome at a time via a Phase-1 ``GenotypeSource``
and maintains a **sliding ±2·W window buffer** of columns. Each missing-variant
model is fit the moment its ±W predictor window closes, from a **local Gram block**
gathered out of an incrementally-maintained band Gram over buffered chip columns
(never recomputing ZᵀZ over the sample axis per target — that is O(n·p²)/target and
does not finish at scale). The per-fold out-of-fold predictions are reduced
straight into shared length-n ``s_true``/``s_cv`` calibration accumulators and the
per-fold models discarded, so the per-variant ``cv_predictions`` dict (the ~8 EB
calibration blocker) never exists.

Samples are reordered once into contiguous CV-fold blocks (``GlobalFolds``) so all
per-fold statistics and OOF scoring are cheap contiguous slices.

**Parity / deviation.** On a panel with no missing dosages (the dense 1000G
reference), with a pinned ``random_state``, this reproduces the legacy per-variant
fit within statistical-parity tolerance (exported coefficients ~1e-12; per-fold CV
metrics/calibration within a small band from L1-kink convergence). Predictors/targets
are mean-imputed (a shared Gram cannot listwise-delete per-variant-varying rows); on
missing panels that is a documented, quality-validated deviation, not bit-parity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from imputed_prs.compute.device import resolve_streaming_backend
from imputed_prs.compute.gram_solve import (
    LocalGramBlock,
    fit_from_local_gram,
    fit_reference_folds,
    solve_blocks_batched,
)
from imputed_prs.core.harmonizer import normalize_chromosome_array
from imputed_prs.core.types import ImputedVariantModel
from imputed_prs.core.window_index import ChromosomeIndex
from imputed_prs.models.trainer import compute_residual_variance


# ---------------------------------------------------------------------------
# Deterministic CV folds as a sample reordering (contiguous fold blocks).
# ---------------------------------------------------------------------------
class GlobalFolds:
    """Deterministic global CV fold assignment, materialized as a row permutation.

    Reproduces ``KFold(n_splits, shuffle=True, random_state).split(arange(n))``: the
    per-fold *held-out* index sets are concatenated into ``perm`` so that, in permuted
    order, fold ``k`` is the contiguous block ``[bounds[k], bounds[k+1])``. On a panel
    with no missingness this is exactly the legacy per-variant fold membership (every
    variant has the full sample set), giving tight parity.
    """

    def __init__(self, n_samples: int, cv_folds: int, random_state: Optional[int]):
        self.n = n_samples
        self.n_folds = cv_folds
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        parts: List[np.ndarray] = []
        bounds = [0]
        for _train, val in kf.split(np.arange(n_samples)):
            parts.append(val)
            bounds.append(bounds[-1] + len(val))
        self.perm = np.concatenate(parts).astype(np.int64)
        self.bounds = np.asarray(bounds, dtype=np.int64)

    @classmethod
    def from_partition(cls, fold_indices: Sequence[np.ndarray]) -> "GlobalFolds":
        """Build folds from an explicit disjoint sample partition, bypassing KFold.

        ``fold_indices[k]`` is the natural-order sample-row index set held out in fold
        ``k`` (the reference-CV outer chunks from
        ``ImputationEvaluator.cross_validate``). Concatenated they **must** be a
        complete, disjoint partition of ``range(n)`` — otherwise the additive
        ``S_full − S_fold(k)`` subtraction silently corrupts every training Gram (R5),
        so this validates the partition and raises on any gap/overlap.
        """
        parts = [np.asarray(f, dtype=np.int64).ravel() for f in fold_indices]
        perm = (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        )
        n = int(perm.size)
        if not np.array_equal(np.sort(perm), np.arange(n, dtype=np.int64)):
            raise ValueError(
                "GlobalFolds.from_partition requires a complete, disjoint partition "
                "of range(n); the concatenated fold indices are not a permutation of "
                "[0, n)."
            )
        obj = cls.__new__(cls)  # bypass __init__ (no KFold — partition is given)
        obj.n = n
        obj.n_folds = len(parts)
        obj.perm = perm
        bounds = [0]
        for part in parts:
            bounds.append(bounds[-1] + len(part))
        obj.bounds = np.asarray(bounds, dtype=np.int64)
        return obj

    def permute(self, rows_natural: np.ndarray) -> np.ndarray:
        """Reorder a (n,) or (n, m) array from natural into fold-block order."""
        return rows_natural[self.perm]

    def fold_slice(self, k: int) -> slice:
        return slice(int(self.bounds[k]), int(self.bounds[k + 1]))


# ---------------------------------------------------------------------------
# Harmonized stream plan (built by the orchestrator; see backend seam).
# ---------------------------------------------------------------------------
@dataclass
class TargetVar:
    """A missing PRS variant to impute (and contribute to calibration)."""

    prs_variant_id: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: Optional[str]
    beta: float
    effect_flip: bool  # True ⇒ effect allele is the reference REF ⇒ dosage = 2 − ALT


@dataclass
class ObservedVar:
    """A PRS variant that is on the platform (observed) — a calibration term."""

    beta: float
    effect_flip: bool


@dataclass
class StreamPlan:
    """Everything the streaming imputation fitter needs, keyed by source variant_id."""

    sample_ids: List[str]
    platform_variant_info: pd.DataFrame  # variant_id, chromosome, position, ref/alt
    chip_ids: Dict[str, int]  # source variant_id -> platform row index (predictors)
    targets: Dict[str, TargetVar]  # source variant_id -> target spec
    observed: Dict[str, ObservedVar]  # source variant_id -> observed calibration term
    window_size: int = 1_000_000
    max_predictors: Optional[int] = None
    alpha: float = 0.01
    l1_ratio: float = 0.5
    cv_folds: int = 5
    random_state: Optional[int] = None
    # Observed-variant fallback models (P1.8): each observed PRS variant that
    # resolves to a reference row is trained the same way as a missing target
    # (effect-oriented target, local-window platform predictors with its own locus
    # auto-excluded) so it can be recovered when a user's upload cannot call it.
    # Keyed by source (reference) variant_id, disjoint from ``targets``. Fallback
    # models do NOT contribute to the calibration accumulators (the observed term
    # already does, exactly, via ``observed``).
    fallback_targets: Dict[str, TargetVar] = field(default_factory=dict)
    # PRS-side bookkeeping for the orchestrator's dispositions (PRS variant_ids).
    observed_prs_ids: set = field(default_factory=set)  # kept-observed PRS ids
    fallback_no_target_ids: set = field(default_factory=set)  # observed, no ref row


@dataclass
class StreamingFitResult:
    """Output of the streaming fit: models + the two calibration accumulators."""

    models: Dict[str, ImputedVariantModel]
    s_true: np.ndarray  # (n,) in permuted order; order-independent for calibration
    s_cv: np.ndarray  # (n,)
    n_trained: int = 0
    n_intercept_only: int = 0
    n_failed: int = 0
    failures: Dict[str, str] = field(default_factory=dict)
    # Observed-variant fallback models keyed by PRS variant_id (see StreamPlan).
    fallback_models: Dict[str, ImputedVariantModel] = field(default_factory=dict)

    @classmethod
    def reduce(cls, partials: Sequence["_ImputeChromPartial"], n: int) -> "StreamingFitResult":
        """Order-independent merge of per-chromosome partials (Phase 7 fan-out).

        Chromosome shards are key-disjoint (a target belongs to exactly one chromosome),
        so ``models``/``fallback_models``/``failures`` merge by dict-union (asserted
        disjoint — a spuriously shared id is loud, not silent). ``s_true``/``s_cv`` are
        additive over units; summed in canonical ``_chrom_sort_key`` order they are
        bit-identical to the pre-Phase-7 in-place running sum (same order, same terms).
        """
        ordered = sorted(partials, key=lambda p: _chrom_sort_key(p.chrom))
        models: Dict[str, ImputedVariantModel] = {}
        fallback_models: Dict[str, ImputedVariantModel] = {}
        failures: Dict[str, str] = {}
        s_true = np.zeros(n, dtype=np.float64)
        s_cv = np.zeros(n, dtype=np.float64)
        n_io = 0
        for p in ordered:
            assert models.keys().isdisjoint(p.models), (
                "chromosome shards must have disjoint target ids"
            )
            models.update(p.models)
            fallback_models.update(p.fallback_models)
            failures.update(p.failures)
            s_true += p.s_true
            s_cv += p.s_cv
            n_io += p.n_intercept_only
        return cls(
            models=models,
            s_true=s_true,
            s_cv=s_cv,
            n_trained=len(models),
            n_intercept_only=n_io,
            n_failed=len(failures),
            failures=failures,
            fallback_models=fallback_models,
        )


@dataclass
class _ImputeChromPartial:
    """One chromosome's contribution to a streaming fit (Phase 7 shard unit).

    ``s_true``/``s_cv`` are this chromosome's terms only (zero elsewhere); the parent
    sums them across shards. ``cv_collector`` is populated only in reference-CV mode
    (``{fold_k -> {prs_id -> model}}``); ``None`` for a plain fit.
    """

    chrom: str
    models: Dict[str, ImputedVariantModel]
    fallback_models: Dict[str, ImputedVariantModel]
    failures: Dict[str, str]
    s_true: np.ndarray
    s_cv: np.ndarray
    n_intercept_only: int
    cv_collector: Optional[Dict[int, Dict[str, ImputedVariantModel]]] = None


def reduce_cv_collectors(partials: Sequence[object], n_folds: int):
    """Merge per-chromosome reference-CV collectors → ``(fold_models, failures)``.

    Shared by the imputation and projection reference-CV fan-out. Per-outer-fold model
    dicts are key-disjoint across chromosomes, so the merge is a per-fold dict-union in
    canonical chromosome order; returns ``{k: list(models)}`` matching the serial path.
    """
    ordered = sorted(partials, key=lambda p: _chrom_sort_key(p.chrom))
    merged = {k: {} for k in range(n_folds)}
    failures: Dict[str, str] = {}
    for p in ordered:
        for k in range(n_folds):
            merged[k].update(p.cv_collector[k])
        failures.update(p.failures)
    return {k: list(merged[k].values()) for k in range(n_folds)}, failures


# ---------------------------------------------------------------------------
# Incrementally-maintained band Gram over buffered chip (predictor) columns.
# ---------------------------------------------------------------------------
class _ChipGramBuffer:
    """Sliding buffer of chip columns with an incremental full + per-fold Gram.

    Columns are appended in ascending position and evicted from the front (FIFO),
    so an evict is a prefix compaction. ``gather`` returns a target's local Gram
    sub-block by pure indexing — no sample-axis matmul per target.
    """

    def __init__(
        self,
        n_samples: int,
        folds: GlobalFolds,
        capacity: int = 256,
        lazy_fold_gram: bool = False,
    ):
        self.n = n_samples
        self.folds = folds
        self.K = folds.n_folds
        self.cap = capacity
        self.m = 0
        # ``lazy_fold_gram`` (projection): do NOT maintain the O(cap²) full Gram or the
        # O(K·cap²) per-fold Gram incrementally. Both are recomputed on-demand at ``gather``
        # from the resident band ``Z`` over each unit's ≤max_predictors fit predictors, so a
        # chromosome-spanning merged region (cap ~ thousands) never allocates a (K, cap, cap)
        # tensor. Imputation keeps the incremental path (many small units ⇒ per-unit recompute
        # would dominate); this is the Finding-#1 band-limited per-fold Gram fix.
        self.lazy_fold_gram = lazy_fold_gram
        self.Z = np.zeros((n_samples, capacity), dtype=np.float64)
        self.Gfull = None if lazy_fold_gram else np.zeros((capacity, capacity), dtype=np.float64)
        self.Ghold = (
            None if lazy_fold_gram else np.zeros((self.K, capacity, capacity), dtype=np.float64)
        )
        self.zsum = np.zeros(capacity, dtype=np.float64)
        self.zsqsum = np.zeros(capacity, dtype=np.float64)
        self.zsum_h = np.zeros((self.K, capacity), dtype=np.float64)
        self.zsqsum_h = np.zeros((self.K, capacity), dtype=np.float64)
        self.pos = np.zeros(capacity, dtype=np.int64)
        self.af = np.zeros(capacity, dtype=np.float64)
        self.pidx = np.full(capacity, -1, dtype=np.int64)
        self.slot_of: Dict[int, int] = {}
        self._karange = np.arange(self.K)  # fold axis for the (K, p, p) gather

    def _grow(self) -> None:
        new_cap = self.cap * 2
        # NB: np.resize reflows a 2D array row-major (corrupting columns); allocate
        # and slice-copy so the existing buffered columns stay column-aligned.
        Znew = np.zeros((self.n, new_cap), dtype=np.float64)
        Znew[:, : self.cap] = self.Z
        self.Z = Znew
        if not self.lazy_fold_gram:  # else the Grams are recomputed on-demand at gather
            G = np.zeros((new_cap, new_cap))
            G[: self.cap, : self.cap] = self.Gfull
            self.Gfull = G
            Gh = np.zeros((self.K, new_cap, new_cap))
            Gh[:, : self.cap, : self.cap] = self.Ghold
            self.Ghold = Gh
        for name in ("zsum", "zsqsum", "pos", "af", "pidx"):
            arr = getattr(self, name)
            new = np.zeros(new_cap, dtype=arr.dtype)
            new[: self.cap] = arr
            setattr(self, name, new)
        for name in ("zsum_h", "zsqsum_h"):
            arr = getattr(self, name)
            new = np.zeros((self.K, new_cap))
            new[:, : self.cap] = arr
            setattr(self, name, new)
        self.cap = new_cap

    def add(self, col_perm: np.ndarray, platform_idx: int, position: int, af: float) -> None:
        """Append one chip column (already permuted + mean-imputed) to the band Gram."""
        if self.m >= self.cap:
            self._grow()
        s = self.m
        m = self.m
        if not self.lazy_fold_gram:  # incremental band Gram (imputation); else recomputed at gather
            Za = self.Z[:, :m]
            dots = Za.T @ col_perm  # (m,)
            self.Gfull[:m, s] = dots
            self.Gfull[s, :m] = dots
            self.Gfull[s, s] = float(col_perm @ col_perm)
        for k in range(self.K):
            sl = self.folds.fold_slice(k)
            ck = col_perm[sl]
            if not self.lazy_fold_gram:
                dk = self.Z[sl, :m].T @ ck
                self.Ghold[k, :m, s] = dk
                self.Ghold[k, s, :m] = dk
                self.Ghold[k, s, s] = float(ck @ ck)
            self.zsum_h[k, s] = float(ck.sum())
            self.zsqsum_h[k, s] = float(ck @ ck)
        self.Z[:, s] = col_perm
        self.zsum[s] = float(col_perm.sum())
        self.zsqsum[s] = float(col_perm @ col_perm)
        self.pos[s] = position
        self.af[s] = af
        self.pidx[s] = platform_idx
        self.slot_of[platform_idx] = s
        self.m += 1

    def add_batch(self, cols, platform_indices, positions, afs) -> None:
        """Append a group of chip columns at once (batched-accumulation seam).

        The CPU buffer just loops ``add`` so it stays **bit-for-bit** the per-column path
        (the golden oracle); the GPU buffer overrides this to fold the whole group's band-Gram
        update into GEMMs (``Zᵀ·NewCols``), which is where the on-device accumulation actually
        wins at scale. The streaming driver calls this once per stream block.
        """
        for col, pidx, pos, af in zip(cols, platform_indices, positions, afs):
            self.add(col, pidx, pos, af)

    def evict_below(self, min_pos: int) -> None:
        """Drop leading (oldest, lowest-position) columns with ``pos < min_pos``."""
        m = self.m
        if m == 0:
            return
        c = int(np.searchsorted(self.pos[:m], min_pos, side="left"))
        if c == 0:
            return
        keep = m - c
        # Overlapping in-place shifts: copy RHS first.
        self.Z[:, :keep] = self.Z[:, c:m].copy()
        if not self.lazy_fold_gram:
            self.Gfull[:keep, :keep] = self.Gfull[c:m, c:m].copy()
            self.Ghold[:, :keep, :keep] = self.Ghold[:, c:m, c:m].copy()
        for name in ("zsum", "zsqsum", "pos", "af", "pidx"):
            arr = getattr(self, name)
            arr[:keep] = arr[c:m].copy()
        for name in ("zsum_h", "zsqsum_h"):
            arr = getattr(self, name)
            arr[:, :keep] = arr[:, c:m].copy()
        self.m = keep
        self.slot_of = {int(self.pidx[i]): i for i in range(keep)}

    def clear(self) -> None:
        self.m = 0
        self.slot_of = {}

    def gather(self, platform_indices: Sequence[int]):
        """Return ``(idx, block)`` for a target's predictors, in the given order.

        ``idx`` are the buffer slots of the predictors; ``block`` holds the local Gram
        pieces gathered by pure indexing. Advanced (fancy) indexing already returns a
        **fresh** array independent of the buffer, so no explicit ``.copy()`` is needed
        — a later ``add``/``evict`` cannot alias into a gathered block. The K per-fold
        held-out Gram sub-blocks are gathered in one vectorised ``(K, p, p)`` op and the
        per-fold moment vectors as ``(K, p)``.

        Crucially, the cross-products ``Zᵀy`` and out-of-fold predictions are **not**
        gathered here — they are computed once per *batch* of co-windowed targets as
        banded BLAS-3 GEMMs against the contiguous ``Z[:, :m]`` band (see
        ``_fit_chunk``), so no per-target ``Z`` copy is ever made. ``idx`` lets the
        caller slice the batched cross-product ``C[idx, j]`` for this target.
        """
        idx = np.fromiter(
            (self.slot_of[int(p)] for p in platform_indices), dtype=np.int64,
            count=len(platform_indices),
        )
        if self.lazy_fold_gram:
            # Recompute the (p, p) full + (K, p, p) per-fold Gram on-demand over just this
            # unit's fit predictors, from the resident band ``Z`` — never a (K, cap, cap)
            # tensor. Both come from the same ``Zsub`` slice, so ``G == Σ_k fold_G[k]`` and
            # ``G − fold_G[k]`` is an exact held-in training Gram (Finding-#1 fix).
            Zsub = self.Z[:, idx]  # (n, p) fresh copy, released after the solve
            p = idx.shape[0]
            fold_G = np.empty((self.K, p, p), dtype=np.float64)
            for k in range(self.K):
                Zk = Zsub[self.folds.fold_slice(k)]  # (n_k, p) view
                fold_G[k] = Zk.T @ Zk
            return idx, {
                "G": Zsub.T @ Zsub,
                "fold_G": fold_G,
                "zsum": self.zsum[idx],
                "zsqsum": self.zsqsum[idx],
                "fold_zsum": self.zsum_h[:, idx],
                "fold_zsqsum": self.zsqsum_h[:, idx],
                "af": self.af[idx],
            }
        ix = np.ix_(idx, idx)
        return idx, {
            "G": self.Gfull[ix],
            # (K, p, p) gathered directly via a 3-array np.ix_ — avoids the large
            # (K, p, cap) intermediate that ``Ghold[:, idx][:, :, idx]`` would build.
            "fold_G": self.Ghold[np.ix_(self._karange, idx, idx)],
            "zsum": self.zsum[idx],
            "zsqsum": self.zsqsum[idx],
            "fold_zsum": self.zsum_h[:, idx],  # (K, p)
            "fold_zsqsum": self.zsqsum_h[:, idx],  # (K, p)
            "af": self.af[idx],
        }


# ---------------------------------------------------------------------------
# Column preparation: orient + mean-impute + permute (all-samples-per-block exact).
# ---------------------------------------------------------------------------
def _prepare_column(raw: np.ndarray, flip: bool, folds: GlobalFolds):
    """Orient (2−d if flip), compute AF from non-NaN, mean-impute, permute rows.

    Returns ``(col_perm float64, af, n_nonnan)``. Because a variant-block holds every
    sample, the column mean is exact; flip commutes with mean-imputation.
    """
    oriented = (2.0 - raw) if flip else raw.astype(np.float64, copy=True)
    mask = np.isnan(oriented)
    n_nonnan = int(oriented.size - mask.sum())
    if n_nonnan > 0:
        col_mean = float(np.nansum(oriented) / n_nonnan)
    else:
        col_mean = 0.0
    if mask.any():
        oriented = oriented.copy()
        oriented[mask] = col_mean
    af = col_mean / 2.0
    return folds.permute(oriented), af, n_nonnan


@dataclass
class _OpenTarget:
    source_id: str
    spec: TargetVar
    col: np.ndarray  # (n,) permuted, effect-oriented, mean-imputed target dosage
    af: float
    is_fallback: bool = False  # observed-variant fallback (no calibration reduction)


# ---------------------------------------------------------------------------
# Shared batched-Gram fit kernel (imputation targets AND projection regions).
# ---------------------------------------------------------------------------
@dataclass
class _FitJob:
    """One unit to fit from the shared band buffer: an imputation target or a
    projection region. ``col`` is the (permuted) target — a single effect-oriented
    dosage for imputation, or the region contribution S_R = Σ β_j x_eff_j for
    projection. ``pred_idx`` are its predictor platform indices (legacy order).

    ``calib_coef`` weights this unit's contribution to the calibration accumulators
    (β for imputation, 1.0 for projection since S_R already carries the betas);
    ``is_calibrating`` is False for observed-variant fallbacks (train a model but do
    not touch calibration). ``store``/``fail`` route the fitted model / failure back
    to the method-specific collections.
    """

    col: np.ndarray
    pred_idx: np.ndarray
    calib_coef: float
    is_calibrating: bool
    store: object  # callable(result, pred_idx, pred_af) -> None
    fail: object  # callable(exc) -> None


# --- Batched-solve selection (Phase 8) ------------------------------------------------
# The streaming ElasticNet solve switches from the per-target sklearn coordinate descent
# (the exact oracle) to the batched FISTA/ridge solve (gram_solve.solve_blocks_batched)
# once the panel is large enough that per-target Python-call overhead (millions of fits)
# dominates. The batched path is a *different optimizer* (statistical parity ~1e-3 coef /
# ~5e-3 CV metrics), so it is gated by **sample count** rather than chunk width: at 500K
# samples the (n×T) cross-product bound shrinks a chunk to ~64 targets, so a target-count
# threshold would never fire, whereas every test / verification panel (n ≤ 2504, incl.
# 1000G) stays comfortably on the per-target oracle. IMPUTED_PRS_SOLVE=per_target|batched
# forces one path (tests pin the exact oracle, or force the batched path independent of n).
_BATCH_MIN_SAMPLES = 10_000

# The override is resolved once in the parent (``_resolve_solve_mode``) and carried on the
# fitter, then re-published into this module global at each chromosome task's entry
# (``_set_active_solve_mode`` in ``_run_one_chromosome``) so the choice is per-fit and
# deterministic across the process pool. Reading the env *inside* a worker would be wrong: a
# loky pool is reused across fits, so a worker could see a stale prior override (auto is
# env-free — ``n_samples ≥ _BATCH_MIN_SAMPLES`` — so only the test/debug overrides could leak).
_ACTIVE_SOLVE_MODE = "auto"


def _resolve_solve_mode() -> str:
    """Read the solver-selection override from the env. Call in the PARENT at fit setup."""
    return os.environ.get("IMPUTED_PRS_SOLVE", "auto")


def _set_active_solve_mode(mode: str) -> None:
    """Publish the fitter's resolved mode into the module global (call at task entry)."""
    global _ACTIVE_SOLVE_MODE
    _ACTIVE_SOLVE_MODE = mode


def _use_batched_solve(n_samples: int, n_units: int, alpha: float) -> bool:
    """Whether a chunk of ``n_units`` co-located units should use the batched solve.

    Requires ``alpha > 0`` (the batched solve is all-or-nothing on a singular Gram and has
    no per-item lstsq fallback; ``alpha > 0`` makes every ``G_std + …·I`` SPD). In ``auto``
    mode the batched path engages for large panels (``n_samples ≥ _BATCH_MIN_SAMPLES``);
    ``IMPUTED_PRS_SOLVE`` overrides for pinning.
    """
    if alpha <= 0.0 or n_units == 0:
        return False
    if _ACTIVE_SOLVE_MODE == "batched":
        return True
    if _ACTIVE_SOLVE_MODE == "per_target":
        return False
    return n_samples >= _BATCH_MIN_SAMPLES


def _run_fit_batch(jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv, batch_cap):
    """Fit a group of co-located units, chunked to bound the (n×T) working arrays."""
    n_io = 0
    for s in range(0, len(jobs), batch_cap):
        n_io += _run_fit_chunk(
            jobs[s : s + batch_cap], buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv
        )
    return n_io


def _batch_cross_products(jobs, buf, folds):
    """Shared banded cross-products for a chunk of co-located units.

    The per-unit O(n·p) cross-products (``Zᵀy``) — the cost that dominates at 500K
    samples — are lifted to banded BLAS-3 GEMMs over the contiguous ``Z[:, :m]`` band
    shared by every unit in the chunk (one GEMM for the full ``Zᵀy`` + one per fold
    for the held-out ``Zᵀy``). Returns the dict of stacked targets + full/per-fold
    moments consumed by :func:`_assemble_block` (used by both the single-model fit and
    the reference-CV fit).
    """
    n, K, T = folds.n, folds.n_folds, len(jobs)
    m = buf.m
    Zband = buf.Z[:, :m]  # (n, m) contiguous-row band view — never copied

    Y = np.empty((n, T), dtype=np.float64)
    for j, job in enumerate(jobs):
        Y[:, j] = job.col
    ysum_all = Y.sum(axis=0)
    ysqsum_all = np.einsum("ij,ij->j", Y, Y)
    C = Zband.T @ Y  # (m, T) full cross-product
    Ck: List[np.ndarray] = []
    fold_ysum = np.empty((K, T), dtype=np.float64)
    fold_ysqsum = np.empty((K, T), dtype=np.float64)
    fold_n = np.empty(K, dtype=np.int64)
    for k in range(K):
        sl = folds.fold_slice(k)
        Yk = Y[sl]
        fold_ysum[k] = Yk.sum(axis=0)
        fold_ysqsum[k] = np.einsum("ij,ij->j", Yk, Yk)
        fold_n[k] = sl.stop - sl.start
        Ck.append(Zband[sl].T @ Yk)  # (m, T) held-out cross-product
    return {
        "n": n, "K": K, "T": T, "m": m, "Zband": Zband, "Y": Y,
        "ysum_all": ysum_all, "ysqsum_all": ysqsum_all, "C": C, "Ck": Ck,
        "fold_ysum": fold_ysum, "fold_ysqsum": fold_ysqsum, "fold_n": fold_n,
    }


def _assemble_block(job, j, buf, cross):
    """Assemble unit ``j``'s :class:`LocalGramBlock` from the shared cross-products
    (+ its gathered predictor Gram). Returns ``(block, idx, pred_af)``; ``idx`` is
    ``None`` for a no-predictor unit. ``buf.gather`` may raise on a missing slot — the
    caller wraps this (record-don't-crash) and routes it to ``job.fail``.

    ``fold_ysum``/``fold_ysqsum``/``fold_n`` are populated even for a no-predictor unit
    (they are needed by :func:`fit_reference_folds` for per-fold intercept-only models;
    the single-model path early-returns on ``p==0`` before touching them, so this is
    behaviour-preserving there).
    """
    n, K = cross["n"], cross["K"]
    C, Ck = cross["C"], cross["Ck"]
    ysum_all, ysqsum_all = cross["ysum_all"], cross["ysqsum_all"]
    fold_ysum, fold_ysqsum, fold_n = (
        cross["fold_ysum"], cross["fold_ysqsum"], cross["fold_n"],
    )
    pred_idx = job.pred_idx
    if len(pred_idx):
        idx, gg = buf.gather(pred_idx)
        block = LocalGramBlock(
            n=n, G=gg["G"], c=C[idx, j], zsum=gg["zsum"],
            zsqsum=gg["zsqsum"], ysum=float(ysum_all[j]),
            ysqsum=float(ysqsum_all[j]), fold_G=gg["fold_G"],
            fold_c=[Ck[k][idx, j] for k in range(K)],
            fold_zsum=gg["fold_zsum"], fold_zsqsum=gg["fold_zsqsum"],
            fold_ysum=fold_ysum[:, j], fold_ysqsum=fold_ysqsum[:, j],
            fold_n=fold_n,
        )
        pred_af = gg["af"]
    else:
        idx = None
        block = LocalGramBlock(
            n=n, G=np.empty((0, 0)), c=np.empty(0), zsum=np.empty(0),
            zsqsum=np.empty(0), ysum=float(ysum_all[j]),
            ysqsum=float(ysqsum_all[j]),
            fold_ysum=fold_ysum[:, j], fold_ysqsum=fold_ysqsum[:, j],
            fold_n=fold_n,
        )
        pred_af = np.empty(0)
    return block, idx, pred_af


def _run_fit_chunk(jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv):
    """Batched banded-Gram fit of one chunk of co-located units.

    Each unit's Gram sub-block (O(p²)) and ElasticNet solve stay per-unit (cheap,
    sample-free); the shared cross-products come from :func:`_batch_cross_products`.
    This is also the natural GPU seam (Phase 3): the banded GEMMs map onto batched
    cuBLAS. ``s_true``/``s_cv`` are mutated in place. Returns #calibrating
    intercept-only models.
    """
    cross = _batch_cross_products(jobs, buf, folds)
    n, K, T, m, Zband, Y = (
        cross["n"], cross["K"], cross["T"], cross["m"], cross["Zband"], cross["Y"],
    )

    # Solve each unit, then scatter each fold model's raw coefficients into the band slots
    # so the OOF becomes a single GEMM per fold below.
    Wk = [np.zeros((m, T), dtype=np.float64) for _ in range(K)]
    bk = np.zeros((K, T), dtype=np.float64)
    coef_vec = np.zeros(T, dtype=np.float64)  # nonzero ⇒ contributes to calibration
    n_io = 0

    # Assemble each unit's Gram block (record-don't-crash on a missing-slot gather).
    assembled = []  # (j, job, block, idx, pred_af)
    for j, job in enumerate(jobs):
        try:
            block, idx, pred_af = _assemble_block(job, j, buf, cross)
        except Exception as exc:  # noqa: BLE001 - mirror legacy: record, don't crash
            job.fail(exc)
            continue
        assembled.append((j, job, block, idx, pred_af))

    # Solve — batched (many targets against the shared band) at scale, else the per-target
    # sklearn oracle. The batched solve is all-or-nothing, so a rare failure falls back to
    # the resilient per-target path (which records per-unit failures without crashing).
    results = None
    if _use_batched_solve(n, len(assembled), alpha):
        try:
            results = solve_blocks_batched(
                [a[2] for a in assembled], alpha, l1_ratio, cv_folds
            )
        except Exception:  # noqa: BLE001 - degrade to the per-target path
            results = None
    if results is None:
        results = []
        for a in assembled:
            try:
                results.append(
                    fit_from_local_gram(
                        a[2], alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv_folds
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record, don't crash
                a[1].fail(exc)
                results.append(None)

    for (j, job, block, idx, pred_af), result in zip(assembled, results):
        if result is None:
            continue
        try:
            job.store(result, job.pred_idx, pred_af)
            if job.is_calibrating:
                if result.fold_models:
                    for k, fm in enumerate(result.fold_models):
                        Wk[k][idx, j] = fm.coef
                        bk[k, j] = fm.intercept
                else:  # intercept-only ⇒ constant OOF = intercept on every sample
                    bk[:, j] = result.intercept
                coef_vec[j] = job.calib_coef
                if result.is_intercept_only:
                    n_io += 1
        except Exception as exc:  # noqa: BLE001 - mirror legacy: record, don't crash
            job.fail(exc)

    # Batched OOF (one GEMM per fold over the shared band) + calibration reduction.
    # The zero-padded band slots contribute exact 0.0, so this equals the old
    # per-unit ``Zpred[sl] @ coef`` reduction to BLAS-blocking precision (~1e-13,
    # within the sanctioned statistical-parity band). Weight = β (imputation) or 1.0
    # (projection: S_R already carries the betas).
    if coef_vec.any():
        oof = np.empty((n, T), dtype=np.float64)
        for k in range(K):
            sl = folds.fold_slice(k)
            oof[sl] = Zband[sl] @ Wk[k] + bk[k]
        s_true += Y @ coef_vec
        s_cv += oof @ coef_vec
    return n_io


def _run_cv_batch(jobs, buf, folds, alpha, l1_ratio, cv_folds, batch_cap):
    """Reference-CV analogue of :func:`_run_fit_batch`, chunked to bound working arrays."""
    for s in range(0, len(jobs), batch_cap):
        _run_cv_chunk(jobs[s : s + batch_cap], buf, folds, alpha, l1_ratio, cv_folds)


def _run_cv_chunk(jobs, buf, folds, alpha, l1_ratio, cv_folds):
    """Leave-one-fold-out reference-CV fit of one chunk (host/CPU only).

    Shares :func:`_batch_cross_products`/:func:`_assemble_block` with the single-model
    path, but ``folds`` here are the **reference-CV outer folds**, so each unit's
    per-fold slabs are the outer-fold statistics. Per unit it calls
    :func:`fit_reference_folds` (the additive ``S_full − S_fold(k)`` per-fold solve) and
    hands the ``K`` per-fold results + per-fold predictor AF to the CV ``store``. No
    OOF / ``s_cv`` reduction — reference CV scores raw.

    Per-fold predictor AF is derived from the shared per-slot moments
    ``(zsum − fold_zsum[k]) / (2·n_train)`` so one predictor slot has one AF across every
    target in a fold (satisfies ``build_chip_axis``'s consistency guard, R6).
    """
    cross = _batch_cross_products(jobs, buf, folds)
    K = cross["K"]
    for j, job in enumerate(jobs):
        try:
            block, _idx, _pred_af = _assemble_block(job, j, buf, cross)
            fold_results = fit_reference_folds(
                block, alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv_folds
            )
            if block.n_predictors > 0:
                pred_af_per_fold = []
                for k in range(K):
                    n_tr = block.n - int(block.fold_n[k])
                    denom = 2.0 * n_tr if n_tr > 0 else 1.0
                    pred_af_per_fold.append((block.zsum - block.fold_zsum[k]) / denom)
            else:
                pred_af_per_fold = [np.empty(0) for _ in range(K)]
            job.store(fold_results, job.pred_idx, pred_af_per_fold)
        except Exception as exc:  # noqa: BLE001 - mirror legacy: record, don't crash
            job.fail(exc)


class StreamingImputationFitter:
    """Fit all missing-variant models by streaming the panel once per chromosome."""

    def __init__(self, plan: StreamPlan, device: str = "cpu"):
        self.plan = plan
        self.folds = GlobalFolds(len(plan.sample_ids), plan.cv_folds, plan.random_state)
        # device="auto" engages the GPU only when n is large enough to beat CPU (size guard).
        self.backend = resolve_streaming_backend(device, self.folds.n)
        self.W = plan.window_size
        self.chrom_index = ChromosomeIndex(plan.platform_variant_info)
        pvi = plan.platform_variant_info
        self._pv_id = pvi["variant_id"].to_numpy()
        self._pv_chrom = pvi["chromosome"].astype(str).to_numpy()
        self._pv_pos = pvi["position"].to_numpy()
        self._pv_alt = pvi["alt_allele"].astype(str).to_numpy()
        self._pv_ref = pvi["ref_allele"].astype(str).to_numpy()
        self.s_true = np.zeros(self.folds.n, dtype=np.float64)
        self.s_cv = np.zeros(self.folds.n, dtype=np.float64)
        # Batch width for the banded cross-product / OOF GEMMs. Bounded so the two
        # (n×T) working arrays (stacked targets Y + their OOF) stay ~256 MB even at
        # 500K samples; the ≥16 floor keeps the GEMMs wide enough to run near BLAS-3
        # peak. At 1000G scale n is tiny, so the cap never binds.
        self._batch_cap = max(16, min(4096, (256 * 1024 * 1024) // (max(self.folds.n, 1) * 8)))
        # Set by run_reference_cv: {fold_k -> {prs_id -> ImputedVariantModel}}. When
        # non-None the fitter is in leave-one-fold-out reference-CV mode (see _fit_batch).
        self._cv_collector: Optional[Dict[int, Dict[str, ImputedVariantModel]]] = None
        # Solver-selection override, resolved once here in the parent (pickled to workers).
        self._solve_mode = _resolve_solve_mode()

    def run(self, source, *, n_workers: int = 1) -> StreamingFitResult:
        """Stream the panel and fit every unit, optionally sharding by chromosome.

        ``n_workers > 1`` fans the per-chromosome accumulation + local solves across a
        process pool (Phase 7); the per-chromosome partials are reduced in canonical
        order. ``n_workers=1`` (default) runs a serial in-process map — bit-identical to
        the pre-Phase-7 loop. GPU keeps a single process (``fan_out_chromosomes`` clamps).
        """
        from imputed_prs.compute.parallel import fan_out_chromosomes

        device = getattr(self.backend, "device_name", "cpu")
        partials = fan_out_chromosomes(
            self, source, self._stream_chromosomes(), n_workers=n_workers, device=device
        )
        return StreamingFitResult.reduce(partials, self.folds.n)

    def _stream_chromosomes(self) -> List[str]:
        chset = set()
        for spec in list(self.plan.targets.values()) + list(
            self.plan.fallback_targets.values()
        ):
            chset.add(str(spec.chromosome))
        # Stream every chromosome that carries a target/fallback; predictors ride along.
        # Preserve a stable, human order (1..22 then others).
        return sorted(chset, key=lambda c: _chrom_sort_key(c))

    def _run_one_chromosome(self, source, chrom) -> "_ImputeChromPartial":
        """Stream one chromosome and return its partial (Phase 7 shard unit).

        All accumulators are **local** (the models/failure dicts, the two calibration
        vectors, and — in reference-CV mode — the per-fold collector), so this is a pure
        function of ``(self.plan, self.folds, source, chrom)`` with no shared-``self``
        mutation: safe to run in a worker process and reduce in the parent. The fresh
        per-chromosome band buffer + within-chromosome windows make it bit-identical
        regardless of sharding. ``self._cv_collector`` is read only as a **mode marker**.
        """
        _set_active_solve_mode(self._solve_mode)  # per-fit, deterministic across the pool
        plan = self.plan
        buf = self.backend.make_buffer(self.folds.n, self.folds)
        models: Dict[str, ImputedVariantModel] = {}
        fallback_models: Dict[str, ImputedVariantModel] = {}
        failures: Dict[str, str] = {}
        s_true = np.zeros(self.folds.n, dtype=np.float64)
        s_cv = np.zeros(self.folds.n, dtype=np.float64)
        cv_collector = (
            {k: {} for k in range(self.folds.n_folds)}
            if self._cv_collector is not None
            else None
        )
        open_targets: List[_OpenTarget] = []
        n_intercept_only = 0
        frontier = -1
        # Fallback-training failures are discarded (the dense path ignores them too):
        # an observed variant that fails to train a fallback is still scored directly.
        fb_failures: Dict[str, str] = {}

        def close_ready(force: bool):
            nonlocal n_intercept_only
            cutoff = float("inf") if force else frontier
            ready: List[_OpenTarget] = []
            still_open: List[_OpenTarget] = []
            for tgt in open_targets:
                if tgt.spec.position + self.W < cutoff:
                    ready.append(tgt)
                else:
                    still_open.append(tgt)
            open_targets[:] = still_open
            if ready:
                # Fit every co-windowed target that closes together as one batch, so
                # their cross-products / OOF collapse into banded GEMMs (_fit_chunk).
                n_intercept_only += self._fit_batch(
                    ready, buf, models, fallback_models, failures, fb_failures,
                    s_true, s_cv, cv_collector,
                )

        region = _region_for(source, chrom)
        for block in source.iter_variant_blocks(region=region):
            info = block.variant_info
            dos = block.dosages  # (n, b) float32, natural sample order, raw ALT
            ids = info["variant_id"].to_numpy()
            positions = info["position"].to_numpy()
            chip_cols, chip_pidx, chip_pos, chip_af = [], [], [], []
            for j in range(len(ids)):
                sid = ids[j]
                is_chip = sid in plan.chip_ids
                is_target = sid in plan.targets
                fb = plan.fallback_targets.get(sid)
                obs = plan.observed.get(sid)
                if not (is_chip or is_target or fb is not None or obs is not None):
                    continue
                raw = dos[:, j]
                pos = int(positions[j])
                if is_chip:  # raw ALT-counted predictor column
                    col_perm, af, _ = _prepare_column(raw, flip=False, folds=self.folds)
                    chip_cols.append(col_perm)
                    chip_pidx.append(plan.chip_ids[sid])
                    chip_pos.append(pos)
                    chip_af.append(af)
                if obs is not None:  # observed PRS calibration term (effect-oriented)
                    x_eff, _, _ = _prepare_column(raw, flip=obs.effect_flip, folds=self.folds)
                    s_true += obs.beta * x_eff
                    s_cv += obs.beta * x_eff
                if is_target:
                    spec = plan.targets[sid]
                    col_perm, af, _ = _prepare_column(
                        raw, flip=spec.effect_flip, folds=self.folds
                    )
                    open_targets.append(_OpenTarget(sid, spec, col_perm, af))
                if fb is not None:  # observed-variant fallback (effect-oriented target)
                    col_perm, af, _ = _prepare_column(
                        raw, flip=fb.effect_flip, folds=self.folds
                    )
                    open_targets.append(
                        _OpenTarget(sid, fb, col_perm, af, is_fallback=True)
                    )
                frontier = max(frontier, pos)
            # One batched band-Gram update per stream block (GPU: GEMM accumulation; the
            # band is only read in close_ready, and every column is added before it either
            # way, so this is exactly the per-column path).
            if chip_cols:
                buf.add_batch(chip_cols, chip_pidx, chip_pos, chip_af)
            close_ready(force=False)
            buf.evict_below(frontier - 2 * self.W)

        close_ready(force=True)
        buf.clear()
        return _ImputeChromPartial(
            chrom=str(chrom),
            models=models,
            fallback_models=fallback_models,
            failures=failures,
            s_true=s_true,
            s_cv=s_cv,
            n_intercept_only=n_intercept_only,
            cv_collector=cv_collector,
        )

    def _fit_batch(
        self, batch, buf, models, fallback_models, failures, fb_failures,
        s_true, s_cv, cv_collector,
    ) -> int:
        """Build a fit job per closing target and hand them to the shared batched kernel.

        Predictor selection (``chrom_index.window``) is buffer-independent, so it is
        done here up front; a window failure is recorded (record-don't-crash) and the
        target skipped. Returns the number of non-fallback intercept-only models.

        In reference-CV mode (``self._cv_collector`` set) each target routes to a CV
        store that collects per-outer-fold models via :func:`_run_cv_chunk`; observed-
        variant fallbacks are skipped entirely (the evaluator scores observed terms
        directly, so they are never trained during CV).
        """
        cv = cv_collector is not None
        jobs: List[_FitJob] = []
        for tgt in batch:
            spec = tgt.spec
            if cv and tgt.is_fallback:
                continue
            dest = fallback_models if tgt.is_fallback else models
            fmap = fb_failures if tgt.is_fallback else failures
            try:
                win = self.chrom_index.window(
                    spec.chromosome, spec.position, window_size=self.W,
                    exclude_target=True, max_variants=self.plan.max_predictors,
                )
            except Exception as exc:  # noqa: BLE001
                fmap[spec.prs_variant_id] = f"{type(exc).__name__}: {exc}"
                continue
            store = (
                self._cv_storer(spec, tgt.af, cv_collector)
                if cv
                else self._impute_storer(spec, tgt.af, dest)
            )
            jobs.append(_FitJob(
                col=tgt.col, pred_idx=win.variant_indices, calib_coef=float(spec.beta),
                is_calibrating=not tgt.is_fallback,
                store=store,
                fail=self._impute_failer(spec.prs_variant_id, fmap),
            ))
        if cv:
            _run_cv_batch(
                jobs, buf, self.folds, self.plan.alpha, self.plan.l1_ratio,
                self.plan.cv_folds, self._batch_cap,
            )
            return 0
        return self.backend.run_fit_batch(
            jobs, buf, self.folds, self.plan.alpha, self.plan.l1_ratio,
            self.plan.cv_folds, s_true, s_cv, self._batch_cap,
        )

    def run_reference_cv(self, source, outer_folds: "GlobalFolds", *, n_workers: int = 1):
        """Single-pass leave-one-fold-out reference CV over the panel.

        Streams the panel **once** with the buffer's folds set to ``outer_folds`` (the
        reference-CV outer partition), accumulating the full-panel band Gram and each
        per-fold slab; every closing target is fit for *all* ``K`` training folds by the
        additive subtraction ``S_full − S_fold(k)`` (:func:`_run_cv_chunk`). Replaces the
        ``k`` independent refit passes of ``ImputationEvaluator.cross_validate``.
        ``n_workers > 1`` shards that single pass by chromosome across processes (still
        one pass total — the additive subtraction stays within each shard).

        Returns ``(fold_models, failures)`` where ``fold_models[k]`` is the list of
        ``ImputedVariantModel`` trained on all samples except outer fold ``k``.

        The per-fold solve is host-side (numpy), so CV runs on the CPU buffer even when
        the fitter was constructed with a GPU device (the device-native kernel returns
        device tensors that :func:`fit_reference_folds` cannot consume — device CV is a
        Phase-3 follow-up).
        """
        from imputed_prs.compute.parallel import fan_out_chromosomes

        if getattr(self.backend, "device_name", "cpu") != "cpu":
            from imputed_prs.compute.device import get_backend

            self.backend = get_backend("cpu")
        self.folds = outer_folds
        self._batch_cap = max(
            16, min(4096, (256 * 1024 * 1024) // (max(self.folds.n, 1) * 8))
        )
        # Marker only (non-None ⇒ CV mode): the real per-fold collector is allocated
        # per chromosome inside _run_one_chromosome and returned in each partial.
        self._cv_collector = {}
        try:
            partials = fan_out_chromosomes(
                self, source, self._stream_chromosomes(), n_workers=n_workers, device="cpu"
            )
            fold_models, failures = reduce_cv_collectors(partials, outer_folds.n_folds)
        finally:
            self._cv_collector = None
        return fold_models, failures

    def _impute_storer(self, spec, af, dest):
        def store(result, pred_idx, pred_af):
            dest[spec.prs_variant_id] = self._to_model(spec, af, result, pred_idx, pred_af)
        return store

    def _cv_storer(self, spec, af, cv_collector):
        """Store the K per-outer-fold models for one target into ``cv_collector``.

        ``af`` is the target's full-panel effect-allele frequency, used only for the
        cosmetic ``allele_frequency``/``residual_variance`` fields (not on the held-out
        scoring path); the predictors carry per-fold training AFs from ``pred_af_list``.
        ``cv_collector`` is the chromosome-local ``{fold_k -> {prs_id -> model}}`` map.
        """
        def store(fold_results, pred_idx, pred_af_list):
            for k, res in enumerate(fold_results):
                cv_collector[k][spec.prs_variant_id] = self._to_model(
                    spec, af, res, pred_idx, pred_af_list[k]
                )
        return store

    @staticmethod
    def _impute_failer(vid, fmap):
        def fail(exc):
            fmap[vid] = f"{type(exc).__name__}: {exc}"
        return fail

    def _to_model(self, spec, af, result, pred_idx, pred_af) -> ImputedVariantModel:
        af_clip = float(np.clip(af, 0.0, 1.0))
        return ImputedVariantModel(
            variant_id=spec.prs_variant_id,
            chromosome=str(spec.chromosome),
            position=int(spec.position),
            effect_allele=spec.effect_allele,
            other_allele=spec.other_allele,
            beta=float(spec.beta),
            allele_frequency=af_clip,
            imputation_r2=float(result.cv_r2),
            residual_variance=compute_residual_variance(af_clip, result.cv_r2),
            intercept=float(result.intercept),
            predictor_variant_ids=self._pv_id[pred_idx].tolist(),
            coefficients=np.asarray(result.coefficients).copy(),
            is_intercept_only=bool(result.is_intercept_only),
            predictor_chromosomes=self._pv_chrom[pred_idx].tolist(),
            predictor_positions=self._pv_pos[pred_idx].tolist(),
            predictor_counted_alleles=self._pv_alt[pred_idx].tolist(),
            predictor_other_alleles=self._pv_ref[pred_idx].tolist(),
            predictor_allele_frequencies=pred_af,
        )


def collect_reference_variant_info(source, chroms: Sequence[str]) -> pd.DataFrame:
    """Gather reference variant metadata (no dosages retained) for classification.

    Iterates the source's variant blocks per chromosome and concatenates their
    ``variant_info``. For VCF this re-reads dosages (discarded) — acceptable for
    validation; the production PGEN backend can serve this cheaply from ``.pvar``.
    """
    cols = ["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]
    frames = []
    for chrom in chroms:
        for block in source.iter_variant_blocks(region=_region_for(source, chrom)):
            frames.append(block.variant_info)
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def build_stream_plan(
    ref_info: pd.DataFrame,
    prs_df: pd.DataFrame,
    platform_variant_set,
    *,
    sample_ids: Sequence[str],
    window_size: int = 1_000_000,
    max_predictors: Optional[int] = None,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
):
    """Metadata-only harmonization → ``StreamPlan`` (no dosage matrix needed).

    Reproduces the dense path's partition → allele-reclassification → platform
    matching (``core/linear_imputation_prs.py`` Steps 7c–8) using ``would_resolve``
    instead of ``resolve``, and builds the observed-variant fallback targets
    (Step 12) plus the PRS-side bookkeeping the orchestrator's dispositions need
    (``observed_prs_ids``, ``fallback_no_target_ids``). ``exclude_ambiguous`` is not
    applied here (default False; its AF-based QC would fold into the streaming pass —
    a follow-up). Returns ``(plan, drop_reasons)`` with drop reasons matching the
    dense granularity (``reference_contig_missing`` / ``allele_mismatch`` /
    ``not_in_reference``).
    """
    from imputed_prs.core.harmonizer import (
        ReferenceAlleleResolver,
        _normalize_chromosome,
        partition_variants,
    )

    resolver = ReferenceAlleleResolver(ref_info)
    reference_index = resolver.locus_to_rows
    gv_ids = ref_info["variant_id"].to_numpy()
    gv_pos = ref_info["position"].to_numpy()
    gv_chrom = ref_info["chromosome"].astype(str).to_numpy()
    gv_chrom_norm = normalize_chromosome_array(ref_info["chromosome"]).tolist()
    gv_ref = ref_info["ref_allele"].astype(str).to_numpy()
    gv_alt = ref_info["alt_allele"].astype(str).to_numpy()
    geno_var_to_idx: Dict[str, int] = {}
    for idx in range(len(gv_ids)):
        geno_var_to_idx.setdefault(gv_ids[idx], idx)
        geno_var_to_idx.setdefault(f"{gv_chrom_norm[idx]}:{int(gv_pos[idx])}", idx)

    part = partition_variants(prs_df, set(platform_variant_set))
    observed_ids = set(part.observed)
    missing_ids = set(part.missing)

    p_vid = prs_df["variant_id"].to_numpy()
    p_chrom = prs_df["chromosome"].to_numpy()
    p_pos = prs_df["position"].to_numpy()
    p_eff = prs_df["effect_allele"].to_numpy()
    p_oth = (
        prs_df["other_allele"].to_numpy()
        if "other_allele" in prs_df.columns
        else np.array([None] * len(prs_df), dtype=object)
    )
    p_beta = prs_df["beta"].to_numpy()

    def _other(i):
        v = p_oth[i]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return str(v)

    # Step 7c: reclassify observed -> missing when allele-incompatible with reference.
    reclass = set()
    for i in range(len(p_vid)):
        if p_vid[i] not in observed_ids:
            continue
        locus = f"{_normalize_chromosome(str(p_chrom[i]))}:{int(p_pos[i])}"
        if locus not in reference_index:
            continue
        if resolver.would_resolve(p_chrom[i], int(p_pos[i]), p_eff[i], _other(i)) is None:
            reclass.add(p_vid[i])
    observed_ids -= reclass
    missing_ids |= reclass

    # Step 8: platform variant info + chip predictor index (first ref row per platform id).
    platform_rows = []
    chip_ids: Dict[str, int] = {}
    for var_id in platform_variant_set:
        idx = geno_var_to_idx.get(var_id)
        if idx is None:
            idx = geno_var_to_idx.get(str(var_id).lower())
        if idx is None:
            continue
        ref_vid = gv_ids[idx]
        if ref_vid in chip_ids:
            continue
        chip_ids[ref_vid] = len(platform_rows)
        platform_rows.append(
            dict(variant_id=ref_vid, chromosome=gv_chrom[idx], position=int(gv_pos[idx]),
                 ref_allele=gv_ref[idx], alt_allele=gv_alt[idx])
        )
    cols = ["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]
    platform_info = pd.DataFrame(platform_rows) if platform_rows else pd.DataFrame(columns=cols)

    reference_contigs = set(gv_chrom_norm)

    # Observed calibration terms + fallbacks + missing targets, keyed by reference
    # variant_id. Fallback targets mirror the dense Step-12 observed fallbacks: every
    # kept-observed variant that resolves to a reference row is trained the same way a
    # missing target is (effect-oriented target, local-window predictors, own locus
    # auto-excluded), so it can be recovered when a user's upload cannot call it.
    observed: Dict[str, ObservedVar] = {}
    fallback_targets: Dict[str, TargetVar] = {}
    observed_prs_ids: set = set()
    fallback_no_target_ids: set = set()
    targets: Dict[str, TargetVar] = {}
    drop_reasons: Dict[str, str] = {}
    for i in range(len(p_vid)):
        vid = p_vid[i]
        wr = resolver.would_resolve(p_chrom[i], int(p_pos[i]), p_eff[i], _other(i))
        if vid in observed_ids:
            observed_prs_ids.add(vid)
            if wr is None:
                # Kept observed but locus absent from the reference: scored directly
                # from the user upload, but there is no panel to train a fallback from
                # (and it is absent from dense X_full / calibration too).
                fallback_no_target_ids.add(vid)
                continue
            ref_idx, flip = wr
            observed[gv_ids[ref_idx]] = ObservedVar(beta=float(p_beta[i]), effect_flip=flip)
            fallback_targets[gv_ids[ref_idx]] = TargetVar(
                prs_variant_id=str(vid), chromosome=str(gv_chrom[ref_idx]),
                position=int(gv_pos[ref_idx]), effect_allele=str(p_eff[i]),
                other_allele=_other(i), beta=float(p_beta[i]), effect_flip=flip,
            )
        elif vid in missing_ids:
            if wr is None:
                # Match the dense drop-reason granularity (Step 8).
                chrom_n = _normalize_chromosome(str(p_chrom[i]))
                locus = f"{chrom_n}:{int(p_pos[i])}"
                if chrom_n not in reference_contigs:
                    drop_reasons[vid] = "reference_contig_missing"
                elif locus in reference_index:
                    drop_reasons[vid] = "allele_mismatch"
                else:
                    drop_reasons[vid] = "not_in_reference"
                continue
            ref_idx, flip = wr
            targets[gv_ids[ref_idx]] = TargetVar(
                prs_variant_id=str(vid), chromosome=str(gv_chrom[ref_idx]),
                position=int(gv_pos[ref_idx]), effect_allele=str(p_eff[i]),
                other_allele=_other(i), beta=float(p_beta[i]), effect_flip=flip,
            )

    plan = StreamPlan(
        sample_ids=list(sample_ids), platform_variant_info=platform_info,
        chip_ids=chip_ids, targets=targets, observed=observed, window_size=window_size,
        max_predictors=max_predictors, alpha=alpha, l1_ratio=l1_ratio,
        cv_folds=cv_folds, random_state=random_state,
        fallback_targets=fallback_targets, observed_prs_ids=observed_prs_ids,
        fallback_no_target_ids=fallback_no_target_ids,
    )
    return plan, drop_reasons


def streaming_fit_imputation(
    source,
    prs_df: pd.DataFrame,
    platform_variant_set,
    *,
    window_size: int = 1_000_000,
    max_predictors: Optional[int] = None,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    device: str = "cpu",
):
    """End-to-end streaming imputation: scan metadata, harmonize, fit, accumulate.

    Returns ``(result, plan, drop_reasons)``. Calibration is finalized separately via
    ``evaluation.streaming_calibration.finalize_imputation_calibration``.
    """
    from imputed_prs.core.harmonizer import _normalize_chromosome

    chroms = sorted(
        {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
        key=_chrom_sort_key,
    )
    ref_info = collect_reference_variant_info(source, chroms)
    plan, drop_reasons = build_stream_plan(
        ref_info, prs_df, platform_variant_set, sample_ids=source.sample_ids,
        window_size=window_size, max_predictors=max_predictors, alpha=alpha,
        l1_ratio=l1_ratio, cv_folds=cv_folds, random_state=random_state,
    )
    fitter = StreamingImputationFitter(plan, device=device)
    result = fitter.run(source)
    return result, plan, drop_reasons


def _chrom_sort_key(chrom: str):
    c = str(chrom).replace("chr", "")
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if c.isdigit():
        return (0, int(c))
    return (1, order.get(c, 99), c)


def _region_for(source, chrom: str) -> str:
    """Map a normalized chromosome to the source's raw contig spelling for a region.

    ``VcfGenotypeSource`` requires the file's own contig name (e.g. ``chr22``, not
    ``22``); we discover it from ``source.contigs``. Sources without a ``contigs``
    property (e.g. in-memory test doubles) get the normalized chromosome unchanged.
    """
    from imputed_prs.core.harmonizer import _normalize_chromosome

    contigs = getattr(source, "contigs", None)
    if not contigs:
        return str(chrom)
    for raw in contigs:
        if _normalize_chromosome(str(raw)) == str(chrom):
            return str(raw)
    return str(chrom)
