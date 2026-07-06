"""Panel-scale, allele-oriented batch scoring for the evaluators (Phase 4).

The per-user :class:`~imputed_prs.models.predictor.PRSPredictor` scores one sample
at a time in Python; the evaluators score a whole reference panel and, at 2M
variants, their per-model / per-predictor loops become millions of Python
iterations. This module contracts those loops into a handful of vectorized array
ops and a single ``scipy.sparse`` CSR mat-mul, orienting each chip variant **once**
regardless of how many models reference it.

Design invariants (validated in ``tests/test_vectorized_predictor.py``):

- **Orientation is byte-identical** to the per-predictor oracle
  (``evaluation/_scoring.py::oriented_predictor_matrix``): float32 gather →
  **float32** flip (``np.float32(2.0) - dosage``, matching the NEP-50 weak-scalar
  promotion of ``2.0 - dosage`` inside ``match_oriented_dosage``) → cast float64 →
  per-column ``2*AF`` NaN-fill.
- **The estimated (imputed / projected) score is pure float64.** It differs from
  the per-model oracle loop only by the CSR mat-mul's float re-association
  (~1e-14 per dot product), so it is used only for larger inputs and validated at
  ``atol≈1e-9`` against the oracle — never at the golden ``atol=1e-12`` (bit
  identity is impossible: ``scipy`` canonicalizes CSR indices and the SpMM
  reorders additions). Tiny inputs stay on the oracle via the evaluator's
  size-select.
- **scipy.sparse is confined to this module.**

``scipy``'s CSR is CPU-only; Phase 4 prediction is intentionally CPU/CPU-only and
independent of the Phase-3 GPU work.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from imputed_prs.core.harmonizer import ReferenceAlleleResolver
from imputed_prs.core.types import GenotypeData, VariantInfo


def _is_effective_intercept_only(model) -> bool:
    """Mirror the oracle's intercept-only test (``_predicted_prs_numeric``).

    A model contributes no predictor terms — its raw prediction is just the
    intercept — iff it is flagged intercept-only *or* carries no predictors.
    """
    return bool(model.is_intercept_only) or not model.predictor_variant_ids


@dataclass(frozen=True)
class ChipAxis:
    """Global chip-predictor column axis for a set of models.

    Each unique predictor ``variant_id`` becomes one column ``k`` (first-occurrence
    order). ``ref_rows[k]``/``flips[k]`` are the reference row and effect-orientation
    flip that :meth:`ReferenceAlleleResolver.would_resolve` selects for that chip
    variant; ``resolved[k]`` is False when no allele-compatible reference row exists
    (the column then holds the ``2*AF`` population mean). ``counted_af[k]`` is the
    counted-allele frequency used for that mean fill.
    """

    chip_index: Dict[str, int]
    ref_rows: np.ndarray  # (K,) int64
    flips: np.ndarray  # (K,) bool
    resolved: np.ndarray  # (K,) bool
    counted_af: np.ndarray  # (K,) float64

    @property
    def n_chip(self) -> int:
        return len(self.chip_index)


def build_chip_axis(models: Sequence, resolver: ReferenceAlleleResolver) -> ChipAxis:
    """Build the global chip axis + per-column orientation plan for ``models``.

    Works for both ``ImputedVariantModel`` and ``ProjectionRegionModel`` (duck-typed
    on the index-aligned ``predictor_*`` fields). Each chip variant is resolved once
    via :meth:`ReferenceAlleleResolver.would_resolve` — metadata only, no dosage
    materialized. Intercept-only-effective models contribute no columns.

    A **consistency guard** raises if the same predictor ``variant_id`` reappears
    with a different ``(counted_allele, other_allele, allele_frequency)``: under the
    invariant "same variant_id → same reference row/ALT/AF" this never fires, but it
    prevents a silently divergent shared column.
    """
    chip_index: Dict[str, int] = {}
    ref_rows: List[int] = []
    flips: List[bool] = []
    resolved: List[bool] = []
    counted_af: List[float] = []
    meta: Dict[str, Tuple[str, Optional[str], float]] = {}

    for model in models:
        if _is_effective_intercept_only(model):
            continue
        chroms = model.predictor_chromosomes
        positions = model.predictor_positions
        counted = model.predictor_counted_alleles
        others = model.predictor_other_alleles
        afs = model.predictor_allele_frequencies
        for i, pid in enumerate(model.predictor_variant_ids):
            c_al = counted[i]
            o_al = others[i]
            af = float(afs[i])
            if pid in chip_index:
                prev = meta[pid]
                if c_al != prev[0] or o_al != prev[1] or af != prev[2]:
                    raise ValueError(
                        f"Inconsistent predictor metadata for chip variant {pid!r}: "
                        f"{(c_al, o_al, af)} vs {prev}"
                    )
                continue
            chip_index[pid] = len(chip_index)
            meta[pid] = (c_al, o_al, af)
            match = resolver.would_resolve(chroms[i], positions[i], c_al, o_al)
            if match is None:
                ref_rows.append(0)  # unused; resolved=False keeps the 2*AF fill
                flips.append(False)
                resolved.append(False)
            else:
                row, flip = match
                ref_rows.append(int(row))
                flips.append(bool(flip))
                resolved.append(True)
            counted_af.append(af)

    return ChipAxis(
        chip_index=chip_index,
        ref_rows=np.asarray(ref_rows, dtype=np.int64),
        flips=np.asarray(flips, dtype=bool),
        resolved=np.asarray(resolved, dtype=bool),
        counted_af=np.asarray(counted_af, dtype=np.float64),
    )


def oriented_chip_matrix(dosage_matrix: np.ndarray, axis: ChipAxis) -> np.ndarray:
    """Build the ``(n_samples, K)`` effect-oriented chip dosage matrix (float64).

    Byte-identical to column-stacking ``oriented_predictor_matrix`` over the chip
    variants: unresolved columns and NaN samples take the ``2*AF`` population mean.
    The flip is done in **float32** (before the float64 cast) to reproduce the
    weak-scalar ``2.0 - dosage`` in ``match_oriented_dosage`` exactly.
    """
    K = axis.n_chip
    n_samples = int(dosage_matrix.shape[0])
    Z = np.empty((n_samples, K), dtype=np.float64)
    if K == 0:
        return Z

    fill = 2.0 * axis.counted_af  # (K,) float64
    Z[:] = fill[None, :]

    res = axis.resolved
    if not res.any():
        return Z

    cols = np.nonzero(res)[0]
    rows = axis.ref_rows[cols]
    z_raw = dosage_matrix[:, rows]  # float32 (reference dosages), NaN preserved
    flip = axis.flips[cols]
    # Flip in the source dtype so f32 dosages match the oracle's f32 ``2 - dosage``.
    z_or = np.where(flip[None, :], np.asarray(2.0, dtype=z_raw.dtype) - z_raw, z_raw)
    z_or = z_or.astype(np.float64)

    nan_mask = np.isnan(z_or)
    if nan_mask.any():
        fill_res = fill[cols]
        z_or = np.where(nan_mask, np.broadcast_to(fill_res, z_or.shape), z_or)

    Z[:, cols] = z_or
    return Z


def build_coef_csr(
    models: Sequence, chip_index: Dict[str, int]
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the ``(n_target, K)`` CSR coefficient matrix + intercepts + betas.

    Built directly from ``indptr/indices/data`` (no COO list-append, which would be
    ~n_target*n_pred Python appends at 2M targets). Intercept-only-effective models
    contribute zero nonzeros, so their raw prediction is exactly the intercept.
    For an imputation model ``coefficients`` predicts a dosage; the caller clips to
    ``[0, 2]`` and multiplies by ``beta``.
    """
    n_target = len(models)
    K = len(chip_index)

    counts = np.zeros(n_target, dtype=np.int64)
    for j, model in enumerate(models):
        if not _is_effective_intercept_only(model):
            counts[j] = len(model.coefficients)
    indptr = np.empty(n_target + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])

    indices = np.fromiter(
        (
            chip_index[pid]
            for model in models
            if not _is_effective_intercept_only(model)
            for pid in model.predictor_variant_ids
        ),
        dtype=np.int64,
        count=nnz,
    )
    if nnz:
        data = np.concatenate(
            [
                np.asarray(model.coefficients, dtype=np.float64)
                for model in models
                if not _is_effective_intercept_only(model)
            ]
        )
    else:
        data = np.zeros(0, dtype=np.float64)

    intercepts = np.array([float(m.intercept) for m in models], dtype=np.float64)
    betas = np.array([float(m.beta) for m in models], dtype=np.float64)

    W = sparse.csr_matrix((data, indices, indptr), shape=(n_target, K))
    return W, intercepts, betas


def panel_impute_prs(
    Z: np.ndarray,
    W: sparse.csr_matrix,
    intercepts: np.ndarray,
    betas: np.ndarray,
    *,
    block_size: int = 8192,
) -> np.ndarray:
    """Imputed PRS component for every sample: ``sum_j clip(z·w_j + b_j, 0, 2)·beta_j``.

    Target-blocked so the dense ``(n_samples, n_target)`` clip intermediate is never
    resident — only one ``(block, n_samples)`` slab at a time. Returns ``(n_samples,)``.
    """
    n_samples = int(Z.shape[0])
    n_target = int(W.shape[0])
    prs = np.zeros(n_samples, dtype=np.float64)
    if n_target == 0:
        return prs

    # (K, n_samples) C-contiguous so the per-block CSR SpMM reads it without copying.
    zt = np.ascontiguousarray(Z.T)
    for b0 in range(0, n_target, block_size):
        b1 = min(b0 + block_size, n_target)
        raw = W[b0:b1].dot(zt)  # (block, n_samples)
        raw += intercepts[b0:b1][:, None]
        np.clip(raw, 0.0, 2.0, out=raw)
        prs += betas[b0:b1] @ raw
    return prs


def build_projection_weff(
    region_models: Sequence, chip_index: Dict[str, int]
) -> Tuple[np.ndarray, float]:
    """Collapse region models to one effective weight vector + constant.

    Projection has no clip and its coefficients already bake in the betas, so the
    projected PRS is linear: ``sum_R (z·a_R + c_R) = z·w_eff + const`` with
    ``w_eff[k] = sum_R a_R[k]`` and ``const = sum_R c_R``. Every region contributes
    its intercept; only non-intercept regions contribute coefficients.
    """
    K = len(chip_index)
    w_eff = np.zeros(K, dtype=np.float64)
    const = 0.0

    cols_iter = []
    vals_iter = []
    for model in region_models:
        const += float(model.intercept)
        if _is_effective_intercept_only(model):
            continue
        cols_iter.append(
            np.fromiter(
                (chip_index[pid] for pid in model.predictor_variant_ids),
                dtype=np.int64,
                count=len(model.predictor_variant_ids),
            )
        )
        vals_iter.append(np.asarray(model.coefficients, dtype=np.float64))

    if cols_iter:
        cols = np.concatenate(cols_iter)
        vals = np.concatenate(vals_iter)
        np.add.at(w_eff, cols, vals)  # accumulates duplicates (shared predictors)
    return w_eff, const


def panel_project_prs(Z: np.ndarray, w_eff: np.ndarray, const: float) -> np.ndarray:
    """Projected PRS component for every sample: ``Z @ w_eff + const``."""
    return Z @ w_eff + const


def accumulate_true_prs(
    dosage_matrix: np.ndarray,
    resolver: ReferenceAlleleResolver,
    placed: Sequence[Tuple[str, int, str, Optional[str], float]],
    *,
    block_size: int = 8192,
) -> np.ndarray:
    """Vectorized gold-standard PRS: ``sum_v oriented_dosage_v · beta_v`` over placed variants.

    ``placed`` is an ordered list of ``(chromosome, position, effect_allele,
    other_allele, beta)``. Matches the per-variant oracle
    (``_compute_true_prs``) numerically: the oriented dosage is flipped in float32,
    NaN samples are skipped (contribute 0), and ``dosage * beta`` is formed in
    **float32** (the oracle's weak-scalar product) before accumulating in float64 —
    so only the block-sum re-association (~1e-14) differs from the sequential loop.
    Unresolved placed variants are skipped, exactly as ``match is None`` is skipped.
    """
    n_samples = int(dosage_matrix.shape[0])
    rows: List[int] = []
    flips: List[bool] = []
    betas: List[float] = []
    for chrom, pos, effect, other, beta in placed:
        match = resolver.would_resolve(chrom, pos, effect, other)
        if match is None:
            continue
        row, flip = match
        rows.append(int(row))
        flips.append(bool(flip))
        betas.append(float(beta))

    prs = np.zeros(n_samples, dtype=np.float64)
    if not rows:
        return prs

    rows_arr = np.asarray(rows, dtype=np.int64)
    flips_arr = np.asarray(flips, dtype=bool)
    betas_f32 = np.asarray(betas, dtype=np.float32)

    for b0 in range(0, len(rows_arr), block_size):
        b1 = min(b0 + block_size, len(rows_arr))
        z_raw = dosage_matrix[:, rows_arr[b0:b1]]  # (n, block) float32
        fl = flips_arr[b0:b1]
        z_or = np.where(fl[None, :], np.asarray(2.0, dtype=z_raw.dtype) - z_raw, z_raw)
        # NaN samples are skipped in the oracle; contributing 0 is equivalent.
        z_or = np.where(np.isnan(z_or), np.asarray(0.0, dtype=z_or.dtype), z_or)
        # Form ``dosage * beta`` in float32 (the oracle's weak-scalar product),
        # then accumulate the block in float64.
        contrib = z_or * betas_f32[b0:b1][None, :]
        prs += contrib.astype(np.float64).sum(axis=1)
    return prs


class VectorizedPredictor:
    """Standalone panel scorer: observed + (imputed | projected), all vectorized.

    A thin convenience over the module functions for scoring a whole panel in one
    call (used by benchmarks). The evaluator wires the same functions directly.
    Exactly one of ``imputed_models`` / ``region_models`` must be given. The observed
    component mirrors ``evaluation/_scoring.py::observed_component_numeric`` (float64,
    resolver-oriented) and is kept here to avoid a circular import.
    """

    def __init__(
        self,
        observed_variants: Sequence[VariantInfo],
        *,
        imputed_models: Optional[Sequence] = None,
        region_models: Optional[Sequence] = None,
    ):
        if (imputed_models is None) == (region_models is None):
            raise ValueError(
                "Provide exactly one of imputed_models or region_models"
            )
        self.observed_variants = list(observed_variants)
        self.imputed_models = imputed_models
        self.region_models = region_models

    def _observed_component(
        self, dosage_matrix: np.ndarray, resolver: ReferenceAlleleResolver
    ) -> np.ndarray:
        out = np.zeros(int(dosage_matrix.shape[0]), dtype=np.float64)
        for var in self.observed_variants:
            match = resolver.resolve(
                var.chromosome,
                var.position,
                var.effect_allele,
                var.other_allele,
                dosage_matrix,
            )
            if match is None:
                continue
            dosages = np.asarray(match[1], dtype=np.float64)
            valid = ~np.isnan(dosages)
            out[valid] += dosages[valid] * var.beta
        return out

    def predict_panel(
        self, genotype_data: GenotypeData, *, block_size: int = 8192
    ) -> np.ndarray:
        """Raw (uncalibrated) observed + model PRS for every sample ``(n_samples,)``."""
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)
        dosage_matrix = genotype_data.dosage_matrix
        prs = self._observed_component(dosage_matrix, resolver)

        models = self.imputed_models if self.imputed_models is not None else self.region_models
        axis = build_chip_axis(models, resolver)
        Z = oriented_chip_matrix(dosage_matrix, axis)

        if self.imputed_models is not None:
            W, intercepts, betas = build_coef_csr(models, axis.chip_index)
            prs = prs + panel_impute_prs(
                Z, W, intercepts, betas, block_size=block_size
            )
        else:
            w_eff, const = build_projection_weff(models, axis.chip_index)
            prs = prs + panel_project_prs(Z, w_eff, const)
        return prs
