"""Streaming sufficient-statistics driver for **projection** training (Phase 2, M3).

Projection regresses each merged genomic region's PRS contribution
``S_R = Σ_j β_j x_eff_j`` (over the region's *missing* PRS variants) on the platform
(chip) dosages in the region. Structurally this is the same local ElasticNet as
imputation — same CV, standardization, back-transform, intercept-only fallbacks — so it
reuses the **shared band-Gram buffer and batched-GEMM fit kernel** from
``sufficient_stats`` (``_ChipGramBuffer``, ``_run_fit_batch``). The only method-specific
pieces live here:

* the target is a region contribution ``S_R`` (accumulated from the region's missing
  variants as they stream by), not a single dosage;
* predictors are all chip variants in the merged region ``[start, end]`` (honouring
  ``max_predictors`` by distance-to-centre), not a per-target ±W window;
* the region buffer must hold columns back to the earliest still-open region's start
  (merged regions can span ≫ 2W), while still keeping the ±2W tail that the
  imputation-style **observed-variant fallbacks** need;
* calibration reduces each region's out-of-fold ``S_R`` prediction with coefficient
  ``1.0`` (the betas are already inside ``S_R``), plus the observed β·x_eff terms.

Parity/deviation is identical to imputation: exact on no-missing data with a pinned
``random_state`` (mean-imputation + global folds), a documented deviation under
missingness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from imputed_prs.compute.device import resolve_streaming_backend
from imputed_prs.compute.gram_solve import fit_from_local_gram  # noqa: F401 (kernel dep)
from imputed_prs.compute.sufficient_stats import (
    GlobalFolds,
    ObservedVar,
    TargetVar,
    _chrom_sort_key,
    _FitJob,
    _OpenTarget,
    _prepare_column,
    _region_for,
    _run_cv_batch,
    reduce_cv_collectors,
)
from imputed_prs.core.harmonizer import normalize_chromosome_array
from imputed_prs.core.regions import merge_variant_windows
from imputed_prs.core.types import ImputedVariantModel, ProjectionRegionModel
from imputed_prs.core.window_index import ChromosomeIndex
from imputed_prs.models.trainer import compute_residual_variance


# ---------------------------------------------------------------------------
# Plan (built by the orchestrator's metadata-only harmonization; see backend seam).
# ---------------------------------------------------------------------------
@dataclass
class RegionSpec:
    """A merged region whose target is S_R = Σ β_j x_eff_j over its missing variants."""

    region_id: str
    chromosome: str
    start: int
    end: int
    # PRS-variant (missing) metadata for the emitted ProjectionRegionModel, region order.
    prs_variant_ids: List[str]
    betas: np.ndarray
    prs_positions: List[int]
    prs_effect_alleles: List[str]
    prs_other_alleles: List[Optional[str]]


@dataclass
class ProjectionStreamPlan:
    """Everything the streaming projection fitter needs, from metadata-only harmonization."""

    sample_ids: List[str]
    platform_variant_info: pd.DataFrame
    chip_ids: Dict[str, int]  # source variant_id -> platform row index (predictors)
    regions: List[RegionSpec]  # sorted by (chromosome, start), non-overlapping
    # reference variant_id -> [(region_index, beta, effect_flip)]; drives the streamed
    # S_R accumulation (a ref row may back >1 missing PRS variant at a multiallelic locus).
    region_members: Dict[str, List[Tuple[int, float, bool]]]
    observed: Dict[str, ObservedVar]  # ref variant_id -> observed calibration term
    # Observed-variant fallbacks (imputation-style single-variant models, P2.4), keyed
    # by ref variant_id; disjoint from region targets.
    fallback_targets: Dict[str, TargetVar] = field(default_factory=dict)
    window_size: int = 1_000_000
    max_predictors: Optional[int] = None
    alpha: float = 0.01
    l1_ratio: float = 0.5
    cv_folds: int = 5
    random_state: Optional[int] = None
    observed_prs_ids: set = field(default_factory=set)
    fallback_no_target_ids: set = field(default_factory=set)


@dataclass
class ProjectionStreamResult:
    """Output: region models + the two calibration accumulators (+ diagonal SE var)."""

    region_models: Dict[str, ProjectionRegionModel]
    s_true: np.ndarray  # (n,) permuted order; order-independent for calibration
    s_cv: np.ndarray  # (n,)
    diag_var: float = 0.0  # Σ region.cv_mse — the diagonal-SE lower-bound term
    n_regions_trained: int = 0
    n_intercept_only: int = 0
    n_regions_failed: int = 0
    failures: Dict[str, str] = field(default_factory=dict)
    fallback_models: Dict[str, ImputedVariantModel] = field(default_factory=dict)
    has_calibration_terms: bool = False

    @classmethod
    def reduce(cls, partials: Sequence["_ProjectChromPartial"], n: int) -> "ProjectionStreamResult":
        """Order-independent merge of per-chromosome partials (Phase 7 fan-out).

        Region/fallback/failure dicts are key-disjoint across chromosomes (dict-union);
        ``s_true``/``s_cv`` sum in canonical ``_chrom_sort_key`` order. ``diag_var`` is
        derived as ``Σ region.cv_mse`` over the merged models in that same canonical order
        (bit-identical to the serial ``self.diag_var +=``, which accumulated in region-close
        order per chromosome). ``has_calibration_terms`` is the OR of the shards.
        """
        ordered = sorted(partials, key=lambda p: _chrom_sort_key(p.chrom))
        region_models: Dict[str, ProjectionRegionModel] = {}
        fallback_models: Dict[str, ImputedVariantModel] = {}
        failures: Dict[str, str] = {}
        s_true = np.zeros(n, dtype=np.float64)
        s_cv = np.zeros(n, dtype=np.float64)
        n_io = 0
        has_terms = False
        for p in ordered:
            assert region_models.keys().isdisjoint(p.region_models), (
                "chromosome shards must have disjoint region ids"
            )
            region_models.update(p.region_models)
            fallback_models.update(p.fallback_models)
            failures.update(p.failures)
            s_true += p.s_true
            s_cv += p.s_cv
            n_io += p.n_intercept_only
            has_terms = has_terms or p.has_obs or bool(p.region_models)
        diag_var = sum(float(m.cv_mse) for m in region_models.values())
        return cls(
            region_models=region_models,
            s_true=s_true,
            s_cv=s_cv,
            diag_var=diag_var,
            n_regions_trained=len(region_models),
            n_intercept_only=n_io,
            n_regions_failed=len(failures),
            failures=failures,
            fallback_models=fallback_models,
            has_calibration_terms=has_terms,
        )


@dataclass
class _ProjectChromPartial:
    """One chromosome's contribution to a streaming projection fit (Phase 7 shard unit)."""

    chrom: str
    region_models: Dict[str, ProjectionRegionModel]
    fallback_models: Dict[str, ImputedVariantModel]
    failures: Dict[str, str]
    s_true: np.ndarray
    s_cv: np.ndarray
    has_obs: bool
    n_intercept_only: int
    cv_collector: Optional[Dict[int, Dict[str, ProjectionRegionModel]]] = None


# ---------------------------------------------------------------------------
# Fitter.
# ---------------------------------------------------------------------------
@dataclass
class _OpenRegion:
    idx: int  # index into plan.regions
    spec: RegionSpec
    s_r: np.ndarray  # (n,) permuted, accumulating Σ β_j x_eff_j


class StreamingProjectionFitter:
    """Fit all region projection models by streaming the panel once per chromosome."""

    def __init__(self, plan: ProjectionStreamPlan, device: str = "cpu"):
        self.plan = plan
        self.folds = GlobalFolds(len(plan.sample_ids), plan.cv_folds, plan.random_state)
        # device="auto" engages the GPU only when n is large enough to beat CPU (size guard).
        self.backend = resolve_streaming_backend(device, self.folds.n)
        self.W = plan.window_size
        self.chrom_index = ChromosomeIndex(plan.platform_variant_info)
        pvi = plan.platform_variant_info
        self._pv_id = pvi["variant_id"].to_numpy()
        self._pv_chrom = pvi["chromosome"].astype(str).to_numpy()
        self._pv_chrom_norm = np.asarray(normalize_chromosome_array(pvi["chromosome"]))
        self._pv_pos = pvi["position"].to_numpy()
        self._pv_alt = pvi["alt_allele"].astype(str).to_numpy()
        self._pv_ref = pvi["ref_allele"].astype(str).to_numpy()
        self.s_true = np.zeros(self.folds.n, dtype=np.float64)
        self.s_cv = np.zeros(self.folds.n, dtype=np.float64)
        self.diag_var = 0.0
        self._has_terms = False
        # Same batch-cap rationale as imputation (bound the (n×T) working arrays).
        self._batch_cap = max(16, min(4096, (256 * 1024 * 1024) // (max(self.folds.n, 1) * 8)))
        # Set by run_reference_cv: {fold_k -> {region_id -> ProjectionRegionModel}}.
        # When non-None the fitter is in leave-one-fold-out reference-CV mode.
        self._cv_collector = None
        # Regions grouped by chromosome, each sorted by start (== sorted by end, since
        # merged regions are non-overlapping), for the in-order close sweep.
        self._regions_by_chrom: Dict[str, List[int]] = {}
        for i, r in enumerate(plan.regions):
            self._regions_by_chrom.setdefault(str(r.chromosome), []).append(i)
        for lst in self._regions_by_chrom.values():
            lst.sort(key=lambda i: (plan.regions[i].start, plan.regions[i].end))

    def run(self, source, *, n_workers: int = 1) -> ProjectionStreamResult:
        """Stream the panel and fit every region, optionally sharding by chromosome.

        ``n_workers > 1`` fans the per-chromosome accumulation + solves across a process
        pool; the per-chromosome partials are reduced in canonical order. ``n_workers=1``
        (default) runs a serial in-process map, bit-identical to the pre-Phase-7 loop.
        """
        from imputed_prs.compute.parallel import fan_out_chromosomes

        device = getattr(self.backend, "device_name", "cpu")
        partials = fan_out_chromosomes(
            self, source, self._stream_chromosomes(), n_workers=n_workers, device=device
        )
        return ProjectionStreamResult.reduce(partials, self.folds.n)

    def run_reference_cv(self, source, outer_folds: GlobalFolds, *, n_workers: int = 1):
        """Single-pass leave-one-fold-out reference CV over the panel (projection).

        Streams once with the buffer's folds set to ``outer_folds`` (the reference-CV
        outer partition); every closing region is fit for all ``K`` training folds by the
        additive subtraction ``S_full − S_fold(k)``. ``n_workers > 1`` shards that single
        pass by chromosome across processes. Returns ``(fold_models, failures)`` where
        ``fold_models[k]`` is the list of ``ProjectionRegionModel`` trained on all samples
        except outer fold ``k``. Observed-variant fallbacks are not trained (the evaluator
        scores observed terms directly). Runs on the CPU buffer (host-side per-fold solve;
        device CV is a Phase-3 follow-up).
        """
        from imputed_prs.compute.parallel import fan_out_chromosomes

        if getattr(self.backend, "device_name", "cpu") != "cpu":
            from imputed_prs.compute.device import get_backend

            self.backend = get_backend("cpu")
        self.folds = outer_folds
        self._batch_cap = max(
            16, min(4096, (256 * 1024 * 1024) // (max(self.folds.n, 1) * 8))
        )
        # Marker only (non-None ⇒ CV mode); the real per-fold collector is per-chromosome.
        self._cv_collector = {}
        try:
            partials = fan_out_chromosomes(
                self, source, self._stream_chromosomes(), n_workers=n_workers, device="cpu"
            )
            fold_models, failures = reduce_cv_collectors(partials, outer_folds.n_folds)
        finally:
            self._cv_collector = None
        return fold_models, failures

    def _cv_region_storer(self, region, s_r, cv_collector):
        """Store the K per-outer-fold region models for one region into ``cv_collector``."""
        def store(fold_results, pred_idx, pred_af_list):
            for k, res in enumerate(fold_results):
                cv_collector[k][region.region_id] = self._to_region_model(
                    region, s_r, res, pred_idx, pred_af_list[k]
                )
        return store

    def _stream_chromosomes(self) -> List[str]:
        chset = {str(r.chromosome) for r in self.plan.regions}
        chset |= {str(t.chromosome) for t in self.plan.fallback_targets.values()}
        return sorted(chset, key=_chrom_sort_key)

    def _run_one_chromosome(self, source, chrom) -> "_ProjectChromPartial":
        """Stream one chromosome and return its partial (Phase 7 shard unit).

        All accumulators are **local** (region/fallback/failure dicts, the two calibration
        vectors, ``has_obs``, and — in CV mode — the per-fold collector), so this is a pure
        function with no shared-``self`` mutation, safe to run in a worker and reduce in the
        parent. ``self.diag_var``/``self._has_terms`` written by the storers land on the
        throwaway worker ``self`` and are discarded — ``diag_var`` is re-derived (``Σ cv_mse``)
        and ``has_calibration_terms`` OR-ed in ``ProjectionStreamResult.reduce``.
        """
        plan = self.plan
        # Projection units are few + wide (merged regions span ≫ 2W), so the per-fold Gram
        # is materialised on-demand per region (≤max_predictors) rather than kept as a
        # (K, cap, cap) band tensor — Finding-#1 band-limited per-fold Gram (Phase 3E).
        buf = self.backend.make_buffer(self.folds.n, self.folds, lazy_fold_gram=True)
        region_models: Dict[str, ProjectionRegionModel] = {}
        fallback_models: Dict[str, ImputedVariantModel] = {}
        failures: Dict[str, str] = {}
        s_true = np.zeros(self.folds.n, dtype=np.float64)
        s_cv = np.zeros(self.folds.n, dtype=np.float64)
        has_obs = False
        cv_collector = (
            {k: {} for k in range(self.folds.n_folds)}
            if self._cv_collector is not None
            else None
        )
        region_ptr = 0  # next region (in start order) not yet closed
        chrom_regions = self._regions_by_chrom.get(str(chrom), [])
        open_sr: Dict[int, np.ndarray] = {}  # region_idx -> accumulating S_R
        open_fallbacks: List[_OpenTarget] = []
        n_intercept_only = 0
        frontier = -1
        fb_failures: Dict[str, str] = {}

        cv = cv_collector is not None

        def close_ready(force: bool):
            nonlocal n_intercept_only, region_ptr
            cutoff = float("inf") if force else frontier
            jobs: List[_FitJob] = []
            # Regions close in start order once the frontier passes their end.
            while region_ptr < len(chrom_regions):
                ridx = chrom_regions[region_ptr]
                region = plan.regions[ridx]
                if region.end < cutoff:
                    s_r = open_sr.pop(ridx, None)
                    if s_r is None:
                        s_r = np.zeros(self.folds.n, dtype=np.float64)
                    job = self._region_job(
                        region, s_r, buf, region_models, failures, cv_collector
                    )
                    if job is not None:
                        jobs.append(job)
                    region_ptr += 1
                else:
                    break
            # Observed-variant fallbacks close like imputation targets (±W window).
            # In CV mode they are never trained (observed terms are scored directly),
            # so they are drained without producing fit jobs.
            still_open: List[_OpenTarget] = []
            for tgt in open_fallbacks:
                if tgt.spec.position + self.W < cutoff:
                    if not cv:
                        job = self._fallback_job(tgt, fallback_models, fb_failures)
                        if job is not None:
                            jobs.append(job)
                else:
                    still_open.append(tgt)
            open_fallbacks[:] = still_open
            if jobs:
                if cv:
                    _run_cv_batch(
                        jobs, buf, self.folds, plan.alpha, plan.l1_ratio,
                        plan.cv_folds, self._batch_cap,
                    )
                else:
                    n_intercept_only += self.backend.run_fit_batch(
                        jobs, buf, self.folds, plan.alpha, plan.l1_ratio, plan.cv_folds,
                        s_true, s_cv, self._batch_cap,
                    )

        region = _region_for(source, chrom)
        for block in source.iter_variant_blocks(region=region):
            info = block.variant_info
            dos = block.dosages
            ids = info["variant_id"].to_numpy()
            positions = info["position"].to_numpy()
            chip_cols, chip_pidx, chip_pos, chip_af = [], [], [], []
            for j in range(len(ids)):
                sid = ids[j]
                is_chip = sid in plan.chip_ids
                members = plan.region_members.get(sid)
                fb = plan.fallback_targets.get(sid)
                obs = plan.observed.get(sid)
                if not (is_chip or members or fb is not None or obs is not None):
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
                    has_obs = True
                if members:  # missing PRS variant → accumulate into its region's S_R
                    for region_idx, beta, flip in members:
                        x_eff, _, _ = _prepare_column(raw, flip=flip, folds=self.folds)
                        sr = open_sr.get(region_idx)
                        if sr is None:
                            sr = np.zeros(self.folds.n, dtype=np.float64)
                            open_sr[region_idx] = sr
                        sr += beta * x_eff
                if fb is not None:  # observed-variant fallback (effect-oriented target)
                    col_perm, af, _ = _prepare_column(raw, flip=fb.effect_flip, folds=self.folds)
                    open_fallbacks.append(_OpenTarget(sid, fb, col_perm, af, is_fallback=True))
                frontier = max(frontier, pos)
            # One batched band-Gram update per stream block (GPU: GEMM accumulation); the band
            # is only read in close_ready, so this matches the per-column path exactly.
            if chip_cols:
                buf.add_batch(chip_cols, chip_pidx, chip_pos, chip_af)
            close_ready(force=False)
            self._evict(buf, chrom_regions, region_ptr, frontier)

        close_ready(force=True)
        buf.clear()
        return _ProjectChromPartial(
            chrom=str(chrom),
            region_models=region_models,
            fallback_models=fallback_models,
            failures=failures,
            s_true=s_true,
            s_cv=s_cv,
            has_obs=has_obs,
            n_intercept_only=n_intercept_only,
            cv_collector=cv_collector,
        )

    def _evict(self, buf, chrom_regions, region_ptr, frontier) -> None:
        """Evict chip columns below the earliest still-open region's start, but never
        above ``frontier - 2W`` (which the observed-fallback ±W windows still need)."""
        floor = frontier - 2 * self.W
        if region_ptr < len(chrom_regions):
            floor = min(floor, self.plan.regions[chrom_regions[region_ptr]].start)
        buf.evict_below(floor)

    # -- job builders -------------------------------------------------------
    def _region_predictors(self, region: RegionSpec) -> np.ndarray:
        """Platform indices in [start, end]; under max_predictors, closest to centre.

        Reproduces ``projection_trainer._find_platform_variants_in_region`` exactly
        (same mask, same centre, same ``np.argsort`` order under truncation)."""
        mask = (
            (self._pv_chrom_norm == region.chromosome)
            & (self._pv_pos >= region.start)
            & (self._pv_pos <= region.end)
        )
        idx = np.where(mask)[0]
        mp = self.plan.max_predictors
        if mp is not None and len(idx) > mp:
            center = (region.start + region.end) // 2
            distances = np.abs(self._pv_pos[idx] - center)
            idx = idx[np.argsort(distances)[:mp]]
        return idx

    def _region_job(
        self, region, s_r, buf, region_models, failures, cv_collector
    ) -> Optional[_FitJob]:
        try:
            pred_idx = self._region_predictors(region)
        except Exception as exc:  # noqa: BLE001
            failures[region.region_id] = f"{type(exc).__name__}: {exc}"
            return None
        store = (
            self._cv_region_storer(region, s_r, cv_collector)
            if cv_collector is not None
            else self._region_storer(region, s_r, region_models)
        )
        return _FitJob(
            col=s_r, pred_idx=pred_idx, calib_coef=1.0, is_calibrating=True,
            store=store,
            fail=self._failer(region.region_id, failures),
        )

    def _fallback_job(self, tgt, fallback_models, fb_failures) -> Optional[_FitJob]:
        spec = tgt.spec
        try:
            win = self.chrom_index.window(
                spec.chromosome, spec.position, window_size=self.W,
                exclude_target=True, max_variants=self.plan.max_predictors,
            )
        except Exception as exc:  # noqa: BLE001
            fb_failures[spec.prs_variant_id] = f"{type(exc).__name__}: {exc}"
            return None
        return _FitJob(
            col=tgt.col, pred_idx=win.variant_indices, calib_coef=float(spec.beta),
            is_calibrating=False,  # fallbacks train a model but do not touch calibration
            store=self._fallback_storer(spec, tgt.af, fallback_models),
            fail=self._failer(spec.prs_variant_id, fb_failures),
        )

    def _region_storer(self, region, s_r, region_models):
        def store(result, pred_idx, pred_af):
            self.diag_var += float(result.cv_mse)
            region_models[region.region_id] = self._to_region_model(
                region, s_r, result, pred_idx, pred_af
            )
            self._has_terms = True
        return store

    def _fallback_storer(self, spec, af, dest):
        def store(result, pred_idx, pred_af):
            dest[spec.prs_variant_id] = self._to_fallback_model(spec, af, result, pred_idx, pred_af)
        return store

    @staticmethod
    def _failer(key, fmap):
        def fail(exc):
            fmap[key] = f"{type(exc).__name__}: {exc}"
        return fail

    # -- model construction -------------------------------------------------
    def _to_region_model(self, region, s_r, result, pred_idx, pred_af) -> ProjectionRegionModel:
        return ProjectionRegionModel(
            region_id=region.region_id,
            chromosome=str(region.chromosome),
            start=int(region.start),
            end=int(region.end),
            prs_variant_ids=list(region.prs_variant_ids),
            betas=np.asarray(region.betas, dtype=np.float64).copy(),
            predictor_variant_ids=self._pv_id[pred_idx].tolist(),
            coefficients=np.asarray(result.coefficients).copy(),
            intercept=float(result.intercept),
            cv_mse=float(result.cv_mse),
            cv_r2=float(result.cv_r2),
            is_intercept_only=bool(result.is_intercept_only),
            mean_prs_contribution=float(s_r.mean()),
            predictor_allele_frequencies=np.asarray(pred_af, dtype=np.float64),
            predictor_chromosomes=self._pv_chrom[pred_idx].tolist(),
            predictor_positions=self._pv_pos[pred_idx].tolist(),
            predictor_counted_alleles=self._pv_alt[pred_idx].tolist(),
            predictor_other_alleles=self._pv_ref[pred_idx].tolist(),
            prs_positions=list(region.prs_positions),
            prs_effect_alleles=list(region.prs_effect_alleles),
            prs_other_alleles=list(region.prs_other_alleles),
            target_variance=float(s_r.var()),
        )

    def _to_fallback_model(self, spec, af, result, pred_idx, pred_af) -> ImputedVariantModel:
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
            predictor_allele_frequencies=np.asarray(pred_af, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Metadata-only harmonization → ProjectionStreamPlan (no dosage matrix needed).
# ---------------------------------------------------------------------------
def build_projection_stream_plan(
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
    """Reproduce the dense projection Steps 7–9 (partition → allele-reclassify →
    platform match → region decomposition) from reference *metadata* only, using
    ``would_resolve`` instead of indexing the dosage matrix. Returns
    ``(plan, drop_reasons)``; drop reasons match the dense granularity.

    ``exclude_ambiguous`` is not applied here (its AF-based QC would fold into the
    streaming pass — a follow-up, guarded at the orchestrator).
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

    # Observed calibration terms + imputation-style fallbacks (mirrors dense Step 11).
    observed: Dict[str, ObservedVar] = {}
    fallback_targets: Dict[str, TargetVar] = {}
    observed_prs_ids: set = set()
    fallback_no_target_ids: set = set()
    # Missing variants that resolve → the target matrix for region decomposition.
    missing_records: List[dict] = []  # {prs_vid, chrom, pos, ref_vid, beta, flip, eff, oth}
    drop_reasons: Dict[str, str] = {}
    for i in range(len(p_vid)):
        vid = p_vid[i]
        wr = resolver.would_resolve(p_chrom[i], int(p_pos[i]), p_eff[i], _other(i))
        if vid in observed_ids:
            observed_prs_ids.add(vid)
            if wr is None:
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
            missing_records.append(dict(
                prs_vid=str(vid), chrom=str(p_chrom[i]), pos=int(p_pos[i]),
                ref_vid=gv_ids[ref_idx], beta=float(p_beta[i]), flip=flip,
                eff=str(p_eff[i]), oth=_other(i),
            ))

    regions, region_members = _decompose_regions(missing_records, window_size)

    plan = ProjectionStreamPlan(
        sample_ids=list(sample_ids), platform_variant_info=platform_info,
        chip_ids=chip_ids, regions=regions, region_members=region_members,
        observed=observed, fallback_targets=fallback_targets, window_size=window_size,
        max_predictors=max_predictors, alpha=alpha, l1_ratio=l1_ratio,
        cv_folds=cv_folds, random_state=random_state,
        observed_prs_ids=observed_prs_ids, fallback_no_target_ids=fallback_no_target_ids,
    )
    return plan, drop_reasons


def _decompose_regions(missing_records, window_size):
    """Merge the resolved missing variants' ±W windows into regions (dense parity via
    ``merge_variant_windows``) and map each ref row to its region(s) for S_R streaming."""
    if not missing_records:
        return [], {}
    missing_df = pd.DataFrame({
        "variant_id": [r["prs_vid"] for r in missing_records],
        "chromosome": [r["chrom"] for r in missing_records],
        "position": [r["pos"] for r in missing_records],
    })
    decomp = merge_variant_windows(missing_df, window_size=window_size)
    regions: List[RegionSpec] = []
    region_members: Dict[str, List[Tuple[int, float, bool]]] = {}
    for region_idx, gr in enumerate(decomp.regions):
        members = [missing_records[m] for m in gr.prs_variant_indices]
        regions.append(RegionSpec(
            region_id=f"chr{gr.chromosome}:{gr.start}-{gr.end}",
            chromosome=str(gr.chromosome), start=int(gr.start), end=int(gr.end),
            prs_variant_ids=[m["prs_vid"] for m in members],
            betas=np.array([m["beta"] for m in members], dtype=np.float64),
            prs_positions=[m["pos"] for m in members],
            prs_effect_alleles=[m["eff"] for m in members],
            prs_other_alleles=[m["oth"] for m in members],
        ))
        for m in members:
            region_members.setdefault(m["ref_vid"], []).append(
                (region_idx, m["beta"], m["flip"])
            )
    return regions, region_members


def streaming_fit_projection(
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
    """End-to-end streaming projection: scan metadata, harmonize → regions, fit, accumulate.

    Returns ``(result, plan, drop_reasons)``. Calibration is finalized separately via
    ``evaluation.streaming_calibration.finalize_projection_calibration``.
    """
    from imputed_prs.compute.sufficient_stats import collect_reference_variant_info
    from imputed_prs.core.harmonizer import _normalize_chromosome

    chroms = sorted(
        {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
        key=_chrom_sort_key,
    )
    ref_info = collect_reference_variant_info(source, chroms)
    plan, drop_reasons = build_projection_stream_plan(
        ref_info, prs_df, platform_variant_set, sample_ids=source.sample_ids,
        window_size=window_size, max_predictors=max_predictors, alpha=alpha,
        l1_ratio=l1_ratio, cv_folds=cv_folds, random_state=random_state,
    )
    result = StreamingProjectionFitter(plan, device=device).run(source)
    return result, plan, drop_reasons
