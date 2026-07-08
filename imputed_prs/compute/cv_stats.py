"""Phase 6: single-pass leave-one-fold-out reference-CV orchestration.

Given a streaming ``GenotypeSource`` and a reference-CV outer partition, accumulate the
sufficient statistics **once** and assemble each training fold's models by the additive
subtraction ``S_full − S_fold(k)`` — replacing the ``k`` independent refit passes of
``ImputationEvaluator.cross_validate`` (and the projection analogue). The per-fold
models are then scored on their held-out fold by the Phase-5 vectorized numeric scorer.

Leaf module in the compute DAG: ``cv_stats → {sufficient_stats, projection_stream} →
gram_solve`` (never imported *by* those, to keep the dependency acyclic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from imputed_prs.core.types import (
    ImputedVariantModel,
    ProjectionRegionModel,
    VariantInfo,
)


def _cv_checkpoint(
    checkpoint_dir,
    *,
    predictor_ids,
    prs_terms,
    source,
    ref_info,
    fold_indices,
    solve_mode,
    window_size,
    max_predictors,
    cv_folds,
    random_state,
    alpha,
    l1_ratio,
):
    """Build the Phase-10 reference-CV ``CheckpointStore`` (mode='cv'), or ``None``.

    Reference CV forces the CPU buffer, so the key's device is always ``cpu``; ``fold_key``
    digests the outer partition so a different fold split lands in a different directory.
    """
    if checkpoint_dir is None:
        return None
    from imputed_prs.io.checkpoint import (
        CheckpointStore,
        fold_partition_key,
        make_checkpoint_key,
    )

    source_file = getattr(source, "path", None) or getattr(source, "_pgen_path", None)
    key = make_checkpoint_key(
        sample_ids=source.sample_ids,
        predictor_ids=predictor_ids,
        prs_terms=prs_terms,
        window_size=window_size,
        max_predictors=max_predictors,
        cv_folds=cv_folds,
        random_state=random_state,
        alpha=alpha,
        l1_ratio=l1_ratio,
        device="cpu",
        solve_mode=solve_mode,
        mode="cv",
        fold_key=fold_partition_key(fold_indices),
        source_file=source_file,
        n_variants=len(ref_info),
    )
    return CheckpointStore(checkpoint_dir, key)


@dataclass
class ReferenceCVModels:
    """Per-outer-fold models produced by one streaming reference-CV pass.

    ``fold_imputed_models[k]`` (imputation) or ``fold_region_models[k]`` (projection) are
    the units trained on **all samples except outer fold ``k``**, assembled by additive
    subtraction. ``observed_variants`` is fold-independent (the on-platform PRS terms,
    scored directly from held-out dosages). ``fold_indices`` is the outer partition used.
    """

    observed_variants: List[VariantInfo]
    fold_indices: List[np.ndarray]
    fold_imputed_models: Optional[Dict[int, List[ImputedVariantModel]]] = None
    fold_region_models: Optional[Dict[int, List[ProjectionRegionModel]]] = None
    failures: Dict[str, str] = field(default_factory=dict)


def _observed_variant_infos(
    prs_df: pd.DataFrame, observed_prs_ids: Sequence[str]
) -> List[VariantInfo]:
    """Fold-independent observed PRS terms (on-platform), in PRS order.

    No per-variant fallbacks are attached: the reference-CV evaluator scores observed
    terms directly from the held-out reference dosages (every variant is present), so the
    upload-time fallback models a normal fit builds are unnecessary here.
    """
    out: List[VariantInfo] = []
    subset = prs_df[prs_df["variant_id"].isin(set(observed_prs_ids))]
    for _, row in subset.iterrows():
        other_allele = row.get("other_allele")
        if other_allele is not None and pd.isna(other_allele):
            other_allele = None
        out.append(
            VariantInfo(
                variant_id=row["variant_id"],
                chromosome=str(row["chromosome"]),
                position=int(row["position"]),
                effect_allele=row["effect_allele"],
                other_allele=other_allele,
                beta=float(row["beta"]),
                fallback=None,
            )
        )
    return out


def streaming_reference_cv_impute(
    source,
    prs_df: pd.DataFrame,
    platform_variant_set,
    *,
    fold_indices: Sequence[np.ndarray],
    window_size: int = 1_000_000,
    max_predictors: Optional[int] = None,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    device: str = "cpu",
    n_workers: int = 1,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> ReferenceCVModels:
    """One streaming pass → per-outer-fold **imputation** models by additive subtraction.

    Builds the same harmonized stream plan a normal streaming fit does (reusing
    ``build_stream_plan``), then runs a single leave-one-fold-out pass
    (``StreamingImputationFitter.run_reference_cv``) over ``fold_indices`` — the
    reference-CV outer partition. Returns the per-fold imputation models plus the
    fold-independent observed terms.
    """
    from imputed_prs.compute.sufficient_stats import (
        GlobalFolds,
        StreamingImputationFitter,
        _chrom_sort_key,
        build_stream_plan,
        collect_reference_variant_info,
    )
    from imputed_prs.core.harmonizer import _normalize_chromosome

    chroms = sorted(
        {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
        key=_chrom_sort_key,
    )
    ref_info = collect_reference_variant_info(source, chroms)
    plan, _drop_reasons = build_stream_plan(
        ref_info,
        prs_df,
        platform_variant_set,
        sample_ids=source.sample_ids,
        window_size=window_size,
        max_predictors=max_predictors,
        alpha=alpha,
        l1_ratio=l1_ratio,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    outer_folds = GlobalFolds.from_partition(fold_indices)
    fitter = StreamingImputationFitter(plan, device=device)
    from imputed_prs.io.checkpoint import imputation_plan_terms

    predictor_ids, prs_terms = imputation_plan_terms(plan)
    checkpoint = _cv_checkpoint(
        checkpoint_dir,
        predictor_ids=predictor_ids,
        prs_terms=prs_terms,
        source=source,
        ref_info=ref_info,
        fold_indices=fold_indices,
        solve_mode=getattr(fitter, "_solve_mode", "default"),
        window_size=window_size,
        max_predictors=max_predictors,
        cv_folds=cv_folds,
        random_state=random_state,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    fold_models, failures = fitter.run_reference_cv(
        source, outer_folds, n_workers=n_workers, checkpoint=checkpoint
    )
    return ReferenceCVModels(
        observed_variants=_observed_variant_infos(prs_df, plan.observed_prs_ids),
        fold_indices=list(fold_indices),
        fold_imputed_models=fold_models,
        failures=failures,
    )


def streaming_reference_cv_project(
    source,
    prs_df: pd.DataFrame,
    platform_variant_set,
    *,
    fold_indices: Sequence[np.ndarray],
    window_size: int = 1_000_000,
    max_predictors: Optional[int] = None,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    device: str = "cpu",
    n_workers: int = 1,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> ReferenceCVModels:
    """One streaming pass → per-outer-fold **projection** region models by subtraction.

    Projection analogue of :func:`streaming_reference_cv_impute`: builds the projection
    stream plan (merged regions) and runs a single leave-one-fold-out pass
    (``StreamingProjectionFitter.run_reference_cv``) over the reference-CV outer partition.
    """
    from imputed_prs.compute.projection_stream import (
        StreamingProjectionFitter,
        build_projection_stream_plan,
    )
    from imputed_prs.compute.sufficient_stats import (
        GlobalFolds,
        _chrom_sort_key,
        collect_reference_variant_info,
    )
    from imputed_prs.core.harmonizer import _normalize_chromosome

    chroms = sorted(
        {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
        key=_chrom_sort_key,
    )
    ref_info = collect_reference_variant_info(source, chroms)
    plan, _drop_reasons = build_projection_stream_plan(
        ref_info,
        prs_df,
        platform_variant_set,
        sample_ids=source.sample_ids,
        window_size=window_size,
        max_predictors=max_predictors,
        alpha=alpha,
        l1_ratio=l1_ratio,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    outer_folds = GlobalFolds.from_partition(fold_indices)
    fitter = StreamingProjectionFitter(plan, device=device)
    from imputed_prs.io.checkpoint import projection_plan_terms

    predictor_ids, prs_terms = projection_plan_terms(plan)
    checkpoint = _cv_checkpoint(
        checkpoint_dir,
        predictor_ids=predictor_ids,
        prs_terms=prs_terms,
        source=source,
        ref_info=ref_info,
        fold_indices=fold_indices,
        solve_mode=getattr(fitter, "_solve_mode", "default"),
        window_size=window_size,
        max_predictors=max_predictors,
        cv_folds=cv_folds,
        random_state=random_state,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    fold_models, failures = fitter.run_reference_cv(
        source, outer_folds, n_workers=n_workers, checkpoint=checkpoint
    )
    return ReferenceCVModels(
        observed_variants=_observed_variant_infos(prs_df, plan.observed_prs_ids),
        fold_indices=list(fold_indices),
        fold_region_models=fold_models,
        failures=failures,
    )
