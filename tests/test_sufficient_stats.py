"""M2 oracle: streaming imputation fitter vs the dense legacy path.

Builds a synthetic reference panel + a fake block-yielding GenotypeSource, then
asserts the streaming sufficient-statistics fitter reproduces, for every missing
target, the legacy ``fit_single_variant_model`` (coefficients, intercept, cv_r2,
predictor order) and the dense calibration accumulators (s_true, s_cv). A small
window is used so the sliding buffer actually evicts across blocks.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from imputed_prs.compute.sufficient_stats import (
    GlobalFolds,
    ObservedVar,
    StreamPlan,
    StreamingImputationFitter,
    TargetVar,
    _ChipGramBuffer,
)
from imputed_prs.core.harmonizer import filter_to_local_window
from imputed_prs.models.elastic_net import fit_single_variant_model


# --- Fake streaming source -------------------------------------------------
@dataclass
class _Block:
    variant_info: pd.DataFrame
    dosages: np.ndarray

    @property
    def n_variants(self):
        return self.dosages.shape[1]


class FakeSource:
    """Yields position-sorted variant blocks per chromosome from an in-memory panel."""

    def __init__(self, variant_info, dosage_matrix, sample_ids, block_size=5):
        self._info = variant_info.reset_index(drop=True)
        self._dos = dosage_matrix
        self._sample_ids = list(sample_ids)
        self._block_size = block_size

    @property
    def sample_ids(self):
        return self._sample_ids

    def iter_variant_blocks(self, region=None, block_size=None):
        bs = block_size or self._block_size
        info = self._info
        if region is not None:
            mask = info["chromosome"].astype(str) == str(region)
            rows = np.nonzero(mask.to_numpy())[0]
        else:
            rows = np.arange(len(info))
        rows = rows[np.argsort(info["position"].to_numpy()[rows], kind="stable")]
        for start in range(0, len(rows), bs):
            sel = rows[start : start + bs]
            yield _Block(info.iloc[sel].reset_index(drop=True), self._dos[:, sel])


# --- Synthetic panel -------------------------------------------------------
def build_panel(n_samples=300, seed=7):
    rng = np.random.RandomState(seed)
    records = []
    for chrom in ("1", "2"):
        for i in range(20):
            records.append(
                dict(
                    variant_id=f"v{chrom}_{i}",
                    chromosome=chrom,
                    position=100_000 * (i + 1),
                    ref_allele="A",
                    alt_allele="G",
                )
            )
    info = pd.DataFrame(records)
    n_var = len(info)
    freqs = rng.uniform(0.15, 0.85, size=n_var)
    dosage = rng.binomial(2, freqs, size=(n_samples, n_var)).astype(np.float32)
    sample_ids = [f"s{i}" for i in range(n_samples)]
    return info, dosage, sample_ids


def build_plan_and_prs(info, dosage, sample_ids, W, max_pred, alpha, l1, seed):
    """Split panel into chip predictors + observed/missing PRS variants."""
    rng = np.random.RandomState(seed)
    n_var = len(info)
    # Chip = every variant whose index is not ≡ 2 (mod 5): ~80% are predictors.
    chip_rows = [i for i in range(n_var) if i % 5 != 2]
    # PRS = a spread of variants, some on-chip (observed), some off-chip (missing).
    prs_rows = list(range(0, n_var, 2))

    platform_info = info.iloc[chip_rows].reset_index(drop=True)
    chip_ids = {info.iloc[r]["variant_id"]: pi for pi, r in enumerate(chip_rows)}

    targets, observed, prs_specs = {}, {}, []
    for r in prs_rows:
        vid = info.iloc[r]["variant_id"]
        beta = float(rng.uniform(-0.6, 0.6))
        # Alternate effect allele between ALT ("G", no flip) and REF ("A", flip).
        flip = r % 4 == 0
        effect = "A" if flip else "G"
        other = "G" if flip else "A"
        prs_specs.append((r, vid, beta, flip, effect, other))
        if vid in chip_ids:
            observed[vid] = ObservedVar(beta=beta, effect_flip=flip)
        else:
            targets[vid] = TargetVar(
                prs_variant_id=vid, chromosome=str(info.iloc[r]["chromosome"]),
                position=int(info.iloc[r]["position"]), effect_allele=effect,
                other_allele=other, beta=beta, effect_flip=flip,
            )

    plan = StreamPlan(
        sample_ids=sample_ids, platform_variant_info=platform_info, chip_ids=chip_ids,
        targets=targets, observed=observed, window_size=W, max_predictors=max_pred,
        alpha=alpha, l1_ratio=l1, cv_folds=5, random_state=seed,
    )
    return plan, platform_info, chip_ids, prs_specs, chip_rows


def dense_oracle(info, dosage, plan, platform_info, chip_ids, prs_specs, chip_rows):
    """Legacy per-variant fit + dense calibration on the same panel."""
    Z = dosage[:, chip_rows].astype(np.float64)  # raw ALT predictors
    W, max_pred = plan.window_size, plan.max_predictors
    dense_models = {}
    cv_preds = {}
    for r, vid, beta, flip, effect, other in prs_specs:
        if vid not in plan.targets:
            continue
        target = dosage[:, r].astype(np.float64)
        y_eff = (2.0 - target) if flip else target
        win = filter_to_local_window(
            str(info.iloc[r]["chromosome"]), int(info.iloc[r]["position"]),
            platform_info, W, exclude_target=True, max_variants=max_pred,
        )
        preds = Z[:, win.variant_indices]
        res = fit_single_variant_model(
            y_eff, preds, l1_ratio=plan.l1_ratio, alpha=plan.alpha,
            cv_folds=plan.cv_folds, random_state=plan.random_state,
        )
        dense_models[vid] = (res, win.variant_indices)
        cv_preds[vid] = res.cv_predictions

    # Dense calibration (natural order).
    n = dosage.shape[0]
    s_true = np.zeros(n)
    s_cv = np.zeros(n)
    for r, vid, beta, flip, effect, other in prs_specs:
        col = dosage[:, r].astype(np.float64)
        x_eff = (2.0 - col) if flip else col
        if vid in plan.observed:
            s_true += beta * x_eff
            s_cv += beta * x_eff
        elif vid in dense_models:
            s_true += beta * x_eff
            s_cv += beta * cv_preds[vid]
    return dense_models, s_true, s_cv


def _fill_buffer(cols, afs, folds, lazy, cap=8):
    buf = _ChipGramBuffer(cols[0].shape[0], folds, capacity=cap, lazy_fold_gram=lazy)
    for j, c in enumerate(cols):
        buf.add(c, platform_idx=j, position=j * 10, af=afs[j])
    return buf


def test_lazy_fold_gram_matches_incremental():
    """Phase-3E band-limited per-fold Gram: the projection ``lazy_fold_gram`` buffer
    recomputes the full + per-fold Gram on-demand at ``gather`` and must return blocks
    identical to the incrementally-maintained buffer — across add / grow / evict — while
    never allocating the (K, cap, cap) per-fold tensor."""
    rng = np.random.RandomState(3)
    n, ncols, K = 160, 40, 5
    folds = GlobalFolds(n, K, random_state=11)
    cols, afs = [], []
    for j in range(ncols):
        raw = rng.randint(0, 3, size=n).astype(float)
        if j % 4 == 0:  # inject missingness → non-integer mean-imputed entries (float path)
            raw[rng.rand(n) < 0.1] = np.nan
            mask = np.isnan(raw)
            raw[mask] = np.nanmean(raw)
        cols.append(folds.permute(raw))
        afs.append(float(raw.mean() / 2))

    eager = _fill_buffer(cols, afs, folds, lazy=False)
    lazy = _fill_buffer(cols, afs, folds, lazy=True)

    # The lazy buffer must NOT hold the O(cap²)/O(K·cap²) Grams — that is the whole fix.
    assert lazy.Gfull is None and lazy.Ghold is None
    assert eager.Gfull is not None and eager.Ghold is not None

    def assert_gather_matches(pred):
        ie, ge = eager.gather(pred)
        il, gl = lazy.gather(pred)
        assert np.array_equal(ie, il)
        for key in ("G", "fold_G", "zsum", "zsqsum", "fold_zsum", "fold_zsqsum", "af"):
            np.testing.assert_allclose(gl[key], ge[key], atol=1e-9, rtol=1e-9)
        # G == Σ_k fold_G[k] exactly (same Zsub slice) ⇒ G − fold_G[k] is an exact train Gram.
        np.testing.assert_allclose(gl["G"], np.asarray(gl["fold_G"]).sum(0), atol=1e-9)

    assert_gather_matches([3, 17, 5, 28, 11, 0, 39])  # scattered order, spans grows
    eager.evict_below(150)
    lazy.evict_below(150)
    assert_gather_matches([20, 35, 16, 39])  # after a front eviction


def test_grow_and_underdetermined():
    """Dense window: >256 buffered chip cols (exercises the Gram buffer's _grow) and
    p > n (underdetermined), the regime where a naive buffer-resize corrupts columns.
    """
    rng = np.random.RandomState(1)
    n = 200
    recs = [
        dict(variant_id=f"d{i}", chromosome="1", position=1000 + i * 2000,
             ref_allele="A", alt_allele="G")
        for i in range(350)
    ]
    info = pd.DataFrame(recs)
    nv = len(info)
    dosage = rng.binomial(2, rng.uniform(0.2, 0.8, nv), size=(n, nv)).astype(np.float32)
    sids = [f"s{i}" for i in range(n)]
    chip_rows = [i for i in range(nv) if i % 7 != 0]  # ~300 predictors > n
    tr = 175
    vid = info.iloc[tr]["variant_id"]
    platform = info.iloc[chip_rows].reset_index(drop=True)
    chip_ids = {info.iloc[r]["variant_id"]: pi for pi, r in enumerate(chip_rows)}
    targets = {vid: TargetVar(vid, "1", int(info.iloc[tr]["position"]), "G", "A", 0.3, False)}
    plan = StreamPlan(
        sids, platform, chip_ids, targets, {}, window_size=1_000_000,
        max_predictors=None, alpha=0.01, l1_ratio=0.5, cv_folds=5, random_state=1,
    )
    res = StreamingImputationFitter(plan).run(FakeSource(info, dosage, sids, block_size=16))

    win = filter_to_local_window(
        "1", int(info.iloc[tr]["position"]), platform, 1_000_000,
        exclude_target=True, max_variants=None,
    )
    Z = dosage[:, chip_rows].astype(np.float64)
    den = fit_single_variant_model(
        dosage[:, tr].astype(np.float64), Z[:, win.variant_indices],
        l1_ratio=0.5, alpha=0.01, cv_folds=5, random_state=1,
    )
    assert res.failures == {}
    assert len(win.variant_indices) > 256  # confirms _grow fired
    np.testing.assert_allclose(res.models[vid].coefficients, den.coefficients, atol=1e-9)
    np.testing.assert_allclose(res.models[vid].intercept, den.intercept, atol=1e-9)


@pytest.mark.parametrize(
    "W,max_pred,alpha,l1",
    [(300_000, None, 0.01, 0.5), (300_000, 3, 0.01, 0.5), (500_000, None, 0.05, 0.5)],
)
def test_streaming_matches_dense(W, max_pred, alpha, l1):
    seed = 7
    info, dosage, sample_ids = build_panel(seed=seed)
    plan, platform_info, chip_ids, prs_specs, chip_rows = build_plan_and_prs(
        info, dosage, sample_ids, W, max_pred, alpha, l1, seed
    )

    fitter = StreamingImputationFitter(plan)
    result = fitter.run(FakeSource(info, dosage, sample_ids, block_size=5))

    dense_models, dense_s_true, dense_s_cv = dense_oracle(
        info, dosage, plan, platform_info, chip_ids, prs_specs, chip_rows
    )

    assert result.failures == {}
    assert set(result.models) == set(dense_models)

    for vid, (dense_res, dense_pred_idx) in dense_models.items():
        m = result.models[vid]
        # Predictor selection + order must match the legacy window exactly.
        assert m.predictor_variant_ids == platform_info["variant_id"].to_numpy()[
            dense_pred_idx
        ].tolist()
        np.testing.assert_allclose(
            m.coefficients, dense_res.coefficients, atol=1e-7, rtol=1e-5
        )
        np.testing.assert_allclose(m.intercept, dense_res.intercept, atol=1e-7)
        np.testing.assert_allclose(m.imputation_r2, dense_res.cv_r2, atol=1e-5)
        assert m.is_intercept_only == dense_res.is_intercept_only

    # Calibration accumulators (streaming is in permuted order → permute the oracle).
    perm = fitter.folds.perm
    np.testing.assert_allclose(result.s_true, dense_s_true[perm], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(result.s_cv, dense_s_cv[perm], atol=1e-6, rtol=1e-5)


def test_run_collect_then_solve_matches_run():
    """Phase 9 grid solve: collect the Gram blocks ONCE, then ``solve_collected(alpha, l1)``
    reproduces a single ``run()`` at that ``(alpha, l1)`` — for two different combos from
    the *same* collected blocks (the blocks are alpha/l1-independent). n=300 (<10K) keeps
    both on the per-target oracle, so coefficients / intercept / CV-R² match exactly.
    """
    info, dosage, sample_ids = build_panel(seed=7)
    W = 300_000
    combos = [(0.01, 0.5), (0.05, 0.9)]

    def src():
        return FakeSource(info, dosage, sample_ids, block_size=5)

    # Collect once (blocks depend on the window structure, not on alpha/l1). Any combo's
    # plan yields the same structure; use the first for the collecting fitter.
    plan0 = build_plan_and_prs(info, dosage, sample_ids, W, None, *combos[0], 7)[0]
    fitter = StreamingImputationFitter(plan0)
    collected = fitter.run_collect(src())
    assert collected  # non-empty

    for alpha, l1 in combos:
        plan = build_plan_and_prs(info, dosage, sample_ids, W, None, alpha, l1, 7)[0]
        single = StreamingImputationFitter(plan).run(src())
        grid = fitter.solve_collected(collected, alpha, l1)

        assert set(grid.models) == set(single.models)
        assert set(grid.fallback_models) == set(single.fallback_models)
        assert grid.n_intercept_only == single.n_intercept_only
        for vid, sm in single.models.items():
            gm = grid.models[vid]
            assert gm.predictor_variant_ids == sm.predictor_variant_ids
            np.testing.assert_allclose(gm.coefficients, sm.coefficients, atol=1e-9)
            np.testing.assert_allclose(gm.intercept, sm.intercept, atol=1e-9)
            np.testing.assert_allclose(gm.imputation_r2, sm.imputation_r2, atol=1e-9)
            assert gm.is_intercept_only == sm.is_intercept_only
