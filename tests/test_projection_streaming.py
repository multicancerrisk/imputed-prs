"""M3 oracle: streaming projection fitter vs the dense legacy region path.

Builds a synthetic reference panel + a fake block-yielding source, then asserts the
streaming sufficient-statistics projection fitter reproduces, for every merged region,
the legacy ``fit_single_region_model`` (coefficients, intercept, cv_r2, predictor
order) and the dense projection calibration accumulators (s_true, s_cv). A small window
makes the region-scoped buffer actually evict across blocks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from imputed_prs.compute.projection_stream import (
    StreamingProjectionFitter,
    build_projection_stream_plan,
    streaming_fit_projection,
)
from imputed_prs.core.regions import merge_variant_windows
from imputed_prs.models.projection import fit_single_region_model
from imputed_prs.models.projection_trainer import _find_platform_variants_in_region


# --- Fake streaming source (mirrors test_sufficient_stats) -----------------
@dataclass
class _Block:
    variant_info: pd.DataFrame
    dosages: np.ndarray

    @property
    def n_variants(self):
        return self.dosages.shape[1]


class FakeSource:
    def __init__(self, variant_info, dosage_matrix, sample_ids, block_size=5, contigs=None):
        self._info = variant_info.reset_index(drop=True)
        self._dos = dosage_matrix
        self._sample_ids = list(sample_ids)
        self._block_size = block_size
        self._contigs = contigs

    @property
    def sample_ids(self):
        return self._sample_ids

    @property
    def contigs(self):
        return self._contigs

    def iter_variant_blocks(self, region=None, block_size=None):
        bs = block_size or self._block_size
        info = self._info
        if region is not None:
            rows = np.nonzero((info["chromosome"].astype(str) == str(region)).to_numpy())[0]
        else:
            rows = np.arange(len(info))
        rows = rows[np.argsort(info["position"].to_numpy()[rows], kind="stable")]
        for start in range(0, len(rows), bs):
            sel = rows[start : start + bs]
            yield _Block(info.iloc[sel].reset_index(drop=True), self._dos[:, sel])


# --- Synthetic panel -------------------------------------------------------
def build_panel(n_samples=300, seed=11):
    rng = np.random.RandomState(seed)
    records = []
    for chrom in ("1", "2"):
        for i in range(24):
            records.append(dict(
                variant_id=f"v{chrom}_{i}", chromosome=chrom,
                position=100_000 * (i + 1), ref_allele="A", alt_allele="G",
            ))
    info = pd.DataFrame(records)
    nv = len(info)
    freqs = rng.uniform(0.15, 0.85, size=nv)
    dosage = rng.binomial(2, freqs, size=(n_samples, nv)).astype(np.float32)
    sids = [f"s{i}" for i in range(n_samples)]
    return info, dosage, sids


def build_prs_and_platform(info, seed, all_missing=False):
    """PRS = every even-indexed variant. Chip is chosen so PRS variants are missing
    (``all_missing`` → chip is the odd variants, disjoint from PRS; else chip drops
    only the i%4==0 evens, leaving a mix of observed/missing PRS)."""
    rng = np.random.RandomState(seed)
    nv = len(info)
    prs_rows = list(range(0, nv, 2))
    if all_missing:
        chip_rows = [i for i in range(nv) if i % 2 == 1]  # disjoint → all PRS missing
    else:
        chip_rows = [i for i in range(nv) if i % 4 != 0]  # i%4==0 evens → missing PRS
    platform_ids = [info.iloc[r]["variant_id"] for r in chip_rows]
    recs = []
    for r in prs_rows:
        flip = (r % 3 == 0)
        recs.append(dict(
            variant_id=info.iloc[r]["variant_id"], chromosome=str(info.iloc[r]["chromosome"]),
            position=int(info.iloc[r]["position"]),
            effect_allele=("A" if flip else "G"), other_allele=("G" if flip else "A"),
            beta=float(rng.uniform(-0.6, 0.6)),
        ))
    return pd.DataFrame(recs), set(platform_ids), chip_rows


def dense_oracle(info, dosage, prs_df, platform_ids, plan, W, max_pred, alpha, l1, cv, seed):
    """Legacy per-region projection fit + dense calibration on the same panel.

    Predictors are selected from ``plan.platform_variant_info`` (the same set-iteration
    order the streaming plan and the real orchestrator use), so predictor identity and
    order match by construction — only the fit itself is under test."""
    observed_ids = {v for v in prs_df["variant_id"] if v in platform_ids}
    missing_ids = {v for v in prs_df["variant_id"] if v not in platform_ids}
    plat_info = plan.platform_variant_info
    id2row = {info.iloc[i]["variant_id"]: i for i in range(len(info))}

    def oriented(vid, flip):
        col = dosage[:, id2row[vid]].astype(np.float64)
        return (2.0 - col) if flip else col

    # Missing target matrix + region decomposition (dense).
    missing_prs = prs_df[prs_df["variant_id"].isin(missing_ids)].reset_index(drop=True)
    X_cols = []
    for _, r in missing_prs.iterrows():
        X_cols.append(oriented(r["variant_id"], r["effect_allele"] == "A"))
    X = np.column_stack(X_cols) if X_cols else np.empty((dosage.shape[0], 0))
    decomp = merge_variant_windows(missing_prs, window_size=W)
    # Z aligned to plan.platform_variant_info column order.
    Z = (np.column_stack([oriented(v, False) for v in plat_info["variant_id"]])
         if len(plat_info) else np.empty((dosage.shape[0], 0)))

    region_models, cv_preds = {}, {}
    for gr in decomp.regions:
        idxs = gr.prs_variant_indices
        betas = missing_prs.iloc[idxs]["beta"].to_numpy(dtype=np.float64)
        target = X[:, idxs] @ betas
        pred_ids, plat_idx = _find_platform_variants_in_region(gr, plat_info, max_pred)
        preds = Z[:, plat_idx] if len(plat_idx) else np.empty((dosage.shape[0], 0))
        res = fit_single_region_model(target, preds, l1_ratio=l1, alpha=alpha,
                                      cv_folds=cv, random_state=seed)
        rid = f"chr{gr.chromosome}:{gr.start}-{gr.end}"
        region_models[rid] = (res, pred_ids, target)
        cv_preds[rid] = res.cv_predictions

    # Dense calibration (natural order).
    n = dosage.shape[0]
    s_true = np.zeros(n)
    s_cv = np.zeros(n)
    for _, r in prs_df.iterrows():
        vid = r["variant_id"]
        flip = (r["effect_allele"] == "A")
        x_eff = oriented(vid, flip)
        s_true += r["beta"] * x_eff
        if vid in observed_ids:
            s_cv += r["beta"] * x_eff
    for rid, cvp in cv_preds.items():
        s_cv += cvp
    return region_models, s_true, s_cv, observed_ids


@pytest.mark.parametrize(
    "all_missing,max_pred,W,alpha,l1",
    [
        (True, None, 250_000, 0.01, 0.5),   # pure regions, no observed
        (False, None, 250_000, 0.01, 0.5),  # mixed observed + missing
        (False, 3, 300_000, 0.05, 0.5),     # max_predictors truncation (distance order)
    ],
)
def test_streaming_projection_matches_dense(all_missing, max_pred, W, alpha, l1):
    seed = 11
    info, dosage, sids = build_panel(seed=seed)
    prs_df, platform_ids, chip_rows = build_prs_and_platform(info, seed, all_missing)
    ref_info = info.copy()

    plan, drops = build_projection_stream_plan(
        ref_info, prs_df, platform_ids, sample_ids=sids, window_size=W,
        max_predictors=max_pred, alpha=alpha, l1_ratio=l1, cv_folds=5, random_state=seed,
    )
    result = StreamingProjectionFitter(plan).run(
        FakeSource(info, dosage, sids, block_size=5)
    )

    dense_models, dense_s_true, dense_s_cv, observed_ids = dense_oracle(
        info, dosage, prs_df, platform_ids, plan, W, max_pred, alpha, l1, 5, seed
    )

    assert result.failures == {}
    assert set(result.region_models) == set(dense_models)
    assert len(dense_models) > 0

    plat_info = plan.platform_variant_info
    for rid, (dres, dpred_ids, _target) in dense_models.items():
        m = result.region_models[rid]
        assert list(m.predictor_variant_ids) == list(dpred_ids)
        np.testing.assert_allclose(m.coefficients, dres.coefficients, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(m.intercept, dres.intercept, atol=1e-7)
        np.testing.assert_allclose(m.cv_r2, dres.cv_r2, atol=1e-5)
        assert m.is_intercept_only == dres.is_intercept_only

    perm = StreamingProjectionFitter(plan).folds.perm
    np.testing.assert_allclose(result.s_true, dense_s_true[perm], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(result.s_cv, dense_s_cv[perm], atol=1e-6, rtol=1e-5)

    # Observed variants each get an imputation-style fallback model.
    if not all_missing:
        assert len(observed_ids) > 0
        assert set(result.fallback_models) == observed_ids


def test_streaming_fit_projection_endtoend():
    """The end-to-end entry point runs (metadata scan → harmonize → fit) via a source."""
    seed = 5
    info, dosage, sids = build_panel(seed=seed)
    prs_df, platform_ids, _ = build_prs_and_platform(info, seed, all_missing=False)
    src = FakeSource(info, dosage, sids, block_size=6)
    result, plan, drops = streaming_fit_projection(
        src, prs_df, platform_ids, window_size=250_000, alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=seed,
    )
    assert result.n_regions_trained > 0
    assert result.has_calibration_terms
    assert result.diag_var >= 0.0
