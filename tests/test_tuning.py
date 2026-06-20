"""Tests for the hyperparameter tuning module.

Covers the P3.2 contract: tuning runs on the *same* local-window (imputation) and
region (projection) matrices used in training, over a bounded, stratified sample,
so cost scales with the sample rather than the chip.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.harmonizer import filter_to_local_window
from imputed_prs.core.regions import merge_variant_windows
from imputed_prs.core.types import GridSearchResult
from imputed_prs.models.elastic_net import fit_single_variant_model
from imputed_prs.models.projection import fit_single_region_model
from imputed_prs.models.projection_trainer import _find_platform_variants_in_region
from imputed_prs.models.tuning import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _build_local_window_datasets,
    global_hyperparameter_search,
    projection_hyperparameter_search,
    select_stratified_sample,
    tune_single_variant_model,
)

OPTUNA_AVAILABLE = importlib.util.find_spec("optuna") is not None


# ---------------------------------------------------------------------------
# Fixtures: data laid out across two chromosomes so local windows are meaningful
# ---------------------------------------------------------------------------


@pytest.fixture
def imputation_data():
    """(Z, X, missing_variant_info, platform_variant_info) across chr1 + chr2."""
    rng = np.random.default_rng(42)
    n_samples, n_platform, n_missing = 300, 40, 12
    Z = rng.binomial(2, 0.3, (n_samples, n_platform)).astype(float)
    X = rng.binomial(2, 0.25, (n_samples, n_missing)).astype(float)
    platform_variant_info = pd.DataFrame({
        "variant_id": [f"p{i}" for i in range(n_platform)],
        "chromosome": ["1"] * 20 + ["2"] * 20,
        "position": list(range(100_000, 100_000 + 20 * 50_000, 50_000)) * 2,
        "ref_allele": ["A"] * n_platform,
        "alt_allele": ["G"] * n_platform,
    })
    missing_variant_info = pd.DataFrame({
        "variant_id": [f"m{i}" for i in range(n_missing)],
        "chromosome": ["1"] * 6 + ["2"] * 6,
        "position": list(range(125_000, 125_000 + 6 * 100_000, 100_000)) * 2,
        "effect_allele": ["G"] * n_missing,
        "other_allele": ["A"] * n_missing,
        "beta": list(np.linspace(-0.5, 0.5, n_missing)),
    })
    return Z, X, missing_variant_info, platform_variant_info


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------


class TestSelectStratifiedSample:
    def test_deterministic(self):
        keys = [("1", 0, 0)] * 10 + [("2", 1, 1)] * 10
        a = select_stratified_sample(keys, 6, random_state=123)
        b = select_stratified_sample(keys, 6, random_state=123)
        assert a == b
        assert len(a) == 6

    def test_returns_all_when_target_none(self):
        keys = [("1", 0, 0)] * 5
        assert select_stratified_sample(keys, None, random_state=1) == list(range(5))

    def test_returns_all_when_target_ge_n(self):
        keys = [("1", 0, 0), ("2", 0, 0)]
        assert select_stratified_sample(keys, 10, random_state=1) == [0, 1]

    def test_empty_or_nonpositive(self):
        assert select_stratified_sample([], 5) == []
        assert select_stratified_sample([("1", 0, 0)] * 3, 0) == []
        assert select_stratified_sample([("1", 0, 0)] * 3, -2) == []

    def test_spread_across_strata(self):
        # Two strata, ample target -> both represented (not just the first block).
        keys = [("1", 0, 0)] * 10 + [("2", 1, 1)] * 10
        sample = select_stratified_sample(keys, 4, random_state=7)
        assert any(i < 10 for i in sample)
        assert any(i >= 10 for i in sample)

    def test_none_random_state_is_deterministic(self):
        keys = [("1", 0, 0)] * 6 + [("2", 1, 1)] * 6
        a = select_stratified_sample(keys, 5, random_state=None)
        b = select_stratified_sample(keys, 5, random_state=None)
        assert a == b
        assert len(a) == 5


# ---------------------------------------------------------------------------
# Dataset builder: tuning matrices must equal training matrices
# ---------------------------------------------------------------------------


class TestBuildLocalWindowDatasets:
    def test_matches_filter_to_local_window(self, imputation_data):
        """Each built dataset equals the trainer's exact local-window slice."""
        Z, X, mvi, pvi = imputation_data
        window_size, max_predictors = 1_000_000, None
        sample_indices = list(range(len(mvi)))
        datasets, kept = _build_local_window_datasets(
            Z, X, mvi, pvi, window_size, max_predictors, sample_indices
        )
        assert kept == sample_indices
        for ds_idx, col in enumerate(kept):
            wr = filter_to_local_window(
                target_chrom=str(mvi.iloc[col]["chromosome"]),
                target_pos=int(mvi.iloc[col]["position"]),
                variant_info=pvi,
                window_size=window_size,
                exclude_target=True,
                max_variants=max_predictors,
            )
            if wr.n_variants > 0:
                expected = Z[:, wr.variant_indices]
            else:
                expected = np.empty((Z.shape[0], 0))
            np.testing.assert_array_equal(datasets[ds_idx][0], expected)
            np.testing.assert_array_equal(datasets[ds_idx][1], X[:, col])

    def test_windows_are_not_full_chip(self, imputation_data):
        """A local window must use fewer predictors than the whole Z matrix."""
        Z, X, mvi, pvi = imputation_data
        datasets, _ = _build_local_window_datasets(
            Z, X, mvi, pvi, 1_000_000, None, list(range(len(mvi)))
        )
        # Each variant only sees its own chromosome's platform variants (20), not 40.
        assert all(d[0].shape[1] < Z.shape[1] for d in datasets)


# ---------------------------------------------------------------------------
# Imputation global search
# ---------------------------------------------------------------------------


class TestGlobalHyperparameterSearch:
    def test_basic(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        res = global_hyperparameter_search(
            Z, X, mvi, pvi, window_size=1_000_000, random_state=42
        )
        assert isinstance(res, GridSearchResult)
        assert res.best_l1_ratio in DEFAULT_L1_RATIOS
        assert res.best_alpha in DEFAULT_ALPHAS
        assert res.n_variants_sampled == len(mvi)  # None cap -> all variants
        assert len(res.grid_results) == len(DEFAULT_L1_RATIOS) * len(DEFAULT_ALPHAS)

    def test_custom_grids(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        res = global_hyperparameter_search(
            Z, X, mvi, pvi, window_size=1_000_000,
            l1_ratios=[0.2, 0.8], alphas=[0.005, 0.05], random_state=1
        )
        assert res.best_l1_ratio in [0.2, 0.8]
        assert res.best_alpha in [0.005, 0.05]
        assert len(res.grid_results) == 4

    def test_max_tuning_variants_caps_sample(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        res = global_hyperparameter_search(
            Z, X, mvi, pvi, window_size=1_000_000,
            max_tuning_variants=5, random_state=42
        )
        assert res.n_variants_sampled == 5

    def test_reproducible(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        kw = dict(window_size=1_000_000, max_tuning_variants=6, random_state=99)
        a = global_hyperparameter_search(Z, X, mvi, pvi, **kw)
        b = global_hyperparameter_search(Z, X, mvi, pvi, **kw)
        assert a.best_l1_ratio == b.best_l1_ratio
        assert a.best_alpha == b.best_alpha
        assert a.grid_results == b.grid_results

    def test_cost_scales_with_sample_not_chip(self, imputation_data, monkeypatch):
        """Fit count == grid x sampled, NOT grid x all-missing."""
        Z, X, mvi, pvi = imputation_data
        import imputed_prs.models.tuning as tuning_mod
        real = tuning_mod.fit_single_variant_model
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(tuning_mod, "fit_single_variant_model", counting)
        res = global_hyperparameter_search(
            Z, X, mvi, pvi, window_size=1_000_000,
            max_tuning_variants=4, l1_ratios=[0.1, 0.5], alphas=[0.01, 0.1],
            random_state=0,
        )
        assert res.n_variants_sampled == 4
        assert calls["n"] == 2 * 2 * 4  # G x sampled
        assert calls["n"] != 2 * 2 * len(mvi)  # not G x all-missing (12)

    def test_no_predictors_returns_inf(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        empty_Z = np.empty((Z.shape[0], 0))
        empty_pvi = pvi.iloc[0:0]
        res = global_hyperparameter_search(
            empty_Z, X, mvi, empty_pvi, window_size=1_000_000, random_state=1
        )
        assert res.best_mean_cv_mse == float("inf")
        assert res.best_l1_ratio == DEFAULT_L1_RATIOS[0]
        assert res.best_alpha == DEFAULT_ALPHAS[0]
        assert res.n_variants_failed == res.n_variants_sampled

    def test_single_variant(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        res = global_hyperparameter_search(
            Z, X[:, :1], mvi.iloc[:1], pvi, window_size=1_000_000, random_state=1
        )
        assert res.n_variants_sampled == 1

    def test_shape_mismatch_raises(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        with pytest.raises(ValidationError):
            global_hyperparameter_search(
                Z[:50], X, mvi, pvi, window_size=1_000_000
            )

    def test_empty_grids_raise(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        with pytest.raises(ValidationError):
            global_hyperparameter_search(
                Z, X, mvi, pvi, window_size=1_000_000, l1_ratios=[]
            )
        with pytest.raises(ValidationError):
            global_hyperparameter_search(
                Z, X, mvi, pvi, window_size=1_000_000, alphas=[]
            )

    def test_invalid_grid_values_raise(self, imputation_data):
        Z, X, mvi, pvi = imputation_data
        with pytest.raises(ValidationError):
            global_hyperparameter_search(
                Z, X, mvi, pvi, window_size=1_000_000, l1_ratios=[1.5]
            )
        with pytest.raises(ValidationError):
            global_hyperparameter_search(
                Z, X, mvi, pvi, window_size=1_000_000, alphas=[-0.1]
            )


# ---------------------------------------------------------------------------
# Per-variant tuning helper
# ---------------------------------------------------------------------------


class TestTuneSingleVariantModel:
    def test_returns_grid_member(self, imputation_data):
        Z, X, _, _ = imputation_data
        result = tune_single_variant_model(
            X[:, 0], Z[:, :6],
            l1_ratios=[0.1, 0.9], alphas=[0.001, 0.1],
            cv_folds=5, random_state=42,
        )
        assert result.l1_ratio in [0.1, 0.9]
        assert result.alpha in [0.001, 0.1]

    def test_intercept_only_when_no_predictors(self, imputation_data):
        Z, X, _, _ = imputation_data
        result = tune_single_variant_model(
            X[:, 0], np.empty((X.shape[0], 0)), random_state=42
        )
        assert result.is_intercept_only

    def test_picks_lowest_cv_mse(self, imputation_data):
        """The chosen result's cv_mse is the best non-intercept-only grid fit."""
        Z, X, _, _ = imputation_data
        target, pred = X[:, 0], Z[:, :6]
        l1s, alphas = [0.1, 0.9], [0.001, 0.1]
        chosen = tune_single_variant_model(
            target, pred, l1_ratios=l1s, alphas=alphas, cv_folds=5, random_state=42
        )
        per_fit = []
        for l1 in l1s:
            for a in alphas:
                r = fit_single_variant_model(
                    target, pred, l1_ratio=l1, alpha=a, cv_folds=5, random_state=42
                )
                if not r.is_intercept_only:
                    per_fit.append(r.cv_mse)
        if per_fit:
            assert chosen.cv_mse == pytest.approx(min(per_fit))


# ---------------------------------------------------------------------------
# Projection region search
# ---------------------------------------------------------------------------


class TestProjectionHyperparameterSearch:
    def test_basic(self, imputation_data):
        Z, X, prs, pvi = imputation_data
        res = projection_hyperparameter_search(
            Z, X, prs, pvi, window_size=1_000_000, random_state=42
        )
        assert isinstance(res, GridSearchResult)
        assert res.best_l1_ratio in DEFAULT_L1_RATIOS
        assert res.best_alpha in DEFAULT_ALPHAS
        n_regions = len(merge_variant_windows(prs, 1_000_000).regions)
        assert res.n_variants_sampled == n_regions

    def test_matches_training_region_matrices(self, imputation_data):
        """best_mean_cv_mse equals a direct per-region fit_single_region_model mean."""
        Z, X, prs, pvi = imputation_data
        L, A, folds, rs = 0.5, 0.01, 3, 42
        res = projection_hyperparameter_search(
            Z, X, prs, pvi, window_size=1_000_000,
            l1_ratios=[L], alphas=[A], cv_folds=folds, random_state=rs,
        )
        # Reconstruct exactly what ProjectionRegionTrainer would score.
        prs_reset = prs.reset_index(drop=True)
        decomposition = merge_variant_windows(prs_reset, 1_000_000)
        mses = []
        for region in decomposition.regions:
            idx = region.prs_variant_indices
            betas = prs_reset.iloc[idx]["beta"].to_numpy(dtype=np.float64)
            target = X[:, idx] @ betas
            _, platform_idx = _find_platform_variants_in_region(region, pvi, None)
            predictor = (
                Z[:, platform_idx] if len(platform_idx) else np.empty((Z.shape[0], 0))
            )
            r = fit_single_region_model(
                target, predictor, l1_ratio=L, alpha=A, cv_folds=folds, random_state=rs
            )
            if not r.is_intercept_only:
                mses.append(r.cv_mse)
        expected = float(np.mean(mses))
        assert res.best_mean_cv_mse == pytest.approx(expected)

    def test_max_tuning_regions_caps_sample(self, imputation_data):
        Z, X, prs, pvi = imputation_data
        # 30kb windows -> 12 non-merging single-variant regions (each with
        # predictors), so a cap of 2 is well below the total.
        res = projection_hyperparameter_search(
            Z, X, prs, pvi, window_size=30_000,
            max_tuning_regions=2, random_state=42,
        )
        assert res.n_variants_sampled == 2

    def test_reproducible(self, imputation_data):
        Z, X, prs, pvi = imputation_data
        kw = dict(window_size=1_000_000, random_state=7)
        a = projection_hyperparameter_search(Z, X, prs, pvi, **kw)
        b = projection_hyperparameter_search(Z, X, prs, pvi, **kw)
        assert a.best_l1_ratio == b.best_l1_ratio
        assert a.best_alpha == b.best_alpha
        assert a.grid_results == b.grid_results

    def test_cost_scales_with_sampled_regions(self, imputation_data, monkeypatch):
        Z, X, prs, pvi = imputation_data
        import imputed_prs.models.projection as projection_mod
        real = projection_mod.fit_single_region_model
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(projection_mod, "fit_single_region_model", counting)
        res = projection_hyperparameter_search(
            Z, X, prs, pvi, window_size=30_000,
            max_tuning_regions=2, l1_ratios=[0.1, 0.5], alphas=[0.01, 0.1],
            random_state=0,
        )
        assert res.n_variants_sampled == 2
        assert calls["n"] == 2 * 2 * 2  # G x sampled regions

    def test_separate_grids(self, imputation_data):
        Z, X, prs, pvi = imputation_data
        res = projection_hyperparameter_search(
            Z, X, prs, pvi, window_size=1_000_000,
            l1_ratios=[0.2, 0.8], alphas=[0.005, 0.05], random_state=1,
        )
        assert res.best_l1_ratio in [0.2, 0.8]
        assert res.best_alpha in [0.005, 0.05]
        assert len(res.grid_results) == 4

    def test_shape_mismatch_raises(self, imputation_data):
        Z, X, prs, pvi = imputation_data
        with pytest.raises(ValidationError):
            projection_hyperparameter_search(Z[:50], X, prs, pvi, window_size=1_000_000)


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestImportPaths:
    def test_public_exports(self):
        from imputed_prs.models import (
            global_hyperparameter_search as g,
            projection_hyperparameter_search as p,
            select_stratified_sample as s,
            tune_single_variant_model as t,
        )
        assert all(callable(f) for f in (g, p, s, t))


# ---------------------------------------------------------------------------
# Optuna (only runs when optuna is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="optuna not installed")
class TestOptunaHyperparameterSearch:
    def test_basic(self, imputation_data):
        from imputed_prs.models.tuning import optuna_hyperparameter_search

        Z, X, mvi, pvi = imputation_data
        res = optuna_hyperparameter_search(
            Z, X, mvi, pvi, window_size=1_000_000,
            max_tuning_variants=5, n_trials=5, seed=42,
        )
        assert 0.0 <= res.best_l1_ratio <= 1.0
        assert res.best_alpha > 0
        assert res.n_trials == 5
        assert res.n_variants_sampled == 5

    def test_reproducible(self, imputation_data):
        from imputed_prs.models.tuning import optuna_hyperparameter_search

        Z, X, mvi, pvi = imputation_data
        kw = dict(window_size=1_000_000, max_tuning_variants=5, n_trials=5, seed=7)
        a = optuna_hyperparameter_search(Z, X, mvi, pvi, **kw)
        b = optuna_hyperparameter_search(Z, X, mvi, pvi, **kw)
        assert a.best_l1_ratio == b.best_l1_ratio
        assert a.best_alpha == b.best_alpha

    def test_shape_mismatch_raises(self, imputation_data):
        from imputed_prs.models.tuning import optuna_hyperparameter_search

        Z, X, mvi, pvi = imputation_data
        with pytest.raises(ValidationError):
            optuna_hyperparameter_search(Z[:50], X, mvi, pvi, window_size=1_000_000)
