"""Phase 9 — opt-in persisted sufficient-statistics cache (io/stats_cache.py).

The cache stores the ``run_collect`` block list keyed on (reference, chip set, window
params) — everything the accumulation depends on, but NOT ``(alpha, l1)``. These tests
pin the four guarantees the wiring (Commit 6) relies on:

  * the key is deterministic and order-independent in the chip set;
  * a stored → loaded block list round-trips **exactly** (arrays ``array_equal``) and
    re-solves to the **same** coefficients / CV-R² as the un-cached blocks;
  * any changed key component (reference, chip set, window param) ⇒ miss ⇒ recompute;
  * a corrupt / partial / absent entry is tolerated silently (miss, never raises).
"""

import json

import numpy as np
import pytest

from imputed_prs.io import stats_cache
from imputed_prs.io.stats_cache import (
    clear_stats_cache,
    get_stats_cache_info,
    load_collected,
    make_stats_key,
    store_collected,
)
from imputed_prs.compute.sufficient_stats import StreamingImputationFitter
from tests.test_sufficient_stats import FakeSource, build_panel, build_plan_and_prs

pytestmark = pytest.mark.filterwarnings("ignore")

W = 300_000
KEY_KW = dict(window_size=W, max_predictors=None, cv_folds=5, random_state=7)


def _collect():
    """Stream a small two-chromosome panel once → its collected Gram blocks + the plan."""
    info, dosage, sample_ids = build_panel(seed=7)
    plan = build_plan_and_prs(info, dosage, sample_ids, W, None, 0.01, 0.5, 7)[0]
    fitter = StreamingImputationFitter(plan)
    collected = fitter.run_collect(FakeSource(info, dosage, sample_ids, block_size=5))
    return collected, plan, fitter


def _key_for(plan, **overrides):
    kw = dict(
        sample_ids=plan.sample_ids,
        chip_ids=list(plan.chip_ids),
        **KEY_KW,
    )
    kw.update(overrides)
    return make_stats_key(**kw)


# --- Key ---------------------------------------------------------------------
def test_key_deterministic_and_chipset_order_independent():
    ids = ["a", "b", "c", "d"]
    k1 = make_stats_key(sample_ids=["s0", "s1"], chip_ids=ids, **KEY_KW)
    k2 = make_stats_key(sample_ids=["s0", "s1"], chip_ids=list(reversed(ids)), **KEY_KW)
    assert k1.digest == k2.digest  # chip-set hash sorts → order-independent
    assert k1.chip_set_hash == k2.chip_set_hash
    # A genuinely different chip set flips the key.
    k3 = make_stats_key(sample_ids=["s0", "s1"], chip_ids=ids + ["e"], **KEY_KW)
    assert k3.digest != k1.digest


def test_key_components_all_matter():
    base = make_stats_key(sample_ids=["s0", "s1"], chip_ids=["a", "b"], **KEY_KW)
    variants = [
        dict(sample_ids=["s0", "s2"]),          # different sample set
        dict(chip_ids=["a", "c"]),              # different chip set
        dict(window_size=W + 1),                # window param
        dict(max_predictors=10),
        dict(cv_folds=10),
        dict(random_state=99),
        dict(fold_key="fold-3"),                # reference-CV partition
    ]
    for override in variants:
        kw = dict(sample_ids=["s0", "s1"], chip_ids=["a", "b"], **KEY_KW)
        kw.update(override)
        assert make_stats_key(**kw).digest != base.digest, override


def test_ref_digest_tracks_source_file(tmp_path):
    f = tmp_path / "ref.vcf"
    f.write_text("x")
    k1 = make_stats_key(sample_ids=["s0"], chip_ids=["a"], source_file=f, **KEY_KW)
    f.write_text("xxxxx")  # size + mtime change
    k2 = make_stats_key(sample_ids=["s0"], chip_ids=["a"], source_file=f, **KEY_KW)
    assert k1.digest != k2.digest
    # No source_file → the in-RAM degradation path still produces a stable key.
    k3 = make_stats_key(sample_ids=["s0"], chip_ids=["a"], n_variants=42, **KEY_KW)
    k4 = make_stats_key(sample_ids=["s0"], chip_ids=["a"], n_variants=42, **KEY_KW)
    assert k3.digest == k4.digest
    assert k3.digest != k1.digest


# --- Round-trip --------------------------------------------------------------
def test_store_load_blocks_exact(tmp_path):
    collected, plan, _ = _collect()
    assert collected
    # Multi-chromosome panel → exercises multi-shard store/load.
    assert len({c.spec.chromosome for c in collected}) >= 2

    key = _key_for(plan)
    assert store_collected(key, collected, cache_dir=tmp_path) is not None
    loaded = load_collected(key, cache_dir=tmp_path)
    assert loaded is not None
    assert len(loaded) == len(collected)

    for orig, got in zip(collected, loaded):
        assert got.spec == orig.spec
        assert got.is_fallback == orig.is_fallback
        assert got.af == orig.af
        assert np.array_equal(got.pred_idx, orig.pred_idx)
        assert np.array_equal(got.pred_af, orig.pred_af)
        bo, bg = orig.block, got.block
        assert bg.n == bo.n and bg.ysum == bo.ysum and bg.ysqsum == bo.ysqsum
        for attr in ("G", "c", "zsum", "zsqsum"):
            assert np.array_equal(getattr(bg, attr), getattr(bo, attr)), attr
        # Live fold scalars are ndarrays, reconstructed ones lists — compare by value.
        for attr in ("fold_n", "fold_ysum", "fold_ysqsum"):
            assert np.array_equal(getattr(bg, attr), getattr(bo, attr)), attr
        for attr in ("fold_G", "fold_c", "fold_zsum", "fold_zsqsum"):
            for a, b in zip(getattr(bg, attr), getattr(bo, attr)):
                assert np.array_equal(a, b), attr


def test_cached_blocks_resolve_identically(tmp_path):
    """The real deliverable: solving cache-loaded blocks == solving the live blocks,
    bit-for-bit, at two different grid points."""
    collected, plan, fitter = _collect()
    key = _key_for(plan)
    store_collected(key, collected, cache_dir=tmp_path)
    loaded = load_collected(key, cache_dir=tmp_path)
    assert loaded is not None

    for alpha, l1 in [(0.01, 0.5), (0.05, 0.9)]:
        live = fitter.solve_collected(collected, alpha, l1)
        warm = fitter.solve_collected(loaded, alpha, l1)
        assert set(warm.models) == set(live.models)
        assert warm.n_intercept_only == live.n_intercept_only
        for vid, lm in live.models.items():
            wm = warm.models[vid]
            assert wm.predictor_variant_ids == lm.predictor_variant_ids
            np.testing.assert_array_equal(wm.coefficients, lm.coefficients)
            assert wm.intercept == lm.intercept
            np.testing.assert_array_equal(wm.imputation_r2, lm.imputation_r2)


# --- Invalidation + corruption tolerance -------------------------------------
def test_key_change_is_a_miss(tmp_path):
    collected, plan, _ = _collect()
    store_collected(_key_for(plan), collected, cache_dir=tmp_path)
    # Same digest dir, but a load key whose components differ → invalidation miss.
    assert load_collected(_key_for(plan, window_size=W + 1), cache_dir=tmp_path) is None
    assert load_collected(_key_for(plan, cv_folds=10), cache_dir=tmp_path) is None


def test_absent_entry_is_a_miss(tmp_path):
    _, plan, _ = _collect()
    assert load_collected(_key_for(plan), cache_dir=tmp_path) is None


def test_corrupt_shard_tolerated(tmp_path):
    collected, plan, _ = _collect()
    key = _key_for(plan)
    entry = store_collected(key, collected, cache_dir=tmp_path)
    # Truncate a chromosome shard to garbage → load must miss silently, not raise.
    shard = next(entry.glob("chr*.npz"))
    shard.write_bytes(b"not a real npz")
    assert load_collected(key, cache_dir=tmp_path) is None


def test_corrupt_manifest_tolerated(tmp_path):
    collected, plan, _ = _collect()
    key = _key_for(plan)
    entry = store_collected(key, collected, cache_dir=tmp_path)
    (entry / "manifest.json").write_text("{ this is not json")
    assert load_collected(key, cache_dir=tmp_path) is None


def test_schema_version_mismatch_is_a_miss(tmp_path):
    collected, plan, _ = _collect()
    key = _key_for(plan)
    entry = store_collected(key, collected, cache_dir=tmp_path)
    manifest = json.loads((entry / "manifest.json").read_text())
    manifest["schema_version"] = 999
    (entry / "manifest.json").write_text(json.dumps(manifest))
    assert load_collected(key, cache_dir=tmp_path) is None


# --- Maintenance helpers -----------------------------------------------------
def test_info_and_clear(tmp_path):
    collected, plan, _ = _collect()
    assert get_stats_cache_info(cache_dir=tmp_path)["n_entries"] == 0
    store_collected(_key_for(plan), collected, cache_dir=tmp_path)
    store_collected(_key_for(plan, random_state=99), collected, cache_dir=tmp_path)

    info = get_stats_cache_info(cache_dir=tmp_path)
    assert info["n_entries"] == 2
    assert info["size_bytes"] > 0
    assert all(e["n_targets"] == len(collected) for e in info["entries"])

    assert clear_stats_cache(cache_dir=tmp_path) == 2
    assert get_stats_cache_info(cache_dir=tmp_path)["n_entries"] == 0


def test_default_cache_dir_constant():
    # Guard the documented default location (mirrors io/pgs_catalog conventions).
    assert stats_cache.DEFAULT_CACHE_DIR.parts[-2:] == ("imputed_prs", "stats_cache")
