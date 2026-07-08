"""Phase 10 — opt-in disk checkpoint for resumable streaming fits (io/checkpoint.py).

These unit tests pin the guarantees the fan-out wiring (Commit 2/3) relies on:

  * the key is deterministic and binds the PRS content (betas/orientation) + hyperparameters
    that change the per-chromosome bits a checkpoint stores;
  * a saved → loaded partial round-trips **bit-identically** (arrays ``array_equal``);
  * atomic per-shard commit: an interrupted write leaves no shard (recompute), and tmp
    debris is swept on init;
  * a corrupt / absent shard is tolerated silently (``None``, never raises);
  * an incompatible manifest (schema / key component) resets; a git-sha drift **warns** but
    keeps the shards; ``save`` is best-effort and never raises into the fit.
"""

import json

import numpy as np
import pytest

from imputed_prs.compute.sufficient_stats import _ImputeChromPartial
from imputed_prs.core.types import ImputedVariantModel
from imputed_prs.io import checkpoint as ckpt
from imputed_prs.io.checkpoint import (
    CheckpointStore,
    clear_checkpoint,
    get_checkpoint_info,
    make_checkpoint_key,
)


# --- fixtures ----------------------------------------------------------------
def _model(vid, n_pred=2):
    return ImputedVariantModel(
        variant_id=vid,
        chromosome="22",
        position=1000,
        effect_allele="A",
        other_allele="G",
        beta=0.5,
        allele_frequency=0.3,
        imputation_r2=0.8,
        residual_variance=0.1,
        intercept=0.6,
        predictor_variant_ids=[f"{vid}_p{i}" for i in range(n_pred)],
        coefficients=np.arange(n_pred, dtype=np.float64) + 0.1,
        is_intercept_only=False,
        predictor_allele_frequencies=np.linspace(0.1, 0.4, n_pred),
    )


def _partial(chrom="22", n=10, cv=False):
    models = {"rs1": _model("rs1"), "rs2": _model("rs2", n_pred=3)}
    cv_collector = None
    if cv:
        cv_collector = {0: {"rs1": _model("rs1")}, 1: {"rs2": _model("rs2", 3)}}
    return _ImputeChromPartial(
        chrom=chrom,
        models=models,
        fallback_models={"rs9": _model("rs9")},
        failures={"rsX": "no predictors"},
        s_true=np.linspace(0.0, 1.0, n),
        s_cv=np.linspace(1.0, 2.0, n),
        n_intercept_only=1,
        cv_collector=cv_collector,
    )


def _assert_partial_equal(a, b):
    assert a.chrom == b.chrom
    assert a.n_intercept_only == b.n_intercept_only
    assert a.failures == b.failures
    np.testing.assert_array_equal(a.s_true, b.s_true)
    np.testing.assert_array_equal(a.s_cv, b.s_cv)
    for da, db in ((a.models, b.models), (a.fallback_models, b.fallback_models)):
        assert set(da) == set(db)
        for k in da:
            np.testing.assert_array_equal(da[k].coefficients, db[k].coefficients)
            assert da[k].intercept == db[k].intercept
            assert da[k].predictor_variant_ids == db[k].predictor_variant_ids
    if a.cv_collector is None:
        assert b.cv_collector is None
    else:
        assert set(a.cv_collector) == set(b.cv_collector)
        for f in a.cv_collector:
            assert set(a.cv_collector[f]) == set(b.cv_collector[f])


def _key(**over):
    kw = dict(
        sample_ids=["s0", "s1", "s2"],
        predictor_ids=["c1", "c2", "c3"],
        prs_terms=[("t", "rs1", False, 0.5), ("o", "rs2", True, -0.3)],
        window_size=1_000_000,
        max_predictors=None,
        cv_folds=5,
        random_state=7,
        alpha=0.01,
        l1_ratio=0.5,
        device="cpu",
        solve_mode="auto",
        mode="fit",
    )
    kw.update(over)
    return make_checkpoint_key(**kw)


# --- key ---------------------------------------------------------------------
def test_key_deterministic_and_predictor_order_independent():
    assert _key().digest == _key().digest
    assert _key(predictor_ids=["c3", "c1", "c2"]).digest == _key().digest


@pytest.mark.parametrize(
    "over",
    [
        dict(alpha=0.1),
        dict(l1_ratio=0.9),
        dict(prs_terms=[("t", "rs1", False, 0.9), ("o", "rs2", True, -0.3)]),  # beta change
        dict(prs_terms=[("t", "rs1", True, 0.5), ("o", "rs2", True, -0.3)]),  # flip change
        dict(device="mps"),
        dict(solve_mode="batched"),
        dict(mode="cv", fold_key="abc"),
        dict(window_size=500_000),
        dict(random_state=99),
    ],
)
def test_key_component_change_changes_digest(over):
    assert _key(**over).digest != _key().digest


# --- round trip --------------------------------------------------------------
@pytest.mark.parametrize("cv", [False, True])
def test_save_load_round_trip_bit_identical(tmp_path, cv):
    store = CheckpointStore(tmp_path, _key())
    p = _partial(cv=cv)
    assert store.save("22", p) is True
    loaded = store.load("22")
    assert loaded is not None
    _assert_partial_equal(p, loaded)


def test_load_absent_is_none(tmp_path):
    store = CheckpointStore(tmp_path, _key())
    assert store.load("22") is None
    assert store.done_chromosomes() == []


def test_done_chromosomes_tracks_committed_shards(tmp_path):
    store = CheckpointStore(tmp_path, _key())
    store.save("22", _partial("22"))
    store.save("1", _partial("1"))
    assert store.done_chromosomes() == ["1", "22"]


# --- atomicity / crash safety ------------------------------------------------
def test_interrupted_write_leaves_no_shard_and_tmp_is_swept(tmp_path):
    key = _key()
    store = CheckpointStore(tmp_path, key)
    # Simulate a kill mid-write: a lone tmp shard, never os.replace'd into place.
    tmp = store.dir / f".chr22{ckpt._SHARD_SUFFIX}.tmp.9999"
    tmp.write_bytes(b"partial junk")
    assert store.load("22") is None  # tmp is not a shard
    assert store.done_chromosomes() == []
    # A fresh store on the same (compatible) dir sweeps the debris but keeps real shards.
    store.save("1", _partial("1"))
    CheckpointStore(tmp_path, key)
    assert not tmp.exists()
    assert (store.dir / f"chr1{ckpt._SHARD_SUFFIX}").exists()


def test_corrupt_shard_is_a_miss(tmp_path):
    store = CheckpointStore(tmp_path, _key())
    store.save("22", _partial())
    (store.dir / f"chr22{ckpt._SHARD_SUFFIX}").write_bytes(b"not a real joblib dump")
    assert store.load("22") is None  # tolerated, no raise


# --- invalidation ------------------------------------------------------------
def test_different_config_uses_a_separate_dir(tmp_path):
    a = CheckpointStore(tmp_path, _key(alpha=0.01))
    a.save("22", _partial())
    b = CheckpointStore(tmp_path, _key(alpha=0.1))
    assert b.load("22") is None  # b's digest dir never sees a's shard
    assert a.load("22") is not None  # a's shard untouched
    assert a.dir != b.dir


def test_incompatible_manifest_resets(tmp_path):
    key = _key()
    store = CheckpointStore(tmp_path, key)
    store.save("22", _partial())
    # Force an incompatible schema_version under the same digest dir (truncation-collision
    # analogue) → a new store on the same key must wipe the stale shard and re-header.
    manifest_path = store.dir / ckpt._MANIFEST_NAME
    m = json.loads(manifest_path.read_text())
    m["schema_version"] = 999
    manifest_path.write_text(json.dumps(m))
    fresh = CheckpointStore(tmp_path, key)
    assert fresh.load("22") is None
    reset = json.loads((fresh.dir / ckpt._MANIFEST_NAME).read_text())
    assert reset["schema_version"] == ckpt._CHECKPOINT_SCHEMA_VERSION


def test_git_drift_warns_but_keeps_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(ckpt, "_git_sha", lambda: "cafef00d")
    key = _key()
    store = CheckpointStore(tmp_path, key)
    store.save("22", _partial())
    # Rewrite provenance to an older sha while keeping schema/key compatible.
    manifest_path = store.dir / ckpt._MANIFEST_NAME
    m = json.loads(manifest_path.read_text())
    m["provenance"]["git_sha"] = "deadbeef"
    manifest_path.write_text(json.dumps(m))
    with pytest.warns(UserWarning, match="not be bit-identical"):
        resumed = CheckpointStore(tmp_path, key)
    assert resumed.load("22") is not None  # shard survives a git drift


# --- best-effort -------------------------------------------------------------
def test_save_never_raises_on_write_error(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path, _key())

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt.joblib, "dump", _boom)
    assert store.save("22", _partial()) is False  # swallowed, no raise
    assert store.load("22") is None


def test_unwritable_dir_disables_checkpointing(tmp_path, monkeypatch):
    monkeypatch.setattr(ckpt.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    store = CheckpointStore(tmp_path / "nope", _key())
    assert store._ok is False
    assert store.save("22", _partial()) is False
    assert store.load("22") is None


# --- maintenance -------------------------------------------------------------
def test_info_and_clear(tmp_path):
    CheckpointStore(tmp_path, _key(alpha=0.01)).save("22", _partial())
    CheckpointStore(tmp_path, _key(alpha=0.1)).save("1", _partial("1"))
    info = get_checkpoint_info(tmp_path)
    assert info["n_entries"] == 2
    assert sum(e["n_shards"] for e in info["entries"]) == 2
    assert info["size_bytes"] > 0
    assert clear_checkpoint(tmp_path) == 2
    assert get_checkpoint_info(tmp_path)["n_entries"] == 0
