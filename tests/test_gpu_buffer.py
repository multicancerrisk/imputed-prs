"""Phase 3D: device-resident band buffer parity with the numpy ``_ChipGramBuffer``.

``GpuBackend.make_buffer`` returns ``_GpuChipGramBuffer`` — the sliding chip band and its
incremental full + per-fold Gram live on the GPU so the O(n) accumulation matmuls run there.
Driven with the *same* stream of columns (adds + an eviction), its Gram and column moments must
match the validated numpy buffer within float32 tolerance (exact for integer dosages; a small
band after mean-imputation). ``gather`` must return the same sub-block. Runs only when torch is
installed (skipped in the torch-free ``.venv``).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imputed_prs.compute.gpu_backend import GpuBackend  # noqa: E402
from imputed_prs.compute.sufficient_stats import (  # noqa: E402
    GlobalFolds,
    _ChipGramBuffer,
    _prepare_column,
)

pytestmark = pytest.mark.gpu

N = 300
K = 5
SEED = 11


def _gpu_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    pytest.skip("no GPU (mps/cuda) available")


def _columns(n_cols, with_nan, seed):
    """A stream of (col_perm, platform_idx, position, af) in ascending position."""
    rng = np.random.RandomState(seed)
    folds = GlobalFolds(N, K, SEED)
    cols = []
    for c in range(n_cols):
        freq = rng.uniform(0.1, 0.9)
        raw = rng.binomial(2, freq, size=N).astype(np.float64)
        if with_nan:
            raw[rng.rand(N) < 0.05] = np.nan  # ~5% missing → mean-imputed
        col_perm, af, _ = _prepare_column(raw, flip=False, folds=folds)
        cols.append((col_perm, c, 1000 * (c + 1), af))
    return folds, cols


def _feed(buf, cols):
    for col_perm, pidx, pos, af in cols:
        buf.add(col_perm, pidx, pos, af)


def _to_np(x):
    return x.detach().cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("with_nan", [False, True])
def test_device_buffer_gram_matches_numpy(with_nan):
    dev = _gpu_device()
    folds, cols = _columns(24, with_nan, seed=1)
    cpu = _ChipGramBuffer(N, folds)
    gpu = GpuBackend(dev).make_buffer(N, folds)
    _feed(cpu, cols)
    _feed(gpu, cols)

    m = cpu.m
    assert gpu.m == m
    # Integer dosages accumulate exactly in float32; mean-imputation adds a small band.
    atol = 1e-3 if not with_nan else 1e-2
    rtol = 1e-4
    np.testing.assert_allclose(_to_np(gpu.Gfull[:m, :m]), cpu.Gfull[:m, :m], rtol=rtol, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.Ghold[:, :m, :m]), cpu.Ghold[:, :m, :m], rtol=rtol, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsum[:m]), cpu.zsum[:m], rtol=rtol, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsqsum[:m]), cpu.zsqsum[:m], rtol=rtol, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsum_h[:, :m]), cpu.zsum_h[:, :m], rtol=rtol, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsqsum_h[:, :m]), cpu.zsqsum_h[:, :m], rtol=rtol, atol=atol)


def test_device_buffer_evict_and_gather_match_numpy():
    dev = _gpu_device()
    folds, cols = _columns(24, with_nan=False, seed=2)
    cpu = _ChipGramBuffer(N, folds)
    gpu = GpuBackend(dev).make_buffer(N, folds)
    _feed(cpu, cols)
    _feed(gpu, cols)

    # Evict the leading third of the band (positions are 1000*(c+1)).
    min_pos = cols[8][2] + 1
    cpu.evict_below(min_pos)
    gpu.evict_below(min_pos)
    m = cpu.m
    assert gpu.m == m and m > 0
    assert cpu.slot_of == gpu.slot_of  # host bookkeeping identical
    np.testing.assert_allclose(_to_np(gpu.Gfull[:m, :m]), cpu.Gfull[:m, :m], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(_to_np(gpu.Ghold[:, :m, :m]), cpu.Ghold[:, :m, :m], rtol=1e-4, atol=1e-3)

    # Gather an out-of-order subset of surviving predictors.
    surviving = sorted(cpu.slot_of.keys())
    pred = [surviving[i] for i in (3, 0, 5, 1)]
    c_idx, c_gg = cpu.gather(pred)
    g_idx_host, _g_idx_dev, g_gg = gpu.gather(pred)
    np.testing.assert_array_equal(g_idx_host, c_idx)
    np.testing.assert_allclose(_to_np(g_gg["G"]), c_gg["G"], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(_to_np(g_gg["fold_G"]), c_gg["fold_G"], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(_to_np(g_gg["zsum"]), c_gg["zsum"], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(g_gg["af"], c_gg["af"], rtol=0, atol=1e-12)


def test_device_lazy_fold_gram_gather_matches_numpy():
    """Phase-3E band-limited per-fold Gram: the projection ``lazy_fold_gram`` device buffer
    keeps no (K, cap, cap) tensor (``Gfull``/``Ghold`` are ``None``) and recomputes the full +
    per-fold Gram on-demand at ``gather`` from the resident band ``Z``. Fed the same stream
    (batched, crossing capacity, with an eviction), its gather must match the incrementally
    -maintained numpy buffer within float32 tolerance."""
    dev = _gpu_device()
    folds, cols = _columns(300, with_nan=True, seed=6)  # > cap 256 → _grow; NaN → float band
    cpu = _ChipGramBuffer(N, folds)  # eager reference
    gpu = GpuBackend(dev).make_buffer(N, folds, lazy_fold_gram=True)
    _feed(cpu, cols)
    for i in range(0, len(cols), 64):  # batched flush exercises the gated add_batch GEMMs
        blk = cols[i : i + 64]
        gpu.add_batch([c for c, *_ in blk], [p for _, p, *_ in blk],
                      [ps for *_, ps, _ in blk], [af for *_, af in blk])
    assert gpu.m == cpu.m
    assert gpu.Gfull is None and gpu.Ghold is None  # the memory fix: no resident Gram

    min_pos = cols[40][2] + 1  # evict the leading band, then gather survivors
    cpu.evict_below(min_pos)
    gpu.evict_below(min_pos)
    assert gpu.slot_of == cpu.slot_of
    surviving = sorted(cpu.slot_of.keys())
    pred = [surviving[i] for i in (10, 0, 33, 7, 21)]
    c_idx, c_gg = cpu.gather(pred)
    g_idx_host, _dev, g_gg = gpu.gather(pred)
    np.testing.assert_array_equal(g_idx_host, c_idx)
    for key in ("G", "fold_G", "zsum", "zsqsum", "fold_zsum", "fold_zsqsum"):
        np.testing.assert_allclose(_to_np(g_gg[key]), c_gg[key], rtol=1e-4, atol=1e-2)
    # G == Σ_k fold_G[k] on-device (same Zsub) ⇒ exact held-in training Gram.
    np.testing.assert_allclose(
        _to_np(g_gg["G"]), _to_np(g_gg["fold_G"]).sum(0), rtol=1e-4, atol=1e-2
    )


@pytest.mark.parametrize("with_nan", [False, True])
def test_device_add_batch_matches_per_column(with_nan):
    """The batched-GEMM ``add_batch`` (GPU accumulation win) must give the same band Gram as
    the numpy per-column path, fed as blocks of varying size (crossing the initial capacity)."""
    dev = _gpu_device()
    folds, cols = _columns(300, with_nan, seed=4)  # > default cap 256 → exercises _grow mid-batch
    cpu = _ChipGramBuffer(N, folds)
    gpu = GpuBackend(dev).make_buffer(N, folds)
    _feed(cpu, cols)  # numpy: per-column
    # gpu: flush in blocks of different widths, like real stream blocks.
    i = 0
    for width in (1, 7, 32, 100, 160):
        blk = cols[i : i + width]
        if not blk:
            break
        gpu.add_batch([c for c, *_ in blk], [p for _, p, *_ in blk],
                      [ps for *_, ps, _ in blk], [af for *_, af in blk])
        i += width
    assert gpu.m == cpu.m == i
    m = cpu.m
    atol = 1e-3 if not with_nan else 1e-2
    np.testing.assert_allclose(_to_np(gpu.Gfull[:m, :m]), cpu.Gfull[:m, :m], rtol=1e-4, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.Ghold[:, :m, :m]), cpu.Ghold[:, :m, :m], rtol=1e-4, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsum[:m]), cpu.zsum[:m], rtol=1e-4, atol=atol)
    np.testing.assert_allclose(_to_np(gpu.zsqsum_h[:, :m]), cpu.zsqsum_h[:, :m], rtol=1e-4, atol=atol)
    assert gpu.slot_of == cpu.slot_of


def test_device_buffer_grows_past_initial_capacity():
    """Adding > initial cap columns must grow the device band without corrupting it."""
    dev = _gpu_device()
    folds, cols = _columns(300, with_nan=False, seed=3)  # > default cap 256
    cpu = _ChipGramBuffer(N, folds)
    gpu = GpuBackend(dev).make_buffer(N, folds)
    _feed(cpu, cols)
    _feed(gpu, cols)
    m = cpu.m
    assert gpu.m == m == 300
    assert gpu.cap >= 300
    np.testing.assert_allclose(_to_np(gpu.Gfull[:m, :m]), cpu.Gfull[:m, :m], rtol=1e-4, atol=1e-3)
