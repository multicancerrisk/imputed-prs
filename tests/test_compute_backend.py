"""Phase 3C: CPU-vs-GPU parity for the batched compute backend.

Runs only when ``torch`` is installed (``importorskip``), so it is skipped in the
torch-free CPU ``.venv`` and exercised in ``.venv-gpu``. The CPU backend is the reference;
the GPU backend (torch on the best available device) must match it within statistical-parity
tolerance — float32 on MPS adds ~1e-5, and the ElasticNet FISTA reaches the same optimum as
the CPU coordinate descent (ridge, ``l1_ratio=0``, is exact). Two levels:

  * block level — ``GpuBackend._solve_blocks`` vs ``fit_from_local_gram`` per unit;
  * end to end — a streaming ``fit()`` on ``device="cpu"`` vs the GPU device, comparing
    imputed models, calibration, and prediction through the public API.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from imputed_prs.compute.gpu_backend import GpuBackend  # noqa: E402
from imputed_prs.compute.gram_solve import fit_from_local_gram  # noqa: E402
from imputed_prs.core.linear_imputation_prs import LinearImputationPRS  # noqa: E402
from imputed_prs.core.linear_projection_prs import LinearProjectionPRS  # noqa: E402
from tests.test_gram_solve import build_block, make_dosage_data  # noqa: E402
from tests.test_projection_backend import (  # noqa: E402
    SEED as PROJ_SEED,
    WINDOW as PROJ_WINDOW,
    _prs_and_platform,
)
from tests.test_projection_backend import _write_vcf as _write_proj_vcf  # noqa: E402
from tests.test_streaming_backend import (  # noqa: E402
    SEED,
    WINDOW,
    _synthetic_prs,
    _write_synthetic_vcf,
)

pytestmark = pytest.mark.gpu


def _gpu_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    pytest.skip("no GPU (mps/cuda) available")


# --------------------------------------------------------------- block-level parity
def _blocks(n=500, cv=5):
    out = []
    for p in (8, 20, 3, 1, 12, 40):
        X, y = make_dosage_data(n, p, seed=100 + p)
        out.append(build_block(X, y, cv, seed=7))
    return out


@pytest.mark.parametrize("alpha,l1_ratio", [
    (0.01, 0.0), (0.1, 0.0),           # ridge fast-path (exact)
    (0.01, 0.5), (0.1, 0.9), (0.01, 0.1),  # elastic net (FISTA)
])
@pytest.mark.parametrize("device", ["torch_cpu", "gpu"])
def test_solve_blocks_matches_cpu(alpha, l1_ratio, device):
    dev = "cpu" if device == "torch_cpu" else _gpu_device()
    cv = 5
    blocks = _blocks(cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns on ElasticNet(l1_ratio=0)
        cpu = [fit_from_local_gram(b, alpha=alpha, l1_ratio=l1_ratio, cv_folds=cv) for b in blocks]
    gpu = GpuBackend(dev)._solve_blocks(blocks, alpha, l1_ratio, cv)

    for a, b in zip(cpu, gpu):
        assert a.is_intercept_only == b.is_intercept_only
        np.testing.assert_allclose(b.coefficients, a.coefficients, atol=1e-3, rtol=1e-3)
        assert abs(a.intercept - b.intercept) < 1e-3
        assert abs(a.cv_r2 - b.cv_r2) < 5e-3
        assert abs(a.cv_mse - b.cv_mse) < 5e-3


@pytest.mark.parametrize("alpha,l1_ratio", [(0.1, 0.0), (0.01, 0.5), (0.1, 0.9)])
def test_gpu_and_numpy_batched_solve_agree(alpha, l1_ratio):
    """The numpy batched solve (Phase 8, gram_solve.solve_blocks_batched) and the torch
    GpuBackend._solve_blocks are ports of one algorithm and must stay in lockstep — agree
    within float32/optimizer tolerance on the same blocks (ridge exact; FISTA statistical).
    """
    from imputed_prs.compute.gram_solve import solve_blocks_batched  # noqa: PLC0415

    cv = 5
    blocks = _blocks(cv=cv)
    npy = solve_blocks_batched(blocks, alpha, l1_ratio, cv)
    gpu = GpuBackend(_gpu_device())._solve_blocks(blocks, alpha, l1_ratio, cv)
    for a, b in zip(npy, gpu):
        assert a.is_intercept_only == b.is_intercept_only
        np.testing.assert_allclose(b.coefficients, a.coefficients, atol=2e-3, rtol=2e-3)
        assert abs(a.intercept - b.intercept) < 2e-3
        assert abs(a.cv_r2 - b.cv_r2) < 5e-3


# --------------------------------------------------------------- end-to-end parity
@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    pytest.importorskip("cyvcf2")
    path = tmp_path_factory.mktemp("gpu") / "panel.vcf"
    _write_synthetic_vcf(path)
    prs_df, platform = _synthetic_prs()
    return path, prs_df, platform


def _fit(path, prs_df, platform, device, l1_ratio):
    model = LinearImputationPRS(
        window_size=WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=l1_ratio,
        cv_folds=5, random_state=SEED, backend="streaming", device=device, verbose=0,
    )
    model.fit(reference_genotypes=path, prs_definition=prs_df,
              platform_variants=platform, genome_build="GRCh38")
    return model


@pytest.mark.parametrize("l1_ratio", [0.5, 0.0])
def test_streaming_fit_cpu_vs_gpu(panel, l1_ratio):
    path, prs_df, platform = panel
    dev = _gpu_device()
    cpu = _fit(path, prs_df, platform, "cpu", l1_ratio)
    gpu = _fit(path, prs_df, platform, dev, l1_ratio)

    cmods = {m.variant_id: m for m in cpu.imputed_models}
    gmods = {m.variant_id: m for m in gpu.imputed_models}
    assert set(cmods) == set(gmods)
    assert len(cmods) > 10
    for vid, cm in cmods.items():
        gm = gmods[vid]
        assert list(gm.predictor_variant_ids) == list(cm.predictor_variant_ids)
        assert gm.is_intercept_only == cm.is_intercept_only
        np.testing.assert_allclose(
            np.asarray(gm.coefficients), np.asarray(cm.coefficients), atol=1e-3, rtol=1e-3
        )
        assert abs(gm.intercept - cm.intercept) < 1e-3
        assert abs(gm.imputation_r2 - cm.imputation_r2) < 5e-3

    cc, gc = cpu.calibration_params, gpu.calibration_params
    assert (cc is None) == (gc is None)
    if cc is not None:
        assert abs(cc.scaling_factor - gc.scaling_factor) < 1e-3
        assert abs(cc.calibration_r2 - gc.calibration_r2) < 1e-3
        assert abs(cc.calibration_intercept - gc.calibration_intercept) < 1e-3

    # Prediction through the public API on a synthetic typed user.
    rng = np.random.RandomState(3)
    eff = dict(zip(prs_df["variant_id"], prs_df["effect_allele"]))
    oth = dict(zip(prs_df["variant_id"], prs_df["other_allele"]))
    pos = dict(zip(prs_df["variant_id"], prs_df["position"]))
    user = pd.DataFrame({
        "rsid": platform,
        "chrom": ["1"] * len(platform),
        "pos": [pos[v] for v in platform],
        "genotype": ["".join(rng.choice([eff[v], oth[v]], size=2)) for v in platform],
    })
    rc = cpu.predict(user, apply_calibration=True)
    rg = gpu.predict(user, apply_calibration=True)
    assert abs(rc.prs - rg.prs) < 1e-3
    assert abs(rc.prs_imputed_component - rg.prs_imputed_component) < 1e-3


# ---------------------------------------------- end-to-end projection parity (device buffer)
@pytest.fixture(scope="module")
def proj_panel(tmp_path_factory):
    pytest.importorskip("cyvcf2")
    path = tmp_path_factory.mktemp("gpu_proj") / "panel.vcf"
    _write_proj_vcf(path)
    prs_df, platform = _prs_and_platform()
    return path, prs_df, platform


def _fit_proj(path, prs_df, platform, device, l1_ratio):
    model = LinearProjectionPRS(
        window_size=PROJ_WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=l1_ratio,
        cv_folds=5, random_state=PROJ_SEED, backend="streaming", device=device, verbose=0,
    )
    model.fit(reference_genotypes=path, prs_definition=prs_df,
              platform_variants=platform, genome_build="GRCh38")
    return model


@pytest.mark.parametrize("l1_ratio", [0.5, 0.0])
def test_streaming_projection_cpu_vs_gpu(proj_panel, l1_ratio):
    """Projection shares the device buffer + kernel; region S_R targets are larger-magnitude,
    so validate the device path reproduces the CPU region models + calibration directly."""
    path, prs_df, platform = proj_panel
    dev = _gpu_device()
    cpu = _fit_proj(path, prs_df, platform, "cpu", l1_ratio)
    gpu = _fit_proj(path, prs_df, platform, dev, l1_ratio)

    cmods = {m.region_id: m for m in cpu.region_models}
    gmods = {m.region_id: m for m in gpu.region_models}
    assert set(cmods) == set(gmods)
    assert len(cmods) >= 2  # both chr1 clusters → multiple merged regions
    for rid, cm in cmods.items():
        gm = gmods[rid]
        assert list(gm.predictor_variant_ids) == list(cm.predictor_variant_ids)
        assert gm.is_intercept_only == cm.is_intercept_only
        # S_R has larger magnitude than a single dosage ⇒ looser absolute band than imputation.
        np.testing.assert_allclose(
            np.asarray(gm.coefficients), np.asarray(cm.coefficients), atol=1e-2, rtol=1e-2
        )
        assert abs(gm.intercept - cm.intercept) < 1e-2
        assert abs(gm.cv_r2 - cm.cv_r2) < 5e-3

    cc, gc = cpu.calibration_params, gpu.calibration_params
    assert (cc is None) == (gc is None)
    if cc is not None:
        assert abs(cc.scaling_factor - gc.scaling_factor) < 1e-3
        assert abs(cc.calibration_r2 - gc.calibration_r2) < 1e-3
