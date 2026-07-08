"""Phase 11 — hermetic mini-1000G scale-regression gate (the CI scale canary).

A few-chromosome × few-hundred-variant × ~100-sample synthetic panel exercised end-to-end
through the **streaming** backend: fit both methods, export→reload→predict at ``atol=1e-12``,
dense-vs-streaming parity, a streaming reference CV, cross-``n_workers`` bit-identity, and the
``GenotypeSource`` seam. Catches scaling regressions WITHOUT the 14 GB reference panel or torch,
so CI can run it on a clean checkout in seconds.

Reuses the existing scale-test builders/guards (the streaming test modules already cross-import,
so importing across them is the established pattern here — there is no shared fixture module):
``_multichrom_panel`` (the panel), ``_assert_imputation_identical`` / ``_assert_projection_identical``
(the Phase-7 bit-identity guards), and ``_write_panel_vcf`` (multi-contig VCF for the path leg).
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.evaluation import ImputationEvaluator
from imputed_prs.io.genotype_source import InMemoryGenotypeSource
from tests.test_batched_solve_streaming import _multichrom_panel
from tests.test_parallel_streaming import (
    _assert_imputation_identical,
    _assert_projection_identical,
    _write_panel_vcf,
)
from threadpoolctl import threadpool_limits

pytestmark = pytest.mark.filterwarnings("ignore")

# "mini-1000G": 3 chromosomes × ~100 variants × 100 samples — a few hundred variants, large
# enough that streaming shards multiple chromosomes and trains many targets, small enough to run
# in seconds. ~2/3 on-platform (chip predictors), ~1/3 missing imputation targets per chromosome.
N_CHROM, PER_CHROM, N_SAMPLES, SEED = 3, 100, 100, 99
PROV = dict(reference_panel_id="mini_1000g", training_ancestry="ALL", genome_build="GRCh37")


@pytest.fixture(scope="module")
def mini_panel():
    return _multichrom_panel(n_chrom=N_CHROM, per_chrom=PER_CHROM, n_samples=N_SAMPLES, seed=SEED)


def _fit(cls, ref, prs, plat, **over):
    kw = dict(window_size=100_000, backend="streaming", device="cpu", tuning_scope="none",
              cv_folds=3, alpha=0.01, l1_ratio=0.5, random_state=17, verbose=0)
    kw.update(over)
    model = cls(**kw)
    model.fit(reference_genotypes=ref, prs_definition=prs, platform_variants=plat, **PROV)
    return model


def _probe_df(prs, n=30):
    """Deterministic het probe on the panel's biallelic SNPs (effect=G / other=A)."""
    d = prs.head(n)
    return pd.DataFrame({
        "rsid": d["variant_id"].astype(str).tolist(),
        "genotype": [f"{e}{o}" for e, o in zip(d["effect_allele"], d["other_allele"])],
    })


def test_streaming_fit_both_methods(mini_panel):
    """Both methods fit end-to-end via the streaming sufficient-statistics path."""
    gd, prs, plat = mini_panel
    imp = _fit(LinearImputationPRS, gd, prs, plat)
    proj = _fit(LinearProjectionPRS, gd, prs, plat)
    assert len(imp._imputed_models) > 5  # multiple missing targets trained across chromosomes
    assert len(proj._region_models) >= 1
    assert imp._calibration_params is not None
    assert np.isfinite(imp._calibration_params.scaling_factor)
    assert proj._calibration_params is not None


def test_dense_vs_streaming_parity(mini_panel):
    """Streaming Gram-CD reproduces the dense sklearn oracle within statistical-parity band."""
    gd, prs, plat = mini_panel
    s = _fit(LinearImputationPRS, gd, prs, plat, backend="streaming")
    d = _fit(LinearImputationPRS, gd, prs, plat, backend="dense")
    sm = {m.variant_id: m for m in s._imputed_models}
    dm = {m.variant_id: m for m in d._imputed_models}
    assert set(sm) == set(dm) and len(sm) > 5
    for vid, mm in sm.items():
        np.testing.assert_allclose(
            np.asarray(mm.coefficients), np.asarray(dm[vid].coefficients), atol=2e-3, rtol=2e-3
        )
        assert abs(mm.intercept - dm[vid].intercept) < 5e-3
    # aggregate calibration (over all targets' OOF predictions)
    assert abs(s._calibration_params.scaling_factor - d._calibration_params.scaling_factor) < 1e-2


@pytest.mark.parametrize("cls", [LinearImputationPRS, LinearProjectionPRS])
def test_export_reload_predict_golden(mini_panel, tmp_path, cls):
    """export → reload → predict is exact at atol=1e-12 on the real-shaped streaming artifact."""
    gd, prs, plat = mini_panel
    model = _fit(cls, gd, prs, plat)
    probe = _probe_df(prs)
    r1 = model.predict(probe, apply_calibration=True)
    paths = model.export(str(tmp_path / cls.__name__))
    reloaded = cls.load(str(paths["json"]))
    r2 = reloaded.predict(probe, apply_calibration=True)
    np.testing.assert_allclose(float(r2.prs), float(r1.prs), rtol=0, atol=1e-12)


def test_reference_cv_streaming(mini_panel, tmp_path):
    """Additive single-pass reference CV runs and matches the dense refit oracle within band.

    ``cross_validate`` reads a genotype *path*, so this leg materializes the panel as a VCF
    (guarded by cyvcf2, a core dep present in CI) rather than passing the in-RAM panel.
    """
    pytest.importorskip("cyvcf2")
    gd, prs, plat = mini_panel
    vcf = str(_write_panel_vcf(gd, tmp_path / "cv.vcf"))
    model = _fit(LinearImputationPRS, vcf, prs, plat)
    ev = ImputationEvaluator(model, verbose=0)
    common = dict(reference_genotypes=vcf, prs_definition=prs, platform_variants=plat,
                  n_folds=3, random_state=42)
    cv_stream = ev.cross_validate(backend="streaming", **common)
    cv_dense = ev.cross_validate(backend="dense", **common)
    assert np.isfinite(cv_stream.mean_r2)
    assert abs(cv_stream.mean_r2 - cv_dense.mean_r2) < 1e-2


@pytest.mark.parametrize("n_workers", [2])
def test_bit_identical_across_workers(mini_panel, n_workers):
    """Chromosome fan-out is zero-halo: n_workers ∈ {1, 2} give a bit-identical artifact."""
    gd, prs, plat = mini_panel
    with threadpool_limits(limits=1):
        base_i = _fit(LinearImputationPRS, gd, prs, plat, n_workers=1)
        got_i = _fit(LinearImputationPRS, gd, prs, plat, n_workers=n_workers)
        base_p = _fit(LinearProjectionPRS, gd, prs, plat, n_workers=1)
        got_p = _fit(LinearProjectionPRS, gd, prs, plat, n_workers=n_workers)
    _assert_imputation_identical(base_i, got_i)
    _assert_projection_identical(base_p, got_p)


def test_in_memory_source_streams(mini_panel):
    """A GenotypeSource passed as reference_genotypes streams + fits (Phase-1 seam)."""
    gd, prs, plat = mini_panel
    model = _fit(LinearImputationPRS, InMemoryGenotypeSource(gd), prs, plat)
    assert len(model._imputed_models) > 5


def test_vcf_path_streaming(mini_panel, tmp_path):
    """Streaming from a multi-contig VCF-on-disk (region pushdown) fits end-to-end."""
    pytest.importorskip("cyvcf2")
    gd, prs, plat = mini_panel
    vcf = _write_panel_vcf(gd, tmp_path / "mini.vcf")
    model = _fit(LinearImputationPRS, str(vcf), prs, plat)
    assert len(model._imputed_models) > 5
