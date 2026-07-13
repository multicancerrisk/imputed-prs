"""Phase 7 — process fan-out determinism gate.

The streaming fit / reference-CV shards by chromosome across a process pool
(``n_workers``). Chromosome shards are zero-halo, so the artifact must be **bit-identical**
regardless of worker count. These tests compare ``n_workers ∈ {1, 2, 4}`` under a fixed
single-thread BLAS pin (so the only variable is the reduce, not GEMM reduction order),
covering imputation + projection fit and reference CV, plus a VCF-path source (picklable,
region-pushdown) and the ``resolve_n_workers`` sizing/GPU-clamp logic.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.compute.parallel import resolve_n_workers
from imputed_prs.core.types import GenotypeData

N_SAMPLES = 60
N_CHROM = 4  # so n_workers=4 spawns a distinct pool from 2 and 1
SEED = 20260706


def _build_panel():
    """A 4-chromosome integer-dosage panel: 4 chip predictors + 1 missing target / chrom."""
    rng = np.random.default_rng(SEED)
    vrows, cols, prows, platform = [], [], [], []
    for c in range(1, N_CHROM + 1):
        chrom, base = str(c), c * 1_000_000
        for i in range(5):
            vid = f"{chrom}:{base + i * 500}"
            vrows.append({
                "variant_id": vid, "chromosome": chrom, "position": base + i * 500,
                "ref_allele": "A", "alt_allele": "G",
            })
            cols.append(rng.integers(0, 3, size=N_SAMPLES).astype(np.float32))
            prows.append({
                "variant_id": vid, "chromosome": chrom, "position": base + i * 500,
                "effect_allele": "G", "other_allele": "A", "beta": 0.1 * (i + 1),
            })
            if i < 4:
                platform.append(vid)
    gd = GenotypeData(
        dosage_matrix=np.stack(cols, axis=1),
        variant_info=pd.DataFrame(vrows),
        sample_ids=[f"S{i}" for i in range(N_SAMPLES)],
    )
    return gd, pd.DataFrame(prows), platform


@pytest.fixture(scope="module")
def panel():
    return _build_panel()


def _write_panel_vcf(gd, path):
    """Serialize a panel's integer dosages to a multi-contig VCF (0/0, 0/1, 1/1)."""
    contigs = sorted({str(c) for c in gd.variant_info["chromosome"]}, key=int)
    lines = [
        "##fileformat=VCFv4.2",
        *[f"##contig=<ID={c},length=300000000>" for c in contigs],
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(gd.sample_ids),
    ]
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    vi = gd.variant_info
    for j in range(len(vi)):
        row = vi.iloc[j]
        calls = "\t".join(gt[int(round(v))] for v in gd.dosage_matrix[:, j])
        lines.append(
            f"{row['chromosome']}\t{row['position']}\t{row['variant_id']}\t"
            f"{row['ref_allele']}\t{row['alt_allele']}\t.\t.\t.\tGT\t{calls}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _fit_imputation(panel, n_workers):
    gd, prs, plat = panel
    m = LinearImputationPRS(
        backend="streaming", tuning_scope="none", cv_folds=3,
        random_state=17, n_workers=n_workers, verbose=0,
    )
    m.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
          genome_build="GRCh37")
    return m


def _fit_projection(panel, n_workers):
    gd, prs, plat = panel
    m = LinearProjectionPRS(
        backend="streaming", tuning_scope="none", cv_folds=3,
        random_state=17, n_workers=n_workers, verbose=0,
    )
    m.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
          genome_build="GRCh37")
    return m


def _assert_calibration_equal(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    assert da.keys() == db.keys()
    for k in da:
        va, vb = da[k], db[k]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb), k
        else:
            assert va == vb, (k, va, vb)


def _assert_imputation_identical(a, b):
    ma = {m.variant_id: m for m in a._imputed_models}
    mb = {m.variant_id: m for m in b._imputed_models}
    assert set(ma) == set(mb)
    for vid in ma:
        assert np.array_equal(
            np.asarray(ma[vid].coefficients), np.asarray(mb[vid].coefficients)
        ), vid
        assert ma[vid].intercept == mb[vid].intercept, vid
        assert ma[vid].predictor_variant_ids == mb[vid].predictor_variant_ids, vid
    _assert_calibration_equal(a._calibration_params, b._calibration_params)


def _assert_projection_identical(a, b):
    ma = {m.region_id: m for m in a._region_models}
    mb = {m.region_id: m for m in b._region_models}
    assert set(ma) == set(mb)
    for rid in ma:
        assert np.array_equal(
            np.asarray(ma[rid].coefficients), np.asarray(mb[rid].coefficients)
        ), rid
        assert ma[rid].intercept == mb[rid].intercept, rid
        assert ma[rid].cv_mse == mb[rid].cv_mse, rid
    _assert_calibration_equal(a._calibration_params, b._calibration_params)


@pytest.mark.parametrize("n_workers", [2, 4])
def test_streaming_imputation_bit_identical_across_workers(panel, n_workers):
    with threadpool_limits(limits=1):
        base = _fit_imputation(panel, 1)
        got = _fit_imputation(panel, n_workers)
    assert len(base._imputed_models) == N_CHROM  # a target closed on each chromosome
    _assert_imputation_identical(base, got)


@pytest.mark.parametrize("n_workers", [2, 4])
def test_streaming_projection_bit_identical_across_workers(panel, n_workers):
    with threadpool_limits(limits=1):
        base = _fit_projection(panel, 1)
        got = _fit_projection(panel, n_workers)
    assert len(base._region_models) == N_CHROM
    _assert_projection_identical(base, got)


def _fold_indices():
    rng = np.random.default_rng(3)
    return [np.asarray(idx) for idx in np.array_split(rng.permutation(N_SAMPLES), 3)]


@pytest.mark.parametrize("n_workers", [2, 4])
def test_reference_cv_imputation_bit_identical(panel, n_workers):
    gd, prs, plat = panel
    fi = _fold_indices()

    def run(nw):
        d = LinearImputationPRS(
            backend="streaming", tuning_scope="none", cv_folds=3,
            random_state=17, n_workers=nw, verbose=0,
        )
        return d._reference_cv_fold_models(
            gd, prs, platform_variants=plat, fold_indices=fi
        )

    with threadpool_limits(limits=1):
        base, got = run(1), run(n_workers)
    assert base is not None and got is not None
    for k in base.fold_imputed_models:
        ba = {m.variant_id: m for m in base.fold_imputed_models[k]}
        gb = {m.variant_id: m for m in got.fold_imputed_models[k]}
        assert set(ba) == set(gb) and ba, (k, set(ba), set(gb))
        for vid in ba:
            assert np.array_equal(
                np.asarray(ba[vid].coefficients), np.asarray(gb[vid].coefficients)
            ), (k, vid)
            assert ba[vid].intercept == gb[vid].intercept, (k, vid)


@pytest.mark.parametrize("n_workers", [2, 4])
def test_reference_cv_projection_bit_identical(panel, n_workers):
    gd, prs, plat = panel
    fi = _fold_indices()

    def run(nw):
        d = LinearProjectionPRS(
            backend="streaming", tuning_scope="none", cv_folds=3,
            random_state=17, n_workers=nw, verbose=0,
        )
        return d._reference_cv_fold_models(
            gd, prs, platform_variants=plat, fold_indices=fi
        )

    with threadpool_limits(limits=1):
        base, got = run(1), run(n_workers)
    assert base is not None and got is not None
    for k in base.fold_region_models:
        ba = {m.region_id: m for m in base.fold_region_models[k]}
        gb = {m.region_id: m for m in got.fold_region_models[k]}
        assert set(ba) == set(gb) and ba, (k, set(ba), set(gb))
        for rid in ba:
            assert np.array_equal(
                np.asarray(ba[rid].coefficients), np.asarray(gb[rid].coefficients)
            ), (k, rid)


def test_vcf_path_source_fan_out_bit_identical(panel, tmp_path):
    """Path-based VcfGenotypeSource: picklable + region-pushdown across worker procs."""
    pytest.importorskip("cyvcf2")
    gd, prs, plat = panel
    vcf_path = _write_panel_vcf(gd, tmp_path / "multichrom.vcf")

    def run(nw):
        m = LinearImputationPRS(
            backend="streaming", tuning_scope="none", cv_folds=3,
            random_state=17, n_workers=nw, verbose=0,
        )
        m.fit(reference_genotypes=vcf_path, prs_definition=prs,
              platform_variants=plat, genome_build="GRCh37")
        return m

    with threadpool_limits(limits=1):
        base, got = run(1), run(2)
    _assert_imputation_identical(base, got)


def test_sensitivity_reproducible_across_workers(panel, tmp_path):
    """sensitivity_analysis fans out independent combos; result must be worker-invariant."""
    pytest.importorskip("cyvcf2")
    from imputed_prs.evaluation.evaluator import ImputationEvaluator

    gd, prs, plat = panel
    vcf_path = _write_panel_vcf(gd, tmp_path / "sens.vcf")
    grid = {"l1_ratio": [0.1, 0.5], "alpha": [0.01, 0.1]}  # 4 combos, kept small

    def run(nw):
        # The outer evaluator needs a fitted model; its n_workers drives the combo fan-out.
        model = LinearImputationPRS(
            n_workers=nw, tuning_scope="none", cv_folds=3, random_state=17, verbose=0
        )
        model.fit(reference_genotypes=gd, prs_definition=prs, platform_variants=plat,
                  genome_build="GRCh37")
        ev = ImputationEvaluator(model, verbose=0)
        return ev.sensitivity_analysis(
            reference_genotypes=vcf_path, prs_definition=prs, platform_variants=plat,
            parameter_grid=grid, cv_folds=3, random_state=17,
        )

    with threadpool_limits(limits=1):
        r1, r2 = run(1), run(2)
    # The fan-out guarantees a deterministic winner + canonical combo order. Per-combo
    # metrics carry the pre-existing dense-path PYTHONHASHSEED set/dict-ordering jitter
    # (~1e-7; the benchmark harness pins PYTHONHASHSEED=0 to remove it), exposed here only
    # because a spawned worker gets a different hash seed than the bare-pytest parent — so
    # metrics are compared within tolerance, structure + decision exactly.
    assert r1.best_params == r2.best_params
    assert [pr["params"] for pr in r1.parameter_results] == [
        pr["params"] for pr in r2.parameter_results
    ]
    for a, b in zip(r1.parameter_results, r2.parameter_results):
        ma, mb = a.get("metrics"), b.get("metrics")
        assert (ma is None) == (mb is None)
        if ma is not None:
            assert ma.r2 == pytest.approx(mb.r2, abs=1e-4)


def test_resolve_worker_count(monkeypatch):
    import imputed_prs.compute.parallel as par

    monkeypatch.setattr(par, "_performance_cores", lambda: 6)
    # Pin the logical-CPU count so the min(perf_cores, ncpu) clamp doesn't mask the intent
    # on hosts/runners with fewer than 6 cores (e.g. the 4-core CI runner).
    monkeypatch.setattr(par.os, "cpu_count", lambda: 8)
    assert resolve_n_workers(1) == 1
    assert resolve_n_workers(None) == 1
    assert resolve_n_workers(-1) == 6  # performance cores
    assert resolve_n_workers(2) == 2
    # Process fan-out is CPU-only: any GPU device clamps to a single process.
    assert resolve_n_workers(4, device="mps") == 1
    assert resolve_n_workers(-1, device="cuda") == 1
    # Never exceed the logical CPU count.
    assert resolve_n_workers(10_000) <= (par.os.cpu_count() or 1)
