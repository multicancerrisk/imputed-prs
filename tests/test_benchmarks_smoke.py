"""Full-pipeline smoke test: synthetic data -> fit both methods -> oracle -> export ->
determinism -> load-only points -> scaling projection. Exercises the real library through
the harness (isolated subprocesses), so it is gated on cyvcf2 and is a bit slower than the
hermetic tests."""
from __future__ import annotations

import types

import pytest

pytest.importorskip("cyvcf2")


def test_smoke_pipeline_runs(tmp_path):
    from benchmarks.harness import load_results
    from benchmarks.smoke import run_smoke

    args = types.SimpleNamespace(
        workdir=tmp_path / "work",
        results_dir=tmp_path / "res",
        methods=["imputation", "projection"],
    )
    assert run_smoke(args) == 0

    # projection artifact written
    assert (tmp_path / "res" / "smoke" / "projection.json").exists()

    # both methods produced a completed fit with an oracle payload
    results = load_results(tmp_path / "res" / "smoke")
    fits = [r for r in results if r.spec.operation == "fit" and r.outcome == "completed"]
    assert len(fits) >= 2
    assert all(r.result and "summary" in r.result for r in fits)


def test_synthetic_vcf_is_loadable(tmp_path):
    """The synthetic VCF generator must produce something load_genotypes can read."""
    from imputed_prs.io import load_genotypes

    from benchmarks.smoke import make_synthetic_vcf

    info = make_synthetic_vcf(tmp_path / "t.vcf", n_samples=12, n_variants=20, seed=0)
    gd = load_genotypes(path=str(tmp_path / "t.vcf"))
    assert gd.n_samples == 12
    assert gd.n_variants == 20
    assert len(info) == 20
