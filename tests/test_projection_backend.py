"""Phase-2 M3: streaming projection backend end-to-end parity with the dense oracle.

``backend="streaming"`` trains region projection models from banded sufficient
statistics without materializing the reference dosage matrix. On a panel with no
missing dosages and a pinned ``random_state`` it must reproduce the dense in-RAM path
across the whole public API: region models, calibration, observed variants + their
per-variant fallbacks, dispositions, and ``predict``. Two position clusters on chr1
create multiple merged regions so the region-scoped buffer actually evicts.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.linear_projection_prs import LinearProjectionPRS

pytestmark = pytest.mark.filterwarnings("ignore")

N_SAMPLES = 80
WINDOW = 200_000
SEED = 42
# Two clusters on chr1 separated by a gap > 2*WINDOW → distinct merged regions.
_CLUSTERS = [range(100_000, 100_000 + 40 * 10_000, 10_000),
             range(2_000_000, 2_000_000 + 40 * 10_000, 10_000)]
_ALLELES = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]


def _positions():
    return [p for cl in _CLUSTERS for p in cl]


def _write_vcf(path, seed=7):
    rng = np.random.RandomState(seed)
    samples = [f"S{i}" for i in range(N_SAMPLES)]
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1,length=249250621>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for v, pos in enumerate(_positions()):
        ref, alt = _ALLELES[v % len(_ALLELES)]
        p = 0.2 + 0.6 * rng.rand()
        dos = rng.binomial(2, p, size=N_SAMPLES)
        lines.append(
            f"1\t{pos}\trs{v}\t{ref}\t{alt}\t.\t.\t.\tGT\t"
            + "\t".join(gt[int(d)] for d in dos)
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _prs_and_platform(seed=0):
    """PRS over the panel; ~1/3 on-platform (observed), the rest missing (projected)."""
    rng = np.random.RandomState(seed)
    rows, platform = [], []
    for v, pos in enumerate(_positions()):
        ref, alt = _ALLELES[v % len(_ALLELES)]
        flip = (v % 3 == 0)
        rows.append(dict(
            variant_id=f"rs{v}", chromosome="1", position=pos,
            effect_allele=(ref if flip else alt), other_allele=(alt if flip else ref),
            beta=float(rng.uniform(-0.5, 0.5)),
        ))
        if v % 3 == 1:
            platform.append(f"rs{v}")
    return pd.DataFrame(rows), platform


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    pytest.importorskip("cyvcf2")
    path = tmp_path_factory.mktemp("proj_stream") / "panel.vcf"
    _write_vcf(path)
    prs_df, platform = _prs_and_platform()
    return path, prs_df, platform


def _fit(path, prs_df, platform, backend):
    model = LinearProjectionPRS(
        window_size=WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=SEED, backend=backend, verbose=0,
    )
    model.fit(reference_genotypes=path, prs_definition=prs_df,
              platform_variants=platform, genome_build="GRCh38")
    return model


@pytest.fixture(scope="module")
def fitted(panel):
    path, prs_df, platform = panel
    return _fit(path, prs_df, platform, "dense"), _fit(path, prs_df, platform, "streaming")


class TestProjectionStreamingParity:
    def test_region_models_match(self, fitted):
        dense, stream = fitted
        dmods = {m.region_id: m for m in dense.region_models}
        smods = {m.region_id: m for m in stream.region_models}
        assert set(dmods) == set(smods)
        assert len(dmods) >= 2  # multiple merged regions (both clusters)
        for rid, dm in dmods.items():
            sm = smods[rid]
            assert list(sm.predictor_variant_ids) == list(dm.predictor_variant_ids)
            assert list(sm.prs_variant_ids) == list(dm.prs_variant_ids)
            # Statistical-parity band (not bit-parity): S_R has larger magnitude than a
            # single dosage, so the two coordinate-descent solvers converge to within
            # tol of the same optimum at a looser absolute scale than imputation.
            np.testing.assert_allclose(sm.coefficients, dm.coefficients, rtol=1e-5, atol=1e-7)
            assert abs(sm.intercept - dm.intercept) < 1e-7
            assert abs(sm.cv_r2 - dm.cv_r2) < 1e-6
            assert sm.is_intercept_only == dm.is_intercept_only
            np.testing.assert_allclose(sm.betas, dm.betas, rtol=0, atol=1e-12)

    def test_observed_and_fallbacks_match(self, fitted):
        dense, stream = fitted
        dobs = {v.variant_id: v for v in dense.observed_variants}
        sobs = {v.variant_id: v for v in stream.observed_variants}
        assert set(dobs) == set(sobs)
        d_fb = {vid for vid, v in dobs.items() if v.fallback is not None}
        s_fb = {vid for vid, v in sobs.items() if v.fallback is not None}
        assert d_fb == s_fb
        assert len(d_fb) > 0
        for vid in d_fb:
            np.testing.assert_allclose(
                np.asarray(sobs[vid].fallback.coefficients),
                np.asarray(dobs[vid].fallback.coefficients), rtol=0, atol=1e-9,
            )

    def test_calibration_matches(self, fitted):
        dense, stream = fitted
        dc, sc = dense.calibration_params, stream.calibration_params
        assert (dc is None) == (sc is None)
        assert dc is not None
        assert abs(dc.scaling_factor - sc.scaling_factor) < 1e-6
        assert abs(dc.calibration_r2 - sc.calibration_r2) < 1e-6
        assert abs(dc.calibration_intercept - sc.calibration_intercept) < 1e-5
        assert abs(dc.diagonal_model_se_lower_bound
                   - sc.diagonal_model_se_lower_bound) < 1e-6

    def test_dispositions_match(self, fitted):
        dense, stream = fitted
        dd = dense.variant_dispositions.set_index("variant_id")["status"].to_dict()
        sd = stream.variant_dispositions.set_index("variant_id")["status"].to_dict()
        assert dd == sd
        assert set(dd.values()) >= {"observed", "projected"}

    def test_predict_matches(self, panel, fitted):
        _, prs_df, platform = panel
        dense, stream = fitted
        rng = np.random.RandomState(3)
        pos_of = dict(zip(prs_df["variant_id"], prs_df["position"]))
        eff_of = dict(zip(prs_df["variant_id"], prs_df["effect_allele"]))
        oth_of = dict(zip(prs_df["variant_id"], prs_df["other_allele"]))
        user = pd.DataFrame({
            "rsid": platform,
            "chrom": ["1"] * len(platform),
            "pos": [pos_of[v] for v in platform],
            "genotype": ["".join(rng.choice([eff_of[v], oth_of[v]], size=2)) for v in platform],
        })
        rd = dense.predict(user, apply_calibration=True)
        rs = stream.predict(user, apply_calibration=True)
        assert abs(rd.prs - rs.prs) < 1e-6
        assert abs((rd.prs_scaled or 0.0) - (rs.prs_scaled or 0.0)) < 1e-5


class TestProjectionBackendSelection:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError, match="backend must be"):
            LinearProjectionPRS(backend="bogus")

    def test_auto_small_input_uses_dense(self, panel):
        path, prs_df, platform = panel
        model = LinearProjectionPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="auto", verbose=0,
        )
        needed = set(prs_df["variant_id"]) | set(platform)
        from imputed_prs.io.genotype_source import make_genotype_source
        source = make_genotype_source(path, variant_ids=needed)
        assert model._auto_should_stream(source, needed) is False

    def test_auto_unsupported_format_falls_back_to_dense(self, tmp_path, monkeypatch):
        """backend='auto' with a PLINK .bed must fall through to the dense loader,
        not raise the streaming factory's DataLoadError (format regression guard)."""
        import imputed_prs.core.linear_projection_prs as mod
        prs_df, platform = _prs_and_platform()
        bed = tmp_path / "ref.bed"
        bed.touch()
        sentinel = RuntimeError("DENSE PATH REACHED")

        def _dense_reached(*a, **k):
            raise sentinel

        monkeypatch.setattr(mod, "load_genotypes", _dense_reached)
        model = LinearProjectionPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="auto", verbose=0,
        )
        with pytest.raises(RuntimeError, match="DENSE PATH REACHED"):
            model.fit(reference_genotypes=bed, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")

    def test_streaming_unsupported_format_raises(self, tmp_path):
        """Explicit backend='streaming' on an unreadable format surfaces the error."""
        from imputed_prs.core.exceptions import DataLoadError
        prs_df, platform = _prs_and_platform()
        bed = tmp_path / "ref.bed"
        bed.touch()
        model = LinearProjectionPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="streaming", verbose=0,
        )
        with pytest.raises(DataLoadError, match="streaming GenotypeSource"):
            model.fit(reference_genotypes=bed, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")


class TestProjectionStreamingGuardrails:
    def test_exclude_ambiguous_not_supported(self, panel):
        path, prs_df, platform = panel
        model = LinearProjectionPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="streaming", exclude_ambiguous=True, verbose=0,
        )
        with pytest.raises(NotImplementedError, match="exclude_ambiguous"):
            model.fit(reference_genotypes=path, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")

    def test_tuning_scope_warns_loudly(self, panel):
        """Streaming projection with tuning enabled warns unconditionally."""
        path, prs_df, platform = panel
        model = LinearProjectionPRS(
            window_size=WINDOW, tuning_scope="global", alpha=0.01, l1_ratio=0.5,
            cv_folds=5, random_state=SEED, backend="streaming", verbose=0,
        )
        with pytest.warns(UserWarning, match="tuning_scope"):
            model.fit(reference_genotypes=path, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")
