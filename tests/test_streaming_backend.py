"""Phase-2 streaming backend: end-to-end parity with the dense oracle.

``backend="streaming"`` trains from banded sufficient statistics without ever
materializing the reference dosage matrix. On a panel with no missing dosages and a
pinned ``random_state`` it must reproduce the dense in-RAM path (the correctness
oracle) to statistical-parity tolerance across the *whole public API*: imputed
models, calibration params, observed variants + per-variant fallbacks, dispositions,
and ``predict``. These tests build a synthetic VCF large enough for real ±W windows
(multiple predictors per target) and compare the two backends directly.
"""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import ValidationError
from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

pytestmark = pytest.mark.filterwarnings("ignore")

N_SAMPLES = 60
N_VARIANTS = 180
SPACING = 10_000  # bp between variants → ~40 predictors in a ±200kb window
WINDOW = 200_000
SEED = 42


def _write_synthetic_vcf(path, n_samples=N_SAMPLES, n_variants=N_VARIANTS, seed=7):
    """Write a chr1 VCF with random biallelic genotypes (no missing calls)."""
    rng = np.random.RandomState(seed)
    samples = [f"S{i}" for i in range(n_samples)]
    alleles = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1,length=249250621>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    gt_str = {0: "0/0", 1: "0/1", 2: "1/1"}
    for v in range(n_variants):
        pos = 100_000 + v * SPACING
        ref, alt = alleles[v % len(alleles)]
        # Allele freq in [0.2, 0.8] so targets have variance and predictors inform.
        p = 0.2 + 0.6 * rng.rand()
        dos = rng.binomial(2, p, size=n_samples)
        gts = "\t".join(gt_str[int(d)] for d in dos)
        lines.append(f"1\t{pos}\trs{v}\t{ref}\t{alt}\t.\t.\t.\tGT\t{gts}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _synthetic_prs(seed=0):
    """A PRS over the synthetic panel; ~1/3 on-platform (observed), rest missing."""
    rng = np.random.RandomState(seed)
    alleles = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]
    rows, platform = [], []
    for v in range(N_VARIANTS):
        pos = 100_000 + v * SPACING
        ref, alt = alleles[v % len(alleles)]
        flip = (v % 3 == 0)  # exercise both effect orientations
        rows.append(dict(
            variant_id=f"rs{v}", chromosome="1", position=pos,
            effect_allele=(ref if flip else alt),
            other_allele=(alt if flip else ref),
            beta=float(rng.uniform(-0.5, 0.5)),
        ))
        if v % 3 == 1:
            platform.append(f"rs{v}")
    return pd.DataFrame(rows), platform


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    pytest.importorskip("cyvcf2")
    path = tmp_path_factory.mktemp("stream") / "panel.vcf"
    _write_synthetic_vcf(path)
    prs_df, platform = _synthetic_prs()
    return path, prs_df, platform


def _fit(path, prs_df, platform, backend):
    # device="cpu" pins the numeric path: this file validates the streaming
    # sufficient-statistics algorithm vs the dense oracle at float64, independent of
    # whether torch/MPS is installed. GPU (float32) parity is covered by
    # tests/test_compute_backend.py.
    model = LinearImputationPRS(
        window_size=WINDOW, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=5, random_state=SEED, backend=backend, device="cpu", verbose=0,
    )
    model.fit(reference_genotypes=path, prs_definition=prs_df,
              platform_variants=platform, genome_build="GRCh38")
    return model


@pytest.fixture(scope="module")
def fitted(panel):
    path, prs_df, platform = panel
    return _fit(path, prs_df, platform, "dense"), _fit(path, prs_df, platform, "streaming")


class TestStreamingParity:
    def test_imputed_models_match(self, fitted):
        dense, stream = fitted
        dmods = {m.variant_id: m for m in dense.imputed_models}
        smods = {m.variant_id: m for m in stream.imputed_models}
        assert set(dmods) == set(smods)
        assert len(dmods) > 10  # a meaningful number of targets were trained
        for vid, dm in dmods.items():
            sm = smods[vid]
            assert list(sm.predictor_variant_ids) == list(dm.predictor_variant_ids)
            np.testing.assert_allclose(
                np.asarray(sm.coefficients), np.asarray(dm.coefficients),
                rtol=0, atol=1e-9,
            )
            assert abs(sm.intercept - dm.intercept) < 1e-9
            assert abs(sm.imputation_r2 - dm.imputation_r2) < 1e-8
            assert sm.is_intercept_only == dm.is_intercept_only

    def test_observed_and_fallbacks_match(self, fitted):
        dense, stream = fitted
        dobs = {v.variant_id: v for v in dense.observed_variants}
        sobs = {v.variant_id: v for v in stream.observed_variants}
        assert set(dobs) == set(sobs)
        # Same variants got a fallback, with matching coefficients.
        d_fb = {vid for vid, v in dobs.items() if v.fallback is not None}
        s_fb = {vid for vid, v in sobs.items() if v.fallback is not None}
        assert d_fb == s_fb
        assert len(d_fb) > 0
        for vid in d_fb:
            df, sf = dobs[vid].fallback, sobs[vid].fallback
            assert list(sf.predictor_variant_ids) == list(df.predictor_variant_ids)
            np.testing.assert_allclose(
                np.asarray(sf.coefficients), np.asarray(df.coefficients),
                rtol=0, atol=1e-9,
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
        assert abs(dc.raw_empirical_residual_sd
                   - sc.raw_empirical_residual_sd) < 1e-6

    def test_dispositions_match(self, fitted):
        dense, stream = fitted
        dd = dense.variant_dispositions.set_index("variant_id")["status"].to_dict()
        sd = stream.variant_dispositions.set_index("variant_id")["status"].to_dict()
        assert dd == sd

    def test_predict_matches(self, panel, fitted):
        _, prs_df, platform = panel
        dense, stream = fitted
        # A user typed on the platform variants (genotype strings from the ALT allele).
        rng = np.random.RandomState(3)
        pos_of = dict(zip(prs_df["variant_id"], prs_df["position"]))
        eff_of = dict(zip(prs_df["variant_id"], prs_df["effect_allele"]))
        oth_of = dict(zip(prs_df["variant_id"], prs_df["other_allele"]))
        user = pd.DataFrame({
            "rsid": platform,
            "chrom": ["1"] * len(platform),
            "pos": [pos_of[v] for v in platform],
            "genotype": [
                "".join(rng.choice([eff_of[v], oth_of[v]], size=2)) for v in platform
            ],
        })
        rd = dense.predict(user, apply_calibration=True)
        rs = stream.predict(user, apply_calibration=True)
        assert abs(rd.prs - rs.prs) < 1e-6
        assert abs(rd.prs_observed_component - rs.prs_observed_component) < 1e-6
        assert abs(rd.prs_imputed_component - rs.prs_imputed_component) < 1e-6
        assert abs((rd.prs_scaled or 0.0) - (rs.prs_scaled or 0.0)) < 1e-5


class TestBackendSelection:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError, match="backend must be"):
            LinearImputationPRS(backend="bogus")

    def test_auto_small_input_uses_dense(self, panel):
        """auto keeps test-sized inputs on the dense oracle (golden gate stays exact)."""
        path, prs_df, platform = panel
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="auto", verbose=0,
        )
        needed = set(prs_df["variant_id"]) | set(platform)
        from imputed_prs.io.genotype_source import make_genotype_source
        source = make_genotype_source(path, variant_ids=needed)
        assert model._auto_should_stream(source, needed) is False

    def test_auto_large_input_streams(self, panel, monkeypatch):
        """With the threshold dropped, auto selects streaming and still fits."""
        import imputed_prs.core.linear_imputation_prs as mod
        monkeypatch.setattr(mod, "_AUTO_STREAMING_BYTES_THRESHOLD", 0)
        path, prs_df, platform = panel
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="auto", verbose=0,
        )
        needed = set(prs_df["variant_id"]) | set(platform)
        from imputed_prs.io.genotype_source import make_genotype_source
        source = make_genotype_source(path, variant_ids=needed)
        assert model._auto_should_stream(source, needed) is True
        # And an end-to-end auto fit produces trained models.
        model.fit(reference_genotypes=path, prs_definition=prs_df,
                  platform_variants=platform, genome_build="GRCh38")
        assert len(model.imputed_models) > 0

    def test_auto_unsupported_format_falls_back_to_dense(self, tmp_path, monkeypatch):
        """backend='auto' with a PLINK .bed (which the streaming source can't read)
        must fall through to the dense loader, not raise the factory's DataLoadError.
        Regression: make_genotype_source was previously called unconditionally."""
        import imputed_prs.core.linear_imputation_prs as mod
        prs_df, platform = _synthetic_prs()
        bed = tmp_path / "ref.bed"
        bed.touch()
        sentinel = RuntimeError("DENSE PATH REACHED")

        def _dense_reached(*a, **k):
            raise sentinel

        monkeypatch.setattr(mod, "load_genotypes", _dense_reached)
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="auto", verbose=0,
        )
        with pytest.raises(RuntimeError, match="DENSE PATH REACHED"):
            model.fit(reference_genotypes=bed, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")

    def test_streaming_unsupported_format_raises(self, tmp_path):
        """Explicit backend='streaming' on an unreadable format surfaces the error."""
        from imputed_prs.core.exceptions import DataLoadError
        prs_df, platform = _synthetic_prs()
        bed = tmp_path / "ref.bed"
        bed.touch()
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="streaming", verbose=0,
        )
        with pytest.raises(DataLoadError, match="streaming GenotypeSource"):
            model.fit(reference_genotypes=bed, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")


class TestStreamingGuardrails:
    def test_exclude_ambiguous_not_supported(self, panel):
        path, prs_df, platform = panel
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="none", random_state=SEED,
            backend="streaming", exclude_ambiguous=True, verbose=0,
        )
        with pytest.raises(NotImplementedError, match="exclude_ambiguous"):
            model.fit(reference_genotypes=path, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")

    def test_tuning_scope_warns_loudly(self, panel):
        """Streaming with tuning enabled warns unconditionally (never silently drops
        tuning), independent of verbose level."""
        path, prs_df, platform = panel
        model = LinearImputationPRS(
            window_size=WINDOW, tuning_scope="global", alpha=0.01, l1_ratio=0.5,
            cv_folds=5, random_state=SEED, backend="streaming", verbose=0,
        )
        with pytest.warns(UserWarning, match="tuning_scope"):
            model.fit(reference_genotypes=path, prs_definition=prs_df,
                      platform_variants=platform, genome_build="GRCh38")
