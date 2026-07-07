"""Phase 9: PGEN flows through fit() end-to-end and reads identically to VCF.

Synthesizes a small panel as both ``.pgen`` (via ``pgenlib.PgenWriter`` -- no
plink2 on this machine) and a VCF twin of the *same* genotypes, then fits an
imputation model from each and asserts:

  * a ``.pgen`` reference under ``backend="auto"`` streams and fits (closing the
    small-``.pgen``-under-auto gap: the dense loader has no PGEN branch),
  * the ``.pgen`` fit equals the VCF fit (dosages are bit-identical, so the
    streaming solve matches), and
  * ``backend="dense"`` + ``.pgen`` raises a clear, actionable error.

Skipped when pgenlib (the ``scale`` extra) is absent.
"""

import numpy as np
import pandas as pd
import pytest

pgenlib = pytest.importorskip("pgenlib")
pytest.importorskip("cyvcf2")

from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import ValidationError

# Panel: 20 samples, 6 chr1 variants inside a 1 Mb window. The chip = rs1/rs3/rs5
# (predictors); rs2/rs4/rs6 are the "missing" PRS variants imputed from the chip.
_RNG = np.random.RandomState(0)
N_SAMPLES = 20
VARIANTS = [  # (chrom, pos, id, ref, alt)
    ("1", 100000, "rs1", "A", "G"),
    ("1", 100500, "rs2", "C", "T"),
    ("1", 101000, "rs3", "G", "A"),
    ("1", 101500, "rs4", "T", "C"),
    ("1", 102000, "rs5", "A", "C"),
    ("1", 102500, "rs6", "G", "T"),
]
GENO = _RNG.randint(0, 3, size=(len(VARIANTS), N_SAMPLES)).astype(np.int8)  # ALT counts
SAMPLES = [f"S{i}" for i in range(N_SAMPLES)]
PLATFORM = ["rs1", "rs3", "rs5"]
_GT = {0: "0/0", 1: "0/1", 2: "1/1"}


@pytest.fixture
def prs_df():
    return pd.DataFrame(
        {
            "variant_id": [v[2] for v in VARIANTS],
            "chromosome": [v[0] for v in VARIANTS],
            "position": [v[1] for v in VARIANTS],
            "effect_allele": [v[4] for v in VARIANTS],  # ALT
            "other_allele": [v[3] for v in VARIANTS],  # REF
            "beta": [0.1, -0.2, 0.15, 0.05, -0.1, 0.2],
        }
    )


@pytest.fixture
def pgen_path(tmp_path):
    stem = tmp_path / "panel"
    w = pgenlib.PgenWriter(
        str(stem.with_suffix(".pgen")).encode(),
        sample_ct=N_SAMPLES,
        variant_ct=len(VARIANTS),
    )
    for v in range(len(VARIANTS)):
        w.append_biallelic(np.ascontiguousarray(GENO[v]))
    w.close()
    pvar = ["#CHROM\tPOS\tID\tREF\tALT"]
    pvar += [f"{c}\t{p}\t{i}\t{r}\t{a}" for c, p, i, r, a in VARIANTS]
    stem.with_suffix(".pvar").write_text("\n".join(pvar) + "\n")
    stem.with_suffix(".psam").write_text("#IID\n" + "\n".join(SAMPLES) + "\n")
    return stem.with_suffix(".pgen")


@pytest.fixture
def vcf_path(tmp_path):
    vcf = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES),
    ]
    for vi, (c, p, i, r, a) in enumerate(VARIANTS):
        gts = "\t".join(_GT[int(GENO[vi, s])] for s in range(N_SAMPLES))
        vcf.append(f"{c}\t{p}\t{i}\t{r}\t{a}\t.\t.\t.\tGT\t{gts}")
    path = tmp_path / "panel.vcf"
    path.write_text("\n".join(vcf) + "\n")
    return path


def _imputed_by_id(model):
    return {m.variant_id: m for m in model._imputed_models}


def test_pgen_fit_auto_streams_small_panel(pgen_path, prs_df):
    """The small-``.pgen``-under-auto gap: fit succeeds (streams) instead of raising."""
    model = LinearImputationPRS(backend="auto", random_state=0).fit(
        reference_genotypes=str(pgen_path),
        prs_definition=prs_df,
        platform_variants=PLATFORM,
    )
    assert model._imputed_models  # imputed models produced for rs2/rs4/rs6


def test_pgen_fit_matches_vcf(pgen_path, vcf_path, prs_df):
    """.pgen and VCF references (bit-identical dosages) yield the same streaming fit."""
    common = dict(prs_definition=prs_df, platform_variants=PLATFORM)
    m_pgen = LinearImputationPRS(backend="streaming", random_state=0).fit(
        reference_genotypes=str(pgen_path), **common
    )
    m_vcf = LinearImputationPRS(backend="streaming", random_state=0).fit(
        reference_genotypes=str(vcf_path), **common
    )

    a, b = _imputed_by_id(m_pgen), _imputed_by_id(m_vcf)
    assert a.keys() == b.keys() and a  # same imputed set, non-empty
    for vid in a:
        assert a[vid].predictor_variant_ids == b[vid].predictor_variant_ids
        np.testing.assert_allclose(a[vid].coefficients, b[vid].coefficients, atol=1e-9)
        assert a[vid].intercept == pytest.approx(b[vid].intercept, abs=1e-9)
        assert a[vid].imputation_r2 == pytest.approx(b[vid].imputation_r2, abs=1e-9)


def test_pgen_dense_backend_raises(pgen_path, prs_df):
    """backend='dense' + .pgen → a clear, actionable error (no dense PGEN reader)."""
    with pytest.raises(ValidationError, match="PGEN input requires the streaming backend"):
        LinearImputationPRS(backend="dense", random_state=0).fit(
            reference_genotypes=str(pgen_path),
            prs_definition=prs_df,
            platform_variants=PLATFORM,
        )
