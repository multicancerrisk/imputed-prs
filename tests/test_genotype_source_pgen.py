"""Cross-backend verification: PgenGenotypeSource == VcfGenotypeSource (Phase 1, E).

Synthesizes a tiny .pgen/.pvar/.psam (via pgenlib.PgenWriter -- no plink2 on
this machine) plus a VCF twin of the *same* genotypes, and asserts the PGEN
stream yields dosages identical to the trusted VCF stream. Skipped when pgenlib
(the ``scale`` extra) is absent.
"""

import numpy as np
import pandas as pd
import pytest

pgenlib = pytest.importorskip("pgenlib")
pytest.importorskip("cyvcf2")

from imputed_prs.io.genotype_source import (
    PgenGenotypeSource,
    VcfGenotypeSource,
    _read_pvar,
    _read_psam,
)

# genotypes: (n_variants=4, n_samples=5), ALT-allele counts, -9 = missing.
GENO = np.array([
    [0, 1, 2, 0, 1],
    [2, 2, 0, 1, -9],   # sample 4 missing
    [0, 1, 1, 2, 0],
    [-9, 0, 2, 1, 1],   # sample 0 missing
], dtype=np.int8)
SAMPLES = ["S0", "S1", "S2", "S3", "S4"]
VARIANTS = [  # (chrom, pos, id, ref, alt)
    ("1", 100, "rs1", "A", "G"),
    ("1", 200, "rs2", "C", "T"),
    ("2", 100, "rs3", "G", "A"),
    ("2", 300, "rs4", "T", "C"),
]
_GT = {0: "0/0", 1: "0/1", 2: "1/1", -9: "./."}


@pytest.fixture
def pgen_and_vcf(tmp_path):
    stem = tmp_path / "panel"
    # .pgen
    w = pgenlib.PgenWriter(str(stem.with_suffix(".pgen")).encode(),
                           sample_ct=len(SAMPLES), variant_ct=len(VARIANTS))
    for v in range(len(VARIANTS)):
        w.append_biallelic(np.ascontiguousarray(GENO[v]))
    w.close()
    # .pvar
    pvar = ["#CHROM\tPOS\tID\tREF\tALT"]
    pvar += [f"{c}\t{p}\t{i}\t{r}\t{a}" for c, p, i, r, a in VARIANTS]
    stem.with_suffix(".pvar").write_text("\n".join(pvar) + "\n")
    # .psam
    stem.with_suffix(".psam").write_text("#IID\n" + "\n".join(SAMPLES) + "\n")
    # VCF twin
    vcf = [
        "##fileformat=VCFv4.2", "##contig=<ID=1>", "##contig=<ID=2>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES),
    ]
    for vi, (c, p, i, r, a) in enumerate(VARIANTS):
        gts = "\t".join(_GT[int(GENO[vi, s])] for s in range(len(SAMPLES)))
        vcf.append(f"{c}\t{p}\t{i}\t{r}\t{a}\t.\t.\t.\tGT\t{gts}")
    vcf_path = tmp_path / "panel.vcf"
    vcf_path.write_text("\n".join(vcf) + "\n")
    return stem.with_suffix(".pgen"), vcf_path


def _concat(source, **kw):
    infos, mats = [], []
    for blk in source.iter_variant_blocks(**kw):
        infos.append(blk.variant_info)
        mats.append(blk.dosages)
    if not infos:
        return pd.DataFrame(), np.empty((len(source.sample_ids), 0), np.float32)
    return pd.concat(infos, ignore_index=True), np.hstack(mats)


def test_pgen_dosages_match_vcf(pgen_and_vcf):
    pgen_path, vcf_path = pgen_and_vcf
    pgen = PgenGenotypeSource(pgen_path)
    vcf = VcfGenotypeSource(vcf_path)

    assert pgen.sample_ids == vcf.sample_ids == SAMPLES

    p_info, p_dos = _concat(pgen, block_size=2)  # >1 block
    v_info, v_dos = _concat(vcf, block_size=2)

    # Same variant_info schema/values and bit-identical dosages (NaN-aware).
    pd.testing.assert_frame_equal(p_info, v_info)
    np.testing.assert_array_equal(p_dos, v_dos)
    assert p_dos.shape == (5, 4)
    assert np.isnan(p_dos[4, 1]) and np.isnan(p_dos[0, 3])  # the two missing calls


def test_pgen_region_filter(pgen_and_vcf):
    pgen_path, _ = pgen_and_vcf
    pgen = PgenGenotypeSource(pgen_path)
    info, dos = _concat(pgen, region="2:250-350")
    assert list(info["variant_id"]) == ["rs4"]
    assert dos.shape == (5, 1)


def test_pgen_variant_id_filter(pgen_and_vcf):
    pgen_path, _ = pgen_and_vcf
    pgen = PgenGenotypeSource(pgen_path, variant_ids={"rs1", "rs3"})
    info, _ = _concat(pgen)
    assert set(info["variant_id"]) == {"rs1", "rs3"}


def test_pgen_sample_subset(pgen_and_vcf):
    pgen_path, _ = pgen_and_vcf
    pgen = PgenGenotypeSource(pgen_path, samples=["S0", "S2", "S4"])
    assert pgen.sample_ids == ["S0", "S2", "S4"]
    _, dos = _concat(pgen)
    assert dos.shape == (3, 4)
    # rows are the subset, in .psam order: S0,S2,S4 of variant 0 -> 0,2,1
    np.testing.assert_array_equal(dos[:, 0], [0.0, 2.0, 1.0])


class TestPvarPsamParsers:
    def test_pvar_normalizes_and_types(self, pgen_and_vcf):
        pgen_path, _ = pgen_and_vcf
        info = _read_pvar(pgen_path.with_suffix(".pvar"))
        assert list(info["chromosome"]) == ["1", "1", "2", "2"]
        assert info["position"].dtype == np.int64
        assert list(info["alt_allele"]) == ["G", "T", "A", "C"]

    def test_psam_reads_iids(self, pgen_and_vcf):
        pgen_path, _ = pgen_and_vcf
        assert _read_psam(pgen_path.with_suffix(".psam")) == SAMPLES
