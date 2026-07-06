"""Tests for the streaming VcfGenotypeSource (Phase 1, Workstream E).

The streamed dosages must be bit-identical to the eager load_genotypes on the
same variants (both use the shared ``variant_to_records`` splitter), across
multiallelic splits, missing calls, and block boundaries. Region pushdown and
the contig-naming guard are also covered.
"""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cyvcf2")

from imputed_prs.core.exceptions import DataLoadError
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.genotype_source import (
    InMemoryGenotypeSource,
    VariantBlock,
    VcfGenotypeSource,
)

_VCF = """##fileformat=VCFv4.2
##contig=<ID=1>
##contig=<ID=2>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\tS5\tS6
1\t100\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t200\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t1/1\t0/0\t0/1\t./.\t0/0
1\t300\trs3\tA\tG,T\t.\t.\t.\tGT\t0/1\t1/2\t2/2\t0/0\t0/1\t0/2
2\t100\trs4\tG\tC\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0
2\t200\trs5\tT\tA\t.\t.\t.\tGT\t0/0\t0/1\t0/1\t1/1\t0/0\t0/1
"""


@pytest.fixture
def vcf_path(tmp_path):
    p = tmp_path / "ref.vcf"
    p.write_text(_VCF)
    return p


@pytest.fixture
def bgzipped_vcf(tmp_path, vcf_path):
    """bgzip + tabix via bcftools for region-query tests; skip if unavailable."""
    if not shutil.which("bcftools"):
        pytest.skip("bcftools not available for tabix indexing")
    gz = tmp_path / "ref.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(gz), str(vcf_path)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(gz)], check=True)
    return gz


def _concat_blocks(source, **kw):
    infos, mats = [], []
    for block in source.iter_variant_blocks(**kw):
        assert isinstance(block, VariantBlock)
        assert block.dosages.shape == (len(source.sample_ids), block.n_variants)
        assert block.dosages.dtype == np.float32
        infos.append(block.variant_info)
        mats.append(block.dosages)
    if not infos:
        return pd.DataFrame(), np.empty((len(source.sample_ids), 0), dtype=np.float32)
    return (pd.concat(infos, ignore_index=True), np.hstack(mats))


class TestDosageIdentity:
    def test_stream_matches_eager_full_scan(self, vcf_path):
        eager = load_genotypes(vcf_path)
        src = VcfGenotypeSource(vcf_path)
        info, dosages = _concat_blocks(src, block_size=2)  # forces >1 block

        # Same variants, same order (both scan the file in order), incl. the
        # multiallelic split into two rows.
        pd.testing.assert_frame_equal(info, eager.variant_info)
        np.testing.assert_array_equal(dosages, eager.dosage_matrix)  # NaN==NaN by position
        assert src.sample_ids == eager.sample_ids
        assert info.shape[0] == 6  # 5 records, rs3 splits into 2

    def test_block_size_does_not_change_result(self, vcf_path):
        src = VcfGenotypeSource(vcf_path)
        one = _concat_blocks(src, block_size=1000)
        many = _concat_blocks(src, block_size=1)  # every record its own block
        pd.testing.assert_frame_equal(one[0], many[0])
        np.testing.assert_array_equal(one[1], many[1])

    def test_missing_call_is_nan(self, vcf_path):
        # rs2 sample S5 is ./. -> NaN in both paths.
        src = VcfGenotypeSource(vcf_path)
        info, dosages = _concat_blocks(src)
        rs2_col = info.index[info["variant_id"] == "rs2"][0]
        assert np.isnan(dosages[4, rs2_col])


class TestFilter:
    def test_variant_id_filter(self, vcf_path):
        src = VcfGenotypeSource(vcf_path, variant_ids={"rs1", "rs4"})
        info, dosages = _concat_blocks(src)
        assert set(info["variant_id"]) == {"rs1", "rs4"}
        # Matches the eager loader's filtered result.
        eager = load_genotypes(vcf_path, variant_ids={"rs1", "rs4"})
        pd.testing.assert_frame_equal(info, eager.variant_info)
        np.testing.assert_array_equal(dosages, eager.dosage_matrix)


class TestRegionPushdown:
    def test_region_returns_only_in_region(self, bgzipped_vcf):
        src = VcfGenotypeSource(bgzipped_vcf)
        info, _ = _concat_blocks(src, region="2")
        assert set(info["chromosome"]) == {"2"}
        assert set(info["variant_id"]) == {"rs4", "rs5"}

    def test_region_subrange(self, bgzipped_vcf):
        src = VcfGenotypeSource(bgzipped_vcf)
        info, _ = _concat_blocks(src, region="1:250-350")
        assert list(info["variant_id"]) == ["rs3", "rs3"]  # multiallelic split

    def test_region_matches_eager_subset(self, bgzipped_vcf):
        src = VcfGenotypeSource(bgzipped_vcf)
        info, dosages = _concat_blocks(src, region="2")
        eager = load_genotypes(bgzipped_vcf, variant_ids={"rs4", "rs5"})
        pd.testing.assert_frame_equal(
            info.sort_values("variant_id").reset_index(drop=True),
            eager.variant_info.sort_values("variant_id").reset_index(drop=True),
        )


class TestContigGuard:
    def test_wrong_contig_naming_raises(self, bgzipped_vcf):
        # File contigs are "1"/"2"; a "chr1" region would silently return nothing.
        src = VcfGenotypeSource(bgzipped_vcf)
        with pytest.raises(DataLoadError, match="raw naming"):
            list(src.iter_variant_blocks(region="chr1"))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            VcfGenotypeSource(tmp_path / "nope.vcf")


class TestInMemoryGenotypeSource:
    """Blocks over an in-RAM GenotypeData must be a bit-identical column slice of
    the dense read (Phase 4), position-sorted, honoring sample/variant subsets."""

    def test_blocks_match_dense(self, vcf_path):
        gd = load_genotypes(vcf_path)
        src = InMemoryGenotypeSource(gd)
        info, dosages = _concat_blocks(src, block_size=2)  # force >1 block
        pd.testing.assert_frame_equal(
            info, gd.variant_info.reset_index(drop=True)
        )
        np.testing.assert_array_equal(dosages, gd.dosage_matrix)

    def test_region_filter_position_sorted(self, vcf_path):
        gd = load_genotypes(vcf_path)
        src = InMemoryGenotypeSource(gd)
        info, _ = _concat_blocks(src, region="1")
        assert set(info["chromosome"].astype(str)) == {"1"}
        assert list(info["position"]) == sorted(info["position"])

    def test_region_span(self, vcf_path):
        gd = load_genotypes(vcf_path)
        src = InMemoryGenotypeSource(gd)
        info, _ = _concat_blocks(src, region="1:150-350")
        assert list(info["position"]) == sorted(info["position"])
        assert info["position"].min() >= 150 and info["position"].max() <= 350

    def test_sample_subset(self, vcf_path):
        gd = load_genotypes(vcf_path)
        rows = np.array([0, 2, 4])
        src = InMemoryGenotypeSource(gd, sample_indices=rows)
        assert src.sample_ids == [gd.sample_ids[i] for i in rows]
        _, dosages = _concat_blocks(src)
        np.testing.assert_array_equal(dosages, gd.dosage_matrix[rows, :])

    def test_variant_filter(self, vcf_path):
        gd = load_genotypes(vcf_path)
        src = InMemoryGenotypeSource(gd, variant_ids={"rs1", "rs4"})
        info, _ = _concat_blocks(src)
        assert set(info["variant_id"]) == {"rs1", "rs4"}
