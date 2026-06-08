"""Tests for genotype loading from VCF and PLINK formats."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.types import GenotypeData
from imputed_prs.io.genotype_loader import (
    _build_variant_lookup,
    _detect_genotype_format,
    _normalize_chromosome,
    _variant_matches,
    load_genotypes,
    load_genotypes_plink,
    load_genotypes_vcf,
)


class TestNormalizeChromosome:
    """Tests for chromosome normalization."""

    def test_strips_chr_prefix(self):
        """Test that 'chr' prefix is stripped."""
        assert _normalize_chromosome("chr1") == "1"
        assert _normalize_chromosome("Chr22") == "22"
        assert _normalize_chromosome("CHR10") == "10"

    def test_uppercase_sex_chromosomes(self):
        """Test that sex chromosomes are uppercased."""
        assert _normalize_chromosome("chrX") == "X"
        assert _normalize_chromosome("chrY") == "Y"
        assert _normalize_chromosome("x") == "X"
        assert _normalize_chromosome("y") == "Y"

    def test_mitochondrial_normalized(self):
        """Test that MT is normalized to M."""
        assert _normalize_chromosome("chrMT") == "M"
        assert _normalize_chromosome("MT") == "M"
        assert _normalize_chromosome("chrM") == "M"

    def test_plain_numbers_unchanged(self):
        """Test that plain chromosome numbers are unchanged."""
        assert _normalize_chromosome("1") == "1"
        assert _normalize_chromosome("22") == "22"


class TestDetectFormat:
    """Tests for format detection."""

    def test_vcf_extensions(self):
        """Test VCF format detection."""
        assert _detect_genotype_format(Path("data.vcf")) == "vcf"
        assert _detect_genotype_format(Path("data.vcf.gz")) == "vcf"
        assert _detect_genotype_format(Path("data.vcf.bgz")) == "vcf"
        assert _detect_genotype_format(Path("data.bcf")) == "vcf"

    def test_plink_extensions(self):
        """Test PLINK format detection."""
        assert _detect_genotype_format(Path("data.bed")) == "plink"
        assert _detect_genotype_format(Path("data.bim")) == "plink"
        assert _detect_genotype_format(Path("data.fam")) == "plink"

    def test_plink_prefix_detection(self, tmp_path):
        """Test PLINK detection from prefix when .bed exists."""
        # Create a .bed file
        bed_file = tmp_path / "data.bed"
        bed_file.touch()

        # Check that prefix is detected as PLINK
        assert _detect_genotype_format(tmp_path / "data") == "plink"

    def test_unknown_extension(self):
        """Test unknown format returns 'unknown'."""
        assert _detect_genotype_format(Path("data.txt")) == "unknown"
        assert _detect_genotype_format(Path("data.csv")) == "unknown"


class TestBuildVariantLookup:
    """Tests for variant lookup building."""

    def test_empty_input(self):
        """Test that None input returns empty sets."""
        rsid_set, chrpos_set = _build_variant_lookup(None)
        assert rsid_set == set()
        assert chrpos_set == set()

    def test_rsid_extraction(self):
        """Test rsID extraction."""
        variant_ids = {"rs123", "rs456", "RS789"}
        rsid_set, chrpos_set = _build_variant_lookup(variant_ids)
        assert "rs123" in rsid_set
        assert "rs456" in rsid_set
        assert "rs789" in rsid_set  # lowercase
        assert len(chrpos_set) == 0

    def test_chrpos_extraction(self):
        """Test chr:pos extraction."""
        variant_ids = {"1:12345", "chr22:67890", "X:100"}
        rsid_set, chrpos_set = _build_variant_lookup(variant_ids)
        assert len(rsid_set) == 0
        assert "1:12345" in chrpos_set
        assert "22:67890" in chrpos_set  # chr prefix stripped
        assert "X:100" in chrpos_set

    def test_mixed_ids(self):
        """Test mixed rsID and chr:pos extraction."""
        variant_ids = {"rs123", "1:12345", "rs456"}
        rsid_set, chrpos_set = _build_variant_lookup(variant_ids)
        assert "rs123" in rsid_set
        assert "rs456" in rsid_set
        assert "1:12345" in chrpos_set


class TestVariantMatches:
    """Tests for variant matching."""

    def test_empty_sets_match_all(self):
        """Test that empty filter sets match all variants."""
        assert _variant_matches("rs123", "1", 100, set(), set()) is True

    def test_rsid_match(self):
        """Test rsID matching."""
        rsid_set = {"rs123", "rs456"}
        assert _variant_matches("rs123", "1", 100, rsid_set, set()) is True
        assert _variant_matches("RS123", "1", 100, rsid_set, set()) is True
        assert _variant_matches("rs789", "1", 100, rsid_set, set()) is False

    def test_chrpos_match(self):
        """Test chr:pos matching."""
        chrpos_set = {"1:100", "22:200"}
        assert _variant_matches("rs123", "1", 100, set(), chrpos_set) is True
        assert _variant_matches("rs123", "22", 200, set(), chrpos_set) is True
        assert _variant_matches("rs123", "1", 999, set(), chrpos_set) is False


class TestLoadGenotypesVCF:
    """Tests for VCF loading."""

    @pytest.fixture
    def minimal_vcf_path(self):
        """Return path to minimal test VCF."""
        return Path(__file__).parent / "fixtures" / "minimal.vcf"

    def test_load_all_variants(self, minimal_vcf_path):
        """Test loading all variants from VCF."""
        data = load_genotypes_vcf(minimal_vcf_path)

        assert isinstance(data, GenotypeData)
        assert data.n_samples == 3
        assert data.n_variants == 3
        assert data.sample_ids == ["Sample1", "Sample2", "Sample3"]

    def test_dosage_values(self, minimal_vcf_path):
        """Test that dosage values are correct."""
        data = load_genotypes_vcf(minimal_vcf_path)

        # First variant: Sample1=0/0, Sample2=0/1, Sample3=1/1
        assert data.dosage_matrix[0, 0] == pytest.approx(0.0)
        assert data.dosage_matrix[1, 0] == pytest.approx(1.0)
        assert data.dosage_matrix[2, 0] == pytest.approx(2.0)

        # Second variant: Sample1=0/1, Sample2=1/1, Sample3=0/0
        assert data.dosage_matrix[0, 1] == pytest.approx(1.0)
        assert data.dosage_matrix[1, 1] == pytest.approx(2.0)
        assert data.dosage_matrix[2, 1] == pytest.approx(0.0)

    def test_variant_info(self, minimal_vcf_path):
        """Test variant info DataFrame."""
        data = load_genotypes_vcf(minimal_vcf_path)

        assert "variant_id" in data.variant_info.columns
        assert "chromosome" in data.variant_info.columns
        assert "position" in data.variant_info.columns
        assert "ref_allele" in data.variant_info.columns
        assert "alt_allele" in data.variant_info.columns

        assert data.variant_info["variant_id"].tolist() == ["rs1", "rs2", "rs3"]
        assert data.variant_info["chromosome"].tolist() == ["1", "1", "22"]
        assert data.variant_info["position"].tolist() == [100, 200, 1000]

    def test_filter_by_rsid(self, minimal_vcf_path):
        """Test filtering by rsID."""
        data = load_genotypes_vcf(minimal_vcf_path, variant_ids={"rs1", "rs3"})

        assert data.n_variants == 2
        assert data.variant_info["variant_id"].tolist() == ["rs1", "rs3"]

    def test_filter_by_chrpos(self, minimal_vcf_path):
        """Test filtering by chr:pos."""
        data = load_genotypes_vcf(minimal_vcf_path, variant_ids={"1:100", "22:1000"})

        assert data.n_variants == 2
        assert data.variant_info["position"].tolist() == [100, 1000]

    def test_no_matching_variants_raises(self, minimal_vcf_path):
        """Test that no matching variants raises ValidationError."""
        with pytest.raises(ValidationError, match="No variants matched"):
            load_genotypes_vcf(minimal_vcf_path, variant_ids={"rs999", "rs888"})

    def test_file_not_found_raises(self):
        """Test that missing file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            load_genotypes_vcf("/nonexistent/path.vcf")

    def test_source_file_recorded(self, minimal_vcf_path):
        """Test that source file is recorded."""
        data = load_genotypes_vcf(minimal_vcf_path)
        assert str(minimal_vcf_path) in data.source_file


class TestLoadGenotypesVCFWithMissingData:
    """Tests for VCF loading with missing genotypes."""

    @pytest.fixture
    def vcf_with_missing(self, tmp_path):
        """Create VCF with missing genotypes."""
        vcf_content = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
1	100	rs1	A	G	.	.	.	GT	0/0	./.	1/1
1	200	rs2	C	T	.	.	.	GT	./.	1/1	./.
"""
        vcf_path = tmp_path / "missing.vcf"
        vcf_path.write_text(vcf_content)
        return vcf_path

    def test_missing_as_nan(self, vcf_with_missing):
        """Test that missing genotypes are NaN."""
        data = load_genotypes_vcf(vcf_with_missing)

        # First variant: Sample2 is missing
        assert data.dosage_matrix[0, 0] == pytest.approx(0.0)
        assert np.isnan(data.dosage_matrix[1, 0])
        assert data.dosage_matrix[2, 0] == pytest.approx(2.0)

        # Second variant: Sample1 and Sample3 are missing
        assert np.isnan(data.dosage_matrix[0, 1])
        assert data.dosage_matrix[1, 1] == pytest.approx(2.0)
        assert np.isnan(data.dosage_matrix[2, 1])


class TestLoadGenotypesVCFWithDosage:
    """Tests for VCF loading with DS/GP fields."""

    @pytest.fixture
    def vcf_with_ds(self, tmp_path):
        """Create VCF with DS (dosage) field."""
        vcf_content = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
1	100	rs1	A	G	.	.	.	GT:DS	0/0:0.1	0/1:1.2	1/1:1.9
"""
        vcf_path = tmp_path / "dosage.vcf"
        vcf_path.write_text(vcf_content)
        return vcf_path

    def test_ds_field_used_in_auto(self, vcf_with_ds):
        """Test that DS field is used in auto mode when available."""
        data = load_genotypes_vcf(vcf_with_ds, dosage_field="auto")

        # Should use DS values, not GT
        assert data.dosage_matrix[0, 0] == pytest.approx(0.1, abs=0.01)
        assert data.dosage_matrix[1, 0] == pytest.approx(1.2, abs=0.01)
        assert data.dosage_matrix[2, 0] == pytest.approx(1.9, abs=0.01)

    def test_explicit_gt_field(self, vcf_with_ds):
        """Test that explicit GT field ignores DS."""
        data = load_genotypes_vcf(vcf_with_ds, dosage_field="GT")

        # Should use GT values (0, 1, 2)
        assert data.dosage_matrix[0, 0] == pytest.approx(0.0)
        assert data.dosage_matrix[1, 0] == pytest.approx(1.0)
        assert data.dosage_matrix[2, 0] == pytest.approx(2.0)


class TestLoadGenotypesPLINK:
    """Tests for PLINK loading."""

    @pytest.fixture
    def mock_plink_data(self):
        """Create mock PLINK data structures."""
        bim = pd.DataFrame({
            "chrom": ["1", "1", "22"],
            "snp": ["rs1", "rs2", "rs3"],
            "cm": [0.0, 0.0, 0.0],
            "pos": [100, 200, 1000],
            "a0": ["A", "C", "G"],
            "a1": ["G", "T", "A"],
        })

        fam = pd.DataFrame({
            "fid": ["FAM1", "FAM2", "FAM3"],
            "iid": ["Sample1", "Sample2", "Sample3"],
            "father": ["0", "0", "0"],
            "mother": ["0", "0", "0"],
            "gender": [1, 2, 1],
            "trait": [-9, -9, -9],
        })

        # Genotype matrix (variants x samples): 0=hom_ref, 1=het, 2=hom_alt
        genotype = np.array([
            [0, 1, 2],  # rs1
            [1, 2, 0],  # rs2
            [2, 1, 0],  # rs3
        ], dtype=np.float32)

        # Create mock dask array
        class MockDaskArray:
            def __init__(self, data):
                self._data = data

            def compute(self):
                return self._data

            def __getitem__(self, key):
                return MockDaskArray(self._data[key])

        return bim, fam, MockDaskArray(genotype)

    def test_load_plink(self, tmp_path, mock_plink_data):
        """Test loading PLINK files."""
        bim, fam, genotype = mock_plink_data

        # Create dummy .bed file
        bed_path = tmp_path / "test.bed"
        bed_path.touch()

        with patch("pandas_plink.read_plink1_bin") as mock_read:
            mock_read.return_value = (bim, fam, genotype)

            data = load_genotypes_plink(tmp_path / "test")

        assert isinstance(data, GenotypeData)
        assert data.n_samples == 3
        assert data.n_variants == 3
        assert data.sample_ids == ["Sample1", "Sample2", "Sample3"]

    def test_plink_dosage_values(self, tmp_path, mock_plink_data):
        """Test PLINK dosage values."""
        bim, fam, genotype = mock_plink_data

        bed_path = tmp_path / "test.bed"
        bed_path.touch()

        with patch("pandas_plink.read_plink1_bin") as mock_read:
            mock_read.return_value = (bim, fam, genotype)

            data = load_genotypes_plink(tmp_path / "test")

        # First variant: Sample1=0, Sample2=1, Sample3=2
        assert data.dosage_matrix[0, 0] == pytest.approx(0.0)
        assert data.dosage_matrix[1, 0] == pytest.approx(1.0)
        assert data.dosage_matrix[2, 0] == pytest.approx(2.0)

    def test_plink_variant_filter(self, tmp_path, mock_plink_data):
        """Test PLINK variant filtering."""
        bim, fam, genotype = mock_plink_data

        bed_path = tmp_path / "test.bed"
        bed_path.touch()

        with patch("pandas_plink.read_plink1_bin") as mock_read:
            mock_read.return_value = (bim, fam, genotype)

            data = load_genotypes_plink(tmp_path / "test", variant_ids={"rs1", "rs3"})

        assert data.n_variants == 2
        assert data.variant_info["variant_id"].tolist() == ["rs1", "rs3"]

    def test_plink_file_not_found(self):
        """Test PLINK file not found raises error."""
        with pytest.raises(DataLoadError, match="not found"):
            load_genotypes_plink("/nonexistent/path")


class TestLoadGenotypes:
    """Tests for auto-detecting load_genotypes function."""

    @pytest.fixture
    def minimal_vcf_path(self):
        """Return path to minimal test VCF."""
        return Path(__file__).parent / "fixtures" / "minimal.vcf"

    def test_auto_detect_vcf(self, minimal_vcf_path):
        """Test auto-detection of VCF format."""
        data = load_genotypes(minimal_vcf_path)

        assert isinstance(data, GenotypeData)
        assert data.n_variants == 3
        assert data.n_samples == 3

    def test_auto_detect_plink(self, tmp_path):
        """Test auto-detection of PLINK format."""
        # Create dummy .bed file
        bed_path = tmp_path / "test.bed"
        bed_path.touch()

        with patch("imputed_prs.io.genotype_loader.load_genotypes_plink") as mock_load:
            mock_load.return_value = GenotypeData(
                dosage_matrix=np.zeros((3, 2)),
                variant_info=pd.DataFrame({"variant_id": ["rs1", "rs2"]}),
                sample_ids=["S1", "S2", "S3"],
            )

            data = load_genotypes(tmp_path / "test")

        mock_load.assert_called_once()

    def test_unknown_format_raises(self):
        """Test that unknown format raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Unknown genotype format"):
            load_genotypes("/path/to/file.txt")


class TestGenotypeDataProperties:
    """Tests for GenotypeData dataclass."""

    def test_n_samples(self):
        """Test n_samples property."""
        data = GenotypeData(
            dosage_matrix=np.zeros((10, 5)),
            variant_info=pd.DataFrame(),
            sample_ids=[f"S{i}" for i in range(10)],
        )
        assert data.n_samples == 10

    def test_n_variants(self):
        """Test n_variants property."""
        data = GenotypeData(
            dosage_matrix=np.zeros((10, 5)),
            variant_info=pd.DataFrame(),
            sample_ids=[f"S{i}" for i in range(10)],
        )
        assert data.n_variants == 5


class TestImportErrors:
    """Tests for handling missing dependencies.

    Note: These tests verify the error handling works but can't easily
    test the actual ImportError paths since the imports happen at module
    load time. The error messages are verified through manual testing.
    """

    def test_vcf_loads_successfully(self):
        """Test that VCF loading works when cyvcf2 is available."""
        # This implicitly verifies the import path works
        minimal_vcf = Path(__file__).parent / "fixtures" / "minimal.vcf"
        data = load_genotypes_vcf(minimal_vcf)
        assert data.n_variants > 0

    def test_plink_function_exists(self):
        """Test that PLINK loading function is importable."""
        # This verifies pandas-plink is installed and importable
        from pandas_plink import read_plink1_bin
        assert callable(read_plink1_bin)


class TestMultiAllelicSplitting:
    """Multi-allelic records split into per-ALT rows with allele-specific dosage."""

    def test_multiallelic_record_split(self, tmp_path):
        """A REF=A ALT=G,T record yields two rows with allele-specific dosages."""
        pytest.importorskip("cyvcf2")
        vcf = tmp_path / "multi.vcf"
        # 5 samples: genotypes mix ALT1 (G, allele idx 1) and ALT2 (T, allele idx 2)
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=1000000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
            "\tS1\tS2\tS3\tS4\tS5\n"
            # S1=G/G(2 of ALT1) S2=A/G(1 ALT1) S3=A/T(1 ALT2) S4=T/T(2 ALT2) S5=A/A(ref)
            "1\t500\trsm\tA\tG,T\t.\t.\t.\tGT\t1/1\t0/1\t0/2\t2/2\t0/0\n"
        )
        data = load_genotypes_vcf(vcf)

        assert data.n_variants == 2  # one row per ALT
        vi = data.variant_info
        assert list(vi["ref_allele"]) == ["A", "A"]
        assert set(vi["alt_allele"]) == {"G", "T"}

        # ALT=G dosage counts only G copies: S1=2, S2=1, others 0
        g_idx = vi.index[vi["alt_allele"] == "G"][0]
        np.testing.assert_array_equal(
            data.dosage_matrix[:, g_idx], [2.0, 1.0, 0.0, 0.0, 0.0]
        )
        # ALT=T dosage counts only T copies: S3=1, S4=2, others 0
        t_idx = vi.index[vi["alt_allele"] == "T"][0]
        np.testing.assert_array_equal(
            data.dosage_matrix[:, t_idx], [0.0, 0.0, 1.0, 2.0, 0.0]
        )

    def test_biallelic_unchanged(self, tmp_path):
        """A biallelic record still yields a single row (fast path)."""
        pytest.importorskip("cyvcf2")
        vcf = tmp_path / "bi.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=1000000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n"
            "1\t500\trsb\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"
        )
        data = load_genotypes_vcf(vcf)
        assert data.n_variants == 1
        np.testing.assert_array_equal(data.dosage_matrix[:, 0], [0.0, 1.0, 2.0])
