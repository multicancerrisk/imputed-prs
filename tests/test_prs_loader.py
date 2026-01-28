"""Tests for PRS definition loading."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.io.prs_loader import load_prs_from_dataframe, load_prs_from_file


class TestLoadPrsFromDataframe:
    """Tests for load_prs_from_dataframe function."""

    def test_standard_column_names(self):
        """Test loading DataFrame with standard column names."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'effect_allele': ['A', 'G'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'effect_allele' in result.columns
        assert 'beta' in result.columns
        assert len(result) == 2

    def test_alias_rsid(self):
        """Test that 'rsid' is mapped to 'variant_id'."""
        df = pd.DataFrame({
            'rsid': ['rs123', 'rs456'],
            'effect_allele': ['A', 'G'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'rsid' not in result.columns
        assert result['variant_id'].tolist() == ['rs123', 'rs456']

    def test_alias_effect_weight(self):
        """Test that 'effect_weight' is mapped to 'beta'."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'effect_allele': ['A', 'G'],
            'effect_weight': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'beta' in result.columns
        assert 'effect_weight' not in result.columns
        assert result['beta'].tolist() == pytest.approx([0.05, -0.03])

    def test_alias_allele1(self):
        """Test that 'allele1' is mapped to 'effect_allele'."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'allele1': ['A', 'G'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'effect_allele' in result.columns
        assert 'allele1' not in result.columns
        assert result['effect_allele'].tolist() == ['A', 'G']

    def test_case_insensitive_columns(self):
        """Test that column matching is case-insensitive."""
        df = pd.DataFrame({
            'RSID': ['rs123', 'rs456'],
            'Effect_Allele': ['A', 'G'],
            'BETA': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'effect_allele' in result.columns
        assert 'beta' in result.columns

    def test_missing_variant_id_raises(self):
        """Test that missing variant_id column raises ValidationError."""
        df = pd.DataFrame({
            'effect_allele': ['A', 'G'],
            'beta': [0.05, -0.03],
        })
        with pytest.raises(ValidationError, match="variant_id"):
            load_prs_from_dataframe(df)

    def test_missing_effect_allele_raises(self):
        """Test that missing effect_allele column raises ValidationError."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'beta': [0.05, -0.03],
        })
        with pytest.raises(ValidationError, match="effect_allele"):
            load_prs_from_dataframe(df)

    def test_missing_beta_raises(self):
        """Test that missing beta column raises ValidationError."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'effect_allele': ['A', 'G'],
        })
        with pytest.raises(ValidationError, match="beta"):
            load_prs_from_dataframe(df)

    def test_alleles_uppercase(self):
        """Test that alleles are converted to uppercase."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'effect_allele': ['a', 'g'],
            'other_allele': ['t', 'c'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert result['effect_allele'].tolist() == ['A', 'G']
        assert result['other_allele'].tolist() == ['T', 'C']

    def test_chromosome_prefix_stripped(self):
        """Test that 'chr' prefix is stripped from chromosome."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456', 'rs789'],
            'chromosome': ['chr1', 'CHR22', '3'],
            'effect_allele': ['A', 'G', 'T'],
            'beta': [0.05, -0.03, 0.01],
        })
        result = load_prs_from_dataframe(df)
        assert result['chromosome'].tolist() == ['1', '22', '3']

    def test_optional_columns_preserved(self):
        """Test that optional columns (chromosome, position, other_allele) are preserved."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'chromosome': ['1', '2'],
            'position': [12345, 67890],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'chromosome' in result.columns
        assert 'position' in result.columns
        assert 'other_allele' in result.columns
        assert result['position'].tolist() == [12345, 67890]

    def test_null_variant_ids_dropped(self):
        """Test that rows with null variant_id are dropped."""
        df = pd.DataFrame({
            'variant_id': ['rs123', np.nan, 'rs789'],
            'effect_allele': ['A', 'G', 'T'],
            'beta': [0.05, -0.03, 0.01],
        })
        result = load_prs_from_dataframe(df)
        assert len(result) == 2
        assert result['variant_id'].tolist() == ['rs123', 'rs789']

    def test_multiple_aliases_snp(self):
        """Test that 'snp' alias is mapped to 'variant_id'."""
        df = pd.DataFrame({
            'snp': ['rs123', 'rs456'],
            'a1': ['A', 'G'],
            'weight': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'effect_allele' in result.columns
        assert 'beta' in result.columns

    def test_position_as_nullable_int(self):
        """Test that position column uses nullable Int64 dtype."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'position': [12345, np.nan],
            'effect_allele': ['A', 'G'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert result['position'].dtype == 'Int64'
        assert result['position'].iloc[0] == 12345
        assert pd.isna(result['position'].iloc[1])


class TestLoadPrsFromFile:
    """Tests for load_prs_from_file function."""

    def test_load_csv(self):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("variant_id,effect_allele,beta\n")
            f.write("rs123,A,0.05\n")
            f.write("rs456,G,-0.03\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prs_from_file(path)
            assert len(result) == 2
            assert result['variant_id'].tolist() == ['rs123', 'rs456']
            assert result['effect_allele'].tolist() == ['A', 'G']
        finally:
            path.unlink()

    def test_load_tsv(self):
        """Test loading from TSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("variant_id\teffect_allele\tbeta\n")
            f.write("rs123\tA\t0.05\n")
            f.write("rs456\tG\t-0.03\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prs_from_file(path)
            assert len(result) == 2
            assert result['variant_id'].tolist() == ['rs123', 'rs456']
        finally:
            path.unlink()

    def test_load_gzipped(self):
        """Test loading from gzipped file."""
        with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as f:
            path = Path(f.name)

        try:
            with gzip.open(path, 'wt') as f:
                f.write("variant_id,effect_allele,beta\n")
                f.write("rs123,A,0.05\n")
                f.write("rs456,G,-0.03\n")

            result = load_prs_from_file(path)
            assert len(result) == 2
            assert result['variant_id'].tolist() == ['rs123', 'rs456']
        finally:
            path.unlink()

    def test_file_not_found_raises(self):
        """Test that missing file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="File not found"):
            load_prs_from_file("/nonexistent/path/file.csv")

    def test_comment_lines_skipped(self):
        """Test that lines starting with # are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            f.write("variant_id,effect_allele,beta\n")
            f.write("rs123,A,0.05\n")
            f.write("rs456,G,-0.03\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prs_from_file(path)
            assert len(result) == 2
            assert result['variant_id'].tolist() == ['rs123', 'rs456']
        finally:
            path.unlink()

    def test_whitespace_separated(self):
        """Test loading from whitespace-separated file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("variant_id effect_allele beta\n")
            f.write("rs123 A 0.05\n")
            f.write("rs456 G -0.03\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prs_from_file(path)
            assert len(result) == 2
            assert result['variant_id'].tolist() == ['rs123', 'rs456']
        finally:
            path.unlink()

    def test_load_with_aliases_from_file(self):
        """Test that column aliases work when loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("rsid,allele1,effect_weight\n")
            f.write("rs123,a,0.05\n")
            f.write("rs456,g,-0.03\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prs_from_file(path)
            assert 'variant_id' in result.columns
            assert 'effect_allele' in result.columns
            assert 'beta' in result.columns
            # Test alleles are uppercased
            assert result['effect_allele'].tolist() == ['A', 'G']
        finally:
            path.unlink()

    def test_missing_required_columns_in_file(self):
        """Test that missing required columns raise ValidationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("variant_id,some_column\n")
            f.write("rs123,value\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="Missing required columns"):
                load_prs_from_file(path)
        finally:
            path.unlink()
