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
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
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
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
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
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
            'effect_weight': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'beta' in result.columns
        assert 'effect_weight' not in result.columns
        assert result['beta'].tolist() == pytest.approx([0.05, -0.03])

    def test_alias_allele1(self):
        """Test that 'allele1' is mapped to 'effect_allele' (explicit effect, no error)."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'allele1': ['A', 'G'],
            'allele2': ['T', 'C'],
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
            'CHR': ['1', '2'],
            'POS': [100, 200],
            'Effect_Allele': ['A', 'G'],
            'Other_Allele': ['T', 'C'],
            'BETA': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'chromosome' in result.columns
        assert 'position' in result.columns
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
            'chromosome': ['1', '2'],
            'position': [100, 200],
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
            'position': [100, 200, 300],
            'effect_allele': ['A', 'G', 'T'],
            'other_allele': ['G', 'A', 'C'],
            'beta': [0.05, -0.03, 0.01],
        })
        result = load_prs_from_dataframe(df)
        assert result['chromosome'].tolist() == ['1', '22', '3']

    def test_optional_columns_preserved(self):
        """Test that optional columns (other_allele) are preserved."""
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
            'chromosome': ['1', '2', '3'],
            'position': [100, 200, 300],
            'effect_allele': ['A', 'G', 'T'],
            'other_allele': ['G', 'A', 'C'],
            'beta': [0.05, -0.03, 0.01],
        })
        result = load_prs_from_dataframe(df)
        assert len(result) == 2
        assert result['variant_id'].tolist() == ['rs123', 'rs789']

    def test_multiple_aliases_snp(self):
        """Test that 'snp'/'a1'/'weight' aliases are mapped to canonical names."""
        df = pd.DataFrame({
            'snp': ['rs123', 'rs456'],
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'a1': ['A', 'G'],
            'a2': ['T', 'C'],
            'weight': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert 'variant_id' in result.columns
        assert 'effect_allele' in result.columns
        assert 'beta' in result.columns

    def test_position_coerced_to_nullable_int(self):
        """Test that valid positions get the nullable Int64 dtype."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'chromosome': ['1', '2'],
            'position': [12345, 67890],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
            'beta': [0.05, -0.03],
        })
        result = load_prs_from_dataframe(df)
        assert result['position'].dtype == 'Int64'
        assert result['position'].iloc[0] == 12345

    def test_null_position_value_dropped_with_warning(self):
        """A row with a null position value is dropped (with a warning)."""
        df = pd.DataFrame({
            'variant_id': ['rs123', 'rs456'],
            'chromosome': ['1', '2'],
            'position': [12345, np.nan],
            'effect_allele': ['A', 'G'],
            'other_allele': ['T', 'C'],
            'beta': [0.05, -0.03],
        })
        with pytest.warns(UserWarning, match="position"):
            result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rs123']
        assert result['position'].dtype == 'Int64'


class TestAltAsEffect:
    """Tests for the alt-as-effect guard and its escape hatch."""

    def _alt_ref_df(self):
        return pd.DataFrame({
            'variant_id': ['rs1', 'rs2'],
            'chromosome': ['1', '2'],
            'position': [100, 200],
            'alt': ['A', 'G'],
            'ref': ['T', 'C'],
            'beta': [0.1, -0.2],
        })

    def test_alt_as_effect_raises_by_default(self):
        """Inferring effect_allele from an 'alt' column raises by default."""
        with pytest.raises(ValidationError, match="allow_alt_as_effect"):
            load_prs_from_dataframe(self._alt_ref_df())

    def test_alt_as_effect_allowed_with_flag(self):
        """With the escape hatch, 'alt' is accepted as the effect allele."""
        result = load_prs_from_dataframe(self._alt_ref_df(), allow_alt_as_effect=True)
        assert result['effect_allele'].tolist() == ['A', 'G']
        assert result['other_allele'].tolist() == ['T', 'C']

    def test_explicit_effect_allele_with_alt_column_ok(self):
        """An explicit effect_allele takes precedence over an alt column; no error."""
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'effect_allele': ['A'],
            'alt': ['T'],
            'other_allele': ['G'],
            'beta': [0.1],
        })
        result = load_prs_from_dataframe(df)
        assert result['effect_allele'].tolist() == ['A']

    def test_a1_allele_does_not_trigger_alt_guard(self):
        """'a1'/'allele1' denote the effect allele explicitly and must not raise."""
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'a1': ['A'],
            'a2': ['G'],
            'beta': [0.1],
        })
        result = load_prs_from_dataframe(df)
        assert result['effect_allele'].tolist() == ['A']


class TestRequiredColumns:
    """Tests for the now-required chromosome/position columns."""

    def test_missing_chromosome_raises(self):
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'position': [100],
            'effect_allele': ['A'],
            'beta': [0.1],
        })
        with pytest.raises(ValidationError, match="chromosome"):
            load_prs_from_dataframe(df)

    def test_missing_position_raises(self):
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'effect_allele': ['A'],
            'beta': [0.1],
        })
        with pytest.raises(ValidationError, match="position"):
            load_prs_from_dataframe(df)


class TestRowValidation:
    """Tests for row-level drop+warn cleaning."""

    def test_non_numeric_beta_dropped(self):
        df = pd.DataFrame({
            'variant_id': ['rsGOOD', 'rsBAD'],
            'chromosome': ['1', '1'],
            'position': [100, 200],
            'effect_allele': ['A', 'C'],
            'other_allele': ['G', 'T'],
            'beta': [0.1, 'not_a_number'],
        })
        with pytest.warns(UserWarning, match="beta"):
            result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rsGOOD']

    def test_invalid_effect_allele_dropped(self):
        df = pd.DataFrame({
            'variant_id': ['rsGOOD', 'rsX', 'rsEMPTY'],
            'chromosome': ['1', '1', '1'],
            'position': [100, 200, 300],
            'effect_allele': ['A', 'X', ''],
            'other_allele': ['G', 'A', 'A'],
            'beta': [0.1, 0.2, 0.3],
        })
        with pytest.warns(UserWarning, match="effect_allele"):
            result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rsGOOD']

    def test_multichar_indel_allele_kept(self):
        """Multi-character A/C/G/T alleles (indels) are valid and preserved."""
        df = pd.DataFrame({
            'variant_id': ['rsSNP', 'rsINDEL'],
            'chromosome': ['1', '1'],
            'position': [100, 200],
            'effect_allele': ['A', 'ATG'],
            'other_allele': ['G', 'A'],
            'beta': [0.1, 0.2],
        })
        result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rsSNP', 'rsINDEL']
        assert result['effect_allele'].tolist() == ['A', 'ATG']

    def test_malformed_other_allele_blanked_not_dropped(self):
        """A malformed other_allele is blanked to <NA>, keeping the row."""
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'effect_allele': ['A'],
            'other_allele': ['?'],
            'beta': [0.1],
        })
        with pytest.warns(UserWarning, match="other_allele"):
            result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rs1']
        assert pd.isna(result['other_allele'].iloc[0])


class TestGenomeBuildColumn:
    """Tests for the optional genome_build column."""

    def test_genome_build_alias_normalized(self):
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'effect_allele': ['A'],
            'other_allele': ['G'],
            'beta': [0.1],
            'genome_build': ['hg19'],
        })
        result = load_prs_from_dataframe(df)
        assert result['genome_build'].tolist() == ['GRCh37']

    def test_unrecognized_genome_build_raises(self):
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'effect_allele': ['A'],
            'other_allele': ['G'],
            'beta': [0.1],
            'genome_build': ['hg99'],
        })
        with pytest.raises(ValidationError, match="genome_build"):
            load_prs_from_dataframe(df)


class TestDuplicateResolution:
    """Tests for duplicate variant_id resolution."""

    def test_identical_duplicates_collapsed(self):
        df = pd.DataFrame({
            'variant_id': ['rs1', 'rs1'],
            'chromosome': ['1', '1'],
            'position': [100, 100],
            'effect_allele': ['A', 'A'],
            'other_allele': ['G', 'G'],
            'beta': [0.1, 0.1],
        })
        result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rs1']

    def test_multiallelic_same_rsid_kept(self):
        """Same rsID at different alleles (multi-allelic) is preserved."""
        df = pd.DataFrame({
            'variant_id': ['rs1', 'rs1'],
            'chromosome': ['1', '1'],
            'position': [100, 100],
            'effect_allele': ['C', 'G'],
            'other_allele': ['T', 'A'],
            'beta': [0.1, 0.2],
        })
        result = load_prs_from_dataframe(df)
        assert len(result) == 2
        assert set(result['effect_allele']) == {'C', 'G'}

    def test_conflicting_beta_dropped_with_warning(self):
        """Same id+locus+alleles with conflicting beta is dropped."""
        df = pd.DataFrame({
            'variant_id': ['rsKEEP', 'rsCON', 'rsCON'],
            'chromosome': ['1', '2', '2'],
            'position': [100, 200, 200],
            'effect_allele': ['A', 'T', 'T'],
            'other_allele': ['G', 'A', 'A'],
            'beta': [0.1, 0.3, 0.9],
        })
        with pytest.warns(UserWarning, match="conflicting"):
            result = load_prs_from_dataframe(df)
        assert result['variant_id'].tolist() == ['rsKEEP']


class TestOtherAlleleColumn:
    """Tests for permissive other_allele handling."""

    def test_missing_other_allele_warns(self):
        df = pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'effect_allele': ['A'],
            'beta': [0.1],
        })
        with pytest.warns(UserWarning, match="other_allele"):
            result = load_prs_from_dataframe(df)
        assert 'other_allele' not in result.columns


class TestPgsCompatibility:
    """A PGS-Catalog-shaped frame (explicit effect/other allele) loads unchanged."""

    def test_pgs_style_frame_loads_clean(self):
        df = pd.DataFrame({
            'variant_id': ['rs1', 'rs2', 'rsMA', 'rsMA'],
            'chromosome': ['1', '2', '3', '3'],
            'position': [100, 200, 300, 300],
            'effect_allele': ['A', 'ATG', 'C', 'G'],   # includes an indel
            'other_allele': ['G', 'A', 'T', 'A'],       # shared rsID, distinct alleles
            'beta': [0.1, 0.2, 0.3, 0.4],
        })
        result = load_prs_from_dataframe(df)
        # No alt rule fires (explicit effect_allele), nothing dropped, multi-allelic kept.
        assert len(result) == 4
        assert result['effect_allele'].tolist() == ['A', 'ATG', 'C', 'G']


class TestLoadPrsFromFile:
    """Tests for load_prs_from_file function."""

    def test_load_csv(self):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("variant_id,chromosome,position,effect_allele,other_allele,beta\n")
            f.write("rs123,1,100,A,G,0.05\n")
            f.write("rs456,2,200,G,A,-0.03\n")
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
            f.write("variant_id\tchromosome\tposition\teffect_allele\tother_allele\tbeta\n")
            f.write("rs123\t1\t100\tA\tG\t0.05\n")
            f.write("rs456\t2\t200\tG\tA\t-0.03\n")
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
                f.write("variant_id,chromosome,position,effect_allele,other_allele,beta\n")
                f.write("rs123,1,100,A,G,0.05\n")
                f.write("rs456,2,200,G,A,-0.03\n")

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
            f.write("variant_id,chromosome,position,effect_allele,other_allele,beta\n")
            f.write("rs123,1,100,A,G,0.05\n")
            f.write("rs456,2,200,G,A,-0.03\n")
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
            f.write("variant_id chromosome position effect_allele other_allele beta\n")
            f.write("rs123 1 100 A G 0.05\n")
            f.write("rs456 2 200 G A -0.03\n")
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
            f.write("rsid,chr,pos,allele1,allele2,effect_weight\n")
            f.write("rs123,1,100,a,g,0.05\n")
            f.write("rs456,2,200,g,a,-0.03\n")
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

    def test_alt_as_effect_flag_forwarded_from_file(self):
        """load_prs_from_file forwards allow_alt_as_effect to the dataframe loader."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("variant_id,chromosome,position,alt,ref,beta\n")
            f.write("rs123,1,100,A,G,0.05\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="allow_alt_as_effect"):
                load_prs_from_file(path)
            result = load_prs_from_file(path, allow_alt_as_effect=True)
            assert result['effect_allele'].tolist() == ['A']
        finally:
            path.unlink()


class TestFitThreading:
    """The allow_alt_as_effect flag is reachable through fit()."""

    def _alt_ref_df(self):
        return pd.DataFrame({
            'variant_id': ['rs1'],
            'chromosome': ['1'],
            'position': [100],
            'alt': ['A'],
            'ref': ['G'],
            'beta': [0.1],
        })

    def test_imputation_fit_raises_on_alt_as_effect_by_default(self):
        from imputed_prs.core.linear_imputation_prs import LinearImputationPRS

        with pytest.raises(ValidationError, match="allow_alt_as_effect"):
            LinearImputationPRS().fit(
                "dummy_ref.vcf", self._alt_ref_df(), platform_variants=["rs1"]
            )

    def test_projection_fit_raises_on_alt_as_effect_by_default(self):
        from imputed_prs.core.linear_projection_prs import LinearProjectionPRS

        with pytest.raises(ValidationError, match="allow_alt_as_effect"):
            LinearProjectionPRS().fit(
                "dummy_ref.vcf", self._alt_ref_df(), platform_variants=["rs1"]
            )

    def test_imputation_fit_forwards_flag(self, monkeypatch):
        """fit(allow_alt_as_effect=True) forwards the flag to the loader."""
        import imputed_prs.core.linear_imputation_prs as mod

        captured = {}

        def fake_load(df, allow_alt_as_effect=False):
            captured["flag"] = allow_alt_as_effect
            raise RuntimeError("stop-after-load")

        monkeypatch.setattr(mod, "load_prs_from_dataframe", fake_load)

        with pytest.raises(RuntimeError, match="stop-after-load"):
            mod.LinearImputationPRS().fit(
                "dummy_ref.vcf",
                self._alt_ref_df(),
                platform_variants=["rs1"],
                allow_alt_as_effect=True,
            )
        assert captured["flag"] is True
