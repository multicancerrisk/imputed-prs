"""PRS definition loading and normalization."""

import csv
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError


COLUMN_ALIASES = {
    'variant_id': ['rsid', 'snp', 'snp_id', 'variant', 'id', 'marker', 'markername'],
    'chromosome': ['chr', 'chrom', '#chrom', 'chr_name'],
    'position': ['pos', 'bp', 'chr_position', 'base_pair_location'],
    'effect_allele': ['allele1', 'a1', 'alt', 'effect', 'ea', 'risk_allele'],
    'other_allele': ['allele2', 'a2', 'ref', 'non_effect_allele', 'nea', 'reference_allele'],
    'beta': ['effect_weight', 'weight', 'effect_size', 'log_or'],
}

REQUIRED_COLUMNS = ['variant_id', 'effect_allele', 'beta']


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical names using aliases.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially non-standard column names.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with normalized column names.
    """
    df = df.copy()

    # Create lowercase mapping of current columns
    lowercase_to_original = {col.lower(): col for col in df.columns}

    # Build rename mapping
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        # Check if canonical name already exists (case-insensitive)
        if canonical.lower() in lowercase_to_original:
            original_col = lowercase_to_original[canonical.lower()]
            if original_col != canonical:
                rename_map[original_col] = canonical
            continue

        # Check aliases
        for alias in aliases:
            if alias.lower() in lowercase_to_original:
                original_col = lowercase_to_original[alias.lower()]
                rename_map[original_col] = canonical
                break

    df.rename(columns=rename_map, inplace=True)
    return df


def _validate_prs_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValidationError
        If required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def _coerce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to appropriate data types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with coerced data types.
    """
    df = df.copy()

    # Drop rows where variant_id is null before conversion
    df = df[df['variant_id'].notna()].copy()

    # variant_id -> str
    df['variant_id'] = df['variant_id'].astype(str)

    # chromosome -> str, strip 'chr' prefix if present
    if 'chromosome' in df.columns:
        df['chromosome'] = df['chromosome'].astype(str)
        df['chromosome'] = df['chromosome'].str.replace(r'^chr', '', regex=True, case=False)

    # position -> Int64 (nullable int)
    if 'position' in df.columns:
        df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')

    # effect_allele -> str, uppercase
    df['effect_allele'] = df['effect_allele'].astype(str).str.upper()

    # other_allele -> str, uppercase
    if 'other_allele' in df.columns:
        df['other_allele'] = df['other_allele'].astype(str).str.upper()

    # beta -> float
    df['beta'] = pd.to_numeric(df['beta'], errors='coerce').astype(float)

    return df


def _detect_delimiter(file_path: Path) -> str:
    """
    Detect the delimiter used in a file.

    Parameters
    ----------
    file_path : Path
        Path to the file.

    Returns
    -------
    str
        Detected delimiter character.
    """
    import gzip

    # Determine if file is gzipped
    open_func = gzip.open if str(file_path).endswith('.gz') else open
    mode = 'rt' if str(file_path).endswith('.gz') else 'r'

    with open_func(file_path, mode) as f:
        # Read first non-comment line
        for line in f:
            if not line.startswith('#'):
                break
        else:
            # No non-comment lines found
            return ','

        # Try csv.Sniffer
        try:
            dialect = csv.Sniffer().sniff(line, delimiters='\t,; ')
            return dialect.delimiter
        except csv.Error:
            pass

        # Fallback: check for tab, then comma, then whitespace
        if '\t' in line:
            return '\t'
        elif ',' in line:
            return ','
        else:
            return r'\s+'


def load_prs_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a PRS definition DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PRS variant weights.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical column names.

    Raises
    ------
    ValidationError
        If required columns are missing or data is invalid.
    """
    # Normalize column names
    df = _normalize_column_names(df)

    # Validate required columns
    _validate_prs_dataframe(df)

    # Coerce data types
    df = _coerce_data_types(df)

    return df


def load_prs_from_file(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a PRS definition from a file.

    Parameters
    ----------
    path : str or Path
        Path to PRS file. Supports CSV, TSV, and gzipped formats.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical column names.

    Raises
    ------
    DataLoadError
        If file cannot be read or parsed.
    ValidationError
        If required columns are missing.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        # Detect delimiter
        delimiter = _detect_delimiter(path)

        # Read file, skipping comment lines
        df = pd.read_csv(
            path,
            sep=delimiter,
            comment='#',
            engine='python' if delimiter == r'\s+' else 'c',
        )
    except Exception as e:
        raise DataLoadError(f"Failed to read file {path}: {e}") from e

    # Use the DataFrame loader for validation and normalization
    return load_prs_from_dataframe(df)
