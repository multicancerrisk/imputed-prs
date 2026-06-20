"""PRS definition loading and normalization."""

import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.harmonizer import _normalize_build, _normalize_chromosome


logger = logging.getLogger(__name__)

# Valid allele pattern: one or more A/C/G/T characters (after uppercasing). Multi-character
# matches (e.g. "ATG") are accepted so indel weights survive, while symbolic indel tokens
# ("I", "D", "-"), empty strings, and numeric junk are rejected.
_ALLELE_RE = re.compile(r"[ACGT]+")

# Canonical genome builds recognized for the optional genome_build column.
_RECOGNIZED_BUILDS = frozenset({"GRCh37", "GRCh38"})


COLUMN_ALIASES = {
    'variant_id': ['rsid', 'snp', 'snp_id', 'variant', 'id', 'marker', 'markername'],
    'chromosome': ['chr', 'chrom', '#chrom', 'chr_name'],
    'position': ['pos', 'bp', 'chr_position', 'base_pair_location'],
    'effect_allele': ['allele1', 'a1', 'alt', 'effect', 'ea', 'risk_allele'],
    'other_allele': ['allele2', 'a2', 'ref', 'non_effect_allele', 'nea', 'reference_allele'],
    'beta': ['effect_weight', 'weight', 'effect_size', 'log_or'],
    'genome_build': ['build', 'assembly', 'genome_assembly'],
}

REQUIRED_COLUMNS = ['variant_id', 'chromosome', 'position', 'effect_allele', 'beta']


def _normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names to canonical names using aliases.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially non-standard column names.

    Returns
    -------
    (pd.DataFrame, dict)
        Copy of the DataFrame with canonical column names, plus a map from each
        resolved canonical name to the (lower-cased) source column it came from.
        The source map lets callers detect unsafe inferences such as ``effect_allele``
        being taken from an ``alt`` column.
    """
    df = df.copy()

    # Create lowercase mapping of current columns
    lowercase_to_original = {col.lower(): col for col in df.columns}

    # Build rename mapping and record which source produced each canonical column
    rename_map = {}
    source_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        # Check if canonical name already exists (case-insensitive)
        if canonical.lower() in lowercase_to_original:
            original_col = lowercase_to_original[canonical.lower()]
            source_map[canonical] = canonical.lower()
            if original_col != canonical:
                rename_map[original_col] = canonical
            continue

        # Check aliases
        for alias in aliases:
            if alias.lower() in lowercase_to_original:
                original_col = lowercase_to_original[alias.lower()]
                rename_map[original_col] = canonical
                source_map[canonical] = alias.lower()
                break

    df.rename(columns=rename_map, inplace=True)
    return df, source_map


def _check_alt_as_effect(source_map: Dict[str, str], allow_alt_as_effect: bool) -> None:
    """
    Guard against silently treating an ``alt`` column as the effect allele.

    For generic GWAS summary statistics the ALT allele is not guaranteed to be the
    effect allele, so inferring ``effect_allele`` from an ``alt`` column would
    silently mis-orient every weight. Raise unless the caller opts in.

    Raises
    ------
    ValidationError
        If ``effect_allele`` was inferred from an ``alt`` column and
        ``allow_alt_as_effect`` is False.
    """
    if source_map.get('effect_allele') == 'alt' and not allow_alt_as_effect:
        raise ValidationError(
            "effect_allele was inferred from an 'alt' column, but ALT is not "
            "guaranteed to be the effect allele in generic summary statistics. "
            "Provide an explicit 'effect_allele' column, or pass "
            "allow_alt_as_effect=True to treat ALT as the effect allele."
        )


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


def _validate_genome_build_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize an optional ``genome_build`` column.

    Recognized aliases (hg19/hg38/b37/b38/...) are normalized in place to
    GRCh37/GRCh38. Any unrecognized build is a hard error, since downstream
    build-compatibility checks rely on a correct build.

    Raises
    ------
    ValidationError
        If the column contains an unrecognized build.
    """
    if 'genome_build' not in df.columns:
        return df

    df = df.copy()
    normalized = df['genome_build'].apply(
        lambda b: _normalize_build(str(b)) if pd.notna(b) else None
    )
    bad = sorted({
        str(orig)
        for orig, norm in zip(df['genome_build'], normalized)
        if pd.notna(orig) and norm not in _RECOGNIZED_BUILDS
    })
    if bad:
        raise ValidationError(
            f"Unrecognized genome_build value(s): {bad}. Expected one of "
            f"{sorted(_RECOGNIZED_BUILDS)} or a known alias (hg19, hg38, b37, b38)."
        )
    df['genome_build'] = normalized

    distinct = {b for b in normalized if b is not None}
    if len(distinct) > 1:
        warnings.warn(
            f"PRS definition declares multiple genome builds: {sorted(distinct)}.",
            UserWarning,
            stacklevel=2,
        )
    return df


def _is_valid_allele(allele: object) -> bool:
    """Return True if ``allele`` is a non-empty string of A/C/G/T characters."""
    if allele is None or (isinstance(allele, float) and pd.isna(allele)):
        return False
    return _ALLELE_RE.fullmatch(str(allele)) is not None


def _warn_dropped(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Emit a UserWarning describing rows that are about to be dropped."""
    n = int(mask.sum())
    if n == 0:
        return
    sample_ids = df.loc[mask, 'variant_id'].head(5).tolist()
    warnings.warn(
        f"Dropping {n} PRS variant(s) with {reason}; examples: {sample_ids}.",
        UserWarning,
        stacklevel=2,
    )
    logger.warning("Dropped %d PRS variant(s) with %s", n, reason)


def _coerce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to appropriate data types and drop rows with unusable values.

    Rows with a null variant_id, blank/normalized-missing chromosome, null position,
    non-numeric beta, or a malformed effect_allele are dropped (with a warning). A
    malformed other_allele is blanked to ``<NA>`` rather than dropping the row, since
    other_allele is permissive for research loading.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized column names; chromosome/position are guaranteed
        present by the required-columns check that runs first.

    Returns
    -------
    pd.DataFrame
        DataFrame with coerced data types and unusable rows removed.
    """
    df = df.copy()

    # Drop rows where variant_id is null before conversion
    df = df[df['variant_id'].notna()].copy()

    # variant_id -> str
    df['variant_id'] = df['variant_id'].astype(str)

    # chromosome -> normalized str (strip 'chr', MT->M), matching partition_variants
    df['chromosome'] = df['chromosome'].apply(
        lambda c: _normalize_chromosome(str(c)) if pd.notna(c) else ''
    )

    # position -> Int64 (nullable int)
    df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')

    # effect_allele -> str, uppercase
    df['effect_allele'] = df['effect_allele'].astype(str).str.upper()

    # other_allele -> str, uppercase
    if 'other_allele' in df.columns:
        df['other_allele'] = df['other_allele'].astype(str).str.upper()

    # beta -> float (non-numeric becomes NaN, filtered below)
    df['beta'] = pd.to_numeric(df['beta'], errors='coerce').astype(float)

    # --- Row-level validity filtering (drop + warn) ---
    chrom_bad = df['chromosome'].isin(['', 'NAN', 'NONE'])
    _warn_dropped(df, chrom_bad, "missing/blank chromosome")

    pos_bad = df['position'].isna()
    _warn_dropped(df, pos_bad, "missing position")

    beta_bad = df['beta'].isna()
    _warn_dropped(df, beta_bad, "non-numeric beta")

    effect_bad = ~df['effect_allele'].map(_is_valid_allele)
    _warn_dropped(df, effect_bad, "invalid effect_allele")

    drop_mask = chrom_bad | pos_bad | beta_bad | effect_bad
    df = df[~drop_mask].copy()

    # Malformed other_allele -> <NA> (keep the row; other_allele is permissive)
    if 'other_allele' in df.columns:
        other_bad = ~df['other_allele'].map(_is_valid_allele)
        if other_bad.any():
            warnings.warn(
                f"Blanked {int(other_bad.sum())} malformed other_allele value(s).",
                UserWarning,
                stacklevel=2,
            )
            df.loc[other_bad, 'other_allele'] = pd.NA

    return df


def _resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve duplicate variant_ids.

    Rows identical across id+locus+alleles+beta are collapsed to one. Rows sharing a
    variant_id but differing in locus/alleles are kept (legitimate multi-allelic
    variants, e.g. a shared rsID across ALT alleles). Rows with the same
    id+locus+alleles but conflicting beta are dropped with a warning, since the
    intended weight is ambiguous.

    Parameters
    ----------
    df : pd.DataFrame
        Coerced DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates resolved.
    """
    identity_cols = ['variant_id', 'chromosome', 'position', 'effect_allele']
    if 'other_allele' in df.columns:
        identity_cols.append('other_allele')

    # Collapse fully-identical rows (same identity AND same beta).
    df = df.drop_duplicates(subset=identity_cols + ['beta']).copy()

    # Any remaining duplicates on identity => conflicting beta for the same locus/alleles.
    conflict_mask = df.duplicated(subset=identity_cols, keep=False)
    if conflict_mask.any():
        conflict_ids = sorted(set(df.loc[conflict_mask, 'variant_id']))
        warnings.warn(
            f"Dropping {int(conflict_mask.sum())} PRS row(s) for {len(conflict_ids)} "
            f"variant(s) with conflicting beta at the same locus/alleles; examples: "
            f"{conflict_ids[:5]}.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "Dropped %d conflicting-beta PRS row(s)", int(conflict_mask.sum())
        )
        df = df[~conflict_mask].copy()

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


def load_prs_from_dataframe(
    df: pd.DataFrame,
    allow_alt_as_effect: bool = False,
) -> pd.DataFrame:
    """
    Validate and normalize a PRS definition DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PRS variant weights.
    allow_alt_as_effect : bool, default False
        If True, permit ``effect_allele`` to be taken from an ``alt`` column. By
        default this raises, because ALT is not guaranteed to be the effect allele
        in generic summary statistics.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical column names. Rows with unusable values
        are dropped (with warnings) and duplicate variant_ids are resolved.

    Raises
    ------
    ValidationError
        If required columns are missing, ALT is inferred as the effect allele
        without opt-in, or the genome_build column has an unrecognized value.
    """
    # Normalize column names (and learn which source produced each canonical column)
    df, source_map = _normalize_column_names(df)

    # Guard against silently treating ALT as the effect allele
    _check_alt_as_effect(source_map, allow_alt_as_effect)

    # Validate required columns (variant_id, chromosome, position, effect_allele, beta)
    _validate_prs_dataframe(df)

    # other_allele is recommended but permissive for research loading
    if 'other_allele' not in df.columns:
        warnings.warn(
            "PRS definition has no other_allele column; strand-complement and "
            "multi-allelic handling will be limited, and the deployable export "
            "requires it.",
            UserWarning,
            stacklevel=2,
        )

    # Validate/normalize the optional genome_build column
    df = _validate_genome_build_column(df)

    # Coerce data types and drop rows with unusable values
    df = _coerce_data_types(df)

    # Resolve duplicate variant_ids
    df = _resolve_duplicates(df)

    return df.reset_index(drop=True)


def load_prs_from_file(
    path: Union[str, Path],
    allow_alt_as_effect: bool = False,
) -> pd.DataFrame:
    """
    Load a PRS definition from a file.

    Parameters
    ----------
    path : str or Path
        Path to PRS file. Supports CSV, TSV, and gzipped formats.
    allow_alt_as_effect : bool, default False
        Forwarded to :func:`load_prs_from_dataframe`.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical column names.

    Raises
    ------
    DataLoadError
        If file cannot be read or parsed.
    ValidationError
        If required columns are missing or the schema is invalid.
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
    return load_prs_from_dataframe(df, allow_alt_as_effect=allow_alt_as_effect)
