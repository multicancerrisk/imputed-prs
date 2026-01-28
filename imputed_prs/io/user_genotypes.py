"""User genotype loading for DTC genetic testing files.

This module provides functions for loading user genotype data from direct-to-consumer
(DTC) genetic testing files (23andMe, AncestryDNA, etc.) and converting them to
dosage values for PRS calculation.

Supported input formats:
- File paths: DTC format auto-detected via snps package
- SNPs objects: Direct usage of pre-loaded snps.SNPs instances
- DataFrames: Must have 'rsid' index or column and 'genotype' column
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError

if TYPE_CHECKING:
    from snps import SNPs

logger = logging.getLogger(__name__)


def load_user_genotypes(
    input_data: Union[str, Path, "SNPs", pd.DataFrame],
    expected_variants: Optional[Set[str]] = None,
) -> Dict[str, Optional[float]]:
    """Load user genotypes and convert to dosage values.

    Supports multiple input types:
    - File path: DTC format auto-detected via snps package (23andMe, AncestryDNA, etc.)
    - SNPs object: Direct usage of pre-loaded snps.SNPs instance
    - DataFrame: Must have 'rsid' index or column and 'genotype' column

    Args:
        input_data: User genotype data (file path, SNPs object, or DataFrame).
        expected_variants: Optional set of variant IDs to extract. If None,
            returns all variants found in the input.

    Returns:
        Dictionary mapping variant_id to dosage value (0.0, 1.0, 2.0) or None
        for missing/invalid genotypes.

    Raises:
        DataLoadError: If file cannot be loaded or parsed.
        ValidationError: If input data format is invalid.
    """
    # Import SNPs here to check instance type without requiring it at module load
    try:
        from snps import SNPs
    except ImportError:
        SNPs = None

    if isinstance(input_data, (str, Path)):
        return _load_from_file(input_data, expected_variants)
    elif SNPs is not None and isinstance(input_data, SNPs):
        return _load_from_snps(input_data, expected_variants)
    elif isinstance(input_data, pd.DataFrame):
        return _load_from_dataframe(input_data, expected_variants)
    else:
        raise ValidationError(f"Unsupported input type: {type(input_data)}")


def genotype_to_dosage(genotype: str) -> Optional[float]:
    """Convert genotype string to dosage value.

    Counts alleles to determine dosage:
    - Homozygous (AA, GG, etc.) -> 2.0
    - Heterozygous (AG, CT, etc.) -> 1.0
    - Missing/invalid -> None

    Note: For PRS calculation, the effect allele direction is determined by
    the PRS weights, not the raw genotypes. This function simply counts
    alleles without assuming which is the effect allele.

    Args:
        genotype: Genotype string (e.g., "AA", "AG", "GG", "--", "I", "D")

    Returns:
        Dosage value (0.0, 1.0, 2.0) or None if missing/invalid.
    """
    if genotype is None or pd.isna(genotype):
        return None

    genotype = str(genotype).strip().upper()

    # Missing data indicators
    if genotype in ("--", "", "NA", "N/A", "NULL", "..", "NN", "00"):
        return None

    # Single character (haploid or indel)
    if len(genotype) == 1:
        if genotype in ("I", "D", "N", "-", "0"):
            return None
        # Single allele - treat as missing (conservative approach)
        return None

    # Standard diploid genotype (2 characters)
    if len(genotype) == 2:
        allele1, allele2 = genotype[0], genotype[1]

        # Check for missing alleles
        if allele1 in ("-", "N", "0") or allele2 in ("-", "N", "0"):
            return None

        # Indel genotypes
        if allele1 in ("I", "D") or allele2 in ("I", "D"):
            return None

        # Count matching alleles (homozygous = 2, heterozygous = 1)
        if allele1 == allele2:
            return 2.0  # Homozygous
        else:
            return 1.0  # Heterozygous

    return None  # Unrecognized format


def detect_genome_build(input_data: Union[str, Path, "SNPs"]) -> Optional[int]:
    """Detect genome build from user genotype data.

    Uses the snps package's build detection algorithm which examines
    known SNP positions to determine if the data is GRCh37 (build 37)
    or GRCh38 (build 38).

    Args:
        input_data: File path or SNPs object.

    Returns:
        Build number (37 or 38) or None if not detected.

    Raises:
        DataLoadError: If file cannot be loaded.
    """
    try:
        from snps import SNPs
    except ImportError:
        raise DataLoadError(
            "snps package is required for genome build detection. "
            "Install with: pip install snps"
        )

    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        try:
            snps_obj = SNPs(str(path))
        except Exception as e:
            raise DataLoadError(f"Failed to parse genotype file: {e}") from e
    else:
        snps_obj = input_data

    return snps_obj.build


def _load_from_file(
    path: Union[str, Path],
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a DTC file using the snps package.

    Args:
        path: Path to the genotype file.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.

    Raises:
        DataLoadError: If file cannot be loaded.
    """
    try:
        from snps import SNPs
    except ImportError:
        raise DataLoadError(
            "snps package is required for loading DTC genotype files. "
            "Install with: pip install snps"
        )

    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        snps_obj = SNPs(str(path))
    except Exception as e:
        raise DataLoadError(f"Failed to parse genotype file: {e}") from e

    if snps_obj.snps is None or snps_obj.snps.empty:
        raise DataLoadError(f"No genotype data found in {path}")

    logger.info(
        f"Loaded {len(snps_obj.snps)} variants from {path} "
        f"(source: {snps_obj.source}, build: {snps_obj.build})"
    )

    return _load_from_snps(snps_obj, expected_variants)


def _load_from_snps(
    snps_obj: "SNPs",
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a snps.SNPs object.

    Args:
        snps_obj: Pre-loaded SNPs object.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.
    """
    df = snps_obj.snps  # DataFrame with index=rsid, columns=[chrom, pos, genotype]

    dosages: Dict[str, Optional[float]] = {}

    if expected_variants is not None:
        # Only process expected variants
        for variant_id in expected_variants:
            if variant_id in df.index:
                genotype = df.loc[variant_id, "genotype"]
                dosages[variant_id] = genotype_to_dosage(genotype)
            else:
                dosages[variant_id] = None  # Mark as missing
    else:
        # Process all variants
        for variant_id in df.index:
            genotype = df.loc[variant_id, "genotype"]
            dosages[variant_id] = genotype_to_dosage(genotype)

    return dosages


def _load_from_dataframe(
    df: pd.DataFrame,
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a DataFrame.

    Args:
        df: DataFrame with variant IDs as index or in 'rsid'/'variant_id' column,
            and genotypes in 'genotype' column.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.

    Raises:
        ValidationError: If DataFrame format is invalid.
    """
    # Make a copy to avoid modifying the input
    df = df.copy()

    # Ensure we have a genotype column
    if "genotype" not in df.columns:
        raise ValidationError("DataFrame must have a 'genotype' column")

    # Set index from rsid or variant_id column if not already indexed
    if df.index.name != "rsid" and "rsid" not in str(df.index.dtype):
        if "rsid" in df.columns:
            df = df.set_index("rsid")
        elif "variant_id" in df.columns:
            df = df.set_index("variant_id")
        # Otherwise assume index is already variant IDs

    dosages: Dict[str, Optional[float]] = {}

    if expected_variants is not None:
        for variant_id in expected_variants:
            if variant_id in df.index:
                genotype = df.loc[variant_id, "genotype"]
                dosages[variant_id] = genotype_to_dosage(genotype)
            else:
                dosages[variant_id] = None
    else:
        for variant_id in df.index:
            genotype = df.loc[variant_id, "genotype"]
            dosages[variant_id] = genotype_to_dosage(genotype)

    return dosages
