"""Load reference genotypes from VCF and PLINK formats.

This module provides functions for loading genotype data from common file formats
used in genetic studies:
- VCF (Variant Call Format) files, including gzipped and bgzipped variants
- PLINK binary files (.bed/.bim/.fam)

The primary use case is loading reference panel data (e.g., 1000 Genomes Project)
for training imputation models.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.types import GenotypeData

logger = logging.getLogger(__name__)


def load_genotypes(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
    **kwargs,
) -> GenotypeData:
    """Load genotypes with automatic format detection.

    Detects the file format from the extension and calls the appropriate loader.

    Args:
        path: Path to genotype file (VCF or PLINK prefix).
        variant_ids: Optional set of variant IDs to filter to. Supports both
            rsID format (e.g., "rs123") and chr:pos format (e.g., "1:12345").
        samples: Optional list of sample IDs to include.
        **kwargs: Additional arguments passed to the format-specific loader.

    Returns:
        GenotypeData containing the loaded genotypes.

    Raises:
        DataLoadError: If the file cannot be loaded.
        ValidationError: If no variants match the filter.
    """
    path = Path(path)
    fmt = _detect_genotype_format(path)

    if fmt == "vcf":
        return load_genotypes_vcf(path, variant_ids=variant_ids, samples=samples, **kwargs)
    elif fmt == "plink":
        return load_genotypes_plink(path, variant_ids=variant_ids, samples=samples)
    else:
        raise DataLoadError(f"Unknown genotype format for file: {path}")


def load_genotypes_vcf(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
    dosage_field: str = "auto",
) -> GenotypeData:
    """Load genotypes from a VCF file.

    Args:
        path: Path to VCF file (.vcf, .vcf.gz, or .bcf).
        variant_ids: Optional set of variant IDs to filter to. Supports both
            rsID format (e.g., "rs123") and chr:pos format (e.g., "1:12345").
        samples: Optional list of sample IDs to include.
        dosage_field: How to extract dosage values. Options:
            - "auto": Try DS, then GT, then GP (default)
            - "GT": Convert genotype calls to dosage (0/0=0, 0/1=1, 1/1=2)
            - "DS": Use dosage field directly
            - "GP": Compute expected dosage from genotype probabilities

    Returns:
        GenotypeData containing the loaded genotypes.

    Raises:
        DataLoadError: If the file cannot be loaded.
        ValidationError: If no variants match the filter.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"VCF file not found: {path}")

    return _load_vcf_with_cyvcf2(path, variant_ids, samples, dosage_field)


def load_genotypes_plink(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
) -> GenotypeData:
    """Load genotypes from PLINK binary files.

    Args:
        path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        variant_ids: Optional set of variant IDs to filter to. Supports both
            rsID format (e.g., "rs123") and chr:pos format (e.g., "1:12345").
        samples: Optional list of sample IDs to include.

    Returns:
        GenotypeData containing the loaded genotypes.

    Raises:
        DataLoadError: If the files cannot be loaded.
        ValidationError: If no variants match the filter.
    """
    path = Path(path)
    # Remove extension if provided
    if path.suffix in (".bed", ".bim", ".fam"):
        path = path.with_suffix("")

    return _load_plink_with_pandas_plink(path, variant_ids, samples)


def _detect_genotype_format(path: Path) -> str:
    """Detect genotype file format from extension.

    Args:
        path: Path to the genotype file.

    Returns:
        Format string: "vcf" or "plink".
    """
    path_str = str(path).lower()

    # VCF formats
    if path_str.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")):
        return "vcf"

    # PLINK formats
    if path_str.endswith((".bed", ".bim", ".fam")):
        return "plink"

    # Check if PLINK files exist with this prefix
    if Path(str(path) + ".bed").exists():
        return "plink"

    return "unknown"


def _normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome name.

    Strips 'chr' prefix and normalizes sex/mitochondrial chromosomes.

    Args:
        chrom: Chromosome string (e.g., "chr1", "CHR22", "chrX").

    Returns:
        Normalized chromosome (e.g., "1", "22", "X").
    """
    chrom = str(chrom).upper()
    # Strip chr prefix
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    # Normalize common aliases
    if chrom == "MT":
        chrom = "M"
    return chrom


def _build_variant_lookup(
    variant_ids: Optional[Set[str]],
) -> Tuple[Set[str], Set[str]]:
    """Build lookup sets for variant filtering.

    Args:
        variant_ids: Set of variant IDs in rsID or chr:pos format.

    Returns:
        Tuple of (rsid_set, chrpos_set) for matching.
    """
    if variant_ids is None:
        return set(), set()

    rsid_set = set()
    chrpos_set = set()

    for vid in variant_ids:
        if vid.startswith("rs"):
            rsid_set.add(vid.lower())
        elif ":" in vid:
            # Normalize chr:pos format
            parts = vid.split(":")
            if len(parts) >= 2:
                chrom = _normalize_chromosome(parts[0])
                pos = parts[1]
                chrpos_set.add(f"{chrom}:{pos}")
        else:
            # Assume it's an rsID-like identifier
            rsid_set.add(vid.lower())

    return rsid_set, chrpos_set


def _variant_matches(
    variant_id: str,
    chrom: str,
    pos: int,
    rsid_set: Set[str],
    chrpos_set: Set[str],
) -> bool:
    """Check if a variant matches the filter sets.

    Args:
        variant_id: Variant ID (rsID or similar).
        chrom: Normalized chromosome.
        pos: Position.
        rsid_set: Set of rsIDs to match.
        chrpos_set: Set of chr:pos strings to match.

    Returns:
        True if variant matches either set.
    """
    # Empty sets mean no filtering
    if not rsid_set and not chrpos_set:
        return True

    # Check rsID match
    if variant_id and variant_id.lower() in rsid_set:
        return True

    # Check chr:pos match
    chrpos = f"{chrom}:{pos}"
    if chrpos in chrpos_set:
        return True

    return False


def _load_vcf_with_cyvcf2(
    path: Path,
    variant_ids: Optional[Set[str]],
    samples: Optional[List[str]],
    dosage_field: str,
) -> GenotypeData:
    """Load VCF using cyvcf2 library.

    Args:
        path: Path to VCF file.
        variant_ids: Optional set of variant IDs to filter.
        samples: Optional list of sample IDs.
        dosage_field: Dosage extraction method.

    Returns:
        GenotypeData with loaded genotypes.
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        raise DataLoadError(
            "cyvcf2 is required for VCF loading. Install with: pip install cyvcf2"
        )

    try:
        vcf = VCF(str(path), samples=samples)
    except Exception as e:
        raise DataLoadError(f"Failed to open VCF file {path}: {e}")

    # Get sample list
    sample_ids = list(vcf.samples)
    if not sample_ids:
        raise ValidationError("VCF file contains no samples")

    # Build variant lookup
    rsid_set, chrpos_set = _build_variant_lookup(variant_ids)
    filter_variants = bool(rsid_set or chrpos_set)

    # Collect data
    dosages: List[np.ndarray] = []
    variant_records: List[Dict] = []
    n_missing_total = 0
    n_genotypes_total = 0

    for variant in vcf:
        chrom = _normalize_chromosome(variant.CHROM)
        pos = variant.POS
        var_id = variant.ID if variant.ID else f"{chrom}:{pos}"

        # Apply variant filter
        if filter_variants:
            if not _variant_matches(var_id, chrom, pos, rsid_set, chrpos_set):
                continue

        # Extract dosage
        dosage = _extract_dosage(variant, dosage_field, len(sample_ids))
        if dosage is None:
            logger.warning(f"Could not extract dosage for variant {var_id}, skipping")
            continue

        dosages.append(dosage)
        variant_records.append({
            "variant_id": var_id,
            "chromosome": chrom,
            "position": pos,
            "ref_allele": variant.REF,
            "alt_allele": variant.ALT[0] if variant.ALT else None,
        })

        # Track missing rate
        n_missing = np.sum(np.isnan(dosage))
        n_missing_total += n_missing
        n_genotypes_total += len(dosage)

    vcf.close()

    if not dosages:
        if filter_variants:
            raise ValidationError(
                f"No variants matched the filter. Requested {len(variant_ids)} variants."
            )
        else:
            raise ValidationError("No variants found in VCF file")

    # Build output
    # Stack as (n_variants x n_samples) then transpose to (n_samples x n_variants)
    dosage_matrix = np.vstack(dosages).T.astype(np.float32)

    variant_info = pd.DataFrame(variant_records)

    # Log missing rate
    if n_genotypes_total > 0:
        missing_rate = n_missing_total / n_genotypes_total
        if missing_rate > 0.01:
            logger.warning(
                f"Missing genotype rate: {missing_rate:.1%} "
                f"({n_missing_total}/{n_genotypes_total})"
            )

    logger.info(
        f"Loaded {len(variant_records)} variants for {len(sample_ids)} samples from {path}"
    )

    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=sample_ids,
        source_file=str(path),
    )


def _extract_dosage(variant, dosage_field: str, n_samples: int) -> Optional[np.ndarray]:
    """Extract dosage values from a VCF variant record.

    Args:
        variant: cyvcf2 Variant object.
        dosage_field: Extraction method ("auto", "GT", "DS", "GP").
        n_samples: Expected number of samples.

    Returns:
        Array of dosage values (0-2, NaN for missing), or None if extraction fails.
    """
    if dosage_field == "auto":
        # Try DS first, then GT, then GP
        dosage = _dosage_from_ds(variant, n_samples)
        if dosage is not None:
            return dosage
        dosage = _dosage_from_gt(variant, n_samples)
        if dosage is not None:
            return dosage
        dosage = _dosage_from_gp(variant, n_samples)
        return dosage
    elif dosage_field.upper() == "DS":
        return _dosage_from_ds(variant, n_samples)
    elif dosage_field.upper() == "GT":
        return _dosage_from_gt(variant, n_samples)
    elif dosage_field.upper() == "GP":
        return _dosage_from_gp(variant, n_samples)
    else:
        raise ValidationError(f"Unknown dosage_field: {dosage_field}")


def _dosage_from_gt(variant, n_samples: int) -> Optional[np.ndarray]:
    """Convert GT (genotype) field to dosage.

    Converts:
    - 0/0 -> 0
    - 0/1 or 1/0 -> 1
    - 1/1 -> 2
    - ./. or missing -> NaN

    Args:
        variant: cyvcf2 Variant object.
        n_samples: Expected number of samples.

    Returns:
        Array of dosage values, or None if GT not available.
    """
    try:
        gt = variant.gt_types
        if gt is None:
            return None

        # cyvcf2 gt_types: 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT
        dosage = np.zeros(n_samples, dtype=np.float32)
        dosage[gt == 0] = 0.0  # HOM_REF
        dosage[gt == 1] = 1.0  # HET
        dosage[gt == 3] = 2.0  # HOM_ALT
        dosage[gt == 2] = np.nan  # UNKNOWN/MISSING

        return dosage
    except Exception:
        return None


def _dosage_from_ds(variant, n_samples: int) -> Optional[np.ndarray]:
    """Extract DS (dosage) field directly.

    Args:
        variant: cyvcf2 Variant object.
        n_samples: Expected number of samples.

    Returns:
        Array of dosage values, or None if DS not available.
    """
    try:
        ds = variant.format("DS")
        if ds is None:
            return None

        dosage = ds.flatten().astype(np.float32)
        # Replace -1 (missing) with NaN
        dosage[dosage < 0] = np.nan

        return dosage
    except Exception:
        return None


def _dosage_from_gp(variant, n_samples: int) -> Optional[np.ndarray]:
    """Compute expected dosage from GP (genotype probability) field.

    Computes: P(het) + 2*P(hom_alt) = GP[1] + 2*GP[2]

    Args:
        variant: cyvcf2 Variant object.
        n_samples: Expected number of samples.

    Returns:
        Array of dosage values, or None if GP not available.
    """
    try:
        gp = variant.format("GP")
        if gp is None:
            return None

        # GP is typically (n_samples, 3) for biallelic: P(0/0), P(0/1), P(1/1)
        if gp.shape[1] < 3:
            return None

        # Expected dosage = P(het) + 2*P(hom_alt)
        dosage = (gp[:, 1] + 2 * gp[:, 2]).astype(np.float32)

        # Handle missing (typically -1 in GP)
        missing_mask = np.any(gp < 0, axis=1)
        dosage[missing_mask] = np.nan

        return dosage
    except Exception:
        return None


def _load_plink_with_pandas_plink(
    path: Path,
    variant_ids: Optional[Set[str]],
    samples: Optional[List[str]],
) -> GenotypeData:
    """Load PLINK files using pandas-plink library.

    Args:
        path: Path prefix for PLINK files.
        variant_ids: Optional set of variant IDs to filter.
        samples: Optional list of sample IDs.

    Returns:
        GenotypeData with loaded genotypes.
    """
    try:
        from pandas_plink import read_plink1_bin
    except ImportError:
        raise DataLoadError(
            "pandas-plink is required for PLINK loading. "
            "Install with: pip install pandas-plink"
        )

    bed_path = Path(str(path) + ".bed")
    if not bed_path.exists():
        raise DataLoadError(f"PLINK file not found: {bed_path}")

    try:
        # Read PLINK files
        (bim, fam, genotype) = read_plink1_bin(str(bed_path))
    except Exception as e:
        raise DataLoadError(f"Failed to read PLINK files: {e}")

    # Extract sample IDs
    sample_ids = fam["iid"].tolist()

    # Build variant info from bim
    variant_info = pd.DataFrame({
        "variant_id": bim["snp"].values,
        "chromosome": [_normalize_chromosome(c) for c in bim["chrom"].values],
        "position": bim["pos"].values.astype(int),
        "ref_allele": bim["a0"].values,  # Reference allele
        "alt_allele": bim["a1"].values,  # Alternate allele
    })

    # Apply variant filter
    if variant_ids:
        rsid_set, chrpos_set = _build_variant_lookup(variant_ids)

        mask = np.zeros(len(variant_info), dtype=bool)
        for i, row in variant_info.iterrows():
            if _variant_matches(
                row["variant_id"],
                row["chromosome"],
                row["position"],
                rsid_set,
                chrpos_set,
            ):
                mask[i] = True

        if not mask.any():
            raise ValidationError(
                f"No variants matched the filter. Requested {len(variant_ids)} variants."
            )

        variant_info = variant_info[mask].reset_index(drop=True)
        # genotype is a dask array (variants x samples), we need to slice it
        genotype = genotype[mask, :]

    # Apply sample filter
    if samples:
        sample_mask = fam["iid"].isin(samples)
        if not sample_mask.any():
            raise ValidationError("No samples matched the filter")
        sample_ids = fam.loc[sample_mask, "iid"].tolist()
        genotype = genotype[:, sample_mask.values]

    # Convert to numpy and transpose to (samples x variants)
    # PLINK genotypes: 0=hom_ref, 1=het, 2=hom_alt, NaN=missing
    dosage_matrix = genotype.compute().T.astype(np.float32)

    logger.info(
        f"Loaded {len(variant_info)} variants for {len(sample_ids)} samples from {path}"
    )

    return GenotypeData(
        dosage_matrix=dosage_matrix,
        variant_info=variant_info,
        sample_ids=sample_ids,
        source_file=str(path),
    )
