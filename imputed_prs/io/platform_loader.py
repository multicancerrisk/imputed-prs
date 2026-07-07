"""Platform manifest loading for genotyping platform variant lists.

This module provides functions to load variant lists from genotyping platforms
like 23andMe V5 and AncestryDNA V2. It supports loading from:
- Pre-built platform definitions bundled with the package
- Custom manifest files (CSV, TSV, plain text, gzipped)
- Direct lists of variant IDs
"""

import gzip
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.types import PlatformInfo

logger = logging.getLogger(__name__)

# Supported platforms
SUPPORTED_PLATFORMS = (
    "23andme_v3",
    "23andme_v4",
    "23andme_v5",
    "ancestrydna_v1",
    "ancestrydna_v2",
)

# Validation patterns for variant IDs
VALID_RSID_PATTERN = re.compile(r"^rs\d+$", re.IGNORECASE)
VALID_CHRPOS_PATTERN = re.compile(r"^(\d{1,2}|X|Y|MT?):(\d+)$", re.IGNORECASE)

# Module-level cache for loaded platforms
_platform_cache: Dict[str, Tuple[Set[str], PlatformInfo]] = {}


def _get_package_data_dir() -> Path:
    """Get the path to the package data/platforms directory.

    Returns
    -------
    Path
        Path to the data/platforms directory.
    """
    return Path(__file__).parent.parent / "data" / "platforms"


def _validate_platform_name(name: str) -> str:
    """Normalize and validate a platform name.

    Parameters
    ----------
    name : str
        Platform name to validate.

    Returns
    -------
    str
        Normalized platform name (lowercase).

    Raises
    ------
    ValidationError
        If the platform name is not supported.
    """
    normalized = name.strip().lower()

    if normalized not in SUPPORTED_PLATFORMS:
        raise ValidationError(
            f"Unknown platform: '{name}'. "
            f"Supported platforms: {', '.join(SUPPORTED_PLATFORMS)}"
        )

    return normalized


def _validate_variant_id(variant_id: str) -> bool:
    """Check if a variant ID has a valid format.

    Valid formats are:
    - rsID: rs followed by digits (e.g., rs123)
    - chr:pos: chromosome:position (e.g., 1:12345, X:67890)

    Parameters
    ----------
    variant_id : str
        Variant ID to validate.

    Returns
    -------
    bool
        True if valid format, False otherwise.
    """
    if not variant_id or not isinstance(variant_id, str):
        return False

    variant_id = variant_id.strip()

    if VALID_RSID_PATTERN.match(variant_id):
        return True

    if VALID_CHRPOS_PATTERN.match(variant_id):
        return True

    return False


def _detect_manifest_format(path: Path) -> str:
    """Detect the format of a manifest file.

    Parameters
    ----------
    path : Path
        Path to the manifest file.

    Returns
    -------
    str
        Detected format: "csv", "tsv", or "plain".
    """
    # Read first few lines to detect format
    is_gzipped = path.suffix == ".gz" or str(path).endswith(".gz")

    try:
        if is_gzipped:
            with gzip.open(path, "rt") as f:
                first_lines = [f.readline() for _ in range(5)]
        else:
            with open(path) as f:
                first_lines = [f.readline() for _ in range(5)]
    except Exception as e:
        raise DataLoadError(f"Failed to read manifest file {path}: {e}") from e

    # Check if it looks like CSV/TSV with headers
    first_line = first_lines[0].strip()

    if "\t" in first_line:
        return "tsv"
    elif "," in first_line:
        return "csv"
    else:
        # Plain text with one variant per line
        return "plain"


def _load_variants_from_gzip(path: Path) -> Set[str]:
    """Load variant IDs from a gzipped plain text file.

    Parameters
    ----------
    path : Path
        Path to the gzipped file.

    Returns
    -------
    Set[str]
        Set of variant IDs.

    Raises
    ------
    DataLoadError
        If the file cannot be read.
    """
    try:
        with gzip.open(path, "rt") as f:
            variants = set()
            for line in f:
                variant_id = line.strip()
                if variant_id:
                    variants.add(variant_id)
            return variants
    except Exception as e:
        raise DataLoadError(f"Failed to read gzipped variant file {path}: {e}") from e


def _load_platform_metadata(name: str) -> PlatformInfo:
    """Load platform metadata from JSON file.

    Parameters
    ----------
    name : str
        Normalized platform name.

    Returns
    -------
    PlatformInfo
        Platform metadata.

    Raises
    ------
    DataLoadError
        If the metadata file cannot be loaded.
    """
    data_dir = _get_package_data_dir()
    metadata_path = data_dir / f"{name}.json"

    if not metadata_path.exists():
        raise DataLoadError(
            f"Platform metadata file not found: {metadata_path}. "
            "The platform data may not be installed correctly."
        )

    try:
        with open(metadata_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in platform metadata file: {e}") from e

    return PlatformInfo(
        name=data["name"],
        display_name=data["display_name"],
        description=data["description"],
        genome_build=data["genome_build"],
        n_variants=data["n_variants"],
        chip_technology=data["chip_technology"],
        company=data["company"],
        version=data["version"],
        date_introduced=data.get("date_introduced"),
        source_url=data.get("source_url"),
    )


def _download_chip_clusters():
    """Download chip cluster data via snps.resources.

    Returns
    -------
    pandas.DataFrame
        DataFrame with chip cluster information.

    Raises
    ------
    DataLoadError
        If the download fails.
    """
    try:
        from snps.resources import Resources

        r = Resources()
        chip_clusters = r.get_chip_clusters()
        return chip_clusters
    except ImportError as e:
        raise DataLoadError(
            "The 'snps' package is required for downloading platform data. "
            "Install it with: pip install snps"
        ) from e
    except Exception as e:
        raise DataLoadError(f"Failed to download chip cluster data: {e}") from e


def _extract_platform_variants(cluster_df, cluster_name: str) -> Set[str]:
    """Extract variant IDs for a specific platform from cluster data.

    Parameters
    ----------
    cluster_df : pandas.DataFrame
        DataFrame with chip cluster data. Expected columns: 'chrom', 'pos', 'clusters'
        where 'clusters' is a comma-separated string of cluster names.
    cluster_name : str
        Name of the cluster to filter on (e.g., 'v5' for 23andMe V5).

    Returns
    -------
    Set[str]
        Set of variant IDs in chr:pos format.
    """
    # Filter to rows where this cluster appears in the clusters column
    # The 'clusters' column contains comma-separated cluster names like "c1,c3,v5"
    # Use non-capturing groups (?:) to avoid pandas warning
    mask = cluster_df["clusters"].str.contains(rf"(?:^|,){cluster_name}(?:,|$)", regex=True)
    platform_variants = cluster_df[mask]

    # Create chr:pos variant IDs
    variant_ids = set()
    for _, row in platform_variants.iterrows():
        variant_id = f"{row['chrom']}:{row['pos']}"
        variant_ids.add(variant_id)

    return variant_ids


def _get_or_generate_variant_file(name: str) -> Set[str]:
    """Get variant list from file, or generate it from chip clusters.

    Parameters
    ----------
    name : str
        Normalized platform name.

    Returns
    -------
    Set[str]
        Set of variant IDs.
    """
    data_dir = _get_package_data_dir()
    variant_file = data_dir / f"{name}_variants.txt.gz"

    # If variant file exists and has content, load it
    if variant_file.exists():
        variants = _load_variants_from_gzip(variant_file)
        if variants:
            return variants

    # Otherwise, download and extract from chip clusters
    logger.info(f"Downloading chip cluster data for {name}...")

    cluster_df = _download_chip_clusters()

    # Map platform names to cluster columns
    cluster_mapping = {
        "23andme_v3": "c4",
        "23andme_v4": "c1",
        "23andme_v5": "v5",
        "ancestrydna_v1": "c3",
        "ancestrydna_v2": "c5",
    }

    cluster_name = cluster_mapping.get(name)
    if cluster_name is None:
        raise DataLoadError(f"No cluster mapping defined for platform: {name}")

    variants = _extract_platform_variants(cluster_df, cluster_name)

    # Cache the variants to file for future use
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(variant_file, "wt") as f:
            for variant_id in sorted(variants):
                f.write(f"{variant_id}\n")
        logger.info(f"Cached {len(variants)} variants to {variant_file}")
    except Exception as e:
        logger.warning(f"Could not cache variant file: {e}")

    return variants


def load_platform_variants_from_list(
    variant_list: List[str], name: Optional[str] = None
) -> Set[str]:
    """Load platform variants from a list of variant IDs.

    Parameters
    ----------
    variant_list : list of str
        List of variant IDs (rsIDs or chr:pos format).
    name : str, optional
        Name for the custom platform (for logging purposes).

    Returns
    -------
    Set[str]
        Set of valid, deduplicated variant IDs.

    Raises
    ------
    ValidationError
        If the variant list is empty or contains no valid IDs.

    Examples
    --------
    >>> variants = load_platform_variants_from_list(["rs123", "rs456", "1:12345"])
    >>> len(variants)
    3
    """
    if not variant_list:
        raise ValidationError("Variant list cannot be empty")

    valid_variants = set()
    invalid_count = 0

    for variant_id in variant_list:
        if isinstance(variant_id, str):
            variant_id = variant_id.strip()
            if _validate_variant_id(variant_id):
                valid_variants.add(variant_id)
            else:
                invalid_count += 1

    if invalid_count > 0:
        warnings.warn(
            f"Filtered {invalid_count} invalid variant IDs from the list",
            UserWarning,
        )

    if not valid_variants:
        raise ValidationError(
            "No valid variant IDs found in the list. "
            "Valid formats are rsIDs (e.g., rs123) or chr:pos (e.g., 1:12345)"
        )

    platform_name = name or "custom"
    logger.info(f"Loaded {len(valid_variants)} variants for platform '{platform_name}'")

    return valid_variants


def load_platform_from_manifest(
    path: str, name: Optional[str] = None
) -> Tuple[Set[str], Optional[PlatformInfo]]:
    """Load platform variants from a manifest file.

    Supports CSV, TSV, and plain text formats, with optional gzip compression.

    Parameters
    ----------
    path : str
        Path to the manifest file.
    name : str, optional
        Name for the custom platform (for logging purposes).

    Returns
    -------
    tuple
        (Set[str], None): Set of variant IDs and None (no metadata for custom manifests).

    Raises
    ------
    DataLoadError
        If the file cannot be read.
    ValidationError
        If no valid variants are found.

    Examples
    --------
    >>> variants, info = load_platform_from_manifest("my_platform.txt")
    >>> print(f"Loaded {len(variants)} variants")
    """
    file_path = Path(path)

    if not file_path.exists():
        raise DataLoadError(f"Manifest file not found: {path}")

    format_type = _detect_manifest_format(file_path)
    is_gzipped = file_path.suffix == ".gz" or str(file_path).endswith(".gz")

    variants = set()

    if format_type == "plain":
        # Plain text with one variant per line
        if is_gzipped:
            variants = _load_variants_from_gzip(file_path)
        else:
            try:
                with open(file_path) as f:
                    for line in f:
                        variant_id = line.strip()
                        if variant_id:
                            variants.add(variant_id)
            except Exception as e:
                raise DataLoadError(f"Failed to read manifest file: {e}") from e
    else:
        # CSV or TSV format
        import pandas as pd

        try:
            sep = "\t" if format_type == "tsv" else ","
            compression = "gzip" if is_gzipped else None
            df = pd.read_csv(file_path, sep=sep, compression=compression)

            # Look for variant ID column
            variant_columns = ["rsid", "variant_id", "snp", "name", "id"]
            variant_col = None

            for col in df.columns:
                if col.lower() in variant_columns:
                    variant_col = col
                    break

            if variant_col is None:
                # Use first column
                variant_col = df.columns[0]
                logger.info(f"Using first column '{variant_col}' as variant IDs")

            for val in df[variant_col]:
                if pd.notna(val):
                    variant_id = str(val).strip()
                    if variant_id:
                        variants.add(variant_id)

        except Exception as e:
            raise DataLoadError(f"Failed to read manifest file: {e}") from e

    # Validate variants
    valid_variants = set()
    invalid_count = 0

    for variant_id in variants:
        if _validate_variant_id(variant_id):
            valid_variants.add(variant_id)
        else:
            invalid_count += 1

    if invalid_count > 0:
        warnings.warn(
            f"Filtered {invalid_count} invalid variant IDs from manifest",
            UserWarning,
        )

    if not valid_variants:
        raise ValidationError(
            f"No valid variant IDs found in manifest file: {path}. "
            "Valid formats are rsIDs (e.g., rs123) or chr:pos (e.g., 1:12345)"
        )

    platform_name = name or file_path.stem
    logger.info(
        f"Loaded {len(valid_variants)} variants from manifest '{platform_name}'"
    )

    return valid_variants, None


def load_platform_from_name(
    platform_name: str,
) -> Tuple[Set[str], PlatformInfo]:
    """Load platform variants from a pre-built platform definition.

    Parameters
    ----------
    platform_name : str
        Name of the platform (e.g., "23andme_v5", "ancestrydna_v2").
        Case-insensitive.

    Returns
    -------
    tuple
        (Set[str], PlatformInfo): Set of variant IDs and platform metadata.

    Raises
    ------
    ValidationError
        If the platform name is not recognized.
    DataLoadError
        If platform data cannot be loaded.

    Examples
    --------
    >>> variants, info = load_platform_from_name("23andme_v5")
    >>> print(f"Loaded {len(variants)} variants from {info.display_name}")
    """
    normalized_name = _validate_platform_name(platform_name)

    # Check cache first
    if normalized_name in _platform_cache:
        return _platform_cache[normalized_name]

    # Load metadata
    metadata = _load_platform_metadata(normalized_name)

    # Load or generate variants
    variants = _get_or_generate_variant_file(normalized_name)

    # Update metadata with actual variant count if different
    if len(variants) != metadata.n_variants:
        logger.info(
            f"Platform {normalized_name}: expected {metadata.n_variants} variants, "
            f"loaded {len(variants)}"
        )

    # Cache the result
    _platform_cache[normalized_name] = (variants, metadata)

    return variants, metadata


def list_available_platforms() -> List[str]:
    """List all available pre-built platforms.

    Returns
    -------
    list of str
        List of supported platform names.

    Examples
    --------
    >>> platforms = list_available_platforms()
    >>> print(platforms)
    ['23andme_v5', 'ancestrydna_v2']
    """
    return list(SUPPORTED_PLATFORMS)


def get_platform_info(platform_name: str) -> PlatformInfo:
    """Get metadata for a platform without loading variants.

    This is faster than load_platform_from_name() when you only need
    metadata information.

    Parameters
    ----------
    platform_name : str
        Name of the platform (e.g., "23andme_v5").
        Case-insensitive.

    Returns
    -------
    PlatformInfo
        Platform metadata.

    Raises
    ------
    ValidationError
        If the platform name is not recognized.
    DataLoadError
        If metadata cannot be loaded.

    Examples
    --------
    >>> info = get_platform_info("23andme_v5")
    >>> print(f"{info.display_name}: {info.n_variants} variants")
    """
    normalized_name = _validate_platform_name(platform_name)

    # Check if already in cache
    if normalized_name in _platform_cache:
        return _platform_cache[normalized_name][1]

    return _load_platform_metadata(normalized_name)


def resolve_platform_variant_set(
    platform_name: Optional[str] = None,
    platform_manifest: Optional[str] = None,
    platform_variants: Optional[List[str]] = None,
) -> Tuple[Set[str], Optional[PlatformInfo], str]:
    """Resolve ``(variant_set, platform_info, effective_name)`` from one platform source.

    Centralizes the three-way ``platform_name`` / ``platform_manifest`` /
    ``platform_variants`` branch that ``fit``, ``cross_validate``,
    ``sensitivity_analysis`` and the reference-CV path each repeated. Callers resolve
    the reference **once** and thread the returned ``variant_set`` through every fold /
    combo / fit (via the internal ``_platform_variant_set`` hook) so the platform file
    is read once, not once per fold or per combo.

    Exactly one source is expected (callers validate). ``effective_name`` is the
    metadata label the callers derive today (``platform_name`` / the manifest stem /
    the sentinel ``"custom"``); ``platform_info`` is populated only for the by-name
    source (``None`` for manifest / list), matching the current ``fit`` behaviour.
    """
    if platform_name is not None:
        variant_set, info = load_platform_from_name(platform_name)
        return variant_set, info, platform_name
    if platform_manifest is not None:
        variant_set, _ = load_platform_from_manifest(str(platform_manifest))
        return variant_set, None, Path(platform_manifest).stem
    variant_set = load_platform_variants_from_list(platform_variants)
    return variant_set, None, "custom"
