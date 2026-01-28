"""PGS Catalog API integration for downloading and caching PRS scoring files."""

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.io.prs_loader import load_prs_from_dataframe


# Constants
PGS_CATALOG_REST_BASE = "https://www.pgscatalog.org/rest"
PGS_CATALOG_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"
SUPPORTED_BUILDS = ("GRCh37", "GRCh38")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imputed_prs" / "pgs_catalog"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
HM_CODE_VALID_THRESHOLD = 0  # Filter variants with hm_code < 0

# Build aliases for normalization
BUILD_ALIASES = {
    "hg19": "GRCh37",
    "hg38": "GRCh38",
    "grch37": "GRCh37",
    "grch38": "GRCh38",
}


@dataclass
class PGSCatalogMetadata:
    """Metadata for a PGS Catalog score."""

    pgs_id: str
    name: str
    trait_reported: str
    trait_efo: List[str]
    variants_number: int
    genome_build: Optional[str]
    ftp_scoring_file: Optional[str]
    ftp_harmonized_files: Dict[str, str] = field(default_factory=dict)
    publication_doi: Optional[str] = None
    publication_pmid: Optional[str] = None
    date_release: Optional[str] = None


@dataclass
class PGSSearchResult:
    """Search result from PGS Catalog."""

    pgs_id: str
    name: str
    trait_reported: str
    variants_number: int


def _get_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """
    Get or create the cache directory.

    Parameters
    ----------
    cache_dir : Path, optional
        Custom cache directory. Uses DEFAULT_CACHE_DIR if not specified.

    Returns
    -------
    Path
        Path to the cache directory.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _get_metadata_cache_path(pgs_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Get the cache path for metadata JSON."""
    return _get_cache_dir(cache_dir) / f"{pgs_id}_metadata.json"


def _get_scoring_file_cache_path(
    pgs_id: str, build: str, cache_dir: Optional[Path] = None
) -> Path:
    """Get the cache path for a scoring file."""
    return _get_cache_dir(cache_dir) / f"{pgs_id}_hmPOS_{build}.txt.gz"


def _request_with_retry(
    url: str, timeout: int = DEFAULT_TIMEOUT, max_retries: int = MAX_RETRIES
) -> requests.Response:
    """
    Make an HTTP GET request with exponential backoff retry.

    Parameters
    ----------
    url : str
        URL to fetch.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    requests.Response
        The response object.

    Raises
    ------
    DataLoadError
        If all retries fail.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, ...
                wait_time = 2**attempt
                time.sleep(wait_time)

    raise DataLoadError(f"Failed to fetch {url} after {max_retries} attempts: {last_error}")


def _validate_pgs_id(pgs_id: str) -> str:
    """
    Validate and normalize a PGS ID.

    Parameters
    ----------
    pgs_id : str
        PGS ID to validate (e.g., "PGS000004" or "pgs000004").

    Returns
    -------
    str
        Normalized PGS ID (uppercase).

    Raises
    ------
    ValidationError
        If the PGS ID format is invalid.
    """
    pgs_id = pgs_id.strip().upper()

    # Check format: PGS followed by exactly 6 digits
    if not re.match(r"^PGS\d{6}$", pgs_id):
        raise ValidationError(
            f"Invalid PGS ID format: '{pgs_id}'. "
            "Expected format: PGS followed by 6 digits (e.g., PGS000004)"
        )

    return pgs_id


def _validate_genome_build(build: str) -> str:
    """
    Validate and normalize a genome build.

    Parameters
    ----------
    build : str
        Genome build to validate (e.g., "GRCh37", "hg19").

    Returns
    -------
    str
        Normalized genome build (GRCh37 or GRCh38).

    Raises
    ------
    ValidationError
        If the genome build is not supported.
    """
    build_lower = build.strip().lower()

    if build_lower in BUILD_ALIASES:
        return BUILD_ALIASES[build_lower]

    if build in SUPPORTED_BUILDS:
        return build

    raise ValidationError(
        f"Unsupported genome build: '{build}'. "
        f"Supported builds: {', '.join(SUPPORTED_BUILDS)} (or hg19/hg38)"
    )


def _parse_api_metadata(data: Dict[str, Any]) -> PGSCatalogMetadata:
    """
    Parse API JSON response into PGSCatalogMetadata.

    Parameters
    ----------
    data : dict
        JSON response from the PGS Catalog API.

    Returns
    -------
    PGSCatalogMetadata
        Parsed metadata object.
    """
    # Extract trait EFO IDs
    trait_efo = []
    for trait in data.get("trait_efo", []):
        if isinstance(trait, dict) and "id" in trait:
            trait_efo.append(trait["id"])
        elif isinstance(trait, str):
            trait_efo.append(trait)

    # Extract FTP URLs for harmonized files
    ftp_harmonized = {}
    ftp_harmonized_info = data.get("ftp_harmonized_scoring_files", {})
    for build_key, build_info in ftp_harmonized_info.items():
        if isinstance(build_info, dict) and "positions" in build_info:
            ftp_harmonized[build_key] = build_info["positions"]

    # Extract publication info
    publication = data.get("publication", {}) or {}
    pub_doi = publication.get("doi")
    pub_pmid = publication.get("PMID")

    return PGSCatalogMetadata(
        pgs_id=data["id"],
        name=data.get("name", ""),
        trait_reported=data.get("trait_reported", ""),
        trait_efo=trait_efo,
        variants_number=data.get("variants_number", 0),
        genome_build=data.get("original_genome_build"),
        ftp_scoring_file=data.get("ftp_scoring_file"),
        ftp_harmonized_files=ftp_harmonized,
        publication_doi=pub_doi,
        publication_pmid=pub_pmid,
        date_release=data.get("date_release"),
    )


def _load_cached_metadata(
    pgs_id: str, cache_dir: Optional[Path] = None
) -> Optional[PGSCatalogMetadata]:
    """
    Load metadata from cache if it exists.

    Parameters
    ----------
    pgs_id : str
        PGS ID.
    cache_dir : Path, optional
        Custom cache directory.

    Returns
    -------
    PGSCatalogMetadata or None
        Cached metadata if found, None otherwise.
    """
    cache_path = _get_metadata_cache_path(pgs_id, cache_dir)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return PGSCatalogMetadata(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            # Invalid cache file, ignore it
            return None
    return None


def _save_metadata_to_cache(
    metadata: PGSCatalogMetadata, cache_dir: Optional[Path] = None
) -> None:
    """
    Save metadata to cache.

    Parameters
    ----------
    metadata : PGSCatalogMetadata
        Metadata to cache.
    cache_dir : Path, optional
        Custom cache directory.
    """
    cache_path = _get_metadata_cache_path(metadata.pgs_id, cache_dir)
    data = {
        "pgs_id": metadata.pgs_id,
        "name": metadata.name,
        "trait_reported": metadata.trait_reported,
        "trait_efo": metadata.trait_efo,
        "variants_number": metadata.variants_number,
        "genome_build": metadata.genome_build,
        "ftp_scoring_file": metadata.ftp_scoring_file,
        "ftp_harmonized_files": metadata.ftp_harmonized_files,
        "publication_doi": metadata.publication_doi,
        "publication_pmid": metadata.publication_pmid,
        "date_release": metadata.date_release,
    }
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def _get_harmonized_file_url(pgs_id: str, build: str) -> str:
    """
    Construct the URL for a harmonized scoring file.

    Parameters
    ----------
    pgs_id : str
        PGS ID (validated and normalized).
    build : str
        Genome build (validated and normalized).

    Returns
    -------
    str
        URL to the harmonized scoring file.
    """
    # URL structure: {FTP_BASE}/{pgs_id}/ScoringFiles/Harmonized/{pgs_id}_hmPOS_{build}.txt.gz
    return (
        f"{PGS_CATALOG_FTP_BASE}/{pgs_id}/ScoringFiles/Harmonized/"
        f"{pgs_id}_hmPOS_{build}.txt.gz"
    )


def _download_scoring_file(url: str, cache_path: Path) -> None:
    """
    Download a gzipped scoring file to the cache.

    Parameters
    ----------
    url : str
        URL to download from.
    cache_path : Path
        Path to save the file.

    Raises
    ------
    DataLoadError
        If download fails.
    """
    try:
        response = _request_with_retry(url)
        with open(cache_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        raise DataLoadError(f"Failed to download scoring file from {url}: {e}") from e


def _read_scoring_file(path: Path) -> pd.DataFrame:
    """
    Read a scoring file, skipping comment lines.

    Parameters
    ----------
    path : Path
        Path to the gzipped scoring file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame from the file.

    Raises
    ------
    DataLoadError
        If file cannot be read.
    """
    try:
        return pd.read_csv(path, sep="\t", comment="#", compression="gzip")
    except Exception as e:
        raise DataLoadError(f"Failed to read scoring file {path}: {e}") from e


def _process_harmonized_columns(
    df: pd.DataFrame, filter_failed_mappings: bool = True
) -> pd.DataFrame:
    """
    Process harmonized columns, preferring hm_* columns and filtering failed mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from harmonized scoring file.
    filter_failed_mappings : bool
        If True, filter out variants with hm_code < 0.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with renamed columns.
    """
    df = df.copy()

    # Filter failed mappings if requested
    if filter_failed_mappings and "hm_code" in df.columns:
        df = df[df["hm_code"] >= HM_CODE_VALID_THRESHOLD].copy()

    # Column mapping: prefer harmonized columns
    column_renames = {}

    # Use hm_rsID if available, otherwise fall back to rsID
    if "hm_rsID" in df.columns:
        column_renames["hm_rsID"] = "rsID"
        # Drop original rsID if present
        if "rsID" in df.columns:
            df = df.drop(columns=["rsID"])

    # Use hm_chr if available
    if "hm_chr" in df.columns:
        column_renames["hm_chr"] = "chr_name"
        if "chr_name" in df.columns:
            df = df.drop(columns=["chr_name"])

    # Use hm_pos if available
    if "hm_pos" in df.columns:
        column_renames["hm_pos"] = "chr_position"
        if "chr_position" in df.columns:
            df = df.drop(columns=["chr_position"])

    df = df.rename(columns=column_renames)

    # Generate variant_id from position if rsID is not available or all null
    rsid_col = "rsID" if "rsID" in df.columns else None
    has_valid_rsids = rsid_col and df[rsid_col].notna().any()

    if not has_valid_rsids:
        # Generate variant_id from chr:pos:effect_allele
        chr_col = "chr_name" if "chr_name" in df.columns else None
        pos_col = "chr_position" if "chr_position" in df.columns else None

        if chr_col and pos_col and "effect_allele" in df.columns:
            df["variant_id"] = (
                df[chr_col].astype(str)
                + ":"
                + df[pos_col].astype(str)
                + ":"
                + df["effect_allele"].astype(str)
            )

    return df


def fetch_pgs_catalog_metadata(
    pgs_id: str, use_cache: bool = True, cache_dir: Optional[Path] = None
) -> PGSCatalogMetadata:
    """
    Fetch metadata for a PGS Catalog score.

    Parameters
    ----------
    pgs_id : str
        PGS Catalog score ID (e.g., "PGS000004").
    use_cache : bool
        If True, use cached metadata if available.
    cache_dir : Path, optional
        Custom cache directory.

    Returns
    -------
    PGSCatalogMetadata
        Score metadata.

    Raises
    ------
    ValidationError
        If PGS ID format is invalid.
    DataLoadError
        If metadata cannot be fetched.
    """
    pgs_id = _validate_pgs_id(pgs_id)

    # Check cache first
    if use_cache:
        cached = _load_cached_metadata(pgs_id, cache_dir)
        if cached is not None:
            return cached

    # Fetch from API
    url = f"{PGS_CATALOG_REST_BASE}/score/{pgs_id}"
    response = _request_with_retry(url)

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON response from PGS Catalog API: {e}") from e

    metadata = _parse_api_metadata(data)

    # Cache the result
    _save_metadata_to_cache(metadata, cache_dir)

    return metadata


def download_pgs_catalog_score(
    pgs_id: str,
    genome_build: str = "GRCh37",
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    filter_failed_mappings: bool = True,
) -> Tuple[pd.DataFrame, PGSCatalogMetadata]:
    """
    Download a PGS Catalog scoring file and normalize it.

    Parameters
    ----------
    pgs_id : str
        PGS Catalog score ID (e.g., "PGS000004").
    genome_build : str
        Genome build ("GRCh37" or "GRCh38", or "hg19"/"hg38").
    cache_dir : Path, optional
        Custom cache directory.
    use_cache : bool
        If True, use cached files if available.
    filter_failed_mappings : bool
        If True, filter out variants with hm_code < 0 (failed mappings).

    Returns
    -------
    tuple
        (DataFrame, PGSCatalogMetadata): Normalized scoring DataFrame and metadata.

    Raises
    ------
    ValidationError
        If PGS ID or genome build is invalid.
    DataLoadError
        If files cannot be downloaded or parsed.
    """
    pgs_id = _validate_pgs_id(pgs_id)
    genome_build = _validate_genome_build(genome_build)

    # Get metadata
    metadata = fetch_pgs_catalog_metadata(pgs_id, use_cache=use_cache, cache_dir=cache_dir)

    # Check for cached scoring file
    cache_path = _get_scoring_file_cache_path(pgs_id, genome_build, cache_dir)

    if not use_cache or not cache_path.exists():
        # Download the file
        url = _get_harmonized_file_url(pgs_id, genome_build)
        _download_scoring_file(url, cache_path)

    # Read and process the file
    df = _read_scoring_file(cache_path)
    df = _process_harmonized_columns(df, filter_failed_mappings=filter_failed_mappings)

    # Normalize using the standard loader
    df = load_prs_from_dataframe(df)

    return df, metadata


def search_pgs_catalog(
    trait_query: str, limit: int = 10
) -> List[PGSSearchResult]:
    """
    Search the PGS Catalog for scores by trait.

    Parameters
    ----------
    trait_query : str
        Search query for traits (e.g., "breast cancer").
    limit : int
        Maximum number of results to return.

    Returns
    -------
    list
        List of PGSSearchResult objects.

    Raises
    ------
    ValidationError
        If query is empty.
    DataLoadError
        If search fails.
    """
    if not trait_query or not trait_query.strip():
        raise ValidationError("Search query cannot be empty")

    trait_query = trait_query.strip()

    # Search for traits
    url = f"{PGS_CATALOG_REST_BASE}/trait/search?term={requests.utils.quote(trait_query)}"
    response = _request_with_retry(url)

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON response from PGS Catalog API: {e}") from e

    # Collect PGS IDs from matching traits
    pgs_ids = set()
    results_data = data.get("results", [])
    for trait in results_data:
        associated_scores = trait.get("associated_pgs_ids", [])
        pgs_ids.update(associated_scores)
        if len(pgs_ids) >= limit * 2:  # Fetch extra in case some fail
            break

    # Fetch metadata for each PGS ID
    results = []
    for pgs_id in list(pgs_ids)[:limit * 2]:
        try:
            metadata = fetch_pgs_catalog_metadata(pgs_id)
            results.append(
                PGSSearchResult(
                    pgs_id=metadata.pgs_id,
                    name=metadata.name,
                    trait_reported=metadata.trait_reported,
                    variants_number=metadata.variants_number,
                )
            )
            if len(results) >= limit:
                break
        except (ValidationError, DataLoadError):
            # Skip scores that fail to fetch
            continue

    return results


def clear_pgs_catalog_cache(cache_dir: Optional[Path] = None) -> int:
    """
    Clear all cached PGS Catalog files.

    Parameters
    ----------
    cache_dir : Path, optional
        Custom cache directory.

    Returns
    -------
    int
        Number of files removed.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    if not cache.exists():
        return 0

    count = 0
    for file_path in cache.iterdir():
        if file_path.is_file():
            file_path.unlink()
            count += 1

    return count


def get_pgs_catalog_cache_info(cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about the PGS Catalog cache.

    Parameters
    ----------
    cache_dir : Path, optional
        Custom cache directory.

    Returns
    -------
    dict
        Cache statistics including:
        - path: Cache directory path
        - n_files: Number of cached files
        - size_bytes: Total size in bytes
        - size_mb: Total size in megabytes
        - cached_scores: List of cached PGS IDs
    """
    cache = cache_dir or DEFAULT_CACHE_DIR

    if not cache.exists():
        return {
            "path": str(cache),
            "n_files": 0,
            "size_bytes": 0,
            "size_mb": 0.0,
            "cached_scores": [],
        }

    files = list(cache.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    # Extract unique PGS IDs from filenames
    pgs_ids = set()
    for f in files:
        match = re.match(r"(PGS\d{6})", f.name)
        if match:
            pgs_ids.add(match.group(1))

    return {
        "path": str(cache),
        "n_files": len([f for f in files if f.is_file()]),
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "cached_scores": sorted(pgs_ids),
    }


# Alias for backward compatibility with API design spec
fetch_pgs_catalog_score = download_pgs_catalog_score
