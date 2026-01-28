"""I/O module for reading and writing files."""

from imputed_prs.io.prs_loader import (
    load_prs_from_dataframe,
    load_prs_from_file,
)
from imputed_prs.io.pgs_catalog import (
    fetch_pgs_catalog_metadata,
    download_pgs_catalog_score,
    search_pgs_catalog,
    clear_pgs_catalog_cache,
    get_pgs_catalog_cache_info,
    PGSCatalogMetadata,
    PGSSearchResult,
)

__all__ = [
    # PRS loader
    "load_prs_from_dataframe",
    "load_prs_from_file",
    # PGS Catalog
    "fetch_pgs_catalog_metadata",
    "download_pgs_catalog_score",
    "search_pgs_catalog",
    "clear_pgs_catalog_cache",
    "get_pgs_catalog_cache_info",
    "PGSCatalogMetadata",
    "PGSSearchResult",
]
