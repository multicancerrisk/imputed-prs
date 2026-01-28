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
from imputed_prs.io.platform_loader import (
    load_platform_variants_from_list,
    load_platform_from_manifest,
    load_platform_from_name,
    list_available_platforms,
    get_platform_info,
)
from imputed_prs.io.genotype_loader import (
    load_genotypes,
    load_genotypes_vcf,
    load_genotypes_plink,
)
from imputed_prs.io.user_genotypes import (
    load_user_genotypes,
    genotype_to_dosage,
    detect_genome_build,
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
    # Platform loader
    "load_platform_variants_from_list",
    "load_platform_from_manifest",
    "load_platform_from_name",
    "list_available_platforms",
    "get_platform_info",
    # Genotype loader
    "load_genotypes",
    "load_genotypes_vcf",
    "load_genotypes_plink",
    # User genotype loader
    "load_user_genotypes",
    "genotype_to_dosage",
    "detect_genome_build",
]
