"""I/O module for reading and writing files."""

from imputed_prs.io.prs_loader import (
    load_prs_from_dataframe,
    load_prs_from_file,
)
from imputed_prs.io.pgs_catalog import (
    fetch_pgs_catalog_metadata,
    download_pgs_catalog_score,
    fetch_pgs_catalog_score,
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
    load_user_genotype_strings,
    load_raw_user_genotypes,
    genotype_to_dosage,
    count_allele,
    detect_genome_build,
)
from imputed_prs.io.exporters.json_export import export_to_json
from imputed_prs.io.exporters.arrow_export import export_to_arrow, export_to_parquet
from imputed_prs.io.exporters.hdf5_export import export_to_hdf5
from imputed_prs.io.exporters.csv_export import export_variant_table
from imputed_prs.io.loaders import (
    load_model_hdf5,
    load_model_json,
    load_model_arrow,
    load_model_parquet,
    load_model_csv,
)

__all__ = [
    # PRS loader
    "load_prs_from_dataframe",
    "load_prs_from_file",
    # PGS Catalog
    "fetch_pgs_catalog_metadata",
    "download_pgs_catalog_score",
    "fetch_pgs_catalog_score",
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
    "load_user_genotype_strings",
    "load_raw_user_genotypes",
    "genotype_to_dosage",
    "count_allele",
    "detect_genome_build",
    # Exporters
    "export_to_json",
    "export_to_arrow",
    "export_to_parquet",
    "export_to_hdf5",
    "export_variant_table",
    # Loaders
    "load_model_hdf5",
    "load_model_json",
    "load_model_arrow",
    "load_model_parquet",
    "load_model_csv",
]
