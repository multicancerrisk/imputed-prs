"""Platform definitions for genetic testing platforms.

This package contains:
- JSON metadata files for each supported platform (23andme_v5.json, ancestrydna_v2.json)
- Variant list files (*.txt.gz) that are downloaded on first use from Zenodo via the snps library

Supported platforms:
- 23andme_v5: 23andMe V5 chip (Illumina GSA, GRCh37)
- ancestrydna_v2: AncestryDNA V2 chip (Illumina OmniExpress Plus, GRCh37)

Variant data is sourced from the chip cluster data at:
https://zenodo.org/records/5047472
"""
