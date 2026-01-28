# Imputed PRS

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Calculate Polygenic Risk Scores via linear imputation of missing variants.

## Overview

The library enables the calculation of a specified Polygenic Risk Score (PRS) on data from a fixed target genotyping platform, by performing linear imputation of missing variants at prediction time. Instead of requiring the computationally expensive haplotype-based imputation (e.g., IMPUTE5, Minimac), we train lightweight linear imputation models using a reference panel. These models predict the missing PRS variants from observed, genotyped variants using regularized regression, leveraging linkage disequilibrium (LD).

Key innovations:
- **Linear imputation**: Uses ElasticNet regression to impute missing variants from nearby platform variants
- **Calibrated uncertainty**: Provides confidence intervals that account for imputation error
- **Cross-validation**: All imputation quality metrics are computed out-of-fold to prevent overfitting

## Features

- **PGS Catalog integration**: Search, download, and use scores directly from the [PGS Catalog](https://www.pgscatalog.org/)
- **DTC platform support**: Built-in support for 23andMe (v3-v5) and AncestryDNA (v1-v2)
- **Multiple export formats**: JSON, HDF5, Parquet, and Arrow for deployment flexibility
- **Calibrated uncertainty**: Confidence intervals account for imputation error
- **Evaluation tools**: Cross-validation, sensitivity analysis, and diagnostic plots
- **Hyperparameter tuning**: Grid search and Bayesian optimization (via Optuna)

## Installation

```bash
pip install imputed-prs
```

### Optional dependencies

```bash
# For diagnostic plotting
pip install imputed-prs[plotting]

# For Bayesian hyperparameter optimization
pip install imputed-prs[optuna]

# For development
pip install imputed-prs[dev]
```

### Development installation

```bash
git clone https://github.com/multicancerrisk/imputed-prs.git
cd imputed-prs
pip install -e ".[dev,plotting,optuna]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[dev,plotting,optuna]"
```

## Quick Start

### Train a model

```python
from imputed_prs import LinearImputationPRS

# Initialize model
model = LinearImputationPRS(
    window_size=1_000_000,  # 1 Mb window for predictors
    cv_folds=5,             # 5-fold cross-validation
    n_jobs=-1,              # Use all CPU cores
)

# Train on reference genotypes (e.g., 1000 Genomes)
model.fit(
    reference_genotypes="1000g_eur.vcf.gz",
    prs_definition="PGS000004",  # Download from PGS Catalog
    platform_name="23andme_v5",   # Target platform
)

# View training summary
print(model.summary)
```

### Predict on user genotypes

```python
# Load user genotypes and compute PRS
result = model.predict("user_genotypes.txt")

print(f"PRS: {result.prs:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Variants used: {result.n_variants_used}")
print(f"Variants imputed: {result.n_variants_imputed}")
```

### Export trained model

```python
# Export to multiple formats for deployment
paths = model.export(
    output_dir="./models",
    model_name="breast_cancer_prs",
    formats=["json", "hdf5"],
)
print(f"Exported to: {paths}")

# Later, load and use
loaded_model = LinearImputationPRS.load("./models/breast_cancer_prs.h5")
result = loaded_model.predict("new_user.txt")
```

## Supported Platforms

Built-in support for common DTC genotyping platforms:

```python
from imputed_prs import list_available_platforms, get_platform_info

# List available platforms
platforms = list_available_platforms()
print(platforms)
# ['23andme_v3', '23andme_v4', '23andme_v5', 'ancestrydna_v1', 'ancestrydna_v2']

# Get platform details
info = get_platform_info("23andme_v5")
print(f"{info.display_name}: {info.n_variants:,} variants ({info.genome_build})")
```

### Custom platforms

You can also use custom platform manifests:

```python
# From a manifest file
model.fit(
    reference_genotypes="reference.vcf.gz",
    prs_definition="PGS000004",
    platform_manifest="my_platform_variants.txt",
)

# From a list of variant IDs
model.fit(
    reference_genotypes="reference.vcf.gz",
    prs_definition="PGS000004",
    platform_variants=["rs123", "rs456", "rs789", ...],
)
```

## PGS Catalog Integration

Search and download scores directly from the PGS Catalog:

```python
from imputed_prs import search_pgs_catalog, fetch_pgs_catalog_score

# Search for scores by trait
results = search_pgs_catalog("breast cancer", limit=5)
for r in results:
    print(f"{r.pgs_id}: {r.name} ({r.variants_number} variants)")

# Download a score (with caching)
prs_df, metadata = fetch_pgs_catalog_score(
    "PGS000004",
    genome_build="GRCh37",
)
print(f"Downloaded {len(prs_df)} variants for {metadata.trait_reported}")
```

## Export Formats

Trained models can be exported to multiple formats:

| Format | Use Case | Dependencies |
|--------|----------|--------------|
| **JSON** | Web deployment, JavaScript clients | None |
| **HDF5** | Python/R analysis, large models | h5py |
| **Parquet** | Data warehouses, Spark/Dask | pyarrow |
| **Arrow** | Zero-copy IPC, streaming | pyarrow |
| **CSV** | Variant table inspection | None |

```python
# Export to all formats
paths = model.export(
    output_dir="./models",
    formats=["json", "hdf5", "parquet", "arrow", "csv"],
)
```

## Evaluation & Diagnostics

### Cross-validation

```python
from imputed_prs import LinearImputationPRS, ImputationEvaluator

model = LinearImputationPRS().fit(...)
evaluator = ImputationEvaluator(model)

# Evaluate on held-out data
metrics = evaluator.evaluate("held_out_genotypes.vcf.gz")
print(f"Correlation: {metrics.correlation:.3f}")
print(f"R²: {metrics.r2:.3f}")

# K-fold cross-validation
cv_result = evaluator.cross_validate(
    reference_genotypes="reference.vcf.gz",
    prs_definition="PGS000004",
    platform_name="23andme_v5",
    n_folds=5,
)
print(f"CV R²: {cv_result.mean_r2:.3f} ± {cv_result.std_r2:.3f}")
```

### Diagnostic plots

```python
from imputed_prs.evaluation.plotting import (
    plot_calibration,
    plot_imputation_quality,
    plot_variance_contribution,
)

# Calibration plot (imputed vs true PRS)
fig, ax = plot_calibration(imputed_prs, true_prs)
fig.savefig("calibration.png")

# Imputation quality distribution
fig, ax = plot_imputation_quality(model.imputed_models)
fig.savefig("quality.png")

# Top variance contributors
fig, ax = plot_variance_contribution(model.imputed_models, top_n=20)
fig.savefig("variance.png")
```

### Variant-level summary

```python
# Get per-variant summary table
variant_df = model.variant_table
print(variant_df[["variant_id", "status", "imputation_r2", "n_predictors"]])
```

## API Reference

For detailed API documentation, see [docs/API.md](docs/API.md).

### Key classes and functions

| Name | Description |
|------|-------------|
| `LinearImputationPRS` | Main class for training and prediction |
| `ImputationEvaluator` | Evaluation tools for fitted models |
| `list_available_platforms()` | List supported genotyping platforms |
| `get_platform_info()` | Get platform metadata |
| `search_pgs_catalog()` | Search PGS Catalog by trait |
| `fetch_pgs_catalog_score()` | Download PRS from PGS Catalog |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Submit a pull request

For bug reports and feature requests, please open an issue on [GitHub](https://github.com/multicancerrisk/imputed-prs/issues).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use imputed-prs in your research, please cite:

```bibtex
@software{imputed_prs,
  title = {imputed-prs: Polygenic Risk Score calculation with linear imputation},
  url = {https://github.com/multicancerrisk/imputed-prs},
}
```
