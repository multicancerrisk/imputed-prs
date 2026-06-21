# Imputed PRS

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Estimate a target Polygenic Risk Score on a fixed genotyping platform, using lightweight linear models trained from a reference panel.

## Overview

The library estimates a specified Polygenic Risk Score (PRS) from data on a fixed target genotyping platform (e.g. a 23andMe or AncestryDNA chip), where many of the PRS variants are not directly genotyped. Instead of requiring computationally expensive haplotype-based imputation (e.g., IMPUTE5, Minimac), we train lightweight **linear** models from a reference panel that exploit linkage disequilibrium (LD) between observed, genotyped variants and the missing PRS variants. The trained model is small enough to export and run client-side, which is the eventual deployment target (a browser-based TypeScript scorer).

The library ships **two co-equal methods** for this:

- **`LinearProjectionPRS`** — directly learns, per genomic region, platform-variant weights that approximate that region's PRS contribution. This is the **deployment-preferred** method.
- **`LinearImputationPRS`** — imputes each missing PRS variant's dosage from nearby platform variants with ElasticNet regression, then scores the reconstructed PRS.

Both share the same training inputs, the same allele-aware prediction path, the same calibrated/empirical uncertainty machinery, and the same browser-ready v2 JSON export. See [Imputation vs projection](#imputation-vs-projection) for how to choose.

Key ideas:
- **Linear, LD-based reconstruction**: ElasticNet regression over a local genomic window — no haplotype reference at prediction time.
- **Allele-aware scoring**: user uploads are oriented to the model's effect/counted alleles using per-variant allele metadata (strand-safe, multiallelic-safe).
- **Calibrated + empirical uncertainty**: confidence intervals account for reconstruction error, using an LD-aware empirical residual SD with a per-prediction diagonal lower bound.
- **No silent drops**: every PRS variant has a universal per-variant observed fallback; variants that cannot be scored are surfaced, never dropped silently.
- **Out-of-fold metrics**: all reconstruction-quality and calibration metrics are computed cross-validated to prevent overfitting.

## Features

- **Two methods**: projection (`LinearProjectionPRS`, deployment-preferred) and imputation (`LinearImputationPRS`).
- **PGS Catalog integration**: Search, download, and use scores directly from the [PGS Catalog](https://www.pgscatalog.org/).
- **DTC platform support**: Built-in support for 23andMe (v3-v5) and AncestryDNA (v1-v2).
- **Allele-aware prediction**: file/DataFrame uploads are oriented to the model's alleles; strand-ambiguous (palindromic A/T, C/G) SNP handling is configurable.
- **Calibrated + empirical uncertainty**: intervals that account for reconstruction error (LD-aware empirical residual SD with a per-prediction diagonal lower bound).
- **Universal observed fallback**: a per-variant fallback model means no PRS variant is silently dropped when an upload can't resolve it.
- **Browser-ready v2 export**: portable JSON carrying per-variant allele metadata + provenance, designed for a client-side TypeScript scorer. See [docs/export-format.md](docs/export-format.md).
- **Masking-validation harness**: mask a reference panel down to a platform, score it through the deployed path, and compare to the full PRS (`imputed_prs.evaluation.run_masking_validation`).
- **Evaluation tools**: cross-validation, sensitivity analysis, and diagnostic plots.
- **Hyperparameter tuning**: grid search and Bayesian optimization (via Optuna).

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

### Train a projection model (deployment-preferred)

```python
from imputed_prs import LinearProjectionPRS

# Initialize model
model = LinearProjectionPRS(
    window_size=1_000_000,  # 1 Mb window for regions/predictors
    cv_folds=5,             # 5-fold cross-validation
    n_jobs=-1,              # Use all CPU cores
)

# Train on reference genotypes (e.g., 1000 Genomes).
# reference_panel_id + training_ancestry are provenance, and are REQUIRED
# for a deployable export (the browser scorer validates compatibility against them).
model.fit(
    reference_genotypes="1000g_eur.vcf.gz",
    prs_definition="PGS000004",        # Download from PGS Catalog
    platform_name="23andme_v5",         # Target platform
    reference_panel_id="1000G_phase3_EUR",
    training_ancestry="EUR",
)

# Predict, then export the browser-ready JSON artifact
result = model.predict("user_genotypes.txt")
print(f"PRS: {result.prs:.3f}  95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

paths = model.export(output_dir="./models", model_name="breast_cancer_prs")  # JSON only
loaded = LinearProjectionPRS.load("./models/breast_cancer_prs.json")
```

### Train an imputation model

```python
from imputed_prs import LinearImputationPRS

model = LinearImputationPRS(
    window_size=1_000_000,  # 1 Mb window for predictors
    cv_folds=5,             # 5-fold cross-validation
    n_jobs=-1,              # Use all CPU cores
)

model.fit(
    reference_genotypes="1000g_eur.vcf.gz",
    prs_definition="PGS000004",  # Download from PGS Catalog
    platform_name="23andme_v5",   # Target platform
    reference_panel_id="1000G_phase3_EUR",
    training_ancestry="EUR",
)

# View training summary
print(model.summary)
```

### Predict on user genotypes

```python
# Load user genotypes and compute PRS. File or DataFrame inputs are scored
# allele-aware (the upload is oriented to the model's effect/counted alleles).
result = model.predict("user_genotypes.txt")

print(f"PRS: {result.prs:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"SE: {result.se:.3f}")
print(f"Variants used: {result.n_variants_used}")
print(f"Variants imputed: {result.n_variants_imputed}")
print(f"Scored via fallback: {result.n_observed_scored_via_fallback}")
print(f"Unresolved (never silently dropped): {result.unresolved_observed_ids}")
```

`predict` also takes optional keyword-only arguments to guard against scoring an
incompatible upload:

```python
result = model.predict(
    "user_genotypes.txt",
    apply_calibration=True,   # default; also returns *_scaled fields
    genome_build="GRCh37",    # checked against the model's build
    platform_id="23andme_v5", # checked against the platform the model was trained for
    strict=True,              # default: raise on a build/platform mismatch
)
```

With `strict=True` (default), an incompatible genome build raises `IncompatibleBuildError`
and a declared platform mismatch raises `IncompatiblePlatformError`. With `strict=False`,
each mismatch is downgraded to a blocking `UserWarning` and scoring proceeds.

> **Legacy dict input (allele-blind).** `predict` also accepts a numeric
> `Dict[str, float]` of `variant_id -> dosage`, but this path is **legacy and
> allele-blind**: it bypasses allele orientation, so a strand- or allele-flipped
> upload will be silently mis-scored. Prefer a **file path or DataFrame** for
> correct, allele-aware scoring; use the dict only for trusted, already-oriented
> dosages. On the dict path the orientation diagnostics (`n_observed_scored_direct`,
> `unresolved_observed_ids`, ...) come back as `None`.

### Export trained model

```python
# Imputation models export to several formats.
paths = model.export(
    output_dir="./models",
    model_name="breast_cancer_prs",
    formats=["json", "hdf5"],
)
print(f"Exported to: {paths}")

# Imputation models load from HDF5 or JSON:
loaded_model = LinearImputationPRS.load("./models/breast_cancer_prs.h5")
result = loaded_model.predict("new_user.txt")

# Projection models export and load JSON only:
proj_paths = projection_model.export(output_dir="./models", model_name="breast_cancer_prs")
loaded_projection = LinearProjectionPRS.load("./models/breast_cancer_prs.json")
```

### Imputation vs projection

Both methods take the same training inputs and produce the same `PredictionResult`
shape, scored through the same allele-aware path. They differ in *what* they
reconstruct:

- **Projection (`LinearProjectionPRS`)** learns, for each genomic region, a single
  linear map from platform variants directly to that region's PRS contribution. It
  never reconstructs individual missing dosages, so the artifact is more compact and
  the scorer is simpler — which is why it is the **deployment-preferred** method.
  It exports/loads **JSON only**.
- **Imputation (`LinearImputationPRS`)** reconstructs each missing PRS variant's
  dosage and then applies that variant's beta. This exposes richer per-variant
  diagnostics (per-variant imputation R², a `variant_table`, evaluation metrics) and
  supports more export formats (`json`, `hdf5`, `arrow`, `parquet`, `csv`), which is
  useful for analysis and inspection.

For browser deployment, prefer projection. For per-variant analysis and diagnostics,
imputation gives you more to inspect. Both carry the same provenance and allele
metadata required for a deployable export.

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

Trained models export to portable formats. **Which formats are available depends on
the method**: imputation supports five, projection supports JSON only (the
browser-deployable artifact).

| Format | Use Case | Dependencies | Imputation | Projection |
|--------|----------|--------------|:---------:|:----------:|
| **JSON** | Web deployment, JavaScript/TypeScript clients | None | yes | yes |
| **HDF5** | Python/R analysis, large models | h5py | yes | — |
| **Parquet** | Data warehouses, Spark/Dask | pyarrow | yes | — |
| **Arrow** | Zero-copy IPC, streaming | pyarrow | yes | — |
| **CSV** | Variant table inspection | None | yes | — |

Loading mirrors this: `LinearImputationPRS.load` reads HDF5 and JSON; `LinearProjectionPRS.load` reads JSON only.

The JSON export is the **v2 browser-deployable artifact** (`format_version "2.0"`): it
carries per-variant allele metadata so a client-side scorer can orient raw uploads,
plus a provenance block used to validate compatibility. By default the export requires
`other_allele` on every scored variant and the provenance fields to be set (pass
`require_other_allele=False` / `require_provenance=False` for a non-deployable research
export). See **[docs/export-format.md](docs/export-format.md)** for the full field
reference and the [JSON Schema](schemas/imputation_model_v2.schema.json).

```python
# Imputation: export to all formats
paths = imputation_model.export(
    output_dir="./models",
    formats=["json", "hdf5", "parquet", "arrow", "csv"],
)

# Projection: JSON only (the default)
proj_paths = projection_model.export(output_dir="./models")
```

### Strand-ambiguous SNPs and tuning caps

Both constructors expose options for strand-ambiguous (palindromic A/T, C/G) SNPs and
for bounding hyperparameter tuning:

```python
from imputed_prs import LinearImputationPRS, LinearProjectionPRS

LinearImputationPRS(
    exclude_ambiguous=True,        # drop palindromic SNPs above the MAF threshold
    ambiguous_maf_threshold=0.4,   # MAF above which ambiguous SNPs are excluded
    max_tuning_variants=50,        # cap variants sampled for global tuning
)

LinearProjectionPRS(
    exclude_ambiguous=True,
    ambiguous_maf_threshold=0.4,
    max_tuning_regions=50,         # cap regions sampled for global tuning
)
```

By default `exclude_ambiguous=False` (training keeps palindromic variants), and the
JSON export records the deploy-time ambiguity policy in its provenance
(`ambiguous_policy`, default `"exclude_unless_platform_strand_known"`) for the browser
scorer to enforce. Each observed/predictor entry also carries an `ambiguous` flag.

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
# Get per-variant summary table (imputation models)
variant_df = model.variant_table
print(variant_df[["variant_id", "status", "imputation_r2", "n_predictors"]])
```

### Masking validation

Internal cross-validation only tells you how well the model predicts its own held-out
fold. The masking-validation harness answers the question that actually matters for
deployment: *what will a real platform upload score?* It masks a reference panel down
to a platform's variants (as if the user only ran that chip), scores the masked panel
**through the deployed library path**, and compares the estimate to the full PRS
computed from the complete panel. It works for both methods.

```python
from imputed_prs.evaluation import run_masking_validation

report = run_masking_validation(
    model,                              # a fitted imputation or projection model
    reference_genotypes="1000g_eur.vcf.gz",
    platform_name="23andme_v5",         # optional; defaults to the model's platform
    evaluation_ancestry="EUR",          # recorded; a caveat is emitted if it differs
)
# report (a MaskingValidationReport) carries correlation / R² / Spearman,
# top-decile concordance, the empirical approximation error, and interval coverage.
```

These metrics measure the *approximation error* of the masked-platform estimate
against the full computed PRS on the same population — they are **not** external or
clinical calibration. Cross-ancestry runs are structured for: pass a different-ancestry
panel and an `evaluation_ancestry` label.

## API Reference

For detailed API documentation, see [docs/API.md](docs/API.md). For the
browser-deployable JSON artifact, see [docs/export-format.md](docs/export-format.md).

### Key classes and functions

| Name | Description |
|------|-------------|
| `LinearProjectionPRS` | Projection method — deployment-preferred (JSON export only) |
| `LinearImputationPRS` | Imputation method — per-variant diagnostics, multi-format export |
| `ImputationEvaluator` | Evaluation tools for fitted models |
| `run_masking_validation()` | Mask a panel to a platform and score it through the deployed path (`imputed_prs.evaluation`) |
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
