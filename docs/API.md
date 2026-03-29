# API Reference

Complete API documentation for the imputed-prs library.

## Module Structure

```
imputed_prs/
├── __init__.py                    # Public API exports
├── core/
│   ├── linear_imputation_prs.py   # LinearImputationPRS class
│   ├── linear_projection_prs.py   # LinearProjectionPRS class
│   ├── types.py                   # Data types and dataclasses
│   ├── regions.py                 # Region decomposition for projection
│   └── exceptions.py              # Custom exceptions
├── models/
│   ├── elastic_net.py             # Per-variant ElasticNet (imputation)
│   ├── trainer.py                 # Imputation model trainer
│   ├── predictor.py               # Imputation PRS predictor
│   ├── bounding.py                # Dosage clipping and variance adjustment
│   ├── projection.py              # Per-region ElasticNet (projection)
│   ├── projection_trainer.py      # Projection model trainer
│   ├── projection_predictor.py    # Projection PRS predictor
│   ├── tuning.py                  # Hyperparameter tuning
│   └── metrics.py                 # Model quality metrics
├── io/
│   ├── prs_loader.py              # PRS file loading
│   ├── pgs_catalog.py             # PGS Catalog integration
│   ├── platform_loader.py         # Platform manifest loading
│   ├── genotype_loader.py         # Reference genotype loading
│   ├── user_genotypes.py          # DTC genotype loading
│   └── exporters/                 # Export format handlers
└── evaluation/
    ├── evaluator.py               # ImputationEvaluator class
    ├── projection_evaluator.py    # ProjectionEvaluator class
    ├── calibration.py             # CV-based calibration
    ├── metrics.py                 # Metric computation
    └── plotting.py                # Diagnostic plots
```

## Quick Import Examples

```python
# Main API (most common)
from imputed_prs import (
    LinearImputationPRS,
    LinearProjectionPRS,
    ImputationEvaluator,
    list_available_platforms,
    get_platform_info,
    search_pgs_catalog,
    fetch_pgs_catalog_score,
)

# Projection evaluator
from imputed_prs.evaluation import ProjectionEvaluator

# Data types
from imputed_prs.core.types import (
    PredictionResult,
    PlatformInfo,
    EvaluationMetrics,
    CalibrationParams,
    ProjectionRegionModel,
    ProjectionTrainingResult,
)

# Region decomposition types
from imputed_prs.core.regions import (
    GenomicRegion,
    RegionDecompositionResult,
    merge_variant_windows,
)

# Exceptions
from imputed_prs.core.exceptions import (
    ImputedPRSError,
    DataLoadError,
    ValidationError,
)

# Plotting (requires matplotlib)
from imputed_prs.evaluation.plotting import (
    plot_calibration,
    plot_imputation_quality,
    plot_variance_contribution,
)
```

---

## Method Comparison

The library provides two approaches for computing PRS when the genotyping platform is missing some PRS variants. Both methods share the same I/O loaders, calibration procedure, and `PredictionResult` output type. See the [statistical theory](statistical-theory/README.md) for the mathematical foundations.

| Feature | `LinearImputationPRS` | `LinearProjectionPRS` |
|---------|----------------------|----------------------|
| **Unit of prediction** | Per-variant dosage | Per-region PRS contribution |
| **`tuning_scope` parameter** | Yes (`global`/`per_variant`/`none`) | No (fixed parameters) |
| **`evaluation_genotypes` in `fit()`** | Yes | No |
| **Dosage clipping** | Yes (truncated normal variance) | No (target is PRS, not dosage) |
| **Missing predictor handling** | Fallback to intercept-only | Mean-substitution ($2 \cdot AF$) |
| **Uncertainty source** | $\sum \beta_j^2 \sigma^2_{adj,j}$ | $\sum_R \text{cv\_mse}_R$ |
| **Export/Load** | Yes | Not yet implemented |
| **Per-variant diagnostics** | Yes (`imputation_r2` per variant) | No (`cv_r2` per region) |
| **Key model property** | `imputed_models` | `region_models` |

---

## Main API Class: LinearImputationPRS

The primary class for training imputation models and computing PRS predictions.

### Constructor

```python
LinearImputationPRS(
    window_size: int = 1_000_000,
    tuning_scope: Literal["global", "per_variant", "none"] = "global",
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    max_predictors: Optional[int] = None,
    verbose: int = 1,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `1_000_000` | Size of genomic window (bp) for selecting predictor variants. Larger windows include more potential predictors but increase computation. |
| `tuning_scope` | `str` | `"global"` | Hyperparameter tuning strategy: `"global"` (tune once), `"per_variant"` (tune each variant), or `"none"` (use provided values). |
| `l1_ratio` | `float` | `0.5` | ElasticNet L1/L2 mixing parameter. 0=Ridge, 1=Lasso. Only used when `tuning_scope="none"`. |
| `alpha` | `float` | `0.01` | ElasticNet regularization strength. Only used when `tuning_scope="none"`. |
| `cv_folds` | `int` | `5` | Number of cross-validation folds for training and calibration. |
| `n_jobs` | `int` | `1` | Number of parallel jobs for training. Use `-1` for all CPUs. |
| `random_state` | `int` | `None` | Random seed for reproducibility. |
| `max_predictors` | `int` | `None` | Maximum number of predictor variants per model. If `None`, uses all variants in window. |
| `verbose` | `int` | `1` | Verbosity level. 0=silent, 1=progress bar, 2=debug output. |

**Example:**

```python
model = LinearImputationPRS(
    window_size=500_000,      # 500 kb window
    tuning_scope="global",    # Tune hyperparameters globally
    cv_folds=5,
    n_jobs=-1,                # Use all CPUs
    verbose=1,
)
```

---

### fit()

Train imputation models on reference genotype data.

```python
def fit(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    genome_build: Optional[str] = None,
    prs_id: Optional[str] = None,
    model_name: Optional[str] = None,
    evaluation_genotypes: Optional[Union[str, Path]] = None,
) -> "LinearImputationPRS"
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_genotypes` | `str` or `Path` | Path to reference genotype file (VCF or PLINK format). |
| `prs_definition` | `str`, `Path`, or `DataFrame` | PRS definition as PGS Catalog ID (e.g., `"PGS000004"`), file path, or DataFrame with variant weights. |
| `platform_name` | `str` | Name of pre-built platform (e.g., `"23andme_v5"`). Mutually exclusive with `platform_manifest` and `platform_variants`. |
| `platform_manifest` | `str` or `Path` | Path to custom platform manifest file. |
| `platform_variants` | `List[str]` | List of platform variant IDs. |
| `genome_build` | `str` | Genome build (`"GRCh37"` or `"GRCh38"`). Auto-detected if `None`. |
| `prs_id` | `str` | PRS identifier for metadata. |
| `model_name` | `str` | Human-readable model name for metadata. |
| `evaluation_genotypes` | `str` or `Path` | Optional holdout genotypes for external evaluation. |

**Returns:** `self` (for method chaining)

**Raises:**
- `ValidationError`: If inputs are invalid or incompatible.
- `DataLoadError`: If files cannot be loaded.

**Example:**

```python
model = LinearImputationPRS()
model.fit(
    reference_genotypes="1000g_eur.vcf.gz",
    prs_definition="PGS000004",
    platform_name="23andme_v5",
    model_name="breast_cancer_23andme_v5",
)
```

---

### predict()

Compute PRS for user genotypes.

```python
def predict(
    self,
    user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
    apply_calibration: bool = True,
) -> PredictionResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_genotypes` | `str`, `Path`, `DataFrame`, or `dict` | User genotype data as file path (DTC format auto-detected), DataFrame with `variant_id` and `genotype` columns, or dict mapping variant_id to dosage values. |
| `apply_calibration` | `bool` | Whether to apply calibration scaling. Default: `True`. |

**Returns:** `PredictionResult` with PRS value, uncertainty estimates, and diagnostics.

**Raises:**
- `ModelNotFittedError`: If `fit()` has not been called.
- `DataLoadError`: If user genotype file cannot be loaded.

**Example:**

```python
# From file
result = model.predict("user_23andme.txt")

# From dict
dosages = {"rs123": 1.0, "rs456": 2.0, "rs789": 0.0}
result = model.predict(dosages)

print(f"PRS: {result.prs:.3f} (95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")
```

---

### export()

Export trained model to portable formats.

```python
def export(
    self,
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
    formats: Optional[List[str]] = None,
    include_variance_scaling: bool = True,
) -> Dict[str, Path]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` or `Path` | | Directory for output files. |
| `model_name` | `str` | `None` | Base name for output files. Uses internal name if `None`. |
| `formats` | `List[str]` | `["json", "hdf5"]` | List of formats: `"json"`, `"arrow"`, `"parquet"`, `"hdf5"`, `"csv"`. |
| `include_variance_scaling` | `bool` | `True` | Whether to include variance/SE components. |

**Returns:** Dict mapping format name to output file path.

**Raises:**
- `ModelNotFittedError`: If `fit()` has not been called.
- `ValueError`: If an unsupported format is requested.

**Example:**

```python
paths = model.export(
    output_dir="./models",
    model_name="my_prs_model",
    formats=["json", "hdf5"],
)
print(paths)  # {'json': Path('models/my_prs_model.json'), 'hdf5': Path('models/my_prs_model.h5')}
```

---

### load() (classmethod)

Load a trained model from file.

```python
@classmethod
def load(cls, path: Union[str, Path]) -> "LinearImputationPRS"
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path to saved model file (HDF5 or JSON format). |

**Returns:** Loaded `LinearImputationPRS` instance ready for prediction.

**Raises:**
- `DataLoadError`: If file cannot be loaded or format is unsupported.

**Example:**

```python
model = LinearImputationPRS.load("models/my_model.h5")
result = model.predict("user_genotypes.txt")
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_fitted` | `bool` | Whether the model has been fitted. |
| `variant_table` | `pd.DataFrame` | Per-variant summary with status and quality metrics. |
| `summary` | `Dict[str, Any]` | Model summary with counts and quality statistics. |
| `evaluation_metrics` | `EvaluationMetrics` or `None` | Evaluation metrics from training (if available). |
| `calibration_params` | `CalibrationParams` or `None` | Calibration parameters from CV training. |
| `observed_variants` | `List[VariantInfo]` | List of observed (directly measured) variants. |
| `imputed_models` | `List[ImputedVariantModel]` | List of imputed variant models. |

**Example:**

```python
# Check if fitted
if model.is_fitted:
    # Get summary
    print(model.summary)
    # {'n_total_variants': 1000, 'n_observed': 200, 'n_imputed': 800, ...}

    # Get variant table
    df = model.variant_table
    print(df[df['imputation_r2'] > 0.8])  # High-quality imputed variants
```

---

## Main API Class: LinearProjectionPRS

The projection-based class for training region-level models and computing PRS predictions. Instead of imputing individual missing variant dosages, it directly learns platform-variant weights to approximate each region's PRS contribution.

### Constructor

```python
LinearProjectionPRS(
    window_size: int = 1_000_000,
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    max_predictors: Optional[int] = None,
    verbose: int = 1,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `1_000_000` | Size of genomic window (bp) for defining regions and selecting predictor variants. Overlapping windows merge into regions. |
| `l1_ratio` | `float` | `0.5` | ElasticNet L1/L2 mixing parameter. 0=Ridge, 1=Lasso. Applied directly to all regions. |
| `alpha` | `float` | `0.01` | ElasticNet regularization strength. Applied directly to all regions. |
| `cv_folds` | `int` | `5` | Number of cross-validation folds for training and calibration. |
| `n_jobs` | `int` | `1` | Number of parallel jobs for training. Use `-1` for all CPUs. |
| `random_state` | `int` | `None` | Random seed for reproducibility. |
| `max_predictors` | `int` | `None` | Maximum number of predictor variants per region. If `None`, uses all variants in region. |
| `verbose` | `int` | `1` | Verbosity level. 0=silent, 1=progress bar, 2=debug output. |

**Note:** Unlike `LinearImputationPRS`, there is no `tuning_scope` parameter. The provided `l1_ratio` and `alpha` are used directly for all regions.

**Example:**

```python
model = LinearProjectionPRS(
    window_size=1_000_000,    # 1 Mb window
    alpha=0.01,
    l1_ratio=0.5,
    n_jobs=-1,                # Use all CPUs
    verbose=1,
)
```

---

### fit()

Train projection models on reference genotype data.

```python
def fit(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    genome_build: Optional[str] = None,
    prs_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> "LinearProjectionPRS"
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_genotypes` | `str` or `Path` | Path to reference genotype file (VCF or PLINK format). |
| `prs_definition` | `str`, `Path`, or `DataFrame` | PRS definition as PGS Catalog ID (e.g., `"PGS000004"`), file path, or DataFrame with variant weights. |
| `platform_name` | `str` | Name of pre-built platform (e.g., `"23andme_v5"`). Mutually exclusive with `platform_manifest` and `platform_variants`. |
| `platform_manifest` | `str` or `Path` | Path to custom platform manifest file. |
| `platform_variants` | `List[str]` | List of platform variant IDs. |
| `genome_build` | `str` | Genome build (`"GRCh37"` or `"GRCh38"`). Auto-detected if `None`. |
| `prs_id` | `str` | PRS identifier for metadata. |
| `model_name` | `str` | Human-readable model name for metadata. |

**Note:** Unlike `LinearImputationPRS.fit()`, there is no `evaluation_genotypes` parameter. Use `ProjectionEvaluator` for external evaluation.

**Returns:** `self` (for method chaining)

**Raises:**
- `ValidationError`: If inputs are invalid or incompatible.
- `DataLoadError`: If files cannot be loaded.

**Example:**

```python
model = LinearProjectionPRS()
model.fit(
    reference_genotypes="1000g_eur.vcf.gz",
    prs_definition="PGS000004",
    platform_name="23andme_v5",
    model_name="breast_cancer_23andme_v5",
)
```

---

### predict()

Compute PRS for user genotypes.

```python
def predict(
    self,
    user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
    apply_calibration: bool = True,
) -> PredictionResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_genotypes` | `str`, `Path`, `DataFrame`, or `dict` | User genotype data as file path (DTC format auto-detected), DataFrame with `variant_id` and `genotype` columns, or dict mapping variant_id to dosage values. |
| `apply_calibration` | `bool` | Whether to apply calibration scaling. Default: `True`. |

**Returns:** `PredictionResult` with PRS value, uncertainty estimates, and diagnostics. The `prs_imputed_component` field contains the sum of regional projection predictions. The `n_truncated` field is always 0 (no dosage clipping in projection).

**Raises:**
- `ModelNotFittedError`: If `fit()` has not been called.
- `DataLoadError`: If user genotype file cannot be loaded.

**Example:**

```python
# From file
result = model.predict("user_23andme.txt")

# From dict
dosages = {"rs123": 1.0, "rs456": 2.0, "rs789": 0.0}
result = model.predict(dosages)

print(f"PRS: {result.prs:.3f} (95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_fitted` | `bool` | Whether the model has been fitted. |
| `observed_variants` | `List[VariantInfo]` | List of observed (directly measured) variants. |
| `region_models` | `List[ProjectionRegionModel]` | List of trained projection region models. |
| `calibration_params` | `CalibrationParams` or `None` | Calibration parameters from CV training. |
| `summary` | `Dict[str, Any]` | Model summary with region counts and quality statistics. |
| `variant_table` | `pd.DataFrame` | Per-region summary table with columns: `region_id`, `chromosome`, `start`, `end`, `n_prs_variants`, `n_predictors`, `cv_r2`, `cv_mse`, `is_intercept_only`. |

**Note:** `export()` and `load()` are not yet implemented for the projection method.

**Example:**

```python
# Check if fitted
if model.is_fitted:
    # Get summary
    print(model.summary)
    # {'n_observed_variants': 200, 'n_missing_variants': 800, 'n_regions': 50, ...}

    # Get region table
    df = model.variant_table
    print(df[df['cv_r2'] > 0.8])  # High-quality regions
```

---

## Evaluation: ImputationEvaluator

Evaluator class for assessing fitted imputation models on held-out data.

### Constructor

```python
ImputationEvaluator(
    model: LinearImputationPRS,
    verbose: int = 1,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `LinearImputationPRS` | A fitted model to evaluate. |
| `verbose` | `int` | Verbosity level (0=silent, 1=progress, 2=debug). |

**Raises:**
- `ModelNotFittedError`: If the model has not been fitted.

---

### evaluate()

Evaluate the model on held-out genotype data.

```python
def evaluate(
    self,
    evaluation_genotypes: Union[str, Path, GenotypeData],
    percentile_thresholds: List[int] = None,
) -> EvaluationMetrics
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluation_genotypes` | `str`, `Path`, or `GenotypeData` | | Path to genotype file or pre-loaded data. |
| `percentile_thresholds` | `List[int]` | `[1, 5, 10]` | Percentile thresholds for concordance. |

**Returns:** `EvaluationMetrics` comparing imputed vs true PRS.

**Example:**

```python
evaluator = ImputationEvaluator(model)
metrics = evaluator.evaluate("held_out_data.vcf.gz")
print(f"R²: {metrics.r2:.3f}, Correlation: {metrics.correlation:.3f}")
```

---

### cross_validate()

Perform k-fold cross-validation.

```python
def cross_validate(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> CrossValidationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_genotypes` | `str` or `Path` | | Path to reference genotype file. |
| `prs_definition` | `str`, `Path`, or `DataFrame` | | PRS definition. |
| `platform_name` | `str` | `None` | Name of pre-built platform. |
| `platform_manifest` | `str` or `Path` | `None` | Path to platform manifest. |
| `platform_variants` | `List[str]` | `None` | List of platform variant IDs. |
| `n_folds` | `int` | `5` | Number of CV folds (must be >= 2). |
| `random_state` | `int` | `None` | Random seed for reproducibility. |

**Returns:** `CrossValidationResult` with fold metrics and aggregated statistics.

**Example:**

```python
cv_result = evaluator.cross_validate(
    reference_genotypes="reference.vcf.gz",
    prs_definition="PGS000004",
    platform_name="23andme_v5",
    n_folds=5,
)
print(f"Mean R²: {cv_result.mean_r2:.3f} ± {cv_result.std_r2:.3f}")
```

---

### sensitivity_analysis()

Analyze model sensitivity to hyperparameters.

```python
def sensitivity_analysis(
    self,
    reference_genotypes: Union[str, Path],
    prs_definition: Union[str, Path, pd.DataFrame],
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[List[str]] = None,
    parameter_grid: Optional[Dict[str, List[Any]]] = None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> SensitivityResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameter_grid` | `Dict[str, List]` | See below | Dictionary mapping parameter names to lists of values to try. |

Default parameter grid:
```python
{
    "window_size": [500_000, 1_000_000, 2_000_000],
    "l1_ratio": [0.1, 0.5, 0.9],
    "alpha": [0.001, 0.01, 0.1],
}
```

**Returns:** `SensitivityResult` with results for each parameter combination.

---

## Evaluation: ProjectionEvaluator

Evaluator class for assessing fitted projection models on held-out data.

### Constructor

```python
ProjectionEvaluator(
    model: LinearProjectionPRS,
    verbose: int = 1,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `LinearProjectionPRS` | A fitted projection model to evaluate. |
| `verbose` | `int` | Verbosity level (0=silent, 1=progress, 2=debug). |

**Raises:**
- `ModelNotFittedError`: If the model has not been fitted.

---

### evaluate()

Evaluate the model on held-out genotype data.

```python
def evaluate(
    self,
    evaluation_genotypes: Union[str, Path, GenotypeData],
) -> EvaluationMetrics
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `evaluation_genotypes` | `str`, `Path`, or `GenotypeData` | Path to genotype file or pre-loaded data. |

**Returns:** `EvaluationMetrics` comparing projected vs true PRS.

**Example:**

```python
evaluator = ProjectionEvaluator(model)
metrics = evaluator.evaluate("held_out_data.vcf.gz")
print(f"R²: {metrics.r2:.3f}, Correlation: {metrics.correlation:.3f}")
```

**Note:** Unlike `ImputationEvaluator`, `ProjectionEvaluator` does not yet provide `cross_validate()` or `sensitivity_analysis()` methods.

---

## Convenience Functions

### Platform Functions

#### list_available_platforms()

List all available pre-built platforms.

```python
def list_available_platforms() -> List[str]
```

**Returns:** List of supported platform names.

**Example:**

```python
from imputed_prs import list_available_platforms

platforms = list_available_platforms()
# ['23andme_v3', '23andme_v4', '23andme_v5', 'ancestrydna_v1', 'ancestrydna_v2']
```

---

#### get_platform_info()

Get metadata for a platform without loading variants.

```python
def get_platform_info(platform_name: str) -> PlatformInfo
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `platform_name` | `str` | Name of the platform (case-insensitive). |

**Returns:** `PlatformInfo` with platform metadata.

**Raises:**
- `ValidationError`: If platform name is not recognized.

**Example:**

```python
from imputed_prs import get_platform_info

info = get_platform_info("23andme_v5")
print(f"{info.display_name}: {info.n_variants:,} variants")
print(f"Build: {info.genome_build}, Chip: {info.chip_technology}")
```

---

### PGS Catalog Functions

#### search_pgs_catalog()

Search the PGS Catalog for scores by trait.

```python
def search_pgs_catalog(
    trait_query: str,
    limit: int = 10,
) -> List[PGSSearchResult]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trait_query` | `str` | | Search query for traits (e.g., `"breast cancer"`). |
| `limit` | `int` | `10` | Maximum number of results to return. |

**Returns:** List of `PGSSearchResult` objects.

**Raises:**
- `ValidationError`: If query is empty.
- `DataLoadError`: If search fails.

**Example:**

```python
from imputed_prs import search_pgs_catalog

results = search_pgs_catalog("type 2 diabetes", limit=5)
for r in results:
    print(f"{r.pgs_id}: {r.name} ({r.variants_number} variants)")
```

---

#### fetch_pgs_catalog_score()

Download a PGS Catalog scoring file and normalize it.

```python
def fetch_pgs_catalog_score(
    pgs_id: str,
    genome_build: str = "GRCh37",
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    filter_failed_mappings: bool = True,
) -> Tuple[pd.DataFrame, PGSCatalogMetadata]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pgs_id` | `str` | | PGS Catalog score ID (e.g., `"PGS000004"`). |
| `genome_build` | `str` | `"GRCh37"` | Genome build (`"GRCh37"`, `"GRCh38"`, `"hg19"`, or `"hg38"`). |
| `cache_dir` | `Path` | `None` | Custom cache directory. Uses `~/.cache/imputed_prs/pgs_catalog` by default. |
| `use_cache` | `bool` | `True` | Whether to use cached files if available. |
| `filter_failed_mappings` | `bool` | `True` | Filter out variants with failed harmonization. |

**Returns:** Tuple of (normalized DataFrame, `PGSCatalogMetadata`).

**Example:**

```python
from imputed_prs import fetch_pgs_catalog_score

prs_df, metadata = fetch_pgs_catalog_score("PGS000004", genome_build="GRCh37")
print(f"Downloaded {len(prs_df)} variants for {metadata.trait_reported}")
print(f"DOI: {metadata.publication_doi}")
```

---

#### clear_pgs_catalog_cache()

Clear all cached PGS Catalog files.

```python
def clear_pgs_catalog_cache(cache_dir: Optional[Path] = None) -> int
```

**Returns:** Number of files removed.

---

## Data Loaders

### load_prs_from_file()

Load a PRS definition from a file.

```python
def load_prs_from_file(path: Union[str, Path]) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path to PRS file. Supports CSV, TSV, and gzipped formats. |

**Returns:** Normalized DataFrame with canonical column names (`variant_id`, `chromosome`, `position`, `effect_allele`, `other_allele`, `beta`).

**Raises:**
- `DataLoadError`: If file cannot be read.
- `ValidationError`: If required columns are missing.

---

### load_prs_from_dataframe()

Validate and normalize a PRS definition DataFrame.

```python
def load_prs_from_dataframe(df: pd.DataFrame) -> pd.DataFrame
```

Accepts various column name aliases:
- `variant_id`: rsid, snp, snp_id, variant, id, marker
- `chromosome`: chr, chrom, #chrom, chr_name
- `position`: pos, bp, chr_position, base_pair_location
- `effect_allele`: allele1, a1, alt, effect, ea, risk_allele
- `other_allele`: allele2, a2, ref, non_effect_allele, nea
- `beta`: effect_weight, weight, effect_size, log_or

---

### load_genotypes()

Load genotypes with automatic format detection.

```python
def load_genotypes(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
) -> GenotypeData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path to genotype file (VCF or PLINK prefix). |
| `variant_ids` | `Set[str]` | Optional set of variant IDs to filter to. Supports rsID and chr:pos formats. |
| `samples` | `List[str]` | Optional list of sample IDs to include. |

**Returns:** `GenotypeData` containing the loaded genotypes.

---

### load_genotypes_vcf()

Load genotypes from a VCF file.

```python
def load_genotypes_vcf(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
    dosage_field: str = "auto",
) -> GenotypeData
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dosage_field` | `str` | `"auto"` | How to extract dosage: `"auto"`, `"GT"`, `"DS"`, or `"GP"`. |

---

### load_genotypes_plink()

Load genotypes from PLINK binary files.

```python
def load_genotypes_plink(
    path: Union[str, Path],
    variant_ids: Optional[Set[str]] = None,
    samples: Optional[List[str]] = None,
) -> GenotypeData
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path prefix for PLINK files (without .bed/.bim/.fam extension). |

---

### load_user_genotypes()

Load user genotypes and convert to dosage values.

```python
def load_user_genotypes(
    input_data: Union[str, Path, SNPs, pd.DataFrame],
    expected_variants: Optional[Set[str]] = None,
) -> Dict[str, Optional[float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_data` | Various | File path (DTC format auto-detected), SNPs object, or DataFrame. |
| `expected_variants` | `Set[str]` | Optional set of variant IDs to extract. |

**Returns:** Dictionary mapping variant_id to dosage value (0.0, 1.0, 2.0) or `None` for missing.

Supports DTC formats: 23andMe, AncestryDNA, and others via the `snps` package.

---

### genotype_to_dosage()

Convert genotype string to dosage value.

```python
def genotype_to_dosage(genotype: str) -> Optional[float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `genotype` | `str` | Genotype string (e.g., "AA", "AG", "GG", "--"). |

**Returns:** Dosage value (0.0, 1.0, 2.0) or `None` if missing/invalid.

---

### detect_genome_build()

Detect genome build from user genotype data.

```python
def detect_genome_build(input_data: Union[str, Path, SNPs]) -> Optional[int]
```

**Returns:** Build number (37 or 38) or `None` if not detected.

---

## Export Functions

### export_to_json()

Export trained model to JSON format.

```python
def export_to_json(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    calibration_params: Optional[CalibrationParams] = None,
    evaluation_metrics: Optional[EvaluationMetrics] = None,
    platform_name: Optional[str] = None,
    prs_id: Optional[str] = None,
    genome_build: Optional[str] = None,
    model_name: Optional[str] = None,
    include_variance_scaling: bool = True,
    training_summary: Optional[Dict[str, Any]] = None,
) -> Path
```

---

### export_to_hdf5()

Export trained model to HDF5 format.

```python
def export_to_hdf5(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    # ... same parameters as export_to_json
) -> Path
```

---

### export_to_arrow()

Export trained model to Arrow IPC format.

```python
def export_to_arrow(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    # ... same parameters as export_to_json
) -> Path
```

---

### export_to_parquet()

Export trained model to Parquet format.

```python
def export_to_parquet(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    # ... same parameters as export_to_json
) -> Dict[str, Path]
```

Creates a directory with multiple Parquet files for different components.

---

### export_variant_table()

Export variant summary table to CSV.

```python
def export_variant_table(
    output_path: Union[str, Path],
    observed_variants: List[VariantInfo],
    imputed_models: List[ImputedVariantModel],
    include_variance_scaling: bool = True,
) -> Path
```

---

## Data Types Reference

### PredictionResult

Output from PRS prediction. Used by both `LinearImputationPRS` and `LinearProjectionPRS`.

```python
@dataclass
class PredictionResult:
    prs: float                      # Raw PRS value
    se: float                       # Standard error
    ci_lower: float                 # Lower 95% CI bound
    ci_upper: float                 # Upper 95% CI bound
    prs_observed_component: float   # Contribution from observed variants
    prs_imputed_component: float    # Contribution from missing variants (see note)
    n_variants_used: int            # Total variants contributing
    n_variants_imputed: int         # Count of missing variants (see note)
    n_variants_intercept_only: int  # Count using intercept-only models
    n_user_variants_missing: int    # User variants not available
    n_truncated: int                # Dosages clipped (always 0 for projection)
    prs_scaled: Optional[float]     # Scaled PRS (if calibrated)
    se_scaled: Optional[float]      # Scaled SE (if calibrated)
    ci_lower_scaled: Optional[float]
    ci_upper_scaled: Optional[float]
```

**Note on shared fields:** The `prs_imputed_component` field represents the missing-variant contribution for both methods: for imputation, it is the sum of imputed dosages times effect sizes ($\sum \hat{x}_j \beta_j$); for projection, it is the sum of regional projection predictions ($\sum_R \hat{S}_R$). Similarly, `n_variants_imputed` counts the missing PRS variants covered by either per-variant imputation models or region models. The `n_truncated` field is always 0 for projection (no dosage clipping).

---

### PlatformInfo

Metadata about a genotyping platform.

```python
@dataclass
class PlatformInfo:
    name: str              # Internal identifier (e.g., "23andme_v5")
    display_name: str      # Human-readable name
    description: str       # Brief description
    genome_build: str      # "GRCh37" or "GRCh38"
    n_variants: int        # Number of variants
    chip_technology: str   # Underlying chip (e.g., "Illumina GSA")
    company: str           # Company name
    version: str           # Version identifier
    date_introduced: Optional[str]
    source_url: Optional[str]
```

---

### VariantInfo

Represents a single variant in a PRS definition.

```python
@dataclass
class VariantInfo:
    variant_id: str       # rsID or unique identifier
    chromosome: str       # "1"-"22", "X", "Y", "MT"
    position: int         # Genomic position
    effect_allele: str    # Allele associated with the effect
    other_allele: Optional[str]
    beta: float           # Effect size
```

---

### ImputedVariantModel

Stores the trained imputation model for a missing variant.

```python
@dataclass
class ImputedVariantModel:
    variant_id: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: Optional[str]
    beta: float                   # PRS effect size
    allele_frequency: float       # Population AF
    imputation_r2: float          # CV R² of imputation
    residual_variance: float      # Residual variance
    intercept: float              # Model intercept
    predictor_variant_ids: List[str]  # IDs of predictor variants
    coefficients: np.ndarray      # Regression coefficients
    is_intercept_only: bool       # True if no predictors
```

---

### CalibrationParams

Internal CV calibration parameters.

```python
@dataclass
class CalibrationParams:
    scaling_factor: float         # Slope from calibration regression
    scaling_factor_se: float      # SE of scaling factor
    calibration_intercept: float
    calibration_r2: float         # R² of calibration fit
    sd_cv_predicted: float        # SD of CV-predicted PRS
    sd_true: float                # SD of true PRS
    sd_scaled: float              # SD of scaled predictions
    attenuation_factor: float     # Ratio sd_cv/sd_true
    n_calibration: int            # Sample size for calibration
```

---

### EvaluationMetrics

Metrics from model evaluation.

```python
@dataclass
class EvaluationMetrics:
    correlation: float       # Pearson correlation
    r2: float               # R-squared
    mae: float              # Mean absolute error
    rmse: float             # Root mean squared error
    spearman_rho: float     # Spearman rank correlation
    calibration_slope: float
    calibration_intercept: float
```

---

### TrainingResult

Result from training imputation models.

```python
@dataclass
class TrainingResult:
    models: Dict[str, ImputedVariantModel]
    cv_predictions: Dict[str, np.ndarray]
    n_variants_trained: int
    n_variants_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]  # Includes mean_r2, median_r2, etc.
```

---

### GenotypeData

Container for loaded reference genotype data.

```python
@dataclass
class GenotypeData:
    dosage_matrix: np.ndarray     # (n_samples x n_variants), values 0-2
    variant_info: pd.DataFrame    # variant_id, chromosome, position, ref_allele, alt_allele
    sample_ids: List[str]
    genome_build: Optional[str]
    source_file: Optional[str]

    @property
    def n_samples(self) -> int: ...

    @property
    def n_variants(self) -> int: ...
```

---

### CrossValidationResult

Result from cross-validation evaluation.

```python
@dataclass
class CrossValidationResult:
    fold_metrics: List[EvaluationMetrics]
    mean_correlation: float
    std_correlation: float
    mean_r2: float
    std_r2: float
    mean_mae: float
    mean_rmse: float
    mean_spearman: float
    percentile_concordance: Dict[str, float]
    n_folds: int
    n_samples_per_fold: List[int]
```

---

### SensitivityResult

Result from sensitivity analysis.

```python
@dataclass
class SensitivityResult:
    parameter_results: List[Dict[str, Any]]
    best_params: Dict[str, float]
    best_metrics: EvaluationMetrics
    quality_summaries: List[Dict[str, Any]]
```

---

### ProjectionRegionModel

Stores the trained projection model for a single genomic region.

```python
@dataclass
class ProjectionRegionModel:
    region_id: str                        # e.g., "chr1:1000000-3000000"
    chromosome: str
    start: int                            # Region start position
    end: int                              # Region end position
    prs_variant_ids: List[str]            # Missing PRS variants in this region
    betas: np.ndarray                     # Effect sizes, shape: (n_prs_variants,)
    predictor_variant_ids: List[str]      # Platform variants used as predictors
    coefficients: np.ndarray              # Learned weights a_R, shape: (n_predictors,)
    intercept: float                      # Model intercept (mean S_R for intercept-only)
    cv_mse: float                         # Cross-validated MSE
    cv_r2: float                          # Cross-validated R-squared
    is_intercept_only: bool               # True if no predictors or all zero
    mean_prs_contribution: float          # Mean of S_R across training samples
    predictor_allele_frequencies: np.ndarray  # AFs for mean-substitution at inference
```

---

### ProjectionTrainingResult

Result from training projection models for all regions.

```python
@dataclass
class ProjectionTrainingResult:
    region_models: Dict[str, ProjectionRegionModel]  # region_id -> model
    cv_predictions: Dict[str, np.ndarray]  # region_id -> out-of-fold predictions
    n_regions_trained: int
    n_regions_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]       # mean_r2, median_r2, n_high_quality, etc.
```

---

### GenomicRegion

A contiguous genomic interval containing one or more missing PRS variants, formed by merging overlapping per-variant windows.

```python
@dataclass
class GenomicRegion:
    chromosome: str            # Normalized (e.g., "1", "X")
    start: int                 # Start position (inclusive)
    end: int                   # End position (inclusive)
    prs_variant_ids: List[str]     # Missing PRS variant IDs in this region
    prs_variant_indices: List[int] # Indices into the missing PRS DataFrame
```

---

### RegionDecompositionResult

Result of decomposing PRS variants into non-overlapping regions.

```python
@dataclass
class RegionDecompositionResult:
    regions: List[GenomicRegion]
    n_regions: int                 # Total number of merged regions
    n_variants_in_regions: int     # Total PRS variants covered
    variants_per_region: List[int] # Count of PRS variants per region
    max_region_span_bp: int        # Largest region span in base pairs
```

---

## Exceptions

All exceptions inherit from `ImputedPRSError` for easy catching.

```python
from imputed_prs.core.exceptions import (
    ImputedPRSError,        # Base exception
    DataLoadError,          # Failed to load files
    ValidationError,        # Invalid input data
    IncompatibleBuildError, # Genome build mismatch
    MissingVariantsError,   # Required variants not found
    ModelNotFittedError,    # predict() called before fit()
)
```

**Example:**

```python
from imputed_prs import LinearImputationPRS
from imputed_prs.core.exceptions import ImputedPRSError, ModelNotFittedError

try:
    model = LinearImputationPRS()
    model.predict("user.txt")  # Error: not fitted
except ModelNotFittedError:
    print("Need to fit the model first!")
except ImputedPRSError as e:
    print(f"Library error: {e}")
```

---

## Plotting Functions

Requires `matplotlib`. Install with `pip install imputed-prs[plotting]`.

### plot_calibration()

Create calibration scatter plot comparing imputed vs true PRS.

```python
def plot_calibration(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    show_identity: bool = True,
    show_regression: bool = True,
    alpha: float = 0.5,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
```

**Example:**

```python
from imputed_prs.evaluation.plotting import plot_calibration

fig, ax = plot_calibration(imputed_scores, true_scores)
fig.savefig("calibration_plot.png", dpi=150)
```

---

### plot_imputation_quality()

Create histogram of imputation R² values by quality tier.

```python
def plot_imputation_quality(
    models: List[ImputedVariantModel],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    bins: int = 20,
    show_tiers: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
```

Quality tiers:
- **Excellent** (green): R² > 0.8
- **Good** (blue): 0.6 < R² ≤ 0.8
- **Moderate** (orange): 0.4 < R² ≤ 0.6
- **Poor** (red): R² ≤ 0.4

**Example:**

```python
from imputed_prs.evaluation.plotting import plot_imputation_quality

fig, ax = plot_imputation_quality(model.imputed_models)
fig.savefig("quality_distribution.png")
```

---

### plot_variance_contribution()

Create bar chart of top variance-contributing variants.

```python
def plot_variance_contribution(
    models: List[ImputedVariantModel],
    top_n: int = 20,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    color_by_quality: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
```

Variance contribution is calculated as `beta² × residual_variance`.

---

### plot_truncation_diagnostics()

Visualize dosage truncation patterns across predictions.

```python
def plot_truncation_diagnostics(
    predictions: Union[List[PredictionResult], pd.DataFrame],
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
```

---

## Metrics Functions

### compute_prs_metrics()

Compute comprehensive metrics comparing imputed vs true PRS.

```python
def compute_prs_metrics(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
) -> EvaluationMetrics
```

**Returns:** `EvaluationMetrics` with correlation, R², MAE, RMSE, Spearman rho, and calibration slope/intercept.

---

### compute_percentile_concordance()

Compute top/bottom percentile concordance.

```python
def compute_percentile_concordance(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
    percentiles: List[int] = [1, 5, 10],
) -> Dict[str, float]
```

**Returns:** Dict with keys like `'top_1_concordance'`, `'bottom_5_concordance'`, plus `'quintile_kappa'` for Cohen's kappa on quintile assignments.
