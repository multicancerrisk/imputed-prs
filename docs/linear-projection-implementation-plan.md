# Linear Projection PRS: Implementation Plan

This document specifies the design and implementation steps for adding a **linear projection** PRS method to the `imputed-prs` library. A coding agent should implement the phases sequentially, verifying each phase before proceeding to the next.

## Table of Contents

1. [Overview and Mathematical Background](#1-overview-and-mathematical-background)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Data Types and Region Merging](#3-phase-1-data-types-and-region-merging)
4. [Phase 2: Per-Region Model Fitting](#4-phase-2-per-region-model-fitting)
5. [Phase 3: Projection Predictor](#5-phase-3-projection-predictor)
6. [Phase 4: Main LinearProjectionPRS Class](#6-phase-4-main-linearprojectionprs-class)
7. [Phase 5: Evaluation](#7-phase-5-evaluation)
8. [Phase 6: Equivalence Tests and Comparison Script](#8-phase-6-equivalence-tests-and-comparison-script)
9. [Summary Table and Dependency Order](#9-summary-table-and-dependency-order)
10. [Key Design Decisions and Rationale](#10-key-design-decisions-and-rationale)

---

## 1. Overview and Mathematical Background

### The Problem

A PRS sums effect-weighted dosages across many variants: $S = \sum_j x_j \beta_j$. When a genotyping platform is missing some PRS variants, we need a way to approximate $S$ from the available (platform) genotypes $z$.

### Linear Imputation (Current Approach)

For each missing variant $j$, train a separate model to predict its dosage from nearby platform variants:

$$\hat{x}_j = z_{L_j}^T w_j + \gamma_j$$

Then compute $\hat{S} = \sum_{j \in O} z_j \beta_j + \sum_{j \in M} \hat{x}_j \beta_j$, where $O$ is the set of observed PRS variants and $M$ is the set of missing PRS variants.

**Implementation**: `imputed_prs/models/elastic_net.py:fit_single_variant_model()` trains per-variant models; `imputed_prs/models/predictor.py:PRSPredictor` combines observed and imputed components.

### Linear Projection (Proposed Approach)

Instead of imputing individual dosages, directly learn weights $a$ for platform variants such that $z^T a \approx x^T \beta$ (the true PRS). The OLS solution is:

$$a = (Z^T Z)^{-1} Z^T X \beta$$

where $Z$ is the $n \times p$ matrix of platform genotypes and $X$ is the $n \times q$ matrix of PRS variant genotypes from a reference population.

### Equivalence Proof (Without Regularization or Windowing)

**Claim**: Under OLS with all platform variants as predictors, the two approaches produce identical predictions.

**Proof**: Let $P_Z = Z(Z^T Z)^{-1}Z^T$ be the projection matrix.

Projection approach:

$$\hat{S}_{proj} = P_Z \cdot S_{true} = P_Z(X_O \beta_O + X_M \beta_M)$$

Since $X_O$ columns are in the column space of $Z$ (observed PRS variants are on the platform), $P_Z X_O = X_O$. Therefore:

$$\hat{S}_{proj} = X_O \beta_O + P_Z X_M \beta_M$$

Imputation approach: Each missing variant's OLS imputation is $\hat{x}_j = P_Z x_j$, so $\hat{X}_M = P_Z X_M$:

$$\hat{S}_{imp} = X_O \beta_O + P_Z X_M \beta_M = \hat{S}_{proj} \quad \square$$

### Where the Approaches Diverge

With **elastic net regularization**, the approaches optimize different objectives:

- **Imputation** solves $|M|$ independent problems, each minimizing per-variant prediction error. The regularization is agnostic to $\beta_j$ -- it doesn't know which variants matter most for the PRS.

- **Projection** solves a single problem minimizing PRS prediction error directly. The regularization is informed by $\beta_j$ -- a variant with a tiny beta contributes little to the PRS error, so the model won't waste regularization budget on it.

**Consequence**: With heterogeneous betas, the projection approach can allocate model capacity more efficiently, focusing on the variants that matter most for PRS accuracy.

### Region-Based Decomposition

The projection approach needs a windowing strategy analogous to the imputation approach's per-variant windows. The natural unit is a **region**: a contiguous genomic interval formed by merging overlapping per-variant windows.

For each missing PRS variant $j$, define its window as $[pos_j - W, pos_j + W]$ (same window size $W$ as in the imputation approach, default 1 Mb). Overlapping windows on the same chromosome merge into a single region. For each region $R$:

- **Target**: $S_R = \sum_{j \in R, j \in M} x_j \beta_j$ (PRS contribution from **missing** variants in $R$)
- **Predictors**: Platform variants within $R$'s boundaries on the same chromosome
- **Model**: Elastic net regression of $S_R$ on $Z_R$

Observed PRS variants are handled separately with exact dosages (same as in the imputation approach).

**Why missing-only target?** This isolates the comparison. Both approaches use the exact same observed-variant component. The only difference is how they predict the missing-variant component: per-variant (imputation) vs. per-region (projection).

**Single-variant regions**: When a region contains only one missing PRS variant, the projection target is $S_R = x_j \beta_j$ and the solution is $a_R = w_j \cdot \beta_j$ (imputation weights scaled by $\beta_j$). The approaches are equivalent in this case (modulo regularization path). **Multi-variant regions** are where the projection approach can potentially outperform by jointly optimizing.

---

## 2. Architecture Overview

### Parallel Module Structure

| Imputation (existing) | Projection (new) | Shared |
|----------------------|-----------------|--------|
| `models/elastic_net.py` | `models/projection.py` | -- |
| `models/trainer.py` | `models/projection_trainer.py` | `core/harmonizer.py` |
| `models/predictor.py` | `models/projection_predictor.py` | `predictor.py:compute_observed_prs()` |
| `models/bounding.py` | *(not needed)* | -- |
| `core/linear_imputation_prs.py` | `core/linear_projection_prs.py` | I/O loaders, harmonizer |
| `evaluation/evaluator.py` | `evaluation/projection_evaluator.py` | `evaluation/metrics.py`, `evaluation/calibration.py` |
| -- | `core/regions.py` *(new)* | -- |

### Reusable Existing Components

These files are used as-is, with no modifications to their behavior:

| Component | File | What's Reused |
|-----------|------|---------------|
| Variant partitioning | `core/harmonizer.py` | `partition_variants()`, `filter_to_local_window()`, `align_effect_alleles()` |
| I/O loaders | `io/genotype_loader.py`, `io/prs_loader.py`, `io/platform_loader.py` | All loading functions |
| Calibration | `evaluation/calibration.py` | `estimate_cv_calibration()`, `compute_cv_predicted_prs()` |
| Evaluation metrics | `evaluation/metrics.py` | `compute_prs_metrics()`, `compute_percentile_concordance()` |
| Observed PRS computation | `models/predictor.py` | `compute_observed_prs()` |
| Data types | `core/types.py` | `VariantInfo`, `PredictionResult`, `CalibrationParams`, `EvaluationMetrics`, `GenotypeData` |

### Files Modified (Additive Only)

These files receive only additive changes (new imports/exports). No existing behavior is altered.

| File | Change |
|------|--------|
| `imputed_prs/core/types.py` | Add `ProjectionRegionModel`, `ProjectionTrainingResult` dataclasses |
| `imputed_prs/core/__init__.py` | Add imports and `__all__` entries for new types and `LinearProjectionPRS` |
| `imputed_prs/__init__.py` | Add `LinearProjectionPRS` to top-level imports and `__all__` |
| `imputed_prs/models/__init__.py` | Add imports and `__all__` entries for new projection modules |
| `imputed_prs/evaluation/__init__.py` | Add `ProjectionEvaluator` to imports and `__all__` |

---

## 3. Phase 1: Data Types and Region Merging

**Goal**: Define new data types and implement the interval-merging algorithm. Pure computation with no dependencies on training or prediction.

### 3.1 New File: `imputed_prs/core/regions.py`

#### `GenomicRegion` Dataclass

```python
@dataclass
class GenomicRegion:
    """A contiguous genomic interval containing one or more missing PRS variants.

    Created by merging overlapping per-variant windows.

    Attributes:
        chromosome: Chromosome identifier (normalized, e.g., "1", "X").
        start: Start position (inclusive). Derived from min(pos - window_size)
            across all PRS variants whose windows contributed to this region,
            clamped to >= 0.
        end: End position (inclusive). Derived from max(pos + window_size).
        prs_variant_ids: List of missing PRS variant IDs in this region.
        prs_variant_indices: Indices into the missing_prs_df for variants
            in this region. Used to index into the X matrix.
    """
    chromosome: str
    start: int
    end: int
    prs_variant_ids: List[str]
    prs_variant_indices: List[int]
```

#### `RegionDecompositionResult` Dataclass

```python
@dataclass
class RegionDecompositionResult:
    """Result of decomposing PRS variants into non-overlapping regions.

    Attributes:
        regions: List of GenomicRegion objects, sorted by (chromosome, start).
        n_regions: Total number of merged regions.
        n_variants_in_regions: Total PRS variants covered.
        variants_per_region: List of counts (how many PRS variants per region).
        max_region_span_bp: Largest region span in base pairs.
    """
    regions: List[GenomicRegion]
    n_regions: int
    n_variants_in_regions: int
    variants_per_region: List[int]
    max_region_span_bp: int
```

#### `merge_variant_windows()` Function

```python
def merge_variant_windows(
    prs_variants: pd.DataFrame,
    window_size: int = 1_000_000,
) -> RegionDecompositionResult:
    """Merge overlapping per-variant windows into non-overlapping genomic regions.

    Algorithm:
    1. For each missing PRS variant, compute window = [pos - W, pos + W].
       Clamp start to >= 0.
    2. Group by chromosome (normalized).
    3. Within each chromosome, sort by start position.
    4. Sweep-line merge: if current interval overlaps or is adjacent to previous,
       extend previous; otherwise start a new interval.
    5. Track which variant IDs/indices belong to each merged region.

    Args:
        prs_variants: DataFrame with columns: variant_id, chromosome, position.
            These are the missing PRS variants (not observed ones).
        window_size: Window size in base pairs on each side. Default: 1,000,000.

    Returns:
        RegionDecompositionResult with merged regions sorted by (chromosome, start).
    """
```

**Algorithm detail**: Chromosome normalization should use the same `_normalize_chromosome()` logic from `core/harmonizer.py` (strip "chr" prefix, uppercase). This ensures consistent chromosome representation.

### 3.2 Additions to `imputed_prs/core/types.py`

#### `ProjectionRegionModel` Dataclass

```python
@dataclass
class ProjectionRegionModel:
    """Trained projection model for a single genomic region.

    For region R, this stores the learned weights a_R such that
    z_R^T a_R + intercept approximates S_R = sum(x_j * beta_j) for
    missing PRS variants j in R.

    Attributes:
        region_id: Unique identifier, format "chr{chrom}:{start}-{end}".
        chromosome: Chromosome.
        start: Region start position.
        end: Region end position.
        prs_variant_ids: Missing PRS variant IDs in this region.
        betas: Effect sizes for PRS variants in this region. Shape: (n_prs_variants,).
        predictor_variant_ids: Platform variant IDs used as predictors.
        coefficients: Learned regression coefficients. Shape: (n_predictors,).
            Empty array for intercept-only models.
        intercept: Model intercept (mean of S_R for intercept-only models).
        cv_mse: Cross-validated MSE for this region's PRS contribution.
        cv_r2: Cross-validated R-squared for this region's PRS contribution.
        is_intercept_only: True if no predictors available or all coefficients
            shrunk to zero.
        mean_prs_contribution: Mean of S_R across training samples.
        predictor_allele_frequencies: Allele frequencies for each predictor
            variant. Shape: (n_predictors,). Used for mean-substitution of
            missing predictors at inference time.
    """
    region_id: str
    chromosome: str
    start: int
    end: int
    prs_variant_ids: List[str]
    betas: np.ndarray
    predictor_variant_ids: List[str]
    coefficients: np.ndarray
    intercept: float
    cv_mse: float
    cv_r2: float
    is_intercept_only: bool
    mean_prs_contribution: float
    predictor_allele_frequencies: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary, handling numpy arrays."""
        result = asdict(self)
        result["betas"] = self.betas.tolist()
        result["coefficients"] = self.coefficients.tolist()
        result["predictor_allele_frequencies"] = self.predictor_allele_frequencies.tolist()
        return result
```

#### `ProjectionTrainingResult` Dataclass

```python
@dataclass
class ProjectionTrainingResult:
    """Result from training projection models for all regions.

    Attributes:
        region_models: Dict mapping region_id to ProjectionRegionModel.
        cv_predictions: Dict mapping region_id to out-of-fold CV predictions
            of S_R for each sample. Shape of each: (n_samples,).
        n_regions_trained: Number of regions successfully trained.
        n_regions_failed: Number of regions where training failed.
        n_intercept_only: Number of regions using intercept-only models.
        training_summary: Summary statistics dict with keys:
            mean_r2, median_r2, std_r2, min_r2, max_r2,
            n_high_quality (r2 > 0.8), n_medium_quality (0.4-0.8),
            n_low_quality (r2 <= 0.4), mean_n_predictors,
            mean_n_prs_variants_per_region.
    """
    region_models: Dict[str, ProjectionRegionModel]
    cv_predictions: Dict[str, np.ndarray]
    n_regions_trained: int
    n_regions_failed: int
    n_intercept_only: int
    training_summary: Dict[str, Any]
```

### 3.3 Export Changes

**`imputed_prs/core/__init__.py`**: Add to imports and `__all__`:
- `ProjectionRegionModel`
- `ProjectionTrainingResult`
- `GenomicRegion` (from `regions.py`)
- `RegionDecompositionResult` (from `regions.py`)
- `merge_variant_windows` (from `regions.py`)

### 3.4 Tests: `tests/test_regions.py`

Follow the pytest class pattern used in `tests/test_harmonizer.py`.

```python
class TestMergeVariantWindows:
    def test_single_variant_single_region(self):
        """One variant produces one region spanning [pos-W, pos+W]."""

    def test_two_variants_same_chrom_overlapping(self):
        """Two variants with overlapping windows merge into 1 region."""

    def test_two_variants_same_chrom_non_overlapping(self):
        """Two variants far apart produce 2 separate regions."""

    def test_multiple_chromosomes(self):
        """Variants on different chromosomes never merge."""

    def test_chain_merge(self):
        """Chain of variants where each overlaps the next: all merge into 1."""
        # e.g., positions 0, 1.5M, 3M with W=1M:
        # [0, 1M] overlaps [0.5M, 2.5M] overlaps [2M, 4M] -> one region [0, 4M]

    def test_variant_ids_tracked_correctly(self):
        """Each region contains exactly the variant IDs whose windows contributed."""

    def test_variant_indices_tracked_correctly(self):
        """prs_variant_indices match DataFrame row indices."""

    def test_empty_dataframe(self):
        """Empty input produces 0 regions."""

    def test_position_at_zero(self):
        """Variant at position 0: region start is clamped to 0 (not negative)."""

    def test_regions_sorted(self):
        """Regions are returned sorted by (chromosome, start)."""

    def test_window_size_zero(self):
        """Window size 0: each variant is its own region (point intervals don't overlap)."""

    def test_result_statistics(self):
        """n_regions, n_variants_in_regions, variants_per_region, max_region_span_bp are correct."""

    def test_chromosome_normalization(self):
        """'chr1' and '1' are treated as the same chromosome."""


class TestGenomicRegion:
    def test_construction(self):
        """GenomicRegion can be constructed with all required fields."""


class TestProjectionRegionModel:
    def test_to_dict(self):
        """to_dict() converts numpy arrays to lists."""

    def test_region_id_format(self):
        """region_id follows 'chr{chrom}:{start}-{end}' format."""


class TestProjectionTrainingResult:
    def test_construction(self):
        """ProjectionTrainingResult can be constructed with all required fields."""
```

### 3.5 Regression Verification

```bash
pytest tests/test_harmonizer.py tests/test_types.py -v
```

Both must pass unchanged.

---

## 4. Phase 2: Per-Region Model Fitting

**Goal**: Implement the core per-region elastic net fitting function and the region-level trainer. This is the computational heart of the projection approach.

### 4.1 New File: `imputed_prs/models/projection.py`

#### `SingleRegionModelResult` Dataclass

```python
@dataclass
class SingleRegionModelResult:
    """Result from fitting a single region projection model.

    Mirrors SingleVariantModelResult but for a region whose target is
    S_R = X_R @ beta_R (a continuous PRS contribution, not a dosage).

    Attributes:
        coefficients: Regression coefficients for predictor variants.
            Shape: (n_predictors,). Empty array for intercept-only.
        intercept: Model intercept (mean of target for intercept-only).
        cv_predictions: Out-of-fold CV predictions. Shape: (n_samples,).
            Contains NaN for samples excluded due to missing values.
        cv_mse: Mean squared error from cross-validation.
        cv_r2: R-squared from cross-validation.
        is_intercept_only: True if no predictors or all coefficients zero.
        n_predictors: Number of predictor variants.
        n_samples: Total number of samples.
        l1_ratio: ElasticNet L1 ratio used.
        alpha: ElasticNet alpha used.
    """
    coefficients: np.ndarray
    intercept: float
    cv_predictions: np.ndarray
    cv_mse: float
    cv_r2: float
    is_intercept_only: bool
    n_predictors: int
    n_samples: int
    l1_ratio: float
    alpha: float
```

#### `fit_single_region_model()` Function

```python
def fit_single_region_model(
    target_prs_contribution: np.ndarray,
    predictor_dosages: np.ndarray,
    l1_ratio: float = 0.5,
    alpha: float = 0.01,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> SingleRegionModelResult:
    """Fit ElasticNet projection model for a single genomic region.

    The target is S_R = X_R @ beta_R (regional PRS contribution from missing
    variants). The predictors are platform variant dosages Z_R in the region.

    This mirrors fit_single_variant_model() from elastic_net.py but:
    - Target is a weighted sum (continuous, possibly negative), not a dosage.
    - No dosage bounds [0, 2] apply to the target.
    - The intercept represents the mean regional PRS contribution.

    The structure follows fit_single_variant_model() exactly:
    1. Input validation and float64 conversion.
    2. Create valid mask (non-NaN target AND all predictors non-NaN).
    3. Handle edge cases -> intercept-only:
       - No predictors (n_predictors == 0)
       - Too few valid samples (< cv_folds)
       - Zero variance in target (std < 1e-10)
    4. K-fold cross-validation loop:
       - Split valid data into k folds
       - Fit sklearn.linear_model.ElasticNet on each fold's train data
       - Collect out-of-fold predictions
    5. Fit final model on all valid data.
    6. Compute CV R^2 from out-of-fold predictions.
    7. Detect if all coefficients shrunk to zero -> mark as intercept-only.

    Args:
        target_prs_contribution: S_R values for each sample. Shape: (n_samples,).
        predictor_dosages: Platform variant dosages. Shape: (n_samples, n_predictors).
        l1_ratio: ElasticNet L1/L2 mixing (0=Ridge, 1=Lasso). Default: 0.5.
        alpha: Regularization strength. Default: 0.01.
        cv_folds: Number of CV folds. Default: 5.
        random_state: Random seed. Default: None.

    Returns:
        SingleRegionModelResult with fitted model parameters and CV metrics.
    """
```

**Implementation notes**:
- Use `sklearn.linear_model.ElasticNet` (same as `elastic_net.py`)
- Use `sklearn.model_selection.KFold` for CV (same as `elastic_net.py`)
- The `_fit_intercept_only_region()` helper returns intercept = `np.nanmean(target)`, cv_predictions = constant, cv_mse = variance of target

### 4.2 New File: `imputed_prs/models/projection_trainer.py`

#### `ProjectionRegionTrainer` Class

```python
class ProjectionRegionTrainer:
    """Orchestrates training of projection models for all genomic regions.

    Mirrors ImputationModelTrainer but operates at region granularity instead
    of per-variant.

    The training flow:
    1. Decompose missing PRS variants into non-overlapping regions.
    2. For each region, compute the PRS contribution target and fit an
       ElasticNet model using platform variants in the region as predictors.
    3. Collect cross-validated predictions for downstream calibration.
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        verbose: int = 1,
    ):
        """Initialize the trainer.

        Args:
            window_size: Size of genomic window (bp) for defining regions
                and selecting predictor variants. Default: 1,000,000 (1 Mb).
            l1_ratio: ElasticNet L1/L2 mixing parameter. Default: 0.5.
            alpha: ElasticNet regularization strength. Default: 0.01.
            cv_folds: Number of CV folds. Default: 5.
            n_jobs: Number of parallel jobs (1=sequential, -1=all CPUs). Default: 1.
            random_state: Random seed. Default: None.
            max_predictors: Max predictor variants per region. Default: None (no limit).
            verbose: Verbosity level (0=silent, 1=progress, 2=debug). Default: 1.
        """

    def fit_all_regions(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        prs_variants: pd.DataFrame,
        platform_variant_info: pd.DataFrame,
    ) -> ProjectionTrainingResult:
        """Fit projection models for all regions.

        Args:
            Z: Platform genotype dosages (n_samples, n_platform_variants).
            X: Missing PRS variant dosages (n_samples, n_missing_variants).
            prs_variants: DataFrame for missing PRS variants with columns:
                variant_id, chromosome, position, effect_allele, beta.
                Must have same number of rows as X.shape[1].
            platform_variant_info: DataFrame with columns:
                variant_id, chromosome, position.
                Must have same number of rows as Z.shape[1].

        Returns:
            ProjectionTrainingResult with trained region models and CV predictions.

        Algorithm:
        1. Call merge_variant_windows(prs_variants, self.window_size).
        2. For each region R:
           a. Get variant indices: R.prs_variant_indices
           b. Get betas: prs_variants.iloc[indices]["beta"].values
           c. Compute target: S_R = X[:, indices] @ betas  (shape: n_samples)
           d. Find platform variants in [R.start, R.end] on R.chromosome:
              Use filter_to_local_window() from harmonizer, called with
              target_chrom=R.chromosome, target_pos=(R.start+R.end)//2,
              window_size=(R.end-R.start)//2 + 1.
              Or: directly filter platform_variant_info to same chromosome
              and position within [R.start, R.end].
           e. Extract Z_R = Z[:, platform_indices_in_region]
           f. Optionally limit to max_predictors closest to region center.
           g. Call fit_single_region_model(S_R, Z_R, ...)
           h. Compute predictor allele frequencies:
              af_k = np.nanmean(Z[:, k]) / 2.0 for each predictor k
           i. Wrap into ProjectionRegionModel:
              - region_id = f"chr{R.chromosome}:{R.start}-{R.end}"
              - mean_prs_contribution = np.nanmean(S_R)
              - Store predictor_allele_frequencies for inference fallback
        3. Collect all region models and CV predictions.
        4. Compute training summary via _compute_projection_training_summary().
        """
```

#### Standalone Function for Parallelization

```python
def _fit_one_region(
    region: GenomicRegion,
    Z: np.ndarray,
    X: np.ndarray,
    prs_variants: pd.DataFrame,
    platform_variant_info: pd.DataFrame,
    window_size: int,
    l1_ratio: float,
    alpha: float,
    cv_folds: int,
    random_state: Optional[int],
    max_predictors: Optional[int],
) -> Tuple[str, Optional[ProjectionRegionModel], Optional[np.ndarray], bool]:
    """Fit projection model for a single region (parallelizable).

    Returns:
        Tuple of (region_id, model_or_None, cv_predictions_or_None, is_intercept_only).
    """
```

**Parallelization**: Use `joblib.Parallel` with `prefer="threads"`, same pattern as `trainer.py`.

#### Summary Computation

```python
def _compute_projection_training_summary(
    region_models: Dict[str, ProjectionRegionModel],
) -> Dict[str, Any]:
    """Compute summary statistics across all trained region models.

    Returns dict with keys: mean_r2, median_r2, std_r2, min_r2, max_r2,
    n_high_quality, n_medium_quality, n_low_quality, mean_n_predictors,
    mean_n_prs_variants_per_region.
    """
```

### 4.3 Export Changes

**`imputed_prs/models/__init__.py`**: Add to imports and `__all__`:
- `ProjectionRegionTrainer` (from `projection_trainer.py`)
- `fit_single_region_model` (from `projection.py`)
- `SingleRegionModelResult` (from `projection.py`)

### 4.4 Tests: `tests/test_projection.py`

Follow the patterns in `tests/test_elastic_net.py` and `tests/test_trainer.py`.

```python
def create_projection_test_data(
    n_samples: int = 100,
    n_platform_variants: int = 50,
    n_missing_variants: int = 10,
    random_state: int = 42,
):
    """Create synthetic test data for projection tests.

    Similar to test_trainer.py:create_test_data() but ensures that the
    PRS contribution S_R = X @ betas can be predicted from Z.

    Returns:
        Tuple of (Z, X, prs_variants, platform_variant_info).
    """
    # Generate Z (platform dosages) as binomial(2, 0.3)
    # Generate X (missing dosages) with linear relationship to nearby Z columns
    # Generate betas from uniform(0.1, 0.5)
    # All variants on chromosome "1" with positions that create
    # overlapping windows (to test multi-variant regions)


class TestFitSingleRegionModel:
    def test_basic_fitting(self):
        """Known linear relationship: verify coefficients recover true weights."""

    def test_no_predictors_intercept_only(self):
        """Empty predictor matrix -> intercept-only, intercept == mean(target)."""

    def test_zero_variance_target(self):
        """Constant target -> intercept-only."""

    def test_too_few_samples(self):
        """Fewer valid samples than cv_folds -> intercept-only."""

    def test_cv_predictions_shape(self):
        """cv_predictions has shape (n_samples,)."""

    def test_cv_predictions_nan_for_invalid(self):
        """Samples with NaN in target/predictors have NaN in cv_predictions."""

    def test_reproducibility(self):
        """Same random_state produces identical results."""

    def test_negative_target_values(self):
        """S_R can be negative (unlike dosages): verify no clipping occurs."""

    def test_all_coefficients_zero_intercept_only(self):
        """Strong regularization shrinks all coefficients to zero -> intercept-only."""


class TestProjectionRegionTrainer:
    def test_basic_training(self):
        """Training with synthetic data produces ProjectionTrainingResult."""

    def test_region_count_correct(self):
        """Number of region models matches expected from merge_variant_windows."""

    def test_cv_predictions_per_region_shape(self):
        """Each region's cv_predictions has shape (n_samples,)."""

    def test_training_summary_keys(self):
        """Training summary contains all expected keys."""

    def test_empty_prs_variants(self):
        """No missing variants -> 0 regions, empty result."""

    def test_parallel_vs_sequential(self):
        """n_jobs=1 and n_jobs=2 produce same results (with same random_state)."""

    def test_max_predictors_respected(self):
        """max_predictors limits predictor count per region."""

    def test_target_computation_correctness(self):
        """S_R = X[:, indices] @ beta is computed correctly.
        Verify with manual calculation on known inputs."""

    def test_predictor_allele_frequencies_stored(self):
        """predictor_allele_frequencies matches manual AF computation."""

    def test_input_validation(self):
        """Missing DataFrame columns or shape mismatches raise ValidationError."""
```

### 4.5 Regression Verification

```bash
pytest tests/test_trainer.py tests/test_elastic_net.py -v
```

Both must pass unchanged.

---

## 5. Phase 3: Projection Predictor

**Goal**: Implement the inference pipeline that computes PRS from trained projection region models.

### 5.1 New File: `imputed_prs/models/projection_predictor.py`

#### `compute_projected_prs()` Function

```python
def compute_projected_prs(
    user_dosages: Dict[str, Optional[float]],
    region_models: List[ProjectionRegionModel],
) -> Tuple[float, float, int, int]:
    """Compute PRS contribution from projection regions.

    For each region model:
    1. Gather predictor dosages from user data.
    2. For any missing predictor, substitute 2 * AF (population mean dosage)
       using model.predictor_allele_frequencies.
    3. Compute prediction: S_hat_R = z_R^T @ a_R + intercept_R.
    4. No dosage clipping (target is PRS contribution, not a dosage).

    Key differences from compute_imputed_prs():
    - No dosage clipping [0, 2] on output.
    - Missing predictors get population mean (2*AF) instead of falling
      back to intercept-only for the entire variant.
    - Variance is sum of cv_mse across regions (global SE), not
      per-variant beta^2 * adjusted_residual_variance.

    Args:
        user_dosages: Dict mapping variant_id to dosage (0-2) or None.
        region_models: List of ProjectionRegionModel objects.

    Returns:
        Tuple of (prs_projected, total_variance, n_regions_used,
                  n_predictors_substituted):
            - prs_projected: Sum of S_hat_R across all regions.
            - total_variance: Sum of cv_mse across all regions.
            - n_regions_used: Count of regions computed.
            - n_predictors_substituted: Count of individual predictors where
              AF-based mean was substituted for missing user data.
    """
```

**Implementation detail for missing predictor handling**:

```python
for i, pred_id in enumerate(model.predictor_variant_ids):
    dosage = user_dosages.get(pred_id)
    if dosage is None:
        # Substitute population mean dosage
        dosage = 2.0 * model.predictor_allele_frequencies[i]
        n_predictors_substituted += 1
    predictor_dosages.append(dosage)
```

This differs from the imputation approach's all-or-nothing fallback (`predictor.py:98-100`): the projection approach substitutes individual missing predictors while keeping the rest, preserving partial information.

#### `ProjectionPredictor` Class

```python
class ProjectionPredictor:
    """Full PRS prediction combining observed and projected components.

    Mirrors PRSPredictor but uses region-based projection models
    instead of per-variant imputation models.

    Prediction: PRS = S_observed + sum_R(S_hat_R)

    Where S_observed = sum(z_j * beta_j) for observed PRS variants
    (computed by compute_observed_prs from predictor.py), and S_hat_R
    is the projected regional PRS contribution.
    """

    def __init__(
        self,
        observed_variants: List[VariantInfo],
        region_models: List[ProjectionRegionModel],
        calibration_params: Optional[CalibrationParams] = None,
    ):

    def predict(
        self,
        user_genotypes: Dict[str, Optional[float]],
        apply_calibration: bool = True,
    ) -> PredictionResult:
        """Compute full PRS with uncertainty quantification.

        Steps:
        1. compute_observed_prs(user_genotypes, self.observed_variants)
           (reused from predictor.py -- no reimplementation)
        2. compute_projected_prs(user_genotypes, self.region_models)
        3. prs_raw = prs_observed + prs_projected
        4. SE = sqrt(total_variance) if total_variance > 0 else 0.0
        5. CI = [prs_raw - 1.96*SE, prs_raw + 1.96*SE]
        6. If apply_calibration and calibration_params:
           prs_scaled = slope * prs_raw + intercept
           se_scaled = |slope| * SE
        7. Return PredictionResult with:
           prs_imputed_component = prs_projected  (reuse existing field)
           n_variants_imputed = sum of n_prs_variants across regions

        Returns:
            PredictionResult (same type as PRSPredictor).
        """
```

### 5.2 Tests: `tests/test_projection_predictor.py`

Follow the pattern of `tests/test_predictor.py`.

```python
class TestComputeProjectedPrs:
    def test_basic_calculation(self):
        """Known coefficients and intercept: verify z^T a + intercept."""

    def test_intercept_only_region(self):
        """Region with is_intercept_only=True uses intercept value."""

    def test_missing_predictor_substitution(self):
        """Missing user variant -> substituted with 2*AF.
        Verify n_predictors_substituted count."""

    def test_all_predictors_missing(self):
        """All predictors missing -> all substituted with 2*AF.
        Result should be close to sum(a_k * 2*AF_k) + intercept."""

    def test_no_dosage_clipping(self):
        """Projected value can be negative: verify no clipping to [0,2]."""

    def test_empty_region_models(self):
        """Empty list -> (0.0, 0.0, 0, 0)."""

    def test_multiple_regions(self):
        """Two regions: verify sum is correct."""

    def test_variance_from_cv_mse(self):
        """total_variance == sum of region.cv_mse."""


class TestProjectionPredictor:
    def test_basic_prediction_mixed(self):
        """Both observed and projected components contribute."""

    def test_all_observed_no_projection(self):
        """No region models: PRS == observed component."""

    def test_all_projected_no_observed(self):
        """No observed variants: PRS == projected component."""

    def test_with_calibration(self):
        """Calibration scaling applied correctly."""

    def test_without_calibration(self):
        """No scaling when apply_calibration=False or params=None."""

    def test_confidence_interval(self):
        """CI = prs +/- 1.96 * SE."""

    def test_returns_prediction_result_type(self):
        """Return type is PredictionResult (same as imputation)."""
```

### 5.3 Regression Verification

```bash
pytest tests/test_predictor.py -v
```

Must pass unchanged.

---

## 6. Phase 4: Main `LinearProjectionPRS` Class

**Goal**: Implement the high-level API class with `fit()` and `predict()` methods, mirroring `LinearImputationPRS`.

### 6.1 New File: `imputed_prs/core/linear_projection_prs.py`

#### Constructor

```python
class LinearProjectionPRS:
    """High-level API for training and using projection-based PRS models.

    Mirrors LinearImputationPRS but uses the linear projection approach:
    instead of imputing individual missing variant dosages, it directly
    learns platform-variant weights to approximate each region's PRS
    contribution.

    Example:
        >>> model = LinearProjectionPRS(window_size=1_000_000, cv_folds=5)
        >>> model.fit(
        ...     reference_genotypes="1000g_eur.vcf.gz",
        ...     prs_definition="PGS000004",
        ...     platform_name="23andme_v5",
        ... )
        >>> result = model.predict("user_genotypes.txt")
        >>> print(f"PRS: {result.prs:.3f} "
        ...       f"(95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        verbose: int = 1,
    ):
```

**Note**: No `tuning_scope` parameter. The provided `l1_ratio` and `alpha` are used directly. This simplifies the initial implementation; hyperparameter tuning can be added later.

**Instance variables** (same pattern as `LinearImputationPRS`):
- `self._is_fitted: bool = False`
- `self._observed_variants: Optional[List[VariantInfo]] = None`
- `self._region_models: Optional[List[ProjectionRegionModel]] = None`
- `self._calibration_params: Optional[CalibrationParams] = None`
- `self._training_result: Optional[ProjectionTrainingResult] = None`
- `self._platform_variant_index: Optional[Dict[str, int]] = None`
- Metadata: `_prs_id`, `_platform_name`, `_genome_build`, `_model_name`

#### `fit()` Method

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
) -> "LinearProjectionPRS":
    """Train projection models on reference genotype data.

    Steps 1-8 are identical to LinearImputationPRS.fit() and are
    duplicated here to avoid modifying the existing class. A future
    refactor may extract shared helpers.

    Steps:
    1. Input validation (exactly one platform source).
    2. Load PRS definition (DataFrame, file, or PGS Catalog ID).
    3. Load platform variant set.
    4. Partition variants (observed vs missing).
    5. Load reference genotypes.
    6. Validate genome build.
    7. Align effect alleles for observed variants.
    8. Build training matrices Z (platform) and X (missing PRS).
    9. Train projection models:
       trainer = ProjectionRegionTrainer(
           window_size=self.window_size,
           l1_ratio=self.l1_ratio,
           alpha=self.alpha,
           cv_folds=self.cv_folds,
           n_jobs=self.n_jobs,
           random_state=self.random_state,
           max_predictors=self.max_predictors,
           verbose=self.verbose,
       )
       training_result = trainer.fit_all_regions(Z, X, missing_prs_df, platform_variant_info)
    10. Compute calibration parameters:
        - Build S_cv: observed component (true genotypes x betas) +
          sum of per-region CV predictions x (implicitly weighted by betas,
          since the CV predictions ARE predictions of S_R = X_R @ beta_R).
          S_cv[i] = sum(X[i, obs_indices] * obs_betas) +
                    sum(region_cv_predictions[region_id][i] for each region)
        - Build S_true: X_full @ all_betas (using all PRS variants).
        - calibration_params = estimate_cv_calibration(S_cv, S_true)
    11. Build VariantInfo list for observed variants.
    12. Populate instance state, set _is_fitted = True.

    Returns:
        self (for method chaining).
    """
```

**Calibration note**: The key difference from imputation calibration is how `S_cv` is constructed. In imputation, `S_cv` uses per-variant CV predictions. In projection, `S_cv` uses per-region CV predictions, which are already the predicted PRS contributions (they include the beta weighting). The existing `estimate_cv_calibration(s_cv, s_true)` function from `evaluation/calibration.py` can be reused directly.

#### `predict()` Method

```python
def predict(
    self,
    user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
    apply_calibration: bool = True,
) -> PredictionResult:
    """Compute PRS for user genotypes using projection models.

    Args:
        user_genotypes: User genotype data as file path, DataFrame, or dict.
        apply_calibration: Whether to apply calibration scaling. Default: True.

    Returns:
        PredictionResult with PRS value and uncertainty estimates.

    Raises:
        ModelNotFittedError: If model has not been fitted.
    """
    # Same user genotype loading logic as LinearImputationPRS.predict()
    # Build ProjectionPredictor with observed_variants, region_models, calibration_params
    # Call predictor.predict(user_dosages, apply_calibration)
    # Return PredictionResult
```

#### Properties

```python
@property
def is_fitted(self) -> bool: ...

@property
def observed_variants(self) -> List[VariantInfo]:
    """List of observed PRS variants (on the platform)."""

@property
def region_models(self) -> List[ProjectionRegionModel]:
    """List of trained projection region models."""

@property
def calibration_params(self) -> Optional[CalibrationParams]: ...

@property
def summary(self) -> Dict[str, Any]:
    """Model summary with region counts and quality statistics.

    Returns dict with keys:
    - n_observed_variants
    - n_missing_variants
    - n_regions
    - n_intercept_only_regions
    - training_summary (from ProjectionTrainingResult)
    - calibration (CalibrationParams as dict, if available)
    """

@property
def variant_table(self) -> pd.DataFrame:
    """Per-region summary table.

    Columns: region_id, chromosome, start, end, n_prs_variants,
    n_predictors, cv_r2, cv_mse, is_intercept_only, prs_variant_ids.
    """
```

### 6.2 Export Changes

**`imputed_prs/core/__init__.py`**: Add `LinearProjectionPRS` to imports and `__all__`.

**`imputed_prs/__init__.py`**: Add `LinearProjectionPRS` to imports and `__all__`.

### 6.3 Tests: `tests/test_linear_projection_prs.py`

Follow the pattern of `tests/test_linear_imputation_prs.py`.

The tests in this file that involve `fit()` need synthetic reference genotype data. Follow the pattern from `test_linear_imputation_prs.py` for creating temporary VCF files or use the approach from `test_evaluator.py` which writes temporary VCFs.

```python
class TestLinearProjectionPRSConstructor:
    def test_default_parameters(self):
        """Default parameters match expected values."""

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""

    def test_initial_unfitted_state(self):
        """is_fitted returns False before fit()."""


class TestLinearProjectionPRSUnfittedErrors:
    def test_predict_raises_error(self):
        """predict() before fit() raises ModelNotFittedError."""

    def test_summary_raises_error(self):
        """summary before fit() raises ModelNotFittedError."""

    def test_region_models_raises_error(self):
        """region_models before fit() raises ModelNotFittedError."""

    def test_variant_table_raises_error(self):
        """variant_table before fit() raises ModelNotFittedError."""


class TestLinearProjectionPRSFitValidation:
    def test_no_platform_source(self):
        """No platform source raises ValidationError."""

    def test_multiple_platform_sources(self):
        """Multiple platform sources raises ValidationError."""


class TestLinearProjectionPRSPredict:
    def test_predict_returns_prediction_result(self):
        """predict() returns PredictionResult type."""

    def test_predict_dict_input(self):
        """predict() accepts Dict[str, float] input."""

    def test_predict_with_calibration(self):
        """Calibration is applied when available and requested."""

    def test_predict_without_calibration(self):
        """No calibration when apply_calibration=False."""


class TestLinearProjectionPRSProperties:
    def test_summary_keys(self):
        """summary dict has all expected keys."""

    def test_variant_table_columns(self):
        """variant_table DataFrame has expected columns."""

    def test_region_models_type(self):
        """region_models returns List[ProjectionRegionModel]."""

    def test_is_fitted_after_fit(self):
        """is_fitted returns True after fit()."""
```

### 6.4 Regression Verification

```bash
pytest tests/test_linear_imputation_prs.py tests/test_api.py -v
```

Both must pass unchanged.

---

## 7. Phase 5: Evaluation

**Goal**: Create `ProjectionEvaluator` that mirrors `ImputationEvaluator`, enabling evaluation of `LinearProjectionPRS` models on held-out data.

### 7.1 New File: `imputed_prs/evaluation/projection_evaluator.py`

```python
class ProjectionEvaluator:
    """Evaluator for fitted LinearProjectionPRS models.

    Mirrors ImputationEvaluator but uses ProjectionPredictor for
    computing projected PRS values.
    """

    def __init__(self, model: "LinearProjectionPRS", verbose: int = 1):
        """Initialize the evaluator.

        Args:
            model: A fitted LinearProjectionPRS model.
            verbose: Verbosity level.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """

    def evaluate(
        self,
        evaluation_genotypes: Union[str, Path, GenotypeData],
    ) -> EvaluationMetrics:
        """Evaluate the model on held-out genotype data.

        Steps:
        1. Load evaluation genotypes (same as ImputationEvaluator).
        2. Compute true PRS (same logic as ImputationEvaluator._compute_true_prs).
        3. Compute projected PRS for all samples via _compute_projected_prs_batch.
        4. Call compute_prs_metrics(projected, true) -> EvaluationMetrics.

        Returns:
            EvaluationMetrics comparing projected vs true PRS.
        """

    def _compute_true_prs(self, genotype_data: GenotypeData) -> np.ndarray:
        """Compute true PRS from full genotype data.

        This is identical to ImputationEvaluator._compute_true_prs().
        Sum(dosage * beta) for all PRS variants (observed + missing),
        using true genotypes.
        """

    def _compute_projected_prs_batch(
        self, genotype_data: GenotypeData
    ) -> np.ndarray:
        """Compute projected PRS for all samples.

        For each sample:
        1. Build user_dosages dict from genotype_data row.
        2. Call ProjectionPredictor.predict(user_dosages, apply_calibration=False).
        3. Store result.prs.

        Returns:
            Array of projected PRS values (n_samples,).
        """

    def _get_all_needed_variant_ids(self) -> Set[str]:
        """Get union of observed variant IDs and predictor variant IDs
        across all region models.
        """
```

**Note**: `_compute_true_prs` has the same logic as in `ImputationEvaluator`. Rather than importing and calling the other evaluator's method (which would create a coupling), duplicate the logic. It's ~30 lines of straightforward computation.

### 7.2 Export Changes

**`imputed_prs/evaluation/__init__.py`**: Add `ProjectionEvaluator` to imports and `__all__`.

### 7.3 Tests: `tests/test_projection_evaluator.py`

Follow the pattern of `tests/test_evaluator.py`.

```python
class TestProjectionEvaluator:
    def test_evaluate_returns_evaluation_metrics(self):
        """evaluate() returns EvaluationMetrics type."""

    def test_evaluate_correlation_positive(self):
        """With well-correlated synthetic data, correlation > 0.5."""

    def test_true_prs_computation(self):
        """_compute_true_prs matches manual dot-product calculation."""

    def test_projected_prs_batch_shape(self):
        """_compute_projected_prs_batch returns array of shape (n_samples,)."""

    def test_unfitted_model_raises_error(self):
        """Unfitted model raises ModelNotFittedError."""

    def test_needed_variant_ids(self):
        """_get_all_needed_variant_ids includes observed + all predictor IDs."""
```

### 7.4 Regression Verification

```bash
pytest tests/test_evaluator.py tests/test_calibration.py -v
```

Both must pass unchanged.

---

## 8. Phase 6: Equivalence Tests and Comparison Script

**Goal**: (A) Prove mathematically that the two approaches are equivalent without regularization. (B) Provide a comparison analysis script.

### 8.1 Tests: `tests/test_projection_equivalence.py`

These are integration tests that exercise both `LinearImputationPRS` and `LinearProjectionPRS`.

```python
class TestEquivalenceWithoutRegularization:
    """When regularization is near-zero, projection and imputation give the same PRS.

    Mathematical basis: Without regularization, the per-variant OLS imputation
    weights w_j satisfy: sum_j w_j * beta_j = (Z^T Z)^{-1} Z^T X beta = a.
    So the effective platform weights from imputation equal the projection weights.
    """

    def test_equivalence_single_region(self):
        """Single region: PRS values from both methods match within tolerance.

        Setup:
        - Synthetic data: 200 samples, 30 platform variants, 5 missing PRS variants.
        - All 5 missing variants on same chromosome, close together (1 region).
        - alpha=1e-10, l1_ratio=0 (near-pure Ridge with minimal regularization).
        - Fit both LinearImputationPRS and LinearProjectionPRS.
        - Predict PRS for each sample.
        - Assert: max |PRS_imp - PRS_proj| < 1e-3.
        """

    def test_equivalence_multiple_regions(self):
        """Multiple regions across chromosomes: equivalence holds per-region.

        Setup:
        - Variants on chr1 and chr2 (2 separate regions).
        - Near-zero regularization.
        - Assert: PRS values match.
        """


class TestDivergenceWithRegularization:
    """With meaningful regularization, the two approaches diverge."""

    def test_divergence_moderate_alpha(self):
        """alpha=0.1: PRS values differ between methods.

        Assert: max |PRS_imp - PRS_proj| > 1e-3.
        This confirms the approaches are not trivially identical.
        """


class TestProjectionAdvantage:
    """Cases where projection should outperform imputation."""

    def test_heterogeneous_betas(self):
        """Widely varying betas: projection achieves higher R^2.

        Setup:
        - Some variants with beta ~ 0.01 (tiny), some with beta ~ 1.0 (large).
        - Moderate regularization.
        - The imputation approach wastes regularization on accurately imputing
          variants with tiny betas. The projection approach focuses on PRS accuracy.
        - Fit both, evaluate R^2 on held-out data.
        - Assert: R^2_projection >= R^2_imputation (or very close).

        Note: This test may have a weak assertion (>=) because the advantage
        is data-dependent. The key is demonstrating the mechanism.
        """


class TestSECalibration:
    """Verify that the projection SE is well-calibrated."""

    def test_coverage_95ci(self):
        """~95% of true PRS values fall within projected 95% CIs.

        Setup:
        - Large synthetic dataset (500+ samples).
        - Fit on 80%, predict on 20%.
        - For each test sample: compute PRS and CI.
        - Count fraction of true PRS values within CI.
        - Assert: 0.85 < coverage < 1.0 (allowing for finite-sample variation).
        """
```

### 8.2 New File: `analysis/comparison/compare_imputation_vs_projection.py`

A CLI script for head-to-head comparison on real or synthetic data.

```python
"""Compare linear imputation vs linear projection PRS approaches.

Trains both LinearImputationPRS and LinearProjectionPRS with matched
parameters on the same reference data, evaluates them side-by-side,
and generates comparison visualizations.

Usage:
    python compare_imputation_vs_projection.py \
        --reference-genotypes 1000g_eur.vcf.gz \
        --prs-definition PGS000004 \
        --platform-name 23andme_v5 \
        --output-dir output/comparison

Output:
    output/comparison/
        comparison_summary.json     -- Side-by-side metrics
        scatter_imputation.png      -- Imputed PRS vs true PRS
        scatter_projection.png      -- Projected PRS vs true PRS
        calibration_comparison.png  -- Calibration curves for both methods
        metrics_table.png           -- Bar chart of R^2, correlation, etc.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.evaluation import ImputationEvaluator, ProjectionEvaluator


def run_comparison(args):
    """Run full comparison between imputation and projection approaches.

    Steps:
    1. Train imputation model, record training time.
    2. Train projection model (same params), record training time.
    3. Evaluate both on the same reference data (in-sample evaluation,
       or split into train/test if --test-fraction is provided).
    4. Compute and display metrics for both.
    5. Generate comparison plots.
    6. Write summary JSON.
    """


def main():
    parser = argparse.ArgumentParser(
        description="Compare imputation vs projection PRS approaches"
    )
    parser.add_argument("--reference-genotypes", required=True,
                        help="Path to reference genotype file (VCF/PLINK)")
    parser.add_argument("--prs-definition", required=True,
                        help="PRS definition (PGS ID, e.g., PGS000004, or file path)")
    parser.add_argument("--platform-name", default=None,
                        help="Pre-built platform name (e.g., 23andme_v5)")
    parser.add_argument("--platform-manifest", default=None,
                        help="Path to platform manifest file")
    parser.add_argument("--output-dir", default="output/comparison",
                        help="Output directory for results")
    parser.add_argument("--window-size", type=int, default=1_000_000,
                        help="Genomic window size in bp (default: 1000000)")
    parser.add_argument("--l1-ratio", type=float, default=0.5,
                        help="ElasticNet L1 ratio (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="ElasticNet alpha (default: 0.01)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (default: -1 for all CPUs)")
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
```

**Output `comparison_summary.json` format**:

```json
{
  "imputation": {
    "r2": 0.95,
    "correlation": 0.97,
    "mae": 0.12,
    "rmse": 0.15,
    "calibration_slope": 1.05,
    "attenuation_factor": 0.93,
    "n_models": 200,
    "n_intercept_only": 15,
    "training_time_seconds": 45.2
  },
  "projection": {
    "r2": 0.96,
    "correlation": 0.98,
    "mae": 0.11,
    "rmse": 0.14,
    "calibration_slope": 1.02,
    "attenuation_factor": 0.96,
    "n_regions": 85,
    "n_intercept_only": 3,
    "training_time_seconds": 12.1
  },
  "parameters": {
    "window_size": 1000000,
    "l1_ratio": 0.5,
    "alpha": 0.01,
    "cv_folds": 5,
    "random_state": 42
  }
}
```

### 8.3 Regression Verification

```bash
# Full test suite: everything must pass
pytest tests/ -v
```

---

## 9. Summary Table and Dependency Order

### Files Created Per Phase

| Phase | New Files | Modified Files (additive only) | New Test Files |
|-------|-----------|-------------------------------|----------------|
| 1 | `imputed_prs/core/regions.py` | `core/types.py`, `core/__init__.py` | `tests/test_regions.py` |
| 2 | `imputed_prs/models/projection.py`, `imputed_prs/models/projection_trainer.py` | `models/__init__.py` | `tests/test_projection.py` |
| 3 | `imputed_prs/models/projection_predictor.py` | *(none)* | `tests/test_projection_predictor.py` |
| 4 | `imputed_prs/core/linear_projection_prs.py` | `core/__init__.py`, `__init__.py` | `tests/test_linear_projection_prs.py` |
| 5 | `imputed_prs/evaluation/projection_evaluator.py` | `evaluation/__init__.py` | `tests/test_projection_evaluator.py` |
| 6 | `analysis/comparison/compare_imputation_vs_projection.py` | *(none)* | `tests/test_projection_equivalence.py` |

### Dependency Chain

```
Phase 1 (types + regions)
    |
    v
Phase 2 (per-region fitting)
    |
    v
Phase 3 (prediction)
    |
    v
Phase 4 (main API class)  -- depends on phases 1, 2, 3
    |
    v
Phase 5 (evaluator)  -- depends on phase 4
    |
    v
Phase 6 (equivalence + comparison)  -- depends on phases 4, 5
```

### Regression Tests Per Phase

| Phase | Existing tests that must pass |
|-------|------------------------------|
| 1 | `test_harmonizer.py`, `test_types.py` |
| 2 | `test_trainer.py`, `test_elastic_net.py` |
| 3 | `test_predictor.py` |
| 4 | `test_linear_imputation_prs.py`, `test_api.py` |
| 5 | `test_evaluator.py`, `test_calibration.py` |
| 6 | Full `pytest tests/` |

---

## 10. Key Design Decisions and Rationale

### Region-based decomposition

**Decision**: Merge overlapping per-variant windows into non-overlapping regions. Each region gets one projection model.

**Why**: (1) Captures the structural advantage of projection -- jointly predicting the PRS contribution from multiple correlated missing variants. (2) Reduces to per-variant imputation when regions contain single variants, making the comparison fair. (3) Uses the same window_size parameter for consistency.

### Missing-only target

**Decision**: The projection target `S_R` includes only missing PRS variant contributions. Observed variants use exact dosages separately.

**Why**: Isolates the comparison. Both approaches use the same exact observed-variant component. The only variable is how the missing-variant component is predicted.

### No dosage clipping

**Decision**: The projection output is not clipped to `[0, 2]`.

**Why**: The projection target is a PRS contribution (sum of dosage * beta), which can be negative and has no natural bounds. The bounding module (`bounding.py`) and truncated-normal variance adjustment are irrelevant for the projection approach.

### CV-MSE variance (global SE)

**Decision**: `SE^2 = sum(cv_mse_R)` across regions. This is a global SE (same for all individuals).

**Why**: The projection approach doesn't produce per-variant residual variances, and the truncated-normal framework doesn't apply (no dosage bounds). The CV-MSE is a direct measure of PRS prediction error, which is what we care about. Individual-specific SEs could be added later via bootstrap.

### Code duplication of `fit()` data-loading

**Decision**: Steps 1-8 of `LinearProjectionPRS.fit()` duplicate the data-loading logic from `LinearImputationPRS.fit()` rather than extracting shared helpers.

**Why**: Modifying `LinearImputationPRS.fit()` to call shared helpers introduces regression risk. The duplication is ~200 lines of straightforward I/O code. A future refactor can extract helpers once both classes are stable.

### Population mean substitution for missing predictors

**Decision**: At inference, if a user is missing a platform variant that's a predictor in a region model, substitute `2 * AF` (population mean dosage).

**Why**: This preserves contributions from other available predictors in the region. The imputation approach's all-or-nothing fallback (intercept-only when any predictor is missing) is more conservative but discards partial information. The population mean substitution is equivalent to saying "no information from this predictor" -- the contribution of `a_k * 2*AF_k` is a constant absorbed into the effective intercept.

### Reusing `PredictionResult`

**Decision**: The projection approach returns the same `PredictionResult` dataclass as the imputation approach. The `prs_imputed_component` field stores the projected component.

**Why**: Allows downstream consumers to work with either model type interchangeably. The field semantics are slightly different (imputed dosages * betas vs. direct PRS projection), but the numerical meaning is the same: the PRS contribution from missing variants.
