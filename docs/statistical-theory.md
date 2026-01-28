# Statistical Theory

This document describes the statistical methods implemented in the imputed-prs library for computing Polygenic Risk Scores (PRS) when some variants are missing from a genotyping platform. It emphasizes implementation-specific details that improve upon or differ from naive approaches.

## Table of Contents

1. [Overview](#overview)
2. [The Linear Imputation Model](#the-linear-imputation-model)
3. [Regularization: Elastic Net](#regularization-elastic-net)
4. [Cross-Validated R² and Quality Metrics](#cross-validated-r-and-quality-metrics)
5. [Residual Variance Estimation](#residual-variance-estimation)
6. [Dosage Bounding and Truncation-Adjusted Variance](#dosage-bounding-and-truncation-adjusted-variance)
7. [Intercept-Only Models](#intercept-only-models)
8. [PRS Calculation](#prs-calculation)
9. [Uncertainty Quantification](#uncertainty-quantification)
10. [Internal Calibration via Cross-Validation](#internal-calibration-via-cross-validation)
11. [Inference-Time Behavior](#inference-time-behavior)
12. [Practical Considerations](#practical-considerations)

---

## Overview

A Polygenic Risk Score (PRS) aggregates the effects of many genetic variants into a single predictive score:

$$
S = \sum_j x_j \cdot \beta_j
$$

Where:
- $x_j$ is the dosage (0, 1, or 2 copies of the effect allele) for variant $j$
- $\beta_j$ is the effect size (weight) for variant $j$ from GWAS

**The Problem**: Consumer genotyping platforms (23andMe, AncestryDNA, etc.) typically measure only a subset of variants in a PRS definition. Missing variants contribute zero to the score, causing systematic underestimation and increased variance.

**The Solution**: This library uses linear imputation to predict missing variant dosages from observed (platform) variants, leveraging linkage disequilibrium (LD) patterns in a reference population.

---

## The Linear Imputation Model

For each missing variant $j$, we train a linear model to predict its dosage from nearby observed variants:

$$
\hat{x}_j = \sum_{k \in L_j} z_k \cdot w_{jk} + \gamma_j
$$

Where:
- $\hat{x}_j$ is the predicted dosage for missing variant $j$
- $z_k$ is the observed dosage for platform variant $k$
- $w_{jk}$ is the regression coefficient for predictor $k$
- $\gamma_j$ is the model intercept
- $L_j$ is the local window of predictor variants

### Local Window Constraint

Predictors are restricted to a genomic window around the target variant:

$$
L_j = \{k : |pos_k - pos_j| \leq \text{window\_size} \text{ and } chr_k = chr_j\}
$$

The default window size is 1 Mb (1,000,000 base pairs). This constraint:
- Captures variants in LD with the target
- Reduces overfitting from distant, uncorrelated variants
- Improves computational efficiency

See `imputed_prs/core/harmonizer.py:filter_to_local_window()` for implementation.

---

## Regularization: Elastic Net

The imputation models use Elastic Net regularization, which combines L1 (Lasso) and L2 (Ridge) penalties:

$$
\min_{w, \gamma} \frac{1}{2n} \|Z_j w + \gamma \mathbf{1} - x_j\|^2 + \lambda \left[ \alpha \|w\|_1 + \frac{1-\alpha}{2} \|w\|_2^2 \right]
$$

Where:
- $Z_j$ is the matrix of predictor dosages for samples in the training set
- $x_j$ is the vector of target dosages
- $\lambda$ (alpha) controls overall regularization strength
- $\alpha$ (l1_ratio) controls the L1/L2 balance:
  - $\alpha = 0$: Pure Ridge regression (L2 only)
  - $\alpha = 1$: Pure Lasso regression (L1 only)
  - $0 < \alpha < 1$: Elastic Net (both penalties)

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.01 | Regularization strength $\lambda$ |
| `l1_ratio` | 0.5 | L1/L2 mixing $\alpha$ (balanced) |

### Hyperparameter Tuning

The library supports three tuning strategies via the `tuning_scope` parameter:

| Strategy | Behavior |
|----------|----------|
| `"global"` | Tune once on a subset, apply to all variants |
| `"per_variant"` | Tune separately for each variant (slower) |
| `"none"` | Use provided `alpha` and `l1_ratio` directly |

See `imputed_prs/models/elastic_net.py:fit_single_variant_model()` for implementation.

---

## Cross-Validated R² and Quality Metrics

Imputation quality is assessed using cross-validated $R^2$ (coefficient of determination):

$$
r^2_j = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:
- $SS_{res} = \sum_i (x_{ij} - \hat{x}_{ij})^2$ — residual sum of squares
- $SS_{tot} = \sum_i (x_{ij} - \bar{x}_j)^2$ — total sum of squares
- Predictions are out-of-fold (each sample predicted by a model not trained on it)

### Important: Negative R² Values

**Key implementation detail**: Cross-validated $R^2$ can be negative when predictions are worse than simply predicting the mean. The library:

1. **Stores the original $R^2$ value** (including negative) in `ImputedVariantModel.imputation_r2`
2. **Clips $R^2$ to $[0, 1]$ only for variance calculations** (see next section)

This preserves information about poor-quality imputations while ensuring valid variance estimates.

```python
# From imputed_prs/models/trainer.py
imputation_r2=result.cv_r2,  # Keep original (can be negative)
```

See `imputed_prs/models/metrics.py:compute_cv_r2()` for implementation.

---

## Residual Variance Estimation

The residual variance quantifies uncertainty in imputed dosages:

$$
\sigma^2_j = 2 \cdot q_j \cdot (1 - q_j) \cdot (1 - r^2_{clipped})
$$

Where:
- $q_j$ is the allele frequency of variant $j$ (0 to 1)
- $r^2_{clipped} = \max(0, \min(1, r^2_j))$ — $R^2$ clipped to valid range
- $2 \cdot q_j \cdot (1 - q_j)$ is the variance of dosage under Hardy-Weinberg equilibrium

### Why Clip R²?

- **Negative $R^2$**: Would produce variance > HWE variance (impossible)
- **$R^2 > 1$**: Could produce negative variance (impossible)

Clipping ensures physically meaningful variance estimates.

```python
# From imputed_prs/models/trainer.py:compute_residual_variance()
r2_clipped = max(0.0, min(1.0, r2))
return 2.0 * q * (1.0 - q) * (1.0 - r2_clipped)
```

---

## Dosage Bounding and Truncation-Adjusted Variance

Genotype dosages must lie in $[0, 2]$. When imputed predictions fall outside this range, they are clipped. **This clipping reduces variance** — a key implementation improvement over naive approaches.

### The Problem with Simple Clipping

If we simply clip predictions and use the original residual variance:
- We overestimate uncertainty for predictions near boundaries
- Confidence intervals may extend outside valid dosage range

### Truncated Normal Distribution Theory

The library models predictions as normally distributed and computes the **variance of the truncated distribution**:

$$
\text{Var}(X \mid 0 \leq X \leq 2) = \sigma^2 \left[ 1 + \frac{\alpha \cdot \phi(\alpha) - \beta \cdot \phi(\beta)}{Z} - \left( \frac{\phi(\alpha) - \phi(\beta)}{Z} \right)^2 \right]
$$

Where:
- $\alpha = \frac{0 - \mu}{\sigma}$ — standardized lower bound
- $\beta = \frac{2 - \mu}{\sigma}$ — standardized upper bound
- $\phi(\cdot)$ is the standard normal PDF
- $\Phi(\cdot)$ is the standard normal CDF
- $Z = \Phi(\beta) - \Phi(\alpha)$ — probability mass within bounds

### Variance Reduction Properties

| Prediction Location | Variance Adjustment |
|---------------------|---------------------|
| Well within $[0, 2]$ (e.g., $\mu = 1.0$) | Nearly unchanged |
| Near boundary (e.g., $\mu = 0.1$) | Moderately reduced |
| Outside bounds (e.g., $\mu = -0.2$) | Significantly reduced |

### Implementation

```python
# From imputed_prs/models/bounding.py:clip_and_adjust_variance()
clipped = max(lower, min(upper, raw_prediction))
sigma = np.sqrt(residual_variance)
adjusted_variance = truncated_normal_variance(raw_prediction, sigma, lower, upper)
return clipped, adjusted_variance
```

This is called during prediction for every imputed variant:

```python
# From imputed_prs/models/predictor.py:compute_imputed_prs()
clipped_dosage, adjusted_variance = clip_and_adjust_variance(
    raw_prediction, model.residual_variance
)
```

---

## Intercept-Only Models

An intercept-only model predicts the population mean for all individuals (no predictor information used). These are created under four conditions:

| Condition | Rationale |
|-----------|-----------|
| **No predictors in window** | No variants within $\pm 1$ Mb on same chromosome |
| **Too few samples** | Fewer valid samples than CV folds (default: $< 5$) |
| **Zero variance in target** | Target variant is monomorphic in reference |
| **All coefficients regularized to zero** | Elastic Net shrinks all weights to zero |

### Detection

```python
# From imputed_prs/models/elastic_net.py

# Condition 1: No predictors
if n_predictors == 0:
    return _fit_intercept_only_model(...)

# Condition 2: Too few samples
if n_valid < cv_folds:
    return _fit_intercept_only_model(...)

# Condition 3: Zero variance
if np.std(y_valid) < 1e-10:
    return _fit_intercept_only_model(...)

# Condition 4: All coefficients zero (after fitting)
is_intercept_only = np.allclose(final_model.coef_, 0, atol=1e-10)
```

### Properties

- **$R^2 = 0$**: Predicting the mean explains no variance
- **Residual variance** = full HWE variance: $2 \cdot q \cdot (1 - q)$
- **Intercept** = mean dosage = $2q$

---

## PRS Calculation

The total PRS combines observed and imputed components:

$$
S = S_{observed} + S_{imputed}
$$

### Observed Component

For variants directly measured on the platform:

$$
S_{observed} = \sum_{j \in O} z_j \cdot \beta_j
$$

Where $O$ is the set of observed (platform) variants in the PRS.

### Imputed Component

For variants not on the platform:

$$
S_{imputed} = \sum_{j \in M} \hat{x}_j \cdot \beta_j
$$

Where $M$ is the set of missing (imputed) variants.

See `imputed_prs/models/predictor.py:compute_observed_prs()` and `compute_imputed_prs()`.

---

## Uncertainty Quantification

The library provides standard errors and confidence intervals for PRS predictions.

### Standard Error

$$
SE(S_{imputed}) = \sqrt{\sum_{j \in M} \beta_j^2 \cdot \sigma^2_{adjusted,j}}
$$

Where:
- $\beta_j$ is the effect size for variant $j$
- $\sigma^2_{adjusted,j}$ is the **truncation-adjusted** residual variance

**Note**: Observed variants contribute zero variance (they are measured exactly).

### 95% Confidence Interval

$$
CI = \left[ S - 1.96 \cdot SE, \; S + 1.96 \cdot SE \right]
$$

This assumes approximate normality of the PRS, which is reasonable when many variants contribute.

### Implementation

```python
# From imputed_prs/models/predictor.py:PRSPredictor.predict()

# Accumulate variance
total_variance += (model.beta ** 2) * adjusted_variance

# Compute SE
se = np.sqrt(total_variance) if total_variance > 0 else 0.0

# Compute CI
ci_lower = prs_raw - 1.96 * se
ci_upper = prs_raw + 1.96 * se
```

---

## Internal Calibration via Cross-Validation

Imputation attenuates PRS predictions (reduces variance). The library estimates calibration parameters to correct for this using internal cross-validation.

### Regression Dilution

When predictors are measured with error (or imputed), regression coefficients are biased toward zero. For PRS:

$$
\mathbb{E}[S_{imputed}] \approx a + b \cdot S_{true} \quad \text{where } b < 1
$$

### Calibration Regression

During training, the library:
1. Computes **CV-predicted PRS** using out-of-fold imputation predictions
2. Computes **true PRS** using actual genotypes
3. Regresses true on CV-predicted to estimate scaling parameters

$$
S_{true} = \alpha + \beta \cdot S_{cv}
$$

### Applying Calibration

At prediction time:

$$
S_{scaled} = \alpha + \beta \cdot S_{raw}
$$

$$
SE_{scaled} = |\beta| \cdot SE
$$

$$
CI_{scaled} = \left[ S_{scaled} - 1.96 \cdot SE_{scaled}, \; S_{scaled} + 1.96 \cdot SE_{scaled} \right]
$$

### Calibration Parameters

| Parameter | Description |
|-----------|-------------|
| `scaling_factor` | Slope $\beta$ from calibration regression |
| `scaling_factor_se` | Standard error of slope |
| `calibration_intercept` | Intercept $\alpha$ |
| `calibration_r2` | $R^2$ of calibration fit |
| `attenuation_factor` | $SD(S_{cv}) / SD(S_{true})$, measures variance loss |

See `imputed_prs/evaluation/calibration.py:estimate_cv_calibration()` for implementation.

---

## Inference-Time Behavior

### Missing Predictor Fallback

At inference time, some predictor variants may be missing from user data. When this occurs:

```python
# From imputed_prs/models/predictor.py:compute_imputed_prs()

if not all_predictors_available:
    # Fall back to intercept-only (mean imputation)
    raw_prediction = model.intercept
```

**Behavior**: The model falls back to predicting the population mean for that variant.

**Implications**:
- The imputed dosage equals $2q$ (twice the allele frequency)
- Full residual variance is used (no variance reduction from predictors)
- This is conservative — uncertainty is maximized when information is missing

### Robust to Partial Data

This fallback mechanism means predictions can be made even when:
- User is missing some platform variants
- Platform manifest differs slightly from training

---

## Practical Considerations

### Interpreting Imputation R²

| $R^2$ Range | Quality | Interpretation |
|-------------|---------|----------------|
| $> 0.8$ | Excellent | Highly accurate imputation |
| $0.6 - 0.8$ | Good | Reliable for most applications |
| $0.4 - 0.6$ | Moderate | Use with caution |
| $0.0 - 0.4$ | Poor | High uncertainty |
| $< 0$ | Very Poor | Worse than mean; treated as intercept-only for variance |

### When to Expect Good Imputation

- **High LD** between missing and observed variants
- **Dense genotyping platform** with many nearby markers
- **Common variants** (higher allele frequency = more information in reference)
- **Large reference panel** (more samples = better LD estimation)

### When to Expect Poor Imputation

- **Rare variants** (low allele frequency)
- **Sparse regions** with few platform variants
- **Population mismatch** between reference and target population
- **Structural variation regions** with complex LD patterns

### Calibration Importance

Always apply calibration when:
- Comparing scores across individuals
- Using scores for risk stratification
- Interpreting scores relative to a reference population

The raw (uncalibrated) PRS underestimates variance, which can:
- Compress the score distribution
- Underestimate tail risks
- Affect percentile assignments

### Standard Error Interpretation

The SE reflects uncertainty from imputation only. It does not include:
- Uncertainty in GWAS effect sizes ($\beta_j$)
- Population stratification effects
- Genotyping errors in observed variants

For complete uncertainty quantification, consider these additional sources.

---

## Summary of Key Implementation Details

| Feature | Implementation Behavior |
|---------|------------------------|
| **Negative $R^2$ handling** | Stored as-is in model; clipped to $[0, 1]$ only for variance calculation |
| **Truncation-adjusted variance** | Uses truncated normal distribution theory, not simple clipping |
| **Missing predictor fallback** | Falls back to intercept-only (mean) prediction at inference |
| **Intercept-only creation** | 4 conditions: no predictors, too few samples, zero variance, all coefficients zero |
| **Calibration** | Internal CV-based regression to correct for attenuation |
| **Window constraint** | Local (default 1 Mb) to capture LD while avoiding overfitting |

---

## References

For implementation details, see:

| Component | Source File |
|-----------|-------------|
| Truncation-adjusted variance | `imputed_prs/models/bounding.py` |
| Residual variance, $R^2$ clipping | `imputed_prs/models/trainer.py` |
| PRS calculation, SE, fallback | `imputed_prs/models/predictor.py` |
| Intercept-only conditions | `imputed_prs/models/elastic_net.py` |
| CV $R^2$ computation | `imputed_prs/models/metrics.py` |
| Calibration estimation | `imputed_prs/evaluation/calibration.py` |

For data type definitions, see the [API Reference](API.md#data-types-reference).
