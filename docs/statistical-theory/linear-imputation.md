# Linear Imputation Method

> See [README.md](README.md) for shared notation, regularization, and calibration theory.

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
L_j = \{k : |pos_k - pos_j| \leq W \text{ and } chr_k = chr_j\}
$$

where the default window size is $W = 1\text{ Mb}$ (1,000,000 base pairs). This constraint:
- Captures variants in LD with the target
- Reduces overfitting from distant, uncorrelated variants
- Improves computational efficiency

See `imputed_prs/core/harmonizer.py:filter_to_local_window()` for implementation.

---

## Cross-Validated R-squared and Quality Metrics

Imputation quality is assessed using cross-validated $R^2$ (coefficient of determination):

$$
r^2_j = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:
- $SS_{res} = \sum_i (x_{ij} - \hat{x}_{ij})^2$ — residual sum of squares
- $SS_{tot} = \sum_i (x_{ij} - \bar{x}_j)^2$ — total sum of squares
- Predictions are out-of-fold (each sample predicted by a model not trained on it)

### Important: Negative R-squared Values

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

### Why Clip R-squared?

- **Negative $R^2$**: Would produce variance > HWE variance (impossible)
- **$R^2 > 1$**: Could produce negative variance (impossible)

Clipping ensures physically meaningful variance estimates.

```python
# From imputed_prs/models/trainer.py:compute_residual_variance()
r2_clipped = max(0.0, min(1.0, r2))
return 2.0 * q * (1.0 - q) * (1.0 - r2_clipped)
```

See `imputed_prs/models/trainer.py:compute_residual_variance()` for implementation.

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
\text{Var}(X \mid 0 \leq X \leq 2) = \sigma^2 \left[ 1 + \frac{\alpha_l \cdot \phi(\alpha_l) - \alpha_u \cdot \phi(\alpha_u)}{\Phi(\alpha_u) - \Phi(\alpha_l)} - \left( \frac{\phi(\alpha_l) - \phi(\alpha_u)}{\Phi(\alpha_u) - \Phi(\alpha_l)} \right)^2 \right]
$$

Where:
- $\alpha_l = \frac{0 - \mu}{\sigma}$ — standardized lower bound
- $\alpha_u = \frac{2 - \mu}{\sigma}$ — standardized upper bound
- $\phi(\cdot)$ is the standard normal PDF
- $\Phi(\cdot)$ is the standard normal CDF
- $\Phi(b) - \Phi(a)$ is the probability mass within bounds

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

See `imputed_prs/models/bounding.py:clip_and_adjust_variance()` for implementation.

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

See `imputed_prs/models/elastic_net.py` for implementation.

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

See `imputed_prs/models/predictor.py` for implementation.

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

See `imputed_prs/models/predictor.py:PRSPredictor.predict()` for implementation.

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

See `imputed_prs/models/predictor.py:compute_imputed_prs()` for implementation.

---

## Practical Considerations

### Interpreting Imputation R-squared

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

### Standard Error Interpretation

The SE reflects uncertainty from imputation only. It does not include:
- Uncertainty in GWAS effect sizes ($\beta_j$)
- Population stratification effects
- Genotyping errors in observed variants

For complete uncertainty quantification, consider these additional sources.

---

## Implementation References

| Concept | Source File |
|---------|-------------|
| Linear imputation model, intercept-only conditions | `imputed_prs/models/elastic_net.py` |
| Residual variance, $R^2$ clipping | `imputed_prs/models/trainer.py` |
| Truncation-adjusted variance | `imputed_prs/models/bounding.py` |
| PRS calculation, SE, missing-predictor fallback | `imputed_prs/models/predictor.py` |
| CV $R^2$ computation | `imputed_prs/models/metrics.py` |
