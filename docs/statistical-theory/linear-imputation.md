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

### Predictor Standardization

Before each ElasticNet fit, the predictor columns are **standardized to zero mean and
unit variance**, then the fitted coefficients are back-transformed to the raw-dosage
scale. This matters because the ElasticNet penalty is not scale-free: on raw dosages,
the effective penalty on a coefficient grows with its column's variance (i.e. with MAF
for genotype dosages), so common and rare predictors would be penalized unequally.
Standardizing makes the penalty comparable across predictors.

Standardization statistics are computed from the **training fold only** inside the CV
loop (no leakage from the validation fold), and again on all valid samples for the
final fit. The final model fitted on standardized columns,
$y = w^{std} \cdot \tfrac{x - \mu}{s} + \gamma^{std}$, is mapped back by the algebraic
identity

$$
w_{jk} = \frac{w^{std}_{jk}}{s_k},
\qquad
\gamma_j = \gamma^{std}_j - \sum_k w^{std}_{jk} \, \frac{\mu_k}{s_k},
$$

so the stored $w_{jk}, \gamma_j$ reproduce the same predictions on **raw** dosages.
Storage, export, and inference ($z^\top w + \gamma$ on raw dosages) are therefore
unchanged — standardization is purely a fitting-time device. Near-constant columns are
left unscaled ($s_k = 1$) so the back-transform never divides by $\approx 0$.

See `imputed_prs/models/elastic_net.py:fit_single_variant_model()` (~line 163) and
`imputed_prs/models/metrics.py:standardize_columns()` /
`backtransform_linear_model()` for implementation.

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

The library provides standard errors and confidence intervals for PRS predictions. The
reported interval is built from an **empirical, score-level approximation error**; the
per-variant sum-of-variances below is retained only as an optimistic lower bound.

### The Diagonal SE (a lower bound, not the reported interval)

The natural per-variant standard error sums each imputed variant's variance:

$$
SE_{diag}(S_{imputed}) = \sqrt{\sum_{j \in M} \beta_j^2 \cdot \sigma^2_{eff,j}}
$$

Where $\beta_j$ is the effect size and $\sigma^2_{eff,j}$ is the **missingness-aware**
residual variance for variant $j$ (the truncation-adjusted residual variance, further
inflated when predictors are missing at inference — see
[Inference-Time Behavior](#inference-time-behavior)). Observed variants that are
measured exactly contribute zero; observed variants recovered via their fallback model
contribute their fallback variance ([below](#observed-variant-fallback)).

This is a **diagonal** quantity: it treats the per-variant errors as independent and so
ignores the off-diagonal terms of the residual covariance $\beta^\top \Sigma \beta$.
Because imputed dosages in LD share correlated errors, $SE_{diag}$ is **optimistically
small** — an LD-blind lower bound, not an honest interval. It is preserved in
`PredictionResult.se_diagonal_lower_bound` (a per-user value, inflated by missingness)
and, at fit time, in `CalibrationParams.diagonal_model_se_lower_bound` (the full-data,
no-missingness value).

### The Reported (Empirical) SE

When the model carries an empirical residual SD (computed during calibration; see
[README](README.md#calibration-regression)), the reported standard error is the
**maximum** of the empirical SD and the per-user diagonal lower bound:

$$
SE = \max\!\left(\,\widehat{\sigma}^{\,raw}_{err},\; SE_{diag}\,\right),
\qquad
\widehat{\sigma}^{\,raw}_{err} = \operatorname{sd}\!\left(S_{true} - S_{cv}\right)
$$

where $S_{cv}$ is the out-of-fold CV-predicted PRS and $S_{true}$ the gold-standard
PRS, over the reference panel. Because $S_{cv}$ and the deployed raw score $S_{raw}$
are built identically (exact observed dosages plus CV/model-predicted imputed terms),
$\widehat{\sigma}^{\,raw}_{err}$ is the honest *expected approximation error* of the
raw score — it captures the full LD-aware residual covariance. The empirical SD is the
baseline for a typical, fully-observed upload; the diagonal $SE_{diag}$ becomes the
binding floor only when a particular upload is missing enough predictors that its
inflated diagonal exceeds that baseline.

For uncalibrated or pre-empirical-residual artifacts (no stored
$\widehat{\sigma}^{\,raw}_{err}$), the library falls back to $SE = SE_{diag}$.

### Calibrated Interval

When calibration is applied, the half-width uses the empirical *post-calibration*
residual SD, floored by the slope-scaled diagonal:

$$
SE_{scaled} = \max\!\left(\,\widehat{\sigma}^{\,cal}_{err},\; |b| \cdot SE_{diag}\,\right),
\qquad
\widehat{\sigma}^{\,cal}_{err} = \operatorname{sd}\!\left(S_{true} - (a + b\,S_{cv})\right)
$$

### 95% Confidence Interval

$$
CI = \left[ S - 1.96 \cdot SE, \; S + 1.96 \cdot SE \right]
$$

This assumes approximate normality of the PRS, reasonable when many variants contribute.

### Implementation

```python
# From imputed_prs/models/predictor.py:PRSPredictor.predict()

# Per-user diagonal lower bound (P3.3 missingness inflation already folded in)
se_diagonal = np.sqrt(total_variance) if total_variance > 0 else 0.0

# Reported SE: empirical residual SD floored by the diagonal lower bound
use_empirical = params is not None and params.raw_empirical_residual_sd is not None
se = max(params.raw_empirical_residual_sd, se_diagonal) if use_empirical else se_diagonal

ci_lower = prs_raw - 1.96 * se
ci_upper = prs_raw + 1.96 * se
```

See `imputed_prs/models/predictor.py:PRSPredictor.predict()` (lines ~589-638) for
implementation and `imputed_prs/evaluation/calibration.py:estimate_cv_calibration()`
for where the empirical residual SDs are estimated.

---

## Inference-Time Behavior

### Missing-Predictor Mean Substitution

At inference time, some of a model's predictor variants may be missing from (or
unresolvable in) the user's upload. The deployed, allele-aware scoring path handles
this **per predictor**, not all-or-nothing: each missing predictor $k$ is substituted
with its population mean dosage $2 q_k$ (where $q_k$ is the counted-allele frequency
stored from training), and the model's remaining, available predictors still
contribute. The point estimate is therefore

$$
\hat{x}_j = \sum_{k \in L_j,\, \text{available}} z_k \, w_{jk}
          + \sum_{k \in L_j,\, \text{missing}} 2 q_k \, w_{jk}
          + \gamma_j ,
$$

then clipped to $[0, 2]$ exactly as a fully-observed prediction. Mean-substituting a
predictor at its expectation does not move the point estimate in expectation; it does
increase uncertainty, which is reflected in the variance (below). This is what
`_predict_model_dosage` does, and it brings imputation in line with the projection
scorer.

> **Note on the legacy path.** The deprecated allele-blind dosage-dict scorer
> (`compute_imputed_prs`) still collapses the *whole* model to its intercept if **any**
> single predictor is missing — the $f = 1$ endpoint of the inflation formula below. The
> per-predictor behavior described here is the canonical upload path
> (`compute_imputed_prs_oriented` / `_predict_model_dosage`).

### Missingness-Aware Variance Inflation

The residual variance reported for a partially-observed prediction is interpolated from
the full-model value toward the intercept-only Hardy-Weinberg variance, in proportion
to the fraction of predictors that were substituted:

$$
\sigma^2_{eff} = \sigma^2_{full} \cdot (1 - f) + 2 q (1 - q) \cdot f,
\qquad
f = \frac{n_{substituted}}{n_{predictors}}
$$

With no predictors (an intercept-only model) or none substituted ($f = 0$),
$\sigma^2_{eff} = \sigma^2_{full}$, so a fully-observed prediction keeps its full-model
confidence. At $f = 1$ (every predictor missing) it equals the intercept-only variance
$2 q (1 - q)$. Because the trainer sets $\sigma^2_{full} = 2 q (1 - q)(1 - r^2)$ with
$r^2 \in [0, 1]$, the Hardy-Weinberg term is always $\geq \sigma^2_{full}$, so
$\sigma^2_{eff}$ grows monotonically with the substituted fraction.

```python
# From imputed_prs/models/predictor.py:_effective_residual_variance()
f = n_substituted / n_pred
intercept_only_variance = hardy_weinberg_variance(model.allele_frequency)  # 2q(1-q)
return model.residual_variance * (1.0 - f) + intercept_only_variance * f
```

See `imputed_prs/models/predictor.py:_effective_residual_variance()` (line ~62) and
`compute_imputed_prs_oriented()` for implementation.

### Observed-Variant Fallback

Every observed (on-platform) PRS variant additionally carries its own **per-variant
fallback imputation model** (`VariantInfo.fallback`), trained to predict the variant's
*effect-allele* dosage from its local-window platform predictors (excluding its own
locus). At scoring time each observed variant is handled **direct-effect-dosage-else-
fallback**: if the upload resolves and calls the locus, it is scored exactly from the
counted effect-allele dosage; only if that fails (locus not found, duplicate conflict,
missing other allele, palindromic-blocked, partial overlap, or no-call) is the fallback
model consulted, scored *identically* to a genuine imputation (same mean-substitution
and truncation-adjusted variance). The fallback's contribution carries variance into
the score's total SE (fallback dosages are imputed, not exact). The upshot: an observed
PRS variant is **never silently dropped** — it is recovered via fallback or, if no
fallback was trainable, surfaced in `PredictionResult.unresolved_observed_ids`.

```python
# From imputed_prs/models/predictor.py:compute_observed_prs_oriented()
if direct_dosage is not None:
    total += direct_dosage * variant.beta            # exact, allele-oriented
elif variant.fallback is not None:
    clipped_dosage, adjusted_variance, _ = _predict_model_dosage(variant.fallback, ...)
    total += clipped_dosage * variant.beta           # recovered via fallback
    fallback_variance += variant.beta ** 2 * adjusted_variance
else:
    unresolved.append(variant.variant_id)            # surfaced, never dropped
```

### Robust to Partial Data

Together these mechanisms mean predictions can be made even when the user is missing
some platform variants or the platform manifest differs slightly from training, while
the reported uncertainty honestly grows with how much had to be substituted or
recovered.

See `imputed_prs/models/predictor.py` (`compute_imputed_prs_oriented`,
`compute_observed_prs_oriented`, `_predict_model_dosage`) for implementation.

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

The reported SE is the empirical, score-level approximation error of the imputation/
fallback path against the full computed PRS (floored by the LD-blind diagonal lower
bound). It reflects uncertainty from imputation only. It does **not** include:
- Uncertainty in GWAS effect sizes ($\beta_j$)
- Population stratification effects
- Genotyping errors in observed variants
- Transfer error to a different ancestry (see [Validation](validation.md))

For complete uncertainty quantification, consider these additional sources.

---

## Implementation References

| Concept | Source File |
|---------|-------------|
| Linear imputation model, intercept-only conditions, predictor standardization | `imputed_prs/models/elastic_net.py` |
| Standardization / back-transform helpers, CV $R^2$ computation | `imputed_prs/models/metrics.py` |
| Residual variance, $R^2$ clipping | `imputed_prs/models/trainer.py` |
| Truncation-adjusted variance | `imputed_prs/models/bounding.py` |
| PRS calc, empirical SE, missing-predictor mean substitution, observed-variant fallback | `imputed_prs/models/predictor.py` |
| Empirical residual SDs (calibration) | `imputed_prs/evaluation/calibration.py` |
