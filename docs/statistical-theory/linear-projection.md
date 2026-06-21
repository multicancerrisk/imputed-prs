# Linear Projection Method

> See [README.md](README.md) for shared notation, regularization, and calibration theory.

---

## Overview

The projection approach directly learns platform-variant weights to approximate each region's PRS contribution, rather than imputing individual missing variant dosages. Instead of solving $|M|$ independent per-variant problems, it solves one problem per genomic region, optimizing for PRS prediction accuracy directly.

---

## Region Decomposition

For each missing PRS variant $j$, define a window $[pos_j - W, pos_j + W]$ (default $W = 1$ Mb, same as imputation). Overlapping windows on the same chromosome merge into a single region $R$.

### Merge Algorithm

1. Compute per-variant windows, clamp start to $\geq 0$
2. Group by chromosome
3. Sort by start position within each chromosome
4. Sweep-line merge: extend current region if overlap, else start new

### Region Contents

For each region $R$:

- **Missing PRS variants**: those whose windows contributed to $R$
- **Predictors**: platform variants within $R$'s boundaries on the same chromosome

Single-variant regions reduce to the imputation case (modulo regularization path).

See `imputed_prs/core/regions.py:merge_variant_windows()` for implementation.

---

## The Linear Projection Model

For region $R$ containing missing PRS variants with effect sizes $\beta_R$:

**Target**: $S_R = X_R \beta_R$ -- the regional PRS contribution from missing variants. This is a continuous scalar per sample (can be negative, unlike dosages).

**Model**:

$$
\hat{S}_R = z_R^T a_R + \gamma_R
$$

Where:
- $z_R$ is the vector of platform variant dosages in $R$
- $a_R$ is the learned projection weight vector
- $\gamma_R$ is the intercept

The ElasticNet objective is the same as for imputation (see README) but with $S_R$ as the target instead of $x_j$. No dosage bounds $[0, 2]$ apply to $S_R$.

### Predictor Standardization

Exactly as in imputation, the region's predictor columns are **standardized to zero
mean and unit variance** before each ElasticNet fit (training-fold statistics inside
the CV loop, all-samples statistics for the final fit), so the L1/L2 penalty is
scale-free across predictors of differing MAF. The fitted coefficients are then
back-transformed to the raw-dosage scale by the algebraic identity
$a_{R,k} = a^{std}_{R,k} / s_k$, $\gamma_R = \gamma^{std}_R - \sum_k a^{std}_{R,k}\,\mu_k/s_k$,
so storage, export, and inference ($z_R^\top a_R + \gamma_R$ on raw dosages) are
unchanged. See [Linear Imputation](linear-imputation.md#predictor-standardization) for
the full derivation; the projection code path uses the same
`standardize_columns` / `backtransform_linear_model` helpers.

See `imputed_prs/models/projection.py:fit_single_region_model()` (~line 196) for implementation.

---

## Equivalence with Imputation (OLS, No Regularization)

**Claim**: Under OLS with all platform variants as predictors and no regularization, the two approaches produce identical PRS predictions.

**Proof**: Let $P_Z = Z(Z^T Z)^{-1} Z^T$ be the projection matrix onto the column space of $Z$.

**Projection approach**:

$$
\hat{S}_{proj} = P_Z \cdot S_{true} = P_Z(X_O \beta_O + X_M \beta_M)
$$

Since observed PRS variants are on the platform, their columns lie in the column space of $Z$, so $P_Z X_O = X_O$:

$$
\hat{S}_{proj} = X_O \beta_O + P_Z X_M \beta_M
$$

**Imputation approach**: Each missing variant's OLS imputation is $\hat{x}_j = P_Z x_j$, so $\hat{X}_M = P_Z X_M$:

$$
\hat{S}_{imp} = X_O \beta_O + \hat{X}_M \beta_M = X_O \beta_O + P_Z X_M \beta_M = \hat{S}_{proj}
$$

---

## Where the Approaches Diverge

With elastic net regularization, the two approaches optimize different objectives:

- **Imputation** solves $|M|$ independent problems, each minimizing per-variant prediction error. The regularization is agnostic to $\beta_j$ -- it does not know which variants matter most for the PRS.

- **Projection** solves one problem per region minimizing PRS prediction error directly. The regularization is informed by $\beta_j$ -- a variant with a tiny beta contributes little to PRS error, so the model will not waste regularization budget on it.

**Consequence**: With heterogeneous effect sizes, the projection approach can allocate model capacity more efficiently, focusing on the variants that matter most for PRS accuracy. With homogeneous betas, both approaches are roughly equivalent.

---

## Intercept-Only Regions

An intercept-only region model is created under the same four conditions as imputation intercept-only models:

| Condition | Rationale |
|-----------|-----------|
| **No platform predictors in region** | No variants within region boundaries on same chromosome |
| **Too few valid samples** | Fewer valid samples than CV folds (default: $< 5$) |
| **Zero variance in target** | $S_R$ is constant across samples (degenerate region) |
| **All coefficients regularized to zero** | ElasticNet shrinks all weights to zero |

### Properties

- **Intercept** $= \bar{S}_R$ (mean of regional PRS contribution across training samples)
- **$R^2 = 0$**

Note the intercept is $\bar{S}_R$ (mean of weighted PRS contribution), not $2q$ as in the imputation case (where the intercept is the mean dosage).

See `imputed_prs/models/projection.py:_fit_intercept_only_region()` for implementation.

---

## PRS Calculation

The total PRS combines observed and projected components:

$$
S = S_{observed} + \sum_{R} \hat{S}_R
$$

Where:
- $S_{observed} = \sum_{j \in O} z_j \beta_j$ (same as imputation)
- $\hat{S}_R = z_R^T a_R + \gamma_R$ for each region

No dosage clipping is needed because the target $S_R$ is a PRS contribution (continuous, possibly negative), not a dosage bounded in $[0, 2]$. This eliminates the need for truncated normal variance adjustments.

See `imputed_prs/models/projection_predictor.py:compute_projected_prs()` for implementation.

---

## Allele Orientation and Model Persistence

### Per-PRS-Variant Allele Metadata

A region model stores not only its predictors' allele metadata but also, for each PRS
variant it covers, the variant's **position, effect allele, and other allele**
(`ProjectionRegionModel.prs_positions`, `prs_effect_alleles`, `prs_other_alleles`,
index-aligned with `prs_variant_ids` / `betas`). These let a standalone scorer
recompute the gold-standard regional contribution without assuming "effect allele $=$
ALT at the first reference row" — it can orient correctly when the effect allele is the
reference allele, on the complementary strand, or at a multiallelic locus.

Concretely, the evaluator's true-PRS computation
(`projection_evaluator.py:_compute_true_prs`, line ~174) orients **every** PRS
variant — observed and region (missing) alike — through
`harmonizer.match_oriented_dosage`, which resolves the locus on `chr:pos`, matches the
PRS `(effect, other)` pair against each candidate reference row's `(ref, alt)`
(directly and via the complementary strand), and returns the **effect-allele-oriented**
dosage (flipping to $2 - \text{dosage}$ when the effect allele is the reference allele,
since the stored per-row dosage counts the ALT allele). This brings projection's true
PRS to parity with the imputation evaluator.

### Allele-Orientation Primitives

Two role-aware primitives keep "which allele is counted" explicit throughout scoring:

- **`match_oriented_dosage`** (training/eval, numeric reference matrix) — counts copies
  of the **effect allele** for a PRS term.
- **`count_allele`** (browser/upload, raw genotype strings) — counts copies of a named
  allele: the **effect allele** for an observed PRS term, but the **ALT allele** for a
  *predictor* (because each predictor coefficient was fitted against the ALT-counted
  reference $Z$ column). A genotype is only counted when its alleles are a subset of the
  declared `{counted, other}` pair; a partial overlap is left unresolved rather than
  silently scored.

This effect-vs-ALT distinction is why predictor metadata carries
`predictor_counted_alleles` (the ALT allele backing each $Z$ column) separately from
the PRS variants' effect alleles.

### Export and Load (JSON)

Projection models are persistable: `LinearProjectionPRS.export(formats=["json"])`
writes a browser-deployable JSON artifact (JSON is the only supported projection
format; HDF5/Arrow/CSV remain imputation-only), and `LinearProjectionPRS.load(path)`
reconstructs a fitted model from a `.json` file. The round-trip restores the region
models including all predictor and PRS-variant allele metadata above, so a loaded model
scores identically to the freshly-fit one.

See `imputed_prs/evaluation/projection_evaluator.py:_compute_true_prs()`,
`imputed_prs/core/harmonizer.py:match_oriented_dosage()`,
`imputed_prs/io/user_genotypes.py:count_allele()`, and
`imputed_prs/core/linear_projection_prs.py` (`export` / `load` / `_load_from_json`)
for implementation.

---

## Uncertainty Quantification

As with imputation, the reported interval is an **empirical, score-level approximation
error**; the per-region sum of CV-MSEs is retained only as an optimistic lower bound.

### The Diagonal SE (a lower bound, not the reported interval)

The natural per-region standard error sums each region's cross-validated MSE:

$$
SE_{diag} = \sqrt{\sum_R \text{cv\\_mse}_{eff,R}}
$$

Here $\text{cv\\_mse}_{eff,R}$ is the region's **missingness-aware** error variance: the
fitted $\text{cv\\_mse}_R$ when all predictors are present, interpolated toward the
intercept-only error variance as predictors go missing (see
[Inference-Time Behavior](#inference-time-behavior)). Unlike imputation, no per-variant
$\beta_j^2 \sigma^2_j$ composition is needed: the training target $S_R = X_R \beta_R$
already incorporates the effect sizes, so the CV-MSE measures PRS prediction error
directly, not dosage prediction error.

This sum is still a **diagonal** quantity — it treats the per-region errors as
independent and so omits the off-diagonal residual covariance between regions in LD.
It is therefore an LD-blind lower bound, preserved in
`PredictionResult.se_diagonal_lower_bound` (per-user, missingness-inflated) and, at fit
time, in `CalibrationParams.diagonal_model_se_lower_bound` (full-data
$\sqrt{\sum_R \text{cv\\_mse}_R}$).

### The Reported (Empirical) SE

When the model carries an empirical residual SD (computed during calibration; see
[README](README.md#calibration-regression)), the reported standard error is the
**maximum** of the empirical SD and the per-user diagonal lower bound:

$$
SE = \max\!\left(\,\widehat{\sigma}^{\,raw}_{err},\; SE_{diag}\,\right),
\qquad
\widehat{\sigma}^{\,raw}_{err} = \operatorname{sd}\!\left(S_{true} - S_{cv}\right)
$$

where $S_{cv}$ is the out-of-fold CV-predicted PRS (exact observed terms plus
$\sum_R \hat{S}_R^{(-f)}$) and $S_{true}$ the gold-standard PRS, over the reference
panel. Because $S_{cv}$ and the deployed raw score are built identically, this is the
honest *expected approximation error* of the projected score and captures the full
LD-aware residual covariance the diagonal omits. The empirical SD is the baseline for a
typical, fully-observed upload; $SE_{diag}$ binds only when a particular upload is
missing enough predictors that its inflated diagonal exceeds that baseline. Uncalibrated
or pre-empirical-residual artifacts fall back to $SE = SE_{diag}$.

### Calibrated Interval

$$
SE_{scaled} = \max\!\left(\,\widehat{\sigma}^{\,cal}_{err},\; |b| \cdot SE_{diag}\,\right),
\qquad
\widehat{\sigma}^{\,cal}_{err} = \operatorname{sd}\!\left(S_{true} - (a + b\,S_{cv})\right)
$$

### 95% Confidence Interval

$$
CI = [S - 1.96 \cdot SE, \; S + 1.96 \cdot SE]
$$

See `imputed_prs/models/projection_predictor.py:ProjectionPredictor.predict()` and
`_region_effective_variance()` for implementation, and
`imputed_prs/evaluation/calibration.py:estimate_cv_calibration()` for the empirical
residual SDs.

---

## Inference-Time Behavior

At inference time, some predictor variants may be missing from the user's genotype data.

### Mean-Substitution for Missing Predictors

For each missing (or unresolvable) predictor variant $k$ in a region, substitute its
population mean dosage $2 q_k$ (where $q_k$ is the counted-allele frequency computed
from the reference panel during training), and let the region's remaining predictors
contribute. The prediction is

$$
\hat{S}_R = \sum_{k \in R,\, \text{available}} z_k \, a_{R,k}
          + \sum_{k \in R,\, \text{missing}} 2 q_k \, a_{R,k}
          + \gamma_R .
$$

No dosage clipping applies — the target is a PRS contribution, not a dosage. The allele
frequencies used for substitution are stored in
`ProjectionRegionModel.predictor_allele_frequencies`. The canonical upload path
(`compute_projected_prs_oriented`) does this allele-aware; the legacy dosage-dict scorer
(`compute_projected_prs`) does it allele-blind, but both substitute per predictor.

The deployed imputation scorer now matches this per-predictor behavior (its legacy
dosage-dict path was the historical all-or-nothing exception; see
[Linear Imputation](linear-imputation.md#inference-time-behavior)).

### Missingness-Aware Variance Inflation

The variance contributed by a partially-observed region is interpolated from the fitted
$\text{cv\\_mse}_R$ toward the region's intercept-only error variance, in proportion to
how many of its predictors were substituted:

$$
\text{cv\\_mse}_{eff,R}
   = \text{cv\\_mse}_R \cdot (1 - f) + \tau^2_R \cdot f,
\qquad
f = \frac{n_{substituted}}{n_{predictors}}
$$

where $\tau^2_R$ is the **variance of the region target $S_R$** across reference
samples — the error variance of predicting with the regional mean (the intercept-only
model) — stored as `ProjectionRegionModel.target_variance`. This is the projection
analogue of imputation's $2 q (1 - q)$ Hardy-Weinberg intercept-only variance. With no
substitution ($f = 0$), $\text{cv\\_mse}_{eff,R} = \text{cv\\_mse}_R$. Note that the
interpolation moves *upward* only when $\tau^2_R \geq \text{cv\\_mse}_R$ (non-negative CV
$R^2$); a region whose model predicts worse out-of-fold than its own mean correctly
interpolates *downward* toward the better intercept-only fallback.

```python
# From imputed_prs/models/projection_predictor.py:_region_effective_variance()
f = n_substituted / n_pred
return model.cv_mse * (1.0 - f) + model.target_variance * f
```

### Observed-Variant Fallback

As with imputation, every observed (on-platform) PRS variant is scored
**direct-effect-dosage-else-fallback** through the shared oriented scorer
(`compute_observed_prs_oriented`), so an observed variant is never silently dropped:
a variant that cannot be resolved/called directly is recovered from its per-variant
fallback model, and one with no fallback is surfaced in
`PredictionResult.unresolved_observed_ids`. (Per-variant fallback models for the
*projection* product are not yet populated in the current code — the projection
predictor notes they are "zero today" pending that work — but the scoring path and
diagnostics are shared with imputation so the two products report uncertainty
uniformly.)

See `imputed_prs/models/projection_predictor.py`
(`compute_projected_prs`, `compute_projected_prs_oriented`,
`_region_effective_variance`) for implementation.

---

## Practical Considerations

### When to Prefer Projection

- **Heterogeneous effect sizes**: When PRS effect sizes vary widely (e.g., some variants with $\beta \approx 0.01$ and others with $\beta \approx 1.0$), projection allocates regularization budget more efficiently
- **Dense multi-variant regions**: When many missing PRS variants cluster together, joint optimization can exploit covariance structure
- **Simpler uncertainty model**: No need for dosage clipping or truncated normal adjustments

### When to Prefer Imputation

- **Per-variant diagnostics**: Imputation provides $R^2$ and residual variance per variant, enabling fine-grained quality assessment
- **Per-variant export**: The imputation approach produces per-variant models that can be exported and inspected individually
- **Per-variant tuning**: Both methods support automatic tuning via `tuning_scope` (`"global"`/`"none"`), but only imputation offers `"per_variant"` (a separately-tuned model per variant). See [README — Hyperparameter Tuning](README.md#hyperparameter-tuning).

### Interpreting Region-Level $R^2$

| $R^2$ Range | Quality | Interpretation |
|-------------|---------|----------------|
| $> 0.8$ | Excellent | Region's PRS contribution is well-predicted |
| $0.4 - 0.8$ | Moderate | Useful but with meaningful uncertainty |
| $\leq 0.4$ | Poor | High uncertainty in this region's contribution |

### SE Interpretation

The reported SE is the empirical, score-level approximation error of the projected
estimate against the full computed PRS (floored by the LD-blind diagonal lower bound).
It reflects uncertainty from the projection model only. It does **not** include:
- Uncertainty in GWAS effect sizes ($\beta_j$)
- Population stratification effects
- Genotyping errors
- Transfer error to a different ancestry (see [Validation](validation.md))

---

## Implementation References

| Component | Source File |
|-----------|-------------|
| Region decomposition | `imputed_prs/core/regions.py` |
| Per-region ElasticNet fitting, predictor standardization | `imputed_prs/models/projection.py` |
| Region-level training orchestration | `imputed_prs/models/projection_trainer.py` |
| PRS calc, empirical SE, missing-predictor mean substitution, missingness inflation | `imputed_prs/models/projection_predictor.py` |
| True PRS / allele orientation (`match_oriented_dosage`) | `imputed_prs/evaluation/projection_evaluator.py`, `imputed_prs/core/harmonizer.py` |
| Oriented allele counting (`count_allele`) | `imputed_prs/io/user_genotypes.py` |
| Region tuning (`projection_hyperparameter_search`) | `imputed_prs/models/tuning.py` |
| Empirical residual SDs (calibration) | `imputed_prs/evaluation/calibration.py` |
| JSON export / load | `imputed_prs/core/linear_projection_prs.py`, `imputed_prs/io/exporters/projection_json_export.py` |
| Data types (`ProjectionRegionModel`) | `imputed_prs/core/types.py` |

For shared components (calibration, evaluation metrics), see [README.md](README.md).
For data type definitions, see the [API Reference](../API.md#data-types-reference).
