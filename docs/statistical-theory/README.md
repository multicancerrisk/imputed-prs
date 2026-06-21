# Statistical Theory: Overview

This document provides the shared statistical foundations for the two PRS computation methods in the `imputed-prs` library: **linear imputation** and **linear projection**. Each method has its own detailed document linked below; this page covers the common problem statement, notation, regularization framework, and calibration procedure.

## Table of Contents

1. [Overview](#overview)
2. [Notation](#notation)
3. [Elastic Net Regularization (Shared)](#elastic-net-regularization-shared)
4. [Internal Calibration via Cross-Validation (Shared)](#internal-calibration-via-cross-validation-shared)
5. [Methods](#methods)
6. [Validation](#validation)

---

## Overview

A Polygenic Risk Score (PRS) aggregates the effects of many genetic variants into a scalar predictive score:

$$
S = \sum_j x_j \beta_j
$$

Where:
- $x_j$ is the dosage (0, 1, or 2 copies of the effect allele) for variant $j$
- $\beta_j$ is the effect size (weight) for variant $j$

**The Problem.** Genotyping platforms (23andMe, AncestryDNA, etc.) measure only a subset of the variants defined in a PRS. When a variant is absent from the platform, there is no direct dosage measurement. Naively substituting zero for missing dosages causes systematic underestimation of the score and compresses the PRS distribution, because the missing component $\sum_{j \in M} x_j \beta_j$ is dropped entirely.

**Two Solutions.** This library implements two complementary approaches to recover the missing PRS contribution:

1. **Linear Imputation** -- predict each missing variant's dosage individually using nearby platform variants, then compute the PRS from observed and imputed dosages.
2. **Linear Projection** -- skip individual dosage prediction and directly predict each genomic region's PRS contribution from platform variants.

Both methods exploit linkage disequilibrium (LD) in a reference panel and share the same regularization and calibration framework described below.

---

## Notation

### Shared Symbols

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of samples in the reference panel |
| $Z$ | Platform genotype dosage matrix ($n \times p$), values in $[0, 2]$ |
| $X$ | Missing PRS variant dosage matrix ($n \times q$), values in $[0, 2]$ |
| $x_j$ | Dosage for variant $j$ (column of $X$ or $Z$) |
| $\beta_j$ | PRS effect size (weight) for variant $j$ |
| $O$ | Set of observed PRS variants (present on the genotyping platform) |
| $M$ | Set of missing PRS variants (absent from the platform) |
| $S$ | Total PRS $= S_{observed} + S_{missing}$ |
| $S_{observed}$ | $\sum_{j \in O} x_j \beta_j$ (observed component, computed exactly) |
| $q_j$ | Population allele frequency of variant $j$ |
| $\alpha$ | ElasticNet regularization strength |
| $\rho$ | ElasticNet L1/L2 mixing ratio (code parameter `l1_ratio`) |
| $b$ | Calibration scaling factor (regression slope) |
| $a$ | Calibration intercept |

### Imputation-Specific Symbols

| Symbol | Meaning |
|--------|---------|
| $L_j$ | Local window of predictor variants for missing variant $j$ |
| $w_{jk}$ | Imputation regression coefficient for predictor $k$ of target variant $j$ |
| $\gamma_j$ | Imputation model intercept for variant $j$ |
| $\hat{x}_j$ | Imputed dosage for missing variant $j$ |
| $r^2_j$ | Cross-validated $R^2$ for imputation of variant $j$ |
| $\sigma^2_j$ | Residual variance for imputed variant $j$ |

### Projection-Specific Symbols

| Symbol | Meaning |
|--------|---------|
| $R$ | A genomic region (contiguous interval from merged windows) |
| $Z_R$ | Platform dosages for variants within region $R$ (submatrix of $Z$) |
| $X_R$ | Missing PRS variant dosages within region $R$ (submatrix of $X$) |
| $\beta_R$ | Effect sizes for PRS variants in region $R$ (subvector of $\beta$) |
| $a_R$ | Learned projection weight vector for region $R$ |
| $\gamma_R$ | Projection model intercept for region $R$ |
| $S_R$ | Regional PRS target: $X_R \beta_R$ |
| $\hat{S}_R$ | Predicted regional PRS contribution: $z_R^T a_R + \gamma_R$ |

**Note on notation overlap.** The calibration intercept $a$ is a scalar estimated once from the full cross-validated PRS. The projection weight vector $a_R$ is a per-region vector of regression coefficients. Context (scalar vs. subscripted vector) distinguishes the two.

---

## Elastic Net Regularization (Shared)

Both methods use ElasticNet regularization, which combines L1 (Lasso) and L2 (Ridge) penalties. The general objective is:

$$
\min_{w, \gamma} \frac{1}{2n} \|Z_j w + \gamma - y\|^2 + \alpha \left[ \rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2 \right]
$$

The target $y$ differs between methods:

| Method | Target $y$ | Interpretation |
|--------|------------|----------------|
| Linear Imputation | $x_j$ (dosage vector for a single missing variant) | Predict individual dosage |
| Linear Projection | $S_R = X_R \beta_R$ (regional PRS contribution) | Predict aggregate PRS component |

Where:
- $n$ is the number of samples
- $Z_j$ is the matrix of predictor dosages (platform variants in the local window or region)
- $\alpha$ (`alpha`) controls overall regularization strength
- $\rho$ (`l1_ratio`) controls the L1/L2 balance

### Mixing Ratio Extremes

| $\rho$ Value | Penalty | Behavior |
|--------------|---------|----------|
| $\rho = 0$ | Pure Ridge (L2 only) | Shrinks all coefficients toward zero; keeps all predictors |
| $\rho = 1$ | Pure Lasso (L1 only) | Sparse solutions; drives some coefficients exactly to zero |
| $0 < \rho < 1$ | Elastic Net (both) | Balances sparsity and grouped selection |

### Default Parameters

| Parameter | Default | Code name |
|-----------|---------|-----------|
| $\alpha$ | 0.01 | `alpha` |
| $\rho$ | 0.5 | `l1_ratio` |

These defaults provide a balanced regularization that works well across a range of LD patterns and variant densities.

```python
# Both methods use the same defaults
LinearImputationPRS(alpha=0.01, l1_ratio=0.5)
LinearProjectionPRS(alpha=0.01, l1_ratio=0.5)
```

### Hyperparameter Tuning

Both methods support hyperparameter tuning via a `tuning_scope` parameter. All tuning
modes evaluate candidate `(\alpha, \rho)` pairs on the **same local-window (imputation)
or region (projection) matrices that training uses** — built with the identical
windowing/region call the trainer makes — so the tuner never optimizes a model that
differs from the one ultimately fit. To keep cost bounded, the search runs over a
**stratified subsample** of fitting units (variants or regions) rather than the full
set; the strata are chosen so the subsample spans the structural regimes that drive
the best hyperparameters.

**Linear Imputation** supports three strategies:

| Strategy | Behavior |
|----------|----------|
| `"global"` | Tune once on a bounded, stratified sample of missing variants, each scored on its own local window; apply the single winning `alpha`/`l1_ratio` to all variants. The sample is stratified by **chromosome × MAF bin × \|beta\| bin** (MAF and \|beta\| bucketed by data quantiles) and capped at `max_tuning_variants`. |
| `"per_variant"` | Grid-search each variant's own local window and give each variant its own `alpha`/`l1_ratio` (slower, more precise). |
| `"none"` | Use the provided `alpha` and `l1_ratio` directly. |

See `imputed_prs/models/tuning.py:global_hyperparameter_search()` (and
`optuna_hyperparameter_search()` for the TPE variant) plus
`imputed_prs/models/elastic_net.py:fit_single_variant_model()` for per-variant fitting.

**Linear Projection** supports `"global"` and `"none"` (there is no `"per_variant"`
mode for projection). `"global"` searches a bounded, stratified sample of regions
(capped by `max_tuning_regions`) on the same region matrices training uses — target
$S_R = X_R \beta_R$, predictors the region's platform variants — and applies the
winning `alpha`/`l1_ratio` to all region models. Because the unit of fitting is a
region (not a single variant), the projection sample is stratified by the region's
**predictor-count bucket × PRS-variant-count bucket** (a region's MAF/$\beta$ are
not single scalars), rather than by the imputation chr/MAF/$|\beta|$ key.

See `imputed_prs/models/projection.py:fit_single_region_model()` for the region
fitting implementation and `imputed_prs/models/tuning.py:projection_hyperparameter_search()`
for the region tuner.

---

## Internal Calibration via Cross-Validation (Shared)

Both methods use internal CV-based calibration to correct for systematic attenuation of PRS predictions. The procedure is conceptually identical; only the source of out-of-fold predictions differs.

### Regression Dilution

When dosages are imputed or regional contributions are projected, the resulting PRS predictions have reduced variance compared to the true PRS. This is a form of regression dilution: noise in the predictors attenuates the relationship between predicted and true scores.

$$
\mathbb{E}[S_{predicted}] \approx a + b \cdot S_{true} \quad \text{where } b < 1
$$

Without correction, the predicted PRS distribution is compressed toward the mean, underestimating tail risks and reducing discriminative power.

### Calibration Regression

During training, the library estimates the attenuation and corrects for it:

1. **Compute CV-predicted PRS ($S_{cv}$).** Using out-of-fold predictions from cross-validation, construct a PRS for each reference sample that was never used in its own prediction. This prevents overfitting the calibration estimate.
2. **Compute true PRS ($S_{true}$).** Using actual genotype dosages from the reference panel, compute the gold-standard PRS for each sample.
3. **Regress true on CV-predicted.** Fit a simple linear regression:

$$
S_{true} = a + b \cdot S_{cv}
$$

The slope $b$ (typically $> 1$) is the scaling factor that inflates predictions back to the correct variance. The intercept $a$ corrects for any mean shift.

The same out-of-fold fit also yields the library's **empirical, score-level
approximation error** — the standard deviations of the per-sample residuals
$S_{true} - S_{cv}$ (raw) and $S_{true} - (a + b\,S_{cv})$ (calibrated). Because these
are residuals of the *whole-score* difference, they capture the full
$\beta^\top \Sigma \beta$ covariance — LD off-diagonals included — that a per-variant
sum-of-variances structurally omits. They are the quantities the prediction interval
actually reports; see each method's *Uncertainty Quantification* section.

See `imputed_prs/evaluation/calibration.py:estimate_cv_calibration()` for implementation.

#### Missing-Data Handling in the Calibration Matrix

The score matrix used to assemble $S_{cv}$ and $S_{true}$ is effect-allele-oriented
reference dosages, one column per placed PRS variant. Samples missing a reference
dosage for a column are filled by **per-column mean imputation**
(`mean_impute_columns`), not by `NaN \to 0`. Under Hardy-Weinberg equilibrium a
column's mean equals the population-expected dosage $2 q_j$, so filling with the column
mean substitutes the unbiased expectation rather than $0$ (which would assert
homozygous *non-effect* and pull both $S_{true}$ and the observed part of $S_{cv}$
toward zero).

> **Honest caveat about what this fixes.** Mean-filling de-biases the **location** of
> each column (its mean is preserved exactly) and lowers reconstruction SSE relative to
> zero-filling. It does **not** make the estimated calibration *scaling* parameters
> ($a$, $b$) uniformly closer to their complete-case values: imputed cells sit exactly
> at the column mean, which **shrinks** the column's variance and distorts its
> covariance with other columns. What is robust here is column-mean (location)
> preservation and a smaller reconstruction error — not an unbiased recovery of the
> regression slope under missingness. See `imputed_prs/evaluation/calibration.py:mean_impute_columns()`.

### How $S_{cv}$ Differs Between Methods

The construction of $S_{cv}$ varies because the two methods produce different types of out-of-fold predictions:

**Linear Imputation.** For each missing variant $j$, the cross-validation loop produces out-of-fold dosage predictions $\hat{x}_j^{(-f)}$. These are assembled into a PRS:

$$
S_{cv} = \underbrace{\sum_{j \in O} x_j \beta_j}_{\text{observed (exact)}} + \underbrace{\sum_{j \in M} \hat{x}_j^{(-f)} \beta_j}_{\text{imputed (CV predictions)}}
$$

See `imputed_prs/evaluation/calibration.py:compute_cv_predicted_prs()`.

**Linear Projection.** For each region $R$, the cross-validation loop produces out-of-fold predictions of the regional PRS contribution $\hat{S}_R^{(-f)}$. These are summed directly:

$$
S_{cv} = \underbrace{\sum_{j \in O} x_j \beta_j}_{\text{observed (exact)}} + \underbrace{\sum_R \hat{S}_R^{(-f)}}_{\text{projected (CV predictions)}}
$$

No per-variant betas are applied to the region predictions because the target $S_R = X_R \beta_R$ already incorporates the effect sizes.

See `imputed_prs/core/linear_projection_prs.py` (Step 10) for the projection calibration assembly.

### Applying Calibration at Prediction Time

At inference, the raw PRS point estimate is rescaled using the learned calibration parameters:

$$
S_{scaled} = a + b \cdot S_{raw}
$$

$$
CI_{scaled} = S_{scaled} \pm 1.96 \cdot SE_{scaled}
$$

The interval half-width $SE_{scaled}$ is **not** simply $|b| \cdot SE$. Since the
empirical residual calibration (see each method's *Uncertainty Quantification*
section), the reported $SE_{scaled}$ is the empirical post-calibration residual SD,
floored by the slope-scaled diagonal lower bound:

$$
SE_{scaled} = \max\!\left(\,\widehat{\sigma}^{\,cal}_{err},\; |b| \cdot SE_{diag}\,\right)
$$

The legacy form $|b| \cdot SE$ is recovered exactly on artifacts that predate this
empirical calibration (no stored residual SD), where $SE = SE_{diag}$. The absolute
value $|b|$ keeps the half-width non-negative regardless of the sign of the slope.

### Calibration Parameters

| Parameter | Field | Description |
|-----------|-------|-------------|
| $b$ | `scaling_factor` | Slope from calibration regression |
| $SE(b)$ | `scaling_factor_se` | Standard error of slope |
| $a$ | `calibration_intercept` | Intercept from calibration regression |
| $R^2$ | `calibration_r2` | $R^2$ of the calibration fit |
| $SD(S_{cv}) / SD(S_{true})$ | `attenuation_factor` | Measures how much variance is lost |
| $\widehat{\sigma}^{\,raw}_{err}$ | `raw_empirical_residual_sd` | $SD(S_{true} - S_{cv})$ — empirical raw-scale approximation error |
| $\widehat{\sigma}^{\,cal}_{err}$ | `calibrated_empirical_residual_sd` | $SD(S_{true} - (a + b\,S_{cv}))$ — empirical calibrated approximation error |
| $SE_{diag}$ (full-data) | `diagonal_model_se_lower_bound` | Fit-time, no-missingness diagonal SE; an optimistic LD-blind reference lower bound |

The last three fields are `None` on artifacts predating the empirical residual
calibration, in which case prediction falls back to the diagonal SE.

See `imputed_prs/core/types.py:CalibrationParams` for the full data type definition.

> **Measuring the resulting error.** Internal calibration corrects attenuation against the *same population's* full PRS — it is not external/clinical calibration. For how to *measure* a deployed model's approximation error on a platform-masked (or cross-ancestry) cohort, see [Validation](validation.md).

---

## Methods

### Linear Imputation

Linear imputation predicts each missing variant's dosage individually from nearby platform variants using an ElasticNet regression model, then computes the PRS from observed and imputed dosages. This per-variant approach provides fine-grained quality metrics ($r^2_j$, $\sigma^2_j$) and supports dosage bounding with truncation-adjusted variance, making it well-suited for detailed uncertainty quantification at the variant level.

See [Linear Imputation](linear-imputation.md) for the full method description.

### Linear Projection

Linear projection skips individual dosage prediction and instead learns, for each genomic region, a single regression model that maps platform variant dosages directly to the region's PRS contribution $S_R = X_R \beta_R$. By predicting the weighted sum directly, projection avoids accumulating per-variant imputation errors and can be more efficient when many missing PRS variants cluster in the same LD region.

See [Linear Projection](linear-projection.md) for the full method description.

---

## Validation

Beyond internal cross-validation, the library provides a **masking-validation harness**: it masks a reference panel down to a genotyping platform's variants, scores the masked panel through the deployed prediction path, and compares the estimate to the full computed PRS. It reports correlation, top-decile concordance, empirical approximation error, and interval coverage, plus a raw-parser round-trip and cross-ancestry caveats — and makes explicit that internal calibration is not external calibration.

See [Validation](validation.md) for the methodology and `imputed_prs/evaluation/validation.py:run_masking_validation()` for the implementation.
