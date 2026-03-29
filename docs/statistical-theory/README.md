# Statistical Theory: Overview

This document provides the shared statistical foundations for the two PRS computation methods in the `imputed-prs` library: **linear imputation** and **linear projection**. Each method has its own detailed document linked below; this page covers the common problem statement, notation, regularization framework, and calibration procedure.

## Table of Contents

1. [Overview](#overview)
2. [Notation](#notation)
3. [Elastic Net Regularization (Shared)](#elastic-net-regularization-shared)
4. [Internal Calibration via Cross-Validation (Shared)](#internal-calibration-via-cross-validation-shared)
5. [Methods](#methods)

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

The two methods differ in their tuning support:

**Linear Imputation** supports the `tuning_scope` parameter with three strategies:

| Strategy | Behavior |
|----------|----------|
| `"global"` | Tune once on a random subset of variants, apply best parameters to all |
| `"per_variant"` | Tune separately for each variant (slower, more precise) |
| `"none"` | Use the provided `alpha` and `l1_ratio` directly |

See `imputed_prs/models/tuning.py` for the tuning implementation and `imputed_prs/models/elastic_net.py:fit_single_variant_model()` for per-variant fitting.

**Linear Projection** currently uses fixed parameters directly (equivalent to `"none"`). The `alpha` and `l1_ratio` passed to `LinearProjectionPRS()` are applied uniformly to all region models.

See `imputed_prs/models/projection.py:fit_single_region_model()` for the region fitting implementation.

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

See `imputed_prs/evaluation/calibration.py:estimate_cv_calibration()` for implementation.

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

At inference, the raw PRS and its uncertainty are rescaled using the learned calibration parameters:

$$
S_{scaled} = a + b \cdot S_{raw}
$$

$$
SE_{scaled} = |b| \cdot SE
$$

$$
CI_{scaled} = S_{scaled} \pm 1.96 \cdot SE_{scaled}
$$

The absolute value $|b|$ ensures the standard error remains non-negative regardless of the sign of the slope.

### Calibration Parameters

| Parameter | Field | Description |
|-----------|-------|-------------|
| $b$ | `scaling_factor` | Slope from calibration regression |
| $SE(b)$ | `scaling_factor_se` | Standard error of slope |
| $a$ | `calibration_intercept` | Intercept from calibration regression |
| $R^2$ | `calibration_r2` | $R^2$ of the calibration fit |
| $SD(S_{cv}) / SD(S_{true})$ | `attenuation_factor` | Measures how much variance is lost |

See `imputed_prs/core/types.py:CalibrationParams` for the full data type definition.

---

## Methods

### Linear Imputation

Linear imputation predicts each missing variant's dosage individually from nearby platform variants using an ElasticNet regression model, then computes the PRS from observed and imputed dosages. This per-variant approach provides fine-grained quality metrics ($r^2_j$, $\sigma^2_j$) and supports dosage bounding with truncation-adjusted variance, making it well-suited for detailed uncertainty quantification at the variant level.

See [Linear Imputation](linear-imputation.md) for the full method description.

### Linear Projection

Linear projection skips individual dosage prediction and instead learns, for each genomic region, a single regression model that maps platform variant dosages directly to the region's PRS contribution $S_R = X_R \beta_R$. By predicting the weighted sum directly, projection avoids accumulating per-variant imputation errors and can be more efficient when many missing PRS variants cluster in the same LD region.

See [Linear Projection](linear-projection.md) for the full method description.
