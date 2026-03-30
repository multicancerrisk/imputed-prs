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

See `imputed_prs/models/projection.py:fit_single_region_model()` for implementation.

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

## Uncertainty Quantification

The standard error for the projected PRS is:

$$
SE = \sqrt{\sum_R \text{cv\\_mse}_R}
$$

Where $\text{cv\\_mse}_R$ is the cross-validated mean squared error for region $R$.

### Why This Differs from Imputation

The imputation SE composes per-variant terms $\beta_j^2 \sigma^2_{adjusted,j}$ because each variant is imputed independently and then multiplied by its effect size. The projection SE uses $\text{cv\\_mse}_R$ directly because the training target $S_R$ already incorporates the effect sizes -- the CV-MSE measures PRS prediction error, not dosage prediction error.

### 95% Confidence Interval

$$
CI = [S - 1.96 \cdot SE, \; S + 1.96 \cdot SE]
$$

See `imputed_prs/models/projection_predictor.py:ProjectionPredictor.predict()` for implementation.

---

## Inference-Time Behavior

At inference time, some predictor variants may be missing from the user's genotype data.

### Mean-Substitution Fallback

For each missing predictor variant $k$, substitute its population mean dosage $2 q_k$ (where $q_k$ is the allele frequency computed from the reference panel during training). This preserves partial information from the available predictors.

This differs from the imputation approach, which uses an all-or-nothing fallback: if any predictor is missing for a variant, the entire model falls back to intercept-only. The projection approach substitutes individual missing predictors while retaining the available ones.

The allele frequencies used for substitution are stored in `ProjectionRegionModel.predictor_allele_frequencies`.

See `imputed_prs/models/projection_predictor.py:compute_projected_prs()` for implementation.

---

## Practical Considerations

### When to Prefer Projection

- **Heterogeneous effect sizes**: When PRS effect sizes vary widely (e.g., some variants with $\beta \approx 0.01$ and others with $\beta \approx 1.0$), projection allocates regularization budget more efficiently
- **Dense multi-variant regions**: When many missing PRS variants cluster together, joint optimization can exploit covariance structure
- **Simpler uncertainty model**: No need for dosage clipping or truncated normal adjustments

### When to Prefer Imputation

- **Per-variant diagnostics**: Imputation provides $R^2$ and residual variance per variant, enabling fine-grained quality assessment
- **Per-variant export**: The imputation approach produces per-variant models that can be exported and inspected individually
- **Hyperparameter tuning**: The imputation method supports automatic tuning via `tuning_scope`

### Interpreting Region-Level $R^2$

| $R^2$ Range | Quality | Interpretation |
|-------------|---------|----------------|
| $> 0.8$ | Excellent | Region's PRS contribution is well-predicted |
| $0.4 - 0.8$ | Moderate | Useful but with meaningful uncertainty |
| $\leq 0.4$ | Poor | High uncertainty in this region's contribution |

### SE Interpretation

The SE reflects uncertainty from the projection model only. It does not include:
- Uncertainty in GWAS effect sizes ($\beta_j$)
- Population stratification effects
- Genotyping errors

---

## Implementation References

| Component | Source File |
|-----------|-------------|
| Region decomposition | `imputed_prs/core/regions.py` |
| Per-region ElasticNet fitting | `imputed_prs/models/projection.py` |
| Region-level training orchestration | `imputed_prs/models/projection_trainer.py` |
| PRS calculation, SE, missing-predictor fallback | `imputed_prs/models/projection_predictor.py` |
| Data types (`ProjectionRegionModel`) | `imputed_prs/core/types.py` |

For shared components (calibration, evaluation metrics), see [README.md](README.md).
For data type definitions, see the [API Reference](../API.md#data-types-reference).
