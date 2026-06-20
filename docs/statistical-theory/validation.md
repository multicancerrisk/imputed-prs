# Statistical Theory: Masking Validation

This document describes the **layered masking-validation harness**
(`imputed_prs/evaluation/validation.py`), which measures how well a trained model
reproduces a target PRS when the user only has a genotyping platform's variants. It
complements the internal cross-validated calibration described in the
[Overview](README.md#internal-calibration-via-cross-validation-shared); read that
section first for shared notation ($S$, $S_{true}$, $S_{cv}$, slope $b$, intercept
$a$).

## Table of Contents

1. [Purpose and Scope](#purpose-and-scope)
2. [Masking Methodology](#masking-methodology)
3. [The Metric Panel](#the-metric-panel)
4. [Internal Calibration Is Not External Calibration](#internal-calibration-is-not-external-calibration)
5. [Cross-Ancestry Caveats](#cross-ancestry-caveats)
6. [The Raw-Parser Round-Trip](#the-raw-parser-round-trip)
7. [Limitations](#limitations)

---

## Purpose and Scope

Internal cross-validation answers *"how well does the model predict its own held-out
fold?"* It cannot tell you what a **real platform upload** will score, because a user
only ever has the genotyping chip's variants — every off-platform PRS variant must be
imputed or projected, never read directly.

Masking validation closes that gap. It takes a reference panel, **masks it down to a
platform's variant set** (blanks every off-platform variant, exactly as if the user
only ran a 23andMe / AncestryDNA chip), scores the masked panel **through the deployed
library path**, and compares that estimate to the **full PRS** computed from the
complete panel.

It answers one question: *how close is the masked-platform estimate to the full
computed PRS?* It does **not** answer *"is this PRS clinically valid?"* — that is
external validation against disease outcomes, which is out of scope (see
[Limitations](#limitations)).

The entry point is `run_masking_validation()`, which emits a
`MaskingValidationReport` metric panel for one PRS × platform.

---

## Masking Methodology

Let $V$ be the variants in the reference panel, $P \subseteq V$ the platform variant
set, and $\beta$ the PRS weights.

**1. True (full) PRS.** Computed on the *complete* panel over every placed PRS
variant, effect-allele-oriented:

$$
S_{true} = \sum_{j} x_j \, \beta_j
$$

This is the gold standard — the same quantity the calibration step regresses against
(`README.md`). It is produced by the evaluators' `compute_score_arrays()` (which wraps
the existing `_compute_true_prs`).

**2. Mask to the platform.** `mask_reference_to_platform()` keeps the on-platform
columns at their true dosages and sets every off-platform column to `NaN`:

$$
\tilde{x}_j =
\begin{cases}
x_j & j \in P \\
\texttt{NaN} & j \notin P
\end{cases}
$$

On-platform membership is decided by `harmonizer.partition_variants()` (rsID-first,
then `chr:pos`), the same matching the trainer uses — a naive `variant_id ∈ P` test
would miss `chr:pos`-matched variants. Columns are **blanked, not dropped**, so
`variant_info` stays intact for allele re-resolution, and `NaN` is treated exactly as
a missing off-platform variant by the scorers.

**3. Estimated PRS.** Computed on the *masked* panel through the same allele-oriented
scoring path as the browser/upload (P1.6): observed (on-platform) variants are scored
directly; off-platform variants are imputed/projected from their on-platform
predictors (which are retained by the mask):

$$
S_{est} = \underbrace{\sum_{j \in O} \tilde{x}_j \, \beta_j}_{\text{observed (exact)}}
        + \underbrace{\sum_{j \in M} \hat{x}_j(\tilde{x}_P) \, \beta_j}_{\text{imputed/projected from platform}}
$$

Computing $S_{true}$ on the **full** panel but $S_{est}$ on the **masked** panel is
what makes this a genuine masking test rather than a self-comparison.

---

## The Metric Panel

`MaskingValidationReport` reports the following over the $n$ samples (NaN pairs
excluded). Accuracy, concordance, and empirical error are computed on the fast,
vectorized batch arrays; coverage and the raw-parser check run a small subsample
through the public `predict()` (see [below](#the-raw-parser-round-trip)).

### Accuracy

Computed by `metrics.compute_prs_metrics()`:

- **Pearson $r$** and $r^2$ between $S_{est}$ and $S_{true}$.
- **Spearman $\rho$** — rank correlation (robust to the monotone calibration map).
- **MAE / RMSE** — $\operatorname{mean}|S_{est}-S_{true}|$ and
  $\sqrt{\operatorname{mean}(S_{est}-S_{true})^2}$.
- **Calibration slope/intercept** — from regressing $S_{true} = a + b\,S_{est}$.

### Concordance

Computed by `metrics.compute_percentile_concordance()`:

- **Top/bottom $p$% concordance** for $p \in \{1, 5, 10\}$ — the fraction of
  individuals in the top (bottom) $p\%$ by $S_{est}$ who are also in the top (bottom)
  $p\%$ by $S_{true}$. The **top-decile** ($p=10$) value is lifted out as the headline
  `top_decile_concordance`.
- **Quintile Cohen's $\kappa$** — agreement of quintile assignments.

### Empirical Approximation Error

The honest, score-level (LD-aware) error of the masked estimate against the full PRS:

$$
\widehat{\sigma}_{err} = \operatorname{sd}\!\left(S_{true} - S_{est}\right),
\qquad
\widehat{\mu}_{err} = \operatorname{mean}\!\left(S_{true} - S_{est}\right)
$$

`empirical_error_sd` is the masking-cohort analog of
`CalibrationParams.raw_empirical_residual_sd` (which is the same quantity measured on
out-of-fold CV scores; see `README.md`). It captures the full $\beta^\top \Sigma
\beta$ residual covariance — including LD off-diagonals — that a diagonal
sum-of-variances would omit. `empirical_error_mean` is the bias (≈ 0 for a
well-behaved model; a non-zero sign reveals systematic under/over-estimation).

### Interval Coverage

`coverage_95` is the fraction of the subsample whose **true** PRS falls inside the
interval the deployed `predict()` actually returns:

$$
\text{coverage}_{95} = \frac{1}{n}\sum_{i} \mathbb{1}\!\left[\, S_{true,i} \in
\left[\text{ci\_lower}_i,\ \text{ci\_upper}_i\right]\,\right]
$$

When the model carries calibration, this uses the **calibrated** interval
(`ci_*_scaled`, built from `calibrated_empirical_residual_sd`) — the interval a user
sees; otherwise it falls back to the raw interval. Well-behaved intervals achieve
≈ 0.95 coverage.

---

## Internal Calibration Is Not External Calibration

This distinction is the most important caveat the harness emits, and it is worth
stating plainly.

The model's internal calibration (slope $b$, intercept $a$) corrects **regression
dilution**: imputation/projection shrink the predicted PRS variance, and the
calibration regression $S_{true} = a + b\,S_{cv}$ inflates it back so that, *within the
training population*, the calibrated estimate matches the full computed PRS in scale.

That is **all** it does. In particular, internal calibration does **not**:

- calibrate the PRS to an **external cohort** (a different sample / array / pipeline);
- calibrate the PRS to **disease risk** or any clinical outcome (absolute risk,
  odds ratios, decision thresholds);
- account for **ancestry transfer** (see below).

Masking validation measures the **approximation error of the masked-platform estimate
against the full computed PRS**, on the same population. A model can post excellent
masking-validation numbers and still be poorly calibrated for clinical use — those are
orthogonal questions. `run_masking_validation()` always emits this statement in
`report.caveats`.

---

## Cross-Ancestry Caveats

LD structure and allele frequencies differ across ancestries. A model trained on one
ancestry will degrade when applied to another, for three compounding reasons:

1. **Imputation/projection weights** are LD-dependent: predictor→target relationships
   learned in the training ancestry transfer imperfectly.
2. **Mean substitution** for missing predictors fills $2 q_j$ using the *training*
   allele frequency $q_j$, which is biased off-ancestry.
3. **Calibration** ($a$, $b$) was estimated on the training population's
   $S_{true}$/$S_{cv}$ and need not hold elsewhere.

The harness is **structured for** cross-ancestry evaluation but does not download
multi-ancestry panels itself: pass `evaluation_ancestry` and a different-ancestry
reference panel. The report records `evaluation_ancestry`, sets the `cross_ancestry`
flag when it differs (case-insensitively) from the model's `training_ancestry`, and
appends a caveat that these metrics are a **lower bound** on within-ancestry
performance. A genuine cross-ancestry run therefore means handing
`run_masking_validation()` a reference panel of the target ancestry.

---

## The Raw-Parser Round-Trip

Beyond the vectorized batch comparison, the harness verifies that the **literal upload
path** agrees with the batch estimate, so the DTC file parser, multi-key resolution,
and oriented scoring cannot silently drift apart.

For a small subsample, masked on-platform dosages are rendered to genotype strings
(`io/user_genotypes.py:render_genotype_string()` — *raw*, ALT-counted, so `predict`'s
`count_allele` re-orients per role) and written to a synthetic **23andMe-format** file.
That file is scored through the public `predict()` (file → `snps` parser → multi-key
resolution → oriented scoring), and the uncalibrated result is compared to the batch
estimate $S_{est}$ for the same sample:

$$
\texttt{raw\_parser\_max\_abs\_diff} = \max_i \left| \text{predict}(\text{file}_i).\text{prs} - S_{est,i} \right|
$$

On hard-called biallelic data the two paths are expected to agree to numerical
precision (`raw_parser_agrees` ⇔ max diff ≤ 1e-6) — this is the P1.6 numeric-vs-string
guarantee extended through the file parser. The check is skipped (not failed) when the
`snps` package is unavailable or the reference is continuous (DS/GP), since neither has
a genotype-string upload representation.

---

## Limitations

- Masking validation measures **approximation error vs the full computed PRS**, not
  external or clinical validity (see
  [above](#internal-calibration-is-not-external-calibration)).
- Coverage and the raw-parser check require **hard-called** reference data (genotype
  strings); on continuous DS/GP panels only the batch accuracy/concordance/error
  panel is produced.
- Concordance needs ≥ 20 valid samples and accuracy needs ≥ 3; below these the
  affected metrics are skipped or `nan` with a caveat rather than fabricated.
- A degenerate (zero-variance) cohort yields `nan` correlation/calibration with a
  caveat — by design, a constant PRS is a real degenerate cohort, not an error.
- When every PRS variant is on-platform, masking is a no-op and the run does not
  exercise imputation/projection (the report says so).

See `imputed_prs/evaluation/validation.py` for the implementation and
`tests/test_statistical_validation.py` (the `TestMaskReferenceToPlatform` /
`TestRunMaskingValidationEndToEnd` / … classes) for worked examples.
