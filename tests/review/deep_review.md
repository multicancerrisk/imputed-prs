## Verdict

The **core idea is plausible** for your stated goal: fixed genotyping platform, fixed PRS, offline-trained reference model, browser-side linear scoring. In fact, the **linear projection** formulation is probably the cleaner approach for deployment because it directly learns the missing-variant PRS contribution rather than first approximating thousands of individual dosages and then summing them. Your docs describe exactly this distinction: linear imputation predicts missing variant dosages, while linear projection predicts the aggregate missing PRS contribution for local regions. ([GitHub][1])

But the current implementation is **not yet safe for real PRS scoring**. The largest problem is not the linearity assumption. The largest problem is that the inference path loses allele orientation. In several places the code treats a user genotype like `"AA"` or `"GG"` as dosage `2.0` without knowing whether A or G is the effect allele or the allele counted during training. That makes observed PRS terms wrong and also makes imputed/projection predictors wrong. This is a deterministic correctness bug, not a statistical nuance. 

I would treat the current codebase as a promising prototype whose **training-side allele harmonization has moved in the right direction**, but whose **browser/user-genotype inference design needs a major rewrite before scientific validation**.

---

## Critical blocker 1: user genotype dosage is wrong

The docs correctly define PRS as a sum of effect-allele dosages times weights. In other words, `x_j` must mean “number of copies of the effect allele.” ([GitHub][1])

The user genotype loader does not compute that. `genotype_to_dosage()` converts any homozygous diploid genotype to `2.0` and any heterozygous genotype to `1.0`; it does not know which allele is being counted. So:

```text
effect allele = A
AA should be 2
AG should be 1
GG should be 0

current code:
AA -> 2
AG -> 1
GG -> 2
```

For the opposite effect allele:

```text
effect allele = G
AA should be 0
AG should be 1
GG should be 2

current code:
AA -> 2
AG -> 1
GG -> 2
```

That means many homozygous non-effect genotypes are scored as homozygous effect genotypes. The loader explicitly says it “simply counts alleles without assuming which is the effect allele,” but it is then passed directly into PRS prediction. 

This breaks the observed component immediately. `compute_observed_prs()` simply retrieves the dosage by variant ID and adds `dosage * beta`; there is no allele-aware conversion at prediction time.  The high-level `predict()` method loads user genotypes, gets these scalar dosages, and passes them directly into `PRSPredictor`. 

It also breaks imputation/projection predictors. The reference training matrix `Z` is based on reference genotype dosages with a specific allele interpretation, but the browser/user path supplies “homozygosity dosage,” not the same counted allele. So even if the model coefficients are statistically good in reference data, inference may feed the model the wrong variables.

**This must be fixed before any validation numbers are meaningful.**

The right architecture is: raw user genotype parsing should return genotype strings plus position/ID metadata, not final dosages. Then model-specific code should count the correct allele for each observed PRS variant and each predictor variant. The exported model must include, for every required variant, the chromosome, position, genome build, acceptable IDs, effect allele or counted allele, other allele, strand/complement handling, and whether ambiguous palindromic SNPs are allowed.

---

## Critical blocker 2: training is allele-aware, but inference is not

The training side has a much better allele-aware function: `match_oriented_dosage()` matches by chromosome/position, compares PRS effect/other alleles against reference/alternate alleles, tries complement alleles, and flips dosage with `2 - dosage` when the effect allele is the reference allele.  Missing PRS targets are built using effect-allele-oriented dosages, which is the right concept. 

The problem is that the browser/user prediction path does not use the same allele-aware machinery. It uses a dictionary from variant ID to scalar dosage, and the scalar has already lost the allele identity. The model cannot recover whether `2.0` meant AA, CC, GG, or TT.

This can create a dangerous failure mode: internal cross-validation can look good because training/evaluation uses allele-oriented reference dosages, while real user predictions are wrong because DTC genotype conversion is not allele-oriented.

---

## Critical blocker 3: predictor allele metadata is not exported or stored

The model objects store predictor variant IDs and coefficients, but not the predictor allele that the coefficient expects. For projection, `ProjectionRegionModel` contains `predictor_variant_ids`, coefficients, intercept, CV metrics, and predictor allele frequencies, but it does not store reference allele, alternate allele, counted allele, or strand-orientation information. 

Similarly, the platform predictor matrix is built from `variant_id`, `chromosome`, and `position`, and `Z` is taken from `genotype_data.dosage_matrix`; the code does not preserve the allele counted by each `Z` column for browser inference.  The JSON exporter is also imputation-oriented and builds a `platform_variant_index` only from observed variants; it does not export the allele-orientation map needed to convert raw user genotypes into the model’s predictor space. 

This is a schema-level issue. A browser model should not just say:

```json
"predictor_variant_ids": ["rs123", "rs456"],
"coefficients": [0.14, -0.03]
```

It needs to say something like:

```json
{
  "variant_id": "rs123",
  "chromosome": "1",
  "position": 1234567,
  "genome_build": "GRCh37",
  "counted_allele": "G",
  "other_allele": "A",
  "strand": "plus",
  "ambiguous": false,
  "training_mean": 0.37,
  "training_scale": 0.62,
  "coefficient": 0.14
}
```

Without this, Python training and JavaScript inference will not be computing the same features.

---

## Major issue: observed variant status is probably too ID-based

A PRS variant should only be considered directly observed if the platform/user data contains the same locus and allele definition. The code appears to build platform predictors largely by ID matching against genotype variants and then later builds observed variants from PRS rows whose IDs are in the observed set. The observed scoring path then simply multiplies user dosage by beta. 

This means an “observed” PRS variant can be accepted by ID/locus availability even though the runtime user dosage is not allele-oriented. For correctness, “observed” should mean:

1. same genome build,
2. same chromosome and position,
3. compatible allele pair,
4. effect allele count can be computed from the user genotype,
5. palindromic SNP policy satisfied,
6. variant representation is not ambiguous or multiallelic without resolution.

A direct PRS hit is only safe when those conditions are met.

---

## Major issue: projection evaluation can be allele-wrong

The projection evaluator contains an explicit warning-like comment: region models do not store per-variant alleles, so true PRS for region variants is computed using the first reference row at each locus and is said to be correct for the common effect-equals-ALT biallelic case. 

That means projection evaluation can be wrong for:

* effect allele equals REF,
* strand-complemented variants,
* multiallelic sites,
* duplicate rsIDs,
* variants where the first reference row at a locus is not the intended allele.

This undermines projection validation. The projection model should store the PRS variant allele metadata used to construct the regional target, or the evaluator should compute true PRS from the original PRS DataFrame plus allele-aware matching.

---

## Statistical issue: uncertainty is too optimistic under LD

The imputation documentation uses a per-variant residual variance formula and propagates uncertainty as a sum of `beta² * residual_variance`. ([GitHub][2]) The code follows that general design: residual variance is based on `2q(1-q)(1-r²)`, and prediction sums `beta² * adjusted_variance`.  

That assumes residual imputation errors are independent across variants. In real LD blocks, residual errors for nearby variants are correlated. For dense PRS regions, summing only diagonal variances will often understate the uncertainty.

Projection partly helps because the target is the regional PRS contribution and the region CV MSE captures within-region aggregate error. But the projection predictor still sums `model.cv_mse` across regions, assuming region residuals are independent and using the same population-average error for every individual. 

A better uncertainty model would use out-of-fold residuals at the score or block level:

```text
residual_i = true_prs_i - predicted_prs_i
```

Then estimate:

* global score-level residual SD by ancestry/platform/PRS,
* block residual covariance across regions,
* missingness-pattern-adjusted residual SD,
* empirical prediction intervals rather than nominal normal intervals.

For browser deployment, the simplest robust version is to export empirical calibration/error parameters from held-out reference data and report them as “expected approximation error,” not as individual clinical confidence intervals.

---

## Statistical issue: calibration is internal, not external validation

The docs describe calibration by regressing true PRS on cross-validated predicted PRS, then applying a scaling factor and intercept. ([GitHub][1]) That is reasonable for correcting attenuation inside the reference panel.

But it should not be interpreted as external calibration for arbitrary users. A model trained and calibrated on 1000 Genomes EUR, for example, may not be calibrated for a 23andMe user with different ancestry, array missingness, genotype calling artifacts, or build/liftover differences.

There is also a code-level concern: calibration construction uses `np.nan_to_num()` when building the effect-oriented placed-variant matrix. That turns missing reference dosages into `0`, which means homozygous non-effect, not “missing.”  Projection calibration has the same pattern.  This can bias both `s_cv` and `s_true`.

For calibration, missing genotypes should be handled consistently: either exclude samples/variants with missing values, use mean imputation with documented assumptions, or train/evaluate on fully imputed reference panels where missingness has already been resolved.

---

## Major issue: missing predictors are handled inconsistently

For linear imputation, if any predictor for a missing variant is absent in the user data, the code falls back to the model intercept, discarding all available predictors for that variant.  It then still adds the model’s residual variance after clipping/adjustment. 

That is not ideal. If the trained model has 50 predictors and the user is missing one, falling all the way back to the mean is unnecessarily lossy. Worse, the residual variance used is the residual variance from the full predictor model, not the residual variance of the intercept-only fallback. If the model falls back to the intercept, the uncertainty should generally revert toward the full Hardy-Weinberg genotype variance or a missingness-pattern-specific residual variance.

Projection handles missing predictors differently: it substitutes `2 * AF` for missing predictor dosages, which is more graceful. But it still uses the same regional CV MSE regardless of how many predictors were substituted.  If many predictors are missing for a user, the uncertainty should increase.

Recommended fix: use mean substitution for both approaches, store predictor means, optionally store standard deviations, and estimate residual error as a function of predictor missingness.

---

## Major issue: ElasticNet is fit on unstandardized genotype dosages

`fit_single_variant_model()` and `fit_single_region_model()` fit `ElasticNet` directly on raw dosage columns. They filter out rows with any missing predictor and then call `ElasticNet(...).fit(X_train, y_train)`.  

Scikit-learn’s `ElasticNet` does not automatically standardize predictors. Genotype columns have different variances depending on allele frequency. The penalty therefore depends on MAF: common variants and rare variants are penalized on different effective scales. This can change which predictors are selected and can make alpha values difficult to interpret across regions.

For a penalized linear model, you should standardize predictors during training, store means and scales, and apply the same transformation in the browser. Alternatively, back-transform coefficients to raw dosage scale after fitting, but then be careful with intercept reconstruction.

---

## Major issue: global hyperparameter tuning optimizes the wrong design

The high-level fit path runs global hyperparameter search with the full platform matrix `Z` and all missing variants. The comment says `sample_indices=None  # Use all variants`.  The tuning helper evaluates a target variant by calling `fit_single_variant_model(target_dosages=target, predictor_dosages=Z, ...)`, so it is tuning on the full chip-wide predictor matrix. 

But the actual trainer later fits each variant using a local genomic window. The trainer calls `filter_to_local_window()` to select local predictors for each target. 

So global tuning is optimizing a different model than the one later used. It is also computationally expensive if `Z` contains hundreds of thousands of chip variants.

Recommended fix:

* tune on the same local-window matrices used by final training,
* sample a bounded number of missing variants, not all variants,
* stratify tuning variants by chromosome, MAF, predictor count, and PRS weight magnitude,
* consider separate alpha values for imputation and projection,
* either implement `per_variant` tuning or remove the option.

---

## Data processing issue: PRS loader aliases are risky

The PRS loader maps many column aliases into canonical names. In particular, it maps `alt` to `effect_allele` and `ref` to `other_allele`, while only `variant_id`, `effect_allele`, and `beta` are required. 

That is unsafe for generic summary-statistic files because ALT is not necessarily the effect allele. For PGS Catalog scoring files, effect allele columns are explicit; for arbitrary GWAS summary statistics, `A1`, `A2`, `ALT`, and `REF` conventions vary.

For this library, I would require a stricter schema:

```text
variant_id
chromosome
position
effect_allele
other_allele
effect_weight
genome_build
```

Then validate:

* non-null effect allele and beta,
* chromosome and position present,
* beta numeric,
* allele strings are compatible,
* duplicates are resolved,
* palindromic SNP policy is explicit,
* genome build matches platform/reference.

---

## Genomics issue: ambiguous SNP handling is incomplete for DTC inference

The code has helper logic for ambiguous A/T and C/G SNPs and an optional `exclude_ambiguous` path. The allele matching function can complement alleles and flip dosages.  

That is good, but it does not solve browser-side inference unless the exported model carries the exact orientation and the user genotype converter applies it. Palindromic SNPs are particularly dangerous in DTC data because the genotype string alone may not tell you strand. A/T and C/G variants should either be excluded unless allele frequency disambiguates them safely, or retained only when platform manifest strand and reference build are known.

---

## Evaluation issue: metrics may be over-optimistic

Internal CV is useful, but for your use case I would require separate validation layers:

1. **Reference-panel internal CV**: checks whether linear models approximate held-out reference genotypes.
2. **Held-out reference cohort**: different samples, same ancestry.
3. **Cross-ancestry validation**: because LD patterns and allele frequencies change.
4. **Platform-realistic masking validation**: mask reference data to 23andMe v5 or Ancestry v2 and compare predicted PRS to full/imputed PRS.
5. **Raw DTC parser validation**: take known genotypes, export model, run browser/JS scoring, and assert exact agreement with Python scoring.

Right now, because the Python training path and user prediction path use different allele semantics, internal evaluation can pass while actual user scoring fails.

---

## Browser deployment concerns

The README emphasizes portable exports and browser-style use cases. ([GitHub][3]) The current export schema shown in `json_export.py` is oriented around observed and imputed variants, but not enough allele metadata for safe raw genotype conversion.  I also did not find an equivalent projection exporter in the inspected code path; the projection class has prediction and properties, while the export code shown serializes imputation models.

For browser deployment, the model artifact needs to be more than coefficients. It needs:

```text
model metadata:
  PRS ID
  platform ID
  genome build
  ancestry/reference panel
  training reference version
  score centering/scaling parameters

observed PRS terms:
  locus
  effect allele
  other allele
  beta
  accepted IDs
  strand/build policy

predictor terms:
  locus
  counted allele used by the model
  other allele
  coefficient
  mean/scale if standardized
  allele frequency
  ambiguity policy

calibration/error:
  intercept
  slope
  empirical residual SD
  validation ancestry/cohort
  warnings for out-of-scope ancestry/platform
```

Without that, JavaScript cannot reliably transform a 23andMe genotype string into the same numeric vector that Python used during model training.

---

## Linear imputation vs linear projection

For your stated product goal, I would prioritize **linear projection** after fixing allele orientation and export.

Linear imputation has a conceptual appeal because it approximates missing variant dosages, but it pays a large multiple-testing/modeling cost: one model per missing PRS variant, then a weighted sum. Errors across variants are correlated, and uncertainty accounting is hard.

Linear projection directly trains the target you care about: the missing PRS component in a local region. Your docs describe this as avoiding per-variant error accumulation and being more efficient for large PRS models. ([GitHub][1]) The projection code also avoids dosage clipping because the target is a PRS contribution, not a genotype dosage, which is conceptually correct. 

However, the projection implementation currently loses per-PRS-variant allele metadata in the model object/evaluator, and it lacks the robust export schema needed for browser use. Fix those before relying on projection validation.

---

## Recommended fix plan

### 1. Redesign the genotype representation

Do not convert user genotypes to scalar dosages until the counted allele is known.

Use something like:

```python
@dataclass
class RawUserGenotype:
    variant_id: str
    chromosome: str | None
    position: int | None
    genotype: str  # "AA", "AG", etc.
    source_build: str | None
```

Then convert with:

```python
def count_allele(genotype: str, counted_allele: str) -> float | None:
    # "AA", counted A -> 2
    # "AG", counted A -> 1
    # "GG", counted A -> 0
```

This same logic must exist in JS and must be tested against Python.

### 2. Store predictor allele orientation in every trained model

Every coefficient should know what allele its input dosage counts. For imputation and projection predictors, export:

```text
variant_id
chromosome
position
ref_allele
alt_allele
counted_allele
training_mean
training_sd
allele_frequency
coefficient
```

For observed PRS variants, export:

```text
effect_allele
other_allele
beta
```

### 3. Make observed variant inclusion allele-aware

A variant should not be marked observed only because the rsID or position is on the platform. It should be observed only if the platform/user genotype can be converted to effect-allele dosage.

### 4. Standardize ElasticNet predictors

Fit on standardized predictors and either export means/scales or back-transform coefficients. This makes alpha comparable across regions and avoids MAF-dependent penalty artifacts.

### 5. Fix missing predictor handling

For imputation, replace all-or-nothing fallback with mean substitution or a reduced model. For uncertainty, estimate missingness-pattern-adjusted residual variance. For projection, increase uncertainty when many predictors are mean-substituted.

### 6. Replace diagonal SE with empirical residual calibration

For deployment, report an empirical approximation error from held-out data. For example:

```text
PRS estimate: X
Approximation error SD: Y
Validation population: 1000G EUR, 23andMe v5 mask
Correlation with full PRS: Z
Top-decile concordance: W
```

Avoid presenting nominal genetic “confidence intervals” as if they were clinical uncertainty intervals.

### 7. Add golden tests before more modeling work

Minimum tests I would add immediately:

| Test                           | Expected behavior                                                     |
| ------------------------------ | --------------------------------------------------------------------- |
| Effect allele is ALT           | ALT homozygote = 2, REF homozygote = 0                                |
| Effect allele is REF           | REF homozygote = 2, ALT homozygote = 0                                |
| Heterozygote                   | dosage = 1 regardless of allele order                                 |
| Palindromic A/T, C/G           | excluded or resolved only by explicit policy                          |
| Strand complement              | correct flip or explicit exclusion                                    |
| User raw 23andMe genotype      | Python and JS produce identical dosages                               |
| Observed PRS-only score        | browser score equals hand calculation                                 |
| One missing predictor          | prediction uses remaining information or documented mean substitution |
| Projection true PRS evaluation | effect=REF and multiallelic cases score correctly                     |
| JSON round trip                | exported model gives identical Python prediction after reload         |

---

## Severity-ranked issue list

| Severity   | Area                  | Problem                                                                         | Fix                                                                     |
| ---------- | --------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Critical   | User inference        | `genotype_to_dosage()` counts homozygosity, not effect/counted allele dosage    | Parse raw genotype first; count model-specific allele later             |
| Critical   | Browser export        | Predictor allele orientation is not exported                                    | Store counted allele, ref/alt, build, strand policy for every predictor |
| Critical   | Observed PRS          | Observed variants are multiplied by beta without allele-aware dosage conversion | Make observed scoring use effect-allele dosage                          |
| High       | Projection evaluation | True PRS for region variants can use wrong allele row                           | Store PRS variant alleles in region model or evaluator                  |
| High       | Uncertainty           | Residual covariance ignored; SE likely optimistic                               | Use empirical CV residuals/block covariance                             |
| High       | Calibration           | Internal CV calibration only; `NaN -> 0` can bias true/CV PRS                   | Use held-out validation and proper missing handling                     |
| High       | Hyperparameter tuning | Tunes full-chip model but trains local-window model                             | Tune on local windows and sampled variants                              |
| Medium     | ElasticNet            | No predictor standardization                                                    | Store means/scales or back-transform coefficients                       |
| Medium     | Missing predictors    | Imputation falls back to intercept if any predictor missing                     | Mean substitution or reduced-model prediction                           |
| Medium     | PRS loading           | `ALT` mapped to effect allele                                                   | Require explicit scoring-file schema                                    |
| Medium     | Export                | Projection export appears missing; JSON is imputation-centric                   | Add projection JSON schema and loader                                   |
| Low/Medium | Diagnostics           | Trainers catch broad exceptions and drop failures                               | Collect failure reasons and expose them                                 |

---

## Bottom line

The library’s statistical concept is worth pursuing, especially the projection approach. But the current implementation has a fundamental allele-orientation break between reference training and user prediction. Until that is fixed, PRS values from raw 23andMe/Ancestry-style uploads can be systematically wrong even when cross-validation metrics look strong.

I would pause model refinement and first rebuild the data model around allele-aware, build-aware, exportable variant representations. After that, re-run validation from scratch, with Python-vs-browser golden tests as a hard gate.

[1]: https://github.com/multicancerrisk/imputed-prs/tree/main/docs/statistical-theory "imputed-prs/docs/statistical-theory at main · multicancerrisk/imputed-prs · GitHub"
[2]: https://github.com/multicancerrisk/imputed-prs/blob/main/docs/statistical-theory/linear-imputation.md "imputed-prs/docs/statistical-theory/linear-imputation.md at main · multicancerrisk/imputed-prs · GitHub"
[3]: https://github.com/multicancerrisk/imputed-prs "GitHub - multicancerrisk/imputed-prs: A library that facilitates the calculation of a specified Polygenic Risk Score (PRS) on data from a fixed target genotyping platform while accounting for missing variants · GitHub"
