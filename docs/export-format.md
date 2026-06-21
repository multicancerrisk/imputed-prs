# Export format: the v2 browser-deployable JSON artifact

This is a reference for the JSON model artifact emitted by `imputed_prs`, aimed at
whoever writes the client-side (TypeScript) scorer. It documents the exact field
names, what each field means, and what a deployable artifact is required to contain.

Both methods emit the **same v2.0 JSON family**:

- **Imputation** — `imputed_prs.io.exporters.json_export.export_to_json`, written by
  `LinearImputationPRS.export(..., formats=["json"])`.
- **Projection** — `imputed_prs.io.exporters.projection_json_export.export_projection_to_json`,
  written by `LinearProjectionPRS.export(...)` (JSON is the only projection format).

The two differ in exactly one structural way: the imputation artifact carries a flat
`imputed_variants` list (one reconstructed PRS variant each), while the projection
artifact carries `region_models` (one linear map per genomic region). Everything else
— the `metadata`/`provenance` blocks, the `observed_variants` shape, the
self-describing predictor objects, calibration — is shared.

Machine-readable JSON Schemas (strict, closed contracts: `additionalProperties: false`)
live at
[`schemas/imputation_model_v2.schema.json`](../schemas/imputation_model_v2.schema.json)
for the imputation artifact and
[`schemas/projection_model_v2.schema.json`](../schemas/projection_model_v2.schema.json)
for the projection artifact. Exported models are validated against these schemas in the
test suite.

Design principle: **every object a scorer needs to count from the user's upload
carries its own allele metadata.** A client must orient each raw genotype to the
allele a coefficient/beta counts — it must never trust positional alignment across
arrays or assume "effect allele == ALT".

---

## Top-level layout

```jsonc
{
  "metadata":        { ... },          // identity + counts; format_version "2.0"
  "provenance":      { ... },          // compatibility metadata the scorer validates
  "observed_variants": [ ... ],        // PRS variants present on the platform
  "imputed_variants":  [ ... ],        // IMPUTATION ONLY: reconstructed PRS variants
  "region_models":     [ ... ],        // PROJECTION ONLY: per-region linear maps
  "platform_variant_index": { ... },   // variant_id -> index into observed_variants
  "calibration_params": { ... },       // optional; present when the model is calibrated
  "evaluation_metrics": { ... },       // IMPUTATION ONLY, optional
  "training_summary":   { ... }        // optional, free-form
}
```

| Key | Imputation | Projection | Required | Notes |
|-----|:---------:|:----------:|:--------:|-------|
| `metadata` | yes | yes | yes | |
| `provenance` | yes | yes | yes | the block the deployed scorer reads |
| `observed_variants` | yes | yes | yes | may be empty |
| `imputed_variants` | yes | — | yes (imputation) | flat list of reconstructed variants |
| `region_models` | — | yes | yes (projection) | per-region linear maps |
| `platform_variant_index` | yes | yes | yes | `{ variant_id: int }` into `observed_variants` |
| `calibration_params` | optional | optional | no | present iff the model was calibrated |
| `evaluation_metrics` | optional | — | no | projection stores none |
| `training_summary` | optional | optional | no | free-form `object` |

`format_version` is the string `"2.0"` (note: a JSON string, not a number) and is the
first thing a loader should check.

> The legacy v1.0 keys are retained inside `metadata` for continuity, but the browser
> should read identity/compatibility from the richer `provenance` block.

---

## `metadata`

Identity and counts. Most fields are nullable (the deploy gate lives in `provenance`,
not here).

Shared fields:

| Field | Type | Notes |
|-------|------|-------|
| `format_version` | string | always `"2.0"` |
| `created_at` | string | ISO-8601 UTC timestamp, `Z`-suffixed |
| `model_name` | string \| null | |
| `prs_id` | string \| null | e.g. `"PGS000004"` |
| `platform_name` | string \| null | e.g. `"23andme_v5"` |
| `genome_build` | string \| null | e.g. `"GRCh37"` |
| `include_variance_scaling` | boolean | whether per-variant variance/SE components were emitted |

Method-specific counts:

| Field | Method | Type |
|-------|--------|------|
| `n_observed_variants` | both | integer |
| `n_imputed_variants` | imputation | integer |
| `n_intercept_only` | imputation | integer (imputed models with no predictors) |
| `n_region_models` | projection | integer |
| `n_intercept_only_regions` | projection | integer |

---

## `provenance`

The compatibility block. The deployed scorer reads this to decide whether a user's
upload is safe to score. **All six fields are always present** (the keys are required),
though some values may be `null` for a non-deployable research export.

| Field | Type | Meaning |
|-------|------|---------|
| `genome_build` | string \| null | build the model was trained on; the scorer must reject/flag an upload from a different build |
| `platform_id` | string \| null | the platform the model targets (this is the platform name; there is no separate platform-id concept) |
| `reference_panel_id` | string \| null | reference panel used for training, e.g. `"1000G_phase3_EUR"` |
| `training_ancestry` | string \| null | ancestry of the training cohort, e.g. `"EUR"` (estimates degrade off-ancestry) |
| `ambiguous_policy` | string | how the scorer must handle palindromic (A/T, C/G) SNPs; default `"exclude_unless_platform_strand_known"` |
| `centering_scaling` | object \| null | the calibration object (same shape as `calibration_params`), or null if uncalibrated |

For a **deployable** export, `genome_build`, `reference_panel_id`, and
`training_ancestry` must all be non-null — the Python exporter raises otherwise (see
[Deploy-time requirements](#deploy-time-requirements)).

---

## `observed_variants` (both methods)

PRS variants that are **directly present on the platform**. The scorer counts the
user's effect-allele dosage for each.

```jsonc
{
  "variant_id":   "rs123",
  "chromosome":   "1",
  "position":     12345,
  "effect_allele":"A",
  "other_allele": "G",
  "beta":         0.12,
  "accepted_ids": ["rs123", "1:12345"],
  "ambiguous":    false,
  "fallback":     null
}
```

| Field | Type | Notes |
|-------|------|-------|
| `variant_id` | string | |
| `chromosome` | string | `"1"`–`"22"`, `"X"`, `"Y"`, `"MT"` |
| `position` | integer | |
| `effect_allele` | string | the allele `beta` is oriented to (the one to count) |
| `other_allele` | string | the complementary biallelic allele — **required for orientation** |
| `beta` | number | effect size (log-OR or beta) |
| `accepted_ids` | string[] | every id that should match this variant in a user file; currently `[variant_id, "chr:pos"]`. Try them all when resolving an upload. |
| `ambiguous` | boolean | true iff `effect_allele`/`other_allele` form a palindrome (A/T or C/G). Apply `provenance.ambiguous_policy`. |
| `fallback` | object \| null | optional per-variant fallback model — same shape as an imputed-variant entry (see below). Used to recover this variant when the upload cannot resolve/call it directly, so it is not silently dropped. |

The `fallback` object, when present, is an **imputed-variant model** (next section)
that predicts *this observed variant's effect-allele dosage* from local platform
predictors. A scorer should use it only when the variant cannot be read directly from
the upload.

---

## `imputed_variants` (imputation only)

One entry per missing PRS variant the model reconstructs. The variant's dosage is
predicted from `predictors`, then multiplied by `beta`.

```jsonc
{
  "variant_id":      "rs999",
  "chromosome":      "2",
  "position":        67890,
  "effect_allele":   "T",
  "other_allele":    "C",
  "beta":            -0.08,
  "allele_frequency":0.31,
  "imputation_r2":   0.84,
  "intercept":       0.62,
  "is_intercept_only": false,
  "predictors":      [ /* predictor objects */ ],
  "residual_variance": 0.05   // present only when include_variance_scaling=true
}
```

| Field | Type | Notes |
|-------|------|-------|
| `variant_id` | string | |
| `chromosome` | string | |
| `position` | integer | |
| `effect_allele` | string | the allele the prediction (and `beta`) is oriented to |
| `other_allele` | string \| null | may be null for the imputed *target* (it is predicted, not counted), so it is **not** deploy-gated |
| `beta` | number | |
| `allele_frequency` | number | effect-allele frequency; `2 * AF` is the mean dosage used when no predictors resolve |
| `imputation_r2` | number | cross-validated reconstruction quality |
| `intercept` | number | regression intercept (the predicted dosage for an intercept-only model) |
| `is_intercept_only` | boolean | true when there are no usable predictors / all coefficients shrank to zero |
| `predictors` | predictor[] | platform variants feeding this model (see below); empty for intercept-only |
| `residual_variance` | number | only emitted when `include_variance_scaling=true`; feeds the uncertainty interval |

### `predictor` objects (shared by imputed models, fallbacks, and projection regions)

Each predictor is **self-describing**: it carries the allele its coefficient counts so
the scorer never relies on cross-array index alignment.

```jsonc
{
  "variant_id":      "rs555",
  "chromosome":      "2",
  "position":        67000,
  "counted_allele":  "A",
  "other_allele":    "G",
  "allele_frequency":0.22,
  "coefficient":     0.41
}
```

| Field | Type | Notes |
|-------|------|-------|
| `variant_id` | string | |
| `chromosome` | string | |
| `position` | integer | |
| `counted_allele` | string | the allele whose dosage this coefficient multiplies (= ALT of the backing reference row) |
| `other_allele` | string | the complementary allele (= REF) — **required**; together with chr/pos it pins the exact reference row, disambiguating multiallelic loci |
| `allele_frequency` | number | frequency of `counted_allele`; mean dosage `2 * AF` is substituted when the predictor is missing from the upload |
| `coefficient` | number | weight applied to the predictor's `counted_allele` dosage |

---

## `region_models` (projection only)

Each entry is one genomic region's linear map: `predictors` (platform variants, with
allele metadata) combine via `coefficients` + `intercept` to approximate that region's
PRS contribution. `prs_variants` records the true PRS variants the region stands in
for (carried for transparency/auditing; they are **projected**, not counted from the
upload).

```jsonc
{
  "region_id":   "chr2:60000000-61000000",
  "chromosome":  "2",
  "start":       60000000,
  "end":         61000000,
  "intercept":   0.30,
  "cv_mse":      0.012,
  "cv_r2":       0.79,
  "is_intercept_only": false,
  "mean_prs_contribution": 0.31,
  "target_variance": 0.044,
  "predictors": [ /* predictor objects, same shape as above */ ],
  "prs_variants": [
    {
      "variant_id":   "rs777",
      "chromosome":   "2",       // denormalized from the region (single-chromosome)
      "position":     60500000,
      "effect_allele":"G",
      "other_allele": "A",       // may be null if the PRS source lacked it
      "beta":         0.05
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `region_id` | string | `"chr{chrom}:{start}-{end}"`; do not parse it — use the explicit fields below |
| `chromosome` | string | regions are single-chromosome by construction |
| `start` / `end` | integer | region span (bp) |
| `intercept` | number | region-model intercept (= mean PRS contribution for an intercept-only region) |
| `cv_mse` | number | cross-validated MSE of this region's PRS-contribution prediction |
| `cv_r2` | number | cross-validated R² for the region |
| `is_intercept_only` | boolean | true when no predictors / all coefficients shrank to zero |
| `mean_prs_contribution` | number | mean of the region target across training samples |
| `target_variance` | number | variance of the region target across reference samples; the error variance of predicting with the regional mean, used to inflate the uncertainty interval as predictors go missing |
| `predictors` | predictor[] | self-describing predictor objects (same shape as imputation) |
| `prs_variants` | object[] | the region's true PRS variants (id, chromosome, position, `effect_allele`, `other_allele` (nullable), `beta`); projected, so **not** deploy-gated |

The exporter validates that all `predictor_*` arrays and all `prs_*` arrays in a region
are index-aligned, and serializes them into these aligned objects.

---

## `calibration_params` and empirical-error fields

The same calibration object appears both at top level (`calibration_params`) and inside
`provenance.centering_scaling`. It is present only when the model was calibrated.

| Field | Type | Meaning |
|-------|------|---------|
| `scaling_factor` | number | slope from regressing the true PRS on the CV-predicted PRS |
| `scaling_factor_se` | number | standard error of `scaling_factor` |
| `calibration_intercept` | number | intercept of the calibration regression |
| `calibration_r2` | number | R² of the calibration fit |
| `sd_cv_predicted` | number | SD of the CV-predicted PRS |
| `sd_true` | number | SD of the true PRS |
| `sd_scaled` | number | SD of the scaled predictions |
| `attenuation_factor` | number | `sd_cv_predicted / sd_true` |
| `n_calibration` | integer | sample size used for calibration |
| `raw_empirical_residual_sd` | number \| null | empirical score-level residual SD on the **raw** scale, `std(s_true - s_cv)` out-of-fold — the honest SD for the raw interval (captures full LD-aware `betaᵀ Σ beta` residual covariance). `null` on pre-P4.1 artifacts. |
| `calibrated_empirical_residual_sd` | number \| null | empirical residual SD after the calibration transform — the SD for the scaled interval. `null` on pre-P4.1 artifacts. |
| `diagonal_model_se_lower_bound` | number \| null | the **full-data** (no-missingness) diagonal SE measured at fit time — imputation `sqrt(Σ beta² · residual_var)`, projection `sqrt(Σ cv_mse)`. A reference/QC lower bound. `null` on pre-P4.1 artifacts. |

> **Interval guidance for the scorer.** The reported SE should be
> `max(empirical_residual_sd, per_prediction_diagonal_SE)`: the empirical SD is the
> LD-aware panel-wide baseline, and the diagonal value (recomputed per user, inflated
> as predictors go missing) is a genuine lower bound that becomes binding under heavy
> upload missingness. The fit-time `diagonal_model_se_lower_bound` above is a *full-data
> reference value* — it is **not** the per-prediction quantity and must not be used
> directly as the user's interval.

---

## `evaluation_metrics` (imputation only, optional)

Held-out evaluation metrics, when the model was trained with an evaluation set.

| Field | Type |
|-------|------|
| `correlation` | number |
| `r2` | number |
| `mae` | number |
| `rmse` | number |
| `spearman_rho` | number |
| `calibration_slope` | number |
| `calibration_intercept` | number |

---

## Deploy-time requirements

The Python exporters gate a *deployable* artifact by default (`require_other_allele=True`,
`require_provenance=True`). A scorer can rely on the following for any artifact produced
with the defaults:

1. **`other_allele` is present on every scored variant.** That means every
   `observed_variants[]` entry and every `predictors[]` entry (including predictors
   inside a `fallback`). Export raises if any is missing it. The imputed *target*
   allele and region `prs_variants` are **predicted/projected, not counted**, so they
   are exempt and `other_allele` there may legitimately be `null`.
2. **Provenance is present.** `genome_build`, `reference_panel_id`, and
   `training_ancestry` are all non-null. The scorer uses build + platform to validate
   compatibility before scoring.
3. **Ambiguous SNPs are excluded by default at deploy time.** Training keeps
   palindromic variants (so they round-trip in the artifact and carry an `ambiguous`
   flag), but `provenance.ambiguous_policy` instructs the scorer to drop them unless the
   platform strand is known (`"exclude_unless_platform_strand_known"`).

Passing `require_other_allele=False` / `require_provenance=False` to the exporter
produces a **non-deployable research export** that may omit these — a TypeScript scorer
should refuse such an artifact (or treat it as untrusted).

### Scoring an upload, in brief

1. Check `metadata.format_version == "2.0"` and validate `provenance` (build, platform).
2. For each `observed_variants[]` entry: resolve the user genotype by any `accepted_ids`,
   orient to `effect_allele` using `other_allele`, count effect-allele dosage, multiply
   by `beta`. If unresolved and a `fallback` exists, reconstruct via the fallback.
3. **Imputation:** for each `imputed_variants[]` entry, predict the dosage from its
   `predictors` (orient each by its `counted_allele`/`other_allele`; substitute `2*AF`
   for a missing predictor), add `intercept`, multiply by `beta`.
   **Projection:** for each `region_models[]` entry, combine `predictors` (oriented,
   `2*AF` for missing) with `coefficients` and `intercept` to get the region's
   contribution directly.
4. Sum all contributions. Apply `calibration_params` (`calibration_intercept + scaling_factor * raw`)
   for the calibrated score, and form the interval per the SE guidance above.
