"""Layered masking-validation harness (P4.3).

Internal cross-validation answers "how well does the model predict its own held-out
fold?" — it cannot tell you what a *real platform upload* will score, because the
user only ever has the genotyping chip's variants. This module closes that gap.

The harness takes a reference panel, **masks it down to a platform's variants**
(blanks every off-platform variant, exactly as if the user only ran a 23andMe /
AncestryDNA chip), scores the masked panel **through the deployed library path**, and
compares that estimate to the **full PRS** computed from the complete panel. It emits
a consolidated metric panel: correlation / R² / Spearman, top-decile concordance,
the empirical score-level approximation error, and the deployed interval's coverage.

It reuses the already-validated, allele-oriented scorers — both evaluators expose
``compute_score_arrays`` (P1.6 keeps eval and the browser path in lock-step) — so the
harness is orchestration, not a third scoring path. It also runs a **raw-parser
round-trip**: a handful of masked samples are written to a synthetic 23andMe file and
scored through the public ``predict`` (file → parser → multi-key resolution → oriented
scoring), asserting agreement with the batch estimate.

Important framing (see :data:`_CALIBRATION_CAVEAT`): these metrics measure the
approximation error of the masked estimate against the *full computed PRS*, on the
same population. They are **not** external/clinical calibration. Cross-ancestry runs
are *structured for* (pass ``evaluation_ancestry``; the report records it and emits a
caveat when it differs from the training ancestry) — a genuine cross-ancestry run
means handing this function a different-ancestry reference panel.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import tempfile

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.harmonizer import partition_variants
from imputed_prs.core.types import GenotypeData, PlatformInfo
from imputed_prs.evaluation.metrics import (
    compute_percentile_concordance,
    compute_prs_metrics,
)
from imputed_prs.evaluation._scoring import is_hard_called
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.platform_loader import (
    load_platform_from_manifest,
    load_platform_from_name,
    load_platform_variants_from_list,
)
from imputed_prs.io.user_genotypes import render_genotype_string

# Minimum valid (non-NaN-pair) samples to compute accuracy metrics / concordance.
# Mirrors the floors enforced inside the metric helpers (3 for correlation, 20 for
# percentile concordance) so the harness can degrade gracefully instead of raising.
_MIN_SAMPLES_METRICS = 3
_MIN_SAMPLES_CONCORDANCE = 20

# Always-present caveat: what these numbers do and do not mean.
_CALIBRATION_CAVEAT = (
    "Internal calibration is NOT external calibration: the model's calibration "
    "corrects imputation/projection attenuation against the same population's full "
    "PRS. It does not calibrate to disease risk or to an external cohort. These "
    "metrics measure the approximation error of the masked-platform estimate vs the "
    "full computed PRS, not clinical validity."
)


@dataclass
class MaskingValidationReport:
    """The metric panel from :func:`run_masking_validation`.

    One method-agnostic report for both products (``model_type`` records which).
    Floats are ``nan`` when a quantity is undefined (e.g. a degenerate, zero-variance
    cohort); see :attr:`caveats` for why.
    """

    # --- Metadata / provenance ------------------------------------------------
    model_type: str
    prs_id: Optional[str]
    platform_id: Optional[str]
    genome_build: Optional[str]
    reference_panel_id: Optional[str]
    training_ancestry: Optional[str]
    evaluation_ancestry: Optional[str]
    n_samples: int
    n_reference_variants: int
    n_platform_variants_retained: int
    n_variants_masked: int
    n_observed: int
    n_imputed_or_regions: int

    # --- Accuracy (batch, all samples) ---------------------------------------
    correlation: float
    r2: float
    spearman_rho: float
    mae: float
    rmse: float
    calibration_slope: float
    calibration_intercept: float

    # --- Concordance (batch) -------------------------------------------------
    percentile_concordance: Dict[str, float]
    top_decile_concordance: float

    # --- Empirical approximation error (batch) -------------------------------
    empirical_error_sd: float
    empirical_error_mean: float

    # --- Interval coverage (predict() subsample) -----------------------------
    coverage_n: int
    coverage_95: Optional[float]
    mean_se: Optional[float]

    # --- Raw-parser round-trip (predict() file subsample) --------------------
    raw_parser_checked: bool
    raw_parser_n: int
    raw_parser_max_abs_diff: Optional[float]
    raw_parser_agrees: Optional[bool]

    # --- Cross-ancestry / caveats --------------------------------------------
    cross_ancestry: bool
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (for notebook/CI artifacts)."""
        from dataclasses import asdict

        return asdict(self)

    def summary(self) -> str:
        """A formatted, human-readable metric panel."""

        def fmt(x: Optional[float]) -> str:
            if x is None:
                return "n/a"
            return f"{x:.4f}" if np.isfinite(x) else "nan"

        lines = [
            "Masking validation — metric panel",
            "=" * 38,
            f"model            : {self.model_type}",
            f"prs / platform   : {self.prs_id} / {self.platform_id}",
            f"build / ancestry : {self.genome_build} / train={self.training_ancestry} "
            f"eval={self.evaluation_ancestry}"
            + ("  [CROSS-ANCESTRY]" if self.cross_ancestry else ""),
            f"samples          : {self.n_samples}",
            f"variants         : {self.n_reference_variants} reference, "
            f"{self.n_platform_variants_retained} retained, {self.n_variants_masked} masked",
            f"prs terms        : {self.n_observed} observed, "
            f"{self.n_imputed_or_regions} imputed/region",
            "-" * 38,
            f"correlation (r)  : {fmt(self.correlation)}   r2: {fmt(self.r2)}",
            f"spearman rho     : {fmt(self.spearman_rho)}",
            f"MAE / RMSE       : {fmt(self.mae)} / {fmt(self.rmse)}",
            f"calibration      : slope={fmt(self.calibration_slope)} "
            f"intercept={fmt(self.calibration_intercept)}",
            f"top-decile conc. : {fmt(self.top_decile_concordance)}",
            f"empirical error  : sd={fmt(self.empirical_error_sd)} "
            f"bias={fmt(self.empirical_error_mean)}",
            f"95% coverage     : {fmt(self.coverage_95)} (n={self.coverage_n}, "
            f"mean_se={fmt(self.mean_se)})",
            f"raw-parser check : "
            + (
                f"agrees={self.raw_parser_agrees} "
                f"(max|diff|={fmt(self.raw_parser_max_abs_diff)}, n={self.raw_parser_n})"
                if self.raw_parser_checked
                else "skipped"
            ),
        ]
        if self.caveats:
            lines.append("-" * 38)
            lines.append("caveats:")
            lines.extend(f"  - {c}" for c in self.caveats)
        return "\n".join(lines)


def mask_reference_to_platform(
    genotype_data: GenotypeData,
    platform_variant_ids: Set[str],
    *,
    platform_info: Optional[PlatformInfo] = None,
) -> GenotypeData:
    """Blank every off-platform variant, simulating a platform-only upload.

    On-platform columns keep their true dosages; off-platform columns are set to
    ``NaN`` across all samples — exactly how the scorers already treat a missing
    off-platform variant. Columns are **blanked, not dropped**, so ``variant_info``
    stays intact for chr:pos/allele re-resolution.

    On-platform membership is decided with :func:`partition_variants` (rsID-first,
    then chr:pos), the same matching the rest of the library uses — a naive
    ``variant_id in set`` test would miss chr:pos-matched variants.

    Args:
        genotype_data: The full reference panel.
        platform_variant_ids: Platform variant IDs (rsIDs and/or ``chr:pos``).
        platform_info: Optional platform metadata (unused for masking; accepted so
            callers can thread it through without a second lookup).

    Returns:
        A new ``GenotypeData`` with off-platform columns set to ``NaN``.
    """
    partition = partition_variants(genotype_data.variant_info, set(platform_variant_ids))
    keep_mask = (
        genotype_data.variant_info["variant_id"].isin(partition.observed).to_numpy()
    )

    masked = genotype_data.dosage_matrix.copy()
    masked[:, ~keep_mask] = np.nan

    return GenotypeData(
        dosage_matrix=masked,
        variant_info=genotype_data.variant_info.copy(),
        sample_ids=list(genotype_data.sample_ids),
        genome_build=genotype_data.genome_build,
        source_file=genotype_data.source_file,
    )


def run_masking_validation(
    model: Any,
    reference_genotypes: Union[str, Path, GenotypeData],
    *,
    platform_name: Optional[str] = None,
    platform_manifest: Optional[Union[str, Path]] = None,
    platform_variants: Optional[Sequence[str]] = None,
    evaluation_ancestry: Optional[str] = None,
    max_samples: Optional[int] = None,
    coverage_max_samples: int = 200,
    raw_parser_max_samples: int = 50,
    run_raw_parser_check: bool = True,
    percentile_thresholds: Sequence[int] = (1, 5, 10),
    random_state: Optional[int] = None,
    verbose: int = 1,
) -> MaskingValidationReport:
    """Mask a reference panel to a platform, score it, and compare to the full PRS.

    Works for a fitted ``LinearImputationPRS`` or ``LinearProjectionPRS``. The true
    PRS is computed on the **full** panel; the estimate on the **masked** panel — that
    asymmetry is what makes this a real masking test rather than a self-comparison.

    Args:
        model: A fitted imputation or projection model.
        reference_genotypes: Reference panel (path or pre-loaded ``GenotypeData``).
        platform_name / platform_manifest / platform_variants: Optional platform
            override (at most one). When omitted, the model's declared platform is
            used; if the model declares none, the platform footprint is reconstructed
            from the variants the model actually uses (observed + predictors).
        evaluation_ancestry: Ancestry label for the validation cohort. Recorded; a
            caveat is emitted if it differs from the model's training ancestry.
        max_samples: Cap the batch panel to this many (deterministically subsampled)
            samples. ``None`` uses all samples.
        coverage_max_samples: Subsample size run through public ``predict`` for the
            interval-coverage check (DataFrame path).
        raw_parser_max_samples: Subsample size for the file-based raw-parser check.
        run_raw_parser_check: Whether to run the DTC file round-trip.
        percentile_thresholds: Percentiles for concordance (default 1/5/10; 10 is the
            headline top-decile).
        random_state: Seed for deterministic subsampling.
        verbose: ``>=1`` prints the metric panel at the end.

    Returns:
        A :class:`MaskingValidationReport`.
    """
    # Lazy imports of the model/evaluator classes to avoid an import cycle
    # (evaluation is imported by the core model modules).
    from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
    from imputed_prs.core.linear_projection_prs import LinearProjectionPRS
    from imputed_prs.evaluation.evaluator import ImputationEvaluator
    from imputed_prs.evaluation.projection_evaluator import ProjectionEvaluator

    if not getattr(model, "is_fitted", False):
        raise ModelNotFittedError("run_masking_validation requires a fitted model.")

    caveats: List[str] = [_CALIBRATION_CAVEAT]

    # 1. Dispatch on model type.
    if isinstance(model, LinearImputationPRS):
        evaluator = ImputationEvaluator(model, verbose=0)
        model_type = "imputation"
        n_imputed_or_regions = len(model.imputed_models)
    elif isinstance(model, LinearProjectionPRS):
        evaluator = ProjectionEvaluator(model, verbose=0)
        model_type = "projection"
        n_imputed_or_regions = len(model.region_models)
    else:
        raise ValidationError(
            "model must be a fitted LinearImputationPRS or LinearProjectionPRS, "
            f"got {type(model).__name__}"
        )

    # 2. Resolve the platform variant set + provenance label.
    platform_set, platform_info, platform_label, platform_caveat = _resolve_platform(
        model, platform_name, platform_manifest, platform_variants
    )
    if platform_caveat:
        caveats.append(platform_caveat)

    # 3. Load the full reference (no variant filter — we need the whole panel for the
    #    true PRS and to know what to mask) and deterministically subsample.
    if isinstance(reference_genotypes, GenotypeData):
        full_gd = reference_genotypes
    else:
        full_gd = load_genotypes(path=reference_genotypes)

    full_gd = _maybe_subsample(full_gd, max_samples, random_state)
    n_samples = full_gd.n_samples

    # 4. True PRS on the FULL panel; estimate on the MASKED panel.
    _, s_true = evaluator.compute_score_arrays(full_gd)
    masked_gd = mask_reference_to_platform(full_gd, platform_set, platform_info=platform_info)
    s_est, _ = evaluator.compute_score_arrays(masked_gd)

    retained_ids = partition_variants(full_gd.variant_info, platform_set).observed
    n_retained = int(full_gd.variant_info["variant_id"].isin(retained_ids).sum())
    n_masked = full_gd.n_variants - n_retained
    if n_masked == 0:
        caveats.append(
            "No off-platform variants were masked (every PRS variant is on-platform): "
            "this run does not exercise imputation/projection."
        )

    # 5. Accuracy + concordance + empirical error (batch).
    metrics, conc, emp_sd, emp_mean, metric_caveats = _batch_metrics(
        s_est, s_true, list(percentile_thresholds)
    )
    caveats.extend(metric_caveats)

    # 6. Interval coverage + raw-parser round-trip (small predict() subsamples).
    coverage_n, coverage_95, mean_se, cov_caveat = _interval_coverage(
        model, masked_gd, s_true, coverage_max_samples
    )
    if cov_caveat:
        caveats.append(cov_caveat)

    (
        raw_checked,
        raw_n,
        raw_max_diff,
        raw_agrees,
        raw_caveat,
    ) = _raw_parser_round_trip(
        model, masked_gd, s_est, raw_parser_max_samples, run_raw_parser_check
    )
    if raw_caveat:
        caveats.append(raw_caveat)

    # 7. Cross-ancestry note.
    training_ancestry = getattr(model, "_training_ancestry", None)
    cross_ancestry = bool(
        evaluation_ancestry
        and training_ancestry
        and evaluation_ancestry.strip().lower() != training_ancestry.strip().lower()
    )
    if cross_ancestry:
        caveats.append(
            f"Cross-ancestry evaluation: training_ancestry={training_ancestry} but "
            f"evaluation_ancestry={evaluation_ancestry}. LD patterns and allele "
            "frequencies differ across ancestries, so imputation/projection accuracy "
            "and mean-substitution (2*AF fills) are expected to degrade. Treat these "
            "metrics as a lower bound on within-ancestry performance."
        )

    report = MaskingValidationReport(
        model_type=model_type,
        prs_id=getattr(model, "_prs_id", None),
        platform_id=platform_label,
        genome_build=getattr(model, "_genome_build", None),
        reference_panel_id=getattr(model, "_reference_panel_id", None),
        training_ancestry=training_ancestry,
        evaluation_ancestry=evaluation_ancestry,
        n_samples=n_samples,
        n_reference_variants=full_gd.n_variants,
        n_platform_variants_retained=n_retained,
        n_variants_masked=n_masked,
        n_observed=len(model.observed_variants),
        n_imputed_or_regions=n_imputed_or_regions,
        correlation=metrics["correlation"],
        r2=metrics["r2"],
        spearman_rho=metrics["spearman_rho"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        calibration_slope=metrics["calibration_slope"],
        calibration_intercept=metrics["calibration_intercept"],
        percentile_concordance=conc,
        top_decile_concordance=conc.get("top_10_concordance", float("nan")),
        empirical_error_sd=emp_sd,
        empirical_error_mean=emp_mean,
        coverage_n=coverage_n,
        coverage_95=coverage_95,
        mean_se=mean_se,
        raw_parser_checked=raw_checked,
        raw_parser_n=raw_n,
        raw_parser_max_abs_diff=raw_max_diff,
        raw_parser_agrees=raw_agrees,
        cross_ancestry=cross_ancestry,
        caveats=caveats,
    )

    if verbose >= 1:
        print(report.summary())

    return report


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_platform(
    model: Any,
    platform_name: Optional[str],
    platform_manifest: Optional[Union[str, Path]],
    platform_variants: Optional[Sequence[str]],
) -> Tuple[Set[str], Optional[PlatformInfo], Optional[str], Optional[str]]:
    """Return ``(variant_id_set, platform_info, label, caveat)``."""
    n_given = sum(
        x is not None for x in (platform_name, platform_manifest, platform_variants)
    )
    if n_given > 1:
        raise ValidationError(
            "Provide at most one of platform_name, platform_manifest, platform_variants."
        )

    if platform_name is not None:
        ids, info = load_platform_from_name(platform_name)
        return set(ids), info, platform_name, None
    if platform_manifest is not None:
        ids, info = load_platform_from_manifest(str(platform_manifest))
        return set(ids), info, str(platform_manifest), None
    if platform_variants is not None:
        ids = load_platform_variants_from_list(list(platform_variants))
        return set(ids), None, "platform_variants", None

    # No override: prefer the model's declared platform, else reconstruct. A model
    # fit from an explicit variant list stores the sentinel ``"custom"`` (not a real
    # platform name), so an unknown name falls through to reconstruction.
    declared = getattr(model, "_platform_name", None)
    if declared:
        try:
            ids, info = load_platform_from_name(declared)
            return set(ids), info, declared, None
        except ValidationError:
            pass

    ids = _model_used_variant_ids(model)
    caveat = (
        "Platform footprint reconstructed from the model's used variants (observed + "
        "predictors): the model declares no built-in platform name and no override "
        "was given. Scoring is faithful (unused platform variants do not affect the "
        "estimate), but the retained-variant count reflects the used subset, not the "
        "full chip."
    )
    return ids, None, "model-derived", caveat


def _model_used_variant_ids(model: Any) -> Set[str]:
    """Variant IDs the model can see: observed terms + every predictor."""
    ids: Set[str] = {v.variant_id for v in model.observed_variants}
    sub_models = getattr(model, "imputed_models", None)
    if sub_models is None:
        sub_models = getattr(model, "region_models", [])
    for m in sub_models:
        ids.update(m.predictor_variant_ids)
    return ids


def _maybe_subsample(
    gd: GenotypeData, max_samples: Optional[int], random_state: Optional[int]
) -> GenotypeData:
    """Deterministically subsample samples to ``max_samples`` (no-op if larger)."""
    if max_samples is None or gd.n_samples <= max_samples:
        return gd
    rng = np.random.default_rng(random_state)
    indices = np.arange(gd.n_samples)
    rng.shuffle(indices)
    indices = np.sort(indices[:max_samples])
    return GenotypeData(
        dosage_matrix=gd.dosage_matrix[indices, :],
        variant_info=gd.variant_info.copy(),
        sample_ids=[gd.sample_ids[i] for i in indices],
        genome_build=gd.genome_build,
        source_file=gd.source_file,
    )


def _batch_metrics(
    s_est: np.ndarray, s_true: np.ndarray, percentiles: List[int]
) -> Tuple[Dict[str, float], Dict[str, float], float, float, List[str]]:
    """Accuracy + concordance + empirical error, degrading gracefully."""
    caveats: List[str] = []
    nan = float("nan")
    metrics = {
        k: nan
        for k in (
            "correlation",
            "r2",
            "spearman_rho",
            "mae",
            "rmse",
            "calibration_slope",
            "calibration_intercept",
        )
    }

    valid = ~(np.isnan(s_est) | np.isnan(s_true))
    e = s_est[valid]
    t = s_true[valid]
    n_valid = e.size

    if n_valid >= 2:
        diff = t - e
        emp_sd = float(np.std(diff, ddof=1))
        emp_mean = float(np.mean(diff))
    else:
        emp_sd = nan
        emp_mean = nan

    if n_valid < _MIN_SAMPLES_METRICS:
        raise ValidationError(
            f"Need at least {_MIN_SAMPLES_METRICS} valid samples for masking "
            f"validation, got {n_valid}."
        )

    if e.std() == 0 or t.std() == 0:
        caveats.append(
            "Degenerate cohort: estimated and/or true PRS has zero variance, so "
            "correlation/calibration are undefined (reported as nan)."
        )
    else:
        m = compute_prs_metrics(s_est, s_true)
        metrics = {
            "correlation": float(m.correlation),
            "r2": float(m.r2),
            "spearman_rho": float(m.spearman_rho),
            "mae": float(m.mae),
            "rmse": float(m.rmse),
            "calibration_slope": float(m.calibration_slope),
            "calibration_intercept": float(m.calibration_intercept),
        }

    if n_valid >= _MIN_SAMPLES_CONCORDANCE:
        conc = compute_percentile_concordance(s_est, s_true, percentiles)
    else:
        conc = {}
        caveats.append(
            f"Percentile concordance skipped: needs >= {_MIN_SAMPLES_CONCORDANCE} "
            f"valid samples, got {n_valid}."
        )

    return metrics, conc, emp_sd, emp_mean, caveats


def _sample_genotype_records(
    gd: GenotypeData, sample_idx: int
) -> List[Tuple[str, str, int, str]]:
    """Render ``(variant_id, chromosome, position, genotype)`` for one sample's
    hard-called dosages. NaN / non-integer dosages render to ``None`` and are
    dropped — exactly the off-platform / no-call variants a real upload omits."""
    vinfo = gd.variant_info
    dm = gd.dosage_matrix
    records: List[Tuple[str, str, int, str]] = []
    for col, row in enumerate(vinfo.itertuples(index=False)):
        geno = render_genotype_string(row.ref_allele, row.alt_allele, dm[sample_idx, col])
        if geno is None:
            continue
        records.append((str(row.variant_id), str(row.chromosome), int(row.position), geno))
    return records


def _interval_coverage(
    model: Any,
    masked_gd: GenotypeData,
    s_true: np.ndarray,
    coverage_max_samples: int,
) -> Tuple[int, Optional[float], Optional[float], Optional[str]]:
    """Fraction of the subsample whose true PRS lies in predict()'s 95% CI.

    Uses the DataFrame ``predict`` path (oriented scorer) with the model's own build
    so the compatibility guard stays silent. Validates the *deployed* (calibrated)
    interval when the model has calibration, else the raw interval. Only meaningful
    for hard-called reference data (the upload path needs genotype strings)."""
    if not is_hard_called(masked_gd.dosage_matrix):
        return 0, None, None, (
            "Interval coverage skipped: reference is continuous (DS/GP), which has no "
            "genotype-string upload representation."
        )

    build = getattr(model, "_genome_build", None)
    n = min(coverage_max_samples, masked_gd.n_samples)
    covered = 0
    checked = 0
    se_values: List[float] = []
    for i in range(n):
        if np.isnan(s_true[i]):
            continue
        records = _sample_genotype_records(masked_gd, i)
        if not records:
            continue
        df = pd.DataFrame(records, columns=["rsid", "chromosome", "position", "genotype"])
        result = model.predict(
            df, apply_calibration=True, genome_build=build, strict=False
        )
        lo = result.ci_lower_scaled if result.ci_lower_scaled is not None else result.ci_lower
        hi = result.ci_upper_scaled if result.ci_upper_scaled is not None else result.ci_upper
        se = result.se_scaled if result.se_scaled is not None else result.se
        if lo <= s_true[i] <= hi:
            covered += 1
        if se is not None and np.isfinite(se):
            se_values.append(float(se))
        checked += 1

    if checked == 0:
        return 0, None, None, "Interval coverage skipped: no scorable samples."
    mean_se = float(np.mean(se_values)) if se_values else None
    return checked, covered / checked, mean_se, None


def _raw_parser_round_trip(
    model: Any,
    masked_gd: GenotypeData,
    s_est: np.ndarray,
    raw_parser_max_samples: int,
    run_check: bool,
) -> Tuple[bool, int, Optional[float], Optional[bool], Optional[str]]:
    """Write masked samples to a synthetic 23andMe file and confirm the public
    ``predict(path)`` matches the (uncalibrated) batch estimate."""
    if not run_check:
        return False, 0, None, None, None
    if not is_hard_called(masked_gd.dosage_matrix):
        return False, 0, None, None, (
            "Raw-parser check skipped: reference is continuous (DS/GP), which has no "
            "DTC-file representation."
        )
    try:
        import snps  # noqa: F401
    except ImportError:
        return False, 0, None, None, (
            "Raw-parser check skipped: the 'snps' package is not installed."
        )

    build = getattr(model, "_genome_build", None)
    n = min(raw_parser_max_samples, masked_gd.n_samples)
    max_diff = 0.0
    checked = 0
    for i in range(n):
        if np.isnan(s_est[i]):
            continue
        records = _sample_genotype_records(masked_gd, i)
        if not records:
            continue
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as fh:
            tmp_path = Path(fh.name)
        try:
            _write_synthetic_23andme(tmp_path, records)
            result = model.predict(
                tmp_path,
                apply_calibration=False,
                genome_build=build,
                strict=False,
            )
            max_diff = max(max_diff, abs(float(result.prs) - float(s_est[i])))
            checked += 1
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if checked == 0:
        return False, 0, None, None, "Raw-parser check skipped: no scorable samples."
    return True, checked, max_diff, max_diff <= 1e-6, None


def _write_synthetic_23andme(
    path: Union[str, Path], records: Sequence[Tuple[str, str, int, str]]
) -> None:
    """Write records as a 23andMe-format DTC file the ``snps`` parser autodetects.

    The first comment line must contain the token ``23andMe`` and the last comment
    line must be exactly ``# rsid<TAB>chromosome<TAB>position<TAB>genotype``; data
    rows are tab-separated ``rsid<TAB>chrom<TAB>pos<TAB>genotype``. Genotypes are
    raw (ALT-counted) strings from :func:`render_genotype_string`; ``predict``'s
    ``count_allele`` re-orients per role, so they must NOT be pre-oriented.
    """
    lines = [
        "# This data file generated by 23andMe (synthetic; imputed-prs validation).",
        "# rsid\tchromosome\tposition\tgenotype",
    ]
    for variant_id, chrom, pos, geno in records:
        lines.append(f"{variant_id}\t{chrom}\t{pos}\t{geno}")
    Path(path).write_text("\n".join(lines) + "\n")
